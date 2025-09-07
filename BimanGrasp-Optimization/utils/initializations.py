"""
Bimanual hand initialization utilities for grasp generation.
Provides functions to initialize hand poses and orientations around target objects.
"""

import math
import numpy as np
import torch
import transforms3d
import trimesh as tm
import pytorch3d.structures
import pytorch3d.ops

from utils.hand_model import HandModel

def initialize_convex_hull(left_hand_model, object_model, args):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices.
    
    Args:
        left_hand_model: HandModel instance for left hand
        object_model: ObjectModel instance containing target objects
        args: Configuration namespace with initialization parameters
        
    Returns:
        tuple: Normal vectors and sample points from object surface
    """
        
    device = left_hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_per_obj = object_model.batch_size_each
    total_batch_size = n_objects * batch_per_obj

    # Initialize translation and rotation tensors
    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    if left_hand_model.handedness != 'left_hand':
        raise ValueError("This function should initialize the left hand model")

    for i in range(n_objects):
        # Get inflated convex hull
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # Sample points from mesh surface
        dense_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_per_obj)
        p = pytorch3d.ops.sample_farthest_points(dense_cloud, K=batch_per_obj)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # Sample initialization parameters
        rand_vals = torch.rand([4, batch_per_obj], dtype=torch.float, device=device)
        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * rand_vals[0]
        cone_angle = args.theta_lower + (args.theta_upper - args.theta_lower) * rand_vals[1]
        azimuth = 2 * math.pi * rand_vals[2]
        roll = 2 * math.pi * rand_vals[3]

        # Solve transformation matrices
        # hand_rot: rotate the hand to align its grasping direction with the +z axis
        # cone_rot: jitter the hand's orientation in a cone  
        # world_rot and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull
        cone_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        world_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_per_obj):
            cone_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(azimuth[j], cone_angle[j], roll[j], axes='rzxz'),
                dtype=torch.float, device=device
            )
            world_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'
                ), dtype=torch.float, device=device
            )
        start_idx = i * batch_per_obj
        end_idx = start_idx + batch_per_obj
        z_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
        translation[start_idx:end_idx] = p - distance.unsqueeze(1) * (world_rot @ cone_rot @ z_vec).squeeze(2)
        hand_rot = torch.tensor(
            transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device
        )
        rotation[start_idx:end_idx] = world_rot @ cone_rot @ (-hand_rot)
    
    # Initialize joint angles using truncated normal distribution  
    # joint_angles_mu: hand-crafted canonicalized hand articulation
    joint_angles_mu = torch.tensor([
        0.1, 0, -0.6, 0, 0, 0, -0.6, 0, -0.1, 0, -0.6, 0,
        0, -0.2, 0, -0.6, 0, 0, -1.2, 0, -0.2, 0
    ], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (left_hand_model.joints_upper - left_hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, left_hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(left_hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
            left_hand_model.joints_lower[i] - 1e-6, left_hand_model.joints_upper[i] + 1e-6
        )

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    # Initialize contact point indices
    # Handle both old and new parameter names for backward compatibility
    n_contact = getattr(args, 'num_contacts', getattr(args, 'n_contact', 4))
    contact_indices = torch.randint(
        left_hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device
    )

    left_hand_model.set_parameters(hand_pose, contact_indices)
    return n, p


def initialize_dual_hand(right_hand_model, object_model, args):
    """
    Initialize both hands' positions and rotations to grasp an object symmetrically.
    
    Args:
        right_hand_model: HandModel instance for right hand
        object_model: ObjectModel instance containing target objects
        args: Configuration namespace with initialization parameters
        
    Returns:
        tuple: (left_hand_model, right_hand_model) with initialized poses
    """
    
    device = right_hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_per_obj = object_model.batch_size_each
    total_batch_size = n_objects * batch_per_obj    
    
    # Create left hand model
    left_hand_model = HandModel(
        mjcf_path='mjcf/left_shadow_hand.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/left_hand_contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device,
        handedness='left_hand'
    )
    
    n, p = initialize_convex_hull(left_hand_model, object_model, args)

    # Compute the right hand's parameters based on the left hand's
    rotation_right = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)
    translation_right = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        start_idx = i * batch_per_obj
        end_idx = start_idx + batch_per_obj
        # Mirror the normal vectors and points for symmetric grasp
        n[start_idx:end_idx, 0] = -n[start_idx:end_idx, 0]
        n[start_idx:end_idx, 1] = -n[start_idx:end_idx, 1]
        n[start_idx:end_idx, 2] = n[start_idx:end_idx, 2]

        p[start_idx:end_idx, 0] = -p[start_idx:end_idx, 0]
        p[start_idx:end_idx, 1] = -p[start_idx:end_idx, 1]
        p[start_idx:end_idx, 2] = p[start_idx:end_idx, 2]
        
        # Sample parameters for right hand
        rand_vals = torch.rand([4, batch_per_obj], dtype=torch.float, device=device)
        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * rand_vals[0]
        cone_angle = args.theta_lower + (args.theta_upper - args.theta_lower) * rand_vals[1]
        azimuth = 2 * math.pi * rand_vals[2]
        roll = 2 * math.pi * rand_vals[3]

        # Solve transformation matrices for right hand
        # hand_rot: rotate the hand to align its grasping direction with the +z axis
        # cone_rot: jitter the hand's orientation in a cone
        # world_rot and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull
        cone_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        world_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_per_obj):
            cone_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(azimuth[j], cone_angle[j], roll[j], axes='rzxz'),
                dtype=torch.float, device=device
            )
            world_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'
                ), dtype=torch.float, device=device
            )
        z_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
        translation_right[start_idx:end_idx] = p - distance.unsqueeze(1) * (world_rot @ cone_rot @ z_vec).squeeze(2)
        hand_rot = torch.tensor(
            transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device
        )
        rotation_right[start_idx:end_idx] = world_rot @ cone_rot @ hand_rot


    # Initialize right hand joint angles
    joint_angles_mu = torch.tensor([
        0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0,
        0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0
    ], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (right_hand_model.joints_upper - right_hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, right_hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(right_hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
            right_hand_model.joints_lower[i] - 1e-6, right_hand_model.joints_upper[i] + 1e-6
        )

    # Assemble right hand pose
    hand_pose_right = torch.cat([
        translation_right,
        rotation_right.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose_right.requires_grad_()

    # Set parameters for right hand model
    # Handle both old and new parameter names for backward compatibility
    n_contact = getattr(args, 'num_contacts', getattr(args, 'n_contact', 4))
    contact_indices = torch.randint(
        right_hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device
    )
    right_hand_model.set_parameters(hand_pose_right, contact_indices)

    return left_hand_model, right_hand_model