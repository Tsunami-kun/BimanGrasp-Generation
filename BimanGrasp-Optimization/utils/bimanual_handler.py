from dataclasses import dataclass
from typing import Callable, Any, Tuple, Dict, List, Optional, Union
import torch
import os
import numpy as np
import transforms3d

from .config import JOINT_NAMES, TRANSLATION_NAMES, ROTATION_NAMES
from .common import robust_compute_rotation_matrix_from_ortho6d


@dataclass
class HandState:
    """Represents the complete state of a single hand."""
    
    hand_pose: torch.Tensor
    contact_point_indices: torch.Tensor
    global_translation: torch.Tensor
    global_rotation: torch.Tensor
    current_status: Any  # Forward kinematics result
    contact_points: torch.Tensor
    grad_hand_pose: Optional[torch.Tensor] = None
    
    def clone(self) -> 'HandState':
        """Create a deep copy of the hand state."""
        return HandState(
            hand_pose=self.hand_pose.clone() if self.hand_pose is not None else None,
            contact_point_indices=self.contact_point_indices.clone() if self.contact_point_indices is not None else None,
            global_translation=self.global_translation.clone() if self.global_translation is not None else None,
            global_rotation=self.global_rotation.clone() if self.global_rotation is not None else None,
            current_status=self.current_status,  # Usually doesn't need deep copy
            contact_points=self.contact_points.clone() if self.contact_points is not None else None,
            grad_hand_pose=self.grad_hand_pose.clone() if self.grad_hand_pose is not None else None
        )


@dataclass
class EnergyTerms:
    """Container for all energy terms in grasp optimization."""
    
    total: torch.Tensor
    force_closure: torch.Tensor
    distance: torch.Tensor
    penetration: torch.Tensor
    self_penetration: torch.Tensor
    joint_limits: torch.Tensor
    wrench_volume: torch.Tensor = None
    
    def __getitem__(self, key: Union[str, int]) -> torch.Tensor:
        """Allow dictionary-like access."""
        if isinstance(key, str):
            attr_map = {
                'total': self.total,
                'fc': self.force_closure,
                'dis': self.distance,
                'pen': self.penetration,
                'spen': self.self_penetration,
                'joints': self.joint_limits,
                'vew': self.wrench_volume
            }
            return attr_map.get(key)
        elif isinstance(key, int):
            # Support indexing for batch operations
            return EnergyTerms(
                total=self.total[key] if self.total is not None else None,
                force_closure=self.force_closure[key] if self.force_closure is not None else None,
                distance=self.distance[key] if self.distance is not None else None,
                penetration=self.penetration[key] if self.penetration is not None else None,
                self_penetration=self.self_penetration[key] if self.self_penetration is not None else None,
                joint_limits=self.joint_limits[key] if self.joint_limits is not None else None,
                wrench_volume=self.wrench_volume[key] if self.wrench_volume is not None else None
            )
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")


@dataclass 
class GraspData:
    """Standardized grasp data format for saving and loading."""
    
    scale: float
    qpos_left: Dict[str, float]
    qpos_left_st: Dict[str, float]  # Starting pose
    qpos_right: Dict[str, float] 
    qpos_right_st: Dict[str, float]  # Starting pose
    energy: float
    E_fc: float
    E_dis: float
    E_pen: float
    E_spen: float
    E_joints: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            'scale': self.scale,
            'qpos_left': self.qpos_left,
            'qpos_left_st': self.qpos_left_st,
            'qpos_right': self.qpos_right,
            'qpos_right_st': self.qpos_right_st,
            'energy': self.energy,
            'E_fc': self.E_fc,
            'E_dis': self.E_dis,
            'E_pen': self.E_pen,
            'E_spen': self.E_spen,
            'E_joints': self.E_joints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraspData':
        """Create from dictionary."""
        return cls(**data)


class BimanualPair:
    """
    Manages a pair of left and right hand models with unified operations.
    Eliminates code duplication in dual-hand computations.
    """
    
    def __init__(self, left_hand_model, right_hand_model, device='cuda'):
        self.left = left_hand_model
        self.right = right_hand_model
        self.device = device
        
        # Cache for performance
        self._cached_states = {'left': None, 'right': None}
    
    def apply_to_both(self, func: Callable, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Apply a function to both hands and return results as (left_result, right_result).
        
        Args:
            func: Function to apply to each hand model
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (left_result, right_result)
        """
        left_result = func(self.left, *args, **kwargs)
        right_result = func(self.right, *args, **kwargs)
        return left_result, right_result
    
    def apply_with_context(self, func: Callable, context: str, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Apply function with handedness context (useful for asymmetric operations).
        
        Args:
            func: Function that takes (hand_model, handedness, *args, **kwargs)
            context: Additional context string
            
        Returns:
            Tuple of (left_result, right_result)
        """
        left_result = func(self.left, 'left', context, *args, **kwargs)
        right_result = func(self.right, 'right', context, *args, **kwargs)
        return left_result, right_result
    
    def get_hand_states(self) -> Tuple[HandState, HandState]:
        """Get current states of both hands."""
        def extract_state(hand_model):
            return HandState(
                hand_pose=hand_model.hand_pose,
                contact_point_indices=hand_model.contact_point_indices,
                global_translation=hand_model.global_translation,
                global_rotation=hand_model.global_rotation,
                current_status=hand_model.current_status,
                contact_points=hand_model.contact_points,
                grad_hand_pose=hand_model.hand_pose.grad
            )
        
        return self.apply_to_both(extract_state)
    
    def save_states(self) -> Tuple[HandState, HandState]:
        """Save current states and return them."""
        states = self.get_hand_states()
        self._cached_states['left'] = states[0].clone()
        self._cached_states['right'] = states[1].clone()
        return states
    
    def restore_states(self, left_state: HandState, right_state: HandState, 
                      reject_mask: torch.Tensor) -> None:
        """
        Restore hand states for rejected samples.
        
        Args:
            left_state: Left hand state to restore
            right_state: Right hand state to restore  
            reject_mask: Boolean mask indicating which samples to restore
        """
        if reject_mask.any():
            # Restore left hand
            self.left.hand_pose[reject_mask] = left_state.hand_pose[reject_mask]
            self.left.contact_point_indices[reject_mask] = left_state.contact_point_indices[reject_mask]
            self.left.global_translation[reject_mask] = left_state.global_translation[reject_mask]
            self.left.global_rotation[reject_mask] = left_state.global_rotation[reject_mask]
            self.left.current_status = self.left.chain.forward_kinematics(self.left.hand_pose[:, 9:])
            self.left.contact_points[reject_mask] = left_state.contact_points[reject_mask]
            if left_state.grad_hand_pose is not None:
                self.left.hand_pose.grad[reject_mask] = left_state.grad_hand_pose[reject_mask]
            
            # Restore right hand
            self.right.hand_pose[reject_mask] = right_state.hand_pose[reject_mask]
            self.right.contact_point_indices[reject_mask] = right_state.contact_point_indices[reject_mask]
            self.right.global_translation[reject_mask] = right_state.global_translation[reject_mask]
            self.right.global_rotation[reject_mask] = right_state.global_rotation[reject_mask]
            self.right.current_status = self.right.chain.forward_kinematics(self.right.hand_pose[:, 9:])
            self.right.contact_points[reject_mask] = right_state.contact_points[reject_mask]
            if right_state.grad_hand_pose is not None:
                self.right.hand_pose.grad[reject_mask] = right_state.grad_hand_pose[reject_mask]
    
    def zero_grad(self) -> None:
        """Zero gradients for both hands."""
        if self.left.hand_pose.grad is not None:
            self.left.hand_pose.grad.data.zero_()
        if self.right.hand_pose.grad is not None:
            self.right.hand_pose.grad.data.zero_()
    
    def compute_joint_limits_energy(self) -> torch.Tensor:
        """Compute joint limits energy for both hands."""
        def single_hand_joint_energy(hand_model):
            upper_violations = torch.sum(
                (hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * 
                (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), 
                dim=-1
            )
            lower_violations = torch.sum(
                (hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * 
                (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), 
                dim=-1
            )
            return upper_violations + lower_violations
        
        left_energy, right_energy = self.apply_to_both(single_hand_joint_energy)
        return left_energy + right_energy
    
    def compute_self_penetration_energy(self) -> torch.Tensor:
        """Compute self-penetration energy for both hands."""
        left_spen, right_spen = self.apply_to_both(lambda h: h.self_penetration())
        return left_spen + right_spen
    
    def compute_object_penetration_energy(self, object_model) -> torch.Tensor:
        """Compute hand-object penetration energy."""
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_surface_points = object_model.surface_points_tensor * object_scale
        
        def compute_penetration(hand_model):
            distances = hand_model.cal_distance(object_surface_points)
            distances = torch.clamp(distances, min=0)  # Only positive penetrations
            return distances.sum(-1)
        
        left_pen, right_pen = self.apply_to_both(compute_penetration)
        return left_pen + right_pen
    
    @property
    def batch_size(self) -> int:
        """Get batch size from hand models."""
        return self.left.hand_pose.shape[0]
    
    @property
    def total_contacts(self) -> int:
        """Get total number of contact points from both hands."""
        left_contacts = self.left.contact_points.shape[1] if self.left.contact_points is not None else 0
        right_contacts = self.right.contact_points.shape[1] if self.right.contact_points is not None else 0
        return left_contacts + right_contacts


def hand_pose_to_dict(hand_pose: torch.Tensor) -> Dict[str, float]:
    """
    Convert hand pose tensor to dictionary format.
    Unified function to replace multiple duplicated implementations.
    
    Args:
        hand_pose: Hand pose tensor with shape (31,) containing translation, rotation, and joint angles
        
    Returns:
        Dictionary with joint names and values
    """
    hand_pose_cpu = hand_pose.detach().cpu()
    
    # Joint angles (elements 9 onwards)
    qpos = dict(zip(JOINT_NAMES, hand_pose_cpu[9:].tolist()))
    
    # Rotation (elements 3-9, convert from 6D rotation to Euler angles)
    rot_6d = hand_pose_cpu[3:9].unsqueeze(0)
    rot_matrix = robust_compute_rotation_matrix_from_ortho6d(rot_6d)[0]
    euler_angles = transforms3d.euler.mat2euler(rot_matrix, axes='sxyz')
    qpos.update(dict(zip(ROTATION_NAMES, euler_angles)))
    
    # Translation (elements 0-3)
    translation = hand_pose_cpu[:3].tolist()
    qpos.update(dict(zip(TRANSLATION_NAMES, translation)))
    
    return qpos


def create_grasp_data(idx: int, object_model, left_hand_model, right_hand_model, 
                     left_hand_pose_st: torch.Tensor, right_hand_pose_st: torch.Tensor,
                     energy_terms: EnergyTerms) -> GraspData:
    """
    Create standardized grasp data for a single sample.
    
    Args:
        idx: Sample index
        object_model: Object model for scale information
        left_hand_model, right_hand_model: Hand models
        left_hand_pose_st, right_hand_pose_st: Starting poses
        energy_terms: Energy terms for this sample
        
    Returns:
        GraspData instance
    """
    # Extract object scale
    batch_idx = idx // object_model.batch_size_each
    sample_idx = idx % object_model.batch_size_each
    scale = object_model.object_scale_tensor[batch_idx][sample_idx].item()
    
    # Convert poses to dictionaries
    qpos_left = hand_pose_to_dict(left_hand_model.hand_pose[idx])
    qpos_left_st = hand_pose_to_dict(left_hand_pose_st[idx])
    qpos_right = hand_pose_to_dict(right_hand_model.hand_pose[idx])
    qpos_right_st = hand_pose_to_dict(right_hand_pose_st[idx])
    
    return GraspData(
        scale=scale,
        qpos_left=qpos_left,
        qpos_left_st=qpos_left_st,
        qpos_right=qpos_right,
        qpos_right_st=qpos_right_st,
        energy=energy_terms.total[idx].item(),
        E_fc=energy_terms.force_closure[idx].item(),
        E_dis=energy_terms.distance[idx].item(),
        E_pen=energy_terms.penetration[idx].item(),
        E_spen=energy_terms.self_penetration[idx].item(),
        E_joints=energy_terms.joint_limits[idx].item()
    )


def save_grasp_results(result_path: str, object_code_list: List[str], batch_size: int,
                      object_model, bimanual_pair: BimanualPair,
                      left_hand_pose_st: torch.Tensor, right_hand_pose_st: torch.Tensor,
                      energy_terms: EnergyTerms, step: Optional[int] = None) -> None:
    """
    Save grasp results in standardized format.
    
    Args:
        result_path: Directory to save results
        object_code_list: List of object codes
        batch_size: Batch size per object
        object_model: Object model
        bimanual_pair: BimanualPair instance
        left_hand_pose_st, right_hand_pose_st: Starting poses
        energy_terms: Energy terms
        step: Optional step number for intermediate saves
    """
    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(batch_size):
            idx = i * batch_size + j
            grasp_data = create_grasp_data(
                idx, object_model, bimanual_pair.left, bimanual_pair.right,
                left_hand_pose_st, right_hand_pose_st, energy_terms
            )
            data_list.append(grasp_data.to_dict())
        
        # Create filename
        filename = object_code
        if step is not None:
            filename += f'_{step}'
        filename += '.npy'
        
        # Save results
        np.save(os.path.join(result_path, filename), data_list, allow_pickle=True)