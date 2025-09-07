"""
Object model for bimanual grasp generation.
Handles mesh loading, surface sampling, and SDF computation.
"""

import os
import numpy as np
import torch
import trimesh as tm
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go

from torchsdf import index_vertices_by_faces, compute_sdf


class ObjectModel:
    """
    Object model for mesh loading, scaling, and distance computation.
    Manages multiple objects with different scales for batch processing.
    """

    def __init__(self, data_root_path: str, batch_size_each: int, num_samples: int = 2000, 
                 device: str = "cuda", size: str = None):
        """
        Initialize object model.
        
        Args:
            data_root_path: Directory containing object mesh files
            batch_size_each: Batch size per object
            num_samples: Number of surface points to sample using FPS
            device: Device for tensor computations
            size: Scale setting ('normal', 'large', 'very_large')
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        # Model state
        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.surface_points_tensor = None
        
        # Configure object scaling based on size parameter
        self.size = size
        scale_options = {
            'normal': [0.09, 0.1, 0.11],
            'large': [0.15, 0.16, 0.17],
            'very_large': [0.18, 0.19, 0.20]
        }
        scale_values = scale_options.get(self.size, [0.09, 0.1, 0.11])  # Default to 'normal'
        self.scale_choice = torch.tensor(scale_values, dtype=torch.float, device=self.device)        
    
                
    def initialize(self, object_code_list):
        """
        Initialize object model with list of objects.
        
        Loads meshes, chooses scales, and samples surface points.
        
        Args:
            object_code_list: List of object codes or single string
            
        Returns:
            Dense point cloud from last processed object
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
            
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []

        dense_point_cloud = None
        for object_code in object_code_list:
            # Random scale selection for this object
            scale_indices = torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each,), device=self.device)
            self.object_scale_tensor.append(self.scale_choice[scale_indices])
            
            # Load object mesh
            mesh_path = os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj")
            mesh = tm.load(mesh_path, force="mesh", process=False)
            self.object_mesh_list.append(mesh)
            
            # Prepare mesh data for SDF computation
            object_verts = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)
            object_faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            
            # Sample surface points if requested
            if self.num_samples > 0:
                faces_float = torch.tensor(mesh.faces, dtype=torch.float, device=self.device)
                pytorch3d_mesh = pytorch3d.structures.Meshes(
                    object_verts.unsqueeze(0), faces_float.unsqueeze(0)
                )
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                    pytorch3d_mesh, num_samples=100 * self.num_samples
                )
                surface_points = pytorch3d.ops.sample_farthest_points(
                    dense_point_cloud, K=self.num_samples
                )[0][0]
                surface_points = surface_points.to(dtype=torch.float, device=self.device)
                self.surface_points_tensor.append(surface_points)
                
        # Stack tensors for batch processing
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        if self.num_samples > 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0)
            # Repeat for each batch item: (n_objects * batch_size_each, num_samples, 3)
            self.surface_points_tensor = self.surface_points_tensor.repeat_interleave(
                self.batch_size_each, dim=0
            )
            
        return dense_point_cloud

    def calculate_distance(self, x, with_closest_points=False):
        """
        Calculate signed distance from points to object surfaces.
        
        Args:
            x: Input points tensor (batch_size, n_points, 3)
            with_closest_points: If True, also return closest surface points
            
        Returns:
            tuple: (distances, normals) or (distances, normals, closest_points)
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        
        distances = []
        normals = []
        closest_points = []
        
        # Scale points for SDF computation
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x_scaled = x / scale.unsqueeze(2)
        
        # Compute SDF for each object
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis_squared, dis_signs, normal, _ = compute_sdf(x_scaled[i], face_verts)
            
            if with_closest_points:
                closest_points.append(x_scaled[i] - torch.sqrt(dis_squared).unsqueeze(1) * normal)
            
            # Convert to signed distance
            dis = torch.sqrt(dis_squared + 1e-8)
            signed_dis = dis * (-dis_signs)
            distances.append(signed_dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        
        # Stack and rescale results
        distances = torch.stack(distances) * scale
        normals = torch.stack(normals)
        
        # Reshape to original format
        distances = distances.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distances, normals, closest_points
            
        return distances, normals
    
    def cal_distance(self, x, with_closest_points=False):
        """
        Backward compatibility wrapper for calculate_distance.
        
        Args:
            x: Input points tensor
            with_closest_points: Whether to return closest points
            
        Returns:
            Same as calculate_distance
        """
        return self.calculate_distance(x, with_closest_points)

    def get_plotly_data(self, i: int, color: str = 'lightgreen', opacity: float = 0.5, pose=None):
        """
        Generate Plotly 3D mesh data for visualization.
        
        Args:
            i: Global batch index
            color: Mesh color
            opacity: Mesh opacity (0-1)
            pose: Optional transformation pose (4x4 matrix)
            
        Returns:
            List containing Plotly Mesh3d object
        """
        model_index = i // self.batch_size_each
        batch_index = i % self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, batch_index].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        
        # Scale vertices
        vertices = mesh.vertices * model_scale
        
        # Apply pose transformation if provided
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
            
        # Create Plotly mesh
        mesh_data = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            color=color, opacity=opacity
        )
        
        return [mesh_data]
