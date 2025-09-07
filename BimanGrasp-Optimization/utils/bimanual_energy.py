"""
Modular energy computation system for bimanual grasp optimization.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .config import EnergyConfig
from .bimanual_handler import BimanualPair, EnergyTerms
from .common import compute_condition_number, stable_log, safe_division


class GraspMatrixComputer:
    """
    Unified computer for grasp matrix analysis, computing both FC and VEW from single SVD.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def build_grasp_matrix(self, contact_points: torch.Tensor, contact_normals: torch.Tensor,
                          batch_size: int, n_contacts: int) -> torch.Tensor:
        """
        Build grasp matrix G for force closure analysis.
        
        Args:
            contact_points: Contact points (batch_size, n_contacts, 3)
            contact_normals: Contact normals (batch_size, n_contacts, 3)
            batch_size: Batch size
            n_contacts: Number of contacts
            
        Returns:
            Grasp matrix G (batch_size, 6, n_contacts)
        """
        G = torch.zeros(batch_size, 6, n_contacts, device=self.device)
        
        for i in range(n_contacts):
            # Force components (linear part)
            G[:, 0:3, i] = contact_normals[:, i, :]
            
            # Torque components (angular part): r × n
            r = contact_points[:, i, :]
            n = contact_normals[:, i, :]
            
            # Cross product components
            G[:, 3, i] = r[:, 1] * n[:, 2] - r[:, 2] * n[:, 1]
            G[:, 4, i] = r[:, 2] * n[:, 0] - r[:, 0] * n[:, 2]
            G[:, 5, i] = r[:, 0] * n[:, 1] - r[:, 1] * n[:, 0]
        
        return G
    
    def compute_fc_and_vew(self, G: torch.Tensor) -> tuple:
        """
        Compute both force closure and VEW from single SVD decomposition.
        
        Args:
            G: Grasp matrix (batch_size, 6, n_contacts)
            
        Returns:
            tuple: (force_closure_energy, vew_energy)
        """
        batch_size = G.shape[0]
        
        if G.shape[2] < 6:  # Not enough contacts
            return (torch.zeros(batch_size, device=self.device), 
                   torch.zeros(batch_size, device=self.device))
        
        try:
            # Single SVD computation for both metrics
            G_reg = G + 1e-6 * torch.randn_like(G)
            U, S, V = torch.svd(G_reg)
            
            # Force Closure: use condition number
            max_sv = S[..., 0]  # Largest singular value
            min_sv = S[..., -1]  # Smallest singular value
            condition_number = max_sv / (min_sv + 1e-8)
            energy_fc = stable_log(condition_number + 1.0) / 10.0
            
            # VEW: use product of first 6 singular values
            log_volume = torch.sum(
                stable_log(S[:, :min(6, S.shape[1])]), 
                dim=1
            )
            energy_vew = torch.exp(-log_volume / 10.0)
            
            return energy_fc, energy_vew
            
        except (RuntimeError, torch.linalg.LinAlgError, ValueError):
            # fallback when SVD fails
            frobenius_norm = torch.norm(G, dim=[1, 2])
            fallback_fc = 1.0 / (frobenius_norm + 0.01)
            fallback_vew = torch.ones(batch_size, device=self.device)
            return fallback_fc, fallback_vew


class ForceClosureComputer:
    """
    Computes force closure energy using unified grasp matrix computer.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.grasp_matrix_computer = GraspMatrixComputer(device)
    
    def compute(self, bimanual_pair: BimanualPair, object_model) -> torch.Tensor:
        """
        Compute force closure energy for bimanual grasp.
        
        Args:
            bimanual_pair: BimanualPair containing both hands
            object_model: Object model for contact normal computation
            
        Returns:
            Force Closure Energy
        """
        if (bimanual_pair.left.contact_points is None or 
            bimanual_pair.right.contact_points is None):
            batch_size = bimanual_pair.batch_size
            return torch.ones(batch_size, device=self.device)
        
        # Get contact information from both hands
        batch_size, n_contact_left, _ = bimanual_pair.left.contact_points.shape
        _, n_contact_right, _ = bimanual_pair.right.contact_points.shape
        n_total_contacts = n_contact_left + n_contact_right
        
        # Compute contact normals
        _, contact_normal_left = object_model.cal_distance(bimanual_pair.left.contact_points)
        _, contact_normal_right = object_model.cal_distance(bimanual_pair.right.contact_points)
        
        # Combine contact points and normals
        all_contact_points = torch.cat([
            bimanual_pair.left.contact_points, 
            bimanual_pair.right.contact_points
        ], dim=1)
        all_contact_normals = torch.cat([contact_normal_left, contact_normal_right], dim=1)
        
        # Normalize contact normals
        all_contact_normals = all_contact_normals / (
            torch.norm(all_contact_normals, dim=-1, keepdim=True) + 1e-8
        )
        
        # Build grasp matrix
        G = self.grasp_matrix_computer.build_grasp_matrix(
            all_contact_points, all_contact_normals, batch_size, n_total_contacts
        )
        
        # Get only force closure energy 
        energy_fc, _ = self.grasp_matrix_computer.compute_fc_and_vew(G)
        return energy_fc


class PenetrationComputer:
    """
    Computes penetration-related energies (object penetration and self-penetration).
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_object_penetration(self, bimanual_pair: BimanualPair, object_model) -> torch.Tensor:
        """
        Compute hand-object penetration energy.
        
        Args:
            bimanual_pair: BimanualPair containing both hands
            object_model: Object model
            
        Returns:
            Penetration energy tensor
        """
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_surface_points = object_model.surface_points_tensor * object_scale
        
        def compute_single_hand_penetration(hand_model):
            distances = hand_model.cal_distance(object_surface_points)
            distances = torch.clamp(distances, min=0)  # Only positive penetrations
            return distances.sum(-1)
        
        left_pen, right_pen = bimanual_pair.apply_to_both(compute_single_hand_penetration)
        return left_pen + right_pen
    
    def compute_self_penetration(self, bimanual_pair: BimanualPair) -> torch.Tensor:
        """
        Compute self-penetration and inter-hand penetration energy.
        
        Args:
            bimanual_pair: BimanualPair containing both hands
            
        Returns:
            Self-penetration energy tensor
        """
        # Self-penetration within each hand
        left_spen, right_spen = bimanual_pair.apply_to_both(lambda h: h.self_penetration())
        
        # Inter-hand penetration 
        surface_points_right = bimanual_pair.right.surface_point.detach().clone()
        inter_pen = bimanual_pair.left.cal_distance(surface_points_right)
        inter_pen = torch.clamp(inter_pen, min=0)
        
        return left_spen + right_spen + inter_pen.sum(-1)


class ContactDistanceComputer:
    """
    Computes contact distance energy (distance from contact points to object surface).
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute(self, bimanual_pair: BimanualPair, object_model) -> torch.Tensor:
        """
        Compute contact distance energy.
        
        Args:
            bimanual_pair: BimanualPair containing both hands
            object_model: Object model
            
        Returns:
            Contact distance energy tensor
        """
        if (bimanual_pair.left.contact_points is None or 
            bimanual_pair.right.contact_points is None):
            batch_size = bimanual_pair.batch_size
            return torch.zeros(batch_size, device=self.device)
        
        # Compute distances for both hands
        distance_left, _ = object_model.cal_distance(bimanual_pair.left.contact_points)
        distance_right, _ = object_model.cal_distance(bimanual_pair.right.contact_points)
        
        # Sum absolute distances
        energy_dis = torch.sum(distance_left.abs(), dim=-1) + torch.sum(distance_right.abs(), dim=-1)
        return energy_dis.to(self.device)


class WrenchVolumeComputer:
    """
    Computes wrench ellipse volume energy using unified grasp matrix computer.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.grasp_matrix_computer = GraspMatrixComputer(device)
    
    def compute(self, G: torch.Tensor) -> torch.Tensor:
        """
        Compute wrench ellipse volume from grasp matrix.
        
        Args:
            G: Grasp matrix (batch_size, 6, n_contacts)
            
        Returns:
            Wrench volume energy tensor
        """
        # get VEW 
        _, energy_vew = self.grasp_matrix_computer.compute_fc_and_vew(G)
        return energy_vew


class BimanualEnergyComputer:
    """
    Main energy computer that combines all energy terms for bimanual grasp optimization.
    """
    
    def __init__(self, config: EnergyConfig = None, device='cuda'):
        self.device = device
        self.config = config or EnergyConfig()
        
        # Initialize component computers
        self.force_closure_computer = ForceClosureComputer(device)
        self.penetration_computer = PenetrationComputer(device)
        self.contact_distance_computer = ContactDistanceComputer(device)
        self.wrench_volume_computer = WrenchVolumeComputer(device)
        self.grasp_matrix_computer = GraspMatrixComputer(device)
    
    def compute_all_energies(self, bimanual_pair: BimanualPair, object_model, 
                           verbose: bool = False) -> EnergyTerms:
        """
        Compute all energy terms for bimanual grasp.
        
        Args:
            bimanual_pair: BimanualPair containing both hands
            object_model: Object model
            verbose: If True, return individual energy terms
            
        Returns:
            EnergyTerms object containing all computed energies
        """
        batch_size = bimanual_pair.batch_size
        
        # Compute non-grasp-matrix energies
        energy_dis = self.contact_distance_computer.compute(bimanual_pair, object_model)
        energy_pen = self.penetration_computer.compute_object_penetration(bimanual_pair, object_model)
        energy_spen = self.penetration_computer.compute_self_penetration(bimanual_pair)
        energy_joints = bimanual_pair.compute_joint_limits_energy()
        
        # Compute FC and VEW together for efficiency (both need grasp matrix)
        if (bimanual_pair.left.contact_points is not None and 
            bimanual_pair.right.contact_points is not None):
            # Get contact information
            _, contact_normal_left = object_model.cal_distance(bimanual_pair.left.contact_points)
            _, contact_normal_right = object_model.cal_distance(bimanual_pair.right.contact_points)
            
            # Build grasp matrix once
            all_contact_points = torch.cat([
                bimanual_pair.left.contact_points, 
                bimanual_pair.right.contact_points
            ], dim=1)
            all_contact_normals = torch.cat([contact_normal_left, contact_normal_right], dim=1)
            all_contact_normals = all_contact_normals / (
                torch.norm(all_contact_normals, dim=-1, keepdim=True) + 1e-8
            )
            
            n_total_contacts = bimanual_pair.total_contacts
            G = self.grasp_matrix_computer.build_grasp_matrix(
                all_contact_points, all_contact_normals, batch_size, n_total_contacts
            )
            
            # Compute both FC and VEW from single SVD
            energy_fc, energy_vew = self.grasp_matrix_computer.compute_fc_and_vew(G)
            
            # Apply VEW weight (only include if enabled)
            if self.config.w_vew <= 0:
                energy_vew = torch.zeros(batch_size, device=self.device)
        else:
            energy_fc = torch.ones(batch_size, device=self.device)
            energy_vew = torch.zeros(batch_size, device=self.device)
        
        # Compute total weighted energy
        energy_total = (energy_fc + 
                  self.config.w_dis * energy_dis + 
                  self.config.w_pen * energy_pen + 
                  self.config.w_spen * energy_spen + 
                  self.config.w_joints * energy_joints + 
                  self.config.w_vew * energy_vew)
        
        return EnergyTerms(
            total=energy_total,
            force_closure=energy_fc,
            distance=energy_dis,
            penetration=energy_pen,
            self_penetration=energy_spen,
            joint_limits=energy_joints,
            wrench_volume=energy_vew
        )

def cal_energy(left_hand_model, right_hand_model, object_model, 
               w_dis=100.0, w_pen=125.0, w_spen=10.0, w_joints=1.0, w_vew=0.5, 
               verbose=False, device='cuda'):
    """
    Backward compatibility wrapper - redirects to calculate_energy.
    """
    return calculate_energy(left_hand_model, right_hand_model, object_model, 
                          w_dis, w_pen, w_spen, w_joints, w_vew, verbose, device)


def calculate_energy(left_hand_model, right_hand_model, object_model, 
               w_dis=100.0, w_pen=100.0, w_spen=10.0, w_joints=1.0, w_vew=0.0, 
               verbose=False, device='cuda'):
    """
    Backward compatibility wrapper for the new modular energy computation system.
    
    This function maintains the same interface as the original calculate_energy function
    but uses the new modular BimanualEnergyComputer internally.
    
    Args:
        left_hand_model, right_hand_model: Hand models
        object_model: Object model
        w_dis, w_pen, w_spen, w_joints, w_vew: Energy weights
        verbose: Return individual energy terms if True
        device: Compute device
        
    Returns:
        If verbose: (total_energy, energy_fc, energy_dis, energy_pen, energy_spen, energy_joints, energy_vew)
        Otherwise: total_energy
    """
    # Create configuration with provided weights
    energy_config = EnergyConfig(
        w_dis=w_dis, w_pen=w_pen, w_spen=w_spen, 
        w_joints=w_joints, w_vew=w_vew
    )
    
    # Create bimanual pair and energy computer
    bimanual_pair = BimanualPair(left_hand_model, right_hand_model, device)
    energy_computer = BimanualEnergyComputer(energy_config, device)
    
    # Compute all energies
    energy_terms = energy_computer.compute_all_energies(bimanual_pair, object_model, verbose=True)
    
    if verbose:
        return (energy_terms.total, energy_terms.force_closure, energy_terms.distance,
                energy_terms.penetration, energy_terms.self_penetration, 
                energy_terms.joint_limits, energy_terms.wrench_volume)
    else:
        return energy_terms.total
