import torch
import numpy as np
import os
import shutil
from typing import Tuple, List, Optional, Union, Dict, Any
import transforms3d
import math
from torch.utils.tensorboard.writer import SummaryWriter


def ensure_directory(path: str, clean: bool = False) -> None:
    """
    Ensure directory exists, optionally cleaning it first.
    
    Args:
        path: Directory path to create
        clean: If True, remove existing directory first
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device(gpu_id: Union[str, int] = "0") -> torch.device:
    """
    Setup compute device and environment variables.
    
    Args:
        gpu_id: GPU device ID or "cpu"
        
    Returns:
        torch.device: Configured device
    """
    # Set environment variables for stability
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    if gpu_id != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    return device


def batch_apply(func: callable, tensor: torch.Tensor, batch_size: int, *args, **kwargs) -> torch.Tensor:
    """
    Apply function to tensor in batches to manage memory usage.
    
    Args:
        func: Function to apply
        tensor: Input tensor
        batch_size: Batch size for processing
        *args, **kwargs: Additional arguments for func
        
    Returns:
        Concatenated results
    """
    results = []
    for i in range(0, tensor.shape[0], batch_size):
        batch = tensor[i:i+batch_size]
        result = func(batch, *args, **kwargs)
        results.append(result)
    
    return torch.cat(results, dim=0)


def safe_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe division with epsilon to avoid division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor  
        eps: Small epsilon value
        
    Returns:
        Division result
    """
    return numerator / (denominator + eps)


def clamp_to_range(tensor: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """
    Clamp tensor values to be within [min_val, max_val] range.
    
    Args:
        tensor: Input tensor
        min_val: Minimum values (can be tensor or scalar)
        max_val: Maximum values (can be tensor or scalar)
        
    Returns:
        Clamped tensor
    """
    return torch.clamp(tensor, min=min_val, max=max_val)


def normalize_tensor(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize tensor along specified dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized tensor
    """
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def compute_euler_from_rotation_matrix(rot_matrix: torch.Tensor, axes: str = 'sxyz') -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles using transforms3d.
    
    Args:
        rot_matrix: Rotation matrices of shape (..., 3, 3)
        axes: Euler angle convention
        
    Returns:
        Euler angles of shape (..., 3)
    """
    original_shape = rot_matrix.shape[:-2]
    rot_matrix = rot_matrix.view(-1, 3, 3)
    
    euler_angles = []
    for i in range(rot_matrix.shape[0]):
        matrix = rot_matrix[i].detach().cpu().numpy()
        euler = transforms3d.euler.mat2euler(matrix, axes=axes)
        euler_angles.append(euler)
    
    euler_tensor = torch.tensor(euler_angles, device=rot_matrix.device, dtype=rot_matrix.dtype)
    return euler_tensor.view(*original_shape, 3)


def compute_transformation_matrix(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Compute 4x4 transformation matrix from translation and rotation.
    
    Args:
        translation: Translation vectors of shape (..., 3)
        rotation: Rotation matrices of shape (..., 3, 3)
        
    Returns:
        Transformation matrices of shape (..., 4, 4)
    """
    batch_shape = translation.shape[:-1]
    device = translation.device
    
    # Create homogeneous transformation matrix
    transform = torch.zeros(*batch_shape, 4, 4, device=device, dtype=translation.dtype)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    
    return transform


def sample_sphere_uniform(n_samples: int, device: torch.device = None) -> torch.Tensor:
    """
    Sample points uniformly on unit sphere.
    
    Args:
        n_samples: Number of samples
        device: Device for tensor creation
        
    Returns:
        Points on sphere of shape (n_samples, 3)
    """
    if device is None:
        device = torch.device('cpu')
        
    # Sample using normal distribution and normalize
    points = torch.randn(n_samples, 3, device=device)
    return normalize_tensor(points, dim=-1)


def sample_cone_uniform(n_samples: int, axis: torch.Tensor, angle: float, device: torch.device = None) -> torch.Tensor:
    """
    Sample directions uniformly within a cone around an axis.
    
    Args:
        n_samples: Number of samples
        axis: Cone axis direction (3,)
        angle: Half-angle of cone in radians
        device: Device for tensor creation
        
    Returns:
        Direction vectors of shape (n_samples, 3)
    """
    if device is None:
        device = torch.device('cpu')
    
    axis = axis.to(device)
    axis = normalize_tensor(axis)
    
    # Sample uniform angles
    theta = 2 * math.pi * torch.rand(n_samples, device=device)
    phi = torch.acos(1 - torch.rand(n_samples, device=device) * (1 - math.cos(angle)))
    
    # Convert to Cartesian coordinates in cone coordinate system
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    # Create rotation matrix to align z-axis with given axis
    z_axis = torch.tensor([0, 0, 1], device=device, dtype=axis.dtype)
    if torch.allclose(axis, z_axis):
        # No rotation needed
        directions = torch.stack([x, y, z], dim=1)
    elif torch.allclose(axis, -z_axis):
        # Flip z
        directions = torch.stack([x, y, -z], dim=1)
    else:
        # General rotation
        v = torch.cross(z_axis, axis)
        s = torch.norm(v)
        c = torch.dot(z_axis, axis)
        
        # Rodrigues' rotation formula
        vx = torch.tensor([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]], device=device, dtype=axis.dtype)
        
        R = torch.eye(3, device=device, dtype=axis.dtype) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
        
        cone_dirs = torch.stack([x, y, z], dim=1)
        directions = (R @ cone_dirs.T).T
    
    return directions


def compute_condition_number(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute condition number of matrices using SVD.
    
    Args:
        matrix: Input matrices of shape (..., M, N)
        eps: Epsilon for numerical stability
        
    Returns:
        Condition numbers of shape (...)
    """
    try:
        U, S, V = torch.svd(matrix)
        max_sv = S[..., 0]  # Largest singular value
        min_sv = S[..., -1]  # Smallest singular value
        condition_number = max_sv / (min_sv + eps)
        return condition_number
    except (RuntimeError, torch.linalg.LinAlgError, ValueError):
        # Fallback: use Frobenius norm when SVD fails
        frobenius_norm = torch.norm(matrix, dim=[-2, -1])
        return 1.0 / (frobenius_norm + eps)


def stable_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Stable logarithm computation.
    
    Args:
        x: Input tensor
        eps: Minimum value to avoid log(0)
        
    Returns:
        Logarithm of (x + eps)
    """
    return torch.log(torch.clamp(x, min=eps))


def truncated_normal_init(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, 
                         a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Initialize tensor with truncated normal distribution.
    
    Args:
        tensor: Tensor to initialize
        mean: Mean of distribution
        std: Standard deviation
        a: Lower bound (in std units)
        b: Upper bound (in std units)
        
    Returns:
        Initialized tensor
    """
    with torch.no_grad():
        # Use torch.nn.init.trunc_normal_ if available, otherwise approximate
        if hasattr(torch.nn.init, 'trunc_normal_'):
            torch.nn.init.trunc_normal_(tensor, mean, std, mean + a * std, mean + b * std)
        else:
            # Fallback: normal initialization with clipping
            torch.nn.init.normal_(tensor, mean, std)
            tensor.clamp_(mean + a * std, mean + b * std)
    
    return tensor


def create_batch_mask(batch_size: int, indices: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    Create boolean mask from indices.
    
    Args:
        batch_size: Size of the batch
        indices: Indices to set to True
        device: Device for tensor creation
        
    Returns:
        Boolean mask of shape (batch_size,)
    """
    if device is None:
        device = indices.device
    
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask


def soft_clamp(x: torch.Tensor, min_val: float = None, max_val: float = None, 
               smoothness: float = 0.1) -> torch.Tensor:
    """
    Soft clamping using smooth functions instead of hard clipping.
    
    Args:
        x: Input tensor
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
        smoothness: Smoothness parameter
        
    Returns:
        Soft-clamped tensor
    """
    result = x
    
    if min_val is not None:
        # Smooth minimum using log-sum-exp
        result = min_val + smoothness * torch.log(1 + torch.exp((result - min_val) / smoothness))
    
    if max_val is not None:
        # Smooth maximum
        result = max_val - smoothness * torch.log(1 + torch.exp((max_val - result) / smoothness))
    
    return result


def weighted_average(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """
    Compute weighted average of tensors.
    
    Args:
        tensors: List of tensors to average
        weights: Corresponding weights
        
    Returns:
        Weighted average tensor
    """
    if len(tensors) != len(weights):
        raise ValueError("Number of tensors must match number of weights")
    
    weighted_sum = sum(w * t for w, t in zip(weights, tensors))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight


class MovingAverage:
    """Exponential moving average tracker."""
    
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.value = None
    
    def update(self, new_value: torch.Tensor) -> torch.Tensor:
        """Update moving average with new value."""
        if self.value is None:
            self.value = new_value.clone()
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value
    
    def get(self) -> torch.Tensor:
        """Get current moving average value."""
        return self.value


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor") -> None:
    """
    Validate tensor has expected shape.
    
    Args:
        tensor: Input tensor
        expected_shape: Expected shape (use -1 for any size)
        name: Tensor name for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    actual_shape = tensor.shape
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(f"{name} has {len(actual_shape)} dimensions, expected {len(expected_shape)}")
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(f"{name} dimension {i} is {actual}, expected {expected}")


def print_tensor_stats(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Print statistics about a tensor (for debugging)."""
    print(f"{name} stats:")
    print(f"  Shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  min: {tensor.min().item():.6f}")
    print(f"  max: {tensor.max().item():.6f}")
    print(f"  mean: {tensor.mean().item():.6f}")
    print(f"  std: {tensor.std().item():.6f}")
    if tensor.requires_grad:
        print(f"  requires_grad: True")
        if tensor.grad is not None:
            print(f"  grad_norm: {tensor.grad.norm().item():.6f}")
    print()


# ============================================================================
# Rotation Utilities (from rot6d.py)
# ============================================================================

def compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D orthogonal representation to rotation matrix.
    Based on Zhou et al. CVPR19: On the Continuity of Rotation Representations.
    
    Args:
        poses: 6D rotation representation tensor of shape (batch, 6)
        
    Returns:
        Rotation matrices of shape (batch, 3, 3)
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """
    Robust 6D rotation conversion that treats both input vectors equally.
    Instead of making 2nd vector orthogonal to first, creates a balanced base.
    
    Args:
        poses: 6D rotation representation tensor of shape (batch, 6)
        
    Returns:
        Rotation matrices of shape (batch, 3, 3)
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """
    Normalize vectors along last dimension.
    
    Args:
        v: Input vectors
        
    Returns:
        Normalized vectors
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute cross product of two 3D vectors.
    
    Args:
        u, v: Input vectors of shape (batch, 3)
        
    Returns:
        Cross product vectors of shape (batch, 3)
    """
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out

class Logger:
    """TensorBoard logger for optimization tracking."""
    
    def __init__(self, log_dir: str, thres_fc: float = 0.3, thres_dis: float = 0.005, thres_pen: float = 0.02):
        """
        Create a Logger for tensorboard scalars.
        
        Args:
            log_dir: Directory for logs
            thres_fc: Force closure threshold for success estimation
            thres_dis: Distance threshold for success estimation  
            thres_pen: Penetration threshold for data filtering
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.thres_fc = thres_fc
        self.thres_dis = thres_dis
        self.thres_pen = thres_pen

    def log(self, energy: torch.Tensor, E_fc: torch.Tensor, E_dis: torch.Tensor, 
            E_pen: torch.Tensor, E_spen: torch.Tensor, E_joints: torch.Tensor, 
            step: int, show: bool = False) -> None:
        """
        Log energy terms and estimate success rate using energy thresholds.
        
        Args:
            energy: Weighted sum of all terms
            E_fc: Force closure energy
            E_dis: Distance energy
            E_pen: Penetration energy
            E_spen: Self-penetration energy
            E_joints: Joint limit energy
            step: Current iteration of optimization
            show: Whether to print current energy terms to console
        """
        success_fc = E_fc < self.thres_fc
        success_dis = E_dis < self.thres_dis
        success_pen = E_pen < self.thres_pen
        success = success_fc * success_dis * success_pen
        
        self.writer.add_scalar('Energy/energy', energy.mean(), step)
        self.writer.add_scalar('Energy/fc', E_fc.mean(), step)
        self.writer.add_scalar('Energy/dis', E_dis.mean(), step)
        self.writer.add_scalar('Energy/pen', E_pen.mean(), step)

        self.writer.add_scalar('Success/success', success.float().mean(), step)
        self.writer.add_scalar('Success/fc', success_fc.float().mean(), step)
        self.writer.add_scalar('Success/dis', success_dis.float().mean(), step)
        self.writer.add_scalar('Success/pen', success_pen.float().mean(), step)

        if show:
            print(f'Step %d energy: %f  fc: %f  dis: %f  pen: %f  spen: %f  joints: %f' % 
                  (step, energy.mean(), E_fc.mean(), E_dis.mean(), E_pen.mean(), E_spen.mean(), E_joints.mean()))
            print(f'success: %f  fc: %f  dis: %f  pen: %f' % 
                  (success.float().mean(), success_fc.float().mean(), success_dis.float().mean(), success_pen.float().mean()))
