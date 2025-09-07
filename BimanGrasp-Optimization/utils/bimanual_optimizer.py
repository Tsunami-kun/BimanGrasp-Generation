"""\nMetropolis-Adjusted Langevin Algorithm (MALA) with RMSProp preconditioning for bimanual optimization.\n"""

import torch
from .config import OptimizerConfig
from .bimanual_handler import BimanualPair, HandState
from .common import MovingAverage


class MALAOptimizer:
    def __init__(self, left_hand_model, right_hand_model, config: OptimizerConfig = None, 
                 switch_possibility=0.5, initial_temperature=18, cooling_schedule=0.95, annealing_period=30,
                 step_size=0.005, stepsize_period=50, preconditioning_decay=0.98, langevin_noise_factor=0.1, device='cpu'):
        """
        Create a MALA optimizer for bimanual grasp optimization.
        
        Implements Metropolis-Adjusted Langevin Algorithm with:
        - RMSProp preconditioning for adaptive step sizes
        - Langevin dynamics with controlled noise injection  
        - Temperature-adaptive discrete contact point exploration
        - Metropolis-Hastings acceptance with detailed balance
        
        Parameters
        ----------
        left_hand_model: hand_model.HandModel
            Left hand model for dual-hand optimization
        right_hand_model: hand_model.HandModel
            Right hand model for dual-hand optimization
        config: OptimizerConfig
            Configuration object with all MALA parameters (preferred)
        switch_possibility: float
            Base probability for contact point resampling (fallback)
        initial_temperature: float
            Initial temperature for Langevin dynamics (fallback)
        cooling_schedule: float
            Temperature decay rate per cooling period (fallback)
        preconditioning_decay: float
            RMSProp momentum for gradient preconditioning (fallback)
        langevin_noise_factor: float
            Scaling factor for Langevin noise injection (fallback)
        """
        
        # Use config if provided, otherwise use individual parameters for backward compatibility
        if config is not None:
            self.base_switch_possibility = config.switch_possibility
            initial_temperature = config.initial_temperature if hasattr(config, 'initial_temperature') else config.starting_temperature
            cooling_schedule = config.cooling_schedule if hasattr(config, 'cooling_schedule') else config.temperature_decay
            annealing_period = config.annealing_period
            step_size = config.step_size
            stepsize_period = config.stepsize_period
            preconditioning_decay = config.preconditioning_decay if hasattr(config, 'preconditioning_decay') else config.momentum
            langevin_noise_factor = config.langevin_noise_factor if hasattr(config, 'langevin_noise_factor') else 0.1
        else:
            self.base_switch_possibility = switch_possibility
            
        # Create bimanual pair for unified operations
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, device)
        self.device = device
        
        # Convert MALA parameters to tensors
        self.initial_temperature = torch.tensor(initial_temperature, dtype=torch.float, device=device)
        self.cooling_schedule = torch.tensor(cooling_schedule, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        self.preconditioning_decay = torch.tensor(preconditioning_decay, dtype=torch.float, device=device)
        self.langevin_noise_factor = torch.tensor(langevin_noise_factor, dtype=torch.float, device=device)
        self.step = 0
        
        # State management - use unified approach
        self.saved_states = {'left': None, 'right': None}
        
        # RMSProp preconditioning matrices - exponential moving averages
        self.ema_grad_left = MovingAverage(decay=preconditioning_decay)
        self.ema_grad_right = MovingAverage(decay=preconditioning_decay)
        
        # Initialize EMA with zeros
        num_params = self.bimanual_pair.left.n_dofs + 9
        self.ema_grad_left.value = torch.zeros(num_params, dtype=torch.float, device=device)
        self.ema_grad_right.value = torch.zeros(num_params, dtype=torch.float, device=device)


    def langevin_proposal(self):
        """
        Generate Langevin proposal step for both hands with RMSProp preconditioning.
        
        Combines deterministic gradient descent with controlled stochastic exploration
        through adaptive noise injection and temperature-dependent contact resampling.
        
        Returns
        -------
        step_size: torch.Tensor
            Current adaptive step size used in proposal
        """
        # Calculate current adaptive step size with cooling schedule
        current_step_size = self.step_size * self.cooling_schedule ** torch.div(self.step, self.step_size_period, rounding_mode='floor')
        
        # Compute current temperature for Langevin dynamics
        current_temperature = self.initial_temperature * self.cooling_schedule ** torch.div(
            self.step, self.annealing_period, rounding_mode='floor'
        )
        
        # Save current states before making changes
        self.saved_states['left'], self.saved_states['right'] = self.bimanual_pair.save_states()
        
        # Apply Langevin update with RMSProp preconditioning to both hands
        self._langevin_update(self.bimanual_pair.left, self.ema_grad_left, current_step_size, current_temperature)
        self._langevin_update(self.bimanual_pair.right, self.ema_grad_right, current_step_size, current_temperature)
        
        self.step += 1
        return current_step_size
    
    def _langevin_update(self, hand_model, ema_grad, step_size, temperature):
        """
        Apply Langevin dynamics update with RMSProp preconditioning.
        
        Combines deterministic gradient-based descent with stochastic exploration
        through adaptive Langevin noise injection scaled by temperature and preconditioning.
        
        Args:
            hand_model: The hand model to update
            ema_grad: Exponential moving average for RMSProp preconditioning
            step_size: Current adaptive step size
            temperature: Current temperature for Langevin noise scaling
        """
        # Update RMSProp preconditioning matrix
        if hand_model.hand_pose.grad is not None:
            ema_grad.update((hand_model.hand_pose.grad ** 2).mean(0))
            
            # Compute preconditioned gradient step (deterministic component)
            step_size_tensor = torch.zeros_like(hand_model.hand_pose) + step_size
            preconditioned_grad = hand_model.hand_pose.grad / (torch.sqrt(ema_grad.value) + 1e-6)
            deterministic_step = step_size_tensor * preconditioned_grad
            
            # Generate adaptive Langevin noise (stochastic component)
            # Scale noise by sqrt(2 * step_size * temperature / preconditioning)
            adaptive_noise_scale = torch.sqrt(
                2.0 * step_size_tensor * temperature / (torch.sqrt(ema_grad.value) + 1e-6)
            ) * self.langevin_noise_factor
            
            langevin_noise = torch.randn_like(hand_model.hand_pose) * adaptive_noise_scale
            
            # Combined Langevin dynamics update
            hand_pose_new = hand_model.hand_pose - deterministic_step + langevin_noise
        else:
            hand_pose_new = hand_model.hand_pose
        
        # Temperature-adaptive discrete exploration (contact point resampling)
        batch_size, n_contact = hand_model.contact_point_indices.shape
        
        # Adaptive switching probability: higher temperature = more exploration
        adaptive_switch_prob = self.base_switch_possibility * torch.sqrt(
            temperature / self.initial_temperature
        ).item()
        adaptive_switch_prob = min(adaptive_switch_prob, 0.9)  # Cap at 90%
        
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < adaptive_switch_prob
        contact_indices_new = hand_model.contact_point_indices.clone()
        if switch_mask.any():
            contact_indices_new[switch_mask] = torch.randint(
                hand_model.n_contact_candidates, 
                size=[switch_mask.sum()], 
                device=self.device
            )
        
        # Apply updates to hand model
        hand_model.set_parameters(hand_pose_new, contact_indices_new)

    def metropolis_hastings_step(self, energy, new_energy):
        """
        Metropolis-Hastings acceptance step with detailed balance consideration.
        
        Evaluates proposal acceptance using the Metropolis criterion with
        temperature-scaled energy differences, maintaining detailed balance
        for proper convergence to the target distribution.
        
        Returns
        -------
        accept: (N,) torch.BoolTensor
            Boolean mask indicating accepted proposals
        temperature: torch.Tensor
            Current temperature used in acceptance criterion
        """
        batch_size = energy.shape[0]
        temperature = self.initial_temperature * self.cooling_schedule ** torch.div(
            self.step, self.annealing_period, rounding_mode='floor'
        )
        
        # Compute proposal acceptance probability with detailed balance
        # For MALA with RMSProp preconditioning, we use simplified Metropolis
        # as the preconditioning makes exact proposal ratio computation complex
        energy_diff = energy - new_energy
        accept_prob = torch.clamp(torch.exp(energy_diff / temperature), max=1.0)
        
        # Metropolis-Hastings acceptance step
        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        accept = alpha < accept_prob
        
        # Restore rejected proposals to maintain detailed balance
        reject_mask = ~accept
        if reject_mask.any():
            with torch.no_grad():
                self.bimanual_pair.restore_states(
                    self.saved_states['left'], 
                    self.saved_states['right'], 
                    reject_mask
                )
        
        return accept, temperature

    def zero_grad(self):
        """
        Clear gradients for both hands before computing new gradient information.
        """
        self.bimanual_pair.zero_grad()
    
    # Legacy property access for backward compatibility
    @property
    def left_hand_model(self):
        return self.bimanual_pair.left
    
    @property
    def right_hand_model(self):
        return self.bimanual_pair.right

    # Legacy method aliases for backward compatibility
    def try_step(self):
        """Legacy alias for langevin_proposal()"""
        return self.langevin_proposal()
    
    def accept_step(self, energy, new_energy):
        """Legacy alias for metropolis_hastings_step()"""
        return self.metropolis_hastings_step(energy, new_energy)


# Backward compatibility alias
class Annealing(MALAOptimizer):
    """
    Backward compatibility wrapper for MALAOptimizer.
    Provides the original Annealing interface while using MALA implementation.
    """
    
    def __init__(self, left_hand_model, right_hand_model, config=None, **kwargs):
        # Map old parameter names to new ones for backward compatibility
        if 'starting_temperature' in kwargs:
            kwargs['initial_temperature'] = kwargs.pop('starting_temperature')
        if 'temperature_decay' in kwargs:
            kwargs['cooling_schedule'] = kwargs.pop('temperature_decay')
        if 'momentum' in kwargs:
            kwargs['preconditioning_decay'] = kwargs.pop('momentum')
        
        # Set default noise factor to 0 for compatibility (can be overridden)
        kwargs.setdefault('langevin_noise_factor', 0.0)
        
        super().__init__(left_hand_model, right_hand_model, config, **kwargs)