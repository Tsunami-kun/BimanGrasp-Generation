import os
import sys
sys.path.append(os.path.realpath('.'))

import cProfile
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import random
import transforms3d
import shutil
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_dual_hand
from utils.bimanual_energy import calculate_energy, cal_energy, BimanualEnergyComputer
from utils.bimanual_optimizer import MALAOptimizer
from utils.common import robust_compute_rotation_matrix_from_ortho6d
from torch.multiprocessing import set_start_method
import plotly.graph_objects as go
from utils.common import Logger
from utils.config import ExperimentConfig, create_config_from_args
from utils.bimanual_handler import (
    BimanualPair, hand_pose_to_dict, save_grasp_results, EnergyTerms
)
from utils.common import setup_device, set_random_seeds, ensure_directory


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = None
        self.bimanual_pair = None
        self.object_model = None
        self.optimizer = None
        self.energy_computer = None
        self.logger = None
        
        # Profiling
        self.profiler = cProfile.Profile()
        
        # State tracking
        self.left_hand_pose_st = None
        self.right_hand_pose_st = None
        
    def setup_environment(self):
        """Setup device, random seeds, and environment variables."""
        self.device = setup_device(self.config.gpu)
        set_random_seeds(self.config.seed)
        np.seterr(all='raise')
        
        print(f"Using device: {self.device}")
        
    def setup_models(self):
        """Initialize hand and object models."""
        print("Setting up models...")
        
        # Create right hand model
        right_hand_model = HandModel(
            mjcf_path=self.config.paths.right_hand_mjcf,
            mesh_path=self.config.paths.mesh_path,
            contact_points_path=self.config.paths.right_contact_points,
            penetration_points_path=self.config.paths.penetration_points,
            device=self.device,
            n_surface_points=self.config.model.n_surface_points,
            handedness='right_hand'
        )
        
        # Create object model
        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=self.config.model.batch_size,
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size
        )
        self.object_model.initialize(self.config.object_code_list)
        
        # Initialize dual hands
        left_hand_model, right_hand_model = initialize_dual_hand(
            right_hand_model, self.object_model, self.config.initialization
        )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)
        
        # Save initial poses for optional debugging
        self.left_hand_pose_st = left_hand_model.hand_pose.detach()
        self.right_hand_pose_st = right_hand_model.hand_pose.detach()
        
        print(f'Left hand contact candidates: {left_hand_model.n_contact_candidates}')
        print(f'Right hand contact candidates: {right_hand_model.n_contact_candidates}')
        print(f'Total batch size: {self.config.total_batch_size}')
    
    def setup_optimization(self):
        """Initialize optimizer and energy computer."""
        
        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)
        
        # Create optimizer
        self.optimizer = MALAOptimizer(
            self.bimanual_pair.left, 
            self.bimanual_pair.right, 
            config=self.config.optimizer,
            device=self.device
        )
        
    def setup_logging(self):
        """Setup experiment logging and result directories."""
        
        # Create directories
        logs_path = self.config.paths.get_experiment_logs_path(self.config.name)
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        ensure_directory(logs_path, clean=True)
        ensure_directory(results_path, clean=True)
        
        # Create logger
        self.logger = Logger(
            log_dir=logs_path,
            thres_fc=self.config.energy.thres_fc,
            thres_dis=self.config.energy.thres_dis,
            thres_pen=self.config.energy.thres_pen
        )
        
        # Save experiment configuration
        config_path = os.path.join(results_path, 'config.txt')
        with open(config_path, 'w') as f:
            f.write(str(self.config))
        
    def run_optimization(self):
        """Run the optimization loop."""
        print("Starting optimization...")
        
        self.profiler.enable()
        
        # Initial energy computation
        energy_terms = self.energy_computer.compute_all_energies(
            self.bimanual_pair, self.object_model, verbose=True
        )
        
        energy_terms.total.sum().backward(retain_graph=True)
        self.logger.log(
            energy_terms.total, energy_terms.force_closure, energy_terms.distance,
            energy_terms.penetration, energy_terms.self_penetration, 
            energy_terms.joint_limits, 0, show=False
        )
        
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        # Main optimization loop
        for step in tqdm(range(1, self.config.optimizer.num_iterations + 1), desc='optimizing'):
            # MALA proposal step with Langevin dynamics
            step_size = self.optimizer.langevin_proposal()
            
            # Zero gradients and compute new energy
            self.optimizer.zero_grad()
            new_energy_terms = self.energy_computer.compute_all_energies(
                self.bimanual_pair, self.object_model, verbose=True
            )
            
            new_energy_terms.total.sum().backward(retain_graph=True)
            
            # Metropolis-Hastings acceptance step
            with torch.no_grad():
                accept, temperature = self.optimizer.metropolis_hastings_step(
                    energy_terms.total, new_energy_terms.total
                )
                
                # Update energies for accepted samples
                energy_terms.total[accept] = new_energy_terms.total[accept]
                energy_terms.distance[accept] = new_energy_terms.distance[accept]
                energy_terms.force_closure[accept] = new_energy_terms.force_closure[accept]
                energy_terms.penetration[accept] = new_energy_terms.penetration[accept]
                energy_terms.self_penetration[accept] = new_energy_terms.self_penetration[accept]
                energy_terms.joint_limits[accept] = new_energy_terms.joint_limits[accept]
                energy_terms.wrench_volume[accept] = new_energy_terms.wrench_volume[accept]
                
                # Log progress
                self.logger.log(
                    energy_terms.total, energy_terms.force_closure, energy_terms.distance,
                    energy_terms.penetration, energy_terms.self_penetration,
                    energy_terms.joint_limits, step, show=False
                )
        
        self.profiler.disable()
        return energy_terms
    
    def save_intermediate_results(self, step: int, energy_terms: EnergyTerms):
        """Save intermediate results during optimization."""
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        save_grasp_results(
            results_path, self.config.object_code_list, self.config.model.batch_size,
            self.object_model, self.bimanual_pair,
            self.left_hand_pose_st, self.right_hand_pose_st,
            energy_terms, step=step
        )
    
    def save_final_results(self, energy_terms: EnergyTerms):
        """Save final optimization results."""
        print("Saving final results...")
        
        results_path = self.config.paths.get_experiment_results_path(self.config.name)
        
        save_grasp_results(
            results_path, self.config.object_code_list, self.config.model.batch_size,
            self.object_model, self.bimanual_pair,
            self.left_hand_pose_st, self.right_hand_pose_st,
            energy_terms, step=None
        )
    
    def print_performance_stats(self):
        """Print performance and profiling statistics."""
        print("\n=== Performance Statistics ===")
        # self.profiler.print_stats()
        
        if HAS_MEMORY_PROFILER:
            try:
                memory_usage = memory_profiler.memory_usage(-1, interval=1)
                print(f"Peak memory usage: {max(memory_usage):.2f} MB")
            except (RuntimeError, OSError) as e:
                print(f"Memory profiling error: {e}")
        else:
            print("Memory profiling unavailable (memory_profiler not installed)")
    
    def run_full_experiment(self) -> EnergyTerms:
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")
        
        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_optimization()
        self.setup_logging()
        
        # Run optimization
        final_energy_terms = self.run_optimization()
        
        # Save results
        self.save_final_results(final_energy_terms)

        # Print statistics
        self.print_performance_stats()
        
        print(f"Experiment completed: {self.config.name}")
        return final_energy_terms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bimanual Grasp Generation')
    
    # Core experiment settings
    parser.add_argument('--name', default='test', type=str, help='Experiment name')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu', default="0", type=str)
    
    # Object and batch configuration
    parser.add_argument('--object_code_list', default=[
        'Cole_Hardware_Dishtowel_Multicolors',
        'Curver_Storage_Bin_Black_Small', 
        'Hasbro_Monopoly_Hotels_Game',
        'Breyer_Horse_Of_The_Year_2015',
        'Schleich_S_Bayala_Unicorn_70432'
    ], type=list, help='List of object codes to process')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per object')
    parser.add_argument('--num_iterations', default=10000, type=int, help='Number of optimization iterations')
    
    
    args = parser.parse_args()
    
    # Create config from args or use defaults
    if hasattr(args, 'config_file') and args.config_file:
        # Future: Load from config file
        config = ExperimentConfig()
        config.update_from_args(args)
    else:
        config = create_config_from_args(args)
    
    # Print configuration summary
    print("=== Experiment Configuration ===")
    print(f"Name: {config.name}")
    print(f"Objects: {len(config.object_code_list)} objects")
    print(f"Batch size: {config.model.batch_size} per object")
    print(f"Total batch: {config.total_batch_size}")
    print(f"iterations: {config.optimizer.num_iterations}")
    print(f"Energy weights: dis={config.energy.w_dis}, pen={config.energy.w_pen}, vew={config.energy.w_vew}")
    print(f"temperature: {config.optimizer.initial_temperature}")
    print(f"Langevin noise: {config.optimizer.langevin_noise_factor}")
    print("=" * 45)
    
    # Run experiment
    experiment = GraspExperiment(config)
    final_energy_terms = experiment.run_full_experiment()
