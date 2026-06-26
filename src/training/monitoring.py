import torch
import numpy as np
import os
import shutil
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import defaultdict
from datetime import datetime

# Force matplotlib to use non-interactive backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io


class TrainingMonitor:
    """
    Comprehensive monitoring class for training using TensorBoard.
    Tracks losses, metrics, learning rates, gradients, and visualizations.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment", flush_sec: int = 5, 
                 hyperparams: Optional[Dict[str, Any]] = None, remove_existing: bool = True):
        """
        Initialize the training monitor.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            experiment_name: Name of the experiment
            flush_sec: How often to flush TensorBoard logs (seconds)
            hyperparams: Dictionary of hyperparameters to include in log directory name
            remove_existing: Whether to remove existing log directory if it exists
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M")
        
        # Create hyperparameter string for directory name
        if hyperparams:
            # Create a readable string from key hyperparameters
            key_params = []
            for key, value in hyperparams.items():
                if isinstance(value, (int, float)):
                    key_params.append(f"{key}_{value}")
                elif isinstance(value, str):
                    key_params.append(f"{key}_{value}")
                elif isinstance(value, (list, tuple)):
                    key_params.append(f"{key}_{'_'.join(map(str, value))}")
            
            param_str = "_".join(key_params)  # Limit to first 5 parameters to avoid too long names
            log_dir_name = f"{timestamp}_{param_str}"
        else:
            log_dir_name = f"{timestamp}_{experiment_name}"
        
        self.log_dir = Path(log_dir) / log_dir_name
        self.experiment_name = experiment_name
        self.flush_sec = flush_sec
        self.hyperparams = hyperparams
        
        # Remove existing log directory if it exists and remove_existing is True
        if remove_existing and self.log_dir.exists():
            print(f"Removing existing log directory: {self.log_dir}")
            shutil.rmtree(self.log_dir)
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=experiment_name,
            flush_secs=1,  # auto-flush every 1 second
            max_queue=100  # larger queue size
        )
        
        # Track metrics
        self.metrics = defaultdict(list)
        self.step_count = 0
        self.epoch_count = 0
        self.global_step_count = 0  # Global step counter across all epochs
        
        # Track best metrics
        self.best_metrics = {}
        
        print(f"Training monitor initialized. Logs will be saved to: {self.log_dir}")
        
        # Log hyperparameters if provided
        if hyperparams:
            self.log_hyperparameters(hyperparams)
    
    def log_losses(self, losses: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        Log training losses to TensorBoard.
        
        Args:
            losses: Dictionary of loss names and values
            step: Current step (if None, uses internal counter)
            prefix: Prefix for loss names (e.g., "train/", "val/")
        """
        if step is None:
            step = self.step_count
        
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            
            full_name = f"{prefix}{loss_name}" if prefix else loss_name
            self.writer.add_scalar(full_name, loss_value, step)
            self.metrics[full_name].append(loss_value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        Log evaluation metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step (if None, uses internal counter)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step_count
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            
            full_name = f"{prefix}{metric_name}" if prefix else metric_name
            self.writer.add_scalar(full_name, metric_value, step)
            self.metrics[full_name].append(metric_value)
            
            # Track best metrics
            if full_name not in self.best_metrics:
                self.best_metrics[full_name] = metric_value
            else:
                # For metrics where higher is better (like accuracy)
                if "accuracy" in full_name.lower() or "precision" in full_name.lower() or "recall" in full_name.lower():
                    if metric_value > self.best_metrics[full_name]:
                        self.best_metrics[full_name] = metric_value
                # For metrics where lower is better (like loss)
                else:
                    if metric_value < self.best_metrics[full_name]:
                        self.best_metrics[full_name] = metric_value
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """
        Log learning rates for all parameter groups.
        
        Args:
            optimizer: PyTorch optimizer
            step: Current step (if None, uses internal counter)
        """
        if step is None:
            step = self.step_count
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
            self.metrics[f'learning_rate/group_{i}'].append(lr)
    
    def log_gradients(self, model: torch.nn.Module, step: Optional[int] = None, prefix: str = ""):
        """
        Log gradient norms for model parameters.
        
        Args:
            model: PyTorch model
            step: Current step (if None, uses internal counter)
            prefix: Prefix for gradient names
        """
        if step is None:
            step = self.step_count
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                full_name = f"{prefix}gradients/{name}" if prefix else f"gradients/{name}"
                self.writer.add_scalar(full_name, grad_norm, step)
                self.metrics[full_name].append(grad_norm)
    
    def log_parameter_histograms(self, model: torch.nn.Module, step: Optional[int] = None, prefix: str = ""):
        """
        Log parameter histograms to TensorBoard.
        
        Args:
            model: PyTorch model
            step: Current step (if None, uses internal counter)
            prefix: Prefix for parameter names
        """
        if step is None:
            step = self.step_count
        
        for name, param in model.named_parameters():
            full_name = f"{prefix}parameters/{name}" if prefix else f"parameters/{name}"
            self.writer.add_histogram(full_name, param.data, step)
    
    def log_trajectory(self, poses: np.ndarray, step: Optional[int] = None, name: str = "trajectory"):
        """
        Log trajectory visualization to TensorBoard.
        
        Args:
            poses: Array of poses (N, 6) where each pose is [x, y, z, qw, qx, qy, qz]
            step: Current step (if None, uses internal counter)
            name: Name for the trajectory plot
        """
        if step is None:
            step = self.step_count
        
        if len(poses) < 2:
            return
        
        # Extract 2D trajectory (x, y)
        x_coords = poses[:, 0]
        y_coords = poses[:, 1]
        
        # Create trajectory plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(x_coords[0], y_coords[0], c='green', s=100, label='Start', zorder=5)
        ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='End', zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{name} - Step {step}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Convert to TensorBoard
        self.writer.add_figure(f'trajectories/{name}', fig, step)
        plt.close(fig)
    
    def log_pose_comparison(self, gt_poses: np.ndarray, pred_poses: np.ndarray, 
                          step: Optional[int] = None, name: str = "pose_comparison"):
        """
        Log pose comparison to TensorBoard.
        """
        gt_xyz = gt_poses[:, :3]
        pred_xyz = pred_poses[:, :3]
        aligned_pred = self._align_trajectories(pred_xyz, gt_xyz)

        # compute x, y min/max
        all_x = np.concatenate([gt_xyz[:, 0], aligned_pred[:, 0]])
        all_y = np.concatenate([gt_xyz[:, 1], aligned_pred[:, 1]])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'g-', label='GT')
        ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], 'r-', label='Pred')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(name)
        ax.axis('equal')
        # fit to a square range
        ax.set_xlim(x_min - 50, x_max + 50)
        ax.set_ylim(y_min - 50, y_max + 50)

        self.writer.add_figure(f'pose_comparisons/{name}', fig, step)
        plt.close(fig)
    
    def log_multiple_pose_comparison(self, poses_dict: Dict[str, np.ndarray], 
                                   step: Optional[int] = None, name: str = "multiple_pose_comparison",
                                   epoch: Optional[int] = None, train_valid: str = "train"):
        """
        Log comparison between multiple pose trajectories.
        
        Args:
            poses_dict: Dictionary of pose names and arrays, e.g., {'gt': gt_poses, 'icp': icp_poses, 'pgo': pgo_poses}
            step: Current step (if None, uses internal counter)
            name: Name for the comparison plot
            epoch: Epoch number for separate figure creation
            train_valid: Whether this is "train" or "valid" data
        """
        if step is None:
            step = self.step_count
        
        # Check if we have enough poses to compare
        valid_poses = {k: v for k, v in poses_dict.items() if v is not None and len(v) > 1}
        if len(valid_poses) < 2:
            print(f"Warning: Need at least 2 valid pose arrays for comparison, got {len(valid_poses)}")
            return
        
        # Create comprehensive comparison plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors, line styles, and linewidths matching train_01_with_video.py
        pose_styles = {
            'gt': {'color': 'y', 'linestyle': '--', 'linewidth': 6, 'zorder': 1, 'label': 'Ground-Truth Pose'},
            'icp': {'color': 'r', 'linestyle': '-', 'linewidth': 4, 'zorder': 3, 'label': 'ICP Pose'},
            'pgo': {'color': 'b', 'linestyle': '-', 'linewidth': 3, 'zorder': 4, 'label': 'PGO Pose'},
            'label': {'color': 'g', 'linestyle': '--', 'linewidth': 3, 'zorder': 5, 'label': 'Label Pose'}
        }
        
        # Plot each trajectory with consistent styling
        for pose_name, poses in valid_poses.items():
            if poses is not None and len(poses) > 1:
                # Get style for this pose type
                style = pose_styles.get(pose_name, {
                    'color': 'gray', 'linestyle': '-', 'linewidth': 2, 'zorder': 6, 'label': f'{pose_name.upper()}'
                })
                
                # Extract x, y coordinates (assuming poses are [x, y, z, qw, qx, qy, qz])
                x_coords = poses[:, 0]
                y_coords = poses[:, 1]
                
                ax.plot(x_coords, y_coords, 
                       color=style['color'], linestyle=style['linestyle'], 
                       linewidth=style['linewidth'], label=style['label'], 
                       zorder=style['zorder'], alpha=0.8)
                
                # Mark start and end points
                ax.scatter(x_coords[0], y_coords[0], c=style['color'], s=100, marker='o', alpha=0.8, zorder=style['zorder'])
                ax.scatter(x_coords[-1], y_coords[-1], c=style['color'], s=100, marker='s', alpha=0.8, zorder=style['zorder'])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Set title based on whether epoch is provided
        if epoch is not None:
            ax.set_title(f'{name} - {train_valid.upper()} - Epoch {epoch:04d} - Step {step}')
        else:
            ax.set_title(f'{name} - {train_valid.upper()} - Step {step}')
            
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Convert to TensorBoard with appropriate naming based on train/valid
        if epoch is not None:
            # Create epoch-specific figure with train/valid separation
            plot_name = f'epoch_{epoch:04d}/{train_valid}/{name}'
            self.writer.add_figure(f'epoch_trajectories_{train_valid}/{name}/{epoch:04d}', fig, step)
        else:
            # Regular step-based figure with train/valid separation
            self.writer.add_figure(f'trajectories_{train_valid}/{name}', fig, step)
        
        plt.close(fig)
    
    def log_loss_breakdown(self, loss_components: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        Log detailed loss breakdown to TensorBoard.
        
        Args:
            loss_components: Dictionary of loss component names and values
            step: Current step (if None, uses internal counter)
            prefix: Prefix for loss names
        """
        if step is None:
            step = self.step_count
        
        for component_name, component_value in loss_components.items():
            if isinstance(component_value, torch.Tensor):
                component_value = component_value.item()
            
            full_name = f"{prefix}loss_components/{component_name}" if prefix else f"loss_components/{component_name}"
            self.writer.add_scalar(full_name, component_value, step)
            self.metrics[full_name].append(component_value)
    
    def log_optimization_metrics(self, trans_loss: float, rot_loss: float, 
                               rot_cov_loss: float, vel_cov_loss: float, trans_cov_loss: float,
                               step: Optional[int] = None, prefix: str = ""):
        """
        Log specific optimization metrics for pose graph optimization.
        
        Args:
            trans_loss: Translation loss
            rot_loss: Rotation loss
            rot_cov_loss: Rotation covariance loss
            vel_cov_loss: Velocity covariance loss
            trans_cov_loss: Translation covariance loss
            step: Current step (if None, uses internal counter)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step_count
        else:
            # Update internal step counter if step is provided
            self.step_count = max(self.step_count, step)
        
        metrics = {
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'rot_cov_loss': rot_cov_loss,
            'vel_cov_loss': vel_cov_loss,
            'trans_cov_loss': trans_cov_loss
        }
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            
            full_name = f"{prefix}weight/optimization/{metric_name}" if prefix else f"weight/optimization/{metric_name}"
            self.writer.add_scalar(full_name, metric_value, step)
            self.metrics[full_name].append(metric_value)
    
    def log_relative_errors(self, rel_rot_err: float, rel_trans_err: float, 
                          step: Optional[int] = None, prefix: str = ""):
        """
        Log relative rotation and translation errors.
        
        Args:
            rel_rot_err: Relative rotation error (in degrees or radians)
            rel_trans_err: Relative translation error (in meters)
            step: Current step (if None, uses internal counter)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step_count
        else:
            # Update internal step counter if step is provided
            self.step_count = max(self.step_count, step)
        
        errors = {
            'relative_rotation_error': rel_rot_err,
            'relative_translation_error': rel_trans_err
        }
        
        for error_name, error_value in errors.items():
            if isinstance(error_value, torch.Tensor):
                error_value = error_value.item()
            
            full_name = f"{prefix}error/errors/{error_name}" if prefix else f"error/errors/{error_name}"
            self.writer.add_scalar(full_name, error_value, step)
            self.metrics[full_name].append(error_value)
    
    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Optional[Dict[str, float]] = None):
        """
        Log comprehensive epoch metrics including relative errors and losses.
        Creates epoch-specific directories for overlay comparison.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary (optional)
        """
        # Create epoch-specific directory
        epoch_dir = self.log_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a separate writer for this epoch
        epoch_writer = SummaryWriter(
            log_dir=str(epoch_dir),
            comment=f"epoch_{epoch:02d}",
            flush_secs=1
        )
        
        # Log training metrics to main writer (for proper epoch ordering)
        for metric_name, metric_value in train_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            
            # Log to main writer for proper epoch ordering
            self.writer.add_scalar(f"epoch_metrics/train/{metric_name}", metric_value, epoch)
            
            # Also log to epoch-specific directory for detailed view
            epoch_writer.add_scalar(f"train/{metric_name}", metric_value, epoch)
        
        # Log validation metrics to main writer (for proper epoch ordering)
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                
                # Log to main writer for proper epoch ordering
                self.writer.add_scalar(f"epoch_metrics/val/{metric_name}", metric_value, epoch)
                
                # Also log to epoch-specific directory for detailed view
                epoch_writer.add_scalar(f"val/{metric_name}", metric_value, epoch)
        
        # Close epoch-specific writer
        epoch_writer.close()
        
        # Log epoch number to main writer
        self.writer.add_scalar('epoch', epoch, epoch)
        self.epoch_count = epoch
    
    def log_epoch_losses(self, epoch: int, train_losses: Dict[str, float], 
                        val_losses: Optional[Dict[str, float]] = None):
        """
        Log epoch summary with all losses and errors.
        Creates epoch-specific directories for overlay comparison.
        
        Args:
            epoch: Current epoch number
            train_losses: Training losses for this epoch
            val_losses: Validation losses for this epoch (optional)
        """
        # Create epoch-specific directory
        epoch_dir = self.log_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a separate writer for this epoch
        epoch_writer = SummaryWriter(
            log_dir=str(epoch_dir),
            comment=f"epoch_{epoch:02d}",
            flush_secs=1
        )
        
        # Log training losses to main writer (for proper epoch ordering)
        for loss_name, loss_value in train_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            
            # Log to main writer for proper epoch ordering
            self.writer.add_scalar(f"epoch_losses/train/{loss_name}", loss_value, epoch)
            
            # Also log to epoch-specific directory for detailed view
            epoch_writer.add_scalar(f"train_loss/{loss_name}", loss_value, epoch)
        
        # Log validation losses to main writer (for proper epoch ordering)
        if val_losses:
            for loss_name, loss_value in val_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                
                # Log to main writer for proper epoch ordering
                self.writer.add_scalar(f"epoch_losses/val/{loss_name}", loss_value, epoch)
                
                # Also log to epoch-specific directory for detailed view
                epoch_writer.add_scalar(f"val_loss/{loss_name}", loss_value, epoch)
        
        # Close epoch-specific writer
        epoch_writer.close()
        
        # Log epoch number to main writer
        self.writer.add_scalar('epoch', epoch, epoch)
        self.epoch_count = epoch
    
    def log_step_detailed_metrics(self, step: int, 
                                pose_losses: Dict[str, float],
                                cov_losses: Dict[str, float], 
                                rel_errors: Dict[str, float],
                                prefix: str = ""):
        """
        Log detailed step-level metrics for pose loss, covariance loss, and relative errors.
        
        Args:
            step: Current step number (if None, uses global step counter)
            pose_losses: Dictionary with 'total', 'rot', 'rot', 'trans' pose losses
            cov_losses: Dictionary with 'total', 'rot', 'vel', 'trans' covariance losses
            rel_errors: Dictionary with 'rot', 'trans' relative errors
            prefix: Prefix for metric names (e.g., "train_", "val_")
        """
        if step is None:
            step = self.global_step_count
        else:
            self.global_step_count = max(self.global_step_count, step)
        
        # Log pose losses (loss)
        for loss_name, loss_value in pose_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            full_name = f"{prefix}loss/pose_loss/{loss_name}" if prefix else f"loss/pose_loss/{loss_name}"
            self.writer.add_scalar(full_name, loss_value, step)
            self.metrics[full_name].append(loss_value)
        
        # Log covariance losses (loss)
        for loss_name, loss_value in cov_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            full_name = f"{prefix}loss/cov_loss/{loss_name}" if prefix else f"loss/cov_loss/{loss_name}"
            self.writer.add_scalar(full_name, loss_value, step)
            self.metrics[full_name].append(loss_value)
        
        # Log relative errors (error)
        for error_name, error_value in rel_errors.items():
            if isinstance(error_value, torch.Tensor):
                error_value = error_value.item()
            full_name = f"{prefix}error/relative_error/{error_name}" if prefix else f"error/relative_error/{error_name}"
            self.writer.add_scalar(full_name, error_value, step)
            self.metrics[full_name].append(error_value)
    
    def log_step_to_epoch_directory(self, epoch: int, step: int,
                                  pose_losses: Dict[str, float],
                                  cov_losses: Dict[str, float], 
                                  rel_errors: Dict[str, float],
                                  is_validation: bool = False):
        """
        Log step-level metrics to epoch-specific directory for real-time monitoring.
        
        Args:
            epoch: Current epoch number
            step: Current step number
            pose_losses: Dictionary with pose losses
            cov_losses: Dictionary with covariance losses
            rel_errors: Dictionary with relative errors
            is_validation: Whether this is validation step
        """
        # Create epoch-specific directory
        epoch_dir = self.log_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a separate writer for this epoch (if not exists)
        if not hasattr(self, f'epoch_{epoch}_writer'):
            epoch_writer = SummaryWriter(
                log_dir=str(epoch_dir),
                comment=f"epoch_{epoch:02d}",
                flush_secs=1
            )
            setattr(self, f'epoch_{epoch}_writer', epoch_writer)
        
        epoch_writer = getattr(self, f'epoch_{epoch}_writer')
        
        # Determine prefix based on validation flag
        prefix = "val" if is_validation else "train"
        
        # Log pose losses
        for loss_name, loss_value in pose_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            epoch_writer.add_scalar(f"{prefix}/pose_loss/{loss_name}", loss_value, step)
        
        # Log covariance losses
        for loss_name, loss_value in cov_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            epoch_writer.add_scalar(f"{prefix}/cov_loss/{loss_name}", loss_value, step)
        
        # Log relative errors
        for error_name, error_value in rel_errors.items():
            if isinstance(error_value, torch.Tensor):
                error_value = error_value.item()
            epoch_writer.add_scalar(f"{prefix}/relative_error/{error_name}", error_value, step)
        
        # Force flush to ensure immediate writing
        epoch_writer.flush()
    
    def get_next_global_step(self, epoch: int, step_in_epoch: int, is_validation: bool = False) -> int:
        """
        Get the next global step number for logging.
        
        Args:
            epoch: Current epoch number
            step_in_epoch: Step number within the current epoch
            is_validation: Whether this is a validation step
            
        Returns:
            Global step number (since each epoch has its own directory, we can use step_in_epoch directly)
        """
        # Since each epoch has its own directory, we can use step_in_epoch directly
        # No need for epoch multiplication or validation offset
        return step_in_epoch
    
    def log_loss_breakdown_epoch(self, epoch: int, losses: Dict[str, float], prefix: str = ""):
        """
        Log detailed loss breakdown for each epoch.
        
        Args:
            epoch: Current epoch number
            losses: Dictionary of loss components
            prefix: Prefix for loss names
        """
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            
            full_name = f"{prefix}loss/losses/{loss_name}" if prefix else f"loss/losses/{loss_name}"
            self.writer.add_scalar(full_name, loss_value, epoch)
            self.metrics[full_name].append(loss_value)
    
    def log_epoch_summary(self, epoch: int, train_losses: Dict[str, float], 
                         val_losses: Optional[Dict[str, float]] = None,
                         train_metrics: Optional[Dict[str, float]] = None,
                         val_metrics: Optional[Dict[str, float]] = None):
        """
        Log epoch summary with all metrics.
        """
        # Log training losses
        self.log_losses(train_losses, step=epoch, prefix="loss/")
        # Log validation losses
        if val_losses:
            self.log_losses(val_losses, step=epoch, prefix="val_loss/")
        # Log training metrics (error)
        if train_metrics:
            self.log_metrics(metrics=train_metrics, step=epoch, prefix="error/")
        # Log validation metrics (val_error)
        if val_metrics:
            self.log_metrics(metrics=val_metrics, step=epoch, prefix="val_error/")
        self.writer.add_scalar('epoch', epoch, epoch)
        self.epoch_count = epoch
    
    def log_epoch_summary_statistics(self, epoch: int, 
                                   train_state_losses: Dict[str, float],
                                   train_cov_losses: Dict[str, float],
                                   train_rel_errors: Dict[str, float],
                                   train_abs_errors: Optional[Dict[str, float]] = None,
                                   val_state_losses: Optional[Dict[str, float]] = None,
                                   val_cov_losses: Optional[Dict[str, float]] = None,
                                   val_rel_errors: Optional[Dict[str, float]] = None,
                                   val_abs_errors: Optional[Dict[str, float]] = None):
        """
        Log epoch summary statistics to epoch_summary directory for overlay comparison.
        
        Args:
            epoch: Current epoch number
            train_state_losses: Training state losses (total, rot, trans)
            train_cov_losses: Training covariance losses (total, rot, vel, pos)
            train_rel_errors: Training relative errors (rot, trans)
            val_state_losses: Validation state losses (optional)
            val_cov_losses: Validation covariance losses (optional)
            val_rel_errors: Validation relative errors (optional)
        """
        # Create epoch_summary directory
        summary_dir = self.log_dir / "epoch_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a separate writer for epoch summary
        summary_writer = SummaryWriter(
            log_dir=str(summary_dir),
            comment="epoch_summary",
            flush_secs=1
        )
        
        # Log training state losses (mean)
        for loss_name, loss_value in train_state_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            summary_writer.add_scalar(f"train/state_loss/{loss_name}", loss_value, epoch)
        
        # Log training covariance losses (mean)
        for loss_name, loss_value in train_cov_losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            summary_writer.add_scalar(f"train/cov_loss/{loss_name}", loss_value, epoch)
        
        # Log training relative errors (RMSE)
        for error_name, error_value in train_rel_errors.items():
            if isinstance(error_value, torch.Tensor):
                error_value = error_value.item()
            summary_writer.add_scalar(f"train/relative_error/{error_name}", error_value, epoch)
        
        # Log training absolute errors
        if train_abs_errors:
            for error_name, error_value in train_abs_errors.items():
                if isinstance(error_value, torch.Tensor):
                    error_value = error_value.item()
                summary_writer.add_scalar(f"train/absolute_error/{error_name}", error_value, epoch)
        
        # Log validation metrics if provided
        if val_state_losses:
            for loss_name, loss_value in val_state_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                summary_writer.add_scalar(f"val/state_loss/{loss_name}", loss_value, epoch)
        
        if val_cov_losses:
            for loss_name, loss_value in val_cov_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                summary_writer.add_scalar(f"val/cov_loss/{loss_name}", loss_value, epoch)
        
        if val_rel_errors:
            for error_name, error_value in val_rel_errors.items():
                if isinstance(error_value, torch.Tensor):
                    error_value = error_value.item()
                summary_writer.add_scalar(f"val/relative_error/{error_name}", error_value, epoch)
        
        if val_abs_errors:
            for error_name, error_value in val_abs_errors.items():
                if isinstance(error_value, torch.Tensor):
                    error_value = error_value.item()
                summary_writer.add_scalar(f"val/absolute_error/{error_name}", error_value, epoch)
        
        # Close summary writer
        summary_writer.close()
    
    def log_step_summary(self, step: int, losses: Dict[str, float], 
                        metrics: Optional[Dict[str, float]] = None,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        model: Optional[torch.nn.Module] = None):
        """
        Log step summary with all metrics.
        
        Args:
            step: Current step number
            losses: Losses for this step
            metrics: Metrics for this step (optional)
            optimizer: Optimizer for logging learning rates (optional)
            model: Model for logging gradients and parameters (optional)
        """
        # Log losses
        self.log_losses(losses, step=step)
        
        # Log metrics
        if metrics:
            self.log_metrics(metrics, step=step)
        
        # Log learning rates
        if optimizer:
            self.log_learning_rate(optimizer, step=step)
        
        # Log gradients
        if model:
            self.log_gradients(model, step=step)
        
        self.step_count = step
    
    def log_model_graph(self, model: torch.nn.Module, dummy_input: torch.Tensor):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            dummy_input: Dummy input tensor for graph visualization
        """
        self.writer.add_graph(model, dummy_input, verbose=False)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text to TensorBoard.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Current step (if None, uses internal counter)
        """
        if step is None:
            step = self.step_count
        
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Convert all values to strings for logging
        text = "## Hyperparameters\n\n"
        for key, value in hyperparams.items():
            text += f"**{key}**: {value}\n\n"
        
        self.log_text("hyperparameters", text)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best metrics achieved so far.
        
        Returns:
            Dictionary of best metric values
        """
        return self.best_metrics.copy()
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Get the history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values
        """
        return self.metrics.get(metric_name, [])
    
    def flush(self):
        """Flush TensorBoard writer."""
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer and print summary."""
        print("\n" + "="*50)
        print("TRAINING MONITOR SUMMARY")
        print("="*50)
        
        if self.best_metrics:
            print("\nBest Metrics:")
            for metric_name, best_value in self.best_metrics.items():
                print(f"  {metric_name}: {best_value:.6f}")
        
        print(f"\nTotal steps logged: {self.step_count}")
        print(f"Total epochs logged: {self.epoch_count}")
        print(f"Log directory: {self.log_dir}")
        print("="*50)
        
        # Close all epoch-specific writers
        for attr_name in dir(self):
            if attr_name.startswith('epoch_') and attr_name.endswith('_writer'):
                epoch_writer = getattr(self, attr_name)
                if hasattr(epoch_writer, 'close'):
                    epoch_writer.close()
        
        self.writer.close()

    def log_pose_comparison(self, gt_poses: np.ndarray, pred_poses: np.ndarray, 
                            step: Optional[int] = None, name: str = "pose_comparison"):
        """
        Log pose comparison to TensorBoard.
        """
        gt_xyz = gt_poses[:, :3]
        pred_xyz = pred_poses[:, :3]
        aligned_pred = self._align_trajectories(pred_xyz, gt_xyz)

        # compute x, y min/max
        all_x = np.concatenate([gt_xyz[:, 0], aligned_pred[:, 0]])
        all_y = np.concatenate([gt_xyz[:, 1], aligned_pred[:, 1]])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'g-', label='GT')
        ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], 'r-', label='Pred')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(name)
        ax.axis('equal')
        # fit to a square range
        ax.set_xlim(x_min - 50, x_max + 50)
        ax.set_ylim(y_min - 50, y_max + 50)

        self.writer.add_figure(f'pose_comparisons/{name}', fig, step)
        plt.close(fig)
    
    def log_covariance_visualization(self, epoch: int, pgo_poses: np.ndarray, covariances: np.ndarray, 
                                    lo_poses: np.ndarray, sample_step: int = 50, 
                                    prefix: str = "", x_limit: tuple = (-150, 150), y_limit: tuple = (-150, 150)):
        """
        Log covariance visualization to TensorBoard.
        
        Args:
            epoch: Current epoch number
            pgo_poses: PGO poses (N, 6) [x, y, z, qw, qx, qy, qz] or (N, 7) [x, y, z, qw, qx, qy, qz]
            covariances: Covariance matrices (N, 9) or (N, 9, 9)
            lo_poses: LO poses (N, 6) for reference trajectory
            sample_step: Step interval for plotting ellipses
            prefix: Prefix for the plot name
            x_limit: X-axis limits
            y_limit: Y-axis limits
        """
        if len(pgo_poses) < 2 or len(covariances) < 2:
            return
        
        # Extract positions from poses (first 3 elements are always x, y, z)
        pgo_xyz = pgo_poses[:, :3]
        lo_xyz = lo_poses[:, :3]
        # Align PGO to LO (simplified alignment)
        aligned_pgo = self._align_trajectories(pgo_xyz, lo_xyz)
        
        # Process covariances
        if covariances.ndim == 2 and covariances.shape[1] >= 9:
            diag = covariances[:, :9]
        elif covariances.ndim == 3 and covariances.shape[1] == 9 and covariances.shape[2] == 9:
            diag = np.diagonal(covariances, axis1=1, axis2=2)
        else:
            print(f"Warning: Unexpected covariance shape {covariances.shape}, skipping visualization")
            return
        # Split into rot, vel, pos
        rot_cov = diag[:, :3]    # (N, 3)
        vel_cov = diag[:, 3:6]   # (N, 3)
        pos_cov = diag[:, 6:9]   # (N, 3)
        
        # Calculate adaptive scales (2x mean of each covariance type)
        # rot_scale = 5.0 * 1/np.mean(rot_cov)
        # vel_scale = 5.0 * 1/np.mean(vel_cov)
        # pos_scale = 5.0 * 1/np.mean(pos_cov)
        
        rot_scale = 1e3
        vel_scale = 1e3
        pos_scale = 1e4
        # Sample indices for ellipses
        n = min(len(aligned_pgo), len(rot_cov))
        idx = np.arange(0, n, sample_step)
        
        # Create subplots for each covariance type
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        
        cov_sets = [
            ('Rotational', rot_cov[idx], rot_scale, 'red'),
            ('Velocity', vel_cov[idx], vel_scale, 'blue'),
            ('Positional', pos_cov[idx], pos_scale, 'green')
        ]
        
        for ax, (title, cov_data, scale, color) in zip(axs, cov_sets):
            # Plot LO trajectory
            ax.plot(lo_xyz[:n, 0], lo_xyz[:n, 1], 
                    linestyle='--', linewidth=3, color='cyan', label='LO trajectory')
            
            # Plot aligned PGO trajectory
            ax.plot(aligned_pgo[:n, 0], aligned_pgo[:n, 1], 
                    '-', color='black', linewidth=1, label='PGO trajectory')
            
            # Plot covariance ellipses
            for i, (x, y) in enumerate(zip(aligned_pgo[idx, 0], aligned_pgo[idx, 1])):
                if i < len(cov_data):
                    c1, c2, _ = cov_data[i]
                    ellipse = Ellipse((x, y), width=scale * c1, height=scale * c2,
                                    angle=0, alpha=0.3, facecolor=color, edgecolor='none')
                    ax.add_patch(ellipse)
            
            ax.set_xlim(x_limit)
            ax.set_ylim(y_limit)
            ax.set_title(f"{title} Covariance (scale={scale:.2e})")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True)
        
        plt.tight_layout()
        
        # Log to TensorBoard
        plot_name = f"{prefix}covariance_visualization" if prefix else "covariance_visualization"
        self.writer.add_figure(plot_name, fig, epoch)
        
        plt.close(fig)

    def _align_trajectories(self, pgo_xyz: np.ndarray, lo_xyz: np.ndarray) -> np.ndarray:
        """
        Simple alignment of PGO trajectory to LO trajectory.
        
        Args:
            pgo_xyz: PGO positions (N, 3)
            lo_xyz: LO positions (N, 3)
        
        Returns:
            aligned_pgo: Aligned PGO positions (N, 3)
        """
        if len(pgo_xyz) < 2 or len(lo_xyz) < 2:
            return pgo_xyz
        
        # Simple start point alignment
        pgo_start = pgo_xyz[0]
        lo_start = lo_xyz[0]
        
        # Translation alignment
        translation = lo_start - pgo_start
        aligned_pgo = pgo_xyz + translation
        
        return aligned_pgo


class RealTimeMonitor:
    """
    Real-time monitoring for live training visualization.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "realtime_experiment"):
        """
        Initialize real-time monitor.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.monitor = TrainingMonitor(log_dir, experiment_name, flush_sec=10)
        self.start_time = time.time()
    
    def log_realtime_metrics(self, losses: Dict[str, float], step: int):
        """
        Log real-time metrics with timing information.
        
        Args:
            losses: Dictionary of losses
            step: Current step
        """
        # Add timing information
        elapsed_time = time.time() - self.start_time
        losses_with_time = losses.copy()
        losses_with_time['elapsed_time'] = elapsed_time
        losses_with_time['steps_per_second'] = step / elapsed_time if elapsed_time > 0 else 0
        
        self.monitor.log_losses(losses_with_time, step, prefix="realtime/")
        self.monitor.flush()
    
    def close(self):
        """Close the real-time monitor."""
        self.monitor.close()


# Utility functions for easy integration
def create_monitor(log_dir: str, experiment_name: str, hyperparams: Optional[Dict[str, Any]] = None, **kwargs) -> TrainingMonitor:
    """
    Create a training monitor with default settings.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        hyperparams: Dictionary of hyperparameters to include in log directory name
        **kwargs: Additional arguments for TrainingMonitor
        
    Returns:
        TrainingMonitor instance
    """
    return TrainingMonitor(log_dir, experiment_name, hyperparams=hyperparams, **kwargs)


def log_training_step(monitor: TrainingMonitor, step: int, losses: Dict[str, float], 
                     optimizer: torch.optim.Optimizer, model: torch.nn.Module):
    """
    Convenience function to log a complete training step.
    
    Args:
        monitor: TrainingMonitor instance
        step: Current step
        losses: Dictionary of losses
        optimizer: PyTorch optimizer
        model: PyTorch model
    """
    monitor.log_step_summary(step, losses, optimizer=optimizer, model=model)
    monitor.log_learning_rate(optimizer, step)
    monitor.log_gradients(model, step)
    monitor.flush()


# Additional utility functions for common monitoring tasks
def log_training_epoch(monitor: TrainingMonitor, epoch: int, 
                      train_losses: Dict[str, float],
                      val_losses: Optional[Dict[str, float]] = None,
                      train_metrics: Optional[Dict[str, float]] = None,
                      val_metrics: Optional[Dict[str, float]] = None):
    """
    Convenience function to log a complete training epoch.
    
    Args:
        monitor: TrainingMonitor instance
        epoch: Current epoch number
        train_losses: Training losses for this epoch
        val_losses: Validation losses for this epoch (optional)
        train_metrics: Training metrics for this epoch (optional)
        val_metrics: Validation metrics for this epoch (optional)
    """
    monitor.log_epoch_summary(epoch, train_losses, val_losses, train_metrics, val_metrics)
    monitor.flush()


def log_pose_metrics(monitor: TrainingMonitor, step: int,
                    pose_losses: Dict[str, float],
                    cov_losses: Dict[str, float],
                    rel_errors: Dict[str, float],
                    prefix: str = ""):
    """
    Convenience function to log pose-related metrics.
    
    Args:
        monitor: TrainingMonitor instance
        step: Current step number
        pose_losses: Dictionary with pose losses
        cov_losses: Dictionary with covariance losses
        rel_errors: Dictionary with relative errors
        prefix: Prefix for metric names
    """
    monitor.log_step_detailed_metrics(step, pose_losses, cov_losses, rel_errors, prefix)
    monitor.flush()


def log_trajectory_comparison(monitor: TrainingMonitor, step: int,
                             gt_poses: np.ndarray, pred_poses: np.ndarray,
                             name: str = "trajectory_comparison"):
    """
    Convenience function to log trajectory comparison.
    
    Args:
        monitor: TrainingMonitor instance
        step: Current step number
        gt_poses: Ground truth poses
        pred_poses: Predicted poses
        name: Name for the comparison
    """
    monitor.log_pose_comparison(gt_poses, pred_poses, step, name)
    monitor.flush()


def log_covariance_analysis(monitor: TrainingMonitor, epoch: int,
                           pgo_poses: np.ndarray, covariances: np.ndarray,
                           lo_poses: np.ndarray, **kwargs):
    """
    Convenience function to log covariance analysis.
    
    Args:
        monitor: TrainingMonitor instance
        epoch: Current epoch number
        pgo_poses: PGO poses
        covariances: Covariance matrices
        lo_poses: LO poses
        **kwargs: Additional arguments for covariance visualization
    """
    monitor.log_covariance_visualization(epoch, pgo_poses, covariances, lo_poses, **kwargs)
    monitor.flush()