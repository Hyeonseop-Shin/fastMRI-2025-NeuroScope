#!/usr/bin/env python3
"""
Visualization script for comparing reconstructed images with ground truth.
Automatically saves images to visual/<model>/checkpoints/<checkpoint>/<image_index>_slice<slice_index>.png

Usage:
    python visualize_reconstruction.py --checkpoint results/<model>/checkpoints/<checkpoint.pt> --image-index <idx>

Example:
    python visualize_reconstruction.py --checkpoint results/fivarnet_f8_i2_attn0_c32_s8_epoch4_fold5_seed2025_acc4-brain/checkpoints/best_model.pt --image-index 0
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils.model.FIVarNet import FIVarNet
from utils.model.VarNet import VarNet
from utils.data.transforms import DataTransform
from utils.common.loss_function import SSIMLoss


class AnatomicalSSIM:
    """Anatomical SSIM calculator matching evaluation strategy"""
    
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        self.win_size = win_size
        self.k1 = k1
        self.k2 = k2
    
    def calculate_ssim(self, prediction, target, anatomy_type, device):
        """Calculate anatomical SSIM matching eval.py implementation"""
        # Create SSIM calculator on the correct device
        ssim_calculator = SSIMLoss(win_size=self.win_size, k1=self.k1, k2=self.k2).to(device)
        
        # Create anatomical mask
        mask = np.zeros(target.shape)
        if anatomy_type == 'knee':
            mask[target > 2e-5] = 1
        elif anatomy_type == 'brain':
            mask[target > 5e-5] = 1
        
        # Apply morphological operations (matching eval.py)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=15)
        mask = cv2.erode(mask, kernel, iterations=14)
        
        # Convert to tensors
        target_tensor = torch.from_numpy(target).to(device)
        prediction_tensor = torch.from_numpy(prediction).to(device)
        mask_tensor = torch.from_numpy(mask).to(device).type(torch.float)
        
        # Apply mask
        target_masked = target_tensor * mask_tensor
        prediction_masked = prediction_tensor * mask_tensor
        
        # Calculate data range on masked target
        data_range = target_masked.max()
        
        # Calculate SSIM (return actual SSIM, not loss)
        ssim_loss = ssim_calculator(prediction_masked.unsqueeze(0), target_masked.unsqueeze(0), data_range.unsqueeze(0))
        ssim_value = 1 - ssim_loss.item()  # Convert loss back to SSIM
        
        return ssim_value, mask  # Return both SSIM and mask


class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self, checkpoint_path: str):
        self.config = self._parse_from_path(checkpoint_path)
    
    def _parse_from_path(self, checkpoint_path: str) -> Dict[str, Any]:
        """Parse model configuration from checkpoint path"""
        model_dir = Path(checkpoint_path).parent.parent.name
        parts = model_dir.split('_')
        
        config = {
            'model': 'fivarnet',
            'feature_cascades': 8,
            'image_cascades': 2,
            'attention_stride': 0,
            'chans': 32,
            'sens_chans': 8,
            'acc': 4,
            'anatomy': 'brain'
        }
        
        for part in parts:
            if part.startswith('f') and part[1:].isdigit():
                config['feature_cascades'] = int(part[1:])
            elif part.startswith('i') and part[1:].isdigit():
                config['image_cascades'] = int(part[1:])
            elif part.startswith('attn') and part[4:].isdigit():
                config['attention_stride'] = int(part[4:])
            elif part.startswith('c') and part[1:].isdigit():
                config['chans'] = int(part[1:])
            elif part.startswith('s') and part[1:].isdigit():
                config['sens_chans'] = int(part[1:])
            elif part.startswith('acc') and len(part) > 3:
                acc_anatomy = part[3:]
                if '-' in acc_anatomy:
                    acc_part, anatomy_part = acc_anatomy.split('-', 1)
                    config['acc'] = int(acc_part)
                    config['anatomy'] = anatomy_part
                else:
                    config['acc'] = int(acc_anatomy)
        
        return config
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        return str(self.config)


class ModelLoader:
    """Handles model loading and configuration"""
    
    @staticmethod
    def load_model(checkpoint_path: str, device: torch.device):
        """Load model from checkpoint"""
        config = ModelConfig(checkpoint_path)
        
        # Build model
        if config['model'].lower() == 'fivarnet':
            model = FIVarNet(
                num_feature_cascades=config['feature_cascades'],
                num_image_cascades=config['image_cascades'],
                attn_stride=config['attention_stride'],
                chans=config['chans'],
                sens_chans=config['sens_chans'],
                acceleration=config['acc']
            )
        elif config['model'].lower() == 'varnet':
            model = VarNet(
                num_cascades=config['feature_cascades'],
                chans=config['chans'],
                sens_chans=config['sens_chans']
            )
        else:
            raise ValueError(f"Unknown model type: {config['model']}")
        
        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, config


class DataLoader:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def get_data_paths(config: ModelConfig):
        """Get leaderboard data paths based on acceleration"""
        acc_dir = f"acc{config['acc']}"
        base_path = f"/root/fastMRI/datasets/leaderboard/{acc_dir}"
        return f"{base_path}/kspace", f"{base_path}/image"
    
    @staticmethod
    def load_sample(kspace_path: str, image_path: str, image_index: int, config: ModelConfig):
        """Load a sample from the leaderboard dataset"""
        # Get filtered files based on anatomy
        kspace_files = sorted([f for f in os.listdir(kspace_path) if f.endswith('.h5')])
        
        # Filter by anatomy (brain_test or knee_test)
        anatomy_prefix = f"{config['anatomy']}_test"
        filtered_files = [f for f in kspace_files if anatomy_prefix in f]
        
        if not filtered_files:
            # Fallback to all files if filtering fails
            filtered_files = kspace_files
        
        if image_index >= len(filtered_files):
            raise ValueError(f"Image index {image_index} out of range. Available: 0-{len(filtered_files)-1}")
        
        filename = filtered_files[image_index]
        
        # Load data
        with h5py.File(os.path.join(kspace_path, filename), 'r') as f:
            kspace = torch.from_numpy(f['kspace'][:])
            mask = f['mask'][:]
        
        with h5py.File(os.path.join(image_path, filename), 'r') as f:
            target = torch.from_numpy(f['image_label'][:])
            attrs = dict(f.attrs)
        
        return kspace, target, mask, attrs, filename
    
    @staticmethod
    def prepare_slice(kspace, mask, target, attrs, slice_idx):
        """Prepare a single slice using DataTransform"""
        transform = DataTransform(isforward=False, max_key='max')
        
        kspace_slice = kspace[slice_idx].numpy()
        target_slice = target[slice_idx].numpy()
        
        mask_t, kspace_t, target_t, maximum, _, _ = transform(
            mask, kspace_slice, target_slice, attrs, 'dummy', slice_idx
        )
        
        return mask_t, kspace_t, target_t


class ModelInference:
    """Handles model inference"""
    
    @staticmethod
    def reconstruct(model, mask_t, kspace_t, device):
        """Reconstruct image using the model"""
        # Add batch dimension and move to device
        mask_batch = mask_t.unsqueeze(0).to(device)
        kspace_batch = kspace_t.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(kspace_batch, mask_batch, use_grad_ckpt=False)
            
            # Handle different output formats
            if torch.is_complex(output):
                output = torch.abs(output)
            elif output.dim() >= 4 and output.shape[-1] == 2:
                output = torch.sqrt(output[..., 0]**2 + output[..., 1]**2)
            
            return output.squeeze(0).cpu().numpy()


class Visualizer:
    """Handles visualization and saving"""
    
    @staticmethod
    def normalize_image(image: np.ndarray, percentile: float = 99) -> np.ndarray:
        """Normalize image for display"""
        vmax = np.percentile(image, percentile)
        return np.clip(image / vmax, 0, 1)
    
    @staticmethod
    def get_output_path(checkpoint_path: str, image_index: int, slice_idx: int) -> str:
        """Generate output path based on checkpoint path and indices"""
        path_parts = Path(checkpoint_path).parts
        
        # Find model directory
        results_idx = next(i for i, part in enumerate(path_parts) if part == 'results')
        model_name = path_parts[results_idx + 1]
        checkpoint_name = Path(checkpoint_path).stem
        
        # Create output directory
        output_dir = Path("visual") / model_name / "checkpoints" / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"image{image_index}_slice{slice_idx}.png"
        return str(output_dir / filename)
    
    @staticmethod
    def create_comparison(reconstruction, target, filename, slice_idx, output_path, anatomy_type, device):
        """Create and save comparison visualization with Anatomical SSIM and mask overlay"""
        # Normalize images
        recon_norm = Visualizer.normalize_image(reconstruction)
        target_norm = Visualizer.normalize_image(target)
        diff = np.abs(recon_norm - target_norm)
        
        # Calculate Anatomical SSIM and get mask
        ssim_calculator = AnatomicalSSIM()
        ssim_result = ssim_calculator.calculate_ssim(reconstruction, target, anatomy_type, device)
        
        # Handle both old and new return formats
        if isinstance(ssim_result, tuple):
            ssim_value, anatomical_mask = ssim_result
        else:
            ssim_value = ssim_result
            # Create mask manually if not returned
            anatomical_mask = np.zeros(target.shape)
            if anatomy_type == 'knee':
                anatomical_mask[target > 2e-5] = 1
            elif anatomy_type == 'brain':
                anatomical_mask[target > 5e-5] = 1
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            anatomical_mask = cv2.erode(anatomical_mask, kernel, iterations=1)
            anatomical_mask = cv2.dilate(anatomical_mask, kernel, iterations=15)
            anatomical_mask = cv2.erode(anatomical_mask, kernel, iterations=14)
        
        # Find contours of the anatomical mask
        contour_result = cv2.findContours((anatomical_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour_result) == 3:  # Older OpenCV versions
            _, contours, _ = contour_result
        else:  # Newer OpenCV versions
            contours, _ = contour_result
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Reconstruction with SSIM value and anatomical area outline
        im1 = axes[0].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Reconstruction\n{filename} - Slice {slice_idx}\nAnatomical SSIM: {ssim_value:.4f}')
        axes[0].axis('off')
        
        # Add yellow contour lines to reconstruction
        for contour in contours:
            if len(contour) > 2:  # Only draw if contour has enough points
                contour_points = contour.squeeze()
                if len(contour_points.shape) == 2:  # Valid contour
                    axes[0].plot(contour_points[:, 0], contour_points[:, 1], 'yellow', linewidth=1.5, alpha=0.8)
        
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Ground truth with anatomical area outline
        im2 = axes[1].imshow(target_norm, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Add yellow contour lines to ground truth
        for contour in contours:
            if len(contour) > 2:  # Only draw if contour has enough points
                contour_points = contour.squeeze()
                if len(contour_points.shape) == 2:  # Valid contour
                    axes[1].plot(contour_points[:, 0], contour_points[:, 1], 'yellow', linewidth=1.5, alpha=0.8)
        
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference
        im3 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path} (Anatomical SSIM: {ssim_value:.4f})")
        plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize reconstructed images vs ground truth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image-index', type=int, default=0,
                       help='Index of the image to visualize')
    parser.add_argument('--slice-index', type=int, default=None,
                       help='Slice index to visualize (default: middle slice)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    return parser.parse_args()
def main():
    """Main function"""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Load model
        print(f"Loading model from: {args.checkpoint}")
        model, config = ModelLoader.load_model(args.checkpoint, device)
        print(f"Model configuration: {config}")
        
        # Load sample data
        kspace_path, image_path = DataLoader.get_data_paths(config)
        print(f"Loading image index: {args.image_index}")
        print(f"Using leaderboard data from: {kspace_path}")
        
        kspace, target, mask, attrs, filename = DataLoader.load_sample(
            kspace_path, image_path, args.image_index, config
        )
        print(f"Loaded file: {filename}")
        print(f"Kspace shape: {kspace.shape}, Target shape: {target.shape}")
        
        # Select slice
        num_slices = target.shape[0]
        slice_idx = args.slice_index if args.slice_index is not None else num_slices // 2
        
        if slice_idx >= num_slices:
            raise ValueError(f"Slice index {slice_idx} out of range. Available: 0-{num_slices-1}")
        
        print(f"Using slice index: {slice_idx} (out of {num_slices} slices)")
        
        # Prepare slice data
        print("Preparing slice data...")
        mask_t, kspace_t, target_t = DataLoader.prepare_slice(kspace, mask, target, attrs, slice_idx)
        
        # Reconstruct image
        print("Reconstructing image...")
        reconstruction = ModelInference.reconstruct(model, mask_t, kspace_t, device)
        target_slice = target[slice_idx].numpy()
        
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Target slice shape: {target_slice.shape}")
        
        # Generate output path and visualize
        output_path = Visualizer.get_output_path(args.checkpoint, args.image_index, slice_idx)
        print("Creating visualization...")
        Visualizer.create_comparison(reconstruction, target_slice, filename, slice_idx, output_path, config['anatomy'], device)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
