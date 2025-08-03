import torch
import numpy as np
from utils.data.augmentation import spatial_augmentation

def to_tensor(data):
    """Convert numpy array to torch tensor"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        return torch.tensor(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentation=False, anatomy_type='knee'):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentation = augmentation
        self.anatomy_type = anatomy_type
        
        self.aug_params = {
            'flip_prop': 0.0, 
            'rotate_prop': 0.0, 
            'rotate_range': (3, 8),
            'scale_prop': 0.0, 
            'scale_range': (0.95, 1.05),
            'shift_prop': 0.0, 
            'shift_range': (3, 8),
            # New brightness/contrast parameters
            'weight_brightness': 1.0,
            'brightness_range': (0.5, 2.0),
            'weight_contrast': 1.0,
            'contrast_range': (0.5, 2.0),
            'anatomy_type': anatomy_type
        }
    
    def set_brightness_contrast_augmentation(self, weight_brightness=0.1, weight_contrast=0.1, 
                                           brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        """
        Configure brightness and contrast augmentation parameters.
        
        Args:
            weight_brightness: Probability of applying brightness augmentation (0.0 to 1.0)
            weight_contrast: Probability of applying contrast augmentation (0.0 to 1.0)
            brightness_range: Range for brightness factor (e.g., (0.8, 1.2))
            contrast_range: Range for contrast factor (e.g., (0.8, 1.2))
        """
        self.aug_params.update({
            'weight_brightness': weight_brightness,
            'brightness_range': brightness_range,
            'weight_contrast': weight_contrast,
            'contrast_range': contrast_range
        })
    
    def __call__(self, mask, input, target, attrs, fname, slice):
        
        if self.augmentation:
            input, target = spatial_augmentation(kspace=input, image=target, **self.aug_params)

        if self.isforward:
            target = maximum = -1
        else:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
    
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace.float(), target, maximum, fname, slice
