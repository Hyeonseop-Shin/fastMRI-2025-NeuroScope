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
    def __init__(self, isforward, max_key, augmentation=False):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentation = augmentation
        
        self.aug_params = {
            'flip_prop': 0.5, 
            'rotate_prop': 0.0, 
            'rotate_range': (3, 8),
            'scale_prop': 0.0, 
            'scale_range': (0.95, 1.05),
            'shift_prop': 0.0, 
            'shift_range': (3, 8)
        }
    
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
