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
    def __init__(self, isforward, max_key, augmentation=False, class_label=None):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentation = augmentation
        self.class_label = class_label
        
        # 클래스별 augmentation 파라미터
        self.aug_params = {
            'acc4-brain': {
                'flip_prop': 0.3, 'rotate_prop': 0.3, 'rotate_range': (2, 5),
                'scale_prop': 0.3, 'scale_range': (0.95, 1.05),
                'shift_prop': 0.3, 'shift_range': (2, 5)
            },
            'acc4-knee': {
                'flip_prop': 0.6, 'rotate_prop': 0.5, 'rotate_range': (5, 12),
                'scale_prop': 0.5, 'scale_range': (0.9, 1.1),
                'shift_prop': 0.5, 'shift_range': (5, 12)
            },
            'acc8-brain': {
                'flip_prop': 0.2, 'rotate_prop': 0.2, 'rotate_range': (1, 3),
                'scale_prop': 0.2, 'scale_range': (0.98, 1.02),
                'shift_prop': 0.2, 'shift_range': (1, 3)
            },
            'acc8-knee': {
                'flip_prop': 0.4, 'rotate_prop': 0.3, 'rotate_range': (3, 8),
                'scale_prop': 0.3, 'scale_range': (0.95, 1.05),
                'shift_prop': 0.3, 'shift_range': (3, 8)
            }
        }
    
    def __call__(self, mask, input, target, attrs, fname, slice):
        
        if self.augmentation:
            kwargs = self.aug_params.get(self.class_label, {}) if self.class_label else {}
            input, target = spatial_augmentation(kspace=input * mask, image=target, **kwargs)

        # Prepare target and maximum
        if self.isforward:
            target = maximum = -1
        else:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
    
        kspace = to_tensor(input)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace.float(), target, maximum, fname, slice

