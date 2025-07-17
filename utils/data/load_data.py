import h5py
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    """Unified dataset class - handles both standard and indexed loading"""
    
    def __init__(self, kspace_root, image_root=None, index_file=None, transform=None, input_key='kspace', target_key='image_label', forward=False):
        self.kspace_root = Path(kspace_root)
        self.image_root = Path(image_root) if image_root and not forward else None
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        
        self.examples = []
        self._build_examples(index_file)
    
    def _build_examples(self, index_file):
        """Build examples from index file or all files"""
        if index_file:
            # Index-based loading
            with open(index_file, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]
        else:
            # Load all files
            file_list = [f.name for f in self.kspace_root.iterdir() if f.suffix == '.h5']
        
        for fname in file_list:
            kspace_path = self.kspace_root / fname
            if not kspace_path.exists():
                continue
            
            with h5py.File(kspace_path, "r") as hf:
                num_slices = hf[self.input_key].shape[0]
            
            for slice_ind in range(num_slices):
                self.examples.append((fname, slice_ind))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        fname, slice_ind = self.examples[i]
        
        # Load kspace
        kspace_path = self.kspace_root / fname
        with h5py.File(kspace_path, "r") as hf:
            input_data = hf[self.input_key][slice_ind]
            mask = np.array(hf["mask"])
        
        # Load image (if not forward)
        if self.forward:
            target = -1
            attrs = -1
        else:
            image_path = self.image_root / fname
            with h5py.File(image_path, "r") as hf:
                target = hf[self.target_key][slice_ind]
                attrs = dict(hf.attrs)
        
        return self.transform(mask, input_data, target, attrs, fname, slice_ind)

def create_data_loaders(data_path, args, shuffle=False, isforward=False, augmentation=False):
    """Create standard data loader"""
    max_key_ = args.max_key if not isforward else -1
    target_key_ = args.target_key if not isforward else -1
    
    # For standard loading, assume data_path contains kspace/ and image/ folders
    kspace_root = Path(data_path) / "kspace"
    image_root = Path(data_path) / "image" if not isforward else None

    dataset = SliceData(
        kspace_root=kspace_root,
        image_root=image_root,
        transform=DataTransform(isforward, max_key_, augmentation=augmentation),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward
    )

    return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle)

def create_indexed_loader(kspace_root, image_root, index_file, args, shuffle=False, isforward=False, augmentation=False):
    """Create index-based data loader for MoE with class-specific augmentation"""
    max_key_ = args.max_key if not isforward else -1
    target_key_ = args.target_key if not isforward else -1
    
    # 인덱스 파일명에서 클래스 추출
    class_label = Path(index_file).stem  # 예: acc4-brain.txt -> acc4-brain

    dataset = SliceData(
        kspace_root=kspace_root,
        image_root=image_root,
        index_file=index_file,
        transform=DataTransform(
            isforward, 
            max_key_, 
            augmentation=augmentation,
            class_label=class_label  # 클래스 정보 전달
        ),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward
    )

    return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle)

def create_moe_data_loaders(kspace_root, image_root, class_indices_dict, args, shuffle=False, isforward=False, augmentation=False):
    """Create multiple data loaders for MoE training"""
    loaders = {}
    for class_label, index_file in class_indices_dict.items():
        loaders[class_label] = create_indexed_loader(
            kspace_root, image_root, index_file, args, shuffle, isforward, augmentation
        )
    return loaders

# Example usage
# if __name__ == '__main__':
    # Standard usage
    # loader = create_data_loaders("/root/Data/train", args)
    
    # MoE usage
    # class_indices = {
    #     'acc4-brain': '/root/Data/train/class_indices/acc4-brain.txt',
    #     'acc4-knee': '/root/Data/train/class_indices/acc4-knee.txt',
    #     'acc8-brain': '/root/Data/train/class_indices/acc8-brain.txt',
    #     'acc8-knee': '/root/Data/train/class_indices/acc8-knee.txt'
    # }
    # moe_loaders = create_moe_data_loaders(
    #     "/root/Data/train/kspace", 
    #     "/root/Data/train/image", 
    #     class_indices, 
    #     args
    # )
    # pass
