
from ast import Load
import h5py
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader

class SliceData(Dataset):
    """Unified dataset class - handles both standard and indexed loading"""
    
    def __init__(self, kspace_root, image_root=None, file_list=None, transform=None, input_key='kspace', target_key='image_label', forward=False):
        self.kspace_root = Path(kspace_root)
        self.image_root = Path(image_root) if image_root and not forward else None
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        
        self.examples = []
        self._build_examples(file_list)
    
    def _build_examples(self, file_list):
        """Build examples from index file or all files"""
        if file_list:   # use_moe, Load files from index file
            for input_path in file_list:
                if not os.path.exists(input_path):
                    print(f"File not found: {input_path}")
                    continue
                
                with h5py.File(input_path, "r") as hf:  
                    num_slices = hf[self.input_key].shape[0]
                
                for slice_ind in range(num_slices):
                    self.examples.append((input_path, slice_ind))

        else:   # no moe, Load all files in kspace root
            for input_name in os.listdir(self.kspace_root):
                input_path = self.kspace_root / input_name
                with h5py.File(input_path, "r") as hf:
                    num_slices = hf[self.input_key].shape[0]
                
                for slice_ind in range(num_slices):
                    self.examples.append((str(input_path), slice_ind))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        input_path, slice_ind = self.examples[index]

        # Load kspace
        with h5py.File(input_path, "r") as hf:
            input_data = hf[self.input_key][slice_ind]
            mask = np.array(hf["mask"])
        
        # Load image (if not forward)
        if self.forward:
            target = -1
            attrs = -1
        else:
            image_path = input_path.replace("kspace", "image")
            with h5py.File(image_path, "r") as hf:
                target = hf[self.target_key][slice_ind]
                attrs = dict(hf.attrs)
        fname = list(map(lambda x: os.path.basename(x), input_path))
        
        return self.transform(mask, input_data, target, attrs, fname, slice_ind)


def split_k_folds(file_list, num_folds=5):
    """Split file list into k folds for cross-validation"""
    np.random.shuffle(file_list)
    fold_size = len(file_list) // num_folds
    folds = [file_list[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    
    # Handle remaining files
    if len(file_list) % num_folds != 0:
        folds[-1].extend(file_list[num_folds * fold_size:])
    
    return folds

def create_data_loaders(train_data_path, val_data_path, args, index_file=None, shuffle=False, isforward=False, augmentation=False):
    """Create standard data loader"""
    max_key_ = args.max_key if not isforward else -1
    target_key_ = args.target_key if not isforward else -1

    train_kspace_root = Path(train_data_path) / "kspace"
    train_image_root = Path(train_data_path) / "image" if not isforward else None
    val_kspace_root = Path(val_data_path) / "kspace"
    val_image_root = Path(val_data_path) / "image" if not isforward else None

    if args.use_moe:
        with open(index_file, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
    else:
        file_list = [str(p) for p in train_kspace_root.glob('*.h5')]
        if not file_list:
            raise ValueError(f"No files found in {train_kspace_root}")
    folds = split_k_folds(file_list, args.num_folds)
    fold_num = args.num_folds if args.k_fold else 1 # first fold always used for validation when k-fold == False


    for val_fold in range(fold_num):
        train_folds = [f for i, f in enumerate(folds) if i != val_fold]
        train_folds = [item for sublist in train_folds for item in sublist]  # Flatten list

        train_dataset = SliceData(
            kspace_root=train_kspace_root,
            image_root=train_image_root,
            file_list=train_folds if args.use_moe else None,
            transform=DataTransform(isforward, max_key_, augmentation=augmentation),
            input_key=args.input_key,
            target_key=target_key_,
            forward=isforward
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=shuffle)

        val_dataset = SliceData(
            kspace_root=val_kspace_root,
            image_root=val_image_root,
            file_list=folds[val_fold] if args.use_moe else None,
            transform=DataTransform(isforward, max_key_, augmentation=augmentation),
            input_key=args.input_key,
            target_key=target_key_,
            forward=isforward
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=shuffle)

        yield train_loader, val_loader


def create_evaluation_loaders(data_path, args, isforward=False):
    """Create evaluation data loader"""
    max_key_ = args.max_key if not isforward else -1
    target_key_ = args.target_key if not isforward else -1

    kspace_root = Path(data_path) / "kspace"
    image_root = Path(data_path) / "image" if not isforward else None

    dataset = SliceData(
        kspace_root=kspace_root,
        image_root=image_root,
        file_list=None,  # Load all files
        transform=DataTransform(isforward, max_key_, augmentation=False),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward
    )
    
    return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

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
