
import numpy as np
import matplotlib.pyplot as plt
import h5py


kspace_path = "D://Data/train/kspace/brain_acc4_1.h5"
image_path = "D://Data/train/image/brain_acc4_1.h5"

with h5py.File(kspace_path, "r") as f:
    print(f.keys())
    kspace_data = f["kspace_input"][:]
    kspace_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kspace_data)))

with h5py.File(image_path, "r") as f:
    print(f.keys())
    image_data = f["image_input"][:]