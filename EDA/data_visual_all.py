import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

kspace_dir = "/root/Data/leaderboard/acc4/kspace"
images_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(images_dir, exist_ok=True)

kspace_files = sorted([f for f in os.listdir(kspace_dir) if f.endswith('.h5')])

for fname in kspace_files:
    path = os.path.join(kspace_dir, fname)
    with h5py.File(path, 'r') as f:
        kspace = f['kspace'][:]
        # (slice, coil, height, width) or (coil, height, width)
        if kspace.ndim == 4:
            kspace_slice = kspace[0]  # 첫 슬라이스
        else:
            kspace_slice = kspace
        img = np.fft.ifft2(kspace_slice, axes=(-2, -1))
        img = np.abs(img)
        rss_img = np.sqrt(np.sum(img**2, axis=0))  # (height, width)

        plt.figure()
        plt.imshow(rss_img, cmap='gray')
        plt.axhline(200, color='red', linestyle='--')  # y=200 위치에 빨간 선
        plt.title(f"{fname} RSS of IFFT (y=200)")
        plt.colorbar()
        plt.savefig(os.path.join(images_dir, f"{fname}_rss_ifft_y200.png"))
        plt.close()