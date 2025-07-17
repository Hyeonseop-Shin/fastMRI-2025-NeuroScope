import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kspace_dir = "/root/Data/leaderboard/acc4/kspace"
kspace_files = sorted([f for f in os.listdir(kspace_dir) if f.endswith('.h5')])

def get_rss_roi_mean(h5_path, y_start=100, y_end=200, x_ranges=[(0, 50), (350, None)]):
    with h5py.File(h5_path, 'r') as f:
        kspace = f['kspace'][:]
        if kspace.ndim == 4:
            kspace_slice = kspace[0]  # (coil, height, width)
        else:
            kspace_slice = kspace
        img = np.fft.ifft2(kspace_slice, axes=(-2, -1))
        img = np.abs(img)
        rss_img = np.sqrt(np.sum(img**2, axis=0))  # (height, width)
        means = []
        for x_start, x_end in x_ranges:
            if x_end is None:
                roi = rss_img[y_start:y_end+1, x_start:]
            else:
                roi = rss_img[y_start:y_end+1, x_start:x_end+1]
            means.append(np.mean(roi))
        return means

data = []
for fname in kspace_files:
    path = os.path.join(kspace_dir, fname)
    roi_means = get_rss_roi_mean(path)
    data.append({
        "file": fname,
        "roi_mean_x0_50": roi_means[0],
        "roi_mean_x350_end": roi_means[1]
    })

df = pd.DataFrame(data)
print(df)

# CSV로 저장
df.to_csv("roi_means_acc4.csv", index=False)
print("Saved to roi_means_acc4.csv")

# 히스토그램 시각화
for feat in ["roi_mean_x0_50", "roi_mean_x350_end"]:
    plt.figure()
    plt.hist(df[feat], bins=20, alpha=0.7)
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.title(f"Histogram of {feat} (ROI y=100~200, x range)")
    plt.tight_layout()