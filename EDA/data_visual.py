import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kspace_dir = "/root/Data/train/kspace"
images_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(images_dir, exist_ok=True)

kspace_files = sorted([f for f in os.listdir(kspace_dir) if f.endswith('.h5')])

results = []
for fname in kspace_files:
    path = os.path.join(kspace_dir, fname)
    with h5py.File(path, 'r') as f:
        kspace = f['kspace'][:]
        if kspace.ndim == 4:
            kspace_slice = kspace[0]
        else:
            kspace_slice = kspace
        img = np.fft.ifft2(kspace_slice, axes=(-2, -1))
        img = np.abs(img)
        rss_img = np.sqrt(np.sum(img**2, axis=0))  # (height, width)

        # y=100~200 영역에서 x축별 평균값 구하기
        roi = rss_img[100:201, :]
        x_profile = np.mean(roi, axis=0)  # (width,)

        # x_profile에서 신호가 있는 x 범위 찾기 (예: 0.01 이상인 구간)
        threshold = 0.01 * np.max(x_profile)
        nonzero_indices = np.where(x_profile > threshold)[0]
        if len(nonzero_indices) > 0:
            x_min, x_max = nonzero_indices[0], nonzero_indices[-1]
        else:
            x_min, x_max = None, None
        results.append({"file": fname, "x_signal_max": x_max})

# DataFrame으로 저장 및 csv로 저장
df = pd.DataFrame(results)
print(df)
df.to_csv("x_signal_max_acc4_val.csv", index=False)
print("Saved to x_signal_max_acc4_val.csv")

# 히스토그램 시각화
plt.figure()
plt.hist(df["x_signal_max"].dropna(), bins=20, alpha=0.7)
plt.xlabel("x_signal_max")
plt.ylabel("Count")
plt.title("Histogram of x_signal_max (y=100~200, threshold=0.01*max)")
plt.tight_layout()
plt.savefig("x_signal_max_hist_val.png")
plt.close()