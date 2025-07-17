import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# acc4, acc8 경로
acc4_dir = "/root/Data/leaderboard/acc4/kspace"
acc8_dir = "/root/Data/leaderboard/acc8/kspace"

def get_mask_ones(h5_path):
    with h5py.File(h5_path, 'r') as f:
        mask = f['mask'][:]
        return int(np.sum(mask))

data = []
for acc_dir, label in [(acc4_dir, "acc4"), (acc8_dir, "acc8")]:
    for fname in sorted(os.listdir(acc_dir)):
        if fname.endswith('.h5'):
            path = os.path.join(acc_dir, fname)
            ones = get_mask_ones(path)
            data.append({"file": fname, "acc_type": label, "mask_ones": ones})

df = pd.DataFrame(data)
print(df)

# mask 개수 분포 요약
print("\nSummary by acc_type:")
print(df.groupby("acc_type")["mask_ones"].describe())

# csv로 저장하고 싶으면 아래 주석 해제
# df.to_csv("acc_mask_summary.csv", index=False)

# 히스토그램 시각화
plt.figure()
for label in df['acc_type'].unique():
    subset = df[df['acc_type'] == label]
    plt.hist(subset['mask_ones'], bins=20, alpha=0.5, label=label)
plt.xlabel("mask_ones")
plt.ylabel("Count")
plt.title("Histogram of mask_ones by acc_type")
plt.legend()
plt.tight_layout()
plt.savefig("mask_ones_hist.png")
plt.close()