
import os
import h5py

train_path = "D://Data/train/image"
val_path = "D://Data/val/image"

train_slice = 0
val_slice = 0
train_slice = {
    "brain_acc4": 0,
    "brain_acc8": 0,
    "knee_acc4": 0,
    "knee_acc8": 0,
}
val_slice = {
    "brain_acc4": 0,
    "brain_acc8": 0,
    "knee_acc4": 0,
    "knee_acc8": 0,
}

for train_file in os.listdir(train_path):
    key = train_file.split("_")
    key = key[0] + "_" + key[1]
    if train_file.endswith(".h5"):
        with h5py.File(os.path.join(train_path, train_file), "r") as f:
            train_slice[key] += f["image_input"].shape[0]

for val_file in os.listdir(val_path):
    key = val_file.split("_")
    key = key[0] + "_" + key[1]
    if val_file.endswith(".h5"):
        with h5py.File(os.path.join(val_path, val_file), "r") as f:
            val_slice[key] += f["image_input"].shape[0]



print("Train slices:", train_slice)
print("Validation slices:", val_slice)