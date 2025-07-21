import os

root_path = "/root/FastMRI_challenge"

python_path = "/root/anaconda3/envs/mri/bin/python"
train_path = os.path.join(root_path, "train.py")
cmd_file_path = os.path.join(root_path, "train.sh")

# Training hyperparameters
batch = 1
epoch = 100
accumulation_step = 2

# Model hyperparameters
model = 'fivarnet'
feature_cascades = 12
image_cascades = 0
use_attention = False
chans = 32
sen_chans = 8

# MoE hyperparameters
use_moe = False
class_split_path = os.path.join(root_path, "class_indices")

# K-Fold hyperparameters
k_fold = False
num_folds = 5

# Data hyperparameters
data_root = "/root"
data_path_train = os.path.join(data_root, "data/train/")
data_path_val = os.path.join(data_root, "data/val/")
data_augmentation = False

# Scheduler hyperparameters
scheduler = "cosine"    
lr = 3e-4            
lr_min1 = 0.00005       
lr_max2 = 0.00015      
lr_min2 = 0.0         
warmup1 = 10   
anneal1 = 40           
warmup2 = 10           
anneal2 = 40           

# Saving hyperparameters
seed = 0
result_path = os.path.join(root_path, "results")
model_name = f"{model}_f{feature_cascades}_i{image_cascades}{'_attn' if use_attention else ''}{'_augmentation' if data_augmentation else ''}_c{chans}_s{sen_chans}_e{epoch}_seed{seed}"

instruction_template = f"{python_path} {train_path} \
-b {batch} \
-e {epoch} \
--accumulation-step {accumulation_step} \
--model {model} \
-f {feature_cascades} \
-i {image_cascades} \
-a {use_attention} \
--chans {chans} \
--sens-chans {sen_chans} \
--use-moe {use_moe} \
--class-split-path {class_split_path} \
--k-fold {k_fold} \
--num-folds {num_folds} \
--data-augmentation {data_augmentation} \
--data-path-train {data_path_train} \
--data-path-val {data_path_val} \
--scheduler {scheduler} \
-m {model} \
--lr {lr} \
--lr-min1 {lr_min1} \
--lr-max2 {lr_max2} \
--lr-min2 {lr_min2} \
--warmup1 {warmup1} \
--anneal1 {anneal1} \
--warmup2 {warmup2} \
--anneal2 {anneal2} \
--seed {seed} \
--result-path {result_path} \
-n {model_name} \
"

with open(cmd_file_path, 'w') as f:
    f.write(instruction_template)


log_path = "/root/FastMRI_challenge/logs"
log_file_path = os.path.join(log_path, f"{model_name}.log")
print(f"{instruction_template} > {log_file_path} 2>&1 &")