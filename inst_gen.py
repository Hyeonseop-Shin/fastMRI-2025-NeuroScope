import os

root_path = "/root/fastMRI/fastMRI-2025-NeuroScope"

python_path = "/venv/mri/bin/python"
train_path = os.path.join(root_path, "train.py")
cmd_file_path = os.path.join(root_path, "train.sh")

# Training hyperparameters
batch = 1
accumulation_step = 2
criterion = "AnatomicalSSIM"
# criterion = "AnatomicalSSIM_L1"
epoch = 5 # 실제로 돌아가는 에폭 수
retrain = False
retrain_epoch = 5
start_epoch = retrain_epoch if retrain else 0 # 처음이면 0, retrain이면 retrain_epoch과 똑같이 설정
acc_only_list = [4, 8]  # 0 for all acc
anatomy_only_list = [
    # 'brain', 
    'knee',
    # 'all',
    ]


# Model hyperparameters
model = 'fivarnet'
feature_cascades = 8
image_cascades = 0
attention_stride = 0
chans = 32
sen_chans = 8

# MoE hyperparameters
use_moe = True
class_split_path = os.path.join(root_path, "class_indices")

# K-Fold hyperparameters
k_fold = True
num_folds = 5

# Data hyperparameters
data_root = "/root/fastMRI/datasets"
data_path_train = os.path.join(data_root, "train/")
data_path_val = os.path.join(data_root, "val/")
data_augmentation = False

# Scheduler hyperparameters
# scheduler = "constant"
scheduler = "cosine"
lr = 3e-4
# lr = 3e-5
lr_min1 = 0.00005       
lr_max2 = 0.00015      
lr_min2 = 0.0         
warmup1 = 10   
anneal1 = 40           
warmup2 = 10           
anneal2 = 40           

# Saving hyperparameters
seed = 2025
result_path = os.path.join(root_path, "results")
# model_name = f"{model}_f{feature_cascades}_i{image_cascades}_attn{attention_stride}{'_augmentation' if data_augmentation else ''}_c{chans}_s{sen_chans}_epoch{epoch}_fold{num_folds}_seed{seed}"
model_name = f"{model}_f{feature_cascades}_i{image_cascades}_attn{attention_stride}_c{chans}_s{sen_chans}_epoch{epoch}_fold{num_folds}_seed{seed}"
model_name = f"{model}_f{feature_cascades}_i{image_cascades}_attn{attention_stride}_c{chans}_s{sen_chans}_no_image_cascade"

instruction_template = [
    f"{python_path} -u {train_path} \
    -g {gpu_num} \
    --acc-only {acc_only} \
    --anatomy-only {anatomy_only} \
    -b {batch} \
    --start-epoch {start_epoch} \
    -e {epoch} \
    --accumulation-step {accumulation_step} \
    --criterion {criterion} \
    --retrain {retrain} \
    --retrain-epoch {retrain_epoch} \
    --model {model} \
    -f {feature_cascades} \
    -i {image_cascades} \
    -a {attention_stride} \
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
    for gpu_num, acc_only in enumerate(acc_only_list)
    for anatomy_only in anatomy_only_list
    ]


with open(cmd_file_path, 'w') as f:
    for instruction in instruction_template:
        f.write(f"{instruction}\n")

log_path = "/root/fastMRI/fastMRI-2025-NeuroScope/logs"
log_path = os.path.join(root_path, "logs")
os.makedirs(log_path, exist_ok=True)

with open(cmd_file_path, 'w') as f:
    log_file_path = [os.path.join(log_path, f"{model_name}_acc{acc_only}-{anatomy_only}.log") 
                     for acc_only in acc_only_list
                     for anatomy_only in anatomy_only_list]
    for instruction, log_file in zip(instruction_template, log_file_path):
        f.write(f"nohup {instruction}> {log_file} 2>&1 &\n")