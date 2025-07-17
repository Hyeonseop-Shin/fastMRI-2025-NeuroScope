import os

python_path = "/root/anaconda3/envs/mri/bin/python"
train_path = "/root/FastMRI_challenge/train.py"
cmd_file_path = "/root/FastMRI_challenge/train.sh"

model = 'fivarnet'
feature_cascades = 12
image_cascades = 0
use_attention = False

chans = 32
sen_chans = 8
seed = 0
epoch = 100
batch = 1
augmentation = False

accumulation_step = 4   # 원하는 값으로 설정
warmup_epochs = 10      # 첫 번째 warmup
anneal1 = 40            # 첫 번째 cosine annealing
warmup2 = 10            # 두 번째 warmup
anneal2 = 40            # 두 번째 cosine annealing
lr = 3e-4               # 첫 번째 warmup의 max lr
lr_min1 = 0.00005       # 첫 번째 annealing의 min lr
lr_max2 = 0.00015       # 두 번째 warmup의 max lr
lr_min2 = 0.0           # 두 번째 annealing의 min lr
scheduler = "cosine"    # 원하는 값으로 설정

model_name = f"{model}_f{feature_cascades}_i{image_cascades}{'_attn' if use_attention else ''}{'_augmentation' if augmentation else ''}_c{chans}_s{sen_chans}_e{epoch}_seed{seed}"

instruction_template = f"{python_path} {train_path} \
-b {batch} \
-e {epoch} \
--seed {seed} \
--chans {chans} \
--sens-chans {sen_chans} \
-f {feature_cascades} \
-i {image_cascades} \
-a {use_attention} \
-m {model} \
-n {model_name} \
--accumulation-step {accumulation_step} \
--lr {lr} \
--warmup-epochs {warmup_epochs} \
--anneal1 {anneal1} \
--warmup2 {warmup2} \
--anneal2 {anneal2} \
--lr-min1 {lr_min1} \
--lr-max2 {lr_max2} \
--lr-min2 {lr_min2} \
--scheduler {scheduler}"

with open(cmd_file_path, 'w') as f:
    f.write(instruction_template)


log_path = "/root/FastMRI_challenge/logs"
log_file_path = os.path.join(log_path, f"{model_name}.log")
print(f"{instruction_template} > {log_file_path} 2>&1 &")