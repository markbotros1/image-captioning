import os
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset
from model.model import ImageCaptionTransformer, Config, load_config

#--------------------------- Training Parameters -------------------------------
# Data filepaths
tr_images_fp = 'data/train2014'
tr_annotations_fp = 'data/annotations/captions_train2014.json'
val_images_fp = 'data/val2014'
val_annotations_fp = 'data/annotations/captions_val2014.json'
# Model initialization
ckpt_path = 'checkpoints/custom_vocab_ckpt.pt'
init_from = 'resume'
iter_num = 0
best_val_loss = 1e9
# WandB logging
wandb_log = True
wandb_project = 'ImageCaptioning'
# Training loop
gradient_accumulation_steps = 8
eval_interval = 100
log_interval = 1
eval_iters = 10
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
compile = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ------------------------------------------------------------------------------


#------------------------ Create Datastets/DataLoaders -------------------------
# Create Datasets
tr_data = ImageCaptionDataset(tr_images_fp, tr_annotations_fp)
val_data = ImageCaptionDataset(val_images_fp, val_annotations_fp)
# Model/DataLoader config
tokenizer = tr_data.get_tokenizer()
cnf = load_config('config.yaml')
cnf['vocab_size'] = tokenizer.vocab_size
cnf['padding_idx'] = tokenizer.pad_token_id
cnf['start_idx'] = tokenizer.bos_token_id
config = Config(**cnf)
# Get DataLoaders
tr_dataloader = DataLoader(tr_data, batch_size=config.batch_size)
val_dataloader = DataLoader(val_data, batch_size=config.batch_size)
# ------------------------------------------------------------------------------


#-------------------------- Create and Compile Model ---------------------------
# Create Model
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model = ImageCaptionTransformer(config)
elif init_from == 'resume':
    print(f"Resuming training from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    checkpoint_model_args = checkpoint['model_args']
    model = ImageCaptionTransformer(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.freeze_cnn()
model.to(config.device)
print(f'Num. Total Parameters: {model.get_num_params(trainable=False)/1e6:.2f}M')
print(f'Num. Trainable Parameters: {model.get_num_params()/1e6:.2f}M')
# Compile Model
if compile:
    print('COMPILING MODEL')
    unoptimized_model = model
    model = torch.compile(model)
#-------------------------------------------------------------------------------


#----------------------- Configure and Create Optimizer ------------------------
# AdamW optimizer
learning_rate =1e-6
max_iters = len(tr_data) / (config.batch_size * gradient_accumulation_steps)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# Learning rate decay settings
decay_lr = True
warmup_iters = 50
lr_decay_iters = 500
min_lr = 1e-7
print('CREATING AND CONFIGURING OPTIMIZER')
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), config.device)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
#-------------------------------------------------------------------------------


#-------------------------- Training Helper Functions --------------------------
d = torch.device(config.device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        _, imgs, capts, tgts, masks = next(iter(val_dataloader))
        logits = model(imgs.to(d), capts.to(d), masks.to(d))
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        tgts = tgts.view(B*T)
        loss = F.cross_entropy(logits, tgts)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
#-------------------------------------------------------------------------------


# Logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, config=config)


################################ Training loop ################################
###############################################################################
print(f'TRAINING: max iters: {int(max_iters)}, eval interval: {eval_interval}')
_, imgs, capts, tgts, masks = next(iter(tr_dataloader)) # first batch
t0 = time.time()
while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: val loss {losses:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val loss": losses
            })
        if losses < best_val_loss or always_save_checkpoint:
            best_val_loss = losses
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': cnf,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, w/ gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        logits = model(imgs.to(d), capts.to(d), masks.to(d))
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        tgts = tgts.view(B*T)
        loss = F.cross_entropy(logits, tgts)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        _, imgs, capts, tgts, masks = next(iter(tr_dataloader))
        
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f'iter {iter_num}: loss {lossf:.4f}, time {dt:.2f}s')
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train loss": lossf
            })
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
################################################################################
################################################################################
