import torch
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset
from model.model import ImageCaptionTransformer, Config, load_config
tr_images_fp = 'data/train2014'
tr_annotations_fp = 'data/annotations/captions_train2014.json'

tr_data = ImageCaptionDataset(tr_images_fp, tr_annotations_fp)
tokenizer = tr_data.get_tokenizer()
cnf = load_config('config.yaml')
cnf['vocab_size'] = tokenizer.vocab_size
cnf['padding_idx'] = tokenizer.pad_token_id
cnf['start_idx'] = tokenizer.bos_token_id
config = Config(**cnf)
tr_dataloader = DataLoader(tr_data, batch_size=config.batch_size)

ckpt_path = 'checkpoints/custom_vocab_ckpt.pt'
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
print(f'Num. Total Parameters: {model.get_num_params(trainable=False)/1e6:.1f}M')
print(f'Num. Trainable Parameters: {model.get_num_params()/1e6:.1f}M')

model = torch.compile(model)

fn, im, ca, tg, _ = next(iter(tr_dataloader))
new = model.generate_caption(im[0].unsqueeze(0).to(config.device), 100, max_tokens=20)
print(tokenizer.decode(new[0], skip_special_tokens=True))


