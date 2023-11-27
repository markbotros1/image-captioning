import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_V2_M_Weights
from transformers import AutoTokenizer

class ImageCaptionDataset(Dataset):
    def __init__(self, images_fp, annotations_fp, test=False):
        anns = COCO(annotations_fp)
        image_ids = anns.getImgIds()
        images = anns.loadImgs(image_ids)
        caption_ids = anns.getAnnIds()
        captions = anns.loadAnns(caption_ids) if not test else None

        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        self.preproc = weights.transforms()
        self.images_fp = images_fp

        self.data = self.get_images_captions(images, captions) \
            if not test else self.get_images(images)
        self.tokenizer = AutoTokenizer.from_pretrained('tokenizer/')
    
    def __getitem__(self, idx):
        x = self.data[idx]
        filename = x[1]
        caption = self.clean_caption(x[2]) if len(x) == 3 else None

        img = Image.open(os.path.join(self.images_fp, filename))
        img = self.preproc(img)

        if caption:
            enc = self.tokenizer.encode_plus(
                caption,
                max_length=100,
                padding='max_length',
                return_tensors='pt')
            cpt = enc['input_ids'][0]
            tgt = torch.cat((
                cpt, 
                torch.tensor([self.tokenizer.pad_token_id])), dim=0)[1:]
            mask = enc['attention_mask'][0]
            mask = mask.float().masked_fill(mask == 0, float('-inf'))
            return filename, img, cpt, tgt, mask
        return filename, img

    def __len__(self):
        return len(self.data)
    
    def clean_caption(self, caption):
        cp = caption.lower().strip()
        if cp[-1] == '.' or cp[-1] == '!':
            cp = cp[:-1]
        return f'{self.tokenizer.bos_token}{cp}{self.tokenizer.eos_token}'
    
    def get_images_captions(self, images, captions):
        i = {}
        for img in images:
            i[img['id']] = img['file_name']

        res = np.empty(len(captions), dtype=object)
        for c in range(len(captions)):
            cpt = captions[c]
            seq = cpt['caption']
            id = cpt['image_id']
            fn = i[id]
            res[c] = (id, fn, seq)
        return res
            
    def get_images(self, images):
        res = np.empty(len(images), dtype=object)
        for i in range(len(images)):
            img = images[i]
            id = img['id']
            fn = img['file_name']
            res[i] = (id, fn)
        return res
    
    def get_tokenizer(self):
        return self.tokenizer