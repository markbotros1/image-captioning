# image-captioning

A multi-modal generative AI model that generates captions for images

![alt text](https://github.com/markbotros1/image-captioning/blob/main/resources/example.png)

## Summary

- Pretrained EfficientNet v2 Convolutional Neural Net backbone
- Encoder/Decoder Transformer to translate CNN image features to text
- Trained on the Common Objects in Context (COCO) dataset—[check it out](https://cocodataset.org/#download)
- Model based on [End-to-End Transformer Based Model for Image Captioning](https://arxiv.org/abs/2203.15350)

## Project Organization

    ├── model              <- Model architecture
    │     ├── blocks.py
    │     ├── layers.py
    │     ├── model.py
    │
    ├── tokenizer          <- Custom tokenizer config
    │     ├── ...
    │
    ├── config.yaml        <- Model configuration and training setup
    |
    ├── dataset.py         <- Functions and class for building dataset for model training
    |
    ├── train.py           <- Model training
    |
    ├── inference.py       <- Model inference/testing
    │
    ├── requirements.txt   <- Requirements file for reproducing the model environment
