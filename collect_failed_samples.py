# Author: Tsung-Hung Hsieh
# Date: 2023-12-12
# Email: ifndef.012@gmail.com
# Description: Collect failed samples from ViT models and CLIP model (zero-shot)
# Python version: 3.11

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification
from torchvision import transforms as tv_transforms
from transformers import DefaultDataCollator
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
from huggingface_hub import notebook_login
import torch
from torch import nn
import re
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import pickle
import random

# CLIP model reference: https://huggingface.co/openai/clip-vit-base-patch32
# ViT-base model reference: https://huggingface.co/google/vit-base-patch16-224-in21k
# ViT-tiny model https://huggingface.co/WinKawaks/vit-tiny-patch16-224
# Food 101 dataset reference: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/



# List of our trained vit models
VIT_MODEL_LIST = [
    'thhsieh/vit_base_food101_finetune',
    'thhsieh/vit_tiny_food101_finetuned',
    'thhsieh/vit_tiny_scratch_food101',
    'thhsieh/vit_scratch_food101_resume_1'
]

# Pack the dataset into batches for vit models
def batchify_vit(dataset, image_processor, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        imgs = image_processor(images=batch['image'], return_tensors='pt')
        labels = torch.as_tensor(batch['label'])
        idxs = torch.arange(i, i + labels.size(0))
        yield idxs, imgs, labels

# Pack the dataset into batches for CLIP model
def batchify_clip(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        imgs = batch['image']
        labels = torch.as_tensor(batch['label'])
        idxs = torch.arange(i, i + labels.size(0))
        yield idxs, imgs, labels


def load_model(checkpoint):
    return AutoModelForImageClassification.from_pretrained(checkpoint)

# Collect failed samples from ViT models and CLIP model (zero-shot)
def collect_failed_samples(model_list: list[str], dataset, batch_size: int, device):
    failed = defaultdict(list)
    for checkpoint in model_list: # Iterate through all ViT models
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = load_model(checkpoint).to(device)
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(dataset)) as pbar:
                for idxs, imgs, labels in batchify_vit(dataset, image_processor, batch_size):
                    logits = model(**imgs.to(device)).logits.cpu()
                    failed[checkpoint].extend(idxs[torch.nonzero(logits.argmax(-1) != labels).flatten()].tolist())
                    pbar.update(logits.size(0))

    # Collect failed samples from CLIP model (zero-shot)
    checkpoint_clip = 'openai/clip-vit-base-patch32'
    texts = [n.replace('_', ' ') for n in dataset.features['label'].names]
    processor_clip = CLIPProcessor.from_pretrained(checkpoint_clip)
    model_clip = CLIPModel.from_pretrained(checkpoint_clip).to(device)
    model_clip.eval()

    with torch.no_grad():
        with tqdm(total=len(dataset)) as pbar:
            for idxs, imgs, labels in batchify_clip(dataset, 256):
                inputs = processor_clip(text=texts, images=imgs, return_tensors='pt', padding=True)
                logits = model_clip(**inputs.to(device)).logits_per_image.cpu()
                failed[checkpoint_clip].extend(idxs[torch.nonzero(logits.argmax(-1) != labels).flatten()].tolist())
                pbar.update(logits.size(0))

    return failed

if __name__ == '__main__':
    # Select gpu if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the dataset
    food101 = load_dataset('food101')

    # Evaluate on the validation set
    dataset_val = food101['validation']

    # Get the failed samples' indices
    failed = collect_failed_samples(VIT_MODEL_LIST, dataset_val, 512, device)

    # Convert the failed samples indices to set
    for key in failed.keys():
        failed[key] = set(failed[key])

    # Save the failed samples for later visualization
    with open('failed_samples.pkl', mode='wb') as f:
        pickle.dump(failed, f)
