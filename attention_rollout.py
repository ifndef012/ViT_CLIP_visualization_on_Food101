# Author: Tsung-Hung Hsieh
# Date: 2023-12-12
# Email: ifndef.012@gmail.com
# Description: Use attention rollout to visualize the attention map of ViT models
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


# Convert the model name to a shorter alias for visualization
model_alias = {
    'thhsieh/vit_base_food101_finetune': 'vit_base_finetune',
    'thhsieh/vit_tiny_food101_finetuned': 'vit_tiny_finetune',
    'thhsieh/vit_tiny_scratch_food101': 'vit_tiny_scratch',
    'thhsieh/vit_scratch_food101_resume_1': 'vit_base_scratch'
}

# Reference https://jacobgil.github.io/deeplearning/vision-transformer-explainability
def attention_rollout(model, input_):
    model.config.output_attentions = True
    with torch.no_grad():
        output = model(input_)
        attention_maps = torch.concat(output.attentions, dim=0)
        num_attentions, num_heads, num_tokens, num_tokens = attention_maps.size()
        attention_maps_fused = torch.mean(attention_maps, dim=1)
        attention_maps_fused_aug = attention_maps_fused + torch.eye(num_tokens)
        attention_maps_fused_aug_normalized = attention_maps_fused_aug / attention_maps_fused_aug.sum(dim=-1, keepdim=True)

        rollout, *ms = attention_maps_fused_aug_normalized
        for m in ms:
            rollout = torch.matmul(m, rollout)
        patch_size = int((num_tokens - 1)**0.5)
        mask = rollout[0, 1:].reshape(patch_size, patch_size)
        mask_normalized = (mask - mask.min()) / (mask.max() - mask.min())
        return mask_normalized

# Put the mask on the image
def apply_mask(mask, img):
    h_img, w_img, c = img.shape
    mask = cv2.resize(mask, (w_img, h_img))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255.0 * mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255.0 * cam)


def load_model(checkpoint):
    return AutoModelForImageClassification.from_pretrained(checkpoint)

def post_process(img): # De-normalize the image tensor and swap the axes to fit opencv
    return img.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5


if __name__ == '__main__':
    # Select gpu if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the failed samples from previously saved pickle file
    with open('failed_samples.pkl', mode='rb') as f:
        failed = pickle.load(f)

    # Load the food 101 dataset
    food101 = load_dataset('food101')
    # We only need the validation set for visualization
    dataset = food101['validation']

    # Draw 3 samples for each model
    num_samples = 3

    # Load the CLIP model
    checkpoint_clip = 'openai/clip-vit-base-patch32'
    ref = failed.pop(checkpoint_clip)
    model_clip = CLIPModel.from_pretrained(checkpoint_clip).to(device)

    # Get the 101 class names for CLIP model prediction
    class_names = dataset.features['label'].names

    # Store the figures for later saving
    figures = {}

    # All the samples' indices
    idx_all = set(range(len(dataset)))

    # Iterate through all the failed samples
    for ckpt, target in failed.items():
        # Load the ViT model and their image processor
        image_processor = AutoImageProcessor.from_pretrained(ckpt)
        model = load_model(ckpt).to(device)
        model.eval()

        # Collect ViT correct & CLIP failed samples
        correct = random.sample(list((idx_all - target) & ref), num_samples)
        # Collect ViT failed & CLIP correct samples
        wrong = random.sample(list(target & (idx_all - ref)), num_samples)

        with torch.no_grad():
            # Create the figure for ViT model correct & CLIP model failed
            fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True, figsize=(10, 10))
            fig.suptitle(f'{model_alias[ckpt]} correct | clip_model failed')
            for ax in axes.flat:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            figures[f'{ckpt}/correct'] = fig

            # Iterate through the samples which ViT model correct and CLIP model failed
            for col, idx in enumerate(correct):
                # Get the sample
                data = dataset[idx]
                # Preprocess the image
                img = image_processor(images=data['image'], return_tensors='pt')['pixel_values']
                # Draw the attention map
                mask_target = attention_rollout(model, img)
                # Apply the attention map on the image
                cam_target = apply_mask(mask_target.numpy(), post_process(img).numpy())

                # Display the attention maps and the image
                axes[1, col].imshow(cam_target)
                axes[1, col].set_xlabel(f'{model_alias[ckpt]}')


                mask_ref = attention_rollout(model_clip.vision_model, img)
                cam_ref = apply_mask(mask_ref.numpy(), post_process(img).numpy())
                axes[2, col].imshow(cam_ref)
                axes[2, col].set_xlabel('clip_base')


                axes[0, col].imshow(post_process(img).numpy())
                axes[0, col].set_title(f'{class_names[data["label"]]}')
                axes[0, col].set_xlabel('origin image')

            # Create the figure for ViT model failed & CLIP model correct
            fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True, figsize=(10, 10))
            fig.suptitle(f'{model_alias[ckpt]} failed | clip_model correct')
            for ax in axes.flat:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            figures[f'{ckpt}/wrong'] = fig

            # Iterate through the samples which ViT model failed and CLIP model correct
            for col, idx in enumerate(wrong):
                # Get the sample
                data = dataset[idx]
                # Preprocess the image
                img = image_processor(images=data['image'], return_tensors='pt')['pixel_values']
                # Draw the attention map
                mask_target = attention_rollout(model, img)
                # Apply the attention map on the image
                cam_target = apply_mask(mask_target.numpy(), post_process(img).numpy())

                # Display the attention maps and the image
                axes[1, col].imshow(cam_target)
                axes[1, col].set_xlabel(f'{model_alias[ckpt]}')


                mask_ref = attention_rollout(model_clip.vision_model, img)
                cam_ref = apply_mask(mask_ref.numpy(), post_process(img).numpy())
                axes[2, col].imshow(cam_ref)
                axes[2, col].set_xlabel('clip_base')


                axes[0, col].imshow(post_process(img).numpy())
                axes[0, col].set_title(f'{class_names[data["label"]]}')
                axes[0, col].set_xlabel('origin image')

    # Save the figures
    for name, fig in figures.items():
        filename = name.replace('/', '_')
        fig.savefig(f'{filename}.png')
