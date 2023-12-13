# Author: Tsung-Hung Hsieh
# Date: 2023-12-12
# Email: ifndef.012@gmail.com
# Description: Train a ViT model on Food101 dataset
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
from argparse import ArgumentParser

# CLIP model reference: https://huggingface.co/openai/clip-vit-base-patch32
# ViT-base model reference: https://huggingface.co/google/vit-base-patch16-224-in21k
# ViT-tiny model https://huggingface.co/WinKawaks/vit-tiny-patch16-224
# Food 101 dataset reference: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

# Image classification reference: https://huggingface.co/docs/transformers/tasks/image_classification

def parse_args():
    """ Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='vit_base_finetune', help='Checkpoint directory')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k', help='Model to use')
    parser.add_argument('--from_scratch', action='store_true', help='Train from scratch')
    return parser.parse_args()


def get_compute_metrics_fn():
    """ Get the compute metrics function for Trainer.
    """
    accuracy = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    return compute_metrics

def get_train_transform(checkpoint: str):
    # Load the image processor
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    # Define the transforms
    _transforms = tv_transforms.Compose([
        tv_transforms.RandomAffine(degrees=10, shear=5),
        tv_transforms.RandomHorizontalFlip(p=0.5),
        tv_transforms.transforms.RandomResizedCrop(size=(image_processor.size['height'], image_processor.size['width'])),
        tv_transforms.transforms.ToTensor(),
        tv_transforms.transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    # Define the transform function
    def transform_data(data):
      data['pixel_values'] = [_transforms(img.convert('RGB')) for img in data.pop('image')]
      return data

    return transform_data


def get_val_transform(checkpoint: str):
    # Load the image processor
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    # Define the transforms
    _transforms = tv_transforms.Compose([
        tv_transforms.transforms.Resize(size=(image_processor.size['height'], image_processor.size['width'])),
        tv_transforms.transforms.ToTensor(),
        tv_transforms.transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    # Define the transform function
    def transform_data(data):
      data['pixel_values'] = [_transforms(img.convert('RGB')) for img in data.pop('image')]
      return data

    return transform_data

def load_untrained_model(checkpoint: str, num_labels: int):
    # Load the config
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    # Load the model from config, this will initialize the model with random weights
    model = AutoModelForImageClassification.from_config(config)
    return model

def load_pretrained_model(checkpoint: str, num_labels: int):
    # Load the model from checkpoint with pre-trained weights
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model

if __name__ == '__main__':
    args = parse_args()
    # Define the model loading function
    load_model = load_untrained_model if args.from_scratch else load_pretrained_model

    # Select gpu if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the dataset
    food101 = load_dataset('food101')

    # Define the training and validation dataset with transform
    dataset_train = food101['train'].with_transform(get_train_transform(args.model_name))
    dataset_val = food101['validation'].with_transform(get_val_transform(args.model_name))

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f'{args.checkpoint_dir}',
        remove_unused_columns=False, # Keep this
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=64,
        num_train_epochs=100,
        warmup_ratio=0.1,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        push_to_hub=False,
        save_total_limit=2,
        report_to='tensorboard',
        hub_strategy='checkpoint'
    )

    # load the model
    model = load_model(args.model_name, num_labels=dataset_train.features['label'].num_classes).to(device)

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=AutoImageProcessor.from_pretrained(args.model_name),
        compute_metrics=get_compute_metrics_fn(),
    )

    # Start training
    trainer.train()
