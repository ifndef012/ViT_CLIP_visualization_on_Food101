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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='vit_base_finetune', help='Checkpoint directory')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k', help='Model to use')
    parser.add_argument('--from_scratch', action='store_true', help='Train from scratch')
    return parser.parse_args()


def get_compute_metrics_fn():
    accuracy = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    return compute_metrics

def get_train_transform(checkpoint: str):
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    _transforms = tv_transforms.Compose([
        tv_transforms.RandomAffine(degrees=10, shear=5),
        tv_transforms.RandomHorizontalFlip(p=0.5),
        tv_transforms.transforms.RandomResizedCrop(size=(image_processor.size['height'], image_processor.size['width'])),
        tv_transforms.transforms.ToTensor(),
        tv_transforms.transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    def transform_data(data):
      data['pixel_values'] = [_transforms(img.convert('RGB')) for img in data.pop('image')]
      return data

    return transform_data


def get_val_transform(checkpoint: str):
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    _transforms = tv_transforms.Compose([
        tv_transforms.transforms.Resize(size=(image_processor.size['height'], image_processor.size['width'])),
        tv_transforms.transforms.ToTensor(),
        tv_transforms.transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    def transform_data(data):
      data['pixel_values'] = [_transforms(img.convert('RGB')) for img in data.pop('image')]
      return data

    return transform_data

def load_untrained_model(checkpoint: str, num_labels: int):
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    model = AutoModelForImageClassification.from_config(config)
    return model

def load_pretrained_model(checkpoint: str, num_labels: int):
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model

if __name__ == '__main__':
    args = parse_args()
    load_model = load_untrained_model if args.from_scratch else load_pretrained_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    food101 = load_dataset('food101')

    dataset_train = food101['train'].with_transform(get_train_transform(args.model_name))
    dataset_val = food101['validation'].with_transform(get_val_transform(args.model_name))

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

    model = load_model(args.model_name, num_labels=dataset_train.features['label'].num_classes).to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=AutoImageProcessor.from_pretrained(args.model_name),
        compute_metrics=get_compute_metrics_fn(),
    )


    trainer.train()
