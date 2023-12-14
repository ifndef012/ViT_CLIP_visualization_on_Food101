# ViT_CLIP_visualization_on_Food101

## Training 

```sh
# E.g. 
python train.py --checkpoint_dir vit_base_finetune --model_name google/vit-base-patch16-224-in21k --from_scratch
```

## Collect Failed Samples

```sh
python collect_failed_samples.py
```

This will output a file named `failed_samples.pkl`, which is used by the next step.

## Visualization 

```sh 
python attention_rollout.py 
```

This will take the `failed_samples.pkl` as input and draw the attention maps.


