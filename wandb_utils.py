import pathlib
from typing import Optional, Dict, List
import wandb
import torch
import numpy as np
import random
import random
import cv2

def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             entity_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.
        Returns the config, if it's not None it will also update it first
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        print('Resuming from wandb path... ', wandb_id_file_path)
        resume_id = wandb_id_file_path.read_text()
        wandb.init(entity=entity_name,
                   project=project_name,
                   name=run_name,
                   resume=resume_id,
                   config=config)
                   # settings=wandb.Settings(start_method="thread"))
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(entity=entity_name, project=project_name, name=run_name, config=config)
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        wandb.config.update(config)

    return config

def denormalize(img, mean, scale):
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale).view(1, 1, -1) + torch.tensor(mean).view(1, 1, -1)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,::-1]*255, np.uint8)
    return img

def create_overlay(img, mask, pred, cat2color):
    mask = mask[0]
    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    for cat, color in cat2color.items():
        mask_color[mask==cat] = color

    pred = pred.argmax(dim=0)
    pred_color= np.zeros((mask.shape[0], mask.shape[1],3))
    for cat, color in cat2color.items():
        pred_color[pred==cat] = color

    collated = np.concatenate([img[:, :, ::-1], mask_color, pred_color], axis=1)
    return collated

def get_viz_img(images, labels, preds, fnames, cat2color):
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.detach().cpu()

    nimages = 2
    indices = np.arange(images.shape[0])
    selected_indices = np.random.choice(indices, nimages)

    images = images[selected_indices]
    labels = labels[selected_indices]
    preds = preds[selected_indices]
    fnames = [fnames[idx] for idx in selected_indices]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    final_collate = []
    for img_idx in range(nimages):
        img = denormalize(images[img_idx], mean, std)
        overlay_img = create_overlay(img, labels[img_idx], preds[img_idx], cat2color)
        fname = fnames[img_idx].split('/')[-1]
        overlay_img = cv2.putText(overlay_img, fname, org, font, fontScale, color, thickness, cv2.LINE_AA)
        final_collate.append(overlay_img)

    return np.concatenate(final_collate, axis=0)
