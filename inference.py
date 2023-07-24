import torch
import torchvision
import argparse
from spatial_apertheid_dataloader import SpatialApertheidDataset
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
from torch_poly_lr_decay import PolynomialLRDecay
from tqdm import tqdm
import wandb
from wandb_utils import init_or_resume_wandb_run, get_viz_img
import pathlib
import os
import numpy as np
from metric_utils import batch_intersectionAndUnion
from dlab_model import DLab
import torch.backends.cudnn as cudnn
import random
from unet import UNet
import cv2
import glob
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser('Spatial Apertheid Challenge', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data_root', default='data_renamed/', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--model_dir', type=str, default='/local/riemann1/home/msiam/spatial_apertheid_challenge/')
    parser.add_argument('--model_type', type=str, default='dlabv3')
    parser.add_argument('--out_dir', type=str, default='outputs/')
    return parser.parse_args()

def inverse_map_mask(mask):
    cls_mapping = { 0: 255, 1: 124, 2: 7, 3: 201}
    temp_mask = np.zeros(mask.shape)
    for cls in np.unique(mask):
        temp_mask[mask==cls] = cls_mapping[int(cls)]
    return temp_mask


def map_mask(mask):
    cls_mapping = {255: 0, 124: 1, 7: 2, 201: 3}
    temp_mask = np.zeros(mask.shape)
    for cls in np.unique(mask):
        temp_mask[mask==cls] = cls_mapping[int(cls)]
    return temp_mask

def score_fn(pred_filenames, gt_filenames):
    all_ious = None
    count_dict = {0:1, 1:1, 2:1, 3:1}
    for predfname, gtfname in tqdm(zip(pred_filenames, gt_filenames)):
        preds = Image.open(predfname)
        preds = preds.convert('L')
        labels = Image.open(gtfname)
        labels = labels.convert('L')
        preds = np.array(preds)
        labels = np.array(labels)
        labels = map_mask(labels)
        preds = map_mask(preds)
        unique_label = np.unique(labels)
        for label in unique_label:
            count_dict[label] += 1

        intersection, union, _ = batch_intersectionAndUnion(preds, labels, 4)
        intersection, union = intersection.cpu(), union.cpu()

        eps = 1e-10
        batch_iou = intersection / (union + eps)

        if all_ious is None:
            all_ious = batch_iou.sum(axis=0)
        else:
            all_ious += batch_iou.sum(axis=0)

    count_list = np.array(list(count_dict.values()))
    count_list[count_list>1] = count_list[count_list>1] - 1
    return all_ious / count_list

@torch.no_grad()
def test(model, test_loader, device, out_dir):
    model.eval()
    all_ious = None

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for idx, (images, labels, fnames) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        labels = labels.to(device)

        preds, _ = model(images)
        preds = preds.argmax(dim=1)
        preds = inverse_map_mask(preds.detach().cpu())

        for bidx in range(preds.shape[0]):
            cv2.imwrite(os.path.join(out_dir, '%s'%fnames[bidx].split('/')[-1]), np.array(preds[bidx]))

    pred_fnames = sorted(glob.glob(os.path.join(out_dir, '*.png')))
    gt_fnames = sorted(glob.glob(os.path.join('data_renamed/labels_4/tiles/*.png')))

    iou = score_fn(pred_fnames, gt_fnames)
    print('Miou score ', iou)

def infer(args):
    test_dataset = SpatialApertheidDataset(args.data_root, "test.txt", "test", img_size=256)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'dlabv3':
        model = DLab(num_classes=4)
    elif args.model_type == 'unet':
        model = UNet(n_classes=4)

    model.to(device)

    model_path = os.path.join(args.model_dir, 'best.pth')
    state_dict = torch.load(model_path)['weights']
    model.load_state_dict(state_dict)

    test(model, test_loader, device, args.out_dir)

if __name__ == "__main__":
    args = parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    infer(args)
