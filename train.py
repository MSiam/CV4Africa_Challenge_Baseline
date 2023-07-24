"""
Ro'ya CV4Africa Challenge Baseline
workshop website: https://ro-ya-cv4africa.github.io/homepage/event_workshop.html
challenge website: https://codalab.lisn.upsaclay.fr/competitions/14259
"""

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
import torch.backends.cudnn as cudnn
import random
from unet import UNet

def parse_args():
    parser = argparse.ArgumentParser('Spatial Apertheid Challenge', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-6, type=float)
    parser.add_argument('--poly_power', default=0.9, type=float)
    parser.add_argument('--end_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data_root', default='data_renamed/', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--val_every_epoch', default=1, type=int)
    parser.add_argument('--log_every_iteration', default=100, type=int)
    parser.add_argument('--vis_every_iteration', default=1000, type=int)
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--experiment_name', default='dlabv3_spatial_apertheid', type=str)
    parser.add_argument('--wandb_user', type=str, default='URUSER')
    parser.add_argument('--wandb_project', type=str, default='URPROJECT')
    parser.add_argument('--out_dir', type=str, default='/local/riemann1/home/msiam/spatial_apertheid_challenge/')
    parser.add_argument('--pretrained', type=str, default='none')
    parser.add_argument('--auxloss', action="store_true")
    parser.add_argument('--overfit_data', action='store_true')
    parser.add_argument('--model_type', type=str, default='dlabv3')
    return parser.parse_args()


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    all_ious = None
    all_intersection = None
    all_union = None
    all_gt = None

    for idx, (images, labels, fnames) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        labels = labels.to(device)

        one_hot_mask = torch.zeros((labels.shape[0], 4, *labels.shape[-2:]))
        one_hot_mask.scatter_(1, labels.cpu(), 1).long()

        preds, _ = model(images)

        intersection, union, _ = batch_intersectionAndUnion(preds, labels, 4)
        intersection, union = intersection.cpu(), union.cpu()

        eps = 1e-10
        batch_iou = intersection / (union + eps)

        if all_ious is None:
            all_intersection = intersection.sum(axis=0)
            all_union = union.sum(axis=0)
            all_ious = batch_iou.sum(axis=0)
            all_gt = one_hot_mask.sum(axis=[0,2,3])
        else:
            all_intersection += intersection.sum(axis=0)
            all_union += union.sum(axis=0)
            all_ious += batch_iou.sum(axis=0)
            all_gt += one_hot_mask.sum(axis=[0,2,3])

    print('Stats per cls: ', all_intersection,' :::: ' , all_union, ' :::: ', all_gt)
    return all_ious / len(val_loader.dataset), images, labels, preds, fnames

def train(args):
    if args.overfit_data:
        # Create dataloader
        train_dataset = SpatialApertheidDataset(args.data_root, "val.txt", "train", img_size=256, overfit=True)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        val_dataset = SpatialApertheidDataset(args.data_root, "val.txt", "val", img_size=256, overfit=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        # Create dataloader
        train_dataset = SpatialApertheidDataset(args.data_root, "train.txt", "train", img_size=256)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        val_dataset = SpatialApertheidDataset(args.data_root, "val.txt", "val", img_size=256)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'dlabv3':
        model = DLab(num_classes=4, pretrained=args.pretrained, auxloss=args.auxloss)
    elif args.model_type == 'unet':
        model = UNet(n_classes=4)

    model.to(device)

    # Loss weights per class
    weight = torch.tensor([0.8177, 0.1276, 0.0147, 0.0400]).to(device)
    #weight = torch.log(1/(1.02+weight))
    weight = 1.0/weight * 5
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    # Create optimizer and scheduler ##########TODO: Add auxiliary classifier
    if args.model_type == 'dlabv3':
        param_dict = [
                {
                    "params": model.classifier.parameters(),
                    "lr": args.lr
                },
                {
                    "params": model.backbone.parameters(),
                    "lr": args.lr_backbone,
                },
        ]
    else:
        param_dict = [
                {
                    "params": model.parameters(),
                    "lr": args.lr
                },
        ]

    optimizer = torch.optim.AdamW(param_dict, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs-1, end_learning_rate=args.end_lr, power=args.poly_power)

    output_path = os.path.join(args.out_dir, args.experiment_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.use_wandb:
        wandb_id_file_path = pathlib.Path(os.path.join(output_path, args.experiment_name + '_wandb.txt'))
        config = init_or_resume_wandb_run(wandb_id_file_path,
                                          entity_name=args.wandb_user,
                                          project_name=args.wandb_project,
                                          run_name=args.experiment_name,
                                          config=args)

    best_iou = 0
    cat2color = {cat: [key]*3 for key, cat in train_dataset.cls_mapping.items()}
    for epoch_idx in range(args.epochs):
        model.train()
        for idx, (images, labels, fnames) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outs, aux_outs = model(images)
            loss = criterion(outs, labels.squeeze())
            if aux_outs is not None:
                loss += criterion(aux_outs, labels.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if aux_outs is not None:
                outs = aux_outs

            if idx % args.log_every_iteration == 0:
                if args.use_wandb:
                    wandb_dict = {'loss': loss, 'lr': optimizer.param_groups[0]["lr"]}
                    if idx % args.vis_every_iteration == 0:
                        viz_img = get_viz_img(images, labels, outs, fnames, cat2color)
                        wandb_dict['viz_img'] = wandb.Image(viz_img)
                    wandb.log(wandb_dict)
                state_dict = {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args}
                torch.save(state_dict, os.path.join(output_path, "final.pth"))

        if epoch_idx % args.val_every_epoch == 0:
            miou, images, labels, preds, fnames = validate(model, val_loader, device)
            if args.use_wandb:
                wandb_dict = {'val_iou': miou.mean()}
                for idx, iou in enumerate(miou):
                    wandb_dict['val_iou_cls%s'%idx] = iou
                viz_img = get_viz_img(images, labels, preds, fnames, cat2color)
                wandb_dict['val_viz_img'] = wandb.Image(viz_img)
                wandb.log(wandb_dict)

            if miou.mean() > best_iou:
                state_dict = {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args}
                torch.save(state_dict, os.path.join(output_path, "best.pth"))
                best_iou = miou.mean()
            print("mIoU of epoch ", epoch_idx, ': ', miou.mean(), ' per cls: ', miou, ' current best: ', best_iou)

        lr_scheduler.step()

if __name__ == "__main__":
    args = parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train(args)
