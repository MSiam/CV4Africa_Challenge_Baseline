"""
Ro'ya CV4Africa Challenge Baseline
workshop website: https://ro-ya-cv4africa.github.io/homepage/event_workshop.html
challenge website: https://codalab.lisn.upsaclay.fr/competitions/14259
"""

import torch
import torch.nn.functional as F


def map_mask(mask):
    cls_mapping = {255: 0, 124: 1, 7: 2, 201: 3}
    temp_mask = np.zeros(mask.shape)
    for cls in np.unique(mask):
        temp_mask[mask==cls] = cls_mapping[int(cls)]
    return temp_mask

def score_fn(pred_filenames, gt_filenames):
    all_ious = None

    for predfname, gtfname in tqdm(zip(pred_filenames, gt_filenames)):
        preds = cv2.imread(predfname, 0)
        labels = cv2.imread(gtfname, 0)
        labels = map_mask(labels)

        intersection, union, _ = batch_intersectionAndUnion(preds, labels, 4)
        intersection, union = intersection.cpu(), union.cpu()

        eps = 1e-10
        batch_iou = intersection / (union + eps)

        if all_ious is None:
            all_ious = batch_iou.sum(axis=0)
        else:
            all_ious += batch_iou.sum(axis=0)

    return all_ious / len(predfilenames)


def batch_intersectionAndUnion(logits: torch.Tensor,
                                  target: torch.Tensor,
                                  num_classes: int,
                                  ignore_index=255):
    """
    inputs:
        logits : shape [batch, num_class, h, w] or [h, w] in this case its direct predictions
        target : shape [batch, H, W] or [h, w]
        num_classes : Number of classes

    returns :
        area_intersection : shape [n_task, shot, num_class]
        area_union : shape [n_task, shot, num_class]
        area_target : shape [n_task, shot, num_class]
    """
    num_classes = 4

    if logits.ndim < 3:
        # No Batch dimension
        preds = torch.tensor(logits).unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)
        n_tasks, h, w = preds.size()
        preds = preds.view(1, *preds.shape)
    elif logits.ndim == 4:
        preds = logits.argmax(dim=1)
        n_tasks, h, w = preds.size()
        preds = preds.unsqueeze(1)

    # preds: Batch x Channel (1) x H x W
    H, W = target.size()[-2:]
    preds = F.interpolate(preds.float(), size=(H, W), mode='nearest')

    preds = preds[:, 0]

    area_intersection = torch.zeros(n_tasks, num_classes)
    area_union = torch.zeros(n_tasks, num_classes)
    area_target = torch.zeros(n_tasks, num_classes)
    for task in range(n_tasks):
        i, u, t = intersectionAndUnion(preds[task], target[task],
                                          num_classes, ignore_index=ignore_index)
        area_intersection[task, :] = i
        area_union[task, :] = u
        area_target[task, :] = t
    return area_intersection, area_union, area_target


def intersectionAndUnion(preds: torch.tensor,
                            target: torch.tensor,
                            num_classes: int,
                            ignore_index=255):
    """
    inputs:
        preds : shape [H, W]
        target : shape [H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [num_class]
        area_union : shape [num_class]
        area_target : shape [num_class]
    """
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape[-2:] == target.shape[-2:]
    preds = preds.view(-1)
    target = target.view(-1)
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]

    # Addind .float() becausue histc not working with long() on CPU
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_intersection
    # print(torch.unique(intersection))
    return area_intersection, area_union, area_target

