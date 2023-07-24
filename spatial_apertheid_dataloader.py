import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import transforms as T
import cv2
import torch
from tqdm import tqdm

def make_train_transform_hr(train_size=None):
    print('using make_train_transform_hr> train_size:%d' % train_size)
    normalize = T.Compose([
        T.ToTensor(),
        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        #T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([200, 250, 300]),
            T.CenterCrop((128, 128)),
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),  # for r50
        ]),
        normalize,
    ])


def make_validation_transforms(val_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([val_size], max_size=int(val_size * 2)),
        normalize,
    ])

class SpatialApertheidDataset(data.Dataset):
    def __init__(self, data_root, file_list, split='train', img_size=None, overfit=False):
        super(SpatialApertheidDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.overfit = overfit
        self.file_list = file_list

        self.classes = []
        self.load_file_list(file_list)
        #self.load_annotations()

        self.classes = [255, 124, 7, 201]
        self.cls_mapping = {}
        for idx_cls, cls in enumerate(self.classes):
            if cls == 255:
                mapped_cls = 0
            else:
                if idx_cls == 0:
                    mapped_cls = idx_cls + 1
                else:
                    mapped_cls = idx_cls
            self.cls_mapping[cls] = mapped_cls

        self.transform = None
        if self.split == 'train':
            self.transform =  make_validation_transforms(img_size) #make_train_transform_hr(img_size)
        elif self.split in ['val', 'test']:
            self.transform = make_validation_transforms(img_size)
        else:
            raise Exception("Invalid split")

    def load_annotations(self):
        self.cls_mapping = {}

        for mask in tqdm(self.mask_files):
            mask = cv2.imread(mask, 0)
            for cls in np.unique(mask):
                if cls not in self.classes:
                    self.classes.append(cls)

        for idx_cls, cls in enumerate(self.classes):
            if cls == 255:
                mapped_cls = 0
            else:
                if idx_cls == 0:
                    mapped_cls = idx_cls + 1
                else:
                    mapped_cls = idx_cls

            self.cls_mapping[cls] = mapped_cls

    def load_file_list(self, file_list):
        self.image_files = []
        self.mask_files = []
        with open(os.path.join(self.data_root, file_list), "r") as f:
            for line in f:
                self.mask_files.append(line.strip())
                self.image_files.append(line.strip().replace("labels_4", "images"))
        if self.overfit:
            self.image_files = self.image_files[:2]
            self.mask_files = self.mask_files[:2]

    def __len__(self):
        return len(self.mask_files)

    def map_mask(self, mask):
        temp_mask = torch.zeros(mask.shape)
        for cls in torch.unique(mask):
            temp_mask[mask==cls] = self.cls_mapping[int(cls)]
        return temp_mask

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        mask = torch.tensor(cv2.imread(self.mask_files[idx], 0)).unsqueeze(0)
        mask = self.map_mask(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask.long(), self.image_files[idx]

def denormalize(img, mean, scale):
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale).view(1, 1, -1) + torch.tensor(mean).view(1, 1, -1)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,::-1]*255, np.uint8)
    return img

def create_overlay(img, mask, cat2color):
    mask = mask[0]
    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    for cat, color in cat2color.items():
        mask_color[mask==cat] = color

    collated = np.concatenate([img, mask_color], axis=1)
    return collated

if __name__ == "__main__":
    spatial_dataset = SpatialApertheidDataset("/home/msiam/Code/spatial_challenge_prepration/spatial_apertheid_challenge_v2/",
                                              "train.txt", "train", img_size=256)
    batch_size = 16
    spatial_loader = data.DataLoader(spatial_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if not os.path.exists("out_images"):
        os.makedirs("out_images")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    cat2color = {cat: [key]*3 for key, cat in spatial_dataset.cls_mapping.items()}
    for idx, (images, labels, fnames) in enumerate(tqdm(spatial_loader)):
        for bidx in range(images.shape[0]):
            img = denormalize(images[bidx], mean, std)
            overlay_img = create_overlay(img, labels[bidx], cat2color)
            fname = fnames[bidx].split('/')[-1]
            cv2.imwrite(os.path.join("out_images/", fname), overlay_img)
