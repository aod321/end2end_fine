import os
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torchvision.transforms.functional as TF
import cv2

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

np.set_printoptions(threshold=np.inf)


class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, parts_root_dir=None, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.parts_root_dir = parts_root_dir
        self.transform = transform
        self.names = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        self.label_id = {'eyebrow1': [2],
                         'eyebrow2': [3],
                         'eye1': [4],
                         'eye2': [5],
                         'nose': [6],
                         'mouth': [7, 8, 9]
                         }

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]
        image = io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        # bg = labels[0] + labels[1] + labels[10]
        bg = 255 - labels[2:10].sum(0)
        labels = np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0)
        parts_image = None
        parts_ground = None
        if self.parts_root_dir:
            parts_label_path = {x: [os.path.join(self.parts_root_dir, '%s' % x,
                                                 'labels', img_name,
                                                 img_name + "_lbl%.2d.png" % i)
                                    for i in self.label_id[x]]
                                for x in self.names}
            part_path = [os.path.join(self.parts_root_dir, '%s' % x, 'images',
                                      img_name + '.jpg')
                         for x in self.names]
            parts_ground = {x: np.array([io.imread(parts_label_path[x][i])
                                         for i in range(len(self.label_id[x]))
                                         ])
                            for x in self.names
                            }
            parts_image = [io.imread(part_path[i])
                           for i in range(6)]

            for x in self.names:
                bg = 255 - np.sum(parts_ground[x], axis=0, keepdims=True)  # [1, 64, 64]
                parts_ground[x] = np.uint8(np.concatenate([bg, parts_ground[x]], axis=0))  # [L + 1, 64, 64]
            sample = {'image': image, 'labels': labels, 'orig': image,
                      'orig_label': labels, 'pflag': 1, 'parts': parts_image, 'parts_ground': parts_ground}
        else:
            sample = {'image': image, 'labels': labels, 'orig': image,
                      'orig_label': labels, 'pflag': 0}

        if self.transform:
            sample = self.transform[0](sample)

        return sample