import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2 as cv
import numpy as np
import random
from skimage.util import random_noise
from PIL import ImageFilter, Image
import cv2

name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
max_1 = 0
max_2 = 0

class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        orig_label = sample['orig_label']
        orig = sample['orig']
        pflag = sample['pflag']
        if pflag:
            parts = sample['parts']
            parts_ground = sample['parts_ground']
            parts_ground = {
                x: np.array([np.array(TF.resize(parts_ground[x][r], (64, 64), Image.ANTIALIAS))
                             for r in range(len(parts_ground[x]))])
                for x in name_list
            }
            resized_parts = np.array([cv2.resize(parts[i], (64, 64), interpolation=cv2.INTER_AREA)
                                      for i in range(len(parts))])

        resized_image = TF.resize(image, self.size, Image.ANTIALIAS)
        resized_labels = [TF.resize(labels[r], self.size, Image.ANTIALIAS)
                          for r in range(len(labels))
                          ]
        if pflag:
            sample = {'image': resized_image,
                      'labels': resized_labels,
                      'orig': orig,
                      'orig_label': orig_label,
                      'pflag': pflag,
                      'parts': resized_parts,
                      'parts_ground': parts_ground
                      }
        else:
            sample = {'image': resized_image,
                      'labels': resized_labels,
                      'orig': orig,
                      'orig_label': orig_label,
                      'pflag': pflag
                      }

        return sample


class ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """

    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        image, labels = sample['image'], sample['labels']
        orig_label = sample['orig_label']
        pflag = sample['pflag']
        if pflag:
            parts_ground = sample['parts_ground']
            parts_ground = {x: [TF.to_pil_image(parts_ground[x][i])
                                for i in range(len(parts_ground[x]))]
                            for x in name_list
                            }

        labels = np.uint8(labels)
        orig_label = np.uint8(orig_label)
        image = TF.to_pil_image(image)

        labels = [TF.to_pil_image(labels[i])
                  for i in range(labels.shape[0])]

        orig_label = [TF.to_pil_image(orig_label[i])
                      for i in range(len(orig_label))]

        if pflag:
            sample = {'image': image,
                      'labels': labels,
                      'orig': TF.to_pil_image(sample['orig']),
                      'orig_label': orig_label,
                      'pflag': pflag,
                      'parts': sample['parts'],
                      'parts_ground': parts_ground
                      }
        else:
            sample = {'image': image,
                      'labels': labels,
                      'orig': TF.to_pil_image(sample['orig']),
                      'orig_label': orig_label,
                      'pflag': pflag
                      }
        return sample


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']
        orig_label = sample['orig_label']
        orig = sample['orig']
        pflag = sample['pflag']
        if pflag:
            parts = sample['parts']
            parts_ground = sample['parts_ground']
            parts_ground = [torch.cat([TF.to_tensor(parts_ground[x][r])
                                       for r in range(len(parts_ground[x]))
                                       ])
                            for x in name_list
                            ]

            parts_ground = torch.stack([parts_ground[r].argmax(dim=0, keepdim=False)
                                        for r in range(len(parts_ground))], dim=0)  # Shape(6, 64, 64)
            parts = torch.stack([TF.to_tensor(parts[i])
                                 for i in range(len(parts))])
            assert parts_ground.shape == (6, 64, 64)

        labels = [TF.to_tensor(labels[r])
                  for r in range(len(labels))
                  ]
        labels = torch.cat(labels, dim=0).float()
        labels = labels.argmax(dim=0).float()  # Shape(H, W)

        orig_label = [TF.to_tensor(TF.resize(orig_label[r], (512, 512), Image.ANTIALIAS))
                      for r in range(len(orig_label))
                      ]
        orig_label = torch.cat(orig_label, dim=0).float()
        orig_label[0] = 1 - torch.sum(orig_label[1:], dim=0, keepdim=True)
        orig_label = orig_label.argmax(dim=0, keepdim=False)        # (N, 512, 512)

        orig = TF.to_tensor(TF.resize(orig, (512, 512), Image.ANTIALIAS))

        if pflag:
            sample = {'image': TF.to_tensor(image),
                      'labels': labels,
                      'orig': orig,
                      'orig_label': orig_label,
                      'pflag': pflag,
                      'parts': parts,
                      'parts_ground': parts_ground
                      }
        else:
            sample = {'image': TF.to_tensor(image),
                      'labels': labels,
                      'orig': orig,
                      'orig_label': orig_label,
                      'pflag': pflag
                      }
        return sample


class Stage2Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        resized_image = np.array([cv2.resize(image[i], self.size, interpolation=cv2.INTER_AREA)
                                  for i in range(len(image))])
        labels = {x: np.array([np.array(TF.resize(TF.to_pil_image(labels[x][r]), self.size, Image.ANTIALIAS))
                               for r in range(len(labels[x]))])
                  for x in name_list
                  }

        sample = {'image': resized_image,
                  'labels': labels
                  }

        return sample


class Stage2ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image = sample['image']
        labels = sample['labels']
        image = torch.stack([TF.to_tensor(image[i])
                             for i in range(len(image))])

        labels = {x: torch.cat([TF.to_tensor(labels[x][r])
                                for r in range(len(labels[x]))
                                ])
                  for x in name_list
                  }

        return {'image': image,
                'labels': labels
                }


class Stage2_ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """

    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        image, labels = sample['image'], sample['labels']
        image = [TF.to_pil_image(image[i])
                 for i in range(len(image))]
        labels = {x: [TF.to_pil_image(labels[x][i])
                      for i in range(len(labels[x]))]
                  for x in name_list
                  }

        return {'image': image,
                'labels': labels
                }


class RandomRotation(transforms.RandomRotation):
    """Rotate the image by angle.

        Override the __call__ of transforms.RandomRotation

    """

    def __call__(self, sample):
        """
            sample (dict of PIL Image and label): Image to be rotated.

        Returns:
            Rotated sample: dict of Rotated image.
        """

        angle = self.get_params(self.degrees)

        img, labels = sample['image'], sample['labels']
        orig, orig_label = sample['orig'], sample['orig_label']

        rotated_img = TF.rotate(img, angle, self.resample, self.expand, self.center)
        rotated_labels = [TF.rotate(labels[r], angle, self.resample, self.expand, self.center)
                          for r in range(len(labels))
                          ]
        rotated_orig = TF.rotate(orig, angle, self.resample, self.expand, self.center)
        rotated_orig_label = [TF.rotate(orig_label[r], angle, self.resample, self.expand, self.center)
                              for r in range(len(orig_label))
                              ]

        sample = {'image': rotated_img,
                  'labels': rotated_labels,
                  'orig': rotated_orig,
                  'orig_label': rotated_orig_label,
                  'pflag': sample['pflag']
                  }

        return sample


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.RandomResizedCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']
        orig, orig_label = sample['orig'], sample['orig_label']

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        croped_img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        croped_labels = [TF.resized_crop(labels[r], i, j, h, w, self.size, self.interpolation)
                         for r in range(len(labels))
                         ]
        croped_orig = TF.resized_crop(orig, i, j, h, w, self.size, self.interpolation)
        croped_orig_label = [TF.resized_crop(orig_label[r], i, j, h, w, self.size, self.interpolation)
                             for r in range(len(orig_label))
                             ]

        sample = {'image': croped_img,
                  'labels': croped_labels,
                  'orig': croped_orig,
                  'orig_label': croped_orig_label,
                  'pflag': sample['pflag']
                  }

        return sample


class HorizontalFlip(object):
    """ HorizontalFlip the given PIL Image
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be

        Returns:
        """
        img, labels = sample['image'], sample['labels']
        orig, orig_label = sample['orig'], sample['orig_label']

        img = TF.hflip(img)

        labels = [TF.hflip(labels[r])
                  for r in range(len(labels))
                  ]
        orig = TF.hflip(orig)

        orig_label = [TF.hflip(orig_label[r])
                      for r in range(len(orig_label))
                      ]
        sample = {'image': img,
                  'labels': labels,
                  'orig': orig,
                  'orig_label': orig_label,
                  'pflag': sample['pflag']
                  }

        return sample


class CenterCrop(transforms.CenterCrop):
    """CenterCrop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.CenterCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']
        orig, orig_label = sample['orig'], sample['orig_label']

        croped_img = TF.center_crop(img, self.size)
        croped_labels = [TF.center_crop(labels[r], self.size)
                         for r in range(len(labels))
                         ]
        croped_orig = TF.center_crop(orig, self.size)
        croped_orig_label = [TF.center_crop(orig_label[r], self.size)
                             for r in range(len(orig_label))
                             ]

        sample = {'image': croped_img,
                  'labels': croped_labels,
                  'orig': croped_orig,
                  'orig_label': croped_orig_label,
                  'pflag': sample['pflag']
                  }

        return sample


class RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        orig, orig_label = sample['orig'], sample['orig_label']
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = TF.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        labels = [TF.affine(labels[r], *ret, resample=self.resample, fillcolor=self.fillcolor)
                  for r in range(len(labels))]
        orig = TF.affine(orig, *ret, resample=self.resample, fillcolor=self.fillcolor)
        orig_label = [TF.affine(orig_label[r], *ret, resample=self.resample, fillcolor=self.fillcolor)
                      for r in range(len(orig_label))]
        sample = {'image': img,
                  'labels': labels,
                  'orig': orig,
                  'orig_label': orig_label,
                  'pflag': sample['pflag']
                  }
        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        orig, orig_label = sample['orig'], sample['orig_label']
        img = np.array(img, np.uint8)
        img = random_noise(img)
        img = TF.to_pil_image(np.uint8(255 * img))

        orig = np.array(orig, np.uint8)
        orig = random_noise(orig)
        orig = TF.to_pil_image(np.uint8(255 * orig))
        sample = {'image': img,
                  'labels': labels,
                  'orig': orig,
                  'orig_label': orig_label,
                  'pflag': sample['pflag']
                  }

        return sample


class Blurfilter(object):
    # img: PIL image
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        img = img.filter(ImageFilter.BLUR)
        orig = sample['orig'].filter(ImageFilter.BLUR)
        sample = {'image': img,
                  'labels': labels,
                  'orig': orig,
                  'orig_label': sample['orig_label'],
                  'pflag': sample['pflag']
                  }

        return sample


class Normalize(object):
    """Normalize Tensors.
    """

    def __call__(self, sample):
        """
        Args:
            sample (dict of Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensors of sample: Normalized Tensor sample. Only the images need to be normalized.
        """

        image_tensor, labels_tensor = sample['image'], sample['labels']
        pflag = sample['pflag']
        # mean = image_tensor.mean(dim=[1, 2]).tolist()
        # std = image_tensor.std(dim=[1, 2]).tolist()
        mean = [0.369, 0.314, 0.282]
        std = [0.282, 0.251, 0.238]
        inplace = True
        if pflag:
            sample = {'image': TF.normalize(image_tensor, mean, std, inplace),
                      'labels': labels_tensor,
                      'orig': TF.normalize(sample['orig'], mean, std, inplace),
                      'orig_label': sample['orig_label'],
                      'pflag': pflag,
                      'parts': torch.stack([TF.normalize(sample['parts'][i], mean, std, inplace)
                                            for i in range(6)]),
                      'parts_ground': sample['parts_ground']
                      }
        else:
            sample = {'image': TF.normalize(image_tensor, mean, std, inplace),
                      'labels': labels_tensor,
                      'orig': TF.normalize(sample['orig'], mean, std, inplace),
                      'orig_label': sample['orig_label'],
                      'pflag': pflag
                      }
        return sample
