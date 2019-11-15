"""Prepare Cityscapes dataset"""
import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from .segbase import SegmentationDataset

class CityscapesDataset(SegmentationDataset):
    NUM_CLASS = 19
    IGNORE_INDEX=-1
    NAME = "cityscapes"

     # image transform
    """
        transforms.ToTensor():
            Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
            Converts a PIL Image or numpy.ndarray (H x W x C) in the range
            [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __init__(self, root = './datasets/Cityscapes', split='train', base_size=1024, crop_size=720, mode=None, transform=input_transform):
        """
        Parameters
            root : string
                Path to Cityscapes folder. Default is './datasets/Cityscapes'
            split: string
                'train', 'val' or 'test'
            transform : callable, optional
                A function that transforms the image
        """
        super(CityscapesDataset, self).__init__(root, split, mode, transform,base_size, crop_size)
        assert os.path.exists(self.root), "Error: data root path is wrong!"
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        # _gtFine_labelIds.png中，像素值从[-1,33]中的有效像素值
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        # reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        # _gtFine_labelIds.png中，像素值从[-1,33]所对应的类别值
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        # [-1, ..., 33]
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        
    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        # 获取mask中各像素值对应于_mapping的索引
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        # 依据上述索引，根据_key，得到对应
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])
        
    # 覆盖了基类的_mask_transform方法
    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))
        
    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

"""
Cityscapes文件夹构成：
Citicapes:
- leftImg8bit
    - train
        - aachen
            - aachen_xxx_leftImg8bit.png
            - ...
        - ....      
    - val
        - frankfurt
            - frankfurt_xxx_leftImg8bit.png
            - ...
        - ...
    - test
        - berloin
            - berlin_xxx_leftImg8bit.png
            - ...
        - ...
- gtFine 
    - train
        - aachen
            - aachen_xxx_gtFine_color.png
            - aachen_xxx_gtFine_labelIds.png
            - ...
        - ....      
    - val
        - frankfurt
            - frankfurt_xxx_gtFine_color.png
            - frankfurt_xxx_gtFine_labelIds.png
            - ...
        - ...
    - test
        - berloin
            - berloin_xxx_gtFine_color.png
            - berloin_xxx_gtFine_labelIds.png
            - ...
        - ...
- trainImages.txt
- trainLabels.txt
- valImages.txt
- valLabels.txt
- testImages.txt
- testLabels.txt
"""

def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    """
                    Example:
                        root = "./Cityscapes/leftImg8bit/train/aachen"
                        filename = "aachen_xxx_leftImg8bit.png"
                        imgpath = "./Cityscapes/leftImg8bit/train/aachen/aachen_xxx_leftImg8bit.png"
                        foldername = "aachen"
                        maskname = "aachen_xxx_gtFine_labelIds.png"
                        maskpath = "./Cityscapes/gtFine/train/aachen/aachen_xxx_gtFine_labelIds"
                    """
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        # "./Cityscapes/gtFine/train" or "./Cityscapes/gtFine/val"
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        # img_paths与mask_paths的顺序是一一对应的
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    pass
