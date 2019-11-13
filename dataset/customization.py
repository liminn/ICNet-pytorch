"""Prepare Customized dataset"""
import cv2
import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from .segbase import SegmentationDataset

class CustomizedDataset(SegmentationDataset):
    NUM_CLASS = 3
    IGNORE_INDEX = -1
    NAME = "product_segmentation"
    
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

    def __init__(self, txt_path = None, split='train', base_size=1024, crop_size=720, mode=None, transform=input_transform):
        """
        Parameters
            txt_path : string
                
            split: string
                'train', 'val' or 'test'
            base_size:
            
            crop_size:
            
            mode:
            
            transform : callable, optional
                A function that transforms the image
        """
        super(CustomizedDataset, self).__init__(split, mode, transform, base_size, crop_size)
        assert os.path.exists(self.txt_path), "Error: txt_path is wrong!"
        self.image_paths, self.mask_paths = self.get_pairs(self.txt_path)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Error: len(self.images) == 0")
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        # general normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.image_paths[index])

    # trimap generation with different/random kernel size
    def random_trimap(self, mask, smooth=False):
        # (cols, row, c) -> (rows, cols ,c)
        mask = np.array(mask).astype(np.float32)    
        h, w = mask.size
        scale_up, scale_down = 0.022, 0.006   # hyper parameter
        dmin = 0        # hyper parameter
        emax = 255 - dmin   # hyper parameter
        
        kernel_size_high = max(10, round((h + w) / 2 * scale_up))
        kernel_size_low  = max(1, round((h + w) /2 * scale_down))
        erode_kernel_size  = np.random.randint(kernel_size_low, kernel_size_high)
        dilate_kernel_size = np.random.randint(kernel_size_low, kernel_size_high)
        
        erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        eroded_alpha = cv2.erode(mask, erode_kernel)
        dilated_alpha = cv2.dilate(mask, dilate_kernel)

        dilated_alpha = np.where(dilated_alpha > dmin, 255, 0)
        eroded_alpha = np.where(eroded_alpha < emax, 0, 255)

        res = dilated_alpha.copy()
        res[((dilated_alpha == 255) & (eroded_alpha == 0))] = 128

        res[res==0] = 0
        res[res==255] = 1
        res[res==128] = 2

        return res.astype('int32')

    # 覆盖了基类的_mask_transform方法
    def _mask_transform(self, mask):
        # mask输入时为PIL image，输出时也是PIL image
        target = self.random_trimap(mask)
        #return torch.LongTensor(np.array(target).astype('int32'))
        return torch.LongTensor(target)
        
    def get_pairs(self, txt_path):
        image_paths = []
        mask_paths = []
        with open(txt_path, 'r') as f:
            list_lines = f.read().splitlines()
        for i in list_lines:
            image_paths.append(i.split('\t')[0])
            mask_paths.append(i.split('\t')[1])
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

if __name__ == '__main__':
    pass
