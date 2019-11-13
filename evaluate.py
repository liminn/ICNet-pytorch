import os
import time
import datetime
import yaml
import shutil
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
from models import ICNet
from dataset import CityscapesDataset
from utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger, get_color_pallete

class Evaluator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get valid dataset images and targets
        self.image_paths, self.mask_paths = _get_city_pairs(cfg["train"]["cityscapes_root"], "val")

        # create network
        self.model = ICNet(nclass = 19, backbone='resnet50').to(self.device)
        
        # load ckpt
        pretrained_net = torch.load(cfg["test"]["ckpt_path"])
        self.model.load_state_dict(pretrained_net)
        
        # evaluation metrics
        self.metric = SegmentationMetric(19)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        model = self.model

        logger.info("Start validation, Total sample: {:d}".format(len(self.image_paths)))
        list_time = []
        lsit_pixAcc = []
        list_mIoU = []

        for i in range(len(self.image_paths)):

            image = Image.open(self.image_paths[i]).convert('RGB') # image shape: (W,H,3)
            mask = Image.open(self.mask_paths[i])                  # mask shape: (W,H)
            
            image = self._img_transform(image)                     # image shape: (3,H,W) [0,1]
            mask = self._mask_transform(mask)                      # mask shape: (H,w)

            image = image.to(self.device)
            mask = mask.to(self.device)

            image = torch.unsqueeze(image, 0)                      # image shape: (1,3,H,W) [0,1]

            with torch.no_grad():
                start_time = time.time()
                outputs = model(image)
                end_time = time.time()
                step_time = end_time-start_time
            self.metric.update(outputs[0], mask)
            pixAcc, mIoU = self.metric.get()
            list_time.append(step_time)
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, time: {:.3f}s".format(
                i + 1, pixAcc * 100, mIoU * 100, step_time))
            
            filename = os.path.basename(self.image_paths[i])
            prefix = filename.split('.')[0]

            # save pred 
            pred = torch.argmax(outputs[0], 1)                
            pred = pred.cpu().data.numpy()
            pred = pred.squeeze(0)
            pred = get_color_pallete(pred, "citys")
            pred.save(os.path.join(outdir, prefix + "_mIoU_{:.3f}.png".format(mIoU)))
            
            # save image
            image = Image.open(self.image_paths[i]).convert('RGB') # image shape: (W,H,3)
            image.save(os.path.join(outdir, prefix + '_src.png'))
            
            # save target
            mask = Image.open(self.mask_paths[i])                   # mask shape: (W,H)
            mask = self._class_to_index(np.array(mask).astype('int32'))
            mask = get_color_pallete(mask, "citys")
            mask.save(os.path.join(outdir, prefix + '_label.png'))

        average_pixAcc = sum(lsit_pixAcc)/len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU)/len(list_mIoU)
        average_time = sum(list_time)/len(list_time)
        self.current_mIoU = average_mIoU
        logger.info("Evaluate: Average mIoU: {:.3f}, Average pixAcc: {:.3f}, Average time: {:.3f}".format(average_mIoU, average_pixAcc, average_time))
        
    def _img_transform(self, image):            
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        image = image_transform(image)
        return image

    def _mask_transform(self, mask):
        mask = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(mask).astype('int32'))
        
    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        for value in values:
            assert (value in self._mapping)
        # 获取mask中各像素值对应于_mapping的索引
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        # 依据上述索引index，根据_key，得到对应的mask图
        return self._key[index].reshape(mask.shape)

def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    """
                    For example:
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
    return img_paths, mask_paths

if __name__ == '__main__':
    # Set config file
    config_path = "./configs/icnet.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
        #print(cfg)
        #print(cfg["model"]["backbone"])
        print(cfg["train"]["specific_gpu_num"])
    
    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    
    outdir = os.path.join(cfg["train"]["ckpt_dir"], "evaluate_output")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    logger = SetupLogger(name = "semantic_segmentation", 
                         save_dir = cfg["train"]["ckpt_dir"], 
                         distributed_rank = 0, 
                         filename='{}_{}_evaluate_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))

    evaluator = Evaluator(cfg)
    evaluator.eval()