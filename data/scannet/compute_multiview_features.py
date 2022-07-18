import os
import sys
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from imageio import imread
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "../..")) # HACK add the root folder

from model.enet import create_enet_for_3d


class EnetDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self._init_resources()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene_id, frame_id = self.data[idx]
        image = self._load_image(os.path.join(self.cfg.SCANNETV2_PATH.frames, scene_id, "color", "{}.jpg".format(frame_id)), [328, 256])

        return scene_id, frame_id, image

    def _init_resources(self):
        self._get_scene_list()
        self.data = []
        for scene_id in self.scene_list:
            frame_list = sorted(os.listdir(os.path.join(self.cfg.SCANNETV2_PATH.frames, scene_id, "color")), key=lambda x:int(x.split(".")[0]))
            for frame_file in frame_list:
                self.data.append(
                    (
                        scene_id,
                        int(frame_file.split(".")[0])
                    )
                )
    
    def _get_scene_list(self):
        with open(self.cfg.SCANNETV2_PATH.scene_list, 'r') as f:
            self.scene_list = sorted(list(set(f.read().splitlines())))

    def _resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims != new_image_dims:
            resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
            image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
            image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        
        return np.array(image)

    def _load_image(self, file, image_dims):
        image = imread(file)
        # preprocess
        image = self._resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image = np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise ValueError

        return image

    def collate_fn(self, data):
        scene_ids, frame_ids, images = zip(*data)
        scene_ids = list(scene_ids)
        frame_ids = list(frame_ids)
        images = torch.stack(images, 0).cuda()

        return scene_ids, frame_ids, images

def create_enet(cfg):
    enet_fixed, enet_trainable, _ = create_enet_for_3d(41, cfg.SCANNETV2_PATH.enet_weights, 21)
    enet = nn.Sequential(
        enet_fixed,
        enet_trainable
    ).cuda()
    enet.eval()
    for param in enet.parameters():
        param.requires_grad = False

    return enet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help='scannet configuration YAML file', default='../../conf/path.yaml')
    opt = parser.parse_args()

    cfg = OmegaConf.load(opt.cfg)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # init
    dataset = EnetDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=dataset.collate_fn)
    enet = create_enet(cfg)

    # feed
    print("extracting multiview features from ENet...")
    for scene_ids, frame_ids, images in tqdm(dataloader):
        features = enet(images)
        batch_size = images.shape[0]
        for batch_id in range(batch_size):
            enet_feature_root = os.path.join(cfg.SCANNETV2_PATH.enet_features, scene_ids[batch_id])
            enet_feature_path = os.path.join(enet_feature_root, "{}.npy".format(frame_ids[batch_id]))

            os.makedirs(enet_feature_root, exist_ok=True)
            np.save(enet_feature_path, features[batch_id].cpu().numpy())

    print("done!")

