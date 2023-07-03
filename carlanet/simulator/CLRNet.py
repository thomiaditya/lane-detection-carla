import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm

class CLRNet:
    """
    CLRNet is a model for doing inference on Lane Detection task.
    """
    def __init__(self, config_path, weight_path):
        self.cfg = Config.fromfile(config_path)
        self.cfg.load_from = weight_path
        self.processes = Process(self.cfg.val_process, self.cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, carla_img):
        """
        Preprocesses the image for inference.

        Args:
            img_path (str): The path to the image.

        Returns:
            dict: The preprocessed image.
        """
        img = carla_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':"", 'ori_img':carla_img})
        return data
    
    def inference(self, data):
        """
        Performs inference on the given image.

        Args:
            data (dict): The preprocessed image.

        Returns:
            dict: The output of the model.
        """
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        return data
    
    def show(self, data):
        """
        Shows the output of the model.

        Args:
            data (dict): The output of the model.
        """
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def run(self, carla_img):
        """
        Runs the model on the given image.

        Args:
            data (dict): The preprocessed image.

        Returns:
            dict: The output of the model.
        """
        data = self.preprocess(carla_img)
        data['lanes'] = self.inference(data)[0]
        # if self.cfg.show or self.cfg.savedir:
        #     self.show(data)
        return data