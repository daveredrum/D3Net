import os, random
import numpy as np
import glob
from importlib import import_module

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from tensorboardX import SummaryWriter

from lib.utils.log import Logger, AverageMeter
from lib.utils.solver import is_power2, is_multiple


class BaseSolver:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.general.task
        self.total_epoch = cfg.train.epochs
        self.curr_epoch = 0
        
        self._init_random_seed()
        self._init_log()
        self._init_data()
        # self.max_iter = self.total_epoch * len(self.loader)


    def _init_random_seed(self):
        if self.cfg.general.manual_seed:
            random.seed(self.cfg.general.manual_seed)
            np.random.seed(self.cfg.general.manual_seed)
            torch.manual_seed(self.cfg.general.manual_seed)
            torch.cuda.manual_seed_all(self.cfg.general.manual_seed)


    def _init_log(self):
        if self.cfg.model.use_checkpoint:
            self.logger = Logger.from_checkpoint(self.cfg)
            self.logger.info(f"=> resume from {self.logger.log_path}")
        else:
            self.logger = Logger(self.cfg)
    
    
    def _init_tb_writer(self):
        self.writer = SummaryWriter(self.logger.log_path)


    def _init_data(self):
        DATA_MODULE = import_module(self.cfg.data.module)
        dataloader = getattr(DATA_MODULE, self.cfg.data.loader)

        if self.cfg.general.task == 'train':
            self.logger.info('=> loading the train and val datasets...')
        else:
            self.logger.info(f'=> loading the {self.cfg.data.split} dataset...')
            
        self.dataset, self.dataloader = dataloader(self.cfg)
        self.logger.info('=> loading dataset completed')


    def _init_model(self):
        assert torch.cuda.is_available(), 'No CUDA available :('
        self.num_devices = torch.cuda.device_count()
        self.logger.info('num of GPUs available: {}'.format(self.num_devices))

        self.logger.info('=> initializing model ...')
        MODEL_MODULE = import_module(self.cfg.model.module)
        model = getattr(MODEL_MODULE, self.cfg.model.classname)
        
        self.model_path = self.logger.model_path
        # self.freeze_backbone = self.cfg.cluster.freeze_backbone
        self.model = model(self.cfg)
        self.model = self.model.cuda()
    
    
    def _init_optim(self):
        optim_class_name = self.cfg.train.optim.classname
        optim = getattr(torch.optim, optim_class_name)
        if optim_class_name == 'Adam':
            self.optimizer = optim(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.train.optim.lr)
        elif optim_class_name == 'SGD':
            self.optimizer = optim(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.train.optim.lr, momentum=self.cfg.train.optim.momentum, weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented


    def _init_criterion(self):
        raise NotImplemented
    
    
    def _resume_from_checkpoint(self):
        if self.cfg.model.use_checkpoint:
            self.logger.info(f'=> restoring checkpoint from {self.cfg.model.use_checkpoint} ...')
            self.start_epoch = self.restore_checkpoint(self.cfg.model.resume_epoch)      # resume from the latest epoch, or specify the epoch to restore
        else: 
            self.start_epoch = 1
            self._load_pretrained_module()
            self._freeze_module()
            self.curr_epoch = self.start_epoch
            
    
    def _load_pretrained_module(self):
        for i, module_name in enumerate(self.cfg.model.pretrained_module):
            self.logger.info(f'=> loading pretrained {module_name}...')
            module = getattr(self.model, module_name)
            ckp = torch.load(self.cfg.model.pretrained_module_path[i])
            module.load_state_dict(ckp)
        
       
    def _freeze_module(self):
        for module_name in self.cfg.model.freeze_module:
            module = getattr(self.model, module_name)
            for param in module.parameters():
                param.requires_grad = False
        
    
    def _load_pretrained_model(self):
        pretrained_path = self.cfg.model.pretrained_path
        self.logger.info(f'=> load pretrained model from {pretrained_path} ...')
        
        model_state = torch.load(pretrained_path)
        self.model.load_state_dict(model_state["model_state_dict"])
        
        self.start_epoch = self.cfg.model.resume_epoch


    def save_checkpoint(self, epoch):
        assert epoch is not None, 'Need to provide epoch to save checkpoint'
        ckp_filename = os.path.join(self.model_path, f'ckp-{epoch:05d}.tar')
        self.logger.info('=> saving checkpoint at ' + ckp_filename)

        save_dict = {
            "cuda": True,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(save_dict, ckp_filename)
        

    def save_model(self, epoch=None):
        model_filename = os.path.join(self.model_path, 'model_best.pth')
        self.logger.info('=> Saving the best model at ' + model_filename)
        
        save_dict = {
            "cuda": True,
            "model_state_dict": self.model.state_dict(),
        }
        if epoch:
            save_dict["epoch"] = epoch
        torch.save(save_dict, model_filename)


    def check_save_condition(self, epoch):
        if not self.cfg.train.save_freq:
            return True
        else:
            return is_power2(epoch) or is_multiple(epoch, self.cfg.train.save_freq) or epoch==0
    
    
    def restore_checkpoint(self, epoch=None):
        if epoch is None:
            cp_lists = sorted(glob.glob(os.path.join(self.model_path, 'ckp-*.tar')))
            if len(cp_lists) > 0:
                ckp_filename = cp_lists[-1]
                epoch = int(ckp_filename[-9:-4])
        else:
            ckp_filename = os.path.join(self.model_path, f'ckp-{epoch:05d}.tar')
            # ckp_filename = os.path.join(self.model_path, f'pointgroup_default_scannet-000000480.pth')
        self.logger.info(f'=> relocating epoch at {epoch} ...')
        assert os.path.isfile(ckp_filename), f'Invalid checkpoint file: {ckp_filename}'

        checkpoint = torch.load(ckp_filename)
        # self.model.load_state_dict(checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["epoch"] + 1
        # return self.cfg.model.resume_epoch
        
        
    def _loss(self):
        raise NotImplemented
        
        
    def _feed(self):
        raise NotImplemented
        
        
    def train(self):
        raise NotImplemented
    
    
    def eval(self):
        raise NotImplemented
    
    def inference(self):
        raise NotImplemented
    