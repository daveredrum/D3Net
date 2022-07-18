import logging
import os
import sys
from datetime import datetime
from omegaconf import OmegaConf

from tensorboardX import SummaryWriter

sys.path.append('../')


class Logger:

    def __init__(self, cfg, log_date=None, log_task=None):
        self.cfg = cfg
        self.use_console = cfg.log.use_console_log
        
        self.log_task = log_task if log_task else cfg.general.task
        self.log_name = f'{cfg.general.task}-logger'
        self.log_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if log_date is None else log_date
        self.log_filenames = {m: f'{self.log_task}.{m}.log' for m in ['full', 'less']}

        self.logdir_path = os.path.join(cfg.LOG_PATH, cfg.general.dataset, cfg.general.model, cfg.general.task)
        self.log_path = os.path.join(self.logdir_path, self.log_date)
        self.model_path = os.path.join(self.cfg.LOG_PATH, self.cfg.general.dataset, cfg.general.model, 'train', self.log_date)
        self.backup_path = os.path.join(self.log_path, 'backup')
        
        self.logfile_exists = True if self.log_date and cfg.general.task == 'train' else False
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)
            os.makedirs(self.backup_path, exist_ok=True)

        self._init_logger()
        self._init_tb_writer()
    
    def _init_logger(self):
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.DEBUG)

        # self.log_format = '[%(asctime)s  %(levelname)s]\n  %(message)s\n'
        self.log_format = '%(message)s'
        self.formatter = logging.Formatter(self.log_format)
        
        fh_mode = 'w' if self.logfile_exists is None else 'a'
        
        # initialize detailed logging file handler
        self.full_fh = logging.FileHandler(os.path.join(self.log_path, self.log_filenames['full']), mode=fh_mode)
        self.full_fh.setLevel(logging.DEBUG)
        self.full_fh.setFormatter(self.formatter)
        self.logger.addHandler(self.full_fh)
        
        # initialize detailed logging file handler
        self.less_fh = logging.FileHandler(os.path.join(self.log_path, self.log_filenames['less']), mode=fh_mode)
        self.less_fh.setLevel(logging.INFO)
        self.less_fh.setFormatter(self.formatter)
        self.logger.addHandler(self.less_fh)     

        if self.use_console:
            self.ch = logging.StreamHandler()
            self.ch.setLevel(logging.DEBUG)
            self.ch.setFormatter(self.formatter)
            self.logger.addHandler(self.ch) 

        self.logger.info('************************ Start Logging ************************')
        

    def _init_tb_writer(self):
        self.tb_writer = SummaryWriter(self.log_path)

    @classmethod
    def from_checkpoint(cls, cfg):
        log_date = cfg.model.use_checkpoint
        logger = cls(cfg, log_date)
        return logger
    
    @classmethod
    def from_evaluation(cls, cfg):
        log_date = cfg.evaluation.use_model
        log_task = cfg.evaluation.task
        logger = cls(cfg, log_date, log_task)
        return logger
    
    def store_backup_config(self):
        backup_file = os.path.join(self.backup_path, 'config.yaml')
        OmegaConf.save(self.cfg, backup_file)

    def info(self, message):
        self.logger.info(message)
        
    def debug(self, message):
        self.logger.debug(message)
        
    def tb_add_scalar(self, name, val, epoch):
        self.tb_writer.add_scalar(name, val, epoch)
    


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        # super(AverageMeter, self).__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Meters:
    def __init__(self, *meter_names):
        # self._init_meters()
        self._meters = {}
        self.add_meter(*meter_names)
        
    def _init_meters(self, *meter_names):
        pass
        # self.meters['iter_time'] = AverageMeter()
        
    def __getitem__(self, key):
        return self._meters[key]
    
    def add_meter(self, *meter_names):
        for name in meter_names:
            self._meters[name] = AverageMeter()
            
    def update(self, meter_name, val, n=1):
        self._meters[meter_name].update(val, n)
        
    def get_avg(self, meter_name):
        return self._meters[meter_name].avg
    
    def get_val(self, meter_name):
        return self._meters[meter_name].val