import logging
import logging.config
from dataclasses import dataclass
from configparser import ConfigParser

config = ConfigParser()
config.read('./config/config.ini')

logging.config.fileConfig('./config/logger.ini')
logger = logging.getLogger()

def get_logger():
    return logger

@dataclass
class TokenizerConf:
    cache_dir: str = './cache/phobert-base-v2'
    vocab_size: int = 64000
    pretrained_model_name: str = 'vinai/phobert-base-v2'

@dataclass
class DatasetConf:
    column: str = config['dataset']['column']
    cache_dir: str = config['dataset']['cache_dir']
    data_dir: str = config['dataset']['data_dir']
    extension: str = config['dataset']['extension']
    n_processes: int = int(config['dataset']['n_processes'])
    threshold: float = float(config['dataset']['threshold'])
    dedup_file: str = config['dataset']['dedup_file']
    temp_column: str = config['dataset']['temp_column']
    seed: int = int(config['dataset']['seed'])
    test_size: int = int(config['dataset']['test_size'])
    split_dir: str = config['dataset']['split_dir']
    train_data: str = config['dataset']['train_data']
    val_data: str = config['dataset']['val_data']
    test_data: str = config['dataset']['test_data']
    
class TrainerConf(object):
    def __init__(self):
        self.data_seed: int = 42
        self.dataloader_num_workers: int = 4
        self.do_train: bool = True
        self.do_eval: bool = True
        self.evaluation_strategy: str =  'steps'
        self.eval_steps: int = 5000
        self.fp16: bool = False
        self.gradient_accumulation_steps: int = 1
        self.learning_rate: float = 2e-5
        self.log_level: str = 'info'
        self.logging_strategy: str = 'steps'
        self.logging_steps: int = 5000
        self.lr_scheduler_type: str = 'linear'
        self.max_steps: int = 500000
        self.mlm_probability = 0.15
        self.per_device_train_batch_size: int = 64
        self.per_device_eval_batch_size: int = 64
        self.prediction_loss_only: bool = True
        self.report_to: str = 'wandb'
        self.save_strategy: str = 'steps'
        self.save_steps: int = 10000
        self.save_total_limit: int = 10
        self.seed: int = 42
        self.warmup_ratio: float = 0.01
        self.warmup_steps: int = 10000
        self.weight_decay: float = 0.015
        self.logging_dir = './log/phobert-base'
        self.output_dir = './checkpoint/phobert-base'
        self.run_name = 'phobert-base'
        self.pretrained_model_name = 'vinai/phobert-base-v2'
        
if __name__=='__main__':
    pass
