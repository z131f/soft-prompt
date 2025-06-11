import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse # 导入argparse模块，用于处理命令行参数

from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics

config = build_config()
print("Configuration:", config)
os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_devices']
logger = get_logger()
set_seed(config['seed'])

if config['model_name'] == 'llava_next_tune':
    from trainer.llava_next_tune import llava_next_tune_trainer
    trainer = llava_next_tune_trainer(config, logger, action='skip')
    trainer.train()
    trainer.eval()
elif config['model_name'] == 'llava_next':
    from trainer.llava_next import llava_next_trainer
    trainer = llava_next_trainer(config, logger)
    trainer.eval()