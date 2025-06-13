import os
import argparse # 导入argparse模块，用于处理命令行参数

from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics
from trainer.utils import load_model_trainer

config, logger = build_config()

if config['model_name'] == 'llava_next_tune':
    trainer = load_model_trainer(config, logger)
    trainer.train()
    acc = trainer.eval()
elif config['model_name'] == 'llava_next':
    trainer = load_model_trainer(config, logger)
    trainer.eval()