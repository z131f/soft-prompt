import os
import argparse # 导入argparse模块，用于处理命令行参数

from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics

config, logger = build_config()

if config['model_name'] == 'llava_next_tune':
    from trainer.llava_next_tune import llava_next_tune_trainer
    acc_list = []
    # config['task'] = 'comp'
    print(f'task: {config["task"]}')
    trainer = llava_next_tune_trainer(config, logger, action='skip')
    trainer.train()
    acc = trainer.eval()
    acc_list.append(acc)
    config['task'] = 'count'
    print(f'task: {config["task"]}')
    trainer = llava_next_tune_trainer(config, logger, action='skip')
    trainer.train()
    acc = trainer.eval()
    acc_list.append(acc)
    config['task'] = 'presence'
    print(f'task: {config["task"]}')
    trainer = llava_next_tune_trainer(config, logger, action='skip')
    trainer.train()
    acc = trainer.eval()
    acc_list.append(acc)
    config['task'] = 'rural_urban'
    print(f'task: {config["task"]}')
    trainer = llava_next_tune_trainer(config, logger, action='skip')
    trainer.train()
    acc = trainer.eval()
    acc_list.append(acc)
    print(f'Accuracy list: {acc_list}')
elif config['model_name'] == 'llava_next':
    from trainer.llava_next import llava_next_trainer
    config['task'] = 'comp'
    print(f'task: {config["task"]}')
    trainer = llava_next_trainer(config, logger)
    trainer.eval()
    config['task'] = 'count'
    print(f'task: {config["task"]}')
    trainer = llava_next_trainer(config, logger)
    trainer.eval()
    config['task'] = 'presence'
    print(f'task: {config["task"]}')
    trainer = llava_next_trainer(config, logger)
    trainer.eval()
    config['task'] = 'rural_urban'
    print(f'task: {config["task"]}')
    trainer = llava_next_trainer(config, logger)
    trainer.eval()