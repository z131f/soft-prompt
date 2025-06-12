import os
import argparse # 导入argparse模块，用于处理命令行参数

from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics

config, logger = build_config()

assert config['model_name'] == 'llava_next_tune'
assert config['action'] == 'skip'
from trainer.utils import load_model_trainer
acc_list = []
config['task'] = 'comp'
print(f'task: {config["task"]}')
trainer = load_model_trainer(config, logger)
trainer.train()
acc = trainer.eval()
acc_list.append(acc)
config['task'] = 'count'
print(f'task: {config["task"]}')
trainer = load_model_trainer(config, logger)
trainer.train()
acc = trainer.eval()
acc_list.append(acc)
config['task'] = 'presence'
print(f'task: {config["task"]}')
trainer = load_model_trainer(config, logger)
trainer.train()
acc = trainer.eval()
acc_list.append(acc)
config['task'] = 'rural_urban'
print(f'task: {config["task"]}')
trainer = load_model_trainer(config, logger)
trainer.train()
acc = trainer.eval()
acc_list.append(acc)
print(f'Accuracy list: {acc_list}')