import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics

config, logger = build_config()


dataset_names = ['RSVQA_LR', 'RSVQA_HR', 'RSIVQA']
dataset_dic = {}

try:
    for da in dataset_names:
        from trainer.utils import load_model_trainer
        config['dataset_name'] = da
        config['action'] = 'begin'
        config['task'] = 'all'
        config['tag'] = 'auto'
        if da == 'RSIVQA':
            config['load_train_num'] = [300, 300, 300]
        else:
            config['load_train_num'] = [300, 300, 300, 300]

        trainer = load_model_trainer(config, logger)
        if 'tune' in config['model_name']:
            trainer.train()
        config['action'] = 'skip'


        if config['dataset_name'] == 'RSVQA_LR':
            acc_dict = {
                'comp': 0.0,
                'count': 0.0,
                'presence': 0.0,
                'rural_urban': 0.0
            }
        elif config['dataset_name'] == 'RSVQA_HR':
            acc_dict = {
                'area': 0.0,
                'comp': 0.0,
                'count': 0.0,
                'presence': 0.0,
            }
        elif config['dataset_name'] == 'RSIVQA':
            acc_dict = {
                'YesOrNo': 0.0,
                'Number': 0.0,
                'Other': 0.0,
            }

        for task in acc_dict.keys():
            config['task'] = task
            print(f'task: {config["task"]}')
            trainer.load_data()
            acc = trainer.eval(is_print=False)
            acc_dict[task] = acc
            print(f'Accuracy for {task}: {acc}')

        acc_dict['avg'] = sum(acc_dict.values()) / len(acc_dict)

        config['task'] = 'all'
        trainer.load_data()
        acc = trainer.eval(is_print=False)
        acc_dict['all'] = acc

        print(f'Accuracy dict: {acc_dict}')

        del trainer
        # torch.cuda.empty_cache()

        dataset_dic[da] = acc_dict
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print(dataset_dic)