config = {
    'dataset_name': 'RSVQA_LR',
    'model_name': 'llava_next',
    'seed': 42,
    'skip_training': False,
}

dataset_config = {
    'RSVQA_LR': {
        'load_train_num': 8000,
        'load_test_num': 1000,
    }
}

model_config = {
    'llava_next': {
        'prompt_num': 10,
        'lr': 5e-5,
    }
}

model_name_map = {
    'llava_next': 'llava-hf/llava-v1.6-mistral-7b-hf'
}
def build_config():
    config['model_site'] = model_name_map[config['model_name']]
    config['load_train_num'] = dataset_config[config['dataset_name']]['load_train_num']
    config['load_test_num'] = dataset_config[config['dataset_name']]['load_test_num']
    config['prompt_num'] = model_config[config['model_name']]['prompt_num']
    config['lr'] = model_config[config['model_name']]['lr']
    return config