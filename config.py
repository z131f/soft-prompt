config = {
    'dataset_name': 'RSVQA_LR',
    'model_name': 'llava_next',
    'seed': 42,
    'per_device_train_batch_size': 1,
}

dataset_config = {
    'RSVQA_LR': {
        'load_train_num': 100,
        'load_test_num': 50,
        'image_size': (256, 256),
    }
}

model_config = {
    'llava_next': {
        'lr': 1e-4,
    }
}

model_name_map = {
    'llava_next': 'llava-hf/llava-v1.6-mistral-7b-hf'
}
def build_config():
    config['model_site'] = model_name_map[config['model_name']]
    config['load_train_num'] = dataset_config[config['dataset_name']]['load_train_num']
    config['load_test_num'] = dataset_config[config['dataset_name']]['load_test_num']
    config['image_size'] = dataset_config[config['dataset_name']]['image_size']
    config['lr'] = model_config[config['model_name']]['lr']
    return config