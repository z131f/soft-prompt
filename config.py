import argparse


config = {
    'dataset_name': 'RSVQA_LR',
    'model_name': 'llava_next',
    'seed': 42,
    'per_device_train_batch_size': 4,
    'load_tag': 'RSVQA_LR_1000_0.0001',
    'cuda_devices': '0,1',
    'action': 'train',
}

dataset_config = {
    'RSVQA_LR': {
        'load_train_num': 1000,
        'load_test_num': 1500,
        'image_size': (256, 256),
        'task': 'all'
    }
}

model_config = {
    'llava_next_tune': {
        'lr': 1e-4,
    },
    'llava_next': {
        'lr': None,
    }
}

model_name_map = {
    'llava_next_tune': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_next': 'llava-hf/llava-v1.6-mistral-7b-hf'
}
def get_args():
    """
    解析命令行参数。
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='训练配置参数') # Training configuration parameters

    # 为每个默认配置项添加命令行参数
    # Add command-line arguments for each default config item
    parser.add_argument('--dataset_name', type=str, default=config['dataset_name'],
                        help='数据集名称，例如 RSVQA_LR') # Dataset name, e.g., RSVQA_LR
    parser.add_argument('--model_name', type=str, default=config['model_name'],
                        help='模型名称，例如 llava_next 或 llava_next_tune') # Model name, e.g., llava_next or llava_next_tune
    parser.add_argument('--per_device_train_batch_size', type=int,
                        default=config['per_device_train_batch_size'],
                        help='每个设备的训练批次大小') # Per device training batch size
    parser.add_argument('--load_tag', type=str, default=config['load_tag'],
                        help='加载标签，用于识别模型或数据加载路径') # Load tag, used to identify model or data load path
    parser.add_argument('--cuda_devices', type=str, default=config['cuda_devices'],
                        help='CUDA设备ID，例如 "0" 或 "0,1"') # CUDA device IDs, e.g., "0" or "0,1"
    parser.add_argument('--action', type=str, default=config['action'],
                        help='要执行的操作，例如 "train" 或 "eval"') # Action to perform, e.g., "train" or "eval"
    parser.add_argument('--task', type=str)

    args = parser.parse_args()
    return args

def build_config(args):
    """
    根据默认配置和命令行参数构建最终配置。
    Builds the final configuration based on default settings and command-line arguments.

    Args:
        args: argparse.Namespace 对象，包含命令行参数。
              argparse.Namespace object containing command-line arguments.
    Returns:
        dict: 最终的配置字典。
              The final configuration dictionary.
    """
    # 创建配置的副本，以避免直接修改全局变量
    # Create a copy of the config to avoid directly modifying the global variable
    current_config = config.copy()

    # 使用命令行参数覆盖默认配置
    # Override default config with command-line arguments

    # 根据 dataset_name 和 model_name 进一步构建配置
    # Further build the configuration based on dataset_name and model_name
    current_config['model_site'] = model_name_map[current_config['model_name']]
    current_config['load_train_num'] = dataset_config[current_config['dataset_name']]['load_train_num']
    current_config['load_test_num'] = dataset_config[current_config['dataset_name']]['load_test_num']
    current_config['image_size'] = dataset_config[current_config['dataset_name']]['image_size']
    current_config['lr'] = model_config[current_config['model_name']]['lr']
    current_config['task'] = dataset_config[current_config['dataset_name']]['task']

    for key, value in vars(args).items():
        # 仅当命令行参数与现有配置键匹配时才更新
        # Only update if the command-line argument matches an existing config key
        if key in current_config:
            current_config[key] = value


    return current_config