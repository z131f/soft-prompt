import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse # 导入argparse模块，用于处理命令行参数

# 创建一个参数解析器
parser = argparse.ArgumentParser(description="运行模型训练和评估，可指定CUDA设备。")
# 添加一个名为'--cuda_devices'的命令行参数
# nargs='?'表示参数是可选的，如果提供了，可以有0个或1个值
# default=None表示如果命令行没有提供这个参数，它的默认值是None
parser.add_argument('--cuda_devices', type=str, nargs='?', default=None,
                    help='指定可见的CUDA设备，例如：0,1,2。')

# 解析命令行参数
args = parser.parse_args()

# 检查是否从命令行接收到cuda_devices参数
if args.cuda_devices:
    # 如果接收到了，就使用命令行提供的设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    print(f"使用命令行指定的CUDA设备: {args.cuda_devices}")
else:
    # 否则，使用默认设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    print("使用默认的CUDA设备: 5,6,7")

from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics

config = build_config()
logger = get_logger()
set_seed(config['seed'])

if config['model_name'] == 'llava_next':
    from scripts.llava_next import llava_next_trainer
    trainer = llava_next_trainer(config, logger, action='skip')
    trainer.train()
    trainer.eval()