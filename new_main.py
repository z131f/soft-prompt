from config import build_config
from utils import get_logger, print_trainable_parameters, set_seed, custom_collate_fn, compute_metrics

config = build_config()
logger = get_logger()
set_seed(config['seed'])

if config['model_name'] == 'llava_next':
    from scripts.llava_next import llava_next_trainer
    trainer = llava_next_trainer(config, logger)
    trainer.train()
    trainer.eval()