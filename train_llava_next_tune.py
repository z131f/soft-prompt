from config import build_config
from trainer.utils import load_model_trainer

config, logger = build_config()

assert config['model_name'] == 'llava_next_tune', "This script is for llava_next_tune only."
assert config['action'] == 'begin', "This script is for training only."

trainer = load_model_trainer(config, logger)

trainer.train()
trainer.eval()