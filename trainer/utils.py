model_name_map = {
    'llava_next_tune': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_next': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_rs': 'BigData-KSU/RS-llava-v1.5-7b-LoRA'
}

def load_model_trainer(config, logger):
    config['model_site'] = model_name_map[config['model_name']]
    if config['model_name'] == 'llava_next_tune':
        from trainer.llava_next_tune import llava_next_tune_trainer
        return llava_next_tune_trainer(config, logger)
    elif config['model_name'] == 'llava_next':
        from trainer.llava_next import llava_next_trainer
        return llava_next_trainer(config, logger)
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")