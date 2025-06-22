model_name_map = {
    'llava_next_tune': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_next': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_rs': 'BigData-KSU/RS-llava-v1.5-7b-LoRA',
    'llava_rs_tune': 'BigData-KSU/RS-llava-v1.5-7b-LoRA',
    'qwen_vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'qwen_vl_tune': 'Qwen/Qwen2.5-VL-7B-Instruct'
}

def load_model_trainer(config, logger):
    config['model_site'] = model_name_map[config['model_name']]
    if config['model_name'] == 'llava_next_tune':
        from trainer.llava_next_tune import llava_next_tune_trainer
        return llava_next_tune_trainer(config, logger)
    elif config['model_name'] == 'llava_next':
        from trainer.llava_next import llava_next_trainer
        return llava_next_trainer(config, logger)
    elif config['model_name'] == 'llava_rs':
        from trainer.llava_rs import llava_rs_trainer
        return llava_rs_trainer(config, logger)
    elif config['model_name'] == 'llava_rs_tune':
        from trainer.llava_rs_tune import llava_rs_tune_trainer
        return llava_rs_tune_trainer(config, logger)
    elif config['model_name'] == 'qwen_vl':
        from trainer.qwen_vl import qwen_vl_trainer
        return qwen_vl_trainer(config, logger)
    elif config['model_name'] == 'qwen_vl_tune':
        from trainer.qwen_vl_tune import qwen_vl_tune_trainer
        return qwen_vl_tune_trainer(config, logger)
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")