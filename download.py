# Load model directly
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoProcessor, AutoModelForCausalLM
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
model_cache = "model_cache"

processor = AutoProcessor.from_pretrained("BigData-KSU/RS-llava-v1.5-7b-LoRA", cache_dir=model_cache)
model = AutoModelForCausalLM.from_pretrained("BigData-KSU/RS-llava-v1.5-7b-LoRA", cache_dir=model_cache)