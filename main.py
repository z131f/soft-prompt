from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

prompt = "在这里输入你的问题"
url = "https://www.ilankelman.org/stopsigns/australia.jpg" # 替换成你的图片 URL
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

# 生成
generate_ids = model.generate(**inputs)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))