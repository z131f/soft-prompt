from PIL import Image
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, CLIPProcessor, AutoModelForImageTextToText

# 原始模型叫做 "liuhaotian/llava-v1.5-7b", 我已经把它下载到本地目录 model_bank 中
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
device = "cuda:1"  # 定义使用的设备（GPU）
model_cache = "model_cache"  # 定义模型缓存路径

# 加载处理器（AutoProcessor），用于处理输入数据
processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(model_name, cache_dir=model_cache, device_map="auto")

# 定义输入的提示文本，包含用户和助手的对话格式
prompt = "USER: <image>\n What's the content of the image? ASSISTANT:"

# 定义图像文件的路径
url = "1.jpg"  # .image_data/test_image_1.jpg

# 打开图像文件
image = Image.open(fp=url)

# 使用处理器处理文本和图像，生成模型输入
inputs = processor(text=prompt, images=image, return_tensors="pt")

# 将输入数据移动到GPU上
for temp_key in inputs.keys():
    inputs[temp_key] = inputs[temp_key].to("cuda:0")

# 使用模型生成输出，限制生成的最大token数为50
generate_ids = model.generate(**inputs, max_new_tokens=100)

# 将生成的token解码为文本，跳过特殊token并清理空格
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# 打印模型的响应
print("\n\nresponse:\n", response)
