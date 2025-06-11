import os
# 设置环境变量，使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging # 导入 logging 以便更好地处理错误
from dataset.RSVQA_LR_Dataset import RSVQA_LR_Dataset
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig
import torch.nn as nn
from transformers.utils import is_torchdynamo_compiling
import numpy as np
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LlavaNextCausalLMOutputWithPast
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
# 引入训练所需的库
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset # 用于自定义数据集
from tqdm import tqdm  # 用于显示进度条
from model.llava_next_tune import ModifiedLlavaNext
from utils import custom_collate_fn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = 'cuda'
# 原始模型名称和缓存路径
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
model_cache = "model_cache"

# 加载原始模型的配置
config = AutoConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)
# 加载处理器
# 增加一个额外的token用于处理图像输入
processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1)

image_folder_path="DATA/RSVQA-LR/images"  # 替换为实际的图像文件夹路径
image_size=(256, 256)  # 替换为实际的图像大小
patch_num=image_size_to_num_patches(image_size=image_size,grid_pinpoints=config.image_grid_pinpoints,patch_size=config.vision_config.image_size)
print(f"patch_num: {patch_num}")
# 图像处理参数

eval_questions_json_path = "DATA/RSVQA-LR/test/LR_split_test_questions.json"
eval_answers_json_path = "DATA/RSVQA-LR/test/LR_split_test_answers.json"
eval_images_json_path = "DATA/RSVQA-LR/test/LR_split_test_images.json"

RSVQA_LR_Eval_Dataset_instance = RSVQA_LR_Dataset(
    processor=processor,
    image_folder_path=image_folder_path,
    questions_json_path=eval_questions_json_path,
    answers_json_path=eval_answers_json_path,
    images_json_path=eval_images_json_path,
    image_size=image_size,
    logger=logger,
    use_num=1000, # 使用部分数据进行评估，方便调试
    add_instruct=True,
    is_eval=True  # 设置为评估模式
)

model = LlavaNextForConditionalGeneration.from_pretrained(model_name, config=config, cache_dir=model_cache, device_map="balanced", torch_dtype=torch.bfloat16)

# --- 训练后的测试部分 (可选) ---
print("\n--- 评估训练结果 ---")

# 确保模型处于评估模式
model.eval()

# 加载用于语义相似度的句子 transformer 模型
# 'all-MiniLM-L6-v2' 在速度和性能之间取得了很好的平衡。
# 如果计算资源允许，你可以考虑使用 'all-mpnet-base-v2' 以获得更高的准确性。
print("正在加载用于语义相似度的句子 transformer 模型...")
sentence_model = SentenceTransformer('all-mpnet-base-v2')
print("句子 transformer 模型加载完成。")

correct_predictions_semantic = 0
total_predictions = 0
semantic_similarity_threshold = 0.75 # 根据你的需求调整此阈值

# 可以使用DataLoader进行批量处理，但为了简单起见，我们这里直接遍历评估数据集
# 如果你的评估数据集很大，请考虑使用 DataLoader
eval_dataloader = torch.utils.data.DataLoader(RSVQA_LR_Eval_Dataset_instance, batch_size=1, collate_fn=custom_collate_fn)
answer_list = RSVQA_LR_Eval_Dataset_instance.get_answers_list() # 获取答案列表

for idx, batch in tqdm(enumerate(eval_dataloader), desc="正在评估"):
    # 将批次数据移动到设备上
    inputs_eval = {k: v.to(device) for k, v in batch.items() if k != 'answer_text'}
    ground_truth_answer = answer_list[idx] # 假设批处理大小为1，简化处理

    with torch.no_grad():
        generate_ids = model.generate(**inputs_eval, max_new_tokens=100)
        predicted_answer = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # 清理预测答案
        # 提示词是 "USER: <image>\n What's the content of the image? ASSISTANT:"
        # 模型可能会生成整个提示词+答案
        if "ASSISTANT:" in predicted_answer:
            predicted_answer = predicted_answer.split("ASSISTANT:")[1].strip()
        elif "USER:" in predicted_answer: # 如果模型生成了意外的前缀
            predicted_answer = predicted_answer.split("USER:")[-1].strip()

    print(f"\n问题: {processor.decode(inputs_eval['input_ids'][0], skip_special_tokens=True)}")
    print(f"预测答案: {predicted_answer}")
    print(f"真实答案: {ground_truth_answer}")

    # --- 语义相似度评分 ---
    # 将答案编码以获取它们的嵌入向量
    embeddings1 = sentence_model.encode(predicted_answer, convert_to_tensor=True)
    embeddings2 = sentence_model.encode(ground_truth_answer, convert_to_tensor=True)

    # 计算余弦相似度
    cosine_similarity = util.cos_sim(embeddings1, embeddings2).item()
    print(f"语义相似度: {cosine_similarity:.4f}")

    # 检查相似度是否高于阈值
    if cosine_similarity >= semantic_similarity_threshold:
        correct_predictions_semantic += 1
    total_predictions += 1

accuracy_semantic = (correct_predictions_semantic / total_predictions) * 100 if total_predictions > 0 else 0
print(f"\n--- 评估总结 (基于语义相似度) ---")
print(f"总评估样本数: {total_predictions}")
print(f"语义相似度正确预测数 (相似度 > {semantic_similarity_threshold}): {correct_predictions_semantic}")
print(f"语义相似度准确率: {accuracy_semantic:.2f}%")