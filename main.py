import os
# 设置环境变量，使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging # 导入 logging 以便更好地处理错误
from dataset.RSVQA_LR_Dataset import RSVQA_LR_Dataset
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
from PIL import Image
import torch
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
from model.LlavaNext import ModifiedLlavaNext
from utils import custom_collate_fn


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = 'cuda'
skip_training = False
# 原始模型名称和缓存路径
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
model_cache = "model_cache"

# 加载原始模型的配置
config = AutoConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)
# 加载处理器
# 增加一个额外的token用于处理图像输入
processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1 + 1)

image_folder_path="DATA/RSVQA-LR/images"  # 替换为实际的图像文件夹路径
questions_json_path="DATA/RSVQA-LR/train/LR_split_train_questions.json"  # 替换为实际的 JSON 文件路径
answers_json_path="DATA/RSVQA-LR/train/LR_split_train_answers.json"  # 替换为实际的 JSON 文件路径
images_json_path="DATA/RSVQA-LR/train/LR_split_train_images.json"  # 替换为实际的 JSON 文件路径
image_size=(256, 256)  # 替换为实际的图像大小
patch_num=image_size_to_num_patches(image_size=image_size,grid_pinpoints=config.image_grid_pinpoints,patch_size=config.vision_config.image_size)
print(f"patch_num: {patch_num}")
# 图像处理参数
# image_size = (672, 672)
# # patch_num = image_size[0] // 336 * image_size[1] // 336 + 1
# patch_num = image_size_to_num_patches(
#                 image_size=image_size,
#                 grid_pinpoints=config.image_grid_pinpoints,
#                 patch_size=config.vision_config.image_size,
#             )

RSVQA_LR_Dataset_instance = RSVQA_LR_Dataset(
    processor=processor,  # 这里需要传入实际的处理器实例
    image_folder_path=image_folder_path,
    questions_json_path=questions_json_path,
    answers_json_path=answers_json_path,
    images_json_path=images_json_path,
    image_size=image_size,
    logger=logger,
    use_num=500
)

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
    use_num=30, # 使用部分数据进行评估，方便调试
    is_eval=True,  # 设置为评估模式
)

model = ModifiedLlavaNext.from_pretrained(model_name, config=config, cache_dir=model_cache, device_map="balanced", patch_num=patch_num, torch_dtype=torch.bfloat16)

training_args = TrainingArguments(
    output_dir="./output/llava_next_modified_finetune", # 训练输出目录
    num_train_epochs=1,                                # 训练轮数
    # num_train_epochs=15,
    per_device_train_batch_size=2,                     # 每个设备的训练批量大小
    gradient_accumulation_steps=4,                     # 梯度累积步数，模拟更大的batch size
    learning_rate=5e-4,                                # 学习率
    weight_decay=0.01,                                 # 权重衰减
    logging_dir="./logs",                              # 日志目录
    logging_steps=10,                                  # 每隔多少步记录一次日志
    save_steps=500,                                    # 每隔多少步保存一次模型
    save_total_limit=2,                                # 最多保存的模型数量
    do_train=True,                                     # 执行训练
    report_to="none",                                  # 不报告到任何平台 (可选，如果你想用wandb等可以设置)
    bf16=True,                                       # 使用半精度训练
)

def print_trainable_parameters(model):
    print("--------------------------------------------------")
    print("Trainable Parameters:")
    trainable_params_count = 0
    all_params_count = 0
    for name, param in model.named_parameters():
        all_params_count += param.numel()
        if param.requires_grad:
            trainable_params_count += param.numel()
            print(f"  - {name} (Shape: {param.shape})")

    print(f"\nTotal trainable parameters: {trainable_params_count}")
    print(f"Total parameters: {all_params_count}")
    if all_params_count > 0:
        print(f"Trainable percentage: {100 * trainable_params_count / all_params_count:.2f}%")
    print("--------------------------------------------------")

# 在训练前打印
print("Before training:")
model.set_train()
# # 4. 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RSVQA_LR_Dataset_instance,
    data_collator=custom_collate_fn, # 使用自定义的 collate_fn
)
print_trainable_parameters(trainer.model)

if not skip_training:
    print("Starting training...")
    trainer.train()

    # 5. 开始训练
    print("Training finished!")

    # # 6. 保存训练好的模型（只有新增的模块）
    output_dir = "./output/llava_next_modified_finetune/trained_modules"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.soft_prompts, os.path.join(output_dir, "soft_prompts.pt"))
    torch.save(model.linear_projector.state_dict(), os.path.join(output_dir, "linear_projector.pt"))
    torch.save(model.global_prompt_self_attention.state_dict(), os.path.join(output_dir, "global_prompt_self_attention.pt"))
    print(f"新模块权重已保存到：{output_dir}")
else:
    loaded_model_dir = "./output/llava_next_modified_finetune/trained_modules"
    print(f"Skipping training. Attempting to load model from {loaded_model_dir}...")
    try:
        # 确保加载的模型目录存在
        if not os.path.exists(loaded_model_dir):
            raise FileNotFoundError(f"Model directory not found: {loaded_model_dir}")

        # 检查每个权重文件是否存在
        soft_prompts_path = os.path.join(loaded_model_dir, "soft_prompts.pt")
        linear_projector_path = os.path.join(loaded_model_dir, "linear_projector.pt")
        global_prompt_self_attention_path = os.path.join(loaded_model_dir, "global_prompt_self_attention.pt")

        if not os.path.exists(soft_prompts_path):
            raise FileNotFoundError(f"soft_prompts.pt not found in {loaded_model_dir}")
        if not os.path.exists(linear_projector_path):
            raise FileNotFoundError(f"linear_projector.pt not found in {loaded_model_dir}")
        if not os.path.exists(global_prompt_self_attention_path):
            raise FileNotFoundError(f"global_prompt_self_attention.pt not found in {loaded_model_dir}")

        # 加载权重
        model.soft_prompts = torch.load(soft_prompts_path, map_location=device)
        model.linear_projector.load_state_dict(torch.load(linear_projector_path, map_location=device))
        model.global_prompt_self_attention.load_state_dict(torch.load(global_prompt_self_attention_path, map_location=device))
        print("Model weights loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}. Please ensure the correct path and saved files.")
        print("Exiting as training is skipped and model loading failed.")
        exit() # 如果加载失败，直接退出程序
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        print("Exiting as training is skipped and model loading failed.")
        exit()

# --- 训练后的测试部分 (可选) ---
print("\n--- 评估训练结果 ---")

# 确保模型处于评估模式
model.eval()

correct_predictions = 0
total_predictions = 0

# 可以使用DataLoader进行批量处理，但为了简单起见，我们这里直接遍历评估数据集
# 如果你的评估数据集很大，请考虑使用 DataLoader
eval_dataloader = torch.utils.data.DataLoader(RSVQA_LR_Eval_Dataset_instance, batch_size=1, collate_fn=custom_collate_fn)
answer_list = RSVQA_LR_Eval_Dataset_instance.get_answers_list()  # 获取答案列表

for idx, batch in tqdm(enumerate(eval_dataloader), desc="正在评估"):
    # 将批次数据移动到设备上
    inputs_eval = {k: v.to(device) for k, v in batch.items() if k != 'answer_text'}
    ground_truth_answer = answer_list[idx] # 假设批处理大小为1，简化处理

    with torch.no_grad():
        generate_ids = model.generate(**inputs_eval, max_new_tokens=100)
        predicted_answer = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print(f"预测答案: {predicted_answer}")
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


    # 简单的精确匹配准确率
    # 对于VQA，你可能需要更复杂的答案比较（例如，基于语义相似度或VQA官方评分）
    if predicted_answer.lower() == ground_truth_answer.lower():
        correct_predictions += 1
    total_predictions += 1

accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
print(f"\n--- 评估总结 ---")
print(f"总评估样本数: {total_predictions}")
print(f"正确预测数: {correct_predictions}")
print(f"准确率: {accuracy:.2f}%")