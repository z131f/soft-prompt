import os
# 设置环境变量，使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging # 导入 logging 以便更好地处理错误
# from dataset.RSVQA_LR_Dataset import RSVQA_LR_Dataset
from dataset.utils import load_dataset
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
from PIL import Image
import torch
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig
import torch.nn as nn
from transformers.utils.import_utils import is_torchdynamo_compiling
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LlavaNextCausalLMOutputWithPast
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
# 引入训练所需的库
from transformers.trainer import Trainer, TrainingArguments
from torch.utils.data import Dataset # 用于自定义数据集
from tqdm import tqdm  # 用于显示进度条
from model.llava_next_tune import ModifiedLlavaNext
from utils import custom_collate_fn, compute_metrics, set_seed, get_logger, print_trainable_parameters
from functools import partial
from config import build_config

config = build_config()

config = {
    'dataset_name': 'RSVQA_LR',
    'model_name': 'llava_next',
    'seed': 42,
    'skip_training': False,
    'load_train_num': 1000,
    'load_test_num': 1000,
    'prompt_num': 10,
    'lr': 5e-5,
}

seed = config['seed']
dataset_num = config['dataset_name']
model_site = config['model_site']
skip_training = config['skip_training']
load_train_num = config['load_train_num']
load_test_num = config['load_test_num']
prompt_num = config['prompt_num']
lr = config['lr']

set_seed(seed)
print(f"随机种子已设置为: {seed}")

logger = get_logger()

# logger.info('test1')

device = 'cuda'
# 原始模型名称和缓存路径
model_cache = "model_cache"

# 加载原始模型的配置
config = AutoConfig.from_pretrained(model_site, cache_dir=model_cache, trust_remote_code=True)
# 加载处理器
# 增加一个额外的token用于处理图像输入
processor = LlavaNextProcessor.from_pretrained(model_site, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1 + 1)
# processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1)
# print(processor.tokenizer.eos_token_id)   2 # 2 是 LlavaNext 的 EOS token ID

# image_folder_path="DATA/RSVQA-LR/images"  # 替换为实际的图像文件夹路径
# questions_json_path="DATA/RSVQA-LR/train/LR_split_train_questions.json"  # 替换为实际的 JSON 文件路径
# answers_json_path="DATA/RSVQA-LR/train/LR_split_train_answers.json"  # 替换为实际的 JSON 文件路径
# images_json_path="DATA/RSVQA-LR/train/LR_split_train_images.json"  # 替换为实际的 JSON 文件路径
# image_size=(256, 256)  # 替换为实际的图像大小
# # image_size=(512, 512)
# patch_num=image_size_to_num_patches(image_size=image_size,grid_pinpoints=config.image_grid_pinpoints,patch_size=config.vision_config.image_size)
# print(f"patch_num: {patch_num}")
# 图像处理参数
# image_size = (672, 672)
# # patch_num = image_size[0] // 336 * image_size[1] // 336 + 1
# patch_num = image_size_to_num_patches(
#                 image_size=image_size,
#                 grid_pinpoints=config.image_grid_pinpoints,
#                 patch_size=config.vision_config.image_size,
#             )

# RSVQA_LR_Dataset_instance = RSVQA_LR_Dataset(
#     processor=processor,  # 这里需要传入实际的处理器实例
#     image_folder_path=image_folder_path,
#     questions_json_path=questions_json_path,
#     answers_json_path=answers_json_path,
#     images_json_path=images_json_path,
#     image_size=image_size,
#     logger=logger,
#     use_num=1500
# )

RSVQA_LR_Dataset_train = load_dataset(
    dataset_name="RSVQA-LR",
    is_eval=False,
    add_instruct=False,
    load_num=load_train_num,
    type="train",
    processor=processor,
)

# eval_questions_json_path = "DATA/RSVQA-LR/test/LR_split_test_questions.json"
# eval_answers_json_path = "DATA/RSVQA-LR/test/LR_split_test_answers.json"
# eval_images_json_path = "DATA/RSVQA-LR/test/LR_split_test_images.json"

# RSVQA_LR_Dataset_eval = RSVQA_LR_Dataset(
#     processor=processor,
#     image_folder_path=image_folder_path,
#     questions_json_path=eval_questions_json_path,
#     answers_json_path=eval_answers_json_path,
#     images_json_path=eval_images_json_path,
#     image_size=image_size,
#     logger=logger,
#     use_num=1000, # 使用部分数据进行评估，方便调试
#     is_eval=True,  # 设置为评估模式
# )
RSVQA_LR_Dataset_eval = load_dataset(
    dataset_name="RSVQA-LR",
    is_eval=True,
    add_instruct=False,
    load_num=load_test_num,
    type="test",
    processor=processor,
)

prompt_num = 10

model = ModifiedLlavaNext.from_pretrained(model_site, config=config, cache_dir=model_cache, device_map="balanced", prompt_num=prompt_num, torch_dtype=torch.bfloat16)

training_args = TrainingArguments(
    output_dir="./output/llava_next_modified_finetune", # 训练输出目录
    num_train_epochs=1,                                # 训练轮数
    per_device_train_batch_size=4,                     # 每个设备的训练批量大小
    per_device_eval_batch_size=2,                      # 每个设备的评估批量大小
    gradient_accumulation_steps=1,                     # 梯度累积步数，模拟更大的batch size
    learning_rate=lr,                                # 学习率
    # learning_rate=1e-4,                                # 学习率 学习率过大
    weight_decay=0.01,                                 # 权重衰减
    logging_dir="logs",                              # 日志目录
    logging_steps=2,                                  # 每隔多少步记录一次日志
    save_steps=500,                                    # 每隔多少步保存一次模型
    save_total_limit=2,                                # 最多保存的模型数量
    do_train=True,                                     # 执行训练
    report_to="none",                                  # 不报告到任何平台 (可选，如果你想用wandb等可以设置)
    bf16=True,                                         # 使用半精度训练
    seed=seed,
)
    # 评估相关参数
'''
    eval_strategy="steps",                        # 每隔一定步数进行评估
    eval_accumulation_steps=2,                     # 评估时的梯度累积步数
    eval_steps=100,                                    # 每隔 500 步进行一次评估
    load_best_model_at_end=True,                       # 训练结束后加载在评估集上表现最好的模型
    metric_for_best_model="semantic_accuracy",         # 用于判断最佳模型的指标
    greater_is_better=True,# 该指标是越大越好
'''


# 在训练前打印
print("Before training:")

print("正在加载用于语义相似度的句子 transformer 模型...")
sentence_model = SentenceTransformer('all-mpnet-base-v2')
print("句子 transformer 模型加载完成。")
model.set_train()
# # 4. 创建 Trainer 实例
compute_metrics_with_args = partial(
    compute_metrics, 
    logger=logger, 
    sentence_model=sentence_model, 
    processor=processor,
    RSVQA_LR_Eval_Dataset_instance=RSVQA_LR_Dataset_eval
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RSVQA_LR_Dataset_train,
    data_collator=custom_collate_fn, # 使用自定义的 collate_fn
    eval_dataset=RSVQA_LR_Dataset_eval, # 添加评估数据集
    compute_metrics=compute_metrics_with_args,             # 添加评估指标计算函数
)
print_trainable_parameters(trainer.model)

if not skip_training:
    print("Starting training...")
    trainer.train()

    # 5. 开始训练
    print("Training finished!")
    
    for obj in trainer.state.log_history:
        logger.info(str(obj))

    # # 6. 保存训练好的模型（只有新增的模块）
    output_dir = "./output/llava_next_modified_finetune/trained_modules"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.soft_prompts, os.path.join(output_dir, "soft_prompts.pt"))
    torch.save(model.q_projector_image.state_dict(), os.path.join(output_dir, "q_projector_image.pt"))
    torch.save(model.k_projector_image.state_dict(), os.path.join(output_dir, "k_projector_image.pt"))
    torch.save(model.v_projector_image.state_dict(), os.path.join(output_dir, "v_projector_image.pt"))
    torch.save(model.k_projector_prompt.state_dict(), os.path.join(output_dir, "k_projector_prompt.pt"))
    torch.save(model.v_projector_prompt.state_dict(), os.path.join(output_dir, "v_projector_prompt.pt"))
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
        q_projector_image_path = os.path.join(loaded_model_dir, "q_projector_image.pt")
        k_projector_image_path = os.path.join(loaded_model_dir, "k_projector_image.pt")
        v_projector_image_path = os.path.join(loaded_model_dir, "v_projector_image.pt")
        k_projector_prompt_path = os.path.join(loaded_model_dir, "k_projector_prompt.pt")
        v_projector_prompt_path = os.path.join(loaded_model_dir, "v_projector_prompt.pt")
        global_prompt_self_attention_path = os.path.join(loaded_model_dir, "global_prompt_self_attention.pt")

        if not os.path.exists(soft_prompts_path):
            raise FileNotFoundError(f"soft_prompts.pt not found in {loaded_model_dir}")
        if not os.path.exists(global_prompt_self_attention_path):
            raise FileNotFoundError(f"global_prompt_self_attention.pt not found in {loaded_model_dir}")

        # 加载权重
        model.soft_prompts = torch.load(soft_prompts_path, map_location=device)
        model.q_projector_image.load_state_dict(torch.load(q_projector_image_path, map_location=device))
        model.k_projector_image.load_state_dict(torch.load(k_projector_image_path, map_location=device))
        model.v_projector_image.load_state_dict(torch.load(v_projector_image_path, map_location=device))
        model.k_projector_prompt.load_state_dict(torch.load(k_projector_prompt_path, map_location=device))
        model.v_projector_prompt.load_state_dict(torch.load(v_projector_prompt_path, map_location=device))
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

# 加载用于语义相似度的句子 transformer 模型
# 'all-MiniLM-L6-v2' 在速度和性能之间取得了很好的平衡。
# 如果计算资源允许，你可以考虑使用 'all-mpnet-base-v2' 以获得更高的准确性。

correct_predictions_semantic = 0
total_predictions = 0
semantic_similarity_threshold = 0.75 # 根据你的需求调整此阈值

# 可以使用DataLoader进行批量处理，但为了简单起见，我们这里直接遍历评估数据集
# 如果你的评估数据集很大，请考虑使用 DataLoader
eval_dataloader = torch.utils.data.DataLoader(RSVQA_LR_Dataset_eval, batch_size=1, collate_fn=custom_collate_fn)
answer_list = RSVQA_LR_Dataset_eval.get_answers_list() # 获取答案列表

for idx, batch in tqdm(enumerate(eval_dataloader), desc="正在评估"):
    # 将批次数据移动到设备上
    inputs_eval = {k: v.to(device) for k, v in batch.items() if k != 'answer_text'}
    ground_truth_answer = answer_list[idx] # 假设批处理大小为1，简化处理

    with torch.no_grad():
        generate_ids = model.generate(**inputs_eval, max_new_tokens=100)
        # [1,3806]
        predicted_answer = processor.batch_decode( # type: ignore
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # 清理预测答案
        # 提示词是 "USER: <image>\n What's the content of the image? ASSISTANT:"
        # 模型可能会生成整个提示词+答案
        if "ASSISTANT:" in predicted_answer:
            predicted_answer = predicted_answer.split("ASSISTANT:")[1].strip()
        elif "USER:" in predicted_answer: # 如果模型生成了意外的前缀
            predicted_answer = predicted_answer.split("USER:")[-1].strip()

    print(f"\n问题: {processor.decode(inputs_eval['input_ids'][0], skip_special_tokens=True)}") # type: ignore
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
