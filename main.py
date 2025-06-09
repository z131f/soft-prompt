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
from model.LlavaNext import ModifiedLlavaNext
from utils import custom_collate_fn, compute_metrics
from functools import partial


# --- 日志配置修改开始 ---

# 获取logger实例，或者根logger（根据您的具体需求，这里使用您之前的方法__name__）
# 如果您想确保所有Trainer的日志也通过此logger，最好使用root_logger = logging.getLogger()
logger_to_configure = logging.getLogger(__name__) # 假设您仍想使用模块名作为logger名称
# 或者使用 root_logger = logging.getLogger() 来捕获所有日志

# 设置最低日志级别为INFO
logger_to_configure.setLevel(logging.INFO)

# 关键步骤：在每次配置前，移除所有现有的 handlers
for handler in logger_to_configure.handlers[:]: # 遍历handlers的副本，避免在迭代时修改列表
    logger_to_configure.removeHandler(handler)
    handler.close() # 关闭 handler 释放资源，尤其是FileHandler会占用文件句柄

# 1. 创建一个 StreamHandler (用于控制台输出)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # 设置控制台输出的最低级别

# 2. 创建一个 FileHandler (用于文件输出)
log_file_path = "app.log" # 定义日志文件名称
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True) # 如果logs目录不存在，则创建
file_handler = logging.FileHandler(os.path.join(log_dir, log_file_path))
file_handler.setLevel(logging.INFO) # 设置文件输出的最低级别

# 创建一个 Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 为两个handler设置Formatter
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将两个handler添加到logger
logger_to_configure.addHandler(console_handler)
logger_to_configure.addHandler(file_handler)
logger = logger_to_configure # 确保使用配置后的logger

# logger.info('test1')

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
# processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1)
# print(processor.tokenizer.eos_token_id)   2 # 2 是 LlavaNext 的 EOS token ID

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
    use_num=5000
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
    use_num=1000, # 使用部分数据进行评估，方便调试
    is_eval=True,  # 设置为评估模式
)

model = ModifiedLlavaNext.from_pretrained(model_name, config=config, cache_dir=model_cache, device_map="balanced", patch_num=patch_num, torch_dtype=torch.bfloat16)

training_args = TrainingArguments(
    output_dir="./output/llava_next_modified_finetune", # 训练输出目录
    num_train_epochs=1,                                # 训练轮数
    per_device_train_batch_size=3,                     # 每个设备的训练批量大小
    per_device_eval_batch_size=2,                      # 每个设备的评估批量大小
    gradient_accumulation_steps=1,                     # 梯度累积步数，模拟更大的batch size
    # learning_rate=5e-5,                                # 学习率
    learning_rate=1e-4,                                # 学习率
    weight_decay=0.01,                                 # 权重衰减
    logging_dir="logs",                              # 日志目录
    logging_steps=5,                                  # 每隔多少步记录一次日志
    save_steps=500,                                    # 每隔多少步保存一次模型
    save_total_limit=2,                                # 最多保存的模型数量
    do_train=True,                                     # 执行训练
    report_to="none",                                  # 不报告到任何平台 (可选，如果你想用wandb等可以设置)
    bf16=True                                         # 使用半精度训练
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

class CustomGenerationTrainer(Trainer):
    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        # inputs 包含 input_ids, attention_mask, labels 等
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"] # 真实的标签，也需要传递

        # 调用 model.generate() 来获取生成的 token ID
        # 确保传递 max_new_tokens 和其他生成参数
        # 注意：这里只生成了回答部分，如果你的任务需要模型从头生成整个序列，
        # 那么 input_ids 可能是空的或者只包含起始 token
        generate_ids = model.generate(**inputs, max_new_tokens=100, pad_token_id=2)

        # prediction_loss_only 为 True 时，只返回 loss
        if prediction_loss_only:
            # 对于生成任务，通常我们不直接计算生成ID的loss，而是通过generate来获取
            # 如果需要loss，可以单独计算
            return (None, None, None) # 返回 (loss, predictions, labels)

        # 返回生成的 token ID 和真实的标签 ID
        # 注意：generated_ids 可能包含 input_ids 的前缀，需要根据情况处理
        # 如果你希望 predictions 只是生成的答案部分，可能需要切片
        # 这里我们假设 generated_ids 包含了完整的序列（prompt + answer）
        # 并且我们只关心 answer 部分的评估，所以需要对 generated_ids 进行处理
        # 最简单的方式是直接返回 generated_ids，然后在 compute_metrics 中处理
        return (None, generate_ids, labels) # loss 为 None，因为我们不在这里计算评估损失

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
    RSVQA_LR_Eval_Dataset_instance=RSVQA_LR_Eval_Dataset_instance
)
trainer = CustomGenerationTrainer(
    model=model,
    args=training_args,
    train_dataset=RSVQA_LR_Dataset_instance,
    data_collator=custom_collate_fn, # 使用自定义的 collate_fn
    eval_dataset=RSVQA_LR_Eval_Dataset_instance, # 添加评估数据集
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

# 加载用于语义相似度的句子 transformer 模型
# 'all-MiniLM-L6-v2' 在速度和性能之间取得了很好的平衡。
# 如果计算资源允许，你可以考虑使用 'all-mpnet-base-v2' 以获得更高的准确性。

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
