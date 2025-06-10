import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import random
import os
import logging


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

def get_logger():
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
    return logger

def set_seed(seed: int):
    """
    固定所有可能产生随机性的操作的种子。
    """
    random.seed(seed) # Python 随机模块
    np.random.seed(seed) # Numpy
    torch.manual_seed(seed) # PyTorch CPU
    torch.cuda.manual_seed_all(seed) # PyTorch CUDA (针对所有 GPU)
    # torch.backends.cudnn.deterministic = True # 如果您的模型对CUDNN的确定性有要求，可以取消注释
    # torch.backends.cudnn.benchmark = False # 同上
    os.environ['PYTHONHASHSEED'] = str(seed) # Python 哈希种子，影响字典和集合的哈希
    # 对于 transformers 库，Trainer 也有一个 seed 参数
    # 但为了全面性，我们在这里设置底层的库


def custom_collate_fn(batch):
    # 过滤掉 None 值（如果 CustomDataset 返回了 None）
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    # 从批次中的第一个元素获取所有键
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if key == "pixel_values":
            # 对于 pixel_values，它们可能是不同数量的图像块，所以需要特殊处理
            # 这里的 pixel_values 已经是预处理后的 5D 张量 (batch_size, num_patches, C, H, W)
            # 或者 4D 张量 (num_patches, C, H, W)
            # 如果是 5D，可以直接堆叠
            # 如果是 4D，并且每个样本的 patch 数量不同，则需要 pad 或使用 list
            # 在 Llava-Next 中，通常是 5D 张量，其中 num_patches 是固定值（4 + 1）
            collated_batch[key] = torch.stack([x[key] for x in batch])
        elif key == "image_sizes":
            collated_batch[key] = torch.stack([x[key] for x in batch])
        else:
            # 对于 input_ids, attention_mask, labels，它们都是张量，需要进行pad
            # 使用 processor 的 tokenizer 的 pad 方法

            # processor.tokenizer.pad_token_id: 32001
            collated_batch[key] = torch.nn.utils.rnn.pad_sequence([x[key] for x in batch], 
                                                                   batch_first=True, 
                                                                   padding_value=32001)
            # 对于 attention_mask，padding_value 应该是 0
            if key == "attention_mask":
                collated_batch[key][collated_batch[key] == 32001] = 0
            # 对于 labels，padding_value 应该是 -100
            elif key == "labels":
                collated_batch[key][collated_batch[key] == 32001] = -100
    
    return collated_batch


def compute_metrics(eval_pred, logger, sentence_model, processor, RSVQA_LR_Eval_Dataset_instance):
    """
    计算评估指标。
    eval_pred 包含模型预测的 logits 和真实的标签。
    """
    predictions = eval_pred.predictions # 现在 predictions 应该是 generated_ids
    label_ids = eval_pred.label_ids

    # 解码预测的 token ID
    # predictions 可能包含 prompt 的部分，需要根据你的模型和任务进行调整
    # 如果 generated_ids 包含了 input_ids 的前缀，你可能需要移除它
    # 比如：decoded_preds = tokenizer.batch_decode(predictions[:, input_len:], ...)
    # 但最简单的是让 generate 负责生成完整的序列，然后在这里解码
    predicted_answers = processor.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # 解码真实的 token ID (从 label_ids 转换为原始文本答案)
    # 注意: 这里需要确保 label_ids 能够被正确解码为原始答案文本
    # 如果你的 label_ids 是针对整个输入序列的，你可能需要提取答案部分
    # 或者，如果你的 eval_dataset 存储了原始答案，你可以从数据集中获取
    
    # 假设 RSVQA_LR_Eval_Dataset_instance 能够提供根据索引获取原始答案的方法
    # 注意：在 compute_metrics 中直接访问数据集的索引会比较复杂，
    # 更好的方式是在 eval_dataset 的 __getitem__ 中把原始答案也带出来
    # 或者在 Trainer 的 predict 方法中拿到对应的 ground_truth。
    # 为了简化，我们这里假设 eval_pred 能够包含原始答案文本，或者我们能通过某种方式获取。
    # 暂时使用一个占位符，实际应用中你需要调整这里。
    
    # 获取评估数据集的真实答案列表
    ground_truth_answers = RSVQA_LR_Eval_Dataset_instance.get_answers_list()
    
    total_samples = len(predicted_answers)
    correct_semantic_predictions = 0
    semantic_similarity_threshold = 0.75

    all_similarities = []

    for i in range(total_samples):
        predicted_answer = predicted_answers[i]
        # 清理预测答案
        if "ASSISTANT:" in predicted_answer:
            predicted_answer = predicted_answer.split("ASSISTANT:")[1].strip()
        elif "USER:" in predicted_answer:
            predicted_answer = predicted_answer.split("USER:")[-1].strip()
        
        # 确保 ground_truth_answers 的长度与 predicted_answers 匹配
        if i < len(ground_truth_answers):
            ground_truth_answer = ground_truth_answers[i]
        else:
            # 如果不匹配，可能是评估集和预测结果对应有问题，这里需要更严格的错误处理
            # 暂时跳过或者报错
            logger.warning(f"Warning: Missing ground truth answer for index {i}. Skipping this sample.")
            continue

        # 计算语义相似度
        embeddings1 = sentence_model.encode(predicted_answer, convert_to_tensor=True)
        embeddings2 = sentence_model.encode(ground_truth_answer, convert_to_tensor=True)
        cosine_similarity = util.cos_sim(embeddings1, embeddings2).item()
        all_similarities.append(cosine_similarity)

        if cosine_similarity >= semantic_similarity_threshold:
            correct_semantic_predictions += 1

    accuracy_semantic = (correct_semantic_predictions / total_samples) if total_samples > 0 else 0
    
    # 可以返回更多指标，例如平均语义相似度
    avg_semantic_similarity = np.mean(all_similarities) if all_similarities else 0

    return {
        "semantic_accuracy": accuracy_semantic,
        "avg_semantic_similarity": avg_semantic_similarity,
    }