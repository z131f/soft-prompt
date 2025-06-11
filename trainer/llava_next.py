import logging
import os # 导入 logging 以便更好地处理错误
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
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LlavaNextCausalLMOutputWithPast, image_size_to_num_patches
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

class llava_next_trainer():
    def __init__(self, config, logger):

        model_site = config['model_site']
        model_cache = 'model_cache'
        image_size = config['image_size']
        seed = config['seed']
        self.load_train_num = config['load_train_num']
        self.load_test_num = config['load_test_num']
        self.dataset_name = config['dataset_name']
        per_device_train_batch_size = config['per_device_train_batch_size']
        self.logger = logger
        print('load model config ...')
        model_config = AutoConfig.from_pretrained(model_site, cache_dir=model_cache, trust_remote_code=True)
        patch_num = image_size_to_num_patches(image_size=image_size,grid_pinpoints=model_config.image_grid_pinpoints,patch_size=model_config.vision_config.image_size)
        print('load model ...')
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_site, config=model_config, cache_dir=model_cache, device_map="balanced", torch_dtype=torch.bfloat16)
        print('load model processor ...')
        self.processor = LlavaNextProcessor.from_pretrained(model_site, cache_dir=model_cache, use_fast=True)
        print('load dataset ...')
        self.__load_data()


    def __load_data(self):
        dataset_name = self.dataset_name
        if dataset_name == 'RSVQA_LR':
            RSVQA_LR_Dataset_train = load_dataset(
                dataset_name="RSVQA_LR",
                is_eval=False,
                add_instruct=True,
                load_num=self.load_train_num,
                type="train",
                processor=self.processor,
            )
            RSVQA_LR_Dataset_eval = load_dataset(
                dataset_name="RSVQA_LR",
                is_eval=True,
                add_instruct=True,
                load_num=self.load_test_num,
                type="test",
                processor=self.processor,
            )
            self.train_dataset = RSVQA_LR_Dataset_train
            self.eval_dataset = RSVQA_LR_Dataset_eval


    def eval(self):
        print("\n--- 评估训练结果 ---")
        print("正在加载用于语义相似度的句子 transformer 模型...")
        sentence_model = SentenceTransformer('all-mpnet-base-v2')
        print("句子 transformer 模型加载完成。")

        # 确保模型处于评估模式
        self.model.eval()

        # 加载用于语义相似度的句子 transformer 模型
        # 'all-MiniLM-L6-v2' 在速度和性能之间取得了很好的平衡。
        # 如果计算资源允许，你可以考虑使用 'all-mpnet-base-v2' 以获得更高的准确性。

        correct_predictions_semantic = 0
        total_predictions = 0
        semantic_similarity_threshold = 0.75 # 根据你的需求调整此阈值

        # 可以使用DataLoader进行批量处理，但为了简单起见，我们这里直接遍历评估数据集
        # 如果你的评估数据集很大，请考虑使用 DataLoader
        eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=1, collate_fn=custom_collate_fn)
        answer_list = self.eval_dataset.get_answers_list() # 获取答案列表

        for idx, batch in tqdm(enumerate(eval_dataloader), desc="正在评估"):
            # 将批次数据移动到设备上
            inputs_eval = {k: v.to('cuda') for k, v in batch.items() if k != 'answer_text'}
            ground_truth_answer = answer_list[idx] # 假设批处理大小为1，简化处理

            with torch.no_grad():
                generate_ids = self.model.generate(**inputs_eval, max_new_tokens=100)
                # [1,3806]
                predicted_answer = self.processor.batch_decode( # type: ignore
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                # 清理预测答案
                # 提示词是 "USER: <image>\n What's the content of the image? ASSISTANT:"
                # 模型可能会生成整个提示词+答案
                if "ASSISTANT:" in predicted_answer:
                    predicted_answer = predicted_answer.split("ASSISTANT:")[1].strip()
                elif "USER:" in predicted_answer: # 如果模型生成了意外的前缀
                    predicted_answer = predicted_answer.split("USER:")[-1].strip()

            print(f"\n问题: {self.processor.decode(inputs_eval['input_ids'][0], skip_special_tokens=True)}") # type: ignore
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