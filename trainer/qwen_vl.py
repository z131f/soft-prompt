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
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
# 引入训练所需的库
from transformers.trainer import Trainer, TrainingArguments
from torch.utils.data import Dataset # 用于自定义数据集
from tqdm import tqdm  # 用于显示进度条
from model.llava_next_tune import ModifiedLlavaNext
from utils import custom_collate_fn, compute_metrics, set_seed, get_logger, print_trainable_parameters
from functools import partial
from config import build_config
from model.qwen_vl_tune import ModifiedQwenVL

class qwen_vl_trainer():
    def __init__(self, config, logger):
        self.config = config

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
        # patch_num = image_size_to_num_patches(image_size=image_size,grid_pinpoints=model_config.image_grid_pinpoints,patch_size=model_config.vision_config.image_size)
        print('load model ...')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_site, config=model_config, cache_dir=model_cache, device_map="balanced", torch_dtype=torch.bfloat16)
        print('load model processor ...')
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_site, cache_dir=model_cache, use_fast=True)
        print('load dataset ...')
        self.load_data()
        self.sentence_model = None


    def load_data(self):
        dataset_name = self.config['dataset_name']
        if dataset_name == 'RSIVQA':
            load_num = {
                'train': self.config['load_train_num'],
                'test': self.config['load_test_num']
            }
            self.train_dataset, self.eval_dataset = load_dataset(
                dataset_name=self.config['dataset_name'],
                is_eval=None,
                add_instruct=True,
                load_num=load_num,
                type=None,
                processor=self.processor,
                task=self.config['task'],
                model_name=self.config['model_name']
            ) # type: ignore
        else:
            self.train_dataset = load_dataset(
                    dataset_name=self.config['dataset_name'],
                    is_eval=False,
                    add_instruct=True,
                    load_num=self.config['load_train_num'],
                    type="train",
                    processor=self.processor,
                    task=self.config['task'],
                    model_name=self.config['model_name']
                )
            self.eval_dataset = load_dataset(
                    dataset_name=self.config['dataset_name'],
                    is_eval=True,
                    add_instruct=True,
                    load_num=self.config['load_test_num'],
                    type="test",
                    processor=self.processor,
                    task=self.config['task'],
                    model_name=self.config['model_name']
                )

    def eval(self, is_print=False, ans_add=''):
        self.model.eval()

        if self.sentence_model is None:
            self.sentence_model = self.sentence_model = SentenceTransformer('all-mpnet-base-v2', device='cuda', cache_folder='model_cache/sentence_transformer')

        correct_predictions_semantic = 0
        total_predictions = 0
        semantic_similarity_threshold = 0.75 

        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset, 
            batch_size=1, # 评估时通常每个样本单独生成，如果可以批量生成，可以调大
            collate_fn=custom_collate_fn,
            shuffle=False 
        )
        
        all_ground_truth_answers = self.eval_dataset.get_answers_list()
        all_sample_types = self.eval_dataset.get_type_list()

        for idx, batch in tqdm(enumerate(eval_dataloader), desc="正在评估", total=len(self.eval_dataset)):
            # 将批次数据移动到设备上。
            # 这里需要保留原始的 input_ids，因为它代表了 prompt 的部分
            inputs_eval = {k: v.to('cuda') for k, v in batch.items() if k not in ['labels', 'answer_text']}
            
            ground_truth_answer = all_ground_truth_answers[idx]
            current_sample_type = all_sample_types[idx]

            with torch.no_grad():
                # generate_ids 将包含 prompt 的 token 以及模型生成的新 token
                generate_ids = self.model.generate(
                    **inputs_eval, 
                    max_new_tokens=100,
                    do_sample=False,
                )

                # 解码完整的生成结果
                full_generated_text = self.processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0] # 假设 batch_size 为 1

                parts = full_generated_text.split("assistant\n", 1) # 1 表示只分割一次

                predicted_answer = parts[1].strip() # .strip() 用于去除可能的前后空白           
                # 进一步清理可能存在的 EOS token 或其他模型特定的生成结束标记
                # 例如，LLaVA 模型有时会在答案后面生成 </s> 或其他特殊 token
                if self.processor.tokenizer.eos_token and self.processor.tokenizer.eos_token in predicted_answer:
                    predicted_answer = predicted_answer.split(self.processor.tokenizer.eos_token)[0].strip()

            # 应用 ans_add 和类型特定后缀
            predicted_answer += ans_add
            if ans_add == '' and current_sample_type == 'area':
                if 'm2' not in predicted_answer:
                    predicted_answer = predicted_answer + 'm2'
            
            # 记录和打印
            if is_print:
                # 打印问题时使用原始 prompt 文本，更清晰
                print(f"\n完整文本: {full_generated_text}") 
                print(f"预测答案: {predicted_answer}")
                print(f"真实答案: {ground_truth_answer}")

            # --- 语义相似度评分 ---
            embeddings1 = self.sentence_model.encode([predicted_answer], convert_to_tensor=True)
            embeddings2 = self.sentence_model.encode([ground_truth_answer], convert_to_tensor=True)

            cosine_similarity = util.cos_sim(embeddings1, embeddings2).item()
            if is_print:
                print(f"语义相似度: {cosine_similarity:.4f}")

            if cosine_similarity >= semantic_similarity_threshold:
                correct_predictions_semantic += 1
            total_predictions += 1

        accuracy_semantic = (correct_predictions_semantic / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"\n--- 评估总结 (基于语义相似度) ---")
        print(f"总评估样本数: {total_predictions}")
        print(f"语义相似度正确预测数 (相似度 > {semantic_similarity_threshold}): {correct_predictions_semantic}")
        print(f"语义相似度准确率: {accuracy_semantic:.2f}%")
        
        return accuracy_semantic