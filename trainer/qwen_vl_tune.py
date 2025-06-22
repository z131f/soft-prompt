import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLConfig
# 引入训练所需的库
from transformers.trainer import Trainer, TrainingArguments
from torch.utils.data import Dataset # 用于自定义数据集
from tqdm import tqdm  # 用于显示进度条
from model.qwen_vl_tune import ModifiedQwenVL
from utils import custom_collate_fn, compute_metrics, set_seed, get_logger, print_trainable_parameters
from functools import partial
from config import build_config

class qwen_vl_tune_trainer():
    def __init__(self, config, logger):
        action = config['action']
        if action is None:
            action = 'begin'
        assert action in ['begin', 'continue', 'skip']
        if action == 'skip':
            print('skip training ...')
            self.skip_training = True
        else:
            self.skip_training = False
            if action == 'continue':
                print('continue training ...')
        self.action = action

        self.config = config

        model_site = config['model_site']
        model_cache = 'model_cache'
        image_size = config['image_size']
        lr = config['lr']
        seed = config['seed']
        self.load_train_num = config['load_train_num']
        self.load_test_num = config['load_test_num']
        self.dataset_name = config['dataset_name']
        per_device_train_batch_size = config['per_device_train_batch_size']
        self.logger = logger

        if self.dataset_name == 'RSVQA_LR':
            self.patch_num = 81
        elif self.dataset_name == 'RSVQA_HR':
            self.patch_num = 324
        elif self.dataset_name == 'RSIVQA':
            self.patch_num = 729
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}. Supported datasets are RSVQA_LR, RSVQA_HR, and RSIVQA.")

        print('load model config ...')
        model_config = Qwen2_5_VLConfig.from_pretrained(model_site, cache_dir=model_cache, trust_remote_code=True)
        print('load model ...')
        self.model = ModifiedQwenVL.from_pretrained(model_site, config=model_config, cache_dir=model_cache, device_map=config['device_map'], torch_dtype=torch.bfloat16,
                                                    patch_num=self.patch_num, extra_fun=self.config['extra_fun'])
        print('load model processor ...')
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_site, cache_dir=model_cache+'/llava_next_processor', use_fast=True, num_additional_image_tokens=1 + 1)
        print('load train args ...')
        tag = config['tag']
        if tag == 'auto':
            tag = config['dataset_name']+'_'+str(config['load_train_num'])+'_'+str(config['task'])
            if config['extra_fun'] is not None:
                tag += '_'+config['extra_fun']
            print(f"Auto-generated tag: {tag}")
        self.output_dir = "./output/qwen_vl/" + tag
        print(f"Output directory: {self.output_dir}")
        self.training_args = TrainingArguments(
            output_dir=self.output_dir, # 训练输出目录
            num_train_epochs=1,                                # 训练轮数
            per_device_train_batch_size=per_device_train_batch_size,                     # 每个设备的训练批量大小
            gradient_accumulation_steps=config['gradient_accumulation_steps'],                     # 梯度累积步数，模拟更大的batch size
            learning_rate=lr,                                # 学习率
            # learning_rate=1e-4,                                # 学习率 学习率过大
            weight_decay=0.01,                                 # 权重衰减
            logging_dir="logs",                              # 日志目录
            logging_steps=100,                                  # 每隔多少步记录一次日志
            save_steps=500,                                    # 每隔多少步保存一次模型
            save_total_limit=2,                                # 最多保存的模型数量
            do_train=True,                                     # 执行训练
            report_to="none",                                  # 不报告到任何平台 (可选，如果你想用wandb等可以设置)
            bf16=True,                                         # 使用半精度训练
            seed=seed,
        )
        print('load dataset ...')
        self.load_data()
        print('load trainer ...')
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=custom_collate_fn, # 使用自定义的 collate_fn
        )
        self.model.set_train()
        print_trainable_parameters(self.model)
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

    
    def __save_model(self):
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.soft_prompts, os.path.join(output_dir, "soft_prompts.pt"))
        #torch.save(self.model.q_projector_image.state_dict(), os.path.join(output_dir, "q_projector_image.pt"))
        torch.save(self.model.k_projector_image.state_dict(), os.path.join(output_dir, "k_projector_image.pt"))
        #torch.save(self.model.v_projector_image.state_dict(), os.path.join(output_dir, "v_projector_image.pt"))
        #torch.save(self.model.k_projector_prompt.state_dict(), os.path.join(output_dir, "k_projector_prompt.pt"))
        torch.save(self.model.q_projector_prompt.state_dict(), os.path.join(output_dir, "q_projector_prompt.pt"))
        #torch.save(self.model.global_prompt_self_attention.state_dict(), os.path.join(output_dir, "global_prompt_self_attention.pt"))
        print(f"新模块权重已保存到：{output_dir}")

    
    def __load_model(self):
        loaded_model_dir = self.output_dir
        print(f"Skipping training. Attempting to load model from {loaded_model_dir}...")
        try:
            # 确保加载的模型目录存在
            if not os.path.exists(loaded_model_dir):
                raise FileNotFoundError(f"Model directory not found: {loaded_model_dir}")

            # 检查每个权重文件是否存在
            soft_prompts_path = os.path.join(loaded_model_dir, "soft_prompts.pt")
            #q_projector_image_path = os.path.join(loaded_model_dir, "q_projector_image.pt")
            k_projector_image_path = os.path.join(loaded_model_dir, "k_projector_image.pt")
            #v_projector_image_path = os.path.join(loaded_model_dir, "v_projector_image.pt")
            q_projector_prompt_path = os.path.join(loaded_model_dir, "q_projector_prompt.pt")
            #v_projector_prompt_path = os.path.join(loaded_model_dir, "v_projector_prompt.pt")
            #global_prompt_self_attention_path = os.path.join(loaded_model_dir, "global_prompt_self_attention.pt")

            if not os.path.exists(soft_prompts_path):
                raise FileNotFoundError(f"soft_prompts.pt not found in {loaded_model_dir}")
            # if not os.path.exists(global_prompt_self_attention_path):
            #     raise FileNotFoundError(f"global_prompt_self_attention.pt not found in {loaded_model_dir}")

            # 加载权重
            self.model.soft_prompts = torch.load(soft_prompts_path, map_location='cuda')
            self.model.k_projector_image.load_state_dict(torch.load(k_projector_image_path, map_location='cuda'))
            #self.model.k_projector_image.load_state_dict(torch.load(k_projector_image_path, map_location='cuda'))
            #self.model.v_projector_image.load_state_dict(torch.load(v_projector_image_path, map_location='cuda'))
            self.model.q_projector_prompt.load_state_dict(torch.load(q_projector_prompt_path, map_location='cuda'))
            #self.model.v_projector_prompt.load_state_dict(torch.load(v_projector_prompt_path, map_location='cuda'))
            #self.model.global_prompt_self_attention.load_state_dict(torch.load(global_prompt_self_attention_path, map_location='cuda'))
            print("Model weights loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}. Please ensure the correct path and saved files.")
            print("Exiting as training is skipped and model loading failed.")
            exit() # 如果加载失败，直接退出程序
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            print("Exiting as training is skipped and model loading failed.")
            exit()

    def train(self):
        if not self.skip_training:
            print("Starting training...")
            if self.action == 'continue':
                print("Continuing training from the last checkpoint...")
                self.__load_model()
            self.trainer.train()

            # 5. 开始训练
            print("Training finished!")
            
            for obj in self.trainer.state.log_history:
                self.logger.info(str(obj))

            self.__save_model()
        else:
            self.__load_model()


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