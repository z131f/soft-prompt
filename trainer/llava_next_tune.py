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
from model.LlavaNext import ModifiedLlavaNext
from utils import custom_collate_fn, compute_metrics, set_seed, get_logger, print_trainable_parameters
from functools import partial
from config import build_config

class llava_next_tune_trainer():
    def __init__(self, config, logger, tag=None, action=None, load_tag=None):
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
        self.load_tag = load_tag

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
        print('load model config ...')
        model_config = AutoConfig.from_pretrained(model_site, cache_dir=model_cache, trust_remote_code=True)
        patch_num = image_size_to_num_patches(image_size=image_size,grid_pinpoints=model_config.image_grid_pinpoints,patch_size=model_config.vision_config.image_size)
        print('load model ...')
        self.model = ModifiedLlavaNext.from_pretrained(model_site, config=model_config, cache_dir=model_cache, device_map="balanced", patch_num=patch_num, torch_dtype=torch.bfloat16)
        print('load model processor ...')
        self.processor = LlavaNextProcessor.from_pretrained(model_site, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1 + 1)
        print('load train args ...')
        if tag is None:
            tag = config['dataset_name']+'_'+str(config['load_train_num'])+'_'+str(config['lr'])
        if tag is None:
            self.output_dir = "./output/llava_next"
        else:
            self.output_dir = "./output/llava_next/" + tag
        self.training_args = TrainingArguments(
            output_dir=self.output_dir, # 训练输出目录
            num_train_epochs=1,                                # 训练轮数
            per_device_train_batch_size=per_device_train_batch_size,                     # 每个设备的训练批量大小
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
        print('load dataset ...')
        self.__load_data()
        print('load trainer ...')
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=custom_collate_fn, # 使用自定义的 collate_fn
        )
        self.model.set_train()
        print_trainable_parameters(self.model)


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
                task=self.config['task']
            )
            RSVQA_LR_Dataset_eval = load_dataset(
                dataset_name="RSVQA_LR",
                is_eval=True,
                add_instruct=True,
                load_num=self.load_test_num,
                type="test",
                processor=self.processor,
                task=self.config['task']
            )
            self.train_dataset = RSVQA_LR_Dataset_train
            self.eval_dataset = RSVQA_LR_Dataset_eval

    
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
        if self.action == 'continue' and not self.skip_training:
            self.output_dir = "./output/llava_next/" + self.load_tag
        else:
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