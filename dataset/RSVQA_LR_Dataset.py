import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
from qwen_vl_utils import process_vision_info
import random


class RSVQA_LR_Dataset(Dataset):
    def __init__(self, processor, model_name, image_folder_path, questions_json_path, answers_json_path, images_json_path, image_size, use_num, logger=None, add_instruct=False, is_eval=False, task='all'):
        self.processor = processor
        self.model_name = model_name
        self.image_folder_path = image_folder_path
        self.image_size = image_size
        self.logger = logger
        self.add_instruct = add_instruct
        self.is_eval = is_eval  # 是否为评估模式
        self.task = task  # 任务类型，默认为 'all'

        self.task_list = ['comp', 'count', 'presence', 'rural_urban']
        use_num = {self.task_list[i]: use_num[i] for i in range(len(self.task_list))} if isinstance(use_num, list) else use_num

        # 从 JSON 文件加载所有数据
        with open(questions_json_path, 'r', encoding='utf-8') as f:
            self.questions_data = json.load(f)["questions"]
        with open(answers_json_path, 'r', encoding='utf-8') as f:
            self.answers_data = json.load(f)["answers"]
        with open(images_json_path, 'r', encoding='utf-8') as f:
            self.images_data = json.load(f)["images"]

        # 创建字典以便高效查找
        self.question_id_to_data = {q["id"]: q for q in self.questions_data}
        self.answer_id_to_data = {a["id"]: a for a in self.answers_data}
        self.image_id_to_data = {img["id"]: img for img in self.images_data}

        # 准备用于迭代的样本列表
        self.samples = self._prepare_samples()
        if isinstance(use_num, int) and use_num > 0:
            self.samples = self.samples[:use_num]
        elif isinstance(use_num, dict):
            new_samples = []
            for sample in self.samples:
                if sample['type'] in use_num and use_num[sample['type']] > 0:
                    new_samples.append(sample)
                    use_num[sample['type']] -= 1
            self.samples = new_samples
        print(f"加载 {len(self.samples)} 个样本。")
        random.shuffle(self.samples)  # 打乱样本顺序

    def get_answers_list(self):
        return [s['answer_text'] for s in self.samples]

    def _prepare_samples(self):
        """
        准备 (image_id, question_id, answer_id, question_text, answer_text) 元组的列表。
        这会将图像、问题和答案关联起来。
        """
        samples = []
        for question_entry in self.questions_data:
            if question_entry['active'] == False:
                continue
            if self.task != 'all' and question_entry['type'] != self.task:
                continue
            q_id = question_entry["id"]
            img_id = question_entry["img_id"]
            question_text = question_entry["question"]
            if self.add_instruct:
                question_text = 'Answer in a word or a number. Question: ' + question_text

            if "answers_ids" in question_entry and len(question_entry["answers_ids"]) > 0:
                for ans_id in question_entry["answers_ids"]:
                    answer_entry = self.answer_id_to_data.get(ans_id)
                    if answer_entry:
                        samples.append({
                            "image_id": img_id,
                            "question_id": q_id,
                            "answer_id": ans_id,
                            "question_text": question_text,
                            "answer_text": answer_entry["answer"],
                            "type": question_entry["type"]
                        })
                    else:
                        if self.logger: self.logger.warning(f"问题 ID {q_id} 的答案 ID {ans_id} 在 answers.json 中未找到。跳过。")
            else:
                if self.logger: self.logger.warning(f"问题 ID {q_id} 没有关联答案。跳过。")

        print(f"共有 {len(samples)} 个样本。")
        return samples

    def __len__(self):
        return len(self.samples)
    
    def get_type_list(self):
        return [s['type'] for s in self.samples]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_id = sample["image_id"]
        question_text = sample["question_text"]
        answer_text = sample["answer_text"]

        image_info = self.image_id_to_data.get(image_id)
        if not image_info:
            if self.logger: self.logger.error(f"图像 ID {image_id} 在 images.json 中未找到。此样本返回 None。")
            return None 

        image_filename = f'{image_id}.tif'
        image_path = os.path.join(self.image_folder_path, image_filename)

        try:
            image = Image.open(image_path).convert("RGB").resize(self.image_size)
        except Exception as e:
            if self.logger: self.logger.error(f"加载图像 {image_path} 时出错：{e}。此样本返回 None。")
            return None 

        # 构建 VQA 模型的完整文本输入
        if self.model_name == 'llava_next' or self.model_name == 'llava_next_tune':
            # 只包含用户问题的对话（用于计算 prompt 长度）
            conversation_prompt_only = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_text},
                        {"type": "image"},
                    ],
                },
            ]
            # 包含用户问题和助手答案的完整对话（用于生成 input_ids 和 labels）
            conversation_full = conversation_prompt_only.copy()
            if not self.is_eval: # 评估模式下不添加答案
                conversation_full.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer_text},
                        ],
                    }
                )
            
            # 使用 processor 生成完整对话的 input_ids
            # 注意这里 apply_chat_template 默认会 tokenize，但为了避免二次处理，可以先获取字符串再 tokenize
            full_text = self.processor.apply_chat_template(conversation_full, tokenize=False)
            inputs = self.processor(
                images=image,
                text=full_text,
                return_tensors="pt",
                # padding="max_length", # 这里使用 max_length 填充，确保批次内长度一致
                # truncation=True,
                # max_length=3800 # 确保足够长以包含所有内容
            )

            # 使用 processor 生成只包含 prompt 的 input_ids，用于确定 prompt 的长度
            prompt_text_only = self.processor.apply_chat_template(conversation_prompt_only, tokenize=False)
            prompt_inputs_only = self.processor(
                images=image,
                text=prompt_text_only,
                return_tensors="pt",
                # 不做 padding 或 truncation，只获取原始长度
            )
            
            # 确保批次维度是1，以便后续操作
            assert inputs['input_ids'].shape[0] == 1, "输入必须是单条数据"

            # 初始化 labels，用 -100 填充
            labels = torch.full_like(inputs["input_ids"], -100)

            # 确定 assistant 回复的起始位置
            # 这是最关键的部分：prompt_inputs_only 的长度就是 assistant 回复前的所有 token 长度
            # 注意这里需要考虑 tokenzier 的特殊 token 和 image token 的处理
            # 实际的 prompt_input_ids 可能包含特殊 token，但它们通常会出现在 text 的开头。
            # 这里我们假设 assistant 的回复紧跟在 prompt 后面。
            
            # 从 full_input_ids 中找到 assistant 回复的起始索引
            # 假设 full_text 是 "prompt assistant_text"
            # 我们可以通过比较 full_input_ids 和 prompt_inputs_only 的前缀来找到 assistant_text 的开始
            
            # 找到 prompt_inputs_only 的有效长度（非填充部分）
            prompt_len = prompt_inputs_only["input_ids"].shape[1]

            # 将 labels 中 assistant 回复的部分设置为对应的 input_ids
            # assistant_start_index 就是 prompt_len
            labels[:, ((inputs["input_ids"]==32001).sum()+prompt_len):] = inputs["input_ids"][:, ((inputs["input_ids"]==32001).sum()+prompt_len):]
            
            inputs["labels"] = labels

        elif self.model_name == 'qwen_vl' or self.model_name == 'qwen_vl_tune':
            # 假设 process_vision_info 已经被正确导入
            # 对于 QWen-VL，同样需要区分 prompt only 和 full conversation
            conversation_prompt_only = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {"type": "text", "text": question_text},
                    ],
                },
            ]
            conversation_full = conversation_prompt_only.copy()
            if not self.is_eval:
                conversation_full.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer_text},
                        ],
                    }
                )
            
            # 获取完整对话的 text
            full_text = self.processor.apply_chat_template(
                conversation_full, tokenize=False, add_generation_prompt=self.is_eval
            )
            image_inputs_full, video_inputs_full = process_vision_info(conversation_full)
            inputs = self.processor(
                text=[full_text],
                images=image_inputs_full,
                videos=video_inputs_full,
                # padding=True,
                return_tensors="pt",
                # truncation=True,
                # max_length=500
            )

            # 获取只包含 prompt 的 text
            prompt_text_only = self.processor.apply_chat_template(
                conversation_prompt_only, tokenize=False, add_generation_prompt=True
            )
            image_inputs_prompt, video_inputs_prompt = process_vision_info(conversation_prompt_only)
            prompt_inputs_only = self.processor(
                text=[prompt_text_only],
                images=image_inputs_prompt,
                videos=video_inputs_prompt,
                return_tensors="pt",
            )
            
            assert inputs['input_ids'].shape[0] == 1, "输入必须是单条数据"

            labels = torch.full_like(inputs["input_ids"], -100)
            
            # 对于 QWen-VL，也用相同的方法找到 prompt 长度
            prompt_len = prompt_inputs_only["input_ids"].shape[1]
            labels[:, prompt_len:] = inputs["input_ids"][:, prompt_len:]
            
            inputs["labels"] = labels

            if self.model_name == 'qwen_vl_tune':
                inputs['attention_mask'] = torch.cat([
                    torch.ones((1, 1), dtype=inputs['attention_mask'].dtype),
                    inputs['attention_mask']
                ], dim=1)
                inputs['labels'] = torch.cat([
                    torch.tensor([[-100]], dtype=inputs['labels'].dtype),
                    inputs['labels'],
                ], dim=1)
                ids_image_start = torch.where(inputs['input_ids']==151655)[1][0]
                inputs['input_ids'] = torch.cat([
                    inputs['input_ids'][:, :ids_image_start],
                    torch.tensor([[151655]], dtype=inputs['input_ids'].dtype),  # 添加 image token
                    inputs['input_ids'][:, ids_image_start:]
                ], dim=1)

        # print(len(inputs["input_ids"][0]), prompt_len)
        
        # 挤压掉批次维度，因为 DataLoader 会重新添加它
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs