import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import os

# 设置日志，用于错误消息
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

class RSVQA_LR_Dataset(Dataset):
    def __init__(self, processor, image_folder_path, questions_json_path, answers_json_path, images_json_path, image_size, use_num,logger=None, add_instruct=False, is_eval=False, task='all'):
        self.processor = processor
        self.image_folder_path = image_folder_path
        self.image_size = image_size
        self.logger = logger
        self.add_instruct = add_instruct
        self.is_eval = is_eval  # 是否为评估模式
        self.task = task  # 任务类型，默认为 'all'

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
        # 每个样本将是 (image_id, question_id, answer_id) 的元组
        self.samples = self._prepare_samples()[:use_num] if use_num > 0 else self._prepare_samples()

    def get_answers_list(self):
        return [s['answer_text'] for s in self.samples]

    def _prepare_samples(self):
        """
        准备 (image_id, question_id, answer_id, question_text, answer_text) 元组的列表。
        这会将图像、问题和答案关联起来。
        """
        samples = []
        # 遍历问题以构建样本
        for question_entry in self.questions_data:
            if self.task != 'all' and question_entry['type'] != self.task:
                continue
            if question_entry['active'] == False:
                continue
            q_id = question_entry["id"]
            img_id = question_entry["img_id"]
            question_text = question_entry["question"]
            if self.add_instruct:
                question_text = 'Answer with a single word.  ' + question_text

            # 每个问题可以有多个答案，但通常我们只用一个进行训练。
            # 假设 questions_json 中的 'answers_ids' 指向该问题的正确答案。
            if "answers_ids" in question_entry and len(question_entry["answers_ids"]) > 0:
                # 为简单起见，我们将取第一个答案 ID。
                # 在真实的 VQA 设置中，您可能需要以不同的方式处理多个答案。
                for ans_id in question_entry["answers_ids"]:
                    answer_entry = self.answer_id_to_data.get(ans_id)
                    if answer_entry:
                        samples.append({
                            "image_id": img_id,
                            "question_id": q_id,
                            "answer_id": ans_id,
                            "question_text": question_text,
                            "answer_text": answer_entry["answer"]
                        })
                    else:
                        self.logger.warning(f"问题 ID {q_id} 的答案 ID {ans_id} 在 answers.json 中未找到。跳过。")
            else:
                self.logger.warning(f"问题 ID {q_id} 没有关联答案。跳过。")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_id = sample["image_id"]
        question_text = sample["question_text"]
        answer_text = sample["answer_text"]

        # 从 images_data 中获取图像的原始名称
        image_info = self.image_id_to_data.get(image_id)
        if not image_info:
            self.logger.error(f"图像 ID {image_id} 在 images.json 中未找到。此样本返回 None。")
            return None # 需要在 collate_fn 中处理

        image_filename = f'{image_id}.tif'
        image_path = os.path.join(self.image_folder_path, image_filename)

        try:
            image = Image.open(image_path).convert("RGB").resize(self.image_size)
        except Exception as e:
            self.logger.error(f"加载图像 {image_path} 时出错：{e}。此样本返回 None。")
            return None # 需要在 collate_fn 中处理

        # 构建 VQA 模型的完整文本输入
        if self.is_eval:
            full_text = f"USER: <image>\n{question_text} ASSISTANT:"
        else:
            full_text = f"USER: <image>\n{question_text} ASSISTANT: {answer_text}"

        # 处理文本和图像
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=3800 # 根据您的模型和数据集调整 max_length
        )

        inputs["labels"] = inputs["input_ids"].clone()

        # 确保图像 token、填充和未知 token 的标签设置为 -100
        if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'pad_token_id'):
            inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100
        if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'unk_token_id'):
            inputs["labels"][inputs["labels"] == self.processor.tokenizer.unk_token_id] = -100
        if hasattr(self.processor, 'image_token_id'):
            inputs["labels"][inputs["labels"] == self.processor.image_token_id] = -100

        # --- 关键修改部分 ---
        # 找到 "ASSISTANT:" 标记的起始位置，这通常是答案的开始
        # 您需要确保您的 tokenizer 不会将 "ASSISTANT:" 拆分成多个 token。
        # 如果会，您需要找到这些 token 序列的开始。
        # 最安全的方法是先对 "ASSISTANT:" 进行 tokenize。

        # 找到 "ASSISTANT:" 在原始 full_text 中的 tokenization 后的起始索引
        # 假设 'ASSISTANT:' 在 tokenizer 后是一个或多个连续的 token。
        # 我们先对 ' ASSISTANT:' （注意前面的空格，避免与单词内部的 'ASSISTANT' 混淆）进行 tokenization
        # 某些 tokenizer 会在单词前加空格，所以尝试两种情况
        assistant_tokens_ids_with_space = self.processor.tokenizer(" ASSISTANT:", add_special_tokens=False).input_ids
        assistant_tokens_ids_no_space = self.processor.tokenizer("ASSISTANT:", add_special_tokens=False).input_ids

        assistant_start_index = -1

        assert inputs['input_ids'].shape[0] == 1, "输入必须是单条数据"
        
        # 尝试匹配带空格的 " ASSISTANT:"
        for i in range(len(inputs["input_ids"][0]) - len(assistant_tokens_ids_with_space) + 1):
            if (inputs["input_ids"][0][i:i+len(assistant_tokens_ids_with_space)] == torch.tensor(assistant_tokens_ids_with_space, device=inputs["input_ids"].device)).all():
                assistant_start_index = i + len(assistant_tokens_ids_with_space)
                break
                
        # 如果没有找到带空格的，尝试匹配不带空格的 "ASSISTANT:"
        if assistant_start_index == -1:
            for i in range(len(inputs["input_ids"][0]) - len(assistant_tokens_ids_no_space) + 1):
                if (inputs["input_ids"][0][i:i+len(assistant_tokens_ids_no_space)] == torch.tensor(assistant_tokens_ids_no_space, device=inputs["input_ids"].device)).all():
                    assistant_start_index = i + len(assistant_tokens_ids_no_space)
                    break

        if assistant_start_index != -1:
            # 如果找到了 "ASSISTANT:" 的位置，将它之前的所有标签设置为 -100
            # 确保这里操作的是inputs["labels"]的第一个维度（批次维度，虽然现在是1）
            # 并且只对有效长度内的token进行操作
            inputs["labels"][:, :assistant_start_index] = -100
        else:
            self.logger.warning(f"未能找到 'ASSISTANT:' 标记，可能未完全屏蔽问题损失。Full text: {full_text}")


        # 挤压掉批次维度，因为 DataLoader 会重新添加它
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs