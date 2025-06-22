from transformers import AutoProcessor
import torch
import shutil
import logging
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import random
from qwen_vl_utils import process_vision_info
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class CombinedVQADataset(Dataset):
    """
    一个用于视觉问答任务的综合数据集类，支持拼接多个子数据集，
    并处理训练集/测试集的划分和数据格式化，支持按答案类型归类和限制样本数量。
    支持分别为训练集和测试集设置样本限制。
    """
    def __init__(self, processor, model_name, image_size, use_num=None, add_instruct=False, task='all', split_save_path=None):
        """
        初始化 CombinedVQADataset。

        Args:
            processor: Hugging Face Transformers 的 AutoProcessor 或类似的处理器，用于图像和文本预处理。
            image_size (tuple): 图像将被调整到的尺寸，例如 (224, 224)。
            use_num: 控制每个任务类型或整体数据集的样本数量。可以是一个整数，
                     一个表示整体限制的字典（如 {'train': N, 'test': M}），
                     或者一个表示按类型限制的字典（如 {'train': {'YesOrNo': N1}, 'test': {'Number': M2}}）。
                     - 如果是 int 且 > 0: 限制训练集总样本数，测试集不限制。
                     - 如果是 dict，且键为 'train' 和/或 'test':
                             - 值可以为 int (限制该集合总样本数)
                             - 值可以为 dict (限制该集合内各类型样本数)
                     - 否则 (None 或 0): 不限制。
            add_instruct (bool): 是否在问题前添加指令。
            task (str): 指定要加载的任务类型 ('YesOrNo', 'Number', 'Other', 'all')。
            split_save_path (str, optional): 用于保存和读取数据集划分结果的文件路径。如果为 None，则不保存/读取。
        """
        print(f'CombinedVQADataset init with image_size: {image_size}, use_num: {use_num}, add_instruct: {add_instruct}, task: {task}, split_save_path: {split_save_path}')
        self.processor = processor
        self.image_size = image_size
        self.add_instruct = add_instruct
        self.task = task
        self.split_save_path = split_save_path
        self.model_name = model_name

        self.task_list = ['YesOrNo', 'Number', 'Other']
        
        # 初始化训练集和测试集的样本数量限制字典和总样本限制
        self._train_use_num_limits_by_type = {task_type: -1 for task_type in self.task_list} # 默认为 -1 (不限制)
        self._test_use_num_limits_by_type = {task_type: -1 for task_type in self.task_list}  # 默认为 -1 (不限制)
        self._train_total_use_num = -1 # 默认为 -1 (不限制)
        self._test_total_use_num = -1  # 默认为 -1 (不限制)

        if isinstance(use_num, int) and use_num > 0:
            # 如果是整数，将其视为训练集的总样本限制
            self._train_total_use_num = use_num
        elif isinstance(use_num, dict):
            # 处理 train 和 test 的具体限制
            if 'train' in use_num:
                if isinstance(use_num['train'], int) and use_num['train'] > 0:
                    self._train_total_use_num = use_num['train']
                elif isinstance(use_num['train'], dict):
                    self._train_use_num_limits_by_type = {task_type: use_num['train'].get(task_type, -1) for task_type in self.task_list}
            
            if 'test' in use_num:
                if isinstance(use_num['test'], int) and use_num['test'] > 0:
                    self._test_total_use_num = use_num['test']
                elif isinstance(use_num['test'], dict):
                    self._test_use_num_limits_by_type = {task_type: use_num['test'].get(task_type, -1) for task_type in self.task_list}

        self.train_samples = []
        self.test_samples = []

        # 尝试从文件加载划分结果
        if self.split_save_path and os.path.exists(self.split_save_path):
            print(f"尝试从 {self.split_save_path} 读取数据集划分结果...")
            try:
                with open(self.split_save_path, 'r', encoding='utf-8') as f:
                    saved_splits = json.load(f)
                    self.train_samples = saved_splits.get('train_samples', [])
                    self.test_samples = saved_splits.get('test_samples', [])
                print(f"成功从 {self.split_save_path} 读取划分结果。训练样本: {len(self.train_samples)}, 测试样本: {len(self.test_samples)}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"读取或解析 {self.split_save_path} 时出错：{e}。将重新进行数据集划分。")
                self.train_samples = []
                self.test_samples = []
        else:
            print(f"未找到划分结果文件 {self.split_save_path} 或未指定保存路径，将进行数据集划分。")


    def _get_answer_type(self, answer_text):
        """
        根据答案文本判断其类型。
        """
        lower_answer = answer_text.lower().strip()
        if lower_answer in ['yes', 'no', 'true', 'false']: # 增加 true/false
            return 'YesOrNo'
        try:
            # 尝试转换为数字，如果是纯数字或包含小数的数字
            float(lower_answer)
            return 'Number'
        except ValueError:
            pass
        return 'Other'

    def add_dataset(self, image_folder_path, qa_file_path_train=None, qa_file_path_test=None, is_split=False, test_split_ratio=0.25):
        """
        添加一个子数据集。

        Args:
            image_folder_path (str): 该子数据集图片所在的文件夹路径。
            qa_file_path_train (str): QA 文件路径。
                                     如果 is_split=False, 这是唯一的 QA 文件。
                                     如果 is_split=True, 这是训练集的 QA 文件。
            qa_file_path_test (str, optional): 如果 is_split=True, 这是测试集的 QA 文件。
            is_split (bool): 该子数据集是否已预先划分好训练集和测试集。
            test_split_ratio (float): 如果 is_split=False，用于划分测试集的比例。默认为 0.25 (四分之一)。
        """
        # 如果已经从文件加载了数据，则不再添加新的数据集，除非train_samples和test_samples为空
        # 这里的逻辑是如果_init_方法成功加载了数据，后续add_dataset就不需要重新读取和划分了
        # 但如果_init_加载失败或者split_save_path没有设置，那么add_dataset仍然需要工作
        if self.split_save_path and self.train_samples and self.test_samples:
            print(f"已从 {self.split_save_path} 加载数据，跳过添加数据集 {image_folder_path}。")
            return

        print(f"正在添加数据集：{image_folder_path}")
        
        def _parse_qa_line(line, file_path, line_idx):
            """内部辅助函数，解析单行 QA 文本并添加 type 属性"""
            try:
                parts = line.strip().split('?')
                if len(parts) < 2:
                    print(f"跳过不符合格式的行：'{line.strip()}' 在文件 {file_path} 第 {line_idx+1} 行")
                    return None
                
                Youtube_part = parts[-1].strip()
                question_with_image_name = "?".join(parts[:-1]).strip()
                
                img_name_q_parts = question_with_image_name.split(':', 1)
                if len(img_name_q_parts) < 2:
                    print(f"跳过不符合格式的行（缺少图片名或问题）：'{line.strip()}' 在文件 {file_path} 第 {line_idx+1} 行")
                    return None

                image_filename = img_name_q_parts[0].strip()
                question_text = img_name_q_parts[1].strip()
                answer_text = Youtube_part

                full_question_text = 'Answer in a word or a number. Question: ' + question_text if self.add_instruct else question_text
                
                answer_type = self._get_answer_type(answer_text)

                sample = {
                    "image_id": image_filename.split('.')[0],
                    "image_filename": image_filename,
                    "image_folder_path": image_folder_path,
                    "question_text": full_question_text,
                    "answer_text": answer_text,
                    "type": answer_type
                }
                return sample
            except Exception as e:
                print(f"处理文件 {file_path} 第 {line_idx+1} 行时出错：{e}")
                return None

        if not is_split:
            if not qa_file_path_train:
                print(f"未划分数据集必须提供 qa_file_path_train: {image_folder_path}")
                return

            with open(qa_file_path_train, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            all_parsed_samples = []
            for line_idx, line in enumerate(lines):
                sample = _parse_qa_line(line, qa_file_path_train, line_idx)
                if sample:
                    all_parsed_samples.append(sample)

            random.shuffle(all_parsed_samples)
            num_test_samples = int(len(all_parsed_samples) * test_split_ratio)
            
            self.test_samples.extend(all_parsed_samples[:num_test_samples])
            self.train_samples.extend(all_parsed_samples[num_test_samples:])
            print(f"数据集 {image_folder_path} (未划分) 已添加。训练样本: {len(all_parsed_samples) - num_test_samples}, 测试样本: {num_test_samples}")

        else:
            if not qa_file_path_train or not qa_file_path_test:
                print(f"已划分数据集必须同时提供 qa_file_path_train 和 qa_file_path_test: {image_folder_path}")
                return

            train_subset = []
            with open(qa_file_path_train, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f.readlines()):
                    sample = _parse_qa_line(line, qa_file_path_train, line_idx)
                    if sample:
                        train_subset.append(sample)

            test_subset = []
            with open(qa_file_path_test, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f.readlines()):
                    sample = _parse_qa_line(line, qa_file_path_test, line_idx)
                    if sample:
                        test_subset.append(sample)
                        
            self.train_samples.extend(train_subset)
            self.test_samples.extend(test_subset)
            print(f"数据集 {image_folder_path} (已划分) 已添加。训练样本: {len(train_subset)}, 测试样本: {len(test_subset)}")

        # 在添加完所有数据集后，如果指定了保存路径，则保存划分结果
    
        
    def save_splits(self):
        """保存当前划分的训练集和测试集到文件。"""
        try:
            os.makedirs(os.path.dirname(self.split_save_path), exist_ok=True)
            with open(self.split_save_path, 'w', encoding='utf-8') as f:
                json.dump({'train_samples': self.train_samples, 'test_samples': self.test_samples}, f, ensure_ascii=False, indent=4)
            print(f"数据集划分结果已保存到 {self.split_save_path}")
        except Exception as e:
            print(f"保存数据集划分结果到 {self.split_save_path} 时出错：{e}")

    def get_dataset(self, split='train'):
        """
        根据指定的划分和初始化的 `task` 及 `use_num` 返回一个子 Dataset 实例。
        """
        if split == 'train':
            filtered_samples = self._filter_and_limit_samples(
                self.train_samples, 
                self.task, 
                self._train_use_num_limits_by_type.copy(), 
                self._train_total_use_num,
            )
            dataset = _VQASubset(self.processor, self.image_size, filtered_samples, False, model_name=self.model_name)
            print(f"返回训练集，包含 {len(dataset)} 个样本 (按任务类型和数量过滤后)。")
            return dataset
        elif split == 'test':
            filtered_samples = self._filter_and_limit_samples(
                self.test_samples, 
                self.task, 
                self._test_use_num_limits_by_type.copy(), 
                self._test_total_use_num,
            )
            dataset = _VQASubset(self.processor, self.image_size, filtered_samples, True, model_name=self.model_name)
            print(f"返回测试集，包含 {len(dataset)} 个样本 (按任务类型和数量过滤后)。")
            return dataset
        else:
            raise ValueError("split 参数必须是 'train' 或 'test'")

    def _filter_and_limit_samples(self, samples_list, task_filter, use_num_limits_by_type, total_use_num):
        """
        根据 task_filter、use_num_limits_by_type (按类型限制) 和 total_use_num (总样本限制) 过滤和限制样本列表。
        """
        filtered_samples = []
        
        if task_filter != 'all':
            samples_to_process = [s for s in samples_list if s['type'] == task_filter]
        else:
            samples_to_process = list(samples_list)

        # 只有在需要限制数量时才打乱，否则保持原始顺序
        # 如果有按类型限制或者有总样本限制，都需要打乱
        if any(limit != -1 for limit in use_num_limits_by_type.values()) or total_use_num != -1:
            random.shuffle(samples_to_process) 

        total_samples_collected = 0
        for sample in samples_to_process:
            if total_use_num != -1 and total_samples_collected >= total_use_num:
                # 如果已经达到总样本数限制，则停止添加
                break

            sample_type = sample['type']
            # 检查是否按类型限制，并且该类型的样本数未达到限制
            if use_num_limits_by_type.get(sample_type, -1) == -1 or use_num_limits_by_type[sample_type] > 0:
                filtered_samples.append(sample)
                total_samples_collected += 1
                if use_num_limits_by_type.get(sample_type, -1) != -1:
                    use_num_limits_by_type[sample_type] -= 1 # 对应类型的计数器减一
        
        return filtered_samples


class _VQASubset(Dataset):
    """
    CombinedVQADataset 内部使用的辅助类，表示一个训练或测试子集。
    它封装了 __len__ 和 __getitem__ 的实际逻辑。
    """
    def __init__(self, processor, image_size, samples, is_eval, model_name):
        self.processor = processor
        self.image_size = image_size
        self.samples = samples
        self.is_eval = is_eval
        self.model_name = model_name
        random.shuffle(self.samples)  # 打乱样本顺序，确保每次迭代的随机性

    def get_answers_list(self):
        return [s['answer_text'] for s in self.samples]

    def __len__(self):
        return len(self.samples)
    
    def get_type_list(self):
        return [s['type'] for s in self.samples]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_filename = sample["image_filename"]
        image_folder_path = sample["image_folder_path"]
        question_text = sample["question_text"]
        answer_text = sample["answer_text"]

        image_path = os.path.join(image_folder_path, image_filename)

        try:
            # 打开图片但先不进行resize
            image = Image.open(image_path).convert("RGB").resize(self.image_size)
        except Exception as e:
            print(f"加载图像 {image_path} 时出错：{e}。此样本返回 None。")
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
            base_path = os.getcwd() # 获取当前工作目录
            # # 组合基路径和相对路径，得到文件的完整绝对路径
            absolute_path = os.path.join(base_path, image_path)

            # # 将路径中的反斜杠（Windows系统可能出现）替换为正斜杠，并添加 'file:///' 前缀
            qwen_path = f"file://{absolute_path.replace('\\', '/')}"

            # print(f"原始相对路径: {image_path}")
            # print(f"转换后的 Qwen 路径: {qwen_path}")
            conversation_prompt_only = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": qwen_path,
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
                # max_length=1000
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
    





# # 为了示例运行，我们继续使用模拟的 Processor 和 Tokenizer
# class MockTokenizer:
#     def __init__(self):
#         self.word_to_id = {"USER": 100, "<image>": 101, "\n": 102, "?": 103, "ASSISTANT": 104, ":": 105, "蓝色": 106, "三棵": 107, "是": 108, "高": 109, "红色": 110, "两只": 111, "轿车": 112, "平静": 113, "没有": 114, "圆形": 115, "绿色": 116, "四条": 117, "否": 118, "一棵": 119, "yes": 120, "no": 121, "true": 122, "false": 123, "5": 124, "10": 125, "25.5": 126, "1": 127, "pad": 0, "unk": 99}
#         self.id_to_word = {v: k for k, v in self.word_to_id.items()}
#         self.pad_token_id = 0
#         self.unk_token_id = 99
#         self.image_token_id = 101

#     def __call__(self, text, add_special_tokens=False):
#         tokens = []
#         processed_text = text.replace("?", " ?").replace(":", " :").replace("\n", " \n ").replace("  ", " ").replace("!", " !").lower()
#         for word in processed_text.split(" "):
#             if word:
#                 tokens.append(self.word_to_id.get(word, self.unk_token_id))
#         return {'input_ids': tokens}

#     def convert_tokens_to_ids(self, token):
#         if token == '<image>':
#             return self.image_token_id
#         return self.word_to_id.get(token)

# class MockProcessor:
#     def __init__(self):
#         self.tokenizer = MockTokenizer()
#         self.image_token_id = self.tokenizer.image_token_id

#     def __call__(self, text, images, return_tensors, padding, truncation, max_length):
#         tokenized = self.tokenizer(text)
#         input_ids = torch.tensor([tokenized['input_ids']], dtype=torch.long)
#         attention_mask = torch.ones_like(input_ids)
        
#         pixel_values = torch.zeros((1, 3, 224, 224)) 

#         current_length = input_ids.shape[1]
#         if padding == "max_length" and current_length < max_length:
#             pad_length = max_length - current_length
#             input_ids = torch.cat([input_ids, torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1)
#             attention_mask = torch.cat([attention_mask, torch.zeros((1, pad_length), dtype=torch.long)], dim=1)
#         elif truncation and current_length > max_length:
#             input_ids = input_ids[:, :max_length]
#             attention_mask = attention_mask[:, :max_length]

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'pixel_values': pixel_values
#         }

# my_processor = MockProcessor()
# image_size = (224, 224)

# # 定义 collate_fn (与之前相同)
# def collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     if not batch:
#         return None

#     keys = batch[0].keys()
#     collated_batch = {}
#     for key in keys:
#         if key == 'pixel_values':
#             collated_batch[key] = torch.stack([d[key] for d in batch])
#         else:
#             collated_batch[key] = torch.stack([d[key] for d in batch])
#     return collated_batch


# # --- 创建模拟数据 (与之前相同，增加了 'true'/'false' 答案) ---
# def create_mock_data():
#     shutil.rmtree("data", ignore_errors=True)
#     os.makedirs("data/dataset1/images", exist_ok=True)
#     with open("data/dataset1/qa.txt", "w", encoding="utf-8") as f:
#         f.write("img001.tif:天空是什么颜色?蓝色\n") # Other
#         f.write("img002.tif:树木有几棵?三棵\n") # Number
#         f.write("img003.tif:路上有汽车吗?是\n") # YesOrNo (是 -> yes)
#         f.write("img004.tif:建筑高吗?高\n") # Other
#         f.write("img005.tif:水是流动的吗?是\n") # YesOrNo
#         f.write("img006.tif:云朵多吗?多\n") # Other
#         f.write("img007.tif:房子是新的吗?不是\n") # Other
#         f.write("img008.tif:地上有草吗?有\n") # YesOrNo
#         f.write("img009.tif:有多少朵花?5\n") # Number
#         f.write("img010.tif:河流宽吗?否\n") # YesOrNo (否 -> no)
#         f.write("img011.tif:有多少个人?10\n") # Number
#         f.write("img012.tif:猫在桌子上吗?yes\n") # YesOrNo
#         f.write("img013.tif:狗在椅子下面吗?no\n") # YesOrNo
#         f.write("img014.tif:温度是多少?25.5\n") # Number
#         f.write("img015.tif:有多少棵树?1\n") # Number
#         f.write("img016.tif:太阳升起来了吗?true\n") # YesOrNo
#         f.write("img017.tif:夜晚来临了吗?false\n") # YesOrNo
#     for i in range(1, 18):
#         Image.new('RGB', image_size, color = (73, 109, 137)).save(f"data/dataset1/images/img{i:03d}.tif")

#     os.makedirs("data/dataset2/images", exist_ok=True)
#     with open("data/dataset2/qa.txt", "w", encoding="utf-8") as f:
#         f.write("picA.tif:花朵是什么颜色?红色\n") # Other
#         f.write("picB.tif:有几只鸟?两只\n") # Other
#         f.write("picC.tif:这辆车是什么型号?轿车\n") # Other
#         f.write("picD.tif:水面平静吗?平静\n") # Other
#         f.write("picE.tif:有车吗?yes\n") # YesOrNo
#     for char in ['A', 'B', 'C', 'D', 'E']:
#         Image.new('RGB', image_size, color = (100, 150, 200)).save(f"data/dataset2/images/pic{char}.tif")

#     os.makedirs("data/dataset3/images", exist_ok=True)
#     with open("data/dataset3/qa.txt", "w", encoding="utf-8") as f:
#         f.write("photo1.tif:天空有飞机吗?没有\n") # Other
#         f.write("photo2.tif:这是一个城市吗?是\n") # YesOrNo
#     for i in range(1, 3):
#         Image.new('RGB', image_size, color = (200, 100, 50)).save(f"data/dataset3/images/photo{i}.tif")

#     os.makedirs("data/dataset4/images", exist_ok=True)
#     with open("data/dataset4/train_qa.txt", "w", encoding="utf-8") as f:
#         f.write("train_img1.tif:这个物体是什么形状?圆形\n") # Other
#         f.write("train_img2.tif:它是什么颜色?绿色\n") # Other
#         f.write("train_img3.tif:有三个苹果吗?no\n") # YesOrNo
#         f.write("train_img4.tif:这是白天吗?true\n") # YesOrNo
#         f.write("train_img5.tif:有多少个盒子?7\n") # Number
#     with open("data/dataset4/test_qa.txt", "w", encoding="utf-8") as f:
#         f.write("test_img1.tif:有几条边?四条\n") # Other
#         f.write("test_img2.tif:这是一个男人吗?yes\n") # YesOrNo
#         f.write("test_img3.tif:有多少个孩子?2\n") # Number
    
#     # 创建模拟图片文件
#     img_names = ["train_img1.tif", "train_img2.tif", "train_img3.tif", "train_img4.tif", "train_img5.tif", "test_img1.tif", "test_img2.tif", "test_img3.tif"]
#     for img_name in img_names:
#         Image.new('RGB', image_size, color = (50, 200, 100)).save(f"data/dataset4/images/{img_name}")

# create_mock_data()


# # --- 测试场景 1: 训练集总样本限制为 10, 测试集总样本限制为 3 (不区分类型) ---
# print("\n--- 测试场景 1: 训练集总样本限制为 10, 测试集总样本限制为 3 (不区分类型) ---")
# combined_dataset_loader_scenario1 = CombinedVQADataset(
#     processor=my_processor,
#     image_size=image_size,
#     use_num={'train': 10, 'test': 3}, # 训练集总数10，测试集总数3
#     task='all',
#     is_eval=False
# )
# combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
# combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
# combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
# combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)

# train_dataset_s1 = combined_dataset_loader_scenario1.get_dataset(split='train')
# test_dataset_s1 = combined_dataset_loader_scenario1.get_dataset(split='test')

# print(f"场景1 - 训练集总样本数: {len(train_dataset_s1)}")
# print(f"场景1 - 测试集总样本数: {len(test_dataset_s1)}")

# train_type_counts_s1 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
# for sample in train_dataset_s1.samples:
#     train_type_counts_s1[sample['type']] += 1
# print(f"场景1 - 训练集类型分布: {train_type_counts_s1}")

# test_type_counts_s1 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
# for sample in test_dataset_s1.samples:
#     test_type_counts_s1[sample['type']] += 1
# print(f"场景1 - 测试集类型分布: {test_type_counts_s1}")


# # --- 测试场景 2: 训练集按类型限制，测试集不限制 ---
# print("\n--- 测试场景 2: 训练集按类型限制，测试集不限制 ---")
# combined_dataset_loader_scenario2 = CombinedVQADataset(
#     processor=my_processor,
#     image_size=image_size,
#     use_num={'train': {'YesOrNo': 3, 'Number': 2, 'Other': 2}, 'test': -1}, # 训练集按类型，测试集不限制
#     task='all',
#     is_eval=False
# )
# combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
# combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
# combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
# combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)

# train_dataset_s2 = combined_dataset_loader_scenario2.get_dataset(split='train')
# test_dataset_s2 = combined_dataset_loader_scenario2.get_dataset(split='test')

# print(f"场景2 - 训练集总样本数: {len(train_dataset_s2)}") # 应该为 3+2+2 = 7
# print(f"场景2 - 测试集总样本数: {len(test_dataset_s2)}") # 应该为所有测试样本

# train_type_counts_s2 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
# for sample in train_dataset_s2.samples:
#     train_type_counts_s2[sample['type']] += 1
# print(f"场景2 - 训练集类型分布: {train_type_counts_s2}")

# test_type_counts_s2 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
# for sample in test_dataset_s2.samples:
#     test_type_counts_s2[sample['type']] += 1
# print(f"场景2 - 测试集类型分布: {test_type_counts_s2}")


# # --- 测试场景 3: 训练集只加载 'Number' 类型 (不限制数量), 测试集只加载 'YesOrNo' 类型 (不限制数量) ---
# print("\n--- 测试场景 3: 训练集只加载 'Number' 类型, 测试集只加载 'YesOrNo' 类型 ---")
# combined_dataset_loader_scenario3 = CombinedVQADataset(
#     processor=my_processor,
#     image_size=image_size,
#     use_num=None, # 不进行总数量限制
#     task='all', # 这里的 task='all' 表示不对整体数据进行初步类型筛选，而是由 get_dataset 内部的 use_num 决定
#     is_eval=False
# )
# # 注意：在这种场景下，需要分开调用 get_dataset，并传入 task 参数来筛选
# combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
# combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
# combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
# combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)

# # 重新实例化 CombinedVQADataset 来展示 'task' 参数的作用
# # 或者，可以修改 get_dataset 让它接收一个 task_override 参数
# # 但为了保持和 `RSVQA_LR_Dataset` 的行为一致（task 是初始化参数），我们在这里重新初始化
# # 实际上，如果你的 task 需求是每次 get_dataset 不同，get_dataset 应该有 task 参数。
# # 考虑到之前你提到了 'task' 参数是在 `__init__` 中设置的，这里保持一致。
# # 所以，如果需要不同 task 的数据集，你需要创建不同的 CombinedVQADataset 实例。
# # 这里为了演示，我们假设 task 是一个全局过滤。

# # 如果你需要 train 是 Number，test 是 YesOrNo，则需要两个不同的 CombinedVQADataset 实例
# # 实例 1: 用于获取 Number 类型的训练集
# combined_dataset_loader_train_num = CombinedVQADataset(
#     processor=my_processor,
#     image_size=image_size,
#     use_num=None,
#     task='Number', # 训练集只取 Number
#     is_eval=False
# )
# combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
# combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
# combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
# combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)
# train_dataset_s3 = combined_dataset_loader_train_num.get_dataset(split='train')

# # 实例 2: 用于获取 YesOrNo 类型的测试集
# combined_dataset_loader_test_yesno = CombinedVQADataset(
#     processor=my_processor,
#     image_size=image_size,
#     use_num=None,
#     task='YesOrNo', # 测试集只取 YesOrNo
#     is_eval=True # 评估模式
# )
# combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
# combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
# combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
# combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)
# test_dataset_s3 = combined_dataset_loader_test_yesno.get_dataset(split='test')


# print(f"场景3 - 训练集总样本数 (只Number类型): {len(train_dataset_s3)}")
# print(f"场景3 - 测试集总样本数 (只YesOrNo类型): {len(test_dataset_s3)}")

# train_type_counts_s3 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
# for sample in train_dataset_s3.samples:
#     train_type_counts_s3[sample['type']] += 1
# print(f"场景3 - 训练集类型分布: {train_type_counts_s3}")

# test_type_counts_s3 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
# for sample in test_dataset_s3.samples:
#     test_type_counts_s3[sample['type']] += 1
# print(f"场景3 - 测试集类型分布: {test_type_counts_s3}")


# # 清理模拟文件
# shutil.rmtree("data")