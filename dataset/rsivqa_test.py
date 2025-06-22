from transformers import AutoProcessor
import torch
import shutil
import logging
import os
from PIL import Image
from RSIVQA_Dateset import CombinedVQADataset

# 为了示例运行，我们继续使用模拟的 Processor 和 Tokenizer
class MockTokenizer:
    def __init__(self):
        self.word_to_id = {"USER": 100, "<image>": 101, "\n": 102, "?": 103, "ASSISTANT": 104, ":": 105, "蓝色": 106, "三棵": 107, "是": 108, "高": 109, "红色": 110, "两只": 111, "轿车": 112, "平静": 113, "没有": 114, "圆形": 115, "绿色": 116, "四条": 117, "否": 118, "一棵": 119, "yes": 120, "no": 121, "true": 122, "false": 123, "5": 124, "10": 125, "25.5": 126, "1": 127, "pad": 0, "unk": 99}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.pad_token_id = 0
        self.unk_token_id = 99
        self.image_token_id = 101

    def __call__(self, text, add_special_tokens=False):
        tokens = []
        processed_text = text.replace("?", " ?").replace(":", " :").replace("\n", " \n ").replace("  ", " ").replace("!", " !").lower()
        for word in processed_text.split(" "):
            if word:
                tokens.append(self.word_to_id.get(word, self.unk_token_id))
        return {'input_ids': tokens}

    def convert_tokens_to_ids(self, token):
        if token == '<image>':
            return self.image_token_id
        return self.word_to_id.get(token)

class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.image_token_id = self.tokenizer.image_token_id

    def __call__(self, text, images, return_tensors, padding, truncation, max_length):
        tokenized = self.tokenizer(text)
        input_ids = torch.tensor([tokenized['input_ids']], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        pixel_values = torch.zeros((1, 3, 224, 224)) 

        current_length = input_ids.shape[1]
        if padding == "max_length" and current_length < max_length:
            pad_length = max_length - current_length
            input_ids = torch.cat([input_ids, torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros((1, pad_length), dtype=torch.long)], dim=1)
        elif truncation and current_length > max_length:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }

my_processor = MockProcessor()
image_size = (224, 224)

# 定义 collate_fn (与之前相同)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if key == 'pixel_values':
            collated_batch[key] = torch.stack([d[key] for d in batch])
        else:
            collated_batch[key] = torch.stack([d[key] for d in batch])
    return collated_batch


# --- 创建模拟数据 (与之前相同，增加了 'true'/'false' 答案) ---
def create_mock_data():
    shutil.rmtree("data", ignore_errors=True)
    os.makedirs("data/dataset1/images", exist_ok=True)
    with open("data/dataset1/qa.txt", "w", encoding="utf-8") as f:
        f.write("img001.tif:天空是什么颜色?蓝色\n") # Other
        f.write("img002.tif:树木有几棵?三棵\n") # Number
        f.write("img003.tif:路上有汽车吗?是\n") # YesOrNo (是 -> yes)
        f.write("img004.tif:建筑高吗?高\n") # Other
        f.write("img005.tif:水是流动的吗?是\n") # YesOrNo
        f.write("img006.tif:云朵多吗?多\n") # Other
        f.write("img007.tif:房子是新的吗?不是\n") # Other
        f.write("img008.tif:地上有草吗?有\n") # YesOrNo
        f.write("img009.tif:有多少朵花?5\n") # Number
        f.write("img010.tif:河流宽吗?否\n") # YesOrNo (否 -> no)
        f.write("img011.tif:有多少个人?10\n") # Number
        f.write("img012.tif:猫在桌子上吗?yes\n") # YesOrNo
        f.write("img013.tif:狗在椅子下面吗?no\n") # YesOrNo
        f.write("img014.tif:温度是多少?25.5\n") # Number
        f.write("img015.tif:有多少棵树?1\n") # Number
        f.write("img016.tif:太阳升起来了吗?true\n") # YesOrNo
        f.write("img017.tif:夜晚来临了吗?false\n") # YesOrNo
    for i in range(1, 18):
        Image.new('RGB', image_size, color = (73, 109, 137)).save(f"data/dataset1/images/img{i:03d}.tif")

    os.makedirs("data/dataset2/images", exist_ok=True)
    with open("data/dataset2/qa.txt", "w", encoding="utf-8") as f:
        f.write("picA.tif:花朵是什么颜色?红色\n") # Other
        f.write("picB.tif:有几只鸟?两只\n") # Other
        f.write("picC.tif:这辆车是什么型号?轿车\n") # Other
        f.write("picD.tif:水面平静吗?平静\n") # Other
        f.write("picE.tif:有车吗?yes\n") # YesOrNo
    for char in ['A', 'B', 'C', 'D', 'E']:
        Image.new('RGB', image_size, color = (100, 150, 200)).save(f"data/dataset2/images/pic{char}.tif")

    os.makedirs("data/dataset3/images", exist_ok=True)
    with open("data/dataset3/qa.txt", "w", encoding="utf-8") as f:
        f.write("photo1.tif:天空有飞机吗?没有\n") # Other
        f.write("photo2.tif:这是一个城市吗?是\n") # YesOrNo
    for i in range(1, 3):
        Image.new('RGB', image_size, color = (200, 100, 50)).save(f"data/dataset3/images/photo{i}.tif")

    os.makedirs("data/dataset4/images", exist_ok=True)
    with open("data/dataset4/train_qa.txt", "w", encoding="utf-8") as f:
        f.write("train_img1.tif:这个物体是什么形状?圆形\n") # Other
        f.write("train_img2.tif:它是什么颜色?绿色\n") # Other
        f.write("train_img3.tif:有三个苹果吗?no\n") # YesOrNo
        f.write("train_img4.tif:这是白天吗?true\n") # YesOrNo
        f.write("train_img5.tif:有多少个盒子?7\n") # Number
    with open("data/dataset4/test_qa.txt", "w", encoding="utf-8") as f:
        f.write("test_img1.tif:有几条边?四条\n") # Other
        f.write("test_img2.tif:这是一个男人吗?yes\n") # YesOrNo
        f.write("test_img3.tif:有多少个孩子?2\n") # Number
    
    # 创建模拟图片文件
    img_names = ["train_img1.tif", "train_img2.tif", "train_img3.tif", "train_img4.tif", "train_img5.tif", "test_img1.tif", "test_img2.tif", "test_img3.tif"]
    for img_name in img_names:
        Image.new('RGB', image_size, color = (50, 200, 100)).save(f"data/dataset4/images/{img_name}")

create_mock_data()


# --- 测试场景 1: 训练集总样本限制为 10, 测试集总样本限制为 3 (不区分类型) ---
print("\n--- 测试场景 1: 训练集总样本限制为 10, 测试集总样本限制为 3 (不区分类型) ---")
combined_dataset_loader_scenario1 = CombinedVQADataset(
    processor=my_processor,
    image_size=image_size,
    use_num={'train': 10, 'test': 3}, # 训练集总数10，测试集总数3
    task='all',
    is_eval=False
)
combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
combined_dataset_loader_scenario1.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)

train_dataset_s1 = combined_dataset_loader_scenario1.get_dataset(split='train')
test_dataset_s1 = combined_dataset_loader_scenario1.get_dataset(split='test')

print(f"场景1 - 训练集总样本数: {len(train_dataset_s1)}")
print(f"场景1 - 测试集总样本数: {len(test_dataset_s1)}")

train_type_counts_s1 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
for sample in train_dataset_s1.samples:
    train_type_counts_s1[sample['type']] += 1
print(f"场景1 - 训练集类型分布: {train_type_counts_s1}")

test_type_counts_s1 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
for sample in test_dataset_s1.samples:
    test_type_counts_s1[sample['type']] += 1
print(f"场景1 - 测试集类型分布: {test_type_counts_s1}")


# --- 测试场景 2: 训练集按类型限制，测试集不限制 ---
print("\n--- 测试场景 2: 训练集按类型限制，测试集不限制 ---")
combined_dataset_loader_scenario2 = CombinedVQADataset(
    processor=my_processor,
    image_size=image_size,
    use_num={'train': {'YesOrNo': 3, 'Number': 2, 'Other': 2}, 'test': -1}, # 训练集按类型，测试集不限制
    task='all',
    is_eval=False
)
combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
combined_dataset_loader_scenario2.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)

train_dataset_s2 = combined_dataset_loader_scenario2.get_dataset(split='train')
test_dataset_s2 = combined_dataset_loader_scenario2.get_dataset(split='test')

print(f"场景2 - 训练集总样本数: {len(train_dataset_s2)}") # 应该为 3+2+2 = 7
print(f"场景2 - 测试集总样本数: {len(test_dataset_s2)}") # 应该为所有测试样本

train_type_counts_s2 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
for sample in train_dataset_s2.samples:
    train_type_counts_s2[sample['type']] += 1
print(f"场景2 - 训练集类型分布: {train_type_counts_s2}")

test_type_counts_s2 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
for sample in test_dataset_s2.samples:
    test_type_counts_s2[sample['type']] += 1
print(f"场景2 - 测试集类型分布: {test_type_counts_s2}")


# --- 测试场景 3: 训练集只加载 'Number' 类型 (不限制数量), 测试集只加载 'YesOrNo' 类型 (不限制数量) ---
print("\n--- 测试场景 3: 训练集只加载 'Number' 类型, 测试集只加载 'YesOrNo' 类型 ---")
combined_dataset_loader_scenario3 = CombinedVQADataset(
    processor=my_processor,
    image_size=image_size,
    use_num=None, # 不进行总数量限制
    task='all', # 这里的 task='all' 表示不对整体数据进行初步类型筛选，而是由 get_dataset 内部的 use_num 决定
    is_eval=False
)
# 注意：在这种场景下，需要分开调用 get_dataset，并传入 task 参数来筛选
combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
combined_dataset_loader_scenario3.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)

# 重新实例化 CombinedVQADataset 来展示 'task' 参数的作用
# 或者，可以修改 get_dataset 让它接收一个 task_override 参数
# 但为了保持和 `RSVQA_LR_Dataset` 的行为一致（task 是初始化参数），我们在这里重新初始化
# 实际上，如果你的 task 需求是每次 get_dataset 不同，get_dataset 应该有 task 参数。
# 考虑到之前你提到了 'task' 参数是在 `__init__` 中设置的，这里保持一致。
# 所以，如果需要不同 task 的数据集，你需要创建不同的 CombinedVQADataset 实例。
# 这里为了演示，我们假设 task 是一个全局过滤。

# 如果你需要 train 是 Number，test 是 YesOrNo，则需要两个不同的 CombinedVQADataset 实例
# 实例 1: 用于获取 Number 类型的训练集
combined_dataset_loader_train_num = CombinedVQADataset(
    processor=my_processor,
    image_size=image_size,
    use_num=None,
    task='Number', # 训练集只取 Number
    is_eval=False
)
combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
combined_dataset_loader_train_num.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)
train_dataset_s3 = combined_dataset_loader_train_num.get_dataset(split='train')

# 实例 2: 用于获取 YesOrNo 类型的测试集
combined_dataset_loader_test_yesno = CombinedVQADataset(
    processor=my_processor,
    image_size=image_size,
    use_num=None,
    task='YesOrNo', # 测试集只取 YesOrNo
    is_eval=True # 评估模式
)
combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset1/images", qa_file_path_train="data/dataset1/qa.txt", is_split=False)
combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset2/images", qa_file_path_train="data/dataset2/qa.txt", is_split=False)
combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset3/images", qa_file_path_train="data/dataset3/qa.txt", is_split=False)
combined_dataset_loader_test_yesno.add_dataset(image_folder_path="data/dataset4/images", qa_file_path_train="data/dataset4/train_qa.txt", qa_file_path_test="data/dataset4/test_qa.txt", is_split=True)
test_dataset_s3 = combined_dataset_loader_test_yesno.get_dataset(split='test')


print(f"场景3 - 训练集总样本数 (只Number类型): {len(train_dataset_s3)}")
print(f"场景3 - 测试集总样本数 (只YesOrNo类型): {len(test_dataset_s3)}")

train_type_counts_s3 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
for sample in train_dataset_s3.samples:
    train_type_counts_s3[sample['type']] += 1
print(f"场景3 - 训练集类型分布: {train_type_counts_s3}")

test_type_counts_s3 = {'YesOrNo': 0, 'Number': 0, 'Other': 0}
for sample in test_dataset_s3.samples:
    test_type_counts_s3[sample['type']] += 1
print(f"场景3 - 测试集类型分布: {test_type_counts_s3}")


# 清理模拟文件
shutil.rmtree("data")