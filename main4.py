from PIL import Image
import torch
import os
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig
import torch.nn as nn
from transformers.utils import is_torchdynamo_compiling
import numpy as np
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LlavaNextCausalLMOutputWithPast
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches

# 引入训练所需的库
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset # 用于自定义数据集

logger = logging.getLogger(__name__)

device = 'cuda:3'

# 设置环境变量，使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 原始模型名称和缓存路径
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
model_cache = "model_cache"

# 加载原始模型的配置
config = AutoConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)

# 图像处理参数
image_size = (672, 672)
# patch_num = image_size[0] // 336 * image_size[1] // 336 + 1
patch_num = image_size_to_num_patches(
                image_size=image_size,
                grid_pinpoints=config.image_grid_pinpoints,
                patch_size=config.vision_config.image_size,
            )

class ModifiedLlavaNext(LlavaNextForConditionalGeneration):
    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)
        # Add your new layer here
        self.image_seq_length = config.image_seq_length
        self.soft_prompts = nn.Parameter(torch.randn(patch_num, config.vision_config.intermediate_size))  # 添加soft prompt参数
        # self.linear_projector = nn.Linear(config.vision_config.intermediate_size * 2, config.vision_config.intermediate_size)
        input_dim = config.vision_config.intermediate_size * 2
        output_dim = config.vision_config.intermediate_size

        self.linear_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim), # Example: double the input dimension for the first hidden layer
            nn.GELU(), # Non-linear activation (Gaussian Error Linear Unit)
            nn.Linear(input_dim, output_dim), # Example: reduce to original intermediate_size * 2
        )
        self.global_prompt_self_attention = nn.MultiheadAttention(
            embed_dim=config.vision_config.intermediate_size,  # 或 config.hidden_size，取决于你将在哪里使用它
            num_heads=4,  # 你可以根据你的需求调整这个数字
            batch_first=True # 如果你的输入批次维度在前，则设置为 True
        )
        # (self.soft_prompts.unsqueeze(0).expand(config.image_seq_length,-1,-1) * self.image_seq_weight.unsqueeze(2))

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
    ):
        image_num_patches = [
            patch_num for imsize in image_sizes
        ]
        # print(f"image_num_patches: {image_num_patches}")
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0) # type: ignore
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        # print(f"pixel_values shape: {pixel_values.shape}")
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_features.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        
        # print(f"selected_image_feature shape: {selected_image_feature.shape}")

        image_features = self.multi_modal_projector(selected_image_feature)
        # print(f"image_features shape: {image_features.shape}")
        image_features = torch.split(image_features, image_num_patches, dim=0)
     
        return image_features
    
    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_select_strategy (`str`)
                The feature selection strategy used to select the vision feature from the vision backbone.
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )

                if (
                    np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                    and vision_feature_select_strategy == "default"
                ):
                    logger.warning_once( # type: ignore
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS."
                    )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    # [4096, 48, 48]
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                    # [4096, 48, 49]
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes, # type: ignore
                vision_feature_layer=vision_feature_layer, # type: ignore
                vision_feature_select_strategy=vision_feature_select_strategy, # type: ignore
            )
            # 结果 [5, 576, 4096]，对应5个patch，每个patch有576个token，每个token有4096维特征
            # TODO: 可以在这里插入修改代码，先concat p tensor，再线性层降维，还有生成后面需要的额外token
            image_features_processed = []
            for image_feature in image_features:
                image_features_with_soft_prompt = torch.cat((image_feature, self.soft_prompts.unsqueeze(1).expand(-1,self.image_seq_length,-1)), dim=-1)
                image_features_processed.append(self.linear_projector(image_features_with_soft_prompt))
            image_features = tuple(image_features_processed)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )
            # 结果 [2928, 4096] 5*576 = 2880， + 48个额外的token
            # 新添加的48个token叫 newline，可能表示图像换行符

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device) # type: ignore
            # TODO: 只需要在这里把特殊token conncat 到image_features后面就行了

            global_prompt_token, attn_weights = self.global_prompt_self_attention(self.soft_prompts,self.soft_prompts,self.soft_prompts)
            global_prompt_token = torch.mean(global_prompt_token, dim=0).unsqueeze(0)
            image_features = torch.cat((image_features, global_prompt_token), dim=0)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel(): # type: ignore
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype) # type: ignore
            # print(f"image_features shape: {image_features.shape}, inputs_embeds shape: {inputs_embeds.shape}") # type: ignore
            # print(f"special_image_mask shape: {special_image_mask.shape}, special_image_mask: {special_image_mask}")
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features) # type: ignore
            # print(f"inputs_embeds shape after scatter: {inputs_embeds.shape}") # type: ignore

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None, # type: ignore
        )

# --- 训练代码部分 ---

# 1. 冻结原始模型的参数
modified_config = LlavaNextConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)
modified_model = ModifiedLlavaNext.from_pretrained(model_name, config=modified_config, cache_dir=model_cache, device_map="auto")

# 冻结所有参数
for param in modified_model.parameters():
    param.requires_grad = False

# 解冻你新增的模块的参数
modified_model.soft_prompts.requires_grad = True
for param in modified_model.linear_projector.parameters():
    param.requires_grad = True
for param in modified_model.global_prompt_self_attention.parameters():
    param.requires_grad = True

# 确认哪些参数正在训练
print("Parameters being trained:")
for name, param in modified_model.named_parameters():
    if param.requires_grad:
        print(name)

# 加载处理器
processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1 + 1) # 增加一个额外的token用于处理图像输入

# 2. 准备数据集
# 你需要将这部分替换为你的实际数据集加载逻辑。
# 这里只是一个虚拟数据集的示例。
class CustomDataset(Dataset):
    def __init__(self, processor, image_paths, texts, image_size):
        self.processor = processor
        self.image_paths = image_paths
        self.texts = texts
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        try:
            image = Image.open(image_path).convert("RGB").resize(self.image_size)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # 返回一个虚拟的空数据或者跳过这个样本
            return None # 这在 collate_fn 中需要处理

        # 处理文本和图像
        # 这里需要将 labels 和 input_ids 对齐
        # 对于视觉-语言模型，通常 labels 会是文本部分的 input_ids
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=3000)
        
        # 对于生成任务，通常 labels 等于 input_ids，但是会忽略图像token和prompt token的损失
        inputs["labels"] = inputs["input_ids"].clone()
        
        # 确保 labels 中图像 token 和任何你想忽略损失的 token 设置为 -100
        # self.config.image_token_index 是图像占位符的token ID
        # 你的 prompt token 和 soft_prompts 相关的 token 位置也可能需要设置为 -100
        # 找到图像 token 的位置并将其 labels 设为 -100
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100 # 忽略pad token的损失
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.unk_token_id] = -100 # 忽略unk token的损失
        inputs["labels"][inputs["labels"] == self.processor.image_token_id] = -100 # 忽略图像token的损失

        # 将所有张量从 batch 维度中挤压出来，因为 DataLoader 会重新添加批次维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

# 示例数据 (请替换为你的实际数据)
# 假设你有一个图像文件列表和对应的文本描述列表
dummy_image_paths = ["1.jpg", "1.jpg"] # 替换为你的图像路径列表
dummy_texts = ["USER: <image>\n What's the content of the image? ASSISTANT: This is a test image.", "USER: <image>\n Describe the scene. ASSISTANT: A detailed description of the image."]

# 创建数据集实例
train_dataset = CustomDataset(processor, dummy_image_paths, dummy_texts, image_size)

print(f'processor.tokenizer.pad_token_id: {processor.tokenizer.pad_token_id}')
# processor.tokenizer.pad_token_id: 32001

# 定义 collate_fn 来处理批量数据
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
            collated_batch[key] = torch.nn.utils.rnn.pad_sequence([x[key] for x in batch], 
                                                                   batch_first=True, 
                                                                   padding_value=processor.tokenizer.pad_token_id)
            # 对于 attention_mask，padding_value 应该是 0
            if key == "attention_mask":
                collated_batch[key][collated_batch[key] == processor.tokenizer.pad_token_id] = 0
            # 对于 labels，padding_value 应该是 -100
            elif key == "labels":
                collated_batch[key][collated_batch[key] == processor.tokenizer.pad_token_id] = -100
    
    return collated_batch


# 3. 设置 TrainingArguments
training_args = TrainingArguments(
    output_dir="./output/llava_next_modified_finetune", # 训练输出目录
    # num_train_epochs=3,                                # 训练轮数
    num_train_epochs=15,
    per_device_train_batch_size=1,                     # 每个设备的训练批量大小
    gradient_accumulation_steps=4,                     # 梯度累积步数，模拟更大的batch size
    learning_rate=5e-5,                                # 学习率
    weight_decay=0.01,                                 # 权重衰减
    logging_dir="./logs",                              # 日志目录
    logging_steps=10,                                  # 每隔多少步记录一次日志
    save_steps=500,                                    # 每隔多少步保存一次模型
    save_total_limit=2,                                # 最多保存的模型数量
    do_train=True,                                     # 执行训练
    fp16=True,                                         # 使用混合精度训练
    report_to="none",                                  # 不报告到任何平台 (可选，如果你想用wandb等可以设置)
)

# # 4. 创建 Trainer 实例
# trainer = Trainer(
#     model=modified_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     data_collator=custom_collate_fn, # 使用自定义的 collate_fn
# )

# # 5. 开始训练
# print("Starting training...")
# trainer.train()
# print("Training finished!")

# # 6. 保存训练好的模型（只有新增的模块）
# # 如果你只训练了新增模块，保存整个模型会包含基础模型的权重
# # 如果你想只保存你训练的 soft_prompts, linear_projector 和 global_prompt_self_attention
# # 你需要手动保存它们的 state_dict
# # 或者在训练结束后，Trainer 会保存整个模型，你可以加载后手动提取你需要的权重

# # 示例：手动保存新增模块的权重
# output_dir = "./output/llava_next_modified_finetune/trained_modules"
# os.makedirs(output_dir, exist_ok=True)

# torch.save(modified_model.soft_prompts, os.path.join(output_dir, "soft_prompts.pt"))
# torch.save(modified_model.linear_projector.state_dict(), os.path.join(output_dir, "linear_projector.pt"))
# torch.save(modified_model.global_prompt_self_attention.state_dict(), os.path.join(output_dir, "global_prompt_self_attention.pt"))
# print(f"新模块权重已保存到：{output_dir}")

# --- 训练后的测试部分 (可选) ---
# 你可以再次运行你原始的测试代码来验证微调后的模型效果

# # 定义输入的提示文本，包含用户和助手的对话格式
prompt_test = "USER: <image>\n What's the content of the image? ASSISTANT:"

# 定义图像文件的路径
url_test = "1.jpg"  # 替换为你的图像路径

try:
    # 打开图像文件
    image_test = Image.open(fp=url_test).resize(image_size)

    # 使用处理器处理文本和图像，生成模型输入
    inputs_test = processor(text=prompt_test, images=image_test, return_tensors="pt")

    # 将输入数据移动到GPU上
    for temp_key in inputs_test.keys():
        inputs_test[temp_key] = inputs_test[temp_key].to(device)

    # 使用修改后的模型生成输出
    modified_model.eval() # 切换到评估模式
    with torch.no_grad():
        generate_ids_test = modified_model.generate(**inputs_test, max_new_tokens=100)

    # 将生成的token解码为文本，跳过特殊token并清理空格
    response_test = processor.batch_decode(
        generate_ids_test, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 打印模型的响应
    print("\n\nresponse from modified model after finetuning:\n", response_test)

except FileNotFoundError:
    print(f"Error: Image file not found at {url_test}")
except Exception as e:
    print(f"An error occurred during post-finetune processing: {e}")