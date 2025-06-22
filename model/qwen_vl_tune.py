import random
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLConfig
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
import torch
from torch import nn
from transformers.utils import is_torchdynamo_compiling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import seaborn as sns

class ModifiedQwenVL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig, patch_num=0, extra_fun=None):
        super().__init__(config)
        self.config = config
        self.patch_num = patch_num # 81 324
        self.token_output_dim = 3584
        self.extra_fun = extra_fun
        if self.extra_fun is not None:
            if self.extra_fun == 'pool1':
                self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
                self.patch_num = self.patch_num // 2
            elif self.extra_fun == 'pool2':
                self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
                self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
                self.patch_num = self.patch_num // 2 // 2
        self.soft_prompts = nn.Parameter(torch.randn(self.patch_num, self.token_output_dim))  # 添加soft prompt参数
        self.k_projector_image = nn.Linear(self.token_output_dim, self.token_output_dim)
        self.q_projector_prompt = nn.Linear(self.token_output_dim, self.token_output_dim)

    def set_train(self):
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 解冻你新增的模块的参数
        self.soft_prompts.requires_grad = True
        for param in self.k_projector_image.parameters():
            param.requires_grad = True
        for param in self.q_projector_prompt.parameters():
            param.requires_grad = True

        # 确认哪些参数正在训练
        print("Parameters being trained:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                # 修改
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) # 81 3584
                assert image_embeds.shape[1] == 3584

                if self.extra_fun == 'pool1':
                    #print(f"原始特征形状: {image_embeds.shape}")
                    image_embeds_reshaped = image_embeds.permute(1, 0).unsqueeze(0)
                    #print(f"转换后的特征形状 (适配池化层): {image_embeds_reshaped.shape}")
                    embd_pooled = self.pool1(image_embeds_reshaped)
                    #print(f"池化后的形状 (来自池化层): {embd_pooled.shape}")
                    embd_final = embd_pooled.squeeze(0).permute(1, 0)
                    #print(f"最终输出特征形状: {embd_final.shape}")
                elif self.extra_fun == 'pool2':
                    #print(f"原始特征形状: {image_embeds.shape}")
                    image_embeds_reshaped = image_embeds.permute(1, 0).unsqueeze(0)
                    #print(f"转换后的特征形状 (适配池化层): {image_embeds_reshaped.shape}")
                    embd_pooled1 = self.pool1(image_embeds_reshaped)
                    #print(f"第一次池化后的形状: {embd_pooled1.shape}")
                    embd_pooled2 = self.pool2(embd_pooled1)
                    #print(f"第二次池化后的形状: {embd_pooled2.shape}")
                    embd_final = embd_pooled2.squeeze(0).permute(1, 0)
                    #print(f"最终输出特征形状: {embd_final.shape}")
                else:
                    embd_final = image_embeds

                assert embd_final.shape[0] == self.patch_num, print(f"输入图像特征数量与预设数量不一致: {embd_final.shape[0]} != {self.patch_num}")

                soft_prompts_q = self.q_projector_prompt(self.soft_prompts)
                image_feature_k = self.k_projector_image(embd_final)
                score = image_feature_k @ soft_prompts_q.T / np.sqrt(self.token_output_dim)
                weight = torch.softmax(torch.diag(score), dim=-1).to(self.soft_prompts.device)
                global_prompt = (weight.unsqueeze(1) * self.soft_prompts).sum(dim=0)
                image_embeds = torch.cat([image_embeds, global_prompt.unsqueeze(0)], dim=0)

                # tem
                # heatmap_data = weight.float().reshape((18,18)).cpu().detach().numpy()
                # # 2. 绘制热力图
                # plt.figure(figsize=(8, 7)) # 可以调整图的大小
                # sns.heatmap(heatmap_data, annot=False, cmap='viridis', fmt=".2f", linewidths=.5)
                # # annot=True 会在每个单元格显示数值
                # # cmap='viridis' 是一个常用的颜色映射，你也可以尝试 'plasma', 'magma', 'coolwarm' 等
                # # fmt=".2f" 将数值格式化为保留两位小数的浮点数
                # # linewidths=.5 增加单元格之间的线条，使其更清晰
                # # plt.title('Weight Heatmap (9x9)')
                # plt.xlabel('Column Index')
                # plt.ylabel('Row Index')
                # tag = random.randint(0, 999999)
                # plt.savefig(f'images/weight_heatmap_{str(tag)}.png', bbox_inches='tight', dpi=300)
                # tem end

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device) # type: ignore

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype) # type: ignore
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds) # type: ignore

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype) # type: ignore
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device) # type: ignore

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype) # type: ignore
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds) # type: ignore

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device) # type: ignore

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0) # type: ignore
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )