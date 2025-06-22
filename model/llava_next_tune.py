from typing import List, Optional, Tuple, Union
import numpy as np
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LlavaNextCausalLMOutputWithPast
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape, image_size_to_num_patches
import torch
from torch import nn
from transformers.utils import is_torchdynamo_compiling


class ModifiedLlavaNext(LlavaNextForConditionalGeneration):
    def __init__(self, config: LlavaNextConfig, patch_num, extra_fun=None):
        super().__init__(config)
        # Add your new layer here
        self.patch_num = patch_num
        self.extra_fun = extra_fun
        if self.extra_fun is not None:
            if self.extra_fun == 'pool1':
                self.patch_num -= 1
            elif self.extra_fun == 'pool2':
                self.patch_num -= 2
            else:
                raise ValueError(f"Unsupported extra_fun: {self.extra_fun}. Supported values are 'pool1' and 'pool2'.")
        self.image_seq_length = config.image_seq_length
        self.soft_prompts = nn.Parameter(torch.randn(self.patch_num, config.vision_config.intermediate_size))  # 添加soft prompt参数
        # self.linear_projector = nn.Linear(config.vision_config.intermediate_size * 2, config.vision_config.intermediate_size)
        # self.input_dim = config.vision_config.intermediate_size * 2
        self.output_dim = config.vision_config.intermediate_size

        # self.q_projector_image = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size // 2)
        self.k_projector_image = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size)
        # self.v_projector_image = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size)

        self.q_projector_prompt = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size)
        # self.k_projector_prompt = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size // 2)
        # self.v_projector_prompt = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size)

        # 获取权重和偏置的维度
        # out_features, in_features = self.linear_projector.weight.shape # (intermediate_size, intermediate_size * 2)


        # self.linear_projector = nn.Sequential(
        #     nn.Linear(input_dim, input_dim), # Example: double the input dimension for the first hidden layer
        #     nn.GELU(), # Non-linear activation (Gaussian Error Linear Unit)
        #     nn.Linear(input_dim, output_dim), # Example: reduce to original intermediate_size * 2
        # )
        # self.global_prompt_self_attention = nn.MultiheadAttention(
        #     embed_dim=config.vision_config.intermediate_size,  # 或 config.hidden_size，取决于你将在哪里使用它
        #     num_heads=1,  # 你可以根据你的需求调整这个数字
        #     batch_first=True # 如果你的输入批次维度在前，则设置为 True
        # )
        # (self.soft_prompts.unsqueeze(0).expand(config.image_seq_length,-1,-1) * self.image_seq_weight.unsqueeze(2))

    def set_train(self):
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 解冻你新增的模块的参数
        self.soft_prompts.requires_grad = True
        # for param in self.linear_projector.parameters():
        #     param.requires_grad = True
        # for param in self.q_projector_image.parameters():
        #     param.requires_grad = True
        for param in self.k_projector_image.parameters():
            param.requires_grad = True
        for param in self.q_projector_prompt.parameters():
            param.requires_grad = True
        # for param in self.v_projector_image.parameters():
        #     param.requires_grad = True
        # for param in self.k_projector_prompt.parameters():
        #     param.requires_grad = True
        # for param in self.v_projector_prompt.parameters():
        #     param.requires_grad = True
        # for param in self.global_prompt_self_attention.parameters():
        #     param.requires_grad = True

        # 确认哪些参数正在训练
        print("Parameters being trained:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

        # # 1. 初始化偏置为全0
        # nn.init.constant_(self.linear_projector.bias, 0)

        # # 2. 初始化权重
        # with torch.no_grad():
        #     out_features = self.output_dim
        #     in_features = self.input_dim

        #     # 创建一个与权重形状相同的全零张量
        #     weight_data = torch.zeros((out_features, in_features), dtype=self.linear_projector.weight.dtype)

        #     # 创建一个单位矩阵
        #     identity_matrix = torch.eye(out_features, dtype=self.linear_projector.weight.dtype)

        #     # 将单位矩阵放置在 weight_data 的前一半列
        #     # weight_data[:, :out_features] 是指所有行，前 out_features 列
        #     weight_data[:, :out_features].copy_(identity_matrix)

        #     # 将初始化后的数据赋值给线性层的权重
        #     self.linear_projector.weight.copy_(weight_data)

    # def get_image_features(
    #     self,
    #     pixel_values: torch.FloatTensor,
    #     image_sizes: torch.Tensor,
    #     vision_feature_layer: Union[int, List[int]],
    #     vision_feature_select_strategy: str,
    # ):
    #     image_num_patches = [
    #         self.patch_num for imsize in image_sizes
    #     ]
    #     # print(f"image_num_patches: {image_num_patches}")
    #     if pixel_values.dim() == 5:
    #         # stacked if input is (batch_size, num_patches, num_channels, height, width)
    #         _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
    #         pixel_values = torch.cat(_pixel_values_list, dim=0) # type: ignore
    #     elif pixel_values.dim() != 4:
    #         # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
    #         raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

    #     # print(f"pixel_values shape: {pixel_values.shape}")
    #     image_features = self.vision_tower(pixel_values, output_hidden_states=True)
    #     if isinstance(vision_feature_layer, int):
    #         selected_image_feature = image_features.hidden_states[vision_feature_layer]
    #     else:
    #         hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
    #         selected_image_feature = torch.cat(hs_pool, dim=-1)

    #     if vision_feature_select_strategy == "default":
    #         selected_image_feature = selected_image_feature[:, 1:]
    #     elif vision_feature_select_strategy == "full":
    #         selected_image_feature = selected_image_feature
        
    #     # print(f"selected_image_feature shape: {selected_image_feature.shape}")

    #     image_features = self.multi_modal_projector(selected_image_feature)
    #     # print(f"image_features shape: {image_features.shape}")
    #     image_features = torch.split(image_features, image_num_patches, dim=0)
     
    #     return image_features
    
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
        # print("权重：\n", self.linear_projector.weight) # 打印前5列看看是不是1
        # print("\n偏置初始化完成：\n", self.linear_projector.bias) # 打印偏置看看是不是0


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
            global_prompt_list = []
            for image_feature in image_features:
                if self.extra_fun is not None:
                    modified_image_feature = image_feature
                    if self.extra_fun == 'pool1':
                        fea1 = image_feature[:-2, :, :]
                        fea2 = image_feature[-2:, :, :]
                        fea2 = torch.mean(fea2, dim=0, keepdim=True)
                        modified_image_feature = torch.cat([fea1, fea2], dim=0)
                    elif self.extra_fun == 'pool2':
                        if image_feature.shape[0] == 3:
                            modified_image_feature = torch.mean(image_feature, dim=0, keepdim=True)
                        elif image_feature.shape[0] == 5:
                            fea1 = image_feature[0, :, :].unsqueeze(0)
                            fea2 = image_feature[1:3, :, :]
                            fea2 = torch.mean(fea2, dim=0, keepdim=True)
                            fea3 = image_feature[3:, :, :]
                            fea3 = torch.mean(fea3, dim=0, keepdim=True)
                            modified_image_feature = torch.cat([fea1, fea2, fea3], dim=0)
                        else:
                            raise ValueError(f"feature shape {image_feature.shape} is not supported for extra_fun {self.extra_fun}. Supported shapes are (3, 576, 4096) and (5, 576, 4096).")
                else:
                    modified_image_feature = image_feature

                soft_prompts_q = self.q_projector_prompt(self.soft_prompts)
                image_feature_k = self.k_projector_image(modified_image_feature)
                score = (image_feature_k @ soft_prompts_q.T / np.sqrt(self.output_dim)).mean(dim=1) # [patch prompt]
                weight = torch.softmax(torch.diag(score), dim=-1).to(self.soft_prompts.device) # [patch prompt]
                global_prompt = (weight.unsqueeze(1) * self.soft_prompts).sum(dim=0)
                global_prompt_list.append(global_prompt)

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

            # global_prompt_token, attn_weights = self.global_prompt_self_attention(self.soft_prompts,self.soft_prompts,self.soft_prompts)
            # global_prompt_token = torch.mean(global_prompt_token, dim=0).unsqueeze(0)
            new_feature_list = []
            images_input_num = input_ids.shape[0]
            images_feature_dim = (input_ids[0] == self.config.image_token_index).sum()-1
            for input_index in range(images_input_num):
                new_feature_list.append(image_features[images_feature_dim*(input_index):(images_feature_dim*(input_index+1))].to(inputs_embeds.device, inputs_embeds.dtype))
                new_feature_list.append(global_prompt_list[input_index].unsqueeze(0).to(inputs_embeds.device, inputs_embeds.dtype))
                # new_feature_list.append(global_prompt_token.to(inputs_embeds.device, inputs_embeds.dtype))
            image_features = torch.cat(new_feature_list, dim=0)

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