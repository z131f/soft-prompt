from PIL import Image
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
from transformers.utils import is_torchdynamo_compiling
import math
import numpy as np
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LLAVA_NEXT_INPUTS_DOCSTRING, LlavaNextCausalLMOutputWithPast
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape

logger = logging.getLogger(__name__)

# 原始模型名称
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
device = "cuda:1"  # 定义使用的设备（GPU）
model_cache = "model_cache"  # 定义模型缓存路径


# 加载原始模型的配置
config = AutoConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)

class ModifiedLlavaNext(LlavaNextForConditionalGeneration):
    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)
        # Add your new layer here
        self.new_linear = nn.Linear(config.vision_config.intermediate_size, config.vision_config.intermediate_size)

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
        print(f"image_num_patches: {image_num_patches}")
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0) # type: ignore
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        print(f"pixel_values shape: {pixel_values.shape}")
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
                    logger.warning_once(
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
    

# 使用修改后的模型类
modified_config = LlavaNextConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)
modified_model = ModifiedLlavaNext.from_pretrained(model_name, config=modified_config, cache_dir=model_cache, device_map="auto", trust_remote_code=True)

# 加载处理器（AutoProcessor），用于处理输入数据
processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True, num_additional_image_tokens=1+1)  # 增加一个额外的token用于处理图像输入
# 现在只用填充这个token位置就好了

# 定义输入的提示文本，包含用户和助手的对话格式
prompt = "USER: <image>\n What's the content of the image? ASSISTANT:"

# 定义图像文件的路径
url = "1.jpg"  # 替换为你的图像路径

image_size = (672, 672)  # 定义图像的resize大小
patch_num = image_size[0] // 336 * image_size[1] // 336 + 1  # 对应4个patch和一个额外patch
# 通过resize图像来固定patch数，不用修改processer了

try:
    # 打开图像文件
    image = Image.open(fp=url).resize(image_size)  # 调整图像大小为512x512

    # 使用处理器处理文本和图像，生成模型输入
    inputs = processor(text=prompt, images=image, return_tensors="pt") # type: ignore

    # 将输入数据移动到GPU上
    for temp_key in inputs.keys():
        inputs[temp_key] = inputs[temp_key].to("cuda:0")

    # 使用修改后的模型生成输出
    generate_ids = modified_model.generate(**inputs, max_new_tokens=100)

    # 将生成的token解码为文本，跳过特殊token并清理空格
    response = processor.batch_decode( # type: ignore
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 打印模型的响应
    print("\n\nresponse from modified model:\n", response)

except FileNotFoundError:
    print(f"Error: Image file not found at {url}")
except Exception as e:
    print(f"An error occurred during processing: {e}")