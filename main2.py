from PIL import Image
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
import math
import numpy as np
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextPreTrainedModel, LLAVA_NEXT_INPUTS_DOCSTRING, LlavaNextCausalLMOutputWithPast
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from transformers.models.llava_next import LlavaNextForConditionalGeneration

logger = logging.getLogger(__name__)

# 原始模型名称
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
device = "cuda:1"  # 定义使用的设备（GPU）
model_cache = "model_cache"  # 定义模型缓存路径


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit # type: ignore


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints) # type: ignore
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches

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
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0) # type: ignore
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

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

        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        
        return image_features

# 使用修改后的模型类
modified_config = LlavaNextConfig.from_pretrained(model_name, cache_dir=model_cache, trust_remote_code=True)
modified_model = ModifiedLlavaNext.from_pretrained(model_name, config=modified_config, cache_dir=model_cache, device_map="auto", trust_remote_code=True)

# 加载处理器（AutoProcessor），用于处理输入数据
processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache, use_fast=True)

# 定义输入的提示文本，包含用户和助手的对话格式
prompt = "USER: <image>\n What's the content of the image? ASSISTANT:"

# 定义图像文件的路径
url = "1.jpg"  # 替换为你的图像路径

try:
    # 打开图像文件
    image = Image.open(fp=url)

    # 使用处理器处理文本和图像，生成模型输入
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # 将输入数据移动到GPU上
    for temp_key in inputs.keys():
        inputs[temp_key] = inputs[temp_key].to("cuda:0")

    # 使用修改后的模型生成输出
    generate_ids = modified_model.generate(**inputs, max_new_tokens=100)

    # 将生成的token解码为文本，跳过特殊token并清理空格
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 打印模型的响应
    print("\n\nresponse from modified model:\n", response)

except FileNotFoundError:
    print(f"Error: Image file not found at {url}")
except Exception as e:
    print(f"An error occurred during processing: {e}")

print("\nreturn_tensors=\"pt\" 是指返回 PyTorch 张量。")