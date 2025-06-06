import torch


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

            # processor.tokenizer.pad_token_id: 32001
            collated_batch[key] = torch.nn.utils.rnn.pad_sequence([x[key] for x in batch], 
                                                                   batch_first=True, 
                                                                   padding_value=32001)
            # 对于 attention_mask，padding_value 应该是 0
            if key == "attention_mask":
                collated_batch[key][collated_batch[key] == 32001] = 0
            # 对于 labels，padding_value 应该是 -100
            elif key == "labels":
                collated_batch[key][collated_batch[key] == 32001] = -100
    
    return collated_batch