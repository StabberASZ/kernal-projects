# prepare_data_and_model.py (Final Version with Collator)

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image


# --- 新增：自定义整理函数 (Collator) ---
# 这个类负责将 __getitem__ 返回的多个样本打包成一个批次
class VLMCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # batch 是一个列表，列表中的每个元素都是 __getitem__ 返回的字典
        images = [item['image'] for item in batch]
        prompts = [item['prompt'] for item in batch]

        # 使用 processor 对整个批次的文本和图像进行一次性处理
        # padding=True 会自动将文本填充到批次中最长的长度
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # 为损失计算创建 labels
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs


# --- 修改：自定义数据集类 ---
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item['image'].convert('RGB')
        caption = item['answer'][0]
        text_prompt = "Please carefully observe the image and come up with a caption for the image."

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]}
        ]

        # 这里只生成用于处理的文本提示 (prompt)，不进行填充或转换为张量
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        # 返回原始的 image 和 prompt，交由 collator 处理
        return {"image": image, "prompt": prompt}


# 主加载函数
def load_finetune_resources(train_batch=4, test_batch=8):
    """
    加载并准备用于VLM微调的模型、处理器和数据加载器。
    """
    print("Loading pre-trained model and processor...")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)

    # 将 processor.tokenizer 的 pad_token 设置为 eos_token
    # 这对于开源模型在微调时是常见的做法
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading COCO-Caption2017 dataset...")

    full_dataset = load_dataset("lmms-lab/COCO-Caption2017", split='val[:5%]')

    print("Splitting 'val' data into new train and validation sets...")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

    coco_dataset_train = split_dataset['train']
    coco_dataset_val = split_dataset['test']

    print("Creating custom datasets...")
    train_dataset = ImageCaptioningDataset(coco_dataset_train, processor)
    valid_dataset = ImageCaptioningDataset(coco_dataset_val, processor)

    # 实例化我们自定义的 collator
    collator = VLMCollator(processor)

    print("Creating data loaders...")
    # 在 DataLoader 中指定 collate_fn
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch, shuffle=False, collate_fn=collator)

    test_loader = valid_loader

    return model, processor, train_loader, valid_loader, test_loader