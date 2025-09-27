# fine_vlm_modified.py

import torch
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

# 导入新的数据和模型加载函数
from prepare_data_and_model import load_finetune_resources

#  set_random_seed 和 CLEAN_MODEL_DIR 在 utils.py 中定义
from utils import set_random_seed, CLEAN_MODEL_DIR


def main(task_name, train_batch=4):
    num_epochs = 100

    # 使用新函数加载模型、处理器和数据加载器
    # 这里的批大小可以根据GPU显存进行调整
    model, processor, train_loader, valid_loader, test_loader = (
        load_finetune_resources(train_batch=train_batch, test_batch=4))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 对于生成模型，损失函数由模型内部计算，所以不需要单独定义 criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    model = model.to(device)
    best_val_loss = float('inf')

    # 如果目录不存在，则创建
    if not os.path.exists(CLEAN_MODEL_DIR):
        os.makedirs(CLEAN_MODEL_DIR)

    print(f"Starting fine-tuning for task: {task_name}")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"):
            # 将批次中的所有张量移动到GPU
            inputs = {k: v.to(device) for k, v in batch.items()}

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播，模型会返回包含 loss 的输出
            outputs = model(**inputs)
            loss = outputs.loss

            # 记录训练损失
            total_train_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}')

        # 每10个epoch或第一个epoch后进行验证
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**inputs)
                    loss = outputs.loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(test_loader)
            print(f'Average Validation Loss: {avg_val_loss:.4f}')

            # 如果验证损失降低，则保存模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(CLEAN_MODEL_DIR, f"{task_name}_best.pth")
                torch.save(model.state_dict(), save_path)
                print(f'New best model saved with validation loss: {best_val_loss:.4f} to {save_path}')
                print('-----------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 参数从 task_id 改为 task_name，更有描述性
    parser.add_argument('--task_name', type=str, default='Qwen-VL-COCO-Caption-Finetune')
    args = parser.parse_args()
    set_random_seed(3407)
    main(task_name=args.task_name, train_batch=1)