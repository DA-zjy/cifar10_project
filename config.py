# config.py

import torch

# --- 训练配置 ---
DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 64 #Batch如何选择？
LEARNING_RATE = 5e-4 # 0.001

# --- 数据集配置 ---
DATASET_PATH = "./data"

# --- 模型保存配置 ---
CHECKPOINT_PATH = "./checkpoints"