import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import logging

# 导入自定义模块
from src.data.dataset import SpaSemDataset
from src.data.collator import SpaSemCollator
from src.modeling.SpaSem import SpaSem
from src.loss import MaskedMLMLoss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpatialGlue Dual-Stream Struct-MIF")

    # ================= 数据与路径 =================
    parser.add_argument("--data_dir", type=str, default="./processed_pt", help="预处理好的 .pt 图数据目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存路径")

    # ================= 模型架构参数 =================
    parser.add_argument("--esm_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--gvp_node_out_dim", type=int, default=128, help="统一的隐层维度 (影响 GVP, GAT 和 Fusion)")
    parser.add_argument("--gvp_layers", type=int, default=6, help="GVP 几何引擎的层数")
    parser.add_argument("--dropout", type=float, default=0.1)

    # [核心新增] 双流控制开关
    # 默认开启双流，如果命令行加上 --disable_dual_stream 则关闭双流，退回单流 Baseline
    parser.add_argument("--disable_dual_stream", action="store_true", help="禁用双流注意力机制，仅使用 GVP 单流")

    # ================= 训练超参数 =================
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mask_prob", type=float, default=0.15)

    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 并发读取线程数")
    parser.add_argument("--accum_steps", type=int, default=2, help="梯度累加的步数")

    return parser.parse_args()


def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. 初始化 Dataset 和 Collator
    logger.info(f"Loading dataset from {args.data_dir}...")
    train_dataset = SpaSemDataset(
        root_dir=args.data_dir,
        tokenizer_name=args.esm_model_name,
        max_len=1024,
        mask_prob=args.mask_prob
    )

    collator = SpaSemCollator(
        tokenizer=train_dataset.tokenizer,
        max_length=1024
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 4. 初始化模型 (最新版 SpatialGlue 双流架构)
    use_dual = not args.disable_dual_stream
    logger.info(f"Initializing SpaSem model... (Dual-Stream enabled: {use_dual})")

    model = SpaSem(
        esm_model_name=args.esm_model_name,
        gvp_node_out_dim=args.gvp_node_out_dim,
        gvp_layers=args.gvp_layers,
        dropout=args.dropout,
        use_dual_stream=use_dual  # 接收命令行指令
    ).to(device)

    # 冻结 ESM 塔的参数 (极其重要，否则显存爆炸且破坏预训练语义)
    for param in model.esm_encoder.parameters():
        param.requires_grad = False

    # 5. 设置优化器和损失函数
    # 只优化需要梯度的参数（GVP, GAT, Attention, Head 等）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # 使用我们自定义的忽略 -100 的 Loss
    criterion = MaskedMLMLoss(ignore_index=-100)

    # 6. 开始训练
    logger.info("Starting training...")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        # 每一个 epoch 开始前清空梯度
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        # 使用 enumerate 拿到当前的 step 索引
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)

            # 前向传播
            logits = model(batch)

            # 计算 Loss，并根据 accum_steps 缩小，保证数值尺度正确
            loss = criterion(logits, batch.y)
            loss = loss / args.accum_steps

            # 反向传播 (累计梯度，但不立即更新权重)
            loss.backward()

            # 当达到累加步数，或者已经是当前 epoch 的最后一个 batch 时，更新权重
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()  # 更新完后立刻清空梯度

            # 记录真实的 loss.item() (因为上面除以了 accum_steps，这里要乘回来显示)
            current_loss = loss.item() * args.accum_steps
            total_loss += current_loss

            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # 保存 Checkpoint
        save_path = os.path.join(args.save_dir, f"SpaSem_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()