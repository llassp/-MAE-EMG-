import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import logging
import math
from tqdm import tqdm
from dataset import EMGPretrainDataset, EMGPretrainDatasetFromArray, create_train_val_split


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_start_lr=1e-6, base_lr=1e-4, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pretrain.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        inputs = batch.to(device)
        optimizer.zero_grad()

        pred, target, mask_indices = model(inputs)

        loss = model.compute_loss(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    return total_loss / num_batches


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            inputs = batch.to(device)
            pred, target, mask_indices = model(inputs)
            loss = model.compute_loss(pred, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, checkpoint_path)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='EMG MAE Pretraining')
    parser.add_argument('--data_dir', type=str, default='data/new_data/', help='Path to raw EMG data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for logs')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--encoder_type', type=str, default='mae', choices=['mae', 'mae_ila'],
                        help='Encoder type: mae (default) or mae_ila (MAE with Inter-Layer Attention)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir)

    logging.info('Loading and processing data...')
    train_data, val_data = create_train_val_split(args.data_dir, val_ratio=0.2, seed=args.seed)
    logging.info(f'Train samples: {len(train_data)}, Val samples: {len(val_data)}')

    train_dataset = EMGPretrainDatasetFromArray(train_data)
    val_dataset = EMGPretrainDatasetFromArray(val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    config = {
        'img_size': 64,
        'patch_size': 8,
        'in_channels': 3,
        'encoder_dim': 128,
        'encoder_layers': 6,
        'encoder_heads': 4,
        'encoder_ffn_dim': 512,
        'decoder_dim': 64,
        'decoder_layers': 2,
        'decoder_heads': 2,
        'mask_ratio': 0.75,
        'dropout': 0.1
    }

    from model import MAEModel, MAEModelWithInterLayerAttention
    if args.encoder_type == 'mae_ila':
        logging.info('Using MAE with Inter-Layer Attention')
        model = MAEModelWithInterLayerAttention(**config).to(device)
    else:
        model = MAEModel(**config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total parameters: {total_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        logging.info(f'Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}')

    logging.info(f'Starting training for {args.epochs} epochs...')

    for epoch in range(start_epoch, args.epochs):
        logging.info(f'[Epoch {epoch+1}/{args.epochs}]')
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]

        logging.info(f'[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, args.checkpoint_dir, 'best_model.pth')
            logging.info(f'New best model saved with val_loss: {best_val_loss:.4f}')

        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, config, args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            logging.info(f'Checkpoint saved at epoch {epoch+1}')

    save_checkpoint(model, optimizer, args.epochs - 1, val_loss, config, args.checkpoint_dir, 'final_model.pth')
    logging.info(f'Training completed! Best val_loss: {best_val_loss:.4f}')
    logging.info(f'Final model saved to {args.checkpoint_dir}/final_model.pth')


if __name__ == '__main__':
    main()
