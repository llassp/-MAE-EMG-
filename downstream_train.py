import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import logging
import json
import math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from model import MAEModel, DownstreamLSTM
from dataset import EMGDownstreamDataset
from visualize import confusion_matrix_visualization, training_curve_visualization


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lrs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = base_lrs
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.min_lr + (self.base_lrs[i] - self.min_lr) * self.current_step / self.warmup_steps
                param_group['lr'] = lr
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.min_lr + (self.base_lrs[i] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'downstream.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        frames, labels = batch
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })

    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total


def test(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100 * np.mean(all_preds == all_labels)
    per_class_acc = []

    for i in range(len(class_names)):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = 100 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)

    return accuracy, per_class_acc, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, checkpoint_path)
    return checkpoint_path


import numpy as np


def main():
    parser = argparse.ArgumentParser(description='EMG Downstream Classification')
    parser.add_argument('--data_dir', type=str, default='../new_data/', help='Path to raw EMG data')
    parser.add_argument('--pretrain_ckpt', type=str, default='checkpoints/best_model.pth', help='Pretrained model path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/downstream/', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/downstream/', help='Directory for logs')
    parser.add_argument('--phase1_epochs', type=int, default=50, help='Phase 1 epochs')
    parser.add_argument('--phase2_epochs', type=int, default=150, help='Phase 2 epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--phase1_lr', type=float, default=1e-4, help='Phase 1 learning rate')
    parser.add_argument('--phase2_lr', type=float, default=5e-5, help='Phase 2 head learning rate')
    parser.add_argument('--phase2_encoder_lr', type=float, default=1e-5, help='Phase 2 encoder learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir)

    logging.info('Loading pretrained model...')
    pretrain_model = MAEModel()
    if os.path.exists(args.pretrain_ckpt):
        ckpt = torch.load(args.pretrain_ckpt, weights_only=False)
        pretrain_model.load_state_dict(ckpt['model_state_dict'])
        logging.info(f'Pretrained weights loaded: Epoch {ckpt["epoch"]}, Val Loss: {ckpt["val_loss"]:.4f}')
    else:
        logging.warning(f'Pretrained checkpoint not found at {args.pretrain_ckpt}, using random init')
    pretrain_model = pretrain_model.to(device)

    encoder_params = sum(p.numel() for p in pretrain_model.parameters())

    logging.info('Building downstream dataset...')
    train_dataset, val_dataset, test_dataset = EMGDownstreamDataset.build_splits(
        args.data_dir, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = DownstreamLSTM(pretrain_model, num_classes=16, freeze_encoder=True).to(device)

    head_params = sum(p.numel() for p in model.classifier.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    pos_params = model.temporal_pos_embed.numel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f'Encoder params: {encoder_params:,}')
    logging.info(f'LSTM params: {lstm_params:,}')
    logging.info(f'Pos embed params: {pos_params:,}')
    logging.info(f'Classifier params: {head_params:,}')
    logging.info(f'Total params: {total_params:,}')
    logging.info(f'Phase 1 trainable params: {trainable_params:,}')

    training_history = {
        'phase1': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'phase2': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    }

    logging.info('=' * 60)
    logging.info('PHASE 1: Frozen Encoder, Training Classification Head')
    logging.info('=' * 60)

    model.set_encoder_grad(freeze=True)
    optimizer_p1 = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.phase1_lr,
        weight_decay=0.01
    )

    phase1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=args.phase1_epochs, eta_min=1e-6
    )

    best_val_acc_p1 = 0
    patience_counter_p1 = 0
    patience = 15

    for epoch in range(args.phase1_epochs):
        logging.info(f'[P1 Epoch {epoch+1}/{args.phase1_epochs}]')

        train_loss, train_acc = train_epoch(model, train_loader, optimizer_p1, device)
        val_loss, val_acc = validate(model, val_loader, device)
        phase1_scheduler.step()

        training_history['phase1']['train_loss'].append(train_loss)
        training_history['phase1']['train_acc'].append(train_acc)
        training_history['phase1']['val_loss'].append(val_loss)
        training_history['phase1']['val_acc'].append(val_acc)

        logging.info(f'[P1 Epoch {epoch+1}/{args.phase1_epochs}] '
                    f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | '
                    f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%')

        if val_acc > best_val_acc_p1:
            best_val_acc_p1 = val_acc
            save_checkpoint(model, optimizer_p1, epoch, val_acc, args.checkpoint_dir, 'phase1_best.pth')
            logging.info(f'Phase 1 best model saved with Val Acc: {val_acc:.1f}%')
            patience_counter_p1 = 0
        else:
            patience_counter_p1 += 1

        if patience_counter_p1 >= patience:
            logging.info(f'Early stopping triggered at epoch {epoch+1}')
            break

    logging.info(f'Phase 1 completed. Best Val Acc: {best_val_acc_p1:.1f}%')

    if best_val_acc_p1 < 10:
        logging.warning('预训练特征判别性不足，建议检查预训练质量')
    else:
        logging.info('Phase 1验证通过，开始Phase 2端到端微调')

    logging.info('=' * 60)
    logging.info('PHASE 2: Unfrozen Encoder, End-to-end Fine-tuning')
    logging.info('=' * 60)

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'phase1_best.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.set_encoder_grad(freeze=False)

    param_groups = model.get_param_groups(
        base_lr_encoder=args.phase2_encoder_lr,
        base_lr_head=args.phase2_lr
    )

    optimizer_p2 = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps_p2 = len(train_loader) * args.phase2_epochs
    warmup_steps_p2 = int(total_steps_p2 * args.warmup_ratio)

    phase2_scheduler = WarmupCosineScheduler(
        optimizer_p2,
        warmup_steps_p2,
        total_steps_p2,
        base_lrs=[args.phase2_encoder_lr, args.phase2_lr],
        min_lr=1e-7
    )

    best_val_acc_p2 = 0
    patience_counter_p2 = 0
    patience = 30

    for epoch in range(args.phase2_epochs):
        logging.info(f'[P2 Epoch {epoch+1}/{args.phase2_epochs}]')

        train_loss, train_acc = train_epoch(model, train_loader, optimizer_p2, device)
        val_loss, val_acc = validate(model, val_loader, device)
        phase2_scheduler.step()

        training_history['phase2']['train_loss'].append(train_loss)
        training_history['phase2']['train_acc'].append(train_acc)
        training_history['phase2']['val_loss'].append(val_loss)
        training_history['phase2']['val_acc'].append(val_acc)

        lrs = phase2_scheduler.get_last_lr()
        logging.info(f'[P2 Epoch {epoch+1}/{args.phase2_epochs}] '
                    f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | '
                    f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | '
                    f'LR(enc): {lrs[0]:.2e} | LR(head): {lrs[1]:.2e}')

        if val_acc > best_val_acc_p2:
            best_val_acc_p2 = val_acc
            save_checkpoint(model, optimizer_p2, epoch, val_acc, args.checkpoint_dir, 'best_model.pth')
            logging.info(f'Phase 2 best model saved with Val Acc: {val_acc:.1f}%')
            patience_counter_p2 = 0
        else:
            patience_counter_p2 += 1

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer_p2, epoch, val_acc, args.checkpoint_dir, f'phase2_epoch_{epoch+1}.pth')

        if patience_counter_p2 >= patience:
            logging.info(f'Early stopping triggered at epoch {epoch+1}')
            break

    logging.info(f'Phase 2 completed. Best Val Acc: {best_val_acc_p2:.1f}%')

    logging.info('=' * 60)
    logging.info('Testing on Test Set')
    logging.info('=' * 60)

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, per_class_acc, all_preds, all_labels = test(
        model, test_loader, device, EMGDownstreamDataset.CLASS_NAMES
    )

    logging.info(f'Test Accuracy: {test_acc:.1f}%')
    logging.info('Per-class Accuracy:')
    for i, name in enumerate(EMGDownstreamDataset.CLASS_NAMES):
        logging.info(f'  {i:2d}. {name:20s}: {per_class_acc[i]:.1f}%')

    confusion_matrix_visualization(
        all_labels, all_preds,
        EMGDownstreamDataset.CLASS_NAMES,
        os.path.join(args.log_dir, 'confusion_matrix.png')
    )

    training_curve_visualization(
        training_history,
        os.path.join(args.log_dir, 'training_curve.png')
    )

    history_path = os.path.join(args.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    logging.info('Downstream training completed!')
    logging.info(f'Best Phase 1 Val Acc: {best_val_acc_p1:.1f}%')
    logging.info(f'Best Phase 2 Val Acc: {best_val_acc_p2:.1f}%')
    logging.info(f'Test Accuracy: {test_acc:.1f}%')
    logging.info(f'Models saved to: {args.checkpoint_dir}')
    logging.info(f'Visualizations saved to: {args.log_dir}')


if __name__ == '__main__':
    main()
