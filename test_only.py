import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import numpy as np

from model import MAEModel, DownstreamLSTM
from dataset import EMGDownstreamDataset
from visualize import confusion_matrix_visualization, training_curve_visualization


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading pretrained model...')
    pretrain_model = MAEModel()
    pretrain_model = pretrain_model.to(device)

    model = DownstreamLSTM(pretrain_model, num_classes=16, freeze_encoder=True).to(device)

    checkpoint_path = 'checkpoints/downstream/best_model.pth'
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]} with Val Acc: {checkpoint["val_acc"]:.1f}%')

    print('Building dataset...')
    _, val_dataset, test_dataset = EMGDownstreamDataset.build_splits(
        '../new_data/', seed=42
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    print('Running inference on test set...')
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = 100 * np.mean(all_preds == all_labels)
    print(f'Test Accuracy: {test_acc:.1f}%')

    print('\nPer-class Accuracy:')
    class_names = EMGDownstreamDataset.CLASS_NAMES
    for i in range(len(class_names)):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = 100 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            print(f'  {i:2d}. {class_names[i]:20s}: {acc:.1f}%')

    confusion_matrix_visualization(
        all_labels, all_preds,
        class_names,
        'logs/downstream/confusion_matrix.png'
    )

    history_path = 'logs/downstream/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        training_curve_visualization(history, 'logs/downstream/training_curve.png')
    else:
        print('training_history.json not found, skipping training curve visualization')

    print('\nVisualization complete!')


if __name__ == '__main__':
    main()
