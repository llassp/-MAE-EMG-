import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import glob
import pandas as pd


def reconstruct_visualization(model, dataset, n_samples=8, save_path='logs/reconstruction.png', device='cpu'):
    model.eval()

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    original_images = []
    masked_images_list = []
    reconstructed_images = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img = dataset[idx].unsqueeze(0).to(device)

            patch_size = model.patch_size
            num_patches = model.num_patches
            img_size = model.img_size

            patch_h = img_size // patch_size
            patch_w = patch_size

            patches = img.reshape(1, 3, patch_h, patch_size, patch_h, patch_size)
            patches = patches.permute(0, 2, 4, 3, 5, 1)
            patches = patches.reshape(1, num_patches, patch_size * patch_size * 3)

            B = 1
            rand_indices = torch.argsort(torch.rand(B, num_patches), dim=-1)
            num_mask = model.num_mask
            mask_indices = rand_indices[:, :num_mask]
            unmask_indices = rand_indices[:, num_mask:]

            masked_patch = patches.clone()
            gray_value = 0.5
            for j in range(num_mask):
                patch_idx = mask_indices[0, j].item()
                masked_patch[0, patch_idx] = torch.tensor([gray_value] * (patch_size * patch_size * 3))

            masked_img = masked_patch.reshape(1, num_patches, 3, patch_size, patch_size)
            masked_img = masked_img.reshape(1, 3, patch_h, patch_size, patch_h, patch_size)
            masked_img = masked_img.permute(0, 2, 4, 3, 5, 1)
            masked_img = masked_img.reshape(1, 3, img_size, img_size)

            original_np = img[0].cpu().permute(1, 2, 0).numpy()
            original_np = np.clip(original_np, 0, 1)

            masked_np = masked_img[0].cpu().permute(1, 2, 0).numpy()
            masked_np = np.clip(masked_np, 0, 1)

            pred, target, _ = model(img)

            pred_patches = torch.zeros(1, num_patches, patch_size * patch_size * 3)
            for j in range(num_mask):
                patch_idx = mask_indices[0, j].item()
                pred_patches[0, patch_idx] = pred[0, j]

            recon_img = model.reconstruct_from_patches(pred_patches, B)
            recon_np = recon_img[0].cpu().permute(1, 2, 0).numpy()
            recon_np = np.clip(recon_np, 0, 1)

            original_images.append(original_np)
            masked_images_list.append(masked_np)
            reconstructed_images.append(recon_np)

    for i in range(n_samples):
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masked_images_list[i])
        axes[i, 1].set_title('Masked (75%)' if i == 0 else '')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(reconstructed_images[i])
        axes[i, 2].set_title('Reconstructed' if i == 0 else '')
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Reconstruction visualization saved to {save_path}')


def tsne_visualization(model, data_dir, n_per_class=100, save_path='logs/tsne.png', device='cpu'):
    model.eval()

    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))

    all_features = []
    all_labels = []

    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        if 'labels' not in df.columns and 'movement_type' not in df.columns:
            continue

        label_col = 'labels' if 'labels' in df.columns else 'movement_type'
        emg_columns = [col for col in df.columns if col not in ['labels', 'fma', 'movement_type']]

        data = df[emg_columns].values
        labels = df[label_col].values

        from scipy.signal import butter, filtfilt
        def bandpass_filter(data, lowcut=1, highcut=100, fs=2000, order=4):
            nyquist = fs / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
            return filtered_data

        def compute_rms(data, window=64, stride=10):
            n_samples = data.shape[0]
            n_channels = data.shape[1]
            n_rms = (n_samples - window) // stride + 1
            rms_features = np.zeros((n_rms, n_channels))
            for i in range(n_rms):
                start = i * stride
                end = start + window
                window_data = data[start:end, :]
                rms_features[i, :] = np.sqrt(np.mean(window_data ** 2, axis=0))
            return rms_features

        def sliding_window(rms_features, window=64, stride=10):
            n_rms = rms_features.shape[0]
            n_windows = (n_rms - window) // stride + 1
            windows = []
            for i in range(n_windows):
                start = i * stride
                end = start + window
                window_data = rms_features[start:end, :]
                windows.append(window_data.T)
            return np.array(windows)

        import matplotlib.cm
        def normalize_and_colormap(windows):
            images = []
            for window in windows:
                window_min = window.min()
                window_max = window.max()
                if window_max - window_min > 1e-8:
                    normalized = (window - window_min) / (window_max - window_min)
                else:
                    normalized = window - window_min
                colored = matplotlib.cm.jet(normalized)
                rgb = colored[:, :, :3]
                rgb_float = rgb.astype(np.float32)
                images.append(rgb_float)
            images = np.array(images)
            images = images.transpose(0, 3, 1, 2)
            images = images / 255.0
            return images

        filtered_data = bandpass_filter(data)
        rms_features = compute_rms(filtered_data)
        windows = sliding_window(rms_features)
        images = normalize_and_colormap(windows)

        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]

            if len(label_indices) > n_per_class:
                selected_indices = np.random.choice(label_indices, n_per_class, replace=False)
            else:
                selected_indices = label_indices

            for idx in selected_indices:
                if idx < len(images):
                    img_tensor = torch.from_numpy(images[idx]).unsqueeze(0).float().to(device)

                    features = model.encode(img_tensor)
                    features = features.mean(dim=1)

                    all_features.append(features.cpu().numpy())
                    all_labels.append(label)

    if len(all_features) == 0:
        print("No labeled data found. Please ensure parquet files contain labels.")
        return

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    print(f'T-SNE on {len(all_features)} samples from {len(np.unique(all_labels))} classes')

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
    features_2d = tsne.fit_transform(all_features)

    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=f'Class {label}', alpha=0.6, s=30)

    plt.title('t-SNE Visualization of EMG Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f't-SNE visualization saved to {save_path}')

    if len(unique_labels) >= 8:
        print("特征判别性良好，可进行下游训练" if len(unique_labels) >= 8 else "建议继续预训练或调整参数")
    else:
        print("建议继续预训练或调整参数")


def confusion_matrix_visualization(y_true, y_pred, class_names, save_path='logs/downstream/confusion_matrix.png'):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 14))
    im = plt.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            plt.text(j, i, f'{cm_normalized[i, j]:.1%}', ha='center', va='center', color=text_color, fontsize=8)

    plt.colorbar(im, label='Accuracy')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    accuracy = 100 * np.mean(np.array(y_true) == np.array(y_pred))
    plt.title(f'Test Accuracy: {accuracy:.1f}%')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names, rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')

    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print('Classification Report:')
    print(report)


def training_curve_visualization(history, save_path='logs/downstream/training_curve.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    p1_epochs = range(1, len(history['phase1']['train_loss']) + 1)
    p2_start = len(history['phase1']['train_loss'])
    p2_epochs = range(p2_start + 1, p2_start + len(history['phase2']['train_loss']) + 1)

    ax1.plot(p1_epochs, history['phase1']['train_loss'], 'b-', label='Train Loss (P1)', linewidth=2)
    ax1.plot(p1_epochs, history['phase1']['val_loss'], 'b--', label='Val Loss (P1)', linewidth=2)
    ax1.plot(p2_epochs, history['phase2']['train_loss'], 'r-', label='Train Loss (P2)', linewidth=2)
    ax1.plot(p2_epochs, history['phase2']['val_loss'], 'r--', label='Val Loss (P2)', linewidth=2)
    ax1.axvline(x=p2_start, color='gray', linestyle=':', linewidth=2, label='Phase Change')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(p1_epochs, history['phase1']['train_acc'], 'b-', label='Train Acc (P1)', linewidth=2)
    ax2.plot(p1_epochs, history['phase1']['val_acc'], 'b--', label='Val Acc (P1)', linewidth=2)
    ax2.plot(p2_epochs, history['phase2']['train_acc'], 'r-', label='Train Acc (P2)', linewidth=2)
    ax2.plot(p2_epochs, history['phase2']['val_acc'], 'r--', label='Val Acc (P2)', linewidth=2)
    ax2.axvline(x=p2_start, color='gray', linestyle=':', linewidth=2, label='Phase Change')

    best_p1_idx = np.argmax(history['phase1']['val_acc'])
    best_p1_val = history['phase1']['val_acc'][best_p1_idx]
    ax2.scatter([best_p1_idx + 1], [best_p1_val], color='blue', s=100, zorder=5, marker='*')
    ax2.annotate(f'P1 Best: {best_p1_val:.1f}%', xy=(best_p1_idx + 1, best_p1_val),
                xytext=(best_p1_idx - 10, best_p1_val + 5), fontsize=9)

    best_p2_idx = np.argmax(history['phase2']['val_acc'])
    best_p2_val = history['phase2']['val_acc'][best_p2_idx]
    ax2.scatter([p2_start + 1 + best_p2_idx], [best_p2_val], color='red', s=100, zorder=5, marker='*')
    ax2.annotate(f'P2 Best: {best_p2_val:.1f}%', xy=(p2_start + 1 + best_p2_idx, best_p2_val),
                xytext=(p2_start + best_p2_idx - 10, best_p2_val + 5), fontsize=9)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curve saved to {save_path}')
