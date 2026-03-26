import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.cm


class EMGPretrainDataset(Dataset):
    def __init__(self, data_dir, rms_window=64, rms_stride=10, sliding_window=64, sliding_stride=10):
        self.data_dir = data_dir
        self.rms_window = rms_window
        self.rms_stride = rms_stride
        self.sliding_window = sliding_window
        self.sliding_stride = sliding_stride

        self.emg_data = self._load_and_process()

    def _load_and_process(self):
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        all_data = []

        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            emg_columns = [col for col in df.columns if col.startswith('channel_')]
            data = df[emg_columns].values.astype(np.float64)
            all_data.append(data)

        emg_raw = np.vstack(all_data)

        emg_filtered = self._bandpass_filter(emg_raw)

        rms_features = self._compute_rms(emg_filtered)

        windows = self._sliding_window(rms_features)

        images = self._normalize_and_colormap(windows)

        return images

    def _bandpass_filter(self, data, lowcut=1, highcut=100, fs=2000, order=4):
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')

        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

        return filtered_data

    def _compute_rms(self, data):
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        n_rms = (n_samples - self.rms_window) // self.rms_stride + 1

        rms_features = np.zeros((n_rms, n_channels))

        for i in range(n_rms):
            start = i * self.rms_stride
            end = start + self.rms_window
            window = data[start:end, :]
            rms_features[i, :] = np.sqrt(np.mean(window ** 2, axis=0))

        return rms_features

    def _sliding_window(self, rms_features):
        n_rms = rms_features.shape[0]
        n_windows = (n_rms - self.sliding_window) // self.sliding_stride + 1

        windows = []
        for i in range(n_windows):
            start = i * self.sliding_stride
            end = start + self.sliding_window
            window = rms_features[start:end, :]
            windows.append(window.T)

        return np.array(windows)

    def _normalize_and_colormap(self, windows):
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

        return images

    def __getitem__(self, index):
        return torch.from_numpy(self.emg_data[index]).float()

    def __len__(self):
        return len(self.emg_data)


def create_train_val_split(data_dir, val_ratio=0.2, seed=42):
    full_dataset = EMGPretrainDataset(data_dir)

    n_samples = len(full_dataset)
    indices = list(range(n_samples))

    np.random.seed(seed)
    np.random.shuffle(indices)

    val_size = int(n_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = full_dataset.emg_data[train_indices]
    val_data = full_dataset.emg_data[val_indices]

    return train_data, val_data


class EMGPretrainDatasetFromArray(Dataset):
    def __init__(self, data_array):
        self.data = data_array

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

    def __len__(self):
        return len(self.data)


class EMGDownstreamDataset(Dataset):
    CLASS_NAMES = [
        "DiameterGrasp", "ForearmPronation", "ForearmSupination", "HookGrasp",
        "MassAdduction", "MassExtension", "MassFlexion", "PinchGrasp",
        "PinchGraspMiddle", "PinchGraspPinkie", "PinchGraspRing", "Rest",
        "SphereGrasp", "ThumbAdduction", "WristDorsiFlexion", "WristVolarFlexion"
    ]

    def __init__(self, data_array, labels, sequence_length=7):
        self.data = data_array
        self.labels = labels
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        frames = self.data[index]
        label = self.labels[index]
        return torch.from_numpy(frames).float(), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build_splits(data_dir, sequence_length=7, rms_window=64, rms_stride=10,
                     sliding_window=64, sliding_stride=10, seed=42):
        from sklearn.model_selection import train_test_split

        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
        all_data = []
        all_labels = []

        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            emg_columns = [col for col in df.columns if col.startswith('channel_')]
            emg_data = df[emg_columns].values.astype(np.float64)

            label_col = None
            for col in ['labels', 'movement_type', 'action', 'gesture']:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                raise ValueError(f"Could not find label column in {file_path}")

            labels_raw = df[label_col].values

            if isinstance(labels_raw[0], str):
                label_map = {name: i for i, name in enumerate(EMGDownstreamDataset.CLASS_NAMES)}
                labels = np.array([label_map.get(l, -1) for l in labels_raw])
                labels = labels[labels >= 0]
            else:
                labels = labels_raw.astype(int)

            filtered_data = EMGDownstreamDataset._bandpass_filter(emg_data)
            rms_features, rms_labels = EMGDownstreamDataset._compute_rms_with_labels(
                filtered_data, labels, rms_window, rms_stride
            )
            windows, window_labels = EMGDownstreamDataset._sliding_window_with_labels(
                rms_features, rms_labels, sliding_window, sliding_stride
            )

            all_data.append(windows)
            all_labels.append(window_labels)

        all_data = np.vstack(all_data)
        all_labels = np.concatenate(all_labels)

        images = EMGDownstreamDataset._normalize_and_colormap(all_data)

        sequences, seq_labels = EMGDownstreamDataset._build_sequences(
            images, all_labels, sequence_length
        )

        print(f"Total sequences: {len(sequences)}")
        print(f"Label distribution: {np.bincount(seq_labels)}")

        indices = np.arange(len(sequences))
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=seed, stratify=seq_labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=seq_labels[temp_idx]
        )

        train_data = sequences[train_idx]
        train_labels = seq_labels[train_idx]
        val_data = sequences[val_idx]
        val_labels = seq_labels[val_idx]
        test_data = sequences[test_idx]
        test_labels = seq_labels[test_idx]

        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

        train_dataset = EMGDownstreamDataset(train_data, train_labels, sequence_length)
        val_dataset = EMGDownstreamDataset(val_data, val_labels, sequence_length)
        test_dataset = EMGDownstreamDataset(test_data, test_labels, sequence_length)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def _bandpass_filter(data, lowcut=1, highcut=100, fs=2000, order=4):
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')

        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

        return filtered_data

    @staticmethod
    def _compute_rms(data, window=64, stride=10):
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

    @staticmethod
    def _compute_rms_with_labels(data, labels, window=64, stride=10):
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        n_rms = (n_samples - window) // stride + 1

        rms_features = np.zeros((n_rms, n_channels))
        rms_labels = np.zeros(n_rms, dtype=int)

        for i in range(n_rms):
            start = i * stride
            end = start + window
            window_data = data[start:end, :]
            rms_features[i, :] = np.sqrt(np.mean(window_data ** 2, axis=0))
            rms_labels[i] = labels[start]

        return rms_features, rms_labels

    @staticmethod
    def _sliding_window_with_labels(rms_features, labels, window=64, stride=10):
        n_rms = rms_features.shape[0]
        n_windows = (n_rms - window) // stride + 1

        windows = []
        window_labels = []

        for i in range(n_windows):
            start = i * stride
            end = start + window
            window_data = rms_features[start:end, :]
            window_label_values = labels[start:end]

            mode_result = np.argmax(np.bincount(window_label_values.flatten().astype(int)))
            mode_count = np.sum(window_label_values.flatten() == mode_result)
            purity = mode_count / len(window_label_values.flatten())

            if purity > 0.7:
                windows.append(window_data.T)
                window_labels.append(mode_result)

        return np.array(windows), np.array(window_labels)

    @staticmethod
    def _normalize_and_colormap(windows):
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

        return images

    @staticmethod
    def _build_sequences(images, labels, sequence_length):
        sequences = []
        seq_labels = []

        for i in range(len(images) - sequence_length + 1):
            seq_labels_values = labels[i:i + sequence_length]
            if np.all(seq_labels_values == seq_labels_values[0]):
                sequences.append(images[i:i + sequence_length])
                seq_labels.append(seq_labels_values[0])

        return np.array(sequences), np.array(seq_labels)
