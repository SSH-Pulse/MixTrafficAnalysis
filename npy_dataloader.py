import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import math

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_data(k, batch_size=256, sample_frequency=50, split_ratio=0.8):
    """Loads the dataset, processes it, and returns the train/test DataLoader."""
    pcap_file = f'D:/wy/traffic/raw_data/ss_tls/open/filter{k}/data_{k}.npy'
    label_file = f'D:/wy/traffic/raw_data/ss_tls/open/filter{k}/labels_{k}.npy'

    # Load the data
    pcap_data = np.load(pcap_file)
    label_data = np.load(label_file)

    # Shuffle and select random indices
    random_indices = np.random.choice(label_data.shape[0], 25600, replace=False)
    pcap_data, label_data = pcap_data[random_indices], label_data[random_indices, 0:1]

    pack_len, pack_time = pcap_data[:, :, 0:1].squeeze(2), pcap_data[:, :, 1:2].squeeze(2)
    sign_list, p_len_value_list = spilt_data_sign(pack_len)
    _, p_time_value_list = spilt_data_sign(pack_time)

    # Generate gravity features
    gravity_feas = gravity_features(label_data.shape[0], k, p_len_value_list, p_time_value_list, sign_list,
                                    sample_frequency)

    # Convert to tensors
    gra_data = torch.from_numpy(gravity_feas)
    len_data = torch.from_numpy(p_len_value_list)
    label_data = label_data.squeeze().astype(float)
    label_data = torch.from_numpy(label_data.squeeze()).long()

    # Split data
    split_index = int(len(label_data) * split_ratio)
    train_len_data, train_time_data = len_data[:split_index], gra_data[:split_index]
    train_label = label_data[:split_index]
    test_len_data, test_time_data = len_data[split_index:], gra_data[split_index:]
    test_label = label_data[split_index:]

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(train_len_data, train_time_data, train_label)
    test_dataset = torch.utils.data.TensorDataset(test_len_data, test_time_data, test_label)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

    return train_dataloader, test_dataloader


def gravity_features(max_samples, k, m, v_0, direction, sample_frequency):
    """Generates gravity features from raw data using the given parameters."""
    gravity_feas = np.empty((max_samples, k, sample_frequency, 4))
    t_values = np.linspace(1e-5, 1, sample_frequency, dtype=float)
    for i, t in enumerate(t_values):
        gravity_f = gravity_feature(m, v_0, direction, t)
        gravity_feas[:, :, i, :] = gravity_f
    return gravity_feas


def spilt_data_sign(p_len):
    """Splits the packet length data into sign and absolute value."""
    sign_array = np.where(p_len >= 0, 1, -1)
    abs_array = np.abs(p_len)
    return sign_array, abs_array


def gravity_feature(m, v_0, direction, t):
    """Calculates the gravity feature for a single time value."""
    sp_speed = 1 / (v_0 + 1e-3)
    sp_speed_norm = direction * 1e-3 * sp_speed
    t = np.full(sp_speed.shape, t)
    cz_speed_norm = t
    sp_s = sp_shift(sp_speed, t)
    sp_s_norm = direction * sp_s
    cz_s = cz_shift(t)
    k_e = k_energy(m, sp_speed, t)
    k_e_norm = norm(k_e)
    p_e = p_energy(m, cz_s)
    p_e_norm = p_e / 1515
    return np.stack((sp_speed_norm, cz_speed_norm, k_e_norm, p_e_norm), axis=-1)


def sp_shift(v_0, t):
    """Calculates the displacement in the horizontal direction."""
    return v_0 * t


def cz_shift(t):
    """Calculates the displacement in the vertical direction due to gravity."""
    return t ** 2


def k_energy(m, v_0, t):
    """Calculates kinetic energy."""
    return 0.5 * m * (v_0 ** 2 + (9.8 * t) ** 2)


def p_energy(m, cz_s):
    """Calculates potential energy."""
    return m * cz_s


def norm(gravity_t):
    """Normalizes the gravity values."""
    max_value, min_value = np.max(gravity_t), np.min(gravity_t)
    if max_value - min_value == 0:
        print('The maximum and minimum values are the same, normalization will result in invalid values.')
    return (gravity_t - min_value) / (max_value - min_value)
