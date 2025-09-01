import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.augmentation import run_augmentation_single
import warnings

warnings.filterwarnings('ignore')


class CustomDataset(Dataset):
    """
    Unified dataset class to replace all old dataset providers.
    Supports CSV or NumPy data, scaling, time features, train/val/test splits, and augmentation.
    """

    def __init__(self, args, root_path, data_file=None, flag='train', size=None,
                 features='S', target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        self.root_path = root_path
        self.data_file = data_file
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # sequence lengths
        if size is None:
            self.seq_len = 96
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # load CSV or NumPy
        file_path = os.path.join(self.root_path, self.data_file)
        if file_path.endswith('.csv'):
            df_raw = pd.read_csv(file_path)
        elif file_path.endswith('.npy'):
            data_array = np.load(file_path)
            df_raw = pd.DataFrame(data_array)
            df_raw.columns = [f'feat{i}' for i in range(df_raw.shape[1])]
        else:
            raise ValueError("Unsupported file type. Only .csv or .npy allowed.")

        # ensure target is last column
        if self.target in df_raw.columns:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            if 'date' in cols:
                cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]] if 'date' in df_raw.columns else df_raw[cols + [self.target]]

        # split data
        n = len(df_raw)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_val = n - num_train - num_test

        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, n]

        border1 = border1s[['train', 'val', 'test'].index(self.flag)]
        border2 = border2s[['train', 'val', 'test'].index(self.flag)]

        # select features
        if self.features in ['M', 'MS']:
            df_data = df_raw.drop(columns=['date']) if 'date' in df_raw.columns else df_raw
        else:  # 'S'
            df_data = df_raw[[self.target]]

        # scaling
        if self.scale:
            train_data = df_data.iloc[0:border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # time features
        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']].iloc[border1:border2].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date'])
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.dt.month
                df_stamp['day'] = df_stamp.date.dt.day
                df_stamp['weekday'] = df_stamp.date.dt.weekday
                df_stamp['hour'] = df_stamp.date.dt.hour
                df_stamp['minute'] = df_stamp.date.dt.minute
                data_stamp = df_stamp.drop(columns=['date']).values
            else:
                data_stamp = time_features(df_stamp['date'].values, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # augmentation
        if self.flag == 'train' and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] if self.data_stamp is not None else None
        seq_y_mark = self.data_stamp[r_begin:r_end] if self.data_stamp is not None else None

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
