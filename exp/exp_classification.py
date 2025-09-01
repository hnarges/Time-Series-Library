from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        self.args.seq_len = self.args.seq_len
        self.args.pred_len = self.args.pred_len
        self.args.enc_in = train_data.data_x.shape[1]

        # ✅ Use class_names to determine number of classes
        self.args.num_class = len(train_data.class_names)

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        dataset, dataloader = data_provider(self.args, flag)
        return dataset, dataloader

    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, preds, trues = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        self.model.train()
        return total_loss, accuracy
