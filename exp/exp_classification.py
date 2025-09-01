import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # Get dataset and loader
        train_dataset, train_loader = self._get_data(flag='train')
        test_dataset, test_loader = self._get_data(flag='test')

        # Input/output settings
        self.args.seq_len = self.args.seq_len
        self.args.pred_len = self.args.pred_len
        self.args.enc_in = train_dataset.data_x.shape[1]

        # ✅ Fix class issue
        if hasattr(train_dataset, 'class_names'):
            self.args.num_class = len(train_dataset.class_names)
        else:
            self.args.num_class = len(set(train_dataset.data_y[:, -1]))

        # Initialize model
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        dataset, loader = data_provider(self.args, flag)
        return dataset, loader

    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def vali(self, vali_dataset, vali_loader, criterion):
        self.model.eval()
        total_loss, preds, trues = [], [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x, padding_mask, label = batch_x.float().to(self.device), \
                                              padding_mask.float().to(self.device), \
                                              label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss.item())
                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.mean(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_dataset, train_loader = self._get_data(flag='train')
        vali_dataset, vali_loader = self._get_data(flag='val')
        test_dataset, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            epoch_loss, iter_count = [], 0
            self.model.train()
            start_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, padding_mask, label = batch_x.float().to(self.device), \
                                              padding_mask.float().to(self.device), \
                                              label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                epoch_loss.append(loss.item())

            train_loss = np.mean(epoch_loss)
            vali_loss, vali_acc = self.vali(vali_dataset, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_dataset, test_loader, criterion)

            print(f"Epoch {epoch+1}/{self.args.train_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Vali Loss: {vali_loss:.4f}, Vali Acc: {vali_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            early_stopping(-vali_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test_flag=0):
        test_dataset, test_loader = self._get_data(flag='test')
        if test_flag:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth')))

        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x, padding_mask, label = batch_x.float().to(self.device), \
                                              padding_mask.float().to(self.device), \
                                              label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        print(f'Test accuracy: {accuracy:.4f}')
        return accuracy

