import os
import time
import warnings
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.device = torch.device('cuda' if args.use_gpu else 'cpu')

    def _build_model(self):
        # Load sample data to get input size
        train_data, _ = self._get_data(flag='train')
        self.args.enc_in = train_data.data_x.shape[1]
        self.args.num_class = len(train_data.class_names)
        
        # Initialize model
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _get_data(self, flag):
        dataset, loader = data_provider(self.args, flag)
        return dataset, loader

    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def vali(self, data_set, data_loader, criterion):
        self.model.eval()
        total_loss = []
        all_preds, all_trues = [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss.item())

                all_preds.append(outputs)
                all_trues.append(label)

        preds = torch.cat(all_preds, 0)
        trues = torch.cat(all_trues, 0)
        probs = torch.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        return np.mean(total_loss), accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        # Checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            start_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                model_optim.zero_grad()
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                model_optim.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss, val_acc = self.vali(val_data, val_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print(f"Epoch {epoch+1}/{self.args.train_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Time: {time.time() - start_time:.1f}s")

            early_stopping(-val_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data('test')
        best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                all_preds.append(outputs)
                all_trues.append(label)

        preds = torch.cat(all_preds, 0)
        trues = torch.cat(all_trues, 0)
        probs = torch.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy
