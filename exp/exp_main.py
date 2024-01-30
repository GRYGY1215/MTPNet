from data.data_provider import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual, process_one_batch
from utils.metrics import metric
from models import MTPNet, MTPNet_Linear

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args.device = self.device
        if args.load_pretrained_model:
            self._load_pretrain_model()

    def _build_model(self):
        model_dict = {
            'MTPNet': MTPNet,
            'MTPNet_Linear': MTPNet_Linear
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.99))
        return optimizer

    def _select_LR_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.args.train_epochs)
        return scheduler

    def _select_criterion(self):
        criterion = nn.L1Loss(reduction='mean') if self.args.loss == 'L1' else nn.MSELoss(reduction='mean')
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        scheduler = self._select_LR_scheduler(optimizer)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            self.model.train()

            train_loss = []
            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()

                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, self.args)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()


            train_loss = np.average(train_loss)

            # Run validation and test
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch)
            test_loss = self.vali(test_data, test_loader, criterion, epoch)

            scheduler.step(epoch)
            self.model.epoch += 1

            print("Epoch: {:3d}, Steps: {:3d}, cost time: {:5.2f} | Train Loss: {:5.4f} Vali Loss: {:5.4f} Test Loss: {:5.4f}".format(
                epoch + 1, train_steps, (time.time() - epoch_time), train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion, epoch=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, self.args)

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.detach().item())

        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, self.args)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                outputs, batch_y = process_one_batch(self.model, batch_x, batch_y, self.args)
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        return


