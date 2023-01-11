import numpy as np
import torch
import wandb
from torch.utils.data import dataloader
from typing import List, Union
from timeit import default_timer as timer


class Trainer:
    """
    세그멘테이션 모델 학습을 위한 클래스
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, metric_fn: Union[torch.nn.Module, torch.nn.Module],
                 optimizer: torch.optim.Optimizer, device: str, len_epoch: int, save_dir,
                 data_loader: torch.utils.data.DataLoader, valid_data_loader: torch.utils.data.DataLoader = None,
                 lr_scheduler: torch.optim.lr_scheduler = None):

        # CUDA // device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # make_dataloder 함수의 결과 값
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler

        self.metric_fn = metric_fn

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = len_epoch

        self.save_dir = save_dir

        self.es_log = {'train_loss' : [], 'val_loss' : []}

        self.not_improved = 0
        self.early_stop = 10
        self.save_period = 10
        self.mnt_best = np.inf

    def _train_epoch(self, epoch: int):
        train_loss = 0
        train_metric = {f'{metric.__name__}': [] for metric in self.metric_fn}

        self.model.train()
        for batch, data in enumerate(self.data_loader):
            x_train = data['input']
            y_train = data['label']
            x_train, y_train = x_train.to(self.device), y_train.to(self.device).long()
            y_pred = self.model(x_train)
            loss = self.criterion(y_pred, y_train)

            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            met_ = {f'{metric.__name__}': metric(self.model, y_pred, y_train, self.device) for metric in self.metric_fn}
            for key, value in met_.items():
                train_metric[key].append(value)


        train_loss /= len(self.data_loader)
        train_iou, train_pa = list(map(lambda x: sum(x) / len(self.data_loader), train_metric.values()))
        print(f'Train Loss : {train_loss:.5f} | Train P.A : {train_pa:.5f}% | Train IOU : {train_iou:.5f} | ', end='')
        self.es_log['train_loss'].append(train_loss)

        if self.do_validation:
            val_loss, val_pa, val_iou = self._valid_epoch(epoch)
            wandb.log({'Train Loss': train_loss, 'Train P.A': train_pa, 'Train IOU': train_iou,
                                'Val Loss': val_loss, 'Val P.A': val_pa, 'Val IOU': val_iou})
        else:
            wandb.log(
                {'Train Loss': train_loss, 'Train P.A': train_pa, 'Train IOU': train_iou})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)

        # Early_Stopping
        best = False

        improved = (self.es_log['val_loss'][-1] <= self.mnt_best)
        if improved:
            self.mnt_best = self.es_log['val_loss'][-1]
            self.not_improved_count = 0
            best = True
        else:
            self.not_improved_count += 1

        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch, save_best=best)

    def _valid_epoch(self, epoch: int):
        val_loss = 0
        val_metric = {f'{metric.__name__}': [] for metric in self.metric_fn}

        self.model.eval()
        with torch.inference_mode():
            for data in self.valid_data_loader:
                x_test = data['input']
                y_test = data['label']
                x_test, y_test = x_test.to(self.device), y_test.to(self.device).long()
                y_pred = self.model(x_test)
                loss = self.criterion(y_pred, y_test)

                val_loss += loss.item()

                met_ = {f'{metric.__name__}': metric(self.model, y_pred, y_train, self.device) for metric in self.metric_fn}
                for key, value in met_.items():
                    val_metric[key].append(value)

            val_loss /= len(self.valid_data_loader)
            iou, p_a = list(map(lambda x: sum(x) / len(self.valid_data_loader), val_metric.values()))
            print(f'Val Loss : {val_loss:.5f} | Val P.A : {p_a:.5f}% | Val IOU : {iou:.5f} | ', end='')
            self.es_log['val_loss'].append(val_loss)

            return val_loss, p_a, iou

    def train(self):
        for epoch in range(self.epochs):
            print(f'\nEpoch : {epoch} | ', end='')
            start_time = timer()
            self._train_epoch(epoch)
            end_time = timer()
            print(f'Training Time : {(end_time-start_time):.2f}sec')

            if self.not_improved_count > self.early_stop:
                print("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                break

        wandb.finish()

    def _save_checkpoint(self, epoch, save_best=False):
        filename = str(self.save_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(self.model.state_dict(), filename)
        if save_best:
            best_path = str(self.save_dir / 'model_best.pth')
            torch.save(self.model.state_dict(), best_path)