import re
import numpy as np
import torch
import wandb
from torch.utils.data import dataloader
from typing import List, Union
from timeit import default_timer as timer
from pathlib import Path
from utils.visual import *


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
        self.idx = 1
        while Path(self.save_dir).is_dir() == True:
            self.save_dir = re.sub('[0-9]+',f'{self.idx}', self.save_dir)
            self.idx += 1
        
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)


        self.es_log = {'train_loss' : [], 'val_loss' : []}

        self.not_improved = 0
        self.early_stop = 30
        self.save_period = 5
        self.mnt_best = np.inf

    def _train_epoch(self, epoch: int):
        train_loss = 0
        train_metric = {f'{metric.__name__}': [] for metric in self.metric_fn}

        self.model.train()
        for batch, data in enumerate(self.data_loader):
            x_train = data['input']
            y_train = data['label']
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
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
        print(f'Train Loss : {train_loss:.5f} | Train P.A : {train_pa:.2f}% | Train IOU : {train_iou:.5f} | ', end='')
        self.es_log['train_loss'].append(train_loss)

        if self.do_validation:
            val_loss, val_pa, val_iou, examples = self._valid_epoch(epoch)
            wandb.log({'Train Loss': train_loss, 'Train P.A': train_pa, 'Train IOU': train_iou,
                                'Val Loss': val_loss, 'Val P.A': val_pa, 'Val IOU': val_iou, 'examples' : examples})
        else:
            wandb.log(
                {'Train Loss': train_loss, 'Train P.A': train_pa, 'Train IOU': train_iou, 'examples' : examples})

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
        examples = []
        self.model.eval()
        with torch.inference_mode():
            for batch, data in enumerate(self.valid_data_loader):
                x_test = data['input']
                y_test = data['label']
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                y_pred = self.model(x_test)
                loss = self.criterion(y_pred, y_test)

                val_loss += loss.item()

                met_ = {f'{metric.__name__}': metric(self.model, y_pred, y_test, self.device) for metric in self.metric_fn}
                for key, value in met_.items():
                    val_metric[key].append(value)

                if batch < 4:
                    pred_mask, target_mask = make_mask((torch.sigmoid(y_pred) > 0.5).float(), y_test)
                    wandb_img = ((x_test.permute(0,2,3,1).cpu().numpy() * 0.5 + 0.5) * 255).astype('int')
                    
                    pred_mask = pred_mask.astype('int')
                    target_mask = target_mask.astype('int')
                    

                    for i in range(wandb_img.shape[0] % 4):
                        

                        class_labels = {
                            0: "BackGround",
                            1: "Damage",
                            2: "Ground Truth"
                        }

                        class_set = wandb.Classes([
                            {"name" : "BackGround", "id" : 0},
                            {"name" : "Damage", "id" : 1},
                            {"name" : "Ground Truth", "id" : 2}
                        ])
                        
                        print(wandb_img.shape, pred_mask.shape, target_mask.shape)

                        example = wandb.Image(wandb_img[i], masks={"Mask" : {"mask_data" : pred_mask[i], "class_labels" : class_labels}, "ground_truth" : {"mask_data" : target_mask[i], "class_labels" : class_labels}}, classes=class_set)
                        examples.append(example)
            
            val_loss /= len(self.valid_data_loader)
            iou, p_a = list(map(lambda x: sum(x) / len(self.valid_data_loader), val_metric.values()))
            print(f'Val Loss : {val_loss:.5f} | Val P.A : {p_a:.2f}% | Val IOU : {iou:.5f} | ', end='')
            self.es_log['val_loss'].append(val_loss)

            return val_loss, p_a, iou, examples

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
            
        filename = str(self.save_dir / 'Epoch_{}.pth'.format(epoch))
        torch.save(self.model.state_dict(), filename)
        if save_best:
            best_path = str(self.save_dir / 'Model_best.pth')
            torch.save(self.model.state_dict(), best_path)