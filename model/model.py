# Copyright 2022 Arkadiusz Choruży


from typing import Any, Union
from pathlib import Path

from numpy import array
from torch import no_grad, Tensor, argmax, float32
from torch.nn import CrossEntropyLoss, Sigmoid
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, Accuracy, F1Score
from torchvision.models import resnet18, alexnet
from simple_arch import ArkNet

LR = 1e-3


class LitModel(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = LR
        self.num_classes = 5
        self.architecture = ArkNet(5)
        self.loss_func = CrossEntropyLoss()
        self.optimizer = Adam
        self.post_transforms = Sigmoid()
        
        metrics = MetricCollection([Accuracy(), F1Score()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')
        
    def forward(self, input):
        return self.architecture(input)
    
    def predict(self, input) -> Tensor:
        self.eval()
        with no_grad():
            out = self.forward(input)
            return self.post_transforms(out)
    
    def configure_optimizers(self, **kwargs):
        return self.optimizer(self.parameters(), lr=self.lr, **kwargs)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        logits = self.post_transforms(y_pred)
        loss = self.loss_func(y_pred, y)
        
        metrics = self.train_metrics(logits, y)
        self.log_dict(metrics)
        self.log('train_loss', loss)
        
        return {'loss': loss, 'y_true': y, 'y_pred': y_pred.detach()}
    
    def training_epoch_end(self, train_step_outs) -> None:
        epoch_loss = train_step_outs[-1]['loss'].item()
        epoch_metrics = self.train_metrics.compute()
        print(f'\nTRAIN ep: {self.current_epoch} | loss: {epoch_loss:.4} |',
              f'accuracy: {epoch_metrics["train_Accuracy"]:.4}\n')
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        logits = self.post_transforms(y_pred)
        loss = self.loss_func(y_pred, y)
        
        metrics = self.valid_metrics(logits, y)
        self.log_dict(metrics)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        
        return {'loss': loss, 'y_true': y, 'y_pred': y_pred}
    
    def validation_epoch_end(self, valid_step_outs) -> None:
        epoch_loss = valid_step_outs[-1]['loss']
        epoch_metrics = self.valid_metrics.compute()
        print(f'\nVALID ep: {self.current_epoch} | loss: {epoch_loss:.4} |',
              f'accuracy: {epoch_metrics["valid_Accuracy"]:.4}\n')
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        logits = self.post_transforms(y_pred)
        
        loss = self.loss_func(y_pred, y)
        
        metrics = self.test_metrics(logits, y)
        self.log_dict(metrics)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return {'loss': loss, 'y_true': y.to(float32).numpy(), 'y_pred': y_pred.to(float32).numpy()}
    
    def test_epoch_end(self, test_step_outs) -> None:
        epoch_loss = test_step_outs[-1]['loss']
        epoch_metrics = self.test_metrics.compute()
        
        self.test_y_true = array([])
        self.test_y_pred = array([])
        for batch_data in test_step_outs:
            self.test_y_true = array([*self.test_y_true, *batch_data['y_true']])
            self.test_y_pred = array([*self.test_y_pred, *batch_data['y_pred']])
        
        print(f'\nTEST | loss: {epoch_loss:.4} |',
              f'accuracy: {epoch_metrics["test_Accuracy"]:.4}\n')