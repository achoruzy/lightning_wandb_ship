# Copyright 2022 Arkadiusz Choruży


from typing import Any, Union
from pathlib import Path

from numpy import array
from torch import no_grad, Tensor, argmax, float32
from torch.nn import CrossEntropyLoss, Sigmoid
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, Accuracy, F1Score, Recall, Precision
from model.simple_arch import SimpleArkNet


class LitModel(LightningModule):
    def __init__(self, lr, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.num_classes = 5
        self.architecture = SimpleArkNet(5)
        self.loss_func = CrossEntropyLoss()
        self.optimizer = Adam
        self.post_transforms = Sigmoid()
        
        metrics = MetricCollection(F1Score(num_classes=self.num_classes, multiclass=True)) # should receive class, probability or logits
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.test_metrics.add_metrics([Accuracy(num_classes=self.num_classes, average='macro'), 
            Precision(num_classes=self.num_classes, average='macro'), Recall(num_classes=self.num_classes, average='macro')])
        
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
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        logits = self.post_transforms(y_pred)
        loss = self.loss_func(y_pred, y)
        
        metrics = self.valid_metrics(logits, y)
        self.log_dict(metrics)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        logits = self.post_transforms(y_pred)
        print(y[0], '\n', logits[0])
        loss = self.loss_func(y_pred, y)
        
        metrics = self.test_metrics(logits, y)
        self.log_dict(metrics)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
    
        return loss