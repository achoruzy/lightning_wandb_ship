# Copyright 2022 Arkadiusz Choruży


from logging import Logger
from pathlib import Path
from typing import List
from datetime import datetime
import argparse

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ml_collections import config_dict

from model.data import ShipDataModule
from model.model import LitModel


EXPERIMENT_NAME = f'ship_classification_{datetime.now()}'

PRECISION = 16
GPUS = 0
NUM_WORKERS = 8
SPLIT = (0.1, 0.1)

SAVE_PATH = Path(__file__).parent/'artifacts'
FILE_NAME = 'model'

train_cfg = config_dict.ConfigDict()
train_cfg.img_size = 32
train_cfg.bs = 4
train_cfg.epochs = 4
train_cfg.lr = 1e-3


def parse_args():
    "Overriding default training argments"
    argparser = argparse.ArgumentParser(description='Training hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=train_cfg.img_size, help='image size')
    argparser.add_argument('--bs', type=int, default=train_cfg.bs, help='batch size')
    argparser.add_argument('--epochs', type=int, default=train_cfg.epochs, help='max qty of training epochs')
    argparser.add_argument('--lr', type=float, default=train_cfg.lr, help='learning rate')
    return argparser.parse_args()


def train(datamodule: LightningDataModule,
          model: LightningModule,
          logger: Logger,
          callbacks: List[Callback]) -> Trainer:
    
    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=GPUS,
                      precision=PRECISION,
                      logger=logger,
                      callbacks=callbacks,
                      enable_model_summary=False)

    trainer.fit(model=model, datamodule=datamodule)
    return trainer
        

def main(test: bool = False,
         verbose: bool = False,
         save_torch: bool = False,
         offline_log: bool = True):
    
    datamodule = ShipDataModule(split=SPLIT, bs=train_cfg.bs, img_size=train_cfg.img_size)
    model = LitModel(lr=train_cfg.lr)

    logger = WandbLogger(project='Ships_wandb_course', name=EXPERIMENT_NAME, offline=offline_log)
    
    callbacks = [
        #EarlyStopping('valid_F1Score', min_delta=0.001, patience=2),
    ]
    if save_torch:
        checkpoint_clbk = ModelCheckpoint(dirpath=SAVE_PATH, filename=FILE_NAME,
                                          monitor='valid_F1Score', save_top_k=1, mode='max')
        callbacks.append(checkpoint_clbk)
        best_model = checkpoint_clbk.best_model_path
        print(f'Training session saved to {best_model}')
    
    trainer = train(datamodule, model, logger, callbacks)

    if test:
        trainer.test(ckpt_path="best", datamodule=datamodule)
    
    if verbose:
        print(trainer.logged_metrics)
    
    
if __name__ == '__main__':
    train_cfg.update(vars(parse_args()))
    main(test=True, verbose=True, save_torch=True, offline_log=True)