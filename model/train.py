# Copyright 2022 Arkadiusz Choruży


from logging import Logger
from pathlib import Path
from typing import Iterable, Union, List
from numpy import argmax
from datetime import datetime

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import ShipDataModule, DATA_DIR
from model import LitModel


EXPERIMENT_NAME = f'ship_classification_{datetime.now()}'

PRECISION = 16
MAX_EPOCHS = 16
GPUS = 0
BATCH_SIZE = 4
NUM_WORKERS = 8
SPLIT = (0.1, 0.1)

SAVE_PATH = './'
FILE_NAME = 'model'


def train(datamodule: LightningDataModule,
          model: LightningModule,
          logger: Logger,
          callbacks: List[Callback]) -> Trainer:
    
    trainer = Trainer(max_epochs=MAX_EPOCHS,
                      gpus=GPUS,
                      precision=PRECISION,
                      logger=logger,
                      callbacks=callbacks,
                      enable_model_summary=False)

    trainer.fit(model=model, datamodule=datamodule)
    return trainer


def get_callbacks(inject_clbk: Iterable[Callback] = [], save_model: bool = False) -> List[Callback]:
    callbacks = [EarlyStopping('valid_loss', min_delta=0.001, patience=2)]
       
    if save_model:
        checkpoint_clbk = ModelCheckpoint(dirpath=SAVE_PATH, filename=FILE_NAME,
                                          monitor='valid_Accuracy', save_top_k=1, mode='max')
        callbacks.append(checkpoint_clbk)
        
    for clbk in inject_clbk:
        callbacks.append(clbk)
    
    return callbacks
        

def main(verbose: bool = False,
         save_torch: bool = False):
    
    datamodule = ShipDataModule(split=SPLIT, bs=BATCH_SIZE)
    model = LitModel()
    logger = WandbLogger(project='Ships_wandb_course', name=EXPERIMENT_NAME, offline=False)
    callbacks = get_callbacks(save_model=save_torch)
    
    trainer = train(datamodule, model, logger, callbacks)
    trainer.test(ckpt_path="best", datamodule=datamodule)
    
    if verbose:
        print(trainer.logged_metrics)
    
    if save_torch:
        ckpt_callback = callbacks[1]
        best_model = ckpt_callback.best_model_path
        print(f'Training session saved to {best_model}')
    
    
if __name__ == '__main__':
    main(verbose=True, save_torch=True)