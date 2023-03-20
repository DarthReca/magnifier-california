import argparse
import logging
import os
import pathlib
from pathlib import Path

import comet_ml
import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils
from lightning_modules import CaliforniaDataModule
from neural_net import MagnifierNet


@hydra.main(config_path="configs", config_name="UnifiedFocal_Magnifier")
def main(cfg: DictConfig):
    # Set common seed
    pl.seed_everything(47, True)
    # Create datamodule
    datamodule = CaliforniaDataModule(**cfg["dataset"])
    # Create model
    pl_model = instantiate(cfg["model"])
    # Setup logger
    logger = loggers.CometLogger(**cfg["logger"], experiment_name=cfg["model"]["name"])
    # Setup checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=11,
    )
    # Setup early stopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )
    # Setup learning rate logger
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # Setup trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_logger],
    )

    if "train" == cfg.mode:
        trainer.fit(pl_model, datamodule=datamodule)
        logger.experiment.log_model(
            "Best model", trainer.checkpoint_callback.best_model_path
        )
    if "test" == cfg.mode:
        trainer.test(pl_model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    logger.experiment.end()


if __name__ == "__main__":
    main()
