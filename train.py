import os
from typing import Any
from os import path as osp
from torch import set_float32_matmul_precision

import argparse

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import OCRDM
from src.lightning_module import OCRModule


def arg_parse() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train(config: Config):
    set_float32_matmul_precision('high')
    datamodule = OCRDM(config.data_config)
    model = OCRModule(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=config.experiment_name,
        auto_connect_frameworks=True,
    )
    task.connect(config.model_dump())

    experiment_save_path = osp.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(),
        ],
        logger=TensorBoardLogger(
            save_dir='lightning_logs',
            name=config.experiment_name,
        ),
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()

    seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config)
