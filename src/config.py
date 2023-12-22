from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class ModelConfig(BaseModel):
    backbone_name: str = 'resnet18'
    pretrained: bool = True
    cnn_output_size: int = 128
    rnn_features_num: int = 48
    rnn_hidden_size: int = 64
    rnn_dropout: float = 0.1
    rnn_bidirectional: bool = True
    rnn_num_layers: int = 2
    num_classes: int = 11


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    batch_size: int
    num_iterations: int
    n_workers: int
    width: int
    height: int
    vocab: str
    text_size: int


class Config(BaseModel):
    project_name: str
    experiment_name: str
    data_config: DataConfig
    n_epochs: int
    num_classes: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model_kwargs: ModelConfig
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
