import argparse
import torch
import os

from src.lightning_module import OCRModule
from src.config import Config
from src.constants import CONFIGS_PATH


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='model checkpoint path')
parser.add_argument('onnx', type=str, help='onnx model path')


def torch_to_onnx(config: Config, checkpoint: str, onnx: str) -> None:
    model = OCRModule.load_from_checkpoint(checkpoint, config=config)
    model.to_onnx(
        file_path=onnx,
        input_sample=torch.randn(1, 3, config.data_config.height, config.data_config.width),
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    args = parser.parse_args()
    config = Config.from_yaml(os.path.join(CONFIGS_PATH, 'config.yaml'))
    torch_to_onnx(config=config, checkpoint=args.checkpoint, onnx=args.onnx)
# /home/16-ocr/lightning_logs/exp1/version_1/checkpoints/epoch=29-step=3000.ckpt
# /home/16-ocr/experiments/exp1/ocr.onnx