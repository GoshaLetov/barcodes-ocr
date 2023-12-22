from torch import nn
import torch
from timm import create_model
from src.config import ModelConfig


class CRNN(nn.Module):
    """Реализует CRNN модель для OCR задачи.
    CNN-backbone берется из timm, в RNN части стоит GRU.
    """

    def __init__(
        self,
        model_kwargs: ModelConfig,
    ) -> None:
        super().__init__()

        # Предобученный бекбон для фичей. Можно обрезать, не обязательно использовать всю глубину.
        self.backbone = create_model(
            model_kwargs.backbone_name,
            pretrained=model_kwargs.pretrained,
            features_only=True,
            out_indices=(2,),
        )

        self.gate = nn.Conv2d(model_kwargs.cnn_output_size, model_kwargs.rnn_features_num, kernel_size=1, bias=False)

        # Рекуррентная часть.
        self.rnn = nn.GRU(
            input_size=576,
            hidden_size=model_kwargs.rnn_hidden_size,
            dropout=model_kwargs.rnn_dropout,
            bidirectional=model_kwargs.rnn_bidirectional,
            num_layers=model_kwargs.rnn_num_layers,
        )

        classifier_in_features = model_kwargs.rnn_hidden_size
        if model_kwargs.rnn_bidirectional:
            classifier_in_features = 2 * model_kwargs.rnn_hidden_size

        # Классификатор.
        self.fc = nn.Linear(classifier_in_features, model_kwargs.num_classes)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        cnn_features = self.backbone(tensor)[0]
        cnn_features = self.gate(cnn_features)
        cnn_features = cnn_features.permute(3, 0, 2, 1)
        cnn_features = cnn_features.reshape(
            cnn_features.shape[0],
            cnn_features.shape[1],
            cnn_features.shape[2] * cnn_features.shape[3],
        )
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.fc(rnn_output)
        return self.softmax(logits)
