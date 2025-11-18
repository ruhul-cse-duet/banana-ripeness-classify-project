
import logging
import os
from functools import lru_cache
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'resnet_Model.pth'
NUM_CLASSES = 4


def to_device(data, target_device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, target_device) for x in data]
    return data.to(target_device, non_blocking=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        logging.info(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']
            )
        )


def _conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class CNNNeuralNet(ImageClassificationBase):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = _conv_block(in_channels, 64)
        self.conv2 = _conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(_conv_block(128, 128), _conv_block(128, 128))

        self.conv3 = _conv_block(128, 256, pool=True)
        self.conv4 = _conv_block(256, 512, pool=True)

        self.res2 = nn.Sequential(_conv_block(512, 512), _conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def _resolve_checkpoint_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidates = [
        os.path.join(project_root, 'model', MODEL_NAME),
        os.path.join(project_root, 'models', MODEL_NAME),
        os.path.join(project_root, MODEL_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1]


@lru_cache(maxsize=1)
def _load_model() -> CNNNeuralNet:
    checkpoint_path = _resolve_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    raw_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state, dict) and 'state_dict' in raw_state:
        state_dict = raw_state['state_dict']
    else:
        state_dict = raw_state if isinstance(raw_state, dict) else {}

    model = CNNNeuralNet(3, NUM_CLASSES).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logging.info("State dict diffs. Missing: %s Unexpected: %s", missing, unexpected)

    model.eval()
    return model


def prediction_img(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model = _load_model()
    x = to_device(img, device)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(probabilities, dim=1)

    return predicted.cpu(), probabilities.cpu()
