# custom_yolov8.py

from ultralytics import YOLO
import torch.nn as nn

class CustomYOLO(YOLO):
    def __init__(self, model='yolov8n.pt', dropout_rate=0.5):
        super().__init__(model)
        self.dropout_rate = dropout_rate
        self._add_dropout_layers()

    def _add_dropout_layers(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                module.add_module('dropout', nn.Dropout2d(p=self.dropout_rate))

    def predict(self, **kwargs):
        self.model.train()
        return super().predict(**kwargs)
