import os
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from src.multicropdataset import strong_transforms
from src.vision_transformer import vit_small
import utils
import src.vision_transformer as vits



class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        return self.linear(x)


class mugs(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
        embed_dim = self.model.embed_dim * (4+ int(0))

        self.num_labels = num_labels
        self.linear = nn.Linear(embed_dim , num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forword(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


if __name__ == '__main__':
    input = torch.randn(2, 3, 224, 224)
    model = vit_small(num_classes=3)
    # pretrained_weights = '/home/yht/mugs/checkpoint0040.pth'
    # utils.load_pretrained_weights(model, pretrained_weights, 'teacher', 'vit_small', 16)
    # output, relation_out= m(input)
    output, _ = model(input)
    print(output)