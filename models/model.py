import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageOps
from torchvision import transforms
import torch.nn.functional as F

class CRNN(torch.nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()

        self.conv_1 = torch.nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = torch.nn.Linear(2304,64)
        self.drop_1 = torch.nn.Dropout(0.2)
        self.lstm = torch.nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = torch.nn.Linear(64,vocab_size + 1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        print(x.size())
        x = self.pool_1(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        x = self.pool_2(x)
        print(x.size())
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1) # potreba pro nasledny vstup do RNN
        x = F.relu(self.linear1(x))
        print(x.size())
        x = self.drop_1(x)
        print(x.size())
        x, _ = self.lstm(x)
        print(x.size())
        x = self.output(x)
        x = x.permute(1,0,2)

        if targets is not None:
            log_probs = F.log_softmax(x,2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = torch.nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss
        return x, None


if __name__ == "__main__":
    crnn = CRNN(19)
    img = torch.rand((1, 1, 75, 300))
    print(img)
    x, loss = crnn(img, torch.rand((1, 5)))
    print(f"loss:{loss}")
    print(f"x:{x}")


