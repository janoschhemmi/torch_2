import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset, DataLoader
#import torchvision
import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Create PyTorch Dataset
class TSDataset(Dataset):
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.tensor(label).long()
        )


class TSDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TSDataset(self.train_sequences)
        self.test_dataset =  TSDataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1

        )


## ________________________________________________________________________________________
class se_block(nn.Module):
    def __init__(self, in_layer, out_layer):
        super(se_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer // 8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer // 8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1, out_layer // 8)
        self.fc2 = nn.Linear(out_layer // 8, out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)

        x_out = torch.add(x, x_se)
        return x_out

## self try
class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()

        self.cbr1 = conbr_block(in_layer, out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer, out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)

    def forward(self, x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out

## block of CNN and batch norm and activation function
class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn(x)
        out = self.relu(x)

        return out

class Unet_try (nn.Module):
    def __init__(self, in_bands, time_steps ,n_classes ):
        super(Unet_try, self).__init__()
        self.in_bands = in_bands
        self.time_steps = time_steps
        self.classes = n_classes

        ## gives 2 times in bands and same n timesteps
        self.conv1_bands = nn.Conv1d(in_bands, in_bands * 2,stride = 1 ,kernel_size=1, padding=0)

        ## creates first downlayer; will give 32 out layers (with 8 in bands)
        self.layer1 = self.down_layer(input_layer=in_bands * 2,
                                      out_layer=in_bands * 2 * 2, kernel = 5, stride = 1,
                                      depth = 2)

        self.layer2 = self.down_layer(input_layer=in_bands * 2 * 2,
                                      out_layer=in_bands * 2 * 2 * 2, kernel=5, stride=1,
                                      depth=2)

        self.layer3 = self.down_layer(input_layer=in_bands * 2 * 2 * 2,
                                      out_layer=in_bands * 2 * 2 * 2 * 2, kernel=5, stride=1,
                                      depth=1)

        ## halfs the input length
        self.maxPool1D1 = nn.MaxPool1d( kernel_size = 2, stride=2)

        ## doubles the input
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.cbr_up1 = conbr_block(int(in_bands * 2 * 2 * 2 * 2 +in_bands*  2 * 2 * 2 ), int(in_bands * 2 * 2 * 2 * 2 ), 5, 1, 1)

        self.cbr_up2 = conbr_block(int(in_bands * 2 * 2 + in_bands * 2 * 2 * 2 * 2,), int(in_bands * 2 * 2 * 2 * 2 ), 5, 1, 1)

        self.conv2_out = nn.Conv2d(in_bands * 2 *2 + in_bands * 2 * 2 * 2 * 2, n_classes, stride=1, kernel_size=1, padding=0)



    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):

        ## depth *2 input bands
        out_pre_1 = self.conv1_bands(x)
        ## depth *2 *2 input bands (28)
        out_1     = self.layer1(out_pre_1)
        ## maxpooling -- lengh = 0.5 input length
        out_1_2 = self.maxPool1D1(out_1)

        ## conv block 2
        out_2 =     self.layer2(out_1_2)
        ## maxpooling -- lengh = 0.25 input length
        out_2_2 = self.maxPool1D1(out_2)

        ## conv block 3 --in_bands * 2 * 2 * 2 * 2
        out_3 = self.layer3(out_2_2)

        ## expand
        out_4 = self.upsample(out_3)

        ## conc
        up_1 = torch.cat([out_2, out_4], 1)
        up_1 = self.cbr_up1(up_1)

        up_2 = self.upsample(up_1)
        up_2 = torch.cat([out_1, up_2], 1)

        out_out = self.conv2_out(up_2)
        return out_out



class UNET_Predictor(pl.LightningModule):
    def __init__(self, in_bands: int, n_classes: int, time_steps :  int):

        super().__init__()
        self.model = Unet_try( in_bands, time_steps ,n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels    = batch["label"]

        loss, outputs = self.forward(sequences, labels)


        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self.forward(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

