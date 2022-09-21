
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
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import multiprocessing

## create Pytorch Dataset ##  Dataset stores the samples and their corresponding labels
class Sequence_Dataset(Dataset):

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

## DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Sequence_Dataset(self.train_sequences)
        self.test_dataset  = Sequence_Dataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=cpu_count()
            num_workers = 2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

## Lightning Modul
"""
A LightningModule organizes your PyTorch code into 6 sections:
--Computations (init).
--Train Loop (training_step)
--Validation Loop (validation_step)
--Test Loop (test_step)
--Prediction Loop (predict_step)
--Optimizers and LR Schedulers (configure_optimizers)
"""


class LSTM_CNN (nn.Module):
    def __init__(self, in_bands, n_classes,
                 Conv1_bands_NF = 8, Conv2_time_NF = 128, Conv3_time_NF = 256,
                 Conv4_time_NF = 64):

        super(LSTM_CNN, self).__init__()
        self.in_bands = in_bands
        self.n_classes = n_classes
        self.Conv1_bands_NF = Conv1_bands_NF
        self.Conv2_time_NF  = Conv2_time_NF
        self.Conv3_time_NF  = Conv3_time_NF
        self.Conv4_time_NF  = Conv4_time_NF

        ## CONV
        ## gives 2 times in bands and same n timesteps
        self.conv1_bands = nn.Conv1d(in_bands, Conv1_bands_NF,stride = 1 ,kernel_size=1, padding=0)

        ## keeps 2 times in bands same time sequence size
        self.conv2_time = nn.Conv1d(Conv1_bands_NF, Conv2_time_NF, stride=1, kernel_size=5, padding=2)
        ## COnv 3 128 --> 256
        self.conv3_time = nn.Conv1d(Conv2_time_NF, Conv3_time_NF, stride = 1, kernel_size = 7, padding = 3)
        ## COnv 4 256 --> 64
        self.conv4_time = nn.Conv1d(Conv3_time_NF, Conv4_time_NF, stride = 1, kernel_size = 3 , padding = 1)

        ## batchnorm
        self.bn_2 = nn.BatchNorm1d(Conv2_time_NF)
        self.bn_3 = nn.BatchNorm1d(Conv3_time_NF)

        ## relu
        self.relu = nn.ReLU()

        ## Each convolutional layer is succeeded by batch normalization,
        # with a momentum of 0.99 and epsilon of 0.001.

        ## Self out layer
        self.FC = nn.Linear(Conv4_time_NF, self.n_classes)

        ##  output layer fc LSTM
        #self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out, self.NumClassesOut)



    def forward(self, x):

        x = self.conv1_bands(x)
        print('Conv1 Band Shape: {}'.format(x.shape))
        x = self.conv2_time(x)
        print('Conv2 Time Shape: {}'.format(x.shape))


        x = self.conv3_time(x)
        x = self.bn_2(x)
        x = self.relu(x)
        print('Conv3 Time Shape: {}'.format(x.shape))

        x = self.conv4_time(x)
        x = self.bn_3(x)
        x = self.relu(x)
        print('Conv4 Time Shape: {}'.format(x.shape))

        out = self.FC(x)

        return out


class Disturbance_Predictor(pl.LightningModule):
    def __init__(self,in_bands: int, n_classes: int):
        super().__init__()
        self.model = LSTM_CNN(in_bands, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self.forward(sequences, labels)
        predictions   = torch.softmax(outputs, dim=1)
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



#multiprocessing.set_start_method("spawn")
"""
from multiprocessing import get_context
try:
    pool = get_context("fork").Pool(num_processes)
except ValueError as exc:
    if "cannot find context for 'fork'" in exc:
         pool = get_context("spawn").Pool(num_processes)
         logging.info("Switching to \"spawn\" as \"fork\" context is not found")
"""


if __name__ ==  '__main__':
    ## read data
    X_train = pd.read_csv(r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_x_8_250smps.csv", sep=';')
    y_train = pd.read_csv(r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\05_df_y_8_250smps.csv", sep=';')
    print(X_train.shape, y_train.shape)

    X_train
    ## data labeler
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train.disturbance)
    print(encoded_labels[:1000])
    print("Label Classes: ", label_encoder.classes_)

    # Adding to y_train df
    y_train["label"] = encoded_labels
    # reshape
    X_train = (X_train.set_index(['id', 'index', 'instances_rep'])
               .rename_axis(['step'], axis=1)
               .stack()
               .unstack('index')
               .reset_index())

    FEATURE_COLUMNS = X_train.columns.tolist()[3:]
    FEATURE_COLUMNS

    ## group per id for individual sequence
    sequences = []
    for instances, group in X_train.groupby("instances_rep"):
        print(id)
        sequence_features = group[FEATURE_COLUMNS]
        label = y_train[y_train.instance_rep == instances].label
        print(label)
        sequences.append((sequence_features, label))
    sequences[0]

    ## split train test
    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2)
    print("Number of Training Sequences: ", len(train_sequences))
    print("Number of Testing Sequences: ", len(test_sequences))

    # Model
    model = Disturbance_Predictor(
        in_bands=len(FEATURE_COLUMNS),
        n_classes=len(label_encoder.classes_)
    )

    ## train init
    save_model_path = r"P:\workspace\jan\fire_detection\dl\models_store\05_FCN_Light"

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_model_path,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min")

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="P:/workspace/jan/fire_detection/dl/models_store/05_FCN_Light/tl_logger/", name="surface_predictor")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    N_EPOCHS = 3
    BATCH_SIZE = 64
    data_module = SequenceDataModule(train_sequences, test_sequences, BATCH_SIZE)

    # Trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=N_EPOCHS
        # gpus = 1,
    )

    trainer.fit(model, data_module)
