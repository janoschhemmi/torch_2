
## Model Architecture

import torch
import torch.nn as nn

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

