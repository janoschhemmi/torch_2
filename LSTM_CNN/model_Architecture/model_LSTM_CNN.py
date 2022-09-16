
class LSTM_CNN (nn.Module):
    def __init__(self, in_bands, time_steps ,n_classes ):
        super(LSTM_CNN, self).__init__()
        self.in_bands = in_bands
        self.time_steps = time_steps
        self.classes = n_classes

        ## gives 2 times in bands and same n timesteps
        self.conv1_bands = nn.Conv1d(in_bands, in_bands * 2,stride = 1 ,kernel_size=1, padding=0)
        ## keeps 2 times in bands same time sequence size
        self.conv2_time = nn.Conv1d(in_bands * 2, in_bands * 2, stride=1, kernel_size=5, padding=2)
        ## relu
        self.relu = nn.ReLU()
        ## batchnorm
        self.bn = nn.BatchNorm1d(out_layer)

        ## 2 COnvs
        self.convblock3 = ConvBlock(in_bands * 2, 128, 7)
        self.convblock4 = ConvBlock(128, 256, 5)
        self.convblock5 = ConvBlock(256, 64, 3)

        self.maxpool = MaxPool2d( (3,2), stride = 1)
        self.fc = nn.Linear(layers[-1], c_out)








    def forward(self, x):

        x = self.conv1_bands(x)
        x = self.conv1_time(x)
        x = self.bn(x)
        x = self.relu(x)

        return out_out

