import torch.nn as nn
import torch.nn.functional as F#, F#, bill y'all pt2

class LWM1CNN(nn.Module):

    def __init__(self):

        super(LWM1CNN, self).__init__() # still u af, lol

        # 5h4p3
        # (batch_size, 3, 128, 128)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # 1st ly batch normalization

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # 2st ly batch normalization

        self.fc1 = nn.Linear(32*32*32, 256)
        self.fc2 = nn.Linear(256, 10) # len(classes) == 10 / for noe at least / as always / we shall see shell ..


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) # despacey half

        x = F.relu(self.bn2(self.conv(x)))
        x = F.max_pool2d(x, 2) # despacey another half

        x = x.view(x.size(0), -1) # phlatten tomatoes

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)

        return x

model = LWM1CNN()
