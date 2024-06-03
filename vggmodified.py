import torch.nn as nn
import torch.nn.functional as F

# VGG Network Modified

class VGG16Modified(nn.Module):
    def __init__(self):
        super(VGG16Modified, self).__init__()
        self.layer1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.layer2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.layer3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.layer3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.layer4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.layer4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.layer5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.layer5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.layer1_1(x), inplace=True)
        x = F.relu(self.layer1_2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.layer2_1(x), inplace=True)
        x = F.relu(self.layer2_2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.layer3_1(x), inplace=True)
        x = F.relu(self.layer3_2(x), inplace=True)
        x = F.relu(self.layer3_3(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.layer4_1(x), inplace=True)
        x = F.relu(self.layer4_2(x), inplace=True)
        x = F.relu(self.layer4_3(x), inplace=True)

        x = F.relu(self.layer5_1(x), inplace=True)
        x = F.relu(self.layer5_2(x), inplace=True)
        x = F.relu(self.layer5_3(x), inplace=True)
        output = x

        return output