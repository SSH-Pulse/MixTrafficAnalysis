import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    """Basic residual block for a ResNet architecture."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu_1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu_2(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture with multiple layers of BasicBlocks."""
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, 1000)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ConvFeatureExtractor(nn.Module):
    """Extractor for convolutional features from input sequences."""
    def __init__(self, c, num_kernels):
        super(ConvFeatureExtractor, self).__init__()
        self.c = c
        self.num_kernels = num_kernels

        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=num_kernels,
            kernel_size=(3, c),
            stride=(1, 1),
            padding=(1, 0)  # Padding only in the s dimension
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.layer1 = self._make_layer(num_kernels, 1, stride=1)
        self.layer2 = self._make_layer(num_kernels * 2, 1, stride=1)
        self.layer3 = self._make_layer(num_kernels * 4, 1, stride=1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.num_kernels, planes, stride))
            self.num_kernels = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        b, l, s, c = x.shape
        x = x.view(b * l, 1, s, c)  # Reshaping for Conv2D input
        x = self.conv2d(x)
        x = x.squeeze(-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = x.view(b, l, -1)
        return x


class CombineFeature(nn.Module):
    """Combine features from length and time data through 1D convolution."""
    def __init__(self, b, l, input_channel, num_kernels):
        super(CombineFeature, self).__init__()
        self.b = b
        self.l = l
        self.input_channel = input_channel
        self.num_kernels = num_kernels

        self.conv1d_layer1 = nn.Conv1d(in_channels=2, out_channels=self.num_kernels, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv1d_layer2 = nn.Conv1d(in_channels=self.num_kernels, out_channels=self.num_kernels // 4, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU()
        self.conv1d_layer3 = nn.Conv1d(in_channels=self.num_kernels // 4, out_channels=1, kernel_size=3, padding=1)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x = x.view(self.b * self.l, 2, self.input_channel)
        x = self.conv1d_layer1(x)
        x = self.relu_1(x)
        x = self.conv1d_layer2(x)
        x = self.relu_2(x)
        x = self.conv1d_layer3(x)
        x = self.relu_3(x)
        x = x.view(self.b, self.l, -1)
        return x


class BiGRUModel(nn.Module):
    """BiGRU Model with feature extraction and GRU layers."""
    def __init__(self, input_size, hidden_size, num_layers, te_shape, droprate, output_classes=2, bidirectional=True):
        super(BiGRUModel, self).__init__()
        self.b, self.l, self.s = te_shape
        self.CombineFeature = CombineFeature(self.b, self.l, input_size, 64)
        self.ConvF = ConvFeatureExtractor(4, input_size // 4)
        self.embed = nn.Embedding(1515, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear((hidden_size) * num_directions, output_classes)
        self.Drop = nn.Dropout(droprate)

    def _init_hidden(self, batch_size):
        num_directions = 2 if self.gru.bidirectional else 1
        hidden = torch.zeros(self.gru.num_layers * num_directions, batch_size, self.gru.hidden_size).to(device)
        return hidden

    def forward(self, len_data, time_data):
        time_t = self.ConvF(time_data)
        len_t = self.embed(len_data)
        combined_t = torch.stack((time_t, len_t), dim=2)

        x = self.CombineFeature(combined_t)
        x = self.Drop(x)

        hidden = self._init_hidden(x.size(0))
        out, _ = self.gru(x, hidden)

        out = out[:, -1, :]
        out = self.fc(out)
        return out
