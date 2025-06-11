import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from network.ResNet import *

class GaussianNoiseGeneratorResnet18(torch.nn.Module):
    def __init__(self, n_channel, input_size):
        super(GaussianNoiseGeneratorResnet18, self).__init__()
        self.n_channel = n_channel
        self.input_size = input_size
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self._build_up()
        self.upper_bound = 0.1

    def _build_up(self):
        self.conv1 = Conv1(self.n_channel, 64, 2)
        #for layers
        self.in_planes = 64
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 64, self.num_blocks[3], stride=2)
        self.upsample = nn.UpsamplingBilinear2d(size=self.input_size)
        self.fc_variance = nn.Conv2d(64, self.n_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.fc_mean = nn.Conv2d(64, self.n_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.upsample(out)
        variance = self.fc_variance(out).abs()
        # mu = self.fc_mean(feat)
        # mu = mu.minimum(torch.tensor(self.upper_bound))  # A Projection Operation
        mu = torch.zeros(x.shape).to(x.device)
        return mu, variance

    def sampling(self, mu, variance, num):
        """
        Args:
            mu:
            variance:
            num:

        Returns:
            Tensor: batch_size * num * dim
        """
        batch_size = mu.shape[0]
        channel = mu.shape[1]
        height = mu.shape[2]
        width = mu.shape[3]
        # noises = torch.empty([batch_size, num, dim]).to(self.device)
        # gaussian_distribution = MultivariateNormal(torch.zeros(dim).to(self.device), torch.eye(dim).to(self.device))
        # noise = gaussian_distribution.sample([batch_size * num])  # ?
        noise = torch.randn(batch_size, num, channel, height, width).to(variance.device)
        # noise = noise.reshape(batch_size, num, dim)
        var = variance.expand(num, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
        m = mu.expand(num, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
        noise = var * noise + m
        # for i in range(batch_size):
        #     mu_i = mu[i, :]
        #     var = variance[i, :]
        #     noises[i].data.copy_(noise)
        return noise


class GaussianNoiseGeneratorDNN3(torch.nn.Module):
    def __init__(self, n_channel, input_dim):
        super(GaussianNoiseGeneratorDNN3, self).__init__() 
        self.n_channel = n_channel
        self.input_dim = input_dim
        self.dim1 = n_channel * input_dim * input_dim
        self._build_up()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.002)
                m.bias.data.zero_()

    def _build_up(self):
        self.fc1 = torch.nn.Linear(self.dim1, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc_variance = torch.nn.Linear(1024, self.dim1)
        self.fc_mean = torch.nn.Linear(1024, self.dim1)

    def forward(self, x):
        feat = x.reshape(x.size(0), -1)
        feat = self.fc1(feat)
        feat = feat.relu()
        feat = self.fc2(feat)
        feat = feat.relu()
        variance = self.fc_variance(feat).abs()
        variance = variance.reshape(variance.size(0), self.n_channel, self.input_dim, self.input_dim)
        mu = torch.zeros(x.shape).to(x.device)
        return mu, variance
    
    def sampling(self, mu, variance, num):
        """
        Args:
            mu:
            variance:
            num:

        Returns:
            Tensor: batch_size * num * dim
        """
        batch_size = mu.shape[0]
        channel = mu.shape[1]
        height = mu.shape[2]
        width = mu.shape[3]
        noise = torch.randn(batch_size, num, channel, height, width).to(variance.device)
        var = variance.expand(num, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
        m = mu.expand(num, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
        noise = var * noise + m
        return noise