import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):
        x= F.relu(self.conv1_1(x))
        x= F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x= F.relu(self.conv2_1(x))
        x= F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x= F.relu(self.conv3_1(x))
        x= F.relu(self.conv3_2(x))
        x= F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x= F.relu(self.conv4_1(x))
        x= F.relu(self.conv4_2(x))
        x= F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x= F.relu(self.conv5_1(x))
        x= F.relu(self.conv5_2(x))
        x= F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, 2)

        return x

class Classifer(nn.Module):
    def __init__(self):
        super(Classifer, self).__init__()
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop7 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.drop7(x))
        x = F.dropout(x, 0.5, self.training)
        return x


class VGG(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG, self).__init__()
        self.features = ConvBlock()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = Classifer()

        self.alpha = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(8192, 1), nn.Sigmoid())
        self.fc8 = nn.Linear(8192, 8)

    def forward(self, x):
        vs = []
        alphas = []
        for i in range(6):
            f = x[i,:,:,:]
            f = f.view(1,3, 224, 224)
            f = self.features(f)
            f = self.avgpool(f)
            f = torch.flatten(f, 1)
            f = self.classifier(f)
            vs.append(f)
            alphas.append(self.alpha(f))
        vs_stack = torch.stack(vs, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        vm = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))

        for i in range(len(vs)):
            vs[i] = torch.cat([vs[i], vm], dim=1)

        vs_stack_4096 = torch.stack(vs, dim=2)
        betas = []
        for index, v in enumerate(vs):
            betas.append(self.beta(v))

        betas_stack = torch.stack(betas, dim=2)

        output = vs_stack_4096.mul(betas_stack* alphas_stack)
        output = output.sum(2)
        output = output.div((betas_stack*alphas_stack).sum(2))

        output = output.view(output.size(0), -1)

        pred_score = self.fc8(output)

        return pred_score

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)