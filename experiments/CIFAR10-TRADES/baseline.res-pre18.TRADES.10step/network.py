import config
from base_model.wide_resnet import WideResNet
from base_model.preact_resnet import PreActResNet18

def create_network():
    # return WideResNet(34)
    return PreActResNet18()


def test():
    net = create_network()
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())
