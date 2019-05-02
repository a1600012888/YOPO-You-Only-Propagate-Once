import config
from base_model.small_cnn import SmallCNN


def create_network():
    return SmallCNN()


def test():
    net = create_network()
    y = net((torch.randn(1, 1, 28, 28)))
    print(y.size())
