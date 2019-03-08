import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])


class MNIST():
    def __init__(self, download=True, transform=transform, batch_size=1):
        train_im = datasets.MNIST(
            root='./data', download=download, transform=transform, train=True)
        test_im = datasets.MNIST(
            root='./data', download=download, transform=transform, train=False)

        self.train_loader = DataLoader(dataset=train_im,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=test_im,
                                      batch_size=batch_size,
                                      shuffle=False)
