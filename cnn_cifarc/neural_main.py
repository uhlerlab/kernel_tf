import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn.functional as F
import neural_trainer
from torch.utils.data import DataLoader
import dataset


def main():

    seed = 1717
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loader, test_loader = dataset.get_cifar_data()
    print("Data shape: ", len(train_loader.dataset), len(test_loader.dataset))

    neural_trainer.train_net(train_loader, test_loader, save=True,
                             lr=1e-4,
                             num_classes=10)
    
        
if __name__ == "__main__":
    main()
