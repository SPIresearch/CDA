import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class RotateEMNIST(Dataset):
    def __init__(self, rotate_angle, root='', train=True, split ='balanced'):
        if train:
            data_file = 'EMNIST/processed/training_' + split + '.pt'
        else:
            data_file = 'EMNIST/processed/test_' + split + '.pt'
        self.data, self.targets = torch.load(os.path.join(root, data_file))
        self.rotate_angle = rotate_angle
        self.train = train


    def __getitem__(self, index):
        from torchvision.transforms.functional import rotate
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, y = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

      
        if not self.train:
            angle =np.random.rand() * 180.
            img = rotate(img, angle)
        else:
            img = rotate(img, self.rotate_angle)
        img = transforms.ToTensor()(img).to(torch.float)

        return img, \
               y, \
               np.array([self.rotate_angle / 180.0], dtype=np.float32)

    def __len__(self):
        return len(self.data)

class EMNIST(Dataset):
    def __init__(self, root='', train=True, tar=False, split ='balanced'):
        if train:
            data_file = 'EMNIST/processed/training_' + split + '.pt'
        else:
            data_file = 'EMNIST/processed/test_' + split + '.pt'
        self.data, self.targets = torch.load(os.path.join(root, data_file))


    def __getitem__(self, index):
        from torchvision.transforms.functional import rotate
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        img, y = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img).to(torch.float)

        
     
        return img, \
               y, \
               np.array(0, dtype=np.float32)

    def __len__(self):
        return len(self.data)