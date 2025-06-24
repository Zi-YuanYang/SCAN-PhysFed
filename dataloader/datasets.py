import glob
import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset


class trainset_loader(Dataset):
    def __init__(self, root):
        # print(root)
        self.file_path = 'input'
        self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path,'') + '*.mat'))

        # print('123')
    def __getitem__(self, index):
        file_A = self.files_A[index]
        # print(file_A)
        file_B = file_A.replace(self.file_path, 'label')
        file_C = file_A.replace('input', 'projection')
        file_D = file_A.replace('input', 'geometry')
        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_B)['data']
        prj_data = scio.loadmat(file_C)['data']

        ## Original 
        geometry1 = scio.loadmat(file_D)['data']
        
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)

        geometry = geometry1[0]
        
        geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0, 1, 4, 5, 7, 8, 10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5, 0, 0.005 - 0.001, 0.004 - 0.0015, 2.0 - 0.5, 2.0 - 0.5, 4.6])
        maxVal = torch.FloatTensor([11, 4, 0.012 + 0.001, 0.014 + 0.0015, 5.0 + 0.5, 4.0 + 0.5, 6])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, label_data, prj_data, option, feature

    def __len__(self):
        return len(self.files_A)

class testset_loader(Dataset):
    def __init__(self, root):
        self.file_path = 'input'
        self.files_A = sorted(glob.glob(os.path.join(root, 'test', self.file_path,'') + '*.mat'))


    def __getitem__(self, index):
        file_A = self.files_A[index]
        # res_name = file_A[64:]
        # print(file_A)
        res_name = os.path.basename(file_A)
        file_B = file_A.replace(self.file_path, 'label')
        # print(file_B)
        file_C = file_A.replace('input', 'projection')
        file_D = file_A.replace('input', 'geometry')
        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_B)['data']
        prj_data = scio.loadmat(file_C)['data']
        
        geometry1 = scio.loadmat(file_D)['data']

        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)


        geometry = geometry1[0]
        
        geometry = torch.FloatTensor(geometry).view(-1)
        # geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0, 1, 4, 5, 7, 8, 10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5, 0, 0.005 - 0.001, 0.004 - 0.0015, 2.0 - 0.5, 2.0 - 0.5, 4.6])
        maxVal = torch.FloatTensor([11, 4, 0.012 + 0.001, 0.014 + 0.0015, 5.0 + 0.5, 4.0 + 0.5, 6])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, label_data, res_name, prj_data, option, feature

    def __len__(self):
        return len(self.files_A)

