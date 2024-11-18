import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class JUFE_10K_Dataset(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        # idx_list = [str(i) for i in range(15)]
        idx_list = [str(i) for i in range(8)]
        column_names = idx_list + ['mos', 'type', 'range', 'degree']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        self.type = self.df['type']
        self.range = self.df['range']
        self.degree = self.df['degree']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(8):
            p1, p2, p3 = self.X.iloc[index, i].split("/")
            path = os.path.join("/mnt/10T/rjl/dataset", p1, p2, p3)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)
        

        # # 8vps乱序
        vs1 = img_list
        vs1 = torch.cat(vs1)

        imgs = vs1

        mos = torch.FloatTensor(np.array(self.mos[index]))
   
        t = torch.from_numpy(np.array(self.type[index])) - 1.0
        t.type(torch.float64)

        r = torch.from_numpy(np.array(self.range[index])) - 1.0
        r.type(torch.float64)

        d = torch.from_numpy(np.array(self.degree[index])) - 1.0
        d.type(torch.float64)

        sample = {
            'd_img_org': imgs,
            'score': mos,
            'type': t,
            'range': r,
            'degree': d
        }

        return sample

class OIQ_10K_Dataset(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        # idx_list = [str(i) for i in range(15)]
        idx_list = [str(i) for i in range(8)]
        column_names = idx_list + ['d', 'mos']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        self.range = self.df['d']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(8):
            p1, p2 = self.X.iloc[index, i].split("/")
            path = os.path.join("/mnt/10T/tzw/methods/dataset/OIQ-10K/viewports_8", p1, p2)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)
        

        vs1 = img_list
        vs1 = torch.cat(vs1)

        imgs = vs1

        mos = torch.FloatTensor(np.array(self.mos[index]))
   
        r = torch.from_numpy(np.array(self.range[index])) - 1.0
        r.type(torch.float64)

        sample = {
            'd_img_org': imgs,
            'score': mos,
            'range': r
        }

        return sample

class OIQA_Dataset(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        # idx_list = [str(i) for i in range(15)]
        idx_list = [str(i) for i in range(8)]
        column_names = idx_list + ['mos', 'type', 'degree']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        self.type = self.df['type']
        self.degree = self.df['degree']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(8):
            p1, p2, p3 = self.X.iloc[index, i].split("/")
            path = os.path.join("/home/d310/10t/wkc/Database/OIQA", p1, p2, p3)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)
        

        # # 8vps乱序
        vs1 = img_list
        vs1 = torch.cat(vs1)

        imgs = vs1

        mos = torch.FloatTensor(np.array(self.mos[index]))
   
        t = torch.from_numpy(np.array(self.type[index])) - 1.0
        t.type(torch.float64)

        d = torch.from_numpy(np.array(self.degree[index])) - 1.0
        d.type(torch.float64)

        sample = {
            'd_img_org': imgs,
            'score': mos,
            'type': t,
            'degree': d
        }

        return sample
    
class CVIQ_Dataset(Dataset):
    def __init__(self, info_csv_path, transform=None):
        super().__init__()
        self.transform = transform
        # idx_list = [str(i) for i in range(15)]
        idx_list = [str(i) for i in range(8)]
        column_names = idx_list + ['mos', 'type', 'degree']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        self.type = self.df['type']
        self.degree = self.df['degree']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(8):
            p1, p2, p3 = self.X.iloc[index, i].split("/")
            path = os.path.join("/home/d310/10t/wkc/Database/CVIQ", p1, p2, p3)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)
        

        # # 8vps乱序
        vs1 = img_list
        vs1 = torch.cat(vs1)

        imgs = vs1

        mos = torch.FloatTensor(np.array(self.mos[index]))
   
        t = torch.from_numpy(np.array(self.type[index])) - 1.0
        t.type(torch.float64)

        d = torch.from_numpy(np.array(self.degree[index])) - 1.0
        d.type(torch.float64)

        sample = {
            'd_img_org': imgs,
            'score': mos,
            'type': t,
            'degree': d
        }

        return sample




if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.Resize((516, 516)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # test_dataset = JUEF_10K_Dataset(info_csv_path="/home/d310/10t/rjl/multitask_OIQA/file/JUFE-10K/test_st_final_viewport8_mos_type_range_degree_1.csv", transform=test_transform)
    test_dataset = OIQ_10K_Dataset(info_csv_path="/home/d310/10t/rjl/multitask_OIQA/file/OIQ-10K/OIQ-10K_test_info.csv", transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=8,
        shuffle=False,
    )
    print(len(test_dataset))

    for data in test_loader:
        imgs = data['d_img_org']
        mos = data['score']
        t = data['range'].cuda("cuda:0")
        
        print(imgs.shape)
        print(mos.shape)
        print(mos)
        print(t.shape)
        print(t)
        
        break



    
        