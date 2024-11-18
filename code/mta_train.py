import os
import torch
import numpy as np
import time
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
# from mta_model import creat_model
# from mta_model_component.mta_model_all_vdp import creat_model 
# from mta_model_component.mta_model_all_sdm import creat_model
# from mta_model_component.mta_model_all_cdc import creat_model
# from mta_model_component.mta_model_all_mdi import creat_model
# from mta_model_component.mta_model_without_Vw_Vs import creat_model 
# from mta_model_component.mta_model_without_Vs import creat_model
# from mta_model_component.mta_model_without_Vw import creat_model
# from mta_model_component.mta_model_without_MFS import creat_model
from mta_model_component.mta_model_without_MAF import creat_model
from config_mta import config_mta
from mta_load_train import train_mta_oiqa, test_mta_oiqa
from mta_dataset import JUFE_10K_Dataset, OIQ_10K_Dataset, OIQA_Dataset, CVIQ_Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config_mta().CUDA_VISIBLE_DEVICES


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum



if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)  

    setup_seed(20)

    config = config_mta()

 
    config.log_file = config.model_name + ".log"
    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name, config.model_name)
    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
 

    
    if config.dataset_name == 'JUFE_10K':
        Dataset = JUFE_10K_Dataset
    elif config.dataset_name == 'OIQ_10K':
        Dataset = OIQ_10K_Dataset
    elif config.dataset_name == 'OIQA':
        Dataset = OIQA_Dataset
    elif config.dataset_name == 'CVIQ':
        Dataset = CVIQ_Dataset
    else:
        print('no this dataset')


    # data load
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = Dataset(
        info_csv_path= config.train_dataset,
        transform=train_transform
    )
    test_dataset = Dataset(
        info_csv_path= config.test_dataset,
        transform=test_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False
    )
    
    device_ids = config.device_ids

    if config.qrtd:
        net = creat_model(config=config, pretrained=False)
        num_loss = 4
        
    elif config.qrt or config.qrd or config.qtd:
        net = creat_model(config=config, pretrained=False)
        num_loss = 3
        
    elif config.qr or config.qd or config.qt:
        net = creat_model(config=config, pretrained=False)
        num_loss = 2
        
    else:
        net = creat_model(config=config, pretrained=False)
        num_loss = 1

    net = nn.DataParallel(net, device_ids=device_ids).cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()

    awl = AutomaticWeightedLoss(num=num_loss)
    
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0

    for epoch in range(0, config.n_epoch):
        # visual(net, test_loader)
        start_time = time.time()

        
        if config.qrtd:
            loss_train, acc_r, acc_t, acc_d, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_r: %.4f, acc_t: %.4f, acc_d: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_r, acc_t, acc_d, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        elif config.qrt:
            loss_train, acc_r, acc_t, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_r: %.4f, acc_t: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_r, acc_t, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        elif config.qrd:
            loss_train, acc_r, acc_d, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_r: %.4f, acc_d: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_r, acc_d, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        elif config.qtd:
            loss_train, acc_t, acc_d, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_t: %.4f, acc_d: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_t, acc_d, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        elif config.qr:
            loss_train, acc_r, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_r: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_r, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        elif config.qt:
            loss_train, acc_t, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_t: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_t, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        elif config.qd:
            loss_train, acc_d, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, acc_d: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, acc_d, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        else:
            loss_train, plcc, srcc, rmse = train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader)
            print("[train epoch %d/%d] loss: %.6f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                    (epoch+1, config.n_epoch, loss_train, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))

    
        if (epoch + 1) % config.val_freq == 0:
            start_time = time.time()
            if config.qrtd:
                acc_r_v, acc_t_v, acc_d_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_r_v: %.4f, acc_t_v: %.4f, acc_d_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_r_v, acc_t_v, acc_d_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            elif config.qrt:
                acc_r_v, acc_t_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_r_v: %.4f, acc_t_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_r_v, acc_t_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            elif config.qrd:
                acc_r_v, acc_d_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_r_v: %.4f, acc_d_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_r_v, acc_d_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            elif config.qtd:
                acc_t_v, acc_d_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_t_v: %.4f, acc_d_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_t_v, acc_d_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            elif config.qr:
                acc_r_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_r_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_r_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            elif config.qt:
                acc_t_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_t_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_t_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            elif config.qd:
                acc_d_v, plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] acc_d_v: %.4f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, acc_d_v, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
            else:
                plcc_v, srcc_v, rmse_v = test_mta_oiqa(config, net, test_loader)
                print("[test epoch %d/%d] plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                        (epoch+1, config.n_epoch, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))

            if plcc_v + srcc_v > main_score:
                main_score = plcc_v + srcc_v
                best_srcc = srcc_v
                best_plcc = plcc_v
                # save weights
                # ckpt_name = "best_ckpt.pt"
                model_save_path = os.path.join(config.ckpt_path, "best_ckpt_"+str(epoch+1)+".pt")
                torch.save(net.module.state_dict(), model_save_path)

    