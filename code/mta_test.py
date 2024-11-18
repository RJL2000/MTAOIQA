import os
import torch
import numpy as np
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from mta_model import creat_model
from config_mta import config_mta
from mta_load_train import fit_function, pearsonr, spearmanr, mean_squared_error
from mta_dataset import JUFE_10K_Dataset, OIQ_10K_Dataset, OIQA_Dataset, CVIQ_Dataset
from tqdm import tqdm


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
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = Dataset(
        info_csv_path= config.test_dataset,
        transform=test_transform
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
        net = creat_model(config=config, pretrained=config.pretrained)
        num_loss = 4
    elif config.qrt or config.qrd or config.qtd:
        net = creat_model(config=config, pretrained=config.pretrained)
        num_loss = 3
    elif config.qr or config.qd or config.qt:
        net = creat_model(config=config, pretrained=config.pretrained)
        num_loss = 2
    else:
        net = creat_model(config=config, pretrained=config.pretrained)
        num_loss = 1

    net = nn.DataParallel(net, device_ids=device_ids).cuda()

    with torch.no_grad():
        acc_t = torch.zeros(1).cuda("cuda:0")
        acc_r = torch.zeros(1).cuda("cuda:0")
        acc_d = torch.zeros(1).cuda("cuda:0")
        net.eval()
        # save data for one epoch
        sample_num = 0

        pred_all = []
        mos_all = []

        pred_t_all = []
        type_all = []

        pred_r_all = []
        range_all = []

        pred_d_all = []
        degree_all = []

        test_loader = tqdm(test_loader)

        for data in test_loader:
        # for data in test_loader:
            d = data['d_img_org'].cuda("cuda:0")
            sample_num += d.shape[0]
            labels_quality = data['score']
            labels_quality = labels_quality.type(torch.FloatTensor).cuda("cuda:0")

            if config.qrtd:
                labels_type = data['type']
                labels_range = data['range']
                labels_degree = data['degree']

                labels_type = labels_type.type(torch.LongTensor).cuda("cuda:0")
                 

                labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
                 

                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                  

                pred_q, pred_t, pred_r, pred_d = net(d)

                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_t_all = np.append(pred_t_all, torch.argmax(pred_t, dim=1).data.cpu().numpy())
                type_all = np.append(type_all, labels_type.data.cpu().numpy())

                pred_r_all = np.append(pred_r_all, torch.argmax(pred_r, dim=1).data.cpu().numpy())
                range_all = np.append(range_all, labels_range.data.cpu().numpy())

                pred_d_all = np.append(pred_d_all, torch.argmax(pred_d, dim=1).data.cpu().numpy())
                degree_all = np.append(degree_all, labels_degree.data.cpu().numpy())


                acc_num_t = (torch.argmax(pred_t, dim=1) == labels_type).sum().item()
                acc_num_r = (torch.argmax(pred_r, dim=1) == labels_range).sum().item()
                acc_num_d = (torch.argmax(pred_d, dim=1) == labels_degree).sum().item()

                acc_t = acc_t + acc_num_t
                acc_r = acc_r + acc_num_r
                acc_d = acc_d + acc_num_d

            elif config.qrt:
                labels_type = data['type']
                labels_range = data['range']

                labels_type = labels_type.type(torch.LongTensor).cuda("cuda:0")
                 

                labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
                 

                pred_q, pred_t, pred_r = net(d)

                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_t_all = np.append(pred_t_all, torch.argmax(pred_t, dim=1).data.cpu().numpy())
                type_all = np.append(type_all, labels_type.data.cpu().numpy())

                pred_r_all = np.append(pred_r_all, torch.argmax(pred_r, dim=1).data.cpu().numpy())
                range_all = np.append(range_all, labels_range.data.cpu().numpy())

                acc_num_t = (torch.argmax(pred_t, dim=1) == labels_type).sum().item()
                acc_num_r = (torch.argmax(pred_r, dim=1) == labels_range).sum().item()
                
                acc_t = acc_t + acc_num_t
                acc_r = acc_r + acc_num_r
            
            elif config.qrd:
                labels_range = data['range']
                labels_degree = data['degree']

                labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
                 

                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                  

                pred_q, pred_r, pred_d = net(d)

                

                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_r_all = np.append(pred_r_all, torch.argmax(pred_r, dim=1).data.cpu().numpy())
                range_all = np.append(range_all, labels_range.data.cpu().numpy())

                pred_d_all = np.append(pred_d_all, torch.argmax(pred_d, dim=1).data.cpu().numpy())
                degree_all = np.append(degree_all, labels_degree.data.cpu().numpy())

                acc_num_r = (torch.argmax(pred_r, dim=1) == labels_range).sum().item()
                acc_num_d = (torch.argmax(pred_d, dim=1) == labels_degree).sum().item()

                acc_r = acc_r + acc_num_r
                acc_d = acc_d + acc_num_d
            elif config.qtd:
                labels_type = data['type']
                labels_degree = data['degree']

                labels_type = labels_type.type(torch.LongTensor).cuda("cuda:0")
                 

                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                  

                pred_q, pred_t, pred_d = net(d)

                

                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_t_all = np.append(pred_t_all, torch.argmax(pred_t, dim=1).data.cpu().numpy())
                type_all = np.append(type_all, labels_type.data.cpu().numpy())

    
                pred_d_all = np.append(pred_d_all, torch.argmax(pred_d, dim=1).data.cpu().numpy())
                degree_all = np.append(degree_all, labels_degree.data.cpu().numpy())


                acc_num_t = (torch.argmax(pred_t, dim=1) == labels_type).sum().item()
               
                acc_num_d = (torch.argmax(pred_d, dim=1) == labels_degree).sum().item()
                acc_t = acc_t + acc_num_t
                acc_d = acc_d + acc_num_d
     
            elif config.qr:

                labels_range = data['range']

                labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
                 

                pred_q, pred_r = net(d)


                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_r_all = np.append(pred_r_all, torch.argmax(pred_r, dim=1).data.cpu().numpy())
                range_all = np.append(range_all, labels_range.data.cpu().numpy())

                acc_num_r = (torch.argmax(pred_r, dim=1) == labels_range).sum().item()
                acc_r = acc_r + acc_num_r
            
            elif config.qt:
                labels_type = data['type']
                

                labels_type = labels_type.type(torch.LongTensor).cuda("cuda:0")
                 

                

                pred_q, pred_t= net(d)

                

                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_t_all = np.append(pred_t_all, torch.argmax(pred_t, dim=1).data.cpu().numpy())
                type_all = np.append(type_all, labels_type.data.cpu().numpy())
                acc_num_t = (torch.argmax(pred_t, dim=1) == labels_type).sum().item()
            
                acc_t = acc_t + acc_num_t
            
            elif config.qd:
                
                labels_degree = data['degree']

                
                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                  

                pred_q, pred_d = net(d)

                

                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

                pred_d_all = np.append(pred_d_all, torch.argmax(pred_d, dim=1).data.cpu().numpy())
                degree_all = np.append(degree_all, labels_degree.data.cpu().numpy())

                acc_num_d = (torch.argmax(pred_d, dim=1) == labels_degree).sum().item()
                acc_d = acc_d + acc_num_d
              
            else:
                pred_q = net(d)

               
                # save results in one epoch
                pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
                mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

        # compute correlation coefficient

        logistic_pred_all = fit_function(mos_all, pred_all)
        plcc = pearsonr(logistic_pred_all, mos_all)[0]
        srcc = spearmanr(logistic_pred_all, mos_all)[0]
        rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)

        if config.qrtd:

            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.test_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
                
                with open(config.test_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')
                
                with open(config.test_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
                
            acc_t = acc_t / sample_num
            acc_r = acc_r / sample_num
            acc_d = acc_d / sample_num

            print("[test] acc_r_t: %.4f, acc_t_t: %.4f, acc_d_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_r, acc_t, acc_d, plcc, srcc, rmse))
        
        elif config.qrt:

            if config.print_pred_file:

                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.test_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
                
                with open(config.test_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')

            acc_t = acc_t / sample_num
            acc_r = acc_r / sample_num
            print( "[test] acc_r_t: %.4f, acc_t_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_r, acc_t, plcc, srcc, rmse))
        
        elif config.qrd:
            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')
                
                with open(config.test_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')
                
                with open(config.test_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
                
            acc_r = acc_r / sample_num
            acc_d = acc_d / sample_num
            print("[test] acc_r_t: %.4f, acc_d_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_r, acc_d, plcc, srcc, rmse))

        elif config.qtd:
            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.test_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
                
                with open(config.test_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
            acc_t = acc_t / sample_num
            acc_d = acc_d / sample_num
            print("[test] acc_t_t: %.4f, acc_d_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_t, acc_d, plcc, srcc, rmse))
        
        elif config.qr:
            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.test_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')

            acc_r = acc_r / sample_num
       
            print("[test] acc_r_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_r, plcc, srcc, rmse))

        elif config.qt:
            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.test_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
            
            acc_t = acc_t / sample_num
            print("[test] acc_t_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_t, plcc, srcc, rmse))

        elif config.qd:
            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')
    
                with open(config.test_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
                
            acc_d = acc_d / sample_num
            print("[test] acc_d_t: %.4f, plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (acc_d, plcc, srcc, rmse))

        else:
            if config.print_pred_file:
                with open(config.test_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

            print("[test] plcc_t: %.4f, srcc_t: %.4f, rmse_t: %.4f" % (plcc, srcc, rmse))