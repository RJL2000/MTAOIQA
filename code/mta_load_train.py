from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from config_mta import config_mta
from scipy.optimize import curve_fit
from scipy.special import expit

import numpy as np
import torch

def mean_squared_error(actual, predicted, squared=True):
    """计算 MSE or RMSE (squared=False)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 计算误差
    error = predicted - actual
    
    # 计算均方根误差
    res = np.mean(error**2)
    if squared==False:
        res = np.sqrt(res)
    
    return res


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + expit(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def train_mta_oiqa(config, net, criterion, criterion2, awl, optimizer, train_loader):
    losses = []
    acc_t = torch.zeros(1).cuda("cuda:0")
    acc_r = torch.zeros(1).cuda("cuda:0")
    acc_d = torch.zeros(1).cuda("cuda:0")
    net.train()
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
    
    # for data in tqdm(train_loader):
    for data in train_loader:
        
        d = data['d_img_org'].cuda("cuda:0")    # [batch, viewports, 3, 224, 224]: [B, 8, 3, 224, 224]

        sample_num += d.shape[0]

        labels_quality = data['score']
        labels_quality = labels_quality.type(torch.FloatTensor).cuda("cuda:0")
        
        if config.qrtd:
            
            labels_type = data['type']
            labels_range = data['range']
            labels_degree = data['degree']

            labels_type = labels_type.type(torch.LongTensor).cuda("cuda:0")
            labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

            labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
            labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

            labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
            labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

            pred_q, pred_t, pred_r, pred_d = net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss_t = criterion2(torch.squeeze(pred_t), labels_one_hot)
            loss_r = criterion2(torch.squeeze(pred_r), labels_one_hot_r)
            loss_d = criterion2(torch.squeeze(pred_d), labels_one_hot_d)

            loss = awl(loss_q, loss_t, loss_r, loss_d)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

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
            labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

            labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
            labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

            pred_q, pred_t, pred_r = net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss_t = criterion2(torch.squeeze(pred_t), labels_one_hot)
            loss_r = criterion2(torch.squeeze(pred_r), labels_one_hot_r)

            loss = awl(loss_q, loss_t, loss_r)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

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
            labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

            labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
            labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

            pred_q, pred_r, pred_d = net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            
            loss_r = criterion2(torch.squeeze(pred_r), labels_one_hot_r)
            loss_d = criterion2(torch.squeeze(pred_d), labels_one_hot_d)

            loss = awl(loss_q, loss_r, loss_d)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

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
            labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

            labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
            labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

            pred_q, pred_t, pred_d = net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss_t = criterion2(torch.squeeze(pred_t), labels_one_hot)
            loss_d = criterion2(torch.squeeze(pred_d), labels_one_hot_d)

            loss = awl(loss_q, loss_t, loss_d)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

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
            labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

            pred_q, pred_r = net(d)     # [B]

            optimizer.zero_grad()

            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss_r = criterion2(torch.squeeze(pred_r), labels_one_hot_r)

            loss = awl(loss_q, loss_r)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

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
            labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

            pred_q, pred_t = net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss_t = criterion2(torch.squeeze(pred_t), labels_one_hot)
            
            loss = awl(loss_q, loss_t)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

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
            labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

            pred_q, pred_d = net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss_d = criterion2(torch.squeeze(pred_d), labels_one_hot_d)

            loss = awl(loss_q, loss_d)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # save results in one epoch
            pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
            mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

            pred_d_all = np.append(pred_d_all, torch.argmax(pred_d, dim=1).data.cpu().numpy())
            degree_all = np.append(degree_all, labels_degree.data.cpu().numpy())

            acc_num_d = (torch.argmax(pred_d, dim=1) == labels_degree).sum().item()
            acc_d = acc_d + acc_num_d
        else:
            pred_q= net(d)     # [B]

            optimizer.zero_grad()
            loss_q = criterion(torch.squeeze(pred_q), labels_quality)
            loss = loss_q

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # save results in one epoch
            pred_all = np.append(pred_all, pred_q.data.cpu().numpy())
            mos_all = np.append(mos_all, labels_quality.data.cpu().numpy())

    # compute correlation coefficient
    loss = np.mean(losses)
    logistic_pred_all = fit_function(mos_all, pred_all)
    plcc = pearsonr(logistic_pred_all, mos_all)[0]
    srcc = spearmanr(logistic_pred_all, mos_all)[0]
    rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)

    if config.qrtd:
        acc_r = acc_r / sample_num
        acc_t = acc_t / sample_num
        acc_d = acc_d / sample_num

        return loss, acc_r, acc_t, acc_d, plcc, srcc, rmse
    
    elif config.qrt:
        acc_r = acc_r / sample_num
        acc_t = acc_t / sample_num

        return loss, acc_r, acc_t, plcc, srcc, rmse
    
    elif config.qrd:
        acc_r = acc_r / sample_num
        acc_d = acc_d / sample_num

        return loss, acc_r, acc_d, plcc, srcc, rmse
    elif config.qtd:
        acc_t = acc_t / sample_num
        acc_d = acc_d / sample_num

        return loss, acc_t, acc_d, plcc, srcc, rmse
    elif config.qr:
        acc_r = acc_r / sample_num
       
        return loss, acc_r, plcc, srcc, rmse
    elif config.qt:
        acc_t = acc_t / sample_num

        return loss, acc_t, plcc, srcc, rmse
    elif config.qd:
        acc_d = acc_d / sample_num

        return loss, acc_d, plcc, srcc, rmse
    else:
        return loss, plcc, srcc, rmse
    

def test_mta_oiqa(config, net, test_loader):
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
                labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

                labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
                labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

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
                labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

                labels_range = labels_range.type(torch.LongTensor).cuda("cuda:0")
                labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

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
                labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

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
                labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

                labels_degree = labels_degree.type(torch.LongTensor).cuda("cuda:0")
                labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

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
                labels_one_hot_r = torch.nn.functional.one_hot(labels_range, num_classes=config.range_num).float()

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
                labels_one_hot = torch.nn.functional.one_hot(labels_type, num_classes=config.type_num).float()

                

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
                labels_one_hot_d = torch.nn.functional.one_hot(labels_degree, num_classes=config.degree_num).float()

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
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.train_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
                
                with open(config.train_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')
                
                with open(config.train_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
                
            acc_t = acc_t / sample_num
            acc_r = acc_r / sample_num
            acc_d = acc_d / sample_num

            return acc_t, acc_r, acc_d, plcc, srcc, rmse 
        
        elif config.qrt:

            if config.print_pred_file:

                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.train_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
                
                with open(config.train_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')

            acc_t = acc_t / sample_num
            acc_r = acc_r / sample_num
            return acc_t, acc_r, plcc, srcc, rmse 
        
        elif config.qrd:
            if config.print_pred_file:
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')
                
                with open(config.train_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')
                
                with open(config.train_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
                
            acc_r = acc_r / sample_num
            acc_d = acc_d / sample_num
            return acc_r, acc_d, plcc, srcc, rmse
        elif config.qtd:
            if config.print_pred_file:
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.train_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
                
                with open(config.train_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
            acc_t = acc_t / sample_num
            acc_d = acc_d / sample_num
            return acc_t, acc_d, plcc, srcc, rmse  
        
        elif config.qr:
            if config.print_pred_file:
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.train_out_range_file, "w", encoding="utf8") as f:
                    f.write("range,pred_r\n")
                    for i in range(len(np.squeeze(pred_r_all))):
                        f.write(str(range_all[i])+','+str(pred_r_all[i])+'\n')

            acc_r = acc_r / sample_num
       
            return acc_r, plcc, srcc, rmse 
        elif config.qt:
            if config.print_pred_file:
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

                with open(config.train_out_type_file, "w", encoding="utf8") as f:
                    f.write("type,pred_t\n")
                    for i in range(len(np.squeeze(pred_t_all))):
                        f.write(str(type_all[i])+','+str(pred_t_all[i])+'\n')
            
            acc_t = acc_t / sample_num
            return acc_t, plcc, srcc, rmse
        elif config.qd:
            if config.print_pred_file:
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')
    
                with open(config.train_out_degree_file, "w", encoding="utf8") as f:
                    f.write("degree,pred_d\n")
                    for i in range(len(np.squeeze(pred_d_all))):
                        f.write(str(degree_all[i])+','+str(pred_d_all[i])+'\n')
                
            acc_d = acc_d / sample_num
            return acc_d, plcc, srcc, rmse 

        else:
            if config.print_pred_file:
                with open(config.train_out_file, "w", encoding="utf8") as f:
                    f.write("mos,pred\n")
                    for i in range(len(np.squeeze(pred_all))):
                        f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

            return plcc, srcc, rmse 
            

            
