import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from re import I
import scipy.io as scio
import random
import numpy as np
import torch
import time
import copy
import logging
import wandb
import math
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from validation import *
# from dataloader.datasets import *
from dataloader.datasets_y import *
from components.N2N import *
from networks.HyperRED_VAE import RED_CNN_NOHYP

from utils.Communication import *
from utils.fednova import FedNova
from networks.HyperRED_BN import RED_CNN2 as Hyper_BN
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F

## DataLoader Rule
def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(logger_name, log_file, level=logging.INFO):

    ## Read the Codes by yourself
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M") # RECORD Time
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    vlog = logging.getLogger(logger_name)
    vlog.setLevel(level)
    vlog.addHandler(fileHandler)

    return vlog

def create_logger(indx, opt):
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

    time_name = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
    mode_name = opt.mode
    model_name = opt.model
    batch_size = opt.batch_size
    com = opt.communication
    epo = opt.epochs
    sup_weight = opt.geo_weight
    unsup_weight = opt.unsup_weight
    #  str(opt.sup_weight) + '_' + str(opt.unsup_weight) + '_' 
    if not os.path.exists((opt.log_path+mode_name+'_'+model_name+'_'+str(batch_size) + '_' + str(com) + '_' + str(epo) + '_' + str(sup_weight)+ '_' + time_name)):
        os.makedirs((opt.log_path+mode_name+'_'+model_name+'_'+str(batch_size) + '_' + str(com) + '_' + str(epo) + '_' + str(sup_weight)+ '_' + time_name))

    log_file = opt.log_path+mode_name+'_'+model_name+'_'+str(batch_size) + '_' + str(com) + '_' + str(epo) + '_' + str(sup_weight)+ '_' + time_name + '/client' + '_' + str(indx) +'.log'
    logger = get_logger('NB', log_file)
    return logger

def my_collate_test(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    label_data = torch.stack([item[1] for item in batch], 0)
    res_name = [item[2] for item in batch]
    prj_data = [item[3] for item in batch]
    option = torch.stack([item[4] for item in batch], 0)
    feature = torch.stack([item[5] for item in batch], 0)
    ana_fe = torch.stack([item[6] for item in batch], 0)
    return input_data, label_data, res_name, prj_data, option, feature, ana_fe

def my_collate(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    label_data = torch.stack([item[1] for item in batch], 0)
    prj_data = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    ana_fe = torch.stack([item[5] for item in batch], 0)
    return input_data, label_data, prj_data, option, feature, ana_fe

def Test_Datasets(test_path, opt):

    ### Build Dataset

    src_dataset_1 = DataLoader(testset_loader(test_path+"geometry_1"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_2 = DataLoader(testset_loader(test_path+"geometry_2"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_3 = DataLoader(testset_loader(test_path+"geometry_3"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_4 = DataLoader(testset_loader(test_path+"geometry_4"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_5 = DataLoader(testset_loader(test_path+"geometry_5"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_6 = DataLoader(testset_loader(test_path+"geometry_6"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_7 = DataLoader(testset_loader(test_path+"geometry_7"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    src_dataset_8 = DataLoader(testset_loader(test_path+"geometry_8"),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    # src_dataset_9 = DataLoader(testset_loader(test_path+"geometry_9"),
    #                         batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)
    # src_dataset_10 = DataLoader(testset_loader(test_path+"geometry_10"),
    #                         batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate_test)

    dataloaders = []
    dataloaders.append(src_dataset_1)
    dataloaders.append(src_dataset_2)
    dataloaders.append(src_dataset_3)
    dataloaders.append(src_dataset_4)
    dataloaders.append(src_dataset_5)
    dataloaders.append(src_dataset_6)
    dataloaders.append(src_dataset_7)
    dataloaders.append(src_dataset_8)
    # dataloaders.append(src_dataset_9)
    # dataloaders.append(src_dataset_10)

    return dataloaders
 
def Train_Dataset(opt):
    ### Build Dataset
    print(opt.data_path)
    dataset_1 = trainset_loader(opt.data_path + "geometry_1")
    train_size = int(opt.data_ratio * dataset_1.__len__())
    # t_set1, v_set1 = random_split(dataset_1, [train_size,dataset_1.__len__()-train_size])
    t_set1, v_set1, e_set1 = random_split(dataset_1,[train_size, int(train_size/2), dataset_1.__len__()-train_size-int(train_size/2)])
    src_dataset_1 = DataLoader(t_set1,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_1 = DataLoader(v_set1,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_2 = trainset_loader(opt.data_path + "geometry_2")
    train_size = int(opt.data_ratio * dataset_2.__len__())
    # t_set2, v_set2 = random_split(dataset_2, [train_size,dataset_2.__len__()-train_size])
    t_set2, v_set2, e_set2 = random_split(dataset_2, [train_size, int(train_size / 2),
                                                      dataset_2.__len__() - train_size - int(train_size / 2)])
    src_dataset_2 = DataLoader(t_set2,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_2 = DataLoader(v_set2,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_3 = trainset_loader(opt.data_path + "geometry_3")
    train_size = int(opt.data_ratio * dataset_3.__len__())
    # t_set3, v_set3 = random_split(dataset_3, [train_size,dataset_3.__len__()-train_size])
    t_set3, v_set3, e_set3 = random_split(dataset_3, [train_size, int(train_size / 2),
                                                      dataset_3.__len__() - train_size - int(train_size / 2)])
    src_dataset_3 = DataLoader(t_set3,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_3 = DataLoader(v_set3,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_4 = trainset_loader(opt.data_path + "geometry_4")
    train_size = int(opt.data_ratio * dataset_4.__len__())
    # t_set4, v_set4 = random_split(dataset_4, [train_size, dataset_4.__len__() - train_size])
    t_set4, v_set4, e_set4 = random_split(dataset_4, [train_size, int(train_size / 2),
                                                      dataset_4.__len__() - train_size - int(train_size / 2)])
    src_dataset_4 = DataLoader(t_set4,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_4 = DataLoader(v_set4,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_5 = trainset_loader(opt.data_path + "geometry_5")
    train_size = int(opt.data_ratio * dataset_5.__len__())
    # t_set5, v_set5 = random_split(dataset_5, [train_size, dataset_5.__len__() - train_size])
    t_set5, v_set5, e_set5 = random_split(dataset_5, [train_size, int(train_size / 2),
                                                      dataset_5.__len__() - train_size - int(train_size / 2)])
    src_dataset_5 = DataLoader(t_set5,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_5 = DataLoader(v_set5,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_6 = trainset_loader(opt.data_path + "geometry_6")
    train_size = int(opt.data_ratio * dataset_6.__len__())
    # t_set6, v_set6 = random_split(dataset_6, [train_size, dataset_6.__len__() - train_size])
    t_set6, v_set6, e_set6 = random_split(dataset_6, [train_size, int(train_size / 2),
                                                      dataset_6.__len__() - train_size - int(train_size / 2)])
    src_dataset_6 = DataLoader(t_set6,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_6 = DataLoader(v_set6,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_7 = trainset_loader(opt.data_path + "geometry_7")
    train_size = int(opt.data_ratio * dataset_7.__len__())
    # t_set7, v_set7 = random_split(dataset_7, [train_size, dataset_7.__len__() - train_size])
    t_set7, v_set7, e_set7 = random_split(dataset_7, [train_size, int(train_size / 2),
                                                      dataset_7.__len__() - train_size - int(train_size / 2)])
    src_dataset_7 = DataLoader(t_set7,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_7 = DataLoader(v_set7,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataset_8 = trainset_loader(opt.data_path + "geometry_8")
    train_size = int(opt.data_ratio * dataset_8.__len__())
    # t_set8, v_set8 = random_split(dataset_8, [train_size, dataset_8.__len__() - train_size])
    t_set8, v_set8, e_set8 = random_split(dataset_8, [train_size, int(train_size / 2),
                                                      dataset_8.__len__() - train_size - int(train_size / 2)])
    src_dataset_8 = DataLoader(t_set8,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    val_dataset_8 = DataLoader(v_set8,
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    # src_dataset_9 = DataLoader(trainset_loader(opt.data_path + "geometry_9"),
    #                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    # src_dataset_10 = DataLoader(trainset_loader(opt.data_path + "geometry_10"),
    #                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataloaders = []
    dataloaders.append(src_dataset_1)
    dataloaders.append(src_dataset_2)
    dataloaders.append(src_dataset_3)
    dataloaders.append(src_dataset_4)
    dataloaders.append(src_dataset_5)
    dataloaders.append(src_dataset_6)
    dataloaders.append(src_dataset_7)
    dataloaders.append(src_dataset_8)
    # dataloaders.append(src_dataset_9)
    # dataloaders.append(src_dataset_10)

    test_dataloaders = []
    test_dataloaders.append(val_dataset_1)
    test_dataloaders.append(val_dataset_2)
    test_dataloaders.append(val_dataset_3)
    test_dataloaders.append(val_dataset_4)
    test_dataloaders.append(val_dataset_5)
    test_dataloaders.append(val_dataset_6)
    test_dataloaders.append(val_dataset_7)
    test_dataloaders.append(val_dataset_8)

    return dataloaders, test_dataloaders

class net():
    def __init__(self,opt):

        ### To Log or some addresses
        self.mode_name = opt.mode
        self.model_name = opt.model
        time_name = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        self.path = opt.model_save_path+ self.mode_name + '_' + self.model_name + '_' + str(opt.batch_size) + '_' + str(opt.communication) + '_' + str(opt.epochs) + '_' + str(opt.geo_weight) + '_' + str(opt.geo_k_weight) + '_' +time_name
        self.logger = [create_logger(idx,opt) for idx in range(opt.num_clients)]

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.checkpoint_interval = opt.checkpoint_interval

        ### Hyper Param Settings 
        self.start = 0
        self.epoch = opt.epochs
        self.com = opt.communication
        self.client_num = opt.num_clients
        self.client_weights = [1 / self.client_num for i in range(self.client_num)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.increase_ratio = opt.increase_ratio
        self.Lambda1 = opt.Lambda1
        self.Lambda2 = opt.Lambda2
        self.open_clients = opt.open_clients

        self.supweight = opt.sup_weight
        self.unsupweight = opt.unsup_weight
        self.MSELoss = nn.MSELoss()

        ### Model Init

        self.model = RED_CNN_NOHYP()

        self.server_model = self.model #.to(self.device)

        self.best_model = copy.deepcopy(self.server_model)

        self.models = [copy.deepcopy(self.server_model) for idx in range(self.client_num)]

        self.best_models = [copy.deepcopy(self.models[i]) for i in range(self.client_num)]

        self.initialize_weights()

        ## Optimizer
        self.optimizer = torch.optim.Adam(self.server_model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)
        self.optimizers = [torch.optim.Adam(self.models[idx].parameters(), lr = opt.lr, weight_decay = opt.weight_decay)
                               for idx in range(self.client_num)]

        self.train_dataset, self.val_dataset = Train_Dataset(opt)
        self.test_dataset = Test_Datasets(opt.test_path, opt)

        self.temp = 0.5

    def initialize_weights(self):
        for module in self.server_model.modules():
            # if isinstance(module, fed_model.prj_module):
            #     nn.init.normal_(module.weight_fed, mean=0.02, std=0.001)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


    def train(self, opt):
        
        print('---------------Start Training---------------')
        iter_index = [1,1,1,1,1,1,1,1]
        com_index = [1,1,1,1,1,1,1,1]
        operation_seed_counter = 0
        best_round = 0
        best_psnr = 0.
        best_ssim = 0.
        best_psnr_client = [0., 0., 0., 0., 0., 0., 0., 0.]
        best_ssim_client = [0., 0., 0., 0., 0., 0., 0., 0.]

        temp_vec = torch.tensor([[0.8333, 0.5000, 0.2889, 0.3615, 0.2500, 0.3333, 0.2857],
                            [0.3333, 0.7500, 0.4222, 0.2538, 0.5000, 0.5000, 1.0000],
                            [0.6667, 0.7500, 0.6667, 0.7308, 0.8750, 0.8333, 0.0707],
                            [0.5975, 0.5859, 1.1111, 0.9615, 0.5000, 0.5000, 0.3549],
                            [0.7460, 0.7031, 0.2222, 0.4385, 0.3750, 0.6667, 0.3671],
                            [0.4406, 0.7129, 0.5333, 0.4077, 0.5000, 0.4333, 0.9673],
                            [0.6882, 0.7373, 0.8889, 0.8077, 0.3750, 0.8333, 0.0380],
                            [0.5873, 0.4883, 0.6667, 0.8077, 0.5000, 0.6667, 0.4010]
                            ])

        for com_iter in range(self.start, self.com):            
            # dynamic_temperature = 1 / (math.tanh((com_iter+1)/60)*200)
            for i_wkr in range(self.client_num):
                for epoch in range(self.epoch):

                    #Init This Epoch
                    st = time.time()
                    all_loss_this_epoch = 0.
                    loss2_this_epoch = 0.
                    loss1_this_epoch = 0.

                    self.models[i_wkr] = self.models[i_wkr].to(self.device)
                    self.server_model = self.server_model.to(self.device)

                    possible_values = torch.cat((temp_vec[:i_wkr], temp_vec[i_wkr+1:])) 
                    
                    time_list =[]

                    for batch_index, data in enumerate(self.train_dataset[i_wkr]):
                       
                        input_data, label_data, prj_data, options, feature_vec, ana_data = data
                        bs, _ = feature_vec.size()
                        
                        indices = torch.randint(0, possible_values.size(1), (bs,))

                        pos_vec = temp_vec[indices].to(self.device)

                        input_data = input_data.to(self.device)
                        label_data = label_data.to(self.device)
                        # options = options.to(self.device)
                        feature_vec = feature_vec.to(self.device)
                        ana_data = ana_data.to(self.device)

                        self.optimizers[i_wkr].zero_grad()

                        time_start = time.time()
                        output, code = self.models[i_wkr](input_data,feature_vec,ana_data) #, mode='train')
                        time_end = time.time()
                        t = time_end - time_start 
                        time_list.append(t)
                        supervised_loss = self.MSELoss(output, label_data)

                        output2, code2 = self.models[i_wkr](input_data, pos_vec, ana_data)
                        # output2, code2 = self.server_model(input_data, pos_vec, ana_data)

                        supervised_loss2 = self.MSELoss(output2,label_data)

                        cosine_sim = F.cosine_similarity(code, code2, dim=1) 

                        code_distance = cosine_sim.abs().mean()

                        loss_all = supervised_loss  + opt.geo_weight * code_distance              


                        loss_all.backward()
                        self.optimizers[i_wkr].step()
                        del input_data, label_data, feature_vec, ana_data, output, code
                        torch.cuda.empty_cache()

                        all_loss_this_epoch += loss_all.item()
                        
                        loss1_this_epoch += supervised_loss.item()     ## Supervsied Loss
                        # loss2_this_epoch += unsupervised_loss.item()     ## Unsupervised Loss
                    print('mean time:',np.mean(time_list))
                    self.models[i_wkr] = self.models[i_wkr].cpu()
                    self.server_model = self.server_model.cpu()

                    avg_all_loss = all_loss_this_epoch / len(self.train_dataset[i_wkr])
                    avg_sup = loss1_this_epoch / len(self.train_dataset[i_wkr])
                    # avg_unsup = loss2_this_epoch / len(self.train_dataset[i_wkr])

                    print(
                        'Com: {:04d} | Worker ID: {:04d} | Epoch: {:04d} / {:04d} | Sup_Loss={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                        .format(com_iter, i_wkr, epoch + 1, self.epoch, np.mean(avg_sup), np.mean(avg_all_loss),
                                time.time() - st))

                    self.logger[i_wkr].info(
                        'Com: {:04d} | Worker ID: {:04d} | Epoch: {:04d} / {:04d} | Sup_Loss={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                        .format(com_iter, i_wkr, epoch + 1, self.epoch, np.mean(avg_sup), np.mean(avg_all_loss),
                                time.time() - st))

                    iter_index[i_wkr] += 1
                    # com_index += 1

            ### Warning in Fed
            self.pre_models = []
            self.pre_models =[copy.deepcopy(self.models[idx]) for idx in range(self.client_num)]
            self.server_model, self.models = communication(opt, self.server_model, self.models, self.client_weights)

            if self.checkpoint_interval != -1 and (com_iter + 1) % self.checkpoint_interval == 0:                
                for check_id in range(self.client_num):
                    torch.save(self.server_model.state_dict(), '%s/model_commu_%04d.pth' % (self.path, com_iter + 1))
                    torch.save(self.models[check_id].state_dict(),
                                '%s/model_worker_id(%04d)_commu_%04d.pth' % (self.path, check_id, com_iter + 1))

            ### Start Close-Set Testing in the training set
            torch.cuda.empty_cache()
            # del input_data, label_data, feature_vec, ana_data, output, code

            psnr_all = [[] for i in range(self.client_num)]
            ssim_all = [[] for i in range(self.client_num)]
            for cli_num in range(self.client_num):
                for batch_, data in enumerate(self.val_dataset[cli_num]):
                    # print(batch_)
                    inp, lab, _, _, fe, ana_data = data
                    if opt.mode.lower() == 'hyperfed' or opt.mode.lower() == 'fedbn' or opt.mode.lower() == 'fedmri' or opt.mode.lower() == 'fedper' or opt.mode.lower() == 'yang_geo':
                        # print(opt.mode.lower())
                        psnr, ssim = val_img(opt, self.models[cli_num], inp, lab, fe, ana_data)
                    else:
                        psnr, ssim = val_img(opt, self.server_model, inp, lab, fe, ana_data)

                    if not isinstance(psnr, list):
                        psnr_all[cli_num].extend([psnr])
                    else:
                        psnr_all[cli_num].extend(psnr)

                    if not isinstance(ssim,list):
                        ssim_all[cli_num].extend([ssim])
                    else:
                        ssim_all[cli_num].extend(ssim)
                    # break
                # print(cli_num)

            psnr_each_client =list(map(lambda x: np.average(x), psnr_all))
            ssim_each_client = list(map(lambda x: np.average(x), ssim_all))
            psnr_average = np.average(psnr_each_client)
            ssim_average = np.average(ssim_each_client)

            print('---------------Close-Validation-Set Test Round {:04d}---------------'.format(com_iter))
            for i_wkr in range(self.client_num): 
                print('Com: {:04d} | Worker ID: {:04d} | PSNR={:.6f}, SSIM={:.6f}'
                        .format(com_iter, i_wkr, psnr_each_client[i_wkr], ssim_each_client[i_wkr]))
                self.logger[i_wkr].info('Com: {:04d} | Worker ID: {:04d} | PSNR={:.6f}, SSIM={:.6f}'
                        .format(com_iter, i_wkr, psnr_each_client[i_wkr], ssim_each_client[i_wkr]))
                # wandb.log({"now_psnr_"+str(i_wkr):psnr_each_client[i_wkr],"now_ssim_"+str(i_wkr):ssim_each_client[i_wkr], "com iter": com_iter})
                com_index[i_wkr] += 1

            print('Com: {:04d} | Average_Performance | PSNR={:.6f}, SSIM={:.6f}'
                .format(com_iter, psnr_average, ssim_average))
            # wandb.log({"now_avg_psnr":psnr_average,"now_avg_ssim_":ssim_average})

            if com_iter > 0 and com_iter % 50 == 0:
                torch.cuda.empty_cache()
                self.test_close(opt, com_iter)

            if best_psnr < psnr_average:
                best_psnr_client = psnr_each_client
                best_ssim_client = ssim_each_client
                best_psnr = psnr_average
                best_ssim = ssim_average
                best_round = com_iter

                self.best_model = copy.deepcopy(self.server_model)
                self.best_models = [copy.deepcopy(self.models[i]) for i in range(self.client_num)]
                if com_iter > 50:
                    self.test_close(opt, com_iter)

                for check_id in range(self.client_num):
                    if com_iter > 50:
                        torch.save(self.server_model.state_dict(),'%s/best_model_commu_%04d.pth' % (self.path, com_iter + 1))
                        torch.save(self.models[check_id].state_dict(),
                                    '%s/best_model_worker_id(%04d)_commu_%04d.pth' % (self.path, check_id, com_iter + 1))
            #
                print('---------------Best Round {:04d}---------------'.format(best_round))
                for i_wkr in range(self.client_num):
                    print('Best Round: {:04d} | Worker ID: {:04d} | Best PSNR={:.6f}, Best SSIM={:.6f}'
                        .format(best_round, i_wkr, best_psnr_client[i_wkr], best_ssim_client[i_wkr]))
                    self.logger[i_wkr].info('Best Round: {:04d} | Worker ID: {:04d} | Best PSNR={:.6f}, Best SSIM={:.6f}'
                        .format(best_round, i_wkr, best_psnr_client[i_wkr], best_ssim_client[i_wkr]))

    #
    def test_open(self,opt):
    
        # psnr_open_all[open_client][client_num]
        psnr_open_all = [[[] for _ in range(opt.client_num)] for _ in range(self.open_clients)]
        ssim_open_all = [[[] for _ in range(opt.client_num)] for _ in range(self.open_clients)]
    
        for open_id in range(self.client_num, self.client_num+self.open_clients):
    
            for cli_num in range(self.client_num):
                for batch_, data in enumerate(self.test_dataset[open_id]):
                    inp, lab, _, _, _, fe = data
                    psnr, ssim = val_img(opt,self.models[cli_num], inp, lab, fe)
                    psnr_open_all[open_id][cli_num].extend(psnr)
                    ssim_open_all[open_id][cli_num].extend(ssim)
    
    
            for batch_,data in enumerate(self.test_dataset[open_id]):
                inp, lab, _, _, _, fe = data
                if opt.mode == 'HyperFed' or opt.mode == 'FedBN':
                    for cli_num in range(self.client_num):
                        psnr, ssim = val_img(opt, self.models[cli_num], inp, lab, fe)
    
    
        for cli_num in range(self.client_num, self.client_num+self.open_clients):
            for batch_, data in enumerate(self.test_dataset[cli_num]):
                inp, lab, _, _, _, fe = data
                if opt.mode == 'HyperFed' or opt.mode == 'FedBN':
                    psnr, ssim = val_img(opt, self.models[cli_num], inp, lab, fe)
                    # psnr =
                else:
                    psnr, ssim = val_img(opt, self.server_model, inp, lab, fe)
                psnr_open_all[cli_num - self.client_num].extend(psnr)
                ssim_open_all[cli_num - self.client_num].extend(ssim)
    
        return psnr_open_all, ssim_open_all

    def test_close(self,opt, com_iter):

        psnr_open_all = [[] for i in range(self.client_num)]
        ssim_open_all = [[] for i in range(self.client_num)]

        for cli_num in range(self.client_num):
            for batch_, data in enumerate(self.test_dataset[cli_num]):
                inp, lab, _, _, _,  fe, ana_data = data
                if opt.mode == 'HyperFed' or opt.mode.lower() == 'fedbn' or opt.mode.lower()=='fedper' or opt.mode.lower() == 'fedmri' or opt.mode.lower() == 'yang_geo':
                    # print(opt.mode.lower())
                    psnr, ssim = val_img(opt, self.models[cli_num], inp, lab, fe, ana_data)
                else:
                    psnr, ssim = val_img(opt, self.server_model, inp, lab, fe, ana_data)
                psnr_open_all[cli_num].extend(psnr)
                ssim_open_all[cli_num].extend(ssim)

            ####### Open Client Set #######
            # psnr_open_all, ssim_open_all = self.test_open(opt)

        psnr_each_open_client =list(map(lambda x: np.average(x), psnr_open_all))
        ssim_each_open_client = list(map(lambda x: np.average(x), ssim_open_all))
        open_psnr_average = np.average(psnr_each_open_client)
        open_ssim_average = np.average(ssim_each_open_client)
        print('---------------Close-Testing-Set Test Round {:04d}---------------'.format(com_iter))
        for i_wkr in range(self.client_num):
            print('CLOSE | Com: {:04d} | Worker ID: {:04d} | PSNR={:.6f}, SSIM={:.6f}'
                    .format(com_iter, i_wkr, psnr_each_open_client[i_wkr], ssim_each_open_client[i_wkr]))
            self.logger[i_wkr].info('OPEN | Com: {:04d} | Worker ID: {:04d} | PSNR={:.6f}, SSIM={:.6f}'
                    .format(com_iter, i_wkr, psnr_each_open_client[i_wkr], ssim_each_open_client[i_wkr]))

        print('CLOSE | Com: {:04d} | Average_Performance | PSNR={:.6f}, SSIM={:.6f}'
            .format(com_iter, open_psnr_average, open_ssim_average))

        for cli_num in range(self.client_num):
            for batch_, data in enumerate(self.test_dataset[cli_num]):
                inp, lab, _, _, _,  fe, ana_data = data
                if opt.mode == 'HyperFed' or opt.mode.lower() == 'fedbn' or opt.mode.lower()=='fedper' or opt.mode.lower() == 'fedmri' or opt.mode.lower() == 'yang_geo':
                    # print(opt.mode.lower())
                    psnr, ssim = val_img(opt, self.models[cli_num], inp, lab, fe, ana_data)
                else:
                    psnr, ssim = val_img(opt, self.server_model, inp, lab, fe, ana_data)
                psnr_open_all[cli_num].extend(psnr)
                ssim_open_all[cli_num].extend(ssim)

            ####### Open Client Set #######
            # psnr_open_all, ssim_open_all = self.test_open(opt)

        psnr_each_open_client =list(map(lambda x: np.average(x), psnr_open_all))
        ssim_each_open_client = list(map(lambda x: np.average(x), ssim_open_all))
        open_psnr_average = np.average(psnr_each_open_client)
        open_ssim_average = np.average(ssim_each_open_client)
        print('---------------Close-Testing-Set Test Round {:04d}---------------'.format(com_iter))
        for i_wkr in range(self.client_num):
            print('CLOSE | Com: {:04d} | Worker ID: {:04d} | PSNR={:.6f}, SSIM={:.6f}'
                    .format(com_iter, i_wkr, psnr_each_open_client[i_wkr], ssim_each_open_client[i_wkr]))
            self.logger[i_wkr].info('OPEN | Com: {:04d} | Worker ID: {:04d} | PSNR={:.6f}, SSIM={:.6f}'
                    .format(com_iter, i_wkr, psnr_each_open_client[i_wkr], ssim_each_open_client[i_wkr]))

        print('CLOSE | Com: {:04d} | Average_Performance | PSNR={:.6f}, SSIM={:.6f}'
            .format(com_iter, open_psnr_average, open_ssim_average))

        return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",type=int,default=1234,help="The random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--model",type=str,default='HyperRED', help='Uformer or NB_Uformer')
    parser.add_argument("--backbone",type=str,default='HyperRED',help='HyperRED or RED')
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--n_cpu", type=int, default=8)

    ###federated paras
    parser.add_argument("--num_clients", type = int, default = 8,help='Number of local clients')
    parser.add_argument("--communication", type = int, default = 100, help = 'Number of communications')
    parser.add_argument("--epochs", type=int, default = 1, help="Number of local training")
    parser.add_argument("--mode", type=str, default='yang_geo', help="FedAvg | FedProx | FedBN | HyperFed | FedMRI | FedNova | FedPer | FedKD ")
    parser.add_argument("--mu", type=float, default = 1e-3, help="the weight of fedprox")
    parser.add_argument("--open_clients", type=int, default=2, help='Number of local clients')

    ###file paras
    parser.add_argument("--model_save_path", type = str, default="saved_models/")
    parser.add_argument('--checkpoint_interval', type = int, default = 50)
    # parser.add_argument("--data_path",type=str,default="../../dataset/FedData/Mat Data/big_data/")
    # parser.add_argument("--data_path",type=str,default="../../dataset/FedData/Mat Data/ct_mask_data/")
    parser.add_argument("--data_path",type=str,default="/storage/dataset/mayov2/")
    parser.add_argument('--log_path',type=str, default="./log/")
    # parser.add_argument('--test_path',type=str, default="../../dataset/FedData/Mat Data/ct_mask_data/")
    parser.add_argument('--test_path',type=str, default="/storage/dataset/mayov2/")
    parser.add_argument('--data_ratio',type = float, default = 0.2)

    ### Neighbor2Neighbor Params
    parser.add_argument("--increase_ratio",type=float, default=2.0)
    parser.add_argument("--Lambda1", type=float, default=1.0)
    parser.add_argument("--Lambda2", type=float, default=1.0)

    ### Supervised Loss Weight and Unsupervised Loss Weight
    parser.add_argument("--sup_weight",type=float,default=0.8,help="The weight of supervised loss")
    parser.add_argument("--unsup_weight",type=float,default=0.2,help="The weight of unsupervised loss")

    parser.add_argument("--geo_weight",type=float, default = 0.0001,help="The weight of geo hyper loss")
    parser.add_argument("--geo_k_weight",type=float, default = 0.00001, help = "The weight of K-LOSS of Hyepr Geo Net")
    # parser.add_argument("--optim",type=str,default='adam', help="adam|sgd|fednova")
    opt = parser.parse_args()   # No parameter Debug


    set_seed(opt.seed)

    network = net(opt)
    network.train(opt)
