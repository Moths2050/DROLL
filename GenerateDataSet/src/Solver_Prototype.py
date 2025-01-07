import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from collections import OrderedDict
from src import NetWorks_1217
from piqa import psnr, ssim
from src import SSIM
import matplotlib.pyplot as plt

class Solver(object):
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.train_start = args.train_start

        self.FBP_Net = NetWorks_1217.FBP_Net(args)
        # print(self.FBP_Net)
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.FBP_Net = nn.DataParallel(self.FBP_Net)
        self.FBP_Net.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()#SSIM.Structural_loss()#nn.MSELoss()
        # self.criterion1 = SSIM.Structural_loss()
        # self.criterion2 = nn.MSELoss()
        self.optimizer = optim.Adam(self.FBP_Net.parameters(), self.lr)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'FBPNET_{}iter.ckpt'.format(iter_))
        torch.save(self.FBP_Net.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'FBPNET_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.FBP_Net.load_state_dict(state_d)
        else:
            self.FBP_Net.load_state_dict(torch.load(f))

    def lr_decay(self):
        lr = self.lr * 0.8
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        if self.train_start > 0:
            self.load_model(self.train_start)  # 加载已有模型

        train_losses = []
        total_iters = 0
        start_time = time.time()

        for epoch in range(0, self.num_epochs):
            self.FBP_Net.train(True)

            # 每次取一个batch组出来进行训练，每个batch组包含了batch_size个图像组
            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add channel
                x = x.unsqueeze(1).float().to(self.device)*100
                # y = y.unsqueeze(1).float().to(self.device)*20# 2020 8 27
                y = y.unsqueeze(1).float().to(self.device)*100  # 2020 12 12

                pred = self.FBP_Net(x)*100
                loss = self.criterion(pred, y)
                # loss1 = self.criterion1(pred, y)
                # loss2 = self.criterion2(pred, y)
                # loss = 0.8*loss1 + 0.2*loss2
                self.FBP_Net.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters,
                                                                                                        epoch,
                                                                                                        self.num_epochs,
                                                                                                        iter_ + 1,
                                                                                                        len(self.data_loader),
                                                                                                        loss.item(),
                                                                                                        time.time() - start_time))

                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)  # 保存模型
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))

    ####################################################################################################################
    def test(self):
        del self.FBP_Net
        # load
        self.FBP_Net = NetWorks_1217.FBP_Net(self.args ).to(self.device)
        self.load_model(self.test_iters)  # 读取模型

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        psnr_avg, ssim_avg = 0, 0
        total_test_set = 1

        start_time = time.time()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                x = x.unsqueeze(1).float().to(self.device)*100
                y = y.unsqueeze(1).float().to(self.device)# 2020 8 27

                pred = self.FBP_Net(x)*100
                pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
                # loss1 = self.criterion1(pred, y)
                # loss2 = self.criterion2(pred, y)
                # loss = 0.8 * loss1 + 0.2 * loss2

                loss = self.criterion(pred, y)
                print(loss.item())

                # print(torch.min(pred))
                # print(torch.min(y))
                """
                calculate psnr ssim
                """
                # print(pred.shape)
                # print(pred.shape)
                # print(torch.min(pred))
                # print(torch.max(pred))
                # print(torch.min(y))
                # print(torch.max(y))
                psnr_l = psnr.psnr(pred,y)
                psnr_avg += psnr_l
                print(psnr_l)
                ssim_l = ssim.ssim(pred,y)
                ssim_avg += ssim_l
                print(ssim_l)
                total_test_set +=1

                ######################
                res = pred.to('cpu').numpy()
                res = res[0, 0, :, :]
                res = (res - res.min()) / (res.max() - res.min())

                res_y = y.to('cpu').numpy()
                res_y = res_y[0, 0, :, :]
                res_y = (res_y - res_y.min()) / (res_y.max() - res_y.min())

                # 构造图像文件名
                f_name_res = '{}_{}.png'.format(self.args.test_patient, i)
                f_name_y   = '{}_{}_y.png'.format(self.args.test_patient, i)

                # 存储当前图像
                img = Image.fromarray(np.uint16(res * 65535))
                img.save(os.path.join(self.args.save_path,'fig', f_name_res))

                img = Image.fromarray(np.uint16(res_y * 65535))
                img.save(os.path.join(self.args.save_path, 'fig', f_name_y))
        print("#############################")
        print(total_test_set)
        print((psnr_avg/total_test_set).item())
        print((ssim_avg/total_test_set).item())
        print("#############################")
        print(time.time() - start_time)