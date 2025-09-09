import os
import pickle
import time

import scipy.io as sio
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import numpy.linalg as npl

import datasets
import plots
import utils
import datetime

from model import AutoEncoder


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False, data_print=False, index=0):
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.print = data_print
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            self.patch, self.dim = 1, 32
            if data_print:
                self.LR, self.EPOCH = 2e-3, 200
                self.para_re, self.para_sad = 0.1, 1.0

            self.weight_decay_param = 4e-5
            self.batch = 1
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=(self.col ** 2))
            self.init_weight = self.data.get("init_weight").float()
            self.init_bundle = self.data.get("bundle").float()
        else:
            raise ValueError("Unknown dataset")

    def run(self, smry):
        bundles = self.init_bundle.to(self.device)
        num_bundle = bundles.size(2)
        net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                          patch=self.patch, dim=self.dim, num_bundle=num_bundle, init_bundle=bundles).to(self.device)
        if smry:
            summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        net.apply(net.weights_init)

        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = utils.SAD(self.L)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        if not self.skip_train:
            time_start = time.time()
            net.train()
            epo_vs_los = []
            for epoch in range(self.EPOCH):

                for i, (x, _) in enumerate(self.loader):
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col).to(self.device)
                    # B*L*H*W
                    edm_vca = self.init_weight.to(self.device)

                    abu_est, re_result, edm_est = net(x, edm_vca)
                    # N*P B*N*L N*L*P
                    loss_re = self.para_re * loss_func(re_result, x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = loss_func2(re_result, x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = self.para_sad * torch.mean(loss_sad).float()

                    total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    if epoch % 10 == 0 and self.print:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)
                    epo_vs_los.append(float(total_loss.data))

                scheduler.step()
            time_end = time.time()

            if self.save:
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})

            if self.print:
                print('Total computational cost:', time_end - time_start)

        else:
            with open(self.save_dir + 'weights_new.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # Testing ================

        net.eval()
        with torch.no_grad():
            x = self.data.get("hs_img")
            x = x.transpose(1, 0).view(1, -1, self.col, self.col).to(self.device)
            # B*L*H*W
            edm_vca = self.init_weight.to(self.device)
            abu_est, re_result, edm_est = net(x, edm_vca)
            # N*P B*N*L N*L*P

        abu_est = abu_est.view(self.col, -1, self.P).detach().cpu().numpy()
        # (col,col,P)
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        # L*P

        edm_est = edm_est.detach().cpu().numpy()
        est_endmem = edm_est
        # N*L*P
        est_endmem = np.mean(est_endmem, axis=0)

        set_edm = true_endmem.transpose()
        est_edm = est_endmem.transpose()
        ref_list = np.arange(0, set_edm.shape[0])
        est_list = np.arange(0, est_edm.shape[0])
        ref2est_table = np.zeros(set_edm.shape[0])

        '''Find the match relationship'''
        for i in range(set_edm.shape[0]):
            best_dis = np.ones((len(ref_list), len(est_list))) * 100.
            ref = set_edm[ref_list].copy()
            est = est_edm[est_list].copy()
            for j in range(len(ref_list)):
                dis = np.arccos(np.dot(ref[j], est.T) / (npl.norm(ref[j]) * npl.norm(est, axis=1) + 1e-6))
                best_dis[j][dis < best_dis[j]] = dis[dis < best_dis[j]]
            best_match = np.argwhere(best_dis == best_dis.min())[0]
            ref_absolute_index = ref_list[best_match[0]]
            est_absolute_index = est_list[best_match[1]]
            ref2est_table[ref_absolute_index] = est_absolute_index
            ref_list = np.delete(ref_list, best_match[0])  # modify the list
            est_list = np.delete(est_list, best_match[1])

        abu_est = abu_est[:, :, ref2est_table.astype(int)]
        est_endmem = est_endmem[:, ref2est_table.astype(int)]
        # L*P
        edm_est = edm_est[:, :, ref2est_table.astype(int)]
        # N*L*P

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": edm_est})

        x = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        re_result = re_result.view(self.col, self.col, -1).detach().cpu().numpy()

        re = utils.compute_re(x, re_result)
        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)

        if self.print:
            print("RE:", re)

            print("Class-wise RMSE value:")
            for i in range(self.P):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)

            print("Class-wise SAD value:")
            for i in range(self.P):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean SAD:", mean_sad)

            plots.plot_abundance(target, abu_est, self.P, self.save_dir)
            plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)

        with open(self.save_dir + "log.csv", 'a') as file:
            file.write(f"LR: {self.LR}, ")
            file.write(f"EPOCH: {self.EPOCH}, ")
            file.write(f"Batch: {self.batch}, ")
            file.write(f"para_re: {self.para_re}, ")
            file.write(f"para_sad: {self.para_sad}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            for i in range(self.P):
                file.write(f"Class{i}_sad: {sad_cls[i]:.4f}, ")

            file.write(f"RMSE: {mean_rmse:.4f}, ")
            for i in range(self.P):
                file.write(f"Class{i}_mse: {rmse_cls[i]:.4f}, ")

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"TIME:{current_time}\n")


# =================================================================

if __name__ == '__main__':
    pass
