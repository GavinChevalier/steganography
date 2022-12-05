from __future__ import division
from __future__ import print_function

import torch
import os
from PIL import Image
import numpy as np
import time
from mapping import *
from demapping import *
from training_utils import *
from etc import config

nz = config["size_of_z_latent"]


def get_trained_model():
    netG = Generator()
    g_state_dict = torch.load(r'\generator.pt')
    netG.load_state_dict(g_state_dict)
    return netG.cuda()


def main(nz, lambda_, order):
    ORDER = order
    if ORDER:
        root_ = r'\sequence_mapping_rule\nz%s_lambda_%s' % (nz, lambda_)
    else:
        root_ = r'\gray_code_mapping_rule\nz%s_lambda_%s' % (nz, lambda_)

    if not os.path.exists(root_):
        os.makedirs(root_)

    file_num = 10
    time_threshold = 500

    def make_s_o(file_index):
        root = root_ + '/%03d' % file_index
        if not os.path.exists(root):
            os.makedirs(root)
        s_o = np.random.randint(low=0, high=2, size=nz * lambda_)
        f = open(root + '/s_o.txt', 'w')
        for num in s_o:
            f.write(str(num))
        f.close()

    def load_s_o(file_index):
        root = root_ + '/%03d' % file_index
        f = open(root + '/s_o.txt')
        s_o = f.readline()
        f.close()
        return s_o

    def Secret_Maker():
        for i in range(file_num):
            make_s_o(file_index=i)

    def Sender():
        netG = get_trained_model()

        def S(file_index, s_o):
            root = root_ + '/%03d' % file_index

            if ORDER:
                z_s_np = secret_mapping_inorder(secret=s_o, lambda_=lambda_)
            else:
                z_s_np = secret_mapping_v2(secret=s_o, lambda_=lambda_)

            z_s = torch.from_numpy(z_s_np).float().cuda()
            img_s = netG(z_s)

            mmin = -1.0
            mmax = 1.0
            img_s.add_(- mmin).div_(mmax - mmin + 1e-5)
            img_s_np = img_s[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            img_s = Image.fromarray(img_s_np)
            img_s.save(root + '/stego.png')

        for i in range(file_num):
            s_o = load_s_o(file_index=i)
            S(file_index=i, s_o=s_o)


    def Receiver():
        netG = get_trained_model()

        def E(file_index, s_o, try_time):
            root = root_ + '/%03d' % file_index
            img_s = Image.open(root + '/stego.png')
            img_s = np.expand_dims(np.array(img_s).transpose(2, 0, 1), 0)
            img_s = torch.from_numpy(img_s).cuda().float() / 255
            mmin = -1.0
            mmax = 1.0
            img_s = img_s.mul_(mmax - mmin).add_(mmin)

            lr = 0.02
            criterion = torch.nn.MSELoss()
            re_noise = torch.randn(1, nz).cuda().requires_grad_()
            optimizer = torch.optim.Adam([re_noise], lr=lr)  # let optimizer optimize the tensor re_noise

            tt = 0
            start = time.time()

            while True:
                optimizer.zero_grad()
                re_img = netG(re_noise)

                loss = criterion(re_img, img_s)
                loss.backward()
                optimizer.step()

                tt += 1

                if tt % 100 == 0:
                    print('File: {:d}  Try: {:d}  Step: {:d}   Loss: {:.8f}'.format(file_index, try_time, tt // 100,
                                                                                    loss.item()))
                temp_time = int(time.time() - start)

                if temp_time > time_threshold:
                    return time_threshold, -1

                if (loss.item() <= 0.00001500 and temp_time > 240 and try_time >= 3) or \
                        (loss.item() <= 0.00001000 and temp_time > 240 and try_time >= 2) or \
                        (loss.item() <= 0.0000800 and temp_time > 240 and try_time >= 1) or \
                        (loss.item() <= 0.00000700 and temp_time > 200) or \
                        loss.item() <= 0.00000600:

                    end = time.time()
                    miss = 0

                    re_noise_np = re_noise.detach().cpu().numpy()

                    re_noise_min = re_noise_np.min()
                    re_noise_max = re_noise_np.max()
                    for i in range(nz):
                        temp = re_noise_np[0][i]
                        if temp < 0:
                            re_noise_np[0][i] = -temp / re_noise_min
                        else:
                            re_noise_np[0][i] = temp / re_noise_max

                    if ORDER:
                        re_s = de_map_inorder(re_noise_np, lambda_)
                    else:
                        re_s = de_map(re_noise_np, lambda_)

                    for i in range(nz * lambda_):
                        if int(re_s[i]) != int(s_o[i]):
                            miss += 1

                    f = open(root + '/re_s.txt', 'w')
                    for i in range(nz * lambda_):
                        f.write(str(int(re_s[i])))
                    f.close()
                    time_cost = int(end - start)
                    acc = 1 - miss / (nz * lambda_)
                    f = open(root + '/miss=%03d acc=%0.4f time=%d s try=%d.txt' % (
                    miss, acc, time_cost + time_threshold * try_time, try_time), 'w')
                    f.close()
                    print('miss', miss)
                    re_img.add_(- mmin).div_(mmax - mmin + 1e-5)
                    np_fake_data = re_img[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                    torch.uint8).numpy()
                    im = Image.fromarray(np_fake_data)
                    im.save(root + '/re_img.png')
                    break

            return time_cost, acc

        sum_time_cost = 0
        sum_acc = 0
        count = 0
        for i in range(file_num):
            try_time = 0
            s_o = load_s_o(file_index=i)
            time_cost, acc = E(file_index=i, s_o=s_o, try_time=try_time)

            while acc == -1:
                sum_time_cost += time_cost
                try_time += 1
                time_cost, acc = E(file_index=i, s_o=s_o, try_time=try_time)
                if try_time == 4:
                    f = open(root_ + '/0000fail_file_index.txt', 'a')
                    f.write('{:03d}'.format(i) + '\n')
                    f.close()
                    break
            sum_time_cost += time_cost

            if acc != -1:
                sum_acc += acc
                count += 1
        avg_time_cost = sum_time_cost / count
        avg_acc = sum_acc / count
        f = open(root_ + '/0000avg_time_cost=%d s  avg_acc=%0.4f.txt' % (avg_time_cost, avg_acc), 'w')
        f.close()

    Secret_Maker()
    Sender()

    # start = time.time()
    # Receiver()
    # end = time.time()
    # sub = end - start
    # f = open(root_ + '/0000sum_time=%d s  avg_time_cost=%d s.txt' % (sub, sub / file_num), 'w')
    # f.close()


main(nz=nz, lambda_=3, order=False)
