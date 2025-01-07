"""
This program is design for make the dataset
use FBP for create sinogram and patch image
"""

import os
import torch
import numpy as np
import argparse

# save png image
from PIL import Image
# read bin
from src import Reader
from src import readNPY
# FBP
from src import circular_trajectory
from src import geometry_fan_2d
from src import filters
from src import weights
from src import SLFilters

# cal At A A->ray-dirven At->pixel-driven
import FanBeam_BP_cuda_lib

# import for debug disp
import matplotlib.pyplot as plt

def get_div1(img):
    [u_x, u_y] = np.gradient(img)
    [u_xx, u_xy] = np.gradient(u_x)
    [u_yx, u_yy] = np.gradient(u_y)
    img_div = u_xx + u_yy
    return img_div

## Add noise
##***********************************************************************************************************
def add_noise(noise_typ, image, mean=0, var=0.1):
    if noise_typ == "gauss":
        row, col = image.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        noisy = np.random.poisson(image)
        return noisy


## Add noise Astra toolbox
##***********************************************************************************************************
def add_noise_to_sino(sinogram_in, I0, var=0.1, mean=0, seed=None):
    """Adds Poisson noise to a sinogram.

    :param sinogram_in: Sinogram to add noise to.
    :type sinogram_in: :class:`numpy.ndarray`
    :param I0: Background intensity. Lower values lead to higher noise.
    :type I0: :class:`float`
    :returns:  :class:`numpy.ndarray` -- the sinogram with added noise.

    """

    if not seed == None:
        curstate = np.random.get_state()
        np.random.seed(seed)

    if isinstance(sinogram_in, np.ndarray):
        sinogramRaw = sinogram_in
    else:
        # sinogramRaw = data2d.get(sinogram_in)
        print("Error!")

    ## Normal
    row, col = sinogramRaw.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)

    ## Poisson
    max_sinogramRaw = sinogramRaw.max()
    sinogramRawScaled = sinogramRaw / max_sinogramRaw
    # to detector count
    sinogramCT = I0 * np.exp(-sinogramRawScaled)
    # add poison noise
    sinogramCT_C = np.zeros_like(sinogramCT)
    for i in range(sinogramCT_C.shape[0]):
        for j in range(sinogramCT_C.shape[1]):
            sinogramCT_C[i, j] = np.random.poisson(sinogramCT[i, j])
    # to density
    sinogramCT_D = (sinogramCT_C + gauss) / I0
    sinogram_out = -max_sinogramRaw * np.log(sinogramCT_D)

    if not seed == None:
        np.random.set_state(curstate)
        print('hihi')

    return sinogram_out


def normalization_np(input):
    return (input - input.min()) / (input.max() - input.min())


def normalization_tensor(input):
    return (input - torch.min(input)) / (torch.max(input) - torch.min(input))


def FBP_Forward(args, spath):
    device = torch.device('cuda')
    # 1.
    phantom, icol, irow = Reader.ReadSlice(spath)
    # phantom, icol, irow = readNPY.NPYReader(spath)
    # normalization
    phantom = normalization_np(phantom)

    phantom_np = phantom
    phantom_gpu = torch.from_numpy(phantom).unsqueeze(0).to(device)

    # 1-1)
    input = torch.zeros(args.batch_size, args.channel, args.row_count, args.col_count, dtype=torch.float).to(device)

    for i in range(args.batch_size):
        input[i, 0, :, :] = phantom_gpu[0, :, :]

    # 2)
    output = torch.zeros(args.batch_size, args.channel, args.proj_count, args.det_count, dtype=torch.float).to(device)

    # 3) radon transform
    FanBeam_BP_cuda_lib.FBP_2D_Aw(input, output)

    ####################################################################################################
    # sinogram to cpu
    sinogram_np = output.squeeze().squeeze().to("cpu").numpy()
    if args.is_add_noise == True:
        # sino_add_noise
        sino_add_noise_np = add_noise_to_sino(sinogram_np, args.dose_level, args.noise_level)
        # create Tensor
        sino_add_noise_tensor = torch.from_numpy(sino_add_noise_np)
        sinogram_noise_gpu = sino_add_noise_tensor.unsqueeze(0).unsqueeze(0).float().to(torch.device('cuda'))

        # Normalize
        sinogram_noise_norm_gpu = normalization_tensor(sinogram_noise_gpu)

        return sinogram_noise_norm_gpu, phantom_np
    ####################################################################################################
    # Normalize
    sinogram_noise_norm_gpu = normalization_tensor(output)

    # debug plot
    ########################################
    '''
    input = input.to("cpu").numpy()
    for i in range(args.batch_size):
        res = input[i, 0, :, :]
        plt.imshow(res, interpolation='nearest', cmap='gray')
        plt.show()

    ################

    output = output.to("cpu").numpy()
    for i in range(args.batch_size):
        res = output[i, 0, :, :]
        plt.imshow(res, interpolation='nearest', cmap = 'gray')
        plt.show()
    '''
    ########################################

    return sinogram_noise_norm_gpu, phantom_np


# filter-backprojection
def FBP_Backward(args, geometry, sinogram):
    device = torch.device('cuda')
    input = np.zeros([args.batch_size, args.channel, args.proj_count, args.det_count], dtype=float)
    sinogram_np = sinogram.to('cpu').numpy()

    # generate weight fanbeam & Sheep-Logan filter
    redundancy_weights = weights.parker_weights_2d(geometry)
    sl_filter_np = SLFilters.SLFilter(args.col_count, args.col_spacing)

    sinogram_weighted_np = np.zeros([args.proj_count, args.det_count], dtype=float)

    for i in range(args.batch_size):
        sinogram_weighted_np = sinogram_np[i, 0, :, :] * redundancy_weights
        # scale data to 0-1
        sinogram_weighted_np = normalization_np(sinogram_weighted_np)
        sinogram_weighted_save_np = sinogram_weighted_np.copy()
        # sinogram_fft = (fft(sing))
        # sinogram_fitled = np.real(ifft(np.multiply(sinogram_fft, reco_filter)))
        sinogram_fitled = SLFilters.SLFiltering(sinogram_weighted_np, sl_filter_np)
        input[i, 0, :, :] = sinogram_fitled

    input = torch.from_numpy(input).to(device)
    input = input.float()
    output = torch.zeros(args.batch_size, args.channel, args.row_count, args.col_count, dtype=torch.float).to(device)

    FanBeam_BP_cuda_lib.FBP_2D_Atw(input, output)
    phantom_recon = output
    phantom_recon = normalization_tensor(phantom_recon)
    # debug plot
    ########################################
    '''
    output = output.to("cpu").numpy()

    for i in range(args.batch_size):
        res = output[i, 0, :, :]
        res = (res - res.min()) / (res.max() - res.min())
        plt.imshow(res, interpolation='nearest', cmap='gray')
        plt.show()
    '''
    ########################################
    return phantom_recon, sinogram_weighted_save_np


# Init FBP layer
def Init(args):
    volume_shape = [args.row_count, args.col_count]  # 0: row, 1: col
    volume_spacing = [args.row_spacing, args.col_spacing]

    # Detector Parameters:
    detector_shape = args.det_count
    detector_spacing = args.det_spacing

    # Trajectory Parameters:
    number_of_projections = args.proj_count
    angular_range = args.proj_range
    proj_range_start = args.proj_range_start
    proj_range_end = args.proj_range_end

    source_detector_distance = args.source_detector
    source_isocenter_distance = args.source_isocenter

    # create Geometry class
    geometry = geometry_fan_2d.GeometryFan2D(volume_shape, volume_spacing, detector_shape, detector_spacing,
                                             number_of_projections,
                                             proj_range_start, proj_range_end, source_detector_distance,
                                             source_isocenter_distance)

    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    ################################################################
    device = torch.device('cuda')

    # 1)
    central_ray_vectors = np.broadcast_to(geometry.central_ray_vectors,
                                          [args.batch_size, *np.shape(geometry.central_ray_vectors)])

    # 2)
    output = torch.zeros(args.batch_size, args.proj_count, args.det_count, dtype=torch.float).to(device)

    # 3)
    central_ray_vectors = torch.from_numpy(central_ray_vectors.copy()).to(device)

    # 4)
    input_int = torch.zeros(1, 4, dtype=torch.int)
    input_int[0, 0] = args.proj_count
    input_int[0, 1] = args.col_count  # col
    input_int[0, 2] = args.row_count  # row
    input_int[0, 3] = args.det_count

    # 5)
    input_float = torch.zeros(1, 8, dtype=torch.float)
    input_float[0, 0] = args.col_spacing
    input_float[0, 1] = args.row_spacing
    input_float[0, 2] = geometry.volume_origin[1]  # col
    input_float[0, 3] = geometry.volume_origin[0]  # row
    input_float[0, 4] = args.det_spacing
    input_float[0, 5] = geometry.detector_origin[0]
    input_float[0, 6] = geometry.source_isocenter_distance
    input_float[0, 7] = geometry.source_detector_distance

    return geometry, central_ray_vectors, input_int, input_float


def example_fan_2d(args):
    #########################
    geometry, central_ray_vectors, input_int, input_float = Init(args)
    FanBeam_BP_cuda_lib.FBP_Allocate(central_ray_vectors, input_int, input_float)
    #########################
    # 创建存储路径
    if not os.path.exists(args.save_dataset_path):
        os.makedirs(args.save_dataset_path)
        print('Create path : {}'.format(args.save_dataset_path))

    # 获取dataset文件夹下所有子文件夹名 不包含zip文件 子文件以病人名字命名
    patients_list = sorted([d for d in os.listdir(args.data_path) if 'zip' not in d])
    # 处理每一个病人文件夹下的数据
    for p_ind, patient in enumerate(patients_list):
        # 构造子文件夹的完整路径
        patient_input_path = os.path.join(args.data_path, patient)

        # 遍历每个子文件
        pi = 0
        for spath in os.listdir(patient_input_path):
            # 批量读取子文件夹下的所有文件
            sinogram, phantom_origin_np = FBP_Forward(args, os.path.join(patient_input_path, spath))

            # # sino_fbp = sinogram.unsqueeze()
            # recon_img, sinogram_weighted_np = FBP_Backward(args, geometry, sinogram)

            # 构造图像文件名 npy
            f_name_src = '{}_{}_{}.npy'.format(patient, pi, '360_fulldose_noisefree_sinoinput')
            # f_name_fbp = '{}_{}_{}.npy'.format(patient, pi, '360_fulldose_fbpinput')
            f_name_dst = '{}_{}_{}.npy'.format(patient, pi, 'imgtarget')
            # f_name_div = '{}_{}_{}.npy'.format(patient, pi, 'edge')

            # recon_img_npy = recon_img.to('cpu').numpy()[0, 0, :, :]
            sinogram_np = sinogram.to('cpu').numpy()[0, 0, :, :]
            # img_div = get_div1(phantom_origin_np)

            np.save(os.path.join(args.save_dataset_path, f_name_src), sinogram_np)
            # np.save(os.path.join(args.save_dataset_path, f_name_fbp), recon_img_npy)
            np.save(os.path.join(args.save_dataset_path, f_name_dst), phantom_origin_np)
            # np.save(os.path.join(args.save_dataset_path, f_name_div), img_div)
            # img = Image.fromarray(np.uint16(phantom_origin_np * 65535))
            # img.save(os.path.join(args.save_dataset_path, f_name_dst))

            # plt.imshow(recon_img_npy, interpolation='nearest', cmap='gray')
            # plt.show()
            # #
            # plt.imshow(phantom_origin_np, interpolation='nearest', cmap='gray')
            # plt.show()
            print(patient, '_', pi)
            pi += 1

    FanBeam_BP_cuda_lib.FBP_Free()
    print("Success!")


def Process_DataSet_Img2Img(proj_cout, is_ND, dose_level, is_noise_free, noise_level):
    # 判断用于生成存储路径
    save_data_path = './DataSet_Img2Img'
    if is_ND == True and is_noise_free == True:
        if dose_level == 1000000:
            save_data_path += '/normal-dose/noise-free/'
        else:
            print("dose_level must mach!")
            return

    if is_ND == True and is_noise_free == False:
        if dose_level == 1000000:
            save_data_path += '/normal-dose/noise/'
        else:
            print("dose_level must mach!")
            return
    if is_ND == False and is_noise_free == True:
        if dose_level == 1000000:
            print("dose_level must mach!")
            return
        else:
            tmp = '/low-dose/' + str(dose_level) + '/noise-free/'
            save_data_path += tmp

    if is_ND == False and is_noise_free == False:
        if dose_level == 1000000:
            print("dose_level must mach!")
            return
        else:
            tmp = '/low-dose/' + str(dose_level) + '/noise/'
            save_data_path += tmp

    # 总的存储路径
    save_data_path = save_data_path + str(proj_cout) + '/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./DataSet_bin_all/')
    parser.add_argument('--save_dataset_path', type=str, default=save_data_path)

    parser.add_argument('--is_ND_noise', type=bool, default=is_ND)
    parser.add_argument('--dose_level', type=int, default=dose_level)

    parser.add_argument('--is_add_noise', type=bool, default=(not is_noise_free))
    parser.add_argument('--noise_level', type=float, default=noise_level)

    parser.add_argument('--row_count', type=int, default=512)
    parser.add_argument('--col_count', type=int, default=512)
    parser.add_argument('--row_spacing', type=float, default=0.5859)
    parser.add_argument('--col_spacing', type=float, default=0.5859)

    parser.add_argument('--det_count', type=int, default=768)  # the gost from the less detector count
    parser.add_argument('--det_spacing', type=float, default=1.0)

    parser.add_argument('--proj_count', type=int, default=proj_cout)  # the gost from the less projection count
    parser.add_argument('--proj_range', type=float, default=2 * np.pi)
    parser.add_argument('--proj_range_start', type=float, default=0.0 * np.pi)
    parser.add_argument('--proj_range_end', type=float, default=2.0 * np.pi)

    parser.add_argument('--source_isocenter', type=float, default=595.0)
    parser.add_argument('--source_detector', type=float, default=1068.0)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--channel', type=int, default=1)

    args = parser.parse_args()
    example_fan_2d(args)


if __name__ == '__main__':
    # Process_DataSet_Img2Img(proj_cout=360, is_ND=True, dose_level=1000000, is_noise_free=False, noise_level=10)
    #Process_DataSet_Img2Img(proj_cout=1024, is_ND=False, dose_level=1000000/4, is_noise_free=False, noise_level=10)
    #Process_DataSet_Img2Img(proj_cout=360, is_ND=True, dose_level=1000000, is_noise_free=True, noise_level=0)
    Process_DataSet_Img2Img(proj_cout=360, is_ND=True, dose_level=1000000, is_noise_free=True, noise_level=0)

