import os
import torch
import numpy as np
import argparse

from matplotlib import pyplot as plt

from PIL import Image
from torchvision import transforms

from src import circular_trajectory
from src import shepp_logan
from src import geometry_fan_2d
from src import filters
from src import weights
from src import SLFilters

import FBP_cuda_lib

class FBPSolver(object):
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader

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

        # Get Phantom:  the data in Phantom--> saves the No.1 row data at first, then the second row
        phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
        # Add required batch dimension
        phantom = np.expand_dims(phantom, axis=0)

        ################################################################
        device = torch.device('cuda')

        ##1. ##############################
        central_ray_vectors = np.broadcast_to(geometry.central_ray_vectors,
                                             [args.batch_size, *np.shape(geometry.central_ray_vectors)])

        ##2. ##############################
        # 1)
        phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())
        input = torch.from_numpy(phantom).to(device)

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

        ##1. ##############################
        FBP_cuda_lib.FBP_Allocate(central_ray_vectors, input_int, input_float)

    def __del__(self):
        FBP_cuda_lib.FBP_Free()


    def test(self):
        ###
        device = torch.device('cuda')
        sl_filter = SLFilters.SLFilter(self.args.col_count, self.args.col_spacing)

        ###
        for i, (x, y) in enumerate(self.data_loader):
            sing = np.squeeze(x, 0).numpy()

            sinogram_fitled = SLFilters.SLFiltering(sing, sl_filter)

            input = np.zeros([self.args.batch_size, self.args.channel, self.args.proj_count, self.args.det_count], dtype=float)
            input[0, 0, :, :] = sinogram_fitled
            input = torch.from_numpy(input).to(device)
            input = input.float()

            output = torch.zeros(self.args.batch_size, self.args.channel, self.args.row_count, self.args.col_count, dtype=torch.float).to(device)

            FBP_cuda_lib.FBP_2D_Atw(input, output)

            y = y.unsqueeze(1).float()

            ######################
            res = output.to('cpu').numpy()
            res = res[0, 0, :, :]
            res = (res - res.min()) / (res.max() - res.min())

            res_y = y
            res_y = res_y[0, 0, :, :]
            res_y = (res_y - res_y.min()) / (res_y.max() - res_y.min())

            # 构造图像文件名
            f_name_res = '{}_{}.png'.format(self.args.test_patient, i)
            f_name_y   = '{}_{}_y.png'.format(self.args.test_patient, i)

            # 存储当前图像
            img = Image.fromarray(np.uint16(res * 65535))
            img.save(os.path.join(self.args.save_path, f_name_res))

            img = Image.fromarray(np.uint16(res_y * 65535))
            img.save(os.path.join(self.args.save_path, f_name_y))

