import torch
import numpy as np
import FBP_cuda_lib

from src import circular_trajectory
from src import geometry_fan_2d

from matplotlib import pyplot as plt

global col_count
col_count  = 50
global row_count
row_count  = 50
global proj_count
proj_count = 36
global det_count
det_count  = 80

global ibatch
ibatch = 1
global ichannel
ichannel = 1


class FanBeam2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        global col_count
        global row_count
        global ibatch
        global ichannel
        output = torch.zeros(ibatch, ichannel, row_count, col_count, dtype=torch.float).to("cuda")
        FBP_cuda_lib.FBP_2D_Atw(input, output)
        #ctx.save_for_backward(central_ray_vectors, input_int, input_float)

        return output

    @staticmethod
    def backward(ctx, grad_):
        #central_ray_vectors, input_int, input_float = ctx.saved_tensors
        global proj_count
        global det_count
        global ibatch
        global ichannel
        output = torch.zeros(ibatch, ichannel, proj_count, det_count, dtype=torch.float).to("cuda")
        FBP_cuda_lib.FBP_2D_Aw(grad_, output)

        return output

class FBP_Layer(torch.nn.Module):

    def __init__(self, args):
        super(FBP_Layer, self).__init__()
        #
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
                                                 proj_range_start, proj_range_end, source_detector_distance, source_isocenter_distance)

        geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

        # 2.1
        device = torch.device('cuda')
        central_ray_vectors = np.broadcast_to(geometry.central_ray_vectors,
                                              [1, *np.shape(geometry.central_ray_vectors)])
        self.central_ray_vectors = torch.from_numpy(central_ray_vectors.copy()).to(device)

        # 2.2
        input_int = torch.zeros(1, 4, dtype=torch.int)
        input_int[0, 0] = args.proj_count
        input_int[0, 1] = args.col_count  # col
        input_int[0, 2] = args.row_count  # row
        input_int[0, 3] = args.det_count
        self.input_int = input_int

        # 2.3
        input_float = torch.zeros(1, 8, dtype=torch.float)
        input_float[0, 0] = args.col_spacing
        input_float[0, 1] = args.row_spacing
        input_float[0, 2] = geometry.volume_origin[1]  # col
        input_float[0, 3] = geometry.volume_origin[0]  # row
        input_float[0, 4] = args.det_spacing
        input_float[0, 5] = geometry.detector_origin[0]
        input_float[0, 6] = geometry.source_isocenter_distance
        input_float[0, 7] = geometry.source_detector_distance
        self.input_float = input_float

        global col_count
        col_count  = input_int[0, 1]
        global row_count
        row_count  = input_int[0, 2]
        global proj_count
        proj_count = input_int[0, 0]
        global det_count
        det_count  = input_int[0, 3]

        global ibatch
        ibatch = args.batch_size
        global ichannel
        ichannel = args.channel

        # 3
        FBP_cuda_lib.FBP_Allocate(self.central_ray_vectors, self.input_int, self.input_float)

    def __del__(self):
        FBP_cuda_lib.FBP_Free()

    def forward(self, input):
        return FanBeam2D.apply(input)