'''
    this layer is designed by leekunpeng@hotmail.com
    Frequency Domain -> Spatial Domain

    Forward : Frequency Domain into Spatial Domain via ifft
    Backward : change Spatial Domain into Frequency Domain via fft


'''

import torch
import torch.fft

'''
linefft:    @input: Spatial data, shape:[batch, channel, proj_nums, det_nums]
            1. fftshift(fft(input))
            2. do normalization
lineifft    @input: Frequency data, shape[batch, channel, proj_nums, det_nums]
            1. ifft(ifftshift(input))
            2. do normalization
'''
def linefft(input):    # input size must be [batch, channel, proj_nums, det_cout]
    length_num_projs = input.shape[2]
    batch_size = input.shape[0]
    output = torch.zeros(input.size(),dtype=torch.complex64)
    for i in range(batch_size):
        for j in range(length_num_projs):  # get a line proj and do fft fftshift
            x = input[i,0,j,:]
            x.cuda()
            line_fft = torch.fft.fftshift(torch.fft.fft(x))
            output[i,0,j,:] = line_fft
    return output



def lineifft(input):    # input size must be [batch, channel, proj_nums, det_cout]
    # print(input.shape)
    length_num_projs = input.shape[2]
    batch_size = input.shape[0]
    output = torch.zeros_like(input,dtype=torch.float).cuda()
    for i in range(batch_size):
        for j in range(length_num_projs):  # 按行fft
            # print(input[i,0,j,:].shape)
            # print(input[i,0,j,:])
            temp = torch.fft.ifft(torch.fft.ifftshift(input[i, 0, j, :]))
            # only save the real part
            temp = torch.real(temp)
            # output[i, 0, j, :] -= torch.min(temp)
            output[i, 0, j, :] = temp
            # return the real part
    # to avoid the negative value
    # output -= torch.min(output)
    # normalization
    output = Normalization(output)
    return output

def Normalization(input):
    output = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
    return output

class Frequency2Spatial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        '''
                    backward part:
                        1. get real part & imaginary part
                        2. use torch.chunk()
                            change shape of frequency input: batch 2 channel proj_nums det_nums -> batch channel proj_nums det_nums
                        3. do ifft

                '''
        # print("start Frequency2Spatial forward")
        # print(input.size())
        temp = torch.chunk(input, 2, dim=1)
        grad_output_complex = torch.complex(temp[0], temp[1]).cuda()

        output = lineifft(grad_output_complex)
        # print(output)
        # print(output.shape)
        # print("forward Frequency2Spatial finished")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
            froward part:
                1. do fft
                2. get real part & imaginary part
                3. use torch.stack()
                    change shape of frequency input: batch channel proj_nums det_nums -> batch 2*channel proj_nums det_nums
        '''

        # print("start Frequency2Spatial backward")
        # print(grad_output.size())

        output = linefft(grad_output)

        fft_real = torch.real(output)
        # print(fft_real.shape)
        fft_image = torch.imag(output)
        # print(fft_image.shape)
        output = torch.cat([fft_real, fft_image], dim=1).cuda()
        # input[batch_size, channels] = proj
        # print(output.shape)
        # print(output)
        # output = torch.real(output)
        # print(output.type())
        # print("backward Frequency2Spatial finished")
        return output


class Frequency2SpatialLayer(torch.nn.Module):

    def __init__(self):
        super(Frequency2SpatialLayer, self).__init__()
        # print("Ferquency2SpatialLayer construct")


    def forward(self, input):
        # print("strat Frequency2Spatial Layer forward")
        return Frequency2Spatial.apply(input)