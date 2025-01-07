import numpy
import struct
import os

def ReadSlice(filepath):
    # 1.
    binfile = open(filepath, 'rb')
    size = os.path.getsize(filepath)

    # 2.
    data_i = binfile.read(4)
    Rows = numpy.array(struct.unpack('i', data_i), dtype=numpy.int32)
    data_i = binfile.read(4)
    Columns = numpy.array(struct.unpack('i', data_i), dtype=numpy.int32)

    # 3.
    Slice = numpy.zeros((Rows[0], Columns[0]), dtype = numpy.float64)   # numpy np.float default refers to float64
    for i in range(Rows[0]):
        for j in range(Columns[0]):
            data_i = binfile.read(8)
            temp = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)
            Slice[i, j] = temp[0]
    # 4.
    binfile.close()

    # 5. Normalize
    Slice = (Slice - Slice.min()) / (Slice.max() - Slice.min())

    return Slice, Columns[0], Rows[0]


def ReadProjection(filepath):
    # 1.
    binfile = open(filepath, 'rb')
    size = os.path.getsize(filepath)

    # 2.
    data_i = binfile.read(8)
    R = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)
    data_i = binfile.read(8)
    D = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)
    data_i = binfile.read(8)
    r = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)

    data_i = binfile.read(4)
    Rows = numpy.array(struct.unpack('i', data_i), dtype=numpy.int)
    data_i = binfile.read(4)
    Columns = numpy.array(struct.unpack('i', data_i), dtype=numpy.int)

    data_i = binfile.read(8)
    Columns_spacing = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)
    data_i = binfile.read(8)
    Rows_spacing = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)

    data_i = binfile.read(8)
    Phi_d = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)
    data_i = binfile.read(8)
    Phi_start = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)

    # 3.
    Projection = numpy.zeros((Rows[0], Columns[0]), dtype = numpy.float64)
    for i in range(Rows[0]):
        for j in range(Columns[0]):
            data_i = binfile.read(8)
            temp = numpy.array(struct.unpack('d', data_i), dtype=numpy.float)
            Projection[i, j] = temp[0]
    # 4.
    binfile.close()

    # 5.
    Projection = (Projection - Projection.min())/(Projection.max() - Projection.min())

    return R[0], D[0], r[0], Columns[0], Rows[0], Columns_spacing[0], Rows_spacing[0], Phi_d[0], Phi_start[0], Projection

def ReadSliceAll(filepath):
    # 1.
    binfile = open(filepath, 'rb')
    size = os.path.getsize(filepath)

    # 2.
    data_i = binfile.read(4)
    Columns = numpy.array(struct.unpack('i', data_i), dtype=numpy.int)
    data_i = binfile.read(4)
    Rows = numpy.array(struct.unpack('i', data_i), dtype=numpy.int)

    data_i = binfile.read(8)
    Columns_spacing = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)
    data_i = binfile.read(8)
    Rows_spacing = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)

    data_i = binfile.read(8)
    SliceLocation = numpy.array(struct.unpack('d', data_i), dtype=numpy.float64)

    # 3.
    Slice = numpy.zeros((Rows[0], Columns[0]), dtype = numpy.float64)
    for i in range(Rows[0]):
        for j in range(Columns[0]):
            data_i = binfile.read(8)
            temp = numpy.array(struct.unpack('d', data_i), dtype=numpy.float)
            Slice[i, j] = temp[0]
    # 4.
    binfile.close()

    # 5.
    Slice = (Slice - Slice.min())/(Slice.max() - Slice.min())

    return Columns[0], Rows[0], Columns_spacing[0], Rows_spacing[0], SliceLocation[0], Slice

