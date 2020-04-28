import argparse
import h5py
import numpy as np
import vtk
from PIL import Image
from gprMax.exceptions import CmdInputError
from tools.outputfiles_merge import get_output_data
from vtk.util.numpy_support import numpy_to_vtk
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Creates a .vti file from Bscan',
                                     usage='cd gprMax; python -m tools.plot_Bscan outputfile output')
    # parser.add_argument('outputfile', help='name of output file including path')
    # parser.add_argument('rx_component', help='name of output component to be plotted', choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    # parser.add_argument('side_length', help='length of the side, to split')
    # parser.add_argument('side_height', help='length of the side, to split')
    # parser.add_argument('threshold', help='val bigger than max * threshold and less than min * threshold set to 0')
    args = parser.parse_args()
    # Open output file and read number of outputs (receivers)
    args.outputfile = '../experements/3D_research/3d_plotting/cross_pipes_merged.out'
    args.rx_component = 'Ez'
    args.side_length = '21'
    args.side_height = '60'
    args.threshold = '0.2'
    f = h5py.File(args.outputfile, 'r')
    nrx = f.attrs['nrx']
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(args.outputfile))

    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        data = []
        side_l = int(args.side_length)
        side_h = int(args.side_height)
        threshold = float(args.threshold)
        threshold = threshold if threshold > 1 else threshold
        min_v = np.min(outputdata) * threshold
        max_v = np.max(outputdata) * threshold
        outputdata[outputdata > max_v] = 0
        outputdata[outputdata < min_v] = 0
        outputdata = outputdata * (1 / threshold)
        for i in range(outputdata.shape[1] // side_l):
            side = outputdata[:, side_l * i: side_l * (i + 1)]
            img = Image.fromarray(side)
            img = img.resize([side_l, side_h])
            side = np.array(img)
            plt.imshow(side)
            plt.show()
            data.append(side)
        del outputdata
        data = np.stack(data)
        shape = data.shape
        data = numpy_to_vtk(data.flatten(order='F'), deep=False, array_type=vtk.VTK_FLOAT)
        imgPtsVTP = vtk.vtkImageData()
        imgPtsVTP.SetDimensions(*shape)
        imgPtsVTP.GetPointData().SetScalars(data)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(args.outputfile.split('.')[0] + '_out.vti')
        writer.SetInputData(imgPtsVTP)
        writer.Write()
