from random import randint

import numpy as np
from PIL import Image

from gprMax.gprMax import api
from tools.outputfiles_merge import merge_files, get_output_data
import pyvista

import matplotlib.pyplot as plt

def randrange_float(start, stop, step):
    return randint(0, int((stop - start) / step)) * step + start


class Cylinder:
    def __init__(self, radius, thickness, x_cord, y_cord, depth, material):
        self.radius = radius
        self.inner_radius = round(self.radius - thickness, 3)
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.depth = depth
        self.material = material

    def __str__(self):
        return '#cylinder: {x_cord} {y_cord} 0 {x_cord} {y_cord} {depth} {radius} {material}\n#cylinder: {x_cord} {y_cord} 0 {x_cord} {y_cord} {depth} {inner_radius} free_space\n'.format(
            x_cord=self.x_cord, y_cord=self.y_cord, radius=self.radius, inner_radius=self.inner_radius,
            depth=self.depth, material=self.material)


def get_cylinders(width, height, boundary, max_cylinders, max_radius, min_radius, d=0.001, material='pvc'):
    number_of_cylinders = randint(1, max_cylinders)
    rows = int((width - boundary) // (max_radius*2)) - 1
    cols = int((height - 2 * boundary) // (max_radius*2)) - 1
    cylinders_map = np.zeros((cols, rows))
    np.put(cylinders_map, np.random.choice(range(cols * rows), number_of_cylinders, replace=False), 1)
    cylinders_locations = np.argwhere(cylinders_map == 1)
    cylinders = []
    for i, j in cylinders_locations:
        x_pos = round(boundary + (j + 1) * (max_radius*2), 3)
        y_pos = round(boundary + (i + 1) * (max_radius*2), 3)
        radius = round(randrange_float(min_radius, max_radius, d), 3)
        cylinders.append(str(Cylinder(radius, d * 5, x_pos, y_pos, d, material)))
    return cylinders


in_file_content = """
#title: Generator of Bscan models for AI
#domain: 0.7 0.4 0.001
#dx_dy_dz: 0.001 0.001 0.001
#time_window: 9e-9

#material: 4 0 1 0 pvc
#material: 6 0 1 0 half_space

#waveform: ricker 1 1.5e9 my_ricker
#hertzian_dipole: z 0.02 0.35 0 my_ricker
#rx: 0.06 0.35 0
#src_steps: 0.001 0 0
#rx_steps: 0.001 0 0

#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.1 my_soil
#fractal_box: 0 0 0 0.7 0.35 0.001 1.5 1 1 1 50 my_soil my_soil_box

{cylinders}

#geometry_view: 0 0 0 0.7 0.4 0.001 0.001 0.001 0.001 in_geometry n
"""
cylinders = ''.join(get_cylinders(0.7, 0.4, 0.01, 5, 0.05, 0.03))

with open('in_file.in', 'w') as in_file:
    in_file.write(in_file_content.format(cylinders=cylinders))

api('in_file.in', geometry_only=True)
# load a vtk file as input
reader = pyvista.read('in_geometry.vti')
shape = np.array([reader.dimensions[1], reader.dimensions[0]]) - 1
material = reader['Material']
in_img = material.reshape(shape)
in_img = Image.fromarray(np.flip(in_img, axis=0))

with open('in_file.in', 'w') as in_file:
    in_file.write(in_file_content.format(cylinders=cylinders).replace('#geometry_view', 'geometry_view'))

api('in_file.in', 620)
merge_files('in_file', removefiles=True)
data, dt = get_output_data('in_file_merged.out', 1, 'Ez')
out_img = Image.fromarray(data)
out_img = out_img.resize(in_img.size)
in_img.convert('RGB').save('in.png', 'PNG')
out_img.convert('RGB').save('out.png', 'PNG')