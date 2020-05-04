import re
from random import randint

import numpy as np
from PIL import Image
import os
from gprMax.gprMax import api
from tools.outputfiles_merge import merge_files, get_output_data
import pyvista


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


def get_cylinders(width, height, boundary, max_cylinders, max_radius, min_radius, d=0.002, material='pvc'):
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
#domain: 1 1 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 9e-9

#material: 4 0 1 0 pvc
#material: 6 0 1 0 half_space

#waveform: ricker 1 1.5e9 my_ricker
#hertzian_dipole: z 0.02 0.95 0 my_ricker
#rx: 0.06 0.95 0
#src_steps: 0.004 0 0
#rx_steps: 0.004 0 0

#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.1 my_soil
#fractal_box: 0 0 0 1 0.95 0.002 1.5 1 1 1 50 my_soil my_soil_box

{cylinders}

#geometry_view: 0 0 0 1 1 0.002 0.002 0.002 0.002 in_geometry n
"""

if not os.path.isdir('x'):
    os.mkdir('x')
if not os.path.isdir('y'):
    os.mkdir('y')

max_x = [int(re.sub('\D', '', filename)) for filename in os.listdir('x')]
max_y = [int(re.sub('\D', '', filename)) for filename in os.listdir('y')]
max_x.append(0)
max_y.append(0)

s = max([max(max_x), max(max_y)]) + 1

for i in range(s, s+10):
    cylinders = ''.join(get_cylinders(1, 0.95, 0.02, 1, 0.05, 0.03))

    with open('in_file.in', 'w') as in_file:
        in_file.write(in_file_content.format(cylinders=cylinders))

    api('in_file.in', geometry_only=True)
    # load a vtk file as input
    reader = pyvista.read('in_geometry.vti')
    shape = np.array([reader.dimensions[1], reader.dimensions[0]]) - 1
    material = reader['Material']
    in_img = material.reshape(shape)
    in_img = np.flip(in_img, axis=0)
    in_img = in_img + abs(np.min(in_img))
    in_img = in_img / np.max(in_img) * 255
    in_img = Image.fromarray(in_img)

    with open('in_file.in', 'w') as in_file:
        in_file.write(in_file_content.format(cylinders=cylinders).replace('#geometry_view', 'geometry_view'))

    api('in_file.in', 155, gpu=[0])
    merge_files('in_file', removefiles=True)
    data, dt = get_output_data('in_file_merged.out', 1, 'Ez')
    data = data + abs(np.min(data))
    data = data / np.max(data) * 255
    out_img = Image.fromarray(data)
    out_img = out_img.resize(in_img.size)
    in_img.convert('RGB').save('x/in_{}.png'.format(i), 'PNG')
    out_img.convert('RGB').save('y/out_{}.png'.format(i), 'PNG')
