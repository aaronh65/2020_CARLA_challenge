import os
import numpy as np
import cv2
import time
np.set_printoptions(suppress=True, precision=6)
from polyfit import approximate
import pickle as pkl
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--bev', action='store_true')
parser.add_argument('--route', type=int)
parser.add_argument('--fit', action='store_true')
args = parser.parse_args()

# gps 
mean = np.array([49.0, 8.0])
scale = np.array([111324.60662786, 73032.1570362])
size = 256
c = size/2
r = 2

fig = plt.gcf()
ax = plt.gca()
def format_ticks(value, tick_number):
    minute = int(value/60)
    return f'{minute:02d}:00'

# get route paths

for split in sorted(os.listdir(args.target_dir)):
    math_path = os.path.join(args.target_dir, split, 'math')
    if not os.path.exists(math_path):
        continue
    if args.route:
        route_name = f'route_{args.route:02d}'
        rpaths = [f'{math_path}/{route}' for route in os.listdir(math_path) if route == route_name]
    else:
        rpaths = [f'{math_path}/{route}' for route in os.listdir(math_path) if 'route' in route]
    rpaths = sorted(rpaths)
    all_errors = [None] * len(rpaths)
    for rpath in rpaths: # per route
        repaths = [f'{rpath}/{rep}' for rep in sorted(os.listdir(rpath)) if 'repetition' in rep]
        for repath in repaths: # per rep
            pkl_paths = sorted([f'{repath}/{fname}' for fname in os.listdir(repath) if '.pkl' in fname])

            N = len(pkl_paths)
            all_poses = np.zeros((N,2))
            all_points = np.zeros((N,4,2))
            all_thetas = np.zeros(N)
            for i, path in enumerate(pkl_paths):
                with open(path, 'rb') as f:
                    data = pkl.load(f)
                all_poses[i] = data['pos']
                all_points[i] = data['points']
                all_thetas[i] = data['theta']

            l2_errors = [0] * (N-5)
            li_errors = [0] * (N-5)
            route = rpath.split('/')[-1]
            rep = repath.split('/')[-1]

            print(f'{split}/{route}/{rep}')
            for n in range(N-5):

                # print time
                min = int(n//120)
                sec = int(n/2 % 60)
                #time = f'{min:02d}:{sec:02d}'

                # gps poses
                poses = all_poses[n:n+5].copy() # 5x2
                poses -= poses[0]
                poses = poses[1:] * scale # world
                poses = poses * 5.5 * -1 # map

                # theta rotation
                theta = all_thetas[n]
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
                #points = points * 5.5 * -1 # map
                points = all_points[n].copy() # 4x2
                points = points - [128, 256]
                approximate(points)
                points = R.dot(points.T).T
                points = points * -1

                l2 = np.linalg.norm(poses-points, axis=1)
                l2_errors[n] = np.sum(l2)
                li = np.amax(l2)

                # bev check against videos
                if args.bev:
                    img = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
                    draw = ImageDraw.Draw(img)
                    poses_draw = poses + size/2
                    for x, y in poses_draw:
                        draw.ellipse((x-r, y-r, x+r, y+r), (0,255,0))
                    draw.ellipse((c-r, c-r, c+r, c+r), (255,255,255))
                    points_draw = points + size/2
                    for x, y in points_draw:
                        cyan = (0, 191, 255)
                        draw.ellipse((x-r, y-r, x+r, y+r), cyan)
                    cv2.imshow('debug', cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
                    cv2.waitKey(500)

            x_plot = np.arange(len(errors))*0.5
            plt.xlabel('Game time')
            plt.ylabel('L2 error')
            ax.xaxis.set_major_locator(MultipleLocator(60))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
            ax.xaxis.set_minor_locator(MultipleLocator(15))
            ax.tick_params(which='both', direction='in')

            plt.plot(x_plot, errors)
            plt.title(f'{split}/{route}/{rep}')
            plt.show()
            plt.clf()







