import os
import numpy as np
import cv2
import time
np.set_printoptions(suppress=True, precision=6)
from polyfit import approximate
import pickle as pkl
import argparse
from PIL import Image, ImageDraw


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

# get route paths
for split in os.listdir(args.target_dir):
    math_path = os.path.join(args.target_dir, split, 'math')
    if not os.path.exists(math_path):
        continue
    if args.route:
        route_name = f'route_{args.route:02d}'
        rpaths = [f'{math_path}/{route}' for route in os.listdir(math_path) if route == route_name]
    else:
        rpaths = [f'{math_path}/{route}' for route in os.listdir(math_path) if 'route' in route]
    rpaths = sorted(rpaths)

    img = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    for rpath in rpaths: # per route
        repaths = [f'{rpath}/{rep}' for rep in os.listdir(rpath) if 'repetition' in rep]
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

            for n in range(N-5):
                print(n)

                # print time
                min = int(n//120)
                sec = int(n/2 % 60)
                time = f'{min:02d}:{sec:02d}'

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

