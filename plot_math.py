import os
import numpy as np
np.set_printoptions(suppress=True, precision=3)
#import json
import pickle as pkl
import argparse
from io import StringIO
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True)
args = parser.parse_args()
fnames = sorted(os.listdir(args.target_dir))
math_path = f'{args.target_dir}/math'
rpaths = [f'{math_path}/{route}' for route in os.listdir(math_path) if 'route' in route]
for rpath in rpaths: # per route
    repaths = [f'{rpath}/{rep}' for rep in os.listdir(rpath) if 'repetition' in rep]
    for repath in repaths: # per rep
        pkl_paths = sorted([f'{repath}/{fname}' for fname in os.listdir(repath) if '.pkl' in fname])

        N = len(pkl_paths)
        all_points = np.zeros((N,4,2))
        all_poses = np.zeros((N,2))
        all_thetas = np.zeros(N)
        for i, path in enumerate(pkl_paths):
            with open(path, 'rb') as f:
                data = pkl.load(f)
            all_points[i] = data['points']
            all_poses[i] = data['pos']
            all_thetas[i] = data['theta']

        #all_points*=5.5
        n = 24
        points = all_points[n]
        poses = all_poses[n:n+5]
        print(poses)
        poses -= poses[0]
        poses = poses[1:]
        print(points)
        print(poses)

        l2 = np.linalg.norm(poses-points, axis=1)
        print(l2)
        x, y = poses.T
        plt.scatter(x, y, label='poses')
        x, y = points.T
        plt.scatter(x, y, label='points')
        plt.legend()
        plt.show()
