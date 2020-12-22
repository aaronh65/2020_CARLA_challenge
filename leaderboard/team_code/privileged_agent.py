import os
import numpy as np
import cv2
#import json
import pickle as pkl
import torch
import torchvision
import carla

from PIL import Image, ImageDraw
from pathlib import Path

#from carla_project.src.image_model import ImageModel
from carla_project.src.map_model import MapModel
from carla_project.src.dataset import preprocess_semantic
from carla_project.src.converter import Converter

from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController

#from polyfit import approximate
import polyfit


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
SAVE_MATH = int(os.environ.get('SAVE_MATH', 0))
RUN_MATH = int(os.environ.get('RUN_MATH', 0))
SAVE_IMAGES = int(os.environ.get('SAVE_IMAGES', 0))
SAVE_PATH_BASE = os.environ.get('SAVE_PATH_BASE', 0)
ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)
DIM=(1371,256)

def get_entry_point():
    return 'PrivilegedAgent'

class PrivilegedAgent(MapAgent):
    def setup(self, path_to_conf_file):
        # make conf file a json file?
        # store hparams for map model
        # can extend to store all of the save paths so we don't rely on os environ stuff
        # can copy this to the corresponding save path
        
        super().setup(path_to_conf_file)
        self.converter = Converter()

        # make hacky hparams namespace for now
        # missing training hparams
        class Bunch(object):
            def __init__(self, adict):
                self.__dict__.update(adict)
        hparams = {}
        hparams['hack'] = True
        hparams['temperature'] = 10
        hparams['heatmap_radius'] = 5
        hparams['command_coefficient'] = 0.01
        hparams['batch_norm'] = False
        hparams_ns = Bunch(hparams)

        self.net = MapModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self.save_math_path = Path(f'{SAVE_PATH_BASE}/math/{ROUTE_NAME}')
        self.save_images_path = Path(f'{SAVE_PATH_BASE}/images/{ROUTE_NAME}')
        #self.save_path.mkdir()


    def tick(self, input_data):
        result = super().tick(input_data)
        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        result['theta'] = theta
        #print((theta * 180 / np.pi)%360)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        result['R'] = R
        gps = self._get_position(result) # method returns position in meters

        
        # transform route waypoints to overhead map view
        route = self._command_planner.run_step(gps) # oriented in world frame
        nodes = np.array([node for node, _ in route]) # (N,2)
        nodes = nodes - gps # center at agent position and rotate
        nodes = R.T.dot(nodes.T) # (2,2) x (2,N) = (2,N)
        nodes = nodes.T * 5.5 # (N,2) # to map frame (5.5 pixels per meter)
        nodes += [128,256]
        nodes = np.clip(nodes, 0, 256)
        commands = [command for _, command in route]

        # populate results
        result['num_waypoints'] = len(route)
        result['route_map'] = nodes
        result['commands'] = commands
        result['target'] = nodes[1]

        return result

    @torch.no_grad()
    def run_step_using_learned_controller(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        #points, (target_cam, _) = self.net.forward(img, target)
        points, (target_cam, _) = self.net.forward(img, target)
        control = self.net.controller(points).cpu().squeeze()

        steer = control[0].item()
        desired_speed = control[1].item()
        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        topdown = Image.fromarray(tick_data['topdown'])
        topdown = topdown.crop((128, 0, 128+256, 256))
        topdown = np.array(topdown)
        topdown = preprocess_semantic(topdown)
        topdown = topdown[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        #points, (target_cam, _) = self.net.forward(topdown, target)
        points = self.net.forward(topdown, target) # world frame
        points_map = points.clone().cpu().squeeze()
        points_map = points_map + 1
        points_map = points_map / 2 * 256
        points_map = np.clip(points_map, 0, 256)
        points_cam = self.converter.map_to_cam(points_map).numpy()
        points_world = self.converter.map_to_world(points_map).numpy()
        points_map = points_map.numpy()

        tick_data['points_map'] = points_map
        tick_data['points_cam'] = points_cam
        tick_data['points_world'] = points_world

        img = tick_data['image']

        if RUN_MATH:
            # there are 5 points including the origin and last pt
            # and 3 points in between a pair of points
            # so there are 16 valid choices total
            #j = 6 # point between 1st and 2nd waypoint
            j = int(os.environ.get('POLY_SELECT', 0))
            # we exclude the origin and end to compute desired speed
            assert 0 < j < 15, 'point choice invalid'

            points_apx = polyfit.approximate(points_world)
            aim = points_apx[j]
            angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
            steer = self._turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)

            desired_speed_1 = np.linalg.norm(points_apx[j] - points_apx[j-1]) * 8.0
            desired_speed_2 = np.linalg.norm(points_apx[j+1] - points_apx[j]) * 8.0
            desired_speed = desired_speed_1 + desired_speed_2
            desired_speed = desired_speed / 2
            
        else:
            aim = (points_world[1] + points_world[0]) / 2.0
            angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
            steer = self._turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)

            desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
            # desired_speed *= (1 - abs(angle)) ** 2

        tick_data['aim_world'] = aim

        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)
        #print(timestamp) # GAMETIME

        # for math project
        if self.step % 10 == 0 and SAVE_MATH:
            self.save_math_data(tick_data)

        if DEBUG or SAVE_IMAGES:

            # transform image model cam points to overhead BEV image (spectator frame?)
            self.debug_display(
                    tick_data, steer, throttle, brake, desired_speed)
            

        return control

    def debug_display(self, tick_data, steer, throttle, brake, desired_speed):

        # make BEV image

        # transform aim from world to map
        aim_world = np.array(tick_data['aim_world'])
        aim_map = self.converter.world_to_map(torch.Tensor(aim_world)).numpy()

        # append to image model points and plot
        points_plot = np.vstack([tick_data['points_map'], aim_map])
        points_plot = points_plot - [128,256] # center at origin
        points_plot = tick_data['R'].dot(points_plot.T).T
        points_plot = points_plot * -1 # why is this required?
        points_plot = points_plot + 256/2 # recenter origin in middle of plot
        _waypoint_img = self._command_planner.debug.img
        for x, y in points_plot:
            ImageDraw.Draw(_waypoint_img).ellipse((x-2, y-2, x+2, y+2), (0, 191, 255))
        x, y = points_plot[-1]
        ImageDraw.Draw(_waypoint_img).ellipse((x-2, y-2, x+2, y+2), (255, 105, 147))

        # make RGB images

        # draw center RGB image
        _rgb = Image.fromarray(tick_data['rgb'])
        _draw_rgb = ImageDraw.Draw(_rgb)
        for x, y in tick_data['points_cam']: # image model waypoints
            #x = (x + 1)/2 * 256
            #y = (y + 1)/2 * 144
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 191, 255))

        # transform aim from world to cam
        aim_world = np.array(tick_data['aim_world'])
        aim_cam = self.converter.world_to_cam(torch.Tensor(aim_world)).numpy()
        x, y = aim_cam
        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 105, 147))

        # draw route waypoints in RGB image
        route_map = np.array(tick_data['route_map'])
        route_map = route_map[:3].squeeze()
        route_cam = self.converter.map_to_cam(torch.Tensor(route_map)).numpy()
        for i, (x, y) in enumerate(route_cam):
            if i == 0: # waypoint we just passed
                if y >= 139 or x <= 2 or x >= 254: # bottom of frame (behind us)
                    continue
                color = (0, 255, 0) # green 
            elif i == 1: # target
                color = (255, 0, 0) # red
            elif i == 2: # beyond target
                color = (139, 0, 139) # darkmagenta
            else:
                continue
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), color)

        _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
        _draw = ImageDraw.Draw(_combined)

        # draw debug text
        text_color = (139, 0, 139) #darkmagenta
        _draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
        _draw.text((5, 50), 'Brake: %s' % brake, text_color)
        _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'], text_color)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed, text_color)
        cur_command, next_command = tick_data['commands'][:2]
        _draw.text((5, 110), f'Current: {cur_command}', text_color)
        _draw.text((5, 130), f'Next: {next_command}', text_color)

        _rgb_img = cv2.resize(np.array(_combined), DIM, interpolation=cv2.INTER_AREA)
        _save_img = Image.fromarray(np.hstack([_rgb_img, _waypoint_img]))
        _save_img = cv2.cvtColor(np.array(_save_img), cv2.COLOR_BGR2RGB)
        if self.step % 10 == 0 and SAVE_IMAGES:
            frame_number = self.step // 10
            rep_number = int(os.environ.get('REP',0))
            save_path = self.save_images_path / f'repetition_{rep_number:02d}' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _save_img)
        if DEBUG:
            cv2.imshow('debug', _save_img)
            cv2.waitKey(1)
 

    def save_math_data(self, tick_data):
        points_map = tick_data['points_map']
        pos = tick_data['gps']
        theta = tick_data['theta']
        data = {
                'points': points_map.tolist(),
                'pos': pos,
                'theta': theta
                }

        frame_number = self.step // 10
        rep_number = int(os.environ.get('REP',0))
        save_path = self.save_math_path / f'repetition_{rep_number:02d}' / f'{frame_number:06d}.pkl'
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
