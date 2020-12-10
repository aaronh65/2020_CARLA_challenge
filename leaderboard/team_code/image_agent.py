import os
import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter

from team_code.base_agent import BaseAgent
from team_code.pid_controller import PIDController


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
SAVE_IMAGES = int(os.environ.get('SAVE_IMAGES', 0))
SAVE_IMAGES_PATH = os.environ.get('SAVE_IMAGES_PATH', 0)
DIM=(1371,256)

def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step, _waypoint_img):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 0, 0))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144

        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

    _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
    _draw = ImageDraw.Draw(_combined)
    #text_color = (220, 220, 220)
    text_color = (70, 130, 180)
    _draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
    _draw.text((5, 50), 'Brake: %s' % brake, text_color)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'], text_color)
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed, text_color)
    cur_command = tick_data['cur_command']
    _draw.text((5, 110), f'Current: {cur_command}', text_color)
    next_command = tick_data['next_command']
    _draw.text((5, 130), f'Next: {next_command}', text_color)



    _rgb_img = cv2.resize(np.array(_combined), DIM, interpolation=cv2.INTER_AREA)
    _save_img = Image.fromarray(np.hstack([_rgb_img, _waypoint_img]))
    _save_img = cv2.cvtColor(np.array(_save_img), cv2.COLOR_BGR2RGB)
    if step % 10 == 0 and SAVE_IMAGES:
        frame_number = step // 10 + 1
        rep_number = int(os.environ.get('REP',0))
        save_path = os.path.join(SAVE_IMAGES_PATH, f'repetition_{rep_number:02d}', f'{frame_number:06d}.png')
        cv2.imwrite(save_path, _save_img)
    if DEBUG:
        cv2.imshow('debug', _save_img)
        cv2.waitKey(1)


class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)


    def tick(self, input_data):
        result = super().tick(input_data)
        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        #print((theta * 180 / np.pi)%360)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        result['rotation'] = R
        gps = self._get_position(result)

        # oriented in world frame
        #far_node, _ = self._command_planner.run_step(gps)
        current_waypoint, next_waypoint = self._command_planner.run_step(gps)
        cur_node, cur_command = current_waypoint
        far_node, far_command = next_waypoint
        result['cur_node'] = cur_node
        result['cur_command'] = str(cur_command).split('.')[1]
        result['next_node'] = far_node
        result['next_command'] = str(far_command).split('.')[1]
        
        target = R.T.dot(far_node - gps) # map/world frame to ego frame
        target *= 5.5 # from converter.PIXELS_PER_WORLD
        target += [128, 256] # ego origin in map frame
        target = np.clip(target, 0, 256)
        result['target'] = target


        # keep track of rotation to make debug plot in map view
        #self.R = R
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

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        points_cam = points.clone().cpu()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()

        points_world = self.converter.cam_to_world(points_cam).numpy()

        # for math project
        #self.save_poly_data(tick_data, points_world)

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2

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

        if DEBUG or SAVE_IMAGES:
            _waypoint_img = self._command_planner.debug.img
            points_map = self.converter.cam_to_map(points_cam).numpy()

            # center at origin, rotate
            points_plot = points_map - [128, 256]
            R = tick_data['rotation']
            #points_plot = self.R.dot(points_plot.T).T
            points_plot = R.dot(points_plot.T).T
            points_plot *= -1 # why is this required?
            points_plot += 256/2 # recenter origin in middle of plot
            for x, y in points_plot:
                ImageDraw.Draw(_waypoint_img).ellipse((x-2, y-2, x+2, y+2), (0,0,255))
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step, _waypoint_img)

        return control

    def save_poly_data(self, tick_data, points_world):
        pos = self._get_position(tick_data)
        data = {
                'points': points_world,
                'pos': pos
                }
        print(f'points_world\n{points_world}')
        print(f'pos\n{pos}')
        print(f'data\n{data}')
