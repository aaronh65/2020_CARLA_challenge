#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

from __future__ import print_function

from dictor import dictor
from collections import deque
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import seaborn as sns
import os
colors = sns.color_palette("Paired")


from srunner.scenariomanager.traffic_events import TrafficEventType
from py_trees.blackboard import Blackboard
from leaderboard.scenarios.route_scenario import NUMBER_CLASS_TRANSLATION
from leaderboard.utils.checkpoint_tools import fetch_dict, save_dict, create_default_json_msg

SAVE_PATH_BASE = os.environ.get('SAVE_PATH_BASE', 0)
ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)

PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80

penalty_dict = {
        TrafficEventType.COLLISION_PEDESTRIAN  : PENALTY_COLLISION_PEDESTRIAN,
        TrafficEventType.COLLISION_VEHICLE  : PENALTY_COLLISION_VEHICLE,
        TrafficEventType.COLLISION_STATIC  : PENALTY_COLLISION_STATIC,
        TrafficEventType.TRAFFIC_LIGHT_INFRACTION  : PENALTY_TRAFFIC_LIGHT,
        TrafficEventType.STOP_INFRACTION  : PENALTY_STOP
        }

string_dict = {
        TrafficEventType.COLLISION_PEDESTRIAN  : f'hit ped ({PENALTY_COLLISION_PEDESTRIAN}x)',
        TrafficEventType.COLLISION_VEHICLE  : f'hit vehicle ({PENALTY_COLLISION_VEHICLE}x)',
        TrafficEventType.COLLISION_STATIC  : f'hit static ({PENALTY_COLLISION_STATIC}x)',
        TrafficEventType.TRAFFIC_LIGHT_INFRACTION  : f'ran light ({PENALTY_TRAFFIC_LIGHT}x)',
        TrafficEventType.STOP_INFRACTION  : f'ran stop ({PENALTY_STOP}x)',
        }


class RouteRecord():
    def __init__(self):
        self.route_id = None
        self.index = None
        self.status = 'Started'
        self.infractions = {
            'collisions_pedestrian': [],
            'collisions_vehicle': [],
            'collisions_layout': [],
            'red_light': [],
            'stop_infraction': [],
            'outside_route_lanes': [],
            'route_dev': [],
            'route_timeout': [],
            'vehicle_blocked': []
        }

        self.scores = {
            'score_route': 0,
            'score_penalty': 0,
            'score_composed': 0
        }

        self.meta = {}


def to_route_record(record_dict):
    record = RouteRecord()
    for key, value in record_dict.items():
        setattr(record, key, value)

    return record


def compute_route_length(config):
    trajectory = config.trajectory

    route_length = 0.0
    previous_location = None
    for location in trajectory:
        if previous_location:
            dist = math.sqrt((location.x-previous_location.x)*(location.x-previous_location.x) +
                             (location.y-previous_location.y)*(location.y-previous_location.y) +
                             (location.z - previous_location.z) * (location.z - previous_location.z))
            route_length += dist
        previous_location = location

    return route_length


class StatisticsManager(object):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """

    def __init__(self): 
        self._master_scenario = None
        self._registry_route_records = []

    def resume(self, endpoint):
        data = fetch_dict(endpoint)

        if data and dictor(data, '_checkpoint.records'):
            records = data['_checkpoint']['records']

            for record in records:
                self._registry_route_records.append(to_route_record(record))

    def set_route(self, route_id, index):

        self._master_scenario = None # BasicScenario.Scenario
        self._route_scenario = None
        route_record = RouteRecord()
        route_record.route_id = route_id
        route_record.index = index

        if index < len(self._registry_route_records):
            # the element already exists and therefore we update it
            self._registry_route_records[index] = route_record
        else:
            self._registry_route_records.append(route_record)

    def set_scenario(self, scenario):
        """
        Sets the scenario from which the statistics will be taken
        """
        self._route_scenario = scenario
        self._master_scenario = scenario.scenario


    def plot_performance(self, score_route_list, infraction_list, checkpoint, tol=1e-4):
        
        fig = plt.gcf()
        fig.set_size_inches(12,8)
        ax = plt.gca()
        
        # compute penalties
        infraction_list = sorted(infraction_list, key=lambda x: x[0])
        inf_time_mult = deque([(time, penalty_dict[itype]) for time, itype in infraction_list])
        score_penalty = [1.0] * len(score_route_list)
        for i in range(1, len(score_penalty)):

            score_penalty[i] = score_penalty[i-1]
            if len(inf_time_mult) == 0:
                continue

            # check for active infraction and apply penalty if so
            inf_time, penalty = inf_time_mult[0]
            if abs(i*0.05 - inf_time) < tol or i*0.05 - inf_time >= 0.05:
                score_penalty[i] = score_penalty[i-1]*penalty
                inf_time_mult.popleft()

        # compute driving scores and reduce to 2 Hz
        score_composed_list = np.multiply(score_penalty, score_route_list) # 20 Hz
        score_composed_plot = score_composed_list[::10] # 2 Hz
        score_route_plot = score_route_list[::10]

        # plot scores
        x_plot = np.arange(len(score_composed_plot)) * 0.5 # 2 Hz
        plt.plot(x_plot, score_route_plot, label='route completion', color=colors[0])
        plt.plot(x_plot, score_composed_plot, label='driving score', color=colors[2])

        # useful for plotting
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        increment = ymax/30
        repeat = 8

        # plot infraction times
        for i, (time, itype) in enumerate(infraction_list):
            offset = increment*(i%repeat + 1) # extra 1 so text shows up below the top of the plot
            plt.vlines(time, ymin, ymax, linestyles='dashed', alpha=0.5, color='red')
            plt.text(time+0.2, ymax-offset, string_dict[itype])

        # plot scenario trigger times
        scenarios = self._route_scenario.scenario_triggerer._triggered_scenarios
        times = self._route_scenario.scenario_triggerer._triggered_scenarios_times
        lookup = self._route_scenario.route_var_name_class_lookup
        for i, (time, route_var_name) in enumerate(zip(times, scenarios)):
            offset = increment*(i%repeat)
            plt.vlines(time, ymin, ymax, linestyles='dashed', alpha=0.5, color='purple')
            plt.text(time+0.2, ymin+offset+ymax/60, lookup[route_var_name])
        
        # label axes and format ticks
        plt.xlabel('Game time')
        plt.ylabel('Score (%)')
        def format_ticks(value, tick_number):
            minute = int(value/60)
            return f'{minute:02d}:00'
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        ax.xaxis.set_minor_locator(MultipleLocator(15))
        ax.tick_params(which='both', direction='in')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        
        # finish up and save
        rep_number = int(os.environ.get('REP', 0))
        split = SAVE_PATH_BASE.split('/')[-1]
        save_path = f'{SAVE_PATH_BASE}/plots/{ROUTE_NAME}/repetition_{rep_number:02d}.png'
        title = f'{split}/{ROUTE_NAME}: repetition {rep_number:02d}'
        title = title.replace('_', ' ')
        plt.title(title)
        plt.legend(frameon=False, loc='lower right')
        plt.savefig(save_path)
        plt.clf()

    def compute_route_statistics(self, config, duration_time_system=-1, duration_time_game=-1, failure="", checkpoint=None):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        index = config.index

        if not self._registry_route_records or index >= len(self._registry_route_records):
            raise Exception('Critical error with the route registry.')

        # fetch latest record to fill in
        route_record = self._registry_route_records[index]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0
        score_route_list = []
        infraction_list = [] # each elem is (time, TrafficEventType)

        route_record.meta['duration_system'] = duration_time_system
        route_record.meta['duration_game'] = duration_time_game
        route_record.meta['route_length'] = compute_route_length(config)

        if self._master_scenario:
            if self._master_scenario.timeout_node.timeout:
                route_record.infractions['route_timeout'].append('Route timeout.')
                failure = "Agent timed out"

            for node in self._master_scenario.get_criteria():
                if node.list_traffic_events:
                    # analyze all traffic events
                    for event in node.list_traffic_events:
                        if event.get_dict():
                            event_dict = event.get_dict()
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            route_record.infractions['collisions_layout'].append(event.get_message())
                            infraction_list.append((event_dict['time'], event.get_type()))

                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            route_record.infractions['collisions_pedestrian'].append(event.get_message())
                            infraction_list.append((event_dict['time'], event.get_type()))

                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            route_record.infractions['collisions_vehicle'].append(event.get_message())
                            infraction_list.append((event_dict['time'], event.get_type()))

                        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            score_penalty *= (1 - event.get_dict()['percentage'] / 100)
                            route_record.infractions['outside_route_lanes'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            route_record.infractions['red_light'].append(event.get_message())
                            infraction_list.append((event_dict['time'], event.get_type()))

                        elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                            route_record.infractions['route_dev'].append(event.get_message())
                            failure = "Agent deviated from the route"

                        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            route_record.infractions['stop_infraction'].append(event.get_message())
                            infraction_list.append((event_dict['time'], event.get_type()))

                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            route_record.infractions['vehicle_blocked'].append(event.get_message())
                            failure = "Agent got blocked"

                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                if event.get_dict():
                                    score_route_list = event.get_dict()['route_completed_list']
                                    score_route = event.get_dict()['route_completed']
                                else:
                                    score_route = 0

        
        # update route scores
        route_record.scores['score_route'] = score_route
        route_record.scores['score_penalty'] = score_penalty
        route_record.scores['score_composed'] = max(score_route*score_penalty, 0.0)

        # plot per-route performance
        if len(score_route_list) > 0:
            self.plot_performance(score_route_list, infraction_list, checkpoint)
        
        # update status
        if target_reached:
            route_record.status = 'Completed'
        else:
            route_record.status = 'Failed'
            if failure:
                route_record.status += ' - ' + failure

        return route_record

    def compute_global_statistics(self, total_routes):
        global_record = RouteRecord()
        global_record.route_id = -1
        global_record.index = -1
        global_record.status = 'Completed'

        if self._registry_route_records:
            for route_record in self._registry_route_records:
                global_record.scores['score_route'] += route_record.scores['score_route']
                global_record.scores['score_penalty'] += route_record.scores['score_penalty']
                global_record.scores['score_composed'] += route_record.scores['score_composed']

                for key in global_record.infractions.keys():
                    route_length_kms = max(route_record.scores['score_route'] * route_record.meta['route_length'] / 1000.0, 0.001)
                    if isinstance(global_record.infractions[key], list):
                        global_record.infractions[key] = len(route_record.infractions[key]) / route_length_kms
                    else:
                        global_record.infractions[key] += len(route_record.infractions[key]) / route_length_kms

                if route_record.status is not 'Completed':
                    global_record.status = 'Failed'
                    if 'exceptions' not in global_record.meta:
                        global_record.meta['exceptions'] = []
                    global_record.meta['exceptions'].append((route_record.route_id,
                                                             route_record.index,
                                                             route_record.status))

        global_record.scores['score_route'] /= float(total_routes)
        global_record.scores['score_penalty'] /= float(total_routes)
        global_record.scores['score_composed'] /= float(total_routes)

        return global_record

    @staticmethod
    def save_record(route_record, index, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        record_list = data['_checkpoint']['records']
        if index > len(record_list):
            print('Error! No enough entries in the list')
            sys.exit(-1)
        elif index == len(record_list):
            record_list.append(stats_dict)
        else:
            record_list[index] = stats_dict

        save_dict(endpoint, data)

    @staticmethod
    def save_global_record(route_record, sensors, total_routes, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        data['_checkpoint']['global_record'] = stats_dict
        data['values'] = ['{:.3f}'.format(stats_dict['scores']['score_composed']),
                          '{:.3f}'.format(stats_dict['scores']['score_route']),
                          '{:.3f}'.format(stats_dict['scores']['score_penalty']),
                          # infractions
                          '{:.3f}'.format(stats_dict['infractions']['collisions_pedestrian']),
                          '{:.3f}'.format(stats_dict['infractions']['collisions_vehicle']),
                          '{:.3f}'.format(stats_dict['infractions']['collisions_layout']),
                          '{:.3f}'.format(stats_dict['infractions']['red_light']),
                          '{:.3f}'.format(stats_dict['infractions']['stop_infraction']),
                          '{:.3f}'.format(stats_dict['infractions']['outside_route_lanes']),
                          '{:.3f}'.format(stats_dict['infractions']['route_dev']),
                          '{:.3f}'.format(stats_dict['infractions']['route_timeout']),
                          '{:.3f}'.format(stats_dict['infractions']['vehicle_blocked'])
                          ]

        data['labels'] = ['Avg. driving score',
                          'Avg. route completion',
                          'Avg. infraction penalty',
                          'Collisions with pedestrians',
                          'Collisions with vehicles',
                          'Collisions with layout',
                          'Red lights infractions',
                          'Stop sign infractions',
                          'Off-road infractions',
                          'Route deviations',
                          'Route timeouts',
                          'Agent blocked'
                          ]

        entry_status = "Finished"
        eligible = True

        route_records = data["_checkpoint"]["records"]
        progress = data["_checkpoint"]["progress"]

        if progress[1] != total_routes:
            raise Exception('Critical error with the route registry.')

        if len(route_records) != total_routes or progress[0] != progress[1]:
            entry_status = "Finished with missing data"
            eligible = False
        else:
            for route in route_records:
                route_status = route["status"]
                if "Agent" in route_status:
                    entry_status = "Finished with agent errors"
                    break

        data['entry_status'] = entry_status
        data['eligible'] = eligible

        save_dict(endpoint, data)

    @staticmethod
    def save_sensors(sensors, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        if not data['sensors']:
            data['sensors'] = sensors

            save_dict(endpoint, data)

    @staticmethod
    def save_entry_status(entry_status, eligible, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        data['entry_status'] = entry_status
        data['eligible'] = eligible
        save_dict(endpoint, data)

    @staticmethod
    def clear_record(endpoint):
        if not endpoint.startswith(('http:', 'https:', 'ftp:')):
            with open(endpoint, 'w') as fd:
                fd.truncate(0)
