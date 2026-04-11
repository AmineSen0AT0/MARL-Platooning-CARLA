#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# Copyright (c) 2026: Amine Zerarga (amine.zerarga@univ-annaba.dz) - Modified for MARL Platooning
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gymnasium as gym
from gymnasium import spaces
import carla

from .render import BirdeyeRender
from .route_planner import RoutePlanner
from .misc import *


class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  # def __init__(self, params):
  #   # FIX 1: Initialize the parent Gymnasium class (Crucial for new versions)
  #   super().__init__()
  #
  #   # parameters
  #   self.display_size = params['display_size']  # rendering screen size
  #   self.max_past_step = params['max_past_step']
  #   self.number_of_vehicles = params['number_of_vehicles']
  #   self.number_of_walkers = params['number_of_walkers']
  #   self.dt = params['dt']
  #   self.task_mode = params['task_mode']
  #   self.max_time_episode = params['max_time_episode']
  #   self.max_waypt = params['max_waypt']
  #   self.obs_range = params['obs_range']
  #   self.lidar_bin = params['lidar_bin']
  #   self.d_behind = params['d_behind']
  #   self.obs_size = int(self.obs_range / self.lidar_bin)
  #   self.out_lane_thres = params['out_lane_thres']
  #   self.desired_speed = params['desired_speed']
  #   self.max_ego_spawn_times = params['max_ego_spawn_times']
  #   self.display_route = params['display_route']
  #   if 'pixor' in params.keys():
  #     self.pixor = params['pixor']
  #     self.pixor_size = params['pixor_size']
  #   else:
  #     self.pixor = False
  #
  #   # Destination
  #   if params['task_mode'] == 'roundabout':
  #     self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
  #   else:
  #     self.dests = None
  #
  #   # action and observation spaces
  #   self.discrete = params['discrete']
  #   self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
  #   self.n_acc = len(self.discrete_act[0])
  #   self.n_steer = len(self.discrete_act[1])
  #   if self.discrete:
  #     self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
  #   else:
  #     # Note: Ensure your params dictionary has 'continuous_accel_range' and 'continuous_steer_range'
  #     self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
  #     params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
  #     params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
  #
  #   observation_space_dict = {
  #     'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
  #     'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
  #     'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
  #
  #     # FIX 2: Relaxed bounds for 'state' to prevent crashes during training
  #     # Original was [-2, 2], we change to [-inf, inf] for safety
  #     'state': spaces.Box(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
  #                         np.array([np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
  #   }
  #
  #   if self.pixor:
  #     observation_space_dict.update({
  #       'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
  #       'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
  #       'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
  #       'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]),
  #                                 dtype=np.float32)
  #     })
  #   self.observation_space = spaces.Dict(observation_space_dict)
  #
  #   # Connect to carla server and get world object
  #   print('connecting to Carla server...')
  #   client = carla.Client('localhost', params['port'])
  #   client.set_timeout(10.0)
  #   # Warning: load_world reloads the map every time. It's slow but safe for now.
  #   self.world = client.load_world(params['town'])
  #   print('Carla server connected!')
  #
  #   # Set weather
  #   self.world.set_weather(carla.WeatherParameters.ClearNoon)
  #
  #   # Get spawn points
  #   self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
  #   self.walker_spawn_points = []
  #   for i in range(self.number_of_walkers):
  #     spawn_point = carla.Transform()
  #     loc = self.world.get_random_location_from_navigation()
  #     if (loc != None):
  #       spawn_point.location = loc
  #       self.walker_spawn_points.append(spawn_point)
  #
  #   # Create the ego vehicle blueprint
  #   # Note: Kept typo 'bluepprint' to match other methods in the file for now
  #   self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')
  #
  #   # Collision sensor
  #   self.collision_hist = []  # The collision history
  #   self.collision_hist_l = 1  # collision history length
  #   self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
  #
  #   # Lidar sensor
  #   self.lidar_data = None
  #   self.lidar_height = 2.1
  #   self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
  #   self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
  #   self.lidar_bp.set_attribute('channels', '32')
  #   self.lidar_bp.set_attribute('range', '5000')
  #
  #   # Camera sensor
  #   self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
  #   self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
  #   self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
  #   # Modify the attributes of the blueprint to set image resolution and field of view.
  #   self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
  #   self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
  #   self.camera_bp.set_attribute('fov', '110')
  #   # Set the time in seconds between sensor captures
  #   self.camera_bp.set_attribute('sensor_tick', '0.02')
  #
  #   # Set fixed simulation step for synchronous mode
  #   self.settings = self.world.get_settings()
  #   self.settings.fixed_delta_seconds = self.dt
  #
  #   # FIX 3: Explicitly enable Synchronous Mode (Vital for PPO)
  #   self.settings.synchronous_mode = True
  #   self.world.apply_settings(self.settings)
  #
  #   # Record the time of total steps and resetting steps
  #   self.reset_step = 0
  #   self.total_step = 0
  #
  #   # Initialize the renderer
  #   self._init_renderer()
  #
  #   # Get pixel grid points
  #   if self.pixor:
  #     x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size))  # make a canvas with coordinates
  #     x, y = x.flatten(), y.flatten()
  #     self.pixel_grid = np.vstack((x, y)).T

  def __init__(self, params):
    # FIX 1: Initialize the parent Gymnasium class (Crucial for new versions)
    super().__init__()

    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range / self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    else:
      self.dests = None

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
    else:
      # Note: Ensure your params dictionary has 'continuous_accel_range' and 'continuous_steer_range'
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                               params['continuous_steer_range'][0]]),
                                     np.array([params['continuous_accel_range'][1],
                                               params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer

    observation_space_dict = {
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),

      # FIX 2: Relaxed bounds for 'state' to prevent crashes during training
      # Original was [-2, 2], we change to [-inf, inf] for safety
      'state': spaces.Box(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                          np.array([np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
    }

    if self.pixor:
      observation_space_dict.update({
        'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]),
                                  dtype=np.float32)
      })
    self.observation_space = spaces.Dict(observation_space_dict)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(10.0)
    # Warning: load_world reloads the map every time. It's slow but safe for now.
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    # Note: Kept typo 'bluepprint' to match other methods in the file for now
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = []  # The collision history
    self.collision_hist_l = 1  # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor
    self.lidar_data = None
    self.lidar_height = 2.1
    self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '32')
    self.lidar_bp.set_attribute('range', '5000')

    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # FIX 3: Explicitly enable Synchronous Mode (Vital for PPO)
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    # Initialize the renderer
    self._init_renderer()

    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size))  # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T

    # ==============================================================================
    # FIX: NEW MULTI-AGENT PLATOON VARIABLES
    # ==============================================================================
    self.num_agents = 5
    self.egos = []  # Will hold our 5 F1TENTH vehicles
    self.routeplanners = []  # Will hold 5 route planners
    self.waypoints_list = [None] * self.num_agents
    self.vehicle_fronts = [50.0] * self.num_agents

    # Load the blueprint for the slow Pace Car here once to save memory
    self.pace_car = None
    self.pace_car_bp = self.world.get_blueprint_library().find('vehicle.carlamotors.carlacola')
    # ==============================================================================
  # def reset(self, seed=None, options=None):
  #   # FIX 1: Add Gymnasium seeding
  #   super().reset(seed=seed)
  #
  #   # Clear sensor objects
  #   self.collision_sensor = None
  #   self.lidar_sensor = None
  #   self.camera_sensor = None
  #
  #   # Delete sensors, vehicles and walkers
  #   self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
  #
  #   # Disable sync mode
  #   self._set_synchronous_mode(False)
  #
  #   # Spawn surrounding vehicles
  #   random.shuffle(self.vehicle_spawn_points)
  #   count = self.number_of_vehicles
  #   if count > 0:
  #     for spawn_point in self.vehicle_spawn_points:
  #       if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
  #         count -= 1
  #       if count <= 0:
  #         break
  #   while count > 0:
  #     if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
  #       count -= 1
  #
  #   # Spawn pedestrians
  #   random.shuffle(self.walker_spawn_points)
  #   count = self.number_of_walkers
  #   if count > 0:
  #     for spawn_point in self.walker_spawn_points:
  #       if self._try_spawn_random_walker_at(spawn_point):
  #         count -= 1
  #       if count <= 0:
  #         break
  #   while count > 0:
  #     if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
  #       count -= 1
  #
  #   # Get actors polygon list
  #   self.vehicle_polygons = []
  #   vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
  #   self.vehicle_polygons.append(vehicle_poly_dict)
  #   self.walker_polygons = []
  #   walker_poly_dict = self._get_actor_polygons('walker.*')
  #   self.walker_polygons.append(walker_poly_dict)
  #
  #   # Spawn the ego vehicle
  #   ego_spawn_times = 0
  #   while True:
  #     if ego_spawn_times > self.max_ego_spawn_times:
  #       # FIX 2: Pass arguments if we need to restart
  #       return self.reset(seed=seed, options=options)
  #
  #     if self.task_mode == 'random':
  #       transform = random.choice(self.vehicle_spawn_points)
  #     if self.task_mode == 'roundabout':
  #       self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
  #       # self.start=[52.1,-4.2, 178.66] # static
  #       transform = self._set_carla_transform(self.start) # Note: Added self. if set_carla_transform is a method
  #     if self._try_spawn_ego_vehicle_at(transform):
  #       break
  #     else:
  #       ego_spawn_times += 1
  #       time.sleep(0.1)
  #
  #   # Add collision sensor
  #   self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
  #   self.collision_sensor.listen(lambda event: get_collision_hist(event))
  #   def get_collision_hist(event):
  #     impulse = event.normal_impulse
  #     intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
  #     self.collision_hist.append(intensity)
  #     if len(self.collision_hist)>self.collision_hist_l:
  #       self.collision_hist.pop(0)
  #   self.collision_hist = []
  #
  #   # Add lidar sensor
  #   # self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
  #   # self.lidar_sensor.listen(lambda data: get_lidar_data(data))
  #   # def get_lidar_data(data):
  #   #   self.lidar_data = data
  #
  #   # Add camera sensor
  #   self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
  #   self.camera_sensor.listen(lambda data: get_camera_img(data))
  #   def get_camera_img(data):
  #     array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
  #     array = np.reshape(array, (data.height, data.width, 4))
  #     array = array[:, :, :3]
  #     array = array[:, :, ::-1]
  #     self.camera_img = array
  #
  #   # Update timesteps
  #   self.time_step=0
  #   self.reset_step+=1
  #
  #   # Enable sync mode
  #   self.settings.synchronous_mode = True
  #   self.world.apply_settings(self.settings)
  #
  #   self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
  #   self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
  #
  #   # Set ego information for render
  #   self.birdeye_render.set_hero(self.ego, self.ego.id)
  #
  #   # FIX 3: Return observation AND info dictionary
  #   return self._get_obs(), {}

  def reset(self, seed=None, options=None):
    # FIX 1: Add Gymnasium seeding
    super().reset(seed=seed)

    # Clear sensor objects
    self.collision_sensor = None
    self.lidar_sensor = None
    self.camera_sensor = None

    # Delete sensors, vehicles and walkers
    self._clear_all_actors(
      ['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker',
       'walker.*'])

    # Disable sync mode
    self._set_synchronous_mode(False)

    # Spawn surrounding vehicles (Keep this for traffic chaos!)
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

    # Spawn pedestrians
    random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        count -= 1

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # ==============================================================================
    # FIX 2: BULLETPROOF PLATOON SPAWNING LOGIC (6 Cars Total)
    # ==============================================================================
    self.num_agents = 6  # Agent 0 is the Pace Car, Agents 1-5 are the Platoon
    self.egos = []
    self.routeplanners = []
    self.waypoints_list = [None] * self.num_agents
    self.vehicle_fronts = [50.0] * self.num_agents

    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        return self.reset(seed=seed, options=options)

      # 1. Pick a random valid spawn point
      transform = random.choice(self.vehicle_spawn_points)
      current_wp = self.world.get_map().get_waypoint(transform.location)
      spawn_success = True

      # 2. Spawn all 6 identical agents in a perfect line
      for i in range(self.num_agents):
        spawn_trans = current_wp.transform
        spawn_trans.location.z += 0.5  # Drop safely from slightly above

        ego = self.world.try_spawn_actor(self.ego_bp, spawn_trans)
        if ego is not None:
          self.egos.append(ego)
        else:
          spawn_success = False
          break

        # Step back 12.0 meters along the lane for the next car to prevent overlap
        wp_list = current_wp.previous(12.0)
        if not wp_list:
          spawn_success = False
          break
        current_wp = wp_list[0]

      # If all 6 cars spawned perfectly, break the loop. Otherwise, clean up and retry.
      if spawn_success and len(self.egos) == self.num_agents:
        break
      else:
        for e in self.egos:
          e.destroy()
        self.egos = []
        ego_spawn_times += 1
    # ==============================================================================

    # Add collision sensor to Agent 0 (The Pace Car)
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.egos[0])
    self.collision_sensor.listen(lambda event: get_collision_hist(event))

    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist) > self.collision_hist_l:
        self.collision_hist.pop(0)

    self.collision_hist = []

    # Add camera sensor to Agent 0 (The Pace Car) so we can watch the track
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.egos[0])
    self.camera_sensor.listen(lambda data: get_camera_img(data))

    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img = array

    # Update timesteps
    self.time_step = 0
    self.reset_step += 1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # Initialize Route Planners for ALL 6 agents
    for i in range(self.num_agents):
      # FIX: Freeze the dice BEFORE the planner even wakes up to build its first 12 waypoints!
      random.seed(42)
      np.random.seed(42)

      planner = RoutePlanner(self.egos[i], self.max_waypt)
      self.routeplanners.append(planner)

      self.waypoints_list[i], _, self.vehicle_fronts[i] = planner.run_step()

      # Unfreeze the dice so the rest of the simulation (like NPC traffic) runs normally
    random.seed()
    np.random.seed()

    # Set ego information for PyGame Render (Focusing on Agent 0)
    self.birdeye_render.set_hero(self.egos[0], self.egos[0].id)

    # Return observation AND info dictionary
    return self._get_obs(), {}
  # def step(self, action):
  #   # Calculate acceleration and steering
  #   if self.discrete:
  #     acc = self.discrete_act[0][action // self.n_steer]
  #     steer = self.discrete_act[1][action % self.n_steer]
  #   else:
  #     acc = action[0]
  #     steer = action[1]
  #
  #   # Convert acceleration to throttle and brake
  #   if acc > 0:
  #     throttle = np.clip(acc / 3, 0, 1)
  #     brake = 0
  #   else:
  #     throttle = 0
  #     # brake = np.clip(-acc / 8, 0, 1)
  #     # FIX: Changed from / 8 to / 3.
  #     # Now when acc is -3.0, brake = 1.0 (100% stopping power!)
  #     brake = np.clip(-acc / 3, 0, 1)
  #
  #   # Apply control
  #   act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
  #   self.ego.apply_control(act)
  #
  #   self.world.tick()
  #
  #   # Append actors polygon list
  #   vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
  #   self.vehicle_polygons.append(vehicle_poly_dict)
  #   while len(self.vehicle_polygons) > self.max_past_step:
  #     self.vehicle_polygons.pop(0)
  #   walker_poly_dict = self._get_actor_polygons('walker.*')
  #   self.walker_polygons.append(walker_poly_dict)
  #   while len(self.walker_polygons) > self.max_past_step:
  #     self.walker_polygons.pop(0)
  #
  #   # route planner
  #   self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
  #
  #   # state information
  #   info = {
  #     'waypoints': self.waypoints,
  #     'vehicle_front': self.vehicle_front
  #   }
  #
  #   # Update timesteps
  #   self.time_step += 1
  #   self.total_step += 1
  #
  #   # FIX: Gymnasium requires 5 values (terminated, truncated)
  #   # 1. Get the observation and reward
  #   obs = self._get_obs()
  #   reward = self._get_reward()
  #
  #   # 2. Handle termination (Crash or Success)
  #   terminated = self._terminal()
  #
  #   # 3. Handle truncation (Time limit exceeded)
  #   # We check if the episode runs longer than max_time_episode
  #   truncated = False
  #   if self.time_step >= self.max_time_episode:
  #     truncated = True
  #
  #   return obs, reward, terminated, truncated, copy.deepcopy(info)

  def step(self, actions):
    # ==============================================================================
    # FIX 3: MULTI-AGENT ACTION LOOP
    # ==============================================================================
    for i in range(self.num_agents):
      action = actions[i]

      # Calculate acceleration and steering
      if self.discrete:
        acc = self.discrete_act[0][action // self.n_steer]
        steer = self.discrete_act[1][action % self.n_steer]
      else:
        acc = action[0]
        steer = action[1]

      # Convert acceleration to throttle and brake
      if acc > 0:
        throttle = np.clip(acc / 3, 0, 1)
        brake = 0
      else:
        throttle = 0
        brake = np.clip(-acc / 3, 0, 1)

      # Apply control to specific agent
      act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
      self.egos[i].apply_control(act)
    # ==============================================================================

    # Tick the physics engine once for all vehicles
    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route planner for all agents

    for i in range(self.num_agents):
      # FIX: Freeze the dice before routing so followers never diverge from the Leader
      random.seed(42)
      np.random.seed(42)
      self.waypoints_list[i], _, _ = self.routeplanners[i].run_step()

    # Unfreeze the dice
    random.seed()
    np.random.seed()

    # state information
    info = {
      'waypoints': self.waypoints_list,
      'vehicle_front': self.vehicle_fronts
    }


    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    # 1. Get the observation and reward
    obs = self._get_obs()
    reward = self._get_reward()

    # 2. Handle termination (Crash or Success)
    terminated = self._terminal()

    # 3. Handle truncation (Time limit exceeded)
    truncated = False
    if self.time_step >= self.max_time_episode:
      truncated = True

    return obs, reward, terminated, truncated, copy.deepcopy(info)
  # REMOVED def seed(...) - It is handled in reset() now.

  # FIX: Update render signature (remove 'mode')
  def render(self):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if
                                               int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size * 3, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      vehicle.set_autopilot()
      return True
    return False

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
    return False

  # def _try_spawn_ego_vehicle_at(self, transform):
  #   """Try to spawn the ego vehicle at specific transform.
  #   Args:
  #     transform: the carla transform object.
  #   Returns:
  #     Bool indicating whether the spawn is successful.
  #   """
  #   vehicle = None
  #   # Check if ego position overlaps with surrounding vehicles
  #   overlap = False
  #   for idx, poly in self.vehicle_polygons[-1].items():
  #     poly_center = np.mean(poly, axis=0)
  #     ego_center = np.array([transform.location.x, transform.location.y])
  #     dis = np.linalg.norm(poly_center - ego_center)
  #     if dis > 8:
  #       continue
  #     else:
  #       overlap = True
  #       break
  #
  #   if not overlap:
  #     vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
  #
  #   if vehicle is not None:
  #     self.ego=vehicle
  #     return True
  #
  #   return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  # def _get_obs(self):
  #   """Get the observations."""
  #   ## Birdeye rendering
  #   self.birdeye_render.vehicle_polygons = self.vehicle_polygons
  #   self.birdeye_render.walker_polygons = self.walker_polygons
  #   self.birdeye_render.waypoints = self.waypoints
  #
  #   # birdeye view with roadmap and actors
  #   birdeye_render_types = ['roadmap', 'actors']
  #   if self.display_route:
  #     birdeye_render_types.append('waypoints')
  #   self.birdeye_render.render(self.display, birdeye_render_types)
  #   birdeye = pygame.surfarray.array3d(self.display)
  #   birdeye = birdeye[0:self.display_size, :, :]
  #   birdeye = display_to_rgb(birdeye, self.obs_size)
  #
  #   # Roadmap
  #   if self.pixor:
  #     roadmap_render_types = ['roadmap']
  #     if self.display_route:
  #       roadmap_render_types.append('waypoints')
  #     self.birdeye_render.render(self.display, roadmap_render_types)
  #     roadmap = pygame.surfarray.array3d(self.display)
  #     roadmap = roadmap[0:self.display_size, :, :]
  #     roadmap = display_to_rgb(roadmap, self.obs_size)
  #     # Add ego vehicle
  #     for i in range(self.obs_size):
  #       for j in range(self.obs_size):
  #         if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
  #           roadmap[i, j, :] = birdeye[i, j, :]
  #
  #   # Display birdeye image
  #   birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
  #   self.display.blit(birdeye_surface, (0, 0))
  #
  #   ## Lidar image generation (DISABLED FOR FPS OPTIMIZATION)
  #   lidar = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
  #
  #   # Fill the middle PyGame window with black so it doesn't crash
  #   lidar_surface = rgb_to_display_surface(lidar, self.display_size)
  #   self.display.blit(lidar_surface, (self.display_size, 0))
  #   ## Display camera image
  #   camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
  #   camera_surface = rgb_to_display_surface(camera, self.display_size)
  #   self.display.blit(camera_surface, (self.display_size * 2, 0))
  #
  #   # Display on pygame
  #   pygame.display.flip()
  #
  #   # State observation
  #   ego_trans = self.ego.get_transform()
  #   ego_x = ego_trans.location.x
  #   ego_y = ego_trans.location.y
  #   ego_yaw = ego_trans.rotation.yaw/180*np.pi
  #   lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
  #   delta_yaw = np.arcsin(np.cross(w,
  #     np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
  #   v = self.ego.get_velocity()
  #   speed = np.sqrt(v.x**2 + v.y**2)
  #   # state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])
  #   # NEW: Normalize the radar into a "Danger" metric (0.0 = Safe, 1.0 = Crash)
  #   # self.vehicle_front is between 0.0 and 50.0 meters.
  #   radar_danger = (50.0 - self.vehicle_front) / 50.0
  #
  #   # Inject the normalized danger into the brain
  #   state = np.array([lateral_dis, - delta_yaw, speed, radar_danger])
  #   if self.pixor:
  #     ## Vehicle classification and regression maps (requires further normalization)
  #     vh_clas = np.zeros((self.pixor_size, self.pixor_size))
  #     vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))
  #
  #     # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
  #     # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
  #     for actor in self.world.get_actors().filter('vehicle.*'):
  #       x, y, yaw, l, w = get_info(actor)
  #       x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
  #       if actor.id != self.ego.id:
  #         if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
  #           x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
  #             local_info=(x_local, y_local, yaw_local, l, w),
  #             d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
  #           cos_t = np.cos(yaw_pixel)
  #           sin_t = np.sin(yaw_pixel)
  #           logw = np.log(w_pixel)
  #           logl = np.log(l_pixel)
  #           pixels = get_pixels_inside_vehicle(
  #             pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
  #             pixel_grid=self.pixel_grid)
  #           for pixel in pixels:
  #             vh_clas[pixel[0], pixel[1]] = 1
  #             dx = x_pixel - pixel[0]
  #             dy = y_pixel - pixel[1]
  #             vh_regr[pixel[0], pixel[1], :] = np.array(
  #               [cos_t, sin_t, dx, dy, logw, logl])
  #
  #     # Flip the image matrix so that the origin is at the left-bottom
  #     vh_clas = np.flip(vh_clas, axis=0)
  #     vh_regr = np.flip(vh_regr, axis=0)
  #
  #     # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
  #     pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]
  #
  #   obs = {
  #     'camera':camera.astype(np.uint8),
  #     'lidar':lidar.astype(np.uint8),
  #     'birdeye':birdeye.astype(np.uint8),
  #     'state': state,
  #   }
  #
  #   if self.pixor:
  #     obs.update({
  #       'roadmap':roadmap.astype(np.uint8),
  #       'vh_clas':np.expand_dims(vh_clas, -1).astype(np.float32),
  #       'vh_regr':vh_regr.astype(np.float32),
  #       'pixor_state': pixor_state,
  #     })
  #
  #   return obs
  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering (Pointed at Platoon Leader: Agent 0)
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints_list[0]  # FIX: Use leader's waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # Roadmap
    if self.pixor:
      roadmap_render_types = ['roadmap']
      if self.display_route:
        roadmap_render_types.append('waypoints')
      self.birdeye_render.render(self.display, roadmap_render_types)
      roadmap = pygame.surfarray.array3d(self.display)
      roadmap = roadmap[0:self.display_size, :, :]
      roadmap = display_to_rgb(roadmap, self.obs_size)
      # Add ego vehicle
      for i in range(self.obs_size):
        for j in range(self.obs_size):
          if abs(birdeye[i, j, 0] - 255) < 20 and abs(birdeye[i, j, 1] - 0) < 20 and abs(birdeye[i, j, 0] - 255) < 20:
            roadmap[i, j, :] = birdeye[i, j, :]

    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    ## Lidar image generation (DISABLED FOR FPS OPTIMIZATION)
    lidar = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    # Fill the middle PyGame window with black so it doesn't crash
    lidar_surface = rgb_to_display_surface(lidar, self.display_size)
    self.display.blit(lidar_surface, (self.display_size, 0))
    ## Display camera image
    camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
    camera_surface = rgb_to_display_surface(camera, self.display_size)
    self.display.blit(camera_surface, (self.display_size * 2, 0))

    # Display on pygame
    pygame.display.flip()

    # ==============================================================================
    # FIX: MULTI-AGENT STATE MATRIX LOOP WITH EUCLIDEAN RADAR
    # ==============================================================================
    state_matrix = np.zeros((self.num_agents, 4), dtype=np.float32)

    for i in range(self.num_agents):
      ego_trans = self.egos[i].get_transform()
      ego_x = ego_trans.location.x
      ego_y = ego_trans.location.y
      ego_yaw = ego_trans.rotation.yaw / 180 * np.pi

      lateral_dis, w = get_preview_lane_dis(self.waypoints_list[i], ego_x, ego_y)
      delta_yaw = np.arcsin(np.cross(w, np.array([np.cos(ego_yaw), np.sin(ego_yaw)])))

      v = self.egos[i].get_velocity()
      speed = np.sqrt(v.x ** 2 + v.y ** 2)

      # BULLETPROOF RADAR: Calculate Euclidean distance to the exact car ahead
      if i == 0:
        front_dist = 50.0  # Agent 0 (Pace Car) has open road
      else:
        front_loc = self.egos[i - 1].get_transform().location
        ego_loc = self.egos[i].get_transform().location
        # Distance formula minus ~2.5 meters to account for the car's length
        front_dist = np.sqrt((front_loc.x - ego_loc.x) ** 2 + (front_loc.y - ego_loc.y) ** 2) - 2.5
        front_dist = np.clip(front_dist, 0.0, 50.0)

      # Normalize the radar into a "Danger" metric (0.0 = Safe, 1.0 = Crash)
      radar_danger = (50.0 - front_dist) / 50.0

      # Inject the normalized danger into the specific agent's state
      state_matrix[i] = [lateral_dis, - delta_yaw, speed, radar_danger]
    # ==============================================================================

    if self.pixor:
      ## Vehicle classification and regression maps (requires further normalization)
      vh_clas = np.zeros((self.pixor_size, self.pixor_size))
      vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

      # Use Leader's position for Pixor to avoid crashes
      leader_trans = self.egos[0].get_transform()
      l_x = leader_trans.location.x
      l_y = leader_trans.location.y
      l_yaw = leader_trans.rotation.yaw / 180 * np.pi
      l_v = self.egos[0].get_velocity()
      l_speed = np.sqrt(l_v.x ** 2 + l_v.y ** 2)

      # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
      # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
      for actor in self.world.get_actors().filter('vehicle.*'):
        x, y, yaw, l, w = get_info(actor)
        x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (l_x, l_y, l_yaw))
        if actor.id != self.egos[0].id:  # FIX: Ignore leader
          if abs(y_local) < self.obs_range / 2 + 1 and x_local < self.obs_range - self.d_behind + 1 and x_local > -self.d_behind - 1:
            x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
              local_info=(x_local, y_local, yaw_local, l, w),
              d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
            cos_t = np.cos(yaw_pixel)
            sin_t = np.sin(yaw_pixel)
            logw = np.log(w_pixel)
            logl = np.log(l_pixel)
            pixels = get_pixels_inside_vehicle(
              pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
              pixel_grid=self.pixel_grid)
            for pixel in pixels:
              vh_clas[pixel[0], pixel[1]] = 1
              dx = x_pixel - pixel[0]
              dy = y_pixel - pixel[1]
              vh_regr[pixel[0], pixel[1], :] = np.array(
                [cos_t, sin_t, dx, dy, logw, logl])

      # Flip the image matrix so that the origin is at the left-bottom
      vh_clas = np.flip(vh_clas, axis=0)
      vh_regr = np.flip(vh_regr, axis=0)

      # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
      pixor_state = [l_x, l_y, np.cos(l_yaw), np.sin(l_yaw), l_speed]

    obs = {
      'camera': camera.astype(np.uint8),
      'lidar': lidar.astype(np.uint8),
      'birdeye': birdeye.astype(np.uint8),
      'state': state_matrix,  # FIX: Return the matrix of all 6 agents!
    }

    if self.pixor:
      obs.update({
        'roadmap': roadmap.astype(np.uint8),
        'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
        'vh_regr': vh_regr.astype(np.float32),
        'pixor_state': pixor_state,
      })

    return obs
  # def _get_reward(self):
  #   """Calculate the step reward."""
  #   # reward for speed tracking
  #   v = self.ego.get_velocity()
  #   speed = np.sqrt(v.x**2 + v.y**2)
  #   r_speed = -abs(speed - self.desired_speed)
  #
  #   # reward for collision
  #   r_collision = 0
  #   if len(self.collision_hist) > 0:
  #     r_collision = -1
  #
  #   # reward for steering:
  #   r_steer = -self.ego.get_control().steer**2
  #
  #   # reward for out of lane
  #   ego_x, ego_y = get_pos(self.ego)
  #   dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
  #   r_out = 0
  #   if abs(dis) > self.out_lane_thres:
  #     r_out = -10  # Here it was -1 so Increased penalty for actually crossing the line
  #
  #   # NEW: Continuous penalty for wandering from the center
  #   r_center = -abs(dis)
  #
  #   # longitudinal speed
  #   lspeed = np.array([v.x, v.y])
  #   lspeed_lon = np.dot(lspeed, w)
  #
  #   # cost for too fast
  #   r_fast = 0
  #   if lspeed_lon > self.desired_speed:
  #     r_fast = -1
  #
  #   # cost for lateral acceleration
  #   r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2
  #
  #   # r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1
  #   # Increased r_steer weight from 5 to 20 to FORCE smooth steering
  #   # Added r_center to force the car to stay in the middle of the lane
  #   # r = 200 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 20 + 0.2 * r_lat + r_center - 0.1
  #   # radar_danger = self._get_obs()['state'][3]
  #   radar_danger = (50.0 - self.vehicle_front) / 50.0
  #   r_tailgate = 0
  #   if radar_danger > 0.6:  # If the car is less than 20 meters away
  #     # The closer it gets without braking, the more it burns points!
  #     r_tailgate = -20.0 * radar_danger
  #
  #   # Add r_tailgate to the final equation
  #   # r = 5000 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 20 + 0.2 * r_lat + r_center + r_tailgate - 0.1
  #   r = 10000 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 20 + 0.2 * r_lat + r_center + r_tailgate - 0.1
  #   return r

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking (Judged based on the Platoon Leader)
    v = self.egos[0].get_velocity()
    speed = np.sqrt(v.x ** 2 + v.y ** 2)
    r_speed = -abs(speed - self.desired_speed)

    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    # reward for steering:
    r_steer = -self.egos[0].get_control().steer ** 2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.egos[0])
    dis, w = get_lane_dis(self.waypoints_list[0], ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -10  # Increased penalty for actually crossing the line

    # Continuous penalty for wandering from the center
    r_center = -abs(dis)

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1

    # cost for lateral acceleration
    r_lat = - abs(self.egos[0].get_control().steer) * lspeed_lon ** 2

    radar_danger = (50.0 - self.vehicle_fronts[0]) / 50.0
    r_tailgate = 0
    if radar_danger > 0.6:  # If the car is less than 20 meters away
      # The closer it gets without braking, the more it burns points!
      r_tailgate = -20.0 * radar_danger

    # Final equation
    r = 10000 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 20 + 0.2 * r_lat + r_center + r_tailgate - 0.1
    return r
  # def _terminal(self):
  #   """Calculate whether to terminate the current episode."""
  #   # Get ego state
  #   ego_x, ego_y = get_pos(self.ego)
  #
  #   # If collides
  #   if len(self.collision_hist)>0:
  #     return True
  #   #
  #   # # If reach maximum timestep
  #   # if self.time_step>self.max_time_episode:
  #   #   return True
  #
  #   # If at destination
  #   if self.dests is not None: # If at destination
  #     for dest in self.dests:
  #       if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
  #         return True
  #
  #   # If out of lane
  #   dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
  #   if abs(dis) > self.out_lane_thres:
  #     return True
  #
  #   return False
  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # 1. If the Platoon Leader collides (Sensor is on Agent 0)
    if len(self.collision_hist) > 0:
      return True

    # 2. Loop through all 5 agents to check for lane departures or destinations
    for i in range(self.num_agents):
      ego_x, ego_y = get_pos(self.egos[i])

      # If at destination
      if self.dests is not None:
        for dest in self.dests:
          if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
            return True

      # If ANY of the 5 cars goes out of lane, terminate the episode!
      dis, _ = get_lane_dis(self.waypoints_list[i], ego_x, ego_y)
      if abs(dis) > self.out_lane_thres:
        return True

    return False
  
  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
