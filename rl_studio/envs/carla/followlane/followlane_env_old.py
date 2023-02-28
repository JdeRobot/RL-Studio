import glob
import os
import random
import sys
import time
import weakref

import carla
from cv_bridge import CvBridge
import cv2
import numpy as np
import pygame

from rl_studio.envs.carla.carla_env import CarlaEnv
from rl_studio.envs.carla.utils.bounding_boxes import ClientSideBoundingBoxes
from rl_studio.envs.carla.utils.logger import logger
from rl_studio.envs.carla.utils.weather import Weather
from rl_studio.envs.carla.utils.environment import apply_sun_presets, apply_weather_presets, apply_weather_values, apply_lights_to_cars, apply_lights_manager, get_args


VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

class FollowLaneEnv(CarlaEnv):

    def __init__(self, **config):
        """ Constructor of the class. """
        
        CarlaEnv.__init__(self, **config)
        print(f"{config}\n")
        #self.actor_list = []
        #self.carla_map = None
        #self.client = None
        #self.world = None
        self.vehicle = None
        self.camera = None
        self.blueprint_library = self.get_blueprint_library()
        #self.vehicle = self.blueprint_library.filter(config["car"])[0]
        self.car_model = random.choice(self.blueprint_library.filter("vehicle.*.*"))
        self.height_image = config['height_image']
        self.width_image = config['width_image']
        self.vehicles = None

        self.display = None
        self.image = None
        self.capture = True

        #-- weather    
        self.set_carlaclient_world_map(config)
        if config["weather"] == "dynamic":
            self.weather = Weather(self.world.get_weather())
        elif config["weather"] == "argparse":
            self.set_arg_weather()
        else:
            self.set_fix_weather()

        #TODO: lights: street and cars
        self.set_street_lights()
        self.set_car_lights()

        #TODO: get random cars and pedestrians

        #TODO: get snapshots for a particular actor
        self.get_snapshot()

        #TODO: Debugging mode: draw boxes
        self.set_synchronous_mode(True)

        pygame.init()
        self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()







    def set_carlaclient_world_map(self, config):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0) # seconds
        self.world = self.client.load_world(config['town'])
        self.carla_map = self.world.get_map()        
        
  
    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.05 #simulator takes 1/0.1 steps, i.e. 10 steps
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)

    def set_fix_weather(self):
        '''
        ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, 
        MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, 
        WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, 
        HardRainSunset.
        '''
        #TODO: carla.WeatherParameters is a enum. Every time weather is changing
        # for i in enum(carla.WeatherParameters )
        #     weather.tick(speed_factor * elapsed_time)
        #     world.set_weather(weather.weather)

        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def set_arg_weather(self, dynamic=None):
        """
        setting weather through params in utils.get_args()
        """
        self.weather = self.world.get_weather()
        args = get_args()
        # apply presets
        apply_sun_presets(args, self.weather)
        apply_weather_presets(args, self.weather)
        # apply weather values individually
        apply_weather_values(args, self.weather)

        self.world.set_weather(self.weather)

    def set_street_lights(self):
        pass

    def set_car_lights(self):
        pass






        #---------- hasta aqui ----------

        #self.ego_vehicle = self.world.get_actors().filter('vehicle.*')[0]
        #self.map_waypoints = self.carla_map.generate_waypoints(0.5)
        #self.weather = self.world.get_weather()
        ###### Waypoints
        #waypoints = self.world.get_map().generate_waypoints(distance=2.0)
        #waypoints_list = self.get_waypoints(waypoints, road_id=3, life_time=200)
        ###### Waypoint Target
        #self.get_target_waypoint(waypoints_list[50], life_time=200)
        ###### Car
        #self.setup_car(waypoints_list[0])
        #self.actor_list.append(self.car)
        # Camera
        #self.setup_camera()
        #self.actor_list.append(self.setup_camera())
        # PYgame
        #self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        #pygame_clock = pygame.time.Clock()
        #vehicles = self.world.get_actors().filter('vehicle.*')


    def reset_HARRISON(self):
        print(f"\nin reset()\n")
        self.collision_hist = []
        self.actor_list = []

        #-- vehicle random pose
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.car_model, self.transform)
        self.actor_list.append(self.vehicle)

        #-- camera definition
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.width_image}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.height_image}")
        self.rgb_cam.set_attribute("fov", f"110")

        #-- camera attach
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor_camera = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor_camera)
        self.sensor_camera.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        #-- collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.rgb_cam is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.rgb_cam        

    def reset(self):
        print(f"\nin reset()\n")
        self.collision_hist = []
        self.actor_list = []

        #-- vehicle random pose
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.car_model, self.transform)
        self.actor_list.append(self.vehicle)

        #-- camera definition
        #self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        #self.rgb_cam.set_attribute("image_size_x", f"{self.width_image}")
        #self.rgb_cam.set_attribute("image_size_y", f"{self.height_image}")
        #self.rgb_cam.set_attribute("fov", f"110")

        #-- camera attach
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        #self.sensor_camera = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        #self.actor_list.append(self.sensor_camera)
        #self.sensor_camera.listen(lambda data: self.process_img(data))

        self.camera = self.setup_camera()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        #-- collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        self.pygame_clock.tick_busy_loop(20)

        self.render(self.display)

        self.vehicles = self.world.get_actors().filter('vehicle.*')
        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles, self.camera)
        ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
        pygame.display.flip()
        pygame.event.pump()


        return self.camera  


    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        return self.camera


    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))



    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if True:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3


        ##### Waypoints
        #waypoints = self.world.get_map().generate_waypoints(distance=2.0)
        #waypoints_list = self.get_waypoints(waypoints, road_id=3, life_time=200)

        ###### Waypoint Target
        #self.get_target_waypoint(waypoints_list[50], life_time=200)

        ###### Car
        #self.setup_car(waypoints_list[0])
        #actor_list.append(self.car)

        # Camera
        #self.setup_camera()
        #actor_list.append(self.setup_camera())
        #blueprint_library: carla.BlueprintLibrary = self.get_blueprint_library()
        #model = blueprint_library.filter("model3")[0]
        #cam = blueprint_library.find("sensor.camera.rgb")
        #colsensor = blueprint_library.find('sensor.other.collision')
        #self.spawn_vehicle(model, cam, colsensor)




    def get_blueprint_library(self):
        return self.world.get_blueprint_library()     
        
    def destroy_all_actors(self):
        for actor in self._actors:
            actor.destroy()
        self._actors = []
  
    def get_snapshot(self):
        '''TODO: get snapshots for a particular actor'''
        pass

    def recorder_file(self, path, name, recorder=False):
        '''TODO: recorder training or inference...
        maybe it can be launch from trainingFollowlane
        '''
        if recorder is True:
            file_name = f"{path}/{name}"
            self.client.start_recorder(file_name, True)







    def run_simulation(self):
        """
        Main program loop.
        """
        actor_list = []
        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            #self.world = self.client.get_world()
            self.world = self.client.load_world('Town02')


            ###### Waypoints
            waypoints = self.world.get_map().generate_waypoints(distance=2.0)
            waypoints_list = self.get_waypoints(waypoints, road_id=3, life_time=200)

            ###### Waypoint Target
            self.get_target_waypoint(waypoints_list[50], life_time=200)

            ###### Car
            self.setup_car(waypoints_list[0])
            actor_list.append(self.car)

            # Camera
            self.setup_camera()
            actor_list.append(self.setup_camera())
            
            # PYgame
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()
            
            # syncronous mode
            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')

            while True:        
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)

                pygame.display.flip()

                pygame.event.pump()
                #if self.control(self.car):
                #    return

        finally:
            self.set_synchronous_mode(False)
            #self.camera.destroy()
            #self.car.destroy()
            pygame.quit()
            for actor in actor_list:
                print(f"actor:{actor}")
                actor.destroy()
            #self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


    def get_waypoints(self, waypoints, road_id=None, life_time=100.0):
        '''
        Retrieve waypoints from server in a desirable road and drawing them
        '''
        filtered_waypoints = []
        for waypoint in waypoints:

            if(waypoint.road_id == road_id):
                # added
                filtered_waypoints.append(waypoint)
                # draw them
                self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                    color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                    persistent_lines=True)

        return filtered_waypoints

    def get_target_waypoint(self, target_waypoint, life_time=100.0):
        '''
        draw target point
        '''
        self.world.debug.draw_string(target_waypoint.transform.location, 'O', draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                    persistent_lines=True)

    def setup_random_pose_and_car(self):
        """
        Spawns a random actor-vehicle to be controled in a random position inside town.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_fix_pose_car(self):
        """
        Spawns an actor-vehicle in always a fix position
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = carla.Transform(carla.Location(x=-14.130021, y=69.766624, z=4), carla.Rotation(pitch=360.000000, yaw=0.073273, roll=0.000000))
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_waypoint_related_pose_car(self, waypoints_list):
        """
        Spawns actor-vehicle to be controled in a certain waypoint.
        """

        #car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        #car_bp = self.world.get_blueprint_library().filter('mini2021')[0]
        #car_bp = self.world.get_blueprint_library().filter('vehicle.citroen.c3')[0]
        car_bp = self.world.get_blueprint_library().filter('vehicle.jeep.wrangler_rubicon')[0]

        spawn_point = waypoints_list.transform
        spawn_point.location.z += 2
        #vehicle = client.get_world().spawn_actor(vehicle_blueprint, spawn_point)

        #location = random.choice(self.world.get_map().get_spawn_points())
        #location = self.world.get_map().get_spawn_points()
        #location = carla.Transform(carla.Location(x=-14.130021, y=69.766624, z=4), carla.Rotation(pitch=360.000000, yaw=0.073273, roll=0.000000))
        #print(f"location:{location}")
        self.car = self.world.spawn_actor(car_bp, spawn_point)

        # Retrieve the closest waypoint.
        #waypoint = self.world.get_map().get_waypoint(self.car.get_location())
        #print(f"waypoint:{waypoint}")

        '''
        waypoint:Waypoint(Transform(Location(x=14.130021, y=69.766624, z=0.000000), Rotation(pitch=360.000000, yaw=0.073273, roll=0.000000)))
        waypoint:Waypoint(Transform(Location(x=-0.036634, y=13.183878, z=0.000000), Rotation(pitch=360.000000, yaw=180.159195, roll=0.000000)))
        '''
        # Disable physics, in this example we're just teleporting the vehicle.
        #self.car.set_simulate_physics(False)
        # Find next waypoint 2 meters ahead.
        #waypoint = random.choice(waypoint.next(2.0))
        # Teleport the vehicle.
        #self.car.set_transform(waypoint.transform)

        #waypoint_list = self.world.get_map().generate_waypoints(2.0)
        #waypoint_tuple_list = self.world.get_map().get_topology()
        #print(f"waypoint_list:{waypoint_list[0]}")
        #print(f"waypoint_tuple_list:{waypoint_tuple_list[0]}")

        # returns car to add in destroy list
        
        
        #return self.car


    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp



         