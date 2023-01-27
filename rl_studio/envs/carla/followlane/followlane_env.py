from cv_bridge import CvBridge
from logger import logger
from rl_studio.envs.carla.carla_env import CarlaEnv
import carla
import time

class FollowLaneEnv(CarlaEnv):

    def __init__(self, **config):
        """ Constructor of the class. """
        
        # init F1env
        CarlaEnv.__init__(self, **config)
        self.data = {}
        self.pose3D_data = None
        self.recording = False
        self.cvbridge = CvBridge()

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        self.world = client.get_world()
        time.sleep(5)
        self.carla_map = self.world.get_map()
        while len(self.world.get_actors().filter('vehicle.*')) == 0:
            logger.info("Waiting for vehicles!")
            time.sleep(1)
        self.ego_vehicle = self.world.get_actors().filter('vehicle.*')[0]
        self.map_waypoints = self.carla_map.generate_waypoints(0.5)
        self.weather = self.world.get_weather()