
from datetime import datetime
import math
import os
import subprocess
import threading
import time
import sys

import carla
from cv_bridge import CvBridge
import cv2
import gymnasium as gym
import json
from PIL import Image
import rosbag
import rospy
import shlex

from rl_studio.envs.carla.utils.logger import logger


class CarlaEnv(gym.Env):

    def __init__(self, **config):
        """ Constructor of the class. """
        # close previous instances of ROS and simulators if hanged.
        self.close_ros_and_simulators()
        #print(f"{os.environ=}\n")
        try:
            carla_root = os.environ["CARLA_ROOT"]
            #print(f"{carla_root = }\n")
            carla_exec = f"{carla_root}/CarlaUE4.sh"
        except KeyError as oe:
            logger.error("CarlaEnv: exception raised searching CARLA_ROOT env variable. {}".format(oe))
                

        try:
            with open("/tmp/.carlalaunch_stdout.log", "w") as out, open("/tmp/.carlalaunch_stderr.log", "w") as err:
                    subprocess.Popen([carla_exec, "-prefernvidia"], stdout=out, stderr=err)
                    #subprocess.Popen(["/home/jderobot/Documents/Projects/carla_simulator_0_9_13/CarlaUE4.sh", "-RenderOffScreen"], stdout=out, stderr=err)
                    #subprocess.Popen(["/home/jderobot/Documents/Projects/carla_simulator_0_9_13/CarlaUE4.sh", "-RenderOffScreen", "-quality-level=Low"], stdout=out, stderr=err)
            logger.info("SimulatorEnv: launching simulator server.")
            time.sleep(5)
            #with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
            #    child = subprocess.Popen(["roslaunch", launch_file], stdout=out, stderr=err)
            #logger.info("SimulatorEnv: launching simulator server.")
        except OSError as oe:
            logger.error("SimulatorEnv: exception raised launching simulator server. {}".format(oe))
            self.close_ros_and_simulators()
            sys.exit(-1)

        # give simulator some time to initialize
        time.sleep(5)


    def close_ros_and_simulators(self):
        """Kill all the simulators and ROS processes."""
        try:
            ps_output = subprocess.check_output(["ps", "-Af"]).decode('utf-8').strip("\n")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing ps command {}".format(ce))
            sys.exit(-1)

        if ps_output.count('gzclient') > 0:
            try:
                subprocess.check_call(["killall", "-9", "gzclient"])
                logger.debug("SimulatorEnv: gzclient killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for gzclient {}".format(ce))

        if ps_output.count('gzserver') > 0:
            try:
                subprocess.check_call(["killall", "-9", "gzserver"])
                logger.debug("SimulatorEnv: gzserver killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for gzserver {}".format(ce))

        if ps_output.count('CarlaUE4.sh') > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4.sh"])
                logger.debug("SimulatorEnv: CARLA server killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for CARLA server {}".format(ce))

        if ps_output.count('CarlaUE4-Linux-Shipping') > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-Shipping"])
                logger.debug("SimulatorEnv: CarlaUE4-Linux-Shipping killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux-Shipping {}".format(ce))

        if ps_output.count('CarlaUE4-Linux-') > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-"])
                logger.debug("SimulatorEnv: CarlaUE4-Linux- killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux- {}".format(ce))

        if ps_output.count('rosout') > 0:
            try:
                import rosnode
                for node in rosnode.get_node_names():
                    if node != '/carla_ros_bridge':
                        subprocess.check_call(["rosnode", "kill", node])

                logger.debug("SimulatorEnv:rosout killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for rosout {}".format(ce))
        
        if ps_output.count('bridge.py') > 0:
            try:
                os.system("ps -ef | grep 'bridge.py' | awk '{print $2}' | xargs kill -9")
                logger.debug("SimulatorEnv:bridge.py killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for bridge.py {}".format(ce))
            except FileNotFoundError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for bridge.py {}".format(ce))

        if ps_output.count('rosmaster') > 0:
            try:
                subprocess.check_call(["killall", "-9", "rosmaster"])
                logger.debug("SimulatorEnv: rosmaster killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for rosmaster {}".format(ce))

        if ps_output.count('roscore') > 0:
            try:
                subprocess.check_call(["killall", "-9", "roscore"])
                logger.debug("SimulatorEnv: roscore killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for roscore {}".format(ce))

        if ps_output.count('px4') > 0:
            try:
                subprocess.check_call(["killall", "-9", "px4"])
                logger.debug("SimulatorEnv: px4 killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for px4 {}".format(ce))

        if ps_output.count('roslaunch') > 0:
            try:
                subprocess.check_call(["killall", "-9", "roslaunch"])
                logger.debug("SimulatorEnv: roslaunch killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for roslaunch {}".format(ce))
        
        if ps_output.count('rosout') > 0:
            try:
                subprocess.check_call(["killall", "-9", "rosout"])
                logger.debug("SimulatorEnv:rosout killed.")
            except subprocess.CalledProcessError as ce:
                logger.error("SimulatorEnv: exception raised executing killall command for rosout {}".format(ce))

'''
def is_gzclient_open():
    """Determine if there is an instance of Gazebo GUI running

    Returns:
        bool -- True if there is an instance running, False otherwise
    """

    try:
        ps_output = subprocess.check_output(["ps", "-Af"], encoding='utf8').strip("\n")
    except subprocess.CalledProcessError as ce:
        logger.error("SimulatorEnv: exception raised executing ps command {}".format(ce))
        sys.exit(-1)

    return ps_output.count('gzclient') > 0


def close_gzclient():
    """Close the Gazebo GUI if opened."""

    if is_gzclient_open():
        try:
            subprocess.check_call(["killall", "-9", "gzclient"])
            logger.debug("SimulatorEnv: gzclient killed.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing killall command for gzclient {}".format(ce))


def open_gzclient():
    """Open the Gazebo GUI if not running"""

    if not is_gzclient_open():
        try:
            with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                subprocess.Popen(["gzclient"], stdout=out, stderr=err)
            logger.debug("SimulatorEnv: gzclient started.")
        except subprocess.CalledProcessError as ce:
            logger.error("SimulatorEnv: exception raised executing gzclient {}".format(ce))
'''
