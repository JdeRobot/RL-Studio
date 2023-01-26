from enum import Enum


class TasksType(Enum):
    FOLLOWLINEGAZEBO = "follow_line_gazebo"
    FOLLOWLANEGAZEBO = "follow_lane_gazebo"
    FOLLOWLANECARLA = "follow_lane_carla"
    AUTOPARKINGGAZEBO = "autoparking_gazebo"
