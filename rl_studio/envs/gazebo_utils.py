import random

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


def set_new_pose(self):
    """
    (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
    """
    pos = random.choice(list(enumerate(self.circuit["gaz_pos"])))[0]
    self.position = pos

    pos_number = self.circuit["gaz_pos"][0]

    state = ModelState()
    state.model_name = "f1_renault"
    state.pose.position.x = self.circuit["gaz_pos"][pos][1]
    state.pose.position.y = self.circuit["gaz_pos"][pos][2]
    state.pose.position.z = self.circuit["gaz_pos"][pos][3]
    state.pose.orientation.x = self.circuit["gaz_pos"][pos][4]
    state.pose.orientation.y = self.circuit["gaz_pos"][pos][5]
    state.pose.orientation.z = self.circuit["gaz_pos"][pos][6]
    state.pose.orientation.w = self.circuit["gaz_pos"][pos][7]

    rospy.wait_for_service("/gazebo/set_model_state")
    try:
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        set_state(state)
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))
    return pos_number
