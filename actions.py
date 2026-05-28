import rospy
from jetarm_sdk import bus_servo_control

# ---- Original garbage sorting actions ----

def goto_left(pub, duration=1.5):
    bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 875), (2, 610), (3, 70), (4, 140), (5, 500), (10, 200)))
    rospy.sleep(duration)

def goto_right(pub, duration=1.5):
    bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 125), (2, 610), (3, 70), (4, 140), (5, 500), (10, 200)))
    rospy.sleep(duration)
   
def go_home(pub, duration=0.8):
    #bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 500), (2, 610), (3, 70), (4, 140), (5, 500), (10, 200)))
    bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))
    #bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 500), (2, 500), (3, 125), (4, 125), (5, 500), (10, 200)))
    rospy.sleep(duration)

def goto_default(pub, duration=1.5):
    go_home(pub, duration)

def place(pub, duration=1.5):
    bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 500), (2, 610), (3, 70), (4, 140), (5, 500), (10, 200)))

def go_back(pub, duration=0.8):
    bus_servo_control.set_servos(pub, int(duration * 1000), ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 550)))
    rospy.sleep(duration)

# ---- Feeding project actions ----

def goto_pick_food(pub, duration=1.5):
    """Move arm to the food pick position (right side, fixed location)."""
    bus_servo_control.set_servos(pub, int(duration * 1000),
        ((1, 125), (2, 610), (3, 70), (4, 140), (5, 500), (10, 200)))
    rospy.sleep(duration)

def goto_detect_pose(pub, duration=1.5):
    """Move arm to face-detection pose (camera facing forward toward user)."""
    bus_servo_control.set_servos(pub, int(duration * 1000),
        ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))
    rospy.sleep(duration)

def open_gripper(pub, duration=0.5):
    """Fully open gripper to release food."""
    bus_servo_control.set_servos(pub, int(duration * 1000), ((10, 580),))
    rospy.sleep(duration)

def close_gripper(pub, duration=0.5):
    """Close gripper to grasp food."""
    bus_servo_control.set_servos(pub, int(duration * 1000), ((10, 220),))
    rospy.sleep(duration)
