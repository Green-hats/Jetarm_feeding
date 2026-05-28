#!/usr/bin/env python3
# coding: utf8
# Interactive calibration tool for detect_arm_joints
# Adjust servo angles with keyboard and find the best face-detection pose.
#
# Keys:
#   1/2/3/4/5/0  - select servo to adjust (0 = gripper)
#   w/s          - increase/decrease selected servo angle by 10
#   W/S          - increase/decrease selected servo angle by 50
#   h            - go home
#   q            - print current angles and quit

import rospy
import sys
from jetarm_sdk import bus_servo_control
from hiwonder_interfaces.msg import MultiRawIdPosDur

SERVO_NAMES = {1: "base", 2: "shoulder", 3: "elbow", 4: "wrist1", 5: "wrist2", 10: "gripper"}
SERVO_IDS = [1, 2, 3, 4, 5, 10]

rospy.init_node("calibrate_detect_pose", anonymous=False)
pub = rospy.Publisher("/controllers/multi_id_pos_dur", MultiRawIdPosDur, queue_size=1)

# Current angles (start from default detect pose)
angles = {1: 500, 2: 560, 3: 130, 4: 115, 5: 500, 10: 200}
current_servo = 1


def print_state():
    sys.stdout.write("\033[H\033[J")
    print("=== Detect Pose Calibration ===")
    print("  Servo %d (%s) selected (keys 1/2/3/4/5/0 to switch)" % (current_servo, SERVO_NAMES[current_servo if current_servo != 0 else 10]))
    print("  w/s: +/-10   W/S: +/-50   h: home   q: quit")
    print()
    for sid in SERVO_IDS:
        marker = " <--" if sid == (current_servo if current_servo != 0 else 10) else ""
        print("  [%d] %-10s  %4d%s" % (sid, SERVO_NAMES[sid], angles[sid], marker))
    print()
    print("Copy this to config_feed.yaml detect_arm_joints:")
    pairs = ", ".join("(%d, %d)" % (sid, angles[sid]) for sid in SERVO_IDS)
    print("  (%s)" % pairs)
    print()
    print("Watching camera feed? Run: python3 feed_main.py (no start_feeding needed, just watch)")
    sys.stdout.flush()


def move_all():
    pos_list = tuple((sid, angles[sid]) for sid in SERVO_IDS)
    bus_servo_control.set_servos(pub, 500, pos_list)


print_state()
move_all()

import termios, tty, select


def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


while not rospy.is_shutdown():
    key = get_key()
    if key is None:
        continue

    if key == 'q':
        print()
        print("Final config for detect_arm_joints:")
        pairs = ", ".join("(%d, %d)" % (sid, angles[sid]) for sid in SERVO_IDS)
        print("  (%s)" % pairs)
        break
    elif key == 'h':
        angles = {1: 500, 2: 560, 3: 130, 4: 115, 5: 500, 10: 200}
        move_all()
        print_state()
    elif key in ('1', '2', '3', '4', '5'):
        current_servo = int(key)
        print_state()
    elif key == '0':
        current_servo = 0
        print_state()
    elif key == 'w':
        sid = current_servo if current_servo != 0 else 10
        if sid == 17:
            angles[17] = min(1000, angles.get(17, 500) + 10)
        else:
            angles[sid] = min(1000, angles[sid] + 10)
        move_all()
        print_state()
    elif key == 's':
        sid = current_servo if current_servo != 0 else 10
        if sid == 17:
            angles[17] = max(0, angles.get(17, 500) - 10)
        else:
            angles[sid] = max(0, angles[sid] - 10)
        move_all()
        print_state()
    elif key == 'W':
        sid = current_servo if current_servo != 0 else 10
        angles[sid] = min(1000, angles[sid] + 50)
        move_all()
        print_state()
    elif key == 'S':
        sid = current_servo if current_servo != 0 else 10
        angles[sid] = max(0, angles[sid] - 50)
        move_all()
        print_state()
