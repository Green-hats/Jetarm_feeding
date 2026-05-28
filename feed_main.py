#!/usr/bin/env python3
# coding: utf8
# Feeding Robot Arm - Main Pipeline Node
# Flow: Idle -> Pick food(right) -> Move to detect pose -> Detect mouth -> Move to mouth -> Release -> Home

import cv2
import rospy
import queue
import numpy as np
import threading
from vision_utils import fps, xyz_quat_to_mat, xyz_euler_to_mat, mat_to_xyz_euler, distance, box_center
from sensor_msgs.msg import Image as RosImage, CameraInfo
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from hiwonder_interfaces.srv import GetRobotPose
from hiwonder_interfaces.msg import MoveAction, MoveGoal, MultiRawIdPosDur
from jetarm_sdk import bus_servo_control
from face_detect import FaceDetector
import actions
import actionlib

CONFIG_NAME = '/config'

# State machine steps
STEP_IDLE = 0
STEP_PICK_FOOD = 1
STEP_APPROACH_DETECT = 2
STEP_DETECT_MOUTH = 3
STEP_MOVE_TO_MOUTH = 4


class FeedingNode:
    def __init__(self, node_name, log_level=rospy.INFO):
        rospy.init_node(node_name, anonymous=False, log_level=log_level)
        self.lock = threading.RLock()
        self.K = None
        self.D = None

        config = rospy.get_param(CONFIG_NAME)
        hand2cam_raw = config.get('hand2cam_tf_matrix')
        if hand2cam_raw is not None:
            self.hand2cam_tf_matrix = np.array(hand2cam_raw, dtype=np.float64)
        else:
            rospy.logerr("hand2cam_tf_matrix not found in config!")
            self.hand2cam_tf_matrix = np.eye(4)

        self.food_pick_pos = np.array(config.get('food_pick_position', [0.15, 0.25, 0.02]), dtype=np.float64)
        self.detect_arm_joints = config.get('detect_arm_joints',
                                             ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))
        self.stability_frames = config.get('stability_frames', 30)
        self.stability_pixels = config.get('stability_pixels', 30)
        self.release_retreat = config.get('release_retreat', 0.06)
        self.pick_pitch = config.get('pick_pitch', 80)
        self.face_model_path = config.get('face_model_path',
                                           '/home/hiwonder/weights/yolov5s-face.pt')

        self.step = STEP_IDLE
        self.target_mouth = None
        self.stability_count = 0
        self.last_mouth_pos = None
        self.finish_percent = 0
        self.image_queue = queue.Queue(maxsize=2)
        self.fps_counter = fps.FPS()

        self.servos_pub = rospy.Publisher("/controllers/multi_id_pos_dur", MultiRawIdPosDur, queue_size=1)

        camera_info_topic = rospy.get_param('~camera_info_topic', '/rgbd_cam/color/camera_info')
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        rospy.loginfo("Loading face detection model: %s", self.face_model_path)
        self.face_detector = FaceDetector(model_path=self.face_model_path, device='cpu')
        rospy.loginfo("Face detector loaded.")

        rospy.sleep(0.5)
        actions.go_home(self.servos_pub)
        self.get_endpoint()

        self.action_client = actionlib.SimpleActionClient('/grasp', MoveAction)

        source_image_topic = rospy.get_param('~source_image_topic', '/rgbd_cam/color/image_rect_color')
        self.image_sub = rospy.Subscriber(source_image_topic, RosImage, self.image_callback, queue_size=1)

        self.trigger_srv = rospy.Service('~start_feeding', Trigger, self.start_feeding_cb)

        rospy.loginfo("Feeding node ready. Call /feeding/start_feeding to begin.")

    def camera_info_callback(self, msg):
        with self.lock:
            K_arr = np.matrix(msg.K).reshape(1, -1, 3)
            D_arr = np.array(msg.D)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K_arr, D_arr, (640, 480), 0, (640, 480))
            self.K, self.D = np.matrix(new_K), np.zeros((5, 1))

    def get_endpoint(self):
        endpoint = rospy.ServiceProxy('/kinematics/get_current_pose', GetRobotPose)().pose
        self.endpoint = xyz_quat_to_mat(
            [endpoint.position.x, endpoint.position.y, endpoint.position.z],
            [endpoint.orientation.w, endpoint.orientation.x, endpoint.orientation.y, endpoint.orientation.z])

    def camera_in_world(self):
        self.get_endpoint()
        return np.matmul(np.array(self.endpoint), self.hand2cam_tf_matrix)

    def pixel_to_world_mouth(self, mouth_center, bbox_width):
        if self.K is None:
            return None
        cam_world = self.camera_in_world()
        K_mat = np.array(self.K)
        depth = self.face_detector.estimate_depth(bbox_width, K_mat[0, 0])
        depth = max(0.05, min(0.30, depth))
        camera_pt = self.face_detector.pixel_to_camera_point(
            mouth_center[0], mouth_center[1], depth, K_mat)
        world_pt = np.matmul(cam_world, camera_pt.reshape(4, 1)).flatten()
        result = world_pt[:3].copy()
        result[0] = max(-0.05, min(0.28, result[0]))
        result[1] = max(-0.25, min(0.25, result[1]))
        result[2] = max(0.02, min(0.30, result[2]))
        rospy.loginfo("depth=%.2f camera_world=(%.2f,%.2f,%.2f) target=(%.3f,%.3f,%.3f)",
                      depth, cam_world[0,3], cam_world[1,3], cam_world[2,3],
                      result[0], result[1], result[2])
        return result

    def start_feeding_cb(self, req):
        success = False
        message = ""
        if self.step == STEP_IDLE:
            threading.Thread(target=self._start_pick_food).start()
            success = True
            message = "Feeding started"
        else:
            message = "Feeding already in progress (step=%d)" % self.step
        return TriggerResponse(success=success, message=message)

    def done_callback(self, state, result):
        rospy.loginfo("action done, state: %d, complete: %s", state, result.result.complete)
        if not result.result.complete:
            bus_servo_control.set_servos(self.servos_pub, 500, ((1, 500),))
            rospy.sleep(1)
            actions.go_home(self.servos_pub)
            bus_servo_control.set_servos(self.servos_pub, 500, ((10, 200),))
            rospy.sleep(0.5)
            self.reset_state()
            return

        if self.finish_percent < 1.0:
            return

        if self.step == STEP_PICK_FOOD:
            rospy.loginfo("Food picked. Moving to detect position...")
            self.step = STEP_APPROACH_DETECT
            threading.Thread(target=self._goto_detect_position).start()

        elif self.step == STEP_MOVE_TO_MOUTH:
            rospy.loginfo("Food delivered. Returning home...")
            actions.go_home(self.servos_pub)
            rospy.sleep(0.5)
            self.reset_state()

    def _start_pick_food(self):
        rospy.loginfo("Picking food at: (%.3f, %.3f, %.3f)",
                       self.food_pick_pos[0], self.food_pick_pos[1], self.food_pick_pos[2])
        self.step = STEP_PICK_FOOD
        goal = MoveGoal()
        goal.grasp.mode = 'pick'
        goal.grasp.position.x = float(self.food_pick_pos[0])
        goal.grasp.position.y = float(self.food_pick_pos[1])
        goal.grasp.position.z = max(float(self.food_pick_pos[2]), 0.005)
        goal.grasp.pitch = self.pick_pitch
        goal.grasp.align_angle = 0
        goal.grasp.grasp_approach.z = 0.03
        goal.grasp.grasp_retreat.z = 0.05
        goal.grasp.grasp_posture = 580
        goal.grasp.pre_grasp_posture = 220
        self.action_client.send_goal(goal, self.done_callback, self.active_callback, self.feedback_callback)

    def _goto_detect_position(self):
        joints = tuple(tuple(pair) for pair in self.detect_arm_joints)
        bus_servo_control.set_servos(self.servos_pub, 1500, joints)
        rospy.sleep(2.0)
        self.get_endpoint()
        self.step = STEP_DETECT_MOUTH
        self.stability_count = 0
        self.last_mouth_pos = None
        rospy.loginfo("Detect position reached. Waiting for mouth detection...")

    def _move_to_mouth(self, world_xyz):
        rospy.loginfo("Moving to mouth: (%.3f, %.3f, %.3f)", *world_xyz)
        self.step = STEP_MOVE_TO_MOUTH
        goal = MoveGoal()
        goal.grasp.mode = 'place'
        goal.grasp.position.x = float(world_xyz[0])
        goal.grasp.position.y = float(world_xyz[1])
        goal.grasp.position.z = max(float(world_xyz[2]), 0.01)
        goal.grasp.pitch = self.pick_pitch
        goal.grasp.align_angle = 0
        goal.grasp.grasp_approach.z = 0.03
        goal.grasp.grasp_retreat.z = self.release_retreat
        goal.grasp.grasp_posture = 370
        goal.grasp.pre_grasp_posture = 580
        self.action_client.send_goal(goal, self.done_callback, self.active_callback, self.feedback_callback)

    def reset_state(self):
        self.step = STEP_IDLE
        self.target_mouth = None
        self.stability_count = 0
        self.last_mouth_pos = None
        bus_servo_control.set_servos(self.servos_pub, 500, ((10, 200),))

    def active_callback(self):
        rospy.loginfo("start move")

    def feedback_callback(self, msg):
        rospy.loginfo("progress: {:.2%}".format(msg.percent))
        self.finish_percent = msg.percent

    def image_process(self):
        ros_image = self.image_queue.get(block=True)
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3),
                               dtype=np.uint8, buffer=ros_image.data)
        result_image = np.copy(rgb_image)

        try:
            if self.step == STEP_DETECT_MOUTH:
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                boxes, landmarks, mouth_centers = self.face_detector.detect(bgr_image)
                if not hasattr(self, '_detect_frame_count'):
                    self._detect_frame_count = 0
                self._detect_frame_count += 1
                if self._detect_frame_count % 30 == 0:
                    rospy.loginfo("Detect running: frame=%d, faces=%d", self._detect_frame_count, len(boxes))

                for i, (box, mc) in enumerate(zip(boxes, mouth_centers)):
                    x1, y1, x2, y2 = box
                    result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    result_image = cv2.circle(result_image, (int(mc[0]), int(mc[1])), 5, (255, 0, 0), -1)
                    result_image = cv2.putText(result_image, "mouth", (x1, y1 - 5),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if len(mouth_centers) > 0:
                    mc = mouth_centers[0]
                    bw = boxes[0][2] - boxes[0][0]
                    rospy.logdebug("Detected mouth at (%.0f, %.0f) bbox_width=%d", mc[0], mc[1], bw)

                    if self.last_mouth_pos is not None:
                        dist = np.sqrt((mc[0] - self.last_mouth_pos[0]) ** 2 +
                                       (mc[1] - self.last_mouth_pos[1]) ** 2)
                        rospy.loginfo("Mouth dist: %.1f, count: %d/%d", dist, self.stability_count, self.stability_frames)
                        if dist < self.stability_pixels:
                            self.stability_count += 1
                            if self.stability_count > self.stability_frames:
                                world_xyz = self.pixel_to_world_mouth(mc, bw)
                                if world_xyz is not None:
                                    rospy.loginfo("Mouth locked: pixel=(%d,%d) world=(%.3f,%.3f,%.3f)",
                                                  int(mc[0]), int(mc[1]), *world_xyz)
                                    self.target_mouth = world_xyz
                                    threading.Thread(target=self._move_to_mouth, args=(world_xyz,)).start()
                                    self.stability_count = 0
                                else:
                                    rospy.logwarn("K is None, cannot convert pixel to world")
                        else:
                            self.stability_count = 0
                    self.last_mouth_pos = mc
                else:
                    self.stability_count = max(0, self.stability_count - 1)
                    self.last_mouth_pos = None

        except Exception as e:
            rospy.logerr(str(e))

        self.fps_counter.update()
        result_image = self.fps_counter.show_fps(result_image)
        cv2.imshow("feeding", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def image_callback(self, ros_image):
        try:
            self.image_queue.put_nowait(ros_image)
        except queue.Full:
            pass


if __name__ == "__main__":
    node = FeedingNode("feeding", log_level=rospy.INFO)
    while not rospy.is_shutdown():
        node.image_process()
