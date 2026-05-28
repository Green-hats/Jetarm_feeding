#!/usr/bin/env python3
# coding: utf8

import cv2
import rospy
import queue
import numpy as np
import threading
from vision_utils import fps, xyz_quat_to_mat, xyz_euler_to_mat, pixels_to_world, box_center, mat_to_xyz_euler, distance, extristric_plane_shift
from sensor_msgs.msg import Image as RosImage, CameraInfo
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from yolov5_trt import YoLov5TRT
from hiwonder_interfaces.srv import GetRobotPose
from hiwonder_interfaces.msg import MoveAction, MoveGoal, MultiRawIdPosDur
from jetarm_sdk import bus_servo_control
import actions
import actionlib

CONFIG_NAME = '/config'

TRT_NUM_CLASSES = 13
TRT_CLASS_NAMES = ( 'Banana Peel','Broken Bones', 'Cigarette End', 'Disposable Chopsticks',
                   'Ketchup', 'Marker', 'Oral Liquid Bottle', 'Plate', 'Plastic Bottle',
                    'Storage Battery', 'Toothbrush', 'Umbrella','tap')

WASTE_CLASS_NAMES = ['residual_waste', 'food_waste', 'hazardous_waste', 'recyclable_waste','tap']

WASTE_CLASSES = {
    'food_waste': ('Banana Peel', 'Broken Bones', 'Ketchup'),
    'hazardous_waste': ('Marker', 'Oral Liquid Bottle', 'Storage Battery'),
    'recyclable_waste': ('Plastic Bottle', 'Toothbrush', 'Umbrella'),
    'residual_waste': ('Plate', 'Cigarette End', 'Disposable Chopsticks'),
    'tap':('tap')
}

COLORS = {
    'recyclable_waste': (0, 0, 255),
    'hazardous_waste': (255, 0, 0),
    'food_waste': (0, 255, 0),
    'residual_waste': (80, 80, 80),
    'tap' : (200,220,100)
}

class WasteClassificationNode:
    def __init__(self, node_name, log_level=rospy.INFO):
        rospy.init_node(node_name, anonymous=True, log_level=log_level)
        self.lock = threading.RLock()
        self.K = None
        self.D = None

        config = rospy.get_param(CONFIG_NAME)
        self.hand2cam_tf_matrix = config['hand2cam_tf_matrix']
        self.extristric = None
        self.roi = None
        self.moving_step = 0

        self.pick_pitch = 80
        self.target = None
        self.count = 0
        self.last_card = None
        self.endpoint = None
        self.image_queue = queue.Queue(maxsize=2)
        self.fps = fps.FPS()

        self.servos_pub = rospy.Publisher("/controllers/multi_id_pos_dur", MultiRawIdPosDur, queue_size=1)
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1) #订阅相机内参(subscribe to the camera intrinsics)
        
        #weights = '/home/hiwonder/weights/garbage_classification/garbage_classification_320s_6_2.engine'
        #lib = '/home/hiwonder/weights/garbage_classification/libmyplugins_320.so'
        weights = '/home/hiwonder/weights/garbage_classification/garbage640_2023_11_15.engine'
        lib = '/home/hiwonder/weights/garbage_classification/libmyplugins640_2023_11_15.so'
        self.yolov5 = YoLov5TRT(weights, lib, TRT_CLASS_NAMES, 0.85)
        rospy.sleep(1)
        actions.go_home(self.servos_pub)
    
        # 识别区域的四个角的世界坐标(the world coordinates of the four corners of the recognition area)
        white_area_cam = config['white_area_pose_cam']
        white_area_center = config['white_area_pose_world']
        self.white_area_center = white_area_center
        self.white_area_cam = white_area_cam
        white_area_height = config['white_area_world_size']['height']
        white_area_width = config['white_area_world_size']['width']
        white_area_lt = np.matmul(white_area_center, xyz_euler_to_mat((white_area_height / 2, white_area_width / 2, 0.0), (0, 0, 0)))
        white_area_lb = np.matmul(white_area_center, xyz_euler_to_mat((-white_area_height / 2 - 0.01, white_area_width / 2, 0.0), (0, 0, 0)))
        white_area_rb = np.matmul(white_area_center, xyz_euler_to_mat((-white_area_height / 2 - 0.01, -white_area_width / 2, 0.0), (0, 0, 0)))
        white_area_rt = np.matmul(white_area_center, xyz_euler_to_mat((white_area_height / 2, -white_area_width / 2, 0.0), (0, 0, 0)))
        self.get_endpoint()
        corners_cam =  np.matmul(np.linalg.inv(np.matmul(self.endpoint, config['hand2cam_tf_matrix'])), [white_area_lt, white_area_lb, white_area_rb, white_area_rt, white_area_center])
        corners_cam = np.matmul(np.linalg.inv(white_area_cam), corners_cam)
        corners_cam = corners_cam[:, :3, 3:].reshape((-1, 3))
        tvec, rmat = config['extristric']

        while self.K is None or self.D is None: # 等待获取相机内参(wait for getting camera intricate)
            rospy.sleep(0.5)

        self.hand2cam_tf_matrix = config['hand2cam_tf_matrix']
        center_imgpts, jac = cv2.projectPoints(corners_cam[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
        self.center_imgpts = np.int32(center_imgpts).reshape(2)
        tvec, rmat = extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.04)
        self.extristric = tvec, rmat
        imgpts, jac = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)
        self.imgpts = np.int32(imgpts).reshape(-1, 2)

        # 计算ROI区域
        x_min = min(self.imgpts, key=lambda p: p[0])[0]  # x轴最小值(the minimum value of x-axis)
        x_max = max(self.imgpts, key=lambda p: p[0])[0]  # x轴最大值(the maximal value of x-axis)
        y_min = min(self.imgpts, key=lambda p: p[1])[1]  # y轴最小值(the minimum value of y-axis)
        y_max = max(self.imgpts, key=lambda p: p[1])[1]  # y轴最大值(the maximal value of y-axis)
        roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
        self.roi = roi

        self.action_client = actionlib.SimpleActionClient('/grasp', MoveAction)
        source_image_topic = rospy.get_param('~source_image_topic', '/camera/image_raw')
        self.image_sub = rospy.Subscriber(source_image_topic, RosImage, self.image_callback, queue_size=1)
        rospy.loginfo("启动完成\r\n\r\n")


    def camera_info_callback(self, msg): # 相机内参回调(camera intrinsics callback)
        with self.lock:
            K = np.matrix(msg.K).reshape(1, -1, 3)
            D = np.array(msg.D)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0, (640, 480))
            self.K, self.D = np.matrix(new_K), np.zeros((5, 1))


    def get_endpoint(self):
        endpoint = rospy.ServiceProxy('/kinematics/get_current_pose', GetRobotPose)().pose
        self.endpoint = xyz_quat_to_mat([endpoint.position.x, endpoint.position.y, endpoint.position.z],
                                        [endpoint.orientation.w, endpoint.orientation.x, endpoint.orientation.y, endpoint.orientation.z])

    def done_callback(self, state, result):  # 动作执行完毕回调(callback function when the action is completed)
        rospy.loginfo("state:%f", state)
        if not result.result.complete:  # 如果在移动中被取消，需要回到初始位置(if canceled or unable to reach the specified position)
            bus_servo_control.set_servos(self.servos_pub, 500, ((1, 500), ))
            rospy.sleep(1)
            actions.go_home(self.servos_pub)
            bus_servo_control.set_servos(self.servos_pub, 500, ((10, 200), ))
            rospy.sleep(0.5)
        elif self.moving_step != 1:
            bus_servo_control.set_servos(self.servos_pub, 500, ((10, 200), ))

        if self.finish_percent == 1:  # 如果完整的完成移动(if the motion is completed in full)
            if self.moving_step == 1:  # 如果完成了夹取(if grasping is completed)
                self.moving_step = 2
                goal = MoveGoal()
                goal.grasp.mode = 'place'
                target_position = rospy.get_param(CONFIG_NAME)["waste_target_position"][self.target[-1]]
                goal.grasp.position.x = target_position[0]
                goal.grasp.position.y = target_position[1]
                goal.grasp.position.z = target_position[2]
                goal.grasp.pitch = self.pick_pitch
                goal.grasp.align_angle = 90  # yaw #- 20/1000* 240
                goal.grasp.grasp_approach.z = 0.06  # 放置时靠近的方向和距离(direction and distance of approach during placement)
                goal.grasp.grasp_retreat.z = 0.06  # 放置后后撤的方向和距离(direction and distance of retreat after placement)
                goal.grasp.grasp_posture = 370  # 夹取前后夹持器的开合角度(the opening and closing angle of the grippers before and after grasping)
                goal.grasp.pre_grasp_posture = 580
                self.action_client.send_goal(goal, self.done_callback, self.active_callback, self.feedback_callback)
            elif self.moving_step == 2:  # 如果完成了放置(if placement is completed)
                actions.go_home(self.servos_pub)
                rospy.sleep(0.5)
                self.get_endpoint()
                self.last_position = None
                self.target = None
                self.count = 0
                self.moving_step = 0
        else:  # 如果被取消或者无法到达指定位置(if canceled or unable to reach the specified position)
            actions.go_home(self.servos_pub)
            self.moving_step = 0
            self.last_position = None

    def active_callback(self):  # 运动开始回调(callback function when the motion starts)
        self.start_move = True
        rospy.loginfo("start move")

    def feedback_callback(self, msg):  # 动作执行进度回调(progress callback during action execution)
        rospy.loginfo("finish action: {:.2%}".format(msg.percent))
        self.finish_percent = msg.percent

    def start_moving(self, pose_t, pose_R):
        rospy.loginfo("开始搬运堆叠...")
        print(pose_t, pose_R)
        self.moving_step = 1
        goal = MoveGoal()
        goal.grasp.mode = 'pick'
        # 物体坐标(object coordinates)
        goal.grasp.position.x = pose_t[0]
        goal.grasp.position.y = pose_t[1]
        goal.grasp.position.z = 0.010
        # 夹取时的姿态角(posture angle during grasping)
        goal.grasp.pitch = self.pick_pitch
        goal.grasp.align_angle = 0  # 总是朝前(always move forward)
        # 夹取时靠近的方向和距离(direction and distance of approach during grasping)
        goal.grasp.grasp_approach.z = 0.03
        # 夹取后后撤方向和距离(direction and distance of retreat after grasping)
        goal.grasp.grasp_retreat.z = 0.05
        # 夹取前后夹持器的开合(the opening and closing angle of the grippers before and after grasping)
        goal.grasp.grasp_posture = 580
        goal.grasp.pre_grasp_posture = 220
        self.action_client.send_goal(goal, self.done_callback, self.active_callback, self.feedback_callback)  # 发送夹取请求(sent grasping requirement)

    def image_process(self):
        ros_image = self.image_queue.get(block=True)
        # 将ros格式图像转换为opencv格式(convert ros format image to opencv format)
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        result_image = np.copy(rgb_image)

        if self.center_imgpts is not None:
            cv2.line(result_image, (self.center_imgpts[0] - 10, self.center_imgpts[1]), (self.center_imgpts[0] + 10, self.center_imgpts[1]), (255, 255, 0), 2)
            cv2.line(result_image, (self.center_imgpts[0], self.center_imgpts[1] - 10), (self.center_imgpts[0], self.center_imgpts[1] + 10), (255, 255, 0), 2)

        try:
            if self.moving_step == 0 and self.roi is not None and self.K is not None and self.D is not None:
                roi_area_mask = np.zeros(shape=(ros_image.height, ros_image.width, 1), dtype=np.uint8)
                roi_area_mask = cv2.drawContours(roi_area_mask, [self.imgpts], -1, 255, cv2.FILLED)
                rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=roi_area_mask)  # 和原图做遮罩，保留需要识别的区域(mask the original image to retain the area that needs to be recognized)
                #roi_img = rgb_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                boxes, confs, classes = self.yolov5.infer(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                cards = []

                for box, cls_conf, cls_id in zip(boxes, confs, classes):
                    x1 = box[0] #+ self.roi[2]
                    y1 = box[1] #+ self.roi[0]
                    x2 = box[2] #+ self.roi[2]
                    y2 = box[3] #+ self.roi[0]
                    waste_name = TRT_CLASS_NAMES[cls_id]
                    if waste_name == "tap":
                        continue
                    waste_class_name = ''
                    for k, v in WASTE_CLASSES.items():
                        if waste_name in v:
                            waste_class_name = k
                            break
                    cards.append((cls_conf, [x1, y1, x2, y2], waste_class_name))
                    result_image = cv2.putText(result_image, waste_name + " " + str(float(cls_conf))[:4], (int(x1), int(y1) - 5),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
                    result_image = cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[waste_class_name], 3)
                if len(cards) > 0:
                    cards = sorted(cards, key=lambda c: c[0], reverse=True)
                    if self.last_card is None:
                        self.last_card = cards[0]
                    center_last = box_center(self.last_card[1])
                    cards = sorted(cards, key=lambda c: distance(box_center(c[1]), center_last))
                    center = box_center(cards[0][1])
                    if distance(center, center_last) < 50:
                        self.count += 1
                        if self.count > 40:
                            projection_matrix = np.row_stack((np.column_stack((self.extristric[1], self.extristric[0])), np.array([[0, 0, 0, 1]])))
                            world_pose = pixels_to_world([center, ], self.K, projection_matrix)[0]  # 像素坐标相对于识别区域中心的相对坐标(pixel coordinates relative to the center of the recognition area)
                            world_pose[1] = -world_pose[1]
                            world_pose[2] = 0.04
                            world_pose = np.matmul(self.white_area_center, xyz_euler_to_mat(world_pose, (0, 0, 0)))  # 转换到相机相对坐标(convert to the camera relative coordinates)
                            world_pose[2] = 0.04
                            #world_pose = np.matmul(self.white_area_cam, xyz_euler_to_mat(world_pose, (0, 0, 0)))  # 转换到相机相对坐标(convert to camera relative coordinates)
                            # pose_end = np.matmul(self.hand2cam_tf_matrix, world_pose)  # 转换的末端相对坐标(relative coordinates of the converted end)
                            # pose_world = np.matmul(self.endpoint, pose_end)  # 转换到机械臂世界坐标(convert to the robotic arm's world coordinates)
                            pose_t, pose_R = mat_to_xyz_euler(world_pose)
                            #pose_t[0] = pose_t[0] - 0.01  # 卡片上的图案不在画面中心(the pattern on the card is not in the center of the screen)
                            print(pose_t)
                            self.target = cards[0]
                            self.target = cards[0]
                            self.moving_step = 1
                            threading.Thread(target=self.start_moving, args=(pose_t, pose_R)).start()
                    else:
                        self.count = 0
                    self.last_card = cards[0]
                else:
                    self.count = 0
                    self.last_card = None
        except Exception as e:
            rospy.logerr(str(e))

        # 计算帧率及发布结果图像(calculate frame rate and publish result image)
        self.fps.update()
        result_image = self.fps.show_fps(result_image)
        cv2.imshow("waste_classification", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def image_callback(self, ros_image: RosImage):
        rospy.logdebug('Received an image! ')
        try:
            self.image_queue.put_nowait(ros_image)
        except Exception as e:
            e = e


if __name__ == "__main__":
    node = WasteClassificationNode("waste_classification", log_level=rospy.INFO)
    while not rospy.is_shutdown():
        node.image_process()

