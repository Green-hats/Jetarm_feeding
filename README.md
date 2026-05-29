# 喂食机械臂项目

## 项目结构

| 文件 | 用途 |
|------|------|
| `feed_main.py` | **喂食主程序** - ROS 节点, YOLOv5-face 嘴巴检测 + actionlib 运动控制 |
| `face_detect.py` | YOLOv5-face PyTorch 推理模块 (人脸+嘴巴检测) |
| `waste_classification.py` | 原始垃圾分拣主程序 (保留) |
| `yolov5_trt.py` | 原始 TensorRT YOLOv5 推理封装 (保留) |
| `actions.py` | 机械臂动作函数 (新增喂食动作: goto_pick_food, open_gripper 等) |
| `transform.py` | 坐标变换工具函数 |
| `utils.py` | 人脸检测辅助函数 |
| `heart.py` | ROS 心跳监测 |
| `config.yaml` | 原始垃圾分拣标定参数 |
| `config_feed.yaml` | **喂食项目配置参数** |

## 喂食项目工作流程

```
Idle → 启动喂食
  ↓
Step 1: 移动到食物抓取位置(右侧固定点) → 抓取食物
  ↓
Step 2: 移动到人脸检测位置(相机朝向用户)
  ↓
Step 3: 相机检测嘴巴坐标(YOLOv5-face) → 稳定性过滤
  ↓
Step 4: 像素坐标 → 世界坐标(深度估计) → 移动到嘴巴位置
  ↓
Step 5: 松开夹爪释放食物 → 回退 → 回home
```

## 模型下载

使用 [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) 模型,
输出人脸框 + 5个关键点(左眼、右眼、鼻子、左嘴角、右嘴角)。

```bash
# 下载预训练权重
# Google Drive: https://drive.google.com/drive/folders/1v0zFpv5CgpSCF-D_0ENRyQH_WvQVdU7B
# 推荐使用 yolov5s-face.pt (适合 Jetson Nano)

# 放置到指定位置
mkdir -p /home/hiwonder/weights
cp yolov5s-face.pt /home/hiwonder/weights/
```

## 使用方法

```bash
# 1. 加载喂食配置
rosparam load config_feed.yaml /config

# 2. 启动喂食节点
python3 feed_main.py

# 3. 如需测试人脸检测模块
python3 face_detect.py /home/hiwonder/weights/yolov5s-face.pt
```

## 配置说明 (config_feed.yaml)

| 参数 | 说明 |
|------|------|
| `food_pick_position` | 食物抓取世界坐标 [x, y, z] (右侧固定点) |
| `detect_arm_joints` | 检测姿态的舵机角度 |
| `stability_frames` | 嘴巴检测稳定性帧数 (需连续检测到) |
| `stability_pixels` | 稳定性像素阈值 |
| `face_model_path` | YOLOv5-face 权重文件路径 |
| `face_width_real` | 平均人脸宽度 (0.14m, 用于深度估计) |

## 嘴巴坐标提取

YOLOv5-face 输出 5 个关键点 (索引 0-4):
- 0: 左眼 left_eye
- 1: 右眼 right_eye
- 2: 鼻子 nose
- 3: **左嘴角 left_mouth**
- 4: **右嘴角 right_mouth**

嘴巴中心 = ((left_mouth.x + right_mouth.x) / 2, (left_mouth.y + right_mouth.y) / 2)

## 深度估计

```
depth = camera_fx * face_width_real / face_bbox_width_pixels
```

基于人脸宽度估算相机到脸的距离, 无需预标定平面。

# 原始垃圾分拣说明（本项目基于垃圾分拣项目修改）

硬件: Hiwonder JetArm + Jetson Nano
架构: Python3 ROS 节点 → 摄像头 → YOLO检测 → 坐标转换 → /grasp actionlib 运动
