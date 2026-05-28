#!/usr/bin/env python3
# coding: utf8
# Face/Mouth detection using YOLOv5 ONNX via OpenCV DNN
# Supports two model formats:
#   best2.onnx: output [1, 5, 8400] (cx,cy,w,h,conf) - no landmarks, mouth estimated from bbox
#   yolov5s-face.onnx: output [1, 25200, 16] - with 5 facial landmarks

import time
import sys
import os
import numpy as np
import cv2

IMG_SIZE = 640
FACE_WIDTH_REAL = 0.14

# Landmark indices for yolov5s-face model
LDMARK_LEFT_EYE = 0
LDMARK_RIGHT_EYE = 1
LDMARK_NOSE = 2
LDMARK_LEFT_MOUTH = 3
LDMARK_RIGHT_MOUTH = 4


class FaceDetector:
    def __init__(self, model_path, device='cpu', conf_thresh=0.3, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = IMG_SIZE
        self.has_landmarks = False

        net = cv2.dnn.readNet(model_path)
        if device == 'cuda' or device.startswith('cuda'):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net = net

        blob = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        net.setInput(blob)
        test_out = net.forward()
        self.out_shape = test_out.shape
        self.has_landmarks = (test_out.shape == (1, 25200, 16))

        try:
            import rospy
            rospy.loginfo("FaceDetector: %s loaded (landmarks=%s, out=%s)" %
                          (os.path.basename(model_path), self.has_landmarks, self.out_shape))
        except:
            print("FaceDetector: %s loaded (landmarks=%s, out=%s)" %
                  (os.path.basename(model_path), self.has_landmarks, self.out_shape))

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def detect(self, bgr_image):
        h0, w0 = bgr_image.shape[:2]
        img, ratio, (dw, dh) = self._letterbox(bgr_image, (self.img_size, self.img_size))
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (self.img_size, self.img_size),
                                     (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()

        if self.has_landmarks:
            return self._process_landmark_output(output, w0, h0, ratio, dw, dh)
        else:
            return self._process_nolandmark_output(output, w0, h0, ratio, dw, dh)

    def _process_landmark_output(self, output, w0, h0, ratio, dw, dh):
        boxes, all_lm, all_mc = [], [], []
        output = output[0]  # (25200, 16)

        for det in output:
            conf = float(det[4])
            if conf < self.conf_thresh:
                continue
            cx, cy, bw, bh = float(det[0]), float(det[1]), float(det[2]), float(det[3])
            x1 = (cx - bw / 2 - dw) / ratio
            y1 = (cy - bh / 2 - dh) / ratio
            x2 = (cx + bw / 2 - dw) / ratio
            y2 = (cy + bh / 2 - dh) / ratio

            pts = []
            for k in range(5):
                lx = (float(det[5 + k * 2]) - dw) / ratio
                ly = (float(det[6 + k * 2]) - dh) / ratio
                pts.append((lx, ly))
            mc = (
                (pts[3][0] + pts[4][0]) / 2,
                (pts[3][1] + pts[4][1]) / 2,
            )
            boxes.append([x1, y1, x2, y2, conf])
            all_lm.append(pts)
            all_mc.append(mc)

        if boxes:
            boxes, all_lm, all_mc = self._nms(boxes, all_lm, all_mc)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes], all_lm, all_mc

    def _process_nolandmark_output(self, output, w0, h0, ratio, dw, dh):
        boxes, all_mc = [], []
        output_t = output[0].T  # (8400, 5)

        for det in output_t:
            conf = float(det[4])
            if conf < self.conf_thresh:
                continue
            cx, cy, bw, bh = float(det[0]), float(det[1]), float(det[2]), float(det[3])
            x1 = (cx - bw / 2 - dw) / ratio
            y1 = (cy - bh / 2 - dh) / ratio
            x2 = (cx + bw / 2 - dw) / ratio
            y2 = (cy + bh / 2 - dh) / ratio

            face_h = y2 - y1
            mc = ((x1 + x2) / 2, y1 + face_h * 0.78)
            boxes.append([x1, y1, x2, y2, conf])
            all_mc.append(mc)

        if boxes:
            nms_boxes, _, nms_mc = self._nms(boxes, [], all_mc)
            return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in nms_boxes], [], nms_mc
        return [], [], []

    def _nms(self, boxes, landmarks, mouth_centers):
        if len(boxes) == 0:
            return [], [], []
        idxs = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
            [b[4] for b in boxes], self.conf_thresh, self.iou_thresh)
        if len(idxs) == 0:
            return [], [], []
        result_b = [boxes[i] for i in idxs.flatten()]
        result_l = [landmarks[i] for i in idxs.flatten()] if landmarks else []
        result_m = [mouth_centers[i] for i in idxs.flatten()]
        return result_b, result_l, result_m

    def estimate_depth(self, bbox_width_pixels, fx):
        if bbox_width_pixels < 1:
            return 0.5
        return fx * FACE_WIDTH_REAL / bbox_width_pixels

    def pixel_to_camera_point(self, u, v, depth, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        return np.array([X, Y, depth, 1.0])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_detect.py <model_path> [image|camera]")
        sys.exit(1)
    detector = FaceDetector(sys.argv[1], 'cpu')
    src = sys.argv[2] if len(sys.argv) > 2 else "0"

    if src.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        img = cv2.imread(src)
        if img is None:
            sys.exit(1)
        boxes, lm, mc = detector.detect(img)
        print("Detected %d face(s)" % len(boxes))
        for b, m in zip(boxes, mc):
            print("  box=%s mouth=(%.0f, %.0f)" % (b, m[0], m[1]))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.circle(img, (int(m[0]), int(m[1])), 5, (0, 0, 255), -1)
        cv2.imwrite("output_detect.jpg", img)
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            boxes, _, mc = detector.detect(frame)
            for b, m in zip(boxes, mc):
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.circle(frame, (int(m[0]), int(m[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, "FPS:%.1f" % (1/max(time.time()-t0,0.001)), (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
            cv2.imshow("Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
