#!/usr/bin/python3
# coding=utf8
# Date:2021/11/30
# Author:Aiden-Wei
from math import *
import numpy as np

# 一些常用的角度转换(some commonly used angle conversions)

def angle_transform(angle, param):
    # param = [min_old, max_old, center_old, min_new, max_new, center_new]
    new_angle = ((angle - param[2])/(param[1] - param[0])) * (param[4] - param[3]) + param[5]

    return new_angle

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6  

def rot2rpy(R):
    assert(isRotationMatrix(R))

    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        r = atan2(R[2, 1], R[2, 2])
        p = atan2(-R[2, 0], sy)
        y = atan2(R[1, 0], R[0, 0])
    else:
        r = atan2(-R[1, 2], R[1, 1])
        p = atan2(-R[2, 0], sy)
        y = 0
    
    return [degrees(r), degrees(p), degrees(y)]

def rpy2rot(r, p, y):
    r = radians(r)
    p = radians(p)
    y = radians(y)
    
    cr = cos(r)
    sr = sin(r)
    cp = cos(p)
    sp = sin(p)
    cy = cos(y)
    sy = sin(y)

    R = np.array([[cp*cy, -cr*sy + cy*sp*sr, cr*cy*sp + sr*sy],
        [cp*sy, cr*cy + sp*sr*sy, cr*sp*sy - cy*sr],
        [-sp, cp*sr, cp*cr]])

    return R

def qua2rot(x, y, z, w):
    rot_matrix = np.array(
        [[1.0 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (w * y + x * z)],
        [2 * (x * y + w * z), 1.0 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1.0 - 2 * (x * x + y * y)]])

    return rot_matrix

def rot2qua(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return [q[1], q[2], q[3], q[0]]

def rpy2qua(roll, pitch, yaw):
    roll = radians(roll)
    pitch = radians(pitch)
    yaw = radians(yaw)

    x = sin(pitch/2)*sin(yaw/2)*cos(roll/2) + cos(pitch/2)*cos(yaw/2)*sin(roll/2)
    y = sin(pitch/2)*cos(yaw/2)*cos(roll/2) + cos(pitch/2)*sin(yaw/2)*sin(roll/2)
    z = cos(pitch/2)*sin(yaw/2)*cos(roll/2) - sin(pitch/2)*cos(yaw/2)*sin(roll/2)
    w = cos(pitch/2)*cos(yaw/2)*cos(roll/2) - sin(pitch/2)*sin(yaw/2)*sin(roll/2)
   
    return [x, y, z, w]

def qua2rpy(x, y, z, w):
    roll = degrees(atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = degrees(asin(2 * (w * y - x * z)))
    yaw = degrees(atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y)))
    
    return roll, pitch, yaw
