
#region imports
import atmos as at

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2 as cv
import taichi as ti
from PIL import Image
#endregion


# Select parallelization method
try:
    ti.init(arch=ti.cuda)
except Exception:
    try:
        ti.init(arch=ti.vulkan)
    except Exception:
        ti.init(arch=ti.cpu)


#region PARAMETERS

# !! WORLD AND CAMERA COORDS HAVE Z FORWARD Y UP

#### SUN ####################################
sunAngle = 0.2
# most of the earth should be lit up
# you can play around with this if you want
sun = np.array([0,np.sin(sunAngle),np.cos(sunAngle)])
#############################################

### CAMERA INTRINSICS #######################
# don't touch please
focalLength     = 50 * 0.001 # 50mm to m
sensorWidth     = 36 * 0.001 # 36mm to m
sensorHeight    = 36 * 0.001 # 36mm to m
xResolution     = 1024
yResolution     = 1024

xPitch = sensorWidth/xResolution
yPitch = sensorWidth/yResolution
dx = focalLength/xPitch
dy = focalLength/yPitch
#############################################

### CAMERA POSITION/ORIENTATION #############
# you can touch this if you know what you are doing.
# if you've taken linear algebra you may recognize
# these as rotation matrices
# if you get lost come back to y=3.14/2, x=0
thetay = 3.14/2
thetax = 0
TPCY = np.array([[np.cos(thetay),  0,  np.sin(thetay)],
                [     0,          1,      0        ],
                [-np.sin(thetay),  0,  np.cos(thetay)]])

TPCX = np.array([[1,  0,  0],
                [0, np.cos(thetax), -np.sin(thetax)],
                [0, np.sin(thetax),  np.cos(thetax)]])
# this matrix converts a vector from world coords to camera coords
TPC = TPCX.dot(TPCY)
# camera to world coords
TCP = np.linalg.matrix_transpose(TPC)


# position in world coords (KM); you can play with this one if you'd like
# if you get lost come back to [-30000, 0, 0]
rp = np.array([-30000,600,500])
# position in camera coords
rc = TPC.dot(rp)
#############################################

#endregion

#region GPU FUNCTIONS

# The important math is in here 
# if you're feeling brave, there is a small upgrade you can make here
@ti.kernel
def render_kernel(
    pos: at.vec3,
    sun_dir: at.vec3,
    tcp: ti.math.mat3,
    x_res: ti.i32,
    y_res: ti.i32,
    dx_: ti.f32,
    dy_: ti.f32,
    out_color: ti.types.ndarray(dtype=ti.f32, ndim=3),
    out_debug: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    for j, i in ti.ndrange(x_res, y_res):
        # find the direction of the pixel relative to the camera origin
        x = (ti.cast(i, ti.f32) - ti.cast(x_res, ti.f32) * 0.5) / dx_
        y = (ti.cast(j, ti.f32) - ti.cast(y_res, ti.f32) * 0.5) / dy_
        ray = at.vec3(x, y, 1.0).normalized()
        # convert view ray from camera to world coords
        ray = tcp @ ray*-1

        # technically this should be within atmos 
        # but I left it out here so that you can mess up atmos and still have a visible Earth   
        surface = at._earth(pos, ray, sun_dir)

        color, debug = at._atmos(pos, ray, sun_dir) # an astute researcher might find reason to have another input or output...

        clampedColor = ti.math.clamp((color+surface)*255, 0.0, 255.0) # said researcher would also have to change this line
        clampedDebug = ti.math.clamp(debug*255, 0.0, 255.0)
        out_color[i, j, 0] = clampedColor.x
        out_color[i, j, 1] = clampedColor.y
        out_color[i, j, 2] = clampedColor.z
        out_debug[i, j, 0] = clampedDebug.x
        out_debug[i, j, 1] = clampedDebug.y
        out_debug[i, j, 2] = clampedDebug.z


def render_gpu(pos, sun_dir):
    lum = np.zeros((xResolution, yResolution), dtype=np.float32)
    color = np.zeros((xResolution, yResolution, 3), dtype=np.float32)
    debug = np.zeros((xResolution, yResolution, 3), dtype=np.float32)
    render_kernel(
        at.vec3(*pos), at.vec3(*sun_dir),
        ti.math.mat3(TCP.tolist()),
        xResolution, yResolution, dx, dy,color, debug
    )
    return color.astype(np.uint8), debug.astype(np.uint8)

#endregion



#region MAIN

if __name__ == "__main__":
    output, debug = render_gpu(rp, sun)

    # Adding a blur cause it looks nice and is better for later algorithms
    output=cv.GaussianBlur(output, (3, 3), 0)
    img = Image.fromarray(output)
    img.save("./output.png")
    img = Image.fromarray(debug)
    img.save("./debug.png") # pink overlay by default
    print("generated image")
    
#endregion