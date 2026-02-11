# Motion Demo
# This example is part of the RB-Y1 SDK examples. See --help for arguments.
#
# Usage example:
#     python 09_demo_motion.py --help
#
# Copyright (c) 2025 Rainbow Robotics. All rights reserved.
#
# DISCLAIMER:
# This is a sample code provided for educational and reference purposes only.
# Rainbow Robotics shall not be held liable for any damages or malfunctions resulting from
# the use or misuse of this demo code. Please use with caution and at your own discretion.
import pyrealsense2 as rs
import rby1_sdk
import numpy as np
import sys
import time
import argparse
import re
from rby1_sdk import *
import cv2
import socket
import struct
import math
import threading


class RealSenseCamera:
    def __init__(self, serial_number=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial_number = serial_number
        self.camera_running = True
        self.color_image = None
        self.depth_image = None
        self.lock = threading.Lock()
        
        self.width = 640
        self.height = 480
        self.fps = 30
        self.focal_length = 0.0
        self.principal_point = [0.0, 0.0]
        self.intrinsics = None
        self.profile = None

    def initialize_camera(self, set_width, set_height, set_fps):
        self.width = set_width
        self.height = set_height
        self.fps = set_fps
        
        if self.serial_number:
            self.config.enable_device(self.serial_number)
            
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        self.profile = self.pipeline.start(self.config)
        
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        
        # Calculate focal length: sqrt(fx^2 + fy^2) as in C++ code
        self.focal_length = math.sqrt(self.intrinsics.fx**2 + self.intrinsics.fy**2)
        self.principal_point = [self.intrinsics.ppx, self.intrinsics.ppy]
        
        print(f"Focal Length: {self.focal_length}")
        print(f"Principal Point: {self.principal_point[0]}, {self.principal_point[1]}")

    def start(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        try:
            while self.camera_running:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

                with self.lock:
                    self.color_image = color_data
                    self.depth_image = depth_data
                    
        except RuntimeError as e:
            print(f"RealSense Error: {e}")
        except Exception as e:
            print(f"Standard Error: {e}")
        finally:
            self.pipeline.stop()

    def stop(self):
        self.camera_running = False

    def capture_image(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        if self.camera_running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("no frame")
            
            # Convert to numpy arrays
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            with self.lock:
                self.color_image = color_data
                self.depth_image = depth_data

    def get_color_image(self):
        with self.lock:
            if self.color_image is None:
                return None
            return self.color_image.copy()

    def get_depth_image(self):
        with self.lock:
            if self.depth_image is None:
                return None
            return self.depth_image.copy()

    def get_principal_point_and_focal_length(self):
        return [self.principal_point[0], self.principal_point[1], self.focal_length]


class Marker_Detection:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.principal_point = [0, 0]
        self.focal_length = 0

    def set_intrinsics_param(self, param):
        self.principal_point = [param[0], param[1]]
        self.focal_length = param[2]

    def convert_pixel2mm(self, center):
        # center: [x, y, z] (z is depth value in mm)
        if not center:
            return center
        
        if self.focal_length == 0:
            return center 

        x = (center[0] - self.principal_point[0]) * center[2] / self.focal_length
        y = (center[1] - self.principal_point[1]) * center[2] / self.focal_length
        z = center[2]
        return [x, y, z]

    def normalize(self, v):
        m = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if m > 1e-9:
            return [v[0]/m, v[1]/m, v[2]/m]
        return v

    def cross(self, a, b):
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]
    
    def get_rotation_matrix(self, corners, depth_data):
        # corners: list of [x, y]
        pts_3d = []
        for i in range(4):
            cx, cy = corners[i]
            d = 0
            # C++ behavior: truncation
            iy, ix = int(cy), int(cx)
            if 0 <= iy < depth_data.shape[0] and 0 <= ix < depth_data.shape[1]:
                d = depth_data[iy, ix]
            
            pts_3d.append(self.convert_pixel2mm([cx, cy, float(d)]))

        # Vector logic from C++
        # x_axis = (tr + br) - (tl + bl) -> (pt1 + pt2) - (pt0 + pt3)
        # y_axis = (bl + br) - (tl + tr) -> (pt3 + pt2) - (pt0 + pt1) -> Y Down
        # corners order in aruco: TL, TR, BR, BL (0, 1, 2, 3)
        
        x_axis = [
            (pts_3d[1][0] + pts_3d[2][0]) - (pts_3d[0][0] + pts_3d[3][0]),
            (pts_3d[1][1] + pts_3d[2][1]) - (pts_3d[0][1] + pts_3d[3][1]),
            (pts_3d[1][2] + pts_3d[2][2]) - (pts_3d[0][2] + pts_3d[3][2])
        ]
        
        y_axis = [
            (pts_3d[3][0] + pts_3d[2][0]) - (pts_3d[0][0] + pts_3d[1][0]),
            (pts_3d[3][1] + pts_3d[2][1]) - (pts_3d[0][1] + pts_3d[1][1]),
            (pts_3d[3][2] + pts_3d[2][2]) - (pts_3d[0][2] + pts_3d[1][2])
        ]
        
        x_axis = self.normalize(x_axis)
        y_axis = self.normalize(y_axis)
        
        z_axis = self.cross(x_axis, y_axis)
        z_axis = self.normalize(z_axis)
        
        # Recompute Y to be orthogonal
        y_axis = self.cross(z_axis, x_axis)
        y_axis = self.normalize(y_axis)
        
        # Rotation Matrix (3x3)
        # Columns correspond to basis vectors
        R = [[0.0]*3 for _ in range(3)]
        R[0][0] = x_axis[0]; R[0][1] = y_axis[0]; R[0][2] = z_axis[0]
        R[1][0] = x_axis[1]; R[1][1] = y_axis[1]; R[1][2] = z_axis[1]
        R[2][0] = x_axis[2]; R[2][1] = y_axis[2]; R[2][2] = z_axis[2]
        
        return R

    def get_rpy_from_matrix(self, R):
        sy = math.sqrt(R[0][0]**2 + R[1][0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2][1], R[2][2])
            pitch = math.atan2(-R[2][0], sy)
            yaw = math.atan2(R[1][0], R[0][0])
        else:
            roll = math.atan2(-R[1][2], R[1][1])
            pitch = math.atan2(-R[2][0], sy)
            yaw = 0
        return [roll, pitch, yaw]

    def detect(self, color_image, depth_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        
        marker_centers_result = []
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            for i in range(len(ids)):
                c = corners[i][0] # c has 4 points
                
                # C++ Logic: Center is average of min/max (AABB center), not centroid
                xs = [pt[0] for pt in c]
                ys = [pt[1] for pt in c]
                xs.sort()
                ys.sort()
                
                x_center = (xs[0] + xs[3]) / 2.0
                y_center = (ys[0] + ys[3]) / 2.0
                
                # Get depth
                z = 0
                # C++ uses simple cast (truncation), not rounding
                iy, ix = int(y_center), int(x_center)
                if 0 <= iy < depth_image.shape[0] and 0 <= ix < depth_image.shape[1]:
                    z = depth_image[iy, ix]
                
                # Draw center
                cv2.circle(color_image, (ix, iy), 5, (0, 0, 255), -1)
                
                # Pixel to MM
                center_pos = self.convert_pixel2mm([x_center, y_center, float(z)])
                
                # Rotation Matrix
                c_list = [[pt[0], pt[1]] for pt in c]
                rot_matrix = self.get_rotation_matrix(c_list, depth_image)
                
                # RPY
                rpy = self.get_rpy_from_matrix(rot_matrix)
                
                # Cartesian Matrix (4x4)
                transform = [
                    rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], center_pos[0],
                    rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], center_pos[1],
                    rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], center_pos[2],
                    0.0, 0.0, 0.0, 1.0
                ]
                
                print(f"id : {ids[i][0]}")
                print(f"Center [{transform[3]}, {transform[7]}, {transform[11]}]")
                print(f"rpy    [{rpy[0]*180/math.pi}, {rpy[1]*180/math.pi}, {rpy[2]*180/math.pi}]")
                
                marker_centers_result.append(transform)
                
        return marker_centers_result

class Marker_Transform:
    def __init__(self, serial_number=None):
        # Setup Transforms
        base_to_marker_data = [0.2, 0.0, 1.0, 180, 0.0, -90.0]
        camera_to_tool_data = [0.0, 0.0, -0.1, 0.0, 180.0, 90.0]
        
        self.base_to_marker_tf = self.make_transform(base_to_marker_data)
        self.camera_to_tool_tf = self.make_transform(camera_to_tool_data)
        
        # Initialize
        self.camera = RealSenseCamera(serial_number)
        self.marker_detection = Marker_Detection()
        
        self.width = 1280 # 848
        self.height = 720 # 480
        self.fps = 30
        
        print("Initializing Camera...")
        self.camera.initialize_camera(self.width, self.height, self.fps)
        
        intrinsics = self.camera.get_principal_point_and_focal_length()
        self.marker_detection.set_intrinsics_param(intrinsics)

    def make_transform(self, data):
        # data: [x, y, z, roll, pitch, yaw] (x,y,z in meters, r,p,y in degrees)
        # The C++ code multiplies x,y,z by 1000 inside make_Transform
        x, y, z = data[0]*1000, data[1]*1000, data[2]*1000 
        roll = data[3] * math.pi / 180
        pitch = data[4] * math.pi / 180
        yaw = data[5] * math.pi / 180
        
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)
        
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = cy * cp
        m[0, 1] = sr * sp * cy - cr * sy
        m[0, 2] = cr * sp * cy + sr * sy
        m[0, 3] = x
        
        m[1, 0] = sy * cp
        m[1, 1] = sr * sp * sy + cr * cy
        m[1, 2] = cr * sp * sy - sr * cy
        m[1, 3] = y
        
        m[2, 0] = -sp
        m[2, 1] = cp * sr
        m[2, 2] = cp * cr
        m[2, 3] = z
        
        return m

    def get_marker_transform(self,Visualization=False):
        base_to_tool_tf = None
        base_to_tool_vec = None
        # print("RealSense Camera Started. Press 'ESC' to exit.")
        # time.sleep(1) # Warmup - Moved or removed for loop performance
        try:
            self.camera.capture_image()
            color_img = self.camera.get_color_image()
            depth_img = self.camera.get_depth_image()
            
            if color_img is None or depth_img is None:
                return None
            
            marker_transforms = self.marker_detection.detect(color_img, depth_img)
            
            for tf_list in marker_transforms:
                # Convert flattened list to 4x4 matrix
                camera_to_marker_tf = np.array(tf_list, dtype=np.float32).reshape(4, 4)
                
                try:
                    camera_to_marker_inv = np.linalg.inv(camera_to_marker_tf)
                    # base_to_tool = base_to_marker * camera_to_marker^-1 * camera_to_tool
                    base_to_tool_tf = self.base_to_marker_tf @ camera_to_marker_inv @ self.camera_to_tool_tf
                    base_to_tool_vec = base_to_tool_tf.flatten()
                    if base_to_tool_vec[3] > 4 :
                        base_to_tool_vec[3] = base_to_tool_vec[3]/1000
                        base_to_tool_vec[7] = base_to_tool_vec[7]/1000
                        base_to_tool_vec[11] = base_to_tool_vec[11]/1000
                except np.linalg.LinAlgError:
                    print("Singular matrix, cannot invert")

            # Visualization Logic
            if Visualization == True:
                min_dist = 280.0
                max_dist = 3000.0
                
                alpha = (0.0 - 200.0) / (max_dist - min_dist)
                beta = 200.0 - (min_dist * alpha)
                
                depth_debug = depth_img.astype(np.float32)
                depth_debug = depth_debug * alpha + beta
                depth_debug = np.clip(depth_debug, 0, 255).astype(np.uint8)
                
                # Mask invalid depth (0) to black (0)
                depth_debug[depth_img == 0] = 0
                
                depth_debug_bgr = cv2.cvtColor(depth_debug, cv2.COLOR_GRAY2BGR)
                
                # Resize for hconcat if dimensions differ
                if depth_debug_bgr.shape[1] != self.width or depth_debug_bgr.shape[0] != self.height:
                    depth_debug_bgr = cv2.resize(depth_debug_bgr, (self.width, self.height))
                if color_img.shape[1] != self.width or color_img.shape[0] != self.height:
                    color_img = cv2.resize(color_img, (self.width, self.height))
                    
                concat_image = cv2.hconcat([color_img, depth_debug_bgr])
                
                cv2.imshow("Preview", concat_image)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'): # ESC or q
                    raise KeyboardInterrupt
                
                if cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            raise
        # finally:
            # self.camera.stop()
            # cv2.destroyAllWindows() # Do not destroy windows every frame if looping
        # print("Camera Stopped.")

        return base_to_tool_vec

    def stop(self):
        self.camera.stop()
        cv2.destroyAllWindows()


D2R = np.pi / 180  # Degree to Radian conversion factor
MINIMUM_TIME = 2
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-4
STOP_POSITION_TRACKING_ERROR = 1e-3
WEIGHT = 1
STOP_COST = WEIGHT * WEIGHT * 2e-3
MIN_DELTA_COST = WEIGHT * WEIGHT * 2e-3
PATIENCE = 10


def cb(rs):
    print(f"Timestamp: {rs.timestamp - rs.ft_sensor_right.time_since_last_update}")
    position = rs.position * 180 / 3.141592
    print(f"torso [deg]: {position[2:2 + 6]}")
    print(f"right arm [deg]: {position[8:8 + 7]}")
    print(f"left arm [deg]: {position[15:15 + 7]}")


def example_joint_position_command_2(robot, model_name):
    print("joint position command example 2")

    # Define joint positions
    if model_name == "a":
        q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
    elif model_name == "m":
        q_joint_torso = np.array([0, 30, -60, 30, 0, 0]) * D2R
        
    q_joint_right_arm = np.array([-45, -30, 0, -90, 0, 45, 0]) * D2R
    q_joint_left_arm = np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R

    # Combine joint positions
    q = np.concatenate([q_joint_torso, q_joint_right_arm, q_joint_left_arm])

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyCommandBuilder().set_command(
                JointPositionCommandBuilder()
                .set_position(q)
                .set_minimum_time(MINIMUM_TIME)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def example_cartesian_command_1(robot, model_name):
    print("move task space")

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    T_torso[:3, :3] = np.eye(3)
    T_torso[:3, 3] = [0, 0, 1]

    angle = -np.pi / 2
    T_right[:3, :3] = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    T_right[:3, 3] = [0.5, -0.3, 1.0]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    target_link,
                    T_torso,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0



def example_cartesian_command_2(robot, model_name):
    print("move Z axis")

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    T_torso[:3, :3] = np.eye(3)
    T_torso[:3, 3] = [0, 0, 1]

    angle = -np.pi / 2
    T_right[:3, :3] = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    T_right[:3, 3] = [0.5, -0.3, 0.9]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    target_link,
                    T_torso,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0





def example_cartesian_command_3(robot, model_name):
    print("move X axis")

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    T_torso[:3, :3] = np.eye(3)
    T_torso[:3, 3] = [0, 0, 1]

    angle = -np.pi / 2
    T_right[:3, :3] = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    T_right[:3, 3] = [0.4, -0.3, 0.9]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    target_link,
                    T_torso,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def example_cartesian_command_4(robot, model_name):
    print("move Y axis")

    # Initialize transformation matrices
    T_torso = np.eye(4)
    T_right = np.eye(4)

    # Define transformation matrices
    T_torso[:3, :3] = np.eye(3)
    T_torso[:3, 3] = [0, 0, 1]

    angle = -np.pi / 2
    T_right[:3, :3] = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    T_right[:3, 3] = [0.4, -0.2, 0.9]

    if model_name == "a":
        target_link = "link_torso_5"
    elif model_name == "m":
        target_link = "link_torso_5"

    # Build command
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    target_link,
                    T_torso,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
            .set_right_arm_command(
                CartesianCommandBuilder()
                .add_target(
                    "base",
                    "ee_right",
                    T_right,
                    LINEAR_VELOCITY_LIMIT,
                    ANGULAR_VELOCITY_LIMIT,
                    ACCELERATION_LIMIT,
                )
                .set_minimum_time(MINIMUM_TIME)
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(1))
                .set_stop_orientation_tracking_error(STOP_ORIENTATION_TRACKING_ERROR)
                .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def go_to_home_pose_1(robot, model_name):
    print("Go to home pose 1")

    if model_name == "a":
        q_joint_torso = np.zeros(6)
    elif model_name == "m":
        q_joint_torso = np.zeros(6)
        
    q_joint_right_arm = np.zeros(7)
    q_joint_left_arm = np.zeros(7)

    q_joint_right_arm[1] = -135 * D2R
    q_joint_left_arm[1] = 135 * D2R

    # Send command to go to ready position
    rv = robot.send_command(
        RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                BodyComponentBasedCommandBuilder()
                .set_torso_command(
                    JointPositionCommandBuilder()
                    .set_minimum_time(MINIMUM_TIME * 2)
                    .set_position(q_joint_torso)
                )
                .set_right_arm_command(
                    JointPositionCommandBuilder()
                    .set_minimum_time(MINIMUM_TIME * 2)
                    .set_position(q_joint_right_arm)
                )
                .set_left_arm_command(
                    JointPositionCommandBuilder()
                    .set_minimum_time(MINIMUM_TIME * 2)
                    .set_position(q_joint_left_arm)
                )
            )
        ),
        10,
    ).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def go_to_home_pose_2(robot, model_name):
    print("Go to home pose 2")

    if model_name =="a":
        target_joint = np.zeros(20)
    elif model_name == "m":
        target_joint = np.zeros(20)
        
    # Send command to go to home pose
    rv = robot.send_command(
        RobotCommandBuilder().set_command(
            ComponentBasedCommandBuilder().set_body_command(
                JointPositionCommandBuilder()
                .set_position(target_joint)
                .set_minimum_time(MINIMUM_TIME)
            )
        ),
        10,
    ).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0


def main(address, model_name, power, servo):
    print("Attempting to connect to the robot...")

    robot = rby1_sdk.create_robot(address, model_name)

    if not robot.connect():
        print("Error: Unable to establish connection to the robot at")
        sys.exit(1)

    print("Successfully connected to the robot")

    print("Starting state update...")
    # robot.start_state_update(cb, 0.1)

    # robot.factory_reset_all_parameters()
    robot.set_parameter("default.acceleration_limit_scaling", "1.0")
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("default.linear_acceleration_limit", "20")
    robot.set_parameter("default.angular_acceleration_limit", "10")
    robot.set_parameter("manipulability_threshold", "1e4")
    # robot.set_time_scale(1.0)

    print("parameters setting is done")

    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)

    if not robot.is_power_on(power):
        rv = robot.power_on(power)
        if not rv:
            print("Failed to power on")
            exit(1)

    print(servo)
    if not robot.is_servo_on(servo):
        rv = robot.servo_on(servo)
        if not rv:
            print("Fail to servo on")
            exit(1)

    control_manager_state = robot.get_control_manager_state()

    if (
        control_manager_state.state == rby1_sdk.ControlManagerState.State.MinorFault
        or control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault
    ):

        if control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault:
            print(
                "Warning: Detected a Major Fault in the Control Manager!!!!!!!!!!!!!!!."
            )
        else:
            print(
                "Warning: Detected a Minor Fault in the Control Manager@@@@@@@@@@@@@@@@."
            )

        print("Attempting to reset the fault...")
        if not robot.reset_fault_control_manager():
            print("Error: Unable to reset the fault in the Control Manager.")
            sys.exit(1)
        print("Fault reset successfully.")

    print("Control Manager state is normal. No faults detected.")

    print("Enabling the Control Manager...")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        print("Error: Failed to enable the Control Manager.")
        sys.exit(1)
    print("Control Manager enabled successfully.")

    BASE, EE = 0, 1
    def compute_T_fk(q):
        model = robot.model()
        dyn_robot = robot.get_dynamics()
        dyn_state = dyn_robot.make_state(["link_torso_5", "ee_right"], model.robot_joint_names)
        q_ = q
        dyn_state.set_q(q_)
        dyn_robot.compute_forward_kinematics(dyn_state)
        T_fk = dyn_robot.compute_transformation(dyn_state, BASE, EE)
        return np.round(T_fk[:3, 3],2)

    marker_transform = Marker_Transform()
    if not example_joint_position_command_2(robot, model_name):
        print("finish motion")
    if not example_cartesian_command_1(robot, model_name):
        print("finish motion")
    result = marker_transform.get_marker_transform()
    print("camera_resulte=")
    print(result)
    print("t5 to ee=")
    print(compute_T_fk(robot.get_state().position))
    if not example_cartesian_command_2(robot, model_name):
        print("finish motion")
    compute_T_fk(robot.get_state().position)
    result = marker_transform.get_marker_transform()
    print("camera_resulte=")
    print(result)
    print("t5 to ee=")
    print(compute_T_fk(robot.get_state().position))
    if not example_cartesian_command_3(robot, model_name):
        print("finish motion")
    compute_T_fk(robot.get_state().position)
    result = marker_transform.get_marker_transform()
    print("camera_resulte=")
    print(result)
    print("t5 to ee=")
    print(compute_T_fk(robot.get_state().position))
    if not example_cartesian_command_4(robot, model_name):
        print("finish motion")
    compute_T_fk(robot.get_state().position)
    result = marker_transform.get_marker_transform()
    print("camera_resulte=")
    print(result)
    print("t5 to ee=")
    print(compute_T_fk(robot.get_state().position))
    # if not go_to_home_pose_1(robot):
    #     print("finish motion")
    if not go_to_home_pose_2(robot, model_name):
        print("finish motion")

    print("end of demo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="09_demo_motion")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default='a', help="Robot Model Name (default: 'a')")
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default=".*",
        help="Servo name regex pattern (default: '.*')",
    )
    args = parser.parse_args()

    main(address=args.address, model_name = args.model, power=args.power, servo=args.servo)
