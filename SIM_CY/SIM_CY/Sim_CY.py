#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
File Name    : SIM.py
Author       : Vishakh Duggal
Created On   : 2025-07-13
Last Updated : 2025-07-13
Description  : [Brief description of what this script does]
               e.g., "Visual servoing for dual-arm coordination in PyBullet"
Python Ver.  : 3.x
Dependencies : [List libraries if needed, e.g., pybullet, numpy, rclpy, cv2]
License      : MIT / Proprietary / GPL-3.0 (as applicable)
=======
"""

# Import all the modules required 
import os
import math 
import numpy as np
import time
import pybullet 
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import cv2
import threading
import csv

# set the path of the URDF 
ROBOT_URDF_PATH_UR5 = "./ur_e_description/urdf/ur5.urdf" # 
ROBOT_URDF_PATH_FRANKA = "./ur_e_description/urdf/fra.urdf" # 
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
ROBOT_BASE_HEIGHT = 0.01  # Depends on your URDF's first link base offset
TABLE_HEIGHT = 0.63
base_z = TABLE_HEIGHT + ROBOT_BASE_HEIGHT


class SIM_CY():
    """
    Main Class which holds all the codes 
    """
    def __init__(self):
        """
        Init of the class which initialises everything 
        """
        # Initialise Pybullet 
        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(True)

        # Initialise the variables 
        self.start_time = time.time() # time for motion
        self.end_effector_index_ur5 = 7 # end effector for UR5
        
        self.rot = 0 # variable for rotation motion 
        self.image = 0 # image holder from camera 
        self.ur5_gripper_link  = 0 # end link
        self.franka_ee_link = 0 # end link
        self.ur5 = 0 # ur5 robot
        self.fran = 0 # franka robot

        # load the robots and tables for the scene 
        self.ur5, self.fran = self.add_scene() # add scene
        self.setup_endlinks() # setup link data 

        self.setupRobotJointInformation() # setup Joint Information 
      
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.end_effector_index_fran = self.find_link_index(self.fran, "panda_rightfinger")

    def find_link_index(self, robot_id, link_name):
        """
    
            find_link_index : Finds the link Index .
            Return : nothing
    
        """
        for i in range(pybullet.getNumJoints(robot_id)):
            name = pybullet.getJointInfo(robot_id, i)[12].decode("utf-8")
          
            if name == link_name:
                return i
        raise ValueError(f"Link '{link_name}' not found")

    def setupRobotJointInformation(self):
        """
        setupRobotJointInformation : Setup Robot Joint Information  .
        Return : nothing
        """
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.control_joints_fran = ["panda_joint1" ,"panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_joint8", "panda_hand_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.loadJointInformationUR5()
        self.loadJointInformationFranka()

    def loadJointInformationUR5(self):
        """
        loadJointInformationUR5 : Loads the UR5 infromation from URDF .
        Return : nothing
        """
        self.joints = AttrDict()
        self.num_joints = pybullet.getNumJoints(self.ur5)
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info  

        
    
    def loadJointInformationFranka(self):
        """
        loadJointInformationFranka : Loads the Franka infromation from URDF .
        Return : nothing
        """
        self.num_joints_fran = pybullet.getNumJoints(self.fran)
        
        self.joints_fran = AttrDict()
        for i in range(self.num_joints_fran):
            info = pybullet.getJointInfo(self.fran, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints_fran[info.name] = info        

        

    def setup_endlinks(self):
        """
        setup_endlinks : Get the End Links Information
        """
        self.ur5_gripper_link = self.find_link_index(self.ur5, "tool0")       # UR5 gripper
        self.franka_ee_link = self.find_link_index(self.fran, "panda_rightfinger") 
        


    def add_ball(self):
        """
        add_ball : Adds ball ( Red to teh Scene )
        """
        ball_radius = 0.03 # Ball radius
        ball_visual = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1]) # set color 
        ball_collision = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=ball_radius) # set collision

        
        # Place on table (e.g., height = 0.63)
        self.ball_position_green = [0.5, 0.0, 0.63 + ball_radius]  # adjust as needed

        self.ball_id_green = pybullet.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=self.ball_position_green
        )

    def add_tables(self):
        """
        add_tables : Adds Tables ( to teh Scene )
        """
        # Load two tables
        tableA = pybullet.loadURDF(TABLE_URDF_PATH, [0.0, 0, 0], useFixedBase=True) # add table 1
        tableB = pybullet.loadURDF(TABLE_URDF_PATH, [1.5, 0, 0], useFixedBase=True) # add table 2 

        pybullet.changeDynamics(        ## prevent ball from rolling on the tables
        bodyUniqueId=tableA,
        linkIndex=-1,  # -1 means base link (for static objects)
        lateralFriction=10.0,       # High friction to prevent sliding
        rollingFriction=1.0,        #  Prevent rolling
        spinningFriction=1.0        #  Prevent spinning
    )


    def add_robots(self):
        """
        
        add_robots : Adds Robots ( to teh Scene )
        
        """

          # Load Robot A (UR5) on Table A
        robotA = pybullet.loadURDF(
            ROBOT_URDF_PATH_UR5,
            basePosition=[0.0, 0, base_z],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),  # Facing +X
            useFixedBase=True
        )

        # Load Robot B (Franka) on Table B
        robotB = pybullet.loadURDF(
            ROBOT_URDF_PATH_FRANKA,
            basePosition=[1.5, 0, base_z],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 3.1415]),  # Facing -X (towards Robot A)
            useFixedBase=True
        )

        # set initial positon Robot A
        ur5_joint_values = [
              0.0,     # joint1
                -0.4,    # joint2
                0.0,     # joint3
                -2.4,    # joint4
                0.0,     # joint5
                2.0,     # joint6
              # wrist_3_joint
        ]

        for joint_id, joint_value in enumerate(ur5_joint_values):
            pybullet.resetJointState(robotA, joint_id, joint_value)

        # set initial positon Robot B
        franka_joint_values = [
        0.0,     # joint1
        -0.4,    # joint2
        0.0,     # joint3
        -2.4,    # joint4
        0.0,     # joint5
        2.0,     # joint6
        0.8,      # joint7
        0.3,
        ]
        for joint_id, joint_value in enumerate(franka_joint_values):
            pybullet.resetJointState(robotB, joint_id, joint_value)  

        return robotA, robotB

    def add_scene(self):
       
        """
        Add Balls, Tables and robot to the systems 
        """
        self.add_tables() # add table 
        self.add_ball() # add_ball
        return self.add_robots()

    def get_current_pose(self, robot, end_effector):
        """
        get_current_pose : Get the current pose of end effector
        """
        linkstate = pybullet.getLinkState(robot, end_effector, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def calculate_ik(self, position, orientation, robot):
        """
        calculate_ik : Calcualte IK for UR5
        """
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        num_joints = 6  # for UR5
        current_joints = [pybullet.getJointState(robot, i)[0] for i in range(num_joints)]

        joint_angles = pybullet.calculateInverseKinematics(  # calculate inverse kinematics 
            robot, self.end_effector_index_ur5, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=current_joints
        )
        return joint_angles
    
    def calculate_ik_fran(self, position, orientation, robot):
        """
        calculate_ik_fran : Calcualte IK for Franka
        """
        quaternion =  orientation#pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*9
        upper_limits = [math.pi]*9
        joint_ranges = [2*math.pi]*9
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
        


        joint_angles = pybullet.calculateInverseKinematics( # calculate inverse kinematics Franka
            robot, self.end_effector_index_fran, position, quaternion, 
            jointDamping=[0.01]*9, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
    
    def getMotion(self, center, orientation):
        """
        getMotion : Get Motion Commands for UR5
        """
        
        current_time = time.time()
        elapsed = current_time - self.start_time  # in seconds

        # Parameters
        r = 0.17
        freq = 0.10  # Hz
        omega = 2 * math.pi * freq
        phase = omega * elapsed

        # Switching control
        cycle_time = 30.0          # total duration of one full cycle
        half_cycle = cycle_time / 2  # duration of each pattern
        t_mod = elapsed % cycle_time
        transition_duration = 1.0    # seconds over which we blend

        # Compute blend factor (0 to 1)
        blend = 0.0
        if abs(t_mod - half_cycle) < transition_duration:
            # Smooth transition using sine-based blend
            blend = 0.5 * (1 + math.sin(math.pi * (t_mod - half_cycle) / transition_duration))
        else:
            blend = 0.0

        # Circular trajectory (Y-Z plane)
        y_circ = center[1] + 0.6 * r * math.cos(phase)
        z_circ = center[2] + 0.6 * r * math.sin(phase)

        # Lissajous trajectory (Y-Z plane), 
        delta = math.pi / 2  # ensures smooth match with circle
        y_liss = center[1] + r * math.cos(2 * phase)  # a = 2
        z_liss = center[2] + r * math.sin(phase)      # b = 1

        # Determine which is active
        if t_mod < half_cycle:
            # Circle is active
            print("Circle")
            y = (1 - blend) * y_circ + blend * y_liss
            z = (1 - blend) * z_circ + blend * z_liss
        else:
            print("Lissajous")
            # Lissajous is active
            y = (1 - blend) * y_liss + blend * y_circ
            z = (1 - blend) * z_liss + blend * z_circ

        # Fixed X and orientation
        x = center[0]
        Rx = Ry = Rz = 0

        return x, y, z, Rx, Ry, Rz
    
    #def getMotion(self, center, orientation):
        

        # In your update loop
    #    current_time = time.time()
    #    elapsed = current_time - self.start_time  # in seconds

        # Motion parameters
    #    x = center[0]  # Fixed X
    #    r = 0.17
    #    freq = 0.13  # Hz
    #    omega = 2 * math.pi * freq

        # Choose trajectory
    #    if  int(elapsed) % 240 < 120:
    #        # Circle in Y-Z plane
            
    #        y = center[1] +  0.6*r * math.cos(omega * self.rot)
    #        z = center[2] + 0.6*r * math.sin(omega * self.rot)
           
    #    else:
            # Lissajous (∞-shaped) in Y-Z plane
    #        a = 2
    #        b = 1
    #        delta = math.pi 
    #        y = center[1] +  r * math.sin(a * self.rot + delta)
    #        z = center[2] +  r * math.sin(b * self.rot )

            

        # Orientation (fixed here)
    #    Rx = Ry = Rz = 0
    #    self.rot = self.rot + 0.00002

    #    return x, y, z, Rx, Ry, Rz
    
    def set_joint_angles(self, joint_angles, robot):
        """
        set_joint_angles : Set Joint Angles for UR5
        """
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray( # move the robot 
            robot, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )

    def pick_ball(self):
        """
        pick_ball : Pick Ball by UR5
        """
        joint_angles = self.calculate_ik(self.ball_position_green, [0, 0, 0], self.ur5)
        self.set_joint_angles(joint_angles, self.ur5)

        gripper_link_index = 6  # replace this with actual link index like 'tool0'
        pybullet.createConstraint(
            parentBodyUniqueId=self.ur5,
            parentLinkIndex=self.ur5_gripper_link,
            childBodyUniqueId=self.ball_id_green,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.1],  # adjust to grip center
            childFramePosition=[0, 0, 0]
        )

    def update_franka_camera(self):
        """
        update_franka_camera : Start Camera attched to Franka and Give Feed
        """
        
        while True:
            # Get camera position and orientation from Franka's end effector
            
            link_state = pybullet.getLinkState(self.fran, self.franka_ee_link)
            cam_pos = link_state[0]
            cam_orient = link_state[1]

            # Convert quaternion to forward and up vectors
            cam_mat = pybullet.getMatrixFromQuaternion(cam_orient)
            forward_vec = np.array([cam_mat[0], cam_mat[3], cam_mat[6]])
            up_vec = np.array([cam_mat[2], cam_mat[5], cam_mat[8]])
            target_pos = np.array(cam_pos) + 0.1 * forward_vec

            view_matrix = pybullet.computeViewMatrix(  # compute view matrix
                cameraEyePosition=cam_pos,
                cameraTargetPosition=target_pos,
                cameraUpVector=up_vec
            )

            projection_matrix = pybullet.computeProjectionMatrixFOV( # compute view FOVC
                fov=60,
                aspect=1.0,
                nearVal=0.01,
                farVal=2.0
            )

            width, height, rgba, _, _ = pybullet.getCameraImage( # compute Generate Image
                width=320,
                height=320,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )

            rgb = np.reshape(rgba, (320, 320, 4))[:, :, :3]
            self.image = rgb.astype(np.uint8)
            time.sleep(0.1)

    def get_current_pose_fran(self, robot):
        """
        get_current_pose_fran : Get Current Pose of the robot Franka
        """
        linkstate = pybullet.getLinkState(robot, 10, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def FollowRobot(self):
        """
        FollowRobot : Follow the robot ( UR5 ) using camera feed in FRANKA 
        """
        with open('./data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['err_z', 'err_y']) 
            while True:
                
                rgb = self.image
                image, center = self.detect_red_ball(rgb.copy())
            
                if center:

                    err_z = center[0] - 320 // 2 # error in position
                    err_y = center[1] - 3 * 320 // 4 # error in position
                    
                    
                    writer.writerow([err_z, err_y])  # Header
                    file.flush()
                    center_robot, orientation = self.get_current_pose_fran(self.fran)

                    joint_angles = self.calculate_ik_fran([1.07556634006991, center_robot[1] - err_z * 0.002, center_robot[2] + err_y * 0.002], orientation, self.fran) # set Joint angles 
                    self.set_joint_angles_fran(joint_angles, self.fran)  # In camera frame: x=left-right, y=up-down 
                else:
                    print("NO CENTERs")
            

                time.sleep(1.0 / 30.0)  # ~30 FPS
    

    def set_joint_angles_fran(self, joint_angles, robot):
        """
        set_joint_angles_fran : Set Joint angles for motion for Franka
        """
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints_fran):
            
           
            joint = self.joints_fran[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)
         
       
        pybullet.setJointMotorControlArray( # set motion for Fran
            robot, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )

    
    def moveURDF(self, center, orientation):
        """
        moveURDF: Move the UR5 Robot in Motion 
        """
       
        while True:
            x,y,z, Rx, Ry,Rz = self.getMotion(center, orientation)
            
            joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz], sim.ur5)
            sim.set_joint_angles(joint_angles, sim.ur5)
            sim.check_collisions()
            time.sleep(0.01)
            
    
    def check_collisions(self):
        """ 
        check_collisions : Check Collision, Currently Not using it 
        """
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False
    
    def detect_red_ball(self, image):
        """
        detect_red_ball : DEtect Red Ball in the Scene and provide coordinate, calc errro and move Franka accordingle
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # required as Pybullet is RGB not BRG
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1) 
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contour 
        center = None
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)
            if radius > 2:  # check ball radius
                center = (int(x), int(y))
                cv2.circle(image, center, int(radius), (0, 255, 0), 2)
                cv2.imshow("Franka Camera View", image)
                cv2.waitKey(1) 
                

        return image, center

if __name__ == "__main__":
    sim = SIM_CY()  # create object 
    
    center, orientation = sim.get_current_pose(sim.ur5, sim.ur5_gripper_link)
    x,y,z, Rx, Ry,Rz = sim.getMotion(center, orientation)
    joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz], sim.ur5)
    sim.set_joint_angles(joint_angles, sim.ur5) # set the robot to initial position
    
    time.sleep(2)
    sim.pick_ball() # pick the red ball 

    time.sleep(2)

    camera_thread = threading.Thread(target=sim.update_franka_camera) # start teh camer a
    camera_thread.daemon = True  # Ensures it exits when the main program exits
    camera_thread.start()

    time.sleep(2)

    follow_thread = threading.Thread(target=sim.FollowRobot) # start Follow thread FRANKA 
    follow_thread.daemon = True  # Ensures it exits when the main program exits
    follow_thread.start()

    motion_thread = threading.Thread(target=sim.moveURDF, args=(center, orientation)) # start the Move UR5 
    motion_thread.daemon = True  # Ensures it exits when the main program exits
    motion_thread.start()



    while True:
        time.sleep(0.1) # wait for infinity 


   

