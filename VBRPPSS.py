import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import pickle

from classes.Robot import Robot
from classes.Realsense import Realsense
from classes.AprilTags import AprilTags
from classes.TrajPlanner import TrajPlanner

camera_robot_transform_file = "Lab6_2transform.npy"


def detect_colored_spheres(image, extract_hsv_values=False, roi_corners=None):
    """
    - Input: Path to an image file containing the robot workspace
    - Returns:
        1. An annotated image with green circles drawn around
           spheres, red center markers, and color labels
        2. List of tuples containing sphere color (string)
           and centroid coordinates (x, y)
    - Description: Detects and classifies colored spheres using
                   computer vision techniques
        * The method should process the image to enhance sphere detection
        * Implement either Hough circle detection or contour analysis
        * Classify each detected sphere by color using HSV color space
        * Output detection results to terminal in the format: "Color: (x, y)"
    """

    # Convert to HSV color space so hue is described in just one channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red":    np.array([(159, 50, 70), (180, 255, 200)], dtype="uint8"),
        "red2":   np.array([(0, 64, 64), (5, 255, 200)], dtype="uint8"),
        "orange": np.array([(5, 20, 30), (20, 255, 200)], dtype="uint8"),
        "yellow": np.array([(21, 20, 20), (65, 255, 200)], dtype="uint8"),
        "blue":   np.array([(95, 64, 64), (140, 255, 200)], dtype="uint8"),
    }

    # Preallocate space for detected spheres (based on an estimated maximum count)
    max_spheres = 10  # Adjust based on expected maximum
    detected_spheres = [None] * max_spheres
    sphere_index = 0

    # Prepare a single morphological kernel for all operations
    kernel = np.ones((5, 5), np.uint8)

    # Draw the bounding box if provided
    if roi_corners:
        x_min, y_min = roi_corners[0]
        x_max, y_max = roi_corners[1]
        cv2.rectangle(image, (x_min, y_min),
                      (x_max, y_max), (255, 255, 255), 2)

    # For each color range
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        # Create a binary mask for the color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        # Handle red wrap-around in HSV space
        if color_name == "red":
            lower2, upper2 = color_ranges["red2"]
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)  # Combine both masks
        elif color_name == "red2":
            continue

        # Morphological operations to clean up the mask and any gaps
        #   Open: Erode followed by dilate to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #   Close: Dilate followed by erode to fill in gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Detect contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the shape and filter based on circularity
            area = cv2.contourArea(contour)
            # Skip small or excessively large objects
            if area < 500 or area > 2000:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:  # Prevent division by zero
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            # Skip non-circular objects
            if circularity < 0.7 or circularity > 1.2:
                continue

            # Calculate the centroid
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            # Skip if moments are invalid
            else:
                continue

            # Check if centroid is within the ROI
            if roi_corners and not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                continue

            # Draw the detected circle and centroid marker
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(image, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                image,
                color_name,
                (center[0] + radius + 5, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Extract HSV values within the detected circle
            if extract_hsv_values:
                mask_circle = np.zeros_like(mask)
                cv2.circle(mask_circle, center, radius, 255, -1)
                sphere_hsv = hsv[mask_circle == 255]

                if sphere_hsv.size > 0:
                    min_hsv = sphere_hsv.min(axis=0)
                    max_hsv = sphere_hsv.max(axis=0)
                else:
                    # Fallback in case of no values
                    min_hsv = max_hsv = [0, 0, 0]

            # Store the detected sphere in the preallocated list
            if extract_hsv_values:
                detected_spheres[sphere_index] = {
                    "color": color_name,
                    "center": {"x": center[0], "y": center[1]},
                    "radius": radius,
                    "hsv_range": [min_hsv, max_hsv],
                }
            else:
                detected_spheres[sphere_index] = {
                    "color": color_name,
                    "center": {"x": center[0], "y": center[1]},
                    "radius": radius,
                }
            sphere_index += 1

            # Stop early if maximum sphere count is reached
            if sphere_index >= max_spheres:
                break

    # Trim unused preallocated space
    detected_spheres = detected_spheres[:sphere_index]

    return detected_spheres, image


def get_sphere_pose(xyr, intrinsics, radius_size=15):
    """
    Args:
        xyr: 3x1 array of sphere center coordinates and radius in image frame
        intrinsics: Camera intrinsic parameters (from RealSense)
        radius_size (float): Physical size of the sphere radius in millimeters
    Returns:
        tuple: (rotation_matrix, translation_vector)
        - rotation_matrix: 3x3 rotation matrix from object to camera frame
        - translation_vector: 3x1 translation vector from object to camera frame in mm
    """

    x = xyr[0]
    y = xyr[1]
    r = xyr[2]

    try:
        # Define 3D model points (sphere "corners" in sphere frame)
        object_points = np.array([
            [0, radius_size, 0],    # top
            [0, -radius_size, 0],   # bottom
            [radius_size, 0, 0],    # right
            [-radius_size, 0, 0],   # left
        ]).astype(np.float32)

        # Define image points for PnP
        image_points = np.array([
            [x, y+r],  # top
            [x, y-r],  # bottom
            [x+r, y],  # right
            [x-r, y],  # left
        ]).astype(np.float32)

        # Construct camera matrix from intrinsics
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])

        # Solve PnP to get tag pose
        _, rvec, tvec = cv2.solvePnP(object_points,
                                     image_points,
                                     camera_matrix,
                                     None)

        # Convert rotation vector to matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)

        return rot_matrix, tvec

    except Exception as e:
        print(f"Error estimating tag pose: {str(e)}")
        return None, None


def goto_pose(robot_obj, target_pose, velocity=80, time_step=0.02):
    # Check if target pose is even reachable
    try:
        robot_obj.get_ik(target_pose)
    except ValueError as e:
        raise ValueError(str(e))

    current_ee_pose = robot_obj.get_ee_pos(
        robot_obj.get_joints_readings()[0])[:4]
    # print(f'current_ee_pose={current_ee_pose}')

    # Calculate Euclidean distance between current and target poses (ignoring pitch)
    distance = np.linalg.norm(
        np.array(current_ee_pose[:3]) - np.array(target_pose[:3]))

    # Calculate trajectory time based on the desired average velocity (mm/s)
    traj_time = distance / velocity  # Time = Distance / Velocity

    # Calculate the required number of points based on the time_step
    # Total points = traj_time / time_step
    points_num = int(traj_time / time_step)

    # Define setpoints
    ee_poses = np.array([
        current_ee_pose,
        target_pose,
    ])

    # Create trajectory between setpoint angles
    tj = TrajPlanner(ee_poses)
    trajectories = tj.get_quintic_traj(traj_time, points_num)

    # Send to first vertex to start
    robot_obj.write_motor_state(True)  # Write position mode
    robot_obj.write_time(traj_time)
    try:
        robot_obj.write_joints(robot_obj.get_ik(trajectories[0, 1:]))
    except ValueError as e:
        raise ValueError(str(e))
    time.sleep(traj_time)  # Wait for trajectory completion

    robot_obj.write_time(time_step)
    start_time = time.time()

    # Move the robot along all trajectories
    for i in range(1, len(trajectories)):
        try:
            robot_obj.write_joints(robot_obj.get_ik(trajectories[i, 1:]))
        except ValueError as e:
            raise ValueError(str(e))
        while time.time() - start_time < (i * time_step):
            pass


def main():
    try:
        # Initalize robot
        robot = Robot()
        robot.write_gripper(True)  # Open gripper
        traj_time = 3
        robot.write_time(traj_time)
        robot.write_motor_state(True)
        # Program
        goto_pose(robot, [25, -100, 150, -90])  # Write joints to home position

        # Initialize camera and detector
        camera = Realsense()
        intrinsics = camera.get_intrinsics()

        # Load the saved calibration transform
        try:
            T_cam_robot = np.load(camera_robot_transform_file)
            print("Loaded camera-robot transformation matrix:")
            # print(T_cam_robot)
        except FileNotFoundError:
            print(
                f"Error: Calibration transform file {camera_robot_transform_file} not found.")
            return

        # Constants
        # Aim the end effector 100 mm above the detected tag
        EE_X_OFFSET = -30  # mm
        EE_Y_OFFSET = 0  # mm
        EE_Z_OFFSET = 108  # mm
        EE_PITCH_OFFSET = 10  # degrees
        RADIUS_SIZE = 15  # mm

        while True:
            # 1. Capture frame from the RealSense camera
            color_frame, delay = None, 0
            while (delay < 55):
                color_frame, _ = camera.get_frames()
                if color_frame is None:
                    print("Warning: No frame captured from the camera.")
                    continue
                delay += 1

            # 2. Detect spheres
            detected_spheres, image = detect_colored_spheres(
                color_frame, roi_corners=[(1, 1), (600, 600)])

            # 3. Show frame
            cv2.imshow("Detected spheres", image)
            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if len(detected_spheres) != 0:
                xyr = (detected_spheres[0]["center"]["x"], detected_spheres[0]
                       ["center"]["y"], detected_spheres[0]["radius"])
                rotation_matrix, translation_vector = get_sphere_pose(
                    xyr, intrinsics, RADIUS_SIZE)

                # Create a (4 x 4) homogeneous transformation matrix for sphere pose
                #   Put them next to each other to get (3 x 4)
                T_sphere_cam = np.hstack([rotation_matrix, translation_vector])
                #   Add row of [0, 0, 0, 1] to make homogenous
                T_sphere_cam = np.vstack(
                    [T_sphere_cam, np.array([0, 0, 0, 1])])
                # print(f"T_sphere_cam:\n{T_sphere_cam})

                # Get sphere frame to robot base frame transformation matrix
                T_sphere_robot = T_cam_robot @ T_sphere_cam
                # Assume ball height is 0 and pitch is -90, so only using first two rows of T_sphere_robot (x and y)
                ball_pose = np.hstack([T_sphere_robot[:2, 3], [0, -90]])
                print("Detected ball pose:", ball_pose)
                # Get desired end effector position by adding offset to sphere position
                ball_offset_pose = ball_pose + \
                    np.array([EE_X_OFFSET, EE_Y_OFFSET,
                             EE_Z_OFFSET, EE_PITCH_OFFSET])
                # Go to first ball from home with offset
                goto_pose(robot, ball_offset_pose)

                # Lower end effector to within grasping distance of ball
                ball_grasp_pose = ball_pose + \
                    np.array([EE_X_OFFSET, EE_Y_OFFSET, 40, EE_PITCH_OFFSET])
                goto_pose(robot, ball_grasp_pose)

                # Grasp ball (close gripper)
                robot.write_gripper(False)

                # Raise end effector to clear floor
                goto_pose(robot, ball_offset_pose)

                # Go to corresponding bin
                if detected_spheres[0]["color"] == "red":
                    goto_pose(robot, [160, 160, 60, -30])
                elif detected_spheres[0]["color"] == "orange":
                    goto_pose(robot, [50, 160, 60, -30])
                elif detected_spheres[0]["color"] == "yellow":
                    goto_pose(robot, [160, -160, 60, -30])
                elif detected_spheres[0]["color"] == "blue":
                    goto_pose(robot, [-50, -160, 60, -30])

                # Release ball (open gripper)
                robot.write_gripper(True)
                goto_pose(robot, [25, -100, 150, -90])
                time.sleep(traj_time)

            # Only write velocities if a tag is detected, otherwise stop motion
            if len(detected_spheres) == 0:
                print("Tracking status: Sphere(s) not detected")
                # Write joints to home position
                goto_pose(robot, [25, -100, 150, -90])
                time.sleep(traj_time)

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
