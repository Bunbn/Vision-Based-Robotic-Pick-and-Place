# Vision-Based-Robotic-Pick-and-Place
Developed and implemented a fully autonomous robotic pick-and-place sorting system that integrates vision-based detection, 3D localization, and motion planning for accurate object sorting in a robotic workspace.

Demonstration:



https://github.com/user-attachments/assets/fdefc919-40c9-4368-a8f0-cb0a83f61745



This project demonstrates the development of a fully autonomous robotic pick-and-place sorting system using the OpenManipulator-X robotic arm and an Intel RealSense D435 camera. The system integrates real-time vision-based object detection, 3D localization, and motion planning to accurately detect, grasp, and sort colored spheres within a defined workspace.

Features
Vision-Based Object Detection:

Utilizes HSV color space for precise color segmentation.
Detects spherical objects using Hough Circle Transform and contour analysis.
Implements image preprocessing techniques (morphological opening/closing) for noise reduction.
3D Localization:

Uses the Perspective-n-Point (PnP) algorithm to calculate the 3D position of objects in the camera frame.
Transforms camera frame coordinates to the robot’s frame using the Kabsch algorithm.
Motion Planning and Control:

Generates smooth and efficient trajectories using cubic/quintic polynomials.
Implements real-time control with velocity clipping to ensure safe operation within joint limits.
System Robustness:

Dynamic error handling to skip unreachable objects and prevent overextension.
Adapts to changes in lighting and workspace conditions for consistent performance.
Autonomous Operation:

Continuously detects, picks, and places objects in sequence.
Enters idle mode and waits for new objects when the workspace is clear.

System Overview

Vision Pipeline:

Convert workspace image to HSV color space for color segmentation.
Detect and classify spheres by color and shape using Hough Circle Transform.
Map detected objects’ coordinates from the camera frame to the robot’s frame.

Sorting Workflow:

Detect all colored spheres in the workspace.
Map sphere positions to robot coordinates.
Generate motion trajectories for the robot to pick and place each sphere in its designated location.
Repeat until all objects are sorted or workspace is clear.
Consistently low error rates (<4%) across various lighting conditions.

Technologies Used:

Robotics: OpenManipulator-X robotic arm, ROS
Computer Vision: OpenCV, HSV color space, Hough Circle Transform, Perspective-n-Point (PnP) algorithm
Programming: Python
Algorithms: Kabsch algorithm, cubic/quintic polynomial trajectory planning, PID control

Challenges and Learnings

Addressed issues with depth perception inaccuracies by switching to PnP-based 3D localization.
Optimized robot speed and reduced trajectory calculation time to minimize lag.
Overcame lighting sensitivity issues by dynamically adjusting image capture timing and camera placement.
Successfully integrated multiple modules from prior work, including inverse kinematics, trajectory planning, and vision-based calibration.
