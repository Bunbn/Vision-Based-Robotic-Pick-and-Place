# Vision-Based-Robotic-Pick-and-Place
Developed and implemented a fully autonomous robotic pick-and-place sorting system that integrates vision-based detection, 3D localization, and motion planning for accurate object sorting in a robotic workspace.

Key Contributions:

Camera Integration & Image Processing: Utilized an Intel RealSense D435 camera and OpenCV to implement a vision system that detects and classifies colored spheres in the robot’s workspace. Employed HSV color space and Hough Circle Transform for accurate color segmentation and sphere detection.
Coordinate Transformation: Applied the Kabsch algorithm for camera-to-robot coordinate frame transformation, enabling precise 3D localization of objects in the robot’s frame.
Motion Planning & Control: Implemented real-time trajectory planning and joint-level control for the OpenManipulator-X robot, optimizing movement to minimize errors and avoid robot joint limits using velocity clipping.
Error Analysis & Performance Optimization: Achieved consistent low error rates in object detection (below 4%) across various lighting conditions. Improved system efficiency by minimizing computation and optimizing movement time.
System Robustness: Incorporated multiple failure detection and prevention mechanisms to handle robot errors and lighting inconsistencies, including dynamic adjustment of camera frame data collection.
Autonomous Operation: Designed the system to autonomously detect, pick, and place objects in sequence, with the robot returning to idle mode upon completion.
Skills & Technologies:

Robotics: OpenManipulator-X, Intel RealSense D435, motion planning, PID control
Programming: Python, OpenCV, ROS
Computer Vision: Object detection, coordinate transformation, Hough Circle Transform, HSV color space, PnP algorithm
Systems Integration: Real-time control loop, trajectory planning, error handling
Outcome: Successfully developed a robust sorting system that demonstrated high accuracy and real-time adaptability in a variety of environments, achieving consistent, reliable performance in object sorting tasks.
