import cv2
import mediapipe as mp
import numpy as np

# Paths to your video files
video_path_1 = 'biomechanics\\IMG_3085.mp4'  # Amateur golfer video
video_path_2 = 'biomechanics\\scheffler.mp4'  # Professional golfer video

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open the input video files
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

# Get the original frame width and height
original_width_1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height_1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_width_2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height_2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))  # Frames per second (assuming both videos have the same FPS)

# Set the desired width for resizing while maintaining the aspect ratio
desired_width = 400
desired_height_1 = int(original_height_1 * (desired_width / original_width_1))
desired_height_2 = int(original_height_2 * (desired_width / original_width_2))
max_height = max(desired_height_1, desired_height_2)  # Maximum height to pad the shorter video

# Helper function to calculate angles
def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Helper function to get pixel coordinates
def get_pixel_coordinates(landmark, width, height):
    if landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return np.array([x, y])
    return None

# Function to draw angles, pose analysis, and head reference line
def draw_pose_analysis(image, results, width, height, angle_info):
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for key joints and the head
        shoulder_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], width, height)
        shoulder_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], width, height)
        hip_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], width, height)
        hip_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], width, height)
        elbow_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], width, height)
        wrist_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], width, height)
        head = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE.value], width, height)

        # Calculate angles
        shoulder_tilt = calculate_angle(shoulder_left, shoulder_right, [shoulder_left[0], shoulder_left[1] + 1])  # Shoulder tilt angle
        elbow_bend = calculate_angle(shoulder_left, elbow_left, wrist_left)  # Elbow bend angle
        hip_rotation = calculate_angle(hip_left, hip_right, [hip_left[0], hip_left[1] + 1])  # Hip rotation angle
        spine_angle = calculate_angle(hip_left, shoulder_left, [hip_left[0], hip_left[1] - 1])  # Spine angle

        # Store angle information
        angle_info['Shoulder Tilt'] = int(shoulder_tilt) if shoulder_tilt else 'N/A'
        angle_info['Elbow Bend'] = int(elbow_bend) if elbow_bend else 'N/A'
        angle_info['Hip Rotation'] = int(hip_rotation) if hip_rotation else 'N/A'
        angle_info['Spine Angle'] = int(spine_angle) if spine_angle else 'N/A'

        # Draw head reference line
        if head is not None:
            line_color = (0, 255, 0)  # Green color for the line
            line_thickness = 2
            # Draw a vertical line down the middle of the head
            cv2.line(image, (head[0], 0), (head[0], height), line_color, line_thickness)

        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

# Function to draw angle tables
def draw_angle_table(image, angle_info, position):
    # Set the starting position for the table
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Draw each angle in the table
    for idx, (label, angle) in enumerate(angle_info.items()):
        text = f"{label}: {angle}"
        cv2.putText(image, text, (x, y + idx * 20), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    is_paused_1 = False
    is_paused_2 = False

    while cap1.isOpened() and cap2.isOpened():
        angle_info_1 = {'Shoulder Tilt': 'N/A', 'Elbow Bend': 'N/A', 'Hip Rotation': 'N/A', 'Spine Angle': 'N/A'}
        angle_info_2 = {'Shoulder Tilt': 'N/A', 'Elbow Bend': 'N/A', 'Hip Rotation': 'N/A', 'Spine Angle': 'N/A'}

        if not is_paused_1:
            ret1, frame1 = cap1.read()
            if not ret1:
                break
            frame1 = cv2.resize(frame1, (desired_width, desired_height_1))
            if desired_height_1 < max_height:
                padding_top = (max_height - desired_height_1) // 2
                padding_bottom = max_height - desired_height_1 - padding_top
                frame1 = cv2.copyMakeBorder(frame1, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image1.flags.writeable = False
            results1 = pose.process(image1)
            image1.flags.writeable = True
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            draw_pose_analysis(image1, results1, desired_width, max_height, angle_info_1)
            draw_angle_table(image1, angle_info_1, (10, 20))  # Table position on the left video

        if not is_paused_2:
            ret2, frame2 = cap2.read()
            if not ret2:
                break
            frame2 = cv2.resize(frame2, (desired_width, desired_height_2))
            if desired_height_2 < max_height:
                padding_top = (max_height - desired_height_2) // 2
                padding_bottom = max_height - desired_height_2 - padding_top
                frame2 = cv2.copyMakeBorder(frame2, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image2.flags.writeable = False
            results2 = pose.process(image2)
            image2.flags.writeable = True
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            draw_pose_analysis(image2, results2, desired_width, max_height, angle_info_2)
            draw_angle_table(image2, angle_info_2, (desired_width - 150, 20))  # Table position on the right video

        combined_frame = np.hstack((image1, image2))
        cv2.imshow('Side-by-Side Golf Swing Comparison', combined_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Exit
            break
        elif key == ord('1'):  # Pause/resume video 1
            is_paused_1 = not is_paused_1
        elif key == ord('2'):  # Pause/resume video 2
            is_paused_2 = not is_paused_2

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
