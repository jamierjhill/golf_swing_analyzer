import cv2
import mediapipe as mp
import numpy as np

# Path to your video file
video_path = 'biomechanics\\IMG_3085.mp4'  # Use double backslashes or a raw string

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open the input video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the original frame width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Set the desired width and calculate the new height to maintain the aspect ratio
desired_width = 400
aspect_ratio = original_height / original_width
desired_height = int(desired_width * aspect_ratio)

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    is_paused = False  # Variable to keep track of the pause state

    # Initialize dictionaries to store min and max angles
    min_angles = {
        'Left Elbow': float('inf'),
        'Right Elbow': float('inf'),
        'Left Knee': float('inf'),
        'Right Knee': float('inf')
    }
    max_angles = {
        'Left Elbow': float('-inf'),
        'Right Elbow': float('-inf'),
        'Left Knee': float('-inf'),
        'Right Knee': float('-inf')
    }

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame while maintaining the aspect ratio
            frame = cv2.resize(frame, (desired_width, desired_height))

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Helper function to convert normalized coordinates to pixel coordinates
                def get_pixel_coordinates(landmark):
                    if landmark:
                        x = int(landmark.x * desired_width)
                        y = int(landmark.y * desired_height)
                        return np.array([x, y])
                    return None

                # Get coordinates for the nose and shoulders
                nose = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE.value])
                shoulder_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                shoulder_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

                # Draw extended line through the shoulders
                if shoulder_left is not None and shoulder_right is not None:
                    # Calculate the direction vector for the shoulder line
                    direction_vector = shoulder_right - shoulder_left
                    unit_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the vector
                    extension_length = 100  # Length to extend the line on each side

                    # Extend the line in both directions
                    extended_point_left = shoulder_left - unit_vector * extension_length
                    extended_point_right = shoulder_right + unit_vector * extension_length

                    # Draw the extended shoulder line
                    cv2.line(image, tuple(extended_point_left.astype(int)), tuple(extended_point_right.astype(int)), (0, 255, 0), 2)  # Green line

                # Draw a line down from the nose
                if nose is not None:
                    line_length = 150  # Length of the line going down from the nose
                    nose_point_down = nose + np.array([0, line_length])  # Extend the line downward

                    # Draw the line down from the nose
                    cv2.line(image, tuple(nose.astype(int)), tuple(nose_point_down.astype(int)), (0, 255, 0), 2)  # Green line

                # Get coordinates for the elbows and knees
                elbow_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                wrist_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                elbow_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                wrist_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                hip_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                knee_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
                ankle_left = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                hip_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                knee_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
                ankle_right = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

                # Function to calculate angle if all points are valid
                def calculate_angle(a, b, c):
                    if a is None or b is None or c is None:
                        return None
                    a = np.array(a)  # First
                    b = np.array(b)  # Mid
                    c = np.array(c)  # End
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    if angle > 180.0:
                        angle = 360 - angle
                    return angle

                # Calculate angles for the elbows and knees
                angle_left_elbow = calculate_angle(shoulder_left, elbow_left, wrist_left)
                angle_right_elbow = calculate_angle(shoulder_right, elbow_right, wrist_right)
                angle_left_knee = calculate_angle(hip_left, knee_left, ankle_left)
                angle_right_knee = calculate_angle(hip_right, knee_right, ankle_right)

                # Update min and max angles
                if angle_left_elbow is not None:
                    min_angles['Left Elbow'] = min(min_angles['Left Elbow'], angle_left_elbow)
                    max_angles['Left Elbow'] = max(max_angles['Left Elbow'], angle_left_elbow)
                if angle_right_elbow is not None:
                    min_angles['Right Elbow'] = min(min_angles['Right Elbow'], angle_right_elbow)
                    max_angles['Right Elbow'] = max(max_angles['Right Elbow'], angle_right_elbow)
                if angle_left_knee is not None:
                    min_angles['Left Knee'] = min(min_angles['Left Knee'], angle_left_knee)
                    max_angles['Left Knee'] = max(max_angles['Left Knee'], angle_left_knee)
                if angle_right_knee is not None:
                    min_angles['Right Knee'] = min(min_angles['Right Knee'], angle_right_knee)
                    max_angles['Right Knee'] = max(max_angles['Right Knee'], angle_right_knee)

                # Helper function to draw angle text
                def draw_angle(image, point, label, angle):
                    if point is None or angle is None:
                        return
                    text_position = (point[0], point[1] - 10)  # Position text above the point
                    cv2.putText(image, f'{label}: {int(angle)}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw angles for elbows and knees
                draw_angle(image, elbow_left, 'LE', angle_left_elbow)  # LE = Left Elbow
                draw_angle(image, elbow_right, 'RE', angle_right_elbow)  # RE = Right Elbow
                draw_angle(image, knee_left, 'LK', angle_left_knee)  # LK = Left Knee
                draw_angle(image, knee_right, 'RK', angle_right_knee)  # RK = Right Knee

                # Draw the full pose landmarks and connections, excluding face landmarks except the nose
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Draw the min and max angles table on the video
            y_offset = 20  # Starting y position for the table
            for joint, min_angle in min_angles.items():
                max_angle = max_angles[joint]
                text = f'{joint}: Min {int(min_angle)}, Max {int(max_angle)}'
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20  # Increment y position for the next line

            # Show the frame in a pop-up window
            cv2.imshow('Mediapipe Pose Analysis with Extended Lines and Angles', image)

        # Handle key events
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord(' '):  # Press spacebar to pause/resume
            is_paused = not is_paused
        elif key == ord('a'):  # Press 'a' to rewind
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - fps * 2))  # Rewind 2 seconds
        elif key == ord('d'):  # Press 'd' to fast-forward
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + fps * 2))  # Fast-forward 2 seconds

    # Properly release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
