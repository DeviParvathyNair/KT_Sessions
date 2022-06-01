import errno
import os
import mediapipe as mp
import cv2
import json


def summary(video_path):
    b_p = os.path.basename(video_path)
    file = os.path.splitext(b_p)
    file = file[0]  # taking the name of the file
    cap = cv2.VideoCapture(video_path)
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if file.count("_") == 2:
        env = "studio"
        s_id = file.split('_')[1]
        signer_id = s_id.split('S')[1]
        gloss_id = file.split('_')[2]
        position = "S"
    else:
        env = "home"
        s_id = file.split('_')[1]
        signer_id = s_id.split('S')[1]
        gloss_id = file.split('_')[2]
        position = file.split('_')[3]
    return signer_id, gloss_id, position, num_of_frames, width, height, fps, env


def pose_estimates(video_path, save_dir):
    # open video file obtained from path
    cap = cv2.VideoCapture(video_path)
    holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_x = []
    pose_y = []
    right_hand_x = []
    right_hand_y = []
    left_hand_x = []
    left_hand_y = []
    while True:
        ret, frame = cap.read()
        if ret:
            converttoRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_model.process(converttoRGB)
            if results.pose_landmarks:
                for ps in results.pose_landmarks.landmark:
                    pose_x.append(ps.x)
                    pose_y.append(ps.y)
            else:
                pose_x.append('NaN')
                pose_y.append('NaN')
            if results.left_hand_landmarks:
                for lh in results.left_hand_landmarks.landmark:
                    left_hand_x.append(lh.x)
                    left_hand_y.append(lh.y)
            else:
                left_hand_x.append('NaN')
                left_hand_y.append('NaN')
            if results.right_hand_landmarks:
                for rh in results.right_hand_landmarks.landmark:
                    right_hand_x.append(rh.x)
                    right_hand_y.append(rh.y)
            else:
                right_hand_x.append('NaN')
                right_hand_y.append('NaN')
        else:
            break
    b_p = os.path.basename(video_path)
    file = os.path.splitext(b_p)
    file = file[0]
    if not os.path.exists(os.path.dirname(save_dir)):
        try:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    pose_path = save_dir + file + '_poseestimate.json'
    with open(pose_path, "w") as op:
        pos_est_dic = {"pose_x": pose_x, "pose_y": pose_y, "hand1_x": right_hand_x, "hand1_y": right_hand_y,
                       "hand2_x": left_hand_x, "hand2_y": left_hand_y}
        json.dump(pos_est_dic, op)
    return pose_path


def crop_video(video_path, save_dir):
    b_p = os.path.basename(video_path)
    file = os.path.splitext(b_p)
    file = file[0]
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists(os.path.dirname(save_dir)):
        try:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    rgb_path = save_dir + file + 'cropped_output.avi'
    out = cv2.VideoWriter(rgb_path, fourcc, 5, (320, 320))
    while True:
        ret, frame = cap.read()
        if ret:
            b = cv2.resize(frame, (320, 320), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(b)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return rgb_path
