import time, pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pynput.mouse import Listener


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, WhiteKernel, DotProduct, ConstantKernel

records = pickle.load(open('records_sample.pkl', 'rb'))
X = [ r['landmark'] for r in records ]
X = np.array(X).reshape(-1, 478*3)
y = [ r['coord'] for r in records ]
y = np.array(y)

print(X.shape, y.shape)

def predict_landmark(landmark):
    X = np.array([ [d.x, d.y, d.z] for d in landmark ])
    X = X.reshape(1, 478*3)
    return model.predict(X)[0]

def predict_landmark_smooth(landmark):
    if len(coords_traj) > 0:
        prev = coords_traj[-1]
        curr = predict_landmark(landmark)
        curr = prev*0.8 + curr*0.2
    else:
        curr = predict_landmark(landmark)
    return curr

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = GaussianProcessRegressor(kernel=RationalQuadratic())
model.fit(X, y)

# 웹캠 화면이 최상단에 뜨게 하기 위한 설정
# 최상단을 유지하려는 경우에만 주석을 해제해주세요
# cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
# cv2.setWindowProperty('window', cv2.WND_PROP_TOPMOST, 1)

coords_traj = []

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            break

        image.flags.writeable = False # 성능 향상을 위해 이미지를 읽기 전용으로 만듭니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                
            image = cv2.flip(image, 1)
            coord = predict_landmark_smooth(face_landmarks.landmark)
            coords_traj.append(coord)
            cv2.putText(image, f'x: {coord[0]:.2f}, y: {coord[1]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('window', image)
        # 이미지 재생을 위한 대기시간
        cv2.waitKey(5)
        
cap.release()