import cv2
import dlib
import numpy as np
import math

# 두 점 사이의 거리를 계산하는 함수
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 점수를 계산하는 함수 (사용자 비율, 이상적 비율)
def calculate_score(user_ratio, ideal_ratio):
    # 차이가 적을수록 100점에 가까워짐
    score = 100 - abs(user_ratio - ideal_ratio) / ideal_ratio * 100
    return max(0, score) # 점수가 0점 이하로 내려가지 않도록 함

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\landmark_model\shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        my_points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

        # --- 마쿼트 마스크의 주요 비율 계산 ---
        
        # 기준 수치 (콧볼 너비)
        standard_width = distance(my_points[31], my_points[35])
        
        # 1. 눈 사이 거리 (이상적 비율: 황금비 1.618 ^ 2)
        eye_width = distance(my_points[36], my_points[45])
        ratio_eyes = eye_width / standard_width if eye_width > 0 else 0
        score1 = calculate_score(ratio_eyes, 1.618 ** 2)

        # 2. 눈썹 사이 거리 (이상적 비율: 1.618)
        eye_width_left = distance(my_points[36], my_points[39])
        inter_eye_dist = distance(my_points[39], my_points[42])
        ratio_eye = inter_eye_dist / eye_width_left if eye_width_left > 0 else 0
        score2 = calculate_score(ratio_eye, 1.0)
        
        # 3. 얼굴 너비 / 얼굴 길이 (이상적 비율: 황금비 1.618)
        face_width = distance(my_points[0], my_points[16])
        face_height = distance(my_points[8], my_points[27])
        ratio_face_shape = face_width / face_height if face_height > 0 else 0
        score3 = calculate_score(ratio_face_shape, 1.618)

        # 최종 평균 점수 계산
        final_score = (score1 + score2 * 2 + score3 * 2) / 5
        
        # --- 결과 시각화 ---
        x, y, w, h = cv2.boundingRect(my_points)
        
        # 각 비율 정보 표시
        cv2.putText(frame, f"1. Mouth/Nose: {score1:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"2. Eye Ratio: {score2:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"3. Face Shape: {score3:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 최종 점수 표시
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 180, 0), -1)
        cv2.putText(frame, f"Total Score: {final_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 얼굴 윤곽선 그리기 (기존 코드 재활용 가능)
        cv2.polylines(frame, [my_points[0:17].astype(np.int32)], False, (0, 255, 0), 1)


    cv2.imshow('Marquardt Mask Final Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()