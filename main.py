import cv2
import dlib
import numpy as np
import math

# 두 점 사이의 거리를 계산하는 함수
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def face_height_distance(left_p1, right_p1, left_p2, right_p2):
    return abs(((left_p1[1] + right_p1[1])/2) - ((left_p2[1] + right_p2[1])/2))

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
        
        # 1. 입술 너비 - 콧볼 너비 (이상적 비율: 황금비 1.618)
        nose_width = distance(my_points[31], my_points[35])
        lip_width = distance(my_points[48], my_points[54])
        ratio_nose_lip = lip_width / nose_width if nose_width > 0 else 0
        score1 = calculate_score(ratio_nose_lip, 1.618)
        
        # 2. 콧볼 너비 - 눈 거리 (이상적 비율: 황금비 1.618 ^ 2)
        eye_width = distance(my_points[36], my_points[45])
        ratio_nose_eye = eye_width / nose_width if nose_width > 0 else 0
        score2 = calculate_score(ratio_nose_eye, 1.618 ** 2)

        # 3. 콧볼 너비 - 왼쪽 눈썹 길이  (이상적 비율: 1.618)
        left_eyebrow_width = distance(my_points[17], my_points[21])
        ratio_eyebrow_left = left_eyebrow_width / nose_width if nose_width > 0 else 0
        score3 = calculate_score(ratio_eyebrow_left, 1.618)
        
        # 4. 콧볼 너비 - 오른쪽 눈썹 길이  (이상적 비율: 1.618)
        right_eyebrow_width = distance(my_points[22], my_points[26])
        ratio_eyebrow_right = right_eyebrow_width / nose_width if nose_width > 0 else 0
        score4 = calculate_score(ratio_eyebrow_right, 1.618)
        
        # 5. 눈과 눈 사이 길이 - 미간 길이 (이상적 비율: 황금비 1.0)
        glabella_width = distance(my_points[21], my_points[22])
        inner_eye_width = distance(my_points[39], my_points[42])
        ratio_glabella_eye = inner_eye_width / glabella_width if glabella_width > 0 else 0
        score5 = calculate_score(ratio_glabella_eye, 1.0)
        
        # 6. 얼굴 길이 (이상적 비율: 황금비 1.618)
        upper_face_height = face_height_distance(my_points[0], my_points[16], my_points[48], my_points[54])
        lower_face_height = face_height_distance(my_points[48], my_points[54], my_points[8], my_points[8])
        ratio_face_height = upper_face_height / lower_face_height if lower_face_height > 0 else 0
        score6 = calculate_score(ratio_face_height, 1.618)

        # 가중치 적용 계산
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        scores = [score1, score2, score3, score4, score5, score6]
        
        # 최종 평균 점수 계산
        final_score = sum(s*w for s,w in zip(scores, weights)) / sum(weights)
        
        # --- 결과 시각화 ---
        x, y, w, h = cv2.boundingRect(my_points)
        
        # 각 비율 정보 표시
        cv2.putText(frame, f"1. nose/lip: {score1:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, f"2. nose/eye: {score2:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, f"3. left eyebrow: {score3:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, f"4. right eyebrow: {score4:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, f"5. grabella: {score5:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, f"6. face shape: {score6:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 최종 점수 표시
        cv2.putText(frame, f"Total Score: {final_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 211, 255), 2)
        
        # 얼굴 윤곽선 그리기 (기존 코드 재활용 가능)
        cv2.polylines(frame, [my_points[0:17].astype(np.int32)], False, (0, 255, 0), 1)


    cv2.imshow('Marquardt Mask Final Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()