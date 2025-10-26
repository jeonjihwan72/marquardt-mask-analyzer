import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False,
                                  max_num_faces=1,
                                  min_detection_confidence = 0.5,
                                  min_tracking_confidence = 0.5)

# 두 점 사이의 거리를 계산하는 함수
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def face_height_distance(left_p1, right_p1, left_p2, right_p2):
    return abs(((left_p1[1] + right_p1[1])/2) - ((left_p2[1] + right_p2[1])/2))

# 점수를 계산하는 함수 (사용자 비율, 이상적 비율)
def calculate_score(user_ratio, ideal_ratio, multiplier=1.5):
    percent_diff = abs(user_ratio - ideal_ratio) / ideal_ratio
    penalty = percent_diff * 100 * multiplier # 페널티를 multiplier 만큼 증폭
    score = 100 - penalty
    return max(0, score)

# Dlib 랜드마크 인덱스 -> MediaPipe 랜드마크 인덱스 (대략적인 매핑)
mp_points_map = {
    'nose_left_wing': 64,  # 콧볼 왼쪽 끝 (Dlib 31보다 바깥쪽)
    'nose_right_wing': 294, # 콧볼 오른쪽 끝 (Dlib 35보다 바깥쪽)
    'nose_tip' : 19,    # 코끝

    'lip_left_corner': 61,  # 입술 왼쪽 끝
    'lip_right_corner': 291,    # 입술 오른쪽 끝

    'eye_left_outer': 130, # 왼쪽 눈 바깥쪽
    'eye_right_outer': 359, # 오른쪽 눈 바깥쪽

    'eyebrow_left_start': 55,  # 왼쪽 눈썹 시작점 (코 쪽)
    'eyebrow_left_end': 70,   # 왼쪽 눈썹 바깥쪽 끝
    'eyebrow_right_start': 285, # 오른쪽 눈썹 시작점 (코 쪽)
    'eyebrow_right_end': 300,  # 오른쪽 눈썹 바깥쪽 끝

    'jaw_left_start': 127, # 턱선 왼쪽 (대략)
    'jaw_right_end': 356,  # 턱선 오른쪽 (대략)
    'chin_tip': 152,      # 턱 끝

    'central_eyebrow' : 9   # 미간 중앙
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe는 RGB 이미지를 처리하므로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # MediaPipe 랜드마크를 numpy 배열로 변환
            # 각 랜드마크는 x, y, z 좌표와 가시성/존재 확률을 가짐
            # 우리는 x, y만 필요
            mp_points = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                                  for lm in face_landmarks.landmark], dtype=np.int32)

            # --- 마쿼트 마스크의 주요 비율 계산 ---
            
            # 랜드마크 존재 여부 확인 후 계산 (없으면 0 처리)
            # 콧볼 끝
            p_nose_left_wing = mp_points[mp_points_map['nose_left_wing']]
            p_nose_right_wing = mp_points[mp_points_map['nose_right_wing']]
            
            # 입술 양 끝
            p_lip_left = mp_points[mp_points_map['lip_left_corner']]
            p_lip_right = mp_points[mp_points_map['lip_right_corner']]
            
            # 눈 바깥쪽
            p_eye_left_outer = mp_points[mp_points_map['eye_left_outer']]
            p_eye_right_outer = mp_points[mp_points_map['eye_right_outer']]
            
            # 왼쪽 눈썹
            p_eyebrow_left_start = mp_points[mp_points_map['eyebrow_left_start']]
            p_eyebrow_left_end = mp_points[mp_points_map['eyebrow_left_end']]
            
            # 오른쪽 눈썹
            p_eyebrow_right_start = mp_points[mp_points_map['eyebrow_right_start']]
            p_eyebrow_right_end = mp_points[mp_points_map['eyebrow_right_end']]
            
            # 턱선
            p_jaw_left_start = mp_points[mp_points_map['jaw_left_start']]
            p_jaw_right_end = mp_points[mp_points_map['jaw_right_end']]
            
            # 턱 끝
            p_chin_tip = mp_points[mp_points_map['chin_tip']]
            
            # 눈썹 끝 선
            p_central_eyebrow = mp_points[mp_points_map['central_eyebrow']]
            
            # 코 끝
            p_nose_tip = mp_points[mp_points_map['nose_tip']]


            # 1. 입술 너비 - 콧볼 너비 (이상적 비율: 황금비 1.618)
            nose_width = distance(p_nose_left_wing, p_nose_right_wing)
            lip_width = distance(p_lip_left, p_lip_right)
            ratio_nose_lip = lip_width / nose_width if nose_width > 0 else 0
            score1 = calculate_score(ratio_nose_lip, 1.618, 1.0)
            
            # 2. 콧볼 너비 - 눈 거리 (이상적 비율: 황금비 1.618 ^ 2)
            eye_outer_width = distance(p_eye_left_outer, p_eye_right_outer) # 눈 전체 너비
            ratio_nose_eye = eye_outer_width / nose_width if nose_width > 0 else 0
            score2 = calculate_score(ratio_nose_eye, 1.618 ** 2, 2.0)

            # 3. 콧볼 너비 - 왼쪽 눈썹 길이  (이상적 비율: 1.618)
            left_eyebrow_width = distance(p_eyebrow_left_start, p_eyebrow_left_end)
            ratio_eyebrow_left = left_eyebrow_width / nose_width if nose_width > 0 else 0
            score3 = calculate_score(ratio_eyebrow_left, 1.618, 1.5)
            
            # 4. 콧볼 너비 - 오른쪽 눈썹 길이  (이상적 비율: 1.618)
            right_eyebrow_width = distance(p_eyebrow_right_start, p_eyebrow_right_end)
            ratio_eyebrow_right = right_eyebrow_width / nose_width if nose_width > 0 else 0
            score4 = calculate_score(ratio_eyebrow_right, 1.618, 1.5)
            
            # 5. 얼굴 길이 (이상적 비율: 황금비 1.618)
            upper_face_height = face_height_distance(p_jaw_left_start, p_jaw_right_end, p_lip_left, p_lip_right)
            lower_face_height = face_height_distance(p_lip_left, p_lip_right, p_chin_tip, p_chin_tip) # 턱 끝은 한 점이므로, 양쪽을 같은 점으로 전달
            ratio_face_height = upper_face_height / lower_face_height if lower_face_height > 0 else 0
            score5 = calculate_score(ratio_face_height, 1.618, 2.0)
            
            # 6. 얼굴 중안부와 하안부 비율 (이상적 비율: 황금비 0.9)
            central_face_area = distance(p_central_eyebrow, p_nose_tip)
            lower_cheek_area = distance(p_nose_tip, p_chin_tip)
            ratio_face_area = lower_cheek_area / central_face_area if lower_cheek_area > 0 else 0
            score6 = calculate_score(ratio_face_area, 0.9, 1.5)

            # 가중치 적용 계산
            # 가중치 순서
            # 1. 입술 - 콧볼
            # 2. 콧볼 - 눈 너비
            # 3. 콧볼 - 왼쪽 눈썹
            # 4. 콧볼 - 오른쪽 눈썹
            # 5. 얼굴 길이
            # 6. 중안부 - 하안부 비율
            weights = [1.0, 1.0, 0.8, 0.8, 1.5, 1.5]
            scores = [score1, score2, score3, score4, score5, score6]
            
            # 최종 평균 점수 계산
            final_score = sum(s*w for s,w in zip(scores, weights)) / sum(weights)
            
            # --- 결과 시각화 ---
            # 얼굴 영역 Bounding Box (MP는 bounding box를 직접 제공하지 않으므로, 랜드마크에서 추출)
            x_min, y_min = np.min(mp_points, axis=0)
            x_max, y_max = np.max(mp_points, axis=0)
            
            # 각 비율 정보 표시
            cv2.putText(frame, f"1. nose/lip: {score1:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"2. nose/eye: {score2:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"3. left eyebrow: {score3:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"4. right eyebrow: {score4:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"5. face shape: {score5:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"6. face ratio: {score6:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            
            # 최종 점수 표시
            cv2.putText(frame, f"Total Score: {final_score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 211, 255), 2)
            
            # 얼굴 윤곽선 및 랜드마크 그리기 (MediaPipe Drawing Utils 사용)
            # Face Mesh는 이미 테셀레이션(삼각형 망)으로 연결된 랜드마크를 제공
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION, # 랜드마크 연결선
                landmark_drawing_spec=None, # 점은 그리지 않음 (아래에서 직접 그릴 것)
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            # 매핑된 주요 랜드마크에 큰 파란 점 찍기 (확인용)
            highlight_points = [
                mp_points_map['nose_left_wing'], mp_points_map['nose_right_wing'],
                mp_points_map['lip_left_corner'], mp_points_map['lip_right_corner'],
                mp_points_map['eye_left_outer'], mp_points_map['eye_right_outer'],
                mp_points_map['eyebrow_left_start'], mp_points_map['eyebrow_left_end'],
                mp_points_map['eyebrow_right_start'], mp_points_map['eyebrow_right_end'],
                mp_points_map['jaw_left_start'], mp_points_map['jaw_right_end'],
                mp_points_map['chin_tip'], mp_points_map['central_eyebrow'], mp_points_map['nose_tip']
            ]
            for idx in highlight_points:
                 cv2.circle(frame, tuple(mp_points[idx]), 3, (255, 0, 0), -1) # 파란색으로 강조
            
            # 모든 랜드마크를 작은 파란 점으로 찍기
            for p in mp_points:
               cv2.circle(frame, tuple(p), 1, (255, 0, 0), -1)

    cv2.imshow('Marquardt Mask Final Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()