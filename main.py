import cv2
import dlib
import numpy as np

# Dlib의 기본 얼굴 탐지기를 호출
detector = dlib.get_frontal_face_detector()

# Dlib의 랜드마크 예측기 호출
predictor = dlib.shape_predictor("./landmark_model/shape_predictor_68_face_landmarks.dat")

# 0번 카메라(기본 웹캠)에 연결
cap = cv2.VideoCapture(0)

# 웹캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()

# 비디오 프레임을 계속해서 읽어오기
while True:
    # 카메라에서 현재 프레임(이미지)을 읽어옴
    # ret: 성공 여부 (True/False), frame: 읽어온 이미지
    ret, frame = cap.read()

    # 프레임을 성공적으로 읽지 못했다면 루프 종료
    if not ret:
        print("오류: 프레임을 읽을 수 없습니다.")
        break
    
    # 성능 향상을 위해 프레임을 흑백으로 변환
    # 얼굴 탐지는 색상 정보가 필요없음
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 흑백 이미지에서 얼굴을 탐지
    faces = detector(gray)
    
    # 찾은 얼굴들의 좌표를 반복하면서 사각형 그리기
    for face in faces:
        x1, y1 = face.left(), face.top()    # 왼쪽 위 좌표
        x2, y2 = face.right(), face.bottom()    # 오른쪽 아래 좌표
        
        # 원본 커러 프레임에 빨간 사각형 그리기
        # (frame, 시작점, 끝점, 색상, 두께)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 얼굴 랜드마크 찾기
        landmarks = predictor(gray, face)
        
        jaw_line_idx = [(i, i+1) for i in range(0, 16)] # 턱선
        
        left_eyebrow_idx = [(i, i+1) for i in range(17, 21)]    # 왼쪽 눈썹
        left_eye_idx = [(i, i+1) for i in range(36, 41)] + [(36, 41)]   # 왼쪽 눈
        
        right_eyebrow_idx = [(i, i+1) for i in range(22,26)] # 오른쪽 눈썹
        right_eye_idx = [(i, i+1) for i in range(42, 47)] + [(42, 47)]  # 오른쪽 눈
        
        notch_idx = [(27,28), (28,29), (29,30)] # 콧대
        nostrill_idx = [(31,32), (32,33), (33,34), (34,35)] # 콧볼, 콧망울
        
        upper_lip_idx = [(48,49), (49,50), (50,51), (51,52), (52,53), (53,54), 
                         (54,64), (64,63), (63,62), (62,61), (61,60), (60,48)]  # 윗입술
        lower_lip_idx = [(48,60), (60,67), (67,66), (66,65), (65,64), (64,54),
                         (54,55), (55,56), (56,57), (57,58), (58,59), (59,48)]  # 아랫입술
               
        # 68개의 랜드마크에 점 찍기
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # 랜드마크 위치에 파란 점 그리기
            # cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # 턱선 그리기
        for idx_tup in jaw_line_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)

        # 왼쪽 눈썹 그리기
        for idx_tup in left_eyebrow_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
        
        # 왼쪽 눈 그리기
        for idx_tup in left_eye_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
        
        # 오른쪽 눈썹 그리기
        for idx_tup in right_eyebrow_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
        
        # 오른쪽 눈 그리기
        for idx_tup in right_eye_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
    
        # 콧대 그리기
        for idx_tup in notch_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
            
        # 콧망울 그리기
        for idx_tup in nostrill_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
            
        # 윗입술 그리기
        for idx_tup in upper_lip_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)
            
        # 아랫입술 그리기
        for idx_tup in lower_lip_idx:
            first_x = landmarks.part(idx_tup[0]).x
            first_y = landmarks.part(idx_tup[0]).y
            
            second_x = landmarks.part(idx_tup[1]).x
            second_y = landmarks.part(idx_tup[1]).y
            
            cv2.line(frame, (first_x, first_y), (second_x, second_y), (0, 211, 255), 1)

    # "Webcam Feed" 라는 이름의 창에 현재 프레임을 보여줌
    cv2.imshow('Facial Landmarks', frame)

    # 'q' 키를 누르면 루프에서 빠져나옴
    # 1ms 동안 키 입력을 기다림
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업이 끝나면 모든 자원을 해제
print("카메라를 닫고 모든 창을 종료합니다.")
cap.release()
cv2.destroyAllWindows()