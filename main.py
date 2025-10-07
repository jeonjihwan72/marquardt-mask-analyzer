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
        
        # 68개의 랜드마크에 점 찍기
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # 랜드마크 위치에 파란 점 그리기
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

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