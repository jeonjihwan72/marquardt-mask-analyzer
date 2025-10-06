import cv2

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

    # "Webcam Feed" 라는 이름의 창에 현재 프레임을 보여줌
    cv2.imshow('Webcam Feed', frame)

    # 'q' 키를 누르면 루프에서 빠져나옴
    # 1ms 동안 키 입력을 기다림
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업이 끝나면 모든 자원을 해제
print("카메라를 닫고 모든 창을 종료합니다.")
cap.release()
cv2.destroyAllWindows()