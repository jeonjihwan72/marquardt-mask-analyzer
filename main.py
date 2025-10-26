import sys
import cv2
import mediapipe as mp
import numpy as np
import math

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QGraphicsDropShadowEffect  # 그림자 효과를 위해 임포트
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor


# -----------------------------------------------------------------
# 1. MediaPipe 처리 및 점수 계산 스레드 (이전과 동일)
# -----------------------------------------------------------------
# (이전 코드의 FaceMeshWorker 클래스 전체를 여기에 붙여넣으세요)
# ... (FaceMeshWorker 클래스 시작) ...
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def face_height_distance(left_p1, right_p1, left_p2, right_p2):
    return abs(((left_p1[1] + right_p1[1])/2) - ((left_p2[1] + right_p2[1])/2))
def calculate_score(user_ratio, ideal_ratio, multiplier=1.5):
    percent_diff = abs(user_ratio - ideal_ratio) / ideal_ratio
    penalty = percent_diff * 100 * multiplier
    score = 100 - penalty
    return max(0, score)
mp_points_map = {
    'nose_left_wing': 64, 'nose_right_wing': 294, 'nose_tip' : 19,
    'lip_left_corner': 61, 'lip_right_corner': 291,
    'eye_left_outer': 130, 'eye_right_outer': 359,
    'eyebrow_left_start': 55, 'eyebrow_left_end': 70,
    'eyebrow_right_start': 285, 'eyebrow_right_end': 300,
    'jaw_left_start': 127, 'jaw_right_end': 356, 'chin_tip': 152,
    'central_eyebrow' : 9
}

class FaceMeshWorker(QThread):
    frame_ready = Signal(np.ndarray)
    scores_ready = Signal(list, float)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False,
                                          max_num_faces=1,
                                          min_detection_confidence = 0.5,
                                          min_tracking_confidence = 0.5)
        cap = cv2.VideoCapture(0)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            scores = [0.0] * 6
            final_score = 0.0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_points = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                                          for lm in face_landmarks.landmark], dtype=np.int32)
                    p_nose_left_wing = mp_points[mp_points_map['nose_left_wing']]
                    p_nose_right_wing = mp_points[mp_points_map['nose_right_wing']]
                    p_lip_left = mp_points[mp_points_map['lip_left_corner']]
                    p_lip_right = mp_points[mp_points_map['lip_right_corner']]
                    p_eye_left_outer = mp_points[mp_points_map['eye_left_outer']]
                    p_eye_right_outer = mp_points[mp_points_map['eye_right_outer']]
                    p_eyebrow_left_start = mp_points[mp_points_map['eyebrow_left_start']]
                    p_eyebrow_left_end = mp_points[mp_points_map['eyebrow_left_end']]
                    p_eyebrow_right_start = mp_points[mp_points_map['eyebrow_right_start']]
                    p_eyebrow_right_end = mp_points[mp_points_map['eyebrow_right_end']]
                    p_jaw_left_start = mp_points[mp_points_map['jaw_left_start']]
                    p_jaw_right_end = mp_points[mp_points_map['jaw_right_end']]
                    p_chin_tip = mp_points[mp_points_map['chin_tip']]
                    p_central_eyebrow = mp_points[mp_points_map['central_eyebrow']]
                    p_nose_tip = mp_points[mp_points_map['nose_tip']]

                    nose_width = distance(p_nose_left_wing, p_nose_right_wing)
                    lip_width = distance(p_lip_left, p_lip_right)
                    ratio_nose_lip = lip_width / nose_width if nose_width > 0 else 0
                    score1 = calculate_score(ratio_nose_lip, 1.618, 1.0)
                    eye_outer_width = distance(p_eye_left_outer, p_eye_right_outer)
                    ratio_nose_eye = eye_outer_width / nose_width if nose_width > 0 else 0
                    score2 = calculate_score(ratio_nose_eye, 1.618 ** 2, 2.0)
                    left_eyebrow_width = distance(p_eyebrow_left_start, p_eyebrow_left_end)
                    ratio_eyebrow_left = left_eyebrow_width / nose_width if nose_width > 0 else 0
                    score3 = calculate_score(ratio_eyebrow_left, 1.618, 1.5)
                    right_eyebrow_width = distance(p_eyebrow_right_start, p_eyebrow_right_end)
                    ratio_eyebrow_right = right_eyebrow_width / nose_width if nose_width > 0 else 0
                    score4 = calculate_score(ratio_eyebrow_right, 1.618, 1.5)
                    upper_face_height = face_height_distance(p_jaw_left_start, p_jaw_right_end, p_lip_left, p_lip_right)
                    lower_face_height = face_height_distance(p_lip_left, p_lip_right, p_chin_tip, p_chin_tip)
                    ratio_face_height = upper_face_height / lower_face_height if lower_face_height > 0 else 0
                    score5 = calculate_score(ratio_face_height, 1.618, 2.0)
                    central_face_area = distance(p_central_eyebrow, p_nose_tip)
                    lower_cheek_area = distance(p_nose_tip, p_chin_tip)
                    ratio_face_area = lower_cheek_area / central_face_area if lower_cheek_area > 0 else 0
                    score6 = calculate_score(ratio_face_area, 0.9, 1.5)

                    weights = [1.0, 1.0, 0.8, 0.8, 1.5, 1.5]
                    scores = [score1, score2, score3, score4, score5, score6]
                    final_score = sum(s*w for s,w in zip(scores, weights)) / sum(weights)
                    
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                    highlight_points = [
                        mp_points_map['nose_left_wing'], mp_points_map['nose_right_wing'],
                        mp_points_map['lip_left_corner'], mp_points_map['lip_right_corner'],
                        mp_points_map['eye_left_outer'], mp_points_map['eye_right_outer'],
                        mp_points_map['eyebrow_left_start'], mp_points_map['eyebrow_left_end'],
                        mp_points_map['eyebrow_right_start'], mp_points_map['eyebrow_right_end'],
                        mp_points_map['chin_tip'], mp_points_map['central_eyebrow'], mp_points_map['nose_tip']
                    ]
                    for idx in highlight_points:
                         cv2.circle(frame, tuple(mp_points[idx]), 3, (255, 0, 0), -1)

            self.frame_ready.emit(frame)
            self.scores_ready.emit(scores, final_score)

        cap.release()
        print("Thread-Worker finished.")

    def stop(self):
        self.running = False
        self.wait()
# ... (FaceMeshWorker 클래스 끝) ...


# -----------------------------------------------------------------
# 2. PySide6 메인 윈도우 (디자인 수정됨)
# -----------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marquardt Mask Analyzer (Dark Theme)")
        self.setGeometry(100, 100, 1280, 720) # 16:9 비율로 크기 조정

        # --- 스타일시트 정의 ---
        self.BG_COLOR = "#2C3E50"      # 짙은 네이비 (배경)
        self.PANEL_COLOR = "#34495E"   # 한 단계 밝은 회색 (패널)
        self.TEXT_COLOR = "#ECF0F1"    # 밝은 회색 (기본 텍스트)
        self.MUTED_COLOR = "#BDC3C7"   # 중간 회색 (보조 텍스트)
        self.ACCENT_COLOR_1 = "#F1C40F" # 노란색 (강조 1)
        self.ACCENT_COLOR_2 = "#E67E22" # 주황색 (강조 2)
        
        # --- 폰트 정의 ---
        self.font_title = QFont("Arial", 22, QFont.Weight.Bold)
        self.font_heading = QFont("Arial", 16, QFont.Weight.Bold)
        self.font_body = QFont("Arial", 14)
        self.font_score = QFont("Arial", 28, QFont.Weight.Bold)

        # --- 전체 레이아웃 (짙은 네이비 배경) ---
        main_widget = QWidget()
        main_widget.setStyleSheet(f"background-color: {self.BG_COLOR}; padding: 25px;")
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(25) # 위젯 사이 간격
        self.setCentralWidget(main_widget)

        # --- 1. 왼쪽: 웹캠 패널 ---
        self.webcam_label = QLabel("웹 캠 로딩 중...")
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setStyleSheet(f"""
            background-color: {self.PANEL_COLOR}; 
            color: {self.TEXT_COLOR}; 
            font-weight: bold;
            font-size: 20px;
            border-radius: 15px;
        """)
        self.webcam_label.setFixedSize(800, 600) # 4:3 비율 (800x600)
        
        # 웹캠 패널 그림자 효과
        shadow_cam = QGraphicsDropShadowEffect()
        shadow_cam.setBlurRadius(25)
        shadow_cam.setColor(QColor(0, 0, 0, 100)) # 옅은 검은색 그림자
        shadow_cam.setOffset(5, 5)
        self.webcam_label.setGraphicsEffect(shadow_cam)
        
        main_layout.addWidget(self.webcam_label, 2) # 비율 2

        # --- 2. 오른쪽: 정보 출력 패널 ---
        info_widget = QWidget()
        info_widget.setStyleSheet(f"""
            background-color: {self.PANEL_COLOR}; 
            border-radius: 15px;
        """)
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(25, 25, 25, 25)
        info_layout.setSpacing(15)
        
        # 정보 타이틀
        title_label = QLabel("실시간 분석 정보")
        title_label.setFont(self.font_title)
        title_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(title_label)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet(f"background-color: {self.MUTED_COLOR};")
        info_layout.addWidget(line)

        # 점수 라벨들
        self.score_labels = []
        score_names = [
            "1. 입술 / 콧볼 너비",
            "2. 눈 / 콧볼 너비",
            "3. 좌측 눈썹 / 콧볼",
            "4. 우측 눈썹 / 콧볼",
            "5. 얼굴 상하 길이비",
            "6. 중/하안부 길이비"
        ]
        
        for name in score_names:
            # 각 항목을 위한 수평 레이아웃 (이름 + 점수)
            hbox = QHBoxLayout()
            
            name_label = QLabel(name)
            name_label.setFont(self.font_body)
            name_label.setStyleSheet(f"color: {self.MUTED_COLOR};")
            
            score_label = QLabel("- 점")
            score_label.setFont(self.font_body)
            score_label.setStyleSheet(f"color: {self.TEXT_COLOR}; font-weight: bold;")
            score_label.setAlignment(Qt.AlignmentFlag.AlignRight) # 점수 우측 정렬
            
            hbox.addWidget(name_label, 3) # 이름이 3의 비율
            hbox.addWidget(score_label, 1) # 점수가 1의 비율
            
            self.score_labels.append(score_label) # 점수 라벨만 리스트에 저장
            info_layout.addLayout(hbox)

        info_layout.addStretch(1) # 라벨들을 위로 밀기

        # 최종 점수 라벨
        final_score_title = QLabel("Total Score")
        final_score_title.setFont(self.font_heading)
        final_score_title.setStyleSheet(f"color: {self.ACCENT_COLOR_2};") # 주황색
        final_score_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.final_score_label = QLabel("-")
        self.final_score_label.setFont(self.font_score)
        self.final_score_label.setStyleSheet(f"color: {self.ACCENT_COLOR_1};") # 노란색
        self.final_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info_layout.addWidget(final_score_title)
        info_layout.addWidget(self.final_score_label)
        
        # 정보 패널 그림자 효과
        shadow_info = QGraphicsDropShadowEffect()
        shadow_info.setBlurRadius(25)
        shadow_info.setColor(QColor(0, 0, 0, 100))
        shadow_info.setOffset(5, 5)
        info_widget.setGraphicsEffect(shadow_info)
        
        main_layout.addWidget(info_widget, 1) # 비율 1

        # --- 스레드 시작 ---
        self.worker = FaceMeshWorker()
        self.worker.frame_ready.connect(self.update_webcam_feed)
        self.worker.scores_ready.connect(self.update_scores)
        self.worker.start()

    @Slot(np.ndarray)
    def update_webcam_feed(self, frame):
        """웹캠 QLabel을 OpenCV 프레임으로 업데이트"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.webcam_label.setPixmap(pixmap.scaled(
                self.webcam_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
        except Exception as e:
            print(f"Error updating webcam feed: {e}")

    @Slot(list, float)
    def update_scores(self, scores, final_score):
        """점수 라벨들 업데이트"""
        try:
            for i, label in enumerate(self.score_labels):
                label.setText(f"{scores[i]:.1f} 점")
            
            self.final_score_label.setText(f"{final_score:.2f}")
        except Exception as e:
            print(f"Error updating scores: {e}")

    def closeEvent(self, event):
        """윈도우가 닫힐 때 스레드 종료"""
        print("Closing window...")
        self.worker.stop()
        event.accept()


# -----------------------------------------------------------------
# 3. 애플리케이션 실행
# -----------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())