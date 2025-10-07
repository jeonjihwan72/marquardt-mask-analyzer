# Marquardt-mask-analyzer

자신의 얼굴이 얼마나 황금 비율에 가까운지 확인해보세요.

## 실행 방법

1. 작업 폴더로 이동하기
``` powershell
cd marqurdt-mask-analyzer
```

2. 가상 환경 생성하기
``` bash
python -m venv venv
```

3. 가상 환경 진입하기
``` bash
venv\Scripts\activate
```

4. 두 가지 라이브러리 설치하기 
    - 가상환경에서 실행하지 않을 경우 라이브러리 간의 충돌이 발생할 수 있습니다.
``` bash
pip install opencv-python mediapipe
```

5. 실행하기
``` bash
python main.py
```

