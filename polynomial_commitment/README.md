# 📌 초기 셋팅

**Python version: 3.10**

1. Visual Studio 설치  
2. WSL 설치 → Ubuntu 선택  
   - Ubuntu 버전: **24.04.2 LTS**
3. Visual Studio → **View > Terminal**
4. Terminal에서 `+` 버튼 클릭 → **WSL 터미널** 호출
5. Terminal에서 아래 명령어 입력 → Ubuntu 환경의 VS Code 실행
   ```bash
   code .
   ```
6. 코드가 들어있는 폴더를 다음 위치로 이동  
   ```
   ~/Project/
   ```
7. 패키지 목록 업데이트
   ```bash
   sudo apt update
   ```
8. Python 3.10 및 필수 빌드 도구 설치
   ```bash
   sudo apt install -y \
     python3.10 \
     python3.10-venv \
     python3.10-distutils \
     python3.10-dev \
     python3-pip \
     build-essential
   ```
9. 가상환경 생성
   ```bash
   python3.10 -m venv venv
   ```
10. 가상환경 활성화
    ```bash
    source venv/bin/activate
    ```
11. pip 및 필수 툴 업데이트
    ```bash
    pip install --upgrade pip setuptools wheel
    ```
12. 프로젝트 패키지 설치
    ```bash
    pip install -r requirements.txt
    ```

---

# ▶️ 파일 실행

터미널에서 아래 명령 중 하나 실행:
```bash
python pl_dl.py
```
또는
```bash
python pl_pe.py
```

---

# ✅ 할 일

1. 논문 분석  
2. 코드 분석  
3. `python pl_pe.py` 확장  
   - 파일 수정, 삭제 관련 기능 아이디어 구상