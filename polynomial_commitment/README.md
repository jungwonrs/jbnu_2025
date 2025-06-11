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
   
# 🧾 실행 로그

**2024-06-11 16시, Ubuntu WSL 환경**

```bash
(venv) seo@seo-desktop:~/projects/polyonmial_pdp$ python pl_pe.py
[DEBUG] verify_standard: lhs=[2606392055563908508724991085594339296726213703398258028867853511797910171860529164431946136502607592444162747853218157503725850620018138511821204406576099, 5334870022447575772760970000771445582737102600690708940398393315196759116110417391826399998776423478520575529045080205197244291884083512417538721862628141], rhs=[2606392055563908508724991085594339296726213703398258028867853511797910171860529164431946136502607592444162747853218157503725850620018138511821204406576099, 5334870022447575772760970000771445582737102600690708940398393315196759116110417391826399998776423478520575529045080205197244291884083512417538721862628141], ok=True
Standard verify → ✔
Trapdoor verify → ✔
991 991
991 991
poly_eval(F, 3, 13) = 9, syn_div rem = 9
```

> ✅ log 완성  
> ⚠️ **GPT 기반 코드작성이라 오류 가능성 있음**