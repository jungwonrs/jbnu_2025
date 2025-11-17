import pandas as pd
import os
import re
from datetime import datetime

RESULTS_DIR = "./results"
FINAL_EXCEL_PATH = os.path.join(RESULTS_DIR, f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

_all_results = []

def add_result(row_data):
    _all_results.append(row_data)

def clean_text_for_excel(text):
    """엑셀 저장용 텍스트 청소부"""
    if not isinstance(text, str):
        return text
    
    # 1. [강력 수정] Fuyu가 뱉는 쓰레기 문자열 삭제
    garbage_tokens = ["|SPEAKER|", "|NEWLINE|", "<s>", "</s>", "<0x0A>"]
    for g in garbage_tokens:
        text = text.replace(g, " ")
    
    # 2. 엑셀 제어 문자 제거
    text = re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', text)
    
    # 3. 중복 공백 제거
    text = " ".join(text.split())
    
    if len(text) > 30000:
        text = text[:30000] + "..."
        
    return text

def save_to_excel():
    if not _all_results:
        print("!!! 저장할 데이터가 없습니다.")
        return

    df = pd.DataFrame(_all_results)

    print("엑셀 저장 전 데이터 클리닝 중...")
    # 전체 데이터프레임에 청소 적용
    df = df.applymap(clean_text_for_excel)

    try:
        df.to_excel(FINAL_EXCEL_PATH, index=False)
        print("="*40)
        print(f"✅ 저장 완료! (특수문자 제거됨)")
        print(f"📂 {os.path.abspath(FINAL_EXCEL_PATH)}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ 엑셀 저장 에러: {e}")
        csv_path = FINAL_EXCEL_PATH.replace(".xlsx", ".csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 대신 CSV로 저장됨: {csv_path}")