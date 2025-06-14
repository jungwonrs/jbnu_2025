import pandas as pd
import random

# 사용할 CSV 파일 목록
csv_files = [
    "train-images-boxable.csv",
    "validation-images.csv",
    "test-images.csv"
]

all_image_ids = set()

# 모든 CSV 파일에서 ImageID 수집
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    if 'ImageID' not in df.columns:
        raise ValueError(f"{csv_file}에 'ImageID' 컬럼이 없습니다.")
    
    ids = df['ImageID'].drop_duplicates()
    all_image_ids.update(ids)

# 리스트로 변환 후 샘플링
all_image_ids = list(all_image_ids)
print(f"전체 이미지 수: {len(all_image_ids)}")

# 무작위로 10,000개 추출
sample_size = 10000
if sample_size > len(all_image_ids):
    raise ValueError("전체 이미지 수가 10,000보다 적습니다.")

sampled_ids = random.sample(all_image_ids, sample_size)

# txt 파일로 저장
with open("sampled_image_ids.txt", "w") as f:
    for image_id in sampled_ids:
        f.write(f"{image_id}\n")

print(f"{sample_size}개의 이미지 ID를 'sampled_image_ids.txt'로 저장했습니다.")
