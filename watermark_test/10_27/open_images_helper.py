import pandas as pd
import random

# 사용할 CSV 파일 목록과 그에 대응하는 split 이름
csv_files = {
    "train": "train-images-boxable.csv",
    "validation": "validation-images.csv",
    "test": "test-images.csv"
}

# 샘플 총 개수와 split별 비율
sample_size = 10000
split_ratios = {
    "train": 0.7,
    "validation": 0.2,
    "test": 0.1
}

sampled_pairs = []

for split, csv_file in csv_files.items():
    df = pd.read_csv(csv_file)

    if 'ImageID' not in df.columns:
        raise ValueError(f"{csv_file}에 'ImageID' 컬럼이 없습니다.")

    unique_ids = df['ImageID'].drop_duplicates().tolist()
    split_sample_size = int(sample_size * split_ratios[split])

    if split_sample_size > len(unique_ids):
        raise ValueError(f"{split} split에서 충분한 샘플을 뽑을 수 없습니다.")

    sampled_ids = random.sample(unique_ids, split_sample_size)
    sampled_pairs.extend((split, image_id) for image_id in sampled_ids)

# 무작위 순서로 섞기 (선택사항)
random.shuffle(sampled_pairs)

# txt 파일로 저장
with open("sampled_image_ids.txt", "w") as f:
    for split, image_id in sampled_pairs:
        f.write(f"{split}/{image_id}\n")

print(f"{sample_size}개의 split/image ID를 'sampled_image_ids.txt'로 저장했습니다.")
