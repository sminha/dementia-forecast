import pandas as pd
import numpy as np
import json

# CSV 경로 설정
lifelog_path = 'lifelog_test.csv'
lifestyle_path = 'lifestyle_test.csv'

# CSV 불러오기
lifelog_df = pd.read_csv(lifelog_path)
lifestyle_df = pd.read_csv(lifestyle_path)

# 첫 번째 샘플만 사용
day1_values = lifelog_df.iloc[0].apply(lambda x: int(x) if isinstance(x, np.int64) else x).tolist()
day2_values = lifelog_df.iloc[1].apply(lambda x: int(x) if isinstance(x, np.int64) else x).tolist() if len(lifelog_df) > 1 else None
day3_values = lifelog_df.iloc[2].apply(lambda x: int(x) if isinstance(x, np.int64) else x).tolist() if len(lifelog_df) > 2 else None

# 컬럼 이름을 가져오기
lifelog_columns = lifelog_df.columns.tolist()
lifestyle_columns = lifestyle_df.columns.tolist()

# lifestyle 데이터
lifestyle_values = lifestyle_df.iloc[0].apply(lambda x: int(x) if isinstance(x, np.int64) else x).to_dict()

# JSON 구조 구성
example_request = {
    "lifelog": {
        "columns": lifelog_columns,  # 컬럼 이름 추가
        "day1": day1_values,
        "day2": day2_values,
        "day3": day3_values
    },
    "lifestyle": {
        "columns": lifestyle_columns,  # 컬럼 이름 추가
        "data": lifestyle_values
    }
}

# 파일로 저장
with open("example_request.json", "w", encoding="utf-8") as f:
    json.dump(example_request, f, ensure_ascii=False, indent=4)

print("✅ example_request.json 파일이 생성되었습니다.")
