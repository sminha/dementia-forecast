# scaler.py
# lifelog and lifestyle data preprocessing and scaling

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from config import *

lifestyle_df = pd.read_csv(tab_data_path)

lifestyle_feat_df = lifestyle_df.drop(columns=['ID', '치매여부_치매1기타0', '가구돌봄유형'], errors='ignore')
lifestyle_feat_df = lifestyle_feat_df.astype(np.float32)

lifestyle_scaler = StandardScaler()
lifestyle_scaler.fit(lifestyle_feat_df)

joblib.dump(lifestyle_scaler, 'lifestyle_scaler.pkl')

lifelog_df = pd.read_csv(ts_data_path)

lifelog_feat_df = lifelog_df.drop(columns=['ID', 'DIAG_NM', 'EMAIL', 'Day'], errors='ignore')
lifelog_feat_df = lifelog_feat_df.select_dtypes(include=[np.number])

lifelog_scaler = StandardScaler()
lifelog_scaler.fit(lifelog_feat_df)

joblib.dump(lifelog_scaler, 'lifelog_scaler.pkl')
