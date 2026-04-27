import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_and_preprocess(file_path, sample_size=300000):
    # 1. Load và Sample dữ liệu
    df = pd.read_csv(file_path)
    df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    # 2. Mã hóa nhãn
    le = LabelEncoder()
    df['ProtocolName_Encoded'] = le.fit_transform(df['ProtocolName'])
    
    # 3. Lọc đặc trưng số và xử lý Inf/NaN
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in features if f not in ['ProtocolName_Encoded', 'L7Protocol']]
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['ProtocolName_Encoded']
    
    # 4. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, X_train.columns

def apply_smote_and_reshape(X_train_scaled, y_train, target_threshold=2000):
    # Tái mã hóa nhãn để đảm bảo tính liên tục 
    le_final = LabelEncoder()
    y_train_fixed = le_final.fit_transform(y_train)
    
    # Lọc class ít mẫu và chạy SMOTE 
    counts = Counter(y_train_fixed)
    valid_classes = [cls for cls, count in counts.items() if count >= 6]
    mask = np.isin(y_train_fixed, valid_classes)
    X_f, y_f = X_train_scaled[mask], y_train_fixed[mask]
    
    sampling_strategy = {label: max(count, target_threshold) for label, count in Counter(y_f).items()}
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=3, random_state=42)
    X_res, y_res = smote.fit_resample(X_f, y_f)
    
    # Reshape 3D cho CNN/LSTM 
    X_res_3d = np.expand_dims(X_res, axis=2)
    return X_res_3d, y_res, le_final