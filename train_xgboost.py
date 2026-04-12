import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_and_preprocess, apply_smote_and_reshape

def train():
    # 1. Tải dữ liệu (Đường dẫn từ gốc TTCS nên không dùng ../)
    print("--- [XGBoost] Bước 1: Tải và chuẩn bị dữ liệu ---")
    X_train_s, X_test_s, y_train, y_test, le_final, cols = load_and_preprocess('data/Dataset-Unicauca-Version2-87Atts.csv')

    # SMOTE và bóp phẳng dữ liệu
    X_res, y_res, _ = apply_smote_and_reshape(X_train_s, y_train)
    if len(X_res.shape) > 2:
        X_res = X_res.reshape(X_res.shape[0], -1)
        X_test_s = X_test_s.reshape(X_test_s.shape[0], -1)

    print("--- [XGBoost] Bước 2: Huấn luyện ---")
    # Đánh số lại nhãn để XGBoost không lỗi
    le_xg = LabelEncoder()
    y_res_encoded = le_xg.fit_transform(y_res)

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        random_state=42,
        tree_method='hist'
    )

    model.fit(X_res, y_res_encoded)

    # 3. Lưu mô hình (Đường dẫn chuẩn: models/...)
    if not os.path.exists('models'): 
        os.makedirs('models')
        
    model.save_model('models/best_xgboost_model.json')
    joblib.dump(le_final, 'models/label_encoder_xgboost.pkl')
    joblib.dump(le_xg, 'models/label_encoder_xgboost_internal.pkl')
    print("--- ✅ Đã huấn luyện và lưu mô hình thành công! ---")

    print("\n--- [XGBoost] Bước 3: Đánh giá F1-Score  ---")
    y_pred_encoded = model.predict(X_test_s)

    # Đảm bảo y_test cùng kiểu dữ liệu với nhãn đã học
    y_test_cast = y_test.astype(y_res.dtype)

    # Kiểm tra nhãn nào thực sự tồn tại trong le_xg
    mask = np.isin(y_test_cast, le_xg.classes_)
    
    if not np.any(mask):
        print("❌ Lỗi: Không có mẫu dữ liệu nào trong tập Test khớp với nhãn mô hình đã học!")
        print(f"Nhãn trong tập Test: {np.unique(y_test_cast)[:5]}...")
        print(f"Nhãn mô hình đã học: {le_xg.classes_[:5]}...")
        return

    y_test_filtered = y_test_cast[mask]
    y_pred_filtered = y_pred_encoded[mask]

    # Chuyển về hệ 0, 1, 2...
    y_test_encoded = le_xg.transform(y_test_filtered)
    
    # Lấy tên App chuẩn
    actual_ids = le_xg.classes_ 
    target_names = [str(le_final.classes_[int(i)]) for i in actual_ids]

    print(classification_report(
        y_test_encoded, 
        y_pred_filtered, 
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=0
    ))

if __name__ == "__main__":
    train()