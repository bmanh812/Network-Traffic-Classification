import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from src.data_loader import load_and_preprocess, apply_smote_and_reshape

def train():
    # 1. Tải dữ liệu
    print("--- [XGBoost] Bước 1: Tải và chuẩn bị dữ liệu ---")
    X_train_s, X_test_s, y_train, y_test, le_final, cols = load_and_preprocess('data/Dataset-Unicauca-Version2-87Atts.csv')

    # SMOTE và bóp phẳng dữ liệu (XGBoost nhận đầu vào 2D)
    X_res, y_res, _ = apply_smote_and_reshape(X_train_s, y_train)
    if len(X_res.shape) > 2:
        X_res = X_res.reshape(X_res.shape[0], -1)
        X_test_s = X_test_s.reshape(X_test_s.shape[0], -1)

    print("--- [XGBoost] Bước 2: Huấn luyện với Trọng số lớp tối ưu ---")
    le_xg = LabelEncoder()
    y_res_encoded = le_xg.fit_transform(y_res)

    # 1. Tinh chỉnh trọng số mẫu (Dùng căn bậc 4 thay vì căn bậc 2)
    # Căn bậc 4 giúp kìm hãm trọng số các nhãn đa số ít hơn nữa, giữ Accuracy cao hơn
    classes = np.unique(y_res_encoded)
    raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_res_encoded)
    adj_weights = np.power(raw_weights, 0.25) 
    class_weights_dict = dict(zip(classes, adj_weights))
    sample_weights = np.array([class_weights_dict[i] for i in y_res_encoded])

    # 2. Cấu hình mô hình "Chuyên gia"
    model = XGBClassifier(
        n_estimators=500,            # Tăng mạnh số cây để mô hình học sâu hơn
        learning_rate=0.02,          # Giảm tốc độ học để cực kỳ ổn định
        max_depth=12,                # Tăng độ sâu để tách biệt các nhãn gần giống nhau
        min_child_weight=1,          # Cho phép các lá nhỏ hơn tồn tại để cứu nhãn hiếm
        gamma=0.2,                   # Tăng tính tổng quát hóa
        subsample=0.8,               # Lấy mẫu ngẫu nhiên dòng để tránh overfitting
        colsample_bytree=0.7,        # Chỉ lấy 70% đặc trưng cho mỗi cây để giảm nhiễu
        objective='multi:softprob',
        tree_method='hist',
        max_delta_step=5,            # Tăng lên 5 để dồn lực mạnh cho các nhãn khó
        reg_alpha=0.1,               # L1 Regularization: Loại bỏ các đặc trưng thừa
        reg_lambda=1.5,              # L2 Regularization: Làm mượt trọng số
        random_state=42
    )

    # 3. Huấn luyện
    model.fit(X_res, y_res_encoded, sample_weight=sample_weights)

    # 3. Lưu mô hình và các bộ mã hóa
    if not os.path.exists('models'): 
        os.makedirs('models')
        
    model.save_model('models/best_xgboost_model.json')
    joblib.dump(le_final, 'models/label_encoder_xgboost.pkl')
    joblib.dump(le_xg, 'models/label_encoder_xgboost_internal.pkl')
    print("--- ✅ Đã huấn luyện và lưu mô hình thành công! ---")

    # 4. Đánh giá F1-Score
    print("\n--- [XGBoost] Bước 3: Đánh giá chi tiết ---")
    y_pred_encoded = model.predict(X_test_s)

    # Ép kiểu dữ liệu nhãn test để so khớp
    y_test_cast = y_test.astype(y_res.dtype)
    mask = np.isin(y_test_cast, le_xg.classes_)
    
    if not np.any(mask):
        print("❌ Lỗi: Không có mẫu dữ liệu nào trong tập Test khớp với nhãn!")
        return

    y_test_filtered = y_test_cast[mask]
    y_pred_filtered = y_pred_encoded[mask]
    y_test_encoded = le_xg.transform(y_test_filtered)
    
    # Lấy tên App chuẩn để hiển thị báo cáo
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