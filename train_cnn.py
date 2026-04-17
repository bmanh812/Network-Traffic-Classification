import numpy as np
import tensorflow as tf
from src.data_loader import load_and_preprocess, apply_smote_and_reshape
from src.models import get_cnn_model
from src.utils import evaluate_model
import joblib
from src.models import focal_loss

# 1. Tải và tiền xử lý dữ liệu
print("--- [CNN] Bước 1: Đang tải dữ liệu (300k dòng) ---")
X_train_s, X_test_s, y_train, y_test, le, cols = load_and_preprocess('data/Dataset-Unicauca-Version2-87Atts.csv')

# 2. SMOTE và Reshape về dạng 3D cho CNN
print("--- [CNN] Bước 2: Cân bằng dữ liệu bằng SMOTE ---")
X_res_3d, y_res, le_final = apply_smote_and_reshape(X_train_s, y_train)
X_test_3d = np.expand_dims(X_test_s, axis=2)

# 3. Khởi tạo mô hình CNN từ module src/models.py
input_shape = (X_res_3d.shape[1], 1)
num_classes = len(le_final.classes_)
model = get_cnn_model(input_shape, num_classes)
print(model.summary())

# 4. Thiết lập Callbacks tối ưu (Giống LSTM để so sánh công bằng)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7),
    tf.keras.callbacks.ModelCheckpoint('models/best_cnn_checkpoint.keras', save_best_only=True)
]

# 5. Huấn luyện mô hình
print("--- [CNN] Bước 3: Bắt đầu huấn luyện ---")

print("--- [CNN] Cấu hình Focal Loss để xử lý dữ liệu mất cân bằng ---")
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

# Chuyển đổi nhãn sang One-hot encoding để khớp với Focal Loss
y_res_onehot = tf.keras.utils.to_categorical(y_res, num_classes=num_classes)

# Tương tự với tập Validation
y_test_filtered = [label if label in le_final.classes_ else le_final.classes_[0] for label in y_test]
y_val_onehot = tf.keras.utils.to_categorical(le_final.transform(y_test_filtered), num_classes=num_classes)

history = model.fit(
    X_res_3d, y_res_onehot,             
    epochs=100,
    batch_size=128,
    validation_data=(X_test_3d, y_val_onehot), 
    callbacks=callbacks,
    verbose=1
)

# 6. Lưu mô hình và bộ mã hóa nhãn
model.save('models/best_cnn_model.h5')
joblib.dump(le_final, 'models/label_encoder_cnn.pkl')
print("--- [CNN] Đã lưu mô hình và LabelEncoder thành công! ---")

# 7. Đánh giá chi tiết (Sẽ gọi từ utils.py)
print("--- [CNN] Bước 4: Đánh giá mô hình trên tập Test ---")
y_pred = model.predict(X_test_3d)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ở cuối file train_cnn.py hoặc train_lstm.py
from src.utils import evaluate_model, plot_confusion_matrix, plot_training_history


# 1. Vẽ biểu đồ học tập ngay sau khi fit xong
plot_training_history(history)

# 2. Dự đoán và đánh giá chi tiết theo ý bạn em
y_pred = model.predict(X_test_3d)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_fixed = le_final.transform(y_test)

# In báo cáo F1-Score
evaluate_model(y_test_fixed, y_pred_classes, target_names=[str(c) for c in le_final.classes_])

# Vẽ ma trận nhầm lẫn để Hiếu dán vào Word
plot_confusion_matrix(y_test_fixed, y_pred_classes, target_names=[str(c) for c in le_final.classes_])