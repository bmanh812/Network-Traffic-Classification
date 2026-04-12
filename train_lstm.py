import numpy as np
import tensorflow as tf
import joblib # Thêm cái này để lưu le_final
from src.data_loader import load_and_preprocess, apply_smote_and_reshape
from src.models import get_lstm_model
from src.utils import evaluate_model, plot_confusion_matrix, plot_training_history

def train(): # <--- PHẢI CÓ HÀM NÀY
    # 1. Tải dữ liệu
    print("--- Đang chuẩn bị dữ liệu cho LSTM ---")
    X_train_s, X_test_s, y_train, y_test, le, cols = load_and_preprocess('data/Dataset-Unicauca-Version2-87Atts.csv')

    # 2. SMOTE và Reshape
    X_res_3d, y_res, le_final = apply_smote_and_reshape(X_train_s, y_train)
    X_test_3d = np.expand_dims(X_test_s, axis=2)

    # 3. Khởi tạo mô hình
    input_shape = (X_res_3d.shape[1], 1)
    num_classes = len(le_final.classes_)
    model = get_lstm_model(input_shape, num_classes)

    # 4. Callbacks (Patience 15 theo ý Mạnh là rất ổn)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
    ]

    # 5. Huấn luyện mô hình
    print("--- Bắt đầu huấn luyện LSTM ---")

    # Tạo "mặt nạ" để lọc những nhãn mà le_final có biết
    valid_test_mask = np.isin(y_test, le_final.classes_)

    # Áp dụng mặt nạ để lọc dữ liệu Test
    X_test_filtered = X_test_3d[valid_test_mask]
    y_test_filtered = y_test[valid_test_mask] # Đổi thành valid_test_mask cho đồng bộ

    # Bây giờ mới nạp vào fit
    history = model.fit(
        X_res_3d, y_res,
        epochs=100,
        batch_size=64,
        validation_data=(X_test_filtered, le_final.transform(y_test_filtered)),
        callbacks=callbacks
    )

    # 6. Lưu mô hình và Label Encoder (Rất quan trọng để sau này so sánh)
    model.save('models/best_lstm_model.h5')
    joblib.dump(le_final, 'models/label_encoder_lstm.pkl') # Lưu riêng để Hiếu so sánh
    print("--- Đã lưu mô hình LSTM thành công! ---")

    # 7. Đánh giá và vẽ biểu đồ
    plot_training_history(history)
    y_pred = model.predict(X_test_3d)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_fixed = le_final.transform(y_test)

    evaluate_model(y_test_fixed, y_pred_classes, target_names=[str(c) for c in le_final.classes_])
    plot_confusion_matrix(y_test_fixed, y_pred_classes, target_names=[str(c) for c in le_final.classes_])

if __name__ == "__main__":
    train()