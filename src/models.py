from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM

def get_cnn_model(input_shape, num_classes):
    # Kiến trúc CNN
    model = Sequential([
        Conv1D(128, 7, dilation_rate=2, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_lstm_model(input_shape, num_classes):
    # Lấy kiến trúc LSTM
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#Focal-losspython train_cnn.py
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    """
    gamma: Độ tập trung vào mẫu khó (càng cao càng soi kỹ mẫu khó).
    alpha: Trọng số cân bằng giữa các lớp.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Tránh lỗi log(0) dẫn đến NaN
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Tính Cross Entropy cơ bản
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Áp dụng công thức Focal: (1 - p)^gamma
        # Nó sẽ làm giảm loss của các mẫu dễ (p cao) và giữ nguyên loss của mẫu khó (p thấp)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed