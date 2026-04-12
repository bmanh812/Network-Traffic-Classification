import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(y_true, y_pred, target_names):
    """
    In báo cáo chi tiết và tính toán các độ đo F1, Precision, Recall.
    """
    print("\n--- BÁO CÁO CHI TIẾT (CLASSIFICATION REPORT) ---")
    # Sử dụng classification_report để lấy F1-Score như bạn Mạnh gợi ý
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report

def plot_confusion_matrix(y_true, y_pred, target_names, title='Confusion Matrix'):
    """
    Vẽ ma trận nhầm lẫn để soi xem mô hình đang nhầm App nào với App nào.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 15))
    # Vẽ heatmap chuyên nghiệp cho báo cáo Tuần 5
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.show()

def plot_training_history(history):
    """
    Vẽ biểu đồ Loss và Accuracy để xem mô hình có bị Overfit hay bão hòa không.
    """
    plt.figure(figsize=(12, 4))
    
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.show()