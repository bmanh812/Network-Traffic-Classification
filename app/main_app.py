import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
import os

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Nhận diện Lưu lượng Mạng", page_icon="🌐", layout="centered")

st.title("🌐 Hệ thống Nhận diện Ứng dụng qua Lưu lượng Mạng")
st.markdown("""
    **Mô hình phân loại đa lớp (Multi-class Classification)**
    Hệ thống hỗ trợ dự đoán 62 loại ứng dụng khác nhau sử dụng Machine Learning và Deep Learning.
""")

# --- 2. HÀM TẢI TẤT CẢ MÔ HÌNH (Sử dụng Cache) ---
@st.cache_resource
def load_all_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..')) 
    
    # Nạp XGBoost
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(root_dir, 'models', 'best_xgboost_model.json'))
    
    # Nạp CNN và LSTM (Thêm compile=False để tránh lỗi đọc cấu trúc)
    cnn = load_model(os.path.join(root_dir, 'models', 'best_cnn_model.h5'), compile=False)
    lstm = load_model(os.path.join(root_dir, 'models', 'best_lstm_model.h5'), compile=False)
    
    # Nạp Label Encoders
    le_final = joblib.load(os.path.join(root_dir, 'models', 'label_encoder_xgboost.pkl'))
    le_xg = joblib.load(os.path.join(root_dir, 'models', 'label_encoder_xgboost_internal.pkl'))
    
    return xgb, cnn, lstm, le_final, le_xg

# Triển khai nạp mô hình
try:
    model_xgb, model_cnn, model_lstm, le_final, le_xg = load_all_models()
    st.success("✅ Đã nạp thành công bộ 3 mô hình AI!")
except Exception as e:
    st.error(f"❌ Lỗi khi nạp mô hình: {e}")

# --- 3. GIAO DIỆN CHỌN MÔ HÌNH VÀ TẢI FILE ---
model_choice = st.selectbox(
    "🤖 Chọn mô hình để dự đoán:",
    ["XGBoost (Accuracy: ~67%)", "CNN 1D (Accuracy: ~60%)", "LSTM (Accuracy: ~68%)"]
)

uploaded_file = st.file_uploader("📂 Tải lên file dữ liệu lưu lượng (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 **Dữ liệu đầu vào (5 dòng đầu):**")
    st.dataframe(df.head())
    
    if st.button("🚀 Thực hiện Dự đoán"):
        with st.spinner(f"Đang phân tích bằng {model_choice}..."):
            X_input = df.values
            
            # --- 4. XỬ LÝ THEO TỪNG MÔ HÌNH ---
            if "XGBoost" in model_choice:
                probabilities = model_xgb.predict_proba(X_input)
                predicted_indices = np.argmax(probabilities, axis=1)
                
                # Giải mã nhãn cho XGBoost (Dùng le_xg trước rồi mới le_final)
                actual_names = [str(le_final.classes_[int(c)]) for c in le_xg.classes_]
                best_idx = predicted_indices[0]
                predicted_app_name = actual_names[best_idx]
                confidence = probabilities[0][best_idx] * 100
                
            else: 
                # Deep Learning (CNN & LSTM) cần dữ liệu 3 chiều: (Samples, Features, 1)
                X_input_3d = np.expand_dims(X_input, axis=2)
                
                if "CNN" in model_choice:
                    probabilities = model_cnn.predict(X_input_3d)
                else:  # LSTM
                    probabilities = model_lstm.predict(X_input_3d)
                
                predicted_indices = np.argmax(probabilities, axis=1)
                best_idx = predicted_indices[0]
                
                # Giải mã nhãn cho Deep Learning (Móc trực tiếp từ le_final)
                predicted_app_name = str(le_final.classes_[best_idx])
                confidence = probabilities[0][best_idx] * 100
            
            # --- 5. HIỂN THỊ KẾT QUẢ ---
            st.markdown("---")
            st.markdown(f"### 🎯 Kết quả phân tích từ {model_choice}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ứng dụng dự đoán", predicted_app_name)
            with col2:
                st.metric("Độ tự tin", f"{confidence:.2f}%")
            
            st.progress(int(confidence) / 100)