import argparse
import os

def main():
    # 1. Tạo bộ điều khiển dòng lệnh (CLI)
    parser = argparse.ArgumentParser(description="Hệ thống phân loại Network Traffic - Nhóm Bộ Ba")
    parser.add_argument('--mode', type=str, choices=['train_cnn', 'train_lstm', 'train_xgboost', 'evaluate'], 
                        help="Chọn chế độ: train_cnn, train_lstm , train_xgboost hoặc evaluate")
    
    args = parser.parse_args()

    # 2. Điều hướng hành động
    if args.mode == 'train_cnn':
        print("--- Đang khởi động tiến trình huấn luyện CNN ---")
        os.system('python train_cnn.py')
        
    elif args.mode == 'train_lstm':
        print("--- Đang khởi động tiến trình huấn luyện LSTM ---")
        os.system('python train_lstm.py')

    elif args.mode == 'train_xgboost':
        print("--- Đang khởi động tiến trình huấn luyện XGBOOST ---")
        os.system('python train_xgboost.py')
        
    elif args.mode == 'evaluate':
        print("--- Đang khởi động tiến trình đánh giá mô hình ---")
        # Sau này mình sẽ viết logic so sánh 2 mô hình ở đây
        print("Tính năng đang được cập nhật cho Tuần 5...")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()