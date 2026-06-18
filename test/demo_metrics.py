"""
=============================================================================
DEMO: Cách Tính Metrics và Hiểu Ý Nghĩa
=============================================================================
File này minh họa cách tính từng metric và giải thích ý nghĩa
=============================================================================
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def demo_confusion_matrix():
    """Demo 1: Hiểu Confusion Matrix"""
    print("="*60)
    print("DEMO 1: CONFUSION MATRIX")
    print("="*60)
    
    # Dữ liệu mẫu
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 REAL, 5 FAKE
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])  # Predictions
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nDữ liệu:")
    print(f"  Thực tế:   {y_true} (5 REAL, 5 FAKE)")
    print(f"  Dự đoán:   {y_pred}")
    
    print("\nConfusion Matrix:")
    print(cm)
    print()
    print("Giải thích:")
    print(f"  [[{cm[0,0]}  {cm[0,1]}]")
    print(f"   [{cm[1,0]}  {cm[1,1]}]]")
    print()
    print(f"  TN (True Negative):  {cm[0,0]} - Dự đoán ĐÚNG là REAL")
    print(f"  FP (False Positive): {cm[0,1]} - Dự đoán SAI thành FAKE (Type I Error)")
    print(f"  FN (False Negative): {cm[1,0]} - Dự đoán SAI thành REAL (Type II Error)")
    print(f"  TP (True Positive):  {cm[1,1]} - Dự đoán ĐÚNG là FAKE")
    print()
    print("Chi tiết:")
    print(f"  - 5 tin REAL: {cm[0,0]} đúng, {cm[0,1]} sai (dự đoán thành FAKE)")
    print(f"  - 5 tin FAKE: {cm[1,1]} đúng, {cm[1,0]} sai (dự đoán thành REAL)")
    print(f"  - Tổng lỗi: {cm[0,1] + cm[1,0]} / 10 = {(cm[0,1] + cm[1,0])/10*100:.0f}%")


def demo_metrics():
    """Demo 2: Tính các metrics"""
    print("\n" + "="*60)
    print("DEMO 2: TÍNH CÁC METRICS")
    print("="*60)
    
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    
    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\nGiá trị từ Confusion Matrix:")
    print(f"  TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    
    print("\n1. ACCURACY (Độ chính xác tổng thể):")
    print(f"   Công thức: (TP + TN) / (TP + TN + FP + FN)")
    print(f"   Tính:      ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN})")
    print(f"   Kết quả:   {accuracy:.4f} = {accuracy*100:.2f}%")
    print(f"   Ý nghĩa:   Dự đoán đúng {accuracy*100:.0f}% trường hợp")
    
    print("\n2. PRECISION (Độ chính xác khi dự đoán FAKE):")
    print(f"   Công thức: TP / (TP + FP)")
    print(f"   Tính:      {TP} / ({TP} + {FP})")
    print(f"   Kết quả:   {precision:.4f} = {precision*100:.2f}%")
    print(f"   Ý nghĩa:   Trong {TP+FP} tin dự đoán FAKE,")
    print(f"              {TP} thực sự là FAKE ({precision*100:.0f}%)")
    print(f"              {FP} là REAL bị nhầm (Type I Error)")
    
    print("\n3. RECALL (Độ bao phủ / phát hiện được bao nhiêu FAKE):")
    print(f"   Công thức: TP / (TP + FN)")
    print(f"   Tính:      {TP} / ({TP} + {FN})")
    print(f"   Kết quả:   {recall:.4f} = {recall*100:.2f}%")
    print(f"   Ý nghĩa:   Trong {TP+FN} tin FAKE thực tế,")
    print(f"              phát hiện được {TP} tin ({recall*100:.0f}%)")
    print(f"              bỏ sót {FN} tin (Type II Error)")
    
    print("\n4. F1-SCORE (Trung bình điều hòa):")
    print(f"   Công thức: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   Tính:      2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"   Kết quả:   {f1:.4f} = {f1*100:.2f}%")
    print(f"   Ý nghĩa:   Cân bằng giữa Precision và Recall")


def demo_trade_offs():
    """Demo 3: Trade-offs giữa Precision và Recall"""
    print("\n" + "="*60)
    print("DEMO 3: TRADE-OFFS PRECISION vs RECALL")
    print("="*60)
    
    print("\nCase 1: Model thận trọng (chỉ dự đoán FAKE khi rất chắc)")
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred1 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # Chỉ dự đoán 3 FAKE
    
    p1 = precision_score(y_true, y_pred1)
    r1 = recall_score(y_true, y_pred1)
    f1_1 = f1_score(y_true, y_pred1)
    
    print(f"   Precision: {p1:.4f} (cao - ít nhầm)")
    print(f"   Recall:    {r1:.4f} (thấp - bỏ sót nhiều)")
    print(f"   F1:        {f1_1:.4f}")
    print(f"   → Ít tin thật bị nhầm, nhưng bỏ sót nhiều tin giả")
    
    print("\nCase 2: Model aggressive (dễ dự đoán FAKE)")
    y_pred2 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Dự đoán 8 FAKE
    
    p2 = precision_score(y_true, y_pred2)
    r2 = recall_score(y_true, y_pred2)
    f1_2 = f1_score(y_true, y_pred2)
    
    print(f"   Precision: {p2:.4f} (thấp - nhầm nhiều)")
    print(f"   Recall:    {r2:.4f} (cao - bắt hết)")
    print(f"   F1:        {f1_2:.4f}")
    print(f"   → Bắt hết tin giả, nhưng nhiều tin thật bị nhầm")
    
    print("\nCase 3: Model cân bằng")
    y_pred3 = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    
    p3 = precision_score(y_true, y_pred3)
    r3 = recall_score(y_true, y_pred3)
    f1_3 = f1_score(y_true, y_pred3)
    
    print(f"   Precision: {p3:.4f}")
    print(f"   Recall:    {r3:.4f}")
    print(f"   F1:        {f1_3:.4f} (cao nhất!)")
    print(f"   → Cân bằng tốt nhất")


def demo_imbalanced_data():
    """Demo 4: Tại sao Accuracy không đủ với imbalanced data"""
    print("\n" + "="*60)
    print("DEMO 4: IMBALANCED DATA - TẠI SAO ACCURACY KHÔNG ĐỦ")
    print("="*60)
    
    # Dataset imbalanced: 95 REAL, 5 FAKE
    y_true = np.array([0]*95 + [1]*5)
    
    print("\nDataset: 95 REAL, 5 FAKE (imbalanced)")
    
    print("\nModel 1: Ngu (luôn dự đoán REAL)")
    y_pred1 = np.array([0]*100)  # Luôn dự đoán REAL
    
    acc1 = accuracy_score(y_true, y_pred1)
    p1 = precision_score(y_true, y_pred1, zero_division=0)
    r1 = recall_score(y_true, y_pred1)
    f1_1 = f1_score(y_true, y_pred1)
    
    print(f"   Accuracy:  {acc1:.4f} = {acc1*100:.0f}% (CAO!)")
    print(f"   Precision: {p1:.4f} (undefined - không dự đoán FAKE)")
    print(f"   Recall:    {r1:.4f} (THẤP - bỏ sót hết 5 FAKE)")
    print(f"   F1:        {f1_1:.4f} (RẤT THẤP)")
    print(f"   → Accuracy cao nhưng model VÔ DỤNG!")
    
    print("\nModel 2: Thông minh (dự đoán đúng 4/5 FAKE)")
    y_pred2 = np.array([0]*95 + [1]*4 + [0]*1)  # Đúng 4 FAKE, bỏ sót 1
    
    acc2 = accuracy_score(y_true, y_pred2)
    p2 = precision_score(y_true, y_pred2)
    r2 = recall_score(y_true, y_pred2)
    f1_2 = f1_score(y_true, y_pred2)
    
    print(f"   Accuracy:  {acc2:.4f} = {acc2*100:.0f}%")
    print(f"   Precision: {p2:.4f}")
    print(f"   Recall:    {r2:.4f} (CAO - bắt 4/5 FAKE)")
    print(f"   F1:        {f1_2:.4f}")
    print(f"   → Accuracy chỉ cao hơn 4%, nhưng model TỐT HƠN NHIỀU!")
    
    print("\n💡 Bài học: Với imbalanced data, xem F1-Score chứ không phải Accuracy!")


def demo_confidence():
    """Demo 5: Độ tin cậy (Confidence)"""
    print("\n" + "="*60)
    print("DEMO 5: ĐỘ TIN CẬY (CONFIDENCE)")
    print("="*60)
    
    print("\nGiả sử model predict_proba cho 3 samples:")
    
    # Sample 1: High confidence
    proba1 = np.array([0.02, 0.98])
    pred1 = np.argmax(proba1)
    conf1 = proba1[pred1]
    
    print(f"\nSample 1:")
    print(f"   Probabilities: REAL={proba1[0]:.2f}, FAKE={proba1[1]:.2f}")
    print(f"   Prediction:    {'FAKE' if pred1==1 else 'REAL'}")
    print(f"   Confidence:    {conf1:.2f} = {conf1*100:.0f}%")
    print(f"   → RẤT CHẮC CHẮN (High confidence)")
    
    # Sample 2: Medium confidence
    proba2 = np.array([0.25, 0.75])
    pred2 = np.argmax(proba2)
    conf2 = proba2[pred2]
    
    print(f"\nSample 2:")
    print(f"   Probabilities: REAL={proba2[0]:.2f}, FAKE={proba2[1]:.2f}")
    print(f"   Prediction:    {'FAKE' if pred2==1 else 'REAL'}")
    print(f"   Confidence:    {conf2:.2f} = {conf2*100:.0f}%")
    print(f"   → KHÁC CHẮC CHẮN (Medium confidence)")
    
    # Sample 3: Low confidence
    proba3 = np.array([0.48, 0.52])
    pred3 = np.argmax(proba3)
    conf3 = proba3[pred3]
    
    print(f"\nSample 3:")
    print(f"   Probabilities: REAL={proba3[0]:.2f}, FAKE={proba3[1]:.2f}")
    print(f"   Prediction:    {'FAKE' if pred3==1 else 'REAL'}")
    print(f"   Confidence:    {conf3:.2f} = {conf3*100:.0f}%")
    print(f"   → KHÔNG CHẮC CHẮN (Low confidence)")
    print(f"   → Cần HUMAN REVIEW!")
    
    print("\n💡 Quy tắc:")
    print("   Confidence > 90%: Tin tưởng cao")
    print("   Confidence 70-90%: Cần cẩn thận")
    print("   Confidence < 70%: Cần human review")


def demo_classification_report():
    """Demo 6: Classification Report đầy đủ"""
    print("\n" + "="*60)
    print("DEMO 6: CLASSIFICATION REPORT ĐẦY ĐỦ")
    print("="*60)
    
    # Data giống LinearSVC trong project (simplified)
    y_true = np.array([0]*50 + [1]*50)  # 50 REAL, 50 FAKE
    y_pred = np.array([0]*48 + [1]*2 + [0]*1 + [1]*49)  # 48 TN, 2 FP, 1 FN, 49 TP
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=["REAL (0)", "FAKE (1)"],
                               digits=4))
    
    print("Giải thích từng cột:")
    print("  - precision:   Trong số dự đoán class này, bao nhiêu % đúng")
    print("  - recall:      Trong số thực tế là class này, bao nhiêu % phát hiện")
    print("  - f1-score:    Trung bình điều hòa precision và recall")
    print("  - support:     Số samples thực tế của class này")
    
    print("\nMacro avg vs Weighted avg:")
    print("  - macro avg:    Trung bình đơn giản (coi tất cả class bằng nhau)")
    print("  - weighted avg: Trung bình có trọng số (theo số lượng samples)")


def main():
    """Run all demos"""
    demo_confusion_matrix()
    input("\nẤn Enter để tiếp tục...")
    
    demo_metrics()
    input("\nẤn Enter để tiếp tục...")
    
    demo_trade_offs()
    input("\nẤn Enter để tiếp tục...")
    
    demo_imbalanced_data()
    input("\nẤn Enter để tiếp tục...")
    
    demo_confidence()
    input("\nẤn Enter để tiếp tục...")
    
    demo_classification_report()
    
    print("\n" + "="*60)
    print("✅ DEMO HOÀN THÀNH!")
    print("="*60)
    print("🔬 Test thực tế: python test/model_benchmarks.py")


if __name__ == "__main__":
    main()
