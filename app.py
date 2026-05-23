"""
=============================================================================
STREAMLIT WEB APPLICATION - Giao diện web để phát hiện tin giả
=============================================================================

CHỨC NĂNG CHÍNH:
1. Cung cấp giao diện web đơn giản để người dùng nhập tin tức
2. Sử dụng models đã train để dự đoán Real/Fake
3. Hiển thị kết quả với confidence score và visualizations
4. Hỗ trợ cả Baseline và Transformer models

NGƯỜI PHỤ TRÁCH: M4 (Web Application)

CÔNG NGHỆ:
- Streamlit: Simple, fast web framework cho ML/Data Science
- Plotly: Interactive charts
- Session state: Lưu trữ models và history

ƯU ĐIỂM STREAMLIT:
✅ Cực kỳ đơn giản, chỉ cần Python
✅ Tự động reload khi code thay đổi
✅ Built-in widgets đẹp (slider, selectbox, etc.)
✅ Không cần HTML/CSS/JavaScript
✅ Perfect cho ML/Data Science projects
=============================================================================
"""

# TODO: Import các thư viện cần thiết
# import streamlit as st
# import sys
# import plotly.graph_objects as go
# from pathlib import Path

# TODO: Add src to path
# sys.path.append('src')

# TODO: Import từ src/
# from data_loader import DataLoader
# from train_baseline import BaselineTrainer
# from train_transformer import TransformerTrainer


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

# TODO: Configure Streamlit page
# st.set_page_config(
#     page_title="Fake News Detection",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# =============================================================================
# CUSTOM CSS (OPTIONAL)
# =============================================================================

# TODO: Add custom CSS để làm đẹp hơn (optional)
# st.markdown("""
#     <style>
#     .main {
#         padding: 2rem;
#     }
#     .stButton>button {
#         width: 100%;
#         background-color: #2563eb;
#         color: white;
#         font-weight: bold;
#         padding: 0.75rem;
#         border-radius: 8px;
#     }
#     .stButton>button:hover {
#         background-color: #1e40af;
#     }
#     </style>
# """, unsafe_allow_html=True)


# =============================================================================
# LOAD MODELS (WITH CACHING)
# =============================================================================

# TODO: Load baseline models với caching
# @st.cache_resource
# def load_baseline_models():
#     """
#     Load baseline models - chỉ chạy 1 lần
#     
#     CHỨC NĂNG:
#     - Khởi tạo BaselineTrainer
#     - Load tất cả models đã train
#     - Return trainer object
#     
#     LƯU Ý: @st.cache_resource giúp load models 1 lần duy nhất
#     """
#     try:
#         trainer = BaselineTrainer()
#         trainer.load_models()
#         return trainer
#     except Exception as e:
#         st.error(f"Không thể load baseline models: {e}")
#         return None


# TODO: Load transformer model với caching
# @st.cache_resource
# def load_transformer_model():
#     """
#     Load transformer model - chỉ chạy 1 lần
#     
#     CHỨC NĂNG:
#     - Khởi tạo TransformerTrainer
#     - Load model đã train
#     - Return trainer object
#     
#     LƯU Ý: Transformer model nặng hơn, có thể mất vài giây
#     """
#     try:
#         trainer = TransformerTrainer()
#         trainer.load_model("models/transformer/best_model")
#         return trainer
#     except Exception as e:
#         st.warning(f"Không thể load transformer model: {e}")
#         return None


# TODO: Initialize data loader
# @st.cache_resource
# def get_data_loader():
#     """Load DataLoader"""
#     return DataLoader()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# TODO: Function để tạo gauge chart cho confidence
# def create_confidence_gauge(confidence, prediction):
#     """
#     Tạo gauge chart hiển thị confidence score
#     
#     INPUT:
#     - confidence: Confidence score (0-100)
#     - prediction: "REAL" hoặc "FAKE"
#     
#     OUTPUT:
#     - Plotly figure object
#     
#     CHỨC NĂNG:
#     - Tạo gauge chart với màu sắc tùy theo prediction
#     - Green cho REAL, Red cho FAKE
#     """
#     color = "#10b981" if prediction == "REAL" else "#ef4444"
#     
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=confidence,
#         title={'text': "Confidence Score"},
#         gauge={
#             'axis': {'range': [0, 100]},
#             'bar': {'color': color},
#             'steps': [
#                 {'range': [0, 50], 'color': "lightgray"},
#                 {'range': [50, 75], 'color': "gray"},
#                 {'range': [75, 100], 'color': "darkgray"}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 90
#             }
#         }
#     ))
#     
#     fig.update_layout(height=300)
#     return fig


# TODO: Function để tạo bar chart cho probabilities
# def create_probability_chart(prob_real, prob_fake):
#     """
#     Tạo bar chart hiển thị probabilities
#     
#     INPUT:
#     - prob_real: Probability của Real (0-100)
#     - prob_fake: Probability của Fake (0-100)
#     
#     OUTPUT:
#     - Plotly figure object
#     """
#     fig = go.Figure(data=[
#         go.Bar(
#             x=['Real', 'Fake'],
#             y=[prob_real, prob_fake],
#             marker_color=['#10b981', '#ef4444'],
#             text=[f'{prob_real:.1f}%', f'{prob_fake:.1f}%'],
#             textposition='auto',
#         )
#     ])
#     
#     fig.update_layout(
#         title="Probability Distribution",
#         yaxis_title="Probability (%)",
#         height=300,
#         showlegend=False
#     )
#     
#     return fig


# =============================================================================
# MAIN APP
# =============================================================================

# TODO: Main function
# def main():
#     """Main Streamlit app"""
#     
#     # =========================================================================
#     # HEADER
#     # =========================================================================
#     
#     st.title("🔍 Fake News Detection System")
#     st.markdown("Hệ thống phát hiện tin giả sử dụng Machine Learning và Deep Learning")
#     st.markdown("---")
#     
#     
#     # =========================================================================
#     # SIDEBAR - MODEL SELECTION & INFO
#     # =========================================================================
#     
#     st.sidebar.header("⚙️ Cài đặt")
#     
#     # Model selection
#     model_type = st.sidebar.selectbox(
#         "Chọn Model",
#         ["Baseline (XGBoost)", "Transformer (BERT)"],
#         help="Baseline nhanh hơn, Transformer chính xác hơn"
#     )
#     
#     # Info
#     st.sidebar.markdown("---")
#     st.sidebar.header("📊 Thông tin")
#     st.sidebar.info("""
#     **Baseline Models:**
#     - Logistic Regression
#     - Random Forest
#     - Naive Bayes
#     - XGBoost ⭐
#     - LightGBM
#     
#     **Transformer Models:**
#     - BERT
#     - PhoBERT (Tiếng Việt)
#     - RoBERTa
#     """)
#     
#     # Instructions
#     st.sidebar.markdown("---")
#     st.sidebar.header("📖 Hướng dẫn")
#     st.sidebar.markdown("""
#     1. Nhập hoặc dán nội dung tin tức
#     2. Chọn model muốn sử dụng
#     3. Nhấn "Phân tích"
#     4. Xem kết quả và confidence score
#     """)
#     
#     
#     # =========================================================================
#     # LOAD MODELS
#     # =========================================================================
#     
#     with st.spinner("Đang load models..."):
#         data_loader = get_data_loader()
#         baseline_trainer = load_baseline_models()
#         transformer_trainer = load_transformer_model()
#     
#     
#     # =========================================================================
#     # MAIN CONTENT - INPUT FORM
#     # =========================================================================
#     
#     st.header("📝 Nhập nội dung tin tức")
#     
#     # Text input
#     text_input = st.text_area(
#         "Nội dung tin tức:",
#         height=200,
#         placeholder="Dán hoặc nhập nội dung tin tức cần kiểm tra...",
#         help="Nhập nội dung tin tức bạn muốn kiểm tra"
#     )
#     
#     # Character count
#     if text_input:
#         st.caption(f"Số ký tự: {len(text_input)}")
#     
#     # Analyze button
#     analyze_button = st.button("🔍 Phân tích", type="primary", use_container_width=True)
#     
#     
#     # =========================================================================
#     # PREDICTION & RESULTS
#     # =========================================================================
#     
#     if analyze_button:
#         if not text_input or len(text_input.strip()) == 0:
#             st.error("⚠️ Vui lòng nhập nội dung tin tức!")
#         else:
#             # Clean text
#             with st.spinner("Đang xử lý..."):
#                 cleaned_text = data_loader.clean_text(text_input)
#             
#             # Predict
#             try:
#                 with st.spinner("Đang phân tích..."):
#                     if "Baseline" in model_type:
#                         if baseline_trainer is None:
#                             st.error("❌ Baseline models chưa được train!")
#                             return
#                         prediction, probabilities = baseline_trainer.predict(
#                             cleaned_text, 
#                             'xgboost'
#                         )
#                     else:  # Transformer
#                         if transformer_trainer is None:
#                             st.error("❌ Transformer model chưa được train!")
#                             return
#                         prediction, probabilities = transformer_trainer.predict(
#                             cleaned_text
#                         )
#                 
#                 # Format results
#                 is_fake = prediction == 1
#                 label = "FAKE" if is_fake else "REAL"
#                 confidence = float(probabilities[prediction]) * 100
#                 prob_real = float(probabilities[0]) * 100
#                 prob_fake = float(probabilities[1]) * 100
#                 
#                 
#                 # =====================================================================
#                 # DISPLAY RESULTS
#                 # =====================================================================
#                 
#                 st.markdown("---")
#                 st.header("📊 Kết quả phân tích")
#                 
#                 # Main result
#                 col1, col2, col3 = st.columns([1, 2, 1])
#                 
#                 with col2:
#                     if is_fake:
#                         st.error(f"### ❌ {label}")
#                         st.markdown(f"**Confidence:** {confidence:.2f}%")
#                     else:
#                         st.success(f"### ✅ {label}")
#                         st.markdown(f"**Confidence:** {confidence:.2f}%")
#                 
#                 st.markdown("---")
#                 
#                 # Visualizations
#                 col1, col2 = st.columns(2)
#                 
#                 with col1:
#                     st.plotly_chart(
#                         create_confidence_gauge(confidence, label),
#                         use_container_width=True
#                     )
#                 
#                 with col2:
#                     st.plotly_chart(
#                         create_probability_chart(prob_real, prob_fake),
#                         use_container_width=True
#                     )
#                 
#                 # Detailed probabilities
#                 st.markdown("---")
#                 st.subheader("📈 Chi tiết Probabilities")
#                 
#                 col1, col2 = st.columns(2)
#                 
#                 with col1:
#                     st.metric(
#                         label="✅ Tin thật (Real)",
#                         value=f"{prob_real:.2f}%",
#                         delta=None
#                     )
#                 
#                 with col2:
#                     st.metric(
#                         label="❌ Tin giả (Fake)",
#                         value=f"{prob_fake:.2f}%",
#                         delta=None
#                     )
#                 
#                 # Model info
#                 st.markdown("---")
#                 st.info(f"🤖 Model sử dụng: **{model_type}**")
#                 
#                 # Warning
#                 st.warning("""
#                 ⚠️ **Lưu ý:** Kết quả chỉ mang tính chất tham khảo. 
#                 Vui lòng kiểm chứng thông tin từ nhiều nguồn đáng tin cậy.
#                 """)
#                 
#             except Exception as e:
#                 st.error(f"❌ Lỗi khi phân tích: {str(e)}")
#     
#     
#     # =========================================================================
#     # FOOTER
#     # =========================================================================
#     
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: gray;'>
#         <p>© 2024 Fake News Detection System | Powered by Streamlit & AI</p>
#     </div>
#     """, unsafe_allow_html=True)


# =============================================================================
# RUN APP
# =============================================================================

# TODO: Run main function
# if __name__ == "__main__":
#     main()


# =============================================================================
# HƯỚNG DẪN SỬ DỤNG
# =============================================================================
"""
CHẠY APP:

1. Cài đặt Streamlit:
   pip install streamlit

2. Chạy app:
   streamlit run app.py

3. App sẽ tự động mở browser tại:
   http://localhost:8501

4. Để chạy trên port khác:
   streamlit run app.py --server.port 8080

5. Để cho phép truy cập từ mạng ngoài:
   streamlit run app.py --server.address 0.0.0.0

STREAMLIT FEATURES:

1. Auto-reload:
   - Khi bạn sửa code, Streamlit tự động reload
   - Không cần restart server

2. Widgets có sẵn:
   - st.text_input(): Input box
   - st.text_area(): Text area
   - st.button(): Button
   - st.selectbox(): Dropdown
   - st.slider(): Slider
   - st.checkbox(): Checkbox
   - st.radio(): Radio buttons
   - st.file_uploader(): File upload

3. Layout:
   - st.columns(): Tạo columns
   - st.sidebar: Sidebar
   - st.expander(): Collapsible section
   - st.tabs(): Tabs

4. Display:
   - st.write(): Display anything
   - st.markdown(): Markdown
   - st.title(), st.header(), st.subheader(): Headers
   - st.success(), st.error(), st.warning(), st.info(): Colored boxes
   - st.metric(): Metric display
   - st.plotly_chart(): Plotly charts
   - st.dataframe(): Display DataFrame

5. Caching:
   - @st.cache_data: Cache data (DataFrames, lists, etc.)
   - @st.cache_resource: Cache resources (models, connections)

6. Session State:
   - st.session_state: Lưu trữ data giữa các reruns
   - Ví dụ: st.session_state['history'] = []

ADVANCED FEATURES (OPTIONAL):

1. File upload:
   uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
   if uploaded_file:
       df = pd.read_csv(uploaded_file)

2. Download button:
   st.download_button(
       label="Download results",
       data=csv_data,
       file_name="results.csv",
       mime="text/csv"
   )

3. Progress bar:
   progress_bar = st.progress(0)
   for i in range(100):
       progress_bar.progress(i + 1)

4. Tabs:
   tab1, tab2 = st.tabs(["Predict", "History"])
   with tab1:
       # Prediction UI
   with tab2:
       # History UI

5. Expander:
   with st.expander("See explanation"):
       st.write("Detailed explanation here")

DEPLOYMENT:

1. Streamlit Cloud (Free):
   - Push code to GitHub
   - Connect to Streamlit Cloud
   - Deploy với 1 click
   - URL: https://your-app.streamlit.app

2. Heroku:
   # setup.sh
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   
   # Procfile
   web: sh setup.sh && streamlit run app.py

3. Docker:
   FROM python:3.9
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]

TIPS:

✅ Sử dụng @st.cache_resource cho models (load 1 lần)
✅ Sử dụng st.spinner() cho loading states
✅ Sử dụng st.columns() cho layout đẹp
✅ Sử dụng Plotly cho interactive charts
✅ Thêm st.sidebar cho settings
✅ Thêm help text cho widgets
✅ Sử dụng st.session_state cho history feature
✅ Test trên mobile (Streamlit responsive by default)

COMMON ISSUES:

1. Models load chậm:
   - Sử dụng @st.cache_resource
   - Load models 1 lần duy nhất

2. App reload liên tục:
   - Kiểm tra code có lỗi không
   - Kiểm tra file watching settings

3. Memory issues:
   - Clear cache: st.cache_data.clear()
   - Giảm model size
   - Sử dụng quantization

4. Port already in use:
   - Đổi port: streamlit run app.py --server.port 8502
   - Kill process cũ
"""
