# Components Architecture

Thư mục này chứa các UI components được tách ra từ `app.py` để dễ bảo trì và mở rộng.

## Cấu trúc

```
components/
├── __init__.py              # Export các components chính
├── constants.py             # Hằng số, config, theme colors
├── utils.py                 # Utility functions (hash, formatting, session state)
├── styles.py                # CSS styles cho Streamlit
├── model_loader.py          # Model loading và prediction logic
├── topbar.py                # Top navigation bar component
├── sidebar.py               # Sidebar với model selection
├── result_card.py           # Prediction result display
├── manual_analysis.py       # Tab phân tích thủ công
└── auto_scanner.py          # Tab quét tự động với feed
```

## Sử dụng

### Chạy app refactored:

```bash
streamlit run app_refactored.py
```

### Import components:

```python
from components import (
    inject_styles,
    render_topbar,
    render_sidebar,
    render_manual_analysis,
    render_auto_scanner,
    render_result_card,
)
```

## Lợi ích của việc tách component

1. **Dễ bảo trì**: Mỗi component có trách nhiệm rõ ràng
2. **Tái sử dụng**: Components có thể dùng lại ở nhiều nơi
3. **Test dễ hơn**: Có thể test từng component riêng lẻ
4. **Cộng tác tốt hơn**: Nhiều người có thể làm việc trên các file khác nhau
5. **Giảm độ phức tạp**: File nhỏ hơn, dễ đọc và hiểu hơn

## Chi tiết từng component

### constants.py
Chứa tất cả config và constants:
- `TAB_OPTIONS`, `MODEL_OPTIONS`
- `PLATFORM_OPTIONS`, `TOPIC_OPTIONS`
- `VERDICT_THEME` - màu sắc cho từng verdict
- `MODEL_NAME_MAPPING` - mapping model name sang predictor

### utils.py
Utility functions không phụ thuộc UI:
- `clamp()` - giới hạn giá trị
- `stable_noise()` - tạo random noise từ hash
- `verdict_from_score()` - convert score sang verdict
- `format_relative()` - format timestamp
- `init_session_state()` - khởi tạo session state
- `update_stats()` - cập nhật thống kê

### styles.py
CSS styles cho toàn bộ app. Tách riêng để dễ customize theme.

### model_loader.py
Logic load và predict với models:
- `load_all_predictors()` - load và cache models
- `real_model_prediction()` - predict với real models
- `simulate_model_prediction()` - fallback simulation

### topbar.py
Top navigation bar với logo và tab selection.

### sidebar.py
Sidebar với:
- Branding
- Model status indicator
- Model selection
- Platform toggles

### result_card.py
Hiển thị prediction results với:
- Verdict chip
- Confidence bar
- Fake/Real probability
- Metadata (model, processing time)
- Warnings cho low confidence

### manual_analysis.py
Tab phân tích thủ công:
- Text input với character counter
- Vietnamese detection warning
- Analyze button
- Result display

### auto_scanner.py
Tab quét tự động:
- Scanner configuration panel
- Platform/topic selection
- Interval settings
- Live feed display
- Statistics summary
- Auto-refresh logic

