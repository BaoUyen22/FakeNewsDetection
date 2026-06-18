"""
Constants and configuration for Verify.AI
"""

TAB_OPTIONS = ["Phân tích thủ công", "Quét tự động"]
MODEL_OPTIONS = ["Linear SVC", "Logistic Regression", "LSTM", "BERT"]

ENABLE_MODEL_COMPARISON = False  # true để bật so sánh nhiều model

PLATFORM_OPTIONS = [
    ("x", "X / Twitter", "X"),
    ("rss", "RSS Feed", "RSS"),
    ("facebook", "Facebook", "FB"),
    ("web", "Web crawl", "WEB"),
]

TOPIC_OPTIONS = [
    "Chính trị",
    "Y tế / COVID",
    "Khoa học",
    "Kinh tế",
    "Môi trường",
    "Giáo dục",
    "Công nghệ",
]

INTERVAL_OPTIONS = {
    "5 phút": 5,
    "15 phút": 15,
    "1 giờ": 60,
    "Thủ công": 0,
}

HEADLINES = {
    "Chính trị": [
        "Nguồn ẩn danh nói có thay đổi chính sách qua đêm.",
        "Bài đăng cắt ghép phát biểu lãnh đạo gây tranh luận mạnh.",
        "Nội dung chưa kiểm chứng về nghị quyết mới lan truyền nhanh.",
    ],
    "Y tế / COVID": [
        "Tin đồn về thuốc tự pha chữa bệnh đang tăng nhanh trên mạng.",
        "Video cũ bị gắn sai ngữ cảnh về vaccine tại bệnh viện trung ương.",
        "Bản tin chưa xác minh khẳng định có biến thể mới tại nhiều tỉnh.",
    ],
    "Khoa học": [
        "Bài viết nói máy phát điện vĩnh cửu đã được chứng minh thành công.",
        "Ảnh thiên văn chỉnh sửa quá mức bị chia sẻ như nghiên cứu thật.",
        "Tin về thí nghiệm gây sốc thiếu dẫn nguồn từ tạp chí uy tín.",
    ],
    "Kinh tế": [
        "Tin nóng về ngân hàng ngừng chi trả bị lan rộng trên các nhóm kín.",
        "Bài đăng dự báo tỷ giá tăng sốc kèm số liệu không rõ nguồn.",
        "Tuyên bố cắt giảm thuế diện rộng chưa có văn bản chính thức.",
    ],
    "Môi trường": [
        "Tin cảnh báo ô nhiễm nguồn nước dùng hình ảnh từ năm trước.",
        "Video đám cháy rừng ở nước ngoài bị gắn thành sự kiện trong nước.",
        "Nội dung thời tiết cực đoan được chia sẻ lại với tiêu đề sai lệch.",
    ],
    "Giáo dục": [
        "Tin giả về lịch thi quốc gia bị chỉnh sửa lan trên nhóm phụ huynh.",
        "Ảnh thông báo tuyển sinh không dấu xác thực được chia sẻ rộng rãi.",
        "Thông tin học phí tăng đột biến chưa có văn bản chính thức.",
    ],
    "Công nghệ": [
        "Tin đồn rò rỉ dữ liệu từ ứng dụng lớn chưa được bên vận hành xác nhận.",
        "Bài viết dùng ảnh giả để nói về điện thoại mới chưa ra mắt.",
        "Nội dung quảng cáo AI thần kỳ không có bằng chứng kỹ thuật.",
    ],
}

VERDICT_THEME = {
    "Tin giả": {
        "bg": "#FCEBEB",
        "border": "#F09595",
        "text": "#791F1F",
        "bar": "#A32D2D",
    },
    "Tin thật": {
        "bg": "#E1F5EE",
        "border": "#5DCAA5",
        "text": "#085041",
        "bar": "#0F6E56",
    },
    "Chưa rõ": {
        "bg": "#FAEEDA",
        "border": "#EF9F27",
        "text": "#633806",
        "bar": "#854F0B",
    },
}

# Model name to predictor mapping
MODEL_NAME_MAPPING = {
    "Linear SVC": ("baseline", "LinearSVC"),
    "Logistic Regression": ("baseline", "Logistic Regression"),
    "LSTM": ("deep_learning", "LSTM"),
    "BERT": ("bert", "BERT (DistilBERT)")
}
