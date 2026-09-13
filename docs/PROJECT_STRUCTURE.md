# Project Structure — Traffic Vehicle Counting

Tài liệu này mô tả cấu trúc thư mục và quy ước lưu dữ liệu của project.

- Pipeline và experiment: xem `PROJECT_PIPELINE.md`
- Phân công thành viên: xem `TEAM_TASKS.md`

---

## 1. Cấu trúc project

```text
traffic-vehicle-counting/
├── data/
│   ├── videos/
│   │   ├── development/
│   │   └── evaluation/
│   ├── ground_truth/
│   └── datasets/
│       ├── ua_detrac/
│       └── vietnam_traffic/
│
├── models/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── counting/
│   ├── statistics/
│   ├── visualization/
│   └── traffic_counting.py
│
├── experiments/
├── evaluation/
│
├── outputs/
│   ├── videos/
│   ├── csv/
│   └── charts/
│
├── docs/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 2. `data/`

Chứa dữ liệu đầu vào của project.

### `data/videos/development/`

Chứa video dùng trong quá trình phát triển:

- viết và debug code;
- thử model, tracker, counting line/zone;
- kiểm tra các module trước khi chạy chính thức.

Video development có thể ngắn và được sử dụng nhiều lần.

Ví dụ:

```text
dev_01_detrac_MVI_20011.mp4
dev_02_detrac_MVI_20012.mp4
```

### `data/videos/evaluation/`

Chứa video dùng cho đánh giá chính thức.

Ưu tiên:

- một cảnh giao thông liên tục;
- thời lượng khoảng 2–3 phút;
- giữ cố định khi so sánh các cấu hình.

Ví dụ:

```text
eval_01_traffic_2m30s.mp4
eval_02_traffic_2m45s.mp4
```

Tóm lại:

```text
development → dùng để xây và chỉnh hệ thống
evaluation  → dùng để đo kết quả chính thức
```

### `data/ground_truth/`

Chứa kết quả đếm thủ công tương ứng với evaluation video.

Ví dụ:

```text
eval_01_manual_counts.csv
eval_02_manual_counts.csv
```

### `data/datasets/ua_detrac/`

Chứa dữ liệu hoặc các file cần thiết từ UA-DETRAC.

Dataset đầy đủ có dung lượng lớn nên không bắt buộc đưa toàn bộ lên GitHub.

### `data/datasets/vietnam_traffic/`

Chứa dữ liệu giao thông Việt Nam dùng cho phần fine-tuning hoặc thử nghiệm mở rộng.

Có thể tổ chức:

```text
vietnam_traffic/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## 3. `models/`

Chứa model weights sử dụng trong project.

Ví dụ:

```text
yolov8n.pt
yolov8s.pt
yolov8m.pt
best_vietnam.pt
```

Các file model lớn có thể để local thay vì commit trực tiếp lên GitHub.

---

## 4. `src/`

Chứa source code chính.

```text
src/
├── detection/       # Vehicle detection
├── tracking/        # Multi-object tracking
├── counting/        # Line / zone counting
├── statistics/      # Tổng hợp số liệu
├── visualization/   # Hiển thị video và biểu đồ
└── traffic_counting.py
```

`traffic_counting.py` là file tích hợp các thành phần của hệ thống.

---

## 5. `experiments/`

Chứa các script dùng để chạy từng experiment.

Ví dụ:

```text
experiment_01_baseline.py
experiment_02_detector.py
experiment_03_tracker.py
experiment_04_counting.py
```

Không đặt logic chính của Detection/Tracking/Counting trong folder này; các script experiment nên gọi lại code từ `src/`.

---

## 6. `evaluation/`

Chứa code đánh giá và bảng tổng hợp kết quả.

Ví dụ:

```text
evaluation/
├── evaluate_counting.py
└── results_summary.csv
```

---

## 7. `outputs/`

Chứa kết quả sinh ra khi chạy hệ thống.

```text
outputs/
├── videos/
├── csv/
└── charts/
```

### `outputs/videos/`

Video đã được xử lý, có bounding box, track ID, counting line và count.

### `outputs/csv/`

Các file counting event hoặc kết quả từng lần chạy.

Ví dụ:

```text
e1_eval01_events.csv
e2_yolov8s_eval01_events.csv
e3_botsort_eval01_events.csv
```

### `outputs/charts/`

Biểu đồ dùng cho phân tích, báo cáo và slide.

---

## 8. `docs/`

Chứa tài liệu nội bộ của project.

```text
docs/
├── PROJECT_PIPELINE.md
├── TEAM_TASKS.md
└── PROJECT_STRUCTURE.md
```

- `PROJECT_PIPELINE.md`: pipeline và kế hoạch experiment.
- `TEAM_TASKS.md`: phân công công việc.
- `PROJECT_STRUCTURE.md`: cấu trúc repository và quy ước lưu file.

---

## 9. `tests/`

Chứa các test cho từng module khi cần.

Ví dụ:

```text
test_detection.py
test_tracking.py
test_counting.py
```

---

## 10. Quy ước đặt tên

Development video:

```text
dev_<number>_<source>.mp4
```

Ví dụ:

```text
dev_01_detrac_MVI_20011.mp4
```

Evaluation video:

```text
eval_<number>_<description>.mp4
```

Ví dụ:

```text
eval_01_traffic_2m30s.mp4
```

Output experiment:

```text
<experiment>_<config>_<video>_<type>
```

Ví dụ:

```text
e2_yolov8s_eval01_events.csv
e3_botsort_eval01_output.mp4
```

---

## 11. Nguyên tắc lưu file

- Dữ liệu đầu vào → `data/`
- Model → `models/`
- Source code → `src/`
- Script experiment → `experiments/`
- Code đánh giá → `evaluation/`
- Kết quả sinh ra → `outputs/`
- Tài liệu nhóm → `docs/`

Không lưu dataset, model hoặc output lớn rải rác ở root repository.
