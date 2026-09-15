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
├── KIET.md                 # bàn giao Statistics / Ground Truth / Evaluation
└── .gitignore
```

---

## 2. `data/`

Chứa dữ liệu đầu vào của project.

### `data/videos/development/`

Chứa video dùng để phát triển, debug và thử Detection, Tracking, Counting trước khi chạy evaluation chính thức.

Hiện folder đã có đủ các dạng luồng giao thông cần thiết:

```text
dev_01_detrac_MVI_20011.mp4   # Đường hai hướng
dev_02_detrac_MVI_20012.mp4   # Đường hai hướng
dev_03_detrac_MVI_20034.mp4   # Đường hai hướng
dev_04_4_corners.mp4           # Ngã tư / nhiều hướng
dev_05_1_line.mp4              # Đường một hướng
```

Các video development dùng để:

- debug pipeline;
- thử model và tracker;
- thử single line, multiple lines / zones;
- kiểm tra đếm theo một hướng, hai hướng và nhiều hướng.

Các video này không dùng để báo cáo kết quả chính thức.

### `data/videos/evaluation/`

Chứa các video dùng để chạy experiment và đánh giá kết quả chính thức.

Hiện folder gồm:

```text
eval_01_detrac_40131_40141.mp4   # Đường hai chiều
eval_02.mp4                       # Nhiều hướng xe đi vào, có thể đếm bằng một counting line
eval_03.mp4                       # Ngã tư / nhiều hướng
```

Mục đích:

- `eval_01`: kiểm tra tracking và counting trên đường hai chiều;
- `eval_02`: kiểm tra trường hợp nhiều luồng xe nhưng vẫn có thể sử dụng một counting line;
- `eval_03`: kiểm tra tình huống ngã tư phức tạp với nhiều hướng di chuyển.

Ba video trên là evaluation set đã chuẩn bị và đã có Manual GT. Trước E1–E4, nhóm cần chốt dùng một video hay toàn bộ set; sau khi chốt thì video/set và time range phải được giữ cố định khi so sánh Detector, Tracker và Counting Strategy.

### Video storage

Do GitHub không phù hợp để lưu nhiều file video dung lượng lớn, toàn bộ video development và evaluation được lưu trên Google Drive.

Google Drive:

https://drive.google.com/drive/folders/1IUK67TbJ1XqrTOKTAeIFHdr-xFzMsTXd?usp=sharing

Cấu trúc:

```text
videos/
├── development/
│   ├── dev_01_detrac_MVI_20011.mp4
│   ├── dev_02_detrac_MVI_20012.mp4
│   ├── dev_03_detrac_MVI_20034.mp4
│   ├── dev_04_4_corners.mp4
│   └── dev_05_1_line.mp4
│
└── evaluation/
    ├── eval_01_detrac_40131_40141.mp4
    ├── eval_02.mp4
    └── eval_03.mp4
```

Repository GitHub chỉ giữ cấu trúc thư mục và source code; video được tải từ Google Drive khi cần chạy project.

Tóm lại:

```text
development → dùng để xây và chỉnh hệ thống
evaluation  → dùng để đo kết quả chính thức
```

### `data/ground_truth/`

Chứa Manual Ground Truth và line config dùng chung cho development/evaluation.

Cấu trúc hiện tại gồm:

```text
data/ground_truth/
├── line_configs/                         # source of truth cho START/END
├── *_manual_counts_by_line.csv           # audit theo từng line
├── *_manual_counts.csv                   # aggregate theo interval đã chọn
├── *_manual_counts_by_10s.csv            # aggregate 10s khi dùng interval mặc định
└── *_manual_counts_by_minute.csv         # aggregate theo phút
```

Ba evaluation video hiện đã có GT đầy đủ: `eval_01_detrac_40131_40141`, `eval_02`, `eval_03`. File `*_manual_count_progress.json` là state local để resume và được `.gitignore` bỏ qua.

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

Chứa tool tạo Ground Truth, code đánh giá và bước tổng hợp nhiều experiment.

```text
evaluation/
├── manual_count_tool.py
├── evaluate_counting.py
├── build_results_summary.py
├── experiments_manifest_template.csv
└── results_summary.csv                  # sinh khi có kết quả official
```

- `manual_count_tool.py`: setup line + manual count + resume.
- `evaluate_counting.py`: so một System Statistics với Manual GT.
- `build_results_summary.py`: gom nhiều `*_evaluation_summary.csv` theo manifest.

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

Chứa tài liệu nội bộ và hướng dẫn vận hành.

```text
docs/
├── PROJECT_PIPELINE.md
├── TEAM_TASKS.md
├── PROJECT_STRUCTURE.md
├── EVALUATION_PROTOCOL.md
├── MANUAL_GROUND_TRUTH_TOOL.md
├── STATISTICS_EVALUATION_USAGE.md
└── development_protocols/              # tài liệu trực quan dev + eval
```

- `PROJECT_PIPELINE.md`, `TEAM_TASKS.md`: tài liệu kế hoạch/phân công gốc của nhóm.
- `EVALUATION_PROTOCOL.md`: protocol hiện hành cho Manual GT và official evaluation.
- `MANUAL_GROUND_TRUTH_TOOL.md`: cách tạo/resume GT.
- `STATISTICS_EVALUATION_USAGE.md`: contract events → statistics → evaluation → summary.
- `development_protocols/`: Markdown và ảnh để người đọc xem line trực quan; config chạy thật nằm trong `data/ground_truth/line_configs/`.

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
eval_01_detrac_40131_40141.mp4
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
