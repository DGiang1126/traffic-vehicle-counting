# Multi-line Vehicle Counting – E4

## Phạm vi đã hoàn thành

- Đếm xe bằng một hoặc nhiều đoạn thẳng hữu hạn.
- Mỗi xe chỉ được đếm một lần trên cùng một line, nhưng có thể được đếm ở các line khác nhau.
- Mỗi line có tên riêng, tên hướng IN và tên hướng OUT do người dùng đặt.
- Hướng IN/OUT được mô tả bằng vector vuông góc với line; thứ tự điểm start/end không mang ý nghĩa nghiệp vụ.
- Streamlit hiển thị trước line, mũi tên xanh IN và mũi tên cam OUT; nút **Đảo chiều IN ↔ OUT** giúp cấu hình trực quan.
- Streamlit cho chọn riêng **Single line/Multi-line** và nhấp trực tiếp hai điểm START/END trên frame; nhập tọa độ thủ công vẫn được giữ làm phương án dự phòng.
- Video kết quả có bounding box, track ID, line, mũi tên hướng và thống kê trực tiếp.
- Sau khi xử lý, chương trình tự thử chuyển video sang H.264 bằng FFmpeg để Streamlit/trình duyệt phát được; nếu FFmpeg không hỗ trợ `libx264`, file `mp4v` vẫn được giữ và giao diện báo lý do.
- Xuất raw events CSV, năm bảng thống kê CSV, ba biểu đồ và JSON thông tin lần chạy.
- Không ghi đè run cũ: tên mặc định chứa video/chế độ/thời điểm; tên trùng được tự động thêm `_02`, `_03`... Mỗi run còn lưu snapshot config và fingerprint phiên bản code.
- Có E4 để so sánh cấu hình một line và nhiều line; hỗ trợ ground truth nếu nhóm đã gán nhãn.

`src/traffic_counting.py` chứa trực tiếp pipeline chính: đọc config, YOLO tracking, multi-line counting, ghi video, CSV và metrics. Streamlit và E4 đều sử dụng hàm `run(...)` trong file này. Các file chính là:

- `src/traffic_counting.py`: pipeline chính và public API của dự án.
- `src/app.py`: giao diện Streamlit.
- `src/counting/counter.py`: thuật toán crossing và multi-line.
- `configs/e4_single_line.json`, `configs/e4_multi_line.json`: cấu hình mẫu.
- `experiments/experiment_04_counting.py`: thí nghiệm E4.
- `evaluation/evaluate_counting.py`: đánh giá với ground truth.

## Cài đặt và chạy trên PowerShell

Đứng ở thư mục gốc `traffic-vehicle-counting`:

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe -m pip install -r requirements-streamlit.txt
```

Chạy Streamlit:

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe -m streamlit run src/app.py
```

Chạy pipeline bằng command line:

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe -m src.traffic_counting `
  --video data/videos/traffic.mp4 `
  --config configs/e4_multi_line.json
```

Chạy E4 cho toàn bộ video evaluation, mỗi video tạo một thư mục riêng và không ghi đè nhau:

```powershell
$python = "C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe"
Get-ChildItem .\data\videos\evaluation\*.mp4 | ForEach-Object {
    & $python -m experiments.experiment_04_counting `
      --video $_.FullName `
      --run-id $_.BaseName
}
```

Nếu đã có ground truth cho một video:

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe -m experiments.experiment_04_counting `
  --video data/videos/evaluation/eval_01.mp4 `
  --single-ground-truth data/ground_truth/eval_01_single.csv `
  --multi-ground-truth data/ground_truth/eval_01_multi.csv
```

Ground-truth CSV tối thiểu cần cột `class,direction`. Nếu muốn đánh giá từng line thì cả file dự đoán và ground truth cần thêm cột `line`.

## Output

Mỗi lần chạy dùng `run_name` riêng:

```text
outputs/
├── videos/<run_name>.mp4
├── csv/<run_name>_events.csv
├── csv/<run_name>/
│   ├── summary_by_class.csv
│   ├── summary_by_time.csv
│   ├── summary_by_direction.csv
│   ├── summary_by_line.csv
│   ├── summary_by_line_class_direction.csv
│   ├── totals.json
│   └── run_metrics.json
└── charts/<run_name>/
    ├── count_by_class.png
    ├── traffic_flow_by_time.png
    └── count_by_line.png
```

Trong `events.csv`, cột `line` lưu đúng tên counting line và cột `direction` lưu đúng tên IN/OUT người dùng đã đặt.

## Cách xác định hướng

Khi tâm xe dịch chuyển từ điểm trước sang điểm hiện tại, chương trình tính tích vô hướng giữa vector chuyển động và vector mũi tên IN:

```text
dot = movement_x * in_x + movement_y * in_y
```

- `dot > 0`: xe đi cùng mũi tên xanh nên ghi tên IN.
- `dot < 0`: xe đi ngược mũi tên xanh nên ghi tên OUT.

Vector IN luôn vuông góc với counting line. Người dùng chỉ cần nhìn preview; nếu phía xanh không đúng, bấm **Đảo chiều IN ↔ OUT**.

## Kiểm thử logic

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe -m unittest tests.test_multi_line_counting -v
```
