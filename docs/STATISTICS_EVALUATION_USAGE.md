# Statistics & Evaluation — Usage Guide

Tài liệu này mô tả cách các phần của nhóm phối hợp với module Statistics/Evaluation của Kiệt.

## 1. Ownership và luồng dữ liệu

```text
Detection -> Tracking -> Counting/Integration -> *_events.csv
                                         |
                                         v
                                  statistics.py
                                         |
                          *_statistics*.csv
                                         |
Manual Ground Truth ----------------> evaluate_counting.py
                                         |
                               *_evaluation*.csv
```

- `*_events.csv` do Counting/Integration sinh ra; Statistics chỉ đọc, không tự tạo event.
- Manual Ground Truth do Evaluation tạo bằng manual count.
- `statistics.py` tổng hợp event thành số liệu hệ thống.
- `evaluate_counting.py` so System Statistics với Manual Ground Truth.

## 2. Contract đầu vào từ Counting

Tối thiểu mỗi `*_events.csv` cần có:

```csv
timestamp,class
12.4,motorcycle
14.1,car
```
Các cột nên có đầy đủ khi pipeline hỗ trợ:

```csv
timestamp,track_id,class,line,confidence
12.4,15,motorcycle,line_1,0.87
```

Nếu branch Counting hiện vẫn xuất `direction`, Statistics vẫn đọc được cột này và tạo thêm thống kê theo direction. `direction` không bắt buộc đối với Statistics/Evaluation hiện tại.

Vehicle classes thống nhất:

```text
motorcycle, car, bus, truck, bicycle
```

Không đưa class khác vào experiment chính thức nếu chưa thống nhất với nhóm.

## 3. Chạy Statistics

Cú pháp tổng quát:

```powershell
python src/statistics/statistics.py EVENTS_CSV `
  [--output-dir OUTPUT_DIR] [--interval SECONDS] `
  [--chart-dir CHART_DIR] [--no-charts]
```

Ví dụ chạy mặc định:

```powershell
python src/statistics/statistics.py outputs/csv/e1_baseline_events.csv
```

Ví dụ đưa CSV và chart vào folder riêng:

```powershell
python src/statistics/statistics.py outputs/csv/e1_baseline_events.csv `
  --output-dir outputs/csv/e1_baseline `
  --chart-dir outputs/charts/e1_baseline `
  --interval 60
```

Mặc định chia theo phút (`60s`). Output:

```text
outputs/csv/e1_baseline_statistics.csv
outputs/csv/e1_baseline_statistics_by_minute.csv
outputs/csv/e1_baseline_statistics_by_line.csv        # nếu có line
outputs/csv/e1_baseline_statistics_by_direction.csv   # nếu có direction
```
Nếu Ground Truth đang chia 10 giây thì Statistics cũng phải dùng 10 giây:

```powershell
python src/statistics/statistics.py outputs/csv/e1_baseline_events.csv --interval 10
```

Lưu ý: tên file lịch sử vẫn là `*_statistics_by_minute.csv`, nhưng nội dung dùng đúng `--interval` đã truyền. Với `--interval 10`, mỗi row là một bin 10 giây, không phải một phút.

Không tự xóa duplicate event trong Statistics. Nếu Counting sinh duplicate thì đó là hành vi cần được Evaluation phản ánh.

## 4. Manual Ground Truth

Format khuyến nghị cho official evaluation là theo phút:

```csv
minute,motorcycle,car,bus,truck,bicycle
0,35,12,1,3,0
1,42,15,2,4,1
```

Tool development hiện cũng hỗ trợ format theo khoảng thời gian:

```csv
start_sec,end_sec,motorcycle,car,bus,truck,bicycle
0,10,0,10,0,0,0
10,20,0,8,0,1,0
```

System và Manual phải dùng cùng video, time range, classes và counting rule.

## 5. Chạy Evaluation

Cú pháp tổng quát:

```powershell
python evaluation/evaluate_counting.py GROUND_TRUTH_CSV SYSTEM_STATISTICS_CSV `
  [--system-by-time SYSTEM_BY_TIME_CSV] `
  [--experiment NAME] [--output-dir OUTPUT_DIR]
```

Ví dụ official theo phút:

```powershell
python evaluation/evaluate_counting.py `
  data/ground_truth/eval_01_detrac_40131_40141_manual_counts_by_minute.csv `
  outputs/csv/e1_baseline_statistics.csv `
  --experiment e1_baseline
```

Ví dụ chỉ định luôn file system theo thời gian và folder output:

```powershell
python evaluation/evaluate_counting.py `
  data/ground_truth/eval_02_manual_counts_by_minute.csv `
  outputs/csv/e2_yolov8s_statistics.csv `
  --system-by-time outputs/csv/e2_yolov8s_statistics_by_minute.csv `
  --experiment e2_yolov8s `
  --output-dir outputs/csv/evaluation/e2_yolov8s
```
Evaluation tự tìm file `*_statistics_by_minute.csv` tương ứng nếu có.

Các output chính:

```text
*_evaluation_summary.csv
*_evaluation_by_class.csv
*_evaluation_by_time.csv
*_evaluation_by_time_class.csv
```

Metric gồm: signed error, absolute error, counting accuracy và MAE theo thời gian. Đánh giá theo class giúp phát hiện trường hợp tổng xe đúng nhưng class bị nhầm.

Nếu time-bin của GT và Statistics không khớp, script sẽ dừng và yêu cầu chạy lại `statistics.py` với cùng interval thay vì so sai dữ liệu.

## 6. Counting line contract

Tất cả line config của development và evaluation hiện nằm chung tại:

```text
data/ground_truth/line_configs/
```

Mỗi video có file `<video_stem>_lines.json`. Mỗi line chỉ có ID trung tính (`line_1`, `line_2`, ...), `START`, `END` và normalized coordinates. Nhìn theo `START -> END`, protocol hiện tại chỉ count crossing từ phía trái sang phía phải của đoạn line hữu hạn.

Nếu Counting team đổi rule (hai chiều, zone, hoặc định nghĩa multi-line khác), phải thống nhất lại trước khi tạo/chạy Ground Truth chính thức. E4 chỉ dùng lại GT cũ khi định nghĩa "xe nào được tính" không thay đổi.

## 7. Checklist trước official experiment

- Cùng evaluation video và time range.
- Cùng 5 vehicle classes.
- Cùng counting line/zone và crossing rule.
- Counting/Integration lưu event riêng cho từng experiment.
- Statistics dùng cùng interval với GT.
- Không so các experiment dùng Ground Truth không tương thích.

## 8. Tạo Ground Truth cho video bất kỳ

Dùng một tool duy nhất:

```powershell
python evaluation/manual_count_tool.py
```

Tool tự liệt kê video trong `data/videos/`, cho chọn số line, nhập/chỉnh START-END, rồi chuyển thẳng sang manual count. Với interval mặc định 10 giây, nó tự xuất đồng thời `*_by_line.csv`, `*_manual_counts.csv`, `*_by_10s.csv`, `*_by_minute.csv` và progress JSON.

Hướng dẫn đầy đủ: `docs/MANUAL_GROUND_TRUTH_TOOL.md`.

## 9. Charts sinh tự động từ Statistics

Mặc định `statistics.py` tạo luôn hai biểu đồ trong `outputs/charts/`:

```text
<experiment>_class_counts.png
<experiment>_traffic_flow.png
```

Đồng thời cập nhật hai alias theo đúng tên deliverable của task:

```text
class_counts.png
traffic_flow.png
```

Các file có prefix experiment được giữ lại để E1–E4 không ghi đè nhau; hai alias luôn trỏ tới kết quả của lần chạy gần nhất. Nếu chỉ cần CSV, dùng `--no-charts`. Có thể đổi thư mục ảnh bằng `--chart-dir <path>`.

## 10. Tổng hợp nhiều experiment

Sau khi mỗi run đã có `*_evaluation_summary.csv`, dùng:

```text
evaluation/build_results_summary.py
```

Cú pháp tổng quát:

```powershell
python evaluation/build_results_summary.py MANIFEST_CSV [--output OUTPUT_CSV]
```

Input là manifest CSV. Template có sẵn tại:

```text
evaluation/experiments_manifest_template.csv
```

Ví dụ manifest tối thiểu:

```csv
experiment,report_label,detector,tracker,counting,fps,evaluation_summary
e1_baseline,E1,YOLOv8n,ByteTrack,single_line,28.4,outputs/csv/e1_baseline_evaluation_summary.csv
e2_yolov8s,E2_s,YOLOv8s,ByteTrack,single_line,22.1,outputs/csv/e2_yolov8s_evaluation_summary.csv
```

Khi có kết quả official, copy template thành `evaluation/experiments_manifest.csv`, điền đúng metadata/FPS và giữ `experiment` khớp với từng file summary.

`experiment` là ID kỹ thuật và phải khớp cột `experiment` bên trong summary. `report_label` là nhãn tùy chọn để trình bày như `E1`, `E2_s`. Các cột bắt buộc là `experiment`, `detector`, `tracker`, `counting`, `evaluation_summary`; `fps` và metadata khác có thể thêm hoặc để trống.
Chạy:

```powershell
python evaluation/build_results_summary.py evaluation/experiments_manifest.csv
```

Mặc định output:

```text
evaluation/results_summary.csv
```

Có thể đổi bằng `--output <path>`. Đường dẫn `evaluation_summary` có thể ghi relative từ project root (`outputs/...`) hoặc relative từ vị trí manifest. Script chặn duplicate experiment, summary trỏ nhầm experiment, file thiếu, schema thiếu và FPS không hợp lệ. Các cột metadata bổ sung trong manifest được giữ lại trong bảng tổng hợp.

Hiện `build_results_summary.py` gom **mỗi row manifest từ một `*_evaluation_summary.csv`**; nó chưa tự aggregate nhiều video thành một experiment. Nếu official E1–E4 dùng cả 3 evaluation video, nhóm phải chốt cách aggregate trước rồi mới mở rộng bước tổng hợp này.
