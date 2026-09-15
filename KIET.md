# KIET — Bàn giao Statistics, Ground Truth & Evaluation

File này tóm tắt phần tui đã làm để người khác có thể tiếp tục chạy pipeline mà không cần hỏi lại từ đầu.

## 1. Phần tui phụ trách

Theo phân công nhóm, phần của tui gồm:

- Manual Ground Truth cho evaluation.
- Statistics từ `*_events.csv`.
- Biểu đồ thống kê.
- Evaluation bằng AE / Accuracy / MAE.
- Tổng hợp nhiều experiment thành `results_summary.csv`.

Luồng bàn giao hiện tại:

```text
Counting/Integration -> *_events.csv -> statistics.py
                                      -> charts
Manual GT ---------------------------> evaluate_counting.py
                                      -> *_evaluation_summary.csv
nhiều summary + manifest ------------> build_results_summary.py
                                      -> evaluation/results_summary.csv
```

`*_events.csv` không do tui sinh trong official run, tui thấy phần này là bên ông Đức Anh sinh ra chứ ta.

## 2. Ground Truth đã chuẩn bị

Tui đã manual-count đủ 3 evaluation video và 5 development:

Ground Truth nằm trong `data/ground_truth/` và line config chuẩn nằm trong:

```text
data/ground_truth/line_configs/
```

Counting rule hiện hành:

```text
nhìn theo START -> END
chỉ count crossing từ phía trái -> phía phải ( nếu muốn 2 chiều thì tạo thêm một line nhưng đổi start với end lại)
counting point = tâm bbox
crossing phải cắt đoạn line hữu hạn
```

Chi tiết và hình minh họa xem `docs/EVALUATION_PROTOCOL.md` và `docs/development_protocols/README.md`.

Trước khi chạy official E1–E4, nhóm vẫn cần chốt dùng một evaluation video hay toàn bộ 3-video set.
## 3. Tool hỡ trợ tạo / chỉnh Ground Truth dễ dàng hơn

Tool chính:

```text
evaluation/manual_count_tool.py
```

Cú pháp:

```powershell
python evaluation/manual_count_tool.py [--video VIDEO] [--interval SECONDS] `
  [--speed SPEED] [--edit-lines] [--setup-only] [--resume] [--restart]
```

Ví dụ chọn video tương tác:

```powershell
python evaluation/manual_count_tool.py
```

Ví dụ mở thẳng `eval_02` và dùng interval 10 giây:

```powershell
python evaluation/manual_count_tool.py `
  --video data/videos/evaluation/eval_02.mp4 `
  --interval 10
```

Tool tự lưu line config, progress và các file GT theo line / interval / minute. Xem hướng dẫn đầy đủ tại `docs/MANUAL_GROUND_TRUTH_TOOL.md`.
## 4. Statistics từ Counting Events

Tool:

```text
src/statistics/statistics.py
```

Cú pháp:

```powershell
python src/statistics/statistics.py EVENTS_CSV `
  [--output-dir DIR] [--interval SECONDS] `
  [--chart-dir DIR] [--no-charts]
```

Input tối thiểu của `*_events.csv`:

```csv
timestamp,class
12.4,motorcycle
14.1,car
```

Các cột `track_id`, `line`, `confidence`, `direction` có thể có thêm; `direction` không bắt buộc.

Ví dụ:

```powershell
python src/statistics/statistics.py outputs/csv/e1_baseline_events.csv --interval 60
```
Output chính:

```text
*_statistics.csv
*_statistics_by_minute.csv
*_statistics_by_line.csv          # nếu input có line
*_statistics_by_direction.csv     # nếu input có direction
outputs/charts/<experiment>_class_counts.png
outputs/charts/<experiment>_traffic_flow.png
```

Lưu ý: tên `*_statistics_by_minute.csv` được giữ vì tương thích cũ; nội dung thật dùng đúng `--interval`. Nếu truyền `--interval 10`, mỗi row là 10 giây.

## 5. Evaluation

Tool:

```text
evaluation/evaluate_counting.py
```

Cú pháp:

```powershell
python evaluation/evaluate_counting.py GROUND_TRUTH_CSV SYSTEM_STATISTICS_CSV `
  [--system-by-time CSV] [--experiment NAME] [--output-dir DIR]
```

Ví dụ:

```powershell
python evaluation/evaluate_counting.py `
  data/ground_truth/eval_01_detrac_40131_40141_manual_counts_by_minute.csv `
  outputs/csv/e1_baseline_statistics.csv `
  --experiment e1_baseline
```
Output:

```text
*_evaluation_summary.csv
*_evaluation_by_class.csv
*_evaluation_by_time.csv
*_evaluation_by_time_class.csv
```

Metric gồm signed error, absolute error, counting accuracy, class-level error và time MAE.

## 6. Tổng hợp nhiều experiment

Tool:

```text
evaluation/build_results_summary.py
```

Cú pháp:

```powershell
python evaluation/build_results_summary.py MANIFEST_CSV [--output OUTPUT_CSV]
```

Template manifest:

```text
evaluation/experiments_manifest_template.csv
```

Ví dụ:

```powershell
Copy-Item evaluation/experiments_manifest_template.csv evaluation/experiments_manifest.csv
python evaluation/build_results_summary.py evaluation/experiments_manifest.csv
```
Mặc định sinh:

```text
evaluation/results_summary.csv
```

Manifest giữ metadata như `experiment`, `report_label`, `detector`, `tracker`, `counting`, `fps` và đường dẫn tới `evaluation_summary`.

## 7. Trạng thái bàn giao hiện tại

Đã code và test:

- `evaluation/manual_count_tool.py`.
- `src/visualization/line_editor.py`.
- `src/statistics/statistics.py`.
- `evaluation/evaluate_counting.py`.
- `evaluation/build_results_summary.py`.
- `class_counts.png` và `traffic_flow.png` sinh tự động từ Statistics.
- 3 evaluation Ground Truth đã hoàn thành.
- Validation synthetic và event schema kiểu Counting đã được thử.

Chưa có official E1–E4 result vì tui vẫn cần `*_events.csv` thật từ Counting/Integration. Khi phía Integration bàn giao events đúng contract, phần Statistics/Evaluation có thể chạy ngay.

FPS trong `results_summary.csv` phải lấy từ experiment/integration runtime; evaluator không tự suy ra FPS từ counting statistics.

## 8. Điểm cần nhóm chốt trước official run

- E1–E4 dùng 1 evaluation video hay cả 3 video.
- Duplicate rule cho multi-line: một physical vehicle được count global một lần hay một lần trên mỗi line.
- Counting của ông Đức Anh phải thống nhất rule `START -> END`, LEFT -> RIGHT với Manual GT trước khi lấy metric official.

Hướng dẫn chi tiết hơn nằm trong `docs/STATISTICS_EVALUATION_USAGE.md` và `docs/EVALUATION_PROTOCOL.md`.
