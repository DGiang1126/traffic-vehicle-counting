# Team Tasks — Traffic Vehicle Counting

> Tài liệu này dùng để chia công việc cho nhóm 4 người, để mọi người biết rõ phần mình phụ trách, output cần làm ra và cách các phần phối hợp với nhau.
>
> Mục tiêu là cả nhóm có thể làm song song, không phải chờ nhau quá nhiều, nhưng đến lúc chạy experiment chính thức thì vẫn dùng cùng một cấu hình và cùng dữ liệu để so sánh công bằng.

---

## 1. Việc chung cả nhóm làm trước

Trước khi chia phần riêng, cả 4 thành viên đều cần:

1. Pull repo mới nhất về máy.
2. Tạo virtual environment.
3. Cài dependencies từ `requirements.txt`.
4. Chạy file starter:

```bash
python src/traffic_counting.py
```

5. Quan sát baseline hiện tại:

- YOLO detect được những xe nào.
- Có miss phương tiện hay không.
- Track ID có ổn định không.
- Counting line hiện tại hoạt động như thế nào.
- Có bị đếm thiếu hoặc đếm trùng không.
- Chart hiện tại hiển thị được gì.

Mục tiêu của bước này là để cả nhóm cùng hiểu:

```text
Baseline hiện tại làm được gì
+
Baseline đang hạn chế ở đâu
```

Sau khi hiểu baseline thì bắt đầu phát triển phần riêng.

---

## 2. Phân công tổng thể

| Thành viên | Phần phụ trách chính | Experiment chính |
|---|---|---|
| Giang | Detection + Integration | E1, E2 |
| Thuận | Tracking | E3 |
| Đức Anh | Counting | E4 |
| Kiệt | Statistics + Evaluation | Hỗ trợ đánh giá E1–E4 |

---

## 3. Nguyên tắc làm việc song song

Các experiment khi báo cáo sẽ đi theo thứ tự:

```text
E1 Baseline
→ E2 Detector
→ E3 Tracker
→ E4 Counting
→ Final System
```

Tuy nhiên, lúc code **không cần chờ experiment trước hoàn thành rồi mới bắt đầu phần sau**.

Trong giai đoạn phát triển, nhóm dùng một cấu hình tạm để các module có thể làm song song:

```text
YOLOv8s
+
ByteTrack
+
1 counting line
```

Cấu hình này dùng để:

- Viết code.
- Debug.
- Kiểm tra input/output giữa các module.
- Kiểm tra phần đang làm đã chạy đúng chưa.

Kết quả ở giai đoạn này chỉ là **development test**, chưa dùng để kết luận model/tracker/counting nào tốt nhất.

Khi một experiment trước đã chốt được kết quả chính thức, ví dụ `Best Detector` hoặc `Best Tracker`, phần sau chỉ cần thay cấu hình tạm bằng cấu hình đã chốt rồi **rerun experiment chính thức**.

Ví dụ:

```text
Development Tracking:
YOLOv8s + ByteTrack / BoT-SORT

Official E3:
Best Detector + ByteTrack
vs
Best Detector + BoT-SORT
```

Khi chạy official experiment, phải giữ cố định:

- Cùng evaluation video.
- Cùng khoảng thời gian.
- Cùng ground truth.
- Cùng vehicle classes.
- Cùng metric.
- Chỉ thay đúng thành phần đang cần so sánh.

---

# 4. Giang — Detection + Integration

## Branch

```text
feature/detection
```

## Nhiệm vụ chính

Phụ trách phần:

```text
Video / Frame
→ YOLO
→ Bounding Box
→ Class
→ Confidence
```

Tách logic detection khỏi file starter để dễ thay model và cấu hình.

File dự kiến:

```text
src/detection/detector.py
```

## Các model cần thử

```text
YOLOv8n
YOLOv8s
YOLOv8m
```

Các class chính:

```text
motorcycle
car
bus
truck
bicycle
```

## Output của detector

Output thống nhất dạng:

```python
{
    "bbox": [x1, y1, x2, y2],
    "class": "motorcycle",
    "confidence": 0.87
}
```

---

## Experiment 1 — Baseline

Cấu hình baseline:

```text
YOLOv8n
+
ByteTrack
+
1 counting line
```

Mục tiêu:

- Xác nhận toàn bộ pipeline chạy được.
- Có một mốc ban đầu để so sánh với các cải tiến sau.
- Ghi lại các lỗi ban đầu.

---

## Experiment 2 — Detector Comparison

So sánh:

```text
YOLOv8n
vs
YOLOv8s
vs
YOLOv8m
```

Giữ cố định:

```text
Cùng evaluation video
Cùng tracker
Cùng counting logic
Cùng ground truth
```

Các chỉ số cần so:

```text
Detection quality
Missed vehicles
Counting accuracy
FPS
```

Nếu có annotation detection thì bổ sung:

```text
Precision
Recall
mAP50
mAP50-95
```

## Output cần bàn giao

Code:

```text
src/detection/detector.py
experiments/experiment_02_detector.py
```

Kết quả:

```text
Bảng so sánh YOLOv8n / YOLOv8s / YOLOv8m
→ Chọn Best Detector
```

---

## Phần Integration

Ngoài Detection, phần Integration gồm:

- Chốt evaluation video chung.
- Chốt vehicle classes.
- Chốt format dữ liệu giữa các module.
- Review Pull Request trước khi merge.
- Kiểm tra Detection → Tracking → Counting → Statistics có nối được với nhau không.
- Chốt cấu hình Final System sau các experiment.

Mục tiêu là đảm bảo khi ghép các phần lại thì toàn bộ pipeline chạy được.

---

# 5. Thuận — Multi-Object Tracking

## Branch

```text
feature/tracking
```

## Nhiệm vụ chính

Phụ trách:

```text
Detection
→ Tracker
→ Track ID
```

File dự kiến:

```text
src/tracking/tracker.py
```

Hai tracker cần triển khai:

```text
ByteTrack
BoT-SORT
```

Mục tiêu là giữ cùng một ID cho cùng một phương tiện qua nhiều frame.

Ví dụ:

```text
Frame 1 → motorcycle #12
Frame 2 → motorcycle #12
Frame 3 → motorcycle #12
```

Những vấn đề cần quan sát:

```text
Track ID có ổn định không?
Có mất track không?
Có ID switch không?
Một xe có bị chia thành nhiều ID không?
```

## Output tracking

```python
{
    "track_id": 12,
    "bbox": [x1, y1, x2, y2],
    "class": "motorcycle",
    "confidence": 0.87
}
```

---

## Experiment 3 — Tracker Comparison

Khi chạy chính thức, giữ cố định:

```text
Detector
Video
conf
imgsz
Counting logic
Counting line
```

Chỉ thay:

```text
ByteTrack
vs
BoT-SORT
```

Các chỉ số cần so:

```text
Counting accuracy
ID switch
Lost track
Track fragmentation
Duplicate count
FPS
```

## Output cần bàn giao

Code:

```text
src/tracking/tracker.py
experiments/experiment_03_tracker.py
```

Kết quả:

```text
Bảng ByteTrack vs BoT-SORT
→ Chọn Best Tracker
```

Nếu có thể, lưu thêm hình/video minh họa:

```text
Track ổn định
ID switch
Lost track
```

để dùng cho slide và báo cáo.

---

# 6. Đức Anh — Counting Line / Zone / Direction

## Branch

```text
feature/counting
```

## Nhiệm vụ chính

Phần này trả lời câu hỏi:

> Khi nào một track được tính là một phương tiện đã đi qua?

File dự kiến:

```text
src/counting/counter.py
```

---

## Phase 1 — Line Crossing

Baseline hiện tại dùng:

```text
1 horizontal counting line
```

Ví dụ:

```text
Frame trước:
cy = 250

line_y = 300

Frame hiện tại:
cy = 320
```

Khi đó:

```text
250 < 300 <= 320
```

→ tạo counting event.

---

## Phase 2 — Chống đếm trùng

Nếu một `track_id` đã được đếm tại một line thì không được count lại sai.

Ví dụ:

```text
track_id #12
đã đi qua line_1
→ không count lại line_1
```

---

## Phase 3 — Direction

Ít nhất thử:

```text
top_to_bottom
bottom_to_top
```

Nếu video phù hợp có thể thêm:

```text
left_to_right
right_to_left
```

---

## Phase 4 — Multiple Lines / Zones

Vì video giao lộ có thể không phù hợp với một line ngang duy nhất, có thể thử:

```text
single line
vs
multiple lines
```

hoặc:

```text
line
vs
zone
```

Không bắt buộc phải làm tất cả nếu video không phù hợp.

---

## Counting Event Output

Mỗi lần xe đi qua line/zone thì trả về:

```python
{
    "timestamp": 83.2,
    "track_id": 12,
    "class": "motorcycle",
    "line": "line_1",
    "direction": "top_to_bottom",
    "confidence": 0.87
}
```

Output này sẽ được dùng cho Statistics và Evaluation.

---

## Experiment 4 — Counting Strategy

Khi đã có:

```text
Best Detector
+
Best Tracker
```

chạy lại chính thức:

```text
Single Line
vs
Multiple Lines / Zone
```

Nếu làm direction thì có thể so:

```text
Without Direction
vs
Direction-aware Counting
```

Các chỉ số cần so:

```text
Counting accuracy
Missed crossing
Duplicate count
Wrong direction
```

## Output cần bàn giao

Code:

```text
src/counting/counter.py
experiments/experiment_04_counting.py
```

Kết quả:

```text
Bảng so sánh các counting strategy
→ Chọn Best Counting Strategy
```

Video demo nên hiển thị được:

```text
Bounding box
Track ID
Class
Counting line / zone
Direction
Count
```

---

# 7. Kiệt — Statistics + Ground Truth + Evaluation

## Branch

```text
feature/evaluation
```

## Mục tiêu của phần này

Phần này dùng để kiểm tra **hệ thống đếm có đúng hay không** và biến output của hệ thống thành số liệu có thể đưa vào báo cáo.

Sau khi pipeline chạy xong, phần này cần trả lời được:

```text
Hệ thống đếm bao nhiêu xe?
Manual đếm bao nhiêu xe?
Sai bao nhiêu xe?
Accuracy bao nhiêu?
Sai nhiều ở class nào?
Lưu lượng xe thay đổi theo thời gian như thế nào?
Experiment nào cho kết quả tốt nhất?
```

Có thể bắt đầu ngay từ:

```text
Code starter
+
Evaluation video
```

không cần chờ Detection, Tracking hay Counting hoàn thiện.

---

## Bước 1 — Chốt cách đếm cho Evaluation Video

Trước khi manual count, cần thống nhất với nhóm:

```text
Evaluation video nào?
Đoạn thời gian nào được dùng?
Đếm những class nào?
Counting line / zone nào?
Đếm một chiều hay hai chiều?
```

Ví dụ:

```text
Video: traffic.mp4
Thời gian: 00:00 → 02:00
Classes: motorcycle, car, bus, truck, bicycle
Counting rule: xe được tính khi đi qua line_1
```

Manual Ground Truth và System Result phải dùng **cùng quy tắc đếm**.

Nếu hệ thống đếm xe đi qua `line_1` thì manual cũng phải xem xe nào thực sự đi qua `line_1`, không phải đếm tất cả xe xuất hiện trong video.

---

## Bước 2 — Tạo Manual Ground Truth

Xem evaluation video và đếm thủ công số xe thực tế đi qua vị trí cần đếm.

Nên chia theo từng phút để sau này tính được traffic flow.

Ví dụ:

```text
Minute 0–1
motorcycle = 35
car = 12
bus = 1
truck = 3
bicycle = 0

Minute 1–2
motorcycle = 42
car = 15
bus = 2
truck = 4
bicycle = 1
```

Lưu tại:

```text
data/ground_truth/manual_counts.csv
```

Ví dụ:

```csv
minute,motorcycle,car,bus,truck,bicycle
0,35,12,1,3,0
1,42,15,2,4,1
```

Đây là dữ liệu chuẩn để so với kết quả hệ thống.

Với E1, E2 và E3 nếu cùng dùng một counting rule thì dùng chung Ground Truth này.

Nếu E4 thay đổi line/zone đến mức **định nghĩa xe nào được tính cũng thay đổi**, Ground Truth phải được kiểm tra hoặc tạo lại cho đúng với counting strategy đó.

---

## Bước 3 — Lưu từng Counting Event của hệ thống

Code starter hiện tại chủ yếu giữ tổng count trong lúc chạy.

Cần bổ sung phần lưu lại **mỗi lần hệ thống count một xe**.

Ví dụ khi xe `track_id = 15` đi qua line:

```python
{
    "timestamp": 12.4,
    "track_id": 15,
    "class": "motorcycle",
    "line": "line_1",
    "direction": "top_to_bottom",
    "confidence": 0.87
}
```

Các event được lưu thành CSV:

```csv
timestamp,track_id,class,line,direction,confidence
12.4,15,motorcycle,line_1,top_to_bottom,0.87
14.1,18,car,line_1,top_to_bottom,0.92
20.8,24,motorcycle,line_1,bottom_to_top,0.81
```

Không nên ghi đè cùng một `events.csv` cho tất cả experiment.

Nên đặt tên theo experiment, ví dụ:

```text
outputs/csv/e1_baseline_events.csv

outputs/csv/e2_yolov8n_events.csv
outputs/csv/e2_yolov8s_events.csv
outputs/csv/e2_yolov8m_events.csv

outputs/csv/e3_bytetrack_events.csv
outputs/csv/e3_botsort_events.csv

outputs/csv/e4_single_line_events.csv
outputs/csv/e4_multi_line_events.csv
```

Như vậy sau này có thể xem lại và so sánh từng cấu hình.

---

## Bước 4 — Viết Statistics

Tạo:

```text
src/statistics/statistics.py
```

File này đọc một file `*_events.csv` và tổng hợp:

```text
Tổng số xe
Số xe theo class
Số xe theo phút
Số xe theo direction (nếu có)
```

Ví dụ từ:

```csv
timestamp,track_id,class
12.4,15,motorcycle
14.1,18,car
20.8,24,motorcycle
```

phải tổng hợp được:

```text
motorcycle = 2
car = 1
total = 3
```

và nếu chia theo phút:

```text
Minute 0 → 48 xe
Minute 1 → 62 xe
```

Output có thể lưu tại:

```text
outputs/csv/e1_baseline_statistics.csv
outputs/csv/e2_yolov8s_statistics.csv
...
```

---

## Bước 5 — Vẽ biểu đồ

Từ kết quả Statistics, tạo ít nhất:

### Bar chart — số xe theo class

Ví dụ:

```text
motorcycle = 77
car = 27
bus = 3
truck = 7
bicycle = 1
```

Output:

```text
outputs/charts/class_counts.png
```

### Line chart — traffic flow theo thời gian

Ví dụ:

```text
00:00–01:00 → 48 xe
01:00–02:00 → 62 xe
```

Output:

```text
outputs/charts/traffic_flow.png
```

Starter đã có bar chart mẫu, có thể tận dụng phần đó rồi tách thành code Statistics/Visualization rõ ràng hơn.

---

## Bước 6 — Viết Evaluation Script

Tạo:

```text
evaluation/evaluate_counting.py
```

Script này nhận:

```text
Manual Ground Truth
+
System Statistics
```

rồi tính các metric.

### Absolute Error

```text
AE = |System Count - Manual Count|
```

Ví dụ:

```text
Manual = 100
System = 94

AE = 6
```

### Counting Accuracy

```text
Accuracy = 1 - |System Count - Manual Count| / Manual Count
```

Ví dụ:

```text
Manual = 100
System = 94

Accuracy = 94%
```

### MAE

Nếu chia video theo nhiều khoảng thời gian thì tính thêm MAE giữa System Count và Manual Count của các khoảng đó.

Evaluation nên cho kết quả theo:

```text
Từng class
+
Toàn bộ video
```

Ví dụ output:

```text
motorcycle:
Manual = 100
System = 94
AE = 6
Accuracy = 94%

car:
Manual = 30
System = 28
AE = 2
Accuracy = 93.33%
```

---

## Bước 7 — Dùng cùng Evaluation Script cho E1–E4

Sau khi phần evaluation đã chạy được với baseline thì không cần viết lại cho từng experiment.

Các experiment khác chỉ cần đưa file kết quả vào cùng pipeline Evaluation.

Luồng xử lý:

```text
E1 Baseline Events
→ Statistics
→ Evaluation
→ Metrics

E2 YOLOv8n Events
→ Statistics
→ Evaluation
→ Metrics

E2 YOLOv8s Events
→ Statistics
→ Evaluation
→ Metrics

E2 YOLOv8m Events
→ Statistics
→ Evaluation
→ Metrics

E3 ByteTrack Events
→ Statistics
→ Evaluation
→ Metrics

E3 BoT-SORT Events
→ Statistics
→ Evaluation
→ Metrics

E4 Counting Strategy Events
→ Statistics
→ Evaluation
→ Metrics
```

Tức là phần code evaluation chỉ xây một lần, sau đó thay input của từng experiment.

---

## Bước 8 — Tổng hợp kết quả cuối cùng

Tạo một file tổng hợp, ví dụ:

```text
evaluation/results_summary.csv
```

Nội dung:

```csv
experiment,detector,tracker,counting,accuracy,fps
E1,YOLOv8n,ByteTrack,1_line,...
E2_n,YOLOv8n,ByteTrack,1_line,...
E2_s,YOLOv8s,ByteTrack,1_line,...
E2_m,YOLOv8m,ByteTrack,1_line,...
E3_byte,BestDetector,ByteTrack,1_line,...
E3_bot,BestDetector,BoT-SORT,1_line,...
E4_single,BestDetector,BestTracker,1_line,...
E4_multi,BestDetector,BestTracker,multi_line,...
```

Từ đó tạo bảng kết quả cho slide/report:

| Experiment | Detector | Tracker | Counting | Accuracy | FPS |
|---|---|---|---|---:|---:|
| E1 | YOLOv8n | ByteTrack | 1 line | ... | ... |
| E2 | Best Detector | ByteTrack | 1 line | ... | ... |
| E3 | Best Detector | Best Tracker | 1 line | ... | ... |
| E4 | Best Detector | Best Tracker | Best Counting | ... | ... |

---

## Output cần bàn giao

Code:

```text
src/statistics/statistics.py
evaluation/evaluate_counting.py
```

Ground Truth:

```text
data/ground_truth/manual_counts.csv
```

Kết quả từng experiment:

```text
outputs/csv/*_events.csv
outputs/csv/*_statistics.csv
```

Charts:

```text
outputs/charts/class_counts.png
outputs/charts/traffic_flow.png
```

Kết quả tổng hợp:

```text
evaluation/results_summary.csv
```

## Khi phần này được xem là hoàn thành

```text
Có Manual Ground Truth đúng với quy tắc đếm
+
Baseline xuất được Counting Events CSV
+
Statistics đọc được Events CSV
+
Có thống kê theo class và theo thời gian
+
Có bar chart và traffic flow chart
+
Evaluation Script so được System với Manual
+
Tính được AE / Counting Accuracy / MAE
+
Có thể dùng cùng script để đánh giá E1–E4
+
Có bảng tổng hợp kết quả cuối
```

---

# 8. Workflow tổng thể của nhóm

## Giai đoạn 1 — Development song song

```text
                    Starter Baseline
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
      Detection        Tracking        Counting
       Giang            Thuận          Đức Anh
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                 Statistics/Evaluation
                        Kiệt
```

Cấu hình development tạm:

```text
YOLOv8s + ByteTrack + 1 line
```

Mục tiêu:

```text
Tất cả module chạy được
+
Input/Output thống nhất
```

---

## Giai đoạn 2 — Official Experiments

```text
E1 — Baseline
YOLOv8n + ByteTrack + 1 Line
        ↓
E2 — Detector Comparison
YOLOv8n / YOLOv8s / YOLOv8m
        ↓
Best Detector
        ↓
E3 — Tracker Comparison
ByteTrack / BoT-SORT
        ↓
Best Tracker
        ↓
E4 — Counting Strategy
Single Line / Multiple Lines / Zone
        ↓
Best Counting Strategy
```

---

## Giai đoạn 3 — Final Integration

```text
Best Detector
+
Best Tracker
+
Best Counting Strategy
+
Statistics
+
Evaluation
        ↓
FINAL SYSTEM
```

Sau đó chạy lại Final System trên cùng evaluation video để lấy kết quả cuối cùng.

---

## Giai đoạn 4 — Optional

Nếu còn thời gian và pretrained YOLO vẫn yếu với giao thông Việt Nam:

```text
E5 — Fine-tune YOLOv8
```

So sánh:

```text
Pretrained YOLO
vs
Fine-tuned YOLO
```

Phần fine-tune chỉ làm sau khi pipeline chính đã chạy ổn.

---

# 9. Branch của từng người

```text
feature/detection
feature/tracking
feature/counting
feature/evaluation
```

Không push trực tiếp vào `main`.

Workflow chung:

```text
git pull
→ checkout branch cá nhân
→ code
→ test
→ commit
→ push branch
→ Pull Request
→ review
→ merge main
```

---

# 10. Quy tắc code chung

Mỗi người ưu tiên code trong folder của phần mình:

```text
Detection
→ src/detection/

Tracking
→ src/tracking/

Counting
→ src/counting/

Statistics / Evaluation
→ src/statistics/
→ evaluation/
```

Hạn chế việc cả 4 người cùng sửa trực tiếp:

```text
src/traffic_counting.py
```

File này dùng để integration các module sau khi từng phần đã ổn định.

---

# 11. Khi nào phần của mỗi người được xem là xong?

## Detection

```text
Có detector module
+
YOLOv8n/s/m đều chạy
+
Có bảng comparison
+
Chọn được Best Detector
```

## Tracking

```text
ByteTrack chạy
+
BoT-SORT chạy
+
Có bảng comparison
+
Chọn được Best Tracker
```

## Counting

```text
Crossing hoạt động
+
Không đếm trùng
+
Có line/zone phù hợp
+
Direction nếu triển khai
+
Có bảng comparison
```

## Statistics & Evaluation

```text
Có Manual Ground Truth
+
Events CSV
+
Statistics CSV
+
Charts
+
Evaluation Script
+
Bảng tổng hợp Experiment
```
