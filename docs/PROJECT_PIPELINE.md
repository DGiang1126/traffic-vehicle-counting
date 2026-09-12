# Traffic Vehicle Counting — Project Pipeline

> Tài liệu nội bộ để cả nhóm thống nhất mục tiêu, pipeline và các thực nghiệm của đề tài.  
> Nội dung sẽ tiếp tục được bổ sung trong quá trình triển khai.

## 1. Mục tiêu đề tài

Nhóm xây dựng hệ thống **đếm và phân loại phương tiện giao thông từ video camera giám sát**.

Hệ thống cần thực hiện:

```text
Video giao thông
→ Vehicle Detection
→ Multi-Object Tracking
→ Counting Line / Zone
→ Counting Event
→ Statistics
→ Evaluation
```

Các loại phương tiện chính:

```text
motorcycle
car
bus
truck
bicycle
```

Output cuối cùng gồm:

- Video có bounding box, class và track ID.
- Tổng số phương tiện.
- Số lượng theo từng loại xe.
- Số lượng theo thời gian.
- Có thể đếm theo hướng di chuyển.
- File CSV lưu các sự kiện đếm.
- Biểu đồ thống kê.
- Kết quả so sánh với số đếm thủ công.

---

## 2. Pipeline chính

### Step 1 — Input Video

Đọc video giao thông bằng OpenCV.

Video dùng trong nhóm gồm:

```text
Development video
→ dùng để code, test, chỉnh hệ thống.

Evaluation video
→ dùng cố định để đánh giá cuối cùng.
```

Evaluation video nên dùng cùng một đoạn cho tất cả experiment để kết quả có thể so sánh công bằng.

---

### Step 2 — Vehicle Detection

Sử dụng **Ultralytics YOLOv8** để phát hiện phương tiện trong từng frame.

Output của detector:

```text
Bounding Box
Class
Confidence
```

Ví dụ:

```text
motorcycle
bbox = [x1, y1, x2, y2]
confidence = 0.87
```

Các model dự kiến thử:

```text
YOLOv8n
YOLOv8s
YOLOv8m
```

Không chọn model chỉ dựa vào cảm tính. Nhóm sẽ chạy experiment để chọn model phù hợp nhất.

---

### Step 3 — Multi-Object Tracking

Detection chỉ nhận diện object ở từng frame riêng.

Tracking giúp giữ cùng một ID cho cùng một phương tiện qua nhiều frame.

Ví dụ:

```text
Frame 1 → motorcycle #12
Frame 2 → motorcycle #12
Frame 3 → motorcycle #12
```

Tracker dự kiến:

```text
ByteTrack
BoT-SORT
```

Mục đích:

- Giữ `track_id`.
- Tránh đếm cùng một xe nhiều lần.
- Theo dõi hướng di chuyển.
- Phát hiện các lỗi như mất track hoặc ID switch.

---

### Step 4 — Counting Line / Zone

Sau khi có `track_id`, hệ thống kiểm tra phương tiện có đi qua vùng cần đếm hay không.

Baseline hiện tại:

```text
1 horizontal counting line
```

Sau này có thể cải tiến thành:

```text
multiple counting lines
hoặc
counting zones
```

Khi tâm của phương tiện đi từ một phía sang phía còn lại của line thì tạo một **counting event**.

Ví dụ:

```text
track_id = 12
class = motorcycle

Frame trước:
cy = 250

Counting line:
y = 300

Frame sau:
cy = 320

→ phương tiện đã crossing line
→ count +1
```

Mỗi track phải được kiểm soát để tránh duplicate count.

---

### Step 5 — Direction

Nếu cần, hệ thống sẽ xác định hướng đi.

Ví dụ:

```text
top_to_bottom
bottom_to_top
left_to_right
right_to_left
```

Không bắt buộc phải làm đủ 4 hướng.

Mục tiêu tối thiểu nếu triển khai direction:

```text
2 hướng
```

---

### Step 6 — Counting Event

Mỗi lần một phương tiện đi qua line/zone, hệ thống lưu một event.

Format dự kiến:

```text
timestamp
track_id
class
line / zone
direction
confidence
```

Ví dụ:

```text
83.2, 54, motorcycle, line_1, top_to_bottom, 0.82
```

Các event này sẽ được ghi ra CSV.

---

### Step 7 — Statistics

Từ counting event, hệ thống tổng hợp:

```text
Tổng số xe
Số xe theo class
Số xe theo phút
Số xe theo hướng
```

Output:

```text
CSV
Bar chart
Line chart
```

Ví dụ:

```text
Minute 0–1
motorcycle = 35
car        = 12
bus        = 1

Minute 1–2
motorcycle = 42
car        = 15
bus        = 2
```

---

### Step 8 — Manual Ground Truth

Nhóm sẽ tự xem một đoạn video evaluation và đếm thủ công.

Ví dụ:

```text
motorcycle = 100
car        = 30
bus        = 4
truck      = 6
```

Đây là ground truth để so với kết quả của hệ thống.

---

### Step 9 — Evaluation

Các metric chính:

#### Absolute Error

```text
AE = |System Count - Manual Count|
```

#### Counting Accuracy

```text
Accuracy =
1 - |System Count - Manual Count| / Manual Count
```

#### MAE

Dùng khi chia video thành nhiều khoảng thời gian.

Ngoài ra có thể ghi:

```text
FPS
Inference time
```

Nếu có annotation cho detection thì có thể thêm:

```text
Precision
Recall
mAP50
mAP50-95
```

---

## 3. Các experiment của nhóm

Các experiment có thứ tự logic khi báo cáo, nhưng trong lúc code **không cần chờ nhau hoàn toàn**.

### Experiment 1 — Baseline

Cấu hình ban đầu:

```text
YOLOv8n
+
ByteTrack
+
1 counting line
```

Mục tiêu:

- Kiểm tra toàn bộ pipeline chạy được.
- Có kết quả baseline.
- Ghi nhận lỗi ban đầu.

Các lỗi cần quan sát:

```text
missed detection
sai class
mất track
ID switch
duplicate count
missed crossing
```

---

### Experiment 2 — Detector Comparison

Mục tiêu:

> Chọn YOLO phù hợp nhất với video giao thông của nhóm.

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
Cùng video
Cùng tracker
Cùng counting logic
Cùng ground truth
```

Ví dụ:

```text
E2.1 YOLOv8n + ByteTrack
E2.2 YOLOv8s + ByteTrack
E2.3 YOLOv8m + ByteTrack
```

So sánh:

```text
Detection quality
Missed vehicles
Counting accuracy
FPS
```

Kết quả:

```text
→ chọn Best Detector
```

---

### Experiment 3 — Tracker Comparison

Sau khi Experiment 2 chọn được detector tốt nhất:

```text
Best Detector + ByteTrack
vs
Best Detector + BoT-SORT
```

Ví dụ:

```text
YOLOv8s + ByteTrack
vs
YOLOv8s + BoT-SORT
```

Giữ cố định:

```text
Detector
Video
conf
imgsz
Counting logic
Counting line
```

So sánh:

```text
Counting accuracy
ID switch
Lost track
Track fragmentation
Duplicate count
FPS
```

Kết quả:

```text
→ chọn Best Tracker
```

---

### Experiment 4 — Counting Strategy

Sau khi có:

```text
Best Detector
+
Best Tracker
```

thì thử các cách counting.

Ví dụ:

```text
1 counting line
vs
multiple lines / zones
```

Có thể thêm:

```text
No direction
vs
Direction-aware counting
```

So sánh:

```text
Counting accuracy
Missed crossing
Duplicate count
```

Kết quả:

```text
→ chọn Best Counting Strategy
```

---

### Experiment 5 — Fine-tune YOLOv8 (Optional)

Chỉ làm nếu YOLO pretrained vẫn gặp vấn đề như:

```text
miss nhiều motorcycle
miss xe nhỏ
nhầm motorcycle / bicycle
```

Pipeline:

```text
Ảnh / frame giao thông Việt Nam
→ Annotation
→ Train / Val / Test
→ Fine-tune YOLOv8
→ best.pt
```

Sau đó so:

```text
Pretrained YOLO
vs
Fine-tuned YOLO
```

Đây là experiment mở rộng, không ưu tiên trước khi pipeline chính chạy ổn.
