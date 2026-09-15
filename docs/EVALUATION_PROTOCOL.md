# Evaluation Protocol — Traffic Vehicle Counting

Tài liệu này chốt quy tắc dùng chung khi tạo Manual Ground Truth và khi đánh giá kết quả hệ thống.

## 1. Evaluation set đã chuẩn bị

Video được lưu tại:

```text
data/videos/evaluation/
```

Hiện đã có Manual Ground Truth đầy đủ cho 3 video:

| Video | Thời lượng | Lines | Manual GT |
| --- | ---: | ---: | ---: |
| `eval_01_detrac_40131_40141.mp4` | ~129.80s | 2 | 105 xe |
| `eval_02.mp4` | ~179.07s | 1 | 199 xe |
| `eval_03.mp4` | ~123.43s | 2 | 49 xe |

Line config của cả 3 nằm tại `data/ground_truth/line_configs/`; hình minh họa nằm trong `docs/development_protocols/`.

**Việc E1–E4 dùng một video hay toàn bộ 3-video set vẫn cần nhóm chốt trước khi chạy official experiment.** Khi đã chốt, mọi cấu hình đem so phải dùng cùng video set và cùng time range.

Yêu cầu:

- Dùng cùng evaluation video/set và cùng time range giữa các experiment.
- Camera cố định, nhìn rõ vùng đếm.
- Không đổi Ground Truth nếu counting rule không đổi.

## 2. Vehicle classes

```text
motorcycle
car
bus
truck
bicycle
```

Các module Detection, Counting, Statistics và Evaluation phải dùng cùng taxonomy này.
## 3. Counting line contract

Một counting line được định nghĩa trực tiếp bởi:

```text
START=(x1,y1) -> END=(x2,y2)
```

Quy tắc hiện tại:

- Nhìn theo hướng `START -> END`, chỉ count crossing từ **phía trái** sang **phía phải**.
- Counting point là tâm bbox `(cx, cy)`.
- Quỹ đạo tâm phải cắt **đoạn START-END hữu hạn**, không phải đường thẳng kéo dài vô hạn.
- Không dùng tên line để suy ra hình học; `line_1`, `line_2`, ... chỉ là ID.
- Không cần trường `direction` riêng trong Ground Truth hiện tại; thứ tự START/END đã encode hướng đếm.
- Mỗi physical vehicle chỉ được tính theo duplicate rule đã thống nhất cho experiment.

Manual Ground Truth và System Result phải dùng **chính xác cùng line/zone và cùng crossing rule**.

Nếu Counting team chuyển sang bidirectional, zone hoặc một định nghĩa multi-line làm thay đổi xe nào được tính, protocol và Ground Truth phải được cập nhật trước khi đánh giá.

## 4. Manual Ground Truth

Official format khuyến nghị:

```csv
minute,motorcycle,car,bus,truck,bicycle
0,0,0,0,0,0
1,0,0,0,0,0
```
Development/debug có thể dùng format chi tiết theo khoảng thời gian:

```csv
start_sec,end_sec,motorcycle,car,bus,truck,bicycle
0,10,0,0,0,0,0
```

Chỉ tính xe thực sự crossing line/zone theo protocol. Xe chỉ xuất hiện trong frame nhưng không crossing thì không tính.

## 5. Statistics & Evaluation contract

Counting/Integration bàn giao `outputs/csv/*_events.csv`. Kiệt dùng:

```text
src/statistics/statistics.py
```

để tạo System Statistics, sau đó dùng:

```text
evaluation/evaluate_counting.py
```

để so với Manual Ground Truth.

Time-bin của Statistics phải trùng Ground Truth khi tính MAE theo thời gian.

Hướng dẫn chạy chi tiết: `docs/STATISTICS_EVALUATION_USAGE.md`.

## 6. Quy tắc thay đổi protocol

Nếu đổi video, time range, classes, counting line/zone, START/END, crossing rule hoặc duplicate rule thì phải kiểm tra ảnh hưởng tới Ground Truth trước khi tái sử dụng.

Mục tiêu là giữ điều kiện đánh giá E1–E4 công bằng: chỉ thay đúng thành phần đang được so sánh.
