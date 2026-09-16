# Task Đức Anh — Counting và Experiment E4

## 1. Thông tin task

- Thành viên phụ trách: Đức Anh.
- Module phụ trách: counting line, xác định hướng di chuyển và Experiment E4.
- Mục tiêu chính: so sánh chiến lược đếm xe bằng một counting line và nhiều
  counting line trên cùng video.
- Phạm vi tích hợp: nhận kết quả tracking từ module chung, tạo counting event,
  sau đó chuyển event cho module statistics và evaluation của nhóm.

## 2. Đầu vào và đầu ra

### Đầu vào

- Video giao thông gốc.
- Danh sách phương tiện đã được detector và tracker xử lý. Mỗi phương tiện cần
  có `track_id`, loại phương tiện, confidence, bounding box và tâm bounding box.
- File JSON định nghĩa một hoặc nhiều counting line.
- Ground truth thủ công nếu chạy đánh giá chính thức.

### Đầu ra

- Danh sách `CountingEvent`.
- CSV sự kiện với schema:

```text
timestamp,frame_index,track_id,class,line,direction,confidence
```

- Video đã vẽ bounding box, track ID, counting line và thống kê trực tiếp.
- Các bảng tổng hợp theo class, thời gian, hướng và line.
- Biểu đồ thống kê.
- Kết quả E4 so sánh single-line và multi-line.

## 3. Các file thuộc task Đức Anh

```text
src/counting/counter.py
src/counting/__init__.py
src/app.py
src/visualization/renderer.py
src/visualization/__init__.py
configs/e4_single_line.json
configs/e4_multi_line.json
experiments/experiment_04_counting.py
tests/test_counting.py
integration_reference/traffic_counting.py
docs/DUC_ANH_TASK.md
docs/COUNTING_INTEGRATION_CHANGE.md
docs/UNIFIED_DIRECTION_UI.md
```

`integration_reference/traffic_counting.py` là file tham khảo tích hợp. Không
chép đè trực tiếp lên `src/traffic_counting.py` của dự án nhóm nếu thành viên
khác đã sửa file đó.

## 4. Những phần đã hoàn thành

### Counting core

- Định nghĩa `LineDefinition` cho đoạn counting line hữu hạn.
- Xác định phương tiện nằm ở phía nào của line bằng tích có hướng.
- Chỉ tạo event khi quỹ đạo tâm phương tiện thực sự cắt đoạn line hữu hạn.
- Không đếm khi phương tiện chỉ cắt phần kéo dài vô hạn của đường thẳng.
- Không đếm trùng một `track_id` nhiều lần trên cùng một line.
- Một `track_id` vẫn có thể được đếm một lần trên mỗi line khác nhau.
- Hỗ trợ một hoặc nhiều line thông qua `MultiLineCounter`.

### Direction

- Hỗ trợ cơ chế direction cũ bằng hai nhãn
  `negative_to_positive` và `positive_to_negative`.
- Hỗ trợ cơ chế mới bằng `direction_vector`.
- `direction_vector` chỉ chiều `IN`; chiều ngược lại là `OUT`.
- Đảo thứ tự `start` và `end` không làm thay đổi ý nghĩa IN/OUT khi sử dụng
  `direction_vector`.
- Hàm `normal_direction()` tạo vector vuông góc với line để Streamlit có thể
  hiển thị mũi tên hướng đi.

### Multi-line

Task có hỗ trợ multi-line. Trong `MultiLineCounter.update()`, mỗi phương tiện
được kiểm tra với toàn bộ line:

```python
for line in self.lines:
    key = (track_id, line.name)
```

Khóa `(track_id, line.name)` có nghĩa là một xe có thể được tính trên `line_1`
và `line_2`, nhưng không bị tính lặp lại nhiều lần trên cùng một line.

### Streamlit

- Cho phép chọn số lượng line.
- Cho phép đặt tên riêng cho từng line.
- Cho phép chỉnh hai điểm đầu và cuối bằng tọa độ chuẩn hóa từ 0 đến 1.
- Tự tạo mũi tên vuông góc với line.
- Hiển thị mũi tên xanh `IN` và chiều ngược lại `OUT`.
- Có tùy chọn `Đảo hướng IN/OUT`.
- Preview line trên frame đầu trước khi chạy.
- Hiển thị video kết quả, CSV sự kiện, bảng thống kê và biểu đồ.

### Experiment E4

- Chạy chiến lược `e4_single_line`.
- Chạy chiến lược `e4_multi_line`.
- Lưu system count, số track ID duy nhất và processing FPS.
- Khi có ground truth, tính absolute error, counting accuracy và MAE.
- Tạo file `e4_counting_comparison.csv`.

### Unit test

- Test crossing từ hai hướng.
- Test chống đếm trùng.
- Test trường hợp tâm xe nằm đúng trên line.
- Test đoạn line hữu hạn.
- Test cùng một xe đi qua nhiều line.
- Test direction không phụ thuộc thứ tự hai đầu line.
- Test đảo hướng IN/OUT.

## 5. Phần phụ thuộc vào dự án nhóm

Task Đức Anh không tự triển khai lại các phần sau:

- Detector YOLO.
- ByteTrack hoặc BoT-SORT.
- Đọc và ghi video hoàn chỉnh.
- Statistics chung của nhóm.
- Evaluation chung của nhóm.

Để chạy end-to-end cần dự án nhóm có các file:

```text
src/traffic_counting.py
src/detection/
src/tracking/
src/statistics/
evaluation/evaluate_counting.py
```

Nếu thiếu `evaluation/evaluate_counting.py`, Experiment E4 sẽ báo:

```text
ModuleNotFoundError: No module named 'evaluation.evaluate_counting'
```

Đây là lỗi thiếu module tích hợp, không phải lỗi của `MultiLineCounter`.

## 6. Tích hợp tối thiểu vào `src/traffic_counting.py`

Trong hàm `build_lines()`, lúc tạo `LineDefinition`, cần có:

```python
LineDefinition(
    name=item["name"],
    start=point(item["start"]),
    end=point(item["end"]),
    negative_to_positive=item.get(
        "negative_to_positive", "negative_to_positive"
    ),
    positive_to_negative=item.get(
        "positive_to_negative", "positive_to_negative"
    ),
    direction_vector=(
        tuple(map(float, item["direction_vector"]))
        if item.get("direction_vector") is not None
        else None
    ),
    forward_direction=item.get("forward_direction", "IN"),
    reverse_direction=item.get("reverse_direction", "OUT"),
)
```

Cấu hình cũ không có `direction_vector` vẫn tiếp tục chạy bằng logic direction
cũ.

## 7. Chạy unit test

Mở PowerShell tại thư mục gốc dự án:

```powershell
$python = "C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe"

& $python -m unittest tests.test_counting -v
```

Kết quả mong đợi là toàn bộ test báo `OK`.

## 8. Chạy Streamlit

```powershell
$python = "C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe"

& $python -m streamlit run .\src\app.py
```

Quy trình sử dụng:

1. Upload video gốc chưa có bounding box và counting line.
2. Chọn model và tracker.
3. Chọn số lượng line.
4. Đặt tên và tọa độ cho từng line.
5. Kiểm tra mũi tên xanh IN trên preview.
6. Bật `Đảo hướng IN/OUT` nếu mũi tên đang chỉ sai phía.
7. Nhấn chạy.
8. Xem và tải video, CSV sự kiện và bảng thống kê.

## 9. Chạy E4 cho một video

### Chưa có ground truth

```powershell
$python = "C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe"

& $python .\experiments\experiment_04_counting.py `
    --video ".\data\videos\evaluation\eval_01.mp4" `
    --single-config ".\configs\e4_single_line.json" `
    --multi-config ".\configs\e4_multi_line.json" `
    --output-dir ".\outputs\evaluation\eval_01"
```

### Có ground truth

```powershell
& $python .\experiments\experiment_04_counting.py `
    --video ".\data\videos\evaluation\eval_01.mp4" `
    --single-config ".\configs\e4_single_line.json" `
    --multi-config ".\configs\e4_multi_line.json" `
    --single-ground-truth ".\data\ground_truth\eval_01_single_manual.csv" `
    --multi-ground-truth ".\data\ground_truth\eval_01_multi_manual.csv" `
    --output-dir ".\outputs\evaluation\eval_01"
```

Single-line và multi-line sử dụng định nghĩa vùng đếm khác nhau nên cần hai file
ground truth tương ứng.

## 10. Chạy toàn bộ video evaluation mà không ghi đè

`experiment_04_counting.py` hiện dùng các tên cố định:

```text
e4_single_line
e4_multi_line
e4_counting_comparison.csv
```

Do đó mỗi video phải có một `output-dir` riêng. Lệnh an toàn:

```powershell
$python = "C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe"

Get-ChildItem ".\data\videos\evaluation\*.mp4" | ForEach-Object {
    $videoName = $_.BaseName
    $videoOutput = ".\outputs\evaluation\$videoName"

    & $python ".\experiments\experiment_04_counting.py" `
        --video $_.FullName `
        --single-config ".\configs\e4_single_line.json" `
        --multi-config ".\configs\e4_multi_line.json" `
        --output-dir $videoOutput
}
```

Kết quả của mỗi video được tách theo tên video:

```text
outputs/evaluation/
├── eval_01/
│   ├── videos/e4_single_line.mp4
│   ├── videos/e4_multi_line.mp4
│   ├── csv/e4_single_line_events.csv
│   ├── csv/e4_multi_line_events.csv
│   └── experiments/e4_counting_comparison.csv
├── eval_02/
└── eval_03/
```

Các video khác nhau không ghi đè nhau. Nếu chạy lại cùng một video vào cùng thư
mục, kết quả cũ có thể bị ghi đè; khi cần giữ nhiều lần chạy, thêm timestamp vào
`$videoOutput`.

## 11. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Counting một line | Đã hoàn thành |
| Counting nhiều line | Đã hoàn thành |
| Chống đếm trùng theo track và line | Đã hoàn thành |
| Direction vector IN/OUT | Đã hoàn thành |
| Streamlit chỉnh nhiều line | Đã hoàn thành |
| Preview mũi tên IN/OUT | Đã hoàn thành |
| Config E4 single-line | Đã hoàn thành |
| Config E4 multi-line | Đã hoàn thành |
| Experiment E4 | Đã hoàn thành code |
| Unit test counting | Đã hoàn thành |
| Chạy chính thức ba video evaluation | Chưa xác nhận trong gói code |
| Ground truth cho từng video | Cần đếm thủ công |
| Chỉ số accuracy/MAE chính thức | Chỉ có sau khi có ground truth |

## 12. Tích hợp Git an toàn

```powershell
git status
git switch -c feature/counting
```

Chỉ đưa các file thuộc task vào branch. Không chép đè toàn bộ repository:

```powershell
git add src/counting tests/test_counting.py
git commit -m "feat: add direction-aware multi-line counter"

git add configs/e4_single_line.json configs/e4_multi_line.json
git commit -m "feat: add E4 counting configs"

git add experiments/experiment_04_counting.py docs/DUC_ANH_TASK.md
git commit -m "experiment: compare single-line and multi-line counting"
```

Với các file tích hợp chung như `src/traffic_counting.py` và `src/app.py`, luôn
xem `git diff` và merge từng thay đổi nhỏ thay vì thay toàn bộ file.
