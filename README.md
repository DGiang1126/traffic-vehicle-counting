# Đức Anh — Counting/E4

## Các file thuộc task

- `src/counting/counter.py`: phát hiện crossing, direction và chống đếm trùng.
- `src/counting/__init__.py`: export API của module.
- `src/traffic_counting.py`: chỉ thêm phần đọc cấu hình direction vector.
- `experiments/experiment_04_counting.py`: chạy single-line và multi-line.
- `configs/e4_single_line.json`: cấu hình E4 một line.
- `configs/e4_multi_line.json`: cấu hình E4 nhiều line.
- `tests/test_counting.py`: unit test cho counting.

## Direction không phụ thuộc thứ tự hai đầu line

`start` và `end` chỉ xác định vị trí đoạn counting line. `direction_vector` là
mũi tên riêng chỉ chiều `IN`.

- `[0, 1]`: hướng xuống là IN.
- `[0, -1]`: hướng lên là IN.
- `[1, 0]`: hướng phải là IN.
- `[-1, 0]`: hướng trái là IN.

Đảo `start` và `end` không làm IN/OUT bị đảo. Nếu bỏ `direction_vector`, chương
trình quay về logic positive/negative cũ để tương thích cấu hình cũ.

Trong Streamlit, người dùng không nhập vector và không nhập hai tên hướng.
Ứng dụng tự tạo vector vuông góc với line, vẽ mũi tên xanh `IN` và cung cấp một
nút `Đảo hướng IN/OUT`. `OUT` luôn là chiều đối diện nên không thể bị chọn trùng
phía với `IN`.

## Chạy test

Tại thư mục gốc dự án:

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe -m unittest tests.test_counting -v
```

## Chạy E4 chưa có ground truth

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe .\experiments\experiment_04_counting.py `
  --video .\data\videos\traffic.mp4
```

Kết quả so sánh được lưu tại:

```text
outputs/experiments/e4_counting_comparison.csv
```

## Chạy E4 chính thức với ground truth

Single-line và multi-line có định nghĩa đếm khác nhau nên nên dùng hai file
ground truth tương ứng:

```powershell
& C:/Users/AnkjoK/miniconda3/envs/aiEnv/python.exe .\experiments\experiment_04_counting.py `
  --video .\data\videos\eval_01.mp4 `
  --single-ground-truth .\data\ground_truth\e4_single_manual.csv `
  --multi-ground-truth .\data\ground_truth\e4_multi_manual.csv
```

## Tích hợp Git an toàn

```powershell
git status
git switch -c feature/counting
```

Không chép đè cả repository. Chỉ đưa từng file phía trên vào branch, chạy test,
xem `git diff`, rồi mới commit:

```powershell
git diff --check
git add src/counting tests/test_counting.py
git commit -m "feat: add direction-aware multi-line counter"

git add src/traffic_counting.py configs/e4_single_line.json configs/e4_multi_line.json
git commit -m "feat: configure endpoint-independent counting directions"

git add experiments/experiment_04_counting.py docs/DUC_ANH_COUNTING_TASK.md
git commit -m "experiment: compare E4 counting strategies"
```
