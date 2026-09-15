# Manual Ground Truth Tool

Công cụ chính:

```text
evaluation/manual_count_tool.py
```

Mục tiêu: dùng **một file duy nhất** để chọn bất kỳ video nào trong `data/videos/`, thiết lập counting line, rồi manual-count và tự xuất nhiều định dạng Ground Truth CSV.

## 1. Chạy tool

Khuyến nghị dùng virtual environment của project:

```powershell
.\.venv\Scripts\Activate.ps1
python evaluation/manual_count_tool.py
```

Nếu không activate `.venv`, máy có thể dùng Python khác chưa cài `opencv-python`.

### Cú pháp tổng quát

```powershell
python evaluation/manual_count_tool.py [--video VIDEO] [--interval SECONDS] `
  [--speed SPEED] [--edit-lines] [--setup-only] [--resume] [--restart]
```

Ví dụ mở thẳng một evaluation video và chia đoạn 10 giây:

```powershell
python evaluation/manual_count_tool.py `
  --video data/videos/evaluation/eval_02.mp4 `
  --interval 10
```

Nếu muốn bỏ progress cũ và đếm lại từ đầu:

```powershell
python evaluation/manual_count_tool.py `
  --video data/videos/evaluation/eval_02.mp4 `
  --restart
```

## 2. Workflow tương tác

Khi chạy không truyền tham số, tool sẽ:

```text
Chọn video
→ chọn số counting line
→ nhập START/END nếu đã có tọa độ
→ chỉnh endpoint trực tiếp trên frame bằng chuột
→ ENTER xác nhận line
→ chuyển sang manual count
→ xem từng đoạn thời gian
→ nhập count theo class
→ tự lưu sau mỗi đoạn
```

Nếu đã có line config, tool cho chọn dùng lại hoặc sửa.

Quy ước hiện tại: nhìn theo `START -> END`, chỉ đếm xe crossing từ phía trái sang phía phải của đoạn line hữu hạn.

## 3. Thiết lập line

Với mỗi line, terminal cho nhập:

```text
x1 y1 x2 y2
```

Trong đó `(x1,y1)` là START và `(x2,y2)` là END. Nếu nhấn Enter, tool dùng tọa độ cũ/default rồi mở cửa sổ để kéo endpoint bằng chuột.

Trong cửa sổ line setup:

```text
Circle = START
Square = END
Drag endpoint = chỉnh tọa độ
ENTER = xác nhận
Q / Esc = hủy
```

Config của video mới được lưu tại:

```text
data/ground_truth/line_configs/<video_stem>_lines.json
```

Các path được ghi vào JSON/progress đều dùng project-relative path (dùng `/`), không lưu đường dẫn tuyệt đối theo máy cá nhân.

Nếu chỉ muốn setup/chỉnh line mà không manual-count, có thể chạy utility trực tiếp.

Cú pháp:

```powershell
python src/visualization/line_editor.py [--video VIDEO] [--edit]
```

Ví dụ:

```powershell
python src/visualization/line_editor.py `
  --video data/videos/evaluation/eval_03.mp4 `
  --edit
```

## 4. Các file tự sinh

Với interval mặc định `10s`, sau khi đếm tool tạo:

```text
data/ground_truth/<video>_manual_counts_by_line.csv
data/ground_truth/<video>_manual_counts.csv
data/ground_truth/<video>_manual_counts_by_10s.csv
data/ground_truth/<video>_manual_counts_by_minute.csv
data/ground_truth/<video>_manual_count_progress.json
```

Ý nghĩa:

- `*_by_line.csv`: chi tiết từng line và từng khoảng thời gian.
- `*_manual_counts.csv`: aggregate theo interval đang chọn, giữ để tương thích pipeline cũ.
- `*_by_10s.csv`: tên rõ ràng cho aggregate 10 giây mặc định.
- `*_by_minute.csv`: tổng theo minute bucket để dùng cho traffic flow/evaluation theo phút.
- `*_progress.json`: lưu tiến độ, line geometry và counts để resume.

Nếu đổi interval, file interval sẽ đổi tên theo giá trị đó. Ví dụ `--interval 15` tạo:

```text
<video>_manual_counts_by_15s.csv
```

## 5. Điều khiển khi manual count

Trong lúc phát video:

```text
SPACE = pause/resume
R = replay đoạn hiện tại
Q / Esc = thoát và lưu tiến độ
```

Sau khi phát hết đoạn, nhập 5 số theo thứ tự:

```text
motorcycle car bus truck bicycle
```

Ví dụ:

```text
2 7 0 1 0
```

## 6. Tùy chọn hữu ích

```powershell
python evaluation/manual_count_tool.py --video data/videos/evaluation/eval_01_detrac_40131_40141.mp4
python evaluation/manual_count_tool.py --interval 10
python evaluation/manual_count_tool.py --speed 0.75
python evaluation/manual_count_tool.py --edit-lines
python evaluation/manual_count_tool.py --setup-only
python evaluation/manual_count_tool.py --resume
python evaluation/manual_count_tool.py --restart
```

Nếu không truyền `--video`, tool tự liệt kê tất cả video trong `data/videos/`, gồm development, evaluation và các video khác.

## 7. Lưu ý multi-line

File aggregate cộng counts của các line trong cùng time-bin. Vì vậy khi protocol yêu cầu mỗi physical vehicle chỉ được tính một lần, các line phải được thiết kế/đếm sao cho cùng một xe không bị ghi ở nhiều line. `*_by_line.csv` luôn được giữ để audit riêng từng line.

Nếu thay START/END của một line và resume, counts cũ của line đó sẽ bị bỏ để tránh dùng Ground Truth với geometry khác.

## 8. Evaluation GT hiện đã chuẩn bị

Ba video evaluation hiện đã được manual-count toàn bộ bằng chính tool này:

```text
eval_01_detrac_40131_40141  -> 2 lines -> 105 xe
eval_02                       -> 1 line  -> 199 xe
eval_03                       -> 2 lines -> 49 xe
```

Mỗi video đã có `*_by_line.csv`, `*_manual_counts.csv`, `*_by_10s.csv`, `*_by_minute.csv` và line config tương ứng trong `data/ground_truth/line_configs/`.

Hình minh họa và protocol dành cho người đọc nằm tại `docs/development_protocols/`; code luôn dùng JSON trong `data/ground_truth/line_configs/` làm source of truth.
