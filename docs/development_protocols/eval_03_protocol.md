# eval_03

Video: `1920x1080`, khoảng `123.43s`, `24.05 FPS`.
Mục tiêu: evaluation trên ngã tư/nhiều hướng với hai counting line.

## Counting line(s) đã chốt

- `line_1`: `START=(1227,549)`, `END=(1653,780)`.
- `line_2`: `START=(388,619)`, `END=(1162,546)`.
- Config máy đọc: `data/ground_truth/line_configs/eval_03_lines.json`.

## Counting rule

- Nhìn theo `START -> END`, chỉ count xe crossing từ phía trái sang phía phải.
- Counting point là tâm bbox `(cx, cy)` và phải cắt đoạn line hữu hạn.
- `line_1`, `line_2` chỉ là ID; hướng đếm nằm trong START/END.
- Ground Truth hiện tại không cần cột `direction` riêng.

![eval_03 counting rule](images/eval_03_rule.jpg)

## Manual Ground Truth

- Đã manual-count toàn bộ video theo interval `10s` và đã có bản tổng hợp theo phút.
- Tổng aggregate hiện tại: `49` xe.
- Files chính: `data/ground_truth/eval_03_manual_counts_by_10s.csv` và `eval_03_manual_counts_by_minute.csv`.
