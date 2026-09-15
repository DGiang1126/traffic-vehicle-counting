# dev_02_detrac_MVI_20012

Video: `960x540`, khoảng `37.44s`.
Mục tiêu: test single-line counting trong cảnh có bus/trạm và occlusion.

## Counting line(s) đã chốt

- `line_1`: `START=(288,416)`, `END=(955,411)`.
- Normalized: `START=(0.3000,0.7704)`, `END=(0.9948,0.7611)`.

## Counting rule

- Nhìn theo hướng `START -> END`, chỉ count xe crossing từ **phía trái** của line sang **phía phải**.
- Counting point: tâm bbox `(cx, cy)`.
- Crossing phải là giao giữa quỹ đạo tâm bbox và **đoạn START-END hữu hạn**.
- Mỗi physical vehicle chỉ count một lần.
- Không dùng hoặc lưu trường `direction` riêng.

![dev_02 counting rule](images/dev_02_rule.jpg)

## Manual/debug rule

- Xe đang dừng ở trạm không tính cho đến khi thực sự crossing line.
- Nếu tracker mất ID do che khuất, manual count vẫn tính physical vehicle một lần.
- Xe đỗ, xe ngoài vùng đếm hoặc xe không crossing line thì không tính.
- Config tọa độ chuẩn nằm trong `data/ground_truth/line_configs/` theo tên video.
