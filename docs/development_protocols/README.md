# Visual Counting Protocols

Thư mục này là tài liệu **cho người đọc**: mô tả trực quan counting line và rule của 5 video development và 3 video evaluation. Code không đọc config từ `docs/`.

## Source of truth cho code

Tất cả line config máy đọc nằm tại:

```text
data/ground_truth/line_configs/
```

Mỗi video có file `<video_stem>_lines.json`. Nếu hình hoặc Markdown khác config JSON thì **JSON trong `data/ground_truth/line_configs/` được ưu tiên**.

## Development references

- `dev_01_protocol.md` … `dev_05_protocol.md`: protocol/debug notes cho 5 video development.
- `images/dev_01_rule.jpg` … `images/dev_05_rule.jpg`: frame minh họa với line hiện tại.

## Evaluation references

- `eval_01_protocol.md`, `eval_02_protocol.md`, `eval_03_protocol.md`: line, rule và trạng thái Manual GT của 3 video evaluation đã chuẩn bị.
- `images/eval_01_rule.jpg`, `images/eval_02_rule.jpg`, `images/eval_03_rule.jpg`: frame minh họa line evaluation.

## Quy ước chung

- Một line được xác định bởi `START=(x1,y1)` và `END=(x2,y2)`.
- Nhìn theo `START -> END`, chỉ count crossing từ phía trái sang phía phải.
- Counting point là tâm bbox `(cx, cy)` và phải cắt đoạn line hữu hạn.
- `line_1`, `line_2`, ... chỉ là ID; hướng đếm nằm trong thứ tự START/END.
- Classes: `motorcycle`, `car`, `bus`, `truck`, `bicycle`.
