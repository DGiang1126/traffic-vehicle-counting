# Unified Direction Design

## Quy tắc duy nhất

Người dùng tạo counting line bằng hai điểm và nhìn mũi tên xanh `IN` trên
preview. Nếu mũi tên sai, bật `Đảo hướng IN/OUT`. Hướng đối diện luôn là `OUT`,
vì vậy người dùng không thể đặt IN và OUT cùng một phía.

Streamlit không hiển thị các khái niệm `positive`, `negative`, `forward` hoặc
`reverse`. Những chi tiết đó chỉ tồn tại bên trong code.

## Luồng dữ liệu

```text
Hai đầu line + nút đảo
→ app.py tính direction_vector
→ lưu runtime config
→ traffic_counting.py tạo LineDefinition
→ counter.py trả về IN/OUT
→ renderer.py vẽ cùng mũi tên lên video
```

## Tương thích code cũ

`counter.py` vẫn đọc `negative_to_positive` và `positive_to_negative` khi cấu
hình cũ không có `direction_vector`. Streamlit mới chỉ sinh schema thống nhất:

```json
{
  "name": "line_1",
  "start": [0.05, 0.65],
  "end": [0.95, 0.65],
  "direction_vector": [0, 1],
  "forward_direction": "IN",
  "reverse_direction": "OUT"
}
```
