# Thay đổi tối thiểu trong file tích hợp chung

Không chép đè toàn bộ `src/traffic_counting.py` nếu thành viên khác đã sửa file
này. Trong hàm `build_lines()`, thêm ba tham số sau vào lúc tạo
`LineDefinition`:

```python
direction_vector=(
    tuple(map(float, item["direction_vector"]))
    if item.get("direction_vector") is not None
    else None
),
forward_direction=item.get("forward_direction", "IN"),
reverse_direction=item.get("reverse_direction", "OUT"),
```

Vị trí đầy đủ:

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

Cấu hình cũ không có `direction_vector` vẫn chạy như trước.
