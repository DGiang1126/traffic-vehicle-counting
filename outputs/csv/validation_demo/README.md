# Statistics / Evaluation validation demo

Dùng Ground Truth thật của `dev_01_detrac_MVI_20011` để test pipeline bằng events có kiểm soát.

## Case 1 - Perfect match

- System total = Manual total = 24.
- Class counts khớp hoàn toàn.
- Expected: overall accuracy 100%, class accuracy 100%, time MAE 0.

## Case 2 - Class swap

- Tổng vẫn 24 nhưng đổi 1 `car` thành `truck`.
- Expected: overall accuracy vẫn 100%, nhưng evaluator phải phát hiện `car` undercount 1 và `truck` overcount 1.

## Case 3 - Time shift

- Tổng và class totals vẫn đúng.
- Dời 1 `car` từ bin 0-10s sang 10-20s.
- Expected: overall accuracy 100%, nhưng time MAE > 0 và hai time-bin bị under/overcount.

## Compatibility checks

- Events có cột `direction`: Statistics sinh thêm `*_statistics_by_direction.csv`.
- Events không có `direction`: vẫn chạy bình thường.
- Class ngoài bộ 5 class chuẩn: bị từ chối với thông báo lỗi rõ ràng.
