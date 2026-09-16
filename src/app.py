"""Streamlit UI for configurable multi-line vehicle counting."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.counting.counter import LineDefinition, normal_direction  # noqa: E402
from src.traffic_counting import run  # noqa: E402
from src.visualization.counting_renderer import draw_counting_lines  # noqa: E402


def _read_first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Không đọc được frame đầu của video")
    return frame


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _number_input(label, value, maximum, key):
    return int(st.number_input(label, min_value=0, max_value=max(0, maximum), value=min(int(value), maximum), key=key))


def _source_points(source: dict, width: int, height: int):
    start = source.get("start", [0.1, 0.5])
    end = source.get("end", [0.9, 0.5])
    start_px = [round(start[0] * width), round(start[1] * height)] if max(start) <= 1 else start
    end_px = [round(end[0] * width), round(end[1] * height)] if max(end) <= 1 else end
    return [int(start_px[0]), int(start_px[1])], [int(end_px[0]), int(end_px[1])]


def _line_editor(index: int, source: dict, width: int, height: int, manual_coordinates: bool) -> dict:
    st.markdown(f"#### Line {index + 1}")
    name = st.text_input("Tên counting line", source.get("name", f"line_{index + 1}"), key=f"line_name_{index}")

    start_px, end_px = _source_points(source, width, height)
    if manual_coordinates:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Điểm đầu của đoạn đếm")
            x1 = _number_input("Start X", start_px[0], width - 1, f"x1_{index}")
            y1 = _number_input("Start Y", start_px[1], height - 1, f"y1_{index}")
        with col2:
            st.caption("Điểm cuối của đoạn đếm")
            x2 = _number_input("End X", end_px[0], width - 1, f"x2_{index}")
            y2 = _number_input("End Y", end_px[1], height - 1, f"y2_{index}")
    else:
        for key, value in (
            (f"x1_{index}", start_px[0]), (f"y1_{index}", start_px[1]),
            (f"x2_{index}", end_px[0]), (f"y2_{index}", end_px[1]),
        ):
            st.session_state.setdefault(key, value)
        x1, y1 = int(st.session_state[f"x1_{index}"]), int(st.session_state[f"y1_{index}"])
        x2, y2 = int(st.session_state[f"x2_{index}"]), int(st.session_state[f"y2_{index}"])
        st.caption(f"Start=({x1}, {y1}) · End=({x2}, {y2}) — chọn lại bằng cách nhấp trên ảnh.")

    in_label = st.text_input("Tên hướng IN", source.get("in_label", "IN"), key=f"in_label_{index}")
    out_label = st.text_input("Tên hướng OUT", source.get("out_label", "OUT"), key=f"out_label_{index}")

    base_vector = normal_direction((x1, y1), (x2, y2)) if (x1, y1) != (x2, y2) else (0.0, 1.0)
    raw_vector = source.get("direction_vector")
    initially_reversed = bool(
        raw_vector and raw_vector[0] * base_vector[0] + raw_vector[1] * base_vector[1] < 0
    )
    reverse = st.checkbox(
        "Đảo chiều IN ↔ OUT",
        value=initially_reversed,
        key=f"reverse_{index}",
        help="Đổi phía mà mũi tên xanh IN đang chỉ tới; không đổi vị trí line.",
    )
    vector = normal_direction((x1, y1), (x2, y2), reverse) if (x1, y1) != (x2, y2) else (0.0, 1.0)
    return {
        "name": name.strip(),
        "start": [x1 / width, y1 / height],
        "end": [x2 / width, y2 / height],
        "direction_vector": [vector[0], vector[1]],
        "in_label": in_label.strip(),
        "out_label": out_label.strip(),
    }


st.set_page_config(page_title="Multi-line Vehicle Counting", layout="wide")
st.title("Multi-line Vehicle Counting")
st.caption("Multi-line · tên line tùy chỉnh · mũi tên vuông góc IN/OUT · xuất video, CSV và biểu đồ")

config_files = sorted((PROJECT_ROOT / "configs").glob("*.json"))
if not config_files:
    st.error("Không có config JSON trong thư mục configs.")
    st.stop()
selected_config = st.sidebar.selectbox("Config ban đầu", config_files, format_func=lambda p: p.name)
base_config = _load_json(selected_config)

uploaded_video = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"])
if uploaded_video is None:
    st.info("Tải video lên để đặt counting line và xem trước hướng IN/OUT.")
    st.stop()

runtime_input_dir = PROJECT_ROOT / "runtime" / "inputs"
runtime_input_dir.mkdir(parents=True, exist_ok=True)
input_path = runtime_input_dir / uploaded_video.name
input_path.write_bytes(uploaded_video.getbuffer())

try:
    first_frame = _read_first_frame(input_path)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
height, width = first_frame.shape[:2]

st.sidebar.markdown("### Mô hình")
known_models = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"]
default_model = str(base_config.get("model", "yolov8n.pt"))
model_index = known_models.index(default_model) if default_model in known_models else 0
model_name = st.sidebar.selectbox("YOLO model", known_models, index=model_index)
tracker = st.sidebar.selectbox(
    "Tracker", ["bytetrack.yaml", "botsort.yaml"],
    index=0 if base_config.get("tracker", "bytetrack.yaml") == "bytetrack.yaml" else 1,
)
confidence = st.sidebar.slider("Confidence", 0.05, 0.95, float(base_config.get("confidence", 0.25)), 0.05)
image_size = st.sidebar.select_slider("Image size", [320, 480, 640, 960, 1280], value=int(base_config.get("image_size", 960)))

existing_lines = base_config.get("counting_lines", [])
st.sidebar.markdown("### Counting lines")
counting_mode = st.sidebar.radio(
    "Chế độ đếm",
    ["Single line", "Multi-line"],
    index=1 if len(existing_lines) > 1 else 0,
    horizontal=True,
)
if counting_mode == "Single line":
    line_count = 1
else:
    line_count = st.sidebar.number_input(
        "Số line", min_value=2, max_value=20, value=max(2, len(existing_lines)), step=1
    )

placement_mode = st.radio(
    "Cách chọn hai điểm của counting line",
    ["Nhấp trực tiếp trên ảnh", "Nhập tọa độ thủ công"],
    horizontal=True,
)
manual_coordinates = placement_mode == "Nhập tọa độ thủ công"

if not manual_coordinates:
    if streamlit_image_coordinates is None:
        st.error(
            "Thiếu streamlit-image-coordinates. Chạy: "
            "python -m pip install -r requirements-streamlit.txt"
        )
        st.stop()

    active_line = st.selectbox(
        "Line đang chọn điểm",
        range(int(line_count)),
        format_func=lambda index: st.session_state.get(
            f"line_name_{index}",
            existing_lines[index].get("name", f"line_{index + 1}")
            if index < len(existing_lines)
            else f"line_{index + 1}",
        ),
    )
    click_frame = first_frame.copy()
    click_definitions = []
    for index in range(int(line_count)):
        source = existing_lines[index] if index < len(existing_lines) else {}
        default_start, default_end = _source_points(source, width, height)
        start = (
            int(st.session_state.get(f"x1_{index}", default_start[0])),
            int(st.session_state.get(f"y1_{index}", default_start[1])),
        )
        end = (
            int(st.session_state.get(f"x2_{index}", default_end[0])),
            int(st.session_state.get(f"y2_{index}", default_end[1])),
        )
        if start != end:
            click_definitions.append(
                LineDefinition(
                    name=st.session_state.get(f"line_name_{index}", source.get("name", f"line_{index + 1}")),
                    start=start,
                    end=end,
                    direction_vector=normal_direction(
                        start, end, bool(st.session_state.get(f"reverse_{index}", False))
                    ),
                    in_label=st.session_state.get(f"in_label_{index}", source.get("in_label", "IN")),
                    out_label=st.session_state.get(f"out_label_{index}", source.get("out_label", "OUT")),
                )
            )
    draw_counting_lines(click_frame, click_definitions)

    click_points_key = f"click_points_{input_path.stem}_{active_line}"
    selected_points = st.session_state.setdefault(click_points_key, [])
    for point in selected_points:
        cv2.circle(click_frame, tuple(point), 8, (0, 0, 255), -1)

    display_width = min(width, 960)
    display_scale = display_width / width
    display_height = max(1, round(height * display_scale))
    display_frame = cv2.resize(click_frame, (display_width, display_height))
    st.info(
        f"Đang đặt điểm cho line {active_line + 1}: "
        "nhấp điểm START rồi nhấp điểm END. Sau điểm thứ hai, line sẽ tự cập nhật."
    )
    clicked = streamlit_image_coordinates(
        Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)),
        key=f"line_point_selector_{input_path.stem}_{active_line}",
    )
    if st.button("Chọn lại hai điểm của line đang chọn"):
        st.session_state[click_points_key] = []
        st.rerun()

    if clicked:
        event_token = f"{clicked.get('x')}:{clicked.get('y')}:{clicked.get('unix_time')}"
        token_key = f"last_click_token_{input_path.stem}_{active_line}"
        if st.session_state.get(token_key) != event_token:
            st.session_state[token_key] = event_token
            point = [
                min(width - 1, max(0, round(float(clicked["x"]) / display_scale))),
                min(height - 1, max(0, round(float(clicked["y"]) / display_scale))),
            ]
            selected_points = [*selected_points, point]
            if len(selected_points) == 2:
                (x1, y1), (x2, y2) = selected_points
                st.session_state[f"x1_{active_line}"] = x1
                st.session_state[f"y1_{active_line}"] = y1
                st.session_state[f"x2_{active_line}"] = x2
                st.session_state[f"y2_{active_line}"] = y2
                st.session_state[click_points_key] = []
            else:
                st.session_state[click_points_key] = selected_points[-1:]
            st.rerun()

editor_column, preview_column = st.columns([1, 1.25])
lines = []
with editor_column:
    for index in range(int(line_count)):
        source = existing_lines[index] if index < len(existing_lines) else {}
        try:
            lines.append(_line_editor(index, source, width, height, manual_coordinates))
        except ValueError as exc:
            st.error(str(exc))

errors = []
names = [line["name"] for line in lines]
if any(not name for name in names):
    errors.append("Tên line không được để trống.")
if len(names) != len(set(names)):
    errors.append("Tên các line phải khác nhau.")
for line in lines:
    if line["start"] == line["end"]:
        errors.append(f"{line['name'] or 'Line'}: hai điểm không được trùng nhau.")
    if not line["in_label"] or not line["out_label"]:
        errors.append(f"{line['name'] or 'Line'}: tên IN/OUT không được trống.")
    if line["in_label"] == line["out_label"]:
        errors.append(f"{line['name'] or 'Line'}: tên IN và OUT phải khác nhau.")

with preview_column:
    st.subheader("Xem trước line và hướng")
    preview = first_frame.copy()
    definitions = []
    for line in lines:
        start = (round(line["start"][0] * width), round(line["start"][1] * height))
        end = (round(line["end"][0] * width), round(line["end"][1] * height))
        if start != end and line["name"] and line["in_label"] and line["out_label"]:
            definitions.append(
                LineDefinition(
                    name=line["name"], start=start, end=end,
                    direction_vector=tuple(line["direction_vector"]),
                    in_label=line["in_label"], out_label=line["out_label"],
                )
            )
    draw_counting_lines(preview, definitions)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.info(
        "Đường màu là counting line. Hai mũi tên được vẽ VUÔNG GÓC với line: "
        "mũi tên XANH chỉ phía/hướng IN, mũi tên CAM chỉ phía/hướng OUT. "
        "Tên cạnh mũi tên chính là tên IN/OUT bạn nhập."
    )
    st.caption("Nếu phía xanh đang ngược ý bạn, bấm ‘Đảo chiều IN ↔ OUT’; không cần đổi điểm start/end.")

if errors:
    for error in errors:
        st.error(error)

classes = st.sidebar.multiselect(
    "Vehicle classes",
    ["car", "motorcycle", "bus", "truck", "bicycle"],
    default=base_config.get("vehicle_classes", ["car", "motorcycle", "bus", "truck", "bicycle"]),
)
mode_name = "single" if counting_mode == "Single line" else f"multi_{int(line_count)}lines"
default_run_name = f"{input_path.stem}_{mode_name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
run_name = st.text_input("Tên lần chạy / tên file output", default_run_name)

runtime_config = {
    "model": model_name,
    "tracker": tracker,
    "confidence": confidence,
    "image_size": image_size,
    "vehicle_classes": classes,
    "counting_lines": lines,
}
st.download_button(
    "Tải config hiện tại",
    data=json.dumps(runtime_config, ensure_ascii=False, indent=2),
    file_name=f"{run_name}.json",
    mime="application/json",
    disabled=bool(errors),
)

if st.button("Bắt đầu xử lý", type="primary", disabled=bool(errors) or not classes):
    runtime_config_dir = PROJECT_ROOT / "runtime" / "configs"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = runtime_config_dir / f"{run_name}.json"
    runtime_config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")
    progress = st.progress(0.0, text="Đang xử lý video...")
    try:
        result = run(
            video_path=input_path,
            config_path=runtime_config_path,
            output_dir=PROJECT_ROOT / "outputs",
            run_name=run_name,
            progress_callback=lambda value: progress.progress(value, text=f"Đang xử lý: {value:.0%}"),
        )
    except Exception as exc:
        st.exception(exc)
    else:
        progress.progress(1.0, text="Hoàn thành")
        st.success(
            f"Đã xử lý {result['number_of_lines']} line, tổng {result['total_events']} lượt xe. "
            f"Run: {result['run_name']}"
        )
        if result.get("video_warning"):
            st.warning(result["video_warning"])
        output_video = Path(result["output_video"])
        with st.expander("Các file đã lưu trong outputs", expanded=True):
            st.code(
                "\n".join(
                    [
                        f"Video: {result['output_video']}",
                        f"Events CSV: {result['events_csv']}",
                        f"Summary: {result['summary_dir']}",
                        f"Charts: {result['chart_dir']}",
                        f"Metrics: {result['metrics_json']}",
                        f"Config snapshot: {result['config_snapshot']}",
                        f"Code version: {result['code_version']}",
                    ]
                )
            )
        st.video(output_video.read_bytes())
        st.download_button("Tải video kết quả", output_video.read_bytes(), output_video.name, "video/mp4")

        events_csv = Path(result["events_csv"])
        st.download_button("Tải events CSV", events_csv.read_bytes(), events_csv.name, "text/csv")
        line_summary = Path(result["summary_dir"]) / "summary_by_line_class_direction.csv"
        st.download_button("Tải thống kê theo line/hướng", line_summary.read_bytes(), line_summary.name, "text/csv")
        st.dataframe(__import__("pandas").read_csv(line_summary), use_container_width=True)

        chart_dir = Path(result["chart_dir"])
        chart_paths = sorted(chart_dir.glob("*.png"))
        if chart_paths:
            chart_columns = st.columns(len(chart_paths))
            for column, chart_path in zip(chart_columns, chart_paths):
                column.image(str(chart_path), caption=chart_path.stem, use_container_width=True)
