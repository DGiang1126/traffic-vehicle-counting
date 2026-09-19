from pathlib import Path
import csv
import re


DATASET_ROOT = Path("data/datasets/vietnam_traffic")
IMAGES_ROOT = DATASET_ROOT / "images"
MAPPING_PATH = DATASET_ROOT / "frame_mapping.csv"


SPLITS = [
    {
        "split": "train",
        "folder": IMAGES_ROOT / "train",
        "camera": "cam01",
        "time_id": "t01",
        "source_video": "vn_intersection_cam01_t01.mp4",
    },
    {
        "split": "val",
        "folder": IMAGES_ROOT / "val",
        "camera": "cam01",
        "time_id": "t02",
        "source_video": "vn_intersection_cam01_t02.mp4",
    },
    {
        "split": "test",
        "folder": IMAGES_ROOT / "test",
        "camera": "cam02",
        "time_id": "t01",
        "source_video": "vn_intersection_cam02_t01.mp4",
    },
]


def extract_original_frame(filename: str):
    match = re.search(r"_f(\d+)", filename)

    if match:
        return int(match.group(1))

    return None


def main():
    rows = []

    for config in SPLITS:
        folder = config["folder"]

        files = sorted(folder.glob("*.jpg"))

        print(f"\nProcessing {config['split']}: {len(files)} images")

        for index, old_path in enumerate(files, start=1):
            original_frame = extract_original_frame(old_path.name)

            new_name = (
                f"{config['split']}_"
                f"{config['camera']}_"
                f"{config['time_id']}_"
                f"{index:03d}.jpg"
            )

            new_path = folder / new_name

            rows.append(
                {
                    "new_filename": new_name,
                    "old_filename": old_path.name,
                    "source_video": config["source_video"],
                    "original_frame": original_frame,
                    "split": config["split"],
                }
            )

            old_path.rename(new_path)

            print(f"{old_path.name} -> {new_name}")

    with open(
        MAPPING_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_filename",
                "old_filename",
                "source_video",
                "original_frame",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nDONE")
    print(f"Mapping saved to: {MAPPING_PATH}")


if __name__ == "__main__":
    main()