from pathlib import Path
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "plate_detection_v2"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"


def main():

    data_yaml = DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    print("Dataset:", DATASET_DIR)
    print("Config:", data_yaml)

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=4,
        patience=25,

        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=2,
        fliplr=0.0,

        project=str(OUTPUT_DIR),
        name="plate_detection_v2",
        seed=42
    )


if __name__ == "__main__":
    main()