from pathlib import Path
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "plate_detection_v2"
)

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "plate_detection_v2"
    / "weights"
    / "best.pt"
)


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATASET_DIR / "data.yaml"),
        split="val"
    )

    print("\n===== DETECTION RESULTS =====")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()