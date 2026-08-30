from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO


def detect_and_crop(
    model_path,
    image_path,
    output_dir,
    confidence=0.20
):
    """
    Detect the number plate in a vehicle image
    and save the highest-confidence plate crop.
    """

    model_path = Path(model_path)
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(str(model_path))

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Run detection
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        verbose=False
    )[0]

    # Check detections
    if len(result.boxes) == 0:
        print("No number plate detected.")
        return None

    # Highest-confidence detection
    best_idx = result.boxes.conf.argmax().item()

    confidence_score = float(
        result.boxes.conf[best_idx].cpu()
    )

    # Bounding box
    x1, y1, x2, y2 = (
        result.boxes.xyxy[best_idx]
        .cpu()
        .numpy()
        .astype(int)
    )

    # Keep coordinates inside image
    height, width = image.shape[:2]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    # Crop plate
    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:
        print("Invalid plate crop.")
        return None

    # Save crop
    output_path = (
        output_dir /
        f"{image_path.stem}_plate.jpg"
    )

    cv2.imwrite(str(output_path), plate_crop)

    print(f"Number plate detected.")
    print(f"Confidence: {confidence_score:.4f}")
    print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
    print(f"Saved crop: {output_path}")

    return output_path


def main():

    parser = argparse.ArgumentParser(
        description="Detect and crop vehicle number plates."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO model."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input vehicle image."
    )

    parser.add_argument(
        "--output",
        default="outputs/cropped_plates",
        help="Directory for cropped plate."
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.20,
        help="Detection confidence threshold."
    )

    args = parser.parse_args()

    detect_and_crop(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        confidence=args.confidence
    )


if __name__ == "__main__":
    main()