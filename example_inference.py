from ultralytics import YOLO
import cv2
import os

def run_inference(image_path="assets/demo_cards.jpg"):
    """
    Run CardEYE inference on a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Please provide a valid image path.")
        return

    # Load CardEYE model (YOLOv8x)
    # Ensure you have 'cardeye.pt' in the models/ directory
    model_path = "models/cardeye.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Ensure the model file is in the models/ directory.")
        return

    model = YOLO(model_path)

    # Run inference
    # imgsz=1280 is recommended for maximum accuracy
    results = model(image_path, imgsz=1280)

    # Process and display results
    for result in results:
        # Plot detections on the frame
        annotated_frame = result.plot()
        
        # Save or display
        output_path = "results_inference.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"Results saved to {output_path}")

        # Print detections
        for box in result.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            name = model.names[class_id]
            print(f"Found: {name} with {conf:.2%} confidence")

if __name__ == "__main__":
    # Example usage
    # run_inference()
    print("CardEYE Example Inference Script")
    print("-------------------------------")
    print("To run, ensure models/cardeye.pt exists and call run_inference('your_image.jpg')")
