from ultralytics import YOLO

def evaluate_checkpoint(weights_path, model_name):
    print(f"\n==========================================")
    print(f" EVALUATING: {model_name} ({weights_path})")
    print(f"==========================================")
    
    model = YOLO(weights_path)
    
    # Run validation across your test or validation split
    metrics = model.val(split="test")  # Change to split="val" if you don't have a separate test split
    
    print("\n--- PER-CLASS METRICS ---")
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        class_name = metrics.names[class_idx]
        ap50 = metrics.box.ap50[i]
        ap50_95 = metrics.box.ap[i]
        print(f"Class [{class_idx}] {class_name:<20} | AP@0.50: {ap50*100:6.2f}% | AP@0.50:0.95: {ap50_95*100:6.2f}%")
        
    print("\n--- DATASET OVERALL MEAN METRICS ---")
    print(f"mAP@0.50      : {metrics.box.map50 * 100:.2f}%")
    print(f"mAP@0.50:0.95 : {metrics.box.map * 100:.2f}%")
    print(f"Precision (P) : {metrics.box.mp * 100:.2f}%")
    print(f"Recall (R)    : {metrics.box.mr * 100:.2f}%")
    print(f"Inference Time: {metrics.speed['inference']:.2f} ms")
    print(f"NMS Time      : {metrics.speed['nms']:.2f} ms")
    print(f"Preprocess    : {metrics.speed['preprocess']:.2f} ms")

if __name__ == "__main__":
    # 1. Your attention model
    evaluate_checkpoint("best.pt", "YOLOv8s-CBAM")
    
    # 2. Your baseline model (if you have the standard yolov8s.pt weights trained on the same data)
    # evaluate_checkpoint("baseline_yolov8s.pt", "YOLOv8s Baseline")