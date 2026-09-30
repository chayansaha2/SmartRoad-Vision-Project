import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("best.pt")

# Create a standard 640x640 dummy RGB frame
dummy_frame = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))

print("\n==========================================")
print("     SYSTEM HARDWARE & LATENCY PROFILING")
print("==========================================")

# Detect runtime hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Active Compute Device : {device.upper()}")
if device == "cuda":
    print(f"GPU Model             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version          : {torch.version.cuda}")
else:
    import platform
    print(f"Host Processor        : {platform.processor()}")

# Warmup run
_ = model.predict(dummy_frame, verbose=False)

# Profile 50 iterations
preprocess_times, inference_times, nms_times = [], [], []

for _ in range(50):
    res = model.predict(dummy_frame, verbose=False)
    speed = res[0].speed
    preprocess_times.append(speed.get("preprocess", 0.0))
    inference_times.append(speed.get("inference", 0.0))
    nms_times.append(speed.get("nms", 0.0))

avg_prep = np.mean(preprocess_times)
avg_inf = np.mean(inference_times)
avg_nms = np.mean(nms_times)
total_lat = avg_prep + avg_inf + avg_nms
fps = 1000.0 / total_lat if total_lat > 0 else 0.0

print("\n--- TIMING BREAKDOWN (Averaged over 50 frames) ---")
print(f"Preprocessing Latency : {avg_prep:6.2f} ms")
print(f"Forward Pass (Model)  : {avg_inf:6.2f} ms")
print(f"NMS Post-Processing   : {avg_nms:6.2f} ms")
print(f"End-to-End Latency    : {total_lat:6.2f} ms")
print(f"Effective Throughput  : {fps:6.1f} FPS")
print("==========================================\n")