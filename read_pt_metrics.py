import torch

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)

print("\n" + "="*50)
print("     STORED TRAINING & VALIDATION CURVES")
print("="*50)

# Check train_metrics
if "train_metrics" in ckpt and ckpt["train_metrics"]:
    print("\n--- TRAIN METRICS DICTIONARY ---")
    for k, v in ckpt["train_metrics"].items():
        print(f"  {k}: {v}")

# Check train_results (stores epoch-by-epoch table)
if "train_results" in ckpt and ckpt["train_results"] is not None:
    res = ckpt["train_results"]
    print("\n--- LAST EPOCH TRAIN RESULTS ---")
    if isinstance(res, dict):
        for k, v in res.items():
            last_val = v[-1] if isinstance(v, (list, tuple)) else v
            print(f"  {k:<30}: {last_val}")
    else:
        print(res)
print("="*50)