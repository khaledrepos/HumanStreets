from pathlib import Path

train_path = Path("datasets/sidewalk_segmentation/images/train")
val_path = Path("datasets/sidewalk_segmentation/images/val")

def count_files(p):
    if not p.exists():
        return 0
    return sum(1 for _ in p.glob('*') if _.is_file())

train_count = count_files(train_path)
val_count = count_files(val_path)

print(f"Train images: {train_count}")
print(f"Val images: {val_count}")
print(f"Total: {train_count + val_count}")
