import os
import random
import cv2

# Configuration
source_path = r"C:\Users\Nvidia\Downloads\ITD-20260430T141146Z-3-001\ITD"
target_path = r"C:\Users\Nvidia\Downloads\Professional_Textile_Dataset"
img_size = (224, 224)

if not os.path.exists(target_path):
    os.makedirs(target_path)

# 1. Find the minimum number of 'good' images in the train set for balancing
classes = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d)) and d != 'Samples']
train_good_counts = []

for cls in classes:
    path = os.path.join(source_path, cls, 'train', 'good')
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))])
        train_good_counts.append(count)

min_train_count = min(train_good_counts)
print(f"--- Balancing Train Data: Limit set to {min_train_count} images per class ---\n")

# 2. Function to process and save images while maintaining directory structure
def process_and_save(src, dst, limit=None):
    if not os.path.exists(src): return
    os.makedirs(dst, exist_ok=True)
    
    files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Apply balancing (undersampling) if a limit is provided
    if limit and len(files) > limit:
        files = random.sample(files, limit)
    
    for f in files:
        img = cv2.imread(os.path.join(src, f))
        if img is not None:
            # Resize image to target dimensions
            resized = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(dst, f), resized)

# 3. Process all classes
for cls in classes:
    print(f"Processing class: {cls}...")
    
    # A. Process Train set (Good images only - Balanced)
    process_and_save(os.path.join(source_path, cls, 'train', 'good'), 
                     os.path.join(target_path, cls, 'train', 'good'), limit=min_train_count)
    
    # B. Process Test set (Anomaly and Good images - Not Balanced)
    process_and_save(os.path.join(source_path, cls, 'test', 'anomaly'), 
                     os.path.join(target_path, cls, 'test', 'anomaly'))
    process_and_save(os.path.join(source_path, cls, 'test', 'good'), 
                     os.path.join(target_path, cls, 'test', 'good'))
    
    # C. Process Ground Truth (Masks for segmentation)
    process_and_save(os.path.join(source_path, cls, 'ground_truth'), 
                     os.path.join(target_path, cls, 'ground_truth'))

print(f"\nDataset preparation completed successfully. Saved at: {target_path}")