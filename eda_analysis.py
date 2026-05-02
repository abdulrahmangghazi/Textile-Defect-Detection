import os

# Set the path to the original dataset
dataset_path = r"C:\Users\Nvidia\Downloads\ITD-20260430T141146Z-3-001\ITD"

print("--- Exploratory Data Analysis (EDA) ---")
total_images = 0

if not os.path.exists(dataset_path):
    print("Error: Dataset directory not found.")
else:
    # Iterate through class folders
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        if os.path.isdir(folder_path) and folder_name != 'Samples':
            print(f"\nClass: '{folder_name}'")
            class_total = 0
            
            # Iterate through subfolders (train, test, ground_truth)
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                
                if os.path.isdir(subfolder_path):
                    subfolder_count = 0
                    for root, dirs, files in os.walk(subfolder_path):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                subfolder_count += 1
                                
                    print(f"  -> {subfolder_name}: {subfolder_count} images")
                    class_total += subfolder_count
            
            print(f"  [Total for '{folder_name}': {class_total} images]")
            total_images += class_total

    print("\n-------------------------------------------------")
    print(f"Total images in the dataset: {total_images}")