import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TextileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for label, cls_name in enumerate(self.classes):
            train_dir = os.path.join(root_dir, cls_name, 'train', 'good')
            if os.path.exists(train_dir):
                for img_name in os.listdir(train_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(train_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define Data Augmentation and Normalization Pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # Use the newly generated balanced dataset path
    dataset_path = r"C:\Users\Nvidia\Downloads\Professional_Textile_Dataset"
    
    if os.path.exists(dataset_path):
        train_dataset = TextileDataset(root_dir=dataset_path, transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        print("--- PyTorch DataLoader Initialized ---")
        print(f"Total Balanced Images Loaded: {len(train_dataset)}")
        print(f"Total Batches (Batch Size 32): {len(train_loader)}")
        print("Ready for Model Training.")
    else:
        print("Dataset not found. Please run dataset_preparation.py first.")