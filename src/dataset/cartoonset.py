import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CartoonDataset(Dataset):
    """
    Dataset class for CartoonSet handling multiple attributes.
    Can be used for single-label (list of length 1) or multi-label tasks.
    """
    def __init__(self, root_dir, target_attributes, transform=None):
        """
        Args:
            root_dir (str): Path to the image folder.
            target_attributes (list): List of attribute names to extract.
                                      Example: ["hair_color", "face_color"]
            transform (callable, optional): Transformations to apply to the image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_attributes = target_attributes
        
        # List of images (sorted for consistency)
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
        
        self.labels = []
        # Temporary dictionary to store max classes per attribute for consistency check
        self.dims_per_attr = {attr: 0 for attr in target_attributes}

        print(f"Loading labels for: {target_attributes} ...")

        for img_file in self.images:
            csv_file = img_file.replace(".png", ".csv")
            csv_path = os.path.join(root_dir, csv_file)
            
            # Read CSV (no header)
            # col 0: attribute name, col 1: variant index, col 2: total variants
            df = pd.read_csv(csv_path, header=None)
            
            current_label_vector = []
            
            for attr in self.target_attributes:
                # Retrieve the row corresponding to the attribute
                row = df[df.iloc[:, 0] == attr]
                
                if not row.empty:
                    class_index = int(row.iloc[0, 1])
                    max_classes = int(row.iloc[0, 2])
                    
                    current_label_vector.append(class_index)
                    
                    # Update max classes detected for this attribute
                    # (Should be constant across dataset, but we take it from file)
                    self.dims_per_attr[attr] = max_classes
                else:
                    raise ValueError(f"Attribute '{attr}' not found in {csv_file}")

            self.labels.append(current_label_vector)

        # Convert to Long Tensor (N_images, N_attributes)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Ordered list of class counts for each attribute
        # Example for ["hair_color", "face_color"] -> [10, 11]
        self.num_classes_list = [self.dims_per_attr[attr] for attr in target_attributes]
        
        print(f"Dataset loaded. Attributes dimensions: {self.dims_per_attr}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Returns a vector of labels. 
        # Ex: tensor([4, 8]) if hair=4 and face=8
        # If single attribute, it will be tensor([4])
        label = self.labels[idx] 

        return img, label