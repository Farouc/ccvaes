# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class CartoonMultiLabelDataset(Dataset):
    """
    PyTorch Dataset for multi-label attribute prediction on the CartoonSet dataset.

    Each image is associated with a CSV file containing attribute annotations.
    The CSV format (no header) is assumed to be:
        column 0: attribute name
        column 1: class index
        column 2: total number of classes for this attribute

    Example attributes:
        ["hair_color", "face_color"]
    """

    def __init__(self, root_dir, target_attributes, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing images and CSV files.
            target_attributes (list[str]): Attributes to extract
                                           (e.g. ["hair_color", "face_color"]).
            transform (callable, optional): Image transformations.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_attributes = target_attributes

        # Sorted list of image files
        self.images = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(".png")]
        )

        # Storage for label vectors
        self.labels = []

        # Track number of classes per attribute
        self.dims_per_attr = {attr: 0 for attr in target_attributes}

        print(f"Loading labels for attributes: {target_attributes} ...")

        for img_file in self.images:
            csv_file = img_file.replace(".png", ".csv")
            csv_path = os.path.join(root_dir, csv_file)

            # Read CSV file (no header)
            # col 0: attribute name
            # col 1: class index
            # col 2: total number of classes
            df = pd.read_csv(csv_path, header=None)

            current_label_vector = []

            for attr in self.target_attributes:
                # Find the row corresponding to the attribute
                row = df[df.iloc[:, 0] == attr]

                if not row.empty:
                    class_index = int(row.iloc[0, 1])
                    max_classes = int(row.iloc[0, 2])

                    current_label_vector.append(class_index)

                    # Update detected number of classes for this attribute
                    # (Expected to be constant across files)
                    self.dims_per_attr[attr] = max_classes
                else:
                    raise ValueError(
                        f"Attribute '{attr}' not found in {csv_file}"
                    )

            self.labels.append(current_label_vector)

        # Convert labels to a tensor of shape (N_images, N_attributes)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # Ordered list of number of classes per attribute
        # Example: ["hair_color", "face_color"] -> [10, 11]
        self.num_classes_list = [
            self.dims_per_attr[attr] for attr in target_attributes
        ]

        print(f"Dataset loaded successfully. Attribute dimensions: {self.dims_per_attr}")

    def __len__(self):
        """Return the total number of images."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load one image and its multi-attribute label vector.

        Returns:
            image (Tensor): Transformed image tensor.
            label (Tensor): Attribute label vector
                            (e.g. tensor([4, 8])).
        """
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
