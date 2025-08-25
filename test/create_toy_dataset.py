import os
from PIL import Image
import numpy as np

def create_toy_dataset(base_dir="datasets/toy_dataset", n_train=5, n_test=2):
    subfolders = ["trainA", "trainB", "testA", "testB"]
    for sub in subfolders:
        folder = os.path.join(base_dir, sub)
        os.makedirs(folder, exist_ok=True)

        # Decide how many images to create
        n_images = n_train if "train" in sub else n_test

        for i in range(n_images):
            # Create a random color image (64x64 pixels)
            img_array = np.random.randint(0, 128, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save it as JPEG
            img.save(os.path.join(folder, f"img_{i}.jpg"))

    print(f"Toy dataset created at: {base_dir}")


if __name__ == "__main__":
    create_toy_dataset()
