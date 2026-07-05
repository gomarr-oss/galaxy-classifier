import h5py 
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
print("Loading dataset...")
with h5py.File("data/Galaxy10_DECals.h5", "r") as f:
    images = f["images"][:]
    labels = f["ans"][:]

print (f"Loaded {len(images)} images")

# Normalize pixel values from 0-255 to 0-1
print("Normalizing images...")
images = images.astype(np.float32) / 255.0

# First split: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split ( images, labels, test_size = 0.15, random_state = 42, stratify = labels )

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split ( X_temp, y_temp, test_size = 0.176, random_state = 42, stratify = y_temp )

print("Saving splits...")
np.save("data/X_train.npy", X_train)
np.save("data/X_val.npy", X_val)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_val.npy", y_val)
np.save("data/y_test.npy", y_test)

print("Done! Splits saved to data/")