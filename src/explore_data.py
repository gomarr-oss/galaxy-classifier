import h5py
import numpy as np
import matplotlib.pyplot as plt

# Class names for the top 10 galaxies
CLASS_NAMES = [ "Disturbed", "Merging", "Round Smooth", "In-between Smooth", "Cigar Smooth", "Barred Spiral", 
                "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]

# Open the dataset
with h5py.File("data/Galaxy10_DECals.h5", "r") as f:
    print("Keys in file:", list(f.keys()))

    images = f["images"][:]
    labels = f["ans"][:]

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Sample labels:", labels[:10])

# Show 10 random galaxies
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    idx = np.random.randint(0, len(images))
    axes[i].imshow(images[idx])
    axes[i].set_title(CLASS_NAMES[labels[idx]])
    axes[i].axis("off")

plt.tight_layout
plt.savefig("results/galaxy_samples.png")
print("Saved sample images to results/galaxy_samples.png")