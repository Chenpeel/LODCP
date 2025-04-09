import numpy as np
data = np.load("data/processed_data/train/batch_10.npz")
print("Images shape:", data['images'].shape)
print("Masks shape:", data['masks'].shape)
