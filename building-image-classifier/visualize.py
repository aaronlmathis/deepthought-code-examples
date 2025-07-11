import numpy as np
import matplotlib.pyplot as plt

# Load the compressed training dataset
data = np.load("processed/prepared_train_dataset.pt")
X_train = data["X_train"]
y_train = data["y_train"]

# Optional: if you saved class names separately
# class_names = data["class_names"]

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Display 12 random images
plt.figure(figsize=(10, 6))
for i in range(12):
    idx = np.random.randint(0, len(X_train))
    img = X_train[idx]
    label = y_train[idx]
    class_name = class_names[label]

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()