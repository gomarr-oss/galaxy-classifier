import torch 
import torch.nn as nn 

class GalaxyClassifier(nn.Module):
    def __init__(self):
        super(GalaxyClassifier, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1, scans whole image
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2, scans block 1 output
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3, scans block 2 output
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x