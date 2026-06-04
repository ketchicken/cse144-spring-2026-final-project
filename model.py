import torch
import torch.nn as nn
from torchvision import models

# Model Building
class ENetV2STN(nn.Module):
    def __init__(self, num_classes=100, dpr=0.5):
        super(ENetV2STN, self).__init__()

        # Spatial transformer localization-network, from pytorch tutorials
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((3, 3))    # Fix size for FC localization
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Set up efficientnet base model
        self.base_model = models.efficientnet_v2_s(weights='DEFAULT')
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
                                        nn.Dropout(p=dpr, inplace=True), # Hyperparameters: p=0.5
                                        nn.Linear(in_features=num_ftrs, out_features=num_classes), # output resized to fit our dataset
                                        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nn.functional.affine_grid(theta, x.size())
        x = nn.functional.grid_sample(x, grid)

        return x

    def forward(self, x):
            # transform the input
            #x_transform = self.stn(x)
            # feed it into the model
            x = self.base_model(x)

            return x
