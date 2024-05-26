import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class AugmentationFactory(nn.Module):
    def __init__(self, augmentation_type="shift", rotate_angle=4, pad=4, contrast_factor=2, scale_factor=1.2):
        super().__init__()
        # Dictionary to hold augmentation strategies
        self.augmentations = {
            "shift": RandomShiftsAug(pad=pad),
            "rotate": RotateDegrees(angle=rotate_angle),
            "contrast": IncreaseContrast(factor=contrast_factor),
            "zoom": Zoom(scale_factor=scale_factor)
        }
        # Set the current augmentation based on the type provided
        self.set_augmentation(augmentation_type)

    def set_augmentation(self, augmentation_type):
        # Set the augmentation strategy based on the input argument
        if augmentation_type in self.augmentations:
            self.augmentation = self.augmentations[augmentation_type]
        else:
            raise ValueError(f"Augmentation type {augmentation_type} is not supported")

    def forward(self, x):
        # Apply the selected augmentation strategy
        return self.augmentation(x)


class AugmentationStrategy(nn.Module):
    def forward(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")


class RandomShiftsAug(AugmentationStrategy):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class RotateDegrees(AugmentationStrategy):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle  # Rotation angle in degrees

    def forward(self, x):
        n, c, h, w = x.size()
        # Randomly decide to rotate clockwise or counterclockwise
        direction = torch.randint(0, 2, (1,)) * 2 - 1  # Generates 0 or 1, then maps to -1 or 1
        angle_rad = self.angle * direction * torch.pi / 180  # Converts angle to radians and applies direction
        angle_rad = torch.as_tensor(angle_rad).float()
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        rotation_matrix = torch.tensor([[cos_a, -sin_a, 0],
                                        [sin_a, cos_a, 0]],
                                       device=x.device,
                                       dtype=x.dtype)
        rotation_matrix = rotation_matrix.repeat(n, 1, 1)
        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='nearest', padding_mode='reflection', align_corners=False)


class IncreaseContrast(AugmentationStrategy):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor  # Factor by which to increase the contrast

    def forward(self, x):
        # Assuming x is a tensor of shape (N, C, H, W) with values in [0, 1]
        mean = x.mean(dim=(2, 3), keepdim=True)  # Calculate mean per channel

        # Apply contrast formula
        x = (x - mean) * self.factor + mean
        # x = torch.clamp(x, 0, 1)  # Clamp values to maintain valid image range [0, 1]
        return x


class Zoom(AugmentationStrategy):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor  # Zoom-in scale factor

    def forward(self, x):
        n, c, h, w = x.size()
        # Inverse the scale factor for zooming in
        scale_factor = 1 / self.scale_factor

        # Create a scaling transformation matrix
        zoom_matrix = torch.tensor([[scale_factor, 0, 0],
                                    [0, scale_factor, 0]],
                                   device=x.device,
                                   dtype=x.dtype)
        zoom_matrix = zoom_matrix.repeat(n, 1, 1)

        # Create the grid for affine transformation
        grid = F.affine_grid(zoom_matrix, x.size(), align_corners=False)

        # Apply the grid transformation to the input image
        zoomed_in_image = F.grid_sample(x, grid, mode='nearest', padding_mode='border', align_corners=True)
        return zoomed_in_image


class Augmenter(nn.Module):
    def __init__(self, strategy: AugmentationStrategy):
        super().__init__()
        self.strategy = strategy

    def forward(self, x):
        return self.strategy(x)
