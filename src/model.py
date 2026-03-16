"""
ResNet-18 for CIFAR-10 — Built from scratch in PyTorch
======================================================

ResNet-18 uses the same ideas as ResNet-50 (skip connections, batch norm,
progressive downsampling), but with a simpler building block:

    ResNet-50: "Bottleneck" block — 1x1 → 3x3 → 1x1 (3 conv layers, squeeze-expand)
    ResNet-18: "Basic" block     — 3x3 → 3x3         (2 conv layers, straightforward)

ResNet-50: 16 bottleneck blocks × 3 convs = 48 + opening + FC = 50 layers, 23.5M params
ResNet-18: 8 basic blocks × 2 convs = 16 + opening + FC = 18 layers, 11.2M params

Same architecture family, same skip connections, same MLOps workflow.
We'll swap to ResNet-50 when you move to cloud GPUs.

TF EQUIVALENT:
    TF doesn't ship ResNet18 in tf.keras.applications — you'd build it manually.
    This is one reason PyTorch is popular in research: more flexibility out of the box.

KEY DIFFERENCE: TF vs PyTorch channel ordering
    TF:      (batch, height, width, channels)  — "channels last"
    PyTorch: (batch, channels, height, width)  — "channels first"
"""

import torch
import torch.nn as nn


# =============================================================================
# BUILDING BLOCK: BasicBlock (simpler than Bottleneck)
# =============================================================================
# ResNet-50's Bottleneck: 1x1 → 3x3 → 1x1 (squeeze, process, expand)
# ResNet-18's BasicBlock: 3x3 → 3x3          (process, process)
#
# The skip connection works identically — add input to output.
#
# TF EQUIVALENT (manual Keras):
#   x = Conv2D(channels, 3, padding='same')(input)
#   x = BatchNormalization()(x)
#   x = Activation('relu')(x)
#   x = Conv2D(channels, 3, padding='same')(x)
#   x = BatchNormalization()(x)
#   x = Add()([x, input])        <-- skip connection
#   x = Activation('relu')(x)

class BasicBlock(nn.Module):
    """
    Two 3x3 convolutions with a skip connection.
    
    Parameters:
        in_channels:  Channels coming in
        out_channels: Channels going out
        stride:       1 = same spatial size, 2 = halve spatial size
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Conv 1: 3x3, may downsample if stride=2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv 2: 3x3, always stride 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection — 1x1 conv if shapes don't match
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.skip is not None:
            identity = self.skip(identity)
        
        out = out + identity
        out = self.relu(out)
        return out


# =============================================================================
# THE FULL MODEL: ResNet-18 for CIFAR-10
# =============================================================================

class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images, 10 classes).
    
    Same CIFAR-10 adaptations as our ResNet-50:
    - 3x3 opening conv (not 7x7) with stride 1
    - No max pooling after opening
    - 10 output classes
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Opening: 3x3 stride 1, no max pool (CIFAR-10 adaptation)
        self.opening = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 4 layers: [2, 2, 2, 2] basic blocks
        self.layer1 = self._make_layer(in_ch=64,  out_ch=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(in_ch=64,  out_ch=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(in_ch=128, out_ch=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(in_ch=256, out_ch=512, blocks=2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride=stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.opening(x)   # (batch, 64, 32, 32)
        x = self.layer1(x)    # (batch, 64, 32, 32)
        x = self.layer2(x)    # (batch, 128, 16, 16)
        x = self.layer3(x)    # (batch, 256, 8, 8)
        x = self.layer4(x)    # (batch, 512, 4, 4)
        x = self.avgpool(x)   # (batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch, 512)
        x = self.fc(x)        # (batch, 10)
        return x


if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-18 for CIFAR-10")
    print(f"Total parameters: {total_params:,}")
    
    fake_input = torch.randn(4, 3, 32, 32)
    output = model(fake_input)
    print(f"Input shape:  {fake_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (raw logits): {output[0].detach().numpy().round(3)}")
    print("\nModel is working correctly!")
