# test_denoising_generator.py

import pytest
import torch
import math

# Ensure the new DenoisingGenerator is imported
from .denoising_generator import DenoisingGenerator


def test_generator_output_shape_64():
    """
    Tests the forward pass for a 64x64 image, mimicking the original
    model's depth.
    """
    batch_size = 2
    eta_dim = 100
    in_channels = 3
    out_channels = 3
    img_size = 64

    # 6 total downsamples to get 64 -> 1x1
    # This means 5 encoder levels + 1 bottleneck level
    num_levels = 5
    # (1, 2, 4, 8, 16) - 5 levels
    channel_mult = tuple(2**i for i in range(num_levels))

    model = DenoisingGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        eta_dim=eta_dim,
        model_channels=64, # Original base channels
        channel_mult=channel_mult # (1, 2, 4, 8, 16)
    )
    model.eval()

    x = torch.randn(batch_size, in_channels, img_size, img_size)
    eta = torch.randn(batch_size, eta_dim)

    with torch.no_grad():
        output = model(x, eta)

    # 1. Check shape
    expected_shape = (batch_size, out_channels, img_size, img_size)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"

    # 2. Check range
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0), \
        "Output values are not within the expected [-1, 1] range."


def test_generator_output_shape_256_efficient():
    """
    Tests the forward pass for a 256x256 image using a
    memory-efficient configuration.
    """
    batch_size = 1 # Use B=1 for 256x256 tests
    eta_dim = 100
    in_channels = 3
    out_channels = 3
    img_size = 256

    # 8 total downsamples to get 256 -> 1x1
    # This means 7 encoder levels + 1 bottleneck level
    num_levels = 7

    # Memory-efficient channels: (1, 2, 4, 8, 8, 16, 16)
    # Max channels = 32 * 16 = 512. Bottleneck = 1024.
    # This is much smaller than 32 * 64 = 2048.
    channel_mult = (1, 2, 4, 8, 8, 16, 16)

    model = DenoisingGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        eta_dim=eta_dim,
        model_channels=32, # Start with fewer base channels
        channel_mult=channel_mult
    )
    model.eval()

    x = torch.randn(batch_size, in_channels, img_size, img_size)
    eta = torch.randn(batch_size, eta_dim)

    with torch.no_grad():
        output = model(x, eta)

    expected_shape = (batch_size, out_channels, img_size, img_size)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"


def test_generator_grayscale_128():
    """
    Tests the generator with grayscale channels (1) and a 128x128 image.
    """
    batch_size = 2
    eta_dim = 50
    in_channels = 1
    out_channels = 1
    img_size = 128

    # 7 total downsamples to get 128 -> 1x1
    # This means 6 encoder levels + 1 bottleneck level
    num_levels = 6
    channel_mult = (1, 2, 4, 8, 16, 32)

    model = DenoisingGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        eta_dim=eta_dim,
        model_channels=64,
        channel_mult=channel_mult
    )
    model.eval()

    x = torch.randn(batch_size, in_channels, img_size, img_size)
    eta = torch.randn(batch_size, eta_dim)

    with torch.no_grad():
        output = model(x, eta)

    expected_shape = (batch_size, out_channels, img_size, img_size)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"


def test_generator_non_one_by_one_bottleneck():
    """
    Tests if the model works when the bottleneck is not 1x1,
    which validates the `eta` vector expansion logic.
    """
    batch_size = 1
    eta_dim = 10
    img_size = 64

    # 4 total downsamples (3 levels + bottleneck)
    # 64 -> 32 -> 16 -> 8 -> 4. Bottleneck will be 4x4.
    num_levels = 3
    channel_mult = (1, 2, 4) # 3 levels

    model = DenoisingGenerator(
        in_channels=3,
        out_channels=3,
        eta_dim=eta_dim,
        model_channels=16, # Keep channels small
        channel_mult=channel_mult
    )
    model.eval()

    x = torch.randn(batch_size, 3, img_size, img_size)
    eta = torch.randn(batch_size, eta_dim)

    with torch.no_grad():
        # This forward pass will fail if the eta.expand() logic is wrong
        output = model(x, eta)

    expected_shape = (batch_size, 3, img_size, img_size)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"
