import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DenoisingGenerator(nn.Module):
    """
    A configurable and memory-efficient symmetric U-Net generator.

    (Note: Model loading has been moved to the factory function.)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 eta_dim=100,
                 model_channels=64,
                 channel_mult=(1, 2, 4, 8, 16)
                 ):
        """
        Initializes the configurable U-Net.

        Args:
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output image channels.
            eta_dim (int): Dimension of the noise vector `eta`.
            model_channels (int): Base channel count.
            channel_mult (tuple): A tuple of multipliers for each level of the
                                  encoder (determines the depth and width).
        """
        super(DenoisingGenerator, self).__init__()
        self.eta_dim = eta_dim
        self.num_levels = len(channel_mult)

        # --- Build Encoder ---
        self.encoders = nn.ModuleList()
        enc_channels = []

        # First encoder block
        first_enc_channels = model_channels * channel_mult[0]
        self.encoders.append(
            nn.Sequential(
                nn.Conv2d(in_channels, first_enc_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
        )
        enc_channels.append(first_enc_channels)

        # Subsequent encoder blocks
        in_c = first_enc_channels
        for i in range(1, self.num_levels):
            out_c = model_channels * channel_mult[i]
            self.encoders.append(self._make_encoder_block(in_c, out_c))
            enc_channels.append(out_c)
            in_c = out_c

        # --- Bottleneck ---
        max_channels = enc_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max_channels, max_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(max_channels * 2)
        )

        # --- Build Decoder ---
        self.decoders = nn.ModuleList()

        # First decoder block (from bottleneck)
        in_c_dec = (max_channels * 2) + eta_dim
        out_c_dec = max_channels
        self.decoders.append(self._make_decoder_block(in_c_dec, out_c_dec))

        # Subsequent decoder blocks (with skip connections)
        reversed_enc_channels = enc_channels[::-1]

        for i in range(self.num_levels - 1):
            in_c_dec = reversed_enc_channels[i] + reversed_enc_channels[i]
            out_c_dec = reversed_enc_channels[i+1]
            self.decoders.append(self._make_decoder_block(in_c_dec, out_c_dec))

        # Final decoder block
        final_in_c = enc_channels[0] + enc_channels[0]
        final_out_c = enc_channels[0]
        self.decoders.append(self._make_decoder_block(final_in_c, final_out_c))

        # --- Final Output Layer ---
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(enc_channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh() # Tanh maps output to [-1, 1].
        )

        # Model loading logic has been removed from __init__
        # and moved to the factory function.

    def _make_encoder_block(self, in_channels, out_channels):
        """Helper to create a standard encoder block."""
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Helper to create a standard decoder block (Upsample + Conv)."""
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, eta):
        # --- Encoder Path ---
        skip_connections = []
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            skip_connections.append(x)

        # --- Bottleneck ---
        bottleneck_out = self.bottleneck(x)

        # --- Eta Injection ---
        eta_reshaped = eta.view(eta.size(0), -1, 1, 1)
        b, c, h, w = bottleneck_out.shape
        eta_reshaped = eta_reshaped.expand(-1, -1, h, w)

        bottleneck_noisy = torch.cat([bottleneck_out, eta_reshaped], dim=1)

        # --- Decoder Path ---
        dec_out = self.decoders[0](bottleneck_noisy)
        reversed_skips = skip_connections[::-1]

        for i in range(self.num_levels):
            skip = reversed_skips[i]
            dec_in = torch.cat([dec_out, skip], dim=1)
            dec_out = self.decoders[i+1](dec_in)

        # --- Final Layer ---
        output = self.final_conv(dec_out)
        return output

def create_denoising_generator(
    image_size,
    num_channels,
    eta_dim=100,
    channel_mult="",
    grayscale=False,
    model_path=None
):
    """
    Factory function to create a DenoisingGenerator.

    This function automatically sets the U-Net depth (`channel_mult`)
    based on the `image_size` to ensure a 1x1 bottleneck, mimicking
    the behavior of `create_unet_adm`.

    Args:
        image_size (int): The size of the input images (e.g., 64, 256).
        num_channels (int): The base channel count, maps to `model_channels`.
        eta_dim (int): Dimension of the noise vector `eta`.
        channel_mult (str): A comma-separated string of channel multipliers.
                            If empty (""), defaults are set based on `image_size`.
        grayscale (bool): If True, sets in/out channels to 1.
        model_path (str, optional): Path to a pre-trained model checkpoint.
    """

    if channel_mult == "":
        # This default logic selects the *depth* of the U-Net.
        # The number of encoder levels is `log2(image_size) - 1`
        # to ensure the bottleneck downsamples to 1x1.

        if image_size == 512:
            # 9 total downsamples -> 8 encoder levels
            channel_mult = (1, 1, 2, 2, 4, 4, 8, 8)
        elif image_size == 256:
            # 8 total downsamples -> 7 encoder levels
            channel_mult = (1, 1, 2, 2, 4, 8, 8)
        elif image_size == 128:
            # 7 total downsamples -> 6 encoder levels
            channel_mult = (1, 2, 2, 4, 4, 8)
        elif image_size == 64:
            # 6 total downsamples -> 5 encoder levels
            channel_mult = (1, 2, 4, 8, 16)
        elif image_size == 32:
             # 5 total downsamples -> 4 encoder levels
            channel_mult = (1, 2, 4, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}. Must be power of 2 >= 32.")
    else:
        # Parse the string, just like the example
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # --- Derived Parameters ---
    in_channels = 1 if grayscale else 3
    out_channels = 1 if grayscale else 3

    # --- Instantiate Model ---
    model = DenoisingGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        eta_dim=eta_dim,
        model_channels=num_channels, # Map factory arg to constructor arg
        channel_mult=channel_mult,
    )

    # --- Handle Model Loading (externally, like the example) ---
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # This logic supports checkpoints saved as dicts (common)
            # or raw state_dicts.
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Loading weights from 'model_state_dict' key in checkpoint.")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Loading weights from raw state_dict checkpoint.")
                model.load_state_dict(checkpoint)

            print(f"Successfully loaded pre-trained DenoisingGenerator from {model_path}")

        except Exception as e:
            print(f"Warning: could not load model from {model_path}. Got exception: {e}")
            print("Initializing model with random weights.")

    return model
