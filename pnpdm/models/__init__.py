from .edm.edm import create_edm_from_unet_adm
from .denoisecnn.denoisecnn import DenoiseCNN
from .denoising_generator.denoising_generator import create_denoising_generator

def get_model(name: str, **kwargs):
    if name == 'edm_from_unet_adm':
        return create_edm_from_unet_adm(**kwargs)
    elif name == 'denoise_cnn':
        return DenoiseCNN(**kwargs)
    elif name == 'denoising_generator':
        return create_denoising_generator(**kwargs)
    else:
        raise NameError(f"Model {name} is not defined.")
