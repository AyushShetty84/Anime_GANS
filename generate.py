import os
import torch
import torchvision.utils as vutils
from model import Generator


def generate_images(generator, device, save_dir, n_images=64, channels_img=3, features_gen=64):
    # Generate n_images from the generator
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_images, 100, 1, 1).to(device)
        generated_images = generator(noise)

    # Save the generated images
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(generated_images):
        image = (image + 1) / 2  # Denormalize the image
        image = image.clamp(0, 1)  # Clip pixel values to [0, 1]
        image = image.cpu()  # Move the tensor to CPU
        filename = os.path.join(save_dir, f"generated_{i}.png")
        vutils.save_image(image, filename, normalize=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHANNELS_IMG = 3
NOISE_DIM = 100
FEATURES_GEN = 64

# Create generator
generator = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

# Load checkpoint
checkpoint = torch.load('models/checkpoint_400.tar', map_location=device)
generator.load_state_dict(checkpoint['gen_state_dict'])

# Generate images
save_path = 'generated_images/image_epoch400.png'
generate_images(generator, device, save_path, channels_img=CHANNELS_IMG, features_gen=FEATURES_GEN)
