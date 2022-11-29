# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import time

def main():
    # Initialize TensorFlow.
    # tflib.init_tf()
    tflib.init_tf({'gpu_options.allow_growth': True})

    # Load pre-trained network.
    # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl

    # create folder if not exist
    os.makedirs(config.generation_dir, exist_ok=True)

    f = open(config.model_dir, 'rb')
    _G, _D, Gs = pickle.load(f)

    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    print('synthetic data size : ', config.synthetic_data_size)

    for i in range(config.synthetic_data_size):
        print('inside loop with i value : ', i)

        # Pick latent vector.
        rnd = np.random.RandomState(5)
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # Save image.
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        png_filename = os.path.join(config.generation_dir, 'example_' +
                                    str(time.time()*1000) + '_' + time_stamp + '.png')

        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        # gray_image = rgb_image.convert('L')
        # gray_image

if __name__ == "__main__":
    main()
