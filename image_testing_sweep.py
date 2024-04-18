import os
import numpy as np
from keras.datasets import mnist
from image_dictionary import ImageDictionary
import random
import matplotlib.pyplot as plt


def plot_images(original_images, recon_images, labels, figname="reconstructed.png"):
    # Select 5 random indices
    random_indices = random.sample(range(len(recon_imgs)), 5)

    fig, axes = plt.subplots(2, 5, figsize=(15, 10))

    for i, ax in enumerate(axes.flatten()):
        # Display the image
        if i < 5:
            ax.imshow(original_images[random_indices[i]], cmap='gray')
            ax.set_title(f"Image {labels[random_indices[i]]}")
        else:
            ax.imshow(recon_images[random_indices[i-5]], cmap='gray')
            ax.set_title(f"Reconstructed {labels[random_indices[i-5]]}")
        ax.axis('off')

    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/"+figname)


if __name__ == "__main__":
    dict_base_name = "dict"
    recon_base_name = "recon"
    dictionary_sizes = [100]#[100, 500, 1000, 5000, 10000]
    patch_sizes = [5]
    strides = [1]

    # Load images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    for dict_size in dictionary_sizes:
        for patch_size in patch_sizes:
            for stride in strides:
                filename = f"{dict_base_name}_{dict_size}_{patch_size}_{stride}.npy"
                if not os.path.exists("codebooks/"+filename):
                    img_dict = ImageDictionary(x_train, n_clusters=dict_size, patch_size=patch_size, stride=stride)
                    img_dict.save(filename)
                else:
                    img_dict = ImageDictionary()
                    img_dict.load(filename)

                img_dict.plot_dictionary(figname=filename[:-4]+".png")

                recon_imgs = img_dict.reconstruct(x_test, save=True, filename=f"{recon_base_name}_{dict_size}_{patch_size}_{stride}.npy")
                plot_images(x_test, recon_imgs, y_test, figname=f"{recon_base_name}_{dict_size}_{patch_size}_{stride}.png")
                
                
                

