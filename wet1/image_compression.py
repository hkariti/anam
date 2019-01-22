import random
import numpy as np
from sklearn.decomposition import PCA

def break_image(image, height=6, width=6):
    width_count = int(image.shape[1]/width)
    height_count = int(image.shape[0]/height)
    patches = []
    for w in range(width_count + 1):
        for h in range(height_count + 1):
            start_w_index = w*width
            end_w_index = (w+1)*width
            start_h_index = h*height
            end_h_index = (h+1)*height
            if h == height_count and w == width_count:
                patch = image[-height:, -width:]
            elif w == width_count:
                patch = image[start_h_index:end_h_index, -width:]
            elif h == height_count:
                patch = image[-height:, start_w_index:end_w_index]
            else:
                patch = image[start_h_index:end_h_index, start_w_index:end_w_index]
            patches.append(patch)
    return patches

def restore_image(patches, image_width=47, image_height=62):
    patch_height, patch_width = patches[0].shape
    width_count = int(image_width / patch_width)
    height_count = int(image_height / patch_height)
    image = np.zeros((image_height, image_width))
    patch_index = 0
    for w in range(width_count + 1):
        for h in range(height_count + 1):
            patch = patches[patch_index]
            patch_index += 1
            start_w_index = w*patch_width
            end_w_index = (w+1)*patch_width
            start_h_index = h*patch_height
            end_h_index = (h+1)*patch_height
            if h == height_count and w == width_count:
                image[-patch_height:, -patch_width:] = patch
            elif w == width_count:
                image[start_h_index:end_h_index, -patch_width:] = patch
            elif h == height_count:
                image[-patch_height:, start_w_index:end_w_index] = patch
            else:
                image[start_h_index:end_h_index, start_w_index:end_w_index] = patch
    return image

def create_patches_db(normalized_images, num_of_patches=1500):
    patches_db = np.zeros((num_of_patches, 36))
    for i in range(num_of_patches):
        random_image_index = np.random.randint(0, normalized_images.shape[0])
        random_image = normalized_images[random_image_index].reshape(62, 47)
        patches = break_image(random_image)
        random_patch = random.choice(patches)
        patches_db[i] = random_patch.reshape(36)
    return patches_db

def compress_restore_image(image, pca_obj):
    patches = break_image(image)
    patches_restored = []
    for p in patches:
        patch_lowd = pca_obj.transform(p.reshape(1, -1))
        patch_restored = pca_obj.inverse_transform(patch_lowd)
        patches_restored.append(patch_restored.reshape((6,6)))
    return restore_image(patches_restored)

def calc_quality(images, d, patches_dataset, num_of_imgs_for_avg=200):
    pca_obj = PCA(d)
    pca_obj.fit(patches_dataset)
    cr = (88 * d) / images[0].shape[0]

    distances = np.zeros(num_of_imgs_for_avg)
    for i in range(num_of_imgs_for_avg):
        image = images[np.random.randint(0, images.shape[0])]
        restored_image = compress_restore_image(image.reshape(62, 47), pca_obj)
        dist = np.abs(image-restored_image.reshape(-1)).mean()
        distances[i] = dist
    error = distances.mean()
    quality = 1- error - 0.04 * cr
    return quality, error, cr

if __name__ == '__main__':
    from sklearn.datasets import fetch_lfw_people
    from matplotlib import pyplot as plt
    print("Fetching dataset")
    dataset = fetch_lfw_people()
    normalized_images = dataset['data']/255
    patches_dataset = create_patches_db(normalized_images)
    d = 5
    patch_size = 36
    quality = np.zeros(patch_size)
    error = np.zeros(patch_size)
    cr = np.zeros(patch_size)
    for d in range(patch_size):
        quality[d], error[d], cr[d] = calc_quality(normalized_images, d, patches_dataset)
        print("d={}, cr={:.3f}, error={:.3f}, quality={:.3f}".format(d, cr[d], error[d], quality[d]))

    print("Best D value is", np.argmax(quality))
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(list(range(patch_size)), quality)
    plt.subplot(3, 1, 2)
    plt.plot(list(range(patch_size)), error)
    plt.subplot(3, 1, 3)
    plt.plot(list(range(patch_size)), cr)
    plt.show()
