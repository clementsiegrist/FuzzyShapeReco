import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from shape_reco.clustering import cmeans

def change_color_fuzzycmeans(cluster_membership, clusters):
    '''

    :param cluster_membership:
    :param clusters:
    :return:
    '''
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

def read_image(path):
    '''

    :param path:
    :return:
    '''
    folder = path
    list_images = os.listdir(folder)
    list_img = []
    for i in list_images:
        img = cv2.imread(os.path.join(folder, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = img.reshape((img.shape[0] * img.shape[1], 1)) #((img.shape[0] * img.shape[1], 3))
        list_img.append(rgb_img)

    return list_img, img.shape

def readimage_3D(path):
    '''

    :param path:
    :return:
    '''
    folder = path
    list_images = os.listdir(folder)
    list_img = []
    for i in list_images:
        img = cv2.imread(os.path.join(folder, i))
        rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
        list_img.append(rgb_img)

    return list_img, img.shape

def fuzzy_cmeanscolor(path, num_clusters):
    '''

    :param path:
    :param num_clusters:
    :return:
    '''

    list_img, shape = readimage_3D(path)
    n_data = len(list_img)
    num_clusters = num_clusters  # [1, 2, 3]

    # Iterates over previously loaded images
    for index, rgb_img in enumerate(list_img):
        img = np.reshape(rgb_img, shape).astype(np.uint8)

        # Iterate over each cluster
        for i, num_clusters in enumerate(num_clusters):
            print('-----------------------------------------------------------')
            print('Image ' + str(index + 1) + '  Nb cluster' + str(index))
            centers, u_fonc, u_fonc0, distance, hist, num_iter, fuzzy_part_coeff = cmeans(
                rgb_img.T, num_clusters, 2, error=0.05, maxiter=10000, init=None, seed=42)

            # Change each pixel color according
            new_img = change_color_fuzzycmeans(u_fonc, centers)
            new_img = np.reshape(new_img,shape).astype(np.uint8)

            # Create plot
            fig, axs = plt.subplots(1, 2, figsize=(5, 5))
            axs[0].imshow(np.reshape(rgb_img, shape))
            axs[1].imshow(new_img, cmap='gray')
            plt.show()

def fuzzy_cmeansgrey(path, num_clusters):
    '''

    :param path:
    :param num_clusters:
    :return:
    '''

    # looping every images
    list_img, shape = read_image(path)
    n_data = len(list_img)
    clusters = num_clusters
    for index, rgb_img in enumerate(list_img):

        # looping every cluster
        print('Image ' + str(index + 1))
        for i, cluster in enumerate(clusters):
            cntr, u, u0, d, jm, p, fpc = cmeans(
                rgb_img.T, cluster, 2, error=0.05, maxiter=10000, init=None, seed=42)

            new_img = change_color_fuzzycmeans(u, cntr)
            new_img = np.reshape(new_img, shape).astype(np.uint8)
            # Create plot
            fig, axs = plt.subplots(1, 2, figsize=(5, 5))
            axs[0].imshow(np.reshape(rgb_img, shape))
            axs[1].imshow(new_img, cmap='gray')
            plt.show()

if __name__ == '__main__':

    pipeline('/Users/clementsiegrist/untitled7/shape_dir')