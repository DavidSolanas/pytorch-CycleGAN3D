import os
import nibabel as nib
from PIL import Image
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt

# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



def compute_fid_numpy():
    # Assuming you have two sets of images: real_images_A, generated_images_A, real_images_B, generated_images_B

    dir_path = 'C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/OASIS3_final/FakeAndRealT1/'

    real_images_A = []
    generated_images_A = []

    for img in os.listdir(os.path.join(dir_path, 'fake')):
        x = nib.load(os.path.join(dir_path, 'fake', img)).get_fdata()

        x = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))

        # Get the indices for the central slices along each axis
        x_slice = x.shape[2] // 2

        # Extract the vectors for the central slices
        x_slice_data =x[:, :, x_slice]
        
        generated_images_A.append(x_slice_data)


    for img in os.listdir(os.path.join(dir_path, 'real')):
        x = nib.load(os.path.join(dir_path, 'real', img)).get_fdata()

        # Get the indices for the central slices along each axis
        x_slice = x.shape[2] // 2

        # Extract the vectors for the central slices
        x_slice_data =x[:, :, x_slice]
        
        real_images_A.append(x_slice_data)


    fids = []
    for real, fake in zip(real_images_A, generated_images_A):
        fid = calculate_fid(real, fake)
        # Create a figure with 1 row and 2 columns
        '''plt.figure(figsize=(10, 5))

        # Plot the first image
        plt.subplot(1, 3, 1)
        plt.imshow(real, cmap='gray')  # Use 'cmap' suitable for your image data
        plt.title('Real')

        # Plot the second image
        plt.subplot(1, 3, 2)
        plt.imshow(fake, cmap='gray')  # Use 'cmap' suitable for your image data
        plt.title('Fake')

        plt.subplot(1, 3, 3)
        plt.imshow(real - fake, cmap='gray')  # Use 'cmap' suitable for your image data
        plt.title('Diff')

        plt.show()'''
        fids.append(fid)

    fids = numpy.array(fids)

    print(f'FID : {fids.mean():.3f} ({fids.std():.3f})')



def compute_fid_numpy_3slices():
    # Assuming you have two sets of images: real_images_A, generated_images_A, real_images_B, generated_images_B

    dir_path = 'C:/Users/david/Documents/Master Ingenieria Informatica/TFM/dataset/InfersCycleGANImages/'

    real_images_A = []
    generated_images_A = []

    files_in_directory = os.listdir(dir_path)
    # Filter files that end with "_fakeB.png"
    real_A_files = [file for file in files_in_directory if file.endswith("_real_B.png")]
    generated_A_files = [file for file in files_in_directory if file.endswith("_real_A.png")]


    for img in real_A_files:
        x = Image.open(os.path.join(dir_path, img))
        x = x.convert('L')
        x = numpy.array(x)
        x = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))
        
        real_images_A.append(x)


    for img in generated_A_files:
        x = Image.open(os.path.join(dir_path, img))
        x = x.convert('L')
        x = numpy.array(x)
        x = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x))
        
        generated_images_A.append(x)


    fids = []
    for real, fake in zip(real_images_A, generated_images_A):
        fid = calculate_fid(real, fake)
        # Create a figure with 1 row and 2 columns
        '''plt.figure(figsize=(10, 5))

        # Plot the first image
        plt.subplot(1, 3, 1)
        plt.imshow(real, cmap='gray')  # Use 'cmap' suitable for your image data
        plt.title('Real')

        # Plot the second image
        plt.subplot(1, 3, 2)
        plt.imshow(fake, cmap='gray')  # Use 'cmap' suitable for your image data
        plt.title('Fake')

        plt.subplot(1, 3, 3)
        plt.imshow(real - fake, cmap='gray')  # Use 'cmap' suitable for your image data
        plt.title('Diff')

        plt.show()'''
        fids.append(fid)

    fids = numpy.array(fids)

    print(f'FID : {fids.mean():.3f} ({fids.std():.3f})')



if __name__ == '__main__':

    compute_fid_numpy_3slices()



