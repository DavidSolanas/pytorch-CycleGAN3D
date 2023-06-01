"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import nibabel as nib


def tensor2im(input_volume, imtype=np.uint8):
    """
    Convierte un tensor 3D de PyTorch de dimensiones 256x256x256 en una imagen 2D de numpy.

    Parámetros:
        input_volume (tensor) -- el tensor de volumen de entrada de PyTorch de dimensiones 256x256x256
        imtype (type)        -- el tipo de dato deseado para el array de numpy convertido

    Returns:
        Tres imágenes 2D de numpy correspondientes a las slices centrales en las tres dimensiones del volumen de entrada.
    """
    if not isinstance(input_volume, np.ndarray):
        if isinstance(input_volume, torch.Tensor):  # obtener los datos de una variable
            volume_tensor = input_volume.data
        else:
            return input_volume
        volume_numpy = volume_tensor[0].cpu().float().numpy()  # convertir a un array de numpy

        volume_numpy = (np.transpose(volume_numpy, (1, 0, 2)) + 1) / 2.0 * 255.0  # post-procesamiento: transposición y escalado
        volume_numpy = np.flip(volume_numpy, axis=0)
        volume_numpy = np.flip(volume_numpy, axis=2)
        # re-normalizar para tener mismo rango todas las imágenes
        volume_numpy = (volume_numpy - volume_numpy.min()) * (255.0 / (volume_numpy.max() - volume_numpy.min()))

    else:  # si es un array de numpy, no hacer nada
        volume_numpy = (np.transpose(input_volume, (1, 0, 2)) + 1) / 2.0 * 255.0
        volume_numpy = np.flip(volume_numpy, axis=0)
        volume_numpy = np.flip(volume_numpy, axis=2)
        volume_numpy = (volume_numpy - volume_numpy.min()) * (255.0 / (volume_numpy.max() - volume_numpy.min()))
        
    return volume_numpy.astype(imtype)


def tensor3d2im(input_volume: torch.Tensor, imtype=np.uint8):
    """"Converts a 3D Tensor array into a numpy image array. It returns
    the 3 central slices of the volume.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_volume, np.ndarray):
        if isinstance(input_volume, torch.Tensor):  # get the data from a variable
            # Convert back the tensor
            dtype = input_volume.dtype
            mean = torch.as_tensor((0.5,), dtype=dtype, device=input_volume.device)
            std = torch.as_tensor((0.5,), dtype=dtype, device=input_volume.device)
            input_volume = input_volume.mul_(std).add_(mean)
            image_tensor = input_volume.data
        else:
            return input_volume
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array

    else:  # if it is a numpy array, do nothing
        image_numpy = input_volume

    print(image_numpy.min(), image_numpy.max())
    max_val = np.max(image_numpy)
    min_val = np.min(image_numpy)
    if max_val - min_val > 0:
        image_numpy = (image_numpy - min_val) / (max_val - min_val)

    shape = image_numpy.shape
    # Get x, y, and z slices
    x_slice = image_numpy[shape[0] // 2, :, :]  # Middle slice along x-axis
    y_slice = image_numpy[:, shape[1] // 2, :]  # Middle slice along y-axis
    z_slice = image_numpy[:, :, shape[2] // 2]  # Middle slice along z-axis

    # Normalize the slices to the range [0, 1]

    # Convert the slices to RGB images
    x_slice_rgb = np.stack((x_slice, x_slice, x_slice), axis=-1) * 255.0
    y_slice_rgb = np.stack((y_slice, y_slice, y_slice), axis=-1) * 255.0
    z_slice_rgb = np.stack((z_slice, z_slice, z_slice), axis=-1) * 255.0

    return x_slice_rgb.astype(imtype), y_slice_rgb.astype(imtype), z_slice_rgb.astype(imtype)


def tensor2imV2(input_volume, imtype=np.uint8):
    """
    Convierte un tensor 3D de PyTorch de dimensiones 256x256x256 en una imagen 2D de numpy.

    Parámetros:
        input_volume (tensor) -- el tensor de volumen de entrada de PyTorch de dimensiones 256x256x256
        imtype (type)        -- el tipo de dato deseado para el array de numpy convertido

    Returns:
        Tres imágenes 2D de numpy correspondientes a las slices centrales en las tres dimensiones del volumen de entrada.
    """
    if not isinstance(input_volume, np.ndarray):
        if isinstance(input_volume, torch.Tensor):  # obtener los datos de una variable
            volume_tensor = input_volume.data
        else:
            return input_volume
        volume_numpy = volume_tensor[0].cpu().float().numpy()  # convertir a un array de numpy

        volume_numpy = (np.transpose(volume_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-procesamiento: transposición y escalado
        # re-normalizar para tener mismo rango todas las imágenes
        volume_numpy = (volume_numpy - volume_numpy.min()) * (255.0 / (volume_numpy.max() - volume_numpy.min()))

        volume_center_slices = [volume_numpy[:,:,64], volume_numpy[:,64,:], volume_numpy[64,:,:]]  # obtener las slices centrales del volumen
    else:  # si es un array de numpy, no hacer nada
        volume_numpy = (np.transpose(input_volume, (1, 2, 0)) + 1) / 2.0 * 255.0
        volume_numpy = (volume_numpy - volume_numpy.min()) * (255.0 / (volume_numpy.max() - volume_numpy.min()))
        volume_center_slices = [input_volume[:,:,64], input_volume[:,64,:], input_volume[64,:,:]]
    return [s.astype(imtype) for s in volume_center_slices]


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_volume_nifti(volume_numpy, volume_path):
    """Save a numpy volume to the disk

    Parameters:
        volume_numpy (numpy array) -- input numpy array
        volume_path (str)          -- the path of the image
    """

    # Create a Nifti1Image object from the numpy volume
    nifti_img = nib.Nifti1Image(volume_numpy, affine=np.eye(4))
    
    # Save the Nifti image to disk
    nib.save(nifti_img, volume_path)


def save_volume_slices(images_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image1_pil = Image.fromarray(images_numpy[0])
    image2_pil = Image.fromarray(images_numpy[1])
    image3_pil = Image.fromarray(images_numpy[2])
    h, w = images_numpy[0].shape

    if aspect_ratio > 1.0:
        image1_pil = image1_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        image2_pil = image2_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        image3_pil = image3_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image1_pil = image1_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
        image2_pil = image2_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
        image3_pil = image3_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    
    # Create a new image that's three times as wide as the input images
    width, height = image1_pil.size
    result_image = Image.new('RGB', (width * 3, height))

    # Paste the three images side by side into the new image
    result_image.paste(image1_pil, (0, 0))
    result_image.paste(image2_pil, (width, 0))
    result_image.paste(image3_pil, (width * 2, 0))
    result_image.save(image_path)



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
