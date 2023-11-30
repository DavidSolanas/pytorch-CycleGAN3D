"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
from abc import ABC, abstractmethod
from skimage.transform import resize
from scipy.interpolate import interpn



class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    print(x,y,flip)
    return {'crop_pos': (x, y), 'flip': flip}

def get_params3d(opt, size):
    w, h, z = size
    new_h = h
    new_w = w
    new_z = z
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = new_z = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
        new_z = opt.load_size

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    z = random.randint(0, np.maximum(0, new_z - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y, z), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        print('resize',osize)
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        print('crop')
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        print('none')
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        print(params)
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        print('convert')
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            print('normalize')
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform3d(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __resize3d(img)))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'tfm-david':
        # OASIS3 preprocess tfm david
        transform_list.append(transforms.Lambda(lambda img: __cropOASIS3(img)))
        transform_list.append(transforms.Lambda(lambda img: __resizeOASIS3(img, shape=(opt.load_size,opt.load_size,opt.load_size))))
        transform_list.append(transforms.Lambda(lambda img: __normalizeCycleGANOASIS3(img)))


    if not opt.no_flip:
        print('no flip')
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip3d(img, params['flip'])))

    if convert:
        transform_list += [transforms.Lambda(lambda img: to_tensor(img))]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Lambda(lambda img: __normalize3dTensor(img, (0.5,), (0.5,)))]

    return transforms.Compose(transform_list)



def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __cropOASIS3(img: np.ndarray) -> np.ndarray:
    # Precomputed crop for OASIS3 processed volumes
    min_x = 14
    max_x = 179
    min_y = 16
    max_y = 221
    min_z = 60
    max_z = 242
    # print(img.shape)#(1, 240, 240, 155, 4)
    img =  img[min_x-1:max_x+1, min_y-1:max_y+1, min_z-1:max_z+1]
    return img
    # return img[self.buffer]

def __resizeOASIS3(img: np.ndarray, shape=(128, 128, 128)) -> np.ndarray:
    xx = np.arange(shape[1]) 
    yy = np.arange(shape[0])
    zz = np.arange(shape[2])

    xx = xx * img.shape[1] / shape[1]
    yy = yy * img.shape[0] / shape[0] 
    zz = zz * img.shape[2] / shape[2]

    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    sample = np.stack((grid[:, :, :, 1], grid[:, :, :, 0], grid[:, :, :, 2]), 3)
    xxx = np.arange(img.shape[1])
    yyy = np.arange(img.shape[0])
    zzz = np.arange(img.shape[2])  

    img = img[0:img.shape[0],0:img.shape[1],0:img.shape[2]]

    img = interpn((yyy, xxx, zzz), img, sample, method='linear', bounds_error=False, fill_value=0)

    return img

def __normalizeCycleGANOASIS3(img: np.ndarray) -> np.ndarray:
    # Normalize the values to be in the range of 0-1
    img_norm = (img - img.min()) / (img.max() - img.min())

    # Scale the values to be in the range of 0-255
    img_scaled = img_norm * 255

    # Cast the values to np.uint8 data type
    #arr_uint8 = arr_scaled.astype(np.uint8)
    return img_scaled

def __make_power_2_3d(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh, oz = img.shape
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    z = int(round(oz / base) * base)
    if h == oh and w == ow and z == oz:
        return img

    print(ow, oh, oz, '->', w, h, z)
    return img.resize((w, h, z), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __flip3d(img, flip):
    if flip:
        return img.transpose(1, 0, 2) # left to right
    return img

def __resize3d(img, size=(128,128,128), order=3):
    resized = resize(img, size, anti_aliasing=False, order=order)
    return resized


def to_tensor(pic) -> torch.Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    default_float_dtype = torch.get_default_dtype()
    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]
        else:
            pic = pic[None, ...]

        img = torch.from_numpy(pic).contiguous()
        # backward compatibility

        return img.to(dtype=default_float_dtype).div(255)


def __normalize3dTensor(tensor: torch.Tensor, mean, std, inplace: bool = False) -> torch.Tensor:

    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor.float()

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
