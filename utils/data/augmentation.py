
import numpy as np
from typing import Union, Tuple
from skimage.transform import rotate, AffineTransform, warp, rescale


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered IFFT2 (complex output)"""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

def fft2c(image: np.ndarray) -> np.ndarray:
    """Centered FFT2 (complex input)"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def rotate_kspace(kspace: np.ndarray, angle: float) -> np.ndarray:
    image = ifft2c(kspace)

    real_part = np.stack([rotate(img.real, angle, resize=False, mode='edge', preserve_range=True) for img in image])
    imag_part = np.stack([rotate(img.imag, angle, resize=False, mode='edge', preserve_range=True) for img in image])

    rotated_image = real_part + 1j * imag_part
    return fft2c(rotated_image)


def shift_kspace(kspace: np.ndarray, shift: Union[int, Tuple[int,int]]) -> np.ndarray:
    if isinstance(shift, int):
        shift = (shift, shift)

    dx, dy = shift
    C, H, W = kspace.shape

    yy, xx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')  # shape (H, W)
    phase = np.exp(-2j * np.pi * (-dy * yy + dx * xx))  # shape (H, W)

    return kspace * phase[None, :, :]


def scale_kspace(kspace: np.ndarray, factor: Union[float, Tuple[float,float]]) -> np.ndarray:
    if isinstance(factor, (int, float)):
        factor = (factor, factor)
    image = ifft2c(kspace)
    image_scaled = []

    for img in image:
        real_scaled = rescale(img.real, scale=factor, mode='edge', anti_aliasing=True, preserve_range=True).T
        imag_scaled = rescale(img.imag, scale=factor, mode='edge', anti_aliasing=True, preserve_range=True).T

        real_cropped = center_crop(real_scaled, img.shape)
        imag_cropped = center_crop(imag_scaled, img.shape)

        image_scaled.append(real_cropped + 1j * imag_cropped)

    return np.transpose(fft2c(np.stack(image_scaled)), (0, 2, 1))


def flip_kspace(kspace: np.ndarray) -> np.ndarray:
    return np.conj(np.flip(kspace, axis=-2).copy())


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    return rotate(image, angle, resize=False, mode='edge', preserve_range=True)


def shift_image(image: np.ndarray, shift: Union[int, Tuple[int,int]]) -> np.ndarray:
    if isinstance(shift, int):
        shift = (-shift, shift)
    tform = AffineTransform(translation=shift)

    return warp(image, tform, mode='edge', preserve_range=True)


def scale_image(image: np.ndarray, factor: Union[float, Tuple[float,float]]) -> np.ndarray:
    if isinstance(factor, float):
        factor = (factor, factor)

    scaled = rescale(image, scale=factor, mode='edge', anti_aliasing=True, preserve_range=True)
    return center_crop(scaled, crop_size=image.shape)

def flip_image(image: np.ndarray) -> np.ndarray:
    return np.fliplr(image).copy()


def center_crop(image: np.ndarray, crop_size: Union[int, Tuple[int,int]]):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    H, W = image.shape
    target_w, target_h = crop_size

    start_y = max((H - target_h) // 2, 0)
    end_y = start_y + min(H, target_h)
    start_x = max((W - target_w) // 2, 0)
    end_x = start_x + min(W, target_w)

    cropped = image[start_y:end_y, start_x:end_x]

    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]

    if pad_h > 0 or pad_w > 0:
        pad_top = max(pad_h // 2, 0)
        pad_bottom = max(pad_h - pad_top, 0)
        pad_left = max(pad_w // 2, 0)
        pad_right = max(pad_w - pad_left, 0)

        cropped = np.pad(cropped, pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=0)

    return cropped


def rss(kspace: np.ndarray) -> np.ndarray:
    """
    Convert multi-coil k-space to image using Root-Sum-of-Squares (RSS).
    kspace shape: [coil, height, width]
    """
    image_per_coil = ifft2c(kspace)  # shape: [coil, H, W]
    rss_image = np.sqrt(np.sum(np.abs(image_per_coil) ** 2, axis=0))  # shape: [H, W]
    return rss_image

def spatial_augmentation(
        kspace: np.ndarray, 
        image: np.ndarray,
        flip_prop=0.5,
        rotate_prop=0.5, rotate_range=(5, 10),
        scale_prop=0.5, scale_range=(0.9, 1.1),
        shift_prop=0.5, shift_range=(5, 10),
    ) -> Tuple[np.ndarray, np.ndarray]:

    if np.random.random() < rotate_prop:
        angle = np.random.uniform(rotate_range[0], rotate_range[1])
        kspace = rotate_kspace(kspace, angle=angle)
        image = rotate_image(image, angle)
    
    if np.random.random() < scale_prop:
        scale_factor = tuple(np.random.uniform(scale_range[0], scale_range[1], 2))
        kspace = scale_kspace(kspace, scale_factor)
        image = scale_image(image, scale_factor)

    if np.random.random() < shift_prop:
        shift_step = tuple(np.random.randint(shift_range[0], shift_range[1], 2))
        kspace = shift_kspace(kspace, shift_step)
        image = shift_image(image, shift_step)

    if np.random.random() < flip_prop:
        kspace = flip_kspace(kspace)
        image = flip_image(image)
    
    return kspace, image