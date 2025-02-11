import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import warp
import warnings
warnings.simplefilter("ignore", UserWarning)


def radon_old(image, theta=None, circle=True):
    """
    From skimage radon
    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    
    if image.shape[0] != image.shape[1]:
        raise ValueError('The input image must be square')

    if circle:
        center, radius = len(image)//2, len(image)//2
        y, x = np.ogrid[:512, :512]
        dist_from_center = np.sqrt((x - center)**2 + (y-center)**2)
        mask = dist_from_center <= radius
        # extract the value outside of the inscribed circle
        out_med = np.median(image[~mask])
        # only remain the inscribed circle
        image = image * mask

    center = image.shape[0] // 2
    radon_image = np.zeros((image.shape[0], len(theta)), dtype=image.dtype)

    for i, angle in enumerate(theta):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array(
            [
                [cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                [0, 0, 1],
            ]
        )
        rotated = warp(image, R, clip=False)
        rotated[dist_from_center >= (radius-1)] = out_med
        radon_image[:, i] = rotated.sum(0)
    return radon_image


def radon_gpu(image, theta, circle=True, fill_outside=True):
    """
    Modified based skimage radon

    Parameters
    ----------
    image : ndarray
        Input image. The width and height of the image should be the same
    theta : ndarray
        Projection angles (in radians).
    circle : boolean, optional
        The value will be zero outside the inscribed circle
    fill_outside: boolean, optional
        if True, the area outside the inscribed circle will be filled by the median 
    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    
    if image.shape[0] != image.shape[1]:
        raise ValueError('The input image must be square')

    fill_outside = False if not circle else fill_outside
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    size = image.shape[0]

    if circle:
        center, radius = size//2, size//2
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center)**2 + (y-center)**2)
        mask = dist_from_center <= radius
        out_med = float(np.median(image[~mask])) # extract the value outside of the inscribed circle
        image = image * mask # only remain the inscribed circle

    radon_image = torch.zeros((len(theta), size)).type(dtype)
    image = torch.from_numpy(image)[None, None, ...].type(dtype)
    theta = torch.Tensor(theta)
    for i, angle in enumerate(theta):
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot_mat = torch.Tensor([[cos_a, sin_a, 0],
                                [-sin_a, cos_a, 0]]).type(dtype)
        grid = F.affine_grid(rot_mat[None, ...], image.size())
        rotated = F.grid_sample(image, grid).squeeze()
        if fill_outside:
            rotated[dist_from_center >= (radius-1)] = out_med
        radon_image[i, :] = rotated.sum(0)
    return radon_image.detach().cpu().numpy()


if __name__ == '__main__':
    import time
    from PIL import Image
    from fbp import min_max_normalize
    import matplotlib.pyplot as plt
    img = Image.open('Sinogram_Alignment/raw_images/20230222_Sb_m4-1-b2-60s-0.85V_0001.tif')
    img = np.array(img)
    img = min_max_normalize(img)
    theta = np.linspace(0, np.pi, 181)
    start = time.time()
    sinogram = radon_old(img, theta).T
    end = time.time()
    print('Old radon time:', end - start)
    start = time.time()
    sinogram_gpu = radon_gpu(img, theta)
    end = time.time()
    print('New radon time:', end - start)
    sinogram = min_max_normalize(sinogram)
    plt.imshow(sinogram, cmap='gray')
    plt.axis('off')
    plt.savefig('old_sino.png', bbox_inches='tight')
    plt.close()
    sinogram = min_max_normalize(sinogram)
    plt.imshow(sinogram_gpu, cmap='gray')
    plt.axis('off')
    plt.savefig('new_sino.png', bbox_inches='tight')
    plt.close()