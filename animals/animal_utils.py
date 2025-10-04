import numpy as np


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4,
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * (x ** (1 / 2.4)) - a,
    )

def check_input_image(image: np.ndarray) -> bool:
    """
    This function is responsible for checking if the input image is valid

    The function does the following checks:
    - check if it's a np.ndarray instance
    - if it has a dimension of N x M x 3
    - if each values inside are numerical values, instead of from other datatype
    the function returns true iff the above are ALL satisfied
    """

    if not isinstance(image, np.ndarray):
        return False
    if not image.ndim == 3 or not image.shape[2] == 3:
        return False
    if not np.issubdtype(image.dtype, np.number):
        return False

    return True


def get_normalized_image(image: np.ndarray) -> np.ndarray:
    """
    This function returns the normalized version of input image in float32
    """
    image_output = image.astype(np.float32)
    if image_output.max() > 1.0:
        image_output /= 255.0
    image_output = np.clip(image_output, 0.0, 1.0)

    return image_output

def sRGB_to_LMS(image_in_sRGB: np.ndarray) -> np.ndarray:
    """
    The function takes a vector image in sRGB and returns it in LMS
    """
    M_rgb_to_lms = np.array(
        [
            [0.31399022, 0.63951294, 0.04649755],  # L
            [0.15537241, 0.75789446, 0.08670142],  # M
            [0.01775239, 0.10944209, 0.87256922],  # S
        ],
        dtype=np.float32,
    )
    return image_in_sRGB @ M_rgb_to_lms.T

def LMS_to_RGB(image_in_LMS: np.ndarray) -> np.ndarray:
    """
    The function takes a vector image in LMS and returns it in sRGB
    """
    M_LMS_to_sRGB = np.array(
        [
            [ 5.472213,   -4.6419606,   0.16963711],
            [-1.125242,    2.2931712,  -0.16789523],
            [ 0.02980164, -0.19318072,  1.1636479 ]
        ],
    )
    return image_in_LMS @ M_LMS_to_sRGB.T

def merge_L_M(image_in_LMS: np.ndarray, alpha: float) -> np.ndarray:
    """
    **Used for dicromatic mammels like cat, dog**
    The function returns a the image in LMS with L and M merged based on alpha
    """

    LM = alpha * image_in_LMS[:, 0] + (1.0 - alpha) * image_in_LMS[:, 1]
    return np.stack([LM, LM, image_in_LMS[:, 2]], axis=1)
