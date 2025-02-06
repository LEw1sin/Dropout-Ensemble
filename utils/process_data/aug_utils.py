import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

def random_rot_flip(image, label):
    image, label = np.transpose(image, (1, 2, 0)), np.transpose(label, (1, 2, 0))
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    image, label = np.transpose(image, (2, 0, 1)), np.transpose(label, (2, 0, 1))

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def normalize(self, image):
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized_image

    def __call__(self, image, label):
        image_rot_flip, label_rot_flip = random_rot_flip(image, label)
        image_rotate, label_rotate = random_rotate(image, label)
        image, image_rot_flip, image_rotate = self.normalize(image), self.normalize(image_rot_flip), self.normalize(image_rotate)
        _, x_origin, y_origin = image.shape
        _, x_rot_flip, y_rot_flip = image_rot_flip.shape
        _, x_rotate, y_rotate = image_rotate.shape

        image = zoom(image, (1, self.output_size[0] / x_origin, self.output_size[1] / y_origin), order=3) 
        label = zoom(label, (1, self.output_size[0] / x_origin, self.output_size[1] / y_origin), order=0)
        image_rot_flip = zoom(image_rot_flip, (1, self.output_size[0] / x_rot_flip, self.output_size[1] / y_rot_flip), order=3) 
        label_rot_flip = zoom(label_rot_flip, (1, self.output_size[0] / x_rot_flip, self.output_size[1] / y_rot_flip), order=0)
        image_rotate = zoom(image_rotate, (1, self.output_size[0] / x_rotate, self.output_size[1] / y_rotate), order=3)
        label_rotate = zoom(label_rotate, (1, self.output_size[0] / x_rotate, self.output_size[1] / y_rotate), order=0)
        
        return image, label, image_rot_flip, label_rot_flip, image_rotate, label_rotate
    
def padding(image, target_z):
    d, h, w = image.shape
    padded_image = np.zeros((target_z, h, w), dtype=image.dtype)
    padded_image[:d] = image
    mask = np.zeros(target_z, dtype=int)
    mask[:d] = 1  # 将有效的部分标记为 1
    
    return padded_image, mask