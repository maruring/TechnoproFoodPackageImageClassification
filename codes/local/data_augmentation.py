import copy
import os
import glob
from PIL import Image, ImageChops

def rotate_image(src_image, rotate=0):
    """回転処理
    """
    dst_image = copy.deepcopy(src_image)
    return dst_image.rotate(rotate)

def to_gray(src_image):
    """グレースケール化
    """
    dst_image = copy.deepcopy(src_image)
    return dst_image.convert('L')

def change_brightness(src_image, value):
    """明度を変換する
    """
    dst_image = copy.deepcopy(src_image)
    return dst_image.point(lambda x: x * value)

def change_size(src_image, rate):
    """大きさを変換する
    """
    dst_image = copy.deepcopy(src_image)
    return dst_image.resize((int(dst_image.width * rate), int(dst_image.height * rate)), Image.LANCZOS)

def add_noise(src_image, sigma):
    """ガウシアンノイズ付加する

    :param src_image:
    :param sigma:
    :return: np.array
    """
    temp_image = copy.deepcopy(src_image)
    width = temp_image.width
    height = temp_image.height
    noise_image = Image.effect_noise((width, height), sigma).convert('RGB')
    dst_image = ImageChops.multiply(temp_image, noise_image)

    return dst_image


for label in [0, 1]:
    src_dir_path = f"../Dataset/train/{label}/*"
    for image_file_path in glob.glob(src_dir_path):
        image = Image.open(image_file_path)
        image_file_name = os.path.basename(image_file_path)
        for ROTATE in [0, 90, 180, 270]:
            rotated_image = rotate_image(image, ROTATE)
            for is_gray in [True, False]:
                if is_gray:
                    dst_image = to_gray(rotated_image)
                    gray = "GRAY"
                else:
                    dst_image = copy.deepcopy(rotated_image)
                    gray = "COLOR"
                export_path = f"../data/{label}/{ROTATE}_{gray}_{image_file_name}"
                dst_image.save(export_path, quality=95)