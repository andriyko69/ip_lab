from PIL import Image, ImageChops
import time

def calc_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} time: ", time.time() - start_time)
        return result
    return wrapper


@calc_time
def BMP_RLE(im, image_path):
    rle_image_path = 'image_rle.bmp'
    im.save(rle_image_path, format='BMP', compress_level=1)


@calc_time
def TIFF(im, image_path):
    tiff_image_path = 'image_lzw.tiff'
    im.save(tiff_image_path, format='TIFF', compression='tiff_lzw')


@calc_time
def JPEG(im, image_path):
    jpeg_image_path = 'image_standard.jpg'
    im.save(jpeg_image_path, format='JPEG')


@calc_time
def open_image(image_name):
    return Image.open(image_name)


def subtract_images(img1, img2, output_name):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    diff = ImageChops.difference(img1, img2)
    diff.save(output_name)


def main():
    image_name = 'images_9.bmp'
    image = open_image(image_name)

    # Compression
    BMP_RLE(image, 'image_rle.bmp')
    TIFF(image, 'image_lzw.tiff')
    JPEG(image, 'image_standard.jpg')

    # Open compressed images
    imaga_rle = Image.open('image_rle.bmp')
    image_tiff = Image.open('image_lzw.tiff')
    image_jpeg = Image.open('image_standard.jpg')

    subtract_images(image, imaga_rle, 'diff_bmp_rle.png')
    subtract_images(image, image_tiff, 'diff_bmp_tiff.png')
    subtract_images(image, image_jpeg, 'diff_bmp_jpeg.png')

if __name__ == '__main__':
    main()
