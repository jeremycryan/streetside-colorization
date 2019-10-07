from PIL import Image
import os

input_dir = "photos"
grayscale_output_dir = "grayscale_imgs"
color_output_dir = "color_imgs"

output_size = 0.1
crop_margin = 70  # Crop happens before downsample


def get_images(path):
    paths = os.listdir(path)
    img_paths = [item for item in paths if item.lower().endswith(".jpg")]
    for path in img_paths:
        if not path.endswith('_0.jpg') and not path.endswith('_5.jpg'):
            image = Image.open(os.path.join(input_dir, path))
            yield path, image


def process_images(image_iter):
    
    # Make output directory, if it doesn't exist
    if not os.path.exists(color_output_dir):
        os.makedirs(color_output_dir)

    if not os.path.exists(grayscale_output_dir):
        os.makedirs(grayscale_output_dir)

    for path, image in image_iter:

        # Crop image
        width, height = image.size
        rect = crop_margin, crop_margin, width - crop_margin, height - crop_margin
        cropped = image.crop(rect)

        # Resize image
        width, height = cropped.size
        new_width = int(output_size * width)
        new_height = int(output_size * height)
        resized = cropped.resize((new_width, new_height))

        # Write file
        write_path = os.path.join(color_output_dir, path)
        resized.save(write_path, "JPEG")

        # Grayscale-ize them
        resized = resized.convert('L')

        # Write file
        write_path = os.path.join(grayscale_output_dir, path)
        resized.save(write_path, "JPEG")


if __name__ == "__main__":
    image_iter = get_images(input_dir)
    process_images(image_iter)