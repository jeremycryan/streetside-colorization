from PIL import Image
import numpy as np
import pytorch
import pandas

color_dir = 'color_imgs'
grayscale_dir = 'grayscale_imgs'

def get_images(path):
    paths = os.listdir(path)
    img_paths = [item for item in paths if item.lower().endswith(".jpg")]
    for path in img_paths:
        if not path.endswith('_0.jpg') and not path.endswith('_5.jpg'):
            image = Image.open(os.path.join(input_dir, path))
            yield image

if __name__ == '__main__':
	data = np.array([])
	imgs = get_images(color_dir)
	for i in imgs:
		np.concatenate([data, np.asarray(i)], axis = 2)

	print(data)