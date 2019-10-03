from PIL import Image

def rgb_to_grayscale(file):
	img = Image.open('photos/' + file).convert('L')
	img.save('grayscale_imgs/' + file)