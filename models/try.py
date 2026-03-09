import warnings
warnings.filterwarnings("ignore")


from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open(r'C:\Users\Ota\source\dataset\formula_images_processed\formula_images_processed\1a0a0dfbac.png')
model = LatexOCR()
print(model(img))
