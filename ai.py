import warnings
warnings.filterwarnings("ignore")


from PIL import Image
from pix2tex.cli import LatexOCR

model = LatexOCR()


def process(image_path):
    """
    Process the image and return the result.
    """
    img = Image.open(image_path)
    result = model(img)
    return result
