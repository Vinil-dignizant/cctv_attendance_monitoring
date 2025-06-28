# convert_to_ico.py (run once)
from PIL import Image
img = Image.open('assets/logo.png')  # Your 256x256 PNG
img.save('assets/icon.ico', sizes=[(16,16), (32,32), (48,48), (256,256)])