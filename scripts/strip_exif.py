"""Strip EXIF (incl. GPS) from all JPEGs in a folder, in place."""
import sys
from pathlib import Path
from PIL import Image

for p in Path(sys.argv[1]).rglob("*.jp*g"):
    im = Image.open(p)
    data = list(im.getdata())
    clean = Image.new(im.mode, im.size)
    clean.putdata(data)
    clean.save(p, quality=95)
    print("stripped", p)
