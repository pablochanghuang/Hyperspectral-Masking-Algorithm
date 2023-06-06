import os

dr = "data/Core Mask" ## change this for your PC
fs = os.listdir(dr)

for f in fs:
    f_new = f.replace(".png", ".tiff")
    os.rename(dr + "/" + f, dr + "/" + f_new)