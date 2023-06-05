import os
import shutil

def path(*names):
    path = ""
    for name in names:
        path = path+name+'/'
    return path[:-1]

def copyTo(image, label, dst):
    os.makedirs(path(dst, label), exist_ok=True)
    shutil.copy2(image, path(dst, label))