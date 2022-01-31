from doctest import debug_src
import os
from posixpath import basename


# 伪代码
path = 'data/dota/val/labelTxt'

def imagename2txt():
    
    dirs= os.listdir(path)
    with open(dstfile,'w') as dst_file:
        for filename in dirs:
            
            (filepath, basename) = os.path.split(filename)
            (filename, extension) = os.path.splitext(basename)
            dst_file.write(filename+"\n")

if __name__ == '__main__':
    dstfile = 'AerialDetection/DOTA_devkit/imagename2txt.txt'
    imagename2txt()