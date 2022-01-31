from sys import modules
import numpy as np
from numpy.core.fromnumeric import reshape
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks
import torch 

# a = [969, 499, 687, 725, 443, 429, 753, 175]
# b = [332, 4, 443,23 , 56, 565,789, 678]
# a = np.array(a).reshape(1,8).tolist()
# b = np.array(b).reshape(1,8).tolist() 
# c = [a , b]
# h = 1024
# w = 1024

# def _poly2mask(mask_ann, img_h, img_w):
#         """Private function to convert masks represented with polygon to
#         bitmaps.

#         Args:
#             mask_ann (list | dict): Polygon mask annotation input.
#             img_h (int): The height of output mask.
#             img_w (int): The width of output mask.

#         Returns:
#             numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
#         """

#         if isinstance(mask_ann, list):
#             # polygon -- a single object might consist of multiple parts
#             # we merge all parts into one mask rle code
#             rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
#             rle = maskUtils.merge(rles)
#         elif isinstance(mask_ann['counts'], list):
#             # uncompressed RLE
#             rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
#         else:
#             # rle
#             rle = mask_ann
#         mask = maskUtils.decode(rle)
#         return mask
# # result = _poly2mask(c,h,w)
# result = BitmapMasks([_poly2mask(mask,h,w) for mask in c], h,w)
# print(sum(sum(result)).any
# () == 0)
# print(result)

a=[1,2,3,4,6,7,8]
b = 0
for i in a:
    try:
        print(i)
    except:
        pass
    b = b +1
print(b)