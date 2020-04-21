#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import time
import PIL
from PIL import Image
from math import *
from numpy import (dtype, sin)

class RotateProcessor(object):
    def __init__(self):
        self.zoom = 0.5
        self.perc = 50
        self.range = 20
        self.bignore = 0.2
        self.maxskew = 90
        self.skewsteps = 1
        self.escale = 1.0
        self.lo = 0.05
        self.hi = 0.9

    def pil2array(self, im, alpha=0):
        if im.mode == "L":
            a = np.fromstring(im.tobytes(), 'B')
            a.shape = im.size[1], im.size[0]
            return a
        if im.mode == "RGB":
            a = np.fromstring(im.tobytes(), 'B')
            a.shape = im.size[1], im.size[0], 3
            return a
        if im.mode == "RGBA":
            a = np.fromstring(im.tobytes(), 'B')
            a.shape = im.size[1], im.size[0], 4
            if not alpha: a = a[:, :, :3]
            return a
        return self.pil2array(im.convert("L"))

    def array2pil(self, a):
        if a.dtype == dtype("B"):
            if a.ndim == 2:
                return PIL.Image.frombytes("L", (a.shape[1], a.shape[0]), a.tostring())
            elif a.ndim == 3:
                return PIL.Image.frombytes("RGB", (a.shape[1], a.shape[0]), a.tostring())
            else:
                raise Exception("bad image rank")
        elif a.dtype == dtype('float32'):
            return PIL.Image.fromstring("F", (a.shape[1], a.shape[0]), a.tostring())
        else:
            raise Exception("unknown image type")

    def isbytearray(self, a):
        return a.dtype in [dtype('uint8')]

    def isfloatarray(self, a):
        return a.dtype in [dtype('f'), dtype('float32'), dtype('float64')]

    def isintarray(self, a):
        return a.dtype in [dtype('B'), dtype('int16'), dtype('int32'), dtype('int64'), dtype('uint16'), dtype('uint32'),
                           dtype('uint64')]

    def isintegerarray(self, a):
        return a.dtype in [dtype('int32'), dtype('int64'), dtype('uint32'), dtype('uint64')]

    def read_image_gray(self, image, pageno=0):
        """Read an image and returns it as a floating point array.
        The optional page number allows images from files containing multiple
        images to be addressed.  Byte and short arrays are rescaled to
        the range 0...1 (unsigned) or -1...1 (signed)."""
        # if type(fname) == tuple: fname, pageno = fname
        # assert pageno == 0
        # pil = PIL.Image.open(fname)

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        a = self.pil2array(pil)
        if a.dtype == dtype('uint8'):
            a = a / 255.0
        if a.dtype == dtype('int8'):
            a = a / 127.0
        elif a.dtype == dtype('uint16'):
            a = a / 65536.0
        elif a.dtype == dtype('int16'):
            a = a / 32767.0
        elif self.isfloatarray(a):
            pass
        else:
            raise Exception("unknown image type: " + a.dtype)
        if a.ndim == 3:
            a = a[:, :, 0]  # red channel
        return a
    def normalize_raw_image(self, raw):
        ''' perform image normalization '''
        image = raw - np.amin(raw)
        if np.amax(image) == np.amin(image):
            return image
        image /= np.amax(image)
        return image

    def estimate_local_whitelevel(self, image, bignore=0.2, zoom=0.5, perc=80, range=20):
        '''flatten it by estimating the local whitelevel
        zoom for page background estimation, smaller=faster, default: %(default)s
        percentage for filters, default: %(default)s
        range for filters, default: %(default)s
        '''
        d0, d1 = image.shape
        o0, o1 = int(bignore * d0), int(bignore * d1)
        est = image[o0:d0 - o0, o1:d1 - o1]

        image_black = np.sum(est < 0.05)
        image_white = np.sum(est > 0.95)
        extreme = (image_black + image_white) * 1.0 / np.prod(est.shape)

        if np.mean(est) < 0.4:
            # print(np.mean(est), np.median(est))
            image = 1 - image

        if extreme > 0.95:
            flat = image
        else:
            m = cv2.blur(image, (range, range))
            w, h = np.minimum(np.array(image.shape), np.array(m.shape))
            flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)

        return flat

    # 对图像挨个角度旋转，然后做投影，看那个方向上，图像投影方差最大，就是最"正"的角度
    def estimate_skew_angle(self, image, angles):
        estimates = []
        for a in angles:
            matrix = cv2.getRotationMatrix2D((int(image.shape[1] / 2), int(image.shape[0] / 2)), a, 1)
            rotate_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

            # print("rotate img:",rotate_image.shape)
            v = np.mean(rotate_image, axis=1)
            # print("mean shape:",v.shape)
            v = np.var(v)
            # print("var shape:",v)
            estimates.append((v, a))
        _, a = max(estimates)
        # print(a)
        # print(estimates)
        return a

    def estimate_skew(self, flat, maxskew=45, skewsteps=1):
        ''' estimate skew angle and rotate'''
        flat = np.amax(flat) - flat
        flat -= np.amin(flat)
        ma = maxskew
        ms = int(2 * maxskew * skewsteps) # 从45度到-45度，尝试90度的范围

        angle = self.estimate_skew_angle(flat, np.linspace(-ma, ma, ms + 1))

        return angle

    def estimate_thresholds(self, flat, bignore=0.2, escale=1, lo=0.05, hi=0.9):
        '''# estimate low and high thresholds
        ignore this much of the border for threshold estimation, default: %(default)s
        scale for estimating a mask over the text region, default: %(default)s
        lo percentile for black estimation, default: %(default)s
        hi percentile for white estimation, default: %(default)s
        '''
        d0, d1 = flat.shape
        o0, o1 = int(bignore * d0), int(bignore * d1)
        est = flat[o0:d0 - o0, o1:d1 - o1]

        if escale > 0:
            # by default, we use only regions that contain
            # significant variance; this makes the percentile
            # based low and high estimates more reliable

            # 高斯模糊
            v = est - cv2.GaussianBlur(est, (3, 3), escale * 20)
            v = cv2.GaussianBlur(v ** 2, (3, 3), escale * 20) ** 0.5

            v = (v > 0.3 * np.amax(v))
            v = np.asarray(v, np.uint8)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(escale * 50), int(escale * 50)))
            v = cv2.dilate(v, kernel, 1)
            v = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
            v = (v > 0.3 * np.amax(v))
            est = est[v]

        if len(est) != 0:
            est = np.sort(est)
            lo = est[int(lo * len(est))]
            hi = est[int(hi * len(est))]

        # rescale the image to get the gray scale image
        flat -= lo
        flat /= (hi - lo)
        flat = np.clip(flat, 0, 1)
        return flat

    def rotate_angle(self,img,angle):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
        matrix = cv2.getRotationMatrix2D((int(img.shape[1] / 2), int(img.shape[0] / 2)), angle, 1)
        matrix[0, 2] += (widthNew - width) / 2
        matrix[1, 2] += (heightNew - height) / 2
        img_rotate = cv2.warpAffine(img, matrix, (widthNew, heightNew),borderValue=(255,255,255))
        return img_rotate
        #img_resize = cv2.resize(img_rotate, (width * 800 / height, 800))

    # 返回的angel是逆时针方向的，
    # 比如+10'，就意味着正图从中线，逆时针旋转了10度
    # 比如-10'，就是从正图顺时针旋转了10度
    def process(self, image):
        original_image = image
        # perform image normalization
        raw = self.read_image_gray(image)
        image = self.normalize_raw_image(raw)
        # check whether the image is already effectively binarized
        flat = self.estimate_local_whitelevel(image, self.bignore, self.zoom, self.perc, self.range)
        #resize image
        height, width = flat.shape[:2]
        if height > width:
            flat = cv2.resize(flat, (600, int(height * 600 / width)))
        else:
            flat = cv2.resize(flat, (int(width * 600 / height), 600))
        # estimate skew angle and rotate
        result_angle = self.estimate_skew(flat, self.maxskew, self.skewsteps)
        #角度调整 不超过正负45度
        if result_angle < -45:
            result_angle = result_angle + 90
        elif result_angle > 45:
            result_angle = result_angle - 90
        rotate_img = self.rotate_angle(original_image,result_angle)
        return result_angle,rotate_img


if __name__ == "__main__":
    #use demo
    rotate = RotateProcessor()
    start = time.time()
    image_path = '/Users/piginzoo/Downloads/train_images/角度相差较大/angle201912051951405.jpg'
    image = cv2.imread(image_path)
    angle,img_rotate = rotate.process(image)
    end = time.time()
    print(end - start)
