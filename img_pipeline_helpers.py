'''
Copyright (c) 2020, Heiko Bauke
XTrans modification (c) 2021, Robert Freeman

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
'''
import numpy as np
import rawpy

#Pipeline functions mostly from https://www.numbercrunch.de/blog/2020/12/from-numbers-to-images-raw-image-processing-with-python/
#see https://github.com/rabauke/Python_image_processing for ISC license


def pipeline_fix_blacklevel(raw):
    black = np.reshape(np.array(raw.black_level_per_channel, dtype=np.double), (2, 2))
    black = np.tile(black, (raw.raw_image_visible.shape[0]//2, raw.raw_image_visible.shape[1]//2))
    return np.clip( (raw.raw_image_visible - black) / (raw.white_level - black) , 0, 1)


def pipeline_fix_whitelevel(image, raw):
    # code modified to be xtrans friendly
    colors = np.frombuffer(raw.color_desc, dtype=np.byte)
    pattern = np.array(raw.raw_pattern)
    index_a = np.where(raw.raw_colors_visible == 0)
    index_b = np.where(raw.raw_colors_visible == 1)
    index_c = np.where(raw.raw_colors_visible == 2)
    # apply white balance, normalize white balance coefficients to the 2nd coefficient, which is ususally the coefficient for green
    wb_c = raw.camera_whitebalance 
    wbz = np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)
    wbz[index_a] = wb_c[0] / wb_c[1]
    wbz[index_b] = wb_c[1] / wb_c[1]
    wbz[index_c] = wb_c[2] / wb_c[1]
    return np.clip(image * wbz, 0, 1)


def pipeline_demosaic_to_sRGB(image_demosaiced, raw):
    # note: insignificant changes from blog post above
    XYZ_to_cam = np.array(raw.rgb_xyz_matrix[0:3, :], dtype=np.float32)
    sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    sRGB_to_cam = np.dot(XYZ_to_cam, sRGB_to_XYZ)
    norm = np.tile(np.sum(sRGB_to_cam, 1), (3, 1)).transpose()
    sRGB_to_cam = sRGB_to_cam / norm
    cam_to_sRGB = np.linalg.inv(sRGB_to_cam)
    image_sRGB = np.einsum('ij,...j', cam_to_sRGB, image_demosaiced)  # performs the matrix-vector product for each pixel
    # linear to sRGB
    i,j = image_sRGB < 0.0031308, image_sRGB >= 0.0031308
    image_sRGB[i] = 12.92 * image_sRGB[i]
    image_sRGB[j] = 1.055 * image_sRGB[j] ** (1/2.4) - 0.055
    image_sRGB = np.clip(image_sRGB, 0, 1)
    return image_sRGB



if __name__ == "__main__":
    pass