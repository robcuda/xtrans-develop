#!/usr/bin/env python
# MIT License
#
# Deep Joint Demosaicking and Denoising
# Siggraph Asia 2016
# Michael Gharbi, Gaurav Chaurasia, Sylvain Paris, Fredo Durand
# 
# Copyright (c) 2016 Michael Gharbi
# Copyright (c) 2019 adapted by Gabriele Facciolo
# Copyright (c) 2021 adapted by Robert Freeman
#
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run the demosaicking network on an xtrans raw image or a directory containing multiple images."""

import argparse
import skimage.io
import numpy as np
import os
import re
import time
import torch as th
from tqdm import tqdm
import math

import demosaic.modules as modules
import demosaic.converter as converter

import rawpy
import pyexiv2
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2, TiffTags
from img_pipeline_helpers import *

# Pad image to avoid border effects
crop = 48


def _uint2float(I):
    if I.dtype == np.uint8:
        I = I.astype(np.float32)
        I = I*0.00390625
    elif I.dtype == np.uint16:
        I = I.astype(np.float32)
        I = I/65535.0
    else:
        raise ValueError("not a uint type {}".format(I.dtype))

    return I


def _float2uint(I, dtype):
    if dtype == np.uint8:
        I /= 0.00390625
        I += 0.5
        I = np.clip(I,0,255)
        I = I.astype(np.uint8)
    elif dtype == np.uint16:
        I *= 65535.0
        I += 0.5
        I = np.clip(I,0,65535)
        I = I.astype(np.uint16)
    else:
        raise ValueError("not a uint type {}".format(dtype))

    return I


def get_xtrans_offsets(pattern):
    #xtrans raw.raw_pattern falls into two possibilities
    if pattern[0][0] == 0:
        return (2,1)
    return (2,2)


def xtrans_mosaic(im,raw):
    """XTrans Mosaick.

     G b G G r G
     r G r b G b
     G b G G r G
     G r G G b G
     b G b r G r
     G r G G b G
    """
    mask = np.zeros((3, 6, 6), dtype=np.float32)
    g_pos = np.array([(0,0), (0,2), (0,3), (0,5),
            (1,1), (1,4),
            (2,0), (2,2), (2,3), (2,5),
            (3,0), (3,2), (3,3), (3,5), 
            (4,1), (4,4), 
            (5,0), (5,2), (5,3), (5,5)])
    r_pos = np.array([(0,4), 
            (1,0), (1,2), 
            (2,4), 
            (3,1), 
            (4,3), (4,5), 
            (5,1)])
    b_pos = np.array([(0,1), 
            (1,3), (1,5), 
            (2,1), 
            (3,4), 
            (4,0), (4,2), 
            (5,4)])

    mask[0][tuple(r_pos.T)] = 1
    mask[1][tuple(g_pos.T)] = 1
    mask[2][tuple(b_pos.T)] = 1

    _, h, w = im.shape
    mask = np.tile(mask, [1, np.ceil(h / 6).astype(np.int32), np.ceil(w / 6).astype(np.int32)])
    mask = mask[:, :h, :w]

    return mask*im, mask


def demosaick(net, M, noiselevel, tile_size, crop):

    # get the device of the network and apply it to the variables
    dev=next(net.parameters()).device

    M = th.from_numpy(M).to(device=dev, dtype=th.float)

    _, _, h, w = M.shape
    
    out_ref = th.zeros(3, h, w).to(device=dev, dtype=th.float)

    tile_size = min(min(tile_size, h), w)
    
    tot_time_ref = 0

    # mosaic block stride: good for both xtrans and bayer
    mod = 6
    tile_step = tile_size - crop*2
    tile_step = tile_step - (tile_step % mod)

    for start_x in tqdm(range(0, w, tile_step)):
        end_x = start_x + tile_size
        if end_x > w:
            # keep mosaic period
            end_x = w
            start_x = end_x - tile_size
            start_x = start_x - (start_x % mod)
            end_x = start_x + tile_size
        for start_y in range(0, h, tile_step):
            end_y = start_y + tile_size
            if end_y > h:
                end_y = h
                start_y = end_y - tile_size
                start_y = start_y - (start_y % mod)
                end_y = start_y + tile_size

            # noise level is ignored ny the XtransNetwork
            sample = {"mosaic": M[:, :, start_y:end_y, start_x:end_x], "noise_level": 0}

            start = time.time()
            
            # call the network
            outr = net(sample)

            tot_time_ref += time.time()-start

            oh, ow = outr.shape[2:]
            ch = (tile_size-oh) // 2
            cw = (tile_size-ow) // 2
            out_ref[:, start_y + ch: start_y + ch + oh, start_x + cw: start_x + cw + ow] = outr[0]

    tot_time_ref *= 1000
    print("Demosaic Time  {:.0f} ms".format(tot_time_ref))

    out_ref = out_ref.cpu().numpy()

    return out_ref, tot_time_ref


def demosaick_load_model():
    '''
    computes the relative paths to the pretrained models (Caffe)
    '''
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = here+'/pretrained_models/xtrans/'
    model_name = "XtransNetwork" 
    model_ref = modules.get({"model": model_name})
    cvt = converter.Converter(model_path, model_name)
    cvt.convert(model_ref)
    for p in model_ref.parameters():
        p.requires_grad = False
    return model_ref


def save_tiff(nparray_image, out_path):
    #using adobe_deflate for compatability. the tiffinfo config helps optimize the output size
    tif = Image.fromarray(nparray_image)
    tif.save(out_path, compression="tiff_adobe_deflate",tiffinfo={317: 2, 278: 1})
    tif.close()

    
def clone_exif(in_path, out_path):
    #clones the exif data, eg. RAF to TIFF
    original = pyexiv2.Image(in_path)
    exif = original.read_exif()
    clone_to = pyexiv2.Image(out_path)
    clone_to.modify_exif(exif)
    clone_to.close()
    original.close()


def input_name_conversion(name):
    #not robust, but good enough for camera conversions
    return name.replace(".RAF",".tiff")


def process_file( input_fname, output_fname, args, model_ref ): 
    #use rawpy to grab the raw image
    print("Processing:", input_fname)
    raw = rawpy.imread(input_fname)
    
    #there are only two observed xtrans configs, lets determine the pattern offsets
    offset_x, offset_y = get_xtrans_offsets(raw.raw_pattern)

    #image pipeline: preprocess before demosaic
    Iref = pipeline_fix_blacklevel(raw)
    Iref = pipeline_fix_whitelevel(Iref,raw)

    #pad offset for xtrans based on sensor pattern from raw
    Iref = np.pad(Iref, [(offset_y, 0), (offset_x,0)], 'constant')

    I = np.dstack((Iref, Iref, Iref))
    c = crop + (crop % 6)  # Make sure we don't change the pattern's period
    I = np.pad(I, [(c, c), (c, c), (0, 0)], 'symmetric')
    I = np.array(I).transpose(2, 0, 1).astype(np.float32)
    M = xtrans_mosaic(I,raw)
    # the othe field is just the mask
    M = np.array(M)[:1,:,:,:]

    with th.no_grad():
        R, runtime = demosaick(model_ref, M, 0, args.tile_size, crop)

    R = R.squeeze().transpose(1, 2, 0)

    # Remove the padding (crop)
    if crop > 0:
        R = R[c:-c, c:-c, :]

    # Remove the other padding (sensor offset) and adjust color space
    print("Converting to sRGB")
    demosaic = R[ offset_y:, offset_x: ]
    R = pipeline_demosaic_to_sRGB( demosaic, raw )

    if args.tiff16:
        out = _float2uint(R, np.uint16) #for users that really need this
    else:
        out = _float2uint(R, np.uint8) #typical users will probably go for 24bit color

    if output_fname is None:
        output_fname = input_name_conversion( input_fname )
    
    print("Saving output", output_fname)
    
    save_tiff(out, output_fname)
    clone_exif(input_fname, output_fname)


def main(args):
    # Load the network for the specific application
    model_ref = demosaick_load_model()
    if args.gpu:
        model_ref.cuda()
    else:
        model_ref.cpu()

    input_is_file = args.input.lower().endswith(".raf") if args.input else False
    output_is_file = args.output.lower().endswith((".tif",".tif")) if args.output else False

    if input_is_file:
        if args.output and not output_is_file:
            last_slash =  max ( args.input.rfind("/"), args.input.rfind("\\" ) )
            if last_slash:
                output_name = input_name_conversion( args.input[ last_slash+1 : ] )
                args.output = os.path.join( args.output, output_name )
            else:
                args.output = None
        return process_file( args.input, args.output, args, model_ref)

    if args.input is None:
        args.input = os.path.dirname(os.path.abspath(__file__))
        
    savepath = args.output if args.output else args.input
    
    for root, dirs, files in os.walk( args.input ):
        for name in files:
            if name.endswith(".RAF"):
                inpath = os.path.join( root, name )
                outpath = os.path.join( savepath, input_name_conversion( name ) )
                process_file( inpath, outpath, args, model_ref )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=None, help='path to input image.')
    parser.add_argument('--output', type=str, default=None, help='path to output image.')
    parser.add_argument('--tile_size', type=int, default=512, help='split the input into tiles of this size.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='use the GPU for processing.')
    parser.add_argument('--16bit', dest='tiff16', action='store_true', help='save 16bit tiff images instead.')
    parser.set_defaults(gpu=False, linear_input=False)

    args = parser.parse_args()
    main(args)
