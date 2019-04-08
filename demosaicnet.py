#!/usr/bin/env python
# MIT License
#
# Deep Joint Demosaicking and Denoising
# Siggraph Asia 2016
# Michael Gharbi, Gaurav Chaurasia, Sylvain Paris, Fredo Durand
# 
# Copyright (c) 2016 Michael Gharbi
# Copyright (c) 2019 adapted by Gabriele Facciolo
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

"""Run the demosaicking network on an image or a directory containing multiple images."""

import argparse
import cv2
import numpy as np
import os
import re
import time
from tqdm import tqdm

import torch as th

import demosaic.modules as modules
import demosaic.converter as converter

NOISE_LEVELS = [0.0000, 0.0784]  # Min/Max noise levels we trained on

def _psnr(a, b, crop=0, maxval=1.0):
    """Computes PSNR on a cropped version of a,b"""

    if crop > 0:
        aa = a[crop:-crop, crop:-crop, :]
        bb = b[crop:-crop, crop:-crop, :]
    else:
        aa = a
        bb = b

    d = np.mean(np.square(aa-bb))
    d = -10*np.log10(d/(maxval*maxval))
    return d


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


def _blob_to_image(blob):
    # input shape h,w,c
    shape =  blob.data.shape
    sz = shape[1:]
    out = np.copy(blob.data)
    out = np.reshape(out, sz)
    out = out.transpose((1,2,0))
    return out




def bayer_mosaic(im):
    """GRBG Bayer mosaic."""

    mos = np.copy(im)
    mask = np.ones_like(im)

    # red
    mask[0, ::2, 0::2] = 0
    mask[0, 1::2, :] = 0

    # green
    mask[1, ::2, 1::2] = 0
    mask[1, 1::2, ::2] = 0

    # blue
    mask[2, 0::2, :] = 0
    mask[2, 1::2, 1::2] = 0

    return mos*mask, mask



def xtrans_mosaic(im):
    """XTrans Mosaick.

     G b G G r G
     r G r b G b
     G b G G r G
     G r G G b G
     b G b r G r
     G r G G b G
    """
    mask = np.zeros((3, 6, 6), dtype=np.float32)
    g_pos = [(0,0), (0,2), (0,3), (0,5),
           (1,1), (1,4),
           (2,0), (2,2), (2,3), (2,5),
           (3,0), (3,2), (3,3), (3,5), 
           (4,1), (4,4), 
           (5,0), (5,2), (5,3), (5,5)]
    r_pos = [(0,4), 
           (1,0), (1,2), 
           (2,4), 
           (3,1), 
           (4,3), (4,5), 
           (5,1)]
    b_pos = [(0,1), 
           (1,3), (1,5), 
           (2,1), 
           (3,4), 
           (4,0), (4,2), 
           (5,4)]

    for y, x in g_pos:
        mask[1, y, x] = 1

    for y, x in r_pos:
        mask[0, y, x] = 1

    for y, x in b_pos:
        mask[2, y, x] = 1

    mos = np.copy(im)

    _, h, w = mos.shape
    mask = np.tile(mask, [1, np.ceil(h / 6).astype(np.int32), np.ceil(w / 6).astype(np.int32)])
    mask = mask[:, :h, :w]

    return mask*mos, mask






def demosaick_old(net, M, noise, psize, crop):
    start_time = time.time()
    h,w = M.shape[:2]

    psize = min(min(psize,h),w)
    psize -= psize % 2
    patch_step = psize
    patch_step -= 2*crop
    shift_factor = 2

    # Result array
    R = np.zeros(M.shape, dtype = np.float32)

    rangex = range(0,w-2*crop,patch_step)
    rangey = range(0,h-2*crop,patch_step)
    ntiles = len(rangex)*len(rangey)
    with tqdm(total=ntiles, unit='tiles', unit_scale=True) as pbar:
        for start_x in rangex:
            for start_y in rangey:
                end_x = start_x+psize
                end_y = start_y+psize
                if end_x > w:
                    end_x = w
                    end_x = shift_factor*((end_x)/shift_factor)
                    start_x = end_x-psize
                if end_y > h:
                    end_y = h
                    end_y = shift_factor*((end_y)/shift_factor)
                    start_y = end_y-psize


                tileM = M[start_y:end_y, start_x:end_x, :] 
                tileM = tileM[np.newaxis,:,:,:]
                tileM = tileM.transpose((0,3,1,2))

                net.blobs['mosaick'].reshape(*tileM.shape)
                net.blobs['mosaick'].data[...] = tileM

                if 'noise_level' in net.blobs.keys():
                    noise_shape = [1,]
                    net.blobs['noise_level'].reshape(*noise_shape)
                    net.blobs['noise_level'].data[...] = noise

                net.forward()

                out = net.blobs['output']
                out = _blob_to_image(out)
                s = out.shape[0]

                R[start_y+crop:start_y+crop+s,
                  start_x+crop:start_x+crop+s,:] = out

                pbar.update(1)

    R[R<0] = 0.0
    R[R>1] = 1.0

    runtime = (time.time()-start_time)*1000  # in ms

    return R, runtime





def demosaick(net, M, noiselevel, tile_size, crop):

    # get the device of the network and apply it to the variables
    dev=next(net.parameters()).device

    M = th.from_numpy(M).to(device=dev, dtype=th.float)

    _, _, h, w = M.shape

    out_ref = th.zeros(3, h, w).to(device=dev, dtype=th.float)
    
    sigma_noise = noiselevel * th.ones(1).to(device=dev, dtype=th.float)

    tile_size = min(min(tile_size, h), w)

    tot_time_ref = 0

#    if xtrans:
#      mod = 6
#    else:
#      mod = 2
    # good for both xtrans and bayer
    mod = 6
    tile_step = tile_size - crop*2

    tile_step = tile_step - (tile_step % mod)
    tile_step = tile_step - (tile_step % mod)
    #print (tile_step)

#    for start_x in range(0, w, tile_step):
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

            #print(start_x, start_y)
            # noise level is ignored ny the XtransNetwork and BayerNetwork
            sample = {"mosaic": M[:, :, start_y:end_y, start_x:end_x], "noise_level": sigma_noise}

            th.cuda.synchronize()
            start = time.time()

            # call the network
            outr = net(sample)

            th.cuda.synchronize()
            tot_time_ref += time.time()-start

            oh, ow = outr.shape[2:]
            ch = (tile_size-oh) // 2
            cw = (tile_size-ow) // 2
            out_ref[:, start_y + ch: start_y + ch + oh, start_x + cw: start_x + cw + ow] = outr[0]

    tot_time_ref *= 1000
    print("Time  {:.0f} ms".format(tot_time_ref))

    out_ref = out_ref.cpu().numpy()

    return out_ref, tot_time_ref




def demosaick_load_model(noiselevel=0.0, xtrans=False):
    '''
    this function uses the hardcoded paths of the pretrained models 
    '''
    pretrained_xtrans = 'pretrained_models/xtrans/'
    pretrained_bayer = 'pretrained_models/bayer/'
    pretrained_bayer_noise = 'pretrained_models/bayer_noise/'
    
    print("Loading Caffe weights")
    if xtrans:
        model_ref = modules.get({"model": "XtransNetwork"})
        cvt = converter.Converter(pretrained_xtrans, "XtransNetwork")
    else:
        if noiselevel==0.0:

            model_ref = modules.get({"model": "BayerNetwork"})
            cvt = converter.Converter(pretrained_bayer, "BayerNetwork")
        else:
            model_ref = modules.get({"model": "BayerNetworkNoise"})
            cvt = converter.Converter(pretrained_bayer_noise, "BayerNetworkNoise")
    cvt.convert(model_ref)
    for p in model_ref.parameters():
        p.requires_grad = False

    return model_ref




def main(args):

    model_ref = demosaick_load_model(args.noise, xtrans=(args.mosaic_type == 'xtrans') )
    if args.gpu:
        model_ref.cuda()
    else:
        model_ref.cpu()
          
    # this results from padding 
    crop = 36
    print ("Crop", crop)

    regexp = re.compile(r".*\.(png|tif|jpg)")
    if os.path.isdir(args.input):
        print ('dir')
        inputs = [f for f in os.listdir(args.input) if regexp.match(f)]
        inputs = [os.path.join(args.input, f) for f in inputs]
    else:
        inputs = [args.input]

    avg_psnr = 0
    n = 0
    for fname in inputs:
        print ('+ Processing {}'.format(fname))
        Iref = cv2.imread(fname, -1)
        if len(Iref.shape) == 4:  # removes alpha
            Iref = Iref[:, :, :3]
        if len(Iref.shape) == 3:  # CV color storage..
            Iref = cv2.cvtColor(Iref,cv2.COLOR_BGR2RGB) 
        dtype = Iref.dtype
        if dtype not in [np.uint8, np.uint16]:
            raise ValueError('Input type not handled: {}'.format(dtype))
        Iref = _uint2float(Iref)

        if args.linear_input:
            print ("  - Input is linear, mapping to sRGB for processing")
            Iref = np.power(Iref, 1.0/2.2)

        if len(Iref.shape) == 2:
            # Offset the image to match the our mosaic pattern
            if args.offset_x > 0:
                print ('  - offset x')
                # Iref = Iref[:, 1:]
                Iref = np.pad(Iref, [(0, 0), (args.offset_x, 0)], 'reflect')

            if args.offset_y > 0:
                print ('  - offset y')
                # Iref = Iref[1:, :]
                Iref = np.pad(Iref, [(args.offset_y, 0), (0,0)], 'reflect')
            has_groundtruth = False
            Iref = np.dstack((Iref, Iref, Iref))
        else:
            # No need for offsets if we have the ground-truth
            has_groundtruth = True

        if has_groundtruth and args.noise > 0:
            print ('  - adding noise sigma={:.3f}'.format(args.noise))
            I = Iref + np.random.normal(
                    loc=0.0, scale = args.noise , size = Iref.shape )
        else:
            I = Iref

        if crop > 0:
            if args.mosaic_type == 'bayer':
                c = crop + (crop %2)  # Make sure we don't change the pattern's period
                I = np.pad(I, [(c, c), (c, c), (0, 0)], 'reflect')
            else:
                c = crop + (crop % 6)  # Make sure we don't change the pattern's period
                I = np.pad(I, [(c, c), (c, c), (0, 0)], 'reflect')

        if has_groundtruth:
            print ('  - making mosaick')
        else:
            print ('  - formatting mosaick')
            
        #M = _make_mosaic(I, args.mosaic_type)
        
        
        
        
        
        
        I = np.array(I).transpose(2, 0, 1).astype(np.float32)
   
        if args.mosaic_type == 'xtrans':
            M = xtrans_mosaic(I)
        else:
            M = bayer_mosaic(I)
        #im = np.expand_dims(im, 0) 
        # the othe field is just the mask
        M = np.array(M)[:1,:,:,:]


        R, runtime = demosaick(model_ref, M, args.noise, args.tile_size, crop)

        
        R = R.squeeze().transpose(1, 2, 0)

 
        
        
        
        if crop > 0:
            R = R[c:-c, c:-c, :]
            I = I[c:-c, c:-c, :]
            M = M[c:-c, c:-c, :]
        
        if not has_groundtruth:
            if args.offset_x > 0:
                print ('  - remove offset x')
                R = R[:, args.offset_x:]
                I = I[:, args.offset_x:]
                M = M[:, args.offset_x:]

            if args.offset_y > 0:
                print ('  - remove offset y')
                R = R[args.offset_y:, :]
                I = I[args.offset_y:, :]
                M = M[args.offset_y:, :]

        if len(Iref.shape) == 2:
            # Offset the image to match the our mosaic pattern
            if args.offset_x == 1:
                print ('  - offset x')
                Iref = Iref[:, 1:]

            if args.offset_y == 1:
                print ('  - offset y')
                Iref = Iref[1:, :]
            has_groundtruth = False

        if args.linear_input:
            print ("  - Input is linear, mapping output back from sRGB")
            R = np.power(R, 2.2)

        if has_groundtruth:
            p = _psnr(R, Iref, crop=crop)
            avg_psnr += p
            n += 1
            #diff = np.abs((R-Iref))
            #diff /= np.amax(diff)
            #out = np.hstack((Iref, I, M, R, diff))
            #out = _float2uint(out, dtype)
            print ('  PSNR = {:.1f} dB, time = {} ms'.format(p, int(runtime)))
        else:
            print ('  - raw image without groundtruth, bypassing metric')
        out = _float2uint(R, dtype)
        
        
        outputname = os.path.join(args.output, os.path.split(fname)[-1])
        # CV color storage..
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(outputname, out)

    if has_groundtruth and n > 0:
        avg_psnr /= n
        print ('+ Average PSNR = {:.1f} dB'.format(avg_psnr))

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/36111627_7991189675.jpg', help='path to input image or folder.')
    parser.add_argument('--output', type=str, default='output', help='path to output folder.')
    parser.add_argument('--noise', type=float, default=0.0, help='standard deviation of additive Gaussian noise, w.r.t to a [0,1] intensity scale.')
    parser.add_argument('--offset_x', type=int, default=0, help='number of pixels to offset the mosaick in the x-axis.')
    parser.add_argument('--offset_y', type=int, default=0, help='number of pixels to offset the mosaick in the y-axis.')
    parser.add_argument('--tile_size', type=int, default=512, help='split the input into tiles of this size.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='use the GPU for processing.')
    parser.add_argument('--mosaic_type', type=str, default='bayer', choices=['bayer', 'xtrans'], help='type of mosaick (xtrans or bayer)')

    parser.add_argument('--linear_input', dest='linear_input', action='store_true')

    parser.set_defaults(gpu=False, linear_input=False)

    args = parser.parse_args()

    if args.noise > NOISE_LEVELS[1] or args.noise < NOISE_LEVELS[0]:
        msg = 'The model was trained on noise levels in [{}, {}]'.format(
                NOISE_LEVELS[0], NOISE_LEVELS[1])
        raise ValueError(msg)

    main(args)


