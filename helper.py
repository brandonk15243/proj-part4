import numpy as np
import scipy.fftpack
from scipy.fftpack import dct, idct
from huffman import *
from bitarray import bitarray


from PIL import Image
import time
import glob
import os
import bokeh.plotting as bk
from bokeh.io import push_notebook
from bokeh.resources import INLINE
from bokeh.models import GlyphRenderer

bk.output_notebook(INLINE)

import re

#####################################################################
# Image Compression Helpers
#####################################################################
Kr, Kg, Kb = 0.299, 0.587, 0.114

ycbcr_mat = np.array([[Kr, Kg, Kb],
                      [-0.5 * (Kr / (1 - Kb)), -0.5 * (Kg / (1 - Kb)), 0.5],
                      [0.5, -0.5 * (Kg / (1 - Kr)), -0.5 * (Kb / (1 - Kr))]])

ycbcr_inv = np.array([[1, 0, 2 - 2 * Kr],
                      [1, -(Kb / Kg) * (2 - 2 * Kb), -(Kr / Kg) * (2 - 2 * Kr)],
                      [1, 2 - 2 * Kb, 0]])

def RGB2YCbCr(im_rgb):
    # Input:  a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]
    # Output: a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]

    # Your code here
    im_ycbcr = im_rgb.dot(ycbcr_mat.T)
    im_ycbcr[:,:, 0] += -128.
    im_ycbcr = np.clip(im_ycbcr, -128.0, 127.0)
    # End of your code
    return im_ycbcr

def YCbCr2RGB(im_ycbcr):
    # Input:  a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]
    # Output: a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]

    # Your code here
    im_ycbcr[:,:,0] += 128.
    im_rgb = im_ycbcr.dot(ycbcr_inv.T)
    im_rgb = np.clip(im_rgb, 0.0, 255.0)
    # End of your code
    return im_rgb

def mirror_pad(img):
        # Input:  a 3D float array, img, representing an RGB image in range [0.0,255.0]
        # Output: a 3D float array, img_pad, mirror padded so the number of rows and columns are multiples of 16

        M, N = img.shape[0:2]
        pad_r = ((16 - (M % 16)) % 16) # number of rows to pad
        pad_c = ((16 - (N % 16)) % 16) # number of columns to pad
        img_pad = np.pad(img, ((0,pad_r), (0,pad_c), (0,0)), "symmetric") # symmetric padding

        return img_pad

def chroma_downsample(C):
    # Input:  an MxN array, C, of chroma values
    # Output: an (M/2)x(N/2) array, C2, of downsampled chroma values

    # Your code here:
    C2 = np.array(Image.fromarray(C).resize((C.shape[1]//2, C.shape[0]//2), resample=Image.BILINEAR)).astype(float)
    # End of your code
    return C2

def chroma_upsample(C2):
    # Input:  an (M/2)x(N/2) array, C2, of downsampled chroma values
    # Output: an MxN array, C, of chroma values

    # Your code here:
    C = np.array(Image.fromarray(C2).resize((C2.shape[1]*2, C2.shape[0]*2), resample=Image.BILINEAR)).astype(float)
    # End of your code
    return C

def dct2(block):
    # Input:  a 2D array, block, representing an image block
    # Output: a 2D array, block_c, of DCT coefficients

    # Your code here:
    block_c = scipy.fftpack.dct(scipy.fftpack.dct(block, type=2, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
    # End of your code
    return block_c

def idct2(block_c):
    # Input:  a 2D array, block_c, of DCT coefficients
    # Output: a 2D array, block, representing an image block

    # Your code here:
    block = scipy.fftpack.idct(scipy.fftpack.idct(block_c, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
    # End of your code

    return block

def quantize(block_c, mode="y", quality=75):
    # Input:  a 2D float array, block_c, of DCT coefficients
    #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D int array, block_cq, of quantized DCT coefficients

    if mode == "y":
        Q = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61 ],
                      [ 12,  12,  14,  19,  26,  58,  60,  55 ],
                      [ 14,  13,  16,  24,  40,  57,  69,  56 ],
                      [ 14,  17,  22,  29,  51,  87,  80,  62 ],
                      [ 18,  22,  37,  56,  68,  109, 103, 77 ],
                      [ 24,  36,  55,  64,  81,  104, 113, 92 ],
                      [ 49,  64,  78,  87,  103, 121, 120, 101],
                      [ 72,  92,  95,  98,  112, 100, 103, 99 ]])
    elif mode == "c":
        Q = np.array([[ 17,  18,  24,  47,  99,  99,  99,  99 ],
                      [ 18,  21,  26,  66,  99,  99,  99,  99 ],
                      [ 24,  26,  56,  99,  99,  99,  99,  99 ],
                      [ 47,  66,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ]])
    else:
        raise Exception("String argument must be 'y' or 'c'.")

    if quality < 1 or quality > 100:
        raise Exception("Quality factor must be in range [1,100].")

    scalar = 5000 / quality if quality < 50 else 200 - 2 * quality # formula for scaling by quality factor
    Q = Q * scalar / 100. # scale the quantization matrix
    Q[Q<1.] = 1. # do not divide by numbers less than 1

    # Quantize the 8x8 block
    # Your code here
    block_cq = np.around(block_c / Q).astype(int)
    # End of your code
    return block_cq

def unquantize(block_cq, mode="y", quality=75):
    # Input:  a 2D int array, block_cq, of quantized DCT coefficients
    #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D float array, block_c, of "unquantized" DCT coefficients (they will still be quantized)

    if mode == "y":
        Q = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61 ],
                      [ 12,  12,  14,  19,  26,  58,  60,  55 ],
                      [ 14,  13,  16,  24,  40,  57,  69,  56 ],
                      [ 14,  17,  22,  29,  51,  87,  80,  62 ],
                      [ 18,  22,  37,  56,  68,  109, 103, 77 ],
                      [ 24,  36,  55,  64,  81,  104, 113, 92 ],
                      [ 49,  64,  78,  87,  103, 121, 120, 101],
                      [ 72,  92,  95,  98,  112, 100, 103, 99 ]])
    elif mode == "c":
        Q = np.array([[ 17,  18,  24,  47,  99,  99,  99,  99 ],
                      [ 18,  21,  26,  66,  99,  99,  99,  99 ],
                      [ 24,  26,  56,  99,  99,  99,  99,  99 ],
                      [ 47,  66,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ]])
    else:
        raise Exception("String argument must be 'y' or 'c'.")

    if quality < 1 or quality > 100:
        raise Exception("Quality factor must be in range [1,100].")

    scalar = 5000 / quality if quality < 50 else 200 - 2 * quality # formula for scaling by quality factor
    Q = Q * scalar / 100. # scale the quantization matrix
    Q[Q<1.] = 1. # do not divide by numbers less than 1

    # Un-quantize the 8x8 block
    # Your code here
    block_c = np.around(block_cq * Q).astype(np.float_)
    # End of your code

    return block_c

def zigzag(block_cq):
    # Input:  a 2D array, block_cq, of quantized DCT coefficients
    # Output: a list, block_cqz, of zig-zag reordered quantized DCT coefficients

    idx = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41,
           34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
           30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

    # Your code here:
    block_cqz = list(np.ravel(block_cq)[idx])
    # End of your code
    return block_cqz

def unzigzag(block_cqz):
    # Input:  a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    # Output: a 2D array, block_cq, of conventionally ordered quantized DCT coefficients

    idx = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41,
           43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38,
           46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]

    # Your code here:
    b = np.zeros(len(idx))
    b[:len(block_cqz)] = block_cqz
    block_cq = np.reshape(b[idx], (int(np.sqrt(len(idx))), int(np.sqrt(len(idx)))))
    # End of your code
    return block_cq

def zrle(block_cqz):
    # Input:  a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    # Output: a list, block_cqzr, of zero-run-length encoded quantized DCT coefficients

    # Your code here:
    cpy = list(block_cqz)[:]

    # Add first elem
    block_cqzr = [cpy.pop(0)]

    # Loop through
    while sum(cpy[0:]) != 0:
        count = 0
        while cpy[0] == 0 and count < 15:
            count += 1
            cpy.pop(0)
        val = cpy.pop(0)
        block_cqzr.append((count, val))
    block_cqzr.append((0,0))
    # End of your code
    return block_cqzr

def unzrle(block_cqzr):
    # Input:  a list, block_cqzr, of zero-run-length encoded quantized DCT coefficients
    # Output: a list, block_cqz, of zig-zag reordered quantized DCT coefficients

    # Your code here:
    cpy = block_cqzr[:]
    block_cqz = [cpy.pop(0)]
    while cpy:
        zrle = cpy.pop(0)
        # zerolen = zrle[0] if zrle[1] != 0 else zrle[0] + 1
        block_cqz.extend((zrle[0] * [0]) + [zrle[1]])
    # End of your code

    return block_cqz[:-1]

def encode_block(block, mode="y", quality=75):
    # Input:  a 2D array, block, representing an image component block
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a bitarray, dc_bits, of Huffman encoded DC coefficients
    #         a bitarray, ac_bits, of Huffman encoded AC coefficients

    block_c = dct2(block)
    block_cq = quantize(block_c, mode, quality)
    block_cqz = zigzag(block_cq)
    block_cqzr = zrle(block_cqz)
    dc_bits = encode_huffman(block_cqzr[0], mode) # DC
    ac_bits = ''.join(encode_huffman(v, mode) for v in block_cqzr[1:]) # AC

    return bitarray(dc_bits), bitarray(ac_bits)

def decode_block(dc_gen, ac_gen, mode="y", quality=75):
    # Inputs: a generator, dc_gen, that yields decoded Huffman DC coefficients
    #         a generator, ac_gen, that yields decoded Huffman AC coefficients
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D array, block, decoded by and yielded from the two generators

    block_cqzr = [next(dc_gen)] # initialize list by yielding from DC generator
    while block_cqzr[-1] != (0,0):
        block_cqzr.append(next(ac_gen)) # append to list by yielding from AC generator until (0,0) is encountered
    block_cqz = unzrle(block_cqzr)
    block_cq = unzigzag(block_cqz)
    block_c = unquantize(block_cq, mode, quality)
    block = idct2(block_c)

    return block

#####################################################################
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def encode_image(img, quality=75):
    # Inputs:  a 3D float array, img, representing an RGB image in range [0.0,255.0]
    #          an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Outputs: a bitarray, Y_dc_bits, the Y component DC bitstream
    #          a bitarray, Y_ac_bits, the Y component AC bitstream
    #          a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #          a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #          a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #          a bitarray, Cr_ac_bits, the Cr component AC bitstream

    M_orig, N_orig = img.shape[0:2]
    img = mirror_pad(img[:,:,0:3])
    M, N = img.shape[0:2]

    im_ycbcr = RGB2YCbCr(img)
    Y = im_ycbcr[:,:,0]
    Cb = chroma_downsample(im_ycbcr[:,:,1])
    Cr = chroma_downsample(im_ycbcr[:,:,2])

    # Y component
    Y_dc_bits = bitarray()
    Y_ac_bits = bitarray()
    for i in np.r_[0:M:8]:
        for j in np.r_[0:N:8]:
            block = Y[i:i+8,j:j+8]
            dc_bits, ac_bits = encode_block(block, "y", quality)
            Y_dc_bits.extend(dc_bits)
            Y_ac_bits.extend(ac_bits)

    # Cb component
    Cb_dc_bits = bitarray()
    Cb_ac_bits = bitarray()
    for i in np.r_[0:M//2:8]:
        for j in np.r_[0:N//2:8]:
            block = Cb[i:i+8,j:j+8]
            dc_bits, ac_bits = encode_block(block, "c", quality)
            Cb_dc_bits.extend(dc_bits)
            Cb_ac_bits.extend(ac_bits)

    # Cr component
    Cr_dc_bits = bitarray()
    Cr_ac_bits = bitarray()
    for i in np.r_[0:M//2:8]:
        for j in np.r_[0:N//2:8]:
            block = Cr[i:i+8,j:j+8]
            dc_bits, ac_bits = encode_block(block, "c", quality)
            Cr_dc_bits.extend(dc_bits)
            Cr_ac_bits.extend(ac_bits)

    bits = (Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits)

    return bits

def decode_image(bits, M, N, quality=75):
    # Inputs: a tuple, bits, containing the following:
    #              a bitarray, Y_dc_bits, the Y component DC bitstream
    #              a bitarray, Y_ac_bits, the Y component AC bitstream
    #              a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #              a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #              a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #              a bitarray, Cr_ac_bits, the Cr component AC bitstream
    #         ints, M and N, the number of rows and columns in the image
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 3D float array, img, representing an RGB image in range [0.0,255.0]

    Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits = bits # unpack bits tuple

    M_orig = M # save original image dimensions
    N_orig = N
    M = M_orig + ((16 - (M_orig % 16)) % 16) # dimensions of padded image
    N = N_orig + ((16 - (N_orig % 16)) % 16)
    num_blocks = M * N // 64 # number of blocks

    # Y component
    Y_dc_gen = decode_huffman(Y_dc_bits.to01(), "dc", "y")
    Y_ac_gen = decode_huffman(Y_ac_bits.to01(), "ac", "y")
    Y = np.empty((M, N))
    for b in range(num_blocks):
        block = decode_block(Y_dc_gen, Y_ac_gen, "y", quality)
        r = (b*8 // N)*8 # row index (top left corner)
        c = b*8 % N # column index (top left corner)
        Y[r:r+8, c:c+8] = block

    # Cb component
    Cb_dc_gen = decode_huffman(Cb_dc_bits.to01(), "dc", "c")
    Cb_ac_gen = decode_huffman(Cb_ac_bits.to01(), "ac", "c")
    Cb2 = np.empty((M//2, N//2))
    for b in range(num_blocks//4):
        block = decode_block(Cb_dc_gen, Cb_ac_gen, "c", quality)
        r = (b*8 // (N//2))*8 # row index (top left corner)
        c = b*8 % (N//2) # column index (top left corner)
        Cb2[r:r+8, c:c+8] = block

    # Cr component
    Cr_dc_gen = decode_huffman(Cr_dc_bits.to01(), "dc", "c")
    Cr_ac_gen = decode_huffman(Cr_ac_bits.to01(), "ac", "c")
    Cr2 = np.empty((M//2, N//2))
    for b in range(num_blocks//4):
        block = decode_block(Cr_dc_gen, Cr_ac_gen, "c", quality)
        r = (b*8 // (N//2))*8 # row index (top left corner)
        c = b*8 % (N//2) # column index (top left corner)
        Cr2[r:r+8, c:c+8] = block

    Cb = chroma_upsample(Cb2)
    Cr = chroma_upsample(Cr2)

    img = YCbCr2RGB(np.stack((Y,Cb,Cr), axis=-1))

    img = img[0:M_orig,0:N_orig,:] # crop out padded parts

    return img

# Tiff stack player

def Tiff_play(path, display_size, frame_rate):
    image_files = sorted(glob.glob(path), key=numericalSort)
    Nframe = len(image_files)

    im = Image.open(image_files[0])
    xdim, ydim= im.size
    display_array = np.zeros((Nframe,ydim,xdim,4),dtype='uint8')
    
    # load image stack
    for i in range(0,Nframe):
        im = Image.open(image_files[i])
        im = im.convert("RGBA")
        imarray = np.array(im)
        display_array[i] = np.flipud(imarray)
        
    # Play video

    wait_time = 1/frame_rate
    normalized_size = display_size
    max_size = np.maximum(xdim,ydim)
    width = (xdim/max_size * normalized_size).astype('int')
    height = (ydim/max_size * normalized_size).astype('int')
    
    counter = 0
    first_round = True
    try:
        while True:
            if counter == 0 and first_round:
                p = bk.figure(x_range=(0,xdim), y_range=(0,ydim), plot_height = height, plot_width = width)
                p.image_rgba(image=[display_array[counter]], x=0, y=0, dw=xdim, dh=ydim, name='video')
                bk.show(p, notebook_handle=True)
                counter += 1
                first_round = False
            else:
                renderer = p.select(dict(name='video', type=GlyphRenderer))
                source = renderer[0].data_source
                source.data['image'] = [display_array[counter]]
                push_notebook()
                if counter == Nframe-1:
                    counter = 0
                else:
                    counter += 1
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        pass
            
# Image_stack loader

def Tiff_load(path):
    image_files = sorted(glob.glob(path), key=numericalSort)
    Nframe = len(image_files)
    
    im = Image.open(image_files[0])
    xdim, ydim= im.size
    image_stack = np.zeros((Nframe,ydim,xdim,3),dtype='uint8')
    
    for i in range(0,Nframe):
        im = Image.open(image_files[i])
        image_stack[i] = np.array(im)
    
    return image_stack

# Image_stack loader with ffmpeg

def imageStack_load(filename):
    path        = filename[:filename.find('.')]+'/'
    os.system("rm -rf {:s}".format(path))
    os.system("mkdir {:s}".format(path))
    os.system("ffmpeg -i {:s} {:s}frame_%2d.tiff".format(filename, path))
    image_files = sorted(glob.glob(path+"*.tiff"), key=numericalSort)
    Nframe = len(image_files)
    
    im = Image.open(image_files[0])
    xdim, ydim= im.size
    image_stack = np.zeros((Nframe,ydim,xdim,3),dtype='uint8')
    
    for i in range(0,Nframe):
        im = Image.open(image_files[i])
        image_stack[i] = np.array(im)
    
    return image_stack

# Save gif with ffmpeg

def GIF_save(path, framerate):
    os.system("ffmpeg -r {:d} -i {:s}frame_%2d.tiff -compression_level 0 -plays 0 -f apng {:s}animation.png".format(framerate, path,path))

# Compute video PSNR

def psnr(ref, meas, maxVal=255):
    assert np.shape(ref) == np.shape(meas), "Test video must match measured vidoe dimensions"


    dif = (ref.astype(float)-meas.astype(float)).ravel()
    mse = np.linalg.norm(dif)**2/np.prod(np.shape(ref))
    psnr = 10*np.log10(maxVal**2.0/mse)
    return psnr
    
