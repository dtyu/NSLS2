#!/usr/bin/env python
"""
Created on July 29, 2014, last modified on July 30, 2014

@author: Cheng Chang (cchang@bnl.gov)
         Kenneth Lauer (klauer@bnl.gov)
         Brookhaven National Laboratory
         
This code is for Differential Phase Contrast (DPC) imaging based on 
Fourier-shift fitting.

Reference: Yan, H. et al. Quantitative x-ray phase imaging at the nanoscale by 
           multilayer Laue lenses. Sci. Rep. 3, 1307; DOI:10.1038/srep01307 
           (2013).

The dpc_workflow() function, by default, requires a SOFC folder containing the 
test data in your home directory. The default path for the  results (texts and 
JPEGs) is also your home directory.

"""

from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from os.path import expanduser
from scipy.misc import imsave
from scipy.optimize import minimize



def image_reduction(im, roi=None, bad_pixels=None):
    """ 
    Sum the image data along one dimension
        
    Parameters
    ----------
    im : 2-D numpy array
        store the image data
    
    roi : tuple
        store the top-left and bottom-right coordinates of an rectangular ROI
        roi = (11, 22, 33, 44) --> (11, 22) - (33, 44)
        
    bad_pixels : list
        store the coordinates of bad pixels
        [(1, 5), (2, 6)] --> 2 bad pixels --> (1, 5) and (2, 6) 
    
    Returns
    ----------
    xline : 1-D numpu array
        the sum of the image data along x direction
        
    yline : 1-D numpy array
        the sum of the image data along y direction
        
    """
      
    if bad_pixels is not None:
        for x, y in bad_pixels:
            im[x, y] = 0
                
    if roi is not None:
        x1, y1, x2, y2 = roi
        im = im[x1 : x2 + 1, y1 : y2 + 1]
        
    xline = np.sum(im, axis=0)
    yline = np.sum(im, axis=1)
    yline = yline[46 : 61]
        
    return xline, yline



def ifft1D(data):
    """ 
    1D inverse IFFT 
        
    Parameters
    ----------
    data : 1-D numpy array
     
    Returns
    ----------
    f : 1-D complex numpy array
        IFFT result
        zero-frequency component is shifted to the center
         
    """
    
    f = np.fft.fftshift(np.fft.ifft(data))
        
    return f



def _load_image(filename):
    """ 
    Load and preprocess an image
    return a 2D numpy array 
    
    Parameters
    ----------
    filename : string
        the location and name of an image
        
    nx_prb : ???
    ny_prb : ???
    x_raw : ???
    y_raw : ???
    threshold : ???
    
    Return
    ----------
    t : 2-D numpy array
        store the image data
    
    """
        
    if os.path.exists(filename):
    
        nx_prb=256
        ny_prb=256
        x_raw=512
        y_raw=512
        threshold=0

        diff_array = np.zeros((nx_prb, ny_prb))
            
        if 1:
            with open(filename, 'rb') as f:
                np.fromfile(f, dtype='int32', count=2)
                tmp = np.fromfile(f, dtype='int16', count=x_raw * y_raw)
        else:
            tmp = np.arange(x_raw * y_raw)

        tmp.resize(y_raw, x_raw)
        tmp = np.fliplr(np.transpose(tmp * 1.))
        tmp[np.where(tmp < threshold)] = 0.

        t = np.zeros((516, 516))
        t = np.zeros((x_raw + 4, y_raw + 4))
        t[0:x_raw/2-1, 0:y_raw/2-1] = tmp[0:x_raw/2-1, 0:y_raw/2-1]
        t[x_raw/2+5:x_raw+4, 0:y_raw/2-1] = tmp[x_raw/2+1:x_raw, 0:y_raw/2-1]
        t[0:x_raw/2-1, y_raw/2+5:y_raw+4] = tmp[0:x_raw/2-1, y_raw/2+1:y_raw]
        t[x_raw/2+5:x_raw+4, y_raw/2+5:y_raw+4] = tmp[x_raw/2+1:x_raw, 
                                                      y_raw/2+1:y_raw]

        for i in range(y_raw):
            t[x_raw/2-1:x_raw/2+2, i] = tmp[x_raw/2-1, i] / 3.
            t[x_raw/2+2:x_raw/2+5, i] = tmp[x_raw/2, i] / 3.

        for i in range(x_raw):
            t[i, y_raw/2-1:y_raw/2+2] = tmp[i, y_raw/2-1] / 3.
            t[i, y_raw/2+2:y_raw/2+5] = tmp[i, y_raw/2] / 3.

        t[x_raw/2-1:x_raw/2+2, y_raw/2-1:y_raw/2+2] = tmp[x_raw/2-1, y_raw/2-1]/9.
        t[x_raw/2-1:x_raw/2+2, y_raw/2+2:y_raw/2+5] = tmp[x_raw/2-1, y_raw/2]/9.
        t[x_raw/2+2:x_raw/2+5, y_raw/2-1:y_raw/2+2] = tmp[x_raw/2, y_raw/2-1]/9.
        t[x_raw/2+2:x_raw/2+5, y_raw/2+2:y_raw/2+5] = tmp[x_raw/2, y_raw/2]/9.

        if 0:
            plt.close('all')
            plt.figure()
            plt.imshow(np.log(diff_array[:, :] + 0.001))
        
        return t
        
    else:
        print('Please download and decompress the test data to your home directory\
               Google drive link, https://drive.google.com/file/d/0B3v6W1bQwN_AVjdYdERHUDBsMmM/edit?usp=sharing')
        raise Exception('File not found: %s' % filename)
        

def load_image(filename):
    """
    Load an image
    
    Parameters
    ----------
    filename : string
        the location and name of an image
    
    Return
    ----------
    t : 2-D numpy array
        store the image data
        
    """ 
    
    if os.path.exists(filename):  
        t = plt.imread(filename)
    
    else:
        print('Please download and decompress the test data to your home directory\
               Google drive link, https://drive.google.com/file/d/0B3v6W1bQwN_AVjdYdERHUDBsMmM/edit?usp=sharing')
        raise Exception('File not found: %s' % filename) 
    
    return t


def _cache(data, _rss_cache):
    """ 
    Internal function used by fit()
    Cache calculation results
    
    Parameters
    ----------
    data : 1-D numpy array
    
    Global
    ----------
    _rss_cache : dict
        dict[int] = int
        might be updated within _cache()
        
    Return
    ----------
    beta : complex integer
        beta is only dependent on the data length
    
    """
        
    length = len(data)
    
    try:
        beta = _rss_cache[length]
    except:
        beta = 1j * (np.arange(length) - np.floor(length / 2.0))
        _rss_cache[length] = beta
            
    return beta



def _rss(v, xdata, ydata, beta):
    """ 
    Internal function used by fit()
    Cost function to be minimized in nonlinear fitting
    
    Parameters
    ----------
    v : list
        store the fitting value
        v[0], intensity attenuation
        v[1], phase gradient along x or y direction
    
    xdata : 1-D complex numpy array
        auxiliary data in nonlinear fitting
        returning result of ifft1D()
    
    ydata : 1-D complex numpy array
        auxiliary data in nonlinear fitting
        returning result of ifft1D()
    
    beta : complex integer
        returning value of _cache()
        
    Return
    --------
    residue : float
        residue value
    
    """
    
    fitted_curve = xdata * v[0] * np.exp(v[1] * beta)
    residue = np.sum(np.abs(ydata - fitted_curve) ** 2)
    
    return residue



def fit(ref_f, f, _rss_cache, start_point=[1, 0], solver='Nelder-Mead', tol=1e-1, 
        max_iters=2000):
    """ 
    Nonlinear fitting 
    
    Functions
    ----------
    _rss()
    _cache()
    
    Parameters
    ----------
    start_point : 2-element list
        start_point[0], start-searching point for the intensity attenuation
        start_point[1], start-searching point for the phase gradient
    
    solver : string
        method to solve the nonlinear fitting problem
    
    tol : float
        termination criteria of nonlinear fitting
        
    max_iters : integer
        maximum iterations of nonlinear fitting
        
    Returns:
    ----------
    a : float
        intensity attenuation
    g : float
        phase gradient
    
    """
        
    res = minimize(_rss, start_point, args=(ref_f, f, _cache(ref_f, _rss_cache)),
                    method=solver, tol=tol, options=dict(maxiter=max_iters))
                    
    vx = res.x
    a = vx[0]
    g = vx[1]
        
    return a, g



def recon(gx, gy, dx=0.1, dy=0.1, pad=1, w=1.):
    """ 
    Reconstruct the final phase image 
    
    Parameters
    ----------
    gx : 2-D numpy array
        phase gradient along x direction
    
    gy : 2-D numpy array
        phase gradient along y direction
    
    dx : float
        scanning step size in x direction (in micro-meter)
        
    dy : float
        scanning step size in y direction (in micro-meter)
    
    pad : integer
        padding parameter
        default value, pad = 1 --> no padding
                    p p p
        pad = 3 --> p v p
                    p p p
                    
    w : float
        weighting parameter
        
    Return
    ----------
    phi : 2-D numpy array
        final phase image
        
    """
    
    shape = gx.shape
    rows = shape[0]
    cols = shape[1]
    
    gx_padding = np.zeros((pad * rows, pad * cols), dtype='d')
    gy_padding = np.zeros((pad * rows, pad * cols), dtype='d')
    
    gx_padding[(pad / 2) * rows : (pad / 2 + 1) * rows,
               (pad / 2) * cols : (pad / 2 + 1) * cols] = gx
    gy_padding[(pad / 2) * rows : (pad / 2 + 1) * rows, 
               (pad / 2) * cols : (pad / 2 + 1) * cols] = gy
    
    tx = np.fft.fftshift(np.fft.fft2(gx_padding))
    ty = np.fft.fftshift(np.fft.fft2(gy_padding))
    
    c = np.zeros((pad * rows, pad * cols), dtype=complex)
    
    mid_col = (np.floor((pad * cols) / 2.0) + 1)
    mid_row = (np.floor((pad * rows) / 2.0) + 1)
     
    for i in range(pad * rows):
        for j in range(pad * cols):
            kappax = 2 * np.pi * (j + 1 - mid_col) / (pad * cols * dx)
            kappay = 2 * np.pi * (i + 1 - mid_row) / (pad * rows * dy)
            if kappax == 0 and kappay == 0:
                c[i, j] = 0
            else:
                c[i, j] = -1j * (kappax * tx[i][j] + w * kappay * ty[i][j]) / (kappax ** 2 + w * kappay ** 2)

    c = np.fft.ifftshift(c)
    phi_padding = np.fft.ifft2(c)
    phi_padding = -phi_padding.real
    
    phi = phi_padding[(pad / 2) * rows : (pad / 2 + 1) * rows, 
                      (pad / 2) * cols : (pad / 2 + 1) * cols]
    
    return phi



settings = dict(file_format = expanduser("~") + '/SOFC/SOFC_%05d.tif',
                start_point=[1, 0],
                first_image=1,
                ref_image=1,
                pixel_size=55,
                focus_to_det=1.46e6,
                dx=0.1,
                dy=0.1,
                rows = 121,
                cols = 121,
                energy=19.5,
                roi=None,
                pad=1,
                w=1.,
                bad_pixels=None,
                solver='Nelder-Mead')



def dpc_workflow():
    """
    Calculate DPC using functional modules
    
    Steps
    ----------
    1. Set parameters or load parameters from a given dictionary
    2. Load the reference image
    3. Dimension reduction along x and y direction
    4. 1-D IFFT
    5. Same calculation on each diffraction pattern
        5.1. Read a diffraction pattern
        5.2. Dimension reduction along x and y direction
        5.3. 1-D IFFT
        5.4. Nonlinear fitting
    6. Reconstruct the final phase image
    7. Save intermediate and final results
    
    """

    # Step 1.
    roi = settings['roi']
    rows = settings['rows']
    cols = settings['cols']
    energy = settings['energy']
    pixel_size = settings['pixel_size']
    first_image = settings['first_image']
    file_format = settings['file_format']
    focus_to_det = settings['focus_to_det']
    _rss_cache = {}
    
    # Initialize a, gx, gy and phi
    a = np.zeros((rows, cols), dtype='d')
    gx = np.zeros((rows, cols), dtype='d')
    gy = np.zeros((rows, cols), dtype='d')
    phi = np.zeros((rows, cols), dtype='d')

    # Step 2.
    ref = load_image(settings['file_format'] % settings['ref_image'])
    
    # Step 3.
    refx, refy = image_reduction(ref, roi=roi)

    # Step 4.
    ref_fx = ifft1D(refx)
    ref_fy = ifft1D(refy)
 
    # Step 5.
    for i in range(rows):
        print(i)
        for j in range(cols):
    
            # Calculate diffraction pattern index and get its name
            frame_num = first_image + i * cols + j
            filename = file_format % frame_num
            
            try:
                # Step 5.1.
                im = load_image(filename)
                   
                # Step 5.2. 
                imx, imy = image_reduction(im, roi=roi)
                
                # Step 5.3.
                fx = ifft1D(imx)
                fy = ifft1D(imy)
                
                # Step 5.4.
                _a, _gx = fit(ref_fx, fx, _rss_cache)
                _a, _gy = fit(ref_fy, fy, _rss_cache)
                
                # Store one-point intermediate results
                gx[i, j] = _gx
                gy[i, j] = _gy
                a[i, j] = _a
                
            except Exception as ex:
                print('Failed to calculate %s: %s' % (filename, ex))
                gx[i, j] = 0
                gy[i, j] = 0
                a[i, j] = 0

    # Scale gx and gy. Not necessary all the time
    lambda_ = 12.4e-4 / energy
    gx *= - len(ref_fx) * pixel_size / (lambda_ * focus_to_det)
    gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det)
    
    # Step 6.
    phi = recon(gx, gy)
    
    # Step 7.
    imsave(expanduser("~") + '/phi.jpg', phi)
    np.savetxt(expanduser("~") + '/phi.txt', phi)
    imsave(expanduser("~") + '/a.jpg', a)
    np.savetxt(expanduser("~") + '/a.txt', a)
    imsave(expanduser("~") + '/gx.jpg', gx)
    np.savetxt(expanduser("~") + '/gx.txt', gx)
    imsave(expanduser("~") + '/gy.jpg', gy)
    np.savetxt(expanduser("~") + '/gy.txt', gy)
    
    
if __name__ == '__main__':
    
    t_start = time.time()
    
    dpc_workflow()
    
    t_end = time.time()
    t_total = t_end - t_start
    
    print('Running time: %f' % t_total)














