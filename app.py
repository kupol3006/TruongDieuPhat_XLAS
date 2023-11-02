from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import scipy
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageTk
import scipy.fftpack
import math
import random
import scipy.ndimage as sn

app = Flask(__name__)

exercise_folder = os.path.join('exercise')

save_folder = os.path.join('static','saveimages')
list_method_1 = ['inverse_transformation', 'gamma_correction', 'log_transformation', 'contrast_stretching', 'histogram_equalization']
list_method_2 = ['fast_fourier_process', 'butterworth_lowpass_filter', 'butterworth_highpass_filter']

def inverse_transformation(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    im_1 = np.asarray(img_gray)
    im_2 = 255 - im_1
    new_img = Image.fromarray(im_2)
    new_filename ='after_inverse_' + img
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'],new_filename)
    return img_show
def gamma_correction(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    # if request.form['number'].strip() != '':
    #     gamma = float(request.form['number'])
    # else:
    #     gamma = 1
    gamma = 5
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    im_1 = np.asarray(img_gray)
    b1 = im_1.astype(float)

    b2 = np.max(b1)

    b3 = b1/b2
    
    b2 = np.log(b3) * gamma

    c = np.exp(b2) * 255.0

    c1 = c.astype(np.uint8)

    new_img = Image.fromarray(c1)
    new_filename ='after_gamma' + img
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def log_transformation(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    im_1 = np.asarray(img_gray)

    b1 = im_1.astype(float)

    b2 = np.max(b1)
    
    c = (128.0 * np.log(1 + b1))/np.log(1+b2)

    c1 = c.astype(np.uint8)

    new_img = Image.fromarray(c1)
    new_filename ='after_log_' + img
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show


def histogram_equalization(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)

    bl = iml.flatten()

    hist, bins = np.histogram(iml, 256, [0,255])

    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)

    num_cdf_m = (cdf_m-cdf_m.min())*255
    den_cdf_m = (cdf_m.max() - cdf_m.min())
    cdf_m = num_cdf_m/den_cdf_m

    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    im2 = cdf[bl]
    
    im3 = np.reshape(im2, iml.shape)
    new_img = Image.fromarray(im3)
    new_filename ='after_histogram_' + img
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def contrast_stretching(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)
    b = iml.max()
    a = iml.min()

    c = iml.astype(float)

    im2 = 255* (c-a)/(b-a)
    im3 = im2
    im4 = im3.astype(np.uint8)
    new_img = Image.fromarray(im4)

    new_filename ='after_contrast_' + img 
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def fast_fourier_process(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)
    c = abs(scipy.fftpack.fft2(iml))

    d = scipy.fftpack.fftshift(c)
    d = d.astype(np.uint8)

    new_img = Image.fromarray(d)

    new_filename ='after_fast_fourier_' + img 
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def butterworth_lowpass_filter(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)
    c = abs(scipy.fftpack.fft2(iml))

    d = scipy.fftpack.fftshift(c)
    M = d.shape[0]
    N = d.shape[1]

    H = np.ones((M,N))

    center1 = M/2
    center2 = N/2

    d_0 = 30.0

    t1 = 1
    t2 = 2*t1

    for i in range(1,M):
        for j in range (1,N):
            r1 = (i - center1)**2 + (j-center2)**2
            r = math.sqrt(r1)
            if r> d_0:
                H[i, j] = 1/(1+(r/d_0)**t1)

    H = H.astype(float)
    H = Image.fromarray(H)

    con = d * H

    e = abs(scipy.fftpack.ifft2(con))

    e = e.astype(np.uint8)

    new_img = Image.fromarray(e)

    new_filename ='after_butterworth_lowpass_' + img 
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def butterworth_highpass_filter(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)
    c = abs(scipy.fftpack.fft2(iml))

    d = scipy.fftpack.fftshift(c)
    M = d.shape[0]
    N = d.shape[1]

    H = np.ones((M,N))

    center1 = M/2
    center2 = N/2

    d_0 = 30.0

    t1 = 1
    t2 = 2*t1

    for i in range(1,M):
        for j in range (1,N):
            r1 = (i - center1)**2 + (j-center2)**2
            r = math.sqrt(r1)
            if r> d_0:
                H[i, j] = 1/(1+(r/d_0)**t2)

    H = H.astype(float)
    H = Image.fromarray(H)

    con = d * H

    e = abs(scipy.fftpack.ifft2(con))

    e = e.astype(np.uint8)

    new_img = Image.fromarray(e)

    new_filename ='after_butterworth_highpass_' + img 
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def butterworth_highpass_filter_with_max_filter(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)
    c = abs(scipy.fftpack.fft2(iml))

    d = scipy.fftpack.fftshift(c)
    M = d.shape[0]
    N = d.shape[1]

    H = np.ones((M,N))

    center1 = M/2
    center2 = N/2

    d_0 = 30.0

    t1 = 1
    t2 = 2*t1

    for i in range(1,M):
        for j in range (1,N):
            r1 = (i - center1)**2 + (j-center2)**2
            r = math.sqrt(r1)
            if r> d_0:
                H[i, j] = 1/(1+(r/d_0)**t2)

    H = H.astype(float)
    H = Image.fromarray(H)

    con = d * H

    e = abs(scipy.fftpack.ifft2(con))

    e = e.astype(np.uint8)

    b = sn.maximum_filter(e, size=5, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)

    new_img = Image.fromarray(b)

    new_filename ='after_butterworth_highpass_' + img 
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

def butterworth_lowpass_filter_with_min_filter(img):
    path = os.getcwd() +"\\"+ os.path.join(app.config['EXERCISE'], img)
    img_1 = cv2.imread(path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    iml = np.asarray(img_gray)
    c = abs(scipy.fftpack.fft2(iml))

    d = scipy.fftpack.fftshift(c)
    M = d.shape[0]
    N = d.shape[1]

    H = np.ones((M,N))

    center1 = M/2
    center2 = N/2

    d_0 = 30.0

    t1 = 1
    t2 = 2*t1

    for i in range(1,M):
        for j in range (1,N):
            r1 = (i - center1)**2 + (j-center2)**2
            r = math.sqrt(r1)
            if r> d_0:
                H[i, j] = 1/(1+(r/d_0)**t1)

    H = H.astype(float)
    H = Image.fromarray(H)

    con = d * H

    e = abs(scipy.fftpack.ifft2(con))

    e = e.astype(np.uint8)

    b = sn.minimum_filter(e, size=5, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)

    new_img = Image.fromarray(b)

    new_filename ='after_butterworth_lowpass_' + img 
    new_img.save(os.path.join(app.config['SAVE'],new_filename))
    img_show = os.path.join(app.config['SAVE'], new_filename)
    return img_show

app.config['EXERCISE'] = exercise_folder
app.config['SAVE'] = save_folder
@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('homepage.html')
@app.route('/inverse', methods=['GET', 'POST'])
def inverse():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = inverse_transformation(item[0])
        img2 = inverse_transformation(item[1])
        img3 = inverse_transformation(item[2])

        return render_template('inverse_image_render.html', img1=img1, img2=img2, img3=img3)
    return render_template('inverse_image_render.html')

@app.route('/gamma', methods=['GET', 'POST'])
def gamma():
    global gamma
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = gamma_correction(item[0])
        img2 = gamma_correction(item[1])
        img3 = gamma_correction(item[2])
        return render_template('gamma_image_render.html', img1=img1, img2=img2, img3=img3)
    return render_template('gamma_image_render.html')

@app.route('/log', methods=['GET', 'POST'])
def log():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = log_transformation(item[0])
        img2 = log_transformation(item[1])
        img3 = log_transformation(item[2])
        return render_template('log_image_render.html', img1=img1, img2=img2, img3=img3)
    return render_template('log_image_render.html')

@app.route('/histogram', methods=['GET', 'POST'])
def histogram():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = histogram_equalization(item[0])
        img2 = histogram_equalization(item[1])
        img3 = histogram_equalization(item[2])
        return render_template('histogram_image_render.html', img1=img1, img2=img2, img3=img3)
    return render_template('histogram_image_render.html')

@app.route('/contrast', methods=['GET', 'POST'])
def contrast():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = contrast_stretching(item[0])
        img2 = contrast_stretching(item[1])
        img3 = contrast_stretching(item[2])
        
        return render_template('contrast_image_render.html', img1=img1, img2=img2, img3=img3)
    return render_template('contrast_image_render.html')

@app.route('/fast_fourier', methods=['GET', 'POST'])
def fast_fourier():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = fast_fourier_process(item[0])
        img2 = fast_fourier_process(item[1])
        img3 = fast_fourier_process(item[2])
        
        return render_template('fast_fourier_image_render.html', img1=img1, img2=img2, img3=img3)
    return render_template('fast_fourier_image_render.html')

@app.route('/butterworth_lowpass', methods=['GET', 'POST'])
def butterworth_lowpass():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = butterworth_lowpass_filter(item[0])
        img2 = butterworth_lowpass_filter(item[1])
        img3 = butterworth_lowpass_filter(item[2])
        return render_template('butterworth_lowpass_image_render.html',img1=img1, img2=img2, img3=img3)
    return render_template('butterworth_lowpass_image_render.html')
@app.route('/butterworth_highpass', methods=['GET', 'POST'])
def butterworth_highpass():
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        img1 = butterworth_highpass_filter(item[0])
        img2 = butterworth_highpass_filter(item[1])
        img3 = butterworth_highpass_filter(item[2])
        return render_template('butterworth_highpass_image_render.html',img1=img1, img2=img2, img3=img3)
    return render_template('butterworth_highpass_image_render.html')
@app.route('/random_method_1', methods=['GET', 'POST'])
def random_method_1():
    method = random.choice(list_method_1)
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        if method == 'inverse_transformation':
            img1 = inverse_transformation(item[0])
            img2 = inverse_transformation(item[1])
            img3 = inverse_transformation(item[2])
            return render_template('inverse_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
        elif method == 'gamma_correction':
            img1 = gamma_correction(item[0])
            img2 = gamma_correction(item[1])
            img3 = gamma_correction(item[2])
            return render_template('gamma_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
        elif method == 'log_transformation':
            img1 = log_transformation(item[0])
            img2 = log_transformation(item[1])
            img3 = log_transformation(item[2])
            return render_template('log_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
        elif method == 'histogram_equalization':
            img1 = histogram_equalization(item[0])
            img2 = histogram_equalization(item[1])
            img3 = histogram_equalization(item[2])
            return render_template('histogram_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
        else:
            img1 = contrast_stretching(item[0])
            img2 = contrast_stretching(item[1])
            img3 = contrast_stretching(item[2])
            return render_template('contrast_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
    return render_template('homepage.html')

@app.route('/random_method_2', methods=['GET', 'POST'])
def random_method_2():
    method = random.choice(list_method_2)
    if request.method == 'POST':
        item = os.listdir(app.config['EXERCISE'])
        if method == 'fast_fourier_process':
            img1 = fast_fourier_process(item[0])
            img2 = fast_fourier_process(item[1])
            img3 = fast_fourier_process(item[2])
            return render_template('fast_fourier_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
        elif method == 'butterworth_lowpass_filter':
            img1 = butterworth_lowpass_filter_with_min_filter(item[0])
            img2 = butterworth_lowpass_filter_with_min_filter(item[1])
            img3 = butterworth_lowpass_filter_with_min_filter(item[2])
            return render_template('butterworth_lowpass_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
        elif method == 'butterworth_highpass_filter':
            img1 = butterworth_highpass_filter_with_max_filter(item[0])
            img2 = butterworth_highpass_filter_with_max_filter(item[1])
            img3 = butterworth_highpass_filter_with_max_filter(item[2])
            return render_template('butterworth_highpass_image_render.html', img1=img1, img2=img2, img3=img3, method=method)
    return render_template('homepage.html')
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost',5000,app)