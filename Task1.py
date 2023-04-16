from scipy.ndimage.filters import convolve
from PIL import Image
import numpy as np
import copy
import cv2
import sys


########################### for tests #########################################


def mse(img1, img2):
    h, w = img1.shape
    N = h * w
    s = 0
    for y in range(h):
        for x in range(w):
            s += (img1[y, x] - img2[y, x]) ** 2
    return s / N


def psnr(img1, img2):
    L = 255
    m = mse(img1, img2)
    if m == 0:
        return 0
    return 10 * np.log10(L ** 2 / m)


def apply_motion_blur(image):
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(image, -1, kernel_motion_blur), kernel_motion_blur


def add_gaussian_noise(image, noise_level):
      h, w = image.shape
      gauss = np.random.normal(0, noise_level, (h, w))
      gauss = gauss.reshape(h, w)
      noisy = image + gauss
      return noisy


###############################################################################


def open_image(filename):
    image = np.array(Image.open(filename).convert(mode = 'L'), dtype = float)
    return image


def create_image(image, filename):
    mn = image.min()
    mx = image.max()
    image = (image - mn) / (mx - mn) * 255
    image = image.astype(np.uint8)
    np.clip(image, 0, 255)
    img_rgb = cv2.cvtColor(np.clip(image, 0, 255), cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img_rgb, mode = "RGB")
    img.save(filename)
    

def d_L2_residual(img_i, ker, input_img):
    diff = convolve(img_i, ker) - input_img
    return 2 * convolve(diff, ker[::-1, ::-1])


def S_y(img_i, shift_y):
    if shift_y == 0:
        return img_i
    new_img = np.zeros(img_i.shape, dtype = float)
    if shift_y == -1:
        new_img[0 : img_i.shape[0] - 1, :] = img_i[1 : img_i.shape[0], :]
    if shift_y == 1:
        new_img[1 : img_i.shape[0], :] = img_i[0 : img_i.shape[0] - 1, :]
    return new_img


def S_x(img_i, shift_x):
    if shift_x == 0:
        return img_i
    new_img = np.zeros(img_i.shape, dtype = float)
    if shift_x == -1:
        new_img[:, 1 : img_i.shape[1]] = img_i[:, 0 : img_i.shape[1] - 1]
    if shift_x == 1:
        new_img[:, 0 : img_i.shape[1] - 1] = img_i[:, 1 : img_i.shape[1]]
    return new_img


def operators_part_1(img_i, shift_x, shift_y):
    sgn = np.sign(S_x(S_y(img_i, shift_y), shift_x) - img_i)
    return S_x(S_y(sgn, -shift_y), -shift_x) - sgn


def operators_part_2(img_i, shift_x, shift_y):
    sgn = np.sign(S_x(S_y(img_i, shift_y), shift_x) + S_x(S_y(img_i, -shift_y), -shift_x) - 2 * img_i)
    return S_x(S_y(sgn, -shift_y), -shift_x) + S_x(S_y(sgn, shift_y), shift_x) - 2 * sgn


def d_btv_2(img_i):
    res = np.zeros(img_i.shape)
    shifts_set = [[1, 0], [0, 1], [1, 1], [1, -1]]
    for shift in shifts_set:
        res += (operators_part_1(img_i, shift[0], shift[1]) \
            + 0.1 * operators_part_2(img_i, shift[0], shift[1])) / np.linalg.norm(shift)
    return res


def minimize(prev_beta_i, beta_i, img_i, ker, input_img, v_i, noise_level):
    alpha = 0.4 * (1 + noise_level)
    mu = 0.85
    img_i = img_i + prev_beta_i * mu * v_i
    g_i = d_L2_residual(img_i, ker, input_img) + alpha * d_btv_2(img_i)
    v_i = mu * v_i - g_i
    img_i = img_i + beta_i * v_i
    return img_i, v_i


########################### for tests #########################################


"""input_image = r'varmethods\task_testdata\blurred.bmp'
kernel = r'varmethods\task_testdata\kernel.bmp'
output_image = r'varmethods\result.bmp'
noise_level = 20

input_img = open_image(input_image)

ker = open_image(kernel)
print(ker.shape)
ker_sum = (np.sum(np.concatenate(ker)))
ker = ker / ker_sum"""

#input_img, ker = apply_motion_blur(input_img)

#input_img = add_gaussian_noise(input_img, noise_level)

#create_image(input_img, r'varmethods\new_blurred.bmp')


###############################################################################


input_image =  sys.argv[1]
kernel = sys.argv[2]
output_image = sys.argv[3]
noise_level = float(sys.argv[4])

input_img = open_image(input_image)

ker = open_image(kernel)
ker_sum = (np.sum(np.concatenate(ker)))
ker = ker / ker_sum

iter_count = 100
img_i = copy.deepcopy(input_img)

beta_1 = 0.75 if noise_level < 10.0 else 7.5 / noise_level
beta_100 = 0.0075 if noise_level < 10.0 else 0.075 / noise_level
beta_i = 0
v_i = np.zeros(img_i.shape, dtype = float)
for i in range(1, iter_count + 1):
    #print("iter ", i, " begins")
    prev_beta_i = beta_i
    beta_i = beta_1 * (beta_100 / beta_1) ** ((i - 1) / (iter_count - 1))
    img_i, v_i = minimize(prev_beta_i, beta_i, img_i, ker, input_img, v_i, noise_level)
    #print("iter ", i, " ends")
    
    
########################### for tests #########################################    
    
    
#reference_image = r'varmethods\task_testdata\reference.bmp'
#reference_img = open_image(reference_image)

#print(psnr(input_img, reference_img)) # исходная с эталоном
#print(psnr(img_i, reference_img)) # восстановленная с эталоном


###############################################################################

create_image(img_i, output_image)
