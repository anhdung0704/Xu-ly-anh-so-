BT1: 
from PIL import Image  #Nhập Image từ thư viện PIL
import numpy as np     #Nhập thư viện numpy với viết tắt là np
img = Image.open("dog.png")       #Dùng phương thức Image.open để mở file ảnh dog.png gắn vào biến img
img.show()             #Mở ảnh 

import numpy as np             #Nhập thư viện numpy với viết tắt là np
import imageio.v2 as iio       #Nhập thư viện Imageio phiên bản 2 với viết tắt là iio
import matplotlib.pylab as plt     #Nhập thư viện Matplotlib, cụ thể là phần pylab, để vẽ và hiển thị ảnh.
from skimage import io          #Nhập module io từ thư viện skimage
data = io.imread('dog.png', as_gray = True)    #Đọc file ảnh dog.png dưới dạng ảnh xám
plt.imshow(data,cmap='gray')        #Hiển thị ảnh xám vừa đọc.
plt.show()

import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
data = iio.imread('dog.png')            #Đọc ảnh dog.png và lưu vào biến data.
bdata = (data[:,:,1] + data[:,:,2])     #data[:, :, 1] → kênh Green (G), data[:, :, 2] → kênh Blue (B).  
plt.imshow(bdata)                       #Hiển thị ảnh
plt.show


BT3:
import numpy as np 
import imageio.v2 as iio
import matplotlib.pylab as plt
import colorsys

rgb = iio.imread('dog.png')              #Đọc ảnh RGB
rgb2hsv = np.vectorize (colorsys.rgb_to_hsv)      #Chuyển RGB sang HSV
h ,s ,v = rgb2hsv(rgb[:,:,0], rgb[:,:,1], rgb[:,:,2])      
h *= h                          #Tăng độ sắc nét
hsv2rgb = np.vectorize (colorsys.hls_to_rgb)      # Chuyển lại từ HSV sang RGB
rgb2 = hsv2rgb(h,s,v)
rgb2 = np.array(rgb2).transpose((1,2,0))          #Chuyển dữ liệu kết quả về định dạng ảnh
plt.imshow(rgb2)
plt.show()


BT4:
import numpy as np
import imageio.v2 as iio
import colorsys

img = iio.imread('baby.jpeg') / 255.0      #Đọc ảnh và chuẩn hóa với giá trị từ 0-255

hsv_img = np.zeros_like(img)               #Tạo 1 mảng rỗng chứa dữ liệu HSV

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r, g, b = img[i, j]                #Lấy 3 kênh màu RGB tại diểm đó
        h, s, v = colorsys.rgb_to_hsv(r, g, b)     #Chuyển RGB sang HSV

        h_new = 1/3
        v_new = 0.75 * v
        hsv_img[i, j] = [h_new, s, v_new]

rgb_new = np.zeros_like(img)            #Tạo 1 mảng rỗng chứa ảnh mới sau khi chuyển về RGB
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        h, s, v = hsv_img[i, j]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_new[i, j] = [r, g, b]

rgb_new_img = (rgb_new * 255).astype(np.uint8)       #Chuyển về định dạng uint8 và lưu
iio.imwrite('hsv_modified.jpg', rgb_new_img)


BT5:
import numpy as np 
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import colorsys
from skimage import io


a = io.imread('dog.png', as_gray=True)            #Đọc file ảnh dog.png thành ảnh xám


k = np.ones((5,5))/25                    #Tạo kernel trung bình 5x5, 1 phần tử có giá trị 1/25

b = sn.convolve(a, k)                    
b_uint8 = (b * 255).clip(0,255).astype(np.uint8)          #Biến đổi ảnh mờ b từ [0.0 – 1.0] sang [0 – 255] và chuyển sang kiểu uint8
iio.imsave('dog_mean_filter.png', b_uint8)                #Lưu ảnh với tên mới

print(b_uint8)
plt.imshow(b_uint8, cmap='gray')
plt.show()


