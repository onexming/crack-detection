import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from util import crack_detection, slice_image

strat_time = time.time()
plt.ion()
model_data = "model2/model.cpk-43000"
logdir = "visual"

# 图片地址
im_dir = 'test_image/1.jpg'
strides = 175
image_size = 227
# 建立以裂纹识别器
crack_detect = crack_detection(model_data)
image = np.array(Image.open(im_dir))
# 复制一张图片用作展示
image_visual = image.copy()

# 高分辨率图片发处理
for im, y, x in slice_image(image, strides, image_size):
    # print(im.shape, "===", y, "===", x)
    detect_result = crack_detect.run(im)
    if detect_result == 0:  # 如果没有检测到将该区域变黑
        image_visual[y[0]:y[1], x[0]:x[1], :] = np.zeros_like(image_visual[y[0]:y[1], x[0]:x[1], :])
        plt.clf()  # 清除之前画的图,不清楚会越来越慢
        plt.imshow(image_visual)
        plt.pause(0.0001)
        plt.ioff()  # 关闭交互模型
Image.fromarray(image_visual).save("out.jpg")
print("花费时间为：", time.time() - strat_time)
plt.show()
