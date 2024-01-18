# 在图像中检测边缘和轮廓

本章介绍以下主题：

+   加载、显示和保存图像

+   图像翻转和缩放

+   腐蚀和膨胀

+   图像分割

+   模糊和锐化图像

+   在图像中检测边缘

+   直方图均衡化

+   在图像中检测角点

# 介绍

图像处理在几乎所有工程和医学应用中都起着至关重要的作用，用于从灰度/彩色图像中提取和评估感兴趣区域。图像处理方法包括预处理、特征提取和分类。预处理用于增强图像的质量；这包括自适应阈值处理、对比度增强、直方图均衡化和边缘检测。特征提取技术用于从图像中提取显著特征，以供后续分类使用。

本章介绍了构建图像预处理方案的程序。

# 加载、显示和保存图像

本节介绍了如何通过OpenCV-Python处理图像。此外，我们讨论了如何加载、显示和保存图像。

# 如何做...

1.  导入计算机视觉包-`cv2`：

```py
import cv2 
```

1.  使用内置的`imread`函数读取图像：

```py
image = cv2.imread('image_1.jpg')
```

1.  使用内置的`imshow`函数显示原始图像：

```py
cv2.imshow("Original", image) 
```

1.  等待按下任意键：

```py
cv2.waitKey(0) 
```

1.  使用内置的`imwrite`函数保存图像：

```py
cv2.imwrite("Saved Image.jpg", image) 
```

1.  用于执行Python程序`Load_Display_Save.py`的命令如下所示：

![](Images/e2747d29-f33d-4c11-8241-c7bc713334ff.png)

1.  执行`Load_Display_Save.py`后获得的结果如下所示：

![](Images/034ff3a4-7d68-467b-b4a8-0f1bd5117ae8.png)

# 图像翻转

在图像翻转操作中，我们可以水平、垂直、水平和垂直翻转输入图像。

# 如何做...

1.  导入计算机视觉包-`cv2`：

```py
import cv2 
```

1.  使用内置的`imread`函数读取图像：

```py
image = cv2.imread('image_2.jpg')
```

1.  使用内置的`imshow`函数显示原始图像：

```py
cv2.imshow("Original", image) 
```

1.  等待按下任意键：

```py
cv2.waitKey(0) 
```

1.  对测试图像执行所需操作：

```py
# cv2.flip is used to flip images 
# Horizontal flipping of images using value '1' 
flipping = cv2.flip(image, 1) 
```

1.  显示水平翻转的图像：

```py
# Display horizontally flipped image 
cv2.imshow("Horizontal Flipping", flipping) 
```

1.  等待按下任意键：

```py
cv2.waitKey(0) 
```

1.  执行输入图像的垂直翻转：

```py
# Vertical flipping of images using value '0' 
flipping = cv2.flip(image, 0) 
```

1.  显示垂直翻转的图像：

```py
cv2.imshow("Vertical Flipping", flipping) 
```

1.  等待按下任意键：

```py
cv2.waitKey(0) 
```

1.  显示处理后的图像：

```py
# Horizontal & Vertical flipping of images using value '-1' 
flipping = cv2.flip(image, -1) 
# Display horizontally & vertically flipped image 
cv2.imshow("Horizontal & Vertical Flipping", flipping) 
# Wait until any key is pressed 
cv2.waitKey(0)
```

1.  停止执行并显示结果：

```py
# Close all windows 
cv2.destroyAllWindows() 
```

1.  用于执行`Flipping.py` Python程序的命令如下所示：

![](Images/7d0a6ede-4840-4ab4-9776-d44e9f6ebe1b.png)

1.  执行`Flipping.py`后获得的原始和水平翻转的图像如下所示：

![](Images/8ae77f4f-9f1b-47a6-9703-31c2970f9fb1.png)

以下是水平翻转的图片：

![](Images/77684d2e-ae13-4b2c-941c-d37e4b1008b5.png)

1.  执行`Flipping.py`后获得的垂直、水平和垂直翻转的图像如下所示：

![](Images/85e084c3-2d5c-4eae-b489-e57a93e931a0.png)

以下是水平和垂直翻转的图片：

![](Images/e49c540d-e711-4112-85b3-d433c59fea9d.png)

# 图像缩放

图像缩放用于根据要求修改输入图像的尺寸。在OpenCV中通常使用三种类型的缩放操作符，它们是立方、区域和线性插值。

# 如何做...

1.  创建一个新的Python文件并导入以下包：

```py
# Scaling (Resizing) Images - Cubic, Area, Linear Interpolations 
# Interpolation is a method of estimating values between known data points  
# Import Computer Vision package - cv2 
import cv2 
# Import Numerical Python package - numpy as np 
import numpy as np 
```

1.  使用内置的`imread`函数读取图像：

```py
image = cv2.imread('image_3.jpg') 
```

1.  使用内置的`imshow`函数显示原始图像：

```py
cv2.imshow("Original", image) 
```

1.  等待按下任意键：

```py
cv2.waitKey() 
```

1.  根据操作员的命令调整图像大小：

```py
# cv2.resize(image, output image size, x scale, y scale, interpolation) 
```

1.  使用立方插值调整图像大小：

```py
# Scaling using cubic interpolation 
scaling_cubic = cv2.resize(image, None, fx=.75, fy=.75, interpolation = cv2.INTER_CUBIC) 
```

1.  显示输出图像：

```py
# Display cubic interpolated image 
cv2.imshow('Cubic Interpolated', scaling_cubic) 
```

1.  等待按下任意键：

```py
cv2.waitKey()
```

1.  使用区域插值调整图像大小：

```py
# Scaling using area interpolation 
scaling_skewed = cv2.resize(image, (600, 300), interpolation = cv2.INTER_AREA) 
```

1.  显示输出图像：

```py
# Display area interpolated image 
cv2.imshow('Area Interpolated', scaling_skewed)  
```

1.  等待操作员的指示：

```py
# Wait until any key is pressed 
cv2.waitKey() 
```

1.  使用线性插值调整图像大小：

```py
# Scaling using linear interpolation 
scaling_linear  = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR) 
```

1.  显示输出图像：

```py
# Display linear interpolated image 
cv2.imshow('Linear Interpolated', scaling_linear)  
```

1.  等待按下任意键：

```py
cv2.waitKey() 
```

1.  完成图像缩放任务后，终止程序执行：

```py
# Close all windows 
cv2.destroyAllWindows() 
```

1.  用于执行`Scaling.py` Python程序的命令如下所示：

![](Images/e932fc3b-9c5f-452d-83c2-8480e08d0f11.png)

1.  用于缩放的原始图像如下所示：

![](Images/37709e13-bb28-4616-ba74-4bf2b18d9fdc.png)

1.  执行`Scaling.py`文件后获得的线性插值输出如下所示：

![](Images/f451fa07-f8d7-4ce8-88df-6f18f6abe93b.png)

1.  执行`Scaling.py`文件后获得的面积插值输出如下所示：

![](Images/7bb33217-91fc-408c-8107-857eca6a88f4.png)

1.  执行`Scaling.py`文件后获得的立方插值输出如下所示：

![](Images/146eb167-7623-4716-a6ae-0d7ef0e973e3.png)

# 腐蚀和膨胀

腐蚀和膨胀是形态学操作。腐蚀去除图像中对象边界的像素，而膨胀在图像中对象的边界上添加像素。

# 如何做...

1.  导入计算机视觉包 - `cv2`：

```py
import cv2 
```

1.  导入数值Python包 - `numpy as np`：

```py
import numpy as np 
```

1.  使用内置的`imread`函数读取图像：

```py
image = cv2.imread('image_4.jpg')
```

1.  使用内置的`imshow`函数显示原始图像：

```py
cv2.imshow("Original", image) 
```

1.  等待按任意键：

```py
cv2.waitKey(0) 
```

1.  给定形状和类型，用1填充：

```py
# np.ones(shape, dtype) 
# 5 x 5 is the dimension of the kernel, uint8: is an unsigned integer (0 to 255) 
kernel = np.ones((5,5), dtype = "uint8") 
```

1.  `cv2.erode`是用于腐蚀的内置函数：

```py
# cv2.erode(image, kernel, iterations) 
erosion = cv2.erode(image, kernel, iterations = 1) 
```

1.  使用内置的`imshow`函数显示腐蚀后的图像：

```py
cv2.imshow("Erosion", erosion) 
```

1.  等待按任意键：

```py
cv2.waitKey(0) 
```

1.  `cv2.dilate`是用于膨胀的内置函数：

```py
# cv2.dilate(image, kernel, iterations) 
dilation = cv2.dilate(image, kernel, iterations = 1) 
```

1.  使用内置的`imshow`函数显示膨胀后的图像：

```py
cv2.imshow("Dilation", dilation) 
```

1.  等待按任意键：

```py
cv2.waitKey(0) 
```

1.  关闭所有窗口：

```py
cv2.destroyAllWindows()
```

1.  用于执行`Erosion_Dilation.py`文件的命令如下所示：

![](Images/f3c5589b-a112-48f9-8145-a1c2e4b552b7.png)

1.  用于执行`Erosion_Dilation.py`文件的输入图像如下所示：

![](Images/131dfbf0-3273-43fb-82ca-6408ada76501.png)

1.  执行`Erosion_Dilation.py`文件后获得的腐蚀图像如下所示：

![](Images/1a1c5a64-8e8c-4a4f-bda9-c08b3622108d.png)

1.  执行`Erosion_Dilation.py`文件后获得的膨胀图像如下所示：

![](Images/6ad113bb-0abf-4be3-ad07-80b4496dd6d6.png)

# 图像分割

分割是将图像分成不同区域的过程。轮廓是对象边界周围的线条或曲线。本节讨论了使用轮廓进行图像分割。

# 如何做...

1.  导入计算机视觉包 - `cv2`：

```py
import cv2 
# Import Numerical Python package - numpy as np 
import numpy as np 
```

1.  使用内置的`imread`函数读取图像：

```py
image = cv2.imread('image_5.jpg') 
```

1.  使用内置的`imshow`函数显示原始图像：

```py
cv2.imshow("Original", image) 
```

1.  等待按任意键：

```py
cv2.waitKey(0) 
```

1.  执行`Canny`边缘检测系统：

```py
# cv2.Canny is the built-in function used to detect edges 
# cv2.Canny(image, threshold_1, threshold_2) 
canny = cv2.Canny(image, 50, 200) 
```

1.  使用内置的`imshow`函数显示检测到的边缘输出图像：

```py
cv2.imshow("Canny Edge Detection", canny) 
```

1.  等待按任意键：

```py
cv2.waitKey(0)
```

1.  执行轮廓检测系统：

```py
# cv2.findContours is the built-in function to find contours 
# cv2.findContours(canny, contour retrieval mode, contour approximation mode) 
# contour retrieval mode: cv2.RETR_LIST (retrieves all contours)  
# contour approximation mode: cv2.CHAIN_APPROX_NONE (stores all boundary points) 
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
```

1.  在图像上勾画轮廓：

```py
# cv2.drawContours is the built-in function to draw contours 
# cv2.drawContours(image, contours, index of contours, color, thickness) 
cv2.drawContours(image, contours, -1, (255,0,0), 10) 
# index of contours = -1 will draw all the contours 
```

1.  显示图像的勾画轮廓：

```py
# Display contours using imshow built-in function 
cv2.imshow("Contours", image) 
```

1.  等待按任意键：

```py
cv2.waitKey() 
```

1.  终止程序并显示结果：

```py
# Close all windows 
cv2.destroyAllWindows() 
```

1.  执行`Image_Segmentation.py`文件后获得的结果如下所示：

![](Images/5782941a-b4e6-4332-8318-d012e6f2d143.png)

以下是边缘检测输出：

![](Images/b1b74744-bea2-48cd-b867-b9cc58c61326.png)

# 模糊和锐化图像

模糊和锐化是用于增强输入图像的图像处理操作。

# 如何做...

1.  导入计算机视觉包 - `cv2`：

```py
import cv2 
# Import Numerical Python package - numpy as np 
import numpy as np 
```

1.  使用内置的`imread`函数读取图像：

```py
image = cv2.imread('image_6.jpg') 
```

1.  使用内置的`imshow`函数显示原始图像：

```py
cv2.imshow("Original", image) 
```

1.  等待按任意键：

```py
cv2.waitKey(0) 
```

1.  执行模糊操作的像素级操作：

```py
# Blurring images: Averaging, cv2.blur built-in function 
# Averaging: Convolving image with normalized box filter 
# Convolution: Mathematical operation on 2 functions which produces third function. 
# Normalized box filter having size 3 x 3 would be: 
# (1/9)  [[1, 1, 1], 
#         [1, 1, 1], 
#         [1, 1, 1]] 
blur = cv2.blur(image,(9,9)) # (9 x 9) filter is used  
```

1.  显示模糊的图像：

```py
cv2.imshow('Blurred', blur) 
```

1.  等待按任意键：

```py
cv2.waitKey(0)
```

1.  执行锐化操作的像素级操作：

```py
# Sharpening images: Emphasizes edges in an image 
kernel = np.array([[-1,-1,-1],  
                   [-1,9,-1],  
                   [-1,-1,-1]]) 
# If we don't normalize to 1, image would be brighter or darker respectively     
# cv2.filter2D is the built-in function used for sharpening images 
# cv2.filter2D(image, ddepth, kernel) 
# ddepth = -1, sharpened images will have same depth as original image 
sharpened = cv2.filter2D(image, -1, kernel) 
```

1.  显示锐化后的图像：

```py
cv2.imshow('Sharpened', sharpened) 
```

1.  等待按任意键：

```py
cv2.waitKey(0) 
```

1.  终止程序执行：

```py
# Close all windows 
cv2.destroyAllWindows() 
```

1.  用于执行`Blurring_Sharpening.py`的命令如下所示：

![](Images/9d3d7e80-2065-4b49-9fb3-73ba8f019299.png)

1.  用于执行`Blurring_Sharpening.py`文件的输入图像如下所示：

![](Images/0fc19880-c1ed-429d-810f-cc0d99e00029.png)

1.  执行`Blurring_Sharpening.py`文件后获得的模糊图像如下所示：

![](Images/a5d0dda6-40b2-4cd4-a1a5-0caf0c2f3982.png)

1.  执行`Blurring_Sharpening.py`文件后获得的锐化图像如下所示：

![](Images/f4d411d0-8984-4418-b31f-0e9c98cc2bb2.png)

# 在图像中检测边缘

边缘检测用于检测图像中的边界。它提供有关形状和区域属性的详细信息。这包括周长、主轴大小和次轴大小。

# 如何做...

1.  导入必要的包：

```py
import sys 
import cv2 
import numpy as np 
```

1.  读取输入图像：

```py
in_file = sys.argv[1] 
image = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE) 
```

1.  实现Sobel边缘检测方案：

```py
horizontal_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) 
vertical_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) 
laplacian_img = cv2.Laplacian(image, cv2.CV_64F) 
canny_img = cv2.Canny(image, 30, 200) 
```

1.  显示输入图像及其对应的输出：

```py
cv2.imshow('Original', image) 
cv2.imshow('horizontal Sobel', horizontal_sobel) 
cv2.imshow('vertical Sobel', vertical_sobel) 
cv2.imshow('Laplacian image', laplacian_img) 
cv2.imshow('Canny image', canny_img) 
```

1.  等待操作员的指示：

```py
cv2.waitKey() 
```

1.  显示输入图像和相应的结果：

```py
cv2.imshow('Original', image) 
cv2.imshow('horizontal Sobel', horizontal_sobel) 
cv2.imshow('vertical Sobel', vertical_sobel) 
cv2.imshow('Laplacian image', laplacian_img) 
cv2.imshow('Canny image', canny_img) 
```

1.  等待操作员的指示：

```py
cv2.waitKey()
```

1.  用于执行`Detecting_edges.py` Python程序文件的命令，以及输入图像(`baby.jpg`)如下所示：

![](Images/ef7c4c3c-5ff6-4967-8cb2-21225f4a86a0.png)

1.  执行`Detecting_edges.py`文件后获得的输入图像和水平Sobel滤波器输出如下所示：

![](Images/234d88f2-d818-428d-a185-1e4ea0404526.png)

1.  执行`Detecting_edges.py`文件后获得的垂直Sobel滤波器输出和拉普拉斯图像输出如下所示：

![](Images/d10d6ba5-3946-4083-91c1-0cef82083fbe.png)

以下是拉普拉斯图像输出：

![](Images/243aaabd-5c2b-4923-91ef-49916fb9f4a3.png)

1.  执行`Detecting_edges.py`文件后获得的`Canny`边缘检测输出如下所示：

![](Images/8f4d5750-5f51-4c9d-9130-8c64ffd698c6.png)

# 它是如何工作的...

读者可以参考以下文档，了解边缘检测是什么，以及它对测试图片的影响：

[http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.301.927](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.301.927)

# 另请参见

请参阅以下文档：

+   [https://www.tutorialspoint.com/dip/concept_of_edge_detection.htm](https://www.tutorialspoint.com/dip/concept_of_edge_detection.htm)

# 直方图均衡化

直方图均衡化用于增强图像的可见性和对比度。它通过改变图像的强度来执行。这些程序在这里有清晰的描述。

# 如何做...

1.  导入必要的包：

```py
import sys 
import cv2 
import numpy as np 
```

1.  加载输入图像：

```py
in_file = sys.argv[1] 
image = cv2.imread(in_file) 
```

1.  将RGB图像转换为灰度图像：

```py
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Input grayscale image', image_gray) 
```

1.  调整灰度图像的直方图：

```py
image_gray_histoeq = cv2.equalizeHist(image_gray) 
cv2.imshow('Histogram equalized - grayscale image', image_gray_histoeq) 
```

1.  调整RGB图像的直方图：

```py
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) 
image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0]) 
image_histoeq = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR) 
```

1.  显示输出图像：

```py
cv2.imshow('Input image', image) 
cv2.imshow('Histogram equalized - color image', image_histoeq) 
cv2.waitKey()
```

1.  用于执行`histogram.py` Python程序文件的命令，以及输入图像(`finger.jpg`)如下所示：

![](Images/ad53e9d6-fe47-4d0f-a4aa-5f2876f35706.png)

1.  用于执行`histogram.py`文件的输入图像如下所示：

![](Images/7470d11b-e229-4e87-b648-b0f7b4eb6bc9.png)

1.  执行`histogram.py`文件后获得的直方图均衡化灰度图像如下所示：

![](Images/9fa26fce-9eb1-4a54-8163-8bb22ad11a42.png)

1.  执行`histogram.py`文件后获得的直方图均衡化彩色图像如下所示：

![](Images/21bbb467-1918-46bc-ae7c-c2d912a83004.png)

# 在图像中检测角点

角点是图像中用于提取推断图像内容的特殊特征的边界。角点检测经常用于图像配准、视频跟踪、图像拼接、运动检测、3D建模、全景拼接和物体识别。

# 如何做...

1.  导入必要的包：

```py
import sys 
import cv2 
import numpy as np 
```

1.  加载输入图像：

```py
in_file = sys.argv[1] 
image = cv2.imread(in_file) 
cv2.imshow('Input image', image) 
image_gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
image_gray2 = np.float32(image_gray1) 
```

1.  实现Harris角点检测方案：

```py
image_harris1 = cv2.cornerHarris(image_gray2, 7, 5, 0.04) 
```

1.  膨胀输入图像并构造角点：

```py
image_harris2 = cv2.dilate(image_harris1, None) 
```

1.  实现图像阈值处理：

```py
image[image_harris2 > 0.01 * image_harris2.max()] = [0, 0, 0] 
```

1.  显示输入图像：

```py
cv2.imshow('Harris Corners', image) 
```

1.  等待操作员的指示：

```py
cv2.waitKey() 
```

1.  用于执行`Detecting_corner.py` Python程序文件的命令，以及输入图像(`box.jpg`)如下所示：

![](Images/a1cf1797-e1a2-4ff5-917f-b2410b5478db.png)

1.  用于执行`Detecting_corner.py`文件的输入图像如下所示：

![](Images/8e3a22bb-a6f1-4251-baa8-2be67137ef99.png)

1.  执行`Detecting_corner.py`文件后获得的Harris角点如下所示：

![](Images/3934c015-8fa1-4ba6-a941-7c6046008dda.png)

要了解它如何作用于输入图片，请参考以下内容：

+   图像角点检测涉及在给定图片中找到边缘/角点。它可以用于从灰度和RGB图片中提取重要的形状特征。参考这篇关于边缘和角点检测的调查论文：

[https://pdfs.semanticscholar.org/24dd/6c2c08f5601e140aad5b9170e0c7485f6648.pdf](https://pdfs.semanticscholar.org/24dd/6c2c08f5601e140aad5b9170e0c7485f6648.pdf)。
