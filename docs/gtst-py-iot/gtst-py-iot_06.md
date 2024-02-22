# 第六章：构建人脸检测器和人脸识别应用程序

本章介绍以下主题：

+   人脸识别系统简介

+   构建人脸检测器应用程序

+   构建人脸识别应用程序

+   人脸识别系统的应用

# 介绍

近年来，人脸识别已成为最热门的研究领域之一。人脸识别系统是一种具有检测和识别人脸能力的计算机程序。为了识别一个人，它考虑他们独特的面部特征。最近，它已被应用于多个安全和监控设施，以确保高风险区域、住宅区、私人和公共建筑等的安全。

# 构建人脸检测器应用程序

在本节中，我们讨论了如何从网络摄像头图像中检测人脸。需要将 USB 网络摄像头连接到树莓派 3 上，以实现实时人脸检测。

# 如何做...

1.  导入必要的包：

```py
import cv2 
import numpy as np 
```

1.  加载人脸级联文件：

```py
frontalface_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
```

1.  检查人脸级联文件是否已加载：

```py
if frontalface_cascade.empty(): 
  raiseIOError('Unable to load the face cascade classifier xml file') 
```

1.  初始化视频捕获对象：

```py
capture = cv2.VideoCapture(0) 
```

1.  定义缩放因子：

```py
scale_factor = 0.5 
```

1.  直到按下*Esc*键为止执行操作：

```py
# Loop until you hit the Esc key 
while True: 
```

1.  捕获当前帧并调整大小：

```py
  ret, frame = capture.read() 
  frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor,  
            interpolation=cv2.INTER_AREA) 
```

1.  将图像帧转换为灰度：

```py
  gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
```

1.  在灰度图像上运行人脸检测器：

```py
  face_rectangle = frontalface_cascade.detectMultiScale(gray_image, 1.3, 5)
```

1.  绘制矩形框：

```py
  for (x,y,w,h) in face_rectangle: 
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3) 
```

1.  显示输出图像：

```py
    cv2.imshow('Face Detector', frame) 
```

1.  检查是否按下了*Esc*键以终止操作：

```py
  a = cv2.waitKey(1) 
  if a == 10: 
    break 
```

1.  停止视频捕获并终止操作：

```py
capture.release() 
cv2.destroyAllWindows() 
```

人脸检测系统中获得的结果如下所示：

![](img/5cb83a55-bccc-4135-b2ec-90bc2564dc2d.png)

# 构建人脸识别应用程序

人脸识别是在人脸检测之后执行的一种技术。检测到的人脸与数据库中存储的图像进行比较。它从输入图像中提取特征并将其与数据库中存储的人脸特征进行匹配。

# 如何做...

1.  导入必要的包：

```py
import cv2 
import numpy as np   
from sklearn import preprocessing 
```

1.  加载编码和解码任务运算符：

```py
class LabelEncoding(object): 
  # Method to encode labels from words to numbers 
  def encoding_labels(self, label_wordings): 
    self.le = preprocessing.LabelEncoder() 
    self.le.fit(label_wordings) 
```

1.  为输入标签实现从单词到数字的转换：

```py
  def word_to_number(self, label_wordings): 
    return int(self.le.transform([label_wordings])[0]) 
```

1.  将输入标签从数字转换为单词：

```py
  def number_to_word(self, label_number): 
    return self.le.inverse_transform([label_number])[0] 
```

1.  从输入路径提取图像和标签：

```py
def getting_images_and_labels(path_input): 
  label_wordings = [] 
```

1.  迭代输入路径的过程并附加文件：

```py
  for roots, dirs, files in os.walk(path_input): 
    for fname in (x for x in files if x.endswith('.jpg')): 
      fpath = os.path.join(roots, fname) 
      label_wordings.append(fpath.split('/')[-2])
```

1.  初始化变量并解析输入寄存器：

```py
  images = [] 
  le = LabelEncoding() 
  le.encoding_labels(label_wordings) 
  labels = [] 
  # Parse the input directory 
  for roots, dirs, files in os.walk(path_input): 
    for fname in (x for x in files if x.endswith('.jpg')): 
      fpath = os.path.join(roots, fname) 
```

1.  读取灰度图像：

```py
      img = cv2.imread(fpath, 0)  
```

1.  提取标签：

```py
      names = fpath.split('/')[-2] 
```

1.  执行人脸检测：

```py
      face = faceCascade.detectMultiScale(img, 1.1, 2, minSize=(100,100)) 
```

1.  使用面部矩形迭代该过程：

```py
      for (x, y, w, h) in face: 
        images.append(img[y:y+h, x:x+w]) 
        labels.append(le.word_to_number(names)) 
  return images, labels, le 
if __name__=='__main__': 
  path_cascade = "haarcascade_frontalface_alt.xml" 
  train_img_path = 'faces_dataset/train' 
  path_img_test = 'faces_dataset/test' 
```

1.  加载人脸级联文件：

```py
  faceCascade = cv2.CascadeClassifier(path_cascade) 
```

1.  使用局部二值模式初始化人脸检测：

```py
  face_recognizer = cv2.createLBPHFaceRecognizer()
```

1.  从训练人脸数据集中提取人脸特征：

```py
  imgs, labels, le = getting_images_and_labels(train_img_path) 
```

1.  训练人脸检测系统：

```py
  print "nTraining..." 
  face_recognizer.train(imgs, np.array(labels)) 
```

1.  测试人脸检测系统：

```py
  print 'nPerforming prediction on test images...' 
  flag_stop = False 
  for roots, dirs, files in os.walk(path_img_test): 
    for fname in (x for x in files if x.endswith('.jpg')): 
      fpath = os.path.join(roots, fname) 
```

1.  验证人脸识别系统：

```py
      predicting_img = cv2.imread(fpath, 0) 
            # Detect faces 
      face = faceCascade.detectMultiScale(predicting_img, 1.1,  
                    2, minSize=(100,100)) 
            # Iterate through face rectangles 
      for (x, y, w, h) in face: 
        # Predict the output 
        index_predicted, config = face_recognizer.predict( 
predicting_img[y:y+h, x:x+w]) 
        # Convert to word label 
        person_predicted = le.number_to_word(index_predicted) 
        # Overlay text on the output image and display it 
        cv2.putText(predicting_img, 'Prediction: ' +  person_predicted,  
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6) 
        cv2.imshow("Recognizing face", predicting_img) 
      a = cv2.waitKey(0) 
      if a == 27: 
        flag = True 
        break 
    if flag_stop: 
      break 
```

这里显示了人脸识别的输出结果：

![](img/7526565a-5ec3-4065-a9df-d18c86f90bf5.png)

# 工作原理...

人脸识别系统广泛用于实现个人安全系统。读者可以参考文章*基于 OpenCV 的人脸检测系统*，网址为[`ieeexplore.ieee.org/document/6242980/`](http://ieeexplore.ieee.org/document/6242980/)。

另请参阅*用于实时人脸检测系统的人脸检测算法研究*，网址为[`ieeexplore.ieee.org/document/5209668`](http://ieeexplore.ieee.org/document/5209668)。

# 另请参阅

请参考以下文章：

+   [`www.ex-sight.com/technology.htm`](http://www.ex-sight.com/technology.htm)

+   [`www.eurotech.com/en/products/devices/face+recognition+systems`](https://www.eurotech.com/en/products/devices/face+recognition+systems)

+   [`arxiv.org/ftp/arxiv/papers/1403/1403.0485.pdf`](https://arxiv.org/ftp/arxiv/papers/1403/1403.0485.pdf)

# 人脸识别系统的应用

人脸识别广泛应用于安全、医疗保健和营销领域。各行业正在利用深度学习开发新型人脸识别系统，用于识别欺诈、区分人脸和照片之间的差异等。在医疗保健领域，人脸识别结合其他计算机视觉算法用于检测面部皮肤疾病。
