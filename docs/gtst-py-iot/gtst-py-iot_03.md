# 第三章：使用 Python 进行自动化和提高生产力

在本章中，我们将涵盖以下主题：

+   使用 Tkinter 创建图形用户界面

+   创建一个图形启动菜单应用程序

+   在应用程序中显示照片信息

+   自动整理您的照片

# 介绍

到目前为止，我们只专注于命令行应用程序；然而，树莓派不仅仅是命令行。通过使用**图形用户界面**（**GUI**），通常更容易从用户那里获取输入并以更简单的方式提供反馈。毕竟，我们一直在不断处理多个输入和输出，所以为什么在不必要的情况下限制自己只使用命令行的程序格式呢？

幸运的是，Python 可以支持这一点。与其他编程语言（如 Visual Basic 和 C/C++/C#）类似，这可以通过使用提供标准控件的预构建对象来实现。我们将使用一个名为**Tkinter**的模块，它提供了一系列良好的控件（也称为**小部件**）和工具，用于创建图形应用程序。

首先，我们将以`encryptdecrypt.py`为例，演示可以编写和在各种方式中重复使用的有用模块。这是良好编码实践的一个例子。我们应该致力于编写可以进行彻底测试，然后在许多地方重复使用的代码。

接下来，我们将通过创建一个小型图形启动菜单应用程序来扩展我们之前的示例，以运行我们喜爱的应用程序。

然后，我们将探索在我们的应用程序中使用**类**来显示，然后

整理照片。

# 使用 Tkinter 创建图形用户界面

我们将创建一个基本的 GUI，允许用户输入信息，然后程序可以用来加密和解密它。

# 准备工作

您必须确保该文件放置在相同的目录中。

由于我们使用了 Tkinter（Python 的许多可用附加组件之一），我们需要确保它已安装。它应该默认安装在标准的 Raspbian 镜像上。我们可以通过从 Python 提示符导入它来确认它已安装，如下所示：

`Python3`

`>>> import tkinter`

如果未安装，将引发`ImportError`异常，在这种情况下，您可以使用以下命令进行安装（使用*Ctrl* + *Z*退出 Python 提示符）：

`sudo apt-get install python3-tk`

如果模块加载了，您可以使用以下命令来阅读有关模块的更多信息（完成阅读后使用*Q*退出）：

`>>>help(tkinter)`

您还可以使用以下命令获取有关模块内所有类、函数和方法的信息：

`>>>help(tkinter.Button)`

以下`dir`命令将列出在`module`范围内的任何有效命令或变量：

`>>>dir(tkinter.Button)`

您将看到我们自己的模块将包含由三个引号标记的函数的信息；如果我们使用`help`命令，这将显示出来。

命令行将无法显示本章中创建的图形显示，因此您将需要启动树莓派桌面（使用`startx`命令），或者如果您是远程使用它。

确保您已启用**X11 转发**并且运行着**X 服务器**（参见第一章，*使用树莓派 3 计算机入门*）。

# 如何做...

我们将使用`tkinter`模块为`encryptdecrypt.py`脚本生成 GUI。

为了生成 GUI，我们将创建以下`tkencryptdecrypt.py`脚本：

```py
#!/usr/bin/python3 
#tkencryptdecrypt.py 
import encryptdecrypt as ENC 
import tkinter as TK 

def encryptButton(): 
    encryptvalue.set(ENC.encryptText(encryptvalue.get(), 
                                     keyvalue.get())) 

def decryptButton(): 
    encryptvalue.set(ENC.encryptText(encryptvalue.get(), 
                                     -keyvalue.get())) 
#Define Tkinter application 
root=TK.Tk() 
root.title("Encrypt/Decrypt GUI") 
#Set control & test value 
encryptvalue = TK.StringVar() 
encryptvalue.set("My Message")  
keyvalue = TK.IntVar() 
keyvalue.set(20) 
prompt="Enter message to encrypt:" 
key="Key:" 

label1=TK.Label(root,text=prompt,width=len(prompt),bg='green') 
textEnter=TK.Entry(root,textvariable=encryptvalue, 
                   width=len(prompt)) 
encryptButton=TK.Button(root,text="Encrypt",command=encryptButton) 
decryptButton=TK.Button(root,text="Decrypt",command=decryptButton) 
label2=TK.Label(root,text=key,width=len(key)) 
keyEnter=TK.Entry(root,textvariable=keyvalue,width=8) 
#Set layout 
label1.grid(row=0,columnspan=2,sticky=TK.E+TK.W) 
textEnter.grid(row=1,columnspan=2,sticky=TK.E+TK.W) 
encryptButton.grid(row=2,column=0,sticky=TK.E) 
decryptButton.grid(row=2,column=1,sticky=TK.W) 
label2.grid(row=3,column=0,sticky=TK.E) 
keyEnter.grid(row=3,column=1,sticky=TK.W) 

TK.mainloop() 
#End 
```

使用以下命令运行脚本：

```py
python3 tkencryptdecrypt
```

# 它是如何工作的...

我们首先导入两个模块；第一个是我们自己的`encryptdecrypt`模块，第二个是`tkinter`模块。为了更容易看到哪些项目来自哪里，我们使用`ENC`/`TK`。如果您想避免额外的引用，您可以使用`from <module_name> import *`直接引用模块项目。

当我们点击加密和解密按钮时，将调用`encryptButton()`和`decryptButton()`函数；它们将在以下部分中解释。

使用`Tk()`命令创建主 Tkinter 窗口，该命令返回所有小部件/控件可以放置的主窗口。

我们将定义六个控件如下：

+   `Label`：这显示了加密消息的提示输入信息：

+   `Entry`：这提供了一个文本框来接收用户要加密的消息

+   `Button`：这是一个加密按钮，用于触发要加密的消息

+   `Button`：这是一个解密按钮，用于反转加密

+   `Label`：这显示了密钥：字段以提示用户输入加密密钥值

+   `Entry`：这提供了第二个文本框来接收加密密钥的值

这些控件将产生一个类似于以下截图所示的 GUI：

![](img/0c31ff41-4acc-4c81-aa65-3be5d6f61bac.png)加密/解密消息的 GUI

让我们来看一下第一个`label1`的定义：

```py
label1=TK.Label(root,text=prompt,width=len(prompt),bg='green') 
```

所有控件必须链接到应用程序窗口；因此，我们必须指定我们的 Tkinter 窗口`root`。标签使用的文本由`text`设置；在这种情况下，我们将其设置为一个名为`prompt`的字符串，该字符串已经在之前定义了我们需要的文本。我们还设置`width`以匹配消息的字符数（虽然不是必需的，但如果我们稍后向标签添加更多文本，它会提供更整洁的结果），最后，我们使用`bg='green'`设置背景颜色。

接下来，我们为我们的消息定义文本`Entry`框：

```py
textEnter=TK.Entry(root,textvariable=encryptvalue, 
                   width=len(prompt)) 
```

我们将定义`textvariable`——将一个变量链接到框的内容的一种有用的方式，这是一个特殊的字符串变量。我们可以直接使用`textEnter.get()`访问`text`，但我们将使用一个`Tkinter StringVar()`对象来间接访问它。如果需要，这将允许我们将正在处理的数据与处理 GUI 布局的代码分开。`enycrptvalue`变量在使用`.set()`命令时会自动更新它所链接到的`Entry`小部件（并且`.get()`命令会从`Entry`小部件获取最新的值）。

接下来，我们有两个`Button`小部件，加密和解密，如下所示：

```py
encryptButton=TK.Button(root,text="Encrypt",command=encryptButton) 
decryptButton=TK.Button(root,text="Decrypt",command=decryptButton) 
```

在这种情况下，我们可以设置一个函数，当点击`Button`小部件时调用该函数，方法是设置`command`属性。我们可以定义两个函数，当每个按钮被点击时将被调用。在以下代码片段中，我们有`encryptButton()`函数，它将设置控制第一个`Entry`框内容的`encryptvalue StringVar`。这个字符串被设置为我们通过调用`ENC.encryptText()`得到的结果，我们要加密的消息（`encryptvalue`的当前值）和`keyvalue`变量。`decrypt()`函数完全相同，只是我们将`keyvalue`变量设置为负数以解密消息：

```py
def encryptButton(): 
    encryptvalue.set(ENC.encryptText(encryptvalue.get(), 
                                     keyvalue.get())) 
```

然后我们以类似的方式设置最终的`Label`和`Entry`小部件。请注意，如果需要，`textvariable`也可以是整数（数值），但没有内置检查来确保只能输入数字。当使用`.get()`命令时，会遇到`ValueError`异常。

在我们定义了 Tkinter 窗口中要使用的所有小部件之后，我们必须设置布局。在 Tkinter 中有三种定义布局的方法：*place*、*pack*和*grid*。

place 布局允许我们使用精确的像素位置指定位置和大小。pack 布局按照它们被添加的顺序将项目放置在窗口中。grid 布局允许我们以特定的布局放置项目。建议尽量避免使用 place 布局，因为对一个项目进行任何小的更改都可能对窗口中所有其他项目的位置和大小产生连锁效应；其他布局通过确定它们相对于窗口中其他项目的位置来解决这个问题。

我们将按照以下截图中的布局放置这些项目：

![](img/9d8b128f-ce62-43af-bee2-b9fd14f9bfaa.png)加密/解密 GUI 的网格布局

使用以下代码设置 GUI 中前两个项目的位置：

```py
label1.grid(row=0,columnspan=2,sticky= TK.E+TK.W) 
textEnter.grid(row=1,columnspan=2,sticky= TK.E+TK.W) 
```

我们可以指定第一个`Label`和`Entry`框将跨越两列（`columnspan=2`），并且我们可以设置粘性值以确保它们跨越整个宽度。这是通过设置`TK.E`表示东边和`TK.W`表示西边来实现的。如果需要在垂直方向上做同样的操作，我们会使用`TK.N`表示北边和`TK.S`表示南边。如果未指定`column`值，`grid`函数会默认为`column=0`。其他项目也是类似定义的。

最后一步是调用`TK.mainloop()`，这允许 Tkinter 运行；这允许监视按钮点击并调用与它们链接的函数。

# 创建图形应用程序-开始菜单

本示例显示了如何定义我们自己的 Tkinter 对象的变体，以生成自定义控件并动态构建菜单。我们还将简要介绍使用线程来允许其他任务继续运行，同时执行特定任务。

# 准备工作

要查看 GUI 显示，您需要一个显示树莓派桌面的显示器，或者您需要连接到另一台运行 X 服务器的计算机。

# 如何做...

1.  要创建图形开始菜单应用程序，请创建以下`graphicmenu.py`脚本：

```py
#!/usr/bin/python3 
# graphicmenu.py 
import tkinter as tk 
from subprocess import call 
import threading 

#Define applications ["Display name","command"] 
leafpad = ["Leafpad","leafpad"] 
scratch = ["Scratch","scratch"] 
pistore = ["Pi Store","pistore"] 
app_list = [leafpad,scratch,pistore] 
APP_NAME = 0 
APP_CMD  = 1 

class runApplictionThread(threading.Thread): 
    def __init__(self,app_cmd): 
        threading.Thread.__init__(self) 
        self.cmd = app_cmd 
    def run(self): 
        #Run the command, if valid 
        try: 
            call(self.cmd) 
        except: 
            print ("Unable to run: %s" % self.cmd) 

class appButtons: 
    def __init__(self,gui,app_index): 
        #Add the buttons to window 
        btn = tk.Button(gui, text=app_list[app_index][APP_NAME], 
                        width=30, command=self.startApp) 
        btn.pack() 
        self.app_cmd=app_list[app_index][APP_CMD] 
    def startApp(self): 
        print ("APP_CMD: %s" % self.app_cmd) 
        runApplictionThread(self.app_cmd).start()        

root = tk.Tk() 
root.title("App Menu") 
prompt = '      Select an application      ' 
label1 = tk.Label(root, text=prompt, width=len(prompt), bg='green') 
label1.pack() 
#Create menu buttons from app_list 
for index, app in enumerate(app_list): 
    appButtons(root,index) 
#Run the tk window 
root.mainloop() 
#End
```

1.  上面的代码产生了以下应用程序：

![](img/3782c238-32f5-4d5f-bcda-66536357deb0.png)应用程序菜单 GUI

# 它是如何工作的...

我们创建 Tkinter 窗口与之前一样；但是，我们不是单独定义所有项目，而是为应用程序按钮创建一个特殊的类。

我们创建的类充当了`appButtons`项目要包含的蓝图或规范。每个项目将包括一个`app_cmd`的字符串值，一个名为`startApp()`的函数和一个`__init__()`函数。`__init__()`函数是一个特殊函数（称为**构造函数**），当我们创建一个`appButtons`项目时会调用它；它将允许我们创建任何所需的设置。

在这种情况下，`__init__()`函数允许我们创建一个新的 Tkinter 按钮，其中文本设置为`app_list`中的一个项目，当点击按钮时调用`startApp()`函数。使用`self`关键字是为了调用属于该项目的命令；这意味着每个按钮将调用一个具有访问该项目的本地数据的本地定义函数。

我们将`self.app_cmd`的值设置为`app_list`中的命令，并通过`startApp()`函数准备好使用。现在我们创建`startApp()`函数。如果我们直接在这里运行应用程序命令，Tkinter 窗口将会冻结，直到我们打开的应用程序再次关闭。为了避免这种情况，我们可以使用 Python 线程模块，它允许我们同时执行多个操作。

`runApplicationThread()`类是使用`threading.Thread`类作为模板创建的——这个类继承了`threading.Thread`类的所有特性。和之前的类一样，我们也为这个类提供了`__init__()`函数。我们首先调用继承类的`__init__()`函数以确保它被正确设置，然后我们将`app_cmd`的值存储在`self.cmd`中。创建并初始化`runApplicationThread()`函数后，调用`start()`函数。这个函数是`threading.Thread`的一部分，我们的类可以使用它。当调用`start()`函数时，它将创建一个单独的应用程序线程（也就是说，模拟同时运行两个任务），允许 Tkinter 在执行类中的`run()`函数时继续监视按钮点击。

因此，我们可以将代码放在`run()`函数中来运行所需的应用程序（使用`call(self.cmd)`）。

# 还有更多...

使 Python 特别强大的一个方面是它支持**面向对象设计**（**OOD**）中使用的编程技术。这是现代编程语言常用的一种技术，用来帮助将我们希望程序执行的任务转化为代码中有意义的构造和结构。OOD 的原则在于，我们认为大多数问题都由几个对象（GUI 窗口、按钮等）组成，它们相互交互以产生期望的结果。

在前一节中，我们发现可以使用类来创建可以多次重复使用的唯一对象。我们创建了一个`appButton`类，它生成了一个具有该类所有功能的对象，包括其自己的`app_cmd`版本，该版本将被`startApp()`函数使用。`appButton`类型的另一个对象将有其自己不相关的`[app_cmd]`数据，其`startApp()`函数将使用它。

你可以看到，类对于将一组相关的变量和函数集中在一个对象中非常有用，而且类将在一个地方保存它自己的数据。拥有同一类型（类）的多个对象，每个对象内部都有自己的函数和数据，会导致更好的程序结构。传统的方法是将所有信息保存在一个地方，然后来回发送每个项目以供各种函数处理；然而，在大型系统中，这可能变得繁琐。

下图显示了相关函数和数据的组织结构：

![](img/d265bf45-fa70-4e20-be88-fa9b3234d317.png)数据和函数

到目前为止，我们已经使用 Python 模块将程序的不同部分分开。

文件；这使我们能够在概念上将程序的不同部分分开（界面、编码器/解码器或类库，比如 Tkinter）。模块可以提供控制特定硬件的代码，定义互联网接口，或提供常用功能的类库；然而，它最重要的功能是控制接口（在导入项目时可用的函数、变量和类的集合）。一个良好实现的模块应该有一个清晰的接口，其重点是围绕它的使用方式，而不是它的实现方式。这使你能够创建多个可以轻松交换和更改的模块，因为它们共享相同的接口。在我们之前的例子中，想象一下，通过支持`encryptText(input_text,key)`，要将`encryptdecrypt`模块更改为另一个模块是多么容易。复杂的功能可以分解成更小、可管理的块，可以在多个应用程序中重复使用。

Python 一直在使用类和模块。每次你导入一个库，比如`sys`或 Tkinter，或者使用`value.str()`转换一个值，或者使用`for...in`遍历一个列表，你都可以在不用担心细节的情况下使用它们。你不必在你写的每一行代码中都使用类或模块，但它们是你程序员工具箱中有用的工具，适合你正在做的事情时使用。

通过在本书的示例中使用类和模块，我们将了解它们如何使我们能够生成结构良好、易于测试和维护的代码。

# 在应用程序中显示照片信息

在这个例子中，我们将创建一个实用类来处理照片，其他应用程序（作为模块）可以使用它来访问照片元数据并轻松显示预览图像。

# 准备就绪

以下脚本使用了**Python Image Library**（**PIL**）；Python 3 的兼容版本是**Pillow**。

Pillow 没有包含在 Raspbian 仓库中（由`apt-get`使用）；因此，我们需要使用名为**PIP**的**Python 包管理器**来安装 Pillow。

要为 Python 3 安装包，我们将使用 Python 3 版本的 PIP（这需要 50MB 的可用空间）。

以下命令可用于安装 PIP：

```py
sudo apt-get update
sudo apt-get install python3-pip 
```

在使用 PIP 之前，请确保已安装`libjpeg-dev`以允许 Pillow 处理 JPEG 文件。您可以使用以下命令执行此操作：

```py
sudo apt-get install libjpeg-dev

```

现在您可以使用以下 PIP 命令安装 Pillow：

```py
sudo pip-3.2 install pillow  
```

PIP 还可以通过使用`uninstall`而不是`install`来轻松卸载软件包。

最后，您可以通过运行`python3`来确认它已成功安装：

```py
>>>import PIL
>>>help(PIL)  
```

您不应该收到任何错误，并且应该看到有关 PIL 及其用途的大量信息（按*Q*键完成）。按照以下方式检查安装的版本：

```py
>>PIL.PILLOW_VERSION
```

您应该看到`2.7.0`（或类似）。

通过使用以下命令安装 pip-2.x，PIP 也可以与 Python 2 一起使用：

`   sudo apt-get install python-pip`

使用`sudo pip install`安装的任何软件包都将仅为 Python 2 安装。

# 如何做...

要在应用程序中显示照片信息，请创建以下`photohandler.py`脚本：

```py
##!/usr/bin/python3 
#photohandler.py 
from PIL import Image 
from PIL import ExifTags 
import datetime 
import os 

#set module values 
previewsize=240,240 
defaultimagepreview="./preview.ppm" 
filedate_to_use="Exif DateTime" 
#Define expected inputs 
ARG_IMAGEFILE=1 
ARG_LENGTH=2 

class Photo: 
    def __init__(self,filename): 
        """Class constructor""" 
        self.filename=filename 
        self.filevalid=False 
        self.exifvalid=False 
        img=self.initImage() 
        if self.filevalid==True: 
            self.initExif(img) 
            self.initDates() 

    def initImage(self): 
        """opens the image and confirms if valid, returns Image""" 
        try: 
            img=Image.open(self.filename) 
            self.filevalid=True 
        except IOError: 
            print ("Target image not found/valid %s" % 
                   (self.filename)) 
            img=None 
            self.filevalid=False 
        return img 

    def initExif(self,image): 
        """gets any Exif data from the photo""" 
        try: 
            self.exif_info={ 
                ExifTags.TAGS[x]:y 
                for x,y in image._getexif().items() 
                if x in ExifTags.TAGS 
            } 
            self.exifvalid=True 
        except AttributeError: 
            print ("Image has no Exif Tags") 
            self.exifvalid=False 

    def initDates(self): 
        """determines the date the photo was taken""" 
        #Gather all the times available into YYYY-MM-DD format 
        self.filedates={} 
        if self.exifvalid: 
            #Get the date info from Exif info 
            exif_ids=["DateTime","DateTimeOriginal", 
                      "DateTimeDigitized"] 
            for id in exif_ids: 
                dateraw=self.exif_info[id] 
                self.filedates["Exif "+id]= 
                                dateraw[:10].replace(":","-") 
        modtimeraw = os.path.getmtime(self.filename) 
        self.filedates["File ModTime"]="%s" % 
            datetime.datetime.fromtimestamp(modtimeraw).date() 
        createtimeraw = os.path.getctime(self.filename) 
        self.filedates["File CreateTime"]="%s" % 
            datetime.datetime.fromtimestamp(createtimeraw).date() 

    def getDate(self): 
        """returns the date the image was taken""" 
        try: 
            date = self.filedates[filedate_to_use] 
        except KeyError: 
            print ("Exif Date not found") 
            date = self.filedates["File ModTime"] 
        return date 

    def previewPhoto(self): 
        """creates a thumbnail image suitable for tk to display""" 
        imageview=self.initImage() 
        imageview=imageview.convert('RGB') 
        imageview.thumbnail(previewsize,Image.ANTIALIAS) 
        imageview.save(defaultimagepreview,format='ppm') 
        return defaultimagepreview         
```

前面的代码定义了我们的`Photo`类；在*还有更多...*部分和下一个示例中运行它之前，它对我们没有用处。

# 它是如何工作的...

我们定义了一个名为`Photo`的通用类；它包含有关自身的详细信息，并提供

用于访问**可交换图像文件格式**（**EXIF**）信息并生成的函数

一个预览图像。

在`__init__()`函数中，我们为我们的类变量设置值，并调用`self.initImage()`，它将使用 PIL 中的`Image()`函数打开图像。然后我们调用`self.initExif()`和`self.initDates()`，并设置一个标志来指示文件是否有效。如果无效，`Image()`函数将引发`IOError`异常。

`initExif()`函数使用 PIL 从`img`对象中读取 EXIF 数据，如下面的代码片段所示：

```py
self.exif_info={ 
                ExifTags.TAGS[id]:y 
                for id,y in image._getexif().items() 
                if id in ExifTags.TAGS 
               } 
```

前面的代码是一系列复合语句，导致`self.exif_info`被填充为标签名称及其相关值的字典。

`ExifTag.TAGS`是一个包含可能的标签名称及其 ID 的列表的字典，如下面的代码片段所示：

```py
ExifTag.TAGS={ 
4096: 'RelatedImageFileFormat', 
513: 'JpegIFOffset', 
514: 'JpegIFByteCount', 
40963: 'ExifImageHeight', 
...etc...}
```

`image._getexif()`函数返回一个包含图像相机设置的所有值的字典，每个值都与其相关的 ID 链接，如下面的代码片段所示：

```py
Image._getexif()={ 
256: 3264, 
257: 2448, 
37378: (281, 100), 
36867: '2016:09:28 22:38:08', 
...etc...} 
```

`for`循环将遍历图像的 EXIF 值字典中的每个项目，并检查其在`ExifTags.TAGS`字典中的出现；结果将存储在`self.exif_info`中。其代码如下：

```py
self.exif_info={ 
'YResolution': (72, 1), 
 'ResolutionUnit': 2, 
 'ExposureMode': 0,  
'Flash': 24, 
...etc...} 
```

再次，如果没有异常，我们将设置一个标志来指示 EXIF 数据是有效的，或者如果没有 EXIF 数据，我们将引发`AttributeError`异常。

`initDates()`函数允许我们收集所有可能的文件日期和来自 EXIF 数据的日期，以便我们可以选择其中一个作为我们希望用于文件的日期。例如，它允许我们将所有图像重命名为标准日期格式的文件名。我们创建一个`self.filedates`字典，其中包含从 EXIF 信息中提取的三个日期。然后添加文件系统日期（创建和修改），以防没有 EXIF 数据可用。`os`模块允许我们使用`os.path.getctime()`和`os.path.getmtime()`来获取文件创建的时期值。它也可以是文件移动时的日期和时间-最后写入的文件修改时间（例如，通常指图片拍摄的日期）。时期值是自 1970 年 1 月 1 日以来的秒数，但我们可以使用`datetime.datetime.fromtimestamp()`将其转换为年、月、日、小时和秒。添加`date()`只是将其限制为年、月和日。

现在，如果`Photo`类被另一个模块使用，并且我们希望知道拍摄的图像的日期，我们可以查看`self.dates`字典并选择合适的日期。但是，这将要求程序员知道`self.dates`值的排列方式，如果以后更改了它们的存储方式，将会破坏他们的程序。因此，建议我们通过访问函数访问类中的数据，以便实现独立于接口（这个过程称为**封装**）。我们提供一个在调用时返回日期的函数；程序员不需要知道它可能是五个可用日期中的一个，甚至不需要知道它们是作为时期值存储的。使用函数，我们可以确保接口保持不变，无论数据的存储或收集方式如何。

最后，我们希望`Photo`类提供的最后一个函数是`previewPhoto()`。此函数提供了一种生成小缩略图图像并将其保存为**便携式像素图格式**(**PPM**)文件的方法。正如我们将在一会儿发现的那样，Tkinter 允许我们将图像放在其`Canvas`小部件上，但不幸的是，它不直接支持 JPEG，只支持 GIF 或 PPM。因此，我们只需将要显示的图像的小副本保存为 PPM 格式，然后让 Tkinter 在需要时将其加载到`Canvas`上。

总之，我们创建的`Photo`类如下：

| **操作** | **描述** |
| --- | --- |
| `__init__(self,filename)` | 这是对象初始化程序。 |
| `initImage(self)` | 这将返回`img`，一个 PIL 类型的图像对象。 |
| `initExif(self,image)` | 如果存在，这将提取所有的 EXIF 信息。 |
| `initDates(self)` | 这将创建一个包含文件和照片信息中所有可用日期的字典。 |
| `getDate(self)` | 这将返回照片拍摄/创建的日期的字符串。 |
| `previewPhoto(self)` | 这将返回预览缩略图的文件名的字符串。 |

属性及其相应的描述如下：

| **属性** | **描述** |
| --- | --- |
| `self.filename` | 照片的文件名。 |
| `self.filevalid` | 如果文件成功打开，则设置为`True`。 |
| `self.exifvalid` | 如果照片包含 EXIF 信息，则设置为`True`。 |
| `self.exif_info` | 这包含照片的 EXIF 信息。 |
| `self.filedates` | 这包含了文件和照片信息中可用日期的字典。 |

为了测试新类，我们将创建一些测试代码来确认一切是否按我们的预期工作；请参阅以下部分。

# 还有更多...

我们之前创建了`Photo`类。现在我们可以向我们的模块中添加一些测试代码，以确保它按我们的预期运行。我们可以使用`__name__ ="__main__"`属性

与之前一样，以检测模块是否直接运行。

我们可以在`photohandler.py`脚本的末尾添加以下代码段，以生成以下测试应用程序，其外观如下：

![](img/0ced9d12-c53e-4896-9395-4d7563b3b1e4.png)照片查看演示应用程序

在`photohandler.py`的末尾添加以下代码： 

```py
#Module test code 
def dispPreview(aPhoto): 
    """Create a test GUI""" 
    import tkinter as TK 

    #Define the app window 
    app = TK.Tk() 
    app.title("Photo View Demo") 

    #Define TK objects 
    # create an empty canvas object the same size as the image 
    canvas = TK.Canvas(app, width=previewsize[0], 
                       height=previewsize[1]) 
    canvas.grid(row=0,rowspan=2) 
    # Add list box to display the photo data 
    #(including xyscroll bars) 
    photoInfo=TK.Variable() 
    lbPhotoInfo=TK.Listbox(app,listvariable=photoInfo, 
                           height=18,width=45, 
                           font=("monospace",10)) 
    yscroll=TK.Scrollbar(command=lbPhotoInfo.yview, 
                         orient=TK.VERTICAL) 
    xscroll=TK.Scrollbar(command=lbPhotoInfo.xview, 
                         orient=TK.HORIZONTAL) 
    lbPhotoInfo.configure(xscrollcommand=xscroll.set, 
                          yscrollcommand=yscroll.set) 
    lbPhotoInfo.grid(row=0,column=1,sticky=TK.N+TK.S) 
    yscroll.grid(row=0,column=2,sticky=TK.N+TK.S) 
    xscroll.grid(row=1,column=1,sticky=TK.N+TK.E+TK.W) 

    # Generate the preview image 
    preview_filename = aPhoto.previewPhoto() 
    photoImg = TK.PhotoImage(file=preview_filename) 
    # anchor image to NW corner 
    canvas.create_image(0,0, anchor=TK.NW, image=photoImg)  

    # Populate infoList with dates and exif data 
    infoList=[] 
    for key,value in aPhoto.filedates.items(): 
        infoList.append(key.ljust(25) + value) 
    if aPhoto.exifvalid: 
        for key,value in aPhoto.exif_info.items(): 
           infoList.append(key.ljust(25) + str(value)) 
    # Set listvariable with the infoList 
    photoInfo.set(tuple(infoList)) 

    app.mainloop() 

def main(): 
    """called only when run directly, allowing module testing""" 
    import sys 
    #Check the arguments 
    if len(sys.argv) == ARG_LENGTH: 
        print ("Command: %s" %(sys.argv)) 
        #Create an instance of the Photo class 
        viewPhoto = Photo(sys.argv[ARG_IMAGEFILE]) 
        #Test the module by running a GUI 
        if viewPhoto.filevalid==True: 
            dispPreview(viewPhoto) 
    else: 
        print ("Usage: photohandler.py imagefile") 

if __name__=='__main__': 
  main() 
#End 
```

之前的测试代码将运行`main()`函数，该函数获取要使用的照片的文件名，并创建一个名为`viewPhoto`的新`Photo`对象。如果`viewPhoto`成功打开，我们将调用`dispPreview()`来显示图像及其详细信息。

`dispPreview()`函数创建四个 Tkinter 小部件以显示：一个`Canvas`加载缩略图图像，一个`Listbox`小部件显示照片信息，以及两个滚动条来控制`Listbox`。首先，我们创建一个`Canvas`小部件，大小与缩略图图像(`previewsize`)相同。

接下来，我们创建`photoInfo`，它将是我们与`Listbox`小部件关联的`listvariable`参数。由于 Tkinter 没有提供`ListVar()`函数来创建合适的项目，我们使用通用类型`TK.Variable()`，然后确保在设置值之前将其转换为元组类型。添加`Listbox`小部件；我们需要确保`listvariable`参数设置为`photoInfo`，并且将字体设置为`monospace`。这将允许我们使用空格对齐我们的数据值，因为`monospace`是等宽字体，所以每个字符占用的宽度都相同。

我们通过将`Scrollbar`命令参数设置为`lbPhotoInfo.yview`和`lbPhotoInfo.xview`来定义两个滚动条，并将它们链接到`Listbox`小部件。然后，我们使用以下命令调整`Listbox`的参数：

```py
lbPhotoInfo.configure(xscrollcommand=xscroll.set, 
 yscrollcommand=yscroll.set)

```

`configure`命令允许我们在创建小部件后添加或更改小部件的参数，在这种情况下，链接两个滚动条，以便`Listbox`小部件在用户在列表中滚动时也可以控制它们。

与以前一样，我们利用网格布局来确保`Listbox`小部件旁边正确放置了两个滚动条，`Canvas`小部件位于`Listbox`小部件的左侧。

我们现在使用`Photo`对象创建`preview.ppm`缩略图文件（使用`aPhoto.previewPhoto()`函数），并创建一个`TK.PhotoImage`对象，然后可以使用以下命令将其添加到`Canvas`小部件中：

```py
canvas.create_image(0,0, anchor=TK.NW, image=photoImg)

```

最后，我们使用`Photo`类收集的日期信息和 EXIF 信息（确保它首先是有效的）来填充`Listbox`小部件。我们通过将每个项目转换为一系列使用`.ljust(25)`间隔的字符串来实现这一点——它添加左对齐到名称，并填充它使字符串宽度为 25 个字符。一旦我们有了列表，我们将其转换为元组类型并设置`listvariable`（`photoInfo`）参数。

像往常一样，我们调用`app.mainloop()`来开始监视事件以做出响应。

# 自动整理您的照片

现在我们有了一个允许我们收集照片信息的类，我们可以将这些信息应用于执行有用的任务。在这种情况下，我们将使用文件信息自动将一个充满照片的文件夹组织成基于照片拍摄日期的子文件夹的子集。

以下屏幕截图显示了脚本的输出：

![](img/d390e81f-0293-40ab-943b-25d1887f41e3.png)脚本输出以整理文件夹中的照片

# 准备工作

您需要在树莓派上的一个文件夹中放置一些照片。或者，您可以插入一个带有照片的 USB 存储设备或读卡器——它们将位于`/mnt/`中。但是，请确保您首先使用照片的副本测试脚本，以防出现任何问题。

# 如何做...

创建以下脚本`filehandler.py`以自动整理您的照片：

```py
#!/usr/bin/python3 
#filehandler.py 
import os 
import shutil 
import photohandler as PH 
from operator import itemgetter 

FOLDERSONLY=True 
DEBUG=True 
defaultpath="" 
NAME=0 
DATE=1 

class FileList: 
  def __init__(self,folder): 
    """Class constructor""" 
    self.folder=folder 
    self.listFileDates() 

  def getPhotoNamedates(self): 
    """returns the list of filenames and dates""" 
    return self.photo_namedates 

  def listFileDates(self): 
    """Generate list of filenames and dates""" 
    self.photo_namedates = list() 
    if os.path.isdir(self.folder): 
      for filename in os.listdir(self.folder): 
        if filename.lower().endswith(".jpg"): 
          aPhoto = PH.Photo(os.path.join(self.folder,filename)) 
          if aPhoto.filevalid: 
            if (DEBUG):print("NameDate: %s %s"% 
                             (filename,aPhoto.getDate())) 
            self.photo_namedates.append((filename, 
                                         aPhoto.getDate())) 
            self.photo_namedates = sorted(self.photo_namedates, 
                                    key=lambda date: date[DATE]) 

  def genFolders(self): 
    """function to generate folders""" 
    for i,namedate in enumerate(self.getPhotoNamedates()): 
      #Remove the - from the date format 
      new_folder=namedate[DATE].replace("-","") 
      newpath = os.path.join(self.folder,new_folder) 
      #If path does not exist create folder 
      if not os.path.exists(newpath): 
        if (DEBUG):print ("New Path: %s" % newpath) 
        os.makedirs(newpath) 
      if (DEBUG):print ("Found file: %s move to %s" % 
                        (namedate[NAME],newpath)) 
      src_file = os.path.join(self.folder,namedate[NAME]) 
      dst_file = os.path.join(newpath,namedate[NAME]) 
      try: 
        if (DEBUG):print ("File moved %s to %s" % 
                          (src_file, dst_file)) 
        if (FOLDERSONLY==False):shutil.move(src_file, dst_file) 
      except IOError: 
        print ("Skipped: File not found") 

def main(): 
  """called only when run directly, allowing module testing""" 
  import tkinter as TK 
  from tkinter import filedialog 
  app = TK.Tk() 
  app.withdraw() 
  dirname = TK.filedialog.askdirectory(parent=app, 
      initialdir=defaultpath, 
      title='Select your pictures folder') 
  if dirname != "": 
    ourFileList=FileList(dirname) 
    ourFileList.genFolders() 

if __name__=="__main__": 
  main() 
#End 
```

# 它是如何工作的...

我们将创建一个名为`FileList`的类；它将使用`Photo`类来管理

特定文件夹中的照片。这有两个主要步骤：首先需要找到文件夹中的所有图像，然后生成一个包含文件名和照片日期的列表。我们将使用这些信息生成新的子文件夹，并将照片移动到这些文件夹中。

当我们创建`FileList`对象时，我们将使用`listFileDates()`创建列表。然后，我们将确认提供的文件夹是有效的，并使用`os.listdir`获取目录中的所有文件的完整列表。我们将检查每个文件是否是 JPEG 文件，并获取每张照片的日期（使用`Photo`类中定义的函数）。接下来，我们将文件名和日期作为元组添加到`self.photo_namedates`列表中。

最后，我们将使用内置的 `sorted` 函数按日期顺序放置所有文件。虽然我们在这里不需要这样做，但如果我们在其他地方使用这个模块，这个函数将更容易删除重复的日期。

`sorted` 函数需要对列表进行排序，在这种情况下，我们希望按 `date values:` 进行排序。

`   sorted(self.photo_namedates,key=lambda date: date[DATE])`

我们将用 `lambda date:` 替换 `date[DATE]` 作为排序的数值。

一旦 `FileList` 对象被初始化，我们可以通过调用 `genFolders()` 来使用它。首先，我们将日期文本转换为适合我们文件夹的格式（YYYYMMDD），使我们的文件夹可以轻松按日期顺序排序。接下来，它将在当前目录内创建文件夹（如果尚不存在）。最后，它将把每个文件移动到所需的子文件夹中。

我们最终得到了准备测试的 `FileList` 类：

| **操作** | **描述** |
| --- | --- |
| `__init__(self,folder)` | 这是对象初始化程序。 |
| `getPhotoNamedates(self)` | 这将返回一个包含照片文件名和日期的列表。 |
| `listFileDates(self)` | 这将创建一个包含文件夹中照片文件名和日期的列表。 |
| `genFolders(self)` | 这将根据照片的日期创建新文件夹并将文件移动到其中。 |

属性列如下：

| **属性** | **描述** |
| --- | --- |
| `self.folder` | 我们正在处理的文件夹。 |
| `self.photo_namedates` | 这包含文件名和日期的列表。 |

`FileList` 类将所有函数和相关数据封装在一起，将所有内容放在一个逻辑位置：

![](img/eb993e01-c245-49da-a981-a4fafdcc1aa5.png)Tkinter filediaglog.askdirectory() 用于选择照片目录

为了测试这个，我们使用 Tkinter 的 `filedialog.askdirectory()` 小部件来选择照片的目标文件夹。我们使用 `app.withdrawn()` 来隐藏主 Tkinter 窗口，因为这次不需要它。我们只需要创建一个新的 `FileList` 对象，然后调用 `genFolders()` 将所有照片移动到新的位置！

在这个脚本中定义了两个额外的标志，为测试提供了额外的控制。`DEBUG` 允许我们通过将其设置为 `True` 或 `False` 来启用或禁用额外的调试消息。此外，`FOLDERSONLY` 当设置为 `True` 时，只生成文件夹而不移动文件（这对于测试新的子文件夹是否正确非常有帮助）。

运行脚本后，您可以检查所有文件夹是否已正确创建。最后，将 `FOLDERSONLY` 更改为 `True`，下次您的程序将根据日期自动移动和组织照片。建议您只在照片的副本上运行此操作，以防出现错误。
