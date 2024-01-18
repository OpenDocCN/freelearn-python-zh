# 使用谷歌地图

在本章中，您将学习如何在Python应用程序中使用谷歌地图，并探索谷歌提供的不同优势。您将学习以下任务：

+   查找位置或地标的详细信息

+   从经度和纬度值获取完整信息

+   查找两个位置之间的距离

+   在谷歌地图上显示位置

# 介绍

谷歌地图API是一组方法和工具，可用于查找任何位置的完整信息，包括经度和纬度值。您可以使用谷歌地图API方法查找两个位置之间的距离或到达任何位置的方向；甚至可以显示谷歌地图，标记该位置，等等。

更准确地说，谷歌地图服务有一个Python“客户端”库。谷歌地图API包括方向API、距离矩阵API、地理编码API、地理位置API等多个API。要使用任何谷歌地图网络服务，您的Python脚本会向谷歌发送一个请求；为了处理该请求，您需要一个API密钥。您需要按照以下步骤获取API密钥：

1.  访问[https://console.developers.google.com](https://console.developers.google.com)

1.  使用您的谷歌账号登录控制台

1.  选择您现有的项目之一或创建一个新项目。

1.  启用您想要使用的API

1.  复制API密钥并在您的Python脚本中使用它

您需要访问谷歌API控制台，[https://console.developers.google.com](https://console.developers.google.com/apis/dashboard)，并获取API密钥，以便您的应用程序经过身份验证可以使用谷歌地图API网络服务。

API密钥在多个方面有帮助；首先，它有助于识别您的应用程序。API密钥包含在每个请求中，因此它有助于谷歌监视您的应用程序的API使用情况，了解您的应用程序是否已经消耗完每日的免费配额，并因此向您的应用程序收费。

因此，为了在您的Python应用程序中使用谷歌地图API网络服务，您只需要启用所需的API并获取一个API密钥。

# 查找位置或地标的详细信息

在这个教程中，您将被提示输入您想要了解的位置或地标的详细信息。例如，如果您输入“白金汉宫”，该教程将显示宫殿所在地的城市和邮政编码，以及其经度和纬度值。

# 如何做…

`GoogleMaps`类的search方法是这个教程的关键。用户输入的地标或位置被传递给search方法。从search方法返回的对象的`city`、`postal_code`、`lat`和`lng`属性用于分别显示位置的城市、邮政编码、纬度和经度。让我们通过以下逐步过程来看看如何完成这个操作：

1.  基于无按钮对话框模板创建一个应用程序。

1.  通过将六个标签、一个行编辑和一个推送按钮小部件拖放到表单上，向表单添加六个标签、一个行编辑和一个推送按钮小部件。

1.  将第一个标签小部件的文本属性设置为“查找城市、邮政编码、经度和纬度”，将第二个标签小部件的文本属性设置为“输入位置”。

1.  删除第三、第四、第五和第六个标签小部件的文本属性，因为它们的文本属性将通过代码设置；也就是说，输入位置的城市、邮政编码、经度和纬度将通过代码获取并通过这四个标签小部件显示。

1.  将推送按钮小部件的文本属性设置为“搜索”。

1.  将行编辑小部件的objectName属性设置为`lineEditLocation`。

1.  将推送按钮小部件的objectName属性设置为`pushButtonSearch`。

1.  将其余四个标签小部件的objectName属性设置为`labelCity`、`labelPostalCode`、`labelLongitude`和`labelLatitude`。

1.  将应用程序保存为`demoGoogleMap1.ui`。表单现在将显示如下屏幕截图所示：

![](assets/2178b7ed-d1b5-4dac-8323-bdd54599223a.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，它是一个XML文件。通过应用`pyuic5`实用程序将XML文件转换为Python代码。您可以在本书的源代码包中找到生成的Python代码`demoGoogleMap1.py`。

1.  将`demoGoogleMap1.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callGoogleMap1.pyw`的Python文件，并将`demoGoogleMap1.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from geolocation.main import GoogleMaps
from demoGoogleMap1 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonSearch.clicked.connect(self.
        displayDetails)
        self.show()
    def displayDetails(self):
        address = str(self.ui.lineEditLocation.text())
        google_maps = GoogleMaps(api_key=
        'xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        location = google_maps.search(location=address)
        my_location = location.first()
        self.ui.labelCity.setText("City: 
        "+str(my_location.city))
        self.ui.labelPostalCode.setText("Postal Code: " 
        +str(my_location.postal_code))
        self.ui.labelLongitude.setText("Longitude: 
        "+str(my_location.lng))
        self.ui.labelLatitude.setText("Latitude: 
        "+str(my_location.lat))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的...

您可以在脚本中看到，具有objectName属性`pushButtonSearch`的按钮的单击事件连接到`displayDetails`方法。这意味着每当单击按钮时，将调用`displayDetails`方法。在`displayDetails`方法中，您访问用户在行编辑小部件中输入的位置，并将该位置分配给地址变量。通过传递在Google注册时获得的API密钥来定义Google Maps实例。在Google Maps实例上调用`search`方法，传递用户在此方法中输入的位置。`search`方法的结果分配给`my_location`结构。`my_location`结构的city成员包含用户输入的城市。类似地，`my_location`结构的`postal_code`、`lng`和`lat`成员分别包含用户输入位置的邮政编码、经度和纬度信息。城市、邮政编码、经度和纬度信息通过最后四个标签小部件显示。

运行应用程序时，将提示您输入要查找信息的位置。假设您在位置中输入`泰姬陵`，然后单击搜索按钮。泰姬陵地标的城市、邮政编码、经度和纬度信息将显示在屏幕上，如下面的屏幕截图所示：

![](assets/82952079-989a-4c5e-9c83-87569e794297.png)

# 从纬度和经度值获取完整信息

在本教程中，您将学习如何查找已知经度和纬度值的位置的完整详细信息。将点位置（即纬度和经度值）转换为可读地址（地名、城市、国家名称等）的过程称为**反向地理编码**。

应用程序将提示您输入经度和纬度值，然后显示匹配的位置名称、城市、国家和邮政编码。

# 如何做... 

让我们根据以下步骤创建一个基于无按钮对话框模板的应用程序：

1.  通过将七个`QLabel`、两个`QLineEdit`和一个`QPushButton`小部件拖放到表单上，向表单添加七个标签、两个行编辑和一个按钮小部件。

1.  将第一个标签小部件的文本属性设置为`查找位置、城市、国家和邮政编码`，将第二个标签小部件的文本属性设置为`输入经度`，将第三个标签小部件的文本属性设置为`输入纬度`。

1.  删除第四、第五、第六和第七个标签小部件的文本属性，因为它们的文本属性将通过代码设置；也就是说，用户输入经度和纬度的位置的位置、城市、国家和邮政编码将通过代码访问，并通过这四个标签小部件显示。

1.  将Push Button小部件的文本属性设置为`搜索`。

1.  将两个行编辑小部件的objectName属性设置为`lineEditLongitude`和`lineEditLatitude`。

1.  将Push Button小部件的objectName属性设置为`pushButtonSearch`。

1.  将其他四个标签小部件的objectName属性设置为`labelLocation`、`labelCity`、`labelCountry`和`labelPostalCode`。

1.  将应用程序保存为`demoGoogleMap2.ui`。表单现在将显示如下截图所示：

![](assets/7427070b-7dd9-400e-9ac7-7ca614a6ee62.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。使用`pyuic5`实用程序将XML文件转换为Python代码。在本书的源代码包中可以看到生成的Python脚本`demoGoogleMap2.py`。

1.  将`demoGoogleMap2.py`脚本视为头文件，并将其导入到将调用其用户界面设计的文件中。

1.  创建另一个名为`callGoogleMap2.pyw`的Python文件，并将`demoGoogleMap2.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from geolocation.main import GoogleMaps
from demoGoogleMap2 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonSearch.clicked.connect(self.
        displayLocation)
        self.show()
    def displayLocation(self):
        lng = float(self.ui.lineEditLongitude.text())
        lat = float(self.ui.lineEditLatitude.text())
        google_maps = GoogleMaps(api_key=
        'AIzaSyDzCMD-JTg-IbJZZ9fKGE1lipbBiFRiGHA')
        my_location = google_maps.search(lat=lat, lng=lng).
        first()
        self.ui.labelLocation.setText("Location:   
        "+str(my_location))
        self.ui.labelCity.setText("City: 
        "+str(my_location.city))
        self.ui.labelCountry.setText("Country: 
        "+str(my_location.country))
        self.ui.labelPostalCode.setText("Postal Code: 
        "+str(my_location.postal_code))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 操作原理如下：

在脚本中，您可以看到具有objectName属性`pushButtonSearch`的推送按钮的click()事件连接到`displayLocation`方法。这意味着每当单击推送按钮时，将调用`displayLocation`方法。在`displayLocation`方法中，您通过两个Line Edit小部件访问用户输入的经度和纬度，并分别将它们分配给两个变量`lng`和`lat`。通过传递在Google注册时获得的API密钥来定义Google Maps实例。在Google Maps实例上调用`search`方法，传递用户提供的经度和纬度值。在检索到的搜索上调用`first`方法，并将与提供的经度和纬度值匹配的第一个位置分配给`my_location`结构。位置名称通过Label小部件显示。为了显示位置的城市、国家和邮政编码，使用`my_location`结构的`city`、`country`和`postal_code`成员。

运行应用程序时，您将被提示输入经度和纬度值。与提供的经度和纬度相关的位置名称、城市、国家和邮政编码将通过四个标签小部件显示在屏幕上，如下截图所示：

![](assets/9931f5fa-74a4-432d-b0e4-7d1910828e4c.png)

# 查找两个位置之间的距离

在这个教程中，您将学习如何找出用户输入的两个位置之间的距离（以公里为单位）。该教程将简单地提示用户输入两个位置，然后单击“查找距离”按钮，两者之间的距离将被显示。

# 操作步骤如下：

让我们根据没有按钮模板的对话框创建一个应用程序，执行以下步骤：

1.  通过将四个标签、两个行编辑和一个推送按钮小部件拖放到表单上，向表单添加四个`QLabel`、两个`QLineEdit`和一个`QPushButton`小部件。

1.  将第一个标签小部件的文本属性设置为“查找两个位置之间的距离”，将第二个标签小部件的文本属性设置为“输入第一个位置”，将第三个标签小部件的文本属性设置为“输入第二个位置”。

1.  删除第四个标签小部件的文本属性，因为它的文本属性将通过代码设置；也就是说，两个输入位置之间的距离将通过代码计算并显示在第四个标签小部件中。

1.  将推送按钮小部件的文本属性设置为“查找距离”。

1.  将两个行编辑小部件的objectName属性设置为`lineEditFirstLocation`和`lineEditSecondLocation`。

1.  将推送按钮小部件的objectName属性设置为`pushButtonFindDistance`。

1.  将第四个标签小部件的objectName属性设置为`labelDistance`。

1.  将应用程序保存为`demoGoogleMap3.ui`。表单现在将显示如下截图所示：

![](assets/fba0fce7-d01a-4b48-b271-0149a0f77bc4.png)

使用Qt Designer创建的用户界面存储在`.ui`文件中，它是一个XML文件。通过应用`pyuic5`实用程序，将XML文件转换为Python代码。您可以在本书的源代码包中找到生成的Python代码`demoGoogleMap3.py`。

1.  要使用在`demoGoogleMap3.py`文件中创建的GUI，我们需要创建另一个Python脚本并在该脚本中导入`demoGoogleMap3.py`文件。

1.  创建另一个名为`callGoogleMap3.pyw`的Python文件，并将`demoGoogleMap3.py`代码导入其中：

```py
import sys
from PyQt5.QtWidgets import QDialog, QApplication
from googlemaps.client import Client
from googlemaps.distance_matrix import distance_matrix
from demoGoogleMap3 import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonFindDistance.clicked.connect(self.
        displayDistance)
        self.show()
    def displayDistance(self):
        api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        gmaps = Client(api_key)
        data = distance_matrix(gmaps,  
        self.ui.lineEditFirstLocation.text(),         
        self.ui.lineEditSecondLocation.text())
        distance = data['rows'][0]['elements'][0]['distance']
        ['text']
        self.ui.labelDistance.setText("Distance between 
        "+self.ui.lineEditFirstLocation.text()+" 
        and "+self.ui.lineEditSecondLocation.text()+" is 
        "+str(distance))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 它是如何工作的…

您创建一个`Client`类的实例并将其命名为`gmaps`。在创建`Client`实例时，您需要传递在注册Google时获得的API密钥。具有objectName`pushButtonFindDistance`的按钮的click()事件连接到`displayDistance`方法。这意味着每当单击按钮时，将调用`displayDistance`方法。在`displayDistance`方法中，您调用`distance_matrix`方法，传递`Client`实例和用户输入的两个位置，以找出它们之间的距离。`distance_matrix`方法返回一个多维数组，该数组分配给数据数组。从数据数组中，访问并将两个位置之间的距离分配给`distance`变量。最终通过Label小部件显示`distance`变量中的值。

运行应用程序时，将提示您输入要了解其相隔距离的两个位置。输入两个位置后，单击查找距离按钮，两个位置之间的距离将显示在屏幕上，如下截图所示：

![](assets/b51275e5-c2e1-4667-899f-94028d2c718b.png)

# 在Google地图上显示位置

在本教程中，您将学习如何在Google地图上显示位置，如果您知道该位置的经度和纬度值。您将被提示简单输入经度和纬度值，当您单击显示地图按钮时，该位置将显示在Google地图上。

# 如何做…

让我们创建一个基于无按钮对话框模板的应用程序，执行以下步骤：

1.  通过将两个Label、两个Line Edit、一个PushButton和一个QWidget小部件拖放到表单上，向表单添加两个QLabel、两个QLineEdit、一个QPushButton和一个QWidget小部件。

1.  将两个Label小部件的文本属性设置为`Longitude`和`Latitude`。

1.  将Push Button小部件的文本属性设置为`Show Map`。

1.  将两个Line Edit小部件的objectName属性设置为`lineEditLongitude`和`lineEditLatitude`。

1.  将Push Button小部件的objectName属性设置为`pushButtonShowMap`。

1.  将应用程序保存为`showGoogleMap.ui`。现在，表单将显示如下截图所示：

![](assets/5ae4fa15-157c-4c1d-9ff0-0f443d60e9a6.png)

1.  下一步是将`QWidget`小部件提升为`QWebEngineView`，因为要显示Google地图，需要`QWebEngineView`。因为Google地图是一个Web应用程序，我们需要一个QWebEngineView来显示和与Google地图交互。

1.  通过右键单击QWidget小部件并从弹出菜单中选择Promote to ...选项来提升`QWidget`小部件。在出现的对话框中，将Base class name选项保留为默认的QWidget。

1.  在Promoted class name框中输入`QWebEngineView`，在header file框中输入`PyQT5.QtWebEngineWidgets`。

1.  单击Promote按钮，将`QWidget`小部件提升为`QWebEngineView`类，如下截图所示：

![](assets/cef6a3a1-49b1-4336-8cc0-eedfadf522fd.png)

1.  单击关闭按钮关闭Promoted Widgets对话框。使用Qt Designer创建的用户界面存储在`.ui`文件中，这是一个XML文件，需要转换为Python代码。使用`pyuic5`实用程序将XML文件转换为Python代码。生成的Python脚本`showGoogleMap.py`可以在本书的源代码包中找到。

1.  将`showGoogleMap.py`脚本视为头文件，并将其导入到您将调用其用户界面设计的文件中。

1.  创建另一个名为`callGoogleMap.pyw`的Python文件，并将`showGoogleMap.py`代码导入其中：

```py
import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
from showGoogleMap import *
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButtonShowMap.clicked.connect(self.dispSite)
        self.show()
    def dispSite(self):
        lng = float(self.ui.lineEditLongitude.text())
        lat = float(self.ui.lineEditLatitude.text())
        URL="https://www.google.com/maps/@"+self.ui.
        lineEditLatitude.text()+","
        +self.ui.lineEditLongitude.text()+",9z"
        self.ui.widget.load(QUrl(URL))
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
```

# 工作原理…

在脚本中，您可以看到具有objectName属性`pushButtonShowMap`的按钮的点击事件与`dispSite()`方法相连。这意味着，每当点击按钮时，将调用`dispSite()`方法。在`dispSite()`方法中，您通过两个Line Edit小部件访问用户输入的经度和纬度，并分别将它们分配给两个变量`lng`和`lat`。然后，您创建一个URL，从[google.com](https://www.google.com/)调用Google地图，并传递用户输入的纬度和经度值。

URL最初是以文本形式存在的，并且被强制转换为`QUrl`实例，并传递给被提升为`QWebEngineView`以显示网站的小部件。`QUrl`是Qt中提供多种方法和属性来管理URL的类。然后，通过`QWebEngineView`小部件显示具有指定纬度和经度值的Google地图。

运行应用程序时，您将被提示输入您想在Google地图上查看的位置的经度和纬度值。输入经度和纬度值后，当您点击“显示地图”按钮时，Google地图将显示该位置，如下面的屏幕截图所示：

![](assets/15c43a7b-76f1-4c6f-84a3-72240cfc43ec.png)
