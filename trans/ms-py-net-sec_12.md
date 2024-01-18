# 从文档、图像和浏览器中提取地理位置和元数据

本章涵盖了 Python 中用于提取 IP 地址地理位置信息、从图像和文档中提取元数据以及识别网站前端和后端使用的 Web 技术的主要模块。此外，我们还介绍了如何从 Chrome 和 Firefox 浏览器中提取元数据以及与存储在 sqlite 数据库中的下载、cookie 和历史数据相关的信息。

本章将涵盖以下主题：

+   用于地理位置的`pygeoip`和`pygeocoder`模块

+   如何使用`Python Image`库从图像中提取元数据

+   如何使用`pypdf`模块从 PDF 文档中提取元数据

+   如何识别网站使用的技术

+   如何从 Chrome 和 Firefox 等网络浏览器中提取元数据

# 技术要求

本章的示例和源代码可在 GitHub 存储库的`chapter 12`文件夹中找到：[`github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security`](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。

您需要在本地计算机上安装至少 4GB 内存的 Python 发行版。

# 提取地理位置信息

在本节中，我们将回顾如何从 IP 地址或域名中提取地理位置信息。

# 地理位置介绍

从 IP 地址或域名获取地理位置的一种方法是使用提供此类信息的服务。在提供此信息的服务中，我们可以强调 hackertarget.com ([`hackertarget.com/geoip-ip-location-lookup/`](https://hackertarget.com/geoip-ip-location-lookup/))。

通过[hackertarget.com](http://hackertarget.com)，我们可以从 IP 地址获取地理位置：

![](img/04da526f-00a3-4154-b344-e4ec7d7f5970.png)

该服务还提供了一个用于从 IP 地址获取地理位置的 REST API：[`api.hackertarget.com/geoip/?q=8.8.8.8`](https://api.hackertarget.com/geoip/?q=8.8.8.8)。

另一个服务是`api.hostip.info`，它提供了按 IP 地址查询的服务：

![](img/3ca09e55-c965-4e81-b66c-4f4bdd84e344.png)

在下一个脚本中，我们使用此服务和`requests`模块来获取包含地理位置信息的 json 响应。

您可以在`ip_to_geo.py`文件中找到以下代码：

```py
import requests

class IPtoGeo(object):

    def __init__(self, ip_address):

        # Initialize objects to store
        self.latitude = ''
        self.longitude = ''
        self.country = ''
        self.city = ''
        self.ip_address = ip_address
        self._get_location()

    def _get_location(self):
        json_request = requests.get('http://api.hostip.info/get_json.php ip=%s&position=true' % self.ip_address).json()

        self.country = json_request['country_name']
        self.country_code = json_request['country_code']
        self.city = json_request['city']
        self.latitude = json_request['lat']
        self.longitude = json_request['lng']

if __name__ == '__main__':
    ip1 = IPtoGeo('8.8.8.8')
    print(ip1.__dict__)
```

这是前一个脚本的**输出**：

```py
{'latitude': '37.402', 'longitude': '-122.078', 'country': 'UNITED STATES', 'city': 'Mountain View, CA', 'ip_address': '8.8.8.8', 'country_code': 'US'}
```

# Pygeoip 介绍

`Pygeoip`是 Python 中可用的模块之一，允许您从 IP 地址检索地理信息。它基于 GeoIP 数据库，这些数据库根据其类型（城市、地区、国家、ISP）分布在几个文件中。该模块包含几个函数来检索数据，例如国家代码、时区或包含与特定地址相关的所有信息的完整注册。

`Pygeoip`可以从官方 GitHub 存储库下载：[`github.com/appliedsec/pygeoip`](http://github.com/appliedsec/pygeoip)。

如果我们查询模块的帮助，我们会看到必须使用的主要类来实例化允许我们进行查询的对象：

![](img/0ee0956f-9aa4-4942-a081-23a5431e4c64.png)

要构建对象，我们使用一个接受文件作为参数的数据库的构造函数。此文件的示例可以从以下网址下载：[`dev.maxmind.com/geoip/legacy/geolite`](http://dev.maxmind.com/geoip/legacy/geolite)。

![](img/1062cdc6-46f3-4ce5-b7d2-713f98345e54.png)

此类中可用的以下方法允许您从 IP 地址或域名获取国家名称。

您可以在`**geoip.py**`文件中找到以下代码，该文件位于`pygeopip`文件夹中：

```py
import pygeoip
import pprint
gi = pygeoip.GeoIP('GeoLiteCity.dat')
pprint.pprint("Country code: %s " %(str(gi.country_code_by_addr('173.194.34.192'))))
pprint.pprint("Country code: %s " %(str(gi.country_code_by_name('google.com'))))
pprint.pprint("Country name: %s " %(str(gi.country_name_by_addr('173.194.34.192'))))
pprint.pprint("Country code: %s " %(str(gi.country_name_by_name('google.com'))))
```

还有一些方法可以从 IP 和主机地址中获取组织和服务提供商：

![](img/8926cf88-04bd-4d61-91aa-f29823539be8.png)

这是从 IP 地址和域名获取特定组织信息的示例：

```py
gi2 = pygeoip.GeoIP('GeoIPASNum.dat')
pprint.pprint("Organization by addr: %s " %(str(gi2.org_by_addr('173.194.34.192'))))
pprint.pprint("Organization by name: %s " %(str(gi2.org_by_name('google.com'))))
```

还有一些方法可以让我们以字典形式获取有关国家、城市、纬度或经度的数据结构：

![](img/b1cca883-a48c-4887-93eb-6da123e990bd.png)

这是一个从 IP 地址获取地理位置信息的示例：

```py
for record,value in gi.record_by_addr('173.194.34.192').items():
    print(record + "-->" + str(value))
```

我们可以看到上一个脚本返回的所有地理位置信息：

![](img/971e5d90-8cd3-4b50-9499-d2a27403a429.png)

在下一个脚本中，我们有两种方法，`geoip_city()`用于获取有关位置的信息，`geoip_country()`用于获取国家，两者都是从 IP 地址获取的。

在这两种方法中，首先实例化一个带有包含数据库的文件路径的`GeoIP`类。接下来，我们将查询特定记录的数据库，指定 IP 地址或域。这将返回一个包含城市、`region_name`、`postal_code`、`country_name`、`latitude`和`longitude`字段的记录。

您可以在`pygeopip`文件夹中的`pygeoip_test.py`文件中找到以下代码：

```py
import pygeoip

def main():
 geoip_country() 
 geoip_city()

def geoip_city():
 path = 'GeoLiteCity.dat'
 gic = pygeoip.GeoIP(path)
 print(gic.record_by_addr('64.233.161.99'))
 print(gic.record_by_name('google.com'))
 print(gic.region_by_name('google.com'))
 print(gic.region_by_addr('64.233.161.99'))

def geoip_country(): 
 path = 'GeoIP.dat'
 gi = pygeoip.GeoIP(path)
 print(gi.country_code_by_name('google.com'))
 print(gi.country_code_by_addr('64.233.161.99'))
 print(gi.country_name_by_name('google.com'))
 print(gi.country_name_by_addr('64.233.161.99'))

if __name__ == '__main__':
 main()
```

我们可以看到返回的信息对于两种情况是相同的：

![](img/4c071006-cd8a-48ea-8b8b-06e00dfdf1ec.png)

# pygeocoder 简介

`pygeocoder`是一个简化使用 Google 地理位置功能的 Python 模块。使用此模块，您可以轻松找到与坐标对应的地址，反之亦然。我们还可以使用它来验证和格式化地址。

该模块位于官方 Python 存储库中，因此您可以使用`pip`来安装它。在[`pypi.python.org/pypi/pygeocoder`](https://pypi.python.org/pypi/pygeocoder)URL 中，我们可以看到此模块的最新版本：`$ pip install pygeocoder`。

该模块使用 Google Geocoding API v3 服务从特定地址检索坐标：

![](img/7360aef5-9517-48e7-a26e-29c8c7a9d535.png)

该模块的主要类是`Geocoder`类，它允许从地点描述和特定位置进行查询。

在这个截图中，我们可以看到`GeoCoder`类的`help`命令的返回：

![](img/874c59b5-88f5-49f3-a0bf-a8b1b9f20865.png)

从地点描述中获取地点、坐标、纬度、经度、国家和邮政编码的示例。您还可以执行反向过程，即从对应于地理点的纬度和经度的坐标开始，可以恢复该站点的地址。

您可以在`pygeocoder`文件夹中的`PyGeoCoderExample.py`文件中找到以下代码：

```py
from pygeocoder import Geocoder

results = Geocoder.geocode("Mountain View")

print(results.coordinates)
print(results.country)
print(results.postal_code)
print(results.latitude)
print(results.longitude)
results = Geocoder.reverse_geocode(results.latitude, results.longitude)
print(results.formatted_address)
```

我们可以看到上一个脚本返回的所有地理位置信息：

![](img/7c2b63cf-17e6-4c0b-8fd1-3fe517203204.png)

# Python 中的 MaxMind 数据库

还有其他使用 MaxMind 数据库的 Python 模块：

+   **geoip2:** 提供对 GeoIP2 网络服务和数据库的访问

+   [`github.com/maxmind/GeoIP2-python`](https://github.com/maxmind/GeoIP2-python)

+   **maxminddb-geolite2:** 提供一个简单的 MaxMindDB 阅读器扩展

+   [`github.com/rr2do2/maxminddb-geolite2`](https://github.com/rr2do2/maxminddb-geolite2)

在下一个脚本中，我们可以看到如何使用`maxminddb-geolite2`包的示例。

您可以在`geolite2_example.py`文件中找到以下代码：

```py
import socket
from geolite2 import geolite2
import argparse
import json

if __name__ == '__main__':
 # Commandline arguments
 parser = argparse.ArgumentParser(description='Get IP Geolocation info')
 parser.add_argument('--hostname', action="store", dest="hostname",required=True)

# Parse arguments
 given_args = parser.parse_args()
 hostname = given_args.hostname
 ip_address = socket.gethostbyname(hostname)
 print("IP address: {0}".format(ip_address))

# Call geolite2
 reader = geolite2.reader()
 response = reader.get(ip_address)
 print (json.dumps(response['continent']['names']['en'],indent=4))
 print (json.dumps(response['country']['names']['en'],indent=4))
 print (json.dumps(response['location']['latitude'],indent=4))
 print (json.dumps(response['location']['longitude'],indent=4))
 print (json.dumps(response['location']['time_zone'],indent=4))
```

在这个截图中，我们可以看到使用 google.com 作为主机名执行上一个脚本：

`python geolite2_example.py --hostname google.com`

此脚本将显示类似于以下内容的输出：

![](img/39c1d15f-fd21-4639-b49c-70fc88b615cc.png)

# 从图像中提取元数据

在这一部分，我们将回顾如何使用 PIL 模块从图像中提取 EXIF 元数据。

# Exif 和 PIL 模块简介

我们在 Python 中找到的用于处理和操作图像的主要模块之一是`PIL`。`PIL`模块允许我们提取图像的`EXIF`元数据。

**Exif（Exchange Image File Format）**是一个规范，指示在保存图像时必须遵循的规则，并定义了如何在图像和音频文件中存储元数据。这个规范今天在大多数移动设备和数码相机中应用。

`PIL.ExifTags`模块允许我们从这些标签中提取信息：

![](img/f92bcdd3-de77-40ae-b712-31142fd3c023.png)

我们可以在`pillow`模块内的`exiftags`包的官方文档中查看官方文档：

ExifTags 包含一个带有许多众所周知的`EXIF 标签`的常量和名称的字典结构。

在这张图片中，我们可以看到`TAGS.values()`方法返回的所有标签：

![](img/5d180f44-29e7-424e-b1c4-523fcc817f78.png)

# 从图像获取 EXIF 数据

首先，我们导入了`PIL`图像和`PIL TAGS`模块。`PIL`是 Python 中的图像处理模块。它支持许多文件格式，并具有强大的图像处理能力。然后我们遍历结果并打印值。还有许多其他模块支持 EXIF 数据提取，例如`ExifRead`。在这个例子中，为了获取`EXIF`数据，我们可以使用`_getexif()`方法。

您可以在`exiftags`文件夹中的`get_exif_tags.py`文件中找到以下代码：

```py
from PIL import Image
from PIL.ExifTags import TAGS

for (i,j) in Image.open('images/image.jpg')._getexif().items():
    print('%s = %s' % (TAGS.get(i), j))
```

# 了解 Exif 元数据

要获取图像的`EXIF`标签信息，可以使用图像对象的`_getexif()`方法。例如，我们可以编写一个函数，在图像路径中，可以返回`EXIF`标签的信息。

在`exiftags`文件夹中的`extractDataFromImages.py`文件中提供了以下函数：

```py
def get_exif_metadata(image_path):
    exifData = {}
    image = Image.open(image_path)
    if hasattr(image, '_getexif'):
        exifinfo = image._getexif()
        if exifinfo is not None:
            for tag, value in exifinfo.items():
                decoded = TAGS.get(tag, tag)
                exifData[decoded] = value
 decode_gps_info(exifData)
 return exifData
```

这些信息可以通过解码我们以纬度-经度值格式获得的信息来改进，对于它们，我们可以编写一个函数，给定`GPSInfo`类型的`exif`属性，解码该信息：

```py
def decode_gps_info(exif):
    gpsinfo = {}
    if 'GPSInfo' in exif:
    '''
    Raw Geo-references
    for key in exif['GPSInfo'].keys():
        decode = GPSTAGS.get(key,key)
        gpsinfo[decode] = exif['GPSInfo'][key]
    exif['GPSInfo'] = gpsinfo
    '''

     #Parse geo references.
     Nsec = exif['GPSInfo'][2][2][0] / float(exif['GPSInfo'][2][2][1])
     Nmin = exif['GPSInfo'][2][1][0] / float(exif['GPSInfo'][2][1][1])
     Ndeg = exif['GPSInfo'][2][0][0] / float(exif['GPSInfo'][2][0][1])
     Wsec = exif['GPSInfo'][4][2][0] / float(exif['GPSInfo'][4][2][1])
     Wmin = exif['GPSInfo'][4][1][0] / float(exif['GPSInfo'][4][1][1])
     Wdeg = exif['GPSInfo'][4][0][0] / float(exif['GPSInfo'][4][0][1])
     if exif['GPSInfo'][1] == 'N':
         Nmult = 1
     else:
         Nmult = -1
     if exif['GPSInfo'][1] == 'E':
         Wmult = 1
     else:
         Wmult = -1
         Lat = Nmult * (Ndeg + (Nmin + Nsec/60.0)/60.0)
         Lng = Wmult * (Wdeg + (Wmin + Wsec/60.0)/60.0)
         exif['GPSInfo'] = {"Lat" : Lat, "Lng" : Lng}
```

在上一个脚本中，我们将 Exif 数据解析为一个由元数据类型索引的数组。有了完整的数组，我们可以搜索数组，看它是否包含`GPSInfo`的`Exif`标签。如果它包含`GPSInfo`标签，那么我们将知道该对象包含 GPS 元数据，我们可以在屏幕上打印一条消息。

在下面的图像中，我们可以看到我们还在`GPSInfo`对象中获取了有关图像位置的信息：

![](img/e0157494-f454-414c-a0bc-088c111e5445.png)

# 从网络图像中提取元数据

在本节中，我们将构建一个脚本，连接到一个网站，下载网站上的所有图像，然后检查它们的`Exif`元数据。

对于这个任务，我们使用了 Python3 的`urllib`模块，提供了`parse`和`request`包：

[`docs.python.org/3.0/library/urllib.parse.html`](https://docs.python.org/3.0/library/urllib.parse.html)

[`docs.python.org/3.0/library/urllib.request.html`](https://docs.python.org/3.0/library/urllib.request.html)

您可以在`exiftags`文件夹中的`exif_images_web_page.py`文件中找到以下代码。

此脚本包含使用`BeautifulSoup`和`lxml 解析器`在网站中查找图像以及在图像文件夹中下载图像的方法：

```py
def findImages(url):
    print('[+] Finding images on ' + url)
    urlContent = requests.get(url).text
    soup = BeautifulSoup(urlContent,'lxml')
    imgTags = soup.findAll('img')
    return imgTags

def downloadImage(imgTag):
    try:
        print('[+] Dowloading in images directory...'+imgTag['src'])
        imgSrc = imgTag['src']
        imgContent = urlopen(imgSrc).read()
        imgFileName = basename(urlsplit(imgSrc)[2])
        imgFile = open('images/'+imgFileName, 'wb')
        imgFile.write(imgContent)
        imgFile.close()
        return imgFileName
    except Exception as e:
        print(e)
        return ''
```

这是从图像目录中提取图像元数据的函数：

```py
def printMetadata():
    print("Extracting metadata from images in images directory.........")
    for dirpath, dirnames, files in os.walk("images"):
    for name in files:
        print("[+] Metadata for file: %s " %(dirpath+os.path.sep+name))
            try:
                exifData = {}
                exif = get_exif_metadata(dirpath+os.path.sep+name)
                for metadata in exif:
                print("Metadata: %s - Value: %s " %(metadata, exif[metadata]))
            except:
                import sys, traceback
                traceback.print_exc(file=sys.stdout)
```

这是我们的主要方法，它从参数中获取一个 url，并调用`findImages(url)`，`downloadImage(imgTags)`和`printMetadata()`方法：

```py
def main():
    parser = optparse.OptionParser('-url <target url>')
    parser.add_option('-u', dest='url', type='string', help='specify url address')
    (options, args) = parser.parse_args()
    url = options.url
    if url == None:
        print(parser.usage)
        exit(0)
    else:#find and download images and extract metadata
       imgTags = findImages(url) print(imgTags) for imgTag in imgTags: imgFileName = downloadImage(imgTag) printMetadata()
```

# 从 pdf 文档中提取元数据

在本节中，我们将回顾如何使用`pyPDF2`模块从 pdf 文档中提取元数据。

# PyPDF2 简介

Python 中用于从 PDF 文档中提取数据的模块之一是`PyPDF2`。该模块可以直接使用 pip install 实用程序下载，因为它位于官方 Python 存储库中。

在[`pypi.org/project/PyPDF2/`](https://pypi.org/project/PyPDF2/) URL 中，我们可以看到这个模块的最新版本：

![](img/96634174-b564-4c43-a5db-ad1ba4d42650.png)

该模块为我们提供了提取文档信息、加密和解密文档的能力。要提取元数据，我们可以使用`PdfFileReader`类和`getDocumentInfo()`方法，它返回一个包含文档数据的字典：

![](img/084b4e2a-7946-45d5-b287-cacaee766c70.png)

以下函数将允许我们获取“pdf”文件夹中所有 PDF 文档的信息。

你可以在`pypdf`文件夹中的`extractDataFromPDF.py`文件中找到以下代码：

```py
#!usr/bin/env python
# coding: utf-8

from PyPDF2 import PdfFileReader, PdfFileWriter
import os, time, os.path, stat

from PyPDF2.generic import NameObject, createStringObject

class bcolors:
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_metadata():
  for dirpath, dirnames, files in os.walk("pdf"):
    for data in files:
      ext = data.lower().rsplit('.', 1)[-1]
      if ext in ['pdf']:
        print(bcolors.OKGREEN + "------------------------------------------------------------------------------------")
        print(bcolors.OKGREEN + "[--- Metadata : " + bcolors.ENDC + bcolors.BOLD + "%s " %(dirpath+os.path.sep+data) + bcolors.ENDC)
        print(bcolors.OKGREEN + "------------------------------------------------------------------------------------")
        pdf = PdfFileReader(open(dirpath+os.path.sep+data, 'rb'))
        info = pdf.getDocumentInfo()

        for metaItem in info:

          print (bcolors.OKGREEN + '[+] ' + metaItem.strip( '/' ) + ': ' + bcolors.ENDC + info[metaItem])

        pages = pdf.getNumPages()
        print (bcolors.OKGREEN + '[+] Pages:' + bcolors.ENDC, pages)

        layout = pdf.getPageLayout()
        print (bcolors.OKGREEN + '[+] Layout: ' + bcolors.ENDC + str(layout))

```

在这部分代码中，我们使用`getXmpMetadata()`方法获取与文档相关的其他信息，如贡献者、发布者和 PDF 版本：

```py
        xmpinfo = pdf.getXmpMetadata()

        if hasattr(xmpinfo,'dc_contributor'): print (bcolors.OKGREEN + '[+] Contributor:' + bcolors.ENDC, xmpinfo.dc_contributor)
        if hasattr(xmpinfo,'dc_identifier'): print (bcolors.OKGREEN + '[+] Identifier:' + bcolors.ENDC, xmpinfo.dc_identifier)
        if hasattr(xmpinfo,'dc_date'): print (bcolors.OKGREEN + '[+] Date:' + bcolors.ENDC, xmpinfo.dc_date)
        if hasattr(xmpinfo,'dc_source'): print (bcolors.OKGREEN + '[+] Source:' + bcolors.ENDC, xmpinfo.dc_source)
        if hasattr(xmpinfo,'dc_subject'): print (bcolors.OKGREEN + '[+] Subject:' + bcolors.ENDC, xmpinfo.dc_subject)
        if hasattr(xmpinfo,'xmp_modifyDate'): print (bcolors.OKGREEN + '[+] ModifyDate:' + bcolors.ENDC, xmpinfo.xmp_modifyDate)
        if hasattr(xmpinfo,'xmp_metadataDate'): print (bcolors.OKGREEN + '[+] MetadataDate:' + bcolors.ENDC, xmpinfo.xmp_metadataDate)
        if hasattr(xmpinfo,'xmpmm_documentId'): print (bcolors.OKGREEN + '[+] DocumentId:' + bcolors.ENDC, xmpinfo.xmpmm_documentId)
        if hasattr(xmpinfo,'xmpmm_instanceId'): print (bcolors.OKGREEN + '[+] InstanceId:' + bcolors.ENDC, xmpinfo.xmpmm_instanceId)
        if hasattr(xmpinfo,'pdf_keywords'): print (bcolors.OKGREEN + '[+] PDF-Keywords:' + bcolors.ENDC, xmpinfo.pdf_keywords)
        if hasattr(xmpinfo,'pdf_pdfversion'): print (bcolors.OKGREEN + '[+] PDF-Version:' + bcolors.ENDC, xmpinfo.pdf_pdfversion)

        if hasattr(xmpinfo,'dc_publisher'):
          for y in xmpinfo.dc_publisher:
            if y:
              print (bcolors.OKGREEN + "[+] Publisher:\t" + bcolors.ENDC + y) 

      fsize = os.stat((dirpath+os.path.sep+data))
      print (bcolors.OKGREEN + '[+] Size:' + bcolors.ENDC, fsize[6], 'bytes \n\n')

get_metadata()
```

os（操作系统）模块中的“walk”函数对于浏览特定目录中包含的所有文件和目录非常有用。

在这个截图中，我们可以看到先前的脚本正在读取 pdf 文件夹中的文件的输出：

![](img/72fb6e0e-abf0-49bf-bbbb-9e77bd3b737d.png)

它提供的另一个功能是解密使用密码加密的文档：

![](img/9e6c1da5-031c-45fe-8ef0-8206d186fcf1.png)

# Peepdf

`Peepdf`是一个分析 PDF 文件的 Python 工具，允许我们可视化文档中的所有对象。它还具有分析 PDF 文件的不同版本、对象序列和加密文件的能力，以及修改和混淆 PDF 文件的能力：[`eternal-todo.com/tools/peepdf-pdf-analysis-tool`](http://eternal-todo.com/tools/peepdf-pdf-analysis-tool)。

# 识别网站使用的技术

在这一部分中，我们将回顾如何使用 builtwith 和 Wappalyzer 来识别网站使用的技术。

# 介绍 builtwith 模块

构建网站所使用的技术类型将影响您跟踪它的方式。要识别这些信息，您可以利用 Wappalyzer 和 Builtwith 等工具（[`builtwith.com`](https://builtwith.com)）。验证网站所使用的技术类型的有用工具是 builtWith 模块，可以通过以下方式安装：

`pip install builtwith`

该模块有一个名为`parse`的方法，通过 URL 参数传递，并返回网站使用的技术作为响应。以下是一个示例：

```py
>>> import builtwith
>>> builtwith.parse('http://example.webscraping.com')
{u'javascript-frameworks': [u'jQuery', u'Modernizr', u'jQuery UI'],
u'programming-languages': [u'Python'],
u'web-frameworks': [u'Web2py', u'Twitter Bootstrap'],
u'web-servers': [u'Nginx']}
```

文档可在[`bitbucket.org/richardpenman/builtwith`](https://bitbucket.org/richardpenman/builtwith)上找到，模块可在 pypi 存储库的[`pypi.org/project/builtwith/`](https://pypi.org/project/builtwith/)上找到。

# Wappalyzer

另一个用于恢复此类信息的工具是 Wappalyzer。Wappalyzer 具有 Web 应用程序签名数据库，允许您从 50 多个类别中识别 900 多种 Web 技术。

该工具分析网站的多个元素以确定其技术，它分析以下 HTML 元素：

+   服务器上的 HTTP 响应头

+   Meta HTML 标签

+   JavaScript 文件，包括单独的文件和嵌入在 HTML 中的文件

+   特定的 HTML 内容

+   HTML 特定注释

`python-Wappalyzer`是一个用于从 Python 脚本获取此信息的 Python 接口（[`github.com/chorsley/python-Wappalyzer`](https://github.com/chorsley/python-Wappalyzer)）：

`pip install python-Wappalyzer`

我们可以轻松地使用 wappalyzer 模块获取网站前端和后端层中使用的技术的信息：

![](img/8e3d7549-09c9-4ca7-97d8-7249430b6263.png)

# wig - webapp 信息收集器

wig 是一个用 Python3 开发的 Web 应用信息收集工具，可以识别多个内容管理系统和其他管理应用。每个检测到的 CMS 都会显示其最可能的版本。在内部，它从'server'和'x powered-by'头部获取服务器上的操作系统（[`github.com/jekyc/wig`](https://github.com/jekyc/wig)）。

这些是 wig 脚本在 Python3 环境中提供的选项：

![](img/dab0ef28-bbd1-447e-83d9-c69820d648cc.png)

在这张图片中，我们可以看到[testphp.vulneb.com](http://testphp.vulneb.com)网站使用的技术：

![](img/da6a018b-c973-400c-ba94-be4bd3af290b.png)

在这张图片中，我们可以看到它是如何检测到[drupal.com](http://drupal.com)网站使用的 CMS 版本和其他有趣的文件：

![](img/9305f9f7-6133-4d00-8e5c-b4f1643a74f3.png)

# 从 Web 浏览器中提取元数据

在本节中，我们将审查如何从浏览器中提取元数据，例如 chrome 和 firefox。

# 使用 dumpzilla 进行 Python 中的 Firefox 取证

Dumpzilla 是一个非常有用，多功能和直观的工具，专门用于 Mozilla 浏览器的取证分析。Dumpzilla 能够从 Firefox，Iceweasel 和 Seamonkey 浏览器中提取所有相关信息，以便进一步分析，以提供有关遭受攻击，密码和电子邮件的线索。它在 Unix 系统和 Windows 32/64 位下运行。

该应用程序在命令行下运行，我们可以访问大量有价值的信息，其中包括：

+   Cookies + DOM 存储（HTML 5）

+   用户偏好（域权限，代理设置）

+   查看下载历史记录

+   Web 表单数据（搜索，电子邮件，评论等）

+   标记

+   浏览器中保存的密码

+   提取 HTML5 缓存（离线缓存）

+   插件和扩展以及它们使用的路由或 URL

+   添加为例外的 SSL 证书

为了完成对浏览器的取证分析，建议使用缓存中的数据提取应用程序，例如 MozCache ([`mozcache.sourceforge.net`](http://mozcache.sourceforge.net))。

要求：

+   Python 3.x 版本

+   Unix 系统（Linux 或 Mac）或 Windows 系统

+   可选的`Python Magic`模块：[`github.com/ahupp/python-magic`](https://github.com/ahupp/python-magic)

# Dumpzilla 命令行

找到要进行审计的浏览器配置文件目录。这些配置文件位于不同的目录中，具体取决于您的操作系统。第一步是了解存储浏览器用户配置文件信息的目录。

每个操作系统的位置如下：

+   Win7 和 10 配置文件：`'C:\Users\%USERNAME%\AppData\Roaming\Mozilla\Firefox\Profiles\xxxx.default'`

+   MacOS 配置文件：`'/Users/$USER/Library/Application Support/Firefox/Profiles/xxxx.default'`

+   Unix 配置文件：`'/home/$USER/.mozilla/firefox/xxxx.default'`

您可以从 git 存储库下载`dumpzilla` Python 脚本，并使用 Python3 运行该脚本，指向您的浏览器配置文件目录的位置：[`github.com/Busindre/dumpzilla`](https://github.com/Busindre/dumpzilla)。

这些是脚本提供的选项：

```py
python3 dumpzilla.py "/root/.mozilla/firefox/[Your Profile.default]"
```

![](img/e2a8fff8-a533-46c2-b5cb-2d367f649f00.png)

这将返回有关互联网浏览信息的报告，然后显示所收集信息的摘要图表：

![](img/f56fbaef-b459-4144-9333-b27ca3cc2bbd.png)

# 使用 firefeed 进行 Python 中的 Firefox 取证

Firefed 是一个工具，以命令行模式运行，允许您检查 Firefox 配置文件。可以提取存储的密码，偏好设置，插件和历史记录（[`github.com/numirias/firefed`](https://github.com/numirias/firefed)）。

这些是`firefed`脚本提供的选项：

![](img/316eecbb-179b-4dff-be66-172f3b61937d.png)

该工具读取位于您的用户名火狐配置文件中的`profiles.ini`文件。

在 Windows 操作系统中，此文件位于`C:\Users\username\AppData\Roaming\Mozilla\Firefox`。

您还可以使用`%APPDATA%\Mozilla\Firefox\Profiles`命令检测此文件夹。

有关更多信息，请参阅 mozilla 网站的官方文档：[`support.mozilla.org/en-US/kb/profiles-where-firefox-stores-user-data#w_how-do-i-find-my-profile`](https://support.mozilla.org/en-US/kb/profiles-where-firefox-stores-user-data#w_how-do-i-find-my-profile)。

# 使用 Python 进行 Chrome 取证

Google Chrome 将浏览历史记录存储在以下位置的 SQLite 数据库中：

+   Windows 7 和 10：`C:\Users\[USERNAME]\AppData\Local\Google\Chrome\`

+   Linux：`/home/$USER/.config/google-chrome/`

包含浏览历史记录的数据库文件存储在名为“History”的 Default 文件夹下，可以使用任何 SQlite 浏览器（[`sqlitebrowser.org/`](https://sqlitebrowser.org/)）进行检查。

在 Windows 计算机上，该数据库通常可以在以下路径下找到：

`C:\Users\<YOURUSERNAME>\AppData\Local\Google\Chrome\User Data\Default`

例如，在 Windows 操作系统中，路径为`C:\Users\<username>\AppData\Local\Google\Chrome\User Data\Default\History`，我们可以找到存储 Chrome 浏览历史记录的 SQLite 数据库。

这是历史数据库和相关字段的表：

+   **downloads:** `id`, `current_path`, `target_path`, `start_time`, `received_bytes`, `total_bytes`, `state`, `danger_type`, `interrupt_reason`, `end_time`, `opened`, `referrer`, `by_ext_id`, `by_ext_name`, `etag`, `last_modified`, `mime_type`, `original_mime_type`

+   **downloads_url_chains**: `id`, `chain_index`, `url`

+   **keyword_search_terms:** `keyword_id`, `url_id`, `lower_term`, `term`

+   **meta:** `key`, `value`

+   **segment_usage:**  `id`, `segment_id`, `time_slot`, `visit_count`

+   **segments:** `id`, `name`, `url_id`

+   **urls:** `id`, `url`, `title`, `visit_count`, `typed_count`, `last_visit_time`, `hidden`, `favicon_id`

在这张图片中，我们可以看到一个 SQlite 浏览器的截图，显示了历史数据库中可用的表：

![](img/adc00515-d4fc-4183-aa0e-2c22b59573a4.png)

Chrome 将其数据本地存储在一个`SQLite 数据库`中。因此，我们只需要编写一个 Python 脚本，该脚本将连接到数据库，查询必要的字段，并从表中提取数据。

我们可以编写一个 Python 脚本，从下载表中提取信息。您只需要**`导入 sqlite3`**模块，该模块随 Python 安装而来。

您可以在与 Python3.x 兼容的`ChromeDownloads.py`文件中找到以下代码：

```py
import sqlite3
import datetime
import optparse

def fixDate(timestamp):
    #Chrome stores timestamps in the number of microseconds since Jan 1 1601.
    #To convert, we create a datetime object for Jan 1 1601...
    epoch_start = datetime.datetime(1601,1,1)
    #create an object for the number of microseconds in the timestamp
    delta = datetime.timedelta(microseconds=int(timestamp))
    #and return the sum of the two.
    return epoch_start + delta

selectFromDownloads = 'SELECT target_path, referrer, start_time, end_time, received_bytes FROM downloads;'

def getMetadataHistoryFile(locationHistoryFile):
    sql_connect = sqlite3.connect(locationHistoryFile)
    for row in sql_connect.execute(selectFromDownloads):
        print ("Download:",row[0].encode('utf-8'))
        print ("\tFrom:",str(row[1]))
        print ("\tStarted:",str(fixDate(row[2])))
        print ("\tFinished:",str(fixDate(row[3])))
        print ("\tSize:",str(row[4]))

def main():
    parser = optparse.OptionParser('-location <target location>')
    parser.add_option('-l', dest='location', type='string', help='specify url address')

    (options, args) = parser.parse_args()
     location = options.location
     print(location)
     if location == None:
         exit(0)
     else:
         getMetadataHistoryFile(location)

if __name__ == '__main__':
    main()
```

我们可以看到提供脚本使用`-h`参数的选项：

`python .\ChromeDownloads.py -h`

要执行前面的脚本，我们需要传递一个参数，即您的历史文件数据库的位置：

![](img/26627db3-b833-4216-814a-422676282c04.png)

# 使用 Hindsight 进行 Chrome 取证

Hindsight 是一个用于解析用户 Chrome 浏览器数据的开源工具，允许您分析多种不同类型的 Web 工件，包括 URL、下载历史、缓存记录、书签、偏好设置、浏览器扩展、HTTP cookie 和以 cookie 形式的本地存储日志。

该工具可以在 GitHub 和 pip 存储库中找到：

[`github.com/obsidianforensics/hindsight`](https://github.com/obsidianforensics/hindsight)

[`pypi.org/project/pyhindsight/`](https://pypi.org/project/pyhindsight/)

在这个截图中，我们可以看到这个模块的最新版本：

![](img/d756ebc5-251d-4a2f-814c-fd736f30110a.png)

我们可以使用`pip install pyhindsight`命令进行安装。

安装了模块后，我们可以从 GitHub 存储库下载源代码：

[`github.com/obsidianforensics/hindsight`](https://github.com/obsidianforensics/hindsight)

![](img/a5bb0376-8c43-4192-a408-20ec966dec2b.png)

我们可以以两种方式执行它。第一种是使用`hindsight.py`脚本，第二种是通过启动`hindsight_gui.py`脚本，它提供了一个用于输入 Chrome 配置文件位置的 Web 界面。

对于使用`hindsight.py`执行，我们只需要传递一个强制参数(**`-i`,**`--input`)，即您的 Chrome 配置文件的位置，具体取决于您的操作系统：

![](img/99747764-86a9-46c2-a70e-e9069eefbfe9.png)

这些是我们需要了解的 Chrome 配置文件的默认位置，以设置输入参数：

![](img/a369573f-47a5-45f5-89dc-8279e084dd7f.png)

第二种方法是运行“`hindsight_gui.py`”并在浏览器中访问[`localhost:8080`](http://localhost:8080)：

![](img/34b8947b-9ba6-4b35-bc92-289def323701.png)

唯一强制性的字段是配置文件路径：

![](img/de7e2530-62f9-4383-be72-4ad4c5e50beb.png)

如果我们尝试在打开 Chrome 浏览器进程的情况下运行脚本，它将阻止该进程，因为我们需要在运行之前关闭 Chrome 浏览器。

这是当您尝试执行脚本时，chrome 进程正在运行时的错误消息：

![](img/b10ab9d9-ab8e-43d6-a1d4-80bdf11bd1bb.png)

# 总结

本章的目标之一是了解允许我们从文档和图像中提取元数据的模块，以及从 IP 地址和域名中提取地理位置信息的模块。我们讨论了如何获取域名信息，例如特定网页中使用的技术和 CMS。最后，我们回顾了如何从 Chrome 和 Firefox 等网络浏览器中提取元数据。本章中审查的所有工具都允许我们获取可能对我们的渗透测试或审计过程的后续阶段有用的信息。

在下一章节中，我们将探讨用于实现加密和隐写术的编程包和 Python 模块。

# 问题

1.  Python 中哪个模块允许我们从 IP 地址检索地理信息？

1.  哪个模块使用 Google Geocoding API v3 服务来检索特定地址的坐标？

1.  `Pygeocoder`模块的主要类是什么，它允许从地点描述和特定位置进行查询？

1.  哪种方法允许反向过程从对应于纬度和经度的坐标中恢复所述站点的地址？

1.  `pygeoip`模块中的哪个方法允许我们从传递的 IP 地址获取国家名称的值？

1.  `pygeoip`模块中的哪个方法允许我们从 IP 地址获取地理数据（国家、城市、地区、纬度、经度）的字典形式的结构？

1.  `pygeoip`模块中的哪个方法允许我们从域名获取组织的名称？

1.  哪个 Python 模块允许我们从 PDF 文档中提取元数据？

1.  我们可以使用哪个类和方法来获取 PDF 文档的信息？

1.  哪个模块允许我们从 EXIF 标签中提取图像信息？

# 进一步阅读

在这些链接中，您将找到有关本章中提到的工具及其官方文档的更多信息：

+   [`bitbucket.org/xster/pygeocoder/wiki/Home`](https://bitbucket.org/xster/pygeocoder/wiki/Home)

+   [`chrisalbon.com/python/data_wrangling/geocoding_and_reverse_geocoding/`](https://chrisalbon.com/python/data_wrangling/geocoding_and_reverse_geocoding/)

+   [`pythonhosted.org/PyPDF2`](https://pythonhosted.org/PyPDF2)

+   [`www.dumpzilla.org`](http://www.dumpzilla.org)

+   [`tools.kali.org/forensics/dumpzilla`](https://tools.kali.org/forensics/dumpzilla)

+   [`forensicswiki.org/wiki/Google_Chrome`](http://forensicswiki.org/wiki/Google_Chrome)

+   [`sourceforge.net/projects/chromensics`](https://sourceforge.net/projects/chromensics)
