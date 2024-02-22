# 第三章：处理数据

在本章中，我们将涵盖：

+   使用 CSV 和 JSON 数据

+   使用 AWS S3 存储数据

+   使用 MySQL 存储数据

+   使用 PostgreSQL 存储数据

+   使用 Elasticsearch 存储数据

+   如何使用 AWS SQS 构建健壮的 ETL 管道

# 介绍

在本章中，我们将介绍 JSON、CSV 和 XML 格式的数据使用。这将包括解析和将这些数据转换为其他格式的方法，包括将数据存储在关系数据库、Elasticsearch 等搜索引擎以及包括 AWS S3 在内的云存储中。我们还将讨论通过使用 AWS Simple Queue Service（SQS）等消息系统创建分布式和大规模的抓取任务。目标是既了解您可能检索和需要解析的各种数据形式，又了解可以存储您已抓取的数据的各种后端。最后，我们首次介绍了 Amazon Web Service（AWS）的一项服务。在本书结束时，我们将深入研究 AWS，并进行初步介绍。

# 使用 CSV 和 JSON 数据

从 HTML 页面中提取数据是使用上一章节中的技术完成的，主要是使用 XPath 通过各种工具和 Beautiful Soup。虽然我们主要关注 HTML，但 HTML 是 XML（可扩展标记语言）的一种变体。XML 曾经是在 Web 上表达数据的最流行形式之一，但其他形式已经变得流行，甚至超过了 XML。

您将看到的两种常见格式是 JSON（JavaScript 对象表示）和 CSV（逗号分隔值）。CSV 易于创建，是许多电子表格应用程序的常见形式，因此许多网站提供该格式的数据，或者您需要将抓取的数据转换为该格式以进行进一步存储或协作。由于 JSON 易于在 JavaScript（和 Python）等编程语言中使用，并且许多数据库现在支持它作为本机数据格式，因此 JSON 确实已成为首选格式。

在这个示例中，让我们来看看将抓取的数据转换为 CSV 和 JSON，以及将数据写入文件，以及从远程服务器读取这些数据文件。我们将研究 Python CSV 和 JSON 库。我们还将研究使用`pandas`进行这些技术。

这些示例中还隐含了将 XML 数据转换为 CSV 和 JSON 的过程，因此我们不会为这些示例专门设置一个部分。

# 准备工作

我们将使用行星数据页面，并将该数据转换为 CSV 和 JSON 文件。让我们从将行星数据从页面加载到 Python 字典对象列表中开始。以下代码（在（`03/get_planet_data.py`）中找到）提供了执行此任务的函数，该函数将在整个章节中重复使用：

```py
import requests
from bs4 import BeautifulSoup

def get_planet_data():
   html = requests.get("http://localhost:8080/planets.html").text
   soup = BeautifulSoup(html, "lxml")

   planet_trs = soup.html.body.div.table.findAll("tr", {"class": "planet"})

   def to_dict(tr):
      tds = tr.findAll("td")
      planet_data = dict()
      planet_data['Name'] = tds[1].text.strip()
      planet_data['Mass'] = tds[2].text.strip()
      planet_data['Radius'] = tds[3].text.strip()
      planet_data['Description'] = tds[4].text.strip()
      planet_data['MoreInfo'] = tds[5].findAll("a")[0]["href"].strip()
      return planet_data

   planets = [to_dict(tr) for tr in planet_trs]

   return planets

if __name__ == "__main__":
   print(get_planet_data())
```

运行脚本会产生以下输出（简要截断）：

```py
03 $python get_planet_data.py
[{'Name': 'Mercury', 'Mass': '0.330', 'Radius': '4879', 'Description': 'Named Mercurius by the Romans because it appears to move so swiftly.', 'MoreInfo': 'https://en.wikipedia.org/wiki/Mercury_(planet)'}, {'Name': 'Venus', 'Mass': '4.87', 'Radius': '12104', 'Description': 'Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the\r\n heavens. Other civilizations have named it for their god or goddess of love/war.', 'MoreInfo': 'https://en.wikipedia.org/wiki/Venus'}, {'Name': 'Earth', 'Mass': '5.97', 'Radius': '12756', 'Description': "The name Earth comes from the Indo-European base 'er,'which produced the Germanic noun 'ertho,' and ultimately German 'erde,'\r\n Dutch 'aarde,' Scandinavian 'jord,' and English 'earth.' Related forms include Greek 'eraze,' meaning\r\n 'on the ground,' and Welsh 'erw,' meaning 'a piece of land.'", 'MoreInfo': 'https://en.wikipedia.org/wiki/Earth'}, {'Name': 'Mars', 'Mass': '0.642', 'Radius': '6792', 'Description': 'Named by the Romans for their god of war because of its red, bloodlike color. Other civilizations also named this planet\r\n from this attribute; for example, the Egyptians named it "Her Desher," meaning "the red one."', 'MoreInfo':
...
```

可能需要安装 csv、json 和 pandas。您可以使用以下三个命令来完成：

```py
pip install csv
pip install json
pip install pandas
```

# 如何做

我们将首先将行星数据转换为 CSV 文件。

1.  这将使用`csv`执行。以下代码将行星数据写入 CSV 文件（代码在`03/create_csv.py`中）：

```py
import csv
from get_planet_data import get_planet_data

planets = get_planet_data()

with open('../../www/planets.csv', 'w+', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Name', 'Mass', 'Radius', 'Description', 'MoreInfo'])
for planet in planets:
        writer.writerow([planet['Name'], planet['Mass'],planet['Radius'], planet['Description'], planet['MoreInfo']])

```

1.  输出文件放入我们项目的 www 文件夹中。检查它，我们看到以下内容：

```py
Name,Mass,Radius,Description,MoreInfo
Mercury,0.330,4879,Named Mercurius by the Romans because it appears to move so swiftly.,https://en.wikipedia.org/wiki/Mercury_(planet)
Venus,4.87,12104,Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the heavens. Other civilizations have named it for their god or goddess of love/war.,https://en.wikipedia.org/wiki/Venus
Earth,5.97,12756,"The name Earth comes from the Indo-European base 'er,'which produced the Germanic noun 'ertho,' and ultimately German 'erde,' Dutch 'aarde,' Scandinavian 'jord,' and English 'earth.' Related forms include Greek 'eraze,' meaning 'on the ground,' and Welsh 'erw,' meaning 'a piece of land.'",https://en.wikipedia.org/wiki/Earth
Mars,0.642,6792,"Named by the Romans for their god of war because of its red, bloodlike color. Other civilizations also named this planet from this attribute; for example, the Egyptians named it ""Her Desher,"" meaning ""the red one.""",https://en.wikipedia.org/wiki/Mars
Jupiter,1898,142984,The largest and most massive of the planets was named Zeus by the Greeks and Jupiter by the Romans; he was the most important deity in both pantheons.,https://en.wikipedia.org/wiki/Jupiter
Saturn,568,120536,"Roman name for the Greek Cronos, father of Zeus/Jupiter. Other civilizations have given different names to Saturn, which is the farthest planet from Earth that can be observed by the naked human eye. Most of its satellites were named for Titans who, according to Greek mythology, were brothers and sisters of Saturn.",https://en.wikipedia.org/wiki/Saturn
Uranus,86.8,51118,"Several astronomers, including Flamsteed and Le Monnier, had observed Uranus earlier but had recorded it as a fixed star. Herschel tried unsuccessfully to name his discovery ""Georgian Sidus"" after George III; the planet was named by Johann Bode in 1781 after the ancient Greek deity of the sky Uranus, the father of Kronos (Saturn) and grandfather of Zeus (Jupiter).",https://en.wikipedia.org/wiki/Uranus
Neptune,102,49528,"Neptune was ""predicted"" by John Couch Adams and Urbain Le Verrier who, independently, were able to account for the irregularities in the motion of Uranus by correctly predicting the orbital elements of a trans- Uranian body. Using the predicted parameters of Le Verrier (Adams never published his predictions), Johann Galle observed the planet in 1846\. Galle wanted to name the planet for Le Verrier, but that was not acceptable to the international astronomical community. Instead, this planet is named for the Roman god of the sea.",https://en.wikipedia.org/wiki/Neptune
Pluto,0.0146,2370,"Pluto was discovered at Lowell Observatory in Flagstaff, AZ during a systematic search for a trans-Neptune planet predicted by Percival Lowell and William H. Pickering. Named after the Roman god of the underworld who was able to render himself invisible.",https://en.wikipedia.org/wiki/Pluto
```

我们将这个文件写入 www 目录，以便我们可以通过我们的 Web 服务器下载它。

1.  现在可以在支持 CSV 内容的应用程序中使用这些数据，例如 Excel：

![](img/a00f3815-56b8-4bfb-bcd7-e9dbd035caa9.png)在 Excel 中打开的文件

1.  还可以使用`csv`库从 Web 服务器读取 CSV 数据，并首先使用`requests`检索内容。以下代码在`03/read_csv_from_web.py`中：

```py
import requests
import csv

planets_data = requests.get("http://localhost:8080/planets.csv").text
planets = planets_data.split('\n')
reader = csv.reader(planets, delimiter=',', quotechar='"')
lines = [line for line in reader][:-1]
for line in lines: print(line)
```

以下是部分输出

```py
['Name', 'Mass', 'Radius', 'Description', 'MoreInfo']
['Mercury', '0.330', '4879', 'Named Mercurius by the Romans because it appears to move so swiftly.', 'https://en.wikipedia.org/wiki/Mercury_(planet)']
['Venus', '4.87', '12104', 'Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the heavens. Other civilizations have named it for their god or goddess of love/war.', 'https://en.wikipedia.org/wiki/Venus']
['Earth', '5.97', '12756', "The name Earth comes from the Indo-European base 'er,'which produced the Germanic noun 'ertho,' and ultimately German 'erde,' Dutch 'aarde,' Scandinavian 'jord,' and English 'earth.' Related forms include Greek 'eraze,' meaning 'on the ground,' and Welsh 'erw,' meaning 'a piece of land.'", 'https://en.wikipedia.org/wiki/Earth']
```

有一点要指出的是，CSV 写入器留下了一个尾随空白，如果不处理，就会添加一个空列表项。这是通过切片行来处理的：以下语句返回除最后一行之外的所有行：

`lines = [line for line in reader][:-1]`

1.  这也可以很容易地使用 pandas 完成。以下从抓取的数据构造一个 DataFrame。代码在`03/create_df_planets.py`中：

```py
import pandas as pd
planets_df = pd.read_csv("http://localhost:8080/planets_pandas.csv", index_col='Name')
print(planets_df)
```

运行此命令将产生以下输出：

```py
                                               Description Mass Radius
Name 
Mercury Named Mercurius by the Romans because it appea...  0.330 4879
Venus   Roman name for the goddess of love. This plane...   4.87 12104
Earth   The name Earth comes from the Indo-European ba...   5.97 12756
Mars    Named by the Romans for their god of war becau...  0.642 6792
Jupiter The largest and most massive of the planets wa...   1898 142984
Saturn  Roman name for the Greek Cronos, father of Zeu...    568 120536
Uranus  Several astronomers, including Flamsteed and L...   86.8 51118
Neptune Neptune was "predicted" by John Couch Adams an...    102 49528
Pluto   Pluto was discovered at Lowell Observatory in ... 0.0146 2370
```

1.  `DataFrame`也可以通过简单调用`.to_csv()`保存到 CSV 文件中（代码在`03/save_csv_pandas.py`中）：

```py
import pandas as pd
from get_planet_data import get_planet_data

# construct a data from from the list planets = get_planet_data()
planets_df = pd.DataFrame(planets).set_index('Name')
planets_df.to_csv("../../www/planets_pandas.csv")
```

1.  可以使用`pd.read_csv()`非常轻松地从`URL`中读取 CSV 文件，无需其他库。您可以使用`03/read_csv_via_pandas.py`中的代码：

```py
import pandas as pd
planets_df = pd.read_csv("http://localhost:8080/planets_pandas.csv", index_col='Name')
print(planets_df)
```

1.  将数据转换为 JSON 也非常容易。使用 Python 可以使用 Python 的`json`库对 JSON 进行操作。该库可用于将 Python 对象转换为 JSON，也可以从 JSON 转换为 Python 对象。以下将行星列表转换为 JSON 并将其打印到控制台：将行星数据打印为 JSON（代码在`03/convert_to_json.py`中）：

```py
import json
from get_planet_data import get_planet_data
planets=get_planet_data()
print(json.dumps(planets, indent=4))
```

执行此脚本将产生以下输出（省略了部分输出）：

```py
[
    {
        "Name": "Mercury",
        "Mass": "0.330",
        "Radius": "4879",
        "Description": "Named Mercurius by the Romans because it appears to move so swiftly.",
        "MoreInfo": "https://en.wikipedia.org/wiki/Mercury_(planet)"
    },
    {
        "Name": "Venus",
        "Mass": "4.87",
        "Radius": "12104",
        "Description": "Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the heavens. Other civilizations have named it for their god or goddess of love/war.",
        "MoreInfo": "https://en.wikipedia.org/wiki/Venus"
    },
```

1.  这也可以用于轻松地将 JSON 保存到文件（`03/save_as_json.py`）：

```py
import json
from get_planet_data import get_planet_data
planets=get_planet_data()
with open('../../www/planets.json', 'w+') as jsonFile:
   json.dump(planets, jsonFile, indent=4)
```

1.  使用`!head -n 13 ../../www/planets.json`检查输出，显示：

```py
[
    {
        "Name": "Mercury",
        "Mass": "0.330",
        "Radius": "4879",
        "Description": "Named Mercurius by the Romans because it appears to move so swiftly.",
        "MoreInfo": "https://en.wikipedia.org/wiki/Mercury_(planet)"
    },
    {
        "Name": "Venus",
        "Mass": "4.87",
        "Radius": "12104",
        "Description": "Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the heavens. Other civilizations have named it for their god or goddess of love/war.",
```

1.  可以使用`requests`从 Web 服务器读取 JSON 并将其转换为 Python 对象（`03/read_http_json_requests.py`）：

```py
import requests
import json

planets_request = requests.get("http://localhost:8080/planets.json")
print(json.loads(planets_request.text))
```

1.  pandas 还提供了将 JSON 保存为 CSV 的功能（`03/save_json_pandas.py`）：

```py
import pandas as pd
from get_planet_data import get_planet_data

planets = get_planet_data()
planets_df = pd.DataFrame(planets).set_index('Name')
planets_df.reset_index().to_json("../../www/planets_pandas.json", orient='records')
```

不幸的是，目前还没有一种方法可以漂亮地打印从`.to_json()`输出的 JSON。还要注意使用`orient='records'`和使用`rest_index()`。这对于复制与使用 JSON 库示例写入的相同 JSON 结构是必要的。

1.  可以使用`.read_json()`将 JSON 读入 DataFrame，也可以从 HTTP 和文件中读取（`03/read_json_http_pandas.py`）：

```py
import pandas as pd
planets_df = pd.read_json("http://localhost:8080/planets_pandas.json").set_index('Name')
print(planets_df)
```

# 工作原理

`csv`和`json`库是 Python 的标准部分，提供了一种简单的方法来读取和写入这两种格式的数据。

在某些 Python 发行版中，pandas 并不是标准配置，您可能需要安装它。pandas 对 CSV 和 JSON 的功能也更高级，提供了许多强大的数据操作，还支持从远程服务器访问数据。

# 还有更多...

选择 csv、json 或 pandas 库由您决定，但我倾向于喜欢 pandas，并且我们将在整本书中更多地研究其在抓取中的使用，尽管我们不会深入研究其用法。

要深入了解 pandas，请查看`pandas.pydata.org`，或者阅读我在 Packt 出版的另一本书《Learning pandas, 2ed》。

有关 csv 库的更多信息，请参阅[`docs.python.org/3/library/csv.html`](https://docs.python.org/3/library/csv.html)

有关 json 库的更多信息，请参阅[`docs.python.org/3/library/json.html`](https://docs.python.org/3/library/json.html)

# 使用 AWS S3 存储数据

有许多情况下，我们只想将我们抓取的内容保存到本地副本以进行存档、备份或以后进行批量分析。我们还可能希望保存这些网站的媒体以供以后使用。我为广告合规公司构建了爬虫，我们会跟踪并下载网站上基于广告的媒体，以确保正确使用，并且以供以后分析、合规和转码。

这些类型系统所需的存储空间可能是巨大的，但随着云存储服务（如 AWS S3（简单存储服务））的出现，这比在您自己的 IT 部门中管理大型 SAN（存储区域网络）要容易得多，成本也更低。此外，S3 还可以自动将数据从热存储移动到冷存储，然后再移动到长期存储，例如冰川，这可以为您节省更多的钱。

我们不会深入研究所有这些细节，而只是看看如何将我们的`planets.html`文件存储到 S3 存储桶中。一旦您能做到这一点，您就可以保存任何您想要的内容。

# 准备就绪

要执行以下示例，您需要一个 AWS 账户，并且可以访问用于 Python 代码的密钥。它们将是您账户的唯一密钥。我们将使用`boto3`库来访问 S3。您可以使用`pip install boto3`来安装它。此外，您需要设置环境变量进行身份验证。它们看起来像下面这样：

`AWS_ACCESS_KEY_ID=AKIAIDCQ5PH3UMWKZEWA`

`AWS_SECRET_ACCESS_KEY=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`

这些可以在 AWS 门户的 IAM（身份访问管理）部分找到。

将这些密钥放在环境变量中是一个好习惯。在代码中使用它们可能会导致它们被盗。在编写本书时，我将它们硬编码并意外地将它们检入 GitHub。第二天早上，我醒来收到了来自 AWS 的关键消息，说我有成千上万台服务器在运行！GitHub 有爬虫在寻找这些密钥，它们会被找到并用于不正当目的。等我把它们全部关闭的时候，我的账单已经涨到了 6000 美元，全部是在一夜之间产生的。幸运的是，AWS 免除了这些费用！

# 如何做到这一点

我们不会解析`planets.html`文件中的数据，而只是使用 requests 从本地 web 服务器检索它：

1.  以下代码（在`03/S3.py`中找到）读取行星网页并将其存储在 S3 中：

```py
import requests
import boto3

data = requests.get("http://localhost:8080/planets.html").text

# create S3 client, use environment variables for keys s3 = boto3.client('s3')

# the bucket bucket_name = "planets-content"   # create bucket, set s3.create_bucket(Bucket=bucket_name, ACL='public-read')
s3.put_object(Bucket=bucket_name, Key='planet.html',
              Body=data, ACL="public-read")
```

1.  这个应用程序将给出类似以下的输出，这是 S3 信息，告诉您关于新项目的各种事实。

```py

{'ETag': '"3ada9dcd8933470221936534abbf7f3e"',
 'ResponseMetadata': {'HTTPHeaders': {'content-length': '0',
   'date': 'Sun, 27 Aug 2017 19:25:54 GMT',
   'etag': '"3ada9dcd8933470221936534abbf7f3e"',
   'server': 'AmazonS3',
   'x-amz-id-2': '57BkfScql637op1dIXqJ7TeTmMyjVPk07cAMNVqE7C8jKsb7nRO+0GSbkkLWUBWh81k+q2nMQnE=',
   'x-amz-request-id': 'D8446EDC6CBA4416'},
  'HTTPStatusCode': 200,
  'HostId': '57BkfScql637op1dIXqJ7TeTmMyjVPk07cAMNVqE7C8jKsb7nRO+0GSbkkLWUBWh81k+q2nMQnE=',
  'RequestId': 'D8446EDC6CBA4416',
  'RetryAttempts': 0}}
```

1.  这个输出告诉我们对象已成功创建在存储桶中。此时，您可以转到 S3 控制台并查看您的存储桶：

![](img/29fbd119-7ee5-43eb-8b2f-9bc34998ff53.png)S3 中的存储桶

1.  在存储桶中，您将看到`planet.html`文件：

![](img/49cc32c4-5ac3-4177-a397-35385afbcf4e.png)存储桶中的文件

1.  通过点击文件，您可以看到 S3 中文件的属性和 URL：

![](img/6c5b035d-009f-4878-9806-034b4db8e500.png)S3 中文件的属性

# 它是如何工作的

boto3 库以 Pythonic 语法封装了 AWS S3 API。`.client()`调用与 AWS 进行身份验证，并为我们提供了一个用于与 S3 通信的对象。确保您的密钥在环境变量中，否则这将无法工作。

存储桶名称必须是全局唯一的。在撰写本文时，这个存储桶是可用的，但您可能需要更改名称。`.create_bucket()`调用创建存储桶并设置其 ACL。`put_object()`使用`boto3`上传管理器将抓取的数据上传到存储桶中的对象。

# 还有更多...

有很多细节需要学习来使用 S3。您可以在以下网址找到 API 文档：[`docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html`](http://docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html)。Boto3 文档可以在以下网址找到：[`boto3.readthedocs.io/en/latest/`](https://boto3.readthedocs.io/en/latest/)。

虽然我们只保存了一个网页，但这个模型可以用来在 S3 中存储任何类型的基于文件的数据。

# 使用 MySQL 存储数据

MySQL 是一个免费的、开源的关系数据库管理系统（RDBMS）。在这个例子中，我们将从网站读取行星数据并将其存储到 MySQL 数据库中。

# 准备工作

您需要访问一个 MySQL 数据库。您可以在本地安装一个，也可以在云中安装，也可以在容器中安装。我正在使用本地安装的 MySQL 服务器，并且将`root`密码设置为`mypassword`。您还需要安装 MySQL python 库。您可以使用`pip install mysql-connector-python`来安装它。

1.  首先要做的是使用终端上的`mysql`命令连接到数据库：

```py
# mysql -uroot -pmypassword
mysql: [Warning] Using a password on the command line interface can be insecure.
Welcome to the MySQL monitor. Commands end with ; or \g.
Your MySQL connection id is 4
Server version: 5.7.19 MySQL Community Server (GPL)

Copyright (c) 2000, 2017, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>
```

1.  现在我们可以创建一个数据库，用来存储我们抓取的信息：

```py
mysql> create database scraping;
Query OK, 1 row affected (0.00 sec)
```

1.  现在使用新的数据库：

```py
mysql> use scraping;
Database changed
```

1.  并在数据库中创建一个行星表来存储我们的数据：

```py

mysql> CREATE TABLE `scraping`.`planets` (
 `id` INT NOT NULL AUTO_INCREMENT,
 `name` VARCHAR(45) NOT NULL,
 `mass` FLOAT NOT NULL,
 `radius` FLOAT NOT NULL,
 `description` VARCHAR(5000) NULL,
 PRIMARY KEY (`id`));
Query OK, 0 rows affected (0.02 sec)

```

现在我们准备好抓取数据并将其放入 MySQL 数据库中。

# 如何做到这一点

1.  以下代码（在`03/store_in_mysql.py`中找到）将读取行星数据并将其写入 MySQL：

```py
import mysql.connector
import get_planet_data
from mysql.connector import errorcode
from get_planet_data import get_planet_data

try:
    # open the database connection
    cnx = mysql.connector.connect(user='root', password='mypassword',
                                  host="127.0.0.1", database="scraping")

    insert_sql = ("INSERT INTO Planets (Name, Mass, Radius, Description) " +
                  "VALUES (%(Name)s, %(Mass)s, %(Radius)s, %(Description)s)")

    # get the planet data
    planet_data = get_planet_data()

    # loop through all planets executing INSERT for each with the cursor
    cursor = cnx.cursor()
    for planet in planet_data:
        print("Storing data for %s" % (planet["Name"]))
        cursor.execute(insert_sql, planet)

    # commit the new records
    cnx.commit()

    # close the cursor and connection
    cursor.close()
    cnx.close()

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
else:
    cnx.close()
```

1.  这将产生以下输出：

```py
Storing data for Mercury
Storing data for Venus
Storing data for Earth
Storing data for Mars
Storing data for Jupiter
Storing data for Saturn
Storing data for Uranus
Storing data for Neptune
Storing data for Pluto
```

1.  使用 MySQL Workbench，我们可以看到记录已写入数据库（您也可以使用 mysql 命令行）：

![](img/c8a2c090-dce7-40f2-b0b3-0c6d72ff3885.png)使用 MySQL Workbench 显示的记录

1.  以下代码可用于检索数据（`03/read_from_mysql.py`）：

```py
import mysql.connector
from mysql.connector import errorcode

try:
  cnx = mysql.connector.connect(user='root', password='mypassword',
                  host="127.0.0.1", database="scraping")
  cursor = cnx.cursor(dictionary=False)

  cursor.execute("SELECT * FROM scraping.Planets")
  for row in cursor:
    print(row)

  # close the cursor and connection
  cursor.close()
  cnx.close()

except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
finally:
  cnx.close()

```

1.  这将产生以下输出：

```py
(1, 'Mercury', 0.33, 4879.0, 'Named Mercurius by the Romans because it appears to move so swiftly.', 'https://en.wikipedia.org/wiki/Mercury_(planet)')
(2, 'Venus', 4.87, 12104.0, 'Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the heavens. Other civilizations have named it for their god or goddess of love/war.', 'https://en.wikipedia.org/wiki/Venus')
(3, 'Earth', 5.97, 12756.0, "The name Earth comes from the Indo-European base 'er,'which produced the Germanic noun 'ertho,' and ultimately German 'erde,' Dutch 'aarde,' Scandinavian 'jord,' and English 'earth.' Related forms include Greek 'eraze,' meaning 'on the ground,' and Welsh 'erw,' meaning 'a piece of land.'", 'https://en.wikipedia.org/wiki/Earth')
(4, 'Mars', 0.642, 6792.0, 'Named by the Romans for their god of war because of its red, bloodlike color. Other civilizations also named this planet from this attribute; for example, the Egyptians named it "Her Desher," meaning "the red one."', 'https://en.wikipedia.org/wiki/Mars')
(5, 'Jupiter', 1898.0, 142984.0, 'The largest and most massive of the planets was named Zeus by the Greeks and Jupiter by the Romans; he was the most important deity in both pantheons.', 'https://en.wikipedia.org/wiki/Jupiter')
(6, 'Saturn', 568.0, 120536.0, 'Roman name for the Greek Cronos, father of Zeus/Jupiter. Other civilizations have given different names to Saturn, which is the farthest planet from Earth that can be observed by the naked human eye. Most of its satellites were named for Titans who, according to Greek mythology, were brothers and sisters of Saturn.', 'https://en.wikipedia.org/wiki/Saturn')
(7, 'Uranus', 86.8, 51118.0, 'Several astronomers, including Flamsteed and Le Monnier, had observed Uranus earlier but had recorded it as a fixed star. Herschel tried unsuccessfully to name his discovery "Georgian Sidus" after George III; the planet was named by Johann Bode in 1781 after the ancient Greek deity of the sky Uranus, the father of Kronos (Saturn) and grandfather of Zeus (Jupiter).', 'https://en.wikipedia.org/wiki/Uranus')
(8, 'Neptune', 102.0, 49528.0, 'Neptune was "predicted" by John Couch Adams and Urbain Le Verrier who, independently, were able to account for the irregularities in the motion of Uranus by correctly predicting the orbital elements of a trans- Uranian body. Using the predicted parameters of Le Verrier (Adams never published his predictions), Johann Galle observed the planet in 1846\. Galle wanted to name the planet for Le Verrier, but that was not acceptable to the international astronomical community. Instead, this planet is named for the Roman god of the sea.', 'https://en.wikipedia.org/wiki/Neptune')
(9, 'Pluto', 0.0146, 2370.0, 'Pluto was discovered at Lowell Observatory in Flagstaff, AZ during a systematic search for a trans-Neptune planet predicted by Percival Lowell and William H. Pickering. Named after the Roman god of the underworld who was able to render himself invisible.', 'https://en.wikipedia.org/wiki/Pluto')
```

# 工作原理

使用`mysql.connector`访问 MySQL 数据库涉及使用库中的两个类：`connect`和`cursor`。`connect`类打开并管理与数据库服务器的连接。从该连接对象，我们可以创建一个光标对象。该光标用于使用 SQL 语句读取和写入数据。

在第一个例子中，我们使用光标将九条记录插入数据库。直到调用连接的`commit()`方法，这些记录才会被写入数据库。这将执行将所有行写入数据库的操作。

读取数据使用类似的模型，只是我们使用光标执行 SQL 查询（`SELECT`），并遍历检索到的行。由于我们是在读取而不是写入，因此无需在连接上调用`commit()`。

# 还有更多...

您可以从以下网址了解更多关于 MySQL 并安装它：`https://dev.mysql.com/doc/refman/5.7/en/installing.html`。有关 MySQL Workbench 的信息，请访问：`https://dev.mysql.com/doc/workbench/en/`。

# 使用 PostgreSQL 存储数据

在这个示例中，我们将我们的行星数据存储在 PostgreSQL 中。PostgreSQL 是一个开源的关系数据库管理系统（RDBMS）。它由一个全球志愿者团队开发，不受任何公司或其他私人实体控制，源代码可以免费获得。它具有许多独特的功能，如分层数据模型。

# 准备工作

首先确保您可以访问 PostgreSQL 数据实例。同样，您可以在本地安装一个，运行一个容器，或者在云中获取一个实例。

与 MySQL 一样，我们需要首先创建一个数据库。该过程与 MySQL 几乎相同，但命令和参数略有不同。

1.  从终端执行终端上的 psql 命令。这将带您进入 psql 命令处理器：

```py
# psql -U postgres psql (9.6.4) Type "help" for help. postgres=# 
```

1.  现在创建抓取数据库：

```py
postgres=# create database scraping;
CREATE DATABASE
postgres=#
```

1.  然后切换到新数据库：

```py
postgres=# \connect scraping You are now connected to database "scraping" as user "postgres". scraping=# 
```

1.  现在我们可以创建 Planets 表。我们首先需要创建一个序列表：

```py
scraping=# CREATE SEQUENCE public."Planets_id_seq" scraping-#  INCREMENT 1 scraping-#  START 1 scraping-#  MINVALUE 1 scraping-#  MAXVALUE 9223372036854775807 scraping-#  CACHE 1; CREATE SEQUENCE scraping=# ALTER SEQUENCE public."Planets_id_seq" scraping-#  OWNER TO postgres; ALTER SEQUENCE scraping=# 
```

1.  现在我们可以创建表：

```py
scraping=# CREATE TABLE public."Planets" scraping-# ( scraping(# id integer NOT NULL DEFAULT nextval('"Planets_id_seq"'::regclass), scraping(# name text COLLATE pg_catalog."default" NOT NULL, scraping(# mass double precision NOT NULL, scraping(# radius double precision NOT NULL, scraping(# description text COLLATE pg_catalog."default" NOT NULL, scraping(# moreinfo text COLLATE pg_catalog."default" NOT NULL, scraping(# CONSTRAINT "Planets_pkey" PRIMARY KEY (name) scraping(# ) scraping-# WITH ( scraping(# OIDS = FALSE scraping(# )
</span>scraping-# TABLESPACE pg_default; CREATE TABLE scraping=# scraping=# ALTER TABLE public."Planets" scraping-# OWNER to postgres; ALTER TABLE scraping=# \q
```

要从 Python 访问 PostgreSQL，我们将使用`psycopg2`库，因此请确保在 Python 环境中安装了它，使用`pip install psycopg2`。

我们现在准备好编写 Python 将行星数据存储在 PostgreSQL 中。

# 如何操作

我们按照以下步骤进行：

1.  以下代码将读取行星数据并将其写入数据库（代码在`03/save_in_postgres.py`中）：

```py
import psycopg2
from get_planet_data import get_planet_data

try:
  # connect to PostgreSQL
  conn = psycopg2.connect("dbname='scraping' host='localhost' user='postgres' password='mypassword'")

  # the SQL INSERT statement we will use
  insert_sql = ('INSERT INTO public."Planets"(name, mass, radius, description, moreinfo) ' +
          'VALUES (%(Name)s, %(Mass)s, %(Radius)s, %(Description)s, %(MoreInfo)s);')

  # open a cursor to access data
  cur = conn.cursor()

  # get the planets data and loop through each
  planet_data = get_planet_data()
  for planet in planet_data:
    # write each record
    cur.execute(insert_sql, planet)

  # commit the new records to the database
  conn.commit()
  cur.close()
  conn.close()

  print("Successfully wrote data to the database")

except Exception as ex:
  print(ex)

```

1.  如果成功，您将看到以下内容：

```py
Successfully wrote data to the database
```

1.  使用诸如 pgAdmin 之类的 GUI 工具，您可以检查数据库中的数据：

![](img/e1060188-c3d3-4a2d-aaf4-4f9124294d9e.png)在 pgAdmin 中显示的记录

1.  可以使用以下 Python 代码查询数据（在`03/read_from_postgresql.py`中找到）：

```py
import psycopg2

try:
  conn = psycopg2.connect("dbname='scraping' host='localhost' user='postgres' password='mypassword'")

  cur = conn.cursor()
  cur.execute('SELECT * from public."Planets"')
  rows = cur.fetchall()
  print(rows)

  cur.close()
  conn.close()

except Exception as ex:
  print(ex)

```

1.  并导致以下输出（略有截断：

```py
(1, 'Mercury', 0.33, 4879.0, 'Named Mercurius by the Romans because it appears to move so swiftly.', 'https://en.wikipedia.org/wiki/Mercury_(planet)'), (2, 'Venus', 4.87, 12104.0, 'Roman name for the goddess of love. This planet was considered to be the brightest and most beautiful planet or star in the heavens. Other civilizations have named it for their god or goddess of love/war.', 'https://en.wikipedia.org/wiki/Venus'), (3, 'Earth', 5.97, 12756.0, "The name Earth comes from the Indo-European base 'er,'which produced the Germanic noun 'ertho,' and ultimately German 'erde,' Dutch 'aarde,' Scandinavian 'jord,' and English 'earth.' Related forms include Greek 'eraze,' meaning 'on the ground,' and Welsh 'erw,' meaning 'a piece of land.'", 'https://en.wikipedia.org/wiki/Earth'), (4, 'Mars', 0.642, 6792.0, 'Named by the Romans for their god of war because of its red, bloodlike color. Other civilizations also named this planet from this attribute; for example, the Egyptians named it 
```

# 工作原理

使用`psycopg2`库访问 PostgreSQL 数据库涉及使用库中的两个类：`connect`和`cursor`。`connect`类打开并管理与数据库服务器的连接。从该连接对象，我们可以创建一个`cursor`对象。该光标用于使用 SQL 语句读取和写入数据。

在第一个例子中，我们使用光标将九条记录插入数据库。直到调用连接的`commit()`方法，这些记录才会被写入数据库。这将执行将所有行写入数据库的操作。

读取数据使用类似的模型，只是我们使用游标执行 SQL 查询（`SELECT`），并遍历检索到的行。由于我们是在读取而不是写入，所以不需要在连接上调用`commit()`。

# 还有更多...

有关 PostgreSQL 的信息可在`https://www.postgresql.org/`找到。pgAdmin 可以在`https://www.pgadmin.org/`获得。`psycopg`的参考资料位于`http://initd.org/psycopg/docs/usage.html`

# 在 Elasticsearch 中存储数据

Elasticsearch 是基于 Lucene 的搜索引擎。它提供了一个分布式、多租户能力的全文搜索引擎，具有 HTTP Web 界面和无模式的 JSON 文档。它是一个非关系型数据库（通常称为 NoSQL），专注于存储文档而不是记录。这些文档可以是许多格式之一，其中之一对我们有用：JSON。这使得使用 Elasticsearch 非常简单，因为我们不需要将我们的数据转换为/从 JSON。我们将在本书的后面更多地使用 Elasticsearch

现在，让我们去将我们的行星数据存储在 Elasticsearch 中。

# 准备就绪

我们将访问一个本地安装的 Elasticsearch 服务器。为此，我们将使用`Elasticsearch-py`库从 Python 中进行操作。您很可能需要使用 pip 来安装它：`pip install elasticsearch`。

与 PostgreSQL 和 MySQL 不同，我们不需要提前在 Elasticsearch 中创建表。Elasticsearch 不关心结构化数据模式（尽管它确实有索引），因此我们不必经历这个过程。

# 如何做到

将数据写入 Elasticsearch 非常简单。以下 Python 代码使用我们的行星数据执行此任务（`03/write_to_elasticsearch.py`）：

```py
from elasticsearch import Elasticsearch
from get_planet_data import get_planet_data

# create an elastic search object
es = Elasticsearch()

# get the data
planet_data = get_planet_data()

for planet in planet_data:
  # insert each planet into elasticsearch server
  res = es.index(index='planets', doc_type='planets_info', body=planet)
  print (res)
```

执行此操作将产生以下输出：

```py
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF3_T0Z2t9T850q6', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF5QT0Z2t9T850q7', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF5XT0Z2t9T850q8', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF5fT0Z2t9T850q9', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF5mT0Z2t9T850q-', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF5rT0Z2t9T850q_', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF50T0Z2t9T850rA', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF56T0Z2t9T850rB', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
{'_index': 'planets', '_type': 'planets_info', '_id': 'AV4qIF6AT0Z2t9T850rC', '_version': 1, 'result': 'created', '_shards': {'total': 2, 'successful': 1, 'failed': 0}, 'created': True}
```

输出显示了每次插入的结果，为我们提供了 elasticsearch 分配给文档的`_id`等信息。

如果您也安装了 logstash 和 kibana，您可以在 Kibana 内部看到数据：

![Kibana 显示和索引

我们可以使用以下 Python 代码查询数据。此代码检索“planets”索引中的所有文档，并打印每个行星的名称、质量和半径（`03/read_from_elasticsearch.py`）：

```py
from elasticsearch import Elasticsearch

# create an elastic search object
es = Elasticsearch()

res = es.search(index="planets", body={"query": {"match_all": {}}})

```

```py
print("Got %d Hits:" % res['hits']['total'])
for hit in res['hits']['hits']:
 print("%(Name)s %(Mass)s: %(Radius)s" % hit["_source"])Got 9 Hits:
```

这将产生以下输出：

```py
Mercury 0.330: 4879
Mars 0.642: 6792
Venus 4.87: 12104
Saturn 568: 120536
Pluto 0.0146: 2370
Earth 5.97: 12756
Uranus 86.8: 51118
Jupiter 1898: 142984
Neptune 102: 49528
```

# 它是如何工作的

Elasticsearch 既是 NoSQL 数据库又是搜索引擎。您将文档提供给 Elasticsearch，它会解析文档中的数据并自动为该数据创建搜索索引。

在插入过程中，我们使用了`elasticsearch`库的`.index()`方法，并指定了一个名为“planets”的索引，一个文档类型`planets_info`，最后是文档的主体，即我们的行星 Python 对象。`elasticsearch`库将该对象转换为 JSON 并将其发送到 Elasticsearch 进行存储和索引。

索引参数用于通知 Elasticsearch 如何创建索引，它将用于索引和我们在查询时可以用来指定要搜索的一组文档。当我们执行查询时，我们指定了相同的索引“planets”并执行了一个匹配所有文档的查询。

# 还有更多...

您可以在`https://www.elastic.co/products/elasticsearch`找到有关 elasticsearch 的更多信息。有关 python API 的信息可以在`http://pyelasticsearch.readthedocs.io/en/latest/api/`找到

我们还将在本书的后面章节回到 Elasticsearch。

# 如何使用 AWS SQS 构建强大的 ETL 管道

爬取大量站点和数据可能是一个复杂和缓慢的过程。但它可以充分利用并行处理，无论是在本地使用多个处理器线程，还是使用消息队列系统将爬取请求分发给报告爬虫。在类似于提取、转换和加载流水线（ETL）的过程中，可能还需要多个步骤。这些流水线也可以很容易地使用消息队列架构与爬取相结合来构建。

使用消息队列架构给我们的流水线带来了两个优势：

+   健壮性

+   可伸缩性

处理变得健壮，因为如果处理单个消息失败，那么消息可以重新排队进行处理。因此，如果爬虫失败，我们可以重新启动它，而不会丢失对页面进行爬取的请求，或者消息队列系统将把请求传递给另一个爬虫。

它提供了可伸缩性，因为在同一系统或不同系统上可以监听队列上的多个爬虫。然后，可以在不同的核心或更重要的是不同的系统上同时处理多个消息。在基于云的爬虫中，您可以根据需要扩展爬虫实例的数量以处理更大的负载。

可以使用的常见消息队列系统包括：Kafka、RabbitMQ 和 Amazon SQS。我们的示例将利用 Amazon SQS，尽管 Kafka 和 RabbitMQ 都非常适合使用（我们将在本书的后面看到 RabbitMQ 的使用）。我们使用 SQS 来保持使用 AWS 基于云的服务的模式，就像我们在本章早些时候使用 S3 一样。

# 准备就绪

例如，我们将构建一个非常简单的 ETL 过程，该过程将读取主行星页面并将行星数据存储在 MySQL 中。它还将针对页面中的每个*更多信息*链接传递单个消息到队列中，其中 0 个或多个进程可以接收这些请求，并对这些链接执行进一步处理。

要从 Python 访问 SQS，我们将重新使用`boto3`库。

# 如何操作-将消息发布到 AWS 队列

`03/create_messages.py`文件包含了读取行星数据并将 URL 发布到 SQS 队列的代码：

```py
from urllib.request import urlopen
from bs4 import BeautifulSoup

import boto3
import botocore

# declare our keys (normally, don't hard code this)
access_key="AKIAIXFTCYO7FEL55TCQ"
access_secret_key="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"

# create sqs client
sqs = boto3.client('sqs', "us-west-2",
                   aws_access_key_id = access_key, 
                   aws_secret_access_key = access_secret_key)

# create / open the SQS queue
queue = sqs.create_queue(QueueName="PlanetMoreInfo")
print (queue)

# read and parse the planets HTML
html = urlopen("http://127.0.0.1:8080/pages/planets.html")
bsobj = BeautifulSoup(html, "lxml")

planets = []
planet_rows = bsobj.html.body.div.table.findAll("tr", {"class": "planet"})

for i in planet_rows:
  tds = i.findAll("td")

  # get the URL
  more_info_url = tds[5].findAll("a")[0]["href"].strip()

  # send the URL to the queue
  sqs.send_message(QueueUrl=queue["QueueUrl"],
           MessageBody=more_info_url)
  print("Sent %s to %s" % (more_info_url, queue["QueueUrl"]))
```

在终端中运行代码，您将看到类似以下的输出：

```py
{'QueueUrl': 'https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo', 'ResponseMetadata': {'RequestId': '2aad7964-292a-5bf6-b838-2b7a5007af22', 'HTTPStatusCode': 200, 'HTTPHeaders': {'server': 'Server', 'date': 'Mon, 28 Aug 2017 20:02:53 GMT', 'content-type': 'text/xml', 'content-length': '336', 'connection': 'keep-alive', 'x-amzn-requestid': '2aad7964-292a-5bf6-b838-2b7a5007af22'}, 'RetryAttempts': 0}} Sent https://en.wikipedia.org/wiki/Mercury_(planet) to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Venus to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Earth to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Mars to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Jupiter to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Saturn to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Uranus to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Neptune to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Sent https://en.wikipedia.org/wiki/Pluto to https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo
```

现在进入 AWS SQS 控制台。您应该看到队列已经被创建，并且它包含 9 条消息：

![](img/2ad3b7c1-9f39-4d02-ac61-d23619a9c409.png)SQS 中的队列

# 工作原理

该代码连接到给定帐户和 AWS 的 us-west-2 地区。然后，如果队列不存在，则创建队列。然后，对于源内容中的每个行星，程序发送一个消息，该消息包含该行星的*更多信息* URL。

此时，没有人在监听队列，因此消息将一直保留在那里，直到最终被读取或它们过期。每条消息的默认生存期为 4 天。

# 如何操作-读取和处理消息

要处理消息，请运行`03/process_messages.py`程序：

```py
import boto3
import botocore
import requests
from bs4 import BeautifulSoup

print("Starting")

# declare our keys (normally, don't hard code this)
access_key = "AKIAIXFTCYO7FEL55TCQ"
access_secret_key = "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"

# create sqs client
sqs = boto3.client('sqs', "us-west-2", 
          aws_access_key_id = access_key, 
          aws_secret_access_key = access_secret_key)

print("Created client")

# create / open the SQS queue
queue = sqs.create_queue(QueueName="PlanetMoreInfo")
queue_url = queue["QueueUrl"]
print ("Opened queue: %s" % queue_url)

while True:
  print ("Attempting to receive messages")
  response = sqs.receive_message(QueueUrl=queue_url,
                 MaxNumberOfMessages=1,
                 WaitTimeSeconds=1)
  if not 'Messages' in response:
    print ("No messages")
    continue

  message = response['Messages'][0]
  receipt_handle = message['ReceiptHandle']
  url = message['Body']

  # parse the page
  html = requests.get(url)
  bsobj = BeautifulSoup(html.text, "lxml")

  # now find the planet name and albedo info
  planet=bsobj.findAll("h1", {"id": "firstHeading"} )[0].text
  albedo_node = bsobj.findAll("a", {"href": "/wiki/Geometric_albedo"})[0]
  root_albedo = albedo_node.parent
  albedo = root_albedo.text.strip()

  # delete the message from the queue
  sqs.delete_message(
    QueueUrl=queue_url,
    ReceiptHandle=receipt_handle
  )

  # print the planets name and albedo info
  print("%s: %s" % (planet, albedo))
```

使用`python process_messages.py`运行脚本。您将看到类似以下的输出：

```py
Starting Created client Opened queue: https://us-west-2.queue.amazonaws.com/414704166289/PlanetMoreInfo Attempting to receive messages Jupiter: 0.343 (Bond) 0.52 (geom.)[3] Attempting to receive messages Mercury (planet): 0.142 (geom.)[10] Attempting to receive messages Uranus: 0.300 (Bond) 0.51 (geom.)[5] Attempting to receive messages Neptune: 0.290 (bond) 0.41 (geom.)[4] Attempting to receive messages Pluto: 0.49 to 0.66 (geometric, varies by 35%)[1][7] Attempting to receive messages Venus: 0.689 (geometric)[2] Attempting to receive messages Earth: 0.367 geometric[3] Attempting to receive messages Mars: 0.170 (geometric)[8] 0.25 (Bond)[7] Attempting to receive messages Saturn: 0.499 (geometric)[4] Attempting to receive messages No messages
```

# 工作原理

程序连接到 SQS 并打开队列。打开队列以进行读取也是使用`sqs.create_queue`完成的，如果队列已经存在，它将简单地返回队列。

然后，它进入一个循环调用`sqs.receive_message`，指定队列的 URL，每次读取消息的数量，以及如果没有消息可用时等待的最长时间（以秒为单位）。

如果读取了一条消息，将检索消息中的 URL，并使用爬取技术读取 URL 的页面并提取行星的名称和有关其反照率的信息。

请注意，我们会检索消息的接收处理。这是删除队列中的消息所必需的。如果我们不删除消息，它将在一段时间后重新出现在队列中。因此，如果我们的爬虫崩溃并且没有执行此确认，消息将由 SQS 再次提供给另一个爬虫进行处理（或者在其恢复正常时由相同的爬虫处理）。

# 还有更多...

您可以在以下网址找到有关 S3 的更多信息：`https://aws.amazon.com/s3/`。有关 API 详细信息的具体内容，请访问：`https://aws.amazon.com/documentation/s3/`。
