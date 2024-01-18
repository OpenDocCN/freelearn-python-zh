# 通知无服务器应用程序

在本章中，我们将探索 AWS Lambda 函数和 AWS API Gateway。AWS Lambda 使我们能够创建无服务器函数。*无服务器*并不意味着没有服务器；实际上，它意味着这些函数不需要 DevOps 开销，如果您在运行应用程序，例如在 EC2 实例上，就会有。

无服务器架构并非银弹或解决所有问题的方案，但有许多优势，例如定价、几乎不需要 DevOps 以及对不同编程语言的支持。

在 Python 的情况下，像 Zappa 和由亚马逊开发的 AWS Chalice 微框架等工具使创建和部署无服务器函数变得非常容易。

在本章中，您将学习如何：

+   使用 Flask 框架创建服务

+   安装和配置 AWS CLI

+   使用 CLI 创建 S3 存储桶并上传文件

+   安装和配置 Zappa

+   使用 Zappa 部署应用程序

因此，话不多说，让我们马上开始吧！

# 设置环境

让我们首先创建一个文件夹，我们将在其中放置应用程序文件。首先，创建一个名为`notifier`的目录，并进入该目录，以便我们可以创建虚拟环境：

```py
mkdir notifier && cd notifier
```

我们使用`pipenv`创建虚拟环境：

```py
pipenv --python ~/Installs/Python3.6/bin/python3.6
```

请记住，如果 Python 3 在我们的`path`中，您可以简单地调用：

```py
pipenv --three
```

为构建此服务，我们将使用微型 Web 框架 Flask，因此让我们安装它：

```py
pipenv install flask
```

我们还将安装 requests 包，该包在发送请求到订单服务时将被使用：

```py
pipenv install requests
```

现在应该是我们目前所需的一切。接下来，我们将看到如何使用 AWS 简单邮件服务从我们的应用程序发送电子邮件。

# 设置 Amazon Web Services CLI

我们还需要安装 AWS 命令行界面，这将在部署无服务器函数和创建 S3 存储桶时节省大量时间。

安装非常简单，可以通过`pip`安装，AWS CLI 支持 Python 2 和 Python 3，并在不同的操作系统上运行，例如 Linux、macOS 和 Windows。

打开终端并输入以下命令：

```py
pip install awscli --upgrade --user
```

升级选项将告诉 pip 更新所有已安装的要求，`--user`选项意味着 pip 将在我们的本地目录中安装 AWS CLI，因此它不会触及系统全局安装的任何库。在 Linux 系统上，使用`--user`选项安装 Python 包时，该包将安装在`.local/bin`目录中，因此请确保您的`path`中包含该目录。

只需验证安装是否正常工作，输入以下命令：

```py
aws --version
```

您应该看到类似于此的输出：

```py
aws-cli/1.14.30 Python/3.6.2 Linux/4.9.0-3-amd64 botocore/1.8.34
```

在这里，您可以看到 AWS CLI 版本，以及操作系统版本、Python 版本以及当前使用的`botocore`版本。`botocore`是 AWS CLI 使用的核心库。此外，boto 是 Python 的 SDK，允许开发人员编写与 Amazon 服务（如 EC2 和 S3）一起工作的软件。

现在我们需要配置 CLI，并且需要准备一些信息。首先，我们需要`aws_access_key_id`和`aws_secret_access_key`，以及您首选的区域和输出。最常见的值，输出选项，是 JSON。

要创建访问密钥，点击 AWS 控制台页面右上角的用户名下拉菜单，并选择“我的安全凭证”。您将进入此页面：

![](img/7ed0edf2-f265-4321-9cd2-a78f41501ae2.png)

在这里，您可以配置不同的帐户安全设置，例如更改密码或启用多因素身份验证，但您现在应该选择的是访问密钥（访问密钥 ID 和秘密访问密钥）。然后点击“创建新的访问密钥”，将打开一个对话框显示您的密钥。您还将有下载密钥的选项。我建议您下载并将其保存在安全的地方。

在这里[`docs.aws.amazon.com/general/latest/gr/rande.html`](https://docs.aws.amazon.com/general/latest/gr/rande.html)查看 AWS 区域和端点。

现在我们可以`配置`CLI。在命令行中输入：

```py
aws configure
```

您将被要求提供访问密钥、秘密访问密钥、区域和默认输出格式。

# 配置简单电子邮件服务

亚马逊已经有一个名为简单电子邮件服务的服务，我们可以使用它来通过我们的应用程序发送电子邮件。我们将在沙箱模式下运行该服务，这意味着我们还可以向经过验证的电子邮件地址发送电子邮件。如果您计划在生产中使用该服务，可以更改此设置，但是对于本书的目的，只需在沙箱模式下运行即可。

如果您计划在生产环境中运行此应用程序，并希望退出亚马逊 SES 沙箱，您可以轻松地提交支持案例以增加电子邮件限制。要发送请求，您可以转到 SES 主页，在左侧菜单下的“电子邮件发送”部分，您将找到“专用 IP”链接。在那里，您将找到更多信息，还有一个链接，您可以申请增加电子邮件限制。

要使其工作，我们需要有两个可用的电子邮件帐户。在我的情况下，我有自己的域。我还创建了两个电子邮件帐户——`donotreply@dfurtado.com`，这将是我用来发送电子邮件的电子邮件，以及`pythonblueprints@dfurtado.com`，这是将接收电子邮件的电子邮件。在线（视频）游戏商店应用程序中的用户将使用此电子邮件地址，我们将下订单以便稍后测试通知。

# 注册电子邮件

所以让我们开始添加电子邮件。首先我们将注册`donotreply@dfurtado.com`。登录到 AWS 控制台，在搜索栏中搜索“简单电子邮件服务”。在左侧，您将看到一些选项。在身份管理下，单击“电子邮件地址”。您将看到如下屏幕：

![](img/4b285c53-779c-4a7a-8775-d9a40cbddf2b.png)

如您所见，列表为空，因此让我们继续添加两封电子邮件。单击“验证新电子邮件地址”，将出现一个对话框，您可以在其中输入电子邮件地址。只需输入您希望使用的电子邮件，然后单击“验证此电子邮件地址”按钮。通过这样做，将向您指定的电子邮件地址发送验证电子邮件，在其中您将找到验证链接。

对第二个电子邮件重复相同的步骤，该电子邮件将接收消息。

现在，再次转到左侧菜单，单击“电子邮件发送”下的 SMTP 设置。

在那里，您将看到发送电子邮件所需的所有配置，您还需要创建 SMTP 凭据。因此，单击“创建我的 SMTP 凭据”按钮，将打开一个新页面，您可以在其中输入您希望的 IAM 用户名。在我的情况下，我添加了“python-blueprints”。完成后，单击“创建”按钮。凭据创建后，您将看到一个页面，您可以在其中看到 SMTP 用户名和密码。如果愿意，您可以选择下载这些凭据。

# 创建 S3 存储桶

为了向用户发送模板电子邮件，我们需要将我们的模板复制到 S3 存储桶中。我们可以通过网络轻松完成这项工作，或者您可以使用我们刚刚安装的 AWS CLI。在`es-west-2`区域创建 S3 存储桶的命令类似于：

```py
aws s3api create-bucket \
--bucket python-blueprints \
--region eu-west-2 \
--create-bucket-configuration LocationConstraint=eu-west-2
```

在这里，我们使用命令`s3api`，它将为我们提供与 AWS S3 服务交互的不同子命令。我们调用子命令`create-bucket`，正如其名称所示，将创建一个新的 S3 存储桶。对于这个子命令，我们指定了三个参数。首先是`--bucket`，指定 S3 存储桶的名称，然后是`--region`，指定要在哪个区域创建存储桶 - 在这种情况下，我们将在`eu-west-2`创建存储桶。最后，区域外的位置需要设置`LocationConstraint`，以便在我们希望的区域创建存储桶。

# 实现通知服务

现在我们已经准备好了一切，我们将要用作向在线（视频）游戏商店的客户发送电子邮件的模板文件已经放在了`python-blueprints` S3 存储桶中，现在是时候开始实现通知服务了。

让我们继续在`notifier`目录中创建一个名为`app.py`的文件，首先让我们添加一些导入：

```py
import smtplib
from http import HTTPStatus
from smtplib import SMTPAuthenticationError, SMTPRecipientsRefused

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import boto3
from botocore.exceptions import ClientError

from flask import Flask
from flask import request, Response

from jinja2 import Template
import json
```

首先，我们导入 JSON 模块，这样我们就可以序列化和反序列化数据。我们从 HTTP 模块导入`HTTPStatus`，这样我们就可以在服务端点发送响应时使用 HTTP 状态常量。

然后我们导入发送电子邮件所需的模块。我们首先导入`smtplib`，还有一些我们想要处理的异常。

我们还导入`MIMEText`，它将用于从电子邮件内容创建`MIME`对象，以及`MIMEMultipart`，它将用于创建我们将要发送的消息。

接下来，我们导入`boto3`包，这样我们就可以使用*AWS*服务。有一些我们将处理的异常；在这种情况下，这两个异常都与*S3*存储桶有关。

接下来是一些与 Flask 相关的导入，最后但并非最不重要的是，我们导入`Jinja2`包来为我们的电子邮件进行模板化。

继续在`app.py`文件上工作，让我们定义一个常量，用于保存我们创建的*S3*存储桶的名称：

```py
S3_BUCKET_NAME = 'python-blueprints'
```

然后我们创建 Flask 应用程序：

```py
app = Flask(__name__)
```

我们还将添加一个名为`S3Error`的自定义异常：

```py
class S3Error(Exception):
    pass
```

然后我们将定义两个辅助函数。第一个是发送电子邮件的函数：

```py
def _send_message(message):

    smtp = smtplib.SMTP_SSL('email-smtp.eu-west-1.amazonaws.com', 
     465)

    try:
        smtp.login(
            user='DJ********DER*****RGTQ',
            password='Ajf0u*****44N6**ciTY4*****CeQ*****4V')
    except SMTPAuthenticationError:
        return Response('Authentication failed',
                        status=HTTPStatus.UNAUTHORIZED)

    try:
        smtp.sendmail(message['From'], message['To'], 
         message.as_string())
    except SMTPRecipientsRefused as e:
        return Response(f'Recipient refused {e}',
                        status=HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        smtp.quit()

    return Response('Email sent', status=HTTPStatus.OK)
```

在这里，我们定义了函数`_send_message`，它只接受一个参数`message`。我们通过创建一个封装了 SMTP 连接的对象来启动这个函数。我们使用`SMTP_SSL`，因为 AWS Simple Email Service 需要 TLS。第一个参数是 SMTP 主机，我们在 AWS Simple Email Service 中创建的，第二个参数是端口，在需要 SSL 连接的情况下将设置为`456`。

然后我们调用登录方法，传递用户名和密码，这些信息也可以在 AWS Simple Email Service 中找到。在出现`SMTPAuthenticationError`异常的情况下，我们向客户端发送`UNAUTHORIZED`响应。

如果成功登录到 SMTP 服务器，我们调用`sendmail`方法，传递发送消息的电子邮件、目标收件人和消息。我们处理一些收件人拒绝我们消息的情况，如果是这样，我们返回`INTERNAL SERVER ERROR`响应，然后我们退出连接。

最后，我们返回`OK`响应，说明消息已成功发送。

现在，我们创建一个辅助函数，从 S3 存储桶中加载模板文件并返回一个渲染的模板：

```py
def _prepare_template(template_name, context_data):

    s3_client = boto3.client('s3')

    try:
        file = s3_client.get_object(Bucket=S3_BUCKET_NAME, 
        Key=template_name)
    except ClientError as ex:
        error = ex.response.get('Error')
        error_code = error.get('Code')

        if error_code == 'NoSuchBucket':
            raise S3Error(
             f'The bucket {S3_BUCKET_NAME} does not exist') from ex
        elif error_code == 'NoSuchKey':
            raise S3Error((f'Could not find the file "
               {template_name}" '
               f'in the S3 bucket {S3_BUCKET_NAME}')) from ex
        else:
            raise ex

    content = file['Body'].read().decode('utf-8')
    template = Template(content)

    return template.render(context_data)
```

在这里，我们定义了函数`_prepare_template`，它接受两个参数，`template_name`是我们在 S3 存储桶中存储的文件名，`context_data`是一个包含我们将在模板中渲染的数据的字典。

首先，我们创建一个 S3 客户端，然后使用`get_object`方法传递存储桶名称和`Key`。我们将存储桶关键字参数设置为`S3_BUCKET_NAME`，我们在此文件顶部定义了该值为`python-blueprints`。`Key`关键字参数是文件的名称；我们将其设置为在参数`template_name`中指定的值。

接下来，我们访问从 S3 存储桶返回的对象中的`Body`关键字，并调用`read`方法。这将返回一个包含文件内容的字符串。然后，我们创建一个 Jinja2 模板对象，传递模板文件的内容，并最后调用渲染方法传递`context_data`。

现在，让我们实现一个端点，用于向我们收到订单的顾客发送确认电子邮件：

```py
@app.route("/notify/order-received/", methods=['POST'])
def notify_order_received():
    data = json.loads(request.data)

    order_items = data.get('items')

    customer = data.get('order_customer')
    customer_email = customer.get('email')
    customer_name = customer.get('name')

    order_id = data.get('id')
    total_purchased = data.get('total')

    message = MIMEMultipart('alternative')

    context = {
        'order_items': order_items,
        'customer_name': customer_name,
        'order_id': order_id,
        'total_purchased': total_purchased
    }

    try:
        email_content = _prepare_template(
            'order_received_template.html',
            context
        )
    except S3Error as ex:
        return Response(str(ex), 
 status=HTTPStatus.INTERNAL_SERVER_ERROR)

    message.attach(MIMEText(email_content, 'html'))

    message['Subject'] = f'ORDER: #{order_id} - Thanks for your 
    order!'
  message['From'] = 'donotreply@dfurtado.com'
  message['To'] = customer_email

    return _send_message(message)
```

在这里，定义一个名为`notify_order_received`的函数，并使用`@app.route`装饰器定义路由和调用此端点时允许的方法。路由定义为`/notify/order-received/`，`methods`关键字参数接受一个允许的 HTTP 方法列表。在这种情况下，我们只允许 POST 请求。

我们通过获取在请求中传递的所有数据来开始这个函数。在 Flask 应用程序中，可以通过`request.data`访问这些数据；我们使用`json.loads`方法将`request.data`作为参数传递，以便将 JSON 对象反序列化为 Python 对象。然后我们获取项目，这是包含在订单中的所有项目的列表，并获取属性`order_customer`的值，以便获取顾客的电子邮件和顾客的名字。

然后，我们获取订单 ID，可以通过属性`id`访问，最后，我们获取已发送到此端点的数据的属性`total`中的总购买价值。

然后，我们创建一个`MIMEMultiPart`的实例，将`alternative`作为参数传递，这意味着我们将创建一个`MIME`类型设置为 multipart/alternative 的消息。之后，我们配置一个将传递给电子邮件模板的上下文，并使用`_prepare_template`函数传递我们要渲染的模板和包含在电子邮件中显示的数据的上下文。渲染模板的值将存储在变量`email_content`中。

最后，我们对电子邮件消息进行最终设置；我们将渲染的模板附加到消息中，设置主题、发件人和目的地，并调用`_send_message`函数发送消息。

接下来，我们将添加一个端点，用于在用户的订单状态更改为`Shipping`时通知用户。

```py
@app.route("/notify/order-shipped/", methods=['POST'])
def notify_order_shipped():
    data = json.loads(request.data)

    customer = data.get('order_customer')

    customer_email = customer.get('email')
    customer_name = customer.get('name')

    order_id = data.get('id')

    message = MIMEMultipart('alternative')

    try:
        email_content = _prepare_template(
            'order_shipped_template.html',
            {'customer_name': customer_name}
        )
    except S3Error as ex:
        return Response(ex, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    message.attach(MIMEText(email_content, 'html'))

    message['Subject'] = f'Order ID #{order_id} has been shipped'
  message['From'] = 'donotreply@dfurtado.com'
  message['To'] = customer_email

    return _send_message(message)
```

在这里，我们定义一个名为`notify_order_shipped`的函数，并使用`@app.route`装饰器装饰它，传递两个参数和路由，路由设置为`/notify/order-shipped/`，并定义此端点接受的方法为`POST`方法。

我们首先获取在请求中传递的数据 - 基本上与之前的函数`notify_order_received`相同。我们还创建了一个`MIMEMultipart`的实例，并将`MIME`类型设置为 multipart/alternative。接下来，我们使用`_prepare_template`函数加载模板，并使用传递的上下文渲染模板；在这种情况下，我们只传递了顾客的名字。

然后，我们将模板附加到消息中，并进行最终设置，设置主题、发送者和目的地。最后，我们调用`_send_message`发送消息。

接下来，我们将创建两个电子邮件模板，一个用于向用户发送订单确认通知，另一个用于订单已发货时的通知。

# 电子邮件模板

现在我们要创建用于向在线（视频）游戏商店的顾客发送通知邮件的模板。

在应用程序的`root`目录中，创建一个名为`templates`的目录，并创建一个名为`order_received_template.html`的文件，内容如下所示：

```py
<html>
  <head>
  </head>
  <body>
    <h1>Hi, {{customer_name}}!</h1>
    <h3>Thank you so much for your order</h3>
    <p>
      <h3>Order id: {{order_id}}</h3>
    </p>
    <table border="1">
      <thead>
        <tr>
          <th align="left" width="40%">Item</th>
          <th align="left" width="20%">Quantity</th>
          <th align="left" width="20%">Price per unit</th>
        </tr>
      </thead>
      <tbody>
        {% for item in order_items %}
        <tr>
          <td>{{item.name}}</td>
          <td>{{item.quantity}}</td>
          <td>${{item.price_per_unit}}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    <div style="margin-top:20px;">
      <strong>Total: ${{total_purchased}}</strong>
    </div>
  </body>
</html>
```

现在，让我们在同一个目录中创建另一个名为`order_shipped_template.html`的模板，内容如下所示：

```py
<html>
  <head>
  </head>
  <body>
    <h1>Hi, {{customer_name}}!</h1>
    <h3>We just want to let you know that your order is on its way! 
    </h3>
  </body>
</html>
```

如果你阅读过第七章，*使用 Django 创建在线视频游戏商店*，你应该对这种语法很熟悉。与 Django 模板语言相比，Jinja 2 语法有很多相似之处。

现在我们可以将模板复制到之前创建的 S3 存储桶中。打开终端并运行以下命令：

```py
aws s3 cp ./templates s3://python-blueprints --recursive
```

太棒了！接下来，我们将部署我们的项目。

# 使用 Zappa 部署应用程序

现在我们来到了本章非常有趣的部分。我们将使用一个名为**Zappa**（[`github.com/Miserlou/Zappa`](https://github.com/Miserlou/Zappa)）的工具部署我们创建的 Flask 应用程序。Zappa 是一个由**Rich Jones**开发的 Python 工具（Zappa 的主要作者），它使得构建和部署无服务器 Python 应用程序变得非常容易。

安装非常简单。在我们用来开发这个项目的虚拟环境中，你可以运行`pipenv`命令：

```py
pipenv install zappa
```

安装完成后，你可以开始配置。你只需要确保你有一个有效的 AWS 账户，并且 AWS 凭据文件已经就位。如果你从头开始阅读本章并安装和配置了 AWS CLI，那么你应该已经准备就绪了。

要为我们的项目配置 Zappa，你可以运行：

```py
zappa init
```

你会看到 ASCII Zappa 标志（非常漂亮顺便说一句），然后它会开始问一些问题。第一个问题是：

```py
Your Zappa configuration can support multiple production stages, like 'dev', 'staging', and 'production'.
What do you want to call this environment (default 'dev'):
```

你可以直接按*Enter*键默认为`dev`。接下来，Zappa 会询问 AWS S3 存储桶的名称：

```py
Your Zappa deployments will need to be uploaded to a private S3 bucket.
If you don't have a bucket yet, we'll create one for you too.
What do you want call your bucket? (default 'zappa-uc40h2hnc'):
```

在这里，你可以指定一个已存在的环境或创建一个新的环境。然后，Zappa 将尝试检测我们要部署的应用程序：

```py
It looks like this is a Flask application.
What's the modular path to your app's function?
This will likely be something like 'your_module.app'.
We discovered: notify-service.app
Where is your app's function? (default 'notify-service.app'):
```

正如你所看到的，Zappa 自动找到了`notify-service.py`文件中定义的 Flask 应用程序。你可以直接按*Enter*键设置默认值。

接下来，Zappa 会问你是否想要全局部署应用程序；我们可以保持默认值并回答`n`。由于我们在开发环境中部署此应用程序，我们实际上不需要全局部署它。当你的应用程序投入生产时，你可以评估是否需要全局部署。

最后，完整的配置将被显示出来，在这里你需要更改并进行任何必要的修改。你不需要太担心是保存配置还是不保存，因为 Zappa 设置文件只是一个文本文件，以 JSON 格式保存设置。你可以随时编辑文件并手动更改它。

如果一切顺利，你应该在应用程序的根目录下看到一个名为`zappa_settings.json`的文件，内容与下面显示的内容类似：

```py
{
    "dev": {
        "app_function": "notify-service.app",
        "aws_region": "eu-west-2",
        "project_name": "notifier",
        "runtime": "python3.6",
        "s3_bucket": "zappa-43ivixfl0"
    }
}
```

在这里，你可以看到`dev`环境设置。`app_function`指定了我在`notify-service.py`文件中创建的 Flask 应用程序，`aws_region`指定了应用程序将部署在哪个地区 - 在我的情况下，由于我在瑞典，我选择了`eu-west-2`（*伦敦*）这是离我最近的地区。`project_name`将默认为你运行`zappa init`命令的目录名称。

然后我们有运行时，它指的是你在应用程序中使用的 Python 版本。由于我们为这个项目创建的虚拟环境使用的是 Python 3*，*所以这个属性的值应该是 Python 3 的一个版本*-*在我的情况下，我安装了 3.6.2。最后，我们有 Zappa 将用来上传项目文件的 AWS S3 存储桶的名称。

现在，让我们部署刚刚创建的应用程序！在终端上，只需运行以下命令：

```py
zappa deploy dev
```

Zappa 将为您执行许多任务，最后它将显示应用程序部署的 URL。在我的情况下，我得到了：

```py
https://rpa5v43ey1.execute-api.eu-west-2.amazonaws.com/dev
```

你的情况可能会略有不同。因此，我们在 Flask 应用程序中定义了两个端点，`/notify/order-received`和`/notify/order-shipped`。可以使用以下 URL 调用这些端点：

```py
https://rpa5v43ey1.execute-api.eu-west-2.amazonaws.com/dev/notify/order-received
```

```py
https://rpa5v43ey1.execute-api.eu-west-2.amazonaws.com/dev/notify/order-shipped
```

如果你想查看部署的更多信息，可以使用 Zappa 命令：`zappa status`。

在下一节中，我们将学习如何限制对这些端点的访问，并创建一个可以用来进行 API 调用的访问密钥。

# 限制对 API 端点的访问

我们的 Flask 应用程序已经部署，在这一点上任何人都可以向 AWS API Gateway 上配置的端点发出请求。我们想要的是只允许包含访问密钥的请求访问。

为了做到这一点，登录到我们在 AWS 控制台上的帐户，并在 Services 菜单中搜索并选择 Amazon API Gateway*.*。在左侧菜单上的 API 下，你将看到 notifier-dev：

![](img/abc851ca-9fc3-4626-986b-5cb84a46bdb2.png)

太棒了！在这里，我们将定义一个使用计划。点击使用计划，然后点击创建按钮，你将看到一个创建新使用计划的表单。输入名称`up-blueprints`，取消启用节流和启用配额的复选框，然后点击下一步按钮。

下一步是关联 API 阶段。到目前为止，我们只有 dev，所以让我们添加 dev 阶段；点击添加 API 阶段按钮，并在下拉列表中选择 notifier-dev 和阶段 dev。确保点击检查按钮，在下拉菜单的同一行，否则下一步按钮将不会启用。

点击下一步后，你将不得不向我们刚刚创建的使用计划添加一个 API 密钥。在这里你将有两个选项；添加一个新的或选择一个现有的：

![](img/c6112886-16da-4ef5-a61d-759f7705a0f4.png)

让我们添加一个新的。点击标记为创建 API 密钥并添加到使用计划的按钮。API 密钥创建对话框将显示，只需输入名称`notifiers-devs`，然后点击保存。

太棒了！现在，如果你在左侧菜单中选择 API Keys，你应该会在列表中看到新创建的 API 密钥。如果你选择它，你将能够看到有关密钥的所有详细信息：

![](img/a128780e-ca68-4ff5-97de-9a01b6be4413.png)

现在，在左侧菜单中，选择 API -> notifier-dev -> 资源，并在资源选项卡上，选择根路由/。在右侧面板上，你可以看到/方法：

![](img/bdc9760b-78aa-4d93-9ff3-69744d5d3f86.png)

请注意，ANY 表示授权无，API 密钥设置为不需要。让我们更改 API 密钥为必需。在资源面板上，点击 ANY，现在你应该看到一个类似于以下截图的面板：

![](img/adc0d6fd-888b-4f1f-b754-da4c176a292f.png)

点击 Method Request：

![](img/bc95c53f-b152-4b64-becd-b97ba441d619.png)

点击 API Key Required 旁边的笔图标，在下拉菜单中选择值 true*.*。

太棒了！现在，对于 dev 阶段的 API 调用应该限制为请求中包含 API 密钥 notifier-dev。

最后，转到 API Keys，点击 notifier-keys。在右侧面板中，在 API `Key`中，点击显示链接，API 密钥将显示出来。复制该密钥，因为我们将在下一节中使用它。

# 修改订单服务

现在我们已经部署了通知应用程序，我们必须修改我们之前的项目，订单微服务，以利用通知应用程序，并在新订单到达时发送通知，以及在订单状态更改为已发货时发送通知。

我们首先要做的是在`settings.py`文件中包含通知服务 API 密钥和其基本 URL，在名为`order`的目录中，在订单的`root`目录中，并在文件末尾包含以下内容：

```py
NOTIFIER_BASEURL = 'https://rpa5v43ey1.execute-api.eu-west-2.amazonaws.com/dev'

NOTIFIER_API_KEY = 'WQk********P7JR2******kI1K*****r'
```

用您环境中对应的值替换这些值。如果您没有`NOTIFIER_BASEURL`的值，可以通过运行以下命令来获取它：

```py
zappa status
```

您想要的值是 API Gateway URL。

现在，我们要创建两个文件。第一个文件是在`order/main`目录中名为`notification_type.py`的文件。在这个文件中，我们将定义一个包含我们想要在我们的服务中提供的通知类型的枚举：

```py
from enum import Enum, auto

class NotificationType(Enum):
    ORDER_RECEIVED = auto()
    ORDER_SHIPPED = auto()
```

接下来，我们将创建一个帮助函数的文件，该函数将调用通知服务。在`order/main/`目录中创建一个名为`notifier.py`的文件，并包含以下内容：

```py
import requests
import json

from order import settings

from .notification_type import NotificationType

def notify(order, notification_type):
    endpoint = ('notify/order-received/'
                if notification_type is NotificationType.ORDER_RECEIVED
                else 'notify/order-shipped/')

    header = {
        'X-API-Key': settings.NOTIFIER_API_KEY
    }

    response = requests.post(
        f'{settings.NOTIFIER_BASEURL}/{endpoint}',
        json.dumps(order.data),
        headers=header
    )

    return response
```

从顶部开始，我们包括了一些导入语句；我们导入请求以执行对通知服务的请求，因此我们导入 json 模块，以便我们可以将数据序列化为要发送到通知服务的格式。然后我们导入设置，这样我们就可以获得我们在基本 URL 到通知服务和 API 密钥方面定义的常量。最后，我们导入通知类型枚举。

我们在这里定义的`notify`函数接受两个参数，订单和通知类型，这些值在枚举`NotificationType`中定义。

我们首先决定要使用哪个端点，这取决于通知的类型。然后我们在请求的`HEADER`中添加一个`X-API-KEY`条目，其中包含 API 密钥。

之后，我们进行`POST`请求，传递一些参数。第一个参数是端点的 URL，第二个是我们将发送到通知服务的数据（我们使用`json.dumps`函数，以便以 JSON 格式发送数据），第三个参数是包含标头数据的字典。

最后，当我们收到响应时，我们只需返回它。

现在，我们需要修改负责处理`POST`请求以创建新订单的视图，以便在数据库中创建订单时调用`notify`函数。让我们继续打开`order/main`目录中的`view.py`文件，并添加两个导入语句：

```py
from .notifier import notify
from .notification_type import NotificationType
```

这两行可以添加在文件中的第一个类之前。

完美，现在我们需要更改`CreateOrderView`类中的`post`方法。在该方法中的第一个返回语句之前，在我们返回`201`（`CREATED`）响应的地方，包括以下代码：

```py
 notify(OrderSerializer(order),
        NotificationType.ORDER_RECEIVED)
```

因此，在这里我们调用`notify`函数，使用`OrderSerializer`作为第一个参数传递序列化的订单，以及通知类型 - 在这种情况下，我们想发送一个`ORDER_RECEIVED`通知。

我们将允许订单服务应用程序的用户使用 Django 管理界面更新订单。在那里，他们将能够更新订单的状态，因此我们需要实现一些代码来处理用户在 Django 管理界面上进行的数据更改。

为此，我们需要在`order/main`目录中的`admin.py`文件中创建一个`ModelAdmin`类。首先，我们添加一些导入语句：

```py
from .notifier import notify
from .notification_type import NotificationType
from .serializers import OrderSerializer
from .status import Status
```

然后我们添加以下类：

```py
class OrderAdmin(admin.ModelAdmin):

    def save_model(self, request, obj, form, change):
        order_current_status = Status(obj.status)
        status_changed = 'status' in form.changed_data

        if (status_changed and order_current_status is 
           Status.Shipping):
            notify(OrderSerializer(obj), 
            NotificationType.ORDER_SHIPPED)

        super(OrderAdmin, self).save_model(request, obj, form,  
        change)
```

在这里，我们创建了一个名为`OrderAdmin`的类，它继承自`admin.ModelAdmin`，并且我们重写了`save_model`方法，这样我们就有机会在保存数据之前执行一些操作。首先，我们获取订单的当前状态，然后我们检查字段`status`是否在已更改的字段列表之间。

if 语句检查状态字段是否已更改，如果订单的当前状态等于`Status.Shipping`，那么我们调用`notify`函数，传递序列化的订单对象和通知类型`NotificationType.ORDER_SHIPPED`。

最后，我们调用超类的`save_model`方法来保存对象。

这个谜题的最后一部分是替换这个：

```py
admin.site.register(Order)
```

替换为：

```py
admin.site.register(Order, OrderAdmin)
```

这将为`Order`模型注册管理模型`OrderAdmin`。现在，当用户在 Django 管理界面中保存订单时，它将调用`OrderAdmin`类中的`save_model`。

# 测试所有部分的整体功能

现在我们已经部署了通知应用程序，并且还对订单服务进行了所有必要的修改，是时候测试所有应用程序是否能够一起运行了。

打开一个终端，切换到您实施在线（视频）游戏商店的目录，并执行以下命令启动 Django 开发服务器：

```py
python manage.py runserver
```

此命令将启动默认端口`8000`上运行的 Django 开发服务器。

现在让我们启动订单微服务。打开另一个终端窗口，切换到您实施订单微服务的目录，并运行以下命令：

```py
python manage.py runserver 127.0.0.1:8001
```

现在我们可以浏览`http://127.0.0.1:8000`，登录应用程序并向购物车中添加一些商品：

![](img/20a6d3cd-f8f3-4541-8c78-caecf5121171.png)

如您所见，我添加了三件商品，此订单的总金额为 32.75 美元。单击“发送订单”按钮，您应该会在页面上收到订单已发送的通知。

![](img/a31cb284-c9ec-4db6-ad49-a14fe17cd8e8.png)

太好了！到目前为止一切都按预期进行。现在我们检查用户的电子邮件，以验证通知服务是否实际发送了订单确认电子邮件。

还不错，用户刚刚收到了邮件：

![](img/8fd748de-ba5d-48ac-be77-58e7f81d223c.png)

请注意，发件人和收件人的电子邮件是我在 AWS 简单电子邮件服务中注册的。

现在让我们登录订单服务的 Django 管理界面，并更改相同订单的状态，以验证订单已发货的确认电子邮件是否会发送给用户。请记住，只有当订单将其状态字段更改为已发货时，才会发送电子邮件。

浏览`http://localhost:8001/admin/`并使用管理员凭据登录。您将看到一个带有以下菜单的页面：

![](img/063496f6-7f0d-40f4-ad9e-5916bdcdcc20.png)

点击订单，然后选择我们刚刚提交的订单：

![](img/92f7f9f5-dd5c-425b-a89e-269748d2f1b4.png)

在下拉菜单“状态”中，将值更改为“发货”，然后单击“保存”按钮。

现在，如果我们再次验证订单客户的电子邮件，我们应该已经收到另一封确认订单已发货的电子邮件：

![](img/0b0f6c48-27a2-4936-86d4-6d3386d1547c.png)

# 总结

在本章中，您学习了有关无服务器函数架构的更多信息，如何使用 Web 框架 Flask 构建通知服务，以及如何使用伟大的项目 Zappa 将最终应用程序部署到 AWS Lambda。

然后，您学习了如何安装、配置和使用 AWS CLI 工具，并使用它将文件上传到 AWS S3 存储桶。

我们还学习了如何将我们在第七章*Django 在线视频游戏商店*中开发的 Web 应用程序与我们在第八章*订单微服务*中开发的订单微服务与无服务器通知应用程序集成。
