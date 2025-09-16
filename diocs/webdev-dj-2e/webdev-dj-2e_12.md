# 第十二章：12. 构建 REST API

概述

本章介绍了**REST API**和**Django REST 框架**（**DRF**）。你将从为 Bookr 项目实现一个简单的 API 开始。接下来，你将学习模型实例的序列化，这是将数据传递到 Django 应用前端的关键步骤。你将探索不同类型的 API 视图，包括函数式和基于类的类型。到本章结束时，你将能够实现自定义 API 端点，包括简单的身份验证。

# 简介

在上一章中，我们学习了模板和基于类的视图。这些概念极大地帮助我们扩展了提供给用户的前端（即在他们的网络浏览器中）的功能范围。然而，这还不足以构建一个现代 Web 应用程序。Web 应用程序通常使用完全独立的库来构建前端，例如**ReactJS**或**AngularJS**。这些库提供了构建动态用户界面的强大工具；然而，它们不能直接与我们的后端 Django 代码或数据库通信。前端代码仅在网络浏览器中运行，并且无法直接访问后端服务器上的任何数据。因此，我们需要创建一种方法，让这些应用程序能够“与”我们的后端代码“交流”。在 Django 中实现这一点的最佳方法之一就是使用 REST API。

**API**代表**应用程序编程接口**。API 用于促进不同软件组件之间的交互，并且它们使用**HTTP**（**超文本传输协议**）进行通信。这是服务器和客户端之间通信的标准协议，是网络信息传输的基础。API 以 HTTP 格式接收请求并发送响应。

在本章的使用案例中，API 将帮助我们促进 Django 后端和前端 JS 代码之间的交互。例如，想象一下我们想要创建一个前端应用程序，允许用户向 Bookr 数据库添加新书。用户的网络浏览器会向我们的 API 发送一条消息（一个 HTTP 请求），表示他们想要为新书创建一个条目，并可能在该消息中包含一些关于书籍的细节。我们的服务器会发送一个响应来报告书籍是否成功添加。然后网络浏览器将能够向用户显示他们操作的结果。

# REST API

**REST**代表**表示性状态转移**。大多数现代 Web API 都可以归类为 REST API。REST API 是一种简单的 API 类型，它专注于在数据库服务器和前端客户端之间通信和同步对象的**状态**。

例如，想象一下你正在更新你登录账户的网站上的详细信息。当你访问账户详情页面时，网络服务器会告诉你的浏览器与你账户相关的各种详细信息。当你更改该页面的值时，浏览器会将更新后的详细信息发送回网络服务器，并告诉它更新数据库中的这些详细信息。如果操作成功，网站将显示确认消息。

这是一个非常简单的例子，说明了前端和后端系统之间所知的 **解耦** 架构。解耦提供了更大的灵活性，并使得更新或更改架构中的组件变得更加容易。所以，假设你想创建一个新的前端网站。在这种情况下，你根本不需要更改后端代码，只要你的新前端能够构建出与旧的一个相同的 API 请求即可。

REST API 是 *无状态的*，这意味着客户端和服务器在通信过程中都不会存储任何状态。每次请求时，数据都会被处理，并返回响应，而无需协议本身存储任何中间数据。这意味着 API 正在独立处理每个请求。它不需要存储有关会话本身的信息。这与维护会话信息的 **TCP** 等有状态协议形成对比。

因此，正如其名所示，**RESTful 网络服务**是一组用于执行一系列任务的 REST API 的集合。例如，如果我们为 Bookr 应用程序开发一组用于执行特定任务的 REST API，那么我们可以称它为 RESTful 网络服务。

## Django REST 框架

**Django REST 框架**，简称 **DRF**，是一个开源的 Python 库，可用于为 Django 项目开发 REST API。DRF 内置了大多数必要的功能，以帮助开发任何 Django 项目的 API。在本章中，我们将使用它来为我们的 Bookr 项目开发 API。

## 安装和配置

在 PyCharm 的虚拟环境设置中安装 `djangorestframework`。在您的终端应用程序或命令提示符中输入以下代码来完成此操作：

```py
pip install djangorestframework
```

接下来，打开 `settings.py` 文件，并将 `rest_framework` 添加到 `INSTALLED_APPS` 中，如下面的代码片段所示：

```py
INSTALLED_APPS = ['django.contrib.admin',\
                  'django.contrib.auth',\
                  ‹django.contrib.contenttypes›,\
                  'django.contrib.sessions',\
                  'django.contrib.messages',\
                  'django.contrib.staticfiles',\
                  ‹rest_framework›,\
                  ‹reviews›]
```

现在，你已经准备好开始使用 DRF 创建你的第一个简单 API。

## 功能性 API 视图

在 *第三章*，*URL 映射、视图和模板* 中，我们学习了简单的功能性视图，它接受一个请求并返回一个响应。我们可以使用 DRF 编写类似的函数视图。然而，请注意，基于类的视图更常用，将在下一章中介绍。一个功能性视图是通过简单地在一个普通视图上添加以下装饰器来创建的，如下所示：

```py
from rest_framework.decorators import api_view
@api_view
def my_view(request):
     ...
```

这个装饰器将功能视图转换为 DRF 的 `APIView` 子类。这是一种快速地将现有视图作为 API 部分包含进来的方法。

## 练习 12.01：创建一个简单的 REST API

在这个练习中，您将使用 DRF 创建您的第一个 REST API，并使用功能视图实现一个端点。您将创建这个端点来查看数据库中的书籍总数：

注意

您需要在本系统上安装 DRF 才能继续此练习。如果您尚未安装，请确保您参考了本章前面标题为 *安装和配置* 的部分。

1.  在 `bookr/reviews` 文件夹中创建 `api_views.py`。

    REST API 视图的工作方式类似于 Django 的传统视图。我们本可以将 API 视图以及其他视图一起添加到 `views.py` 文件夹中。但是，将我们的 REST API 视图放在单独的文件中将帮助我们保持代码库的整洁。

1.  在 `api_views.py` 中添加以下代码：

    ```py
    from rest_framework.decorators import api_view
    from rest_framework.response import Response
    from .models import Book
    @api_view()
    def first_api_view(request):
        num_books = Book.objects.count()
        api_view decorator, which will be used to convert our functional view into one that can be used with DRF, and the second line imports Response, which will be used to return a response.The `view` function returns a `Response` object containing a dictionary with the number of books in our database (see the highlighted part).Open `bookr/reviews/urls.py` and import the `api_views` module. Then, add a new path to the `api_views` module in the URL patterns that we have developed throughout this course, as follows:

    ```

    从 . 导入 views, api_views

    urlpatterns = [path('api/first_api_view/',)]

    path(api_views.first_api_view)

    …

    ]

    ```py

    Start the Django service with the `python manage.py runserver` command and go to `http://0.0.0.0:8000/api/first_api_view/` to make your first API request. Your screen should appear as in *Figure 12.1*:![Figure 12.1: API view with the number of books    ](img/B15509_12_01.jpg)Figure 12.1: API view with the number of booksCalling this URL endpoint made a default `GET` request to the API endpoint, which returned a JSON key-value pair `("num_books": 0`). Also, notice how DRF provides a nice interface to view and interact with the APIs.
    ```

1.  我们也可以使用 Linux 的 `curl`（客户端 URL）命令发送 HTTP 请求，如下所示：

    ```py
    curl http://0.0.0.0:8000/api/first_api_view/
    {"num_books":0}
    ```

    或者，如果您使用的是 Windows 10，您可以通过命令提示符使用 `curl.exe` 发送等效的 HTTP 请求，如下所示：

    ```py
    curl.exe http://0.0.0.0:8000/api/first_api_view/
    ```

在这个练习中，我们学习了如何使用 DRF 创建 API 视图和使用简单功能视图。现在我们将探讨一种更优雅的方法，使用序列化器在数据库中存储的信息和 API 返回的信息之间进行转换。

# 序列化器

到现在为止，我们对 Django 在我们的应用程序中处理数据的方式已经非常熟悉。从广义上讲，数据库表中的列是在 `models.py` 中的一个类中定义的，当我们访问表中的一行时，我们正在处理该类的实例。理想情况下，我们通常只想将此对象传递给我们的前端应用程序。例如，如果我们想构建一个显示 Bookr 应用程序中书籍列表的网站，我们就会调用每个书籍实例的 `title` 属性，以知道向用户显示什么字符串。然而，我们的前端应用程序对 Python 一无所知，需要通过 HTTP 请求检索这些数据，该请求只返回特定格式的字符串。

这意味着在 Django 和前端（通过我们的 API）之间转换的任何信息都必须通过以 **JavaScript 对象表示法**（**JSON**）格式表示信息来完成。JSON 对象看起来与 Python 字典相似，但有一些额外的规则限制了确切的语法。在我们的上一个示例 *练习 12.01*，*创建一个简单的 REST API* 中，API 返回了以下包含我们数据库中书籍数量的 JSON 对象：

```py
{"num_books": 0}
```

但如果我们想通过我们的 API 返回数据库中实际书籍的完整详细信息呢？DRF 的`serializer`类帮助将复杂的 Python 对象转换为 JSON 或 XML 等格式，以便可以通过 HTTP 协议在网络上传输。DRF 中执行此转换的部分被称为`serializer`。序列化器还执行反序列化，这指的是将序列化数据转换回 Python 对象，以便在应用程序中处理。

## 练习 12.02：创建一个 API 视图以显示书籍列表

在这个练习中，你将使用序列化器创建一个 API，该 API 返回`bookr`应用程序中所有书籍的列表：

1.  在`bookr/reviews`文件夹中创建一个名为`serializers.py`的文件。这是我们放置所有 API 序列化代码的文件。

1.  将以下代码添加到`serializers.py`中：

    ```py
    from rest_framework import serializers
    class PublisherSerializer(serializers.Serializer):
        name = serializers.CharField()
        website = serializers.URLField()
        email = serializers.EmailField()
    class BookSerializer(serializers.Serializer):
        title = serializers.CharField()
        publication_date = serializers.DateField()
        isbn = serializers.CharField()
        publisher = PublisherSerializer()
    ```

    这里，第一行从`rest_framework`模块导入了序列化器。

    在导入之后，我们定义了两个类，`PublisherSerializer`和`BookSerializer`。正如其名称所暗示的，它们分别是`Publisher`和`Book`模型的序列化器。这两个序列化器都是`serializers.Serializer`的子类，并且我们为每个序列化器定义了字段类型，如`CharField`、`URLField`和`EmailField`等。

    查看`bookr/reviews/models.py`文件中的`Publisher`模型。`Publisher`模型有`name`、`website`和`email`属性。因此，为了序列化`Publisher`对象，我们需要在`serializer`类中包含`name`、`website`和`email`属性，我们已经在`PublisherSerializer`中相应地定义了这些属性。同样，对于`Book`模型，我们在`BookSerializer`中定义了`title`、`publication_date`、`isbn`和`publisher`作为所需的属性。由于`publisher`是`Book`模型的外键，我们已将`PublisherSerializer`用作`publisher`属性的序列化器。

1.  打开`bookr/reviews/api_views.py`，删除任何现有的代码，并添加以下代码：

    ```py
    from rest_framework.decorators import api_view
    from rest_framework.response import Response
    from .models import Book
    from .serializers import BookSerializer
    @api_view()
    def all_books(request):
        books = Book.objects.all()
        book_serializer = BookSerializer(books, many=True)
        return Response(book_serializer.data)
    ```

    在第二行，我们从`serializers`模块导入了新创建的`BookSerializer`。

    然后，我们添加一个功能视图`all_books`（如前一个练习所示）。此视图接受包含所有书籍的查询集，然后使用`BookSerializer`对它们进行序列化。`serializer`类还接受一个参数`many=True`，这表示`books`对象是一个`queryset`或多个对象的列表。请记住，序列化将 Python 对象转换为可序列化为 JSON 的格式，如下所示：

    ```py
    [OrderedDict([('title', 'Advanced Deep Learning with Keras'), ('publication_date', '2018-10-31'), ('isbn', '9781788629416'), ('publisher', OrderedDict([('name', 'Packt Publishing'), ('website', 'https://www.packtpub.com/'), ('email', 'info@packtpub.com')]))]), OrderedDict([('title', 'Hands-On Machine Learning for Algorithmic Trading'), ('publication_date', '2018-12-31'), ('isbn', '9781789346411'), ('publisher', OrderedDict([('name', 'Packt Publishing'), ('website', 'https://www.packtpub.com/'), ('email', 'info@packtpub.com')]))]) …
    ```

1.  打开`bookr/reviews/urls.py`，删除之前的`first_api_view`示例路径，并添加如下代码中的`all_books`路径：

    ```py
    from django.urls import path
    from . import views, api_views
    urlpatterns = [path('api/all_books/'),\
                   path(api_views.all_books),\
                   path(name='all_books')
        …
    ]
    ```

    新增的路径在遇到 URL 中的`api/all_books/`路径时会调用视图函数`all_books`。

1.  一旦添加了所有代码，使用`python manage.py runserver`命令运行 Django 服务器，并导航到`http://0.0.0.0:8000/api/all_books/`。你应该会看到类似于*图 12.2*的内容：![图 12.2：在 all_books 端点显示的书籍列表]

    ![img/B15509_12_02.jpg]

图 12.2：在 all_books 端点显示的书籍列表

前面的截图显示，在调用`/api/all_books`端点时，返回了所有书籍的列表。就这样，你已经成功使用序列化器在数据库中高效地返回数据，借助 REST API 的帮助。

到目前为止，我们一直专注于功能视图。然而，你现在将了解到在 DRF 中更常用的是基于类的视图，这将使你的生活变得更加轻松。

## 基于类的 API 视图和通用视图

与我们在*第十一章*，*高级模板和基于类的视图*中学到的类似，我们也可以为 REST API 编写基于类的视图。对于编写视图来说，基于类的视图是开发者中最受欢迎的方式，因为通过编写很少的代码就能实现很多功能。

就像传统的视图一样，DRF 提供了一套通用视图，使得编写基于类的视图变得更加简单。通用视图是在考虑到创建 API 时所需的一些最常见操作而设计的。DRF 提供的通用视图包括`ListAPIView`、`RetrieveAPIView`等等。在*练习 12.02*，*创建一个显示书籍列表的 API 视图*中，我们的功能视图负责创建对象的`queryset`并调用序列化器。同样，我们也可以使用`ListAPIView`来完成相同的事情：

```py
class AllBooks(ListAPIView):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

在这里，对象的`queryset`被定义为类属性。将`queryset`传递给`serializer`由`ListAPIView`上的方法处理。

## 模型序列化器

在*练习 12.02*，*创建一个显示书籍列表的 API 视图*中，我们的序列化器定义如下：

```py
class BookSerializer(serializers.Serializer):
    title = serializers.CharField()
    publication_date = serializers.DateField()
    isbn = serializers.CharField()
    publisher = PublisherSerializer()
```

然而，我们的`Book`模型看起来是这样的（注意模型和序列化器的定义看起来多么相似）：

```py
class Book(models.Model):
    """A published book."""
    title = models.CharField(max_length=70),\
                            (help_text="The title of the book.")
    publication_date = models.DateField\
                      (verbose_name="Date the book was published.")
    isbn = models.CharField(max_length=20),\
                           (verbose_name="ISBN number of the book.")
    publisher = models.ForeignKey(Publisher),\
                                 (on_delete=models.CASCADE)
    contributors = models.ManyToManyField('Contributor'),\
                                         (through="BookContributor")
    def __str__(self):
        return self.title
```

我们宁愿不指定标题必须是`serializers.CharField()`。如果序列化器只需查看模型中`title`是如何定义的，并能够确定要使用什么序列化器字段，那就更容易了。

这就是模型序列化器发挥作用的地方。它们通过利用模型上字段的定义来提供创建序列化器的快捷方式。我们不需要指定`title`应该使用`CharField`进行序列化，我们只需告诉模型序列化器我们想要包含`title`，它就会使用`CharField`序列化器，因为模型上的`title`字段也是一个`CharField`。

例如，假设我们想在`models.py`中为`Contributor`模型创建一个序列化器。我们不需要指定每个字段应该使用哪种序列化器的类型，我们可以给它一个字段名称的列表，让它自己决定其他的事情：

```py
from rest_framework import serializers
from .models import Contributor
class ContributorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contributor
        fields = ['first_names', 'last_names', 'email']
```

在以下练习中，我们将看到如何使用模型序列化器来避免在前面类中重复代码。

## 练习 12.03：创建基于类的 API 视图和模型序列化器

在这个练习中，你将创建基于类的视图来显示所有书籍的列表，同时使用模型序列化器：

1.  打开文件`bookr/reviews/serializers.py`，删除任何现有的代码，并用以下代码替换：

    ```py
    from rest_framework import serializers
    from .models import Book, Publisher
    class PublisherSerializer(serializers.ModelSerializer):
        class Meta:
            model = Publisher
            fields = ['name', 'website', 'email']
    class BookSerializer(serializers.ModelSerializer):
        publisher = PublisherSerializer()
        class Meta:
            model = Book
            fields = ['title', 'publication_date', 'isbn', 'publisher']
    ```

    在这里，我们包含了两个模型序列化类，`PublisherSerializer`和`BookSerializer`。这两个类都继承自父类`serializers.ModelSerializer`。我们不需要指定每个字段如何序列化，相反，我们可以简单地传递一个字段名称列表，字段类型将根据`models.py`中的定义推断。

    虽然在`fields`中提及字段对于模型序列化器来说是足够的，但在某些特殊情况下，例如这个例子，我们可能需要自定义字段，因为`publisher`字段是一个外键。因此，我们必须使用`PublisherSerializer`来序列化`publisher`字段。

1.  接下来，打开`bookr/reviews/api_views.py`，删除任何现有的代码，并添加以下代码：

    ```py
    from rest_framework import generics
    from .models import Book
    from .serializers import BookSerializer
    class AllBooks(generics.ListAPIView):
        queryset = Book.objects.all()
        serializer_class = BookSerializer
    ```

    在这里，我们使用 DRF 的基于类的`ListAPIView`而不是功能视图。这意味着书籍列表被定义为类的属性，我们不需要编写一个直接处理请求并调用序列化器的函数。上一步骤中的书籍序列化器也被导入并分配为这个类的属性。

    打开`bookr/reviews/urls.py`文件，修改`/api/all_books` API 路径以包含新的基于类的视图，如下所示：

    ```py
    urlpatterns = [path('api/all_books/'),\
                   path(api_views.AllBooks.as_view()),\
                   path(name='all_books')]
    ```

    由于我们使用的是基于类的视图，我们必须使用类名以及`as_view()`方法。

1.  完成所有前面的修改后，等待 Django 服务重启，或者使用`python manage.py runserver`命令启动服务器，然后在网络浏览器中打开`http://0.0.0.0:8000/api/all_books/`处的 API。你应该会看到类似于*图 12.3*的内容：![图 12.3：在 all_books 端点显示的书籍列表    ![图片](img/B15509_12_03.jpg)

图 12.3：在 all_books 端点显示的书籍列表

如同我们在*练习 12.02*，*创建一个显示书籍列表的 API 视图*中看到的，这是一个显示在书籍评论应用程序中的所有书籍的列表。在这个练习中，我们使用了模型序列化器来简化我们的代码，并使用通用的基于类的`ListAPIView`来返回数据库中书籍的列表。

## 活动 12.01：创建一个顶级贡献者页面 API 端点

假设你的团队决定创建一个显示数据库中顶级贡献者（即作者、合著者和编辑）的网页。他们决定聘请外部开发者使用 React JavaScript 创建一个应用程序。为了与 Django 后端集成，开发者需要一个提供以下内容的端点：

+   数据库中所有贡献者的列表

+   对于每个贡献者，列出他们所贡献的所有书籍。

+   对于每个贡献者，列出他们所贡献的书籍数量。

+   对于他们所贡献的每本书，列出他们在书中的角色。

最终的 API 视图应如下所示：

![图 12.4：顶级贡献者端点![图片 B15509_12_04.jpg](img/B15509_12_04.jpg)

图 12.4：顶级贡献者端点

要执行此任务，请执行以下步骤：

1.  向`Contributor`类添加一个方法，该方法返回所做的贡献数量。

1.  添加`ContributionSerializer`，它序列化`BookContribution`模型。

1.  添加`ContributorSerializer`，它序列化`Contributor`模型。

1.  添加`ContributorView`，它使用`ContributorSerializer`。

1.  在`urls.py`中添加一个模式以启用对`ContributorView`的访问。

    注意

    本活动的解决方案可以在[`packt.live/2Nh1NTJ`](http://packt.live/2Nh1NTJ)找到。

# 视图集

我们已经看到，我们可以如何通过使用基于类的通用视图来优化我们的代码并使其更加简洁。`AllBooks`视图返回应用程序中所有书籍的列表，而`BookDetail`视图返回单本书的详细信息。使用视图集，我们可以将这两个类合并成一个。

DRF 还提供了一个名为`ModelViewSet`的类。这个类不仅结合了前面讨论中提到的两个视图（即列表和详情），还允许您创建、更新和删除模型实例。实现所有这些功能所需的代码可能非常简单，只需指定序列化和`queryset`即可。例如，一个允许您管理用户模型的所有这些操作的视图可以简洁地定义如下：

```py
class UserViewSet(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User
```

最后，DRF 提供了一个名为`ReadOnlyModelViewSet`的类。这是先前`ModelViewSet`的一个更简单的版本。它与前面的版本相同，只是它只允许您列出和检索特定用户。您不能创建、更新或删除记录。

# 路由器

当与视图集一起使用时，路由器会自动创建视图集所需的 URL 端点。这是因为单个视图集在不同的 URL 上被访问。例如，在先前的`UserViewSet`中，您可以在 URL `/api/users/` 上访问用户列表，并在 URL `/api/users/123` 上访问特定用户记录，其中`123`是该用户记录的主键。以下是一个简单的示例，说明您如何在先前定义的`UserViewSet`的上下文中使用路由器：

```py
from rest_framework import routers
router = routers.SimpleRouter()
router.register(r'users', UserViewSet)
urlpatterns = router.urls
```

现在，让我们通过一个简单的练习来尝试结合路由器和视图集的概念。

## 练习 12.04：使用视图集和路由器

在这个练习中，我们将结合现有的视图来创建视图集，并为视图集创建所需的路由：

1.  打开文件`bookr/reviews/serializers.py`，删除现有的代码，并添加以下代码片段：

    ```py
    reviews/serializers.py
    01  from django.contrib.auth.models import User
    02  from django.utils import timezone
    03  from rest_framework import serializers
    04  from rest_framework.exceptions import NotAuthenticated, PermissionDenied
    05
    06  from .models import Book, Publisher, Review
    07  from .utils import average_rating
    08
    09  class PublisherSerializer(serializers.ModelSerializer):
    You can find the complete code snippet at http://packt.live/3osYJli.
    ```

    在这里，我们向 `BookSerializer` 添加了两个新字段，即 `reviews` 和 `rating`。这些字段的有趣之处在于，它们背后的逻辑被定义为序列化器本身上的一个方法。这就是为什么我们使用 `serializers.SerializerMethodField` 类型来设置 `serializer` 类属性。

1.  打开文件 `bookr/reviews/api_views.py`，删除现有的代码，并添加以下内容：

    ```py
    from rest_framework import viewsets
    from rest_framework.pagination import LimitOffsetPagination
    from .models import Book, Review
    from .serializers import BookSerializer, ReviewSerializer
    class BookViewSet(viewsets.ReadOnlyModelViewSet):
        queryset = Book.objects.all()
        serializer_class = BookSerializer
    class ReviewViewSet(viewsets.ModelViewSet):
        queryset = Review.objects.order_by('-date_created')
        serializer_class = ReviewSerializer
        pagination_class = LimitOffsetPagination
        authentication_classes = []
    ```

    在这里，我们删除了 `AllBook` 和 `BookDetail` 视图，并用 `BookViewSet` 和 `ReviewViewSet` 替换它们。在第一行，我们从 `rest_framework` 模块导入 `ViewSets` 模块。`BookViewSet` 类是 `ReadOnlyModelViewSet` 的子类，这确保了视图仅用于 `GET` 操作。

    接下来，打开 `bookr/reviews/urls.py` 文件，删除以 `api/` 开头的第一个两个 URL 模式，然后添加以下（突出显示）代码：

    ```py
    all_books and book_detail paths into a single path called books.  We have also added a new endpoint under the path reviews which we will need in a later chapter.We start by importing the `DefaultRouter` class from `rest_framework.routers`. Then, we create a `router` object using the `DefaultRouter` class and then register the newly created `BookViewSet` and `ReviewViewSet`, as can be seen from the highlighted code. This ensures that the `BookViewSet` is invoked whenever the API has the `/api/books` path.
    ```

1.  保存所有文件，一旦 Django 服务重启（或者您使用 `python manage.py runserver` 命令手动启动），请访问 URL `http://0.0.0.0:8000/api/books/` 以获取所有书籍的列表。您应该在 API 探索器中看到以下视图：![图 12.5：/api/books 路径下的书籍列表    ](img/B15509_12_05.jpg)

    图 12.5：/api/books 路径下的书籍列表

1.  您也可以使用 URL `http://0.0.0.0:8000/api/books/1/` 访问特定书籍的详细信息。在这种情况下，它将返回具有主键 `1` 的书籍的详细信息（如果它在您的数据库中存在）：![图 12.6：“使用 Keras 进行高级深度学习”的书籍详细信息    ](img/B15509_12_06.jpg)

图 12.6：“使用 Keras 进行高级深度学习”的书籍详细信息

在这个练习中，我们看到了如何使用视图集和路由器将列表视图和详细信息视图合并为一个视图集。使用视图集将使我们的代码更加一致和符合惯例，更容易与其他开发者协作。当与单独的前端应用程序集成时，这一点尤为重要。

# 身份验证

正如我们在 *第九章* 中学到的，*会话和身份验证*，验证我们应用程序的用户非常重要。只允许在应用程序中注册的用户登录并访问应用程序中的信息是一种良好的做法。同样，对于 REST API，我们也需要设计一种方式在传递任何信息之前对用户进行验证和授权。例如，假设 Facebook 的网站通过 API 请求获取一个帖子的所有评论列表。如果他们在这个端点上没有进行身份验证，您就可以用它来程序化地获取任何您想要的帖子的评论。显然，他们不希望允许这样做，因此需要实现某种形式的身份验证。

存在多种认证方案，例如**基本认证**、**会话认证**、**令牌认证**、**远程用户认证**以及各种第三方认证解决方案。在本章范围内，以及针对我们的 Bookr 应用程序，我们将使用**令牌认证**。

注意

关于所有认证方案的进一步阅读，请参阅官方文档[`www.django-rest-framework.org/api-guide/authentication`](https://www.django-rest-framework.org/api-guide/authentication)。

## 基于令牌的认证

基于令牌的认证通过为用户生成一个唯一的令牌来交换用户的用户名和密码。一旦生成令牌，它将被存储在数据库中以供进一步参考，并在每次登录时返回给用户。

这个令牌对每个用户都是唯一的，用户可以使用这个令牌来授权他们发出的每个 API 请求。基于令牌的认证消除了在每次请求中传递用户名和密码的需要。它更加安全，非常适合客户端-服务器通信，例如，基于 JavaScript 的 Web 客户端通过 REST API 与后端应用程序交互。

例如，一个 ReactJS 或 AngularJS 应用程序通过 REST API 与 Django 后端交互。

如果你正在开发一个与后端服务器通过 REST API 交互的移动应用程序，例如，一个与 Django 后端通过 REST API 交互的 Android 或 iOS 应用程序，可以使用相同的架构。

## 练习 12.05：为 Bookr API 实现基于令牌的认证

在这个练习中，你将为`bookr`应用程序的 API 实现基于令牌的认证：

1.  打开`bookr/settings.py`文件，并将`rest_framework.authtoken`添加到`INSTALLED_APPS`中：

    ```py
    INSTALLED_APPS = ['django.contrib.admin',\
                      'django.contrib.auth',\
                      ‹django.contrib.contenttypes›,\
                      'django.contrib.sessions',\
                      'django.contrib.messages',\
                      'django.contrib.staticfiles',\
                      ‹rest_framework›,\
                      ‹rest_framework.authtoken›,\
                      ‹reviews›]
    ```

1.  由于`authtoken`应用程序关联了数据库更改，请在命令行/终端中运行以下`migrate`命令：

    ```py
    python manage.py migrate
    ```

1.  打开`bookr/reviews/api_views.py`文件，删除任何现有的代码，并用以下内容替换：

    ```py
    /reviews/api_views.py
    from django.contrib.auth import authenticate
    from rest_framework import viewsets
    from rest_framework.authentication import TokenAuthentication
    from rest_framework.authtoken.models import Token
    from rest_framework.pagination import LimitOffsetPagination
    from rest_framework.permissions import IsAuthenticated
    from rest_framework.response import Response
    from rest_framework.status import HTTP_404_NOT_FOUND, HTTP_200_OK
    from rest_framework.views import APIView
    You can find the complete code for this file at http://packt.live/2JQebbS.
    ```

    在这里，我们定义了一个名为`Login`的视图。这个视图的目的是允许用户获取（或创建如果尚未存在）一个令牌，他们可以使用这个令牌来通过 API 进行认证。

    我们覆盖了这个视图的`post`方法，因为我们想自定义用户发送数据（即他们的登录详情）时的行为。首先，我们使用 Django 的`auth`库中的`authenticate`方法来检查用户名和密码是否正确。如果正确，那么我们将有一个`user`对象。如果不正确，我们返回一个`HTTP 404`错误。如果我们有一个有效的`user`对象，那么我们只需获取或创建一个令牌，并将其返回给用户。

1.  接下来，让我们将认证类添加到我们的`BookViewSet`中。这意味着当用户尝试访问此视图集时，它将要求他们使用基于令牌的认证进行认证。请注意，可以包含一系列不同的接受认证方法，而不仅仅是其中一个。我们还添加了`permissions_classes`属性，它仅使用 DRF 内置的类来检查给定用户是否有权限查看此模型中的数据：

    ```py
    class BookViewSet(viewsets.ReadOnlyModelViewSet):
        queryset = Book.objects.all()
        serializer_class = BookSerializer
        authentication_classes = [TokenAuthentication]
        permission_classes = [IsAuthenticated]
    ```

    注意

    前面的代码（高亮显示）不会与你在 GitHub 上看到的代码匹配，因为我们将在第 9 步中对其进行修改

1.  打开`bookr/reviews/urls.py`文件，并将以下路径添加到 URL 模式中。

    ```py
    path('api/login', api_views.Login.as_view(), name='login')
    ```

1.  保存文件并等待应用程序重新启动，或者使用`python manage.py runserver`命令手动启动服务器。然后使用 URL `http://0.0.0.0:8000/api/login` 访问应用程序。你的屏幕应该如下所示：![图 12.7：登录页面    ](img/B15509_12_07.jpg)

    图 12.7：登录页面

    `/api/login` API 只接受`POST`请求，因此显示“方法 GET 不允许”。

1.  接下来，在内容中输入以下片段，然后点击`POST`：

    ```py
    {
    "username": "Peter",
    "password": "testuserpassword"
    }
    ```

    你需要将此替换为数据库中你账户的实际用户名和密码。现在你可以看到为用户生成的令牌。这是我们用来访问`BookSerializer`的令牌：

    ![图 12.8：为用户生成的令牌    ](img/B15509_12_08.jpg)

    图 12.8：为用户生成的令牌

1.  尝试使用我们之前创建的 API 访问书籍列表，API 地址为`http://0.0.0.0:8000/api/books/`。请注意，你现在不再被允许访问它。这是因为这个视图集现在要求你使用你的令牌进行认证。

    同样的 API 可以使用命令行中的`curl`访问：

    ```py
    curl -X GET http://0.0.0.0:8000/api/books/
    {"detail":"Authentication credentials were not provided."}
    ```

    由于未提供令牌，显示消息“未提供认证凭据”：

    ![图 12.9：显示认证详情未提供的消息    ](img/B15509_12_09.jpg)

    ```py
    curl -X GET http://0.0.0.0:8000/api/books/ -H "Authorization: Token 724865fcaff6d0aace359620a12ec0b5cc6524fl"
     [{"title":"Advanced Deep Learning with Keras","publication_date":"2018-10-31","isbn":"9781788629416","publisher":{"name":"Packt Publishing","website":"https://www.packtpub.com/","email":"info@packtpub.com"},"rating":4,"reviews":[{"content":"A must read for all","date_created":… (truncated)
    ```

    此操作确保只有应用程序的现有用户才能访问和检索所有书籍的集合。

1.  在继续之前，将`BookViewSet`上的认证和权限类设置为空字符串。未来的章节将不会使用这些认证方法，我们将为了简单起见假设我们的 API 可以被未经认证的用户访问。

    ```py
    class BookViewSet(viewsets.ReadOnlyModelViewSet):
        queryset = Book.objects.all()
        serializer_class = BookSerializer
        authentication_classes = []
        permission_classes = []
    ```

在这个练习中，我们在 Bookr 应用程序中实现了基于令牌的认证。我们创建了一个登录视图，允许我们检索给定认证用户的令牌。然后，我们通过在请求中将令牌作为头部传递，从命令行进行 API 请求。

# 摘要

本章介绍了 REST API，这是大多数现实世界 Web 应用的基本构建块。这些 API 促进了后端服务器与 Web 浏览器之间的通信，因此对于你作为 Django Web 开发者的成长至关重要。我们学习了如何在数据库中序列化数据，以便通过 HTTP 请求进行传输。我们还学习了 DRF 为我们提供的各种选项，以简化我们编写的代码，利用模型本身的现有定义。我们还涵盖了视图集和路由器，并看到了如何通过组合多个视图的功能来进一步压缩代码。我们还学习了身份验证和授权，并为书评应用实现了基于令牌的身份验证。在下一章中，我们将通过学习如何生成 CSV、PDF 和其他二进制文件类型来扩展 Bookr 为用户的功能。
