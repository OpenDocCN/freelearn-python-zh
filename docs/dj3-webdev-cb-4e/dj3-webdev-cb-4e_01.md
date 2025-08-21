# 第一章：开始使用 Django 3.0

在本章中，我们将涵盖以下主题：

+   使用虚拟环境

+   创建项目文件结构

+   使用 pip 处理项目依赖关系

+   为开发、测试、暂存和生产环境配置设置

+   在设置中定义相对路径

+   处理敏感设置

+   在项目中包含外部依赖项

+   动态设置`STATIC_URL`

+   将 UTF-8 设置为 MySQL 配置的默认编码

+   创建 Git 的`ignore`文件

+   删除 Python 编译文件

+   遵守 Python 文件中的导入顺序

+   创建应用程序配置

+   定义可覆盖的应用程序设置

+   使用 Docker 容器处理 Django、Gunicorn、Nginx 和 PostgreSQL

# 介绍

在本章中，我们将看到一些有价值的实践，用于使用 Python 3 在 Django 3.0 中启动新项目时遵循。我们选择了处理可扩展项目布局、设置和配置的最有用的方法，无论是使用 virtualenv 还是 Docker 来管理您的项目。

我们假设您已经熟悉 Django、Git 版本控制、MySQL 以及 PostgreSQL 数据库和命令行使用的基础知识。我们还假设您使用的是基于 Unix 的操作系统，如 macOS 或 Linux。在 Unix-based 平台上开发 Django 更有意义，因为 Django 网站很可能会发布在 Linux 服务器上，这意味着您可以建立在开发或部署时都能工作的例行程序。如果您在 Windows 上本地使用 Django，例行程序是类似的；但是它们并不总是相同的。

无论您的本地平台如何，使用 Docker 作为开发环境都可以通过部署改善应用程序的可移植性，因为 Docker 容器内的环境可以精确匹配部署服务器的环境。我们还应该提到，在本章的配方中，我们假设您已经在本地机器上安装了适当的版本控制系统和数据库服务器，无论您是否使用 Docker 进行开发。

# 技术要求

要使用本书的代码，您将需要最新稳定版本的 Python，可以从[`www.python.org/downloads/`](https://www.python.org/downloads/)下载。在撰写本文时，最新版本为 3.8.X。您还需要 MySQL 或 PostgreSQL 数据库。您可以从[`dev.mysql.com/downloads/`](https://dev.mysql.com/downloads/)下载 MySQL 数据库服务器。PostgreSQL 数据库服务器可以从[`www.postgresql.org/download/`](https://www.postgresql.org/download/)下载。其他要求将在特定的配方中提出。

您可以在 GitHub 存储库的`ch01`目录中找到本章的所有代码，网址为[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 使用虚拟环境

很可能您会在计算机上开发多个 Django 项目。某些模块，如 virtualenv、setuptools、wheel 或 Ansible，可以安装一次，然后为所有项目共享。其他模块，如 Django、第三方 Python 库和 Django 应用程序，需要保持彼此隔离。virtualenv 工具是一个实用程序，它将所有 Python 项目分开，并将它们保留在自己的领域中。在本配方中，我们将看到如何使用它。

# 准备工作

要管理 Python 包，您将需要 pip。如果您使用的是 Python 3.4+，则它将包含在您的 Python 安装中。如果您使用的是其他版本的 Python，可以通过执行[http:/​/​pip.​readthedocs.​org/​en/​stable/installing/](https://pip.pypa.io/en/stable/installing/)的安装说明来安装 pip。让我们升级共享的 Python 模块、pip、setuptools 和 wheel：

```py
$ sudo pip3 install --upgrade pip setuptools wheel
```

虚拟环境已经内置到 Python 3.3 版本以来。

# 如何做...

安装完先决条件后，创建一个目录，其中将存储所有 Django 项目，例如，在您的主目录下创建`projects`。创建目录后，请按以下步骤进行：

1.  转到新创建的目录并创建一个使用共享系统站点包的虚拟环境：

```py
$ cd ~/projects
$ mkdir myproject_website
$ cd myproject_website
$ python3 -m venv env
```

1.  要使用您新创建的虚拟环境，您需要在当前 shell 中执行激活脚本。可以使用以下命令完成：

```py
$ source env/bin/activate
```

1.  根据您使用的 shell，`source`命令可能不可用。另一种使用以下命令来源文件的方法是具有相同结果的（注意点和`env`之间的空格）：

```py
$ . env/bin/activate
```

1.  您将看到命令行工具的提示前缀为项目名称，如下所示：

```py
(env)$
```

1.  要退出虚拟环境，请输入以下命令：

```py
(env)$ deactivate
```

# 它是如何工作的...

创建虚拟环境时，会创建一些特定目录（`bin`、`include`和`lib`），以存储 Python 安装的副本，并定义一些共享的 Python 路径。激活虚拟环境后，您使用`pip`或`easy_install`安装的任何内容都将放在虚拟环境的站点包中，并且不会放在 Python 安装的全局站点包中。

要在虚拟环境中安装最新的 Django 3.0.x，请输入以下命令：

```py
(env)$ pip install "Django~=3.0.0"
```

# 另请参阅

+   *创建项目文件结构*食谱

+   第十二章中的*使用 Docker 容器进行 Django、Gunicorn、Nginx 和 PostgreSQL 部署*食谱

+   第十二章中的*使用 mod_wsgi 在 Apache 上部署分段环境*食谱，*部署*

+   第十二章*, 部署*中的*使用 Apache 和 mod_wsgi 部署生产环境*食谱

+   第十二章*, 部署*中的*在 Nginx 和 Gunicorn 上部署分段环境*食谱

+   第十二章*, 部署*中的*在 Nginx 和 Gunicorn 上部署生产环境*食谱

# 创建项目文件结构

为您的项目保持一致的文件结构可以使您更有条理、更高效。当您定义了基本工作流程后，您可以更快地进入业务逻辑并创建出色的项目。

# 准备工作

如果还没有，请创建一个`~/projects`目录，您将在其中保存所有 Django 项目（您可以在*使用虚拟环境*食谱中了解更多信息）。

然后，为您的特定项目创建一个目录，例如`myproject_website`。在那里的`env`目录中启动虚拟环境。激活它并在其中安装 Django，如前面的食谱中所述。我们建议添加一个`commands`目录，用于与项目相关的本地 shell 脚本，一个用于数据库转储的`db_backups`目录，一个用于网站设计文件的`mockups`目录，最重要的是一个用于您的 Django 项目的`src`目录。

# 如何做...

按照以下步骤为您的项目创建文件结构：

1.  激活虚拟环境后，转到`src`目录并启动一个新的 Django 项目，如下所示：

```py
(env)$ django-admin.py startproject myproject
```

执行的命令将创建一个名为`myproject`的目录，其中包含项目文件。该目录将包含一个名为`myproject`的 Python 模块。为了清晰和方便起见，我们将顶级目录重命名为`django-myproject`。这是您将放入版本控制的目录，因此它将有一个`.git`或类似命名的子目录。

1.  在`django-myproject`目录中，创建一个`README.md`文件，以向新的开发者描述您的项目。

1.  `django-myproject`目录还将包含以下内容：

+   您项目的 Python 包名为`myproject`。

+   您的项目的 pip 要求与 Django 框架和其他外部依赖项（在*使用 pip 处理项目依赖*食谱中了解更多）。

+   `LICENSE`文件中的项目许可证。如果您的项目是开源的，可以从[`choosealicense.com`](https://choosealicense.com)中选择最受欢迎的许可证之一。

1.  在您的项目的根目录`django-myproject`中，创建以下内容：

+   用于项目上传的`media`目录

+   用于收集静态文件的`static`目录

+   用于项目翻译的`locale`目录

+   用于无法使用 pip 要求的项目中包含的外部依赖的`externals`目录

1.  `myproject`目录应包含以下目录和文件：

+   `apps`目录，您将在其中放置项目的所有内部 Django 应用程序。建议您有一个名为`core`或`utils`的应用程序，用于项目的共享功能。

+   用于项目设置的`settings`目录（在*配置开发、测试、暂存和生产环境的设置*食谱中了解更多）。

+   用于特定项目的静态文件的`site_static`目录。

+   项目的 HTML 模板的`templates`目录。

+   项目的 URL 配置的`urls.py`文件。

+   项目的 Web 服务器配置的`wsgi.py`文件。

1.  在您的`site_static`目录中，创建`site`目录作为站点特定静态文件的命名空间。然后，我们将在其中的分类子目录之间划分静态文件。例如，参见以下内容：

+   Sass 文件的`scss`（可选）

+   用于生成压缩的**层叠样式表**（**CSS**）的`css`

+   用于样式图像、网站图标和标志的`img`

+   项目的 JavaScript 的`js`

+   `vendor`用于任何第三方模块，结合所有类型的文件，例如 TinyMCE 富文本编辑器

1.  除了`site`目录，`site_static`目录还可能包含第三方应用程序的覆盖静态目录，例如，它可能包含`cms`，它会覆盖 Django CMS 的静态文件。要从 Sass 生成 CSS 文件并压缩 JavaScript 文件，您可以使用带有图形用户界面的 CodeKit ([`codekitapp.com/`](https://codekitapp.com/))或 Prepros ([`prepros.io/`](https://prepros.io/))应用程序。

1.  将按应用程序分隔的模板放在您的`templates`目录中。如果模板文件表示页面（例如，`change_item.html`或`item_list.html`），则直接将其放在应用程序的模板目录中。如果模板包含在另一个模板中（例如，`similar_items.html`），则将其放在`includes`子目录中。此外，您的模板目录可以包含一个名为`utils`的目录，用于全局可重用的片段，例如分页和语言选择器。

# 它是如何工作的...

完整项目的整个文件结构将类似于以下内容：

```py
myproject_website/
├── commands/
├── db_backups/
├── mockups/
├── src/
│   └── django-myproject/
│       ├── externals/
│       │   ├── apps/
│       │   │   └── README.md
│       │   └── libs/
│       │       └── README.md
│       ├── locale/
│       ├── media/
│       ├── myproject/
│       │   ├── apps/
│       │   │   ├── core/
│       │   │   │   ├── __init__.py
│       │   │   │   └── versioning.py
│       │   │   └── __init__.py
│       │   ├── settings/
│       │   │   ├── __init__.py
│       │   │   ├── _base.py
│       │   │   ├── dev.py
│       │   │   ├── production.py
│       │   │   ├── sample_secrets.json
│       │   │   ├── secrets.json
│       │   │   ├── staging.py
│       │   │   └── test.py
│       │   ├── site_static/
│       │   │   └── site/
│       │   │  django-admin.py startproject myproject     ├── css/
│       │   │       │   └── style.css
│       │   │       ├── img/
│       │   │       │   ├── favicon-16x16.png
│       │   │       │   ├── favicon-32x32.png
│       │   │       │   └── favicon.ico
│       │   │       ├── js/
│       │   │       │   └── main.js
│       │   │       └── scss/
│       │   │           └── style.scss
│       │   ├── templates/
│       │   │   ├── base.html
│       │   │   └── index.html
│       │   ├── __init__.py
│       │   ├── urls.py
│       │   └── wsgi.py
│       ├── requirements/
│       │   ├── _base.txt
│       │   ├── dev.txt
│       │   ├── production.txt
│       │   ├── staging.txt
│       │   └── test.txt
│       ├── static/
│       ├── LICENSE
│       └── manage.py
└── env/
```

# 还有更多...

为了加快按照我们刚刚描述的方式创建项目的速度，您可以使用来自[`github.com/archatas/django-myproject`](https://github.com/archatas/django-myproject)的项目样板。下载代码后，执行全局搜索并替换`myproject`为您的项目的有意义的名称，然后您就可以开始了。

# 另请参阅

+   使用 pip 处理项目依赖的食谱

+   在项目中包含外部依赖的食谱

+   配置开发、测试、暂存和生产环境的设置

+   第十二章**部署**中的*在 Apache 上使用 mod_wsgi 部署暂存环境*食谱

+   第十二章*部署*中的*在 Apache 上使用 mod_wsgi 部署生产环境*食谱

+   第十二章*部署*中的*在 Nginx 和 Gunicorn 上部署暂存环境*食谱

+   在第十二章*，部署*中的*在 Nginx 和 Gunicorn 上部署生产环境*配方

# 使用 pip 处理项目依赖关系

安装和管理 Python 包的最方便的工具是 pip。与逐个安装包不同，可以将要安装的包的列表定义为文本文件的内容。我们可以将文本文件传递给 pip 工具，然后 pip 工具将自动处理列表中所有包的安装。采用这种方法的一个附加好处是，包列表可以存储在版本控制中。

一般来说，拥有一个与您的生产环境直接匹配的单个要求文件是理想的，通常也足够了。您可以在开发机器上更改版本或添加和删除依赖项，然后通过版本控制进行管理。这样，从一个依赖项集（和相关的代码更改）到另一个依赖项集的转换可以像切换分支一样简单。

在某些情况下，环境的差异足够大，您将需要至少两个不同的项目实例：

+   在这里创建新功能的开发环境

+   通常称为托管服务器中的生产环境的公共网站环境

可能有其他开发人员的开发环境，或者在开发过程中需要的特殊工具，但在生产中是不必要的。您可能还需要测试和暂存环境，以便在本地测试项目和在类似公共网站的设置中进行测试。

为了良好的可维护性，您应该能够为开发、测试、暂存和生产环境安装所需的 Python 模块。其中一些模块将是共享的，而另一些将特定于一部分环境。在本配方中，我们将学习如何为多个环境组织项目依赖项，并使用 pip 进行管理。

# 准备工作

在使用此配方之前，您需要准备好一个已安装 pip 并激活了虚拟环境的 Django 项目。有关如何执行此操作的更多信息，请阅读*使用虚拟环境*配方。

# 如何做...

逐步执行以下步骤，为您的虚拟环境 Django 项目准备 pip 要求：

1.  让我们进入您正在版本控制下的 Django 项目，并创建一个包含以下文本文件的 `requirements` 目录：

+   `_base.txt` 用于共享模块

+   `dev.txt` 用于开发环境

+   `test.txt` 用于测试环境

+   `staging.txt` 用于暂存环境

+   `production.txt` 用于生产环境

1.  编辑 `_base.txt` 并逐行添加在所有环境中共享的 Python 模块：

```py
# requirements/_base.txt
Django~=3.0.4
djangorestframework
-e git://github.com/omab/python-social-auth.git@6b1e301c79#egg=python-social-auth
```

1.  如果特定环境的要求与 `_base.txt` 中的要求相同，请在该环境的要求文件中添加包括 `_base.txt` 的行，如下例所示：

```py
# requirements/production.txt
-r _base.txt
```

1.  如果环境有特定要求，请在 `_base.txt` 包含之后添加它们，如下面的代码所示：

```py
# requirements/dev.txt
-r _base.txt
coverage
django-debug-toolbar
selenium
```

1.  您可以在虚拟环境中运行以下命令，以安装开发环境所需的所有依赖项（或其他环境的类似命令），如下所示：

```py
(env)$ pip install -r requirements/dev.txt
```

# 它是如何工作的...

前面的 `pip install` 命令，无论是在虚拟环境中显式执行还是在全局级别执行，都会从 `requirements/_base.txt` 和 `requirements/dev.txt` 下载并安装所有项目依赖项。如您所见，您可以指定您需要的 Django 框架的模块版本，甚至可以直接从 Git 存储库的特定提交中安装，就像我们的示例中对 `python-social-auth` 所做的那样。

在项目中有很多依赖项时，最好坚持使用 Python 模块发布版本的狭窄范围。然后，您可以更有信心地确保项目的完整性不会因依赖项的更新而受到破坏，这可能会导致冲突或向后不兼容。当部署项目或将其移交给新开发人员时，这一点尤为重要。

如果您已经手动逐个使用 pip 安装了项目要求，您可以在虚拟环境中使用以下命令生成`requirements/_base.txt`文件：

```py
(env)$ pip freeze > requirements/_base.txt
```

# 还有更多...

如果您想保持简单，并确信对于所有环境，您将使用相同的依赖项，您可以使用名为`requirements.txt`的一个文件来定义生成要求，如下所示：

```py
(env)$ pip freeze > requirements.txt
```

要在新的虚拟环境中安装模块，只需使用以下命令：

```py
(env)$ pip install -r requirements.txt
```

如果您需要从另一个版本控制系统或本地路径安装 Python 库，则可以从官方文档[`pip.pypa.io/en/stable/user_guide/`](https://pip.pypa.io/en/stable/user_guide)了解有关 pip 的更多信息。

另一种越来越受欢迎的管理 Python 依赖项的方法是 Pipenv。您可以在[`github.com/pypa/pipenv`](https://github.com/pypa/pipenv)获取并了解它。

# 另请参阅

+   *使用虚拟环境* 教程

+   *Django，Gunicorn，Nginx 和 PostgreSQL 的 Docker 容器工作* 教程

+   *在项目中包含外部依赖项* 教程

+   *配置开发、测试、暂存和生产环境的设置* 教程

# 配置开发、测试、暂存和生产环境的设置

如前所述，您将在开发环境中创建新功能，在测试环境中测试它们，然后将网站放到暂存服务器上，让其他人尝试新功能。然后，网站将部署到生产服务器供公众访问。每个环境都可以有特定的设置，您将在本教程中学习如何组织它们。

# 准备工作

在 Django 项目中，我们将为每个环境创建设置：开发、测试、暂存和生产。

# 如何做到...

按照以下步骤配置项目设置：

1.  在`myproject`目录中，创建一个`settings` Python 模块，并包含以下文件：

+   `__init__.py` 使设置目录成为 Python 模块。

+   `_base.py` 用于共享设置

+   `dev.py` 用于开发设置

+   `test.py` 用于测试设置

+   `staging.py` 用于暂存设置

+   `production.py` 用于生产设置

1.  将自动在启动新的 Django 项目时创建的`settings.py`的内容复制到`settings/_base.py`。然后，删除`settings.py`。

1.  将`settings/_base.py`中的`BASE_DIR`更改为指向上一级。它应该首先如下所示：

```py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

更改后，它应如下所示：

```py
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
```

1.  如果一个环境的设置与共享设置相同，那么只需

从`_base.py`中导入所有内容，如下所示：

```py
# myproject/settings/production.py
from ._base import *
```

1.  在其他文件中应用您想要附加或覆盖的特定环境的设置，例如，开发环境设置应该放在`dev.py`中，如下面的代码片段所示：

```py
# myproject/settings/dev.py
from ._base import *
EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
```

1.  修改`manage.py`和`myproject/wsgi.py`文件，以默认使用其中一个环境设置，方法是更改以下行：

```py
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
```

1.  您应该将此行更改为以下内容：

```py
os.environ.setdefault('DJANGO_SETTINGS_MODULE',  'myproject.settings.production')
```

# 它是如何工作的...

默认情况下，Django 管理命令使用`myproject/settings.py`中的设置。使用此食谱中定义的方法，我们可以将所有环境中所需的非敏感设置保留在`config`目录中，并将`settings.py`文件本身忽略在版本控制中，它只包含当前开发、测试、暂存或生产环境所需的设置。

对于每个环境，建议您单独设置`DJANGO_SETTINGS_MODULE`环境变量，可以在 PyCharm 设置中、`env/bin/activate`脚本中或`.bash_profile`中设置。

# 另请参阅

+   *为 Django、Gunicorn、Nginx 和 PostgreSQL 工作的 Docker 容器*食谱

+   *处理敏感设置*食谱

+   *在设置中定义相对路径*食谱

+   *创建 Git 忽略文件*食谱

# 在设置中定义相对路径

Django 要求您在设置中定义不同的文件路径，例如媒体的根目录、静态文件的根目录、模板的路径和翻译文件的路径。对于项目的每个开发者，路径可能会有所不同，因为虚拟环境可以设置在任何地方，用户可能在 macOS、Linux 或 Windows 上工作。即使您的项目包装在 Docker 容器中，定义绝对路径会降低可维护性和可移植性。无论如何，有一种方法可以动态定义这些路径，使它们相对于您的 Django 项目目录。

# 准备工作

已经启动了一个 Django 项目并打开了`settings/_base.py`。

# 如何做...

相应地修改您的与路径相关的设置，而不是将路径硬编码到本地目录中，如下所示：

```py
# settings/_base.py
import os
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# ...
TEMPLATES = [{
    # ...
    DIRS: [
       os.path.join(BASE_DIR, 'myproject', 'templates'),
    ],
    # ...
}]
# ...
LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale'),
]
# ...
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'myproject', 'site_static'),
]
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

# 它是如何工作的...

默认情况下，Django 设置包括`BASE_DIR`值，这是一个绝对路径，指向包含`manage.py`的目录（通常比`settings.py`文件高一级，或比`settings/_base.py`高两级）。然后，我们使用`os.path.join()`函数将所有路径设置为相对于`BASE_DIR`。

根据我们在*创建项目文件结构*食谱中设置的目录布局，我们将在一些先前的示例中插入`'myproject'`作为中间路径段，因为相关文件夹是在其中创建的。

# 另请参阅

+   *创建项目文件结构*食谱

+   *为 Django、Gunicorn、Nginx 和 PostgreSQL 工作的 Docker 容器*食谱

+   *在项目中包含外部依赖项*食谱

# 处理敏感设置

在配置 Django 项目时，您肯定会处理一些敏感信息，例如密码和 API 密钥。不建议将这些信息放在版本控制下。存储这些信息的主要方式有两种：在环境变量中和在单独的未跟踪文件中。在这个食谱中，我们将探讨这两种情况。

# 准备工作

项目的大多数设置将在所有环境中共享并保存在版本控制中。这些可以直接在设置文件中定义；但是，将有一些设置是特定于项目实例的环境或敏感的，并且需要额外的安全性，例如数据库或电子邮件设置。我们将使用环境变量来公开这些设置。

# 如何做...

从环境变量中读取敏感设置，执行以下步骤：

1.  在`settings/_base.py`的开头，定义`get_secret()`函数如下：

```py
# settings/_base.py
import os
from django.core.exceptions import ImproperlyConfigured

def get_secret(setting):
    """Get the secret variable or return explicit exception."""
    try:
        return os.environ[setting]
    except KeyError:
        error_msg = f'Set the {setting} environment variable'
        raise ImproperlyConfigured(error_msg)
```

1.  然后，每当您需要定义敏感值时，使用`get_secret()`函数，如下例所示：

```py
SECRET_KEY = get_secret('DJANGO_SECRET_KEY')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': get_secret('DATABASE_NAME'),
        'USER': get_secret('DATABASE_USER'),
        'PASSWORD': get_secret('DATABASE_PASSWORD'),
        'HOST': 'db',
        'PORT': '5432',
    }
}
```

# 它是如何工作的...

如果在没有设置环境变量的情况下运行 Django 管理命令，您将看到一个错误消息，例如设置`DJANGO_SECRET_KEY`环境变量。

您可以在 PyCharm 配置、远程服务器配置控制台、`env/bin/activate`脚本、`.bash_profile`或直接在终端中设置环境变量，如下所示：

```py
$ export DJANGO_SECRET_KEY="change-this-to-50-characters-long-random-
  string"
$ export DATABASE_NAME="myproject"
$ export DATABASE_USER="myproject"
$ export DATABASE_PASSWORD="change-this-to-database-password"
```

请注意，您应该在 Django 项目配置中使用`get_secret()`函数来获取所有密码、API 密钥和任何其他敏感信息。

# 还有更多...

您还可以使用包含敏感信息的文本文件，这些文件不会被版本控制跟踪，而不是环境变量。它们可以是 YAML、INI、CSV 或 JSON 文件，放置在硬盘的某个位置。例如，对于 JSON 文件，您可以有`get_secret()`函数，如下所示：

```py
# settings/_base.py
import os
import json

with open(os.path.join(os.path.dirname(__file__), 'secrets.json'), 'r') 
 as f:
    secrets = json.loads(f.read())

def get_secret(setting):
    """Get the secret variable or return explicit exception."""
    try:
        return secrets[setting]
    except KeyError:
        error_msg = f'Set the {setting} secret variable'
        raise ImproperlyConfigured(error_msg)
```

这将从设置目录中的`secrets.json`文件中读取，并期望它至少具有以下结构：

```py
{
    "DATABASE_NAME": "myproject",
    "DATABASE_USER": "myproject",
    "DATABASE_PASSWORD": "change-this-to-database-password",
    "DJANGO_SECRET_KEY": "change-this-to-50-characters-long-random-string"
}
```

确保`secrets.json`文件被版本控制忽略，但为了方便起见，您可以创建带有空值的`sample_secrets.json`并将其放在版本控制下：

```py
{
    "DATABASE_NAME": "",
    "DATABASE_USER": "",
    "DATABASE_PASSWORD": "",
    "DJANGO_SECRET_KEY": "change-this-to-50-characters-long-random-string"
}
```

# 另请参阅

+   *创建项目文件结构*配方

+   *Docker 容器中的 Django、Gunicorn、Nginx 和 PostgreSQL*配方

# 在项目中包含外部依赖项

有时，您无法使用 pip 安装外部依赖项，必须直接将其包含在项目中，例如以下情况：

+   当您有一个修补过的第三方应用程序，您自己修复了一个错误或添加了一个未被项目所有者接受的功能时

+   当您需要使用无法在**Python 软件包索引**（**PyPI**）或公共版本控制存储库中访问的私有应用程序时

+   当您需要使用 PyPI 中不再可用的依赖项的旧版本时

*在项目中包含外部依赖项*可以确保每当开发人员升级依赖模块时，所有其他开发人员都将在版本控制系统的下一个更新中收到升级后的版本。

# 准备工作

您应该从虚拟环境下的 Django 项目开始。

# 如何做...

逐步执行以下步骤，针对虚拟环境项目：

1.  如果尚未这样做，请在 Django 项目目录`django-myproject`下创建一个`externals`目录。

1.  然后，在其中创建`libs`和`apps`目录。`libs`目录用于项目所需的 Python 模块，例如 Boto、Requests、Twython 和 Whoosh。`apps`目录用于第三方 Django 应用程序，例如 Django CMS、Django Haystack 和 django-storages。

我们强烈建议您在`libs`和`apps`目录中创建`README.md`文件，其中提到每个模块的用途、使用的版本或修订版本以及它来自哪里。

1.  目录结构应该类似于以下内容：

```py
externals/
 ├── apps/
 │   ├── cms/
 │   ├── haystack/
 │   ├── storages/
 │   └── README.md
 └── libs/
     ├── boto/
     ├── requests/
     ├── twython/
     └── README.md
```

1.  下一步是将外部库和应用程序放在 Python 路径下，以便它们被识别为已安装。这可以通过在设置中添加以下代码来完成：

```py
# settings/_base.py
import os
import sys
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EXTERNAL_BASE = os.path.join(BASE_DIR, "externals")
EXTERNAL_LIBS_PATH = os.path.join(EXTERNAL_BASE, "libs")
EXTERNAL_APPS_PATH = os.path.join(EXTERNAL_BASE, "apps")
sys.path = ["", EXTERNAL_LIBS_PATH, EXTERNAL_APPS_PATH] + sys.path
```

# 工作原理...

如果您可以运行 Python 并导入该模块，则模块应该位于 Python 路径下。将模块放在 Python 路径下的一种方法是在导入位于不寻常位置的模块之前修改`sys.path`变量。根据设置文件指定的`sys.path`的值是一个目录列表，以空字符串开头表示当前目录，然后是项目中的目录，最后是 Python 安装的全局共享目录。您可以在 Python shell 中看到`sys.path`的值，如下所示：

```py
(env)$ python manage.py shell
>>> import sys
>>> sys.path
```

尝试导入模块时，Python 会在此列表中搜索模块，并返回找到的第一个结果。

因此，我们首先定义`BASE_DIR`变量，它是`django-myproject`的绝对路径，或者比`myproject/settings/_base.py`高三级。然后，我们定义`EXTERNAL_LIBS_PATH`和`EXTERNAL_APPS_PATH`变量，它们是相对于`BASE_DIR`的。最后，我们修改`sys.path`属性，将新路径添加到列表的开头。请注意，我们还将空字符串添加为第一个搜索路径，这意味着始终应首先检查任何模块的当前目录，然后再检查其他 Python 路径。

这种包含外部库的方式无法跨平台使用具有 C 语言绑定的 Python 软件包，例如`lxml`。对于这样的依赖关系，我们建议使用在*使用 pip 处理项目依赖关系*配方中介绍的 pip 要求。

# 参见

+   *创建项目文件结构*配方

+   *使用 Docker 容器处理 Django、Gunicorn、Nginx 和 PostgreSQL*配方

+   *使用 pip 处理项目依赖关系*配方

+   *在设置中定义相对路径*配方

+   *在[第十章](http://bells)*中的*Django shell*配方，铃声和口哨*

# 动态设置 STATIC_URL

如果将`STATIC_URL`设置为静态值，则每次更新 CSS 文件、JavaScript 文件或图像时，您和您的网站访问者都需要清除浏览器缓存才能看到更改。有一个绕过清除浏览器缓存的技巧，就是在`STATIC_URL`中显示最新更改的时间戳。每当代码更新时，访问者的浏览器将强制加载所有新的静态文件。

在这个配方中，我们将看到如何在`STATIC_URL`中放置 Git 用户的时间戳。

# 准备工作

确保您的项目处于 Git 版本控制下，并且在设置中定义了`BASE_DIR`，如*在设置中定义相对路径*配方中所示。

# 如何做...

将 Git 时间戳放入`STATIC_URL`设置的过程包括以下两个步骤：

1.  如果尚未这样做，请在 Django 项目中创建`myproject.apps.core`应用。您还应该在那里创建一个`versioning.py`文件：

```py
# versioning.py
import subprocess
from datetime import datetime

def get_git_changeset_timestamp(absolute_path):
    repo_dir = absolute_path
    git_log = subprocess.Popen(
        "git log --pretty=format:%ct --quiet -1 HEAD",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=repo_dir,
        universal_newlines=True,
    )

    timestamp = git_log.communicate()[0]
    try:
        timestamp = datetime.utcfromtimestamp(int(timestamp))
    except ValueError:
        # Fallback to current timestamp
        return datetime.now().strftime('%Y%m%d%H%M%S')
    changeset_timestamp = timestamp.strftime('%Y%m%d%H%M%S')
    return changeset_timestamp
```

1.  在设置中导入新创建的`get_git_changeset_timestamp()`函数，并将其用于`STATIC_URL`路径，如下所示：

```py
# settings/_base.py
from myproject.apps.core.versioning import get_git_changeset_timestamp
# ...
timestamp = get_git_changeset_timestamp(BASE_DIR)
STATIC_URL = f'/static/{timestamp}/'
```

# 它是如何工作的...

`get_git_changeset_timestamp()`函数以`absolute_path`目录作为参数，并调用`git log` shell 命令，参数是显示目录中 HEAD 修订的 Unix 时间戳。我们将`BASE_DIR`传递给函数，因为我们确信它处于版本控制之下。时间戳被解析，转换为由年、月、日、小时、分钟和秒组成的字符串，然后包含在`STATIC_URL`的定义中。

# 还有更多...

这种方法仅在您的每个环境中包含项目的完整 Git 存储库时才有效——在某些情况下，例如当您使用 Heroku 或 Docker 进行部署时，您无法访问远程服务器中的 Git 存储库和`git log`命令。为了使`STATIC_URL`具有动态片段，您必须从文本文件中读取时间戳，例如`myproject/settings/last-modified.txt`，并且应该在每次提交时更新该文件。

在这种情况下，您的设置将包含以下行：

```py
# settings/_base.py
with open(os.path.join(BASE_DIR, 'myproject', 'settings', 'last-update.txt'), 'r') as f:
    timestamp = f.readline().strip()

STATIC_URL = f'/static/{timestamp}/'
```

您可以通过预提交挂钩使 Git 存储库更新`last-modified.txt`。这是一个可执行的 bash 脚本，应该被称为`pre-commit`，并放置在`django-myproject/.git/hooks/`下：

```py
# django-myproject/.git/hooks/pre-commit
#!/usr/bin/env python
from subprocess import check_output, CalledProcessError
import os
from datetime import datetime

def root():
    ''' returns the absolute path of the repository root '''
    try:
        base = check_output(['git', 'rev-parse', '--show-toplevel'])
    except CalledProcessError:
        raise IOError('Current working directory is not a git repository')
    return base.decode('utf-8').strip()

def abspath(relpath):
    ''' returns the absolute path for a path given relative to the root of
        the git repository
    '''
    return os.path.join(root(), relpath)

def add_to_git(file_path):
    ''' adds a file to git '''
    try:
        base = check_output(['git', 'add', file_path])
    except CalledProcessError:
        raise IOError('Current working directory is not a git repository')
    return base.decode('utf-8').strip()

def main():
    file_path = abspath("myproject/settings/last-update.txt")

    with open(file_path, 'w') as f:
        f.write(datetime.now().strftime("%Y%m%d%H%M%S"))

    add_to_git(file_path)

if __name__ == '__main__':
    main()
```

每当您提交到 Git 存储库时，此脚本将更新`last-modified.txt`并将该文件添加到 Git 索引中。

# 参见

+   *创建 Git 忽略文件*配方

# 将 UTF-8 设置为 MySQL 配置的默认编码

MySQL 自称是最流行的开源数据库。在这个食谱中，我们将告诉你如何将 UTF-8 设置为它的默认编码。请注意，如果你不在数据库配置中设置这个编码，你可能会遇到这样的情况，即默认情况下使用 LATIN1 编码你的 UTF-8 编码数据。这将导致数据库错误，每当使用€等符号时。这个食谱还将帮助你免于在将数据库数据从 LATIN1 转换为 UTF-8 时遇到困难，特别是当你有一些表以 LATIN1 编码，另一些表以 UTF-8 编码时。

# 准备工作

确保 MySQL 数据库管理系统和**mysqlclient** Python 模块已安装，并且在项目设置中使用了 MySQL 引擎。

# 操作步骤...

在你喜欢的编辑器中打开`/etc/mysql/my.cnf` MySQL 配置文件，并确保以下设置在`[client]`、`[mysql]`和`[mysqld]`部分中设置如下：

```py
# /etc/mysql/my.cnf
[client]
default-character-set = utf8

[mysql]
default-character-set = utf8

[mysqld]
collation-server = utf8_unicode_ci
init-connect = 'SET NAMES utf8'
character-set-server = utf8
```

如果任何部分不存在，就在文件中创建它们。如果部分已经存在，就将这些设置添加到现有的配置中，然后在命令行工具中重新启动 MySQL，如下所示：

```py
$ /etc/init.d/mysql restart
```

# 它是如何工作的...

现在，每当你创建一个新的 MySQL 数据库时，数据库和所有的表都将默认设置为 UTF-8 编码。不要忘记在开发或发布项目的所有计算机上设置这一点。

# 还有更多...

在 PostgreSQL 中，默认的服务器编码已经是 UTF-8，但如果你想显式地创建一个带有 UTF-8 编码的 PostgreSQL 数据库，那么你可以使用以下命令来实现：

```py
$ createdb --encoding=UTF8 --locale=en_US.UTF-8 --template=template0 myproject
```

# 另请参阅

+   *创建项目文件结构*食谱

+   *使用 Docker 容器进行 Django、Gunicorn、Nginx 和 PostgreSQL 开发*食谱

# 创建 Git 忽略文件

Git 是最流行的分布式版本控制系统，你可能已经在你的 Django 项目中使用它。尽管你正在跟踪大部分文件的更改，但建议你将一些特定的文件和文件夹排除在版本控制之外。通常情况下，缓存、编译代码、日志文件和隐藏系统文件不应该在 Git 仓库中被跟踪。

# 准备工作

确保你的 Django 项目在 Git 版本控制下。

# 操作步骤...

使用你喜欢的文本编辑器，在你的 Django 项目的根目录创建一个`.gitignore`文件，并将以下文件和目录放在其中：

```py
# .gitignore ### Python template
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
db.sqlite3

# Sphinx documentation
docs/_build/

# IPython
profile_default/
ipython_config.py

# Environments
env/

# Media and Static directories
/media/
!/media/.gitkeep

/static/
!/static/.gitkeep

# Secrets
secrets.json
```

# 它是如何工作的...

`.gitignore`文件指定了应该被 Git 版本控制系统有意忽略的模式。我们在这个食谱中创建的`.gitignore`文件将忽略 Python 编译文件、本地设置、收集的静态文件和上传文件的媒体目录。

请注意，我们对媒体和静态文件有特殊的叹号语法：

```py
/media/
!/media/.gitkeep
```

这告诉 Git 忽略`/media/`目录，但保持`/media/.gitkeep`文件在版本控制下被跟踪。由于 Git 版本控制跟踪文件，而不是目录，我们使用`.gitkeep`来确保`media`目录将在每个环境中被创建，但不被跟踪。

# 另请参阅

+   *创建项目文件结构*食谱

+   *使用 Docker 容器进行 Django、Gunicorn、Nginx 和 PostgreSQL 开发*食谱

# 删除 Python 编译文件

当你第一次运行项目时，Python 会将所有的`*.py`代码编译成字节编译文件`*.pyc`，以便后续执行。通常情况下，当你改变`*.py`文件时，`*.pyc`会被重新编译；然而，有时当你切换分支或移动目录时，你需要手动清理编译文件。

# 准备工作

使用你喜欢的编辑器，在你的主目录中编辑或创建一个`.bash_profile`文件。

# 操作步骤...

1.  在`.bash_profile`的末尾添加这个别名，如下所示：

```py
# ~/.bash_profile alias delpyc='
find . -name "*.py[co]" -delete
find . -type d -name "__pycache__" -delete'
```

1.  现在，要清理 Python 编译文件，进入你的项目目录，在命令行上输入以下命令：

```py
(env)$ delpyc
```

# 它是如何工作的...

首先，我们创建一个 Unix 别名，用于搜索当前目录及其子目录中的`*.pyc`和`*.pyo`文件和`__pycache__`目录，并将其删除。当您在命令行工具中启动新会话时，将执行`.bash_profile`文件。

# 还有更多...

如果您想完全避免创建 Python 编译文件，可以在`.bash_profile`、`env/bin/activate`脚本或 PyCharm 配置中设置环境变量`PYTHONDONTWRITEBYTECODE=1`。

# 另请参阅

+   创建 Git 忽略文件的方法

# 尊重 Python 文件中的导入顺序

在创建 Python 模块时，保持与文件结构一致是一个良好的做法。这样可以使您和其他开发人员更容易阅读代码。本方法将向您展示如何构建导入结构。

# 准备就绪

创建虚拟环境并在其中创建 Django 项目。

# 如何做...

对于您正在创建的每个 Python 文件，请使用以下结构。将导入分类为以下几个部分：

```py
# System libraries
import os
import re
from datetime import datetime

# Third-party libraries
import boto
from PIL import Image

# Django modules
from django.db import models
from django.conf import settings

# Django apps
from cms.models import Page

# Current-app modules
from .models import NewsArticle
from . import app_settings
```

# 它是如何工作的...

我们有五个主要的导入类别，如下所示：

+   系统库用于 Python 默认安装的软件包

+   第三方库用于额外安装的 Python 包

+   Django 模块用于 Django 框架中的不同模块

+   Django 应用程序用于第三方和本地应用程序

+   当前应用程序模块用于从当前应用程序进行相对导入

# 还有更多...

在 Python 和 Django 中编码时，请使用 Python 代码的官方样式指南 PEP 8。您可以在[https:/​/​www.​python.​org/​dev/​peps/​pep-​0008/](https://www.python.org/dev/peps/pep-0008/)找到它。

# 另请参阅

+   使用 pip 处理项目依赖的方法

+   在项目中包含外部依赖的方法

# 创建应用程序配置

Django 项目由称为应用程序（或更常见的应用程序）的多个 Python 模块组成，这些模块结合了不同的模块化功能。每个应用程序都可以有模型、视图、表单、URL 配置、管理命令、迁移、信号、测试、上下文处理器、中间件等。Django 框架有一个应用程序注册表，其中收集了所有应用程序和模型，稍后用于配置和内省。自 Django 1.7 以来，有关应用程序的元信息可以保存在每个应用程序的`AppConfig`实例中。让我们创建一个名为`magazine`的示例应用程序，看看如何在那里使用应用程序配置。

# 准备就绪

您可以通过调用`startapp`管理命令或手动创建应用程序模块来创建 Django 应用程序：

```py
(env)$ cd myproject/apps/
(env)$ django-admin.py startapp magazine
```

创建`magazine`应用程序后，在`models.py`中添加`NewsArticle`模型，在`admin.py`中为模型创建管理，并在设置中的`INSTALLED_APPS`中放入`"myproject.apps.magazine"`。如果您还不熟悉这些任务，请学习官方的 Django 教程[`docs.djangoproject.com/en/3.0/intro/tutorial01/`](https://docs.djangoproject.com/en/3.0/intro/tutorial01/)。

# 如何做...

按照以下步骤创建和使用应用程序配置：

1.  修改`apps.py`文件并插入以下内容：

```py
# myproject/apps/magazine/apps.py
from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _

class MagazineAppConfig(AppConfig):
    name = "myproject.apps.magazine"
    verbose_name = _("Magazine")

    def ready(self):
        from . import signals
```

1.  编辑`magazine`模块中的`__init__.py`文件，包含以下内容：

```py
# myproject/apps/magazine/__init__.py
default_app_config = "myproject.apps.magazine.apps.MagazineAppConfig"
```

1.  让我们创建一个`signals.py`文件并在其中添加一些信号处理程序：

```py
# myproject/apps/magazine/signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.conf import settings

from .models import NewsArticle

@receiver(post_save, sender=NewsArticle)
def news_save_handler(sender, **kwargs):
    if settings.DEBUG:
        print(f"{kwargs['instance']} saved.")

@receiver(post_delete, sender=NewsArticle)
def news_delete_handler(sender, **kwargs):
    if settings.DEBUG:
        print(f"{kwargs['instance']} deleted.")
```

# 它是如何工作的...

当您运行 HTTP 服务器或调用管理命令时，会调用`django.setup()`。它加载设置，设置日志记录，并准备应用程序注册表。此注册表分为三个步骤初始化。Django 首先从设置中的`INSTALLED_APPS`导入每个项目的配置。这些项目可以直接指向应用程序名称或配置，例如`"myproject.apps.magazine"`或`"myproject.apps.magazine.apps.MagazineAppConfig"`。

然后 Django 尝试从`INSTALLED_APPS`中的每个应用程序导入`models.py`并收集所有模型。

最后，Django 运行 `ready()` 方法以进行每个应用程序配置。如果有的话，此方法在开发过程中是注册信号处理程序的好时机。`ready()` 方法是可选的。

在我们的示例中，`MagazineAppConfig` 类设置了 `magazine` 应用程序的配置。`name` 参数定义了当前应用程序的模块。`verbose_name` 参数定义了在 Django 模型管理中使用的人类名称，其中模型按应用程序进行呈现和分组。`ready()` 方法导入并激活信号处理程序，当处于 DEBUG 模式时，它会在终端中打印出 `NewsArticle` 对象已保存或已删除的消息。

# 还有更多...

在调用 `django.setup()` 后，您可以按如下方式从注册表中加载应用程序配置和模型：

```py
>>> from django.apps import apps as django_apps
>>> magazine_app_config = django_apps.get_app_config("magazine")
>>> magazine_app_config
<MagazineAppConfig: magazine>
>>> magazine_app_config.models_module
<module 'magazine.models' from '/path/to/myproject/apps/magazine/models.py'>
>>> NewsArticle = django_apps.get_model("magazine", "NewsArticle")
>>> NewsArticle
<class 'magazine.models.NewsArticle'>
```

您可以在官方 Django 文档中阅读有关应用程序配置的更多信息

[`docs.djangoproject.com/en/2.2/ref/applications/`](https://docs.djangoproject.com/en/2.2/ref/applications/)​。

# 另请参阅

+   *使用虚拟环境* 配方

+   *使用 Docker 容器进行 Django、Gunicorn、Nginx 和 PostgreSQL* 配方

+   *定义可覆盖的应用程序设置* 配方

+   第六章*，模型管理*

# 定义可覆盖的应用程序设置

此配方将向您展示如何定义应用程序的设置，然后可以在项目的设置文件中进行覆盖。这对于可重用的应用程序特别有用，您可以通过添加配置来自定义它们。

# 准备工作

按照*准备工作*中*创建应用程序配置* 配方中的步骤来创建您的 Django 应用程序。

# 如何做...

1.  如果只有一两个设置，可以在 `models.py` 中使用 `getattr()` 模式定义应用程序设置，或者如果设置很多并且想要更好地组织它们，可以在 `app_settings.py` 文件中定义：

```py
# myproject/apps/magazine/app_settings.py
from django.conf import settings
from django.utils.translation import gettext_lazy as _

# Example:
SETTING_1 = getattr(settings, "MAGAZINE_SETTING_1", "default value")

MEANING_OF_LIFE = getattr(settings, "MAGAZINE_MEANING_OF_LIFE", 42)

ARTICLE_THEME_CHOICES = getattr(
    settings,
    "MAGAZINE_ARTICLE_THEME_CHOICES",
    [
        ('futurism', _("Futurism")),
        ('nostalgia', _("Nostalgia")),
        ('sustainability', _("Sustainability")),
        ('wonder', _("Wonder")),
    ]
)
```

1.  `models.py` 将包含以下 `NewsArticle` 模型：

```py
# myproject/apps/magazine/models.py
from django.db import models
from django.utils.translation import gettext_lazy as _

class NewsArticle(models.Model):
    created_at = models.DateTimeField(_("Created at"),  
     auto_now_add=True)
    title = models.CharField(_("Title"), max_length=255)
    body = models.TextField(_("Body"))
    theme = models.CharField(_("Theme"), max_length=20)

    class Meta:
        verbose_name = _("News Article")
        verbose_name_plural = _("News Articles")

    def __str__(self):
        return self.title
```

1.  接下来，在 `admin.py` 中，我们将从 `app_settings.py` 导入并使用设置，如下所示：

```py
# myproject/apps/magazine/admin.py
from django import forms
from django.contrib import admin

from .models import NewsArticle

from .app_settings import ARTICLE_THEME_CHOICES

class NewsArticleModelForm(forms.ModelForm):
    theme = forms.ChoiceField(
        label=NewsArticle._meta.get_field("theme").verbose_name,
        choices=ARTICLE_THEME_CHOICES,
        required=not NewsArticle._meta.get_field("theme").blank,
    )
    class Meta:
        fields = "__all__"

@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
 form = NewsArticleModelForm
```

1.  如果要覆盖给定项目的 `ARTICLE_THEME_CHOICES` 设置，应在项目设置中添加 `MAGAZINE_ARTICLE_THEME_CHOICES`：

```py
# myproject/settings/_base.py
from django.utils.translation import gettext_lazy as _
# ...
MAGAZINE_ARTICLE_THEME_CHOICES = [
    ('futurism', _("Futurism")),
    ('nostalgia', _("Nostalgia")),
    ('sustainability', _("Sustainability")),
    ('wonder', _("Wonder")),
    ('positivity', _("Positivity")),
    ('solutions', _("Solutions")),
    ('science', _("Science")),
]
```

# 它是如何工作的...

`getattr(object, attribute_name[, default_value])` Python 函数尝试从 `object` 获取 `attribute_name` 属性，并在找不到时返回 `default_value`。我们尝试从 Django 项目设置模块中读取不同的设置，如果在那里找不到，则使用默认值。

请注意，我们本可以在 `models.py` 中为 `theme` 字段定义 `choices`，但我们改为在管理中创建自定义 `ModelForm` 并在那里设置 `choices`。这样做是为了避免在更改 `ARTICLE_THEME_CHOICES` 时创建新的数据库迁移。

# 另请参阅

+   *创建应用程序配置* 配方

+   第六章，*模型管理*

# 使用 Docker 容器进行 Django、Gunicorn、Nginx 和 PostgreSQL

Django 项目不仅依赖于 Python 要求，还依赖于许多系统要求，如 Web 服务器、数据库、服务器缓存和邮件服务器。在开发 Django 项目时，您需要确保所有环境和所有开发人员都安装了相同的要求。保持这些依赖项同步的一种方法是使用 Docker。使用 Docker，您可以为每个项目单独拥有数据库、Web 或其他服务器的不同版本。

Docker 是用于创建配置、定制的虚拟机的系统，称为容器。它允许我们精确复制任何生产环境的设置。Docker 容器是从所谓的 Docker 镜像创建的。镜像由层（或指令）组成，用于构建容器。可以有一个用于 PostgreSQL 的镜像，一个用于 Redis 的镜像，一个用于 Memcached 的镜像，以及一个用于您的 Django 项目的自定义镜像，所有这些镜像都可以与 Docker Compose 结合成相应的容器。

在这个示例中，我们将使用项目模板来设置一个 Django 项目，其中包括一个由 Nginx 和 Gunicorn 提供的 PostgreSQL 数据库，并使用 Docker Compose 来管理它们。

# 准备工作

首先，您需要安装 Docker Engine，按照[`www.docker.com/get-started`](https://www.docker.com/get-started)上的说明进行操作。这通常包括 Compose 工具，它可以管理需要多个容器的系统，非常适合完全隔离的 Django 项目。如果需要单独安装，Compose 的安装详细信息可在[`docs.docker.com/compose/install/`](https://docs.docker.com/compose/install/)上找到。

# 如何做...

让我们来探索 Django 和 Docker 模板：

1.  例如，从[`github.com/archatas/django_docker`](https://github.com/archatas/django_docker)下载代码到您的计算机的`~/projects/django_docker`目录。

如果您选择另一个目录，例如`myproject_docker`，那么您将需要全局搜索和替换`django_docker`为`myproject_docker`。

1.  打开`docker-compose.yml`文件。需要创建三个容器：`nginx`，`gunicorn`和`db`。如果看起来很复杂，不用担心；我们稍后会详细描述它：

```py
# docker-compose.yml
version: "3.7"

services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./config/nginx/conf.d:/etc/nginx/conf.d
      - static_volume:/home/myproject/static
      - media_volume:/home/myproject/media
    depends_on:
      - gunicorn

  gunicorn:
    build:
      context: .
      args:
        PIP_REQUIREMENTS: "${PIP_REQUIREMENTS}"
    command: bash -c "/home/myproject/env/bin/gunicorn --workers 3 
    --bind 0.0.0.0:8000 myproject.wsgi:application"
    depends_on:
      - db
    volumes:
      - static_volume:/home/myproject/static
      - media_volume:/home/myproject/media
    expose:
      - "8000"
    environment:
      DJANGO_SETTINGS_MODULE: "${DJANGO_SETTINGS_MODULE}"
      DJANGO_SECRET_KEY: "${DJANGO_SECRET_KEY}"
      DATABASE_NAME: "${DATABASE_NAME}"
      DATABASE_USER: "${DATABASE_USER}"
      DATABASE_PASSWORD: "${DATABASE_PASSWORD}"
      EMAIL_HOST: "${EMAIL_HOST}"
      EMAIL_PORT: "${EMAIL_PORT}"
      EMAIL_HOST_USER: "${EMAIL_HOST_USER}"
      EMAIL_HOST_PASSWORD: "${EMAIL_HOST_PASSWORD}"

  db:
    image: postgres:latest
    restart: always
    environment:
      POSTGRES_DB: "${DATABASE_NAME}"
      POSTGRES_USER: "${DATABASE_USER}"
      POSTGRES_PASSWORD: "${DATABASE_PASSWORD}"
    ports:
      - 5432
    volumes:
      - postgres_data:/var/lib/postgresql/data/

volumes:
  postgres_data:
  static_volume:
  media_volume:

```

1.  打开并阅读`Dockerfile`文件。这些是创建`gunicorn`容器所需的层（或指令）：

```py
# Dockerfile
# pull official base image
FROM python:3.8

# accept arguments
ARG PIP_REQUIREMENTS=production.txt

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip setuptools

# create user for the Django project
RUN useradd -ms /bin/bash myproject

# set current user
USER myproject

# set work directory
WORKDIR /home/myproject

# create and activate virtual environment
RUN python3 -m venv env

# copy and install pip requirements
COPY --chown=myproject ./src/myproject/requirements /home/myproject/requirements/
RUN ./env/bin/pip3 install -r /home/myproject/requirements/${PIP_REQUIREMENTS}

# copy Django project files
COPY --chown=myproject ./src/myproject /home/myproject/

```

1.  将`build_dev_example.sh`脚本复制到`build_dev.sh`并编辑其内容。这些是要传递给`docker-compose`脚本的环境变量：

```py
# build_dev.sh
#!/usr/bin/env bash
DJANGO_SETTINGS_MODULE=myproject.settings.dev \
DJANGO_SECRET_KEY="change-this-to-50-characters-long-
 random-string" \
DATABASE_NAME=myproject \
DATABASE_USER=myproject \
DATABASE_PASSWORD="change-this-too" \
PIP_REQUIREMENTS=dev.txt \
docker-compose up --detach --build
```

1.  在命令行工具中，为`build_dev.sh`添加执行权限并运行它以构建容器：

```py
$ chmod +x build_dev.sh
$ ./build_dev.sh
```

1.  如果您现在转到`http://0.0.0.0/en/`，您应该会在那里看到一个 Hello, World!页面。

导航到`http://0.0.0.0/en/admin/`时，您应该会看到以下内容：

```py
OperationalError at /en/admin/
 FATAL: role "myproject" does not exist
```

这意味着你必须在 Docker 容器中创建数据库用户和数据库。

1.  让我们 SSH 到`db`容器中，在 Docker 容器中创建数据库用户、密码和数据库本身：

```py
$ docker exec -it django_docker_db_1 bash
/# su - postgres
/$ createuser --createdb --password myproject
/$ createdb --username myproject myproject
```

当询问时，输入与`build_dev.sh`脚本中数据库相同的密码。

按下[*Ctrl* + *D*]两次以注销 PostgreSQL 用户和 Docker 容器。

如果您现在转到`http://0.0.0.0/en/admin/`，您应该会看到以下内容：

```py
ProgrammingError at /en/admin/ relation "django_session" does not exist LINE 1: ...ession_data", "django_session"."expire_date" FROM "django_se...
```

这意味着您必须运行迁移以创建数据库架构。

1.  SSH 到`gunicorn`容器中并运行必要的 Django 管理命令：

```py
$ docker exec -it django_docker_gunicorn_1 bash
$ source env/bin/activate
(env)$ python manage.py migrate
(env)$ python manage.py collectstatic
(env)$ python manage.py createsuperuser
```

回答管理命令提出的所有问题。

按下[*Ctrl* + *D*]两次以退出 Docker 容器。

如果您现在导航到`[`0.0.0.0/en/admin/`](http://0.0.0.0/en/admin/)`，您应该会看到 Django 管理界面，您可以使用刚刚创建的超级用户凭据登录。

1.  创建类似的脚本`build_test.sh`，`build_staging.sh`和`build_production.sh`，只有环境变量不同。

# 它是如何工作的...

模板中的代码结构类似于虚拟环境中的代码结构。项目源文件位于`src`目录中。我们有`git-hooks`目录用于预提交挂钩，用于跟踪最后修改日期和`config`目录用于容器中使用的服务的配置：

```py
django_docker
├── config/
│   └── nginx/
│       └── conf.d/
│           └── myproject.conf
├── git-hooks/
│   ├── install_hooks.sh
│   └── pre-commit
├── src/
│   └── myproject/
│       ├── locale/
│       ├── media/
│       ├── myproject/
│       │   ├── apps/
│       │   │   └── __init__.py
│       │   ├── settings/
│       │   │   ├── __init__.py
│       │   │   ├── _base.py
│       │   │   ├── dev.py
│       │   │   ├── last-update.txt
│       │   │   ├── production.py
│       │   │   ├── staging.py
│       │   │   └── test.py
│       │   ├── site_static/
│       │   │   └── site/
│       │   │       ├── css/
│       │   │       ├── img/
│       │   │       ├── js/
│       │   │       └── scss/
│       │   ├── templates/
│       │   │   ├── base.html
│       │   │   └── index.html
│       │   ├── __init__.py
│       │   ├── urls.py
│       │   └── wsgi.py
│       ├── requirements/
│       │   ├── _base.txt
│       │   ├── dev.txt
│       │   ├── production.txt
│       │   ├── staging.txt
│       │   └── test.txt
│       ├── static/
│       └── manage.py
├── Dockerfile
├── LICENSE
├── README.md
├── build_dev.sh
├── build_dev_example.sh
└── docker-compose.yml
```

主要的与 Docker 相关的配置位于`docker-compose.yml`和`Dockerfile`。Docker Compose 是 Docker 命令行 API 的包装器。`build_dev.sh`脚本构建并在端口`8000`下运行 Django 项目下的 Gunicorn WSGI HTTP 服务器，端口`80`下的 Nginx（提供静态和媒体文件并代理其他请求到 Gunicorn），以及端口`5432`下的 PostgreSQL 数据库。

在`docker-compose.yml`文件中，请求创建三个 Docker 容器：

+   `nginx`用于 Nginx Web 服务器

+   `gunicorn`用于 Django 项目的 Gunicorn Web 服务器

+   `db`用于 PostgreSQL 数据库

`nginx`和`db`容器将从位于[`hub.docker.com`](https://hub.docker.com)的官方镜像创建。它们具有特定的配置参数，例如它们运行的端口，环境变量，对其他容器的依赖以及卷。

Docker 卷是在重新构建 Docker 容器时保持不变的特定目录。需要为数据库数据文件，媒体，静态文件等定义卷。

`gunicorn`容器将根据`Dockerfile`中的指令构建，该指令由`docker-compose.yml`文件中的构建上下文定义。让我们检查每个层（或指令）：

+   `gunicorn`容器将基于`python:3.7`镜像

+   它将从`docker-compose.yml`文件中获取`PIP_REQUIREMENTS`作为参数

+   它将为容器设置环境变量

+   它将安装并升级 pip，setuptools 和 virtualenv

+   它将为 Django 项目创建一个名为`myproject`的系统用户

+   它将把`myproject`设置为当前用户

+   它将把`myproject`用户的主目录设置为当前工作目录

+   它将在那里创建一个虚拟环境

+   它将从基础计算机复制 pip 要求到 Docker 容器

+   它将安装当前环境的 pip 要求，由`PIP_REQUIREMENTS`变量定义

+   它将复制整个 Django 项目的源代码

`config/nginx/conf.d/myproject.conf`的内容将保存在`nginx`容器中的`/etc/nginx/conf.d/`下。这是 Nginx Web 服务器的配置，告诉它监听端口`80`（默认的 HTTP 端口）并将请求转发到端口`8000`上的 Gunicorn 服务器，除了请求静态或媒体内容：

```py
#/etc/nginx/conf.d/myproject.conf
upstream myproject {
    server django_docker_gunicorn_1:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://myproject;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_redirect off;
    }

    rewrite "/static/\d+/(.*)" /static/$1 last;

    location /static/ {
        alias /home/myproject/static/;
    }

    location /media/ {
        alias /home/myproject/media/;
    }
}
```

您可以在第十二章*，部署*中的*在 Nginx 和 Gunicorn 上部署暂存环境*和*在 Nginx 和 Gunicorn 上部署生产环境*配方中了解更多关于 Nginx 和 Gunicorn 配置的信息。

# 还有更多...

您可以使用`docker-compose down`命令销毁 Docker 容器，并使用构建脚本重新构建它们：

```py
$ docker-compose down
$ ./build_dev.sh
```

如果某些内容不符合预期，您可以使用`docker-compose logs`命令检查日志：

```py
$ docker-compose logs nginx
$ docker-compose logs gunicorn $ docker-compose logs db

```

要通过 SSH 连接到任何容器，您应该使用以下之一：

```py
$ docker exec -it django_docker_gunicorn_1 bash
$ docker exec -it django_docker_nginx_1 bash
$ docker exec -it django_docker_db_1 bash
```

您可以使用`docker cp`命令将文件和目录复制到 Docker 容器上的卷中，并从中复制出来：

```py
$ docker cp ~/avatar.png django_docker_gunicorn_1:/home/myproject/media/ $ docker cp django_docker_gunicorn_1:/home/myproject/media ~/Desktop/

```

如果您想更好地了解 Docker 和 Docker Compose，请查看官方文档[`docs.docker.com/`](https://docs.docker.com/)，特别是[`docs.docker.com/compose/`](https://docs.docker.com/compose/)。

# 另请参阅

+   *创建项目文件结构*配方

+   *在 Apache 上使用 mod_wsgi 部署暂存环境*配方在第十二章*，部署*

+   *在 Apache 上使用 mod_wsgi 部署生产环境*配方在第十二章*，部署*

+   *在 Nginx 和 Gunicorn 上部署暂存环境*配方在第十二章*，部署*

+   *在 Nginx 和 Gunicorn 上部署生产环境*配方在第十二章*，部署*
