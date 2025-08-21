# 第五章：高级应用程序结构

我们的应用程序已经从一个非常简单的例子发展成一个可扩展的基础，可以很容易地构建强大的功能。然而，将整个应用程序代码都放在一个文件中会不必要地使我们的代码混乱。为了使应用程序代码更清晰、更易理解，我们将把整个代码转换为一个 Python 模块，并将代码分割成多个文件。

# 项目作为一个模块

目前，你的文件夹结构应该是这样的：

```py
webapp/
  config.py
  database.db
  main.py
  manage.py
  env/
  migrations/
    versions/
  static/
    css/
    js/
  templates/
    blog/
```

为了将我们的代码转换为一个模块，我们的文件将被转换为这个文件夹结构：

```py
webapp/
  manage.py
  database.db
  webapp/
    __init__.py
    config.py
    forms.py
    models.py
    controllers/
      __init__.py
      blog.py
    static/
      css/
      js/
    templates/
      blog/
  migrations/
    versions/
```

我们将逐步创建这个文件夹结构。要做的第一个改变是在你的应用程序中创建一个包含模块的文件夹。在这个例子中，它将被称为`webapp`，但可以被称为除了博客以外的任何东西，因为控制器被称为博客。如果有两个要从中导入的博客对象，Python 将无法正确地从父目录中导入`blog.py`文件中的对象。

接下来，将`main.py`和`config.py`——静态和模板文件夹，分别移动到你的项目文件夹中，并创建一个控制器文件夹。我们还需要在`project`文件夹中创建`forms.py`和`models.py`文件，以及在控制器文件夹中创建一个`blog.py`文件。此外，`main.py`文件需要重命名为`__init__.py`。

文件名`__init__.py`看起来很奇怪，但它有一个特定的功能。在 Python 中，通过在文件夹中放置一个名为`__init__.py`的文件，可以将文件夹标记为模块。这允许程序从文件夹中的 Python 文件中导入对象和变量。

### 注意

要了解更多关于在模块中组织 Python 代码的信息，请参考官方文档[`docs.python.org/2/tutorial/modules.html#packages`](https://docs.python.org/2/tutorial/modules.html#packages)。

## 重构代码

让我们开始将我们的 SQLAlchemy 代码移动到`models.py`文件中。从`__init__.py`中剪切所有模型声明、标签表和数据库对象，并将它们与 SQLAlchemy 导入一起复制到`models.py`文件中。此外，我们的`db`对象将不再使用`app`对象作为参数进行初始化，因为`models.py`文件中没有`app`对象，导入它将导致循环导入。相反，我们将在初始化模型后将 app 对象添加到`db`对象中。这将在我们的`__init__.py`文件中实现。

你的`models.py`文件现在应该是这样的：

```py
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

tags = db.Table(
    'post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('post.id')),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'))
)

class User(db.Model):
    …

class Post(db.Model):
    …

class Comment(db.Model):
    …

class Tag(db.Model):
    …
```

接下来，`CommentForm`对象以及所有 WTForms 导入都应该移动到`forms.py`文件中。`forms.py`文件将保存所有 WTForms 对象在它们自己的文件中。

`forms.py`文件应该是这样的：

```py
from flask_wtf import Form
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired, Length

class CommentForm(Form):
    …
```

`blog_blueprint`数据函数、它的所有路由以及`sidebar_data`数据函数需要移动到控制器文件夹中的`blog.py`文件中。

`blog.py`文件现在应该是这样的：

```py
import datetime
from os import path
from sqlalchemy import func
from flask import render_template, Blueprint

from webapp.models import db, Post, Tag, Comment, User, tags
from webapp.forms import CommentForm

blog_blueprint = Blueprint(
    'blog',
    __name__,
    template_folder=path.join(path.pardir, 'templates', 'blog')
    url_prefix="/blog"
)

def sidebar_data():
    …
```

现在，每当创建一个新的蓝图时，可以在控制器文件夹中为其创建一个新的文件，将应用程序代码分解为逻辑组。此外，我们需要在控制器文件夹中创建一个空的`__init__.py`文件，以便将其标记为模块。

最后，我们专注于我们的`__init__.py`文件。`__init__.py`文件中应该保留的内容只有`app`对象的创建、`index`路由和`blog_blueprint`在`app`对象上的注册。然而，还有一件事要添加——数据库初始化。通过`db.init_app()`函数，我们将在导入`app`对象后将`app`对象添加到`db`对象中：

```py
from flask import Flask, redirect, url_for
from config import DevConfig

from models import db
from controllers.blog import blog_blueprint

app = Flask(__name__)
app.config.from_object(DevConfig)

db.init_app(app)

@app.route('/')
def index():
    return redirect(url_for('blog.home'))

app.register_blueprint(blog_blueprint)

if __name__ == '__main__':
    app.run()
```

在我们的新结构生效之前，有两件最后需要修复的事情，如果你使用的是 SQLite——`config.py`中的 SQLAlchemy 数据库 URL 需要更新，以及`manage.py`中的导入需要更新。因为 SQLite 数据库的 SQLAlchemy URL 是一个相对文件路径，所以它必须更改为：

```py
from os import path

class DevConfig(object):
    SQLALCHEMY_DATABASE_URI = 'sqlite://' + path.join(
        path.pardir,
        'database.db'
    )
```

要修复`manage.py`的导入，用以下内容替换`main.py`中的导入：

```py
from webapp import app
from webapp.models import db, User, Post, Tag, Comment
```

现在，如果你运行`manage.py`文件，你的应用将以新的结构运行。

# 应用工厂

现在我们以模块化的方式使用蓝图，然而，我们可以对我们的抽象进行另一个改进，即为我们的应用创建一个**工厂**。工厂的概念来自**面向对象编程**（**OOP**）世界，它简单地意味着一个函数或对象创建另一个对象。我们的应用工厂将接受我们在书的开头创建的`config`对象之一，并返回一个 Flask 应用对象。

### 注意

对象工厂设计是由现在著名的《设计模式：可复用面向对象软件的元素》一书所推广的。要了解更多关于这些设计模式以及它们如何帮助简化项目代码的信息，请查看[`en.wikipedia.org/wiki/Structural_pattern`](https://en.wikipedia.org/wiki/Structural_pattern)。

为我们的应用对象创建一个工厂函数有几个好处。首先，它允许环境的上下文改变应用的配置。当服务器创建应用对象进行服务时，它可以考虑服务器中任何必要的更改，并相应地改变提供给应用的配置对象。其次，它使测试变得更加容易，因为它允许快速测试不同配置的应用。第三，可以非常容易地创建使用相同配置的同一应用的多个实例。这对于需要在多个不同的服务器之间平衡网站流量的情况非常有用。

现在应用工厂的好处已经清楚，让我们修改我们的`__init__.py`文件来实现它：

```py
from flask import Flask, redirect, url_for
from models import db
from controllers.blog import blog_blueprint

def create_app(object_name):
    app = Flask(__name__)
    app.config.from_object(object_name)

    db.init_app(app)

    @app.route('/')
    def index():
        return redirect(url_for('blog.home'))

    app.register_blueprint(blog_blueprint)

    return app
```

对文件的更改非常简单；我们将代码包含在一个函数中，该函数接受一个`config`对象并返回一个应用对象。我们需要修改我们的`manage.py`文件，以便与`create_app`函数一起工作，如下所示：

```py
import os
from flask.ext.script import Manager, Server
from flask.ext.migrate import Migrate, MigrateCommand
from webapp import create_app
from webapp.models import db, User, Post, Tag, Comment

# default to dev config
env = os.environ.get('WEBAPP_ENV', 'dev')
app = create_app('webapp.config.%sConfig' % env.capitalize())
…
manager = Manager(app)
manager.add_command("server", Server())
```

当我们创建配置对象时，提到了应用运行的环境可能会改变应用的配置。这段代码有一个非常简单的例子，展示了环境变量的功能，其中加载了一个环境变量，并确定要给`create_app`函数的`config`对象。环境变量是 Bash 中的**全局变量**，可以被许多不同的程序访问。它们可以用以下语法在 Bash 中设置：

```py
$ export WEBAPP_ENV="dev"

```

读取变量时：

```py
$ echo $WEBAPP_ENV
dev

```

您也可以按以下方式轻松删除变量：

```py
$ unset $WEBAPP_ENV
$ echo $WEBAPP_ENV

```

在生产服务器上，您将把`WEBAPP_ENV`设置为`prod`。一旦在第十三章 *部署 Flask 应用*中部署到生产环境，并且当我们到达第十二章 *测试 Flask 应用*时，即可清楚地看到这种设置的真正威力，该章节涵盖了对项目进行测试。

# 总结

我们已经将我们的应用转变为一个更加可管理和可扩展的结构，这将在我们继续阅读本书并添加更多高级功能时为我们节省许多麻烦。在下一章中，我们将为我们的应用添加登录和注册系统，以及其他功能，使我们的网站更加安全。
