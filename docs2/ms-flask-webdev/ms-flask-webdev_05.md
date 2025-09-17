# 第五章：高级应用程序结构

我们的应用程序已经从一个非常简单的示例转变为一个可扩展的基础，在这个基础上可以轻松构建强大的功能。然而，让我们的应用程序完全驻留在单个文件中是不必要的，这会使我们的代码变得杂乱。这是 Flask 的一大优点；您可以在单个文件上编写一个小的 REST 服务或 Web 应用程序，或者一个完整的商业应用程序。框架不会妨碍您，也不会强制任何项目布局。

为了使应用程序代码更清晰易懂，我们将整个代码转换为一个 Python 模块，并将每个特性转换为一个独立的模块。这种模块化方法使您能够轻松且可预测地进行扩展，因此新特性将有一个明显的位置和结构。在本章中，您将学习以下最佳实践：

+   创建一个易于扩展的模块化应用程序

+   应用程序工厂模式

# 模块化应用程序

目前，您的文件夹结构应该如下所示（查看前一章提供的代码）：

```py
./ 
  config.py 
  database.db 
  main.py 
  manage.py 
  env/ 
  migrations/ 
    versions/ 
  templates/ 
    blog/ 
```

为了将我们的代码转换为一个更模块化的应用程序，我们的文件结构如下：

```py
./ 
  manage.py
  main.py
  config.py 
 database.db 
  webapp/ 
    __init__.py
    blog/
      __init__.py 
      controllers.py
      forms.py
      models.py
    main/
      __init__.py
      controllers.py
    templates/ 
      blog/ 
  migrations/ 
    versions/ 
```

首先需要做的是在你的应用程序中创建一个文件夹，用于存放模块。在这个例子中，它将被命名为 `webapp`。

接下来，对于我们应用中的每个模块，我们将创建相应的 Python 模块。如果该模块是一个使用 Web 模板和表单的经典 Web 应用程序，我们将创建以下文件：

```py
./<MODULE_NAME>
  __init__.py -> Declare a python module
  controllers.py -> where our blueprint definition and views are
  models.py -> The module database models definitions
  forms.py -> All the module's web Forms 
```

理念是关注点的分离，因此每个模块将包含所有必要的视图（在 Flask 蓝图中声明并包含在内），Web 表单和模块。这种模块化结构将转化为 URI、模板和 Python 模块的可预测命名空间。继续以抽象方法进行推理，每个模块将具有以下特点：

+   Python 模块（包含 `__init__.py` 的文件夹）使用其名称：`MODULE_NAME`。在模块内部是一个 `controllers` Python 模块，它声明了一个名为 `<MODULE_NAME>_blueprint` 的蓝图，并将其附加到一个 URL，前缀为 `/<MODULE_NAME>`。

+   在 `templates` 内名为 `<MODULE_NAME>` 的模板文件夹。

这种模式将使代码对其他团队成员来说非常可预测，并且非常容易更改和扩展。如果您想创建一个全新的特性，只需使用建议的结构创建一个新的模块，所有团队成员将立即猜出新特性的 URI 命名空间，其中声明了所有视图，以及为该特性定义的数据库模型。如果发现了一些错误，您可以轻松地确定查找错误的位置，并且有一个更受限制的代码库需要关注。

# 代码重构

起初，看起来变化很大，但您会发现，考虑到之前解释的结构，这些变化简单且自然。

首先，我们已经将我们的 SQLAlchemy 代码移动到`blog 模块`文件夹内的`models.py`文件中。我们只想移动模型定义，而不是任何数据库初始化代码。所有初始化代码都将保留在主应用程序模块`webapp`中的`__init__.py`文件内。导入部分和数据库相关对象创建如下所示：

```py
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def page_not_found(error):
    return render_template('404.html'), 404

def create_app(config):
...
```

主要应用程序模块将负责创建 Flask 应用程序（工厂模式，将在下一节中解释）和初始化 SQLAlchemy。

`blog/models.py`文件将导入初始化的`db`对象：

```py
from .. import db

...
class User(db.Model):
...
class Post(db.Model):
...
class Comment(db.Model):
...
class Tag(db.Model):
...
```

接下来，应将`CommentForm`对象以及所有 WTForms 导入移动到`blog/forms.py`文件中。`forms.py`文件将包含与博客功能相关的所有 WTForms 对象。

`forms.py`文件应如下所示：

```py
from flask_wtf import Form 
from wtforms import StringField, TextAreaField 
from wtforms.validators import DataRequired, Length 

class CommentForm(Form): 
  ... 
```

`blog_blueprint`对象、所有其路由以及`sidebar_data`数据函数需要移动到`controllers`文件夹中的`blog/controllers.py`文件。

现在的`blog/controllers.py`文件应如下所示：

```py
from sqlalchemy import func
from flask import render_template, Blueprint, flash, redirect, url_for
from .models import db, Post, Tag, Comment, User, tags
from .forms import CommentForm

blog_blueprint = Blueprint(
    'blog',
    __name__,
    template_folder='../templates/blog',
    url_prefix="/blog"
)

def sidebar_data():
...
```

因此，每当需要一个新的功能，且足够大，可以作为应用程序模块的候选时，就需要一个新的 Python 模块（包含`__init__.py`文件的文件夹）来包含该功能的名称，以及之前描述的文件。我们将把应用程序代码分解成逻辑组。

然后，我们需要将新功能蓝图导入到主`__init__.py`文件中，并在 Flask 中注册它：

```py
from .blog.controllers import blog_blueprint
from .main.controllers import main_blueprint

...
app.register_blueprint(main_blueprint)
app.register_blueprint(blog_blueprint)
```

# 应用程序工厂

现在我们以模块化的方式使用蓝图，我们还可以对我们的抽象进行另一项改进，即创建一个应用程序的**工厂**。工厂的概念来自**面向对象编程**（OOP）世界，它简单意味着一个创建另一个对象的函数或对象。我们的应用程序工厂将接受我们最初在书中创建的`config`对象之一，并返回一个 Flask 应用程序对象。

对象工厂设计被现在著名的书籍《设计模式：可复用面向对象软件元素》（The Gang of Four 所著）所推广。要了解更多关于这些设计模式以及它们如何帮助简化项目代码的信息，请查看[`en.wikipedia.org/wiki/Structural_pattern`](https://en.wikipedia.org/wiki/Structural_pattern)。

为我们的应用程序对象创建一个工厂函数有几个好处。首先，它允许环境上下文改变应用程序的配置。当服务器创建应用程序对象以提供服务时，它可以考虑到服务器上必要的任何变化，并相应地更改提供给应用程序的配置对象。其次，它使得测试变得容易得多，因为它允许快速测试不同配置的应用程序。第三，可以非常容易地创建使用相同配置的同一应用程序的多个实例。这在网络流量在几个不同的服务器之间平衡的情况下非常有用。

现在应用程序工厂的好处已经很明显了，让我们修改我们的`__init__.py`文件来实现一个：

```py
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def page_not_found(error):
    return render_template('404.html'), 404

def create_app(object_name):
    from .blog.controllers import blog_blueprint
    from .main.controllers import main_blueprint

    app = Flask(__name__)
    app.config.from_object(object_name)

    db.init_app(app)
    migrate.init_app(app, db)
    app.register_blueprint(main_blueprint)
    app.register_blueprint(blog_blueprint)
    app.register_error_handler(404, page_not_found)
    return app
```

文件中的更改非常简单：我们将代码包含在一个接受`config`对象并返回应用程序对象的函数中。为了使用环境变量中的正确配置来启动我们的应用程序，我们需要更改`main.py`：

```py
import os
from webapp import create_app

env = os.environ.get('WEBAPP_ENV', 'dev')
app = create_app('config.%sConfig' % env.capitalize())

if __name__ == '__main__':
    app.run()
```

我们还需要修改我们的`manage.py`文件，以便与`create_app`函数一起使用，如下所示：

```py
import os
from webapp import db, migrate, create_app
from webapp.blog.models import User, Post, Tag

env = os.environ.get('WEBAPP_ENV', 'dev')
app = create_app('config.%sConfig' % env.capitalize())

@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db, User=User, Post=Post, Tag=Tag, migrate=migrate)
```

当我们创建配置对象时，提到应用程序运行的环境可能会改变应用程序的配置。此代码提供了一个非常简单的功能示例，其中加载了一个环境变量，并确定要将哪个`config`对象提供给`create_app`函数。环境变量是进程环境的一部分，是动态的名称值。这些环境可以被多个进程、系统范围、用户范围或单个进程共享。它们可以使用以下语法在 Bash 中设置：

```py
    $ export WEBAPP_ENV="dev"
```

使用此方法来读取变量：

```py
    $ echo $WEBAPP_ENV
    dev
```

你也可以轻松地删除变量，如下所示：

```py
    $ unset $WEBAPP_ENV
    $ echo $WEBAPP_ENV
```

在你的生产服务器上，你会将`WEBAPP_ENV`设置为`prod`。一旦你部署到生产环境（第十三章，*部署 Flask 应用程序*），以及当我们到达第十二章，*测试 Flask 应用程序*），该章节涵盖了测试我们的项目，这个设置的真正威力将变得更加清晰。

# 摘要

我们已经将我们的应用程序转换成了一个更加可管理和可扩展的结构，这将在我们进一步阅读本书并添加更多高级功能时节省我们很多麻烦。在下一章中，我们将向我们的应用程序添加登录和注册系统，以及其他使我们的网站更加安全的特性。
