

# 第十章：调试、错误处理和测试

到目前为止，在这本书中，我们一直专注于开发和为它们逐个添加功能的应用程序。了解我们的应用程序有多稳健，以及跟踪它的运行和性能情况非常重要。这反过来又产生了在应用程序出现问题时得到通知的需求。在开发应用程序时，遗漏某些边缘情况是正常的，通常，测试用例也会遗漏它们。如果能在它们出现时了解这些边缘情况，以便相应地处理它们，那就太好了。

有效的日志记录和快速调试的能力是选择应用程序开发框架时的决定性因素之一。框架提供的日志记录和调试支持越好，应用程序开发和维护的过程就越快。更好的日志记录和调试支持水平有助于开发者快速发现应用程序中的问题，在许多情况下，日志记录甚至会在最终用户识别问题之前指出问题。有效的错误处理在最终用户满意度中起着重要作用，并减轻了开发者调试的痛苦。即使其代码完美无缺，应用程序也难免会在某些时候抛出错误。为什么？答案很简单——代码可能完美无缺，但它在其中运行的世界却不是。可能会有无数的问题发生，作为开发者，我们总是想知道任何异常背后的原因。与应用程序一起编写测试用例是软件编写最重要的支柱之一。

Python 的内置日志系统与 Flask 配合得相当好。在本章中，我们将使用这个日志系统，然后再转向一个名为**Sentry**的出色服务，它极大地简化了调试和错误日志记录的痛苦。

既然我们已经讨论了测试对于应用程序开发的重要性，我们现在将看看如何为 Flask 应用程序编写测试用例。我们还将了解如何衡量代码覆盖率并分析我们的应用程序以解决任何瓶颈。

测试本身是一个巨大的主题，与之相关的书籍有很多。在这里，我们将尝试理解使用 Flask 进行测试的基础知识。

在本章中，我们将涵盖以下内容：

+   设置基本的文件日志记录

+   在发生错误时发送电子邮件

+   使用 Sentry 监控异常

+   使用 `pdb` 调试

+   创建应用程序工厂

+   创建第一个简单的测试

+   为视图和逻辑编写更多测试

+   集成 nose2 库

+   使用模拟来避免外部 API 访问

+   确定测试覆盖率

+   使用分析来查找瓶颈

# 设置基本的文件日志记录

默认情况下，Flask 不会为我们记录任何内容，除了带有堆栈跟踪的错误，这些错误被发送到记录器（我们将在本章的其余部分看到更多）。它在开发模式下使用 `run.py` 运行应用程序时确实会创建大量的堆栈跟踪，但在生产系统中，我们没有这种奢侈。幸运的是，记录库提供了一系列的记录处理程序，可以根据需要使用。在本食谱中，我们将了解如何利用 `logging` 库确保从 Flask 应用程序中捕获有效的日志。

## 准备中

我们将从上一章的目录应用程序开始，并使用 `FileHandler` 向文件系统上的指定文件添加一些基本的记录，以对它进行记录。我们将从基本的日志格式开始，然后看看如何格式化日志消息以使其更具信息性。

## 如何操作...

按照以下步骤配置和设置 `logging` 库，以便与我们的应用程序一起使用：

1.  第一个更改是对 `my_app/__init__.py` 文件进行的，该文件作为应用程序的配置文件：

    ```py
    app.config['LOG_FILE'] = 'application.log'
    ```

    ```py
    if not app.debug:
    ```

    ```py
        import logging
    ```

    ```py
        from logging import FileHandler, Formatter
    ```

    ```py
        file_handler = FileHandler(app.config['LOG_FILE'])
    ```

    ```py
        app.logger.setLevel(logging.INFO)
    ```

    ```py
        app.logger.addHandler(file_handler)
    ```

在这里，我们添加了一个配置参数来指定日志文件的位置。这采用从应用程序文件夹的相对路径，除非显式指定了绝对路径。接下来，我们将检查应用程序是否已经在调试模式中，然后我们将添加一个记录级别为 `INFO` 的文件记录处理程序。现在，`DEBUG` 是最低的记录级别，将记录任何级别的所有内容。有关更多详细信息，请参阅 `logging` 库文档（在 *参见* 部分）。

1.  在此之后，将记录器添加到应用程序所需的任何位置，应用程序将开始向指定的文件记录。现在，让我们向 `my_app/catalog/views.py` 添加几个记录器以供演示：

    ```py
    @catalog.route('/')
    ```

    ```py
    @catalog.route('/<lang>/')
    ```

    ```py
    @catalog.route('/<lang>/home')
    ```

    ```py
    @template_or_json('home.html')
    ```

    ```py
    def home():
    ```

    ```py
        products = Product.query.all()
    ```

    ```py
        app.logger.info(
    ```

    ```py
            'Home page with total of %d products'
    ```

    ```py
              % len(products)
    ```

    ```py
        )
    ```

    ```py
        return {'count': len(products)}
    ```

    ```py
    @catalog.route('/<lang>/product/<id>')
    ```

    ```py
    def product(id):
    ```

    ```py
        product = Product.query.filter_by(id=id).first()
    ```

    ```py
        if not product:
    ```

    ```py
            app.logger.warning('Requested product not
    ```

    ```py
              found.')
    ```

    ```py
            abort(404)
    ```

    ```py
        return render_template('product.html',
    ```

    ```py
          product=product)
    ```

在前面的代码中，我们为我们的几个视图处理程序添加了记录器。请注意，`home()` 中的第一个记录器处于 `info` 级别，而 `product()` 中的另一个是 `warning`。如果我们将在 `__init__.py` 中设置的日志级别为 `INFO`，那么两者都将被记录，如果我们设置级别为 `WARNING`，那么只有警告记录器将被记录。

信息

确保如果尚未完成，从 `Flask` 中导入 `abort` – `from flask import abort`。

## 它是如何工作的...

前面的步骤将在根应用程序文件夹中创建一个名为 `application.log` 的文件。根据指定的记录器，记录语句将被记录到 `application.log`，其内容可能类似于以下片段；第一个来自主页，第二个来自请求一个不存在的商品：

```py
Home page with total of 0 products
Requested product not found.
```

重要

要启用记录，要么使用 WSGI 服务器运行您的应用程序（请参阅*第十一章*）或使用终端提示中的 `flask run` 运行（请参阅*第一章*）。

使用`run.py`运行应用程序将始终使其以`debug`标志为`True`的方式运行，这将不会允许日志按预期工作。

记录的信息帮助不大。知道问题何时被记录、以何种级别、哪个文件在什么行号引发了问题等会更好。这可以通过高级日志格式实现。为此，我们需要在配置文件中添加一些语句——即`my_app/__init__.py`：

```py
if not app.debug:
    import logging
    from logging import FileHandler, Formatter
    file_handler = FileHandler(app.config['LOG_FILE'])
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    file_handler.setFormatter(Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(pathname)s:%(lineno)d]'
    ))
```

在前面的代码中，我们向`file_handler`添加了一个格式化器，这将记录时间、日志级别、消息、文件路径和行号。之后，记录的消息将看起来像这样：

```py
2023-01-02 13:01:25,125 INFO: Home page with total of 0 products [in /Users/apple/workspace/flask-cookbook-3/Chapter-10/Chapter-10/my_app/catalog/views.py:72]
2023-01-02 13:01:27,657 WARNING: Requested product not found. [in /Users/apple/workspace/flask-cookbook-3/Chapter-10/Chapter-10/my_app/catalog/views.py:82]
```

## 还有更多…

我们还可能想要记录当页面未找到时（`404`错误）的所有错误。为此，我们只需稍微调整一下`errorhandler`方法：

```py
@app.errorhandler(404)
def page_not_found(e):
    app.logger.error(e)
    return render_template('404.html'), 404
```

## 参见

前往 Python 的`logging`库处理程序文档[`docs.python.org/dev/library/logging.handlers.html`](https://docs.python.org/dev/library/logging.handlers.html)，了解更多关于日志处理程序的信息。

# 发生错误时发送电子邮件

当应用程序出现意外情况时，收到通知是个好主意。设置这一点相当简单，并为错误处理过程增加了许多便利。

## 准备工作

我们将从上一个菜谱中的应用程序开始，向其中添加`mail_handler`，以便在发生错误时我们的应用程序能够发送电子邮件。此外，我们将演示使用 Gmail 作为 SMTP 服务器进行电子邮件设置。

## 如何操作...

首先，将处理程序添加到`my_app/__init__.py`中的配置中。这与我们在上一个菜谱中添加`file_handler`的方式类似：

```py
RECEPIENTS = ['some_receiver@gmail.com']
if not app.debug:
    import logging
    from logging import FileHandler, Formatter
    from logging.handlers import SMTPHandler
    file_handler = FileHandler(app.config['LOG_FILE'])
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    mail_handler = SMTPHandler(
        ("smtp.gmail.com", 587), 'sender@gmail.com',
          RECEPIENTS,
        'Error occurred in your application',
        ('some_email@gmail.com', 'some_gmail_password'),
          secure=())
    mail_handler.setLevel(logging.ERROR)
    app.logger.addHandler(mail_handler)
    for handler in [file_handler, mail_handler]:
        handler.setFormatter(Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
```

在这里，我们有一个电子邮件地址列表，错误通知电子邮件将发送到这些地址。另外，请注意，我们在`mail_handler`的情况下将日志级别设置为`ERROR`。这是因为只有在关键事项的情况下才需要电子邮件。

有关`SMTPHandler`配置的更多详细信息，请参阅文档。

重要

总是要确保以`debug`标志设置为`off`的方式运行您的应用程序，以启用应用程序记录和发送内部应用程序错误（`500`错误）的电子邮件。

## 它是如何工作的…

要引起内部应用程序错误，只需在任何处理程序中拼写一些关键字出错。您将在邮箱中收到一封电子邮件，格式与配置中设置的一样，并附有完整的堆栈跟踪供您参考。

# 使用 Sentry 监控异常

Sentry 是一个工具，它简化了监控异常的过程，同时也为应用程序用户在使用过程中遇到的错误提供了洞察。很可能日志文件中存在被人类眼睛忽视的错误。Sentry 将错误分类到不同的类别，并记录错误的重复次数。这有助于我们根据多个标准了解错误的严重性以及如何相应地处理它们。它有一个很好的 GUI，便于实现所有这些功能。在本食谱中，我们将设置 Sentry 并将其用作有效的错误监控工具。

## 准备工作

Sentry 可作为云服务提供，对开发者和基本用户免费。在本食谱的目的上，这个免费提供的云服务将足够使用。请访问 [`sentry.io/signup/`](https://sentry.io/signup/) 并开始注册过程。话虽如此，我们需要安装 Sentry 的 Python SDK：

```py
$ pip install 'sentry-sdk[flask]'
```

## 如何操作…

Sentry 注册完成后，将显示一个屏幕，询问需要与 Sentry 集成的项目类型。以下是一个截图：

![图 10.1 – Sentry 项目创建屏幕](img/B19111_10_1.jpg)

图 10.1 – Sentry 项目创建屏幕

这将随后显示另一个屏幕，展示如何配置您的 Flask 应用程序以将事件发送到新创建和配置的 Sentry 实例。以下是一个截图：

![图 10.2 – Sentry 项目配置步骤](img/B19111_10_2.jpg)

图 10.2 – Sentry 项目配置步骤

信息

Sentry 也可以免费下载并作为本地应用程序安装。根据您的需求，有多种安装和配置 Sentry 的方法。您可以自由尝试这种方法，因为它超出了本食谱的范围。

在完成之前的设置后，将以下代码添加到您的 Flask 应用程序中的 `my_app/__init__.py` 文件，将 `https://1234:5678@fake-sentry-server/1` 替换为 Sentry 项目 URI：

```py
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
sentry_sdk.init(
    dsn="https://1234:5678@fake-sentry-server/1",
    integrations=[FlaskIntegration()]
)
```

## 它是如何工作的…

在 Sentry 中记录的错误将类似于以下截图：

![图 10.3 – Sentry 错误日志屏幕](img/B19111_10_3.jpg)

图 10.3 – Sentry 错误日志屏幕

还可以在 Sentry 中记录消息和用户定义的异常。我将把这个留给你自己探索。

# 使用 pdb 进行调试

阅读这本书的大部分 Python 开发者可能已经知道 `pdb` 是一个交互式源代码调试器，用于 Python 程序。我们可以在需要的地方设置断点，在源代码行级别进行单步调试，并检查堆栈帧。

许多新开发者可能认为调试器的任务可以使用记录器来处理，但调试器提供了对控制流程的更深入洞察，保留每个步骤的状态，因此，可能节省大量开发时间。在这个菜谱中，让我们看看`pdb`能带来什么。

## 准备工作

我们将在这个菜谱中使用 Python 的内置`pdb`模块，并在上一个菜谱的应用程序中使用它。

## 如何操作…

在大多数情况下，使用`pdb`相当简单。我们只需要在我们想要插入断点以检查特定代码块的地方插入以下语句：

```py
import pdb; pdb.set_trace()
```

这将触发应用程序在此处中断执行，然后我们可以使用调试器命令逐个遍历堆栈帧。

因此，让我们在我们的某个方法中插入这个语句——比如说，产品的处理程序：

```py
@catalog.route('/<lang>/products')
@catalog.route('/<lang>/products/<int:page>')
def products(page=1):
    products = Product.query.paginate(page=page,
      per_page=10)
    import pdb; pdb.set_trace()
    return render_template('products.html',
      products=products)
```

## 它是如何工作的…

每当控制流到达这一行时，调试器提示将会启动；它将如下所示：

```py
> /Users/apple/workspace/flask-cookbook-3/Chapter-10/Chapter-10/my_app/catalog/views.py(93)products()
-> return render_template('products.html', products=products)
(Pdb) u
> /Users/apple/workspace/flask-cookbook-3/Chapter-10/lib/python3.10/site-packages/flask/app.py(1796)dispatch_request()
-> return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)
(Pdb) u
> /Users/apple/workspace/flask-cookbook-3/Chapter-10/lib/python3.10/site-packages/flask/app.py(1820)full_dispatch_request()
-> rv = self.dispatch_request()
(Pdb) u
> /Users/apple/workspace/flask-cookbook-3/Chapter-10/lib/python3.10/site-packages/flask/app.py(2525)wsgi_app()
-> response = self.full_dispatch_request()
(Pdb) u
> /Users/apple/workspace/flask-cookbook-3/Chapter-10/lib/python3.10/site-packages/flask/app.py(2548)__call__()
-> return self.wsgi_app(environ, start_response)
```

注意`(Pdb)`旁边写有`u`。这表示我正在将当前帧在堆栈跟踪中向上移动一级。在该语句中使用的所有变量、参数和属性都将在此上下文中可用，以帮助解决问题或只是理解代码的流程。还有其他调试器命令可能在您的调试日志导航中很有帮助。请参阅以下*参见*部分以获取这些信息。

## 参见

前往[`docs.python.org/3/library/pdb.html#debugger-commands`](https://docs.python.org/3/library/pdb.html#debugger-commands)模块文档，以获取各种调试器命令。

# 创建应用程序工厂

利用工厂模式是组织应用程序对象的好方法，允许有多个具有不同设置的应用程序对象。如在第*第一章*中讨论的，总是可以通过使用不同的配置来创建多个应用程序实例，但应用程序工厂允许您在同一个应用程序进程中拥有多个应用程序对象。它还有助于测试，因为您可以选择为每个测试案例选择一个全新的或不同的应用程序对象，并具有不同的设置。

## 准备工作

我们将使用上一个菜谱中的应用程序，并将其修改为使用应用程序工厂模式。

## 如何操作…

需要进行的以下是一些更改：

1.  我们将在`my_app/__init__.py`中创建一个名为`create_app()`的函数：

    ```py
    def create_app(alt_config={}):
    ```

    ```py
        app = Flask(__name__, template_folder=alt_config
    ```

    ```py
          .get('TEMPLATE_FOLDER', 'templates'))
    ```

    ```py
        app.config['UPLOAD_FOLDER'] =
    ```

    ```py
          os.path.realpath('.') + '/my_app/static/uploads'
    ```

    ```py
        app.config['SQLALCHEMY_DATABASE_URI'] =
    ```

    ```py
          'sqlite:////tmp/test.db'
    ```

    ```py
        app.config['WTF_CSRF_SECRET_KEY'] = 'random key
    ```

    ```py
          for form'
    ```

    ```py
        app.config['LOG_FILE'] = 'application.log'
    ```

    ```py
        app.config.update(alt_config)
    ```

    ```py
        if not app.debug:
    ```

    ```py
            import logging
    ```

    ```py
            from logging import FileHandler, Formatter
    ```

    ```py
            from logging.handlers import SMTPHandler
    ```

    ```py
            file_handler =
    ```

    ```py
              FileHandler(app.config['LOG_FILE'])
    ```

    ```py
            app.logger.setLevel(logging.INFO)
    ```

    ```py
            app.logger.addHandler(file_handler)
    ```

    ```py
            mail_handler = SMTPHandler(
    ```

    ```py
                ("smtp.gmail.com", 587),
    ```

    ```py
                  'sender@gmail.com', RECEPIENTS,
    ```

    ```py
                'Error occurred in your application',
    ```

    ```py
                ('some_email@gmail.com',
    ```

    ```py
                  'some_gmail_password'), secure=())
    ```

    ```py
            mail_handler.setLevel(logging.ERROR)
    ```

    ```py
            # app.logger.addHandler(mail_handler)
    ```

    ```py
            for handler in [file_handler, mail_handler]:
    ```

    ```py
                handler.setFormatter(Formatter(
    ```

    ```py
                    '%(asctime)s %(levelname)s:
    ```

    ```py
                      %(message)s '
    ```

    ```py
                    '[in %(pathname)s:%(lineno)d]'
    ```

    ```py
                ))
    ```

    ```py
        app.secret_key = 'some_random_key'
    ```

    ```py
        return app
    ```

在这个函数中，我们只是在名为`create_app()`的函数内部重新排列了所有的应用程序配置。这将允许我们通过简单地调用这个函数来创建所需数量的应用程序对象。

1.  接下来，我们创建一个名为`create_db()`的方法，它初始化数据库然后创建表：

    ```py
    db = SQLAlchemy()
    ```

    ```py
    def create_db(app):
    ```

    ```py
        db.init_app(app)
    ```

    ```py
        with app.app_context():
    ```

    ```py
            db.create_all()
    ```

    ```py
        return db
    ```

在这个函数中，我们只是将数据库特定的代码移动到了一个函数中。这个方法被保留为独立的，因为您可能希望使用不同的数据库配置与不同的应用程序实例一起使用。

1.  在`my_app/__init__.py`中的最后一步将是调用/执行这些方法并注册蓝图：

    ```py
    def get_locale():
    ```

    ```py
        return g.get('current_lang', 'en')
    ```

    ```py
    app = create_app()
    ```

    ```py
    babel = Babel(app)
    ```

    ```py
    babel.init_app(app, locale_selector=get_locale)
    ```

    ```py
    from my_app.catalog.views import catalog
    ```

    ```py
    app.register_blueprint(catalog)
    ```

    ```py
    db = create_db(app)
    ```

我们通过调用相关方法并初始化扩展来创建了`app`、`db`和`babel`的对象。

应用程序工厂模式的缺点是在导入时不能在蓝图中使用应用程序对象。然而，您始终可以利用`current_app`代理来访问当前的应用程序对象。让我们看看在`my_app/catalog/views.py`中是如何做到这一点的：

```py
from flask import current_app
@catalog.before_request
def before():
    # Existing code
@catalog.context_processor
def inject_url_for():
    # Existing code
# Similarly simply replace all your references to `app` by
  `current_app`. Refer to code provided with the book for a
  complete example.
```

## 它是如何工作的...

应用程序将继续以与上一个配方相同的方式工作。只是代码已经被重新排列以实现应用程序工厂模式。

## 参见

接下来的几个配方将帮助您了解在编写测试用例时如何使用工厂模式。

# 创建第一个简单的测试

测试是任何软件开发期间以及后续的维护和扩展期间最坚实的支柱之一。特别是在网络应用程序的情况下，应用程序将处理高流量，并且始终受到大量最终用户的审查，测试变得非常重要，因为用户反馈决定了应用程序的命运。在这个配方中，我们将看到如何从编写测试开始，并在接下来的配方中看到更复杂的测试。

## 准备工作

我们将从在根应用程序级别创建一个名为`app_tests.py`的新测试文件开始——即在`my_app`文件夹旁边。

## 如何做到这一点...

让我们编写我们的第一个测试用例：

1.  首先，`app_tests.py`测试文件的内容将如下所示：

    ```py
    import os
    ```

    ```py
    from my_app import create_app, db, babel
    ```

    ```py
    import unittest
    ```

    ```py
    import tempfile
    ```

前面的代码描述了此测试套件所需的导入。我们将使用`unittest`来编写我们的测试。需要一个`tempfile`实例来动态创建 SQLite 数据库。

1.  所有测试用例都需要从`unittest.TestCase`派生：

    ```py
    class CatalogTestCase(unittest.TestCase):
    ```

    ```py
        def setUp(self):
    ```

    ```py
            test_config = {}
    ```

    ```py
            self.test_db_file = tempfile.mkstemp()[1]
    ```

    ```py
            test_config['SQLALCHEMY_DATABASE_URI'] =
    ```

    ```py
              'sqlite:///' + self.test_db_file
    ```

    ```py
            test_config['TESTING'] = True
    ```

    ```py
            self.app = create_app(test_config)
    ```

    ```py
            db.init_app(self.app)
    ```

    ```py
            babel.init_app(self.app)
    ```

    ```py
            with self.app.app_context():
    ```

    ```py
                db.create_all()
    ```

    ```py
            from my_app.catalog.views import catalog
    ```

    ```py
            self.app.register_blueprint(catalog)
    ```

    ```py
            self.client = self.app.test_client()
    ```

在每次运行测试之前都会运行前面的方法，并创建一个新的测试客户端。一个测试由这个类中以`test_`前缀开始的方法表示。在这里，我们在应用程序配置中设置了一个数据库名称，它是一个基于时间戳的值，将始终是唯一的。我们还设置了`TESTING`标志为`True`，这禁用了错误捕获以启用更好的测试。请特别注意如何在使用`db`和`babel`初始化之前使用应用程序工厂创建应用程序对象。

最后，我们在`db`上运行`create_all()`方法，在测试数据库中创建我们应用程序的所有表。

1.  在测试执行后删除之前步骤中创建的临时数据库：

    ```py
        def tearDown(self):
    ```

    ```py
            os.remove(self.test_db_file)
    ```

在每次运行测试之后都会调用前面的方法。在这里，我们将删除当前的数据库文件，并为每个测试使用一个新的数据库文件。

1.  最后，编写测试用例：

    ```py
        def test_home(self):
    ```

    ```py
            rv = self.client.get('/')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

之前的代码是我们的第一个测试，其中我们向我们的应用程序在`/` URL 发送了 HTTP `GET`请求并测试了状态码，它应该是`200`，表示成功的`GET`响应。

## 它是如何工作的…

要运行测试文件，只需在终端中执行以下命令：

```py
$ python app_tests.py
```

以下截图显示了表示测试结果的输出：

![图 10.4 – 第一个测试结果](img/B19111_10_4.jpg)

图 10.4 – 第一个测试结果

## 相关内容

参考下一道菜谱，*为视图和逻辑编写更多测试*，以了解如何编写复杂测试的更多内容。

# 为视图和逻辑编写更多测试

在上一道菜谱中，我们开始了为我们的 Flask 应用程序编写测试。在这道菜谱中，我们将基于相同的测试文件添加更多测试到我们的应用程序中；这些测试将涵盖测试视图的行为和逻辑。

## 准备工作

我们将基于上一道菜谱中创建的名为`app_tests.py`的测试文件进行构建。

## 如何做…

在我们编写任何测试之前，我们需要向`setUp()`添加一小部分配置来禁用 CSRF 令牌，因为在测试环境中默认不会生成：

```py
        test_config['WTF_CSRF_ENABLED'] = False
```

以下是一些作为这道菜谱的一部分创建的测试。每个测试将在我们进一步进行时进行描述：

1.  首先，编写一个测试来向产品列表发送`GET`请求：

    ```py
        def test_products(self):
    ```

    ```py
            "Test Products list page"
    ```

    ```py
            rv = self.client.get('/en/products')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('No Previous Page' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            self.assertTrue('No Next Page' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

之前的测试向`/products`端点发送`GET`请求并断言响应的状态码为`200`。它还断言没有上一页和下一页（作为模板逻辑的一部分渲染）。

1.  接下来，创建一个类别并验证它是否已正确创建：

    ```py
        def test_create_category(self):
    ```

    ```py
            "Test creation of new category"
    ```

    ```py
            rv = self.client.get('/en/category-create')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            rv = self.client.post('/en/category-create')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('This field is required.' In
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            rv = self.client.get('/en/categories')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertFalse('Phones' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            rv = self.client.post('/en/category-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'Phones',
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            rv = self.client.get('/en/categories')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('Phones' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            rv = self.client.get('/en/category/1')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('Phones' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

之前的测试创建了一个类别并断言相应的状态消息。当类别成功创建时，我们将被重定向到新创建的类别页面，因此状态码将是`302`。

1.  现在，类似于类别创建，创建一个产品然后验证其创建：

    ```py
        def test_create_product(self):
    ```

    ```py
            "Test creation of new product"
    ```

    ```py
            rv = self.client.get('/en/product-create')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            # Raise a ValueError for a valid category not
    ```

    ```py
              found
    ```

    ```py
            self.assertRaises(ValueError,
    ```

    ```py
              self.client.post, '/en/product-create')
    ```

    ```py
            # Create a category to be used in product
    ```

    ```py
              creation
    ```

    ```py
            rv = self.client.post('/en/category-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'Phones',
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            rv = self.client.post('/en/product-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'iPhone 5',
    ```

    ```py
                'price': 549.49,
    ```

    ```py
                'company': 'Apple',
    ```

    ```py
                'category': 1,
    ```

    ```py
                'image': tempfile.NamedTemporaryFile()
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            rv = self.client.get('/en/products')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('iPhone 5' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

之前的测试创建了一个产品并断言每个调用/请求的相应状态消息。

信息

作为这个测试的一部分，我们在`create_product()`方法中识别了一个小的改进。我们在检查允许的文件类型之前没有初始化`filename`变量。早期的代码只有在`if`条件通过时才能正常工作。现在，我们只是调整了代码，在`if`条件之前将`filename`初始化为`filename = secure_filename(image.filename)`，而不是在条件内部做这件事。

1.  最后，创建多个产品并搜索刚刚创建的产品：

    ```py
        def test_search_product(self):
    ```

    ```py
            "Test searching product"
    ```

    ```py
            # Create a category to be used in product
    ```

    ```py
              creation
    ```

    ```py
            rv = self.client.post('/en/category-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'Phones',
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            # Create a product
    ```

    ```py
            rv = self.client.post('/en/product-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'iPhone 5',
    ```

    ```py
                'price': 549.49,
    ```

    ```py
                'company': 'Apple',
    ```

    ```py
                'category': 1,
    ```

    ```py
                'image': tempfile.NamedTemporaryFile()
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            # Create another product
    ```

    ```py
            rv = self.client.post('/en/product-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'Galaxy S5',
    ```

    ```py
                'price': 549.49,
    ```

    ```py
                'company': 'Samsung',
    ```

    ```py
                'category': 1,
    ```

    ```py
                'image': tempfile.NamedTemporaryFile()
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            self.client.get('/')
    ```

    ```py
            rv = self.client.get('/en/product-
    ```

    ```py
              search?name=iPhone')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('iPhone 5' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            self.assertFalse('Galaxy S5' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            rv = self.client.get('/en/product-
    ```

    ```py
              search?name=iPhone 6')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertFalse('iPhone 6' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

之前的测试首先创建一个类别和两个产品。然后，它搜索一个产品并确保只返回搜索到的产品在结果中。

## 它是如何工作的…

要运行测试文件，只需在终端中执行以下命令：

```py
$ python app_tests.py -v
test_create_category (__main__.CatalogTestCase)
Test creation of new category ... ok
test_create_product (__main__.CatalogTestCase)
Test creation of new product ... ok
test_home (__main__.CatalogTestCase) ... ok
test_products (__main__.CatalogTestCase)
Test Products list page ... ok
test_search_product (__main__.CatalogTestCase)
Test searching product ... ok
----------------------------------------------------------------------
Ran 5 tests in 0.390s
OK
```

命令之后是显示测试结果的输出。

## 相关内容

另一个可以用于单元测试的有趣且流行的库是 **pytest**。它与 Python 内置的 **unittest** 库类似，但具有更多开箱即用的功能。请随意探索它：[`docs.pytest.org/en/stable/`](https://docs.pytest.org/en/stable/)。

# 集成 `nose2` 库

`nose2` 可以用于多种用途，其中最重要的用途仍然是作为测试收集器和运行器。`nose2` 会自动从当前工作目录中的 Python 源文件、目录和包中收集测试。在本食谱中，我们将重点介绍如何使用 `nose2` 运行单个测试，而不是每次都运行一整套测试。

重要

在本书的早期版本中，我们使用了 `nose` 库。自那时起，它已经没有在积极维护，并且可以被认为是过时的。已经创建了一个替代品，名为 `nose2`。这个库的行为与 `nose` 类似，但并不完全相同。然而，为了我们的演示目的，主要功能仍然相似。

## 准备工作

首先，我们需要安装 `nose2` 库：

```py
$ pip install nose2
```

`nose2` 有一个测试文件发现机制，要求文件名应以 `test` 开头。由于在我们的情况下，测试文件名为 `app_tests.py`，我们现在应该将其重命名为 `test_app.py`。在终端中，你可以简单地运行以下命令：

```py
$ mv app_tests.py test_app.py
```

## 如何做到这一点...

我们可以通过运行以下命令使用 `nose2` 执行我们应用程序中的所有测试：

```py
$ nose2 -v
test_create_category (test_app.CatalogTestCase)
Test creation of new category ... ok
test_create_product (test_app.CatalogTestCase)
Test creation of new product ... ok
test_home (test_app.CatalogTestCase) ... ok
test_products (test_app.CatalogTestCase)
Test Products list page ... ok
test_search_product (test_app.CatalogTestCase)
Test searching product ... ok
----------------------------------------------------------------------
Ran 5 tests in 0.241s
OK
```

这将选择我们应用程序中的所有测试并运行它们，即使我们有多个测试文件。

要运行单个测试文件，只需运行以下命令：

```py
$ nose2 test_app
```

现在，如果你想运行单个测试，只需运行以下命令：

```py
$ nose2 test_app.CatalogTestCase.test_home
```

当我们有一个内存密集型应用程序和大量测试用例时，这一点变得很重要。在这种情况下，测试本身可能需要很长时间才能运行，并且每次都这样做可能会让开发者感到非常沮丧。相反，我们更愿意只运行那些与所做的更改相关的测试，或者是在某些更改之后破坏的测试。

## 参见

根据需求，有许多其他方法可以配置 `nose2` 以实现最优和有效的使用。有关更多详细信息，请参阅 [`docs.nose2.io/en/latest/index.html`](https://docs.nose2.io/en/latest/index.html) 上的 `nose2` 文档。

# 使用模拟来避免外部 API 访问

我们了解测试是如何工作的，但现在，让我们想象我们有一个第三方应用程序/服务通过 API 调用与我们的应用程序集成。每次运行测试时都调用这个应用程序/服务并不是一个好主意。有时，这些调用也可能是付费的，并且在测试期间进行调用不仅可能很昂贵，还可能影响该服务的统计信息。我们可以使用 `geoip2` 库并通过模拟来测试它。

## 准备工作

在 Python 3 中，`mock` 已经被包含在 `unittest` 库的标准包中。

为了本食谱的目的，我们首先需要安装 `geoip2` 库和相应的数据库：

```py
$ pip install geoip2
```

您还需要从 MaxMind 的免费网站（[`dev.maxmind.com/geoip/geolite2-free-geolocation-data`](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data)）下载免费的`geoip`数据库到您偏好的位置，然后解压文件。为了简化，我已经将其下载到项目文件夹本身。在您能够下载`geoip`城市数据库之前，您需要创建一个免费账户。

在下载城市数据库后，您应该有一个以`Geolite2- City-`为前缀的文件夹。这个文件夹包含我们将在本食谱中使用的`.mmdb`扩展名的`geoip`数据库。

现在，假设我们想要存储创建产品的用户的位置（想象一下应用程序在多个全球位置管理的场景）。

我们需要对`my_app/catalog/models.py`、`my_app/catalog/views.py`和`templates/product.html`进行一些小的修改。

对于`my_app/catalog/models.py`，我们将添加一个名为`user_timezone`的新字段：

```py
class Product(db.Model):
    # ... Other fields ...
    user_timezone = db.Column(db.String(255))
    def __init__(self, name, price, category, image_path,
      user_timezone=''):
        # ... Other fields initialization ...
        self.user_timezone = user_timezone
```

对于`my_app/catalog/views.py`，我们将修改`create_product()`方法以包含时区：

```py
import geoip2.database, geoip2.errors
@catalog.route('/<lang>/product-create', methods=['GET',
  'POST'])
def create_product():
    form = ProductForm()
    if form.validate_on_submit():
        # ... Non changed code ...
        reader = geoip2.database.Reader(
            'GeoLite2-City_20230113/GeoLite2-City.mmdb'
        )
        try:
            match = reader.city(request.remote_addr)
        except geoip2.errors.AddressNotFoundError:
            match = None
        product = Product(
            name, price, category, filename,
            match and match.location.time_zone or
              'Localhost'
        )
        # ... Non changed code ...
```

在这里，我们使用 IP 查找获取地理位置数据，并在创建产品时传递这些数据。如果没有找到匹配项，那么调用来自`localhost`、`127.0.0.1`或`0.0.0.0`。

此外，我们将在我们的产品模板中添加这个新字段，以便在测试中容易验证。为此，只需在`product.html`模板的某个位置添加`{{ product.user_timezone }}`。

## 如何做到这一点...

首先，修改`test_app.py`以适应对`geoip`查找的模拟：

1.  首先，通过创建修补器来配置对`geoip`查找的模拟：

    ```py
    from unittest import mock
    ```

    ```py
    import geoip2.records
    ```

    ```py
    class CatalogTestCase(unittest.TestCase):
    ```

    ```py
        def setUp(self):
    ```

    ```py
            # ... Non changed code ...
    ```

    ```py
            self.geoip_city_patcher =
    ```

    ```py
              mock.patch('geoip2.models.City',
    ```

    ```py
                location=geoip2.records.Location(time_zone
    ```

    ```py
                  = 'America/Los_Angeles')
    ```

    ```py
            )
    ```

    ```py
            PatchedGeoipCity =
    ```

    ```py
              self.geoip_city_patcher.start()
    ```

    ```py
            self.geoip_reader_patcher =
    ```

    ```py
              mock.patch('geoip2.database.Reader')
    ```

    ```py
            PatchedGeoipReader =
    ```

    ```py
              self.geoip_reader_patcher.start()
    ```

    ```py
            PatchedGeoipReader().city.return_value =
    ```

    ```py
              PatchedGeoipCity
    ```

    ```py
            with self.app.app_context():
    ```

    ```py
                db.create_all()
    ```

    ```py
            from my_app.catalog.views import catalog
    ```

    ```py
            self.app.register_blueprint(catalog)
    ```

    ```py
            self.client = self.app.test_client()
    ```

首先，我们从`geoip2`中导入了`records`，这是我们用来创建测试所需模拟返回值的。然后，我们使用`location`属性在`City`模型上对`geoip2.models.City`进行了修补，并将其预设为`geoip2.records.Location(time_zone = 'America/Los_Angeles')`，然后启动了修补器。这意味着每当创建一个`geoip2.models.City`实例时，它都会被修补，其`location`属性上的时区设置为`'America/Los_Angeles'`。

这随后是对`geoip2.database.Reader`的修补，我们模拟其`city()`方法的返回值为我们之前创建的`PatchedGeoipCity`类。

1.  停止在`setUp`方法中启动的修补器：

    ```py
        def tearDown(self):
    ```

    ```py
            self.geoip_city_patcher.stop()
    ```

    ```py
            self.geoip_reader_patcher.stop()
    ```

    ```py
            os.remove(self.test_db_file)
    ```

我们在`tearDown`中停止了模拟修补器，以确保实际的调用不受影响。

1.  最后，修改创建的产品测试用例以断言位置：

    ```py
        def test_create_product(self):
    ```

    ```py
            "Test creation of new product"
    ```

    ```py
            # ... Non changed code ...
    ```

    ```py
            rv = self.client.post('/en/product-create',
    ```

    ```py
              data={
    ```

    ```py
                'name': 'iPhone 5',
    ```

    ```py
                'price': 549.49,
    ```

    ```py
                'company': 'Apple',
    ```

    ```py
                'category': 1,
    ```

    ```py
                'image': tempfile.NamedTemporaryFile()
    ```

    ```py
            })
    ```

    ```py
            self.assertEqual(rv.status_code, 302)
    ```

    ```py
            rv = self.client.get('/en/product/1')
    ```

    ```py
            self.assertEqual(rv.status_code, 200)
    ```

    ```py
            self.assertTrue('iPhone 5' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

    ```py
            self.assertTrue('America/Los_Angeles' in
    ```

    ```py
              rv.data.decode("utf-8"))
    ```

在这里，在创建产品后，我们断言`America/Los_Angeles`值出现在渲染的产品模板的某个地方。

## 它是如何工作的...

运行测试并查看是否通过：

```py
$ nose2 test_app.CatalogTestCase.test_create_product -v
test_create_product (test_app.CatalogTestCase)
Test creation of new product ... ok
----------------------------------------------------------------
Ran 1 test in 0.079s
OK
```

## 参见

有多种方法可以进行模拟。我演示了其中的一种。你可以从可用的方法中选择任何一种。请参阅[`docs.python.org/3/library/unittest.mock.html`](https://docs.python.org/3/library/unittest.mock.html)上的文档。

# 确定测试覆盖率

在之前的菜谱中，我们涵盖了测试用例的编写，但有一个重要的方面可以衡量测试的范围，称为覆盖率。覆盖率指的是我们的代码有多少被测试覆盖。覆盖率百分比越高，测试就越好（尽管高覆盖率不是良好测试的唯一标准）。在这个菜谱中，我们将检查我们应用程序的代码覆盖率。

小贴士

记住，100%的测试覆盖率并不意味着代码完美无缺。然而，在任何情况下，它都比没有测试或覆盖率低要好。记住，“*如果没有经过测试，* *它就是有缺陷的.*”

## 准备工作

我们将使用一个名为`coverage`的库来完成这个菜谱。以下是其安装命令：

```py
$ pip install coverage
```

## 如何做到这一点…

测量代码覆盖率的最简单方法是使用命令行：

1.  简单地运行以下命令：

    ```py
    $ coverage run --source=<Folder name of the application> --omit=test_app.py,run.py test_app.py
    ```

这里，`--source`表示要考虑在覆盖率中的目录，而`--omit`表示在过程中需要排除的文件。

1.  现在，要直接在终端上打印报告，请运行以下命令：

    ```py
    $ coverage report
    ```

以下截图显示了输出：

![图 10.5 – 测试覆盖率报告](img/B19111_10_5.jpg)

图 10.5 – 测试覆盖率报告

1.  要获得覆盖率报告的漂亮 HTML 输出，请运行以下命令：

    ```py
    $ coverage html
    ```

这将在你的当前工作目录中创建一个名为`htmlcov`的新文件夹。在这个文件夹中，只需在浏览器中打开`index.html`，就可以看到完整的详细视图。

![图 10.6 – 测试覆盖率报告的网页视图](img/B19111_10_6.jpg)

图 10.6 – 测试覆盖率报告的网页视图

或者，我们可以在测试文件中包含一段代码，并在每次运行测试时获取覆盖率报告。我们只需将以下代码片段添加到`test_app.py`中：

1.  在做任何事情之前，添加以下代码以启动覆盖率评估过程：

    ```py
    import coverage
    ```

    ```py
    cov = coverage.coverage(
    ```

    ```py
        omit = [
    ```

    ```py
            '/Users/apple/workspace/flask-cookbook-3/
    ```

    ```py
              Chapter-10/lib/python3.10/site-packages/*',
    ```

    ```py
            'test_app.py'
    ```

    ```py
        ]
    ```

    ```py
    )
    ```

    ```py
    cov.start()
    ```

这里，我们导入了`coverage`库并为它创建了一个对象。这告诉库忽略所有`site-packages`实例（因为我们不想评估我们没有编写的代码）以及测试文件本身。然后，我们启动了确定覆盖率的进程。

1.  在代码的末尾，将最后一个块修改为以下内容：

    ```py
    if __name__ == '__main__':
    ```

    ```py
        try:
    ```

    ```py
            unittest.main()
    ```

    ```py
        finally:
    ```

    ```py
            cov.stop()
    ```

    ```py
            cov.save()
    ```

    ```py
            cov.report()
    ```

    ```py
            cov.html_report(directory = 'coverage')
    ```

    ```py
            cov.erase()
    ```

在前面的代码中，我们首先将`unittest.main()`放在一个`try..finally`块中。这是因为`unittest.main()`在所有测试执行完毕后退出。现在，在执行完这个方法后，必须强制运行与覆盖率相关的代码。我们停止了覆盖率报告，保存了它，然后在控制台上打印了报告，并在删除临时`.coverage`文件（这是作为过程的一部分自动创建的）之前生成了它的 HTML 版本。

## 它是如何工作的…

如果我们在包含覆盖率特定代码后运行测试，那么我们可以运行以下命令：

```py
$ python test_app.py
```

输出将非常类似于*图 10.5*中的那个。

## 参见

还可以使用`nose2`库来确定覆盖率，我们在*集成 nose2 库*菜谱中讨论了它。我将把它留给你自己探索。有关更多信息，请参阅[`docs.nose2.io/en/latest/plugins/coverage.html`](https://docs.nose2.io/en/latest/plugins/coverage.html)。

# 使用性能分析来查找瓶颈

性能分析是在我们决定扩展应用程序时测量性能的重要且方便的工具。在扩展之前，我们想知道是否有任何进程是瓶颈并影响整体性能。Python 有一个内置的性能分析器`cProfile`，可以为我们完成这项工作，但为了使生活更简单，`werkzeug`有`ProfilerMiddleware`，它是基于`cProfile`编写的。在这个菜谱中，我们将使用`ProfilerMiddleware`来确定是否有任何影响性能的因素。

## 准备工作

我们将使用前一个菜谱中的应用程序，并将`ProfilerMiddleware`添加到一个名为`generate_profile.py`的新文件中。

## 如何做…

创建一个新文件`generate_profile.py`，与`run.py`并列，它的工作方式与`run.py`本身一样，但带有`ProfilerMiddleware`：

```py
from werkzeug.middleware.profiler import ProfilerMiddleware
from my_app import app
app.wsgi_app = ProfilerMiddleware(app.wsgi_app,
  restrictions = [10])
app.run(debug=True)
```

在这里，我们从`werkzeug`中导入了`ProfilerMiddleware`，然后修改了我们的 Flask 应用中的`wsgi_app`以使用它，限制输出中打印的前 10 个调用。

## 它是如何工作的…

现在，我们可以使用`generate_profile.py`来运行我们的应用程序：

```py
$ python generate_profile.py
```

然后，我们可以创建一个新的产品。然后，针对该特定调用的输出将类似于以下截图：

![图 10.7 – 性能分析器输出](img/B19111_10_7.jpg)

图 10.7 – 性能分析器输出

从前面的截图可以看出，在这个过程中最密集的调用是对`geoip`数据库的调用。尽管它是一个单独的调用，但它花费了最多的时间。所以，如果我们决定在将来某个时候提高性能，这是需要首先查看的地方。
