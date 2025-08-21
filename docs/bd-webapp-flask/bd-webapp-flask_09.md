# 第九章：扩展，我是如何爱你

我们已经在前几章中使用扩展来增强我们的示例；Flask-SQLAlchemy 用于连接到关系数据库，Flask-MongoEngine 用于连接到 MongoDB，Flask-WTF 用于创建灵活可重用的表单，等等。扩展是一种很好的方式，可以在不妨碍您的代码的情况下为项目添加功能，如果您喜欢我们迄今为止所做的工作，您会喜欢这一章，因为它专门介绍了扩展！

在本章中，我们将了解一些迄今为止忽视的非常流行的扩展。我们要开始了吗？

# 如何配置扩展

Flask 扩展是您导入的模块，（通常）初始化，并用于与第三方库集成。它们通常是从`flask.ext.<extension_name>`（这是扩展模式的一部分）导入的，并应该在 PyPi 存储库中以 BSD、MIT 或其他不太严格的许可证下可用。

扩展最好有两种状态：未初始化和已初始化。这是一个好的做法，因为在实例化扩展时，您的 Flask 应用程序可能不可用。我们在上一章的示例中只有在主模块中导入 Flask-SQLAlchemy 后才进行初始化。好的，知道了，但初始化过程为何重要呢？

嗯，正是通过初始化，扩展才能从应用程序中获取其配置。例如：

```py
from flask import Flask
import logging

# set configuration for your Flask application or extensions
class Config(object):
    LOG_LEVEL = logging.WARNING

app = Flask(__name__)
app.config.from_object(Config)
app.run()
```

在上面的代码中，我们创建了一个配置类，并使用`config.from_object`加载了它。这样，`LOG_LEVEL`就可以在所有扩展中使用，通过对应用实例的控制。

```py
app.config['LOG_LEVEL']
```

将配置加载到`app.config`的另一种方法是使用环境变量。这种方法在部署环境中特别有用，因为您不希望将敏感的部署配置存储在版本控制存储库中（这是不安全的！）。它的工作原理如下：

```py
…
app.config.from_envvar('PATH_TO_CONFIGURATION')
```

如果`PATH_TO_CONFIGURATION`设置为 Python 文件路径，例如`/home/youruser/someconfig.py`，那么`someconfig.py`将加载到配置中。像这样做：

```py
# in the console
export  PATH_TO_CONFIGURATION=/home/youruser/someconfig.py

```

然后创建配置：

```py
# someconfig.py
import logging
LOG_LEVEL = logging.WARNING
```

早期的配置方案都有相同的结果。

### 提示

请注意，`from_envvar`将从运行项目的用户加载环境变量。如果将环境变量导出到您的用户并作为另一个用户（如 www-data）运行项目，则可能无法找到您的配置。

# Flask-Principal 和 Flask-Login（又名蝙蝠侠和罗宾）

如项目页面所述（[`pythonhosted.org/Flask-Principal/`](https://pythonhosted.org/Flask-Principal/)），Flask-Principal 是一个权限扩展。它管理谁可以访问什么以及在什么程度上。通常情况下，您应该与身份验证和会话管理器一起使用它，就像 Flask-Login 的情况一样，这是我们将在本节中学习的另一个扩展。

Flask-Principal 通过四个简单的实体处理权限：**Identity**，**IdentityContext**，**Need**和**Permission**。

+   **Identity**：这意味着 Flask-Principal 识别用户的方式。

+   **IdentityContext**：这意味着针对权限测试的用户上下文。它用于验证用户是否有权执行某些操作。它可以用作装饰器（阻止未经授权的访问）或上下文管理器（仅执行）。

**Need**是您需要满足的标准（啊哈时刻！），以便做某事，比如拥有角色或权限。Principal 提供了一些预设的需求，但您也可以轻松创建自己的需求，因为 Need 只是一个命名元组，就像这样一个：

```py
from collections import namedtuplenamedtuple('RoleNeed', ['role', 'admin'])
```

+   **权限**：这是一组需要，应满足以允许某事。将其解释为资源的守护者。

鉴于我们已经设置好了我们的授权扩展，我们需要针对某些内容进行授权。一个常见的情况是将对管理界面的访问限制为管理员（不要说任何话）。为此，我们需要确定谁是管理员，谁不是。Flask-Login 可以通过提供用户会话管理（登录和注销）来帮助我们。让我们尝试一个例子。首先，确保安装了所需的依赖项：

```py
pip install flask-wtf flask-login flask-principal flask-sqlalchemy

```

然后：

```py
# coding:utf-8
# this example is based in the examples available in flask-login and flask-principal docs

from flask_wtf import Form

from wtforms import StringField, PasswordField, ValidationError
from wtforms import validators

from flask import Flask, flash, render_template, redirect, url_for, request, session, current_app
from flask.ext.login import UserMixin
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext.login import LoginManager, login_user, logout_user, login_required, current_user
from flask.ext.principal import Principal, Permission, Identity, AnonymousIdentity, identity_changed
from flask.ext.principal import RoleNeed, UserNeed, identity_loaded

principal = Principal()
login_manager = LoginManager()
login_manager.login_view = 'login_view'
# you may also overwrite the default flashed login message
# login_manager.login_message = 'Please log in to access this page.'
db = SQLAlchemy()

# Create a permission with a single Need
# we use it to see if an user has the correct rights to do something
admin_permission = Permission(RoleNeed('admin'))
```

由于我们的示例现在太大了，我们将逐步理解它。首先，我们进行必要的导入并创建我们的扩展实例。我们为`login_manager`设置`login_view`，以便它知道如果用户尝试访问需要用户身份验证的页面时应该重定向到哪里。请注意，Flask-Principal 不处理或跟踪已登录的用户。这是 Flask-Login 的魔术！

我们还创建了我们的`admin_permission`。我们的管理员权限只有一个需求：角色管理员。这样，我们定义了我们的权限接受用户时，这个用户需要拥有角色`admin`。

```py
# UserMixin implements some of the methods required by Flask-Login
class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    active = db.Column(db.Boolean, default=False)
    username = db.Column(db.String(60), unique=True, nullable=False)
    password = db.Column(db.String(20), nullable=False)
    roles = db.relationship(
        'Role', backref='roles', lazy='dynamic')

    def __unicode__(self):
        return self.username

    # flask login expects an is_active method in your user model
    # you usually inactivate a user account if you don't want it
    # to have access to the system anymore
    def is_active(self):
        """
        Tells flask-login if the user account is active
        """
        return self.active

class Role(db.Model):
    """
    Holds our user roles
    """
    __tablename__ = 'roles'
    name = db.Column(db.String(60), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __unicode__(self):
        return self.name
```

我们在这里有两个模型，一个用于保存我们的用户信息，另一个用于保存我们的用户角色。角色通常用于对用户进行分类，比如`admin`；您的系统中可能有三个管理员，他们都将拥有管理员角色。因此，如果权限正确配置，他们都将能够执行“管理员操作”。请注意，我们为用户定义了一个`is_active`方法。该方法是必需的，我建议您始终覆盖它，即使`UserMixin`已经提供了实现。`is_active`用于告诉`login`用户是否活跃；如果不活跃，他可能无法登录。

```py
class LoginForm(Form):
    def get_user(self):
        return User.query.filter_by(username=self.username.data).first()

    user = property(get_user)

    username = StringField(validators=[validators.InputRequired()])
    password = PasswordField(validators=[validators.InputRequired()])

    def validate_username(self, field):
        "Validates that the username belongs to an actual user"
        if self.user is None:
            # do not send a very specific error message here, otherwise you'll
            # be telling the user which users are available in your database
            raise ValidationError('Your username and password did not match')

    def validate_password(self, field):
        username = field.data
        user = User.query.get(username)

        if user is not None:
            if not user.password == field.data:
                raise ValidationError('Your username and password did not match')
```

在这里，我们自己编写了`LoginForm`。你可能会说：“为什么不使用`model_form`呢？”嗯，在这里使用`model_form`，您将不得不使用您的应用程序初始化数据库（您目前还没有）并设置上下文。太麻烦了。

我们还定义了两个自定义验证器，一个用于检查`username`是否有效，另一个用于检查`password`和`username`是否匹配。

### 提示

请注意，我们为这个特定表单提供了非常广泛的错误消息。我们这样做是为了避免向可能的攻击者提供太多信息。

```py
class Config(object):
    "Base configuration class"
    DEBUG = False
    SECRET_KEY = 'secret'
    SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/ex03.db'

class Dev(Config):
    "Our dev configuration"
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/dev.db'

def setup(app):
    # initializing our extensions ; )
    db.init_app(app)
    principal.init_app(app)
    login_manager.init_app(app)

    # adding views without using decorators
    app.add_url_rule('/admin/', view_func=admin_view)
    app.add_url_rule('/admin/context/', view_func=admin_only_view)
    app.add_url_rule('/login/', view_func=login_view, methods=['GET', 'POST'])
    app.add_url_rule('/logout/', view_func=logout_view)

    # connecting on_identity_loaded signal to our app
    # you may also connect using the @identity_loaded.connect_via(app) decorator
    identity_loaded.connect(on_identity_loaded, app, False)

# our application factory
def app_factory(name=__name__, config=Dev):
    app = Flask(name)
    app.config.from_object(config)
    setup(app)
    return app
```

在这里，我们定义了我们的配置对象，我们的`app`设置和应用程序工厂。我会说，设置是棘手的部分，因为它使用`app`方法注册视图，而不是装饰器（是的，与使用`@app.route`相同的结果），并且我们将我们的`identity_loaded`信号连接到我们的应用程序，以便用户身份在每个请求中都被加载和可用。我们也可以将其注册为装饰器，就像这样：

```py
@identity_loaded.connect_via(app)

# we use the decorator to let the login_manager know of our load_user
# userid is the model id attribute by default
@login_manager.user_loader
def load_user(userid):
    """
    Loads an user using the user_id

    Used by flask-login to load the user with the user id stored in session
    """
    return User.query.get(userid)

def on_identity_loaded(sender, identity):
    # Set the identity user object
    identity.user = current_user

    # in case you have resources that belong to a specific user
    if hasattr(current_user, 'id'):
        identity.provides.add(UserNeed(current_user.id))

    # Assuming the User model has a list of roles, update the
    # identity with the roles that the user provides
    if hasattr(current_user, 'roles'):
        for role in current_user.roles:
            identity.provides.add(RoleNeed(role.name))
```

`load_user` 函数是 Flask-Login 要求的，用于使用会话存储中存储的`userid`加载用户。如果没有找到`userid`，它应该返回`None`。不要在这里抛出异常。

`on_identity_loaded` 被注册到 `identity_loaded` 信号，并用于加载存储在模型中的身份需求。这是必需的，因为 Flask-Principal 是一个通用解决方案，不知道您如何存储权限。

```py
def login_view():
    form = LoginForm()

    if form.validate_on_submit():
        # authenticate the user...
        login_user(form.user)

        # Tell Flask-Principal the identity changed
        identity_changed.send(
            # do not use current_app directly
            current_app._get_current_object(),
            identity=Identity(form.user.id))
        flash("Logged in successfully.")
        return redirect(request.args.get("next") or url_for("admin_view"))

    return render_template("login.html", form=form)

@login_required  # you can't logout if you're not logged
def logout_view():
    # Remove the user information from the session
    # Flask-Login can handle this on its own = ]
    logout_user()

    # Remove session keys set by Flask-Principal
    for key in ('identity.name', 'identity.auth_type'):
        session.pop(key, None)

    # Tell Flask-Principal the user is anonymous
    identity_changed.send(
        current_app._get_current_object(),
        identity=AnonymousIdentity())

    # it's good practice to redirect after logout
    return redirect(request.args.get('next') or '/')
```

`login_view` 和 `logout_view` 就像它们的名字一样：一个用于认证，另一个用于取消认证用户。在这两种情况下，您只需确保调用适当的 Flask-Login 函数（`login_user` 和 `logout_user`），并发送适当的 Flask-Principal 信号（并在注销时清除会话）。

```py
# I like this approach better ...
@login_required
@admin_permission.require()
def admin_view():
    """
    Only admins can access this
    """
    return render_template('admin.html')

# Meh ...
@login_required
def admin_only_view():
    """
    Only admins can access this
    """
    with admin_permission.require():
        # using context
        return render_template('admin.html')
```

最后，我们有我们的实际视图：`admin_view` 和 `admin_only_view`。它们两者都做同样的事情，它们检查用户是否使用 Flask-Login 登录，然后检查他们是否有足够的权限来访问视图。这里的区别是，在第一种情况下，`admin_view`使用权限作为装饰器来验证用户的凭据，并在第二种情况下作为上下文。

```py
def populate():
    """
    Populates our database with a single user, for testing ; )

    Why not use fixtures? Just don't wanna ...
    """
    user = User(username='student', password='passwd', active=True)
    db.session.add(user)
    db.session.commit()
    role = Role(name='admin', user_id=user.id)
    db.session.add(role)
    db.session.commit()

if __name__ == '__main__':
    app = app_factory()

    # we need to use a context here, otherwise we'll get a runtime error
    with app.test_request_context():
        db.drop_all()
        db.create_all()
        populate()

    app.run()
```

`populate` 用于在我们的数据库中添加适当的用户和角色，以便您进行测试。

### 提示

关于我们之前的例子需要注意的一点：我们在用户数据库中使用了纯文本。在实际的代码中，你不想这样做，因为用户通常会在多个网站使用相同的密码。如果密码是纯文本，任何访问数据库的人都能知道它并测试它是否与敏感网站匹配。[`flask.pocoo.org/snippets/54/`](http://flask.pocoo.org/snippets/54/)中提供的解决方案可能有助于避免这种情况。

现在这是一个你可以与前面的代码一起使用的`base.html`模板的示例：

```py
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{% block title %}{% endblock %}</title>

  <link rel="stylesheet" media="screen,projection"
    href="https://cdnjs.cloudflare.com/ajax/libs/materialize/0.96.1/css/materialize.min.css" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
  <style type="text/css">
    .messages{
      position: fixed;
      list-style: none;
      margin:0px;
      padding: .5rem 2rem;
      bottom: 0; left: 0;
      width:100%;
      background-color: #abc;
      text-align: center;
    }
  </style>
</head>
<body>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
    <ul class='messages'>
        {% for message in messages %}
        <li>{{ message }}</li>
        {% endfor %}
    </ul>
    {% endif %}
  {% endwith %}

  <header>
     <nav>
      <div class="container nav-wrapper">
        {% if current_user.is_authenticated() %}
        <span>Welcome to the admin interface, {{ current_user.username }}</span>
        {% else %}<span>Welcome, stranger</span>{% endif %}

        <ul id="nav-mobile" class="right hide-on-med-and-down">
          {% if current_user.is_authenticated() %}
          <li><a href="{{ url_for('logout_view') }}?next=/admin/">Logout</a></li>
          {% else %}
          <li><a href="{{ url_for('login_view') }}?next=/admin/">Login</a></li>
          {% endif %}
        </ul>
      </div>
    </nav>
  </header>
  <div class="container">
    {% block content %}{% endblock %}
  </div>
  <script type="text/javascript" src="img/jquery-2.1.1.min.js"></script>
  <script src="img/materialize.min.js"></script>
</body>
</html>
```

请注意，我们在模板中使用`current_user.is_authenticated()`来检查用户是否经过身份验证，因为`current_user`在所有模板中都可用。现在，尝试自己编写`login.html`和`admin.html`，并扩展`base.html`。

## 管理员就像老板一样

Django 之所以如此出名的原因之一是因为它有一个漂亮而灵活的管理界面，我们也想要一个！

就像 Flask-Principal 和 Flask-Login 一样，我们将用来构建我们的管理界面的扩展 Flask-Admin 不需要特定的数据库来使用。你可以使用 MongoDB 作为关系数据库（与 SQLAlchemy 或 PeeWee 一起），或者你喜欢的其他数据库。

与 Django 相反，Django 的管理界面专注于应用程序/模型，而 Flask-Admin 专注于页面/模型。你不能（没有一些重编码）将整个蓝图（Flask 的 Django 应用程序等效）加载到管理界面中，但你可以为你的蓝图创建一个页面，并将蓝图模型注册到其中。这种方法的一个优点是你可以轻松选择所有模型将被列出的位置。

在我们之前的例子中，我们创建了两个模型来保存我们的用户和角色信息，所以，让我们为这两个模型创建一个简单的管理员界面。我们确保我们的依赖已安装：

```py
pip install flask-admin

```

然后：

```py
# coding:utf-8

from flask import Flask
from flask.ext.admin import Admin
from flask.ext.admin.contrib.sqla import ModelView
from flask.ext.login import UserMixin
from flask.ext.sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    active = db.Column(db.Boolean, default=False)
    username = db.Column(db.String(60), unique=True, nullable=False)
    password = db.Column(db.String(20), nullable=False)
    roles = db.relationship(
        'Role', backref='roles', lazy='dynamic')

    def __unicode__(self):
        return self.username

    # flask login expects an is_active method in your user model
    # you usually inactivate a user account if you don't want it
    # to have access to the system anymore
    def is_active(self):
        """
        Tells flask-login if the user account is active
        """
        return self.active

class Role(db.Model):
    """
    Holds our user roles
    """
    __tablename__ = 'roles'
    name = db.Column(db.String(60), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __unicode__(self):
        return self.name

# Flask and Flask-SQLAlchemy initialization here
admin = Admin()
admin.add_view(ModelView(User, db.session, category='Profile'))
admin.add_view(ModelView(Role, db.session, category='Profile'))

def app_factory(name=__name__):
    app = Flask(name)
    app.debug = True
    app.config['SECRET_KEY'] = 'secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/ex04.db'

    db.init_app(app)
    admin.init_app(app)
    return app

if __name__ == '__main__':
    app = app_factory()

    # we need to use a context here, otherwise we'll get a runtime error
    with app.test_request_context():
        db.drop_all()
        db.create_all()

    app.run()
```

在这个例子中，我们创建并初始化了`admin`扩展，然后使用`ModelView`向其注册我们的模型，这是一个为我们的模型创建**CRUD**的特殊类。运行此代码，尝试访问`http://127.0.0.1:5000/admin/`；您将看到一个漂亮的管理界面，顶部有一个主页链接，下面是一个包含两个链接的个人资料下拉菜单，指向我们的模型 CRUDs 的**用户**和**角色**。这只是一个非常基本的例子，不算太多，因为你不能拥有一个像那样对所有用户开放的管理界面。

我们向管理员视图添加身份验证和权限验证的一种方法是通过扩展`ModelView`和`IndexView`。我们还将使用一个称为`mixin`的很酷的设计模式：

```py
# coding:utf-8
# permissions.py

from flask.ext.principal import RoleNeed, UserNeed, Permission
from flask.ext.principal import Principal

principal = Principal()

# admin permission role
admin_permission = Permission(RoleNeed('admin'))

# END of FILE

# coding:utf-8
# admin.py

from flask import g
from flask.ext.login import current_user, login_required
from flask.ext.admin import Admin, AdminIndexView, expose
from flask.ext.admin.contrib.sqla import ModelView

from permissions import *

class AuthMixinView(object):
    def is_accessible(self):
        has_auth = current_user.is_authenticated()
        has_perm = admin_permission.allows(g.identity)
        return has_auth and has_perm

class AuthModelView(AuthMixinView, ModelView):
    @expose()
    @login_required
    def index_view(self):
        return super(ModelView, self).index_view()

class AuthAdminIndexView(AuthMixinView, AdminIndexView):
    @expose()
    @login_required
    def index_view(self):
        return super(AdminIndexView, self).index_view()

admin = Admin(name='Administrative Interface', index_view=AuthAdminIndexView())
```

我们在这里做什么？我们重写`is_accessible`方法，这样没有权限的用户将收到一个禁止访问的消息，并重写`AdminIndexView`和`ModelView`的`index_view`，添加`login_required`装饰器，将未经身份验证的用户重定向到登录页面。`admin_permission`验证给定的身份是否具有所需的权限集——在我们的例子中是`RoleNeed('admin')`。

### 提示

如果你想知道 mixin 是什么，请尝试这个链接[`stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful`](http://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful)。

由于我们的模型已经具有**创建、读取、更新、删除**（**CRUD**）和权限控制访问，我们如何修改我们的 CRUD 以仅显示特定字段，或阻止添加其他字段？

就像 Django Admin 一样，Flask-Admin 允许你通过设置类属性来更改你的 ModelView 行为。我个人最喜欢的几个是这些：

+   `can_create`: 这允许用户使用 CRUD 创建模型。

+   `can_edit`: 这允许用户使用 CRUD 更新模型。

+   `can_delete`: 这允许用户使用 CRUD 删除模型。

+   `list_template`、`edit_template`和`create_template`：这些是默认的 CRUD 模板。

+   `list_columns`: 这意味着列在列表视图中显示。

+   `column_editable_list`：这表示可以在列表视图中编辑的列。

+   `form`：这是 CRUD 用来编辑和创建视图的表单。

+   `form_args`：这用于传递表单字段参数。像这样使用它：

```py
form_args = {'form_field_name': {'parameter': 'value'}}  # parameter could be name, for example
```

+   `form_overrides`：像这样使用它来覆盖表单字段：

```py
form_overrides = {'form_field': wtforms.SomeField}
```

+   `form_choices`：允许你为表单字段定义选择。像这样使用它：

```py
form_choices = {'form_field': [('value store in db', 'value display in the combo box')]}
```

一个例子看起来像这样：

```py
class AuthModelView(AuthMixinView, ModelView):
    can_edit= False
    form = MyAuthForm

    @expose()
    @login_required
    def index_view(self):
        return super(ModelView, self).index_view()
```

## 自定义页面

现在，如果你想要在管理界面中添加一个自定义的**报告页面**，你肯定不会使用模型视图来完成这个任务。对于这些情况，像这样添加一个自定义的`BaseView`：

```py
# coding:utf-8
from flask import Flask
from flask.ext.admin import Admin, BaseView, expose

class ReportsView(BaseView):
    @expose('/')
    def index(self):
        # make sure reports.html exists
        return self.render('reports.html')

app = Flask(__name__)
admin = Admin(app)
admin.add_view(ReportsView(name='Reports Page'))

if __name__ == '__main__':
    app.debug = True
    app.run()
```

现在你有了一个带有漂亮的报告页面链接的管理界面。不要忘记编写一个`reports.html`页面，以使前面的示例工作。

那么，如果你不希望链接显示在导航栏中，因为你已经在其他地方有了它，怎么办？覆盖`BaseView.is_visible`方法，因为它控制视图是否会出现在导航栏中。像这样做：

```py
class ReportsView(BaseView):
…
  def is_visible(self):
    return False
```

# 摘要

在这一章中，我们只是学习了一些关于用户授权和认证的技巧，甚至尝试创建了一个管理界面。这是相当多的知识，将在你日常编码中帮助你很多，因为安全性（确保人们只与他们可以和应该互动的内容进行互动）是一个非常普遍的需求。

现在，我的朋友，你知道如何开发健壮的 Flask 应用程序，使用 MVC、TDD、与权限和认证控制集成的关系型和 NoSQL 数据库；表单；如何实现跨站点伪造保护；甚至如何使用开箱即用的管理工具。

我们的研究重点是了解 Flask 开发世界中所有最有用的工具（当然是我认为的），以及如何在一定程度上使用它们。由于范围限制，我们没有深入探讨任何一个，但基础知识肯定是展示过的。

现在，你可以进一步提高对每个介绍的扩展和库的理解，并寻找新的扩展。下一章也试图在这个旅程中启发你，建议阅读材料、文章和教程（等等）。

希望你到目前为止已经喜欢这本书，并且对最后的笔记感到非常愉快。
