# 第四章：请填写这张表格，夫人

你有没有想象过当你在网站上填写表单并点击最后的漂亮的**发送**按钮时会发生什么？好吧，你写的所有数据——评论、名称、复选框或其他任何东西——都会被编码并通过协议发送到服务器，然后服务器将这些信息路由到 Web 应用程序。Web 应用程序将验证数据的来源，读取表单，验证数据的语法和语义，然后决定如何处理它。你看到了吗？那里有一长串事件，每个链接都可能是问题的原因？这就是表单。

无论如何，没有什么可害怕的！Flask 可以帮助你完成这些步骤，但也有专门为此目的设计的工具。在本章中，我们将学习：

+   如何使用 Flask 编写和处理表单

+   如何验证表单数据

+   如何使用 WTForms 验证 Flask 中的表单

+   如何实现跨站点请求伪造保护

这实际上将是一个相当顺利的章节，有很多新信息，但没有复杂的东西。希望你喜欢！

# HTML 表单对于胆小的人

HTML 基本上是 Web 编写的语言。借助称为**标签**的特殊标记，可以为纯文本添加含义和上下文，将其转换为 HTML。对我们来说，HTML 是达到目的的手段。因此，如果你想了解更多，请在你喜欢的浏览器中打开[`www.w3schools.com/html/`](http://www.w3schools.com/html/)。我们没有完全覆盖 HTML 语法，也没有涉及到整个过程中的所有美妙魔法。

虽然我们不会详细介绍 HTML，但我们会专门介绍 HTML；我指的是`<form>`标签。事实是：每当你打开一个网页，有一些空白字段需要填写时，你很可能在填写 HTML 表单。这是从浏览器向服务器传输数据的最简单方式。这是如何工作的？让我们看一个例子：

```py
<!-- example 1 -->
<form method='post' action='.'>
<input type='text' name='username' />
<input type='password' name='passwd' />
<input type='submit' />
</form>
```

在上面的例子中，我们有一个完整的登录表单。它的开始由`<form>`标签定义，具有两个非必需的属性：`method`和`action`。`method`属性定义了当发送表单数据时你希望数据如何发送到服务器。它的值可以是`get`或`post`。只有当表单数据很小（几百个字符）、不敏感（如果其他人看到它并不重要）且表单中没有文件时，才应该使用`get`，这是默认值。这些要求存在的原因是，当使用`get`时，所有表单数据将被编码为参数附加到当前 URL 之后再发送。在我们的例子中，选择的方法是`post`，因为我们的输入字段之一是密码，我们不希望其他人查看我们的密码。使用`get`方法的一个很好的用例是搜索表单。例如：

```py
<!-- example 2 -->
<form action='.'>
<input type='search' name='search' />
</form>
```

在`示例 2`中，我们有一个简单的搜索表单。如果我们在`name`输入中填写搜索词`SearchItem`并点击*Enter*，URL 将如下所示：

[`mydomain.com/?search=SearchItem`](http://mydomain.com/?search=SearchItem)

然后，前面的 URL 将保存到浏览器历史记录中，任何有权访问它的人都可以看到上一个用户在搜索什么。对于敏感数据来说，这是不好的。

无论如何，回到*示例 1*。第二个属性`action`对于告诉浏览器应该接收和响应表单数据的 URL 非常有用。我们使用`'.'`作为它的值，因为我们希望表单数据被发送到当前 URL。

接下来的两行是我们的输入字段。输入字段用于收集用户数据，与名称可能暗示的相反，输入字段可以是`input`、`textarea`或`select`元素。在使用输入字段时，始终记得使用属性`name`对它们进行命名，因为这有助于在 Web 应用程序中处理它们。

在第三行，我们有一个特殊的输入字段，它不一定有任何要发送的数据，即提交输入按钮。默认情况下，如果在`input`元素具有焦点时按下*Enter*，或者按下提交按钮，表单将被发送。我们的*示例 1*是后者。

哇！终于，我们的表单已经编写和解释完毕。有关输入字段可能类型的详尽列表，请查看[`www.w3schools.com/tags/tag_input.asp`](http://www.w3schools.com/tags/tag_input.asp)。

# 处理表单

现在让我们看看如何将*示例 1*中的表单与应用程序集成：

```py
# coding:utf-8

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['get', 'post'])
def login_view():
    # the methods that handle requests are called views, in flask
    msg = ''

    # form is a dictionary like attribute that holds the form data
    if request.method == 'POST':
      username = request.form["username"]
        passwd = request.form["passwd"]

        # static useless validation
        if username == 'you' and passwd == 'flask':
            msg = 'Username and password are correct'
        else:
            msg = 'Username or password are incorrect'
    return render_template('form.html', message=msg)

if __name__=='__main__':
    app.run()
```

在前面的例子中，我们定义了一个名为`login_view`的视图，该视图接受`get`或`post`请求；当请求为`post`时（如果是由`get`请求发送的表单，则我们忽略该表单），我们获取`username`和`passwd`的值；然后我们运行一个非常简单的验证，并相应地更改`msg`的值。

### 提示

注意：在 Flask 中，视图不同于 MVC 中的视图。在 Flask 中，视图是接收请求并返回响应的组件，可以是函数或类。

您看到我们在示例中处理的`request`变量了吗？这是当前活动`request`上下文的代理。这就是为什么`request.form`指向发送的表单数据。

现在，如果您收到一个编码在 URL 中的参数，您将如何获取它，考虑到请求 URL 是`http://localhost:5000/?page=10`？

```py
# inside a flask view
def some_view():
    try:
        page = int(request.args.get('page', 1))
        assert page == 10
    except ValueError:
        page = 1
    ...
```

在分页时，前面的例子是非常常见的。与以前一样，`request.args`只与当前用户请求相关。很简单！

到目前为止，我们用内联验证处理表单验证非常糟糕。不再这样做了！让我们从现在开始尝试一些更花哨的东西。

# WTForms 和你

WTForms（[`github.com/wtforms/wtforms`](https://github.com/wtforms/wtforms)）是一个独立的强大的表单处理库，允许您从类似表单的类生成 HTML 表单，实现字段和表单验证，并包括跨源伪造保护（黑客可能尝试在您的 Web 应用程序中利用的一个恶意漏洞）。我们当然不希望发生这种情况！

首先，要安装 WTForms 库，请使用以下命令：

```py
pip install wtforms

```

现在让我们编写一些表单。WTForms 表单是扩展`Form`类的类。就是这么简单！让我们创建一个登录表单，可以与我们之前的登录示例一起使用：

```py
from wtforms import Form, StringField, PasswordField
class LoginForm(Form):
    username = StringField(u'Username:')
    passwd = PasswordField(u'Password:')
```

在前面的代码中，我们有一个带有两个字段`username`和`passwd`的表单，没有验证。只需在模板中构建一个表单就足够了，就像这样：

```py
<form method='post'>
{% for field in form %}
    {{ field.label }}
    {{ field }}
    {% if field.errors %}
        {% for error in field.errors %}
            <div class="field_error">{{ error }}</div>
        {% endfor %}
    {% endif %}
{% endfor %}
</form>
```

如前面的代码所示，您可以迭代 WTForms 表单的字段，每个字段都有一些有用的属性，您可以使用这些属性使您的 HTML 看起来很好，比如`label`和`errors`。`{{ field }}`将为您呈现一个普通的 HTML 输入元素。有些情况下，您可能希望为输入元素设置特殊属性，例如`required`，告诉浏览器如果为空，则不应提交给定字段。为了实现这一点，调用`field`作为一个函数，就像这样：

```py
{% if field.flags.required %}
{{ field(required='required') }}
{% endif %}
```

您可以根据示例传递任何所需的参数，如`placeholder`或`alt`。Flask-Empty（[`github.com/italomaia/flask-empty`](https://github.com/italomaia/flask-empty)）在其宏中有一个很好的示例。

WTForms 使用标志系统，以允许您检查何时对字段应用了一些验证。如果字段有一个`required`验证规则，`fields.flags`属性中的`required`标志将设置为 true。但是 WTForms 验证是如何工作的呢？

在 Flask 中，验证器是您添加到`validators`字段的可调用对象，或者是格式为`validate_<field>(form, field)`的类方法。它允许您验证字段数据是否符合要求，否则会引发`ValidationError`，解释出了什么问题。让我们看看我们漂亮的登录表单示例如何进行一些验证：

```py
# coding:utf-8
from wtforms import Form, ValidationError
from wtforms import StringField, PasswordField
from wtforms.validators import Length, InputRequired
from werkzeug.datastructures import MultiDict

import re

def is_proper_username(form, field):
    if not re.match(r"^\w+$", field.data):
        msg = '%s should have any of these characters only: a-z0-9_' % field.name
        raise ValidationError(msg)

class LoginForm(Form):
    username = StringField(
        u'Username:', [InputRequired(), is_proper_username, Length(min=3, max=40)])
    password = PasswordField(
        u'Password:', [InputRequired(), Length(min=5, max=12)])

    @staticmethod
    def validate_password(form, field):
        data = field.data
        if not re.findall('.*[a-z].*', data):
            msg = '%s should have at least one lowercase character' % field.name
            raise ValidationError(msg)
        # has at least one uppercase character
        if not re.findall('.*[A-Z].*', data):
            msg = '%s should have at least one uppercase character' % field.name
            raise ValidationError(msg)
        # has at least one number
        if not re.findall('.*[0-9].*', data):
            msg = '%s should have at least one number' % field.name
            raise ValidationError(msg)
        # has at least one special character
        if not re.findall('.*[^ a-zA-Z0-9].*', data):
            msg = '%s should have at least one special character' % field.name
            raise ValidationError(msg)

# testing our form
form = LoginForm(MultiDict([('username', 'italomaia'), ('password', 'lL2m@msbb')]))
print form.validate()
print form.errors
```

在上述代码中，我们有一个完整的表单示例，带有验证，使用类、方法和函数作为验证器以及一个简单的测试。我们的每个字段的第一个参数是字段标签。第二个参数是在调用`form.validate`方法时要运行的验证器列表（这基本上就是`form.validate`做的事情）。每个字段验证器都会按顺序运行，如果发现错误，则会引发`ValidationError`（并停止验证链调用）。

每个验证器都接收表单和字段作为参数，并必须使用它们进行验证。如`validate_password`所示，它是因为命名约定而为字段`password`调用的。`field.data`保存字段输入，因此您通常可以只验证它。

让我们了解每个验证器：

+   `Length`：验证输入值的长度是否在给定范围内（最小、最大）。

+   `InputRequired`：验证字段是否接收到值，任何值。

+   `is_proper_username`：验证字段值是否与给定的正则表达式匹配。（还有一个内置验证器，用于将正则表达式与给定值匹配，称为**Regexp**。您应该尝试一下。）

+   `validate_password`：验证字段值是否符合给定的正则表达式规则组。

在我们的示例测试中，您可能已经注意到了使用`werkzeug`库中称为`MultiDict`的特殊类似字典的类。它被使用是因为`formdata`参数，它可能接收您的`request.form`或`request.args`，必须是`multidict-type`。这基本上意味着您不能在这里使用普通字典。

调用`form.validate`时，将调用所有验证器。首先是字段验证器，然后是`class`方法字段验证器；`form.errors`是一个字典，其中包含在调用 validate 后找到的所有字段错误。然后您可以对其进行迭代，以在模板、控制台等中显示您找到的内容。

# Flask-WTF

Flask 使用扩展以便与第三方库透明集成。WTForms 与 Flask-WTF 是这样的一个很好的例子，我们很快就会看到。顺便说一句，Flask 扩展是一段代码，以可预测的方式与 Flask 集成其配置、上下文和使用。这意味着扩展的使用方式非常相似。现在确保在继续之前在您的虚拟环境中安装了 Flask-WTF：

```py
# oh god, so hard... not!
pip flask-wtf

```

从[`flask-wtf.readthedocs.org/`](http://flask-wtf.readthedocs.org/)，项目网站，我们得到了 Flask-WTF 提供的以下功能列表：

+   与 WTForms 集成

+   使用 CSRF 令牌保护表单

+   与 Flask-Uploads 一起工作的文件上传

+   全局 CSRF 保护

+   Recaptcha 支持

+   国际化集成

我们将在本章中看到前两个功能，而第三个将在第十章中讨论，*现在怎么办？*。最后三个功能将不在本书中涵盖。我们建议您将它们作为作业进行探索。

## 与 WTForms 集成

Flask-WTF 在集成时使用了关于`request`的小技巧。由于`request`实现了对当前请求和请求数据的代理，并且在`request`上下文中可用，扩展`Form`默认会获取`request.form`数据，节省了一些输入。

我们的`login_view`示例可以根据迄今为止讨论的内容进行重写，如下所示：

```py
# make sure you're importing Form from flask_wtf and not wtforms
from flask_wtf import Form

# --//--
@app.route('/', methods=['get', 'post'])
def login_view():
    # the methods that handle requests are called views, in flask
    msg = ''
    # request.form is passed implicitly; implies POST
    form = LoginForm()
    # if the form should also deal with form.args, do it like this:
    # form = LoginForm(request.form or request.args)

    # checks that the submit method is POST and form is valid
    if form.validate_on_submit():
        msg = 'Username and password are correct'
    else:
        msg = 'Username or password are incorrect'
    return render_template('form.html', message=msg)
```

我们甚至可以更进一步，因为我们显然是完美主义者：

```py
# flash allows us to send messages to the user template without
# altering the returned context
from flask import flash
from flask import redirect
@app.route('/', methods=['get', 'post'])
def login_view():
    # msg is no longer necessary. We will use flash, instead
    form = LoginForm()

    if form.validate_on_submit():
        flash(request, 'Username and password are correct')
        # it's good practice to redirect after a successful form submit
        return redirect('/')
    return render_template('form.html', form=form)
```

在模板中，将`{{ message }}`替换为：

```py
{# 
beautiful example from 
http://flask.pocoo.org/docs/0.10/patterns/flashing/#simple-flashing 
#}
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul class='messages'>
    {% for message in messages %}
      <li>{{ message }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endwith %}
```

`get_flashed_messages`默认在模板上下文中可用，并为当前用户提供尚未显示的所有闪现消息。然后我们使用`with`缓存它，检查它是否不为空，然后对其进行迭代。

### 提示

闪现消息在重定向时特别有用，因为它们不受响应上下文的限制。

## 使用 CSRF 令牌保护表单

**跨站点请求伪造**（**CSRF**）发生在一个网站试图利用另一个网站对你的浏览器的信任（假设你是用户）时。基本上，你正在访问的网站会尝试获取或更改你已经访问并进行身份验证的网站的信息。想象一下，你正在访问一个网站，该网站有一张图片，加载了你已经进行身份验证的另一个网站的 URL；想象一下，给定的 URL 请求了前一个网站的一个动作，并且该动作改变了你的账户的某些内容——例如，它的状态被修改为非活动状态。嗯，这就是 CSRF 攻击的一个简单案例。另一个常见的情况是发送 JSONP 请求。如果被攻击的网站，也就是你没有访问的那个网站，接受 JSONP 表单替换（JSONP 用于跨域请求）并且没有 CRSF 保护，那么你将面临更加恶劣的攻击。

WTForms 自带 CSRF 保护；Flask-WTF 将整个过程与 Flask 粘合在一起，使你的生活更轻松。为了在使用该扩展时具有 CSRF 保护，你需要设置`secret_key`，就是这样：

```py
app.secret_key = 'some secret string value' # ex: import os; os.urandom(24)
```

然后，每当你编写一个应该具有 CSRF 保护的表单时，只需确保向其中添加 CSRF 令牌，就像这样：

```py
<form method='post'>{{ form.csrf_token }}
{% for field in form if field.name != 'csrf_token' %}
    <div class="field">
    {{ field.label }} {{ field }}
    </div>
    {% if field.errors %}
        {% for error in field.errors %}
        <div class="field_error">{{ error }}</div>
        {% endfor %}
    {% endif %}
{% endfor %}
<input type='submit' />
</form>
```

当表单被接收时，CSRF 令牌会与用户会话中注册的内容进行检查。如果它们匹配，表单的来源就是安全的。这是一种安全的方法，因为一个网站无法读取另一个网站设置的 cookie。

在不希望表单受到 CSRF 保护的情况下，不要添加令牌。如果希望取消对表单的保护，必须关闭表单的 CSRF 保护，就像这样：

```py
form = Form(csrf_enabled=False)
```

在使用`get`方法但同时又使用表单进行验证的搜索字段的情况下，*可能*需要取消对表单的保护。

## 挑战

创建一个 Web 应用程序，接收一个名字，然后回答：“你好，<NAME>”。如果表单为空发送，应显示错误消息。如果给定的名字是“查克·诺里斯”，答案应该是“旋风踢！”。

创建一个 Web 应用程序，显示一张图片，并询问用户看到了什么。然后应用程序应验证答案是否正确。如果不正确，向用户显示错误消息。否则，祝贺用户并显示一张新图片。使用 Flask-WTF。

创建一个具有四种运算的计算器。它应该有用户可以点击的所有数字和运算符。确保它看起来像一个计算器（因为我们是完美主义者！），并且在用户尝试一些恶意操作时进行投诉，比如将 0 除以 0。

# 总结

学到了这么多...我能说什么呢！试试看也没什么坏处，对吧？嗯，我们已经学会了如何编写 HTML 表单；使用 Flask 读取表单；编写 WTForms 表单；使用纯 Python 和表单验证器验证表单数据；以及编写自定义验证器。我们还看到了如何使用 Flask-WTF 来编写和验证我们的表单，以及如何保护我们的应用程序免受 CSRF 攻击。

在下一章中，我们将看看如何使用出色、易于使用的库将 Web 应用程序数据存储在关系型和非关系型数据库中，并如何将它们与 Flask 集成。还将进行数据库的简要概述，以便更顺畅地吸收知识。
