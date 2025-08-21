# 第十章：认证模块

认证模块在为用户创建空间时节省了大量时间。以下是该模块的主要优势：

+   与用户相关的主要操作得到了简化（连接、帐户激活等）

+   使用该系统可以确保一定级别的安全性

+   页面的访问限制可以很容易地完成

这是一个非常有用的模块，我们甚至在不知不觉中已经使用了它。事实上，对管理模块的访问是通过认证模块执行的。我们在生成数据库时创建的用户是站点的第一个用户。

这一章大大改变了我们之前编写的应用程序。在本章结束时，我们将有：

+   修改我们的 UserProfile 模型，使其与模块兼容

+   创建了一个登录页面

+   修改了添加开发人员和监督员页面

+   增加对连接用户的访问限制

# 如何使用认证模块

在本节中，我们将学习如何通过使我们的应用程序与模块兼容来使用认证模块。

## 配置 Django 应用程序

通常情况下，我们不需要为管理模块在我们的`TasksManager`应用程序中工作做任何特殊的操作。事实上，默认情况下，该模块已启用，并允许我们使用管理模块。但是，可能会在禁用了 Web Django 认证模块的站点上工作。我们将检查模块是否已启用。

在`settings.py`文件的`INSTALLED_APPS`部分中，我们必须检查以下行：

```py
'django.contrib.auth',
```

## 编辑 UserProfile 模型

认证模块有自己的用户模型。这也是我们创建`UserProfile`模型而不仅仅是用户的原因。它是一个已经包含一些字段的模型，比如昵称和密码。要使用管理模块，必须在`Python33/Lib/site-package/django/contrib/auth/models.py`文件中使用用户模型。

我们将修改`models.py`文件中的`UserProfile`模型，将其变为以下内容：

```py
class UserProfile(models.Model):
  user_auth = models.OneToOneField(User, primary_key=True)
  phone = models.CharField(max_length=20, verbose_name="Phone number", null=True, default=None, blank=True)
  born_date = models.DateField(verbose_name="Born date", null=True, default=None, blank=True)
  last_connexion = models.DateTimeField(verbose_name="Date of last connexion", null=True, default=None, blank=True)
years_seniority = models.IntegerField(verbose_name="Seniority", default=0)
def __str__(self):
  return self.user_auth.username
```

我们还必须在`models.py`中添加以下行：

```py
from django.contrib.auth.models import User
```

在这个新模型中，我们有：

+   创建了与导入的用户模型的`OneToOneField`关系

+   删除了用户模型中不存在的字段

`OneToOne`关系意味着对于每个记录的`UserProfile`模型，都会有一个用户模型的记录。在做所有这些的过程中，我们深度修改了数据库。鉴于这些变化，并且因为密码以哈希形式存储，我们将不使用 South 进行迁移。

可以保留所有数据并使用 South 进行迁移，但是我们应该开发一个特定的代码来将`UserProfile`模型的信息保存到用户模型中。该代码还应该为密码生成哈希，但这将是很长的过程，而且不是本书的主题。要重置 South，我们必须执行以下操作：

+   删除`TasksManager/migrations`文件夹以及该文件夹中包含的所有文件

+   删除`database.db`文件

要使用迁移系统，我们必须使用关于模型的章节中已经使用过的以下命令：

```py
manage.py schemamigration TasksManager --initial
manage.py syncdb –migrate
```

删除数据库后，我们必须删除`create_developer.py`中的初始数据。我们还必须删除`developer_detail`的 URL 和`index.html`中的以下行：

```py
<a href="{% url "developer_detail" "2" %}">Detail second developer (The second user must be a developer)</a><br />
```

# 添加用户

允许您添加开发人员和监督员的页面不再起作用，因为它们与我们最近的更改不兼容。我们将更改这些页面以整合我们的样式更改。`create_supervisor.py`文件中包含的视图将包含以下代码：

```py
from django.shortcuts import render
from TasksManager.models import Supervisor
from django import forms
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User
def page(request):
  if request.POST:
    form = Form_supervisor(request.POST)
    if form.is_valid(): 
      name           = form.cleaned_data['name']
      login          = form.cleaned_data['login']
      password       = form.cleaned_data['password']
      specialisation = form.cleaned_data['specialisation']
      email          = form.cleaned_data['email']
      new_user = User.objects.create_user(username = login, email = email, password=password)
      # In this line, we create an instance of the User model with the create_user() method. It is important to use this method because it can store a hashcode of the password in database. In this way, the password cannot be retrieved from the database. Django uses the PBKDF2 algorithm to generate the hash code password of the user.
      new_user.is_active = True
      # In this line, the is_active attribute defines whether the user can connect or not. This attribute is false by default which allows you to create a system of account verification by email, or other system user validation.
      new_user.last_name=name
      # In this line, we define the name of the new user.
      new_user.save()
      # In this line, we register the new user in the database.
      new_supervisor = Supervisor(user_auth = new_user, specialisation=specialisation)
      # In this line, we create the new supervisor with the form data. We do not forget to create the relationship with the User model by setting the property user_auth with new_user instance.
      new_supervisor.save()
      return HttpResponseRedirect(reverse('public_empty')) 
    else:
      return render(request, 'en/public/create_supervisor.html', {'form' : form})
  else:
    form = Form_supervisor()
  form = Form_supervisor()
  return render(request, 'en/public/create_supervisor.html', {'form' : form})
class Form_supervisor(forms.Form):
  name = forms.CharField(label="Name", max_length=30)
  login = forms.CharField(label = "Login")
  email = forms.EmailField(label = "Email")
  specialisation = forms.CharField(label = "Specialisation")
  password = forms.CharField(label = "Password", widget = forms.PasswordInput)
  password_bis = forms.CharField(label = "Password", widget = forms.PasswordInput) 
  def clean(self): 
    cleaned_data = super (Form_supervisor, self).clean() 
    password = self.cleaned_data.get('password') 
    password_bis = self.cleaned_data.get('password_bis')
    if password and password_bis and password != password_bis:
      raise forms.ValidationError("Passwords are not identical.") 
    return self.cleaned_data
```

`create_supervisor.html`模板保持不变，因为我们正在使用 Django 表单。

您可以更改`create_developer.py`文件中的`page()`方法，使其与认证模块兼容（您可以参考可下载的 Packt 代码文件以获得进一步的帮助）：

```py
def page(request):
  if request.POST:
    form = Form_inscription(request.POST)
    if form.is_valid():
      name          = form.cleaned_data['name']
      login         = form.cleaned_data['login']
      password      = form.cleaned_data['password']
      supervisor    = form.cleaned_data['supervisor'] 
      new_user = User.objects.create_user(username = login, password=password)
      new_user.is_active = True
      new_user.last_name=name
      new_user.save()
      new_developer = Developer(user_auth = new_user, supervisor=supervisor)
      new_developer.save()
      return HttpResponse("Developer added")
    else:
      return render(request, 'en/public/create_developer.html', {'form' : form})
  else:
    form = Form_inscription()
    return render(request, 'en/public/create_developer.html', {'form' : form})
```

我们还可以修改`developer_list.html`，内容如下：

```py
{% extends "base.html" %}
{% block title_html %}
    Developer list
{% endblock %}
{% block h1 %}
    Developer list
{% endblock %}
{% block article_content %}
    <table>
        <tr>
            <td>Name</td>
            <td>Login</td>
            <td>Supervisor</td>
        </tr>
        {% for dev in object_list %}
            <tr>
                <!-- The following line displays the __str__ method of the model. In this case it will display the username of the developer -->
                <td><a href="">{{ dev }}</a></td>
                <!-- The following line displays the last_name of the developer -->
                <td>{{ dev.user_auth.last_name }}</td>
                <!-- The following line displays the __str__ method of the Supervisor model. In this case it will display the username of the supervisor -->
                <td>{{ dev.supervisor }}</td>
            </tr>
        {% endfor %}
    </table>
{% endblock %}
```

# 登录和注销页面

现在您可以创建用户，必须创建一个登录页面，以允许用户进行身份验证。我们必须在`urls.py`文件中添加以下 URL：

```py
url(r'^connection$', 'TasksManager.views.connection.page', name="public_connection"),
```

然后，您必须创建`connection.py`视图，并使用以下代码：

```py
from django.shortcuts import render
from django import forms
from django.contrib.auth import authenticate, login
# This line allows you to import the necessary functions of the authentication module.
def page(request):
  if request.POST:
  # This line is used to check if the Form_connection form has been posted. If mailed, the form will be treated, otherwise it will be displayed to the user.
    form = Form_connection(request.POST) 
    if form.is_valid():
      username = form.cleaned_data["username"]
      password = form.cleaned_data["password"]
      user = authenticate(username=username, password=password)
      # This line verifies that the username exists and the password is correct.
      if user:
      # In this line, the authenticate function returns None if authentication has failed, otherwise it returns an object that validates the condition.
        login(request, user)
        # In this line, the login() function allows the user to connect.
    else:
      return render(request, 'en/public/connection.html', {'form' : form})
  else:
    form = Form_connection()
  return render(request, 'en/public/connection.html', {'form' : form})
class Form_connection(forms.Form):
  username = forms.CharField(label="Login")
  password = forms.CharField(label="Password", widget=forms.PasswordInput)
  def clean(self):
    cleaned_data = super(Form_connection, self).clean()
    username = self.cleaned_data.get('username')
    password = self.cleaned_data.get('password')
    if not authenticate(username=username, password=password):
      raise forms.ValidationError("Wrong login or password")
    return self.cleaned_data
```

然后，您必须创建`connection.html`模板，并使用以下代码：

```py
{% extends "base.html" %}
{% block article_content %}
  {% if user.is_authenticated %}
  <-- This line checks if the user is connected.-->
    <h1>You are connected.</h1>
    <p>
      Your email : {{ user.email }}
      <-- In this line, if the user is connected, this line will display his/her e-mail address.-->
    </p>
  {% else %}
  <!-- In this line, if the user is not connected, we display the login form.-->
    <h1>Connexion</h1>
    <form method="post" action="{{ public_connection }}">
      {% csrf_token %}
      <table>
        {{ form.as_table }}
      </table>
      <input type="submit" class="button" value="Connection" />
    </form>
  {% endif %}
{% endblock %}
```

当用户登录时，Django 将在会话变量中保存他/她的数据连接。此示例已经允许我们验证登录和密码对用户是透明的。确实，`authenticate()`和`login()`方法允许开发人员节省大量时间。Django 还为开发人员提供了方便的快捷方式，例如`user.is_authenticated`属性，用于检查用户是否已登录。用户更喜欢网站上有注销链接，特别是在从公共计算机连接时。我们现在将创建注销页面。

首先，我们需要创建带有以下代码的`logout.py`文件：

```py
from django.shortcuts import render
from django.contrib.auth import logout
def page(request):
    logout(request)
    return render(request, 'en/public/logout.html')
```

在先前的代码中，我们导入了身份验证模块的`logout()`函数，并将其与请求对象一起使用。此函数将删除请求对象的用户标识符，并删除其会话数据。

当用户注销时，他/她需要知道网站实际上已断开连接。让我们在`logout.html`文件中创建以下模板：

```py
{% extends "base.html" %}
{% block article_content %}
  <h1>You are not connected.</h1>
{% endblock %}
```

# 限制对已连接成员的访问

当开发人员实现身份验证系统时，通常是为了限制匿名用户的访问。在本节中，我们将看到控制对我们网页访问的两种方式。

## 限制对视图的访问

身份验证模块提供了简单的方法来防止匿名用户访问某些页面。确实，有一个非常方便的装饰器来限制对视图的访问。这个装饰器称为`login_required`。

在接下来的示例中，我们将使用设计师以以下方式限制对`create_developer`模块中的`page()`视图的访问： 

1.  首先，我们必须使用以下行导入装饰器：

```py
from django.contrib.auth.decorators import login_required
```

1.  然后，我们将在视图声明之前添加装饰器：

```py
@login_required
def page(request): # This line already exists. Do not copy it.
```

1.  通过添加这两行，只有已登录用户才能访问添加开发人员的页面。如果尝试在未连接的情况下访问页面，您将意识到这并不是很实用，因为获得的页面是 404 错误。要改进此问题，只需告诉 Django 连接 URL 是什么，通过在`settings.py`文件中添加以下行：

```py
LOGIN_URL = 'public_connection'
```

1.  通过这一行，如果用户尝试访问受保护的页面，他/她将被重定向到登录页面。您可能已经注意到，如果您未登录并单击**创建开发人员**链接，则 URL 包含一个名为 next 的参数。以下是 URL 的屏幕截图：![限制对视图的访问](img/00027.jpeg)

1.  此参数包含用户尝试查看的 URL。身份验证模块在用户连接时将用户重定向到该页面。为此，我们将修改我们创建的`connection.py`文件。我们添加导入`render()`函数以导入`redirect()`函数的行：

```py
from django.shortcuts import render, redirect
```

1.  要在用户登录后重定向用户，我们必须在包含代码 login(request, user)的行之后添加两行。需要添加两行：

```py
if request.GET.get('next') is not None:
  return redirect(request.GET['next'])
```

当用户会话已过期并且希望查看特定页面时，此系统非常有用。

## 限制对 URL 的访问

我们所见的系统不仅仅限制对 CBV 生成的页面的访问。为此，我们将使用相同的装饰器，但这次是在`urls.py`文件中。

我们将添加以下行以导入装饰器：

```py
from django.contrib.auth.decorators import login_required
```

我们需要更改对应于名为`create_project`的 URL 的行：

```py
url (r'^create_project$', login_required(CreateView.as_view(model=Project, template_name="en/public/create_project.html", success_url = 'index')), name="create_project"),
```

使用`login_required`装饰器非常简单，可以让开发人员不浪费太多时间。

# 摘要

在本章中，我们修改了我们的应用程序，使其与认证模块兼容。我们创建了允许用户登录和注销的页面。然后，我们学习了如何限制已登录用户对某些页面的访问。

在下一章中，我们将通过添加 AJAX 请求来提高应用程序的可用性。我们将学习 jQuery 的基础知识，然后学习如何使用它来向服务器发出异步请求。此外，我们还将学习如何处理来自服务器的响应。
