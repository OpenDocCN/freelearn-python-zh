# 第九章：扩展和部署

在本章中，我们将通过利用各种 Django 框架功能来准备我们的应用程序以在生产中部署。我们将添加对多种语言的支持，通过缓存和自动化测试来提高性能，并为生产环境配置项目。本章中有很多有趣和有用的信息，因此在将应用程序发布到网上之前，请确保您仔细阅读！

在本章中，您将学习以下主题：

+   向朋友发送邀请电子邮件

+   国际化（i18n）-提供多种语言的站点

+   缓存-在高流量期间提高站点性能

+   单元测试-自动化测试应用程序的过程

# 向朋友发送邀请电子邮件

使我们的用户邀请他们的朋友具有许多好处。如果他们的朋友已经使用我们的网站，那么他们更有可能加入我们的网站。加入后，他们还会邀请他们的朋友，依此类推，这意味着我们的应用程序会有越来越多的用户。因此，在我们的应用程序中包含“邀请朋友”的功能是一个好主意。

构建此功能需要以下组件：

+   一个邀请数据模型，用于在数据库中存储邀请

+   用户可以在其中输入他们朋友的电子邮件 ID 并发送邀请的表单

+   带有激活链接的邀请电子邮件

+   处理电子邮件中发送的激活链接的机制

在本节中，我们将实现这些组件中的每一个。但是，因为本节涉及发送电子邮件，我们首先需要通过向`settings.py`文件添加一些选项来配置 Django 发送电子邮件。因此，打开`settings.py`文件并添加以下行：

```py
  SITE_HOST = '127.0.0.1:8000'
  DEFAULT_FROM_EMAIL = 'MyTwitter <noreply@mytwitter.com>'
  EMAIL_HOST = 'mail.yourisp.com'
  EMAIL_PORT = ''
  EMAIL_HOST_USER = 'username+mail.yourisp.com'
  EMAIL_HOST_PASSWORD = ''
```

让我们看看前面代码中的每个变量都做了什么：

+   `SITE_HOST`：这是您服务器的主机名。现在将其保留为`127.0.0.1:8000`。在下一章中部署服务器时，我们将更改此设置。

+   `DEFAULT_FROM_EMAIL`：这是出站电子邮件服务器**From**字段中显示的电子邮件地址。对于主机用户名，请输入您的用户名加上您的电子邮件服务器，如前面的代码片段所示。如果您的 ISP 不需要这些字段，请将其留空。

+   `EMAIL_HOST`：这是您的电子邮件服务器的主机名。

+   `EMAIL_PORT`：这是出站电子邮件服务器的端口号。如果将其留空，则将使用默认值（25）。您还需要从 ISP 那里获取此信息。

+   `EMAIL_HOST_USER`和`EMAIL_HOST_PASSWORD`：这是 Django 发送的电子邮件的用户名和密码。

如果您的开发计算机没有运行邮件服务器，很可能是这种情况，那么您需要输入 ISP 的出站电子邮件服务器。联系您的 ISP 以获取更多信息。

要验证您的设置是否正确，请启动交互式 shell 并输入以下内容：

```py
>>> from django.core.mail import EmailMessage
>>> email = EmailMessage('Hello', 'World', to=['your_email@example.com'])
>>> email.send()

```

将`your_email@example.com`参数替换为您的实际电子邮件地址。如果前面的发送邮件调用没有引发异常并且您收到了邮件，那么一切都设置好了。否则，您需要与 ISP 验证您的设置并重试。

但是，如果您没有从 ISP 那里获得任何信息怎么办？然后我们尝试另一种方式：使用 Gmail 发送邮件（当然，不是作为`noreply@mytweet.com`，而是从您的真实电子邮件 ID）。让我们看看您需要对`MyTweeets`项目的`settings.py`文件进行哪些更改。

完全删除以前的`settings.py`文件条目，并添加以下内容：

```py
  EMAIL_USE_TLS = True
  EMAIL_HOST = 'smtp.gmail.com'
  EMAIL_HOST_USER = 'your-gmail-email-id'
  EMAIL_HOST_PASSWORD = 'your-gmail-application-password'
  EMAIL_PORT = 587
  SITE_HOST = '127.0.0.1:8000'
```

如果您遇到错误，例如：

```py
 (534, '5.7.9 Application-specific password required. Learn more at\n5.7.9 http://support.google.com/accounts/bin/answer.py?answer=185833 zr2sm8629305pbb.83 - gsmtp')

```

这意味着`EMAIL_HOST_PASSWORD`参数需要一个应用程序授权密码，而不是您的电子邮件密码。请按照主机部分中提到的链接获取有关如何创建的更多详细信息。

设置好这些东西后，尝试使用以下命令从 shell 再次发送邮件：

```py
>>> from django.core.mail import EmailMessage
>>> email = EmailMessage('Hello', 'World', to=['your_email@example.com'])
>>> email.send()

```

在这里，`your_email@example.com`参数是您想发送邮件的任何电子邮件地址。邮件的发件人地址将是我们传递给以下变量的 Gmail 电子邮件地址：

```py
 EMAIL_HOST_USER = 'your-gmail-email-id'

```

现在，一旦设置正确，使用 Django 发送邮件就像小菜一碟！我们将使用`EmailMessage`函数发送邀请邮件，但首先，让我们创建一个数据模型来存储邀请。

## 邀请数据模型

邀请包括以下信息：

+   收件人姓名

+   收件人邮箱

+   发件人的用户对象

我们还需要为邀请存储一个激活码。该代码将在邀请邮件中发送。该代码将有两个目的：

+   在接受邀请之前，我们可以使用该代码验证邀请是否实际存在于数据库中

+   接受邀请后，我们可以使用该代码从数据库中检索邀请信息，并跟踪发件人和收件人之间的关系

考虑到上述信息，让我们创建邀请数据模型。打开`user_profile/models.py`文件，并将以下代码追加到其中：

```py
  class Invitation(models.Model):
    name = models.CharField(maxlength=50)
    email = models.EmailField()
    code = models.CharField(maxlength=20)
    sender = models.ForeignKey(User)
    def __unicode__(self):
        return u'%s, %s' % (self.sender.username, self.email)
```

在这个模型中没有什么新的或难以理解的。我们只是为收件人姓名、收件人电子邮件、激活码和邀请发件人定义了字段。我们还为调试创建了一个`__unicode__`方法，并在管理界面中启用了该模型。不要忘记运行`python manage.py syncdb`命令来在数据库中创建新模型的表。

我们还将为此创建邀请表单。在`user_profile`目录中创建一个名为`forms.py`的文件，并使用以下代码进行更新：

```py
from django import forms

class InvitationForm(forms.Form):
  email = forms.CharField(widget=forms.TextInput(attrs={'size': 32, 'placeholder': 'Email Address of Friend to invite.', 'class':'form-control search-query'}))
```

创建发送邀请的视图页面类似于创建我们为搜索和推文表单创建的其他页面，通过创建一个名为`template/invite.html`的新文件：

```py
  {% extends "base.html" %}
  {% load staticfiles %}
  {% block content %}
  <div class="row clearfix">
    <div class="col-md-6 col-md-offset-3 column">
      {% if success == "1" %}
        <div class="alert alert-success" role="alert">Invitation Email was successfully sent to {{ email }}</div>
      {% endif %}
      {% if success == "0" %}
        <div class="alert alert-danger" role="alert">Failed to send Invitation Email to {{ email }}</div>
      {% endif %}
      <form id="search-form" action="" method="post">{% csrf_token %}
        <div class="input-group input-group-sm">
        {{ invite.email.errors }}
        {{ invite.email }}
          <span class="input-group-btn">
            <button class="btn btn-search" type="submit">Invite</button>
          </span>
        </div>
      </form>
    </div>
  </div>
  {% endblock %}
```

此方法的 URL 输入如下：

```py
  url(r'^invite/$', Invite.as_view()),
```

现在，我们需要创建`get`和`post`方法来使用此表单发送邀请邮件。

由于发送邮件比推文更具体于用户，我们将在`user_profile`视图中创建此方法，而不是之前使用的推文视图。

使用以下代码更新`user_profile/views.py`文件：

```py
from django.views.generic import View
from django.conf import settings
from django.shortcuts import render
from django.template import Context
from django.template.loader import render_to_string
from user_profile.forms import InvitationForm
from django.core.mail import EmailMultiAlternatives
from user_profile.models import Invitation, User
from django.http import HttpResponseRedirect
import hashlib

class Invite(View):
  def get(self, request):
    params = dict()
    success = request.GET.get('success')
    email = request.GET.get('email')
    invite = InvitationForm()
    params["invite"] = invite
    params["success"] = success
    params["email"] = email
    return render(request, 'invite.html', params)

  def post(self, request):
    form = InvitationForm(self.request.POST)
    if form.is_valid():
      email = form.cleaned_data['email']
      subject = 'Invitation to join MyTweet App'
      sender_name = request.user.username
      sender_email = request.user.email
      invite_code = Invite.generate_invite_code(email)
      link = 'http://%s/invite/accept/%s/' % (settings.SITE_HOST, invite_code)
      context = Context({"sender_name": sender_name, "sender_email": sender_email, "email": email, "link": link})
      invite_email_template = render_to_string('partials/_invite_email_template.html', context)
      msg = EmailMultiAlternatives(subject, invite_email_template, settings.EMAIL_HOST_USER, [email], cc=[settings.EMAIL_HOST_USER])
      user = User.objects.get(username=request.user.username)
      invitation = Invitation()
      invitation.email = email
      invitation.code = invite_code
      invitation.sender = user
      invitation.save()
      success = msg.send()
      return HttpResponseRedirect('/invite?success='+str(success)+'&email='+email)

  @staticmethod
  def generate_invite_code(email):
    secret = settings.SECRET_KEY
    if isinstance(email, unicode):
      email = email.encode('utf-8')
      activation_key = hashlib.sha1(secret+email).hexdigest()
      return activation_key
```

在这里，`get()`方法就像使用`invite.html`文件渲染邀请表单一样简单，并且初始未设置`success`和`email`变量。

`post()`方法使用通常的表单检查和变量提取概念；您将首次看到的代码如下：

```py
  invite_code = Invite.generate_invite_code(email)
```

这实际上是一个静态函数调用，为每个受邀用户生成具有唯一密钥的激活令牌。当您加载名为`_invite_email_template.html`的模板并将以下变量传递给它时，`render_to_string()`方法将起作用：

+   `sender_name`：这是邀请或发件人的姓名

+   `sender_email`：这是发件人的电子邮件地址

+   `email`：这是被邀请人的电子邮件地址

+   `link`：这是邀请接受链接

然后使用该模板来渲染邀请邮件的正文。之后，我们使用`EmailMultiAlternatives()`方法发送邮件，就像我们在上一节的交互式会话中所做的那样。

这里有几点需要注意：

+   激活链接的格式为`http://SITE_HOST/invite/accept/CODE/`。我们将在本节后面编写一个视图来处理此类 URL。

+   这是我们第一次使用模板来渲染除网页以外的其他内容。正如您所见，模板系统非常灵活，允许我们构建电子邮件，以及网页或任何其他文本。

+   我们使用`render_to_string()`和`render()`方法构建消息正文，而不是通常的`render_to_response`调用。如果你还记得，这就是我们在本书早期渲染模板的方式。我们这样做是因为我们不是在渲染网页。

由于`send`方法加载名为`_invite_email_template.html`的模板，请在模板文件夹中创建一个同名文件并插入以下内容：

```py
  Hi,
    {{ sender_name }}({{ sender_email }}) has invited you to join Mytweet.
    Please click {{ link }} to join.
This email was sent to {{ email }}. If you think this is a mistake Please ignore.
```

我们已经完成了“邀请朋友”功能的一半实现。目前，点击激活链接会产生 404 页面未找到错误，因此，接下来，我们将编写一个视图来处理它。

## 处理激活链接

我们取得了良好的进展；用户现在能够通过电子邮件邀请他们的朋友。下一步是构建一个处理邀请中激活链接的机制。以下是我们将要做的概述。

我们将构建一个视图来处理激活链接。此视图验证邀请码实际上是否存在于数据库中，并且注册的用户自动关注发送链接的用户并被重定向到注册页面。

让我们从为视图编写 URL 条目开始。打开`urls.py`文件并添加以下突出显示的行：

```py
 url(r'^invite/accept/(\w+)/$', InviteAccept.as_view()),

```

在`user_profile/view.py`文件中创建一个名为`InviteAccept()`的类。

从逻辑上讲，邀请接受将起作用，因为用户将被要求注册应用程序，如果他们已经注册，他们将被要求关注邀请他们的用户。

为了简单起见，我们将用户重定向到带有激活码的注册页面，这样当他们注册时，他们将自动成为关注者。让我们看一下以下代码：

```py
class InviteAccept(View):
  def get(self, request, code):
    return HttpResponseRedirect('/register?code='+code)
```

然后，我们将用以下代码编写注册页面：

```py
class Register(View):
  def get(self, request):
    params = dict()
    registration_form = RegisterForm()
    code = request.GET.get('code')
    params['code'] = code
    params['register'] = registration_form
    return render(request, 'registration/register.html', params)

  def post(self, request):
    form = RegisterForm(request.POST)
    if form.is_valid():
      username = form.cleaned_data['username']
      email = form.cleaned_data['email']
      password = form.cleaned_data['password']
      try:
        user = User.objects.get(username=username)                
      except:
        user = User()
        user.username = username
        user.email = email
        commit = True
        user = super(user, self).save(commit=False)
        user.set_password(password)
        if commit:
          user.save()
        return HttpResponseRedirect('/login')
```

如你所见，视图遵循邀请电子邮件中发送的 URL 格式。激活码是使用正则表达式从 URL 中捕获的，然后作为参数传递给视图。

这有点耗时，但我们能够充分利用我们的 Django 知识来实现它。您现在可以点击通过电子邮件收到的邀请链接，看看会发生什么。您将被重定向到注册页面；您可以在那里创建一个新账户，登录，并注意新账户和您的原始账户如何成为发送者的关注者。

# 国际化（i18n）-提供多种语言的网站

如果人们无法阅读我们应用的页面，他们就不会使用我们的应用。到目前为止，我们只关注说英语的用户。然而，全世界有许多人不懂英语或更喜欢使用他们的母语。为了吸引这些人，将我们应用的界面提供多种语言是个好主意。这将克服语言障碍，并为我们的应用打开新的前沿，特别是在英语不常用的地区。

正如你可能已经猜到的那样，Django 提供了将项目翻译成多种语言所需的所有组件。负责提供此功能的系统称为**国际化系统**（**i18n**）。翻译 Django 项目的过程非常简单。

按照以下三个步骤进行：

1.  指定应用程序中应翻译的字符串，例如，状态和错误消息是可翻译的，而用户名则不是。

1.  为要支持的每种语言创建一个翻译文件。

1.  启用和配置 i18n 系统。

我们将在以下各小节中详细介绍每个步骤。在本章节的最后，我们的应用将支持多种语言，您将能够轻松翻译任何其他 Django 项目。

## 将字符串标记为可翻译的

翻译应用程序的第一步是告诉 Django 哪些字符串应该被翻译。一般来说，视图和模板中的字符串需要被翻译，而用户输入的字符串则不需要。将字符串标记为可翻译是通过函数调用完成的。函数的名称以及调用方式取决于字符串的位置：在视图、模板、模型或表单中。

这一步比起一开始看起来要容易得多。让我们通过一个例子来了解它。我们将翻译应用程序中的“邀请关注者”功能。翻译应用程序的其余部分的过程将完全相同。打开`user_profile/views.py`文件，并对邀请视图进行突出显示的更改：

```py
from django.utils.translation import ugettext as _
from django.views.generic import View
from django.conf import settings
from django.shortcuts import render
from django.template import Context
from django.template.loader import render_to_string
from user_profile.forms import InvitationForm
from django.core.mail import EmailMultiAlternatives
from user_profile.models import Invitation, User
from django.http import HttpResponseRedirect
import hashlib

class Invite(View):
  def get(self, request):
    params = dict()
    success = request.GET.get('success')
    email = request.GET.get('email')
    invite = InvitationForm()
    params["invite"] = invite
    params["success"] = success
    params["email"] = email
    return render(request, 'invite.html', params)

  def post(self, request):
    form = InvitationForm(self.request.POST)
    if form.is_valid():
      email = form.cleaned_data['email']
      subject = _('Invitation to join MyTweet App')
      sender_name = request.user.username
      sender_email = request.user.email
      invite_code = Invite.generate_invite_code(email)
      link = 'http://%s/invite/accept/%s/' % (settings.SITE_HOST, invite_code)
      context = Context({"sender_name": sender_name, "sender_email": sender_email, "email": email, "link": link})
      invite_email_template = render_to_string('partials/_invite_email_template.html', context)
      msg = EmailMultiAlternatives(subject, invite_email_template, settings.EMAIL_HOST_USER, [email], cc=[settings.EMAIL_HOST_USER])
      user = User.objects.get(username=request.user.username)
      invitation = Invitation()
      invitation.email = email
      invitation.code = invite_code
      invitation.sender = user
      invitation.save()
      success = msg.send()
    return HttpResponseRedirect('/invite?success='+str(success)+'&email='+email)

  @staticmethod
  def generate_invite_code(email):
    secret = settings.SECRET_KEY
    if isinstance(email, unicode):
      email = email.encode('utf-8')
      activation_key = hashlib.sha1(secret+email).hexdigest()
    return activation_key
```

请注意，主题字符串以“`_`”开头；或者，您也可以这样写：

```py
from django.utils.translation import ugettext
  subject = ugettext('Invitation to join MyTweet App')
```

无论哪种方式，它都运行良好。

正如您所看到的，更改是微不足道的：

+   我们从`django.utils.translation`中导入了一个名为`ugettext`的函数。

+   我们使用了`as`关键字为函数（下划线字符）分配了一个更短的名称。我们这样做是因为这个函数将用于在视图中标记字符串为可翻译的，而且由于这是一个非常常见的任务，给函数一个更短的名称是个好主意。

+   我们只需将一个字符串传递给`_`函数即可将其标记为可翻译。

这很简单，不是吗？然而，这里有一个小观察需要做。第一条消息使用了字符串格式化，并且在调用`_()`函数后应用了`%`运算符。这是为了避免翻译电子邮件地址。最好使用命名格式，这样在实际翻译时可以更好地控制。因此，您可能想要定义以下代码：

```py
message= \
_('An invitation was sent to %(email)s.') % {
'email': invitation.email}
```

既然我们知道如何在视图中标记字符串为可翻译的，让我们转到模板。在模板文件夹中打开`invite.html`文件，并修改如下：

```py
{% extends "base.html" %}
{% load staticfiles %}
{% load i18n %}
{% block content %}
<div class="row clearfix">
  <div class="col-md-6 col-md-offset-3 column">
    {% if success == "1" %}
    <div class="alert alert-success" role="alert">
      {% trans Invitation Email was successfully sent to  %}{{ email }}
    </div>
    {% endif %}
    {% if success == "0" %}
    <div class="alert alert-danger" role="alert">Failed to send Invitation Email to {{ email }}</div>
    {% endif %}
      <form id="search-form" action="" method="post">{% csrf_token %}
        <div class="input-group input-group-sm">
        {{ invite.email.errors }}
        {{ invite.email }}
          <span class="input-group-btn">
            <button class="btn btn-search" type="submit">Invite</button>
          </span>
        </div>
      </form>
    </div>
  </div>
  {% endblock %}
```

在这里，我们在模板的开头放置了`{% load i18n %}`参数，以便让它可以访问翻译标签。`<load>`标签通常用于启用默认情况下不可用的额外模板标签。您需要在使用翻译标签的每个模板的顶部放置它。i18n 是国际化的缩写，这是 Django 框架的名称，它提供了翻译功能。

接下来，我们使用了一个名为`trans`的模板标签来标记字符串为可翻译的。这个模板标签与视图中的`gettext`函数完全相同。值得注意的是，如果字符串包含模板变量，`trans`标签将不起作用。在这种情况下，您需要使用`blocktrans`标签，如下所示：

```py
{% blocktrans %} 
```

您可以在`{% endblocktrans %}`块中传递一个变量块，即`{{ variable }}`，以使其对读者更有意义。

现在您知道如何在模板中处理可翻译的字符串了。那么，让我们转到表单和模型。在表单或模型中标记字符串为可翻译与在视图中略有不同。要了解如何完成这一点，请打开`user_profile/forms.py`文件，并修改邀请表单如下：

```py
from django.utils.translation import gettext_lazy as _
class InvitationForm(forms.Form):
  email = forms.CharField(widget=forms.TextInput(attrs={'size': 32, 'placeholder': _('Email Address of Friend to invite.'), 'class':'form-control'}))
```

唯一的区别是我们导入了`gettext_lazy`函数而不是`gettext`。`gettext_lazy`会延迟直到访问其返回值时才翻译字符串。这在这里是必要的，因为表单的属性只在应用程序启动时创建一次。如果我们使用普通的`gettext`函数，翻译后的标签将以默认语言（通常是英语）存储在表单属性中，并且永远不会再次翻译。但是，如果我们使用`gettext_lazy`函数，该函数将返回一个特殊对象，每次访问时都会翻译字符串，因此翻译将正确进行。这使得`gettext_lazy`函数非常适合表单和模型属性。

有了这个，我们完成了为“邀请朋友”视图标记字符串以进行翻译。为了帮助您记住本小节涵盖的内容，这里是标记可翻译字符串所使用的技术的快速总结：

+   在视图中，使用`gettext`函数标记可翻译的字符串（通常导入为`_`）

+   在模板中，使用`trans`模板标记标记不包含变量的可翻译字符串，使用`blocktrans`标记标记包含变量的字符串。

+   在表单和模型中，使用`gettext_lazy`函数标记可翻译的字符串（通常导入为`_`）

当然，也有一些特殊情况可能需要单独处理。例如，您可能希望使用`gettext_lazy`函数而不是`gettext`函数来翻译视图中的默认参数值。只要您理解这两个函数之间的区别，您就应该能够决定何时需要这样做。

## 创建翻译文件

现在我们已经完成了标记要翻译的字符串，下一步是为我们想要支持的每种语言创建一个翻译文件。这个文件包含所有可翻译的字符串及其翻译，并使用 Django 提供的实用程序创建。

让我们创建一个翻译文件。首先，您需要在 Django 安装文件夹内的`bin`目录中找到一个名为`make-messages.py`的文件。找到它的最简单方法是使用操作系统中的搜索功能。找到它后，将其复制到系统路径（在 Linux 和 Mac OS X 中为`/usr/bin/`，在 Windows 中为`c:\windows\`）。

此外，确保在 Linux 和 Mac OS X 中运行以下命令使其可执行（对 Windows 用户来说，这一步是不需要的）：

```py
$ sudo chmod +x /usr/bin/make-messages.py

```

`make-messages.py`实用程序使用一个名为 GNU gettext 的软件包从源代码中提取可翻译的字符串。因此，您需要安装这个软件包。对于 Linux，搜索您的软件包管理器中的软件包并安装它。Windows 用户可以在[`gnuwin32.sourceforge.net/packages/gettext.htm`](http://gnuwin32.sourceforge.net/packages/gettext.htm)找到该软件包的安装程序。

最后，Mac OS X 用户将在[`gettext.darwinports.com/`](http://gettext.darwinports.com/)找到适用于其操作系统的软件包版本以及安装说明。

安装 GNU gettext 软件包后，打开终端，转到您的项目文件夹，在那里创建一个名为`locale`的文件夹，然后运行以下命令：

```py
$ make-messages.py -l de

```

这个命令为德语语言创建了一个翻译文件。`de`变量是德语的语言代码。如果您想要翻译其他语言，将其语言代码放在`de`的位置，并继续为本章的其余部分执行相同的操作。除此之外，如果您想要支持多种语言，为每种语言运行上一个命令，并将说明应用到本节的所有语言。 

一旦您运行了上述命令，它将在`locale/de/LC_MESSAGES/`下创建一个名为`django.po`的文件。这是德语语言的翻译文件。在文本编辑器中打开它，看看它是什么样子的。文件以一些元数据开头，比如创建日期和字符集。之后，您会发现每个可翻译字符串的条目。每个条目包括字符串的文件名和行号，字符串本身，以及下面的空字符串，用于放置翻译。以下是文件中的一个示例条目：

```py
#: user_profile/forms.py
msgid "Friend's Name"
msgstr ""
```

要翻译字符串，只需使用文本编辑器在第三行的空字符串中输入翻译。您也可以使用专门的翻译编辑器，比如`Poedit`（在[`www.poedit.net/`](http://www.poedit.net/)上提供所有主要操作系统的版本），但对于我们的简单文件，普通文本编辑器就足够了。确保在文件的元数据部分设置一个有效的字符。我建议您使用**UTF-8**：

```py
"Content-Type: text/plain; charset=UTF-8\n"
```

您可能会注意到翻译文件包含一些来自管理界面的字符串。这是因为`admin/base_site.html`管理模板使用`trans`模板标记将其字符串标记为可翻译的。无需翻译这些字符串；Django 已经为它们提供了翻译文件。

翻译完成后，您需要将翻译文件编译为 Django 可以使用的格式。这是使用 Django 提供的另一个实用程序`compile-messages.py`命令完成的。找到并将此文件移动到系统路径，并确保它是可执行的，方法与我们使用`make-messages.py`命令相同。

接下来，在项目文件夹中运行以下命令：

```py
$ compile-messages.py

```

如果实用程序报告文件中的错误（例如缺少引号），请更正错误并重试。一旦成功，实用程序将在同一文件夹中创建一个名为`django.mo`的已编译翻译文件，并为本节的下一步做好一切准备。

## 启用和配置 i18n 系统

Django 默认启用了 i18n 系统。您可以通过在`settings.py`文件中搜索以下行来验证这一点：

```py
USE_I18N = True
```

有两种配置 i18n 系统的方法。您可以为所有用户全局设置语言，也可以让用户单独指定其首选语言。我们将在本小节中看到如何同时进行这两种配置。

要全局设置活动语言，请在`settings.py`文件中找到名为`LANGUAGE_CODE`的变量，并将您喜欢的语言代码分配给它。例如，如果您想将德语设置为项目的默认语言，请将语言代码更改如下：

```py
LANGUAGE_CODE = 'de'
```

现在，如果开发服务器尚未运行，请启动它，并转到“邀请朋友”页面。在那里，您会发现字符串已根据您在德语翻译文件中输入的内容进行了更改。现在，将`LANGUAGE_CODE`变量的值更改为'`en`'，并注意页面如何恢复为英语。

第二种配置方法是让用户选择语言。为此，我们应该启用一个名为`LocaleMiddleware`的类。简而言之，中间件是处理请求或响应对象的类。Django 的许多组件都使用中间件类来实现功能。要查看这一点，请打开`settings.py`文件并搜索`MIDDLEWARE_CLASSES`变量。您会在那里找到一个字符串列表，其中一个是`django.contrib.sessions.middleware.SessionMiddleware`，它将会话数据附加到请求对象上。在使用中间件之前，我们不需要了解中间件类是如何实现的。要启用`LocaleMiddleware`，只需将其类路径添加到`MIDDLEWARE_CLASSES`列表中。确保将`LocaleMiddleware`放在`SessionMiddleware`之后，因为区域设置中间件利用会话 API，我们将在下面看到。打开`settings.py`文件并按照以下代码片段中的突出显示的内容修改文件：

```py
MIDDLEWARE_CLASSES = (
'django.middleware.common.CommonMiddleware',
'django.contrib.sessions.middleware.SessionMiddleware',
'django.contrib.auth.middleware.AuthenticationMiddleware',
'django.middleware.doc.XViewMiddleware',
'django.middleware.locale.LocaleMiddleware',
)

```

区域设置中间件通过以下步骤确定用户的活动语言：

1.  它在会话数据中查找名为`django_language`的键。

1.  如果键不存在，则查找名为`django_language`的 cookie。

1.  如果 cookie 不存在，则查看 Accept-Language HTTP 标头中的语言代码。此标头由浏览器发送到 Web 服务器，指示您希望以哪种语言接收内容。

1.  如果一切都失败了，将使用`settings.py`文件中的`LANGUAGE_CODE`变量。

在所有前面的步骤中，Django 会寻找与可用翻译文件匹配的语言代码。为了有效地利用区域设置中间件，我们需要一个视图，使用户能够选择语言并相应地更新会话数据。幸运的是，Django 已经为我们提供了这样的视图。该视图称为**setlanguage**，并且它期望在名为 language 的 GET 变量中包含语言代码。它使用此变量更新会话数据，并将用户重定向到原始页面。要启用此视图，请编辑`urls.py`文件，并向其中添加以下突出显示的行：

```py
urlpatterns = patterns('',
# i18n
(r'^i18n/', include('django.conf.urls.i18n')),
)
```

添加上述行类似于我们为管理界面添加 URL 条目的方式。如果您还记得之前的章节，`include()`函数可以用于在特定路径下包含来自另一个应用程序的 URL 条目。现在，我们可以通过提供链接（例如`/i18n/setlang/language=de`）让用户将语言更改为德语。我们将修改基本模板以在所有页面上添加此类链接。打开`templates/base.html`文件，并向其中添加以下突出显示的行：

```py
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html>
  <head>
    [...]
  </head>
  <body>
    [...]
    <div id="footer">
    Django Mytweets <br />
    Languages:
      <a href="/i18n/setlang/?language=en">en</a>
      <a href="/i18n/setlang/?language=de">de</a>
      [ 218 ]Chapter 11
    </div>
  </body>
</html>
```

此外，我们将通过将以下 CSS 代码附加到`site_media/style.css`文件来为新的页脚设置样式：

```py
#footer {
margin-top: 2em;
text-align: center;
}
```

现在，我们的应用程序的 i18n 功能已经准备就绪。将浏览器指向“邀请朋友”页面，并尝试页面底部的新语言链接。语言应该根据点击的链接而改变。

在我们结束本节之前，这里有一些观察结果：

+   您可以在视图中使用请求`LANGUAGE_CODE`属性访问当前活动的语言。

+   Django 本身被翻译成多种语言。您可以通过在激活英语以外的语言时触发表单错误来查看这一点。错误消息将以所选语言显示，即使您自己没有进行翻译。

+   在模板中，当使用`RequestContext`变量时，可以使用`LANGUAGE_CODE`模板变量访问当前活动的语言。

这一部分有点长，但您从中学到了一个非常重要的功能。通过以多种语言提供我们的应用程序，我们使其能够吸引更广泛的受众，从而具有吸引更多用户的潜力。这实际上适用于任何 Web 应用程序，现在，我们将能够轻松地将任何 Django 项目翻译成多种语言。

在下一节中，我们将转移到另一个主题。当您的应用程序用户基数增长时，服务器的负载将增加，您将开始寻找改进应用程序性能的方法。这就是缓存发挥作用的地方。

因此，请继续阅读以了解这个非常有用的技术！

# 缓存-在高流量期间提高站点性能

Web 应用程序的页面是动态生成的。每次请求页面时，都会执行代码来处理用户输入并生成输出。生成动态页面涉及许多开销，特别是与提供静态 HTML 文件相比。代码可能会连接到数据库，执行昂贵的计算，处理文件等等。同时，能够使用代码生成页面正是使网站动态和交互的原因。

如果我们能同时获得两全其美岂不是太好了？这就是缓存所做的，这是大多数中高流量网站上实现的功能。当请求页面时，缓存会存储页面的生成 HTML，并在以后再次请求相同页面时重用它。这样可以通过避免一遍又一遍地生成相同页面来减少很多开销。当然，缓存页面并不是永久存储的。当页面被缓存时，会为缓存设置一个过期时间。当缓存页面过期时，它会被删除，页面会被重新生成并缓存。过期时间通常在几秒到几分钟之间，取决于网站的流量。过期时间确保缓存定期更新，并且用户接收内容更新的同时，减少生成页面的开销。

尽管缓存对于中高流量网站特别有用，低流量网站也可以从中受益。如果网站突然接收到大量高流量，可能是因为它被主要新闻网站报道，您可以启用缓存以减少服务器负载，并帮助您的网站度过高流量的冲击。稍后，当流量平息时，您可以关闭缓存。因此，缓存对小型网站也很有用。您永远不知道何时会需要它，所以最好提前准备好这些信息。

## 启用缓存

我们将从启用缓存系统开始这一部分。要使用缓存，您首先需要选择一个缓存后端，并在 `settings.py` 文件中的一个名为 `CACHE_BACKEND` 的变量中指定您的选择。此变量的内容取决于您选择的缓存后端。一些可用的选项包括：

+   **简单缓存**：对于这种情况，缓存数据存储在进程内存中。这只对开发过程中测试缓存系统有用，不应在生产中使用。要启用它，请在 `settings.py` 文件中添加以下内容：

```py
CACHE_BACKEND = 'simple:///'
```

+   **数据库缓存**：对于这种情况，缓存数据存储在数据库表中。要创建缓存表，请运行以下命令：

```py
$ python manage.py createcachetable cache_table

```

然后，在 `settings.py` 文件中添加以下内容：

```py
CACHE_BACKEND = 'db://cache_table'
```

在这里，缓存表被称为 `cache_table`。只要不与现有表冲突，您可以随意命名它。

+   **文件系统缓存**：在这里，缓存数据存储在本地文件系统中。要使用它，请在 `settings.py` 文件中添加以下内容：

```py
CACHE_BACKEND = 'file:///tmp/django_cache'
```

在这里，`/tmp/django_cache` 变量用于存储缓存文件。如果需要，您可以指定另一个路径。

+   **Memcached**：Memcached 是一个先进、高效和快速的缓存框架。安装和配置它超出了本书的范围，但如果您已经有一个可用的 Memcached 服务器，可以在 `settings.py` 文件中指定其 IP 和端口，如下所示：

```py
CACHE_BACKEND = 'memcached://ip:port/'
```

如果您不确定在本节中选择哪个后端，请选择简单缓存。然而，实际上，如果您突然遇到高流量并希望提高服务器性能，可以选择 Memcached 或数据库缓存，具体取决于服务器上可用的选项。另一方面，如果您有一个中高流量的网站，我强烈建议您使用 Memcached，因为它绝对是 Django 可用的最快的缓存解决方案。本节中提供的信息无论您选择哪种缓存后端都是一样的。

因此，决定一个缓存后端，并在 `settings.py` 文件中插入相应的 `CACHE_BACKEND` 变量。接下来，您应该指定缓存页面的过期持续时间（以秒为单位）。在 `settings.py` 文件中添加以下内容，以便将页面缓存五分钟：

```py
CACHE_MIDDLEWARE_SECONDS = 60 * 5
```

现在，我们已经完成了启用缓存系统。继续阅读，了解如何利用缓存来提高应用程序的性能。

## 配置缓存

您可以配置 Django 缓存整个站点或特定视图。我们将在本小节中学习如何做到这两点。

### 缓存整个站点

要缓存整个网站，请将`CacheMiddleware`类添加到`settings.py`文件中的`MIDDLEWARE_CLASSES`类中：

```py
MIDDLEWARE_CLASSES = (
'django.middleware.common.CommonMiddleware',
'django.contrib.sessions.middleware.SessionMiddleware',
'django.contrib.auth.middleware.AuthenticationMiddleware',
'django.middleware.cache.CacheMiddleware',
'django.middleware.doc.XViewMiddleware',
'django.middleware.locale.LocaleMiddleware',
)

```

在这里顺序很重要，就像我们添加区域设置中间件时一样。缓存中间件类应该在会话和身份验证中间件类之后添加，在区域设置中间件类之前添加。

这就是您需要缓存 Django 网站的全部内容。从现在开始，每当请求页面时，Django 都会存储生成的 HTML 并在以后重复使用。重要的是要意识到，缓存系统只缓存没有`GET`和`POST`变量的页面。因此，我们的用户仍然可以发布推文和关注朋友，因为这些页面的视图期望 GET 或 POST 变量。另一方面，推文和标签列表等页面将被缓存。

### 缓存特定视图

有时，您可能只想缓存网站的特定页面-可能是一个与您的页面链接的高流量网站，因此大部分流量将被引导到这个特定页面。在这种情况下，只缓存此页面是有意义的。另一个适合缓存的好候选者是生成成本高昂的页面，因此您只希望每五分钟生成一次。我们应用程序中的标签云页面符合后一种情况。每次请求页面时，Django 都会遍历数据库中的所有标签，并计算每个标签的推文数量。这是一个昂贵的操作，因为它需要大量的数据库查询。因此，缓存这个视图是一个好主意。

要根据标签类缓存视图，只需应用一个名为`cache_page`的方法和与之相关的缓存参数。通过编辑`mytweets/urls.py`文件中的以下代码来尝试这一点：

```py
from django.views.decorators.cache import cache_page
...
...
url(r'^search/hashTag$',  cache_page(60 * 15)(SearchHashTag.as_view())),
...
...

```

使用`cache_page()`方法很简单。它允许您指定要缓存的视图。站点缓存中提到的规则也适用于视图缓存。如果视图接收 GET 或 POST 参数，Django 将不会对其进行缓存。

有了这些信息，我们完成了本节。当您首次将网站发布到公众时，缓存是不必要的。然而，当您的网站增长，或者突然接收到大量高流量时，缓存系统肯定会派上用场。因此，在监视应用程序性能时要牢记这一点。

接下来，我们将学习 Django 测试框架。测试有时可能是一项乏味的任务。如果您可以运行一个命令来处理测试您的网站，那不是很好吗？Django 允许您这样做，我们将在下一节中学习。

模板片段可以以以下方式进行缓存：

```py
 % load cache %}
 {% cache 500 sidebar %}
 .. sidebar ..
 {% endcache %}

```

# 单元测试-自动化测试应用程序的过程

在本书的过程中，我们有时修改了先前编写的视图。这在软件开发过程中经常发生。一个人可能会修改甚至重写一个函数来改变实现细节，因为需求已经改变，或者只是为了重构代码，使其更易读。

当您修改一个函数时，您必须再次测试它，以确保您的更改没有引入错误。然而，如果您不断重复相同的测试，测试将变得乏味。如果函数的各个方面没有很好地记录，您可能会忘记测试所有方面。显然，这不是一个理想的情况；我们绝对需要一个更好的机制来处理测试。

幸运的是，已经有了一个解决方案。它被称为单元测试。其思想是编写代码来测试您的代码。测试代码调用您的函数并验证它们的行为是否符合预期，然后打印出结果报告。您只需要编写一次测试代码。以后，每当您想要测试时，只需运行测试代码并检查生成的报告即可。

Python 自带了一个用于单元测试的框架。它位于单元测试模块中。Django 扩展了这个框架，以添加对视图测试的支持。我们将在本节中学习如何使用 Django 单元测试框架。

## 测试客户端

为了与视图交互，Django 提供了一个模拟浏览器功能的类。您可以使用它向应用程序发送请求并接收响应。让我们使用交互式控制台来学习。使用以下命令启动控制台：

```py
$ python manage.py shell

```

导入`Client()`类，创建一个`Client`对象，并使用 GET 请求检索应用程序的主页：

```py
>>>from django.test.client import Client
client = Client()
>>> response = client.get('/')
>>> print response

X-Frame-Options: SAMEORIGIN
Content-Type: text/html; charset=utf-8

<html>
 <head>
 <link href="/static/css/bootstrap.min.css"
 rel="stylesheet" media="screen">
 </head>
 <body>
 <nav class="navbar navbar-default" role="navigation">
 <a class="navbar-brand" href="#">MyTweets</a>
 </nav>
 <div class="container">
 </div>
 <nav class="navbar navbar-default navbar-fixed-bottom" role="navigation">
 <p class="navbar-text navbar-right">Footer </p>
 </nav>
 <script src="img/jquery-2.1.1.min.js"></script>
 <script src="img/bootstrap.min.js"></script>
 <script src="img/base.js"></script>
 </body>
</html>
>>> 

```

尝试向登录视图发送 POST 请求。输出将根据您是否提供正确的凭据而有所不同：

```py
>>> print client.post('/login/',{'username': 'your_username', 'password': 'your_password'})

```

最后，如果有一个只允许已登录用户访问的视图，您可以像这样发送一个请求：

```py
>>> print client.login('/friend/invite/', 'your_username', 'your_password')

```

如您从交互式会话中看到的，`Client()`类提供了三种方法：

+   `get`：这个方法向视图发送一个 GET 请求。它将视图的 URL 作为参数。您可以向该方法传递一个可选的 GET 变量字典。

+   `post`：这个方法向视图发送一个 POST 请求。它将视图的 URL 和一个 POST 变量字典作为参数。

+   `login`：这个方法向一个只允许已登录用户访问的视图发送一个 GET 请求。它将视图的 URL、用户名和密码作为参数。

`Client()`类是有状态的，这意味着它在请求之间保留其状态。一旦您登录，后续的请求将在您登录的状态下处理。`Client()`类的方法返回的响应对象包含以下属性：

+   `status_code`：这是响应的 HTTP 状态

+   `content`：这是响应页面的主体

+   `template`：这是用于渲染页面的`Template`实例；如果使用了多个模板，这个属性将是一个`Template`对象的列表

+   `context`：这是用于渲染模板的`Context`对象

这些字段对于检查测试是否成功或失败非常有用，接下来我们将看到。请随意尝试更多`Client()`类的用法。在继续下一小节之前，了解它的工作原理是很重要的，我们将在下一小节中创建第一个单元测试。

## 测试注册视图

现在您对`Client()`类感到满意了，让我们编写我们的第一个测试。单元测试应该位于应用程序文件夹内名为`tests.py`的模块中。每个测试应该是从`django.test.TestCase`模块派生的类中的一个方法。方法的名称必须以单词 test 开头。有了这个想法，我们将编写一个测试方法，试图注册一个新的用户帐户。因此，在`bookmarks`文件夹内创建一个名为`tests.py`的文件，并在其中输入以下内容：

```py
from django.test import TestCase
from django.test.client import Client
class ViewTest(TestCase):
def setUp(self):
self.client = Client()
def test_register_page(self):
data = {
'username': 'test_user',
'email': 'test_user@example.com',
'password1': 'pass123',
'password2': 'pass123'
}
response = self.client.post('/register/', data)
self.assertEqual(response.status_code, 302)

```

让我们逐行查看代码：

+   首先，我们导入了`TestCase`和`Client`类。

+   接下来，我们定义了一个名为`ViewTest()`的类，它是从`TestCase`类派生的。正如我之前所说，所有测试类都必须从这个基类派生。

+   之后，我们定义了一个名为`setUp()`的方法。当测试过程开始时，将调用这个方法。在这里，我们创建了一个`Client`对象。

+   最后，我们定义了一个名为`test_register_page`的方法。方法的名称以单词 test 开头，表示它是一个测试方法。该方法向注册视图发送一个 POST 请求，并检查状态码是否等于数字`302`。这个数字是重定向的 HTTP 状态。

如果您回忆一下前面的章节，注册视图在请求成功时会重定向用户。

我们使用一个名为`assertEqual()`的方法来检查响应对象。这个方法是从`TestCase`类继承的。如果两个传递的参数不相等，它会引发一个异常。如果引发了异常，测试框架就知道测试失败了；否则，如果没有引发异常，它就认为测试成功了。

`TestCase`类提供了一组方法供测试使用。以下是一些重要的方法列表：

+   `assertEqual`：这期望两个值相等

+   `assertNotEquals`：这期望两个值不相等

+   `assertTrue`：这期望一个值为`True`

+   `assertFalse`：这期望一个值为`False`

现在您了解了测试类，让我们通过发出命令来运行实际测试：

```py
$ python manage.py test

```

输出将类似于以下内容：

```py
Creating test database...
Creating table auth_message
Creating table auth_group
Creating table auth_user
Creating table auth_permission
[...]
Loading 'initial_data' fixtures...
No fixtures found.
.
-------------------------------------------------------------
Ran 1 test in 0.170s
OK
Destroying test database...

```

那么，这里发生了什么？测试框架首先通过创建一个类似于真实数据库中的表的测试数据库来开始。接下来，它运行在测试模块中找到的测试。最后，它打印出结果的报告并销毁测试数据库。

在这里，我们的单个测试成功了。如果测试失败，输出会是什么样子，请修改`tests.py`文件中的`test_register_page`视图，删除一个必需的表单字段：

```py
def test_register_page(self):
data = {
'username': 'test_user',
'email': 'test_user@example.com',
'password1': '1',
# 'password2': '1'
}
response = self.client.post('/register/', data)
self.assertEqual(response.status_code, 302)
```

现在，再次运行`python manage.py test`命令以查看结果：

```py
=============================================================
FAIL: test_register_page (mytweets.user_profile.tests.ViewTest)
-------------------------------------------------------------
Traceback (most recent call last):
File "mytweets/user_profile/tests.py", line 19, in test_
register_page
self.assertEqual(response.status_code, 302)
AssertionError: 200 != 302
-------------------------------------------------------------
Ran 1 test in 0.170s
FAILED (failures=1)

```

我们的测试有效！Django 检测到错误并给了我们发生的确切细节。完成后不要忘记将测试恢复到原始形式。现在，让我们编写另一个测试，一个稍微更高级的测试，以更好地了解测试框架。

还有许多其他情景可以编写单元测试：

+   检查注册是否失败，如果两个密码字段不匹配

+   测试“添加朋友”和“邀请朋友”视图

+   测试“编辑书签”功能

+   测试搜索返回正确结果

上面的列表只是一些例子。编写单元测试以覆盖尽可能多的用例对于保持应用程序的健康和减少错误和回归非常重要。你编写的单元测试越多，当你的应用程序通过所有测试时，你就越有信心。Django 使单元测试变得非常容易，所以要充分利用这一点。

在应用程序的生命周期中的某个时刻，它将从开发模式转移到生产模式。下一节将解释如何为生产环境准备您的 Django 项目。

# 部署 Django

所以，你在你的 Web 应用程序上做了很多工作，现在是时候上线了。为了确保从开发到生产的过渡顺利进行，必须在应用程序上线之前进行一些更改。本节涵盖了这些更改，以帮助您成功上线您的 Web 应用程序。

## 生产 Web 服务器

在本书中，我们一直在使用 Django 自带的开发 Web 服务器。虽然这个服务器非常适合开发过程，但绝对不适合作为生产 Web 服务器，因为它并没有考虑安全性或性能。因此，它绝对不适合生产环境。

在选择 Web 服务器时，有几个选项可供选择，但**Apache**是迄今为止最受欢迎的选择，Django 开发团队实际上也推荐使用它。如何在 Apache 上设置 Django 的详细信息取决于您的托管解决方案。一些托管计划提供预配置的 Django 托管，您只需将项目文件复制到服务器上，而其他托管计划则允许您自己配置一切。

设置 Apache 的详细信息可能会因多种因素而有所不同，超出了本书的范围。如果最终需要自己配置 Apache，请参考 Django 文档[`www.djangoproject.com/documentation/apache_auth/`](http://www.djangoproject.com/documentation/apache_auth/)以获取详细说明。

# 总结

本章涵盖了各种有趣的主题。在本章中，我们为项目开发了一组重要的功能。追随者的网络对于帮助用户社交和共享兴趣非常重要。我们了解了几个在部署 Django 时有用的 Django 框架。我们还学会了如何将 Django 项目从开发环境迁移到生产环境。值得注意的是，我们学到的这些框架都非常易于使用，因此您将能够在未来的项目中有效地利用它们。这些功能在 Web 2.0 应用程序中很常见，现在，您将能够将它们整合到任何 Django 网站中。

在下一章中，我们将学习如何改进应用程序的各个方面，主要是性能和本地化。我们还将学习如何在生产服务器上部署我们的项目。下一章将提供大量有用的信息，所以请继续阅读！
