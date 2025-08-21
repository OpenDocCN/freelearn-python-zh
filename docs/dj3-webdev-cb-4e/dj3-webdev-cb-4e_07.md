# 第七章：安全和性能

在本章中，我们将涵盖以下配方：

+   使表单免受跨站点请求伪造（CSRF）的攻击

+   使用内容安全策略（CSP）使请求安全

+   使用 django-admin-honeypot

+   实施密码验证

+   下载经授权的文件

+   向图像添加动态水印

+   使用 Auth0 进行身份验证

+   缓存方法返回值

+   使用 Memcached 缓存 Django 视图

+   使用 Redis 缓存 Django 视图

# 介绍

如果软件不适当地暴露敏感信息，使用户遭受无休止的等待时间，或需要大量的硬件，那么它将永远无法持久。作为开发人员，我们有责任确保应用程序是安全和高性能的。在本章中，我们将仅仅讨论保持用户（和自己）在 Django 应用程序中安全运行的许多方法之一。然后，我们将介绍一些可以减少处理并以更低的成本（金钱和时间）将数据传递给用户的缓存选项。

# 技术要求

要使用本章中的代码，您需要最新稳定版本的 Python，一个 MySQL 或 PostgreSQL 数据库，以及一个带有虚拟环境的 Django 项目。

您可以在本书的 GitHub 存储库的`ch07`目录中找到本章的所有代码：[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 使表单免受跨站点请求伪造（CSRF）的攻击

如果没有适当的预防措施，恶意网站可能会针对您的网站发起请求，这将导致对服务器进行不希望的更改。例如，他们可能会影响用户的身份验证或未经用户同意地更改内容。Django 捆绑了一个系统来防止此类 CSRF 攻击，我们将在本章中进行审查。

# 准备工作

从我们在第三章中创建的*使用 CRUDL 功能创建应用*中的*ideas*应用开始。

# 如何做…

要在 Django 中启用 CSRF 预防，请按照以下步骤操作：

1.  确保在项目设置中包含`CsrfViewMiddleware`，如下所示：

```py
# myproject/settings/_base.py
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django.middleware.locale.LocaleMiddleware",
]
```

1.  确保使用请求上下文呈现表单视图。例如，在现有的`ideas`应用中，我们有这样的：

```py
# myproject/apps/ideas/views.py from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def add_or_change_idea(request, pk=None):
    # …
    return render(request, "ideas/idea_form.html", context)
```

1.  在表单模板中，确保使用`POST`方法并包括`{% csrf_token %}`标记：

```py
{# ideas/idea_form.html #}
{% extends "base.html" %}
{% load i18n crispy_forms_tags static %}

{% block content %}
    <h1>
        {% if idea %}
            {% blocktrans trimmed with title=idea
             .translated_title %}
                Change Idea "{{ title }}"
            {% endblocktrans %}
        {% else %}
            {% trans "Add Idea" %}
        {% endif %}
    </h1>
    <form action="{{ request.path }}" method="post">
 {% csrf_token %}
        {{ form.as_p }}
        <p>
            <button type="submit">{% trans "Save" %}</button>
        </p>
    </form>
{% endblock %}
```

1.  如果您使用`django-crispy-forms`进行表单布局，则 CSRF 令牌将默认包含在其中：

```py
{# ideas/idea_form.html #}
{% extends "base.html" %}
{% load i18n crispy_forms_tags static %}

{% block content %}
    <h1>
        {% if idea %}
            {% blocktrans trimmed with title=idea
             .translated_title %}
                Change Idea "{{ title }}"
            {% endblocktrans %}
        {% else %}
            {% trans "Add Idea" %}
        {% endif %}
    </h1>
    {% crispy form %}
{% endblock %}
```

# 它是如何工作的…

Django 使用隐藏字段方法来防止 CSRF 攻击。服务器上生成一个令牌，基于请求特定和随机化的信息组合。通过`CsrfViewMiddleware`，此令牌会自动通过请求上下文提供。

虽然不建议禁用此中间件，但可以通过应用`@csrf_protect`装饰器来标记单个视图以获得相同的行为：

```py
from django.views.decorators.csrf import csrf_protect

@csrf_protect
def my_protected_form_view():
    # …
```

同样，我们可以使用`@csrf_exempt`装饰器从 CSRF 检查中排除单个视图，即使中间件已启用：

```py
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def my_unsecured_form_view():
    # …
```

内置的`{% csrf_token %}`标记生成提供令牌的隐藏输入字段，如下例所示：

```py
<input type="hidden" name="csrfmiddlewaretoken" value="29sQH3UhogpseHH60eEaTq0xKen9TvbKe5lpT9xs30cR01dy5QVAtATWmAHvUZFk">
```

在使用`GET`、`HEAD`、`OPTIONS`或`TRACE`方法提交请求的表单中包含令牌被认为是无效的，因为任何使用这些方法的请求首先不应该引起副作用。在大多数情况下，需要 CSRF 保护的 Web 表单将是`POST`表单。

当使用不安全的方法提交受保护的表单而没有所需的令牌时，Django 的内置表单验证将识别此情况并拒绝请求。只有包含有效值令牌的提交才允许继续进行。因此，外部站点将无法更改您的服务器，因为它们将无法知道并包含当前有效的令牌值。

# 还有更多...

在许多情况下，希望增强一个表单，以便可以通过 Ajax 提交。这些也需要使用 CSRF 令牌进行保护，虽然可能在每个请求中作为额外数据注入令牌，但使用这种方法需要开发人员记住为每个`POST`请求这样做。使用 CSRF 令牌标头的替代方法存在，并且使事情更有效。

首先，需要检索令牌值，我们如何做取决于`CSRF_USE_SESSIONS`设置的值。当它为`True`时，令牌存储在会话中而不是 cookie 中，因此我们必须使用`{% csrf_token %}`标签将其包含在 DOM 中。然后，我们可以读取该元素以在 JavaScript 中检索数据：

```py
var input = document.querySelector('[name="csrfmiddlewaretoken"]');
var csrfToken = input && input.value; 
```

当`CSRF_USE_SESSIONS`设置处于默认的`False`状态时，令牌值的首选来源是`csrftoken` cookie。虽然可以自己编写 cookie 操作方法，但有许多可简化此过程的实用程序可用。例如，我们可以使用**js-cookie** API 轻松按名称提取令牌，该 API 可在[`github.com/js-cookie/js-cookie`](https://github.com/js-cookie/js-cookie)上找到，如下所示：

```py
var csrfToken = Cookies.get('crsftoken');
```

一旦令牌被提取，它需要被设置为`XmlHttpRequest`的 CSRF 令牌标头值。虽然可以为每个请求单独执行此操作，但这样做与为每个请求添加数据到请求参数具有相同的缺点。相反，我们可以使用 jQuery 及其在发送请求之前自动附加数据的能力，如下所示：

```py
var CSRF_SAFE_METHODS = ['GET', 'HEAD', 'OPTIONS', 'TRACE'];
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (CSRF_SAFE_METHODS.indexOf(settings.type) < 0
            && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrfToken);
        } 
    }
});
```

# 参见

+   *使用 CRUDL 功能创建应用程序*配方在第三章*，表单和视图*

+   *实施密码验证*配方

+   *下载授权文件*配方

+   *使用 Auth0 进行身份验证*配方

# 使用内容安全策略（CSP）使请求安全

动态多用户网站通常允许用户从各种媒体类型中添加各种数据：图像、视频、音频、HTML、JavaScript 片段等。这打开了用户向网站添加恶意代码的潜力，这些代码可能窃取 cookie 或其他个人信息，在后台调用不需要的 Ajax 请求，或者造成其他伤害。现代浏览器支持额外的安全层，它列入白名单您媒体资源的来源。它被称为 CSP，在这个配方中，我们将向您展示如何在 Django 网站中使用它。

# 准备工作

让我们从一个现有的 Django 项目开始；例如，包含来自第三章*，表单和视图*的`ideas`应用程序。

# 如何做...

要使用 CSP 保护您的项目，请按照以下步骤：

1.  将`django-csp`安装到您的虚拟环境中：

```py
(env)$ pip install django-csp==3.6
```

1.  在设置中，添加`CSPMiddleware`：

```py
# myproject/settings/_base.py
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "csp.middleware.CSPMiddleware",
]
```

1.  在相同的设置文件中，添加`django-csp`设置以列入您信任的包含媒体的来源，例如，jQuery 和 Bootstrap 的 CDN（您将在*它是如何工作的...*部分找到对此的详细解释）：

```py
# myproject/settings/_base.py
CSP_DEFAULT_SRC = [
    "'self'",
    "https://stackpath.bootstrapcdn.com/",
]
CSP_SCRIPT_SRC = [
    "'self'",
    "https://stackpath.bootstrapcdn.com/",
    "https://code.jquery.com/",
    "https://cdnjs.cloudflare.com/",
]
CSP_IMG_SRC = ["*", "data:"]
CSP_FRAME_SRC = ["*"]
```

1.  如果在模板中的任何地方有内联脚本或样式，请使用加密的`nonce`将它们列入白名单，如下所示：

```py
<script nonce="{{ request.csp_nonce }}">
    window.settings = {
        STATIC_URL: '{{ STATIC_URL }}',
        MEDIA_URL: '{{ MEDIA_URL }}',
    }
</script>
```

# 它是如何工作的...

CSP 指令可以添加到头部的 meta 标签或响应头中：

+   `meta`标签的语法如下：

```py
<meta http-equiv="Content-Security-Policy" content="img-src * data:; default-src 'self' https://stackpath.bootstrapcdn.com/ 'nonce-WWNu7EYqfTcVVZDs'; frame-src *; script-src 'self' https://stackpath.bootstrapcdn.com/ https://code.jquery.com/ https://cdnjs.cloudflare.com/">
```

+   我们选择的`django-csp`模块使用**响应头**来创建您希望加载到网站中的源列表。您可以在浏览器检查器的网络部分中检查头，如下所示：

```py
Content-Security-Policy: img-src * data:; default-src 'self' https://stackpath.bootstrapcdn.com/ 'nonce-WWNu7EYqfTcVVZDs'; frame-src *; script-src 'self' https://stackpath.bootstrapcdn.com/ https://code.jquery.com/ https://cdnjs.cloudflare.com/
```

CSP 允许您将资源类型和允许的来源定义在一起。您可以使用的主要指令如下：

+   `default-src`用作所有未设置来源的回退，并在 Django 设置中由`CSP_DEFAULT_SRC`控制。

+   `script-src`用于`<script>`标签，并在 Django 设置中由`CSP_DEFAULT_SRC`控制。

+   `style-src`用于`<style>`和`<link rel="stylesheet">`标签以及 CSS `@import`语句，并由`CSP_STYLE_SRC`设置控制。

+   `img-src`用于`<img>`标签，并由`CSP_IMG_SRC`设置控制。

+   `frame-src`用于`<frame>`和`<iframe>`标签，并由`CSP_FRAME_SRC`设置控制。

+   `media-src`用于`<audio>`、`<video>`和`<track>`标签，并由`CSP_MEDIA_SRC`设置控制。

+   `font-src`用于 Web 字体，并由`CSP_FONT_SRC`设置控制。

+   `connect-src`用于 JavaScript 加载的资源，并由`CSP_CONNECT_SRC`设置控制。

可以在[`developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy `](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy)和[`django-csp.readthedocs.io/en/latest/configuration.html`](https://django-csp.readthedocs.io/en/latest/configuration.html)找到每个指令的值的完整列表。

每个指令的值可以是以下列表中的一个或多个（单引号很重要）：

+   `*`：允许所有来源

+   `'none'`：禁止所有来源

+   `'self'`：允许来自相同域的来源

+   协议；例如，`https:`或`data:`

+   域名；例如，`example.com`或`*.example.com`

+   网站 URL，例如，`https://example.com`

+   `'unsafe-inline'`：允许内联`<script>`或`<style>`标签

+   `'unsafe-eval'`：允许使用`eval()`函数执行脚本

+   `'nonce-<b64-value>'`：通过加密 nonce 允许特定标签

+   `'sha256-...'`：通过源哈希允许资源

没有通用的配置`django-csp`的绝对方法。这总是一个反复试验的过程。不过，以下是我们的指导原则：

1.  首先为现有的工作项目添加 CSP。过早的限制只会使开发网站变得更加困难。

1.  检查所有已硬编码到模板中的脚本、样式、字体和其他静态文件，并将它们列入白名单。

1.  如果允许媒体嵌入到博客文章或其他动态内容中，请允许所有来源的图像、媒体和框架，如下所示：

```py
# myproject/settings/_base.py CSP_IMG_SRC = ["*"]
CSP_MEDIA_SRC = ["*"]
CSP_FRAME_SRC = ["*"]
```

1.  如果您使用内联脚本或样式，请在其中添加`nonce="{{ request.csp_nonce }}"`。

1.  除非通过在模板中硬编码 HTML 是唯一的进入网站的方式，否则避免使用`'unsafe-inline'`和`'unsafe-eval'`CSP 值。

1.  浏览网站，搜索任何未正确加载的内容。如果在开发者控制台中看到以下消息，意味着内容受到 CSP 的限制：

拒绝执行内联脚本，因为它违反了以下内容安全策略指令：“script-src 'self' https://stackpath.bootstrapcdn.com/ https://code.jquery.com/ https://cdnjs.cloudflare.com/”。要启用内联执行，需要使用'unsafe-inline'关键字、哈希（'sha256-P1v4zceJ/oPr/yp20lBqDnqynDQhHf76lljlXUxt7NI='）或 nonce（'nonce-...'）。

这类错误通常是因为一些第三方工具，如 django-cms、Django Debug Toolbar 和 Google Analytics，试图通过 JavaScript 包含资源而发生的。您可以使用资源哈希来将这些资源列入白名单，就像我们在错误消息中看到的那样：

`'sha256-P1v4zceJ/oPr/yp20lBqDnqynDQhHf76lljlXUxt7NI='`。

1.  如果您开发现代的**渐进式 Web 应用**（**PWA**），请考虑检查由`CSP_MANIFEST_SRC`和`CSP_WORKER_SRC`设置控制的清单和 Web Workers 的指令。

# 另请参阅

+   *使表单免受跨站请求伪造（CSRF）*的安全配方

# 使用 django-admin-honeypot

如果您保留 Django 网站的默认管理路径，您将使黑客能够执行暴力攻击，并尝试使用其列表中的不同密码登录。有一个名为 django-admin-honeypot 的应用程序，允许您伪造登录屏幕并检测这些暴力攻击。在本教程中，我们将学习如何使用它。

# 准备就绪

我们可以从任何要保护的 Django 项目开始。例如，您可以扩展上一个教程中的项目。

# 如何做...

按照以下步骤设置 django-admin-honeypot：

1.  在您的虚拟环境中安装模块：

```py
(env)$ pip install django-admin-honeypot==1.1.0
```

1.  在设置中的`INSTALLED_APPS`中添加`"admin_honeypot"`：

```py
# myproject/settings/_base.py INSTALLED_APPS = (
    # …
    "admin_honeypot",
)
```

1.  修改 URL 规则：

```py
# myproject/urls.py from django.contrib import admin
from django.conf.urls.i18n import i18n_patterns
from django.urls import include, path

urlpatterns = i18n_patterns(
    # …
    path("admin/", include("admin_honeypot.urls", 
    namespace="admin_honeypot")),
 path("management/", admin.site.urls),
)
```

# 它是如何工作的...

如果您转到默认的管理 URL，`http://127.0.0.1:8000/en/admin/`，您将看到登录屏幕，但无论您输入什么都将被描述为无效密码：

![](img/7fc7521b-9438-4ccd-9e5c-6b7efa6b23fc.png)

真实网站的管理现在位于`http://127.0.0.1:8000/en/management/`，您可以在那里看到来自蜜罐的跟踪登录。

# 还有更多...

在撰写本文时，django-admin-honeypot 与 Django 3.0 的功能不完善-管理界面会转义 HTML，而应该安全地呈现它。在 django-admin-honeypot 更新并提供新版本之前，我们可以通过进行一些更改来修复它，如下所示：

1.  创建一个名为`admin_honeypot_fix`的应用程序，其中包含以下代码的`admin.py`文件：

```py
# myproject/apps/admin_honeypot_fix/admin.py from django.contrib import admin

from admin_honeypot.admin import LoginAttemptAdmin
from admin_honeypot.models import LoginAttempt
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _

admin.site.unregister(LoginAttempt)

@admin.register(LoginAttempt)
class FixedLoginAttemptAdmin(LoginAttemptAdmin):
    def get_session_key(self, instance):
        return mark_safe('<a href="?session_key=
        %(key)s">%(key)s</a>' % {'key': instance.session_key})
    get_session_key.short_description = _('Session')

    def get_ip_address(self, instance):
        return mark_safe('<a href="?ip_address=%(ip)s">%(ip)s</a>' 
         % {'ip': instance.ip_address})
    get_ip_address.short_description = _('IP Address')

    def get_path(self, instance):
        return mark_safe('<a href="?path=%(path)s">%(path)s</a>' 
         % {'path': instance.path})
    get_path.short_description = _('URL')
```

1.  在同一个应用程序中，创建一个带有新应用程序配置的`apps.py`文件：

```py
# myproject/apps/admin_honeypot_fix/apps.py from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _

class AdminHoneypotConfig(AppConfig):
    name = "admin_honeypot"
    verbose_name = _("Admin Honeypot")

    def ready(self):
 from .admin import FixedLoginAttemptAdmin
```

1.  在设置中的`INSTALLED_APPS`中用新的应用程序配置替换`"admin_honeypot"`：

```py
# myproject/settings/_base.py INSTALLED_APPS = [
    # …
    #"admin_honeypot",
    "myproject.apps.admin_honeypot_fix.apps.AdminHoneypotConfig",
]
```

蜜罐中的登录尝试现在看起来是这样的：

![](img/3c76c52b-0f5c-4272-ab40-01ca4603d480.png)

# 另请参阅

+   *实施密码验证*教程

+   *使用 Auth0 进行身份验证*教程

# 实施密码验证

在软件安全失败列表的前面，有一项是用户选择不安全密码。在本教程中，我们将学习如何通过内置和自定义密码验证器来强制执行最低密码要求，以便用户被引导设置更安全的身份验证。

# 准备就绪

打开项目的设置文件并找到`AUTH_PASSWORD_VALIDATORS`设置。此外，创建一个新的`auth_extra`应用程序，其中包含一个`password_validation.py`文件。

# 如何做...

按照以下步骤为您的项目设置更强大的密码验证：

1.  通过添加一些选项来自定义 Django 中包含的验证器的设置：

```py
# myproject/settings/_base.py
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation."
        "UserAttributeSimilarityValidator",
        "OPTIONS": {"max_similarity": 0.5},
    },
    {
        "NAME": "django.contrib.auth.password_validation." 
        "MinimumLengthValidator",
        "OPTIONS": {"min_length": 12},
    },
    {"NAME": "django.contrib.auth.password_validation." 
    "CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation." 
    "NumericPasswordValidator"},
]
```

1.  在新的`auth_extra`应用程序的`password_validation.py`文件中添加`MaximumLengthValidator`类，如下所示：

```py
# myproject/apps/auth_extra/password_validation.py from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _

class MaximumLengthValidator:
    def __init__(self, max_length=24):
        self.max_length = max_length

    def validate(self, password, user=None):
        if len(password) > self.max_length:
            raise ValidationError(
                self.get_help_text(pronoun="this"),
                code="password_too_long",
                params={'max_length': self.max_length},
            )

    def get_help_text(self, pronoun="your"):
        return _(f"{pronoun.capitalize()} password must contain "
                 f"no more than {self.max_length} characters")
```

1.  在同一文件中，创建`SpecialCharacterInclusionValidator`类：

```py
class SpecialCharacterInclusionValidator:
    DEFAULT_SPECIAL_CHARACTERS = ('$', '%', ':', '#', '!')

    def __init__(self, special_chars=DEFAULT_SPECIAL_CHARACTERS):
        self.special_chars = special_chars

    def validate(self, password, user=None):
        has_specials_chars = False
        for char in self.special_chars:
            if char in password:
                has_specials_chars = True
                break
        if not has_specials_chars:
            raise ValidationError(
                self.get_help_text(pronoun="this"),
                code="password_missing_special_chars"
            )

    def get_help_text(self, pronoun="your"):
        return _(f"{pronoun.capitalize()} password must contain at"
                 " least one of the following special characters: "
                 f"{', '.join(self.special_chars)}")
```

1.  然后，将新的验证器添加到设置中：

```py
# myproject/settings/_base.py
from myproject.apps.auth_extra.password_validation import (
 SpecialCharacterInclusionValidator,
)

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation."
        "UserAttributeSimilarityValidator",
        "OPTIONS": {"max_similarity": 0.5},
    },
    {
        "NAME": "django.contrib.auth.password_validation." 
        "MinimumLengthValidator",
        "OPTIONS": {"min_length": 12},
    },
    {"NAME": "django.contrib.auth.password_validation." 
    "CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation." 
    "NumericPasswordValidator"},
    {
 "NAME": "myproject.apps.auth_extra.password_validation."
        "MaximumLengthValidator",
 "OPTIONS": {"max_length": 32},
 },
 {
 "NAME": "myproject.apps.auth_extra.password_validation."
        "SpecialCharacterInclusionValidator",
 "OPTIONS": {
 "special_chars": ("{", "}", "^", "&")
 + SpecialCharacterInclusionValidator
              .DEFAULT_SPECIAL_CHARACTERS
 },
 },
]
```

# 它是如何工作的...

Django 包含一组默认密码验证器：

+   `UserAttributeSimilarityValidator`确保所选择的任何密码不会与用户的某些属性太相似。默认情况下，相似性比率设置为`0.7`，并且检查的属性是用户名，名字和姓氏以及电子邮件地址。如果这些属性中包含多个单词，则每个单词都会被独立检查。

+   `MinimumLengthValidator`检查输入的密码至少是多少个字符长。默认情况下，密码必须至少为八个字符长。

+   `CommonPasswordValidator`指的是一个包含经常使用的密码列表的文件，因此是不安全的。 Django 默认使用的列表包含 1,000 个这样的密码。

+   `NumericPasswordValidator`验证输入的密码是否完全由数字组成。

当您使用`startproject`管理命令创建新项目时，这些选项将作为初始验证器集合的默认选项添加。在这个配方中，我们已经展示了如何调整这些选项以满足我们项目的需求，将密码的最小长度增加到 12 个字符。

对于`UserAttributeSimilarityValidator`，我们还将`max_similarity`减少到`0.5`，这意味着密码必须与用户属性有更大的差异。

查看`password_validation.py`，我们定义了两个新的验证器：

+   `MaximumLengthValidator`与内置的最小长度验证器非常相似，确保密码不超过默认的 24 个字符

+   `SpecialCharacterInclusionValidator`检查密码中是否包含一个或多个特殊字符，默认情况下定义为`$`、`%`、`:`、`#`和`!`符号

每个验证器类都有两个必需的方法：

+   `validate()`方法执行对`password`参数的实际检查。可选地，当用户已经通过身份验证时，将传递第二个`user`参数。

+   我们还必须提供一个`get_help_text()`方法，该方法返回描述用户验证要求的字符串。

最后，我们将新的验证器添加到设置中，以覆盖默认设置，允许密码的最大长度为 32 个字符，并且能够将符号`{`、`}`、`^`和`&`添加到默认的特殊字符列表中。

# 还有更多...

在`AUTH_PASSWORD_VALIDATORS`中提供的验证器会自动执行`createsuperuser`和`changepassword`管理命令，以及用于更改或重置密码的内置表单。但是，有时您可能希望对自定义密码管理代码使用相同的验证。Django 提供了该级别集成的函数，您可以在`django.contrib.auth.password_validation`模块中的贡献的 Django `auth`应用程序中检查详细信息。

# 另请参阅

+   *下载授权文件*配方

+   *使用 Auth0 进行身份验证*配方

# 下载授权文件

有时，您可能只需要允许特定的人从您的网站下载知识产权。例如，音乐、视频、文学或其他艺术作品只应该对付费会员开放。在这个配方中，您将学习如何使用贡献的 Django auth 应用程序，将图像下载限制仅对经过身份验证的用户。

# 准备工作

让我们从我们在第三章中创建的`ideas`应用开始。

# 如何做...

逐步执行这些步骤：

1.  创建需要身份验证才能下载文件的视图，如下所示：

```py
# myproject/apps/ideas/views.py import os

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponseNotFound
from django.shortcuts import get_object_or_404
from django.utils.text import slugify

from .models import Idea

@login_required
def download_idea_picture(request, pk):
    idea = get_object_or_404(Idea, pk=pk)
    if idea.picture:
        filename, extension = 
        os.path.splitext(idea.picture.file.name)
        extension = extension[1:] # remove the dot
        response = FileResponse(
            idea.picture.file, content_type=f"image/{extension}"
        )
        slug = slugify(idea.title)[:100]
        response["Content-Disposition"] = (
            "attachment; filename="
            f"{slug}.{extension}"
        )
    else:
        response = HttpResponseNotFound(
            content="Picture unavailable"
        )
    return response
```

1.  将下载视图添加到 URL 配置中：

```py
# myproject/apps/ideas/urls.py from django.urls import path

from .views import download_idea_picture

urlpatterns = [
    # …
    path(
 "<uuid:pk>/download-picture/",
 download_idea_picture,
 name="download_idea_picture",
 ),
]
```

1.  在我们项目的 URL 配置中设置登录视图：

```py
# myproject/urls.py from django.conf.urls.i18n import i18n_patterns
from django.urls import include, path

urlpatterns = i18n_patterns(
    # …
    path("accounts/", include("django.contrib.auth.urls")),
    path("ideas/", include(("myproject.apps.ideas.urls", "ideas"), 
     namespace="ideas")),
)
```

1.  创建登录表单的模板，如下所示：

```py
{# registration/login.html #} {% extends "base.html" %}
{% load i18n %}

{% block content %}
    <h1>{% trans "Login" %}</h1>
    <form action="{{ request.path }}" method="POST">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit" class="btn btn-primary">{% trans 
         "Log in" %}</button>
    </form>
{% endblock %}
```

1.  在想法详情的模板中，添加一个下载链接：

```py
{# ideas/idea_detail.html #}
{% extends "base.html" %}
{% load i18n %}

{% block content %}
…
 <a href="{% url 'ideas:download_idea_picture' pk=idea.pk %}" 
     class="btn btn-primary">{% trans "Download picture" %}</a>
{% endblock %}
```

您应该限制用户绕过 Django 直接下载受限文件。要做到这一点，在 Apache web 服务器上，如果您正在运行 Apache 2.4，可以在`media/ideas`目录中放置一个`.htaccess`文件，内容如下：

```py
# media/ideas/.htaccess Require all denied
```

当使用`django-imagekit`时，如本书中的示例所示，生成的图像版本将存储在`media/CACHE`目录中，并从那里提供服务，因此我们的`.htaccess`配置不会影响它。

# 工作原理...

`download_idea_picture`视图从特定想法中流式传输原始上传的图片。设置为`attachment`的`Content-Disposition`标头使文件可下载，而不是立即在浏览器中显示。该文件的文件名也在此标头中设置，类似于`gamified-donation-platform.jpg`。如果某个想法的图片不可用，将显示一个带有非常简单消息的 404 页面：图片不可用。

`@login_required`装饰器将在访问可下载文件时重定向访问者到登录页面，如果他们未登录。默认情况下，登录屏幕如下所示：

![](img/273c8926-f6a8-4087-b876-ae6bd05c0981.png)

# 另请参阅

+   来自第三章的*上传图像*食谱，*表单和视图*

+   来自第三章的*使用自定义模板创建表单布局*食谱，*表单和视图*

+   来自第三章的*使用 django-crispy-forms 创建表单布局*食谱，*表单和视图*

+   来自第四章的*安排 base.html 模板*食谱，*模板和 JavaScript*

+   *实施密码验证*食谱

+   *向图像添加动态水印*食谱

# 向图像添加动态水印

有时，允许用户查看图像，但防止由于知识产权和艺术权利而重新分发是可取的。在这个食谱中，我们将学习如何向在您的网站上显示的图像应用水印。

# 做好准备

让我们从我们在第三章中创建的`core`和`ideas`应用程序开始，*创建具有 CRUDL 功能的应用程序*食谱，*表单和视图*。

# 如何做...

按照以下步骤将水印应用于显示的 idea 图像：

1.  如果尚未这样做，请将`django-imagekit`安装到您的虚拟环境中：

```py
(env)$ pip install django-imagekit==4.0.2
```

1.  在设置中将`"imagekit"`放入`INSTALLED_APPS`：

```py
# myproject/settings/_base.py
INSTALLED_APPS = [
    # …
    "imagekit",
]
```

1.  在`core`应用程序中，创建一个名为`processors.py`的文件，其中包含`WatermarkOverlay`类，如下所示：

```py
# myproject/apps/core/processors.py
from pilkit.lib import Image

class WatermarkOverlay(object):
    def __init__(self, watermark_image):
        self.watermark_image = watermark_image

    def process(self, img):
        original = img.convert('RGBA')
        overlay = Image.open(self.watermark_image)
        img = Image.alpha_composite(original, 
        overlay).convert('RGB')
        return img
```

1.  在`Idea`模型中，将`watermarked_picture_large`规格添加到`picture`字段旁边，如下所示：

```py
# myproject/apps/ideas/models.py import os

from imagekit.models import ImageSpecField
from pilkit.processors import ResizeToFill

from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from django.utils.timezone import now as timezone_now

from myproject.apps.core.models import CreationModificationDateBase, UrlBase
from myproject.apps.core.processors import WatermarkOverlay

def upload_to(instance, filename):
    now = timezone_now()
    base, extension = os.path.splitext(filename)
    extension = extension.lower()
    return f"ideas/{now:%Y/%m}/{instance.pk}{extension}"

class Idea(CreationModificationDateBase, UrlBase):
    # …
    picture = models.ImageField(
        _("Picture"), upload_to=upload_to
    )
    watermarked_picture_large = ImageSpecField(
 source="picture",
 processors=[
 ResizeToFill(800, 400),
 WatermarkOverlay(
 watermark_image=os.path.join(settings.STATIC_ROOT, 
                'site', 'img', 'watermark.png'),
 )
 ],
 format="PNG"
    )
```

1.  使用您选择的图形程序，在透明背景上创建一个带有白色文本或标志的半透明 PNG 图像。将其大小设置为 800 x 400 像素。将图像保存为`site_static/site/img/watermark.png`。它可能看起来像这样：

![](img/e5d2c032-4ad3-426e-9fa5-a932de9e6183.png)

1.  之后运行`collectstatic`管理命令：

```py
(env)$ export DJANGO_SETTINGS_MODULE=myproject.settings.dev
(env)$ python manage.py collectstatic
```

1.  编辑 idea 详细模板，并添加水印图像，如下所示：

```py
{# ideas/idea_detail.html #} {% extends "base.html" %}
{% load i18n %}

{% block content %}
    <a href="{% url "ideas:idea_list" %}">{% trans "List of ideas" 
     %}</a>
    <h1>
        {% blocktrans trimmed with title=idea.translated_title %}
            Idea "{{ title }}"
        {% endblocktrans %}
    </h1>
    <img src="img/{{ idea.watermarked_picture_large.url }}" alt="" />
    {{ idea.translated_content|linebreaks|urlize }}
    <p>
        {% for category in idea.categories.all %}
            <span class="badge badge-pill badge-info">
             {{ category.translated_title }}</span>
        {% endfor %}
    </p>
    <a href="{% url 'ideas:download_idea_picture' pk=idea.pk %}" 
     class="btn btn-primary">{% trans "Download picture" %}</a>
{% endblock %}
```

# 它是如何工作的...

如果我们导航到 idea 详细页面，我们应该看到大图像被我们的水印遮盖，类似于这样：

![](img/527e60a3-66e9-45f5-87bd-f521ce8fc34f.png)

让我们来看看是如何做到的。在详细模板中，`<img>`标签的`src`属性使用了 idea 的图像规格，即`watermarked_picture_large`，以创建一个修改后的图像，然后将其保存在`media/CACHE/`目录下并从那里提供服务。

`django-imagekit`规格使用处理器修改图像。那里使用了两个处理器：

+   `ResizeToFill`将图像调整为 800×400 像素

+   我们的自定义处理器`WatermarkOverlay`将半透明叠加层应用于它

`django-imagekit`处理器必须具有一个`process()`方法，该方法获取来自先前处理器的图像并返回一个新的修改后的图像。在我们的情况下，我们将结果从原始图像和半透明叠加层组合而成。

# 另请参阅

+   *下载授权文件*食谱

# 使用 Auth0 进行身份验证

随着人们每天互动的服务数量的增加，他们需要记住的用户名和密码的数量也在增加。除此之外，用户信息存储的每个额外位置都是在安全漏洞发生时可能被盗窃的另一个位置。为了帮助缓解这一问题，诸如**Auth0**之类的服务允许您在单一安全平台上集中身份验证服务。

除了支持用户名和密码凭据外，Auth0 还可以通过 Google、Facebook 或 Twitter 等社交平台验证用户。您可以使用通过短信或电子邮件发送的一次性代码进行无密码登录，甚至支持不同服务的企业级支持。在本教程中，您将学习如何将 Auth0 应用连接到 Django，并如何集成它以处理用户身份验证。

# 准备就绪

如果尚未这样做，请在 [`auth0.com/`](https://auth0.com/) 创建一个 Auth0 应用，并按照那里的说明进行配置。免费计划提供了两个社交连接，因此我们将激活 Google 和 Twitter 以使用它们登录。您还可以尝试其他服务。请注意，其中一些服务需要您注册应用并获取 API 密钥和密钥。

接下来，我们需要在项目中安装 `python-social-auth` 和其他一些依赖项。将这些依赖项包含在您的 `pip` 要求中：

```py
# requirements/_base.txt
social-auth-app-django~=3.1
python-jose~=3.0
python-dotenv~=0.9
```

`social-auth-app-django` 是 `python-social-auth` 项目的 Django 特定包，允许您使用许多社交连接之一进行网站身份验证。

使用 `pip` 将这些依赖项安装到您的虚拟环境中。

# 如何做...

要将 Auth0 连接到您的 Django 项目，请按照以下步骤进行：

1.  在设置文件中的 `INSTALLED_APPS` 中添加社交身份验证应用，如下所示：

```py
# myproject/settings/_base.py
INSTALLED_APPS = [
    # …
    "social_django",
]
```

1.  现在，添加 `social_django` 应用所需的 Auth0 设置，如下所示：

```py
# myproject/settings/_base.py
SOCIAL_AUTH_AUTH0_DOMAIN = get_secret("AUTH0_DOMAIN")
SOCIAL_AUTH_AUTH0_KEY = get_secret("AUTH0_KEY")
SOCIAL_AUTH_AUTH0_SECRET = get_secret("AUTH0_SECRET")
SOCIAL_AUTH_AUTH0_SCOPE = ["openid", "profile", "email"]
SOCIAL_AUTH_TRAILING_SLASH = False

```

确保您在您的秘密或环境变量中定义 `AUTH0_DOMAIN`，`AUTH0_KEY` 和 `AUTH0_SECRET`。这些变量的值可以在您在本教程的 *准备就绪* 部分的 *第 1 步* 中创建的 Auth0 应用的设置中找到。

1.  我们需要为 Auth0 连接创建一个后端，如下例所示：

```py
# myproject/apps/external_auth/backends.py from urllib import request
from jose import jwt
from social_core.backends.oauth import BaseOAuth2

class Auth0(BaseOAuth2):
    """Auth0 OAuth authentication backend"""

    name = "auth0"
    SCOPE_SEPARATOR = " "
    ACCESS_TOKEN_METHOD = "POST"
    REDIRECT_STATE = False
    EXTRA_DATA = [("picture", "picture"), ("email", "email")]

    def authorization_url(self):
        return "https://" + self.setting("DOMAIN") + "/authorize"

    def access_token_url(self):
        return "https://" + self.setting("DOMAIN") + "/oauth/token"

    def get_user_id(self, details, response):
        """Return current user id."""
        return details["user_id"]

    def get_user_details(self, response):
        # Obtain JWT and the keys to validate the signature
        id_token = response.get("id_token")
        jwks = request.urlopen(
            "https://" + self.setting("DOMAIN") + "/.well-
              known/jwks.json"
        )
        issuer = "https://" + self.setting("DOMAIN") + "/"
        audience = self.setting("KEY")  # CLIENT_ID
        payload = jwt.decode(
            id_token,
            jwks.read(),
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
        )
        first_name, last_name = (payload.get("name") or 
         " ").split(" ", 1)
        return {
            "username": payload.get("nickname") or "",
            "first_name": first_name,
            "last_name": last_name,
            "picture": payload.get("picture") or "",
            "user_id": payload.get("sub") or "",
            "email": payload.get("email") or "",
        }
```

1.  将新后端添加到您的 `AUTHENTICATION_BACKENDS` 设置中，如下所示：

```py
# myproject/settings/_base.py
AUTHENTICATION_BACKENDS = {
    "myproject.apps.external_auth.backends.Auth0",
    "django.contrib.auth.backends.ModelBackend",
}
```

1.  我们希望社交身份验证用户可以从任何模板中访问。因此，我们将为其创建一个上下文处理器：

```py
# myproject/apps/external_auth/context_processors.py
def auth0(request):
    data = {}
    if request.user.is_authenticated:
        auth0_user = request.user.social_auth.filter(
            provider="auth0",
        ).first()
        data = {
            "auth0_user": auth0_user,
        }
    return data
```

1.  接下来，我们需要在设置中注册它：

```py
# myproject/settings/_base.py
TEMPLATES = [
    {
        "BACKEND": 
        "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "myproject", "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors
                 .messages",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "myproject.apps.core.context_processors
                 .website_url",
                "myproject.apps.external_auth
               .context_processors.auth0",
            ]
        },
    }
]
```

1.  现在，让我们为索引页面、仪表板和注销创建视图：

```py
# myproject/apps/external_auth/views.py
from urllib.parse import urlencode

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as log_out
from django.conf import settings

def index(request):
    user = request.user
    if user.is_authenticated:
        return redirect(dashboard)
    else:
        return render(request, "index.html")

@login_required
def dashboard(request):
    return render(request, "dashboard.html")

def logout(request):
    log_out(request)
    return_to = urlencode({"returnTo": 
     request.build_absolute_uri("/")})
    logout_url = "https://%s/v2/logout?client_id=%s&%s" % (
        settings.SOCIAL_AUTH_AUTH0_DOMAIN,
        settings.SOCIAL_AUTH_AUTH0_KEY,
        return_to,
    )
    return redirect(logout_url)
```

1.  创建索引模板，如下所示：

```py
{# index.html #}
{% extends "base.html" %}
{% load i18n utility_tags %}

{% block content %}
<div class="login-box auth0-box before">
    <h3>{% trans "Please log in for the best user experience" %}</h3>
    <a class="btn btn-primary btn-lg" href="{% url "social:begin" 
     backend="auth0" %}">{% trans "Log in" %}</a>
</div>
{% endblock %}
```

1.  相应地创建仪表板模板：

```py
{# dashboard.html #}
{% extends "base.html" %}
{% load i18n %}

{% block content %}
    <div class="logged-in-box auth0-box logged-in">
        <img alt="{% trans 'Avatar' %}" src="img/>         auth0_user.extra_data.picture }}" 
         width="50" height="50" />
        <h2>{% blocktrans with name=request.user
         .first_name %}Welcome, {{ name }}
         {% endblocktrans %}!</h2>

        <a class="btn btn-primary btn-logout" href="{% url 
         "auth0_logout" %}">{% trans "Log out" %}</a>
    </div>
{% endblock %}
```

1.  更新 URL 规则：

```py
# myproject/urls.py
from django.conf.urls.i18n import i18n_patterns
from django.urls import path, include

from myproject.apps.external_auth import views as external_auth_views

urlpatterns = i18n_patterns(
    path("", external_auth_views.index, name="index"),
    path("dashboard/", external_auth_views.dashboard, 
     name="dashboard"),
    path("logout/", external_auth_views.logout, 
     name="auth0_logout"),
    path("", include("social_django.urls")),
    # …
)
```

1.  最后，添加登录 URL 设置：

```py
LOGIN_URL = "/login/auth0"
LOGIN_REDIRECT_URL = "dashboard"
```

# 工作原理...

如果您将浏览器指向项目的索引页面，您将看到一个链接邀请您登录。当您点击它时，您将被重定向到 Auth0 身份验证系统，其屏幕将类似于以下内容：

![](img/6999e777-9040-41a5-93e8-b93bd128971a.png)

这些都是由 `python-social-auth` 和 `Auth0` 后端的 `SOCIAL_AUTH_*` 设置配置的开箱即用功能。

一旦成功完成登录，Auth0 后端将接收来自响应的数据并处理它。相关数据附加到与请求关联的用户对象。在达到 `LOGIN_REDIRECT_URL` 的身份验证结果的仪表板视图中，提取用户详细信息并添加到模板上下文中。然后呈现 `dashboard.html`。结果可能如下所示：

![](img/0da83ec6-d5be-411f-9d3b-865f1de66ba7.png)

仪表板上呈现的注销按钮在按下时将注销用户。

# 另请参阅

+   *实施密码验证* 教程

+   *下载授权文件* 教程

# 缓存方法返回值

如果在请求-响应周期中多次调用具有繁重计算或数据库查询的模型方法，则视图的性能可能会变得非常慢。在本教程中，您将了解一种模式，可以使用它来缓存方法的返回值以供以后重复使用。请注意，我们在这里不使用 Django 缓存框架，只使用 Python 默认提供的内容。

# 准备就绪

选择一个具有耗时方法的模型的应用程序，该方法将在同一请求-响应周期中重复使用。

# 如何做...

执行以下步骤：

1.  这是一个模式，您可以用它来缓存模型的方法返回值，以便在视图、表单或模板中重复使用，如下所示：

```py
class SomeModel(models.Model):
    def some_expensive_function(self):
        if not hasattr(self, "_expensive_value_cached"):
            # do some heavy calculations...
            # ... and save the result to result variable
            self._expensive_value_cached = result
        return self._expensive_value_cached
```

1.  例如，让我们为`ViralVideo`模型创建一个`get_thumbnail_url()`方法。您将在第十章*数据库查询表达式*食谱中更详细地探讨这个问题，标题是《花里胡哨》：

```py
# myproject/apps/viral_videos/models.py
import re
from django.db import models
from django.utils.translation import ugettext_lazy as _

from myproject.apps.core.models import CreationModificationDateBase, UrlBase

class ViralVideo(CreationModificationDateBase, UrlBase):
    embed_code = models.TextField(
        _("YouTube embed code"),
        blank=True)

    # …

    def get_thumbnail_url(self):
        if not hasattr(self, "_thumbnail_url_cached"):
            self._thumbnail_url_cached = ""
            url_pattern = re.compile(
                r'src="img/([^"]+)"'
            )
            match = url_pattern.search(self.embed_code)
            if match:
                video_id = match.groups()[0]
                self._thumbnail_url_cached = (
                    f"https://img.youtube.com/vi/{video_id}/0.jpg"
                )
        return self._thumbnail_url_cached
```

# 它是如何工作的...

在这个通用的例子中，该方法检查模型实例是否存在`_expensive_value_cached`属性。如果不存在，将执行耗时的计算，并将结果赋给这个新属性。在方法结束时，返回缓存的值。当然，如果您有几个繁重的方法，您将需要使用不同的属性名称来保存每个计算出的值。

现在，您可以在模板的页眉和页脚中使用`{{ object.some_expensive_function }}`之类的东西，耗时的计算将只进行一次。

在模板中，您还可以在`{% if %}`条件和值的输出中使用该函数，如下所示：

```py
{% if object.some_expensive_function %}
    <span class="special">
        {{ object.some_expensive_function }}
    </span>
{% endif %}
```

在另一个例子中，我们通过解析视频嵌入代码的 URL，获取其 ID，然后组成缩略图图像的 URL 来检查 YouTube 视频的缩略图。通过这样做，您可以在模板中使用它，如下所示：

```py
{% if video.get_thumbnail_url %}
    <figure>
        <img src="img/{{ video.get_thumbnail_url }}"
             alt="{{ video.title }}" 
        />
        <figcaption>{{ video.title }}</figcaption>
    </figure>
{% endif %}
```

# 还有更多...

我们刚刚描述的方法只有在方法被调用时没有参数时才有效，这样结果将始终相同。但是如果输入有所不同怎么办？自 Python 3.2 以来，有一个装饰器可以使用，基于参数的哈希（至少是可哈希的参数）提供基本的**最近最少使用**（**LRU**）缓存。

例如，让我们看一个人为而琐碎的例子，有一个函数接受两个值，并返回一些昂贵逻辑的结果：

```py
def busy_bee(a, b):
    # expensive logic
    return result
```

如果我们有这样一个函数，并且希望提供一个缓存来存储一些常用输入变化的结果，我们可以很容易地使用`functools`包中的`@lru_cache`装饰器来实现，如下所示：

```py
from functools import lru_cache

@lru_cache(maxsize=100, typed=True)
def busy_bee(a, b):
    # expensive logic
    return result
```

现在，我们提供了一个缓存机制，它将在从输入中计算出的哈希键下存储最多 100 个结果。`typed`选项是在 Python 3.3 中添加的，通过指定`True`，我们使得具有`a=1`和`b=2`的调用将与具有`a=1.0`和`b=2.0`的调用分开存储。根据逻辑操作的方式和返回值的内容，这种变化可能合适也可能不合适。

您可以在[`docs.python.org/3/library/functools.html#functools.lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache)的`functools`文档中了解更多关于`@lru_cache`装饰器的信息。

我们还可以在本食谱中的前面的例子中使用这个装饰器来简化代码，如下所示：

```py
# myproject/apps/viral_videos/models.py
from functools import lru_cache # …

class ViralVideo(CreationModificationDateMixin, UrlMixin):
    # …
    @lru_cache
    def get_thumbnail_url(self):
        # …
```

# 另请参阅

+   第四章*模板和 JavaScript*

+   *使用 Memcached 缓存 Django 视图*食谱

+   *使用 Redis 缓存 Django 视图*食谱

# 使用 Memcached 缓存 Django 视图

Django 允许我们通过缓存最昂贵的部分，如数据库查询或模板渲染，来加快请求-响应周期。Django 本身支持的最快、最可靠的缓存是基于内存的缓存服务器**Memcached**。在这个食谱中，您将学习如何使用 Memcached 来为`viral_videos`应用程序缓存视图。我们将在第十章*数据库查询表达式*食谱中进一步探讨这个问题，标题是《花里胡哨》。

# 准备工作

为了为我们的 Django 项目准备缓存，我们需要做几件事：

1.  让我们安装`memcached`服务。例如，在 macOS 上最简单的方法是使用 Homebrew：

```py
$ brew install memcached
```

1.  然后，您可以使用以下命令启动、停止或重新启动 Memcached 服务：

```py
$ brew services start memcached
$ brew services stop memcached
$ brew services restart memcached
```

在其他操作系统上，您可以使用 apt-get、yum 或其他默认的软件包管理工具安装 Memcached。另一个选项是从源代码编译，如[`memcached.org/downloads`](https://memcached.org/downloads)中所述。

1.  在您的虚拟环境中安装 Memcached Python 绑定，如下：

```py
(env)$ pip install python-memcached==1.59
```

# 如何做...

要为特定视图集成缓存，请执行以下步骤：

1.  在项目设置中设置`CACHES`如下：

```py
# myproject/settings/_base.py
CACHES = {
    "memcached": {
        "BACKEND": 
        "django.core.cache.backends.memcached.MemcachedCache",
        "LOCATION": get_secret("CACHE_LOCATION"),
        "TIMEOUT": 60,  # 1 minute
        "KEY_PREFIX": "myproject",
    },
}
CACHES["default"] = CACHES["memcached"]
```

1.  确保您的秘密或环境变量中的`CACHE_LOCATION`设置为`"localhost:11211"`。

1.  修改`viral_videos`应用的视图，如下：

```py
# myproject/apps/viral_videos/views.py from django.shortcuts import render
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_cookie

@vary_on_cookie
@cache_page(60)
def viral_video_detail(request, pk):
    # …
    return render(
        request,
        "viral_videos/viral_video_detail.html",
        {'video': video}
    )
```

如果您按照下一个配方中的 Redis 设置，您会发现`views.py`文件没有任何变化。这表明我们可以随意更改底层的缓存机制，而无需修改使用它的代码。

# 工作原理...

正如您将在第十章的*使用数据库查询表达式*配方中看到的那样，病毒视频的详细视图显示了经过认证和匿名用户的印象数量。如果您访问一个病毒视频（例如在`http://127.0.0.1:8000/en/videos/1/`）并启用缓存后刷新页面几次，您会注意到印象数量只在一分钟内改变一次。这是因为每个响应对于每个用户都被缓存 60 秒。我们使用`@cache_page`装饰器为视图设置了缓存。

Memcached 是一个键值存储，它默认使用完整的 URL 来为每个缓存页面生成键。当两个访问者同时访问同一页面时，第一个访问者的请求会收到由 Python 代码生成的页面，而第二个访问者会从 Memcached 服务器获取相同的 HTML 代码。

在我们的示例中，为了确保每个访问者即使访问相同的 URL 也会被单独处理，我们使用了`@vary_on_cookie`装饰器。这个装饰器检查了 HTTP 请求中`Cookie`头的唯一性。

您可以从官方文档[`docs.djangoproject.com/en/3.0/topics/cache/`](https://docs.djangoproject.com/en/3.0/topics/cache/)了解更多关于 Django 缓存框架的信息。同样，您也可以在[`memcached.org/`](https://memcached.org/)了解更多关于 Memcached 的信息。

# 另请参阅

+   *缓存方法返回值*配方

+   *使用 Redis 缓存 Django 视图*配方

+   第十章*，花里胡哨*

# 使用 Redis 缓存 Django 视图

尽管 Memcached 在市场上作为缓存机制已经很成熟，并且得到了 Django 的很好支持，但 Redis 是一个提供了 Memcached 所有功能以及更多功能的备用系统。在这里，我们将重新审视*使用 Memcached 缓存 Django 视图*的过程，并学习如何使用 Redis 来实现相同的功能。

# 准备工作

为了为我们的 Django 项目准备缓存，我们需要做几件事：

1.  让我们安装 Redis 服务。例如，在 macOS 上最简单的方法是使用 Homebrew：

```py
$ brew install redis
```

1.  然后，您可以使用以下命令启动、停止或重新启动 Redis 服务：

```py
$ brew services start redis
$ brew services stop redis
$ brew services restart redis
```

在其他操作系统上，您可以使用 apt-get、yum 或其他默认的软件包管理工具安装 Redis。另一个选项是从源代码编译，如[`redis.io/download`](https://redis.io/download)中所述。

1.  在您的虚拟环境中安装 Django 和其依赖的 Redis 缓存后端，如下：

```py
(env)$ pip install redis==3.3.11
(env)$ pip install hiredis==1.0.1
(env)$ pip install django-redis-cache==2.1.0

```

# 如何做...

要为特定视图集成缓存，请执行以下步骤：

1.  在项目设置中设置`CACHES`如下：

```py
# myproject/settings/_base.py
CACHES = {
    "redis": {
        "BACKEND": "redis_cache.RedisCache",
        "LOCATION": [get_secret("CACHE_LOCATION")],
        "TIMEOUT": 60, # 1 minute
        "KEY_PREFIX": "myproject",
    },
}
CACHES["default"] = CACHES["redis"]
```

1.  确保您的秘密或环境变量中的`CACHE_LOCATION`设置为`"localhost:6379"`。

1.  修改`viral_videos`应用的视图，如下：

```py
# myproject/apps/viral_videos/views.py from django.shortcuts import render
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_cookie

@vary_on_cookie
@cache_page(60)
def viral_video_detail(request, pk):
    # …
    return render(
        request,
        "viral_videos/viral_video_detail.html",
        {'video': video}
    )
```

如果您按照上一个教程中的 Memcached 设置进行操作，您会发现在这里的`views.py`中没有任何变化。这表明我们可以随意更改底层缓存机制，而无需修改使用它的代码。

# 它是如何工作的...

就像使用 Memcached 一样，我们使用`@cache_page`装饰器为视图设置缓存。因此，每个用户的每个响应都会被缓存 60 秒。视频详细信息视图（例如`http://127.0.0.1:8000/en/videos/1/`）显示了经过认证和匿名用户的印象数量。启用缓存后，如果您多次刷新页面，您会注意到印象数量每分钟只变化一次。

就像 Memcached 一样，Redis 是一个键值存储，当用于缓存时，它会根据完整的 URL 为每个缓存页面生成密钥。当两个访问者同时访问同一页面时，第一个访问者的请求将接收到由 Python 代码生成的页面，而第二个访问者将从 Redis 服务器获取相同的 HTML 代码。

在我们的示例中，为了确保每个访问者即使访问相同的 URL 也会被单独对待，我们使用了`@vary_on_cookie`装饰器。该装饰器检查 HTTP 请求中`Cookie`头的唯一性。

您可以从官方文档了解有关 Django 缓存框架的更多信息[`docs.djangoproject.com/en/3.0/topics/cache/`](https://docs.djangoproject.com/en/3.0/topics/cache/)。同样，您也可以在[`redis.io/`](https://redis.io/)上了解有关 Memcached 的更多信息。

# 还有更多...

Redis 能够像 Memcached 一样处理缓存，系统内置了大量额外的缓存算法选项。除了缓存，Redis 还可以用作数据库或消息存储。它支持各种数据结构、事务、发布/订阅和自动故障转移等功能。

通过 django-redis-cache 后端，Redis 也可以轻松配置为会话后端，就像这样：

```py
# myproject/settings/_base.py
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
```

# 另请参阅

+   *缓存方法返回值*教程

+   *使用 Memcached 缓存 Django 视图*教程

+   第十章*，花里胡哨*
