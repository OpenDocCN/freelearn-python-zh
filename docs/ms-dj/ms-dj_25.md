# 附录 D。设置

你的 Django 设置文件包含了你的 Django 安装的所有配置。本附录解释了设置的工作原理以及可用的设置。

# 什么是设置文件？

设置文件只是一个具有模块级变量的 Python 模块。以下是一些示例设置：

```py
ALLOWED_HOSTS = ['www.example.com'] DEBUG = False DEFAULT_FROM_EMAIL = 'webmaster@example.com' 

```

### 注意

如果将`DEBUG`设置为`False`，还需要正确设置`ALLOWED_HOSTS`设置。

因为设置文件是一个 Python 模块，所以以下规则适用：

+   它不允许有 Python 语法错误

+   它可以使用常规的 Python 语法动态分配设置，例如：

```py
        MY_SETTING = [str(i) for i in range(30)] 

```

+   它可以从其他设置文件中导入值

## 默认设置

如果不需要，Django 设置文件不必定义任何设置。每个设置都有一个合理的默认值。这些默认值存储在模块`django/conf/global_settings.py`中。以下是 Django 在编译设置时使用的算法：

+   从`global_settings.py`加载设置

+   从指定的设置文件中加载设置，必要时覆盖全局设置

请注意，设置文件不应该从`global_settings`导入，因为那是多余的。

## 查看您已更改的设置

有一种简单的方法来查看您的设置中有哪些与默认设置不同。命令`python manage.py diffsettings`显示当前设置文件与 Django 默认设置之间的差异。有关更多信息，请参阅`diffsettings`文档。

# 在 Python 代码中使用设置

在 Django 应用程序中，通过导入对象`django.conf.settings`来使用设置。例如：

```py
from django.conf import settings
if settings.DEBUG:
     # Do something 

```

请注意，`django.conf.settings`不是一个模块-它是一个对象。因此，无法导入单个设置：

```py
from django.conf.settings import DEBUG  # This won't work. 

```

还要注意，您的代码不应该从`global_settings`或您自己的设置文件中导入。`django.conf.settings`抽象了默认设置和站点特定设置的概念；它提供了一个单一的接口。它还将使用设置的代码与设置的位置解耦。

# 在运行时更改设置

您不应该在应用程序中在运行时更改设置。例如，在视图中不要这样做：

```py
from django.conf import settings
settings.DEBUG = True   # Don't do this! 

```

唯一应该分配设置的地方是在设置文件中。

# 安全

由于设置文件包含敏感信息，例如数据库密码，您应该尽一切努力限制对其的访问。例如，更改文件权限，以便只有您和您的 Web 服务器用户可以读取它。这在共享托管环境中尤为重要。

# 创建自己的设置

没有什么能阻止您为自己的 Django 应用程序创建自己的设置。只需遵循这些约定：

+   设置名称全部大写

+   不要重新发明已经存在的设置

对于序列的设置，Django 本身使用元组，而不是列表，但这只是一种约定。

# DJANGO_SETTINGS_MODULE

当您使用 Django 时，您必须告诉它您正在使用哪些设置。通过使用环境变量`DJANGO_SETTINGS_MODULE`来实现。`DJANGO_SETTINGS_MODULE`的值应该是 Python 路径语法，例如`mysite.settings`。

## django-admin 实用程序

在使用`django-admin`时，您可以设置环境变量一次，或者每次运行实用程序时显式传递设置模块。示例（Unix Bash shell）：

```py
export DJANGO_SETTINGS_MODULE=mysite.settings 
django-admin runserver

```

示例（Windows shell）：

```py
set DJANGO_SETTINGS_MODULE=mysite.settings 
django-admin runserver

```

使用`--settings`命令行参数手动指定设置：

```py
django-admin runserver --settings=mysite.settings

```

## 在服务器上（mod_wsgi）

在您的生产服务器环境中，您需要告诉您的 WSGI 应用程序使用哪个设置文件。使用`os.environ`来实现：

```py
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'

```

阅读第十三章，“部署 Django”，了解有关 Django WSGI 应用程序的更多信息和其他常见元素。

# 在没有设置 DJANGO_SETTINGS_MODULE 的情况下使用设置

在某些情况下，您可能希望绕过`DJANGO_SETTINGS_MODULE`环境变量。例如，如果您仅使用模板系统，您可能不希望设置一个指向设置模块的环境变量。在这些情况下，可以手动配置 Django 的设置。通过调用：

```py
django.conf.settings.configure(default_settings, **settings) 

```

例子：

```py
from django.conf import settings 
settings.configure(DEBUG=True, TEMPLATE_DEBUG=True) 

```

可以将`configure()`作为许多关键字参数传递，每个关键字参数表示一个设置及其值。每个参数名称应全部大写，与上述描述的设置名称相同。如果没有将特定设置传递给`configure()`并且在以后的某个时刻需要，Django 将使用默认设置值。

以这种方式配置 Django 在使用框架的一部分时通常是必要的，而且是推荐的。因此，当通过`settings.configure()`配置时，Django 不会对进程环境变量进行任何修改（请参阅`TIME_ZONE`的文档，了解为什么通常会发生这种情况）。在这些情况下，假设您已经完全控制了您的环境。

## 自定义默认设置

如果您希望默认值来自于`django.conf.global_settings`之外的某个地方，可以将提供默认设置的模块或类作为`configure()`调用中的`default_settings`参数（或作为第一个位置参数）传递。在此示例中，默认设置来自`myapp_defaults`，并且`DEBUG`设置为`True`，而不管它在`myapp_defaults`中的值是什么：

```py
from django.conf import settings 
from myapp import myapp_defaults 

settings.configure(default_settings=myapp_defaults, DEBUG=True) 

```

使用`myapp_defaults`作为位置参数的以下示例是等效的：

```py
settings.configure(myapp_defaults, DEBUG=True) 

```

通常，您不需要以这种方式覆盖默认设置。Django 的默认设置足够温和，您可以放心使用它们。请注意，如果您传入一个新的默认模块，它将完全*替换* Django 的默认设置，因此您必须为可能在导入的代码中使用的每个可能的设置指定一个值。在`django.conf.settings.global_settings`中查看完整列表。

## 要么 configure()，要么 DJANGO_SETTINGS_MODULE 是必需的

如果您没有设置`DJANGO_SETTINGS_MODULE`环境变量，则在使用读取设置的任何代码之前*必须*在某个时刻调用`configure()`。如果您没有设置`DJANGO_SETTINGS_MODULE`并且没有调用`configure()`，Django 将在第一次访问设置时引发`ImportError`异常。如果您设置了`DJANGO_SETTINGS_MODULE`，以某种方式访问设置值，*然后*调用`configure()`，Django 将引发`RuntimeError`，指示已经配置了设置。有一个专门用于此目的的属性：

```py
django.conf.settings.configured 

```

例如：

```py
from django.conf import settings 
if not settings.configured:
     settings.configure(myapp_defaults, DEBUG=True) 

```

此外，多次调用`configure()`或在访问任何设置之后调用`configure()`都是错误的。归根结底：只使用`configure()`或`DJANGO_SETTINGS_MODULE`中的一个。既不是两者，也不是两者都不是。

# 可用设置

Django 有大量可用的设置。为了方便参考，我将它们分成了六个部分，每个部分在本附录中都有相应的表。

+   核心设置（*表 D.1*）

+   身份验证设置（*表 D.2*）

+   消息设置（*表 D.3*）

+   会话设置（*表 D.4*）

+   Django 站点设置（*表 D.5*）

+   静态文件设置（*表 D.6*）

每个表列出了可用设置及其默认值。有关每个设置的附加信息和用例，请参阅 Django 项目网站[`docs.djangoproject.com/en/1.8/ref/settings/`](https://docs.djangoproject.com/en/1.8/ref/settings/)。

### 注意

在覆盖设置时要小心，特别是当默认值是非空列表或字典时，例如`MIDDLEWARE_CLASSES`和`STATICFILES_FINDERS`。确保保留 Django 所需的组件，以便使用您希望使用的 Django 功能。

## 核心设置

| **设置** | **默认值** |
| --- | --- |
| `ABSOLUTE_URL_OVERRIDES` | `{}` (空字典) |
| `ADMINS` | `[]` (空列表) |
| `ALLOWED_HOSTS` | `[]`（空列表） |
| `APPEND_SLASH` | `True` |
| `CACHE_MIDDLEWARE_ALIAS` | `default` |
| `CACHES` | `{ 'default': { 'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', } }` |
| `CACHE_MIDDLEWARE_KEY_PREFIX` | `''`（空字符串） |
| `CACHE_MIDDLEWARE_SECONDS` | `600` |
| `CSRF_COOKIE_AGE` | `31449600 (1 年，以秒计)` |
| `CSRF_COOKIE_DOMAIN` | `None` |
| `CSRF_COOKIE_HTTPONLY` | `False` |
| `CSRF_COOKIE_NAME` | `Csrftoken` |
| `CSRF_COOKIE_PATH` | `'/'` |
| `CSRF_COOKIE_SECURE` | `False` |
| `DATE_INPUT_FORMATS` | `[ '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%b %d %Y', '%b %d, %Y', '%d %b %Y','%d %b, %Y', '%B %d %Y', '%B %d, %Y', '%d %B %Y', '%d %B, %Y', ]` |
| `DATETIME_FORMAT` | `'N j, Y, P'（例如，Feb. 4, 2003, 4 p.m.）` |
| `DATETIME_INPUT_FORMATS` | `[ '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S.%f', '%m/%d/%Y %H:%M', '%m/%d/%Y', '%m/%d/%y %H:%M:%S',``'%m/%d/%y %H:%M:%S.%f', '%m/%d/%y %H:%M', '%m/%d/%y',``]` |
| `DEBUG` | `False` |
| `DEBUG_PROPAGATE_EXCEPTIONS` | `False` |
| `DECIMAL_SEPARATOR` | `'.'`（点） |
| `DEFAULT_CHARSET` | `'utf-8'` |
| `DEFAULT_CONTENT_TYPE` | `'text/html'` |
| `DEFAULT_EXCEPTION_REPORTER_FILTER` | `django.views.debug. SafeExceptionReporterFilter` |
| `DEFAULT_FILE_STORAGE` | `django.core.files.storage. FileSystemStorage` |
| `DEFAULT_FROM_EMAIL` | `'webmaster@localhost'.` |
| `DEFAULT_INDEX_TABLESPACE` | `''`（空字符串） |
| `DEFAULT_TABLESPACE` | `''`（空字符串） |
| `DISALLOWED_USER_AGENTS` | `[]`（空列表） |
| `EMAIL_BACKEND` | `django.core.mail.backends.smtp. EmailBackend` |
| `EMAIL_HOST` | `'localhost'` |
| `EMAIL_HOST_PASSWORD` | `''`（空字符串） |
| `EMAIL_HOST_USER` | `''`（空字符串） |
| `EMAIL_PORT` | `25` |
| `EMAIL_SUBJECT_PREFIX` | `'[Django] '` |
| `EMAIL_USE_TLS` | `False` |
| `EMAIL_USE_SSL` | `False` |
| `EMAIL_SSL_CERTFILE` | `None` |
| `EMAIL_SSL_KEYFILE` | `None` |
| `EMAIL_TIMEOUT` | `None` |
| `FILE_CHARSET` | `'utf-8'` |
| `FILE_UPLOAD_HANDLERS` | `[ 'django.core.files.uploadhandler.` `MemoryFileUploadHandler', 'django.core.files.uploadhandler. TemporaryFileUploadHandler' ]` |
| `FILE_UPLOAD_MAX_MEMORY_SIZE` | `2621440（即 2.5 MB）` |
| `FILE_UPLOAD_DIRECTORY_PERMISSIONS` | `None` |
| `FILE_UPLOAD_PERMISSIONS` | `None` |
| `FILE_UPLOAD_TEMP_DIR` | `None` |
| `FIRST_DAY_OF_WEEK` | `0`（星期日） |
| `FIXTURE_DIRS` | `[]`（空列表） |
| `FORCE_SCRIPT_NAME` | `None` |
| `FORMAT_MODULE_PATH` | `None` |
| `IGNORABLE_404_URLS` | `[]`（空列表） |
| `INSTALLED_APPS` | `[]`（空列表） |
| `INTERNAL_IPS` | `[]`（空列表） |
| `LANGUAGE_CODE` | `'en-us'` |
| `LANGUAGE_COOKIE_AGE` | `None`（在浏览器关闭时过期） |
| `LANGUAGE_COOKIE_DOMAIN` | `None` |
| `LANGUAGE_COOKIE_NAME` | `'django_language'` |
| `LANGUAGES` | 所有可用语言的列表 |
| `LOCALE_PATHS` | `[]`（空列表） |
| `LOGGING` | `一个日志配置字典` |
| `LOGGING_CONFIG` | `'logging.config.dictConfig'` |
| `MANAGERS` | `[]`（空列表） |
| `MEDIA_ROOT` | `''`（空字符串） |
| `MEDIA_URL` | `''`（空字符串） |
| `MIDDLEWARE_CLASSES` | `[ 'django.middleware.common. CommonMiddleware', 'django.middleware.csrf.  CsrfViewMiddleware' ]` |
| `MIGRATION_MODULES` | `{}`（空字典） |
| `MONTH_DAY_FORMAT` | `'F j'` |
| `NUMBER_GROUPING` | `0` |
| `PREPEND_WWW` | `False` |
| `ROOT_URLCONF` | 未定义 |
| `SECRET_KEY` | `''`（空字符串） |
| `SECURE_BROWSER_XSS_FILTER` | `False` |
| `SECURE_CONTENT_TYPE_NOSNIFF` | `False` |
| `SECURE_HSTS_INCLUDE_SUBDOMAINS` | `False` |
| `SECURE_HSTS_SECONDS` | `0` |
| `SECURE_PROXY_SSL_HEADER` | `None` |
| `SECURE_REDIRECT_EXEMPT` | `[]`（空列表） |
| `SECURE_SSL_HOST` | `None` |
| `SECURE_SSL_REDIRECT` | `False` |
| `SERIALIZATION_MODULES` | 未定义 |
| `SERVER_EMAIL` | `'root@localhost'` |
| `SHORT_DATE_FORMAT` | `m/d/Y`（例如，12/31/2003） |
| `SHORT_DATETIME_FORMAT` | `m/d/Y P`（例如，12/31/2003 4 p.m.） |
| `SIGNING_BACKEND` | `'django.core.signing.TimestampSigner'` |
| `SILENCED_SYSTEM_CHECKS` | `[]`（空列表） |
| `TEMPLATES` | `[]`（空列表） |
| `TEMPLATE_DEBUG` | `False` |
| `TEST_RUNNER` | `'django.test.runner.DiscoverRunner'` |
| `TEST_NON_SERIALIZED_APPS` | `[]`（空列表） |
| `THOUSAND_SEPARATOR` | `,（逗号）` |
| `TIME_FORMAT` | `'P'`（例如，下午 4 点） |
| `TIME_INPUT_FORMATS` | `[ '%H:%M:%S',``'%H:%M:%S.%f', '%H:%M',``]` |
| `TIME_ZONE` | `'America/Chicago'` |
| `USE_ETAGS` | `False` |
| `USE_I18N` | `True` |
| `USE_L10N` | `False` |
| `USE_THOUSAND_SEPARATOR` | `False` |
| `USE_TZ` | `False` |
| `USE_X_FORWARDED_HOST` | `False` |
| `WSGI_APPLICATION` | `None` |
| `YEAR_MONTH_FORMAT` | `'F Y'` |
| `X_FRAME_OPTIONS` | `'SAMEORIGIN'` |

表 D.1：Django 核心设置

## 认证

| **设置** | **默认值** |
| --- | --- |
| `AUTHENTICATION_BACKENDS` | `'django.contrib.auth.backends.ModelBackend'` |
| `AUTH_USER_MODEL` | `'auth.User'` |
| `LOGIN_REDIRECT_URL` | `'/accounts/profile/'` |
| `LOGIN_URL` | `'/accounts/login/'` |
| `LOGOUT_URL` | `'/accounts/logout/'` |
| `PASSWORD_RESET_TIMEOUT_DAYS` | `3` |
| `PASSWORD_HASHERS` | `[ 'django.contrib.auth.hashers.PBKDF2PasswordHasher', 'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher', 'django.contrib.auth.hashers.BCryptPasswordHasher', 'django.contrib.auth.hashers.SHA1PasswordHasher', 'django.contrib.auth.hashers.MD5PasswordHasher', 'django.contrib.auth.hashers.UnsaltedMD5PasswordHasher', 'django.contrib.auth.hashers.CryptPasswordHasher' ]` |

表 D.2：Django 身份验证设置

## 消息

| **设置** | **默认值** |
| --- | --- |
| `MESSAGE_LEVEL` | `messages` |
| `MESSAGE_STORAGE` | `'django.contrib.messages.storage.fallback.FallbackStorage'` |
| `MESSAGE_TAGS` | `{ messages.DEBUG: 'debug', messages.INFO: 'info', messages.SUCCESS: 'success', messages.WARNING: 'warning', messages.ERROR: 'error' }` |

表 D.3：Django 消息设置

## 会话

| **设置** | **默认值** |
| --- | --- |
| `SESSION_CACHE_ALIAS` | `default` |
| `SESSION_COOKIE_AGE` | `1209600`（2 周，以秒计）。 |
| `SESSION_COOKIE_DOMAIN` | `None` |
| `SESSION_COOKIE_HTTPONLY` | `True.` |
| `SESSION_COOKIE_NAME` | `'sessionid'` |
| `SESSION_COOKIE_PATH` | `'/'` |
| `SESSION_COOKIE_SECURE` | `False` |
| `SESSION_ENGINE` | `'django.contrib.sessions.backends.db'` |
| `SESSION_EXPIRE_AT_BROWSER_CLOSE` | `False` |
| `SESSION_FILE_PATH` | `None` |
| `SESSION_SAVE_EVERY_REQUEST` | `False` |
| `SESSION_SERIALIZER` | `'django.contrib.sessions.serializers. JSONSerializer'` |

表 D.4：Django 会话设置

## 站点

| **设置** | **默认值** |
| --- | --- |
| `SITE_ID` | `Not defined` |

表 D.5：Django 站点设置

## 静态文件

| **设置** | **默认值** |
| --- | --- |
| `STATIC_ROOT` | `None` |
| `STATIC_URL` | `None` |
| `STATICFILES_DIRS` | `[]`（空列表） |
| `STATICFILES_STORAGE` | `'django.contrib.staticfiles.storage.StaticFilesStorage'` |
| `STATICFILES_FINDERS` | `[``"django.contrib.staticfiles.finders.FileSystemFinder", "django.contrib.staticfiles.finders. AppDirectoriesFinder"``]` |

表 D.6：Django 静态文件设置
