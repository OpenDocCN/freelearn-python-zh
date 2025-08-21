# 第十八章：国际化

当从 JavaScript 源代码创建消息文件时，Django 最初是在美国中部开发的，字面上说，劳伦斯市距离美国大陆的地理中心不到 40 英里。然而，像大多数开源项目一样，Django 的社区逐渐包括来自全球各地的人。随着 Django 社区变得越来越多样化，*国际化*和*本地化*变得越来越重要。

Django 本身是完全国际化的；所有字符串都标记为可翻译，并且设置控制着像日期和时间这样的与区域相关的值的显示。Django 还附带了 50 多种不同的本地化文件。如果您不是以英语为母语，那么 Django 已经被翻译成您的主要语言的可能性很大。

用于这些本地化的相同国际化框架可供您在自己的代码和模板中使用。

因为许多开发人员对国际化和本地化的实际含义理解模糊，所以我们将从一些定义开始。

# 定义

## 国际化

指的是为任何区域的潜在使用设计程序的过程。这个过程通常由软件开发人员完成。国际化包括标记文本（如 UI 元素和错误消息）以供将来翻译，抽象显示日期和时间，以便可以遵守不同的本地标准，提供对不同时区的支持，并确保代码不包含对其用户位置的任何假设。您经常会看到国际化缩写为*I18N*。（18 指的是 I 和 N 之间省略的字母数）。

## 本地化

指的是实际将国际化程序翻译为特定区域的过程。这项工作通常由翻译人员完成。有时您会看到本地化缩写为*L10N*。

以下是一些其他术语，将帮助我们处理常见的语言：

### 区域名称

区域名称，可以是`ll`形式的语言规范，也可以是`ll_CC`形式的组合语言和国家规范。例如：`it`，`de_AT`，`es`，`pt_BR`。语言部分始终为小写，国家部分为大写。分隔符是下划线。

### 语言代码

表示语言的名称。浏览器使用这种格式在`Accept-Language` HTTP 标头中发送它们接受的语言名称。例如：`it`，`de-at`，`es`，`pt-br`。语言代码通常以小写表示，但 HTTP `Accept-Language`标头不区分大小写。分隔符是破折号。

### 消息文件

消息文件是一个纯文本文件，代表单一语言，包含所有可用的翻译字符串以及它们在给定语言中的表示方式。消息文件的文件扩展名为`.po`。

### 翻译字符串

可翻译的文字。

### 格式文件

格式文件是定义给定区域的数据格式的 Python 模块。

# 翻译

为了使 Django 项目可翻译，您必须在 Python 代码和模板中添加最少量的钩子。这些钩子称为翻译字符串。它们告诉 Django：如果该文本在该语言中有翻译，则应将此文本翻译成最终用户的语言。标记可翻译字符串是您的责任；系统只能翻译它知道的字符串。

然后 Django 提供了工具来提取翻译字符串到消息文件中。这个文件是翻译人员以目标语言提供翻译字符串的方便方式。一旦翻译人员填写了消息文件，就必须对其进行编译。这个过程依赖 GNU `gettext`工具集。

完成后，Django 会根据用户的语言偏好即时翻译 Web 应用程序。

基本上，Django 做了两件事：

+   它允许开发人员和模板作者指定其应用程序的哪些部分应该是可翻译的。

+   它使用这些信息根据用户的语言偏好来翻译 Web 应用程序。

Django 的国际化钩子默认打开，这意味着在框架的某些地方有一些与 i18n 相关的开销。如果您不使用国际化，您应该花两秒钟在设置文件中设置`USE_I18N = False`。然后 Django 将进行一些优化，以便不加载国际化机制，这将节省一些开销。还有一个独立但相关的`USE_L10N`设置，用于控制 Django 是否应该实现格式本地化。

# 国际化：在 Python 代码中

## 标准翻译

使用函数`ugettext（）`指定翻译字符串。按照惯例，将其导入为更短的别名`_`，以节省输入。

Python 的标准库`gettext`模块将`_（）`安装到全局命名空间中，作为`gettext（）`的别名。在 Django 中，出于几个原因，我们选择不遵循这种做法：

+   对于国际字符集（Unicode）支持，`ugettext（）`比`gettext（）`更有用。有时，您应该使用`ugettext_lazy（）`作为特定文件的默认翻译方法。在全局命名空间中没有`_（）`时，开发人员必须考虑哪个是最合适的翻译函数。

+   下划线字符（`_`）用于表示 Python 交互式 shell 和 doctest 测试中的先前结果。安装全局`_（）`函数会导致干扰。显式导入`ugettext（）`作为`_（）`可以避免这个问题。

在这个例子中，文本“欢迎来到我的网站。”被标记为翻译字符串：

```py
from django.utils.translation import ugettext as _ 
from django.http import HttpResponse 

def my_view(request): 
    output = _("Welcome to my site.") 
    return HttpResponse(output) 

```

显然，您可以在不使用别名的情况下编写此代码。这个例子与前一个例子相同：

```py
from django.utils.translation import ugettext 
from django.http import HttpResponse 

def my_view(request): 
    output = ugettext("Welcome to my site.") 
    return HttpResponse(output) 

```

翻译也适用于计算值。这个例子与前两个相同：

```py
def my_view(request): 
    words = ['Welcome', 'to', 'my', 'site.'] 
    output = _(' '.join(words)) 
    return HttpResponse(output) 

```

...和变量。再次，这是一个相同的例子：

```py
def my_view(request): 
    sentence = 'Welcome to my site.' 
    output = _(sentence) 
    return HttpResponse(output) 

```

（与前两个示例中使用变量或计算值的警告是，Django 的翻译字符串检测实用程序`django-admin makemessages`将无法找到这些字符串。稍后再讨论`makemessages`。）

您传递给`_（）`或`ugettext（）`的字符串可以使用 Python 的标准命名字符串插值语法指定占位符。示例：

```py
def my_view(request, m, d): 
    output = _('Today is %(month)s %(day)s.') % {'month': m, 'day': d} 
    return HttpResponse(output) 

```

这种技术允许特定语言的翻译重新排列占位符文本。例如，英语翻译可能是“今天是 11 月 26 日。”，而西班牙语翻译可能是“Hoy es 26 de Noviembre。”-月份和日期占位符交换了位置。

因此，当您有多个参数时，应使用命名字符串插值（例如`%(day)s`）而不是位置插值（例如`%s`或`%d`）。如果使用位置插值，翻译将无法重新排列占位符文本。

## 翻译者注释

如果您想给翻译者有关可翻译字符串的提示，可以在前一行添加一个以`Translators`关键字为前缀的注释，例如：

```py
def my_view(request): 
    # Translators: This message appears on the home page only 
    output = ugettext("Welcome to my site.") 

```

该注释将出现在与其下方的可翻译结构相关联的生成的`.po`文件中，并且大多数翻译工具也应该显示该注释。

只是为了完整起见，这是生成的`.po`文件的相应片段：

```py
#. Translators: This message appears on the home page only 
# path/to/python/file.py:123 
msgid "Welcome to my site." 
msgstr "" 

```

这也适用于模板。有关更多详细信息，请参见模板中的翻译注释。

## 标记字符串为 No-Op

使用函数`django.utils.translation.ugettext_noop（）`将字符串标记为翻译字符串而不进行翻译。稍后从变量中翻译字符串。

如果您有应存储在源语言中的常量字符串，因为它们在系统或用户之间交换-例如数据库中的字符串-但应在最后可能的时间点进行翻译，例如在向用户呈现字符串时，请使用此功能。

## 复数形式

使用函数`django.utils.translation.ungettext()`来指定复数形式的消息。

`ungettext`需要三个参数：单数翻译字符串、复数翻译字符串和对象的数量。

当您的 Django 应用程序需要本地化到复数形式比英语中使用的两种形式更多的语言时，此功能非常有用（'object'表示单数，'objects'表示`count`与 1 不同的所有情况，而不考虑其值。）

例如：

```py
from django.utils.translation import ungettext 
from django.http import HttpResponse 

def hello_world(request, count): 
    page = ungettext( 
        'there is %(count)d object', 
        'there are %(count)d objects', 
    count) % { 
        'count': count, 
    } 
    return HttpResponse(page) 

```

在此示例中，对象的数量作为`count`变量传递给翻译语言。

请注意，复数形式很复杂，并且在每种语言中的工作方式都不同。将`count`与 1 进行比较并不总是正确的规则。这段代码看起来很复杂，但对于某些语言来说会产生错误的结果：

```py
from django.utils.translation import ungettext 
from myapp.models import Report 

count = Report.objects.count() 
if count == 1: 
    name = Report._meta.verbose_name 
else: 
    name = Report._meta.verbose_name_plural 

text = ungettext( 
    'There is %(count)d %(name)s available.', 
    'There are %(count)d %(name)s available.', 
    count 
    ) % { 
      'count': count, 
      'name': name 
    } 

```

不要尝试实现自己的单数或复数逻辑，这是不正确的。在这种情况下，考虑以下内容：

```py
text = ungettext( 
    'There is %(count)d %(name)s object available.', 
    'There are %(count)d %(name)s objects available.', 
    count 
    ) % { 
      'count': count, 
      'name': Report._meta.verbose_name, 
    } 

```

使用`ungettext()`时，请确保在文字中包含的每个外推变量使用单个名称。在上面的示例中，请注意我们如何在两个翻译字符串中都使用了`name` Python 变量。这个示例，除了如上所述在某些语言中是不正确的，还会失败：

```py
text = ungettext( 
    'There is %(count)d %(name)s available.', 
    'There are %(count)d %(plural_name)s available.', 
    count 
    ) % { 
      'count': Report.objects.count(), 
      'name': Report._meta.verbose_name, 
      'plural_name': Report._meta.verbose_name_plural 
    } 

```

运行`django-admin compilemessages`时会出现错误：

```py
a format specification for argument 'name', as in 'msgstr[0]', doesn't exist in 'msgid' 

```

## 上下文标记

有时单词有几个含义，例如英语中的*May*，它既指月份名称又指动词。为了使翻译人员能够在不同的上下文中正确翻译这些单词，您可以使用`django.utils.translation.pgettext()`函数，或者如果字符串需要复数形式，则使用`django.utils.translation.npgettext()`函数。两者都将上下文字符串作为第一个变量。

在生成的`.po`文件中，该字符串将出现的次数与相同字符串的不同上下文标记一样多（上下文将出现在`msgctxt`行上），允许翻译人员为每个上下文标记提供不同的翻译。

例如：

```py
from django.utils.translation import pgettext 

month = pgettext("month name", "May") 

```

或：

```py
from django.db import models 
from django.utils.translation import pgettext_lazy 

class MyThing(models.Model): 
    name = models.CharField(help_text=pgettext_lazy( 
        'help text for MyThing model', 'This is the help text')) 

```

将出现在`.po`文件中：

```py
msgctxt "month name" 
msgid "May" 
msgstr "" 

```

上下文标记也受`trans`和`blocktrans`模板标记的支持。

## 延迟翻译

在`django.utils.translation`中使用翻译函数的延迟版本（通过它们的名称中的`lazy`后缀很容易识别）来延迟翻译字符串-当访问值而不是在调用它们时。

这些函数存储字符串的延迟引用-而不是实际的翻译。当字符串在字符串上下文中使用时（例如在模板渲染中），翻译本身将在最后可能的时间点进行。

当这些函数的调用位于模块加载时执行的代码路径中时，这是必不可少的。

这很容易发生在定义模型、表单和模型表单时，因为 Django 实现了这些，使得它们的字段实际上是类级属性。因此，在以下情况下，请确保使用延迟翻译。

### 模型字段和关系

例如，要翻译以下模型中*name*字段的帮助文本，请执行以下操作：

```py
from django.db import models 
from django.utils.translation import ugettext_lazy as _ 

class MyThing(models.Model): 
    name = models.CharField(help_text=_('This is the help text')) 

```

您可以通过使用它们的`verbose_name`选项将`ForeignKey`，`ManyToManyField`或`OneToOneField`关系的名称标记为可翻译：

```py
class MyThing(models.Model): 
    kind = models.ForeignKey(ThingKind, related_name='kinds',  verbose_name=_('kind')) 

```

就像您在`verbose_name`中所做的那样，当需要时，应为关系提供一个小写的详细名称文本，Django 将在需要时自动将其转换为标题大小写。

### 模型详细名称值

建议始终提供明确的`verbose_name`和`verbose_name_plural`选项，而不是依赖于 Django 通过查看模型类名执行的后备英语中心且有些天真的决定详细名称：

```py
from django.db import models 
from django.utils.translation import ugettext_lazy as _ 

class MyThing(models.Model): 
    name = models.CharField(_('name'), help_text=_('This is the help  text')) 

    class Meta: 
        verbose_name = _('my thing') 
        verbose_name_plural = _('my things') 

```

### 模型方法的`short_description`属性值

对于模型方法，你可以使用`short_description`属性为 Django 和管理站点提供翻译：

```py
from django.db import models 
from django.utils.translation import ugettext_lazy as _ 

class MyThing(models.Model): 
    kind = models.ForeignKey(ThingKind, related_name='kinds', 
                             verbose_name=_('kind')) 

    def is_mouse(self): 
        return self.kind.type == MOUSE_TYPE 
        is_mouse.short_description = _('Is it a mouse?') 

```

## 使用延迟翻译对象

`ugettext_lazy()`调用的结果可以在 Python 中任何需要使用 Unicode 字符串（类型为`unicode`的对象）的地方使用。如果你试图在需要字节字符串（`str`对象）的地方使用它，事情将不会按预期工作，因为`ugettext_lazy()`对象不知道如何将自己转换为字节字符串。你也不能在字节字符串中使用 Unicode 字符串，因此这与正常的 Python 行为一致。例如：

```py
# This is fine: putting a unicode proxy into a unicode string. 
"Hello %s" % ugettext_lazy("people") 

# This will not work, since you cannot insert a unicode object 
# into a bytestring (nor can you insert our unicode proxy there) 
b"Hello %s" % ugettext_lazy("people") 

```

如果你看到类似`"hello <django.utils.functional...>"`的输出，你尝试将`ugettext_lazy()`的结果插入到字节字符串中。这是你代码中的一个错误。

如果你不喜欢长长的`ugettext_lazy`名称，你可以将其别名为`_`（下划线），就像这样：

```py
from django.db import models 
from django.utils.translation import ugettext_lazy as _ 

class MyThing(models.Model): 
    name = models.CharField(help_text=_('This is the help text')) 

```

在模型和实用函数中使用`ugettext_lazy()`和`ungettext_lazy()`标记字符串是一个常见的操作。当你在代码的其他地方使用这些对象时，你应该确保不要意外地将它们转换为字符串，因为它们应该尽可能晚地转换（以便正确的区域设置生效）。这就需要使用下面描述的辅助函数。

### 延迟翻译和复数

当使用延迟翻译来处理复数字符串（`[u]n[p]gettext_lazy`）时，通常在字符串定义时不知道`number`参数。因此，你可以授权将一个键名而不是整数作为`number`参数传递。然后在字符串插值期间，`number`将在字典中查找该键下的值。这里有一个例子：

```py
from django import forms 
from django.utils.translation import ugettext_lazy 

class MyForm(forms.Form): 
    error_message = ungettext_lazy("You only provided %(num)d    
      argument", "You only provided %(num)d arguments", 'num') 

    def clean(self): 
        # ... 
        if error: 
            raise forms.ValidationError(self.error_message %  
              {'num': number}) 

```

如果字符串只包含一个未命名的占位符，你可以直接使用`number`参数进行插值：

```py
class MyForm(forms.Form): 
    error_message = ungettext_lazy("You provided %d argument", 
        "You provided %d arguments") 

    def clean(self): 
        # ... 
        if error: 
            raise forms.ValidationError(self.error_message % number) 

```

### 连接字符串：string_concat()

标准的 Python 字符串连接（`''.join([...])`）在包含延迟翻译对象的列表上不起作用。相反，你可以使用`django.utils.translation.string_concat()`，它创建一个延迟对象，只有在结果包含在字符串中时才将其内容连接并转换为字符串。例如：

```py
from django.utils.translation import string_concat 
from django.utils.translation import ugettext_lazy 
# ... 
name = ugettext_lazy('John Lennon') 
instrument = ugettext_lazy('guitar') 
result = string_concat(name, ': ', instrument) 

```

在这种情况下，`result`中的延迟翻译只有在`result`本身在字符串中使用时才会转换为字符串（通常在模板渲染时）。

### 延迟翻译的其他用途

对于任何其他需要延迟翻译的情况，但必须将可翻译的字符串作为参数传递给另一个函数，你可以自己在延迟调用内部包装这个函数。例如：

```py
from django.utils import six  # Python 3 compatibility 
from django.utils.functional import lazy 
from django.utils.safestring import mark_safe 
from django.utils.translation import ugettext_lazy as _ 

mark_safe_lazy = lazy(mark_safe, six.text_type) 

```

然后稍后：

```py
lazy_string = mark_safe_lazy(_("<p>My <strong>string!</strong></p>")) 

```

## 语言的本地化名称

`get_language_info()`函数提供了关于语言的详细信息：

```py
>>> from django.utils.translation import get_language_info 
>>> li = get_language_info('de') 
>>> print(li['name'], li['name_local'], li['bidi']) 
German Deutsch False 

```

字典的`name`和`name_local`属性包含了语言的英文名称和该语言本身的名称。`bidi`属性仅对双向语言为 True。

语言信息的来源是`django.conf.locale`模块。类似的访问这些信息的方式也适用于模板代码。见下文。

# 国际化：在模板代码中

Django 模板中的翻译使用了两个模板标签和与 Python 代码略有不同的语法。为了让你的模板可以访问这些标签，将

在你的模板顶部使用`{% load i18n %}`。与所有模板标签一样，这个标签需要在使用翻译的所有模板中加载，即使是那些从已经加载了`i18n`标签的其他模板继承的模板也是如此。

## trans 模板标签

`{% trans %}`模板标签可以翻译常量字符串（用单引号或双引号括起来）或变量内容：

```py
<title>{% trans "This is the title." %}</title> 
<title>{% trans myvar %}</title> 

```

如果存在`noop`选项，变量查找仍然会发生，但翻译会被跳过。这在需要将来进行翻译的内容中是有用的：

```py
<title>{% trans "myvar" noop %}</title> 

```

在内部，内联翻译使用了`ugettext()`调用。

如果将模板变量（如上面的 `myvar`）传递给标签，则标签将首先在运行时将该变量解析为字符串，然后在消息目录中查找该字符串。

不可能在 `{% trans %}` 内部的字符串中混合模板变量。如果您的翻译需要带有变量（占位符）的字符串，请改用 `{% blocktrans %}`。如果您想要检索翻译后的字符串而不显示它，可以使用以下语法：

```py
{% trans "This is the title" as the_title %} 

```

在实践中，您将使用此功能来获取在多个地方使用的字符串，或者应该用作其他模板标签或过滤器的参数：

```py
{% trans "starting point" as start %} 
{% trans "end point" as end %} 
{% trans "La Grande Boucle" as race %} 

<h1> 
  <a href="/" >{{ race }}</a> 
</h1> 
<p> 
{% for stage in tour_stages %} 
    {% cycle start end %}: {{ stage }}{% if forloop.counter|divisibleby:2 %}<br />{% else %}, {% endif %} 
{% endfor %} 
</p> 

```

`{% trans %}` 也支持使用 `context` 关键字进行上下文标记：

```py
{% trans "May" context "month name" %} 

```

## blocktrans 模板标签

`blocktrans` 标签允许您通过使用占位符标记由文字和变量内容组成的复杂句子进行翻译。

```py
{% blocktrans %}This string will have {{ value }} inside.{% endblocktrans %} 

```

要翻译模板表达式，比如访问对象属性或使用模板过滤器，您需要将表达式绑定到本地变量，以便在翻译块内使用。例如：

```py
{% blocktrans with amount=article.price %} 
That will cost $ {{ amount }}. 
{% endblocktrans %} 

{% blocktrans with myvar=value|filter %} 
This will have {{ myvar }} inside. 
{% endblocktrans %} 

```

您可以在单个 `blocktrans` 标签内使用多个表达式：

```py
{% blocktrans with book_t=book|title author_t=author|title %} 
This is {{ book_t }} by {{ author_t }} 
{% endblocktrans %} 

```

仍然支持以前更冗长的格式：`{% blocktrans with book|title as book_t and author|title as author_t %}`

不允许在 `blocktrans` 标签内部使用其他块标签（例如 `{% for %}` 或 `{% if %}`）。

如果解析其中一个块参数失败，`blocktrans` 将通过使用 `deactivate_all()` 函数临时停用当前活动的语言来回退到默认语言。

此标签还提供了复数形式。使用方法如下：

+   指定并绑定名为 `count` 的计数器值。此值将用于选择正确的复数形式。

+   使用两种形式分隔单数和复数形式

+   `{% plural %}` 标签在 `{% blocktrans %}` 和 `{% endblocktrans %}` 标签内。

一个例子：

```py
{% blocktrans count counter=list|length %} 
There is only one {{ name }} object. 
{% plural %} 
There are {{ counter }} {{ name }} objects. 
{% endblocktrans %} 

```

一个更复杂的例子：

```py
{% blocktrans with amount=article.price count years=i.length %} 
That will cost $ {{ amount }} per year. 
{% plural %} 
That will cost $ {{ amount }} per {{ years }} years. 
{% endblocktrans %} 

```

当您同时使用复数形式功能并将值绑定到本地变量以及计数器值时，请记住 `blocktrans` 结构在内部转换为 `ungettext` 调用。这意味着与 `ungettext` 变量相关的相同注释也适用。

不能在 `blocktrans` 内部进行反向 URL 查找，应该事先检索（和存储）：

```py
{% url 'path.to.view' arg arg2 as the_url %} 
{% blocktrans %} 
This is a URL: {{ the_url }} 
{% endblocktrans %} 

```

`{% blocktrans %}` 还支持使用 `context` 关键字进行上下文标记：

```py
{% blocktrans with name=user.username context "greeting" %} 
Hi {{ name }}{% endblocktrans %} 

```

`{% blocktrans %}` 支持的另一个功能是 `trimmed` 选项。此选项将从 `{% blocktrans %}` 标签的内容开头和结尾删除换行符，替换行开头和结尾的任何空格，并使用空格字符将所有行合并成一行。

这对于缩进 `{% blocktrans %}` 标签的内容而不使缩进字符出现在 PO 文件中的相应条目中非常有用，这样可以使翻译过程更加简单。

例如，以下 `{% blocktrans %}` 标签：

```py
{% blocktrans trimmed %} 
  First sentence. 
  Second paragraph. 
{% endblocktrans %} 

```

如果未指定 `trimmed` 选项，将在 PO 文件中生成条目 `"First sentence. Second paragraph."`，而不是 `"\n First sentence.\n Second sentence.\n"`。

## 传递给标签和过滤器的字符串文字

您可以使用熟悉的 `_()` 语法将作为参数传递给标签和过滤器的字符串文字进行翻译：

```py
{% some_tag _("Page not found") value|yesno:_("yes,no") %} 

```

在这种情况下，标签和过滤器都将看到翻译后的字符串，因此它们不需要知道翻译。

在此示例中，翻译基础设施将传递字符串 "`yes,no`"，而不是单独的字符串 "`yes`" 和 "`no`"。翻译后的字符串需要包含逗号，以便过滤器解析代码知道如何分割参数。例如，德语翻译者可能将字符串 "`yes,no`" 翻译为 "`ja,nein`"（保持逗号不变）。

## 模板中的翻译者注释

与 Python 代码一样，这些翻译者注释可以使用注释指定，可以使用 `comment` 标签：

```py
{% comment %}Translators: View verb{% endcomment %} 
{% trans "View" %} 

{% comment %}Translators: Short intro blurb{% endcomment %} 
<p>{% blocktrans %} 
    A multiline translatable literal. 
   {% endblocktrans %} 
</p> 

```

或者使用 `{#` ... `#}` 单行注释结构：

```py
{# Translators: Label of a button that triggers search #} 
<button type="submit">{% trans "Go" %}</button> 

{# Translators: This is a text of the base template #} 
{% blocktrans %}Ambiguous translatable block of text{% endblocktrans %} 

```

仅供完整性，这些是生成的`.po`文件的相应片段：

```py
#. Translators: View verb 
# path/to/template/file.html:10 
msgid "View" 
msgstr "" 

#. Translators: Short intro blurb 
# path/to/template/file.html:13 
msgid "" 
"A multiline translatable" 
"literal." 
msgstr "" 

# ... 

#. Translators: Label of a button that triggers search 
# path/to/template/file.html:100 
msgid "Go" 
msgstr "" 

#. Translators: This is a text of the base template 
# path/to/template/file.html:103 
msgid "Ambiguous translatable block of text" 
msgstr "" 

```

## 在模板中切换语言

如果要在模板中选择语言，则可以使用`language`模板标签：

```py
{% load i18n %} 

{% get_current_language as LANGUAGE_CODE %} 
<!-- Current language: {{ LANGUAGE_CODE }} --> 
<p>{% trans "Welcome to our page" %}</p> 

{% language 'en' %} 

    {% get_current_language as LANGUAGE_CODE %} 
    <!-- Current language: {{ LANGUAGE_CODE }} --> 
    <p>{% trans "Welcome to our page" %}</p> 

{% endlanguage %} 

```

虽然欢迎来到我们的页面的第一次出现使用当前语言，但第二次将始终是英语。

## 其他标签

这些标签还需要`{% load i18n %}`。

+   `{% get_available_languages as LANGUAGES %}`返回一个元组列表，其中第一个元素是语言代码，第二个是语言名称（翻译为当前活动的区域设置）。

+   `{% get_current_language as LANGUAGE_CODE %}`返回当前用户的首选语言，作为字符串。例如：`en-us`。（请参见本章后面的*django 如何发现语言偏好*。）

+   `{% get_current_language_bidi as LANGUAGE_BIDI %}`返回当前区域设置的方向。如果为 True，则是从右到左的语言，例如希伯来语，阿拉伯语。如果为 False，则是从左到右的语言，例如英语，法语，德语等。

如果启用了`django.template.context_processors.i18n`上下文处理器，则每个`RequestContext`将可以访问`LANGUAGES`，`LANGUAGE_CODE`和`LANGUAGE_BIDI`，如上所定义。

对于新项目，默认情况下不会为`i18n`上下文处理器启用。

您还可以使用提供的模板标签和过滤器检索有关任何可用语言的信息。要获取有关单个语言的信息，请使用`{% get_language_info %}`标签：

```py
{% get_language_info for LANGUAGE_CODE as lang %} 
{% get_language_info for "pl" as lang %} 

```

然后您可以访问这些信息：

```py
Language code: {{ lang.code }}<br /> 
Name of language: {{ lang.name_local }}<br /> 
Name in English: {{ lang.name }}<br /> 
Bi-directional: {{ lang.bidi }} 

```

您还可以使用`{% get_language_info_list %}`模板标签来检索语言列表的信息（例如在`LANGUAGES`中指定的活动语言）。请参阅关于`set_language`重定向视图的部分，了解如何使用`{% get_language_info_list %}`显示语言选择器的示例。

除了`LANGUAGES`风格的元组列表外，`{% get_language_info_list %}`还支持简单的语言代码列表。如果在视图中这样做：

```py
context = {'available_languages': ['en', 'es', 'fr']} 
return render(request, 'mytemplate.html', context) 

```

您可以在模板中迭代这些语言：

```py
{% get_language_info_list for available_languages as langs %} 
{% for lang in langs %} ... {% endfor %} 

```

还有一些简单的过滤器可供使用：

+   `{{ LANGUAGE_CODE|language_name }}`（德语）

+   `{{ LANGUAGE_CODE|language_name_local }}`（德语）

+   `{{ LANGUAGE_CODE|language_bidi }}` (False)

# 国际化：在 JavaScript 代码中

向 JavaScript 添加翻译会带来一些问题：

+   JavaScript 代码无法访问`gettext`实现。

+   JavaScript 代码无法访问`.po`或`.mo`文件；它们需要由服务器传送。

+   JavaScript 的翻译目录应尽可能保持小。

Django 为这些问题提供了一个集成的解决方案：它将翻译传递到 JavaScript 中，因此您可以在 JavaScript 中调用`gettext`等。

## javascript_catalog 视图

这些问题的主要解决方案是`django.views.i18n.javascript_catalog()`视图，它发送一个 JavaScript 代码库，其中包含模仿`gettext`接口的函数，以及一个翻译字符串数组。

这些翻译字符串是根据您在`info_dict`或 URL 中指定的内容来自应用程序或 Django 核心。`LOCALE_PATHS`中列出的路径也包括在内。

您可以这样连接它：

```py
from django.views.i18n import javascript_catalog 

js_info_dict = { 
    'packages': ('your.app.package',), 
} 

urlpatterns = [ 
    url(r'^jsi18n/$', javascript_catalog, js_info_dict), 
] 

```

`packages`中的每个字符串都应该是 Python 点分包语法（与`INSTALLED_APPS`中的字符串格式相同），并且应该引用包含`locale`目录的包。如果指定多个包，所有这些目录都将合并为一个目录。如果您的 JavaScript 使用来自不同应用程序的字符串，则这很有用。

翻译的优先级是这样的，`packages`参数中后面出现的包比出现在开头的包具有更高的优先级，这在相同文字的冲突翻译的情况下很重要。

默认情况下，视图使用`djangojs` `gettext`域。这可以通过修改`domain`参数来更改。

您可以通过将包放入 URL 模式中使视图动态化：

```py
urlpatterns = [ 
    url(r'^jsi18n/(?P<packages>\S+?)/$', javascript_catalog), 
] 

```

通过这种方式，您可以将包作为 URL 中由`+`符号分隔的包名称列表指定。如果您的页面使用来自不同应用的代码，并且这些代码经常更改，您不希望拉入一个大的目录文件，这将特别有用。作为安全措施，这些值只能是`django.conf`或`INSTALLED_APPS`设置中的任何包。

`LOCALE_PATHS`设置中列出的路径中找到的 JavaScript 翻译也总是包含在内。为了保持与用于 Python 和模板的翻译查找顺序算法的一致性，`LOCALE_PATHS`中列出的目录具有最高的优先级，先出现的目录比后出现的目录具有更高的优先级。

## 使用 JavaScript 翻译目录

要使用目录，只需像这样拉入动态生成的脚本：

```py
<script type="text/javascript" src="img/{% url  'django.views.i18n.javascript_catalog' %}"></script> 

```

这使用了反向 URL 查找来查找 JavaScript 目录视图的 URL。加载目录时，您的 JavaScript 代码可以使用标准的`gettext`接口来访问它：

```py
document.write(gettext('this is to be translated')); 

```

还有一个`ngettext`接口：

```py
var object_cnt = 1 // or 0, or 2, or 3, ... 
s = ngettext('literal for the singular case', 
      'literal for the plural case', object_cnt); 

```

甚至还有一个字符串插值函数：

```py
function interpolate(fmt, obj, named); 

```

插值语法是从 Python 借来的，因此`interpolate`函数支持位置和命名插值：

+   位置插值：`obj`包含一个 JavaScript 数组对象，其元素值然后按照它们出现的顺序依次插值到相应的`fmt`占位符中。例如：

```py
        fmts = ngettext('There is %s object. Remaining: %s', 
                 'There are %s objects. Remaining: %s', 11); 
        s = interpolate(fmts, [11, 20]); 
        // s is 'There are 11 objects. Remaining: 20' 

```

+   命名插值：通过将可选的布尔命名参数设置为 true 来选择此模式。`obj`包含一个 JavaScript 对象或关联数组。例如：

```py
        d = { 
            count: 10, 
            total: 50 
        }; 

        fmts = ngettext('Total: %(total)s, there is %(count)s  
          object', 
          'there are %(count)s of a total of %(total)s objects', 
            d.count); 
        s = interpolate(fmts, d, true); 

```

不过，您不应该过度使用字符串插值：这仍然是 JavaScript，因此代码必须进行重复的正则表达式替换。这不像 Python 中的字符串插值那样快，因此只在您真正需要它的情况下使用它（例如，与`ngettext`一起产生正确的复数形式）。

## 性能说明

`javascript_catalog()`视图会在每次请求时从`.mo`文件生成目录。由于它的输出是恒定的-至少对于站点的特定版本来说-它是一个很好的缓存候选者。

服务器端缓存将减少 CPU 负载。可以使用`cache_page()`装饰器轻松实现。要在翻译更改时触发缓存失效，请提供一个版本相关的键前缀，如下例所示，或者将视图映射到一个版本相关的 URL。

```py
from django.views.decorators.cache import cache_page 
from django.views.i18n import javascript_catalog 

# The value returned by get_version() must change when translations change. 
@cache_page(86400, key_prefix='js18n-%s' % get_version()) 
def cached_javascript_catalog(request, domain='djangojs', packages=None): 
    return javascript_catalog(request, domain, packages) 

```

客户端缓存将节省带宽并使您的站点加载更快。如果您使用 ETags（`USE_ETAGS = True`），则已经覆盖了。否则，您可以应用条件装饰器。在下面的示例中，每当重新启动应用程序服务器时，缓存就会失效。

```py
from django.utils import timezone 
from django.views.decorators.http import last_modified 
from django.views.i18n import javascript_catalog 

last_modified_date = timezone.now() 

@last_modified(lambda req, **kw: last_modified_date) 
def cached_javascript_catalog(request, domain='djangojs', packages=None): 
    return javascript_catalog(request, domain, packages) 

```

您甚至可以在部署过程的一部分预先生成 JavaScript 目录，并将其作为静态文件提供。[`django-statici18n.readthedocs.org/en/latest/`](http://django-statici18n.readthedocs.org/en/latest/)。

# 国际化：在 URL 模式中

Django 提供了两种国际化 URL 模式的机制：

+   将语言前缀添加到 URL 模式的根部，以便`LocaleMiddleware`可以从请求的 URL 中检测要激活的语言。

+   通过`django.utils.translation.ugettext_lazy()`函数使 URL 模式本身可翻译。

使用这些功能中的任何一个都需要为每个请求设置一个活动语言；换句话说，您需要在`MIDDLEWARE_CLASSES`设置中拥有`django.middleware.locale.LocaleMiddleware`。

## URL 模式中的语言前缀

这个函数可以在您的根 URLconf 中使用，Django 将自动将当前活动语言代码添加到`i18n_patterns()`中定义的所有 URL 模式之前。示例 URL 模式：

```py
from django.conf.urls import include, url 
from django.conf.urls.i18n import i18n_patterns 
from about import views as about_views 
from news import views as news_views 
from sitemap.views import sitemap 

urlpatterns = [ 
    url(r'^sitemap\.xml$', sitemap, name='sitemap_xml'), 
] 

news_patterns = [ 
    url(r'^$', news_views.index, name='index'), 
    url(r'^category/(?P<slug>[\w-]+)/$',  
        news_views.category, 
        name='category'), 
    url(r'^(?P<slug>[\w-]+)/$', news_views.details, name='detail'), 
] 

urlpatterns += i18n_patterns( 
    url(r'^about/$', about_views.main, name='about'), 
    url(r'^news/', include(news_patterns, namespace='news')), 
) 

```

定义这些 URL 模式后，Django 将自动将语言前缀添加到由`i18n_patterns`函数添加的 URL 模式。例如：

```py
from django.core.urlresolvers import reverse 
from django.utils.translation import activate 

>>> activate('en') 
>>> reverse('sitemap_xml') 
'/sitemap.xml' 
>>> reverse('news:index') 
'/en/news/' 

>>> activate('nl') 
>>> reverse('news:detail', kwargs={'slug': 'news-slug'}) 
'/nl/news/news-slug/' 

```

`i18n_patterns()`只允许在根 URLconf 中使用。在包含的 URLconf 中使用它将引发`ImproperlyConfigured`异常。

## 翻译 URL 模式

URL 模式也可以使用`ugettext_lazy()`函数进行标记翻译。例如：

```py
from django.conf.urls import include, url 
from django.conf.urls.i18n import i18n_patterns 
from django.utils.translation import ugettext_lazy as _ 

from about import views as about_views 
from news import views as news_views 
from sitemaps.views import sitemap 

urlpatterns = [ 
    url(r'^sitemap\.xml$', sitemap, name='sitemap_xml'), 
] 

news_patterns = [ 
    url(r'^$', news_views.index, name='index'), 
    url(_(r'^category/(?P<slug>[\w-]+)/$'),  
        news_views.category, 
        name='category'), 
    url(r'^(?P<slug>[\w-]+)/$', news_views.details, name='detail'), 
] 

urlpatterns += i18n_patterns( 
    url(_(r'^about/$'), about_views.main, name='about'), 
    url(_(r'^news/'), include(news_patterns, namespace='news')), 
) 

```

创建了翻译后，`reverse()`函数将返回活动语言的 URL。例如：

```py
>>> from django.core.urlresolvers import reverse 
>>> from django.utils.translation import activate 

>>> activate('en') 
>>> reverse('news:category', kwargs={'slug': 'recent'}) 
'/en/news/category/recent/' 

>>> activate('nl') 
>>> reverse('news:category', kwargs={'slug': 'recent'}) 
'/nl/nieuws/categorie/recent/' 

```

在大多数情况下，最好只在语言代码前缀的模式块中使用翻译后的 URL（使用`i18n_patterns()`），以避免疏忽翻译的 URL 导致与未翻译的 URL 模式发生冲突的可能性。

## 在模板中进行反向操作

如果在模板中反转了本地化的 URL，它们将始终使用当前语言。要链接到另一种语言的 URL，请使用`language`模板标签。它在封闭的模板部分中启用给定的语言：

```py
{% load i18n %} 

{% get_available_languages as languages %} 

{% trans "View this category in:" %} 
{% for lang_code, lang_name in languages %} 
    {% language lang_code %} 
    <a href="{% url 'category' slug=category.slug %}">{{ lang_name }}</a> 
    {% endlanguage %} 
{% endfor %} 

```

`language`标签期望语言代码作为唯一参数。

# 本地化：如何创建语言文件

一旦应用程序的字符串文字被标记为以后进行翻译，翻译本身需要被编写（或获取）。下面是它的工作原理。

## 消息文件

第一步是为新语言创建一个消息文件。消息文件是一个纯文本文件，代表单一语言，包含所有可用的翻译字符串以及它们在给定语言中的表示方式。消息文件具有`.po`文件扩展名。

Django 附带了一个工具`django-admin makemessages`，它可以自动创建和维护这些文件。

`makemessages`命令（以及稍后讨论的`compilemessages`）使用 GNU `gettext`工具集中的命令：`xgettext`、`msgfmt`、`msgmerge`和`msguniq`。

支持的`gettext`实用程序的最低版本为 0.15。

要创建或更新消息文件，请运行此命令：

```py
django-admin makemessages -l de 

```

...其中`de`是要创建的消息文件的区域名称。例如，`pt_BR`表示巴西葡萄牙语，`de_AT`表示奥地利德语，`id`表示印尼语。

该脚本应该从以下两个地方之一运行：

+   您的 Django 项目的根目录（包含`manage.py`的目录）。

+   您的 Django 应用程序之一的根目录。

该脚本在项目源树或应用程序源树上运行，并提取所有标记为翻译的字符串（请参阅 how-django-discovers-translations 并确保`LOCALE_PATHS`已正确配置）。它在目录`locale/LANG/LC_MESSAGES`中创建（或更新）一个消息文件。在`de`的示例中，文件将是`locale/de/LC_MESSAGES/django.po`。

当您从项目的根目录运行`makemessages`时，提取的字符串将自动分发到适当的消息文件中。也就是说，从包含`locale`目录的应用程序文件中提取的字符串将放在该目录下的消息文件中。从不包含任何`locale`目录的应用程序文件中提取的字符串将放在`LOCALE_PATHS`中列出的第一个目录下的消息文件中，如果`LOCALE_PATHS`为空，则会生成错误。

默认情况下，`django-admin makemessages`检查具有`.html`或`.txt`文件扩展名的每个文件。如果要覆盖默认设置，请使用`-extension`或`-e`选项指定要检查的文件扩展名：

```py
django-admin makemessages -l de -e txt 

```

用逗号分隔多个扩展名和/或多次使用`-e`或`-extension`：

```py
django-admin makemessages -l de -e html,txt -e xml 

```

### 注意

从 JavaScript 源代码创建消息文件时，需要使用特殊的'djangojs'域，而不是`e js`。

如果您没有安装`gettext`实用程序，`makemessages`将创建空文件。如果是这种情况，要么安装`gettext`实用程序，要么只需复制英文消息文件（`locale/en/LC_MESSAGES/django.po`）（如果有的话）并将其用作起点；它只是一个空的翻译文件。

如果您使用 Windows 并且需要安装 GNU `gettext`实用程序以便`makemessages`正常工作，请参阅本章稍后的*在 Windows 上使用 gettext*以获取更多信息。

`.po`文件的格式很简单。每个`.po`文件包含一小部分元数据，例如翻译维护者的联系信息，但文件的大部分是*消息*的列表-翻译字符串和特定语言的实际翻译文本之间的简单映射。

例如，如果您的 Django 应用程序包含了文本`"欢迎来到我的网站。"`的翻译字符串，如下所示：

```py
_("Welcome to my site.") 

```

然后`django-admin makemessages`将创建一个包含以下片段消息的`.po`文件：

```py
#: path/to/python/module.py:23 
msgid "Welcome to my site." 
msgstr "" 

```

一个简单的解释：

+   `msgid`是出现在源中的翻译字符串。不要更改它。

+   `msgstr`是您放置特定于语言的翻译的地方。它起初是空的，所以您有责任更改它。确保您在翻译周围保留引号。

+   为了方便起见，每条消息都包括一个以`#`为前缀的注释行，位于`msgid`行上方，其中包含了翻译字符串所在的文件名和行号。

长消息是一个特殊情况。在那里，`msgstr`（或`msgid`）之后的第一个字符串是一个空字符串。然后内容本身将作为下面几行的一个字符串写入。这些字符串直接连接在一起。不要忘记字符串内的尾随空格；否则，它们将被连接在一起而没有空格！

由于`gettext`工具的内部工作方式，以及我们希望允许 Django 核心和您的应用程序中的非 ASCII 源字符串，您必须将 UTF-8 用作 PO 文件的编码（创建 PO 文件时的默认值）。这意味着每个人都将使用相同的编码，在 Django 处理 PO 文件时这一点很重要。

要重新检查所有源代码和模板以获取新的翻译字符串，并为所有语言更新所有消息文件，请运行以下命令：

```py
django-admin makemessages -a 

```

## 编译消息文件

创建消息文件后，每次对其进行更改时，您都需要将其编译为`gettext`可以使用的更高效的形式。使用`django-admin compilemessages`实用程序进行此操作。

此工具将遍历所有可用的`.po`文件，并创建`.mo`文件，这些文件是为`gettext`使用而优化的二进制文件。在您运行`django-admin makemessages`的同一目录中运行：

```py
django-admin compilemessages 

```

就是这样。您的翻译已经准备好了。

如果您使用 Windows 并且需要安装 GNU `gettext`实用程序以使`django-admin compilemessages`正常工作，请参阅下面有关 Windows 上的`gettext`的更多信息。

Django 仅支持以 UTF-8 编码且没有任何 BOM（字节顺序标记）的`.po`文件，因此如果您的文本编辑器默认在文件开头添加这些标记，那么您需要重新配置它。

## 从 JavaScript 源代码创建消息文件

您可以像其他 Django 消息文件一样使用`django-admin makemessages`工具创建和更新消息文件。唯一的区别是，您需要显式指定在这种情况下称为`djangojs`域的`gettext`术语中的域，通过提供一个`-d djangojs`参数，就像这样：

```py
django-admin makemessages -d djangojs -l de 

```

这将为德语创建或更新 JavaScript 的消息文件。更新消息文件后，只需像处理普通 Django 消息文件一样运行`django-admin compilemessages`。

## Windows 上的 gettext

这仅适用于那些想要提取消息 ID 或编译消息文件（`.po`）的人。翻译工作本身只涉及编辑这种类型的现有文件，但如果您想创建自己的消息文件，或者想测试或编译已更改的消息文件，您将需要`gettext`实用程序：

+   从 GNOME 服务器（[`download.gnome.org/binaries/win32/dependencies/`](https://download.gnome.org/binaries/win32/dependencies/)）下载以下 zip 文件

+   `gettext-runtime-X.zip`

+   `gettext-tools-X.zip`

`X`是版本号；需要版本`0.15`或更高版本。

+   将这两个文件夹中`bin\`目录的内容提取到系统上的同一个文件夹中（即`C:\Program Files\gettext-utils`）

+   更新系统 PATH：

+   `控制面板 > 系统 > 高级 > 环境变量`。

+   在`系统变量`列表中，点击`Path`，点击`Edit`。

+   在`Variable value`字段的末尾添加`;C:\Program Files\gettext-utils\bin`。

您也可以使用其他地方获取的`gettext`二进制文件，只要`xgettext -version`命令正常工作。如果在 Windows 命令提示符中输入`xgettext -version`命令会弹出一个窗口说 xgettext.exe 已经生成错误并将被 Windows 关闭，请不要尝试使用 Django 翻译工具与`gettext`包。

## 自定义 makemessages 命令

如果您想向`xgettext`传递额外的参数，您需要创建一个自定义的`makemessages`命令并覆盖其`xgettext_options`属性：

```py
from django.core.management.commands import makemessages 

class Command(makemessages.Command): 
    xgettext_options = makemessages.Command.xgettext_options +  
      ['-keyword=mytrans'] 

```

如果您需要更灵活性，您还可以向自定义的`makemessages`命令添加一个新参数：

```py
from django.core.management.commands import makemessages 

class Command(makemessages.Command): 

    def add_arguments(self, parser): 
        super(Command, self).add_arguments(parser) 
        parser.add_argument('-extra-keyword', 
                            dest='xgettext_keywords',  
                            action='append') 

    def handle(self, *args, **options): 
        xgettext_keywords = options.pop('xgettext_keywords') 
        if xgettext_keywords: 
            self.xgettext_options = ( 
                makemessages.Command.xgettext_options[:] + 
                ['-keyword=%s' % kwd for kwd in xgettext_keywords] 
            ) 
        super(Command, self).handle(*args, **options) 

```

# 显式设置活动语言

您可能希望明确为当前会话设置活动语言。也许用户的语言偏好是从另一个系统中检索的。例如，您已经介绍了`django.utils.translation.activate()`。这仅适用于当前线程。要使语言在整个会话中持续存在，还要修改会话中的`LANGUAGE_SESSION_KEY`：

```py
from django.utils import translation 
user_language = 'fr' 
translation.activate(user_language) 
request.session[translation.LANGUAGE_SESSION_KEY] = user_language 

```

通常您希望同时使用：`django.utils.translation.activate()`将更改此线程的语言，并修改会话使此偏好在将来的请求中持续存在。

如果您不使用会话，语言将保留在一个 cookie 中，其名称在`LANGUAGE_COOKIE_NAME`中配置。例如：

```py
from django.utils import translation 
from django import http 
from django.conf import settings 
user_language = 'fr' 
translation.activate(user_language) 
response = http.HttpResponse(...) 
response.set_cookie(settings.LANGUAGE_COOKIE_NAME, user_language) 

```

# 在视图和模板之外使用翻译

虽然 Django 提供了丰富的国际化工具供视图和模板使用，但它并不限制使用于 Django 特定的代码。Django 的翻译机制可以用于将任意文本翻译成 Django 支持的任何语言（当然，前提是存在适当的翻译目录）。

您可以加载一个翻译目录，激活它并将文本翻译成您选择的语言，但请记住切换回原始语言，因为激活翻译目录是基于每个线程的，这样的更改将影响在同一线程中运行的代码。

例如：

```py
from django.utils import translation 
def welcome_translated(language): 
    cur_language = translation.get_language() 
    try: 
        translation.activate(language) 
        text = translation.ugettext('welcome') 
    finally: 
        translation.activate(cur_language) 
    return text 

```

使用值'de'调用此函数将给您"`Willkommen`"，而不管`LANGUAGE_CODE`和中间件设置的语言如何。

特别感兴趣的功能是`django.utils.translation.get_language()`，它返回当前线程中使用的语言，`django.utils.translation.activate()`，它激活当前线程的翻译目录，以及`django.utils.translation.check_for_language()`，它检查给定的语言是否受 Django 支持。

# 实现说明

## Django 翻译的特点

Django 的翻译机制使用了 Python 自带的标准`gettext`模块。如果您了解`gettext`，您可能会注意到 Django 在翻译方面的一些特点：

+   字符串域是`django`或`djangojs`。这个字符串域用于区分存储其数据在一个共同的消息文件库中的不同程序（通常是`/usr/share/locale/`）。`django`域用于 Python 和模板翻译字符串，并加载到全局翻译目录中。`djangojs`域仅用于 JavaScript 翻译目录，以确保其尽可能小。

+   Django 不仅仅使用`xgettext`。它使用围绕`xgettext`和`msgfmt`的 Python 包装器。这主要是为了方便。

## Django 如何发现语言偏好

一旦您准备好您的翻译，或者如果您只想使用 Django 提供的翻译，您需要为您的应用程序激活翻译。

在幕后，Django 有一个非常灵活的模型来决定应该使用哪种语言-全局安装、特定用户或两者。

要设置全局安装的语言偏好，请设置`LANGUAGE_CODE`。Django 将使用此语言作为默认翻译-如果通过区域设置中间件采用的方法找不到更好的匹配翻译，则作为最后一次尝试。

如果您只想使用本地语言运行 Django，您只需要设置`LANGUAGE_CODE`并确保相应的消息文件及其编译版本（`.mo`）存在。

如果要让每个用户指定他们喜欢的语言，那么您还需要使用`LocaleMiddleware`。`LocaleMiddleware`基于请求中的数据启用语言选择。它为每个用户定制内容。

要使用`LocaleMiddleware`，请将`'django.middleware.locale.LocaleMiddleware'`添加到您的`MIDDLEWARE_CLASSES`设置中。因为中间件顺序很重要，所以您应该遵循以下准则：

+   确保它是最先安装的中间件之一。

+   它应该放在`SessionMiddleware`之后，因为`LocaleMiddleware`使用会话数据。它应该放在`CommonMiddleware`之前，因为`CommonMiddleware`需要激活的语言来解析请求的 URL。

+   如果使用`CacheMiddleware`，请在其后放置`LocaleMiddleware`。

例如，您的`MIDDLEWARE_CLASSES`可能如下所示：

```py
MIDDLEWARE_CLASSES = [ 
   'django.contrib.sessions.middleware.SessionMiddleware', 
   'django.middleware.locale.LocaleMiddleware', 
   'django.middleware.common.CommonMiddleware', 
] 

```

有关中间件的更多信息，请参见第十七章，*Django 中间件*。

`LocaleMiddleware`尝试通过以下算法确定用户的语言偏好：

+   首先，它会在请求的 URL 中查找语言前缀。只有在您的根 URLconf 中使用`i18n_patterns`函数时才会执行此操作。有关语言前缀以及如何国际化 URL 模式的更多信息，请参见*国际化*。

+   如果失败，它会查找当前用户会话中的`LANGUAGE_SESSION_KEY`键。

+   如果失败，它会查找一个 cookie。使用的 cookie 的名称由`LANGUAGE_COOKIE_NAME`设置。 （默认名称是`django_language`。）

+   如果失败，它会查看`Accept-Language` HTTP 标头。此标头由您的浏览器发送，并告诉服务器您首选的语言（按优先级顺序）。Django 尝试标头中的每种语言，直到找到具有可用翻译的语言。

+   ***** 如果失败，它会使用全局`LANGUAGE_CODE`设置。

**注意：**

+   在这些地方中，语言偏好应该是标准语言格式的字符串。例如，巴西葡萄牙语是`pt-br`。

+   如果基本语言可用但未指定子语言，则 Django 将使用基本语言。例如，如果用户指定`de-at`（奥地利德语），但 Django 只有`de`可用，Django 将使用`de`。

+   只有在`LANGUAGES`设置中列出的语言才能被选择。如果要将语言选择限制为提供的语言的子集（因为您的应用程序没有提供所有这些语言），请将`LANGUAGES`设置为语言列表。例如：

```py

        LANGUAGES = [ 
          ('de', _('German')), 
          ('en', _('English')), 
        ] 

```

此示例将可用于自动选择的语言限制为德语和英语（以及任何子语言，如`de-ch`或`en-us`）。

+   如果您定义了自定义的`LANGUAGES`设置，如前面的项目所述，您可以将语言名称标记为翻译字符串-但使用`ugettext_lazy()`而不是`ugettext()`以避免循环导入。

这里有一个示例设置文件：

```py
from django.utils.translation import ugettext_lazy as _ 

LANGUAGES = [ 
    ('de', _('German')), 
    ('en', _('English')), 
] 

```

一旦`LocaleMiddleware`确定了用户的偏好，它会将这个偏好作为`request.LANGUAGE_CODE`对每个`HttpRequest`可用。请随意在您的视图代码中读取这个值。这里有一个简单的例子：

```py
from django.http import HttpResponse 

def hello_world(request, count): 
    if request.LANGUAGE_CODE == 'de-at': 
        return HttpResponse("You prefer to read Austrian German.") 
    else: 
        return HttpResponse("You prefer to read another language.") 

```

请注意，对于静态（无中间件）翻译，语言在`settings.LANGUAGE_CODE`中，而对于动态（中间件）翻译，它在`request.LANGUAGE_CODE`中。

## Django 如何发现翻译

在运行时，Django 会构建一个内存中的统一的文字翻译目录。为了实现这一点，它会按照一定的顺序查找不同文件路径来加载编译好的消息文件（`.mo`），并确定同一文字的多个翻译的优先级。

+   在`LOCALE_PATHS`中列出的目录具有最高的优先级，出现在前面的优先级高于后面的。

+   然后，它会查找并使用（如果存在）每个已安装应用程序中的`INSTALLED_APPS`列表中的`locale`目录。出现在前面的优先级高于后面的。

+   最后，Django 提供的基础翻译在`django/conf/locale`中被用作后备。

在所有情况下，包含翻译的目录的名称应该使用语言环境的命名规范。例如，`de`，`pt_BR`，`es_AR`等。

通过这种方式，您可以编写包含自己翻译的应用程序，并且可以覆盖项目中的基础翻译。或者，您可以构建一个由多个应用程序组成的大型项目，并将所有翻译放入一个特定于您正在组合的项目的大型共同消息文件中。选择权在您手中。

所有消息文件存储库的结构都是相同的。它们是：

+   在您的设置文件中列出的`LOCALE_PATHS`中搜索`<language>/LC_MESSAGES/django.(po|mo)`

+   `$APPPATH/locale/<language>/LC_MESSAGES/django.(po|mo)`

+   `$PYTHONPATH/django/conf/locale/<language>/LC_MESSAGES/django.(po|mo).`

要创建消息文件，您可以使用`django-admin makemessages`工具。您可以使用`django-admin compilemessages`来生成二进制的`.mo`文件，这些文件将被`gettext`使用。

您还可以运行`django-admin compilemessages`来使编译器处理`LOCALE_PATHS`设置中的所有目录。

# 接下来是什么？

在下一章中，我们将讨论 Django 中的安全性。
