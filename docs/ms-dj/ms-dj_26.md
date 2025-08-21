# 附录 E. 内置模板标签和过滤器

第三章, *模板*, 列出了一些最有用的内置模板标签和过滤器。但是，Django 还附带了许多其他内置标签和过滤器。本附录提供了 Django 中所有模板标签和过滤器的摘要。有关更详细的信息和用例，请参见 Django 项目网站 [`docs.djangoproject.com/en/1.8/ref/templates/builtins/`](https://docs.djangoproject.com/en/1.8/ref/templates/builtins/)。

# 内置标签

## autoescape

控制当前自动转义行为。此标签接受 `on` 或 `off` 作为参数，决定块内是否生效自动转义。块以 `endautoescape` 结束标签关闭。

当自动转义生效时，所有变量内容在放入输出结果之前都会应用 HTML 转义（但在应用任何过滤器之后）。这相当于手动对每个变量应用 `escape` 过滤器。

唯一的例外是已经标记为不需要转义的变量，要么是由填充变量的代码标记的，要么是因为已经应用了 `safe` 或 `escape` 过滤器。示例用法：

```py
{% autoescape on %} 
    {{ body }} 
{% endautoescape %} 

```

## block

定义一个可以被子模板覆盖的块。有关更多信息，请参见 第三章, *模板*, 中的 "模板继承"。

## comment

忽略 `{% comment %}` 和 `{% endcomment %}` 之间的所有内容。第一个标签中可以插入一个可选的注释。例如，当注释掉代码以记录为什么禁用代码时，这是有用的。

`Comment` 标签不能嵌套。

## csrf_token

此标签用于 CSRF 保护。有关 **跨站点请求伪造** (**CSRF**) 的更多信息，请参见 第三章, *模板*, 和 第十九章, *Django 中的安全性*。

## cycle

每次遇到此标签时，产生其中的一个参数。第一次遇到时产生第一个参数，第二次遇到时产生第二个参数，依此类推。一旦所有参数用完，标签就会循环到第一个参数并再次产生它。这个标签在循环中特别有用：

```py
{% for o in some_list %} 
    <tr class="{% cycle 'row1' 'row2' %}"> 
        ... 
    </tr> 
{% endfor %} 

```

第一次迭代生成引用 `row1` 类的 HTML，第二次生成 `row2`，第三次再次生成 `row1`，依此类推。您也可以使用变量。例如，如果有两个模板变量 `rowvalue1` 和 `rowvalue2`，您可以像这样在它们的值之间交替：

```py
{% for o in some_list %} 
    <tr class="{% cycle rowvalue1 rowvalue2 %}"> 
        ... 
    </tr> 
{% endfor %} 

```

您还可以混合变量和字符串：

```py
{% for o in some_list %} 
    <tr class="{% cycle 'row1' rowvalue2 'row3' %}"> 
        ... 
    </tr> 
{% endfor %} 

```

您可以在 `cycle` 标签中使用任意数量的值，用空格分隔。用单引号 (`'`) 或双引号 (`"`) 括起来的值被视为字符串字面量，而没有引号的值被视为模板变量。

## debug

输出大量的调试信息，包括当前上下文和导入的模块。

## extends

表示此模板扩展了父模板。此标签可以以两种方式使用：

+   `{% extends "base.html" %}`（带引号）使用字面值 `"base.html"` 作为要扩展的父模板的名称。

+   `{% extends variable %}` 使用 `variable` 的值。如果变量求值为字符串，Django 将使用该字符串作为父模板的名称。如果变量求值为 `Template` 对象，Django 将使用该对象作为父模板。

## filter

通过一个或多个过滤器过滤块的内容。有关 Django 中过滤器的列表，请参见附录后面的内置过滤器部分。

## firstof

输出第一个不是 `False` 的参数变量。如果所有传递的变量都是 `False`，则不输出任何内容。示例用法：

```py
{% firstof var1 var2 var3 %} 

```

这相当于：

```py
{% if var1 %} 
    {{ var1 }} 
{% elif var2 %} 
    {{ var2 }} 
{% elif var3 %} 
    {{ var3 }} 
{% endif %} 

```

## for

在数组中循环每个项目，使项目在上下文变量中可用。例如，要显示提供的 `athlete_list` 中的运动员列表：

```py
<ul> 
{% for athlete in athlete_list %} 
    <li>{{ athlete.name }}</li> 
{% endfor %} 
</ul> 

```

您可以通过使用`{% for obj in list reversed %}`在列表上进行反向循环。如果需要循环遍历一个列表的列表，可以将每个子列表中的值解压缩为单独的变量。如果需要访问字典中的项目，这也可能很有用。例如，如果您的上下文包含一个名为`data`的字典，则以下内容将显示字典的键和值：

```py
{% for key, value in data.items %} 
    {{ key }}: {{ value }} 
{% endfor %} 

```

## for... empty

`for`标签可以带一个可选的`{% empty %}`子句，如果给定的数组为空或找不到，则显示其文本：

```py
<ul> 
{% for athlete in athlete_list %} 
    <li>{{ athlete.name }}</li> 
{% empty %} 
    <li>Sorry, no athletes in this list.</li> 
{% endfor %} 
</ul> 

```

## 如果

`{% if %}`标签评估一个变量，如果该变量为真（即存在，不为空，并且不是 false 布尔值），则输出块的内容：

```py
{% if athlete_list %} 
    Number of athletes: {{ athlete_list|length }} 
{% elif athlete_in_locker_room_list %} 
    Athletes should be out of the locker room soon! 
{% else %} 
    No athletes. 
{% endif %} 

```

在上面的例子中，如果`athlete_list`不为空，则将通过`{{ athlete_list|length }}`变量显示运动员的数量。正如您所看到的，`if`标签可以带一个或多个`{% elif %}`子句，以及一个`{% else %}`子句，如果所有先前的条件都失败，则将显示该子句。这些子句是可选的。

### 布尔运算符

`if`标签可以使用`and`、`or`或`not`来测试多个变量或否定给定变量：

```py
{% if athlete_list and coach_list %} 
    Both athletes and coaches are available. 
{% endif %} 

{% if not athlete_list %} 
    There are no athletes. 
{% endif %} 

{% if athlete_list or coach_list %} 
    There are some athletes or some coaches. 
{% endif %} 

```

在同一个标签中使用`and`和`or`子句是允许的，例如，`and`的优先级高于`or`：

```py
{% if athlete_list and coach_list or cheerleader_list %} 

```

将被解释为：

```py
if (athlete_list and coach_list) or cheerleader_list 

```

在`if`标签中使用实际括号是无效的语法。如果需要它们来表示优先级，应该使用嵌套的`if`标签。

`if`标签也可以使用`==`、`!=`、`<`、`>`、`<=`、`>=`和`in`运算符，其工作方式如*表 E.1*中所列。

| 运算符 | 示例 |
| --- | --- |
| == | {% if somevar == "x" %} ... |
| != | {% if somevar != "x" %} ... |
| < | {% if somevar < 100 %} ... |
| > | {% if somevar > 10 %} ... |
| <= | {% if somevar <= 100 %} ... |
| >= | {% if somevar >= 10 %} ... |
| In | {% if "bc" in "abcdef" %} |

表 E.1：模板标签中的布尔运算符

### 复杂表达式

所有上述内容都可以组合成复杂的表达式。对于这样的表达式，了解在评估表达式时运算符是如何分组的可能很重要，即优先级规则。运算符的优先级从低到高依次为：

+   `or`

+   `and`

+   `not`

+   `in`

+   `==`、`!=`、`<`、`>`、`<=`和`>=`

这个优先顺序与 Python 完全一致。

### 过滤器

您还可以在`if`表达式中使用过滤器。例如：

```py
{% if messages|length >= 100 %} 
   You have lots of messages today! 
{% endif %} 

```

## ifchanged

检查值是否与循环的上一次迭代不同。

`{% ifchanged %}`块标签在循环内使用。它有两种可能的用法：

+   检查其自身的渲染内容与其先前状态是否不同，仅在内容发生变化时显示内容

+   如果给定一个或多个变量，检查任何变量是否发生了变化

## ifequal

如果两个参数相等，则输出块的内容。示例：

```py
{% ifequal user.pk comment.user_id %} 
    ... 
{% endifequal %} 

```

`ifequal`标签的替代方法是使用`if`标签和`==`运算符。

## ifnotequal

与`ifequal`类似，只是它测试两个参数是否不相等。使用`ifnotequal`标签的替代方法是使用`if`标签和`!=`运算符。

## 包括

加载模板并使用当前上下文进行渲染。这是在模板中包含其他模板的一种方式。模板名称可以是一个变量：

```py
{% include template_name %} 

```

或硬编码（带引号）的字符串：

```py
{% include "foo/bar.html" %} 

```

## 加载

加载自定义模板标签集。例如，以下模板将加载`somelibrary`和`otherlibrary`中注册的所有标签和过滤器，这些库位于`package`包中：

```py
{% load somelibrary package.otherlibrary %} 

```

您还可以使用`from`参数从库中选择性地加载单个过滤器或标签。

在这个例子中，模板标签/过滤器`foo`和`bar`将从`somelibrary`中加载：

```py
{% load foo bar from somelibrary %} 

```

有关更多信息，请参阅*自定义标签*和*过滤器库*。

## lorem

显示随机的 lorem ipsum 拉丁文。这对于在模板中提供示例数据很有用。用法：

```py
{% lorem [count] [method] [random] %} 

```

`{% lorem %}`标签可以使用零个、一个、两个或三个参数。这些参数是：

+   **计数：**生成段落或单词的数量（默认为 1）的数字（或变量）。

+   **方法：**单词的 w，HTML 段落的 p 或纯文本段落块的 b（默认为 b）。

+   **随机：**单词随机，如果给定，则在生成文本时不使用常见段落（Lorem ipsum dolor sit amet...）。

例如，`{% lorem 2 w random %}`将输出两个随机拉丁单词。

## now

显示当前日期和/或时间，使用与给定字符串相符的格式。该字符串可以包含格式说明符字符，如`date`过滤器部分所述。例如：

```py
It is {% now "jS F Y H:i" %} 

```

传递的格式也可以是预定义的格式之一`DATE_FORMAT`、`DATETIME_FORMAT`、`SHORT_DATE_FORMAT`或`SHORT_DATETIME_FORMAT`。预定义的格式可能会根据当前区域设置和格式本地化的启用情况而有所不同，例如：

```py
It is {% now "SHORT_DATETIME_FORMAT" %} 

```

## regroup

通过共同属性对类似对象的列表进行重新分组。

`{% regroup %}`生成*组对象*的列表。每个组对象有两个属性：

+   `grouper`：按其共同属性进行分组的项目（例如，字符串 India 或 Japan）

+   `list`：此组中所有项目的列表（例如，所有`country = "India"`的城市列表）

请注意，`{% regroup %}`不会对其输入进行排序！

任何有效的模板查找都是`regroup`标记的合法分组属性，包括方法、属性、字典键和列表项。

## spaceless

删除 HTML 标签之间的空格。这包括制表符和换行符。例如用法：

```py
{% spaceless %} 
    <p> 
        <a href="foo/">Foo</a> 
    </p> 
{% endspaceless %} 

```

此示例将返回此 HTML：

```py
<p><a href="foo/">Foo</a></p> 

```

## templatetag

输出用于组成模板标记的语法字符之一。由于模板系统没有转义的概念，因此要显示模板标记中使用的位之一，必须使用`{% templatetag %}`标记。参数告诉要输出哪个模板位：

+   `openblock`输出：`{%`

+   `closeblock`输出：`%}`

+   `openvariable`输出：`{{`

+   `closevariable`输出：`}}`

+   `openbrace`输出：`{`

+   `closebrace`输出：`}`

+   `opencomment`输出：`{#`

+   `closecomment`输出：`#}`

示例用法：

```py
{% templatetag openblock %} url 'entry_list' {% templatetag closeblock %} 

```

## url

返回与给定视图函数和可选参数匹配的绝对路径引用（不包括域名的 URL）。结果路径中的任何特殊字符都将使用`iri_to_uri()`进行编码。这是一种在模板中输出链接的方法，而不违反 DRY 原则，因为不必在模板中硬编码 URL：

```py
{% url 'some-url-name' v1 v2 %} 

```

第一个参数是视图函数的路径，格式为`package.package.module.function`。它可以是带引号的文字或任何其他上下文变量。其他参数是可选的，应该是用空格分隔的值，这些值将用作 URL 中的参数。

## verbatim

阻止模板引擎渲染此块标记的内容。常见用途是允许与 Django 语法冲突的 JavaScript 模板层。

## widthratio

用于创建条形图等，此标记计算给定值与最大值的比率，然后将该比率应用于常数。例如：

```py
<img src="img/bar.png" alt="Bar" 
     height="10" width="{% widthratio this_value max_value max_width %}" /> 

```

## with

将复杂变量缓存到更简单的名称下。在多次访问昂贵的方法（例如，多次访问数据库的方法）时很有用。例如：

```py
{% with total=business.employees.count %} 
    {{ total }} employee{{ total|pluralize }} 
{% endwith %} 

```

# 内置过滤器

## add

将参数添加到值。例如：

```py
{{ value|add:"2" }} 

```

如果`value`是`4`，则输出将是`6`。

## addslashes

在引号前添加斜杠。例如，在 CSV 中转义字符串很有用。例如：

```py
{{ value|addslashes }} 

```

如果`value`是`I'm using Django`，输出将是`I'm using Django`。

## capfirst

将值的第一个字符大写。如果第一个字符不是字母，则此过滤器无效。

## center

将值居中在给定宽度的字段中。例如：

```py
"{{ value|center:"14" }}" 

```

如果`value`是`Django`，输出将是`Django`。

## cut

从给定字符串中删除所有`arg`的值。

## date

根据给定的格式格式化日期。使用与 PHP 的`date()`函数类似的格式，但有一些不同之处。

### 注意

这些格式字符在 Django 模板之外不使用。它们旨在与 PHP 兼容，以便设计人员更轻松地过渡。有关格式字符串的完整列表，请参见 Django 项目网站[`docs.djangoproject.com/en/dev/ref/templates/builtins/#date`](https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date)。

例如：

```py
{{ value|date:"D d M Y" }} 

```

如果`value`是`datetime`对象（例如，`datetime.datetime.now()`的结果），输出将是字符串`Fri 01 Jul 2016`。传递的格式可以是预定义的`DATE_FORMAT`、`DATETIME_FORMAT`、`SHORT_DATE_FORMAT`或`SHORT_DATETIME_FORMAT`之一，也可以是使用日期格式说明符的自定义格式。

## 默认

如果值评估为`False`，则使用给定的默认值。否则，使用该值。例如：

```py
{{ value|default:"nothing" }}     

```

## default_if_none

如果（且仅当）值为`None`，则使用给定的默认值。否则，使用该值。

## dictsort

接受一个字典列表并返回按参数中给定的键排序的列表。例如：

```py
{{ value|dictsort:"name" }} 

```

## dictsortreversed

接受一个字典列表并返回按参数中给定的键的相反顺序排序的列表。

## 可被整除

如果值可以被参数整除，则返回`True`。例如：

```py
{{ value|divisibleby:"3" }} 

```

如果`value`是`21`，输出将是`True`。

## 转义

转义字符串的 HTML。具体来说，它进行以下替换：

+   `<`转换为`&lt;`

+   `>`转换为`&gt;`

+   `'`（单引号）转换为`'`

+   `"`（双引号）转换为`&quot;`

+   `&`转换为`&amp;`

转义仅在输出字符串时应用，因此不管在过滤器的链式序列中放置`escape`的位置如何：它始终会被应用，就好像它是最后一个过滤器一样。

## escapejs

转义用于 JavaScript 字符串。这并*不*使字符串在 HTML 中安全使用，但可以保护您免受在使用模板生成 JavaScript/JSON 时的语法错误。

## filesizeformat

格式化值，如“人类可读”的文件大小（即`'13 KB'`、`'4.1 MB'`、`'102 bytes'`等）。例如：

```py
{{ value|filesizeformat }} 

```

如果`value`是`123456789`，输出将是`117.7 MB`。

## 第一

返回列表中的第一项。

## floatformat

在没有参数的情况下使用时，将浮点数四舍五入到小数点后一位，但只有在有小数部分要显示时才会这样做。如果与数字整数参数一起使用，`floatformat`将将数字四舍五入到该小数位数。

例如，如果`value`是`34.23234`，`{{ value|floatformat:3 }}`将输出`34.232`。

## get_digit

给定一个整数，返回请求的数字，其中 1 是最右边的数字。

## iriencode

将**国际化资源标识符**（**IRI**）转换为适合包含在 URL 中的字符串。

## join

使用字符串将列表连接起来，就像 Python 的`str.join(list)`一样。

## 最后

返回列表中的最后一项。

## 长度

返回值的长度。这适用于字符串和列表。

## length_is

如果值的长度是参数，则返回`True`，否则返回`False`。例如：

```py
{{ value|length_is:"4" }} 

```

## linebreaks

用适当的 HTML 替换纯文本中的换行符；单个换行符变成 HTML 换行符（`<br />`），换行符后面跟着一个空行变成段落换行符（`</p>`）。

## linebreaksbr

将纯文本中的所有换行符转换为 HTML 换行符（`<br />`）。

## 行号

显示带有行号的文本。

## ljust

将值左对齐在给定宽度的字段中。例如：

```py
{{ value|ljust:"10" }} 

```

如果`value`是`Django`，输出将是`Django`。

## lower

将字符串转换为全部小写。

## make_list

返回转换为列表的值。对于字符串，它是一个字符列表。对于整数，在创建列表之前，参数被转换为 Unicode 字符串。

## phone2numeric

将电话号码（可能包含字母）转换为其数字等价物。输入不一定是有效的电话号码。这将愉快地转换任何字符串。例如：

```py
{{ value|phone2numeric }} 

```

如果`value`是`800-COLLECT`，输出将是`800-2655328`。

## pluralize

如果值不是`1`，则返回复数后缀。默认情况下，此后缀为`s`。

对于不通过简单后缀复数化的单词，可以指定由逗号分隔的单数和复数后缀。例如：

```py
You have {{ num_cherries }} cherr{{ num_cherries|pluralize:"y,ies" }}. 

```

## 漂亮打印

`pprint.pprint()`的包装器-用于调试。

## 随机

从给定列表返回一个随机项。

## rjust

将值右对齐到给定宽度的字段。例如：

```py
{{ value|rjust:"10" }} 

```

如果`value`是`Django`，输出将是`Django`。

## 安全

将字符串标记为在输出之前不需要进一步的 HTML 转义。当自动转义关闭时，此过滤器没有效果。

## safeseq

将`safe`过滤器应用于序列的每个元素。与操作序列的其他过滤器（如`join`）一起使用时很有用。例如：

```py
{{ some_list|safeseq|join:", " }} 

```

在这种情况下，您不能直接使用`safe`过滤器，因为它首先会将变量转换为字符串，而不是处理序列的各个元素。

## 切片

返回列表的一个切片。使用与 Python 列表切片相同的语法。

## slugify

转换为 ASCII。将空格转换为连字符。删除非字母数字、下划线或连字符的字符。转换为小写。还会去除前导和尾随空格。

## stringformat

根据参数格式化变量，一个字符串格式化说明符。此说明符使用 Python 字符串格式化语法，唯一的例外是省略了前导%。

## 去除标签

尽一切可能去除所有[X]HTML 标记。例如：

```py
{{ value|striptags }} 

```

## 时间

根据给定的格式格式化时间。给定的格式可以是预定义的`TIME_FORMAT`，也可以是与`date`过滤器相同的自定义格式。

## timesince

将日期格式化为自那日期以来的时间（例如，4 天，6 小时）。接受一个可选参数，该参数是包含要用作比较点的日期的变量（没有参数，则比较点是`now`）。

## timeuntil

从现在起测量到给定日期或`datetime`的时间。

## 标题

通过使单词以大写字母开头并将其余字符转换为小写，将字符串转换为标题大小写。

## truncatechars

如果字符串长度超过指定的字符数，则截断字符串。截断的字符串将以可翻译的省略号序列（...）结尾。例如：

```py
{{ value|truncatechars:9 }} 

```

## truncatechars_html

类似于`truncatechars`，只是它知道 HTML 标记。

## truncatewords

在一定数量的单词后截断字符串。

## truncatewords_html

类似于`truncatewords`，只是它知道 HTML 标记。

## unordered_list

递归地获取自我嵌套列表并返回一个不带开放和关闭标签的 HTML 无序列表。

## 上限

将字符串转换为大写。

## urlencode

为在 URL 中使用而转义值。

## urlize

将文本中的 URL 和电子邮件地址转换为可点击的链接。此模板标签适用于以`http://`、`https://`或`www.`为前缀的链接。

## urlizetrunc

将 URL 和电子邮件地址转换为可点击的链接，就像`urlize`一样，但截断超过给定字符限制的 URL。例如：

```py
{{ value|urlizetrunc:15 }} 

```

如果`value`是`Check out www.djangoproject.com`，输出将是`Check out <a href="http://www.djangoproject.com" rel="nofollow">www.djangopr...</a>`。与`urlize`一样，此过滤器只应用于纯文本。

## wordcount

返回单词数。

## wordwrap

在指定的行长度处包装单词。

## yesno

将真、假和（可选）无映射值为字符串 yes、no、maybe，或作为逗号分隔列表传递的自定义映射之一，并根据值返回其中之一：例如：

```py
{{ value|yesno:"yeah,no,maybe" }} 

```

# 国际化标签和过滤器

Django 提供模板标签和过滤器来控制模板中国际化的每个方面。它们允许对翻译、格式化和时区转换进行细粒度控制。

## i18n

此库允许在模板中指定可翻译的文本。要启用它，请将`USE_I18N`设置为`True`，然后使用`{% load i18n %}`加载它。

## l10n

这个库提供了对模板中数值本地化的控制。你只需要使用`{% load l10n %}`加载库，但通常会将`USE_L10N`设置为`True`，以便默认情况下启用本地化。

## tz

这个库提供了对模板中时区转换的控制。像`l10n`一样，你只需要使用`{% load tz %}`加载库，但通常也会将`USE_TZ`设置为`True`，以便默认情况下进行本地时间转换。请参阅模板中的时区。

# 其他标签和过滤器库

## static

要链接到保存在`STATIC_ROOT`中的静态文件，Django 附带了一个`static`模板标签。无论你是否使用`RequestContext`，你都可以使用它。

```py
{% load static %} 
<img src="img/{% static "images/hi.jpg" %}" alt="Hi!" /> 

```

它还能够使用标准上下文变量，例如，假设一个`user_stylesheet`变量被传递给模板：

```py
{% load static %} 
<link rel="stylesheet" href="{% static user_stylesheet %}" type="text/css" media="screen" /> 

```

如果你想要检索静态 URL 而不显示它，你可以使用稍微不同的调用：

```py
{% load static %} 
{% static "images/hi.jpg" as myphoto %} 
<img src="img/{{ myphoto }}"></img> 

```

`staticfiles` contrib 应用程序还附带了一个`static 模板标签`，它使用`staticfiles` `STATICFILES_STORAGE`来构建给定路径的 URL（而不仅仅是使用`STATIC_URL`设置和给定路径的`urllib.parse.urljoin()`）。如果你有高级用例，比如使用云服务来提供静态文件，那就使用它：

```py
{% load static from staticfiles %} 
<img src="img/{% static "images/hi.jpg" %}" alt="Hi!" /> 

```

## get_static_prefix

你应该优先使用`static`模板标签，但如果你需要更多控制`STATIC_URL`被注入到模板的位置和方式，你可以使用`get_static_prefix`模板标签：

```py
{% load static %} 
<img src="img/hi.jpg" alt="Hi!" /> 

```

还有第二种形式，如果你需要多次使用该值，可以避免额外的处理：

```py
{% load static %} 
{% get_static_prefix as STATIC_PREFIX %} 

<img src="img/hi.jpg" alt="Hi!" /> 
<img src="img/hi2.jpg" alt="Hello!" /> 

```

## get_media_prefix

类似于`get_static_prefix`，`get_media_prefix`会用媒体前缀`MEDIA_URL`填充模板变量，例如：

```py
<script type="text/javascript" charset="utf-8"> 
var media_path = '{% get_media_prefix %}'; 
</script> 

```

Django 还附带了一些其他模板标签库，你必须在`INSTALLED_APPS`设置中显式启用它们，并在模板中使用`{% load %}`标签启用它们。
