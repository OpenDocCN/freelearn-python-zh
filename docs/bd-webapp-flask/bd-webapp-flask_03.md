# 第三章：天哪，我喜欢模板！

如前所述，Flask 为您提供了 MVC 中的 VC。在本章中，我们将讨论 Jinja2 是什么，以及 Flask 如何使用 Jinja2 来实现视图层并让您感到敬畏。做好准备！

# Jinja2 是什么，它如何与 Flask 耦合在一起？

Jinja2 是一个库，可以在[`jinja.pocoo.org/`](http://jinja.pocoo.org/)找到；您可以使用它来生成带有捆绑逻辑的格式化文本。与 Python 格式函数不同，Python 格式函数只允许您用变量内容替换标记，您可以在模板字符串中使用控制结构（例如`for`循环），并使用 Jinja2 进行解析。让我们考虑这个例子：

```py
from jinja2 import Template
x = """
<p>Uncle Scrooge nephews</p>
<ul>
{% for i in my_list %}
<li>{{ i }}</li>
{% endfor %}
</ul>
"""
template = Template(x)
# output is an unicode string
print template.render(my_list=['Huey', 'Dewey', 'Louie'])
```

在上面的代码中，我们有一个非常简单的例子，其中我们创建了一个模板字符串，其中包含一个`for`循环控制结构（简称“for 标签”），该结构遍历名为`my_list`的列表变量，并使用大括号`{{ }}`符号打印“li HTML 标签”中的元素。

请注意，您可以在模板实例中调用`render`多次，并使用不同的键值参数，也称为模板上下文。上下文变量可以有任何有效的 Python 变量名——也就是说，任何符合正则表达式*[a-zA-Z_][a-zA-Z0-9_]*格式的内容。

### 提示

有关 Python 正则表达式（**Regex**简称）的完整概述，请访问[`docs.python.org/2/library/re.html`](https://docs.python.org/2/library/re.html)。还可以查看这个用于正则表达式测试的在线工具[`pythex.org/`](http://pythex.org/)。

一个更复杂的例子将使用环境类实例，这是一个中央、可配置、可扩展的类，可以以更有组织的方式加载模板。

您明白我们要说什么了吗？这是 Jinja2 和 Flask 背后的基本原理：它为您准备了一个环境，具有一些响应式默认设置，并让您的轮子转起来。

# 您可以用 Jinja2 做什么？

Jinja2 非常灵活。您可以将其与模板文件或字符串一起使用；您可以使用它来创建格式化文本，例如 HTML、XML、Markdown 和电子邮件内容；您可以组合模板、重用模板和扩展模板；甚至可以使用扩展。可能性是无穷无尽的，并且结合了良好的调试功能、自动转义和完整的 Unicode 支持。

### 注意

自动转义是 Jinja2 的一种配置，其中模板中打印的所有内容都被解释为纯文本，除非另有明确要求。想象一个变量*x*的值设置为`<b>b</b>`。如果启用了自动转义，模板中的`{{ x }}`将打印给定的字符串。如果关闭了自动转义，这是 Jinja2 的默认设置（Flask 的默认设置是开启的），则生成的文本将是`b`。

在介绍 Jinja2 允许我们进行编码之前，让我们先了解一些概念。

首先，我们有前面提到的大括号。双大括号是一个分隔符，允许您从提供的上下文中评估变量或函数，并将其打印到模板中：

```py
from jinja2 import Template
# create the template
t = Template("{{ variable }}")
# – Built-in Types –
t.render(variable='hello you')
>> u"hello you"
t.render(variable=100)
>> u"100"
# you can evaluate custom classes instances
class A(object):
  def __str__(self):
    return "__str__"
  def __unicode__(self):
    return u"__unicode__"
  def __repr__(self):
    return u"__repr__"
# – Custom Objects Evaluation –
# __unicode__ has the highest precedence in evaluation
# followed by __str__ and __repr__
t.render(variable=A())
>> u"__unicode__"
```

在上面的例子中，我们看到如何使用大括号来评估模板中的变量。首先，我们评估一个字符串，然后是一个整数。两者都会产生 Unicode 字符串。如果我们评估我们自己的类，我们必须确保定义了`__unicode__`方法，因为在评估过程中会调用它。如果没有定义`__unicode__`方法，则评估将退回到`__str__`和`__repr__`，依次进行。这很简单。此外，如果我们想评估一个函数怎么办？好吧，只需调用它：

```py
from jinja2 import Template
# create the template
t = Template("{{ fnc() }}")
t.render(fnc=lambda: 10)
>> u"10"
# evaluating a function with argument
t = Template("{{ fnc(x) }}")
t.render(fnc=lambda v: v, x='20')
>> u"20"
t = Template("{{ fnc(v=30) }}")
t.render(fnc=lambda v: v)
>> u"30"
```

要在模板中输出函数的结果，只需像调用任何常规 Python 函数一样调用该函数。函数返回值将被正常评估。如果您熟悉 Django，您可能会注意到这里有一点不同。在 Django 中，您不需要使用括号来调用函数，甚至不需要向其传递参数。在 Flask 中，如果要对函数返回值进行评估，则*始终*需要使用括号。

以下两个示例展示了 Jinja2 和 Django 在模板中函数调用之间的区别：

```py
{# flask syntax #}
{{ some_function() }}

{# django syntax #}
{{ some_function }}
```

您还可以评估 Python 数学运算。看一下：

```py
from jinja2 import Template
# no context provided / needed
Template("{{ 3 + 3 }}").render()
>> u"6"
Template("{{ 3 - 3 }}").render()
>> u"0"
Template("{{ 3 * 3 }}").render()
>> u"9"
Template("{{ 3 / 3 }}").render()
>> u"1"
```

其他数学运算符也可以使用。您可以使用花括号分隔符来访问和评估列表和字典：

```py
from jinja2 import Template
Template("{{ my_list[0] }}").render(my_list=[1, 2, 3])
>> u'1'
Template("{{ my_list['foo'] }}").render(my_list={'foo': 'bar'})
>> u'bar'
# and here's some magic
Template("{{ my_list.foo }}").render(my_list={'foo': 'bar'})
>> u'bar'
```

要访问列表或字典值，只需使用普通的 Python 表示法。对于字典，您还可以使用变量访问表示法访问键值，这非常方便。

除了花括号分隔符，Jinja2 还有花括号/百分比分隔符，它使用`{% stmt %}`的表示法，用于执行语句，这可能是控制语句，也可能不是。它的使用取决于语句，其中控制语句具有以下表示法：

```py
{% stmt %}
{% endstmt %}
```

第一个标签具有语句名称，而第二个是闭合标签，其名称在开头附加了`end`。您必须意识到非控制语句*可能*没有闭合标签。让我们看一些例子：

```py
{% block content %}
{% for i in items %}
{{ i }} - {{ i.price }}
{% endfor %}
{% endblock %}
```

前面的例子比我们之前看到的要复杂一些。它在块语句中使用了控制语句`for`循环（您可以在另一个语句中有一个语句），这不是控制语句，因为它不控制模板中的执行流程。在`for`循环中，您可以看到`i`变量与关联的价格（在其他地方定义）一起打印出来。

您应该知道的最后一个分隔符是`{# comments go here #}`。这是一个多行分隔符，用于声明注释。让我们看两个具有相同结果的例子：

```py
{# first example #}
{#
second example
#}
```

两种注释分隔符都隐藏了`{#`和`#}`之间的内容。可以看到，这个分隔符适用于单行注释和多行注释，非常方便。

## 控制结构

Jinja2 中默认定义了一组不错的内置控制结构。让我们从`if`语句开始学习它。

```py
{% if true %}Too easy{% endif %}
{% if true == true == True %}True and true are the same{% endif %}
{% if false == false == False %}False and false also are the same{% endif %}
{% if none == none == None %}There's also a lowercase None{% endif %}
{% if 1 >= 1 %}Compare objects like in plain python{% endif %}
{% if 1 == 2 %}This won't be printed{% else %}This will{% endif %}
{% if "apples" != "oranges" %}All comparison operators work = ]{% endif %}
{% if something %}elif is also supported{% elif something_else %}^_^{% endif %}
```

`if`控制语句很美妙！它的行为就像`python if`语句一样。如前面的代码所示，您可以使用它以非常简单的方式比较对象。"`else`"和"`elif`"也得到了充分支持。

您可能还注意到了`true`和`false`，非大写，与普通的 Python 布尔值`True`和`False`一起使用。为了避免混淆的设计决策，所有 Jinja2 模板都有`True`、`False`和`None`的小写别名。顺便说一句，小写语法是首选的方式。

如果需要的话，您应该避免这种情况，可以将比较组合在一起以改变优先级评估。请参阅以下示例：

```py
{% if  5 < 10 < 15 %}true{%else%}false{% endif %}
{% if  (5 < 10) < 15 %}true{%else%}false{% endif %}
{% if  5 < (10 < 15) %}true{%else%}false{% endif %}
```

前面示例的预期输出是`true`、`true`和`false`。前两行非常直接。在第三行中，首先，`(10<15)`被评估为`True`，它是`int`的子类，其中`True == 1`。然后评估`5` < `True`，这显然是假的。

`for`语句非常重要。几乎无法想象一个严肃的 Web 应用程序不必在某个时候显示某种列表。`for`语句可以迭代任何可迭代实例，并且具有非常简单的、类似 Python 的语法：

```py
{% for item in my_list %}
{{ item }}{# print evaluate item #}
{% endfor %}
{# or #}
{% for key, value in my_dictionary.items() %}
{{ key }}: {{ value }}
{% endfor %}
```

在第一个语句中，我们有一个开放标签，指示我们将遍历`my_list`项，每个项将被名称`item`引用。名称`item`仅在`for`循环上下文中可用。

在第二个语句中，我们对形成`my_dictionary`的键值元组进行迭代，这应该是一个字典（如果变量名不够具有启发性的话）。相当简单，对吧？`for`循环也为您准备了一些技巧。

在构建 HTML 列表时，通常需要以交替颜色标记每个列表项，以改善可读性，或者使用一些特殊标记标记第一个和/或最后一个项目。这些行为可以通过在 Jinja2 for 循环中访问块上下文中可用的循环变量来实现。让我们看一些例子：

```py
{% for i in ['a', 'b', 'c', 'd'] %}
{% if loop.first %}This is the first iteration{% endif %}
{% if loop.last %}This is the last iteration{% endif %}
{{ loop.cycle('red', 'blue') }}{# print red or blue alternating #}
{{ loop.index }} - {{ loop.index0 }} {# 1 indexed index – 0 indexed index #}
{# reverse 1 indexed index – reverse 0 indexed index #}
{{ loop.revindex }} - {{ loop.revindex0 }} 
{% endfor %}
```

`for`循环语句，就像 Python 一样，也允许使用`else`，但意义略有不同。在 Python 中，当您在`for`中使用`else`时，只有在没有通过`break`命令到达`else`块时才会执行`else`块，就像这样：

```py
for i in [1, 2, 3]:
  pass
else:
  print "this will be printed"
for i in [1, 2, 3]:
  if i == 3:
    break
else:
  print "this will never not be printed"
```

如前面的代码片段所示，`else`块只有在`for`循环中从未被`break`命令中断执行时才会执行。使用 Jinja2 时，当`for`可迭代对象为空时，将执行`else`块。例如：

```py
{% for i in [] %}
{{ i }}
{% else %}I'll be printed{% endfor %}
{% for i in ['a'] %}
{{ i }}
{% else %}I won't{% endfor %}
```

由于我们正在讨论循环和中断，有两件重要的事情要知道：Jinja2 的`for`循环不支持`break`或`continue`。相反，为了实现预期的行为，您应该使用循环过滤，如下所示：

```py
{% for i in [1, 2, 3, 4, 5] if i > 2 %}
value: {{ i }}; loop.index: {{ loop.index }}
{%- endfor %}
```

在第一个标签中，您会看到一个普通的`for`循环和一个`if`条件。您应该将该条件视为一个真正的列表过滤器，因为索引本身只是在每次迭代中计数。运行前面的示例，输出将如下所示：

```py
value:3; index: 1
value:4; index: 2
value:5; index: 3
```

看看前面示例中的最后一个观察——在第二个标签中，您看到`{%-`中的破折号吗？它告诉渲染器在每次迭代之前不应该有空的新行。尝试我们之前的示例，不带破折号，并比较结果以查看有何变化。

现在我们将看看用于从不同文件构建模板的三个非常重要的语句：`block`、`extends`和`include`。

`block`和`extends`总是一起使用。第一个用于定义模板中的“可覆盖”块，而第二个定义了具有块的当前模板的父模板。让我们看一个例子：

```py
# coding:utf-8
with open('parent.txt', 'w') as file:
    file.write("""
{% block template %}parent.txt{% endblock %}
===========
I am a powerful psychic and will tell you your past

{#- "past" is the block identifier #}
{% block past %}
You had pimples by the age of 12.
{%- endblock %}

Tremble before my power!!!""".strip())

with open('child.txt', 'w') as file:
    file.write("""
{% extends "parent.txt" %}

{# overwriting the block called template from parent.txt #}
{% block template %}child.txt{% endblock %}

{#- overwriting the block called past from parent.txt #}
{% block past %}
You've bought an ebook recently.
{%- endblock %}""".strip())
with open('other.txt', 'w') as file:
	file.write("""
{% extends "child.txt" %}
{% block template %}other.txt{% endblock %}""".strip())

from jinja2 import Environment, FileSystemLoader

env = Environment()
# tell the environment how to load templates
env.loader = FileSystemLoader('.')
# look up our template
tmpl = env.get_template('parent.txt')
# render it to default output
print tmpl.render()
print ""
# loads child.html and its parent
tmpl = env.get_template('child.txt')
print tmpl.render()
# loads other.html and its parent
env.get_template('other.txt').render()
```

您是否看到了`child.txt`和`parent.txt`之间的继承？`parent.txt`是一个简单的模板，有两个名为`template`和`past`的`block`语句。当您直接呈现`parent.txt`时，它的块会“原样”打印，因为它们没有被覆盖。在`child.txt`中，我们扩展`parent.txt`模板并覆盖所有其块。通过这样做，我们可以在模板的特定部分中具有不同的信息，而无需重写整个内容。

例如，使用`other.txt`，我们扩展`child.txt`模板并仅覆盖命名为 block 的模板。您可以从直接父模板或任何父模板覆盖块。

如果您正在定义一个`index.txt`页面，您可以在其中有默认块，需要时进行覆盖，从而节省大量输入。

解释最后一个示例，就 Python 而言，非常简单。首先，我们创建了一个 Jinja2 环境（我们之前谈到过这个），并告诉它如何加载我们的模板，然后直接加载所需的模板。我们不必费心告诉环境如何找到父模板，也不必预加载它们。

`include`语句可能是迄今为止最简单的语句。它允许您以非常简单的方式在另一个模板中呈现模板。让我们看一个例子：

```py
with open('base.txt', 'w') as file:
  file.write("""
{{ myvar }}
You wanna hear a dirty joke?
{% include 'joke.txt' %}
""".strip())
with open('joke.txt', 'w') as file:
  file.write("""
A boy fell in a mud puddle. {{ myvar }} 
""".strip())

from jinja2 import Environment, FileSystemLoader

env = Environment()
# tell the environment how to load templates
env.loader = FileSystemLoader('.')
print env.get_template('base.txt').render(myvar='Ha ha!')
```

在前面的示例中，我们在`base.txt`中呈现`joke.txt`模板。由于`joke.txt`在`base.txt`中呈现，它也可以完全访问`base.txt`上下文，因此`myvar`会正常打印。

最后，我们有`set`语句。它允许您在模板上下文中定义变量。它的使用非常简单：

```py
{% set x = 10 %}
{{ x }}
{% set x, y, z = 10, 5+5, "home" %}
{{ x }} - {{ y }} - {{ z }}
```

在前面的示例中，如果`x`是通过复杂计算或数据库查询给出的，如果要在模板中重复使用它，将其*缓存*在一个变量中会更有意义。如示例中所示，您还可以一次为多个变量分配一个值。

## 宏

宏是您在 Jinja2 模板中最接近编码的地方。宏的定义和使用类似于普通的 Python 函数，因此非常容易。让我们尝试一个例子：

```py
with open('formfield.html', 'w') as file:
  file.write('''
{% macro input(name, value='', label='') %}
{% if label %}
<label for='{{ name }}'>{{ label }}</label>
{% endif %}
<input id='{{ name }}' name='{{ name }}' value='{{ value }}'></input>
{% endmacro %}'''.strip())
with open('index.html', 'w') as file:
  file.write('''
{% from 'formfield.html' import input %}
<form method='get' action='.'>
{{ input('name', label='Name:') }}
<input type='submit' value='Send'></input>
</form>
'''.strip())

from jinja2 import Environment, FileSystemLoader

env = Environment()
env.loader = FileSystemLoader('.')
print env.get_template('index.html').render()
```

在前面的例子中，我们创建了一个宏，接受一个`name`参数和两个可选参数：`value`和`label`。在`macro`块内，我们定义了应该输出的内容。请注意，我们可以在宏中使用其他语句，就像在模板中一样。在`index.html`中，我们从`formfield.html`中导入输入宏，就好像`formfield`是一个模块，输入是一个使用`import`语句的 Python 函数。如果需要，我们甚至可以像这样重命名我们的输入宏：

```py
{% from 'formfield.html' import input as field_input %}
```

您还可以将`formfield`作为模块导入并按以下方式使用它：

```py
{% import 'formfield.html' as formfield %}
```

在使用宏时，有一种特殊情况，您希望允许任何命名参数传递到宏中，就像在 Python 函数中一样（例如，`**kwargs`）。使用 Jinja2 宏，默认情况下，这些值在`kwargs`字典中可用，不需要在宏签名中显式定义。例如：

```py
# coding:utf-8
with open('formfield.html', 'w') as file:
    file.write('''
{% macro input(name) -%}
<input id='{{ name }}' name='{{ name }}' {% for k,v in kwargs.items() -%}{{ k }}='{{ v }}' {% endfor %}></input>
{%- endmacro %}
'''.strip())with open('index.html', 'w') as file:
    file.write('''
{% from 'formfield.html' import input %}
{# use method='post' whenever sending sensitive data over HTTP #}
<form method='post' action='.'>
{{ input('name', type='text') }}
{{ input('passwd', type='password') }}
<input type='submit' value='Send'></input>
</form>
'''.strip())

from jinja2 import Environment, FileSystemLoader

env = Environment()
env.loader = FileSystemLoader('.')
print env.get_template('index.html').render()
```

如您所见，即使您没有在宏签名中定义`kwargs`参数，`kwargs`也是可用的。

宏在纯模板上具有一些明显的优势，您可以通过`include`语句注意到：

+   使用宏时，您不必担心模板中的变量名称

+   您可以通过宏签名定义宏块的确切所需上下文

+   您可以在模板中定义一个宏库，并仅导入所需的内容

Web 应用程序中常用的宏包括用于呈现分页的宏，用于呈现字段的宏，以及用于呈现表单的宏。您可能还有其他用例，但这些是相当常见的用例。

### 提示

关于我们之前的例子，使用 HTTPS（也称为安全 HTTP）发送敏感信息，如密码，通过互联网是一个良好的做法。要小心！

## 扩展

扩展是 Jinja2 允许您扩展其词汇的方式。扩展默认情况下未启用，因此只有在需要时才能启用扩展，并且可以在不太麻烦的情况下开始使用它：

```py
env = Environment(extensions=['jinja2.ext.do', 'jinja2.ext.with_'])
```

在前面的代码中，我们有一个示例，其中您创建了一个启用了两个扩展的环境：`do`和`with`。这些是我们将在本章中学习的扩展。

正如其名称所示，`do`扩展允许您“做一些事情”。在`do`标记内，您可以执行 Python 表达式，并完全访问模板上下文。Flask-Empty 是一个流行的 Flask 样板，可在[`github.com/italomaia/flask-empty`](https://github.com/italomaia/flask-empty)上找到，它使用`do`扩展来更新其宏之一中的字典。让我们看看我们如何做到这一点：

```py
{% set x = {1:'home', '2':'boat'} %}
{% do x.update({3: 'bar'}) %}
{%- for key,value in x.items() %}
{{ key }} - {{ value }}
{%- endfor %}
```

在前面的例子中，我们使用一个字典创建了`x`变量，然后用`{3: 'bar'}`更新了它。通常情况下，您不需要使用`do`扩展，但是当您需要时，可以节省大量编码。

`with`扩展也非常简单。每当您需要创建块作用域变量时，都可以使用它。想象一下，您有一个需要在变量中缓存一小段时间的值；这将是一个很好的用例。让我们看一个例子：

```py
{% with age = user.get_age() %}
My age: {{ age }}
{% endwith %}
My age: {{ age }}{# no value here #}
```

如示例所示，`age`仅存在于`with`块内。此外，在`with`块内设置的变量将仅在其中存在。例如：

```py
{% with %}
{% set count = query.count() %}
Current Stock: {{ count }}
Diff: {{ prev_count - count }}
{% endwith %}
{{ count }} {# empty value #}
```

## 过滤器

过滤器是 Jinja2 的一个奇妙之处！这个工具允许您在将常量或变量打印到模板之前对其进行处理。目标是在模板中严格实现您想要的格式。

要使用过滤器，只需使用管道运算符调用它，就像这样：

```py
{% set name = 'junior' %}
{{ name|capitalize }} {# output is Junior #}
```

它的名称被传递给**capitalize**过滤器进行处理，并返回大写的值。要将参数传递给过滤器，只需像调用函数一样调用它，就像这样：

```py
{{ ['Adam', 'West']|join(' ') }} {# output is Adam West #}
```

`join`过滤器将连接传递的可迭代值，将提供的参数放在它们之间。

Jinja2 默认提供了大量可用的过滤器。这意味着我们无法在这里覆盖它们所有，但我们当然可以覆盖一些。`capitalize`和`lower`已经看到了。让我们看一些进一步的例子：

```py
{# prints default value if input is undefined #}
{{ x|default('no opinion') }}
{# prints default value if input evaluates to false #}
{{ none|default('no opinion', true) }}
{# prints input as it was provided #}
{{ 'some opinion'|default('no opinion') }}

{# you can use a filter inside a control statement #}
{# sort by key case-insensitive #}
{% for key in {'A':3, 'b':2, 'C':1}|dictsort %}{{ key }}{% endfor %}
{# sort by key case-sensitive #}
{% for key in {'A':3, 'b':2, 'C':1}|dictsort(true) %}{{ key }}{% endfor %}
{# sort by value #}
{% for key in {'A':3, 'b':2, 'C':1}|dictsort(false, 'value') %}{{ key }}{% endfor %}
{{ [3, 2, 1]|first }} - {{ [3, 2, 1]|last }}
{{ [3, 2, 1]|length }} {# prints input length #}
{# same as in python #}
{{ '%s, =D'|format("I'm John") }}
{{ "He has two daughters"|replace('two', 'three') }}
{# safe prints the input without escaping it first#}
{{ '<input name="stuff" />'|safe }}
{{ "there are five words here"|wordcount }}
```

尝试前面的例子，以确切了解每个过滤器的作用。

阅读了这么多关于 Jinja2 的内容，您可能会想：“Jinja2 很酷，但这是一本关于 Flask 的书。给我看看 Flask 的东西！”好的，好的，我可以做到！

根据我们迄今所见，几乎一切都可以在 Flask 中使用而无需修改。由于 Flask 为您管理 Jinja2 环境，因此您不必担心创建文件加载程序之类的事情。但是，您应该知道的一件事是，由于您不是自己实例化 Jinja2 环境，因此您实际上无法将要激活的扩展传递给类构造函数。

要激活扩展程序，请在应用程序设置期间将其添加到 Flask 中，如下所示：

```py
from flask import Flask
app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.do')  # or jinja2.ext.with_
if __name__ == '__main__':
  app.run()
```

## 搞乱模板上下文

在第二章中所见，*第一个应用，有多难？*，您可以使用`render_template`方法从`templates`文件夹加载模板，然后将其呈现为响应。

```py
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")
```

如果您想向模板上下文添加值，就像本章中的一些示例中所示，您将不得不向`render_template`添加非位置参数：

```py
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html", my_age=28)
```

在上面的示例中，`my_age`将在`index.html`上下文中可用，其中`{{ my_age }}`将被翻译为 28。`my_age`实际上可以具有您想要展示的任何值。

现在，如果您希望*所有*视图在其上下文中具有特定值，例如版本值-一些特殊代码或函数；您该怎么做？Flask 为您提供了`context_processor`装饰器来实现这一点。您只需注释一个返回字典的函数，然后就可以开始了。例如：

```py
from flask import Flask, render_response
app = Flask(__name__)

@app.context_processor
def luck_processor():
  from random import randint
  def lucky_number():
    return randint(1, 10)
  return dict(lucky_number=lucky_number)

@app.route("/")
def hello():
  # lucky_number will be available in the index.html context by default
  return render_template("index.html")
```

# 总结

在本章中，我们看到了如何仅使用 Jinja2 呈现模板，控制语句的外观以及如何使用它们，如何编写注释，如何在模板中打印变量，如何编写和使用宏，如何加载和使用扩展，以及如何注册上下文处理器。我不知道您怎么看，但这一章节感觉像是大量的信息！我强烈建议您运行示例进行实验。熟悉 Jinja2 将为您节省大量麻烦。

下一章，我们将学习使用 Flask 的表单。期待许多示例和补充代码，因为表单是您从 Web 应用程序打开到 Web 的大门。大多数问题都来自 Web，您的大多数数据也是如此。
