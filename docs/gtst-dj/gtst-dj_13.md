# 附录 A. 备忘单

当开发人员学会如何使用技术时，通常需要搜索新的信息或语法。他/她可能会浪费很多时间。本附录的目的是为 Django 开发人员提供快速参考。

# 模型中的字段类型

以下各节涵盖了模型中字段类型的非穷尽列表。

模型字段是将保存在数据库中的字段。根据所选的数据库系统，字段类型可能会因使用的数据库而异。

类型是以以下方式指定其选项的：

```py
Type (option1 = example_data, option2 = example_data) [information]
```

## 数字字段类型

此部分中呈现的字段是数字字段，如整数和小数：

+   `SmallIntegerField()`：定义小整数字段；对于某些数据库，较低值为 256

+   `IntegerField()`：定义整数字段

+   `BigIntegerField()`：精度为 64 位，范围为 -9223372036854775808 到 9223372036854775807

+   `DecimalField（max_digits = 8`，`decimal_places = 2）`

选项的描述如下：

+   `max_digits`：设置组成整数的数字的位数

+   `decimal_places`：设置组成数字的小数部分的位数

## 字符串字段类型

此部分包含包含字符串的字段类型：

+   `CharField（max_length = 250）`

+   `TextField（max_length = 250）`：此字段具有在 Django 表单中呈现为`<textarea>`标签的特点

+   `EmailField（max_length = 250）`：此字段是包含 Django 表单的电子邮件验证程序的`CharField`

选项的描述如下：

+   `max_length`：设置组成字符串的最大字符数

## 时间字段类型

此部分包含包含临时数据的字段类型：

+   `DateField（auto_now = false`，`auto_now_add = true）`

+   `DateTimeField（auto_now = false`，`auto_now_add = true）`

+   `TimeField（auto_now = false`，`auto_now_add = true）`

选项的描述如下：

+   `auto_now`：这会自动将字段设置为每次保存记录时的当前时间

+   `auto_now_add`：这会在创建对象时自动将字段设置为当前时间

## 其他类型的字段

此部分包含不属于先前类别的字段类型：

+   `BooleanField()`

+   `FileField：（upload_to = "path"，max_length="250"）`：此字段用于在服务器上存储文件

+   `ImageField（upload_to = "path"，max_length="250"，height_field =height_img，width_field= width_img）`：此字段对应于`FileField`，但对图像进行特殊处理，如存储图像的高度和宽度

选项的描述如下：

+   `Upload_to`：定义将存储与此字段对应的文件的文件夹。

+   `max_length`：`FileField`和`ImageField`字段实际上是存储上传文件的路径和名称的文本字段。

+   `height_field`和`width_field`：这些以模型的整数字段作为参数。此字段用于存储图像的大小。

## 模型之间的关系

此部分包含定义模型之间关系的字段类型：

+   `ForeignKey（model，related_name = "foreign_key_for_dev"，to_field="field_name"，limit_choices_to=dict_or_Q，on_delete=）`

+   `OneToOneField（model，related_name = "foreign_key_for_dev"，to_field="field_name"，limit_choices_to=dict_or_Q，on_delete=）`

+   `ManyToManyField（model，related_name = "foreign_key_for_dev"，to_field="field_name"，limit_choices_to=dict_or_Q，on_delete=）`

选项的描述如下：

+   `model`：在这里，您必须指定要使用的模型类的名称。

+   `related_name`：这允许您命名关系。当存在多个与同一模型的关系时，这是必不可少的。

+   `to_field`：这定义了与模型的特定字段的关系。默认情况下，Django 会创建与主键的关系。

+   `on_delete`：在删除字段时数据库操作可以是`CASCADE`、`PROTECT`、`SET_NULL`、`SET_DEFAULT`和`DO_NOTHING`。

+   `limit_choices_to`：这定义了限制与关系的记录的查询集。

## 模型元属性

模型元属性应该在模型中的元类中以以下方式定义：

```py
class Product(models.Model):
  name = models.CharField()
  class Meta:
    verbose_name = "product"
```

以下属性用于定义放置它们的模型的信息：

+   `db_tables`：设置存储在数据库中的表的名称

+   `verbose_name`：为用户设置记录的名称

+   `verbose_name_plural`：为用户设置多个记录的名称

+   `ordering`：在列出记录时设置默认顺序

## 模型字段的常见选项

以下选项适用于模型的所有字段：

+   `default`：为字段设置默认值。

+   `null`：为字段启用空值，并且如果在关系字段上定义了此选项，则使关系变为可选。

+   `blank`：允许您将字段留空。

+   `error_messages`：指定一系列错误消息。

+   `help_text`：设置帮助消息。

+   `unique`：定义不包含重复项的字段。

+   `verbose_name`：定义一个可供人类阅读的字段名称。不要首字母大写；Django 会自动完成。

+   `choices`：这定义了字段的可能选择数量。

+   `db_column`：设置在数据库中创建的字段的名称。

# 表单字段

可以在表单中使用所有类型的字段模型。实际上，某些类型的模型字段已经被创建用于在表单中特定的用途。例如，`TextField`模型字段与`CharField`没有任何不同，除了默认情况下，在表单中，`TextField`字段显示一个`<textarea>`标签和一个`<input type="text">`名称。因此，您可以编写一个表单字段如下：

```py
field1 = forms.TextField()
```

## 表单字段的常见选项

以下选项适用于所有表单字段：

+   `error_messages`：指定一系列错误消息

+   `help_text`：设置帮助消息

+   `required`：定义必须填写的字段

+   `initial`：为字段设置默认值

+   `validators`：定义验证字段值的特定验证器

+   `widget`：为字段定义特定的小部件

## 小部件表单

小部件允许您定义呈现表单字段的 HTML 代码。我们将解释小部件可以生成的 HTML 代码，如下所示：

+   `TextInput`：对应于`<input type="text" />`

+   `Textarea`：对应于`<textarea></textarea>`

+   `PasswordInput`：对应于`<input type="password" />`

+   `RadioSelect`：这对应于`<input type="radio" />`

+   `Select`：对应于`<select><option></option></select>`

+   `CheckboxInput`：对应于`<input type="checkbox" />`

+   `FileInput`：这对应于`<input type="file" />`

+   `HiddenInput`：这对应于`<input type="hidden" />`

## 错误消息（表单和模型）

以下是在表单字段输入不正确时可以设置的错误消息的部分列表：

+   `required`：当用户未在字段中填写数据时显示此消息

+   `min_length`：当用户未提供足够的数据时显示此消息

+   `max_length`：当用户超出字段的大小限制时显示此消息

+   `min_value`：当用户输入的值太低时显示此消息

+   `max_value`：当用户输入的值太高时显示此消息

# 模板语言

当开发人员开发模板时，他/她经常需要使用模板语言和过滤器。

## 模板标签

以下是模板语言的关键元素：

+   `{% autoescape on OR off %} {% endautoescape %}`：这自动启动自动转义功能，有助于保护显示数据的浏览器（XSS）。

+   `{% block block_name %} {% endblock %}`: 这设置可以由继承自它们的模板填充的块。

+   `{% comment %} {% endcomment %}`: 这设置一个不会作为 HTML 发送给用户的注释。

+   `{% extends template_name %}`: 这会覆盖一个模板。

+   `{% spaceless %}`: 这会删除 HTML 标签之间的所有空格。

+   `{% include template_name %}`: 这在当前模板中包含一个名为`template_name`的模板。包含的模板块不能被重新定义。

### 字典中的循环

本节向您展示如何循环遍历字典。循环涉及的步骤如下：

+   `{% for var in list_var %}`: 这允许在`list_var`字典中循环

+   `{% empty %}`: 如果字典为空，则显示后续代码

+   `{% endfor %}`: 这表示循环的结束

### 条件语句

本节显示了如何执行条件语句：

+   `{% if cond %}`: 此行检查条件，并在启用时讨论以下代码。

+   `{% elif cond %}`: 如果第一个条件未经验证，此行将检查另一个条件。如果满足此条件，将处理以下代码。

+   `{% else %}`: 如果之前的条件都没有被验证，这行将处理以下代码。

+   `{% endif %}`: 此行结束条件的处理。

## 模板过滤器

以下是不同的模板过滤器：

+   `addslashes`: 这在引号前添加斜杠

+   `capfirst`: 这将首字母大写

+   `lower`: 这将文本转换为小写

+   `upper`: 这将文本转换为大写

+   `title`: 这将每个单词的第一个字符大写

+   `cut`: 这从给定字符串中删除参数的所有值，例如，`{{ value|cut:"*" }}`删除所有`*`字符

+   `linebreaks`: 这将文本中的换行符替换为适当的 HTML 标记

+   `date`: 这显示一个格式化的日期，例如，`{{ value|date:"D d M Y" }}`将显示`Wed 09 Jan 2008`

+   `pluralize`: 这允许您显示复数，如下所示：

```py
You have {{ nb_products }} product{{ nb_products|pluralize }} in our cart.
I received {{ nb_diaries }} diar{{ nb_diaries|pluralize:"y,ies" }}.
```

+   `random`: 这从列表中返回一个随机元素

+   `linenumbers`: 这在左侧显示带有行号的文本

+   `first`: 这显示列表中的第一个项目

+   `last`: 这显示列表中的最后一个项目

+   `safe`: 这设置了一个非转义值

+   `escape`: 这会转义 HTML 字符串

+   `escapejs`: 这会转义字符以在 JavaScript 字符串中使用

+   `default`: 如果原始值等于`None`或`empty`，则定义默认值；例如，使用`{{ value|default:"nothing" }}`，如果值为`""`，它将显示`nothing`。

+   `dictsort`：这将按键的升序对字典进行排序；例如，`{{ value|dictsort:"price"}}`将按`price`对字典进行排序

+   `dictsortreversed`: 这用于按键的降序对字典进行排序

+   `floatformat`: 这格式化一个浮点值，以下是示例：

+   当值为`45.332`时，`{{ value|floatformat:2 }}`显示`45.33`

+   当值为`45.00`时，`{{ value|floatformat:"-2" }}`显示`45`

## 查询集方法

以下是查询集方法：

+   `all()`: 此方法检索模型的所有记录。

+   `filter(condition)`: 此方法允许您过滤查询集。

+   `none()`: 此方法可以返回一个空的查询集。当您想要清空一个查询集时，此方法很有用。

+   `dinstinct(field_name)`: 此方法用于检索字段的唯一值。

+   `values_list(field_name)`: 此方法用于检索字段的数据字典。

+   `get(condition)`: 此方法用于从模型中检索记录。在使用此方法时，您必须确保它只涉及一个记录。

+   `exclude(condition)`: 此方法允许您排除一些记录。

以下元素是聚合方法：

+   `Count()`: 这计算返回的记录数

+   `Sum()`: 这将字段中的值相加

+   `Max()`: 这检索字段的最大值

+   `Min()`: 这检索字段的最小值

+   `Avg()`: 这使用字段的平均值
