# 附录 A.模型定义参考

第四章中的*模型*解释了定义模型的基础知识，并且我们在本书的其余部分中使用它们。然而，还有大量的模型选项可用，其他地方没有涵盖。本附录解释了每个可能的模型定义选项。

# 字段

模型最重要的部分-也是模型的唯一必需部分-是它定义的数据库字段列表。

## 字段名称限制

Django 对模型字段名称只有两个限制：

1.  字段名称不能是 Python 保留字，因为那将导致 Python 语法错误。例如：

```py
        class Example(models.Model): 
        pass = models.IntegerField() # 'pass' is a reserved word! 

```

1.  由于 Django 的查询查找语法的工作方式，字段名称不能连续包含多个下划线。例如：

```py
        class Example(models.Model): 
            # 'foo__bar' has two underscores! 
            foo__bar = models.IntegerField()  

```

您模型中的每个字段都应该是适当`Field`类的实例。Django 使用字段类类型来确定一些事情：

+   数据库列类型（例如，`INTEGER`，`VARCHAR`）

+   在 Django 的表单和管理站点中使用的小部件，如果您愿意使用它（例如，`<input type="text">`，`<select>`）

+   最小的验证要求，这些要求在 Django 的管理界面和表单中使用

每个字段类都可以传递一系列选项参数，例如当我们在第四章中构建书籍模型时，我们的`num_pages`字段如下所示：

```py
num_pages = models.IntegerField(blank=True, null=True) 

```

在这种情况下，我们为字段类设置了`blank`和`null`选项。*表 A.2*列出了 Django 中的所有字段选项。

许多字段还定义了特定于该类的其他选项，例如`CharField`类具有一个必需选项`max_length`，默认为`None`。例如：

```py
title = models.CharField(max_length=100) 

```

在这种情况下，我们将`max_length`字段选项设置为 100，以将我们的书名限制为 100 个字符。

字段类的完整列表按字母顺序排列在*表 A.1*中。

| **字段** | **默认小部件** | **描述** |
| --- | --- | --- |
| `AutoField` | N/A | 根据可用 ID 自动递增的`IntegerField`。 |
| `BigIntegerField` | `NumberInput` | 64 位整数，类似于`IntegerField`，只是它保证适合从`-9223372036854775808`到`9223372036854775807`的数字 |
| `BinaryField` | N/A | 用于存储原始二进制数据的字段。它只支持`bytes`赋值。请注意，此字段功能有限。 |
| `BooleanField` | `CheckboxInput` | 真/假字段。如果需要接受`null`值，则使用`NullBooleanField`。 |
| `CharField` | `TextInput` | 用于小到大的字符串的字符串字段。对于大量的文本，请使用`TextField`。`CharField`有一个额外的必需参数：`max_length`。字段的最大长度（以字符为单位）。 |
| `DateField` | `DateInput` | 日期，在 Python 中由`datetime.date`实例表示。有两个额外的可选参数：`auto_now`，每次保存对象时自动将字段设置为现在，`auto_now_add`，在对象首次创建时自动将字段设置为现在。 |
| `DateTimeField` | `DateTimeInput` | 日期和时间，在 Python 中由`datetime.datetime`实例表示。接受与`DateField`相同的额外参数。 |
| `DecimalField` | `TextInput` | 固定精度的十进制数，在 Python 中由`Decimal`实例表示。有两个必需的参数：`max_digits`和`decimal_places`。 |
| `DurationField` | `TextInput` | 用于存储时间段的字段-在 Python 中由`timedelta`建模。 |
| `EmailField` | `TextInput` | 使用`EmailValidator`验证输入的`CharField`。`max_length`默认为`254`。 |
| `FileField` | `ClearableFileInput` | 文件上传字段。有关`FileField`的更多信息，请参见下一节。 |
| `FilePathField` | `Select` | `CharField`，其选择限于文件系统上某个目录中的文件名。 |
| `FloatField` | `NumberInput` | 由 Python 中的`float`实例表示的浮点数。注意，当`field.localize`为`False`时，默认小部件是`TextInput` |
| `ImageField` | `ClearableFileInput` | 继承自`FileField`的所有属性和方法，但也验证上传的对象是否是有效的图像。额外的`height`和`width`属性。需要在 http://pillow.readthedocs.org/en/latest/上可用的 Pillow 库。 |
| `IntegerField` | `NumberInput` | 一个整数。在 Django 支持的所有数据库中，从`-2147483648`到`2147483647`的值都是安全的。 |
| `GenericIPAddressField` | `TextInput` | 一个 IPv4 或 IPv6 地址，以字符串格式表示（例如，`192.0.2.30`或`2a02:42fe::4`）。 |
| `NullBooleanField` | `NullBooleanSelect` | 像`BooleanField`，但允许`NULL`作为其中一个选项。 |
| `PositiveIntegerField` | `NumberInput` | 一个整数。在 Django 支持的所有数据库中，从`0`到`2147483647`的值都是安全的。 |
| `SlugField` | `TextInput` | Slug 是一个报纸术语。Slug 是某物的一个简短标签，只包含字母、数字、下划线或连字符。 |
| `SmallIntegerField` | `NumberInput` | 像`IntegerField`，但只允许在某个点以下的值。在 Django 支持的所有数据库中，从`-32768`到`32767`的值都是安全的。 |
| `TextField` | `Textarea` | 一个大文本字段。如果指定了`max_length`属性，它将反映在自动生成的表单字段的`Textarea`小部件中。 |
| `TimeField` | `TextInput` | 一个时间，由 Python 中的`datetime.time`实例表示。 |
| `URLField` | `URLInput` | 用于 URL 的`CharField`。可选的`max_length`参数。 |
| `UUIDField` | `TextInput` | 用于存储通用唯一标识符的字段。使用 Python 的`UUID`类。 |

表 A.1：Django 模型字段参考

## FileField 注意事项

不支持`primary_key`和`unique`参数，如果使用将会引发`TypeError`。

+   有两个可选参数：FileField.upload_to

+   `FileField.storage`

### FileField FileField.upload_to

一个本地文件系统路径，将被附加到您的`MEDIA_ROOT`设置，以确定`url`属性的值。这个路径可能包含`strftime()`格式，它将被文件上传的日期/时间替换（这样上传的文件不会填满给定的目录）。这也可以是一个可调用的，比如一个函数，它将被调用来获取上传路径，包括文件名。这个可调用必须能够接受两个参数，并返回一个 Unix 风格的路径（带有正斜杠），以便传递给存储系统。

将传递的两个参数是：

+   **实例：**模型的一个实例，其中定义了 FileField。更具体地说，这是当前文件被附加的特定实例。在大多数情况下，这个对象还没有保存到数据库中，所以如果它使用默认的`AutoField`，它可能还没有主键字段的值。

+   **文件名：**最初给定的文件名。在确定最终目标路径时可能会考虑这个文件名。

### FileField.storage

一个存储对象，用于处理文件的存储和检索。这个字段的默认表单小部件是`ClearableFileInput`。在模型中使用`FileField`或`ImageField`（见下文）需要几个步骤：

+   在您的设置文件中，您需要将`MEDIA_ROOT`定义为一个目录的完整路径，您希望 Django 存储上传的文件在其中。（出于性能考虑，这些文件不存储在数据库中。）将`MEDIA_URL`定义为该目录的基本公共 URL。确保这个目录对 Web 服务器的用户帐户是可写的。

+   将`FileField`或`ImageField`添加到您的模型中，定义`upload_to`选项以指定`MEDIA_ROOT`的子目录，用于上传文件。

+   在数据库中存储的只是文件的路径（相对于 `MEDIA_ROOT`）。您很可能会想要使用 Django 提供的便捷的 `url` 属性。例如，如果您的 `ImageField` 名为 `mug_shot`，您可以在模板中使用 `{{ object.mug_shot.url }}` 获取图像的绝对路径。

请注意，每当处理上传的文件时，都应该密切关注您上传文件的位置和文件类型，以避免安全漏洞。验证所有上传的文件，以确保文件是您认为的文件。例如，如果您盲目地让某人上传文件，而没有进行验证，到您的 Web 服务器文档根目录中，那么某人可能会上传一个 CGI 或 PHP 脚本，并通过访问其 URL 在您的网站上执行该脚本。不要允许这种情况发生。

还要注意，即使是上传的 HTML 文件，由于浏览器可以执行它（尽管服务器不能），可能会带来等同于 XSS 或 CSRF 攻击的安全威胁。`FileField` 实例在数据库中以 `varchar` 列的形式创建，具有默认的最大长度为 100 个字符。与其他字段一样，您可以使用 `max_length` 参数更改最大长度。

### FileField 和 FieldFile

当您在模型上访问 `FileField` 时，会得到一个 `FieldFile` 的实例，作为访问底层文件的代理。除了从 `django.core.files.File` 继承的功能外，此类还具有几个属性和方法，可用于与文件数据交互：

#### FieldFile.url

通过调用底层 `Storage` 类的 `url()` 方法来访问文件的相对 URL 的只读属性。

#### FieldFile.open(mode='rb')

行为类似于标准的 Python `open()` 方法，并以 `mode` 指定的模式打开与此实例关联的文件。

#### FieldFile.close()

行为类似于标准的 Python `file.close()` 方法，并关闭与此实例关联的文件。

#### FieldFile.save(name, content, save=True)

此方法接受文件名和文件内容，并将它们传递给字段的存储类，然后将存储的文件与模型字段关联起来。如果您想手动将文件数据与模型上的 `FileField` 实例关联起来，可以使用 `save()` 方法来持久化该文件数据。

需要两个必需参数：`name` 是文件的名称，`content` 是包含文件内容的对象。可选的 `save` 参数控制在更改与此字段关联的文件后是否保存模型实例。默认为 `True`。

请注意，`content` 参数应该是 `django.core.files.File` 的实例，而不是 Python 的内置文件对象。您可以像这样从现有的 Python 文件对象构造一个 `File`：

```py
from django.core.files import File 
# Open an existing file using Python's built-in open() 
f = open('/tmp/hello.world') 
myfile = File(f) 

```

或者您可以像这样从 Python 字符串构造一个：

```py
from django.core.files.base import ContentFile 
myfile = ContentFile("hello world") 

```

#### FieldFile.delete(save=True)

删除与此实例关联的文件并清除字段上的所有属性。如果在调用 `delete()` 时文件处于打开状态，此方法将关闭文件。

可选的 `save` 参数控制在删除与此字段关联的文件后是否保存模型实例。默认为 `True`。

请注意，当模型被删除时，相关文件不会被删除。如果您需要清理孤立的文件，您需要自行处理（例如，使用自定义的管理命令，可以手动运行或通过例如 `cron` 定期运行）。

# 通用字段选项

*表 A.2* 列出了 Django 中所有字段类型的所有可选字段参数。

| 选项 | 描述 |
| --- | --- |
| `null` | 如果为 `True`，Django 将在数据库中将空值存储为 `NULL`。默认为 `False`。避免在诸如 `CharField` 和 `TextField` 等基于字符串的字段上使用 `null`，因为空字符串值将始终被存储为空字符串，而不是 `NULL`。对于基于字符串和非基于字符串的字段，如果希望在表单中允许空值，还需要设置 `blank=True`。如果要接受带有 `BooleanField` 的 `null` 值，请改用 `NullBooleanField`。 |
| `blank` | 如果为 `True`，则允许该字段为空。默认为 `False`。请注意，这与 `null` 是不同的。`null` 纯粹是与数据库相关的，而 `blank` 是与验证相关的。 |
| `choices` | 一个可迭代对象（例如列表或元组），其中包含正好两个项的可迭代对象（例如 `[(A, B), (A, B) ...]`），用作此字段的选择。如果给出了这个选项，默认的表单小部件将是一个带有这些选择的选择框，而不是标准文本字段。每个元组中的第一个元素是要在模型上设置的实际值，第二个元素是人类可读的名称。 |
| `db_column` | 用于此字段的数据库列的名称。如果没有给出，Django 将使用字段的名称。 |
| `db_index` | 如果为 `True`，将为此字段创建数据库索引。 |
| `db_tablespace` | 用于此字段索引的数据库表空间的名称，如果此字段已被索引。默认值是项目的 `DEFAULT_INDEX_TABLESPACE` 设置（如果设置了），或者模型的 `db_tablespace`（如果有）。如果后端不支持索引的表空间，则将忽略此选项。 |
| `default` | 该字段的默认值。这可以是一个值或一个可调用对象。如果是可调用的，它将在创建新对象时每次被调用。默认值不能是可变对象（模型实例、列表、集合等），因为在所有新模型实例中将使用对该对象的相同实例的引用作为默认值。 |
| `editable` | 如果为 `False`，该字段将不会显示在管理界面或任何其他 `ModelForm` 中。它们也会在模型验证期间被跳过。默认为 `True`。 |
| `error_messages` | `error_messages` 参数允许您覆盖字段将引发的默认消息。传入一个字典，其中键与您想要覆盖的错误消息相匹配。错误消息键包括 `null`、`blank`、`invalid`、`invalid_choice`、`unique` 和 `unique_for_date`。 |
| `help_text` | 要与表单小部件一起显示的额外帮助文本。即使您的字段在表单上没有使用，这也是有用的文档。请注意，此值在自动生成的表单中 *不* 是 HTML 转义的。这样，如果您愿意，可以在 `help_text` 中包含 HTML。 |
| `primary_key` | 如果为 `True`，则该字段是模型的主键。如果您没有为模型中的任何字段指定 `primary_key=True`，Django 将自动添加一个 `AutoField` 来保存主键，因此您不需要在任何字段上设置 `primary_key=True`，除非您想要覆盖默认的主键行为。主键字段是只读的。 |
| `unique` | 如果为 `True`，则此字段必须在整个表中是唯一的。这是在数据库级别和模型验证期间强制执行的。此选项对除 `ManyToManyField`、`OneToOneField` 和 `FileField` 之外的所有字段类型都有效。 |
| `unique_for_date` | 将其设置为 `DateField` 或 `DateTimeField` 的名称，以要求此字段对于日期字段的值是唯一的。例如，如果有一个字段 `title`，其 `unique_for_date="pub_date"`，那么 Django 将不允许输入具有相同 `title` 和 `pub_date` 的两条记录。这是在模型验证期间由 `Model.validate_unique()` 强制执行的，但不是在数据库级别上。 |
| `unique_for_month` | 类似于 `unique_for_date`，但要求该字段相对于月份是唯一的。 |
| `unique_for_year` | 类似于 `unique_for_date`，但要求该字段相对于年份是唯一的。 |
| `verbose_name` | 字段的可读名称。如果未给出详细名称，Django 将使用字段的属性名称自动创建它，将下划线转换为空格。 |
| `validators` | 一个要为此字段运行的验证器列表。 |

表 A.2：Django 通用字段选项

# 字段属性引用

每个`Field`实例都包含几个属性，允许内省其行为。在需要编写依赖于字段功能的代码时，请使用这些属性，而不是`isinstance`检查。这些属性可以与`Model._meta` API 一起使用，以缩小对特定字段类型的搜索。自定义模型字段应实现这些标志。

## 字段属性

### Field.auto_created

布尔标志，指示字段是否自动创建，例如模型继承中使用的`OneToOneField`。

### Field.concrete

布尔标志，指示字段是否与数据库列关联。

### Field.hidden

布尔标志，指示字段是否用于支持另一个非隐藏字段的功能（例如，构成`GenericForeignKey`的`content_type`和`object_id`字段）。`hidden`标志用于区分模型上的字段的公共子集与模型上的所有字段。

### Field.is_relation

布尔标志，指示字段是否包含对一个或多个其他模型的引用，以实现其功能（例如，`ForeignKey`，`ManyToManyField`，`OneToOneField`等）。

### Field.model

返回定义字段的模型。如果字段在模型的超类上定义，则`model`将引用超类，而不是实例的类。

## 具有关系的字段属性

这些属性用于查询关系的基数和其他细节。这些属性存在于所有字段上；但是，只有在字段是关系类型（`Field.is_relation=True`）时，它们才会有有意义的值。

### Field.many_to_many

布尔标志，如果字段具有多对多关系，则为`True`；否则为`False`。Django 中唯一包含此标志为`True`的字段是`ManyToManyField`。

### Field.many_to_one

布尔标志，如果字段具有多对一关系（例如`ForeignKey`），则为`True`；否则为`False`。

### Field.one_to_many

布尔标志，如果字段具有一对多关系（例如`GenericRelation`或`ForeignKey`的反向关系），则为`True`；否则为`False`。

### Field.one_to_one

布尔标志，如果字段具有一对一关系（例如`OneToOneField`），则为`True`；否则为`False`。

### Field.related_model

指向字段相关的模型。例如，在`ForeignKey(Author)`中的`Author`。如果字段具有通用关系（例如`GenericForeignKey`或`GenericRelation`），则`related_model`将为`None`。

# 关系

Django 还定义了一组表示关系的字段。

## ForeignKey

多对一关系。需要一个位置参数：模型相关的类。要创建递归关系（与自身具有多对一关系的对象），请使用`models.ForeignKey('self')`。

如果需要在尚未定义的模型上创建关系，可以使用模型的名称，而不是模型对象本身：

```py
from django.db import models 

class Car(models.Model): 
    manufacturer = models.ForeignKey('Manufacturer') 
    # ... 

class Manufacturer(models.Model): 
    # ... 
    pass 

```

要引用另一个应用程序中定义的模型，可以明确指定具有完整应用程序标签的模型。例如，如果上面的`Manufacturer`模型在另一个名为`production`的应用程序中定义，则需要使用：

```py
class Car(models.Model): 
    manufacturer = models.ForeignKey('production.Manufacturer') 

```

在两个应用程序之间解析循环导入依赖关系时，这种引用可能很有用。在`ForeignKey`上自动创建数据库索引。您可以通过将`db_index`设置为`False`来禁用此功能。

如果您创建外键以确保一致性而不是连接，或者如果您将创建替代索引（如部分索引或多列索引），则可能希望避免索引的开销。

### 数据库表示

在幕后，Django 将`字段名`附加`"_id"`以创建其数据库列名。在上面的示例中，`Car`模型的数据库表将具有`manufacturer_id`列。

您可以通过指定`db_column`来明确更改这一点，但是，除非编写自定义 SQL，否则您的代码不应该处理数据库列名。您将始终处理模型对象的字段名称。

### 参数

`ForeignKey`接受一组额外的参数-全部是可选的-用于定义关系的详细信息。

#### limit_choices_to

设置此字段的可用选择的限制，当使用`ModelForm`或管理员渲染此字段时（默认情况下，查询集中的所有对象都可供选择）。可以使用字典、`Q`对象或返回字典或`Q`对象的可调用对象。例如：

```py
staff_member = models.ForeignKey(User, limit_choices_to={'is_staff': True}) 

```

导致`ModelForm`上的相应字段仅列出`is_staff=True`的`Users`。这在 Django 管理员中可能会有所帮助。可调用形式可能会有所帮助，例如，当与 Python `datetime`模块一起使用以限制日期范围的选择时。例如：

```py
def limit_pub_date_choices(): 
    return {'pub_date__lte': datetime.date.utcnow()} 
limit_choices_to = limit_pub_date_choices 

```

如果`limit_choices_to`是或返回`Q 对象`，对于复杂查询很有用，那么它只会影响在模型的`ModelAdmin`中未列出`raw_id_fields`时管理员中可用的选择。

#### related_name

用于从相关对象返回到此对象的关系的名称。这也是`related_query_name`的默认值（从目标模型返回的反向过滤器名称）。有关完整说明和示例，请参阅相关对象文档。请注意，在定义抽象模型上的关系时，必须设置此值；在这样做时，一些特殊的语法是可用的。如果您希望 Django 不创建反向关系，请将`related_name`设置为`'+'`或以`'+'`结尾。例如，这将确保`User`模型不会有到此模型的反向关系：

```py
user = models.ForeignKey(User, related_name='+') 

```

#### related_query_name

用于从目标模型返回的反向过滤器名称的名称。如果设置了`related_name`，则默认为`related_name`的值，否则默认为模型的名称：

```py
# Declare the ForeignKey with related_query_name 
class Tag(models.Model): 
    article = models.ForeignKey(Article, related_name="tags",
      related_query_name="tag") 
    name = models.CharField(max_length=255) 

# That's now the name of the reverse filter 
Article.objects.filter(tag__name="important") 

```

#### to_field

关系对象上的字段。默认情况下，Django 使用相关对象的主键。

#### db_constraint

控制是否应为此外键在数据库中创建约束。默认值为`True`，这几乎肯定是您想要的；将其设置为`False`可能对数据完整性非常不利。也就是说，有一些情况下您可能希望这样做：

+   您有无效的旧数据。

+   您正在对数据库进行分片。

如果设置为`False`，访问不存在的相关对象将引发其`DoesNotExist`异常。

#### 删除时

当被`ForeignKey`引用的对象被删除时，Django 默认会模拟 SQL 约束`ON DELETE CASCADE`的行为，并删除包含`ForeignKey`的对象。可以通过指定`on_delete`参数来覆盖此行为。例如，如果您有一个可空的`ForeignKey`，并且希望在删除引用对象时将其设置为 null：

```py
user = models.ForeignKey(User, blank=True, null=True, on_delete=models.SET_NULL) 

```

`on_delete`的可能值可以在`django.db.models`中找到：

+   `CASCADE`：级联删除；默认值

+   `PROTECT`：通过引发`ProtectedError`（`django.db.IntegrityError`的子类）来防止删除引用对象

+   `SET_NULL`：将`ForeignKey`设置为 null；只有在`null`为`True`时才可能

+   `SET_DEFAULT`：将`ForeignKey`设置为其默认值；必须设置`ForeignKey`的默认值

#### 可交换

控制迁移框架对指向可交换模型的此`ForeignKey`的反应。如果为`True`-默认值-那么如果`ForeignKey`指向与当前`settings.AUTH_USER_MODEL`的值（或其他可交换模型设置）匹配的模型，则关系将在迁移中使用对设置的引用而不是直接对模型进行存储。

只有在确定模型应始终指向替换模型时才要将其覆盖为`False`，例如，如果它是专门为自定义用户模型设计的配置文件模型。将其设置为`False`并不意味着即使替换了模型，也可以引用可交换模型-`False`只是意味着使用此`ForeignKey`进行的迁移将始终引用您指定的确切模型（例如，如果用户尝试使用您不支持的用户模型，则会严重失败）。如果有疑问，请将其保留为默认值`True`。

## ManyToManyField

多对多关系。需要一个位置参数：模型相关的类，其工作方式与`ForeignKey`完全相同，包括递归和延迟关系。可以使用字段的`RelatedManager`添加、删除或创建相关对象。

### 数据库表示

在幕后，Django 创建一个中间连接表来表示多对多关系。默认情况下，此表名是使用多对多字段的名称和包含它的模型的表名生成的。

由于某些数据库不支持超过一定长度的表名，这些表名将自动截断为 64 个字符，并使用唯一性哈希。这意味着您可能会看到表名如`author_books_9cdf4`；这是完全正常的。您可以使用`db_table`选项手动提供连接表的名称。

### 参数

`ManyToManyField`接受一组额外的参数-全部是可选的-用于控制关系的功能。

#### related_name

与`ForeignKey.related_name`相同。

#### related_query_name

与`ForeignKey.related_query_name`相同。

#### limit_choices_to

与`ForeignKey.limit_choices_to`相同。当在使用`through`参数指定自定义中间表的`ManyToManyField`上使用`limit_choices_to`时，`limit_choices_to`没有效果。

#### 对称的

仅在自身的 ManyToManyFields 的定义中使用。考虑以下模型：

```py
from django.db import models 

class Person(models.Model): 
    friends = models.ManyToManyField("self") 

```

当 Django 处理此模型时，它会识别出它在自身上有一个`ManyToManyField`，因此它不会向`Person`类添加`person_set`属性。相反，假定`ManyToManyField`是对称的-也就是说，如果我是你的朋友，那么你也是我的朋友。

如果不希望在`self`的多对多关系中具有对称性，请将`symmetrical`设置为`False`。这将强制 Django 添加反向关系的描述符，从而允许`ManyToManyField`关系不对称。

#### 通过

Django 将自动生成一个表来管理多对多关系。但是，如果要手动指定中间表，可以使用`through`选项来指定表示要使用的中间表的 Django 模型。

此选项的最常见用法是当您想要将额外数据与多对多关系关联时。如果不指定显式的`through`模型，则仍然有一个隐式的`through`模型类，您可以使用它直接访问创建以保存关联的表。它有三个字段：

+   `id`：关系的主键

+   `<containing_model>_id`：声明`ManyToManyField`的模型的`id`

+   `<other_model>_id`：`ManyToManyField`指向的模型的`id`

此类可用于像普通模型一样查询给定模型实例的关联记录。

#### through_fields

仅在指定自定义中介模型时使用。Django 通常会确定中介模型的哪些字段以自动建立多对多关系。

#### db_table

用于存储多对多数据的表的名称。如果未提供此名称，Django 将基于定义关系的模型的表的名称和字段本身的名称假定默认名称。

#### db_constraint

控制是否应在中介表的外键在数据库中创建约束。默认值为`True`，这几乎肯定是您想要的；将其设置为`False`可能对数据完整性非常不利。

也就是说，以下是一些可能需要这样做的情况：

+   您有不合法的遗留数据

+   您正在对数据库进行分片

传递`db_constraint`和`through`是错误的。

#### swappable

如果此`ManyToManyField`指向可交换模型，则控制迁移框架的反应。如果为`True`-默认值-如果`ManyToManyField`指向与`settings.AUTH_USER_MODEL`（或其他可交换模型设置）的当前值匹配的模型，则关系将存储在迁移中，使用对设置的引用，而不是直接对模型。

只有在确定模型应始终指向替换模型的情况下，才希望将其覆盖为`False`-例如，如果它是专门为自定义用户模型设计的配置文件模型。如果有疑问，请将其保留为默认值`True`。`ManyToManyField`不支持`validators`。`null`没有影响，因为没有办法在数据库级别要求关系。

## OneToOneField

一对一关系。在概念上，这类似于具有`unique=True`的`ForeignKey`，但关系的反向侧将直接返回单个对象。这在作为模型的主键时最有用，该模型以某种方式扩展另一个模型；通过向子模型添加从子模型到父模型的隐式一对一关系来实现多表继承，例如。

需要一个位置参数：将与之相关的类。这与`ForeignKey`的工作方式完全相同，包括递归和延迟关系的所有选项。如果未为`OneToOneField`指定`related_name`参数，Django 将使用当前模型的小写名称作为默认值。使用以下示例：

```py
from django.conf import settings 
from django.db import models 

class MySpecialUser(models.Model): 
    user = models.OneToOneField(settings.AUTH_USER_MODEL) 
    supervisor = models.OneToOneField(settings.AUTH_USER_MODEL, 
      related_name='supervisor_of') 

```

您的生成的`User`模型将具有以下属性：

```py
>>> user = User.objects.get(pk=1)
>>> hasattr(user, 'myspecialuser')
True
>>> hasattr(user, 'supervisor_of')
True

```

当访问相关表中的条目不存在时，将引发`DoesNotExist`异常。例如，如果用户没有由`MySpecialUser`指定的主管：

```py
>>> user.supervisor_of
Traceback (most recent call last):
 ...
DoesNotExist: User matching query does not exist.

```

此外，`OneToOneField`接受`ForeignKey`接受的所有额外参数，以及一个额外参数：

### parent_link

当在继承自另一个具体模型的模型中使用时，`True`表示应使用此字段作为返回到父类的链接，而不是通常通过子类隐式创建的额外`OneToOneField`。有关`OneToOneField`的用法示例，请参见下一章中的*一对一关系*。

# 模型元数据选项

*表 A.3*是您可以在其内部`class Meta`中为模型提供的完整模型元选项列表。有关每个元选项的更多详细信息以及示例，请参阅 Django 文档[`docs.djangoproject.com/en/1.8/ref/models/options/`](https://docs.djangoproject.com/en/1.8/ref/models/options/)。

| **选项** | **说明** |
| --- | --- |
| `abstract` | 如果`abstract = True`，此模型将是一个抽象基类。 |
| `app_label` | 如果模型在`INSTALLED_APPS`之外定义，它必须声明属于哪个应用程序。 |
| `db_table` | 用于模型的数据库表的名称。 |
| - `db_tablespace` | 用于此模型的数据库表空间的名称。如果设置了项目的 `DEFAULT_TABLESPACE` 设置，则默认为该设置。如果后端不支持表空间，则忽略此选项。 |
| - `default_related_name` | 从相关对象返回到此对象的关系的默认名称。默认为 `<model_name>_set`。 |
| - `get_latest_by` | 模型中可排序字段的名称，通常为 `DateField`、`DateTimeField` 或 `IntegerField`。 |
| - `managed` | 默认为 `True`，意味着 Django 将在`migrate`或作为迁移的一部分中创建适当的数据库表，并在`flush`管理命令的一部分中删除它们。 |
| - `order_with_respect_to` | 标记此对象相对于给定字段是可排序的。 |
| - `ordering` | 对象的默认排序，用于获取对象列表时使用。 |
| - `permissions` | 创建此对象时要输入权限表的额外权限。 |
| - `default_permissions` | 默认为 `('add', 'change', 'delete')`。 |
| - `proxy` | 如果 `proxy = True`，则子类化另一个模型的模型将被视为代理模型。 |
| - `select_on_save` | 确定 Django 是否使用 pre-1.6 `django.db.models.Model.save()` 算法。 |
| - `unique_together` | 一起使用的字段集，必须是唯一的。 |
| - `index_together` | 一起使用的字段集，被索引。 |
| - `verbose_name` | 对象的可读名称，单数形式。 |
| - `verbose_name_plural` | 对象的复数名称。 |

表 A.3：模型元数据选项
