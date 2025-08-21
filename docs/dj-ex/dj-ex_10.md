# 第十章：构建一个在线学习平台

在上一章中，你为在线商店项目添加了国际化。你还构建了一个优惠券系统和一个商品推荐引擎。在本章中，你会创建一个新的项目。你会构建一个在线学习平台，这个平台会创建一个自定义的内容管理系统。

在本章中，你会学习如何：

- 为模型创建 fixtures
- 使用模型继承
- 创建自定义 O 型字典
- 使用基于类的视图和 mixins
- 构建表单集
- 管理组和权限
- 创建一个内容管理系统

## 10.1 创建一个在线学习平台

我们最后一个实战项目是一个在线学习平台。在本章中，我们会构建一个灵活的内容管理系统（CMS），允许教师创建课程和管理课程内容。

首先，我们用以下命令为新项目创建一个虚拟环境，并激活它：

```py
mkdir env
virtualenv env/educa
source env/educa/bin/activate
```

用以下命令在虚拟环境中安装 Django：

```py
pip install Django
```

我们将在项目中管理图片上传，所以我们还需要用以下命令安装 Pillow：

```py
pip install Pillow
```

使用以下命令创建一个新项目：

```py
django-admin startproject educa
```

进入新的`educa`目录，并用以下命令创建一个新应用：

```py
cd educa
django-admin startapp courses
```

编辑`educa`项目的`settings.py`文件，把`courses`添加到`INSTALLED_APPS`设置中：

```py
INSTALLED_APPS = [
    'courses',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

现在`courses`应用已经在项目激活了。让我们为课程和课程内容定义模型。

## 10.2 构建课程模型

我们的在线学习平台会提供多种主题的课程。每个课程会划分为可配置的单元数量，而每个单元会包括可配置的内容数量。会有各种类型的内容：文本，文件，图片或者视频。下面这个例子展示了我们的课程目录的数据结构：

```py
Subject 1
	Course 1
		Module 1
			Content 1 (image)
			Content 3 (text)
		Module 2
			Content 4 (text)
			Content 5 (file)
			Content 6 (video)
			...
```

让我们构建课程模型。编辑`courses`应用的`models.py`文件，并添加以下代码：

```py
from django.db import models
from django.contrib.auth.models import User

class Subject(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)

    class Meta:
        ordering = ('title', )

    def __str__(self):
        return self.title

class Course(models.Model):
    owner = models.ForeignKey(User, related_name='courses_created')
    subject = models.ForeignKey(Subject, related_name='courses')
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    overview = models.TextField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created',)

    def __str__(self):
        return self.title

class Module(models.Model):
    course = models.ForeignKey(Course, related_name='modules')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.title
```

这些是初始的`Subject`，`Course`和`Module`模型。`Course`模型有以下字段：

- `owner`：创建给课程的教师
- `subject`：这个课程所属的主题。一个指向`Subject`模型的`ForeignKey`字段。
- `title`：课程标题.
- `slug`：课程别名，之后在 URL 中使用。
- `overview`：一个`TextField`列，表示课程概述。
- `created`：课程创建的日期和时间。因为设置了`auto_now_add=True`，所以创建新对象时，Django 会自动设置这个字段。

每个课程划分为数个单元。因此，`Module`模型包含一个指向`Course`模型的`ForeignKey`字段。

打开终端执行以下命令，为应用创建初始的数据库迁移：

```py
python manage.py makemigrations
```

你会看到以下输出：

```py
Migrations for 'courses':
  courses/migrations/0001_initial.py
    - Create model Course
    - Create model Module
    - Create model Subject
    - Add field subject to course
```

然后执行以下命令，同步迁移到数据库中：

```py
python manage.py migrate
```

你会看到一个输出，其中包括所有已经生效的数据库迁移，包括 Django 的数据库迁移。输出会包括这一行：

```py
Applying courses.0001_initial... OK
```

这个告诉我们，`courses`应用的模型已经同步到数据库中。
			
### 10.2.1 在管理站点注册模型

我们将把课程模型添加到管理站点。编辑`courses`应用目录中的`admin.py`文件，并添加以下代码：

```py
from django.contrib import admin
from .models import Subject, Course, Module

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['title', 'slug']
    prepopulated_fields = {'slug': ('title', )}

class ModuleInline(admin.StackedInline):
    model = Module

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ['title', 'subject', 'created']
    list_filter = ['created', 'subject']
    search_fields = ['title', 'overview']
    prepopulated_fields = {'slug': ('title', )}
    inlines = [ModuleInline]
```

现在`courses`应用的模型已经在管理站点注册。我们用`@admin.register()`装饰器代替`admin.site.register()`函数。它们的功能是一样的。

### 10.2.2 为模型提供初始数据

有时你可能希望用硬编码数据预填充数据库。这在项目创建时自动包括初始数据很有用，来替代手工添加数据。Django 自带一种简单的方式，可以从数据库中加载和转储（dump）数据到 fixtures 文件中。

Django 支持 JSON，XML 或者 YAML 格式的 fixtures。我们将创建一个 fixture，其中包括一些项目的初始`Subject`对象。

首先使用以下命令创建一个超级用户：

```py
python manage.py createsuperuser
```

然后用以下命令启动开发服务器：

```py
python manage.py runserver
```

现在在浏览器中打开`http://127.0.0.1:8000/admin/courses/subject/`。使用管理站点创建几个主题。列表显示页面如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.1.png)

在终端执行以下命令：

```py
python manage.py dumpdata courses --indent=2
```

你会看到类似这样的输出：

```py
[
{
  "model": "courses.subject",
  "pk": 1,
  "fields": {
    "title": "Programming",
    "slug": "programming"
  }
},
{
  "model": "courses.subject",
  "pk": 2,
  "fields": {
    "title": "Physics",
    "slug": "physics"
  }
},
{
  "model": "courses.subject",
  "pk": 3,
  "fields": {
    "title": "Music",
    "slug": "music"
  }
},
{
  "model": "courses.subject",
  "pk": 4,
  "fields": {
    "title": "Mathematics",
    "slug": "mathematics"
  }
}
]
```

`dumpdata`命令从数据库中转储数据到标准输出，默认用 JSON 序列化。返回的数据结构包括模型和它的字段信息，Django 可以把它加载到数据库中。

你可以给这个命令提供应用的名称，或者用`app.Model`格式指定输出数据的模型。你还可以使用`--format`标签指定格式。默认情况下，`dumpdata`输出序列化的数据到标准输出。但是，你可以使用`--output`标签指定一个输出文件。`--indent`标签允许你指定缩进。关于更多`dumpdata`的参数信息，请执行`python manage.py dumpdata --help`命令。

使用以下命令，把这个转储保存到`courses`应用的`fixtures/`目录中：

```py
mkdir courses/fixtures
python manage.py dumpdata courses --indent=2 --output=courses/fixtures/subjects.json
```

使用管理站点移除你创建的主题。然后使用以下命令把 fixture 加载到数据库中：

```py
python manage.py loaddata subjects.json
```

fixture 中包括的所有`Subject`对象已经加载到数据库中。

默认情况下，Django 在每个应用的`fixtures/`目录中查找文件，但你也可以为`loaddata`命令指定 fixture 文件的完整路径。你还可以使用`FIXTURE_DIRS`设置告诉 Django 查找 fixtures 的额外目录。

> Fixtures 不仅对初始数据有用，还可以为应用提供简单的数据，或者测试必需的数据。

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/testing/tools/#topics-testing-fixtures)阅读如何在测试中使用 fixtures。

如果你想在模型迁移中加载 fixtures，请阅读 Django 文档的数据迁移部分。记住，我们在第九章创建了自定义迁移，用于修改模型后迁移已存在的数据。你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/migrations/#data-migrations)阅读数据库迁移的文档。

## 10.3 为不同的内容创建模型

我们计划在课程模型中添加不同类型的内容，比如文本，图片，文件和视频。我们需要一个通用的数据模型，允许我们存储不同的内容。在第六章中，我们已经学习了使用通用关系创建指向任何模型对象的外键。我们将创建一个`Content`模型表示单元内容，并定义一个通过关系，关联到任何类型的内容。

编辑`courses`应用的`models.py`文件，并添加以下导入：

```py
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
```

然后在文件结尾添加以下代码：

```py
class Content(models.Model):
    module = models.ForeignKey(Module, related_name='contents')
    content_type = models.ForeignKey(ContentType)
    object_id = models.PositiveIntegerField()
    item = GenericForeignKey('content_type', 'object_id')
```

这是`Content`模型。一个单元包括多个内容，所以我们定义了一个指向`Module`模型的外键。我们还建立了一个通用关系，从代表不同内容类型的不同模型关联到对象。记住，我们需要三个不同字段来设置一个通用关系。在`Content`模型中，它们分别是：

- `content_type`：一个指向`ContentType`模型的`ForeignKey`字段。
- `object_id`：这是一个`PositiveIntegerField`，存储关联对象的主键。
- `item`：通过组合上面两个字段，指向关联对象的`GenericForeignKey`字段。

在这个模型的数据库表中，只有`content_type`和`object_id`字段有对应的列。`item`字段允许你直接检索或设置关联对象，它的功能建立在另外两个字段之上。

我们将为每种内容类型使用不同的模型。我们的内容模型会有通用字段，但它们存储的实际内容会不同。

### 10.3.1 使用模型继承

Django 支持模型继承，类似 Python 中标准类的继承。Django 为使用模型继承提供了以下三个选择：

- **抽象模型：**当你想把一些通用信息放在几个模型时很有用。不会为抽象模型创建数据库表。
- **多表模型继承：**可用于层次中每个模型本身被认为是一个完整模型的情况下。为每个模型创建一张数据库表。
- **代理模型：**当你需要修改一个模型的行为时很有用。例如，包括额外的方法，修改默认管理器，或者使用不同的元选项。不会为代理模型创建数据库表。

让我们近一步了解它们。

#### 10.3.1.1 抽象模型

一个抽象模型是一个基类，其中定义了你想在所有子模型中包括的字段。Django 不会为抽象模型创建任何数据库表。会为每个子模型创建一张数据库表，其中包括从抽象类继承的字段，和子模型中定义的字段。

要标记一个抽象模型，你需要在它的`Meta`类中包括`abstract=True`。Django 会认为它是一个抽象模型，并且不会为它创建数据库表。要创建子模型，你只需要从抽象模型继承。以下是一个`Content`抽象模型和`Text`子模型的例子：

```py
from django.db import models

class BaseContent(models.Model):
	title = models.CharField(max_length=200)
	created = models.DateTimeField(auto_now_add=True)
	
	class Meta:
		abstract = True

class Text(BaseContent):
	body = models.TextField()
```

在这个例子中，Django 只会为`Text`模型创建数据库表，其中包括`title`，`created`和`body`字段。

#### 10.3.1.2 多表模型继承

在多表继承中，每个模型都有一张相应的数据库表。Django 会在子模型中创建指向父模型的`OneToOneField`字段。

要使用多表继承，你必须从已存在模型中继承。Django 会为原模型和子模型创建数据库表。下面是一个多表继承的例子：

```py
from django.db import models

class BaseContent(models.Model):
	title = models.CharField(max_length=100)
	created = models.DateTimeField(auto_now_add=True)
	
class Text(BaseContent):
	body = models.TextField()
```

Django 会在`Text`模型中包括一个自动生成的`OneToOneField`字段，并为每个模型创建一张数据库表。

#### 10.3.1.3 代理模型

代理模型用于修改模型的行为，比如包括额外的方法或者不同的元选项。这两个模型都在原模型的数据库表上进行操作。在模型的`Meta`类中添加`proxy=True`来创建代理模型。

下面这个例子展示了如何创建一个代理模型：

```py
from django.db import models
from django.utils import timezone

class BaseContent(models.Model):
	title = models.CharField(max_length=100)
	created = models.DateTimeField(auto_now_add=True)
	
class OrderedContent(BaseContent):
	class Meta:
		proxy = True
		ordering = ['created']
		
	def create_delta(self):
		return timezone.now() - self.created
```

我们在这里定义了一个`OrderedContent`模型，它是`Content`模型的代理模型。这个模型为 QuerySet 提供了默认排序和一个额外的`created_delta()`方法。`Content`和`OrderedContent`模型都在同一张数据库表上操作，并且可以用 ORM 通过任何一个模型访问对象。

### 10.3.2 创建内容模型

`courses`应用的`Content`模型包含一个通用关系来关联不同的内容类型。我们将为每种内容模型创建不用的模型。所有内容模型会有一些通用的字段，和一些额外字段存储自定义数据。我们将创建一个抽象模型，它会为所有内容模型提供通用字段。

编辑`courses`应用的`models.py`文件，并添加以下代码：

```py
class ItemBase(models.Model):
    owner = models.ForeignKey(User, related_name='%(class)s_related')
    title = models.CharField(max_length=250)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def __str__(self):
        return self.title

class Text(ItemBase):
    content = models.TextField()

class File(ItemBase):
    file = models.FileField(upload_to='files')

class Image(ItemBase):
    file = models.FileField(upload_to='images')

class Video(ItemBase):
    url = models.URLField()
```

在这段代码中，我们定义了一个`ItemBase`抽象模型。因此我们在`Meta`类中设置了`abstract=True`。在这个模型中，我们定义了`owner`，`title`，`created`和`updated`字段。这些通用字段会用于所有内容类型。`owner`字段允许我们存储哪个用户创建了内容。因为这个字段在抽象类中定义，所以每个子模型需要不同的`related_name`。Django 允许我们在`related_name`属性中为模型的类名指定占位符，比如`%(class)s`。这样，每个子模型的`related_name`会自动生成。因为我们使用`%(class)s_related`作为`related_name`，所以每个子模型对应的反向关系是`text_related`，`file_related`，`image_related`和`video_related`。

我们定义了四个从`ItemBase`抽象模型继承的内容模型。分别是：

- `Text`：存储文本内容。
- `File`：存储文件，比如 PDF。
- `Image`：存储图片文件。
- `Video`：存储视频。我们使用`URLField`字段来提供一个视频的 URL，从而可以嵌入视频。

除了自身的字段，每个子模型还包括`ItemBase`类中定义的字段。会为`Text`，`File`，`Image`和`Video`模型创建对应的数据库表。因为`ItemBase`是一个抽象模型，所以它不会关联到数据库表。

编辑你之前创建的`Content`模型，修改它的`content_type`字段：

```py
content_type = models.ForeignKey(
    ContentType,
    limit_choices_to = {
        'model__in': ('text', 'video', 'image', 'file')
    }
)
```

我们添加了`limit_choices_to`参数来限制`ContentType`对象可用于的通用关系。我们使用了`model__in`字段查找，来过滤`ContentType`对象的`model`属性为`text`，`video`，`image`或者`file`。

让我们创建包括新模型的数据库迁移。在命令行中执行以下命令：

```py
python manage.py makemigrations
```

你会看到以下输出：

```py
Migrations for 'courses':
  courses/migrations/0002_content_file_image_text_video.py
    - Create model Content
    - Create model File
    - Create model Image
    - Create model Text
    - Create model Video
```

然后执行以下命令应用新的数据库迁移：

```py
python manage.py migrate
```

你看到的输出的结尾是：

```py
Running migrations:
  Applying courses.0002_content_file_image_text_video... OK
```

我们已经创建了模型，可以添加不同内容到课程单元中。但是我们的模型仍然缺少了一些东西。课程单元和内容应用遵循特定的顺序。我们需要一个字段对它们进行排序。

## 10.4 创建自定义模板字段

Django 自带一组完整的模块字段，你可以用它们构建自己的模型。但是，你也可以创建自己的模型字段来存储自定义数据，或者修改已存在字段的行为。

我们需要一个字段指定对象的顺序。如果你想用 Django 提供的字段，用一种简单的方式实现这个功能，你可能会想在模型中添加一个`PositiveIntegerField`。这是一个好的开始。我们可以创建一个从`PositiveIntegerField`继承的自定义字段，并提供额外的方法。

我们会在排序字段中添加以下两个功能：

- 没有提供特定序号时，自动分配一个序号。如果存储对象时没有提供序号，我们的字段会基于最后一个已存在的排序对象，自动分配下一个序号。如果两个对象的序号分别是 1 和 2，保存第三个对象时，如果没有给定特定序号，我们应该自动分配为序号 3。
- 相对于其它字段排序对象。课程单元将会相对于它们所属的课程排序，而模块内容会相对于它们所属的单元排序。

在`courses`应用目录中创建一个`fields.py`文件，并添加以下代码：

```py
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

class OrderField(models.PositiveIntegerField):
    def __init__(self, for_fields=None, *args, **kwargs):
        self.for_fields = for_fields
        super().__init__(*args, **kwargs)

    def pre_save(self, model_instance, add):
        if getattr(model_instance, self.attname) is None:
            # no current value
            try:
                qs = self.model.objects.all()
                if self.for_fields:
                    # filter by objects with the same field values
                    # for the fields in "for_fields"
                    query = {field: getattr(model_instance, field) for field in self.for_fields}
                    qs = qs.filter(**query)
                # get the order of the last item
                last_item = qs.latest(self.attname)
                value = last_item.order + 1
            except ObjectDoesNotExist:
                value = 0
            setattr(model_instance, self.attname, value)
            return value
        else:
            return super().pre_save(model_instance, add)
```

这是我们自定义的`OrderField`。它从 Django 提供的`PositiveIntegerField`字段继承。我们的`OrderField`字段有一个可选的`for_fields`参数，允许我们指定序号相对于哪些字段计算。

我们的字段覆写了`PositiveIntegerField`字段的`pre_save()`方法，它会在该字段保存到数据库中之前执行。我们在这个方法中执行以下操作：

1. 我们检查模型实例中是否已经存在这个字段的值。我们使用`self.attname`，这是模型中指定的这个字段的属性名。如果属性的值不是`None`，我们如下计算序号：

 - 我们构建一个`QuerySet`检索这个字段模型所有对象。我们通过访问`self.model`检索字段所属的模型类。 
 - 我们用定义在字段的`for_fields`参数中的模型字段（如果有的话）的当前值过滤`QuerySet`。这样，我们就能相对于给定字段计算序号。
 - 我们用`last_item = qs.lastest(self.attname)`从数据库中检索序号最大的对象。如果没有找到对象，我们假设它是第一个对象，并分配序号 0。
 - 如果找到一个对象，我们在找到的最大序号上加 1。
 - 我们用`setattr()`把计算的序号分配给模型实例中的字段值，并返回这个值。

2. 如果模型实例有当前字段的值，则什么都不做。

> 当你创建自定义模型字段时，让它们是通用的。避免分局特定模型或字段硬编码数据。你的字段应该可以用于所有模型。

你可以在[这里](https://docs.djangoproject.com/en/1.11/howto/custom-model-fields/)阅读更多关于编写自定义模型字段的信息。

让我们在模型中添加新字段。编辑`courses`应用的`models.py`文件，并导入新的字段：

```py
from .fields import OrderField
```

然后在`Module`模型中添加`OrderField`字段：

```py
order = OrderField(blank=True, for_fields=['course'])
```

我们命名新字段为`order`，并通过设置`for_fields=['course']`，指定相对于课程计算序号。这意味着一个新单元会分配给同一个`Course`对象中最新的单元加 1。现在编辑`Module`模型的`__str__()`方法，并如下引入它的序号：

```py
def __str__(self):
    return '{}. {}'.format(self.order, self.title)
```

单元内容也需要遵循特定序号。在`Content`模型中添加一个`OrderField`字段：

```py
order = OrderField(blank=True, for_fields=['module'])
```

这次我们指定序号相对于`module`字段计算。最后，让我们为两个模型添加默认排序。在`Module`和`Content`模型中添加以下`Meta`类：

```py
class Meta:
    ordering = ['order']
```

现在`Module`和`Content`模型看起来是这样的：

```py
class Module(models.Model):
    course = models.ForeignKey(Course, related_name='modules')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    order = OrderField(blank=True, for_fields=['course'])

    class Meta:
        ordering = ['order']

    def __str__(self):
        return '{}. {}'.format(self.order, self.title)

class Content(models.Model):
    module = models.ForeignKey(Module, related_name='contents')
    content_type = models.ForeignKey(
        ContentType,
        limit_choices_to = {
            'model__in': ('text', 'video', 'image', 'file')
        }
    )
    object_id = models.PositiveIntegerField()
    item = GenericForeignKey('content_type', 'object_id')
    order = OrderField(blank=True, for_fields=['module'])

    class Meta:
        ordering = ['order']
```

让我们创建反映新序号字段的模型迁移。打开终端，并执行以下命令：

```py
python manage.py makemigrations courses
```

你会看到以下输出：

```py
You are trying to add a non-nullable field 'order' to content without a default; we can't do that (the database needs something to populate existing rows).
Please select a fix:
 1) Provide a one-off default now (will be set on all existing rows with a null value for this column)
 2) Quit, and let me add a default in models.py
Select an option:
```

Django 告诉我们，因为我们在已存在的模型中添加了新字段，所以必须为数据库中已存在的行提供默认值。如果字段有`null=True`，则可以接受空值，并且 Django 创建迁移时不要求提供默认值。我们可以指定一个默认值，或者取消数据库迁移，并在创建迁移之前在`models.py`文件的`order`字段中添加`default`属性。

输入`1`，然后按下`Enter`，为已存在的记录提供一个默认值。你会看到以下输出：

```py
Please enter the default value now, as valid Python
The datetime and django.utils.timezone modules are available, so you can do e.g. timezone.now
Type 'exit' to exit this prompt
>>>
```

输入`0`作为已存在记录的默认值，然后按下`Enter`。Django 还会要求你为`Module`模型提供默认值。选择第一个选项，然后再次输入`0`作为默认值。最后，你会看到类似这样的输出：

```py
Migrations for 'courses':
  courses/migrations/0003_auto_20170518_0743.py
    - Change Meta options on content
    - Change Meta options on module
    - Add field order to content
    - Add field order to module
```

然后执行以下命令应用新的数据库迁移：

```py
python manage.py migrate
```

这个命令的输出会告诉你迁移已经应用成功：

```py
Applying courses.0003_auto_20170518_0743... OK
```

让我们测试新字段。使用`python manage.py shell`命令打开终端，并如下创建一个新课程：

```py
>>> from django.contrib.auth.models import User
>>> from courses.models import Subject, Course, Module
>>> user = User.objects.latest('id')
>>> subject = Subject.objects.latest('id')
>>> c1 = Course.objects.create(subject=subject, owner=user, title='Course 1', slug='course1')
```

我们已经在数据库中创建了一个课程。现在，让我们添加一些单元到课程中，并查看单元序号是如何自动计算的。我们创建一个初始单元，并检查它的序号：

```py
>>> m1 = Module.objects.create(course=c1, title='Module 1')
>>> m1.order
0
```

`OrderField`设置它的值为 0，因为这是给定课程的第一个`Module`对象。现在我们创建同一个课程的第二个单元：

```py
>>> m2 = Module.objects.create(course=c1, title='Module 2')
>>> m2.order
1
```

`OrderField`在已存在对象的最大序号上加 1 来计算下一个序号。让我们指定一个特定序号来创建第三个单元：

```py
>>> m3 = Module.objects.create(course=c1, title='Module 3', order=5)
>>> m3.order
5
```

如果我们指定了自定义序号，则`OrderField`字段不会介入，并且使用给定的`order`值。

让我们添加第四个单元：

```py
>>> m4 = Module.objects.create(course=c1, title='Module 4')
>>> m4.order
6
```

这个单元的序号已经自动设置了。我们的`OrderField`字段不能保证连续的序号。但是它关注已存在的序号值，总是根据已存在的最大序号值分配下一个序号。

让我们创建第二个课程，并添加一个单元：

```py
>>> c2 = Course.objects.create(subject=subject, owner=user, title='Course 2', slug='course2')
>>> m5 = Module.objects.create(course=c2, title='Module 1')
>>> m5.order
0
```

要计算新的单元序号，该字段只考虑属于同一个课程的已存在单元。因为这个第二个课程的第一个单元，所以序号为 0。这是因为我们在`Module`模型的`order`字段中指定了`for_fields=['course']`。

恭喜你！你已经成功的创建了第一个自定义模型字段。

## 10.5 创建内容管理系统

现在我们已经创建了一个万能的数据模型，接下来我们会创建一个内容管理系统（CMS）。CMS 允许教师创建课程，并管理它们的内容。我们需要以下功能：

- 登录到 CMS
- 教师创建的课程列表
- 创建，编辑和删除课程
- 添加单元到课程，并对它们重新排序
- 添加不同类型的内容到每个单元中，并对它们重新排序

### 10.5.1 添加认证系统

我们将在平台中使用 Django 的认证框架。教师和学生都是 Django 的`User`模型的实例。因此，他们可以使用`django.contrib.auth`的认证视图登录网站。

编辑`educa`项目的主`urls.py`文件，并引入 Django 认证框架的`login`和`logout`视图：

```py
from django.conf.urls import include, url
from django.contrib import admin
from django.contrib.auth import views as auth_views

urlpatterns = [
    url(r'^accounts/login/$', auth_views.login, name='login'),
    url(r'^accounts/logout/$', auth_views.logout, name='logout'),
    url(r'^admin/', admin.site.urls),
]
```

### 10.5.2 创建认证模板

在`courses`应用目录中创建以下文件结构：

```py
templates/
	base.html
	registration/
		login.html
		logged_out.html
```

构建认证模板之前，我们需要为项目准备基础模板。编辑`base.html`模板，并添加以下内容：

```py
{% load staticfiles %}
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{% block title %}Educa{% endblock title %}</title>
    <link href="{% static "css/base.css" %}" rel="stylesheet">
</head>
<body>
    <div id="header">
        <a href="/" class="logo">Educa</a>
        <ul class="menu">
            {% if request.user.is_authenticated %}
                <li><a href="{% url "logout" %}">Sign out</a></li>
            {% else %}
                <li><a href="{% url "login" %}">Sign in</a></li>
            {% endif %}
        </ul>
    </div>
    <div id="content">
        {% block content %}
        {% endblock content %}
    </div>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
    <script>
        $(document).ready(function() {
            {% block domready %}
            {% endblock domready %}
        });
    </script>
</body>
</html>
```

这是基础模板，其它模板会从它扩展。在这个模板中，我们定义了以下块：

- `title`：其它模块用来为每个页面添加自定义标题的块。
- `content`：主要的内容块。所有扩展基础模板的模板必须在这个块中添加内容。
- `domready`：位于 jQuery 的`$(document).ready()`函数内。允许我们在 DOM 完成加载时执行代码。

这个模板中使用的 CSS 样式位于本章实例代码的`courses`应用的`static/`目录中。你可以把它拷贝到项目的相同位置。

编辑`registration/login.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Log-in{% endblock title %}

{% block content %}
    <h1>Log-in</h1>
    <div class="module">
        {% if form.errors %}
            <p>Your username and password didn't match.Please try again.</p>
        {% else %}
            <p>Please, user the following form to log-in:</p>
        {% endif %}
        <div class="login-form">
            <form action="{% url "login" %}" method="post">
                {{ form.as_p }}
                {% csrf_token %}
                <input type="hidden" name="next" value="{{ next }}" />
                <p><input type="submit" value="Log-in"></p>
            </form>
        </div>
    </div>
{% endblock content %}
```

这是 Django 的`login`视图的标准登录模板。编辑`registration/logged_out.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Logged out{% endblock title %}

{% block content %}
    <h1>Logged out</h1>
    <div class="module">
        <p>
            You have been successfully logged out. You can 
            <a href="{% url "login" %}">log-in again</a>.
        </p>
    </div>
{% endblock content %}
```

用户登出后会显示这个模板。执行`python manage.py runserver`命令启动开发服务器，然后在浏览器中打开`http://127.0.0.1:8000/accounts/login/`，你会看到以下登录页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.2.png)

### 10.5.3 创建基于类的视图

我们将构建用于创建，编辑和删除课程的视图。我们将使用基于类的视图。编辑`courses`应用的`views.py`文件，并添加以下代码：

```py
from django.views.generic.list import ListView
from .models import Course

class ManageCourseListView(ListView):
    model = Course
    template_name = 'courses/manage/course/list.html'

    def get_queryset(self):
        qs = super().get_queryset()
        return qs.filter(owner=self.request.user)
```

这是`ManageCourseListView`视图。它从 Django 的通用`ListView`继承。我们覆写了视图的`get_queryset()`方法，只检索当前用户创建的课程。要阻止用户编辑，更新或者删除不是他们创建的课程，我们还需要在创建，更新和删除视图中覆写`get_queryset()`方法。当你需要为数个基于类的视图提供特定行为，推荐方式是使用`minxins`。

### 10.5.4 为基于类的视图使用 mixins

Mixins 是一个类的特殊的多重继承。你可以用它们提供常见的离散功能，把它们添加到其它 mixins 中，允许你定义一个类的行为。有两种主要场景下使用 mixins：

- 你想为一个类提供多个可选的特性
- 你想在数个类中使用某个特性

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/class-based-views/mixins/)阅读如何在基于类的视图中使用 mixins 的文档。

Django 自带几个 mixins，为基于类的视图提供额外的功能。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/class-based-views/mixins/)找到所有 mixins。

我们将创建包括一个常见功能的 mixins 类，并把它用于课程的视图。编辑`courses`应用的`views.py`文件，如下修改：

```py
from django.core.urlresolvers import reverse_lazy
from django.views.generic.list import ListView
from django.views.generic.edit import CreateView
from django.views.generic.edit import UpdateView
from django.views.generic.edit import DeleteView
from .models import Course

class OwnerMixin:
    def get_queryset(self):
        qs = super().get_queryset()
        return qs.filter(owner=self.request.user)

class OwnerEditMixin:
    def form_valid(self, form):
        form.instance.owner = self.request.user
        return super().form_valid(form)

class OwnerCourseMixin(OwnerMixin):
    model = Course

class OwnerCourseEditMixin(OwnerCourseMixin, OwnerEditMixin):
    fields = ['subject', 'title', 'slug', 'overview']
    success_url = reverse_lazy('manage_course_list')
    template_name = 'courses/manage/course/form.html'

class ManageCourseListView(OwnerCourseMixin, ListView):
    template_name = 'courses/manage/course/list.html'

class CourseCreateView(OwnerCourseEditMixin, CreateView):
    pass

class CourseUpdateView(OwnerCourseEditMixin, UpdateView):
    pass

class CourseDeleteView(OwnerCourseMixin, DeleteView):
    template_name = 'courses/manage/course/delete.html'
    success_url = reverse_lazy('manage_course_list')
```

在这段代码中，我们创建了`OwnerMixin`和`OwnerEditMixin`两个 mixins。我们与 Django 提供的`ListView`，`CreateView`，`UpdateView`和`DeleteView`视图一起使用这些 mixins。`OwnerMixin`实现了以下方法：

- `get_queryset()`：视图用这个方法获得基本的 QuerySet。我们的 mixin 会覆写这个方法，通过`owner`属性过滤对象，来检索属于当前用户的对象（request.user）。

`OwnerEditMixin`实现以下方法：

- `form_valid()`：使用 Django 的`ModelFormMixin`的视图会使用这个方法，比如，带表单或者模型表单的视图（比如`CreateView`和`UpdateView`）。当提交的表单有效时，会执行`form_valid()`。这个方法的默认行为是保存实例（对于模型表单），并重定向用户到`success_url`。我们覆写这个方法，在被保存对象的`owner`属性中自动设置当前用户。这样，当保存对象时，我们自动设置了对象的`owner`。

我们的`OwnerMixin`类可用于与包括`owner`属性的任何模型交互的视图。

我们还定义了一个`OwnerCourseMixin`，它从`OwnerMixin`继承，并为子视图提供以下属性：

- `model`：用于 QuerySet 的模型。可以被所有视图使用。

我们用以下属性定义了一个`OwnerCourseEditMixin`：

- `fields`：模型的这个字段构建了`CreateView`和`UpdateView`视图的模型表单。
- `success_url`：当表单提交成功后，`CreateView`和`UpdateView`用它重定向用户。

最后，我们创建从`OwnerCourseMixin`继承的视图：

- `ManageCourseListView`：列出用户创建的课程。它从`OwnerCourseMixin`和`ListView`继承。
- `CourseCreateView`：用模型表单创建一个新的`Course`对象。它用在`OwnerCourseEditMixin`中定义的字段来构建模型表单，它还从`CreateView`继承。
- `CourseUpdateView`：允许编辑一个已存在的`Course`对象。它从`OwnerCourseEditMixin`和`UpdateView`继承。
- `CourseDeleteView`：从`OwnerCourseMixin`和通用的`DeleteView`继承。定义了`success_url`，用于删除对象后重定向用户。

### 10.5.5 使用组和权限

我们已经创建了管理课程的基础视图。当前，任何用户都可以访问这些视图。我们想限制这些视图，只有教师有权限创建和管理课程。Django 的认证框架包括一个权限系统，允许你给用户和组分配权限。我们将为教师用户创建一个组，并分配创建，更新和删除课程的权限。

使用`python manage.py runserver`命令启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/admin/auth/group/add/`，然后创建一个新的`Group`对象。添加组名为`Instructors`，并选择`courses`应用的所有权限，除了`Subject`模型的权限，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.3.png)

正如你所看到，每个模型有三个不同的权限：`Can add`，`Can change`和`Can delete`。为这个组选择权限后，点击`Save`按钮。

Django 自动为模型创建权限，但你也可以创建自定义权限。你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/auth/customizing/#custom-permissions)阅读更多关于添加自定义权限的信息。

打开`http://127.0.0.1:8000/admin/auth/user/add/`，然后添加一个新用户。编辑用户，并把它添加`Instructors`组，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.4.png)

用户从它所属的组中继承权限，但你也可以使用管理站点为单个用户添加独立权限。`is_superuser`设置为`True`的用户自动获得所有权限。

#### 10.5.5.1 限制访问基于类的视图

我们将限制访问视图，只有拥有适当权限的用户才可以添加，修改或删除`Course`对象。认证框架包括一个`permission_required`装饰器来限制访问视图。Django 1.9 将会包括基于类视图的权限 mixins。但是 Django 1.8 不包括它们。因此，我们将使用第三方模块`django-braces`提供的权限 mixins。

> **译者注：**现在 Django 的最新版本是 1.11.X。

Django-braces 是一个第三方模块，其中包括一组通用的 Django mixins。这些 mixins 为基于类的视图提供了额外的特性。你可以在[这里](http://django-braces.readthedocs.org/en/latest/)查看 django-braces 提供的所有 mixins。

使用`pip`命令安装 django-braces：

```py
pip install django-braces
```

我们将使用 django-braces 的两个 mixins 来限制访问视图：

- `LoginRequiredMixin`：重复`login_required`装饰器的功能。
- `PermissionRequiredMixin`：允许有特定权限的用户访问视图。记住，超级用户自动获得所有权限。

编辑`courses`应用的`views.py`文件，添加以下导入：

```py
from braces.views import LoginRequiredMixin
from braces.views import PermissionRequiredMixin
```

让`OwnerCourseMixin`从`LoginRequiredMixin`继承：

```py
class OwnerCourseMixin(OwnerMixin, LoginRequiredMixin):
    model = Course
    fields = ['subject', 'title', 'slug', 'overview']
    success_url = reverse_lazy('manage_course_list')
```

然后在创建，更新和删除视图中添加`permission_required`属性：

```py
class CourseCreateView(PermissionRequiredMixin,
                       OwnerCourseEditMixin, 
                       CreateView):
    permission_required = 'courses.add_course'

class CourseUpdateView(PermissionRequiredMixin,
                       OwnerCourseEditMixin, 
                       UpdateView):
    template_name = 'courses/manage/course/form.html'
    permission_required = 'courses.change_course'

class CourseDeleteView(PermissionRequiredMixin,
                       OwnerCourseMixin, 
                       DeleteView):
    template_name = 'courses/manage/course/delete.html'
    success_url = reverse_lazy('manage_course_list')
    permission_required = 'courses.delete_course'
```

`PermissionRequiredMixin`检查访问视图的用户是否有`permission_required`属性中之指定的权限。现在只有合适权限的用户可以访问我们的视图。

让我们为这些视图创建 URL。在`courses`应用目录中创建`urls.py`文件，并添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^mine/$', views.ManageCourseListView.as_view(), name='manage_course_list'),
    url(r'^create/$', views.CourseCreateView.as_view(), name='course_create'),
    url(r'^(?P<pk>\d+)/edit/$', views.CourseUpdateView.as_view(), name='course_edit'),
    url(r'^(?P<pk>\d+)/delete/$', views.CourseDeleteView.as_view(), name='course_delete'),
]
```

这些是列出，创建，编辑和删除课程视图的 URL 模式。编辑`educa`项目的主`urls.py`文件，在其中包括`courses`应用的 URL 模式：

```py
urlpatterns = [
    url(r'^accounts/login/$', auth_views.login, name='login'),
    url(r'^accounts/logout/$', auth_views.logout, name='logout'),
    url(r'^admin/', admin.site.urls),
    url(r'^course/', include('courses.urls')),
]
```

我们需要为这些视图创建模板。在`courses`应用的`templates/`目录中创建以下目录和文件：

```py
courses/
	manage/
		course/
			list.html
			form.html
			delete.html
```

编辑`courses/manage/course/list.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}My courses{% endblock title %}

{% block content %}
    <h1>My courses</h1>

    <div class="module">
        {% for course in object_list %}
            <div class="course-info">
                <h3>{{ course.title }}</h3>
                <p>
                    <a href="{% url "course_edit" course.id %}">Edit</a>
                    <a href="{% url "course_delete" course.id %}">Delete</a>
                </p>
            </div>
        {% empty %}
            <p>You haven't created any courses yet.</p>
        {% endfor %}
        <p>
            <a href="{% url "course_create" %}" class="button">Create new course</a>
        </p>
    </div>
{% endblock content %}
```

这是`ManageCourseListView`视图的模板。在这个模板中，我们列出了当前用户创建的课程。我们包括了编辑或删除每个课程的链接，和一个创建新课程的链接。

使用`python manage.py runserver`命令启动开发服务器。在浏览器中打开`http://127.0.0.1:8000/accounts/login/?next=/course/mine/`，并用属于`Instructors`组的用户登录。登录后，你会重定向到`http://127.0.0.1:8000/course/mine/`，如下所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.5.png)

这个页面会显示当前用户创建的所有课程。

让我们创建模板，显示创建和更新课程视图的表单。编辑`courses/manage/course/form.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    {% if object %}
        Edit course "{{ object.title }}"
    {% else %}
        Create a new course
    {% endif %}
{% endblock title %}

{% block content %}
    <h1>
        {% if object %}
            Edit course "{{ object.title }}"
        {% else %}
            Create a new course
        {% endif %}
    </h1>
    <div class="module">
        <h2>Course info</h2>
        <form action="." method="post">
            {{ form.as_p }}
            {% csrf_token %}
            <p><input type="submit" value="Save course"></p>
        </form>
    </div>
{% endblock content %}
```

`form.html`模板用于`CourseCreateView`和`CourseUpdateView`视图。在这个模板中，我们检查上下文是否存在`object`变量。如果上下文中存在`object`，我们已经正在更新一个已存在课程，并在页面标题使用它。否则，我们创建一个新的`Course`对象。

在浏览器中打开`http://127.0.0.1:8000/course/mine/`，然后点击`Create new course`。你会看到以下页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.6.png)

填写表单，然后点击`Save course`按钮。课程会被保存，并且你会被重定向到课程列表页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.7.png)

然后点击你刚创建的课程的`Edit`链接。你会再次看到表单，但这次你在编辑已存在的`Course`对象，而不是创建一个新的。

最后，编辑`courses/manage/course/delete.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Delete course{% endblock title %}

{% block content %}
    <h1>Delete course "{{ object.title }}"</h1>

    <div class="module">
        <form action="" method="post">
            {% csrf_token %}
            <p>Are you sure you want to delete "{{ object }}"?</p>
            <input type="submit" class="button" value="Confirm">
        </form>
    </div>
{% endblock content %}
```

这是`CourseDeleteView`视图的模板。这个视图从 Django 提供的`DeleteView`视图继承，它希望用户确认是否删除一个对象。

打开你的浏览器，并点击课程的`Delete`链接。你会看到以下确认页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.8.png)

点击`CONFIRM`按钮。课程会被删除，你会再次被重定向到课程列表页面。

现在教师可以创建，编辑和删除课程。下一步，我们将给教师提供一个内容管理系统，为课程添加单元和内容。我们从管理课程单元开始。

### 10.5.6 使用表单集

Django 自带一个抽象层，可以在同一个页面使用多个表单。这些表单组称为表单集（formsets）。表单集管理多个确定的`Form`或`ModelForm`实例。所有表单会一次性提交，表单集会负责处理一些事情，比如显示的初始表单数量，限制最大的提交表单数量，以及验证所有表单。

表单集包括一个`is_valide()`方法，可以一次验证所有表单。你还可以为表单提供初始数据，并指定显示多少额外的空表单。

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/forms/formsets/)进一步学习表单集，以及在[这里](https://docs.djangoproject.com/en/1.11/topics/forms/modelforms/#model-formsets)学习模型表单集。

#### 10.5.6.1 管理课程单元

因为一个课程分为多个单元，所以这里可以使用表单集。在`courses`应用目录中创建`forms.py`，并添加以下代码：

```py
from django import forms
from django.forms.models import inlineformset_factory
from .models import Course, Module

ModuleFormSet = inlineformset_factory(
    Course,
    Module,
    fields = ['title', 'description'],
    extra = 2,
    can_delete = True
)
```

这是`ModuleFormSet`表单集。我们用 Django 提供的`inlineformset_factory()`函数构建它。内联表单集（inline formsets）是表单集之上的一个小抽象，可以简化关联对象的使用。这个函数允许我们动态构建一个模型表单集，把`Module`对象关联到一个`Course`对象。

我们使用以下参数构建表单集：

- `fields`：在表单集的每个表单中包括的字段。
- `extra`：允许我们在表单集中设置两个额外的空表单。
- `can_delete`：如果设置为`True`，Django 会为每个表单包括一个布尔值字段，该字段渲染为一个复选框。它允许你标记对象为删除。

编辑`courses`应用的`views.py`，并添加以下代码：

```py
from django.shortcuts import redirect, get_object_or_404
from django.views.generic.base import TemplateResponseMixin, View
from .forms import ModuleFormSet

class CourseModuleUpdateView(TemplateResponseMixin, View):
    template_name = 'courses/manage/module/formset.html'
    course = None

    def get_formset(self, data=None):
        return ModuleFormSet(instance=self.course, data=data)

    def dispatch(self, request, pk):
        self.course = get_object_or_404(
            Course, id=pk, owner=request.user
        )
        return super().dispatch(request, pk)

    def get(self, request, *args, **kwargs):
        formset = self.get_formset()
        return self.render_to_response(
            {
                'course': self.course,
                'formset': formset
            }
        )

    def post(self, request, *args, **kwargs):
        formset = self.get_formset(data=request.POST)
        if formset.is_valid():
            formset.save()
            return redirect('manage_course_list')
        return self.render_to_response(
            {
                'course': self.course,
                'formset': formset
            }
        )
```

`CourseModuleUpdateView`视图处理表单集来添加，更新和删除指定课程的单元。这个视图从以下 mixins 和视图继承：

- `TemplateResponseMixin`：这个 mixin 负责渲染模板，并返回一个 HTTP 响应。它需要一个`template_name`属性，指定被渲染的模板，并提供`render_to_response()`方法，传入上下文参数，并渲染模板。
- `View`：Django 提供的基础的基于类的视图。

在这个视图中，我们实现了以下方法：

- `get_formset()`：我们定义这个方法，避免构建表单集的重复代码。我们用可选的`data`为给定的`Course`对象创建`ModuleFormSet`对象。
- `dispatch()`：这个方法由`View`类提供。它接收一个 HTTP 请求作为参数，并尝试委托到与使用的 HTTP 方法匹配的小写方法：GET 请求委托到`get()`方法，POST 请求委托到`post()`方法。在这个方法中，我们用`get_object_or_404()`函数获得属于当前用户，并且 ID 等于`id`参数的`Course`对象。因为 GET 和 POST 请求都需要检索课程，所以我们在`dispatch()`方法中包括这段代码。我们把它保存在视图的`course`属性，让其它方法也可以访问。
- `get()`：GET 请求时执行的方法。我们构建一个空的`ModuleFormSet`表单集，并使用`TemplateResponseMixin`提供的`render_to_response()`方法，把当前`Course`对象和表单集渲染到模板中。
- `post()`：POST 请求时执行的方法。在这个方法中，我们执行以下操作：
 1. 我们用提交的数据构建一个`ModuleFormSet`实例。
 2. 我们执行表单集的`is_valid()`方法，验证表单集的所有表单。
 3. 如果表单集有效，则调用`save()`方法保存它。此时，添加，更新或者标记删除的单元等任何修改都会应用到数据库中。然后我们重定向用户到`manage_course_list` URL。如果表单集无效，则渲染显示错误的模板。

编辑`courses`应用的`urls.py`文件，并添加以下 URL 模式：
 
```py
url(r'^(?P<pk>\d+)/module/$', views.CourseModuleUpdateView.as_view(), name='course_module_update'),
```

在`courses/manage/`模板目录中创建`module`目录。创建`courses/manage/module/formset.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    Edit "{{ course.title }}"
{% endblock title %}

{% block content %}
    <h1>Edit "{{ course.title }}"</h1>
    <div class="module">
        <h2>Course modules</h2>
        <form action="" method="post">
            {{ formset }}
            {{ formset.management_form }}
            {% csrf_token %}
            <input type="submit" class="button" value="Save modules">
        </form>
    </div>
{% endblock content %}
```

在这个模板中，我们创建了一个`<form>`元素，其中包括我们的表单集。我们还用`{{ formset.management_form }}`变量为表单集包括了管理表单。管理表单保存隐藏的字段，用于控制表单的初始数量，总数量，最小数量和最大数量。正如你所看到的，创建表单集很简单。

编辑`courses/manage/course/list.html`模板，在课程编辑和删除链接下面，为`course_module_update` URL 添加以下链接：

```py
<a href="{% url "course_edit" course.id %}">Edit</a>
<a href="{% url "course_delete" course.id %}">Delete</a>
<a href="{% url "course_module_update" course.id %}">Edit modules</a>
```

我们已经包括了编辑课程单元的链接。在浏览器中打开`http://127.0.0.1:8000/course/mine/`，然后点击一个课程的`Edit modules`链接，你会看到如图所示的表单集：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.9.png)

表单集中包括课程中每个`Module`对象的表单。在这些表单之后，显示了两个额外的空表单，这是因为我们为`ModuleFormSet`设置了`extra=2`。当你保存表单集时，Django 会包括另外两个额外字段来添加新单元。

### 10.5.7 添加内容到课程单元

现在我们需要一种添加内容到课程单元的方式。我们有四种不同类型的内容：文本，视频，图片和文件。我们可以考虑创建四个不同的视图，来为每种模型创建内容。但是我们会用更通用的方法：创建一个可以处理创建或更新任何内容模型对象的视图。

编辑`courses`应用的`views.py`文件，并添加以下代码：

```py
from django.forms.models import modelform_factory
from django.apps import apps
from .models import Module, Content

class ContentCreateUpdateView(TemplateResponseMixin, View):
    module = None
    model = None
    obj = None
    template_name = 'courses/manage/content/form.html'

    def get_model(self, model_name):
        if model_name in ['text', 'video', 'image', 'file']:
            return apps.get_model(app_label='courses', model_name=model_name)
        return None

    def get_form(self, model, *args, **kwargs):
        Form = modelform_factory(
            model,
            exclude = [
                'owner',
                'order',
                'created',
                'updated'
            ]
        )
        return Form(*args, **kwargs)

    def dispatch(self, request, module_id, model_name, id=None):
        self.module = get_object_or_404(
            Module,
            id=module_id,
            course__owner=request.user
        )
        self.model = self.get_model(model_name)
        if id:
            self.obj = get_object_or_404(
                self.model,
                id=id,
                owner=request.user
            )
        return super().dispatch(request, module_id, model_name, id)
```

这是`ContentCreateUpdateView`的第一部分。它允许我们创建和更新不同模型的内容。这个视图定义了以下方法：

- `get_model()`：在这里，我们检查给定的模型名称是否为四种内容模型之一：文本，视频，图片或文件。然后我们用 Django 的`apps.get_model()`获得给定模型名的实际类。如果给定的模型名不是四种之一，则返回`None`。
- `get_form()`：我们用表单框架的`modelform_factory()`函数动态构建表单。因为我们要为`Text`，`Video`，`Image`和`File`模型构建表单，所以我们使用`exclude`参数指定要从表单中排出的字段，而让剩下的所有字段自动包括在表单中。这样我们不用根据模型来包括字段。
- `dispatch()`：它接收以下 URL 参数，并用类属性存储相应的单元，模型和内容对象：
 * `module_id`：内容会关联的单元的 ID。
 * `model_name`：内容创建或更新的模型名。
 * `id`：被更新的对象的 ID。创建新对象时为 None。

在`ContentCreateUpdateView`类中添加以下`get()`和`post()`方法：

```py
def get(self, request, module_id, model_name, id=None):
    form = self.get_form(self.model, instance=self.obj)
    return self.render_to_response({
        'form': form,
        'object': self.obj
    })

def post(self, request, module_id, model_name, id=None):
    form = self.get_form(
        self.model,
        instance=self.obj,
        data=request.POST,
        files=request.FILES
    )
    if form.is_valid():
        obj = form.save(commit=False)
        obj.owner = request.user
        obj.save()
        if not id:
            # new content
            Content.objects.create(
                module=self.module,
                item=obj
            )
        return redirect('module_content_list', self.module.id)
    return self.render_to_response({
        'form': form,
        'object': self.obj
    })
```

这些方法分别是：

- `get()`：收到 GET 请求时执行。我们为被更新的`Text`，`Video`，`Image`或者`File`实例构建模型表单。否则我们不会传递实例来创建新对象，因为如果没有提供`id`，则`self.obj`为 None。
- `post()`：收到 POST 请求时执行。我们传递提交的所有数据和文件来构建模型表单。然后验证它。如果表单有效，我们创建一个新对象，并在保存到数据库之前把`request.user`作为它的所有者。我们检查`id`参数。如果没有提供`id`，我们知道用户正在创建新对象，而不是更新已存在的对象。如果这是一个新对象，我们为给定的单元创建一个`Content`对象，并把它关联到新的内容。

编辑`courses`应用的`urls.py`文件，并添加以下 URL 模式：

```py
url(r'^module/(?P<module_id>\d+)/content/(?P<model_name>\w+)/create/$', 
    views.ContentCreateUpdateView.as_view(), 
    name='module_content_create'),
url(r'^module/(?P<module_id>\d+)/content/(?P<model_name>\w+)/(?P<id>\d+)/$',
    views.ContentCreateUpdateView.as_view(),
    name='module_content_update'),
```

这些新的 URL 模式分别是：

- `module_content_create`：用于创建文本，视频，图片或者文件对象，并把它们添加到一个单元。它包括`module_id`和`model_name`参数。第一个参数允许我们把新内容对象链接到给定的单元。第二个参数指定了构建表单的内容模型。
- `module_content_update`：用于更新已存在的文本，视图，图片或者文件对象。它包括`module_id`和`model_name`参数，以及被更新的内容的`id`参数。

在`courses/manage/`模板目录中创建`content`目录。创建`courses/manage/content/form.html`模板，并添加以下内容：

```py
{% extends "base.html" %}

{% block title %}   
    {% if object %}
        Edit content "{{ object.title }}"
    {% else %}
        Add a new content
    {% endif %}
{% endblock title %}     

{% block content %}
    <h1>
        {% if object %}
            Edit content "{{ object.title }}"
        {% else %}
            Add a new content
        {% endif %}
    </h1>
    <div class="module">
        <h2>Course info</h2>
        <form action="" method="post" enctype="multipart/form-data">
            {{ form.as_p }}
            {% csrf_token %}
            <p><input type="submit" value="Save content"></p>
        </form>
    </div>
{% endblock content %}   
```

这是`ContentCreateUpdateView`视图的模板。在这个模板中，我们检查上下文中是否存在`object`变量。如果存在，则表示正在更新一个已存在对象。否则，表示正在创建一个新对象。

因为表单中包含一个上传的`File`和`Image`内容模型文件，所以我们在`<form>`元素中包括了`enctype="multipart/form-data`，

启动开发服务器。为已存在的课程创建一个单元，然后在浏览器中打开`http://127.0.0.1:8000/course/module/6/content/image/create/`。如果修改的话，请修改 URL 中的单元 ID。你会看到创建一个`Image`对象的表单，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.10.png)

先不要提交表单。如果你这么做了，提交会失败，因为我们还没有定义`module_content_list` URL。我们一会创建它。

我们还需要一个视图来删除内容。编辑`courses`应用的`views.py`文件，并添加以下代码：

```py
class ContentDeleteView(View):
    def post(self, request, id):
        content = get_object_or_404(
            Content,
            id=id,
            module__course__owner=request.user
        )
        module = content.module
        content.item.delete()
        content.delete()
        return redirect('module_content_list', module.id)
```

`ContentDeleteView`用给定`id`检索`Content`对象，它会删除关联的`Text`，`Video`，`Image`或`File`对象，最后删除`Content`对象，然后重定向用户到`module_content_list` URL，列出单元剩余的内容。

编辑`courses`应用的`urls.py`文件，并添加以下 URL 模式：

```py
url(r'^content/(?P<id>\d+)/delete/$', views.ContentDeleteView.as_view(), name='module_content_delete'),
```

现在，教师可以很容易的创建，更新和删除内容。

### 10.5.8 管理单元和内容

我们已经构建创建，编辑，删除课程单元和内容的视图。现在，我们需要一个显示某个课程所有单元和列出特定单元所有内容的视图。

编辑`courses`应用的`views.py`文件，并添加以下代码：

```py
class ModuleContentListView(TemplateResponseMixin, View):
    template_name = 'courses/manage/module/content_list.html'

    def get(self, request, module_id):
        module = get_object_or_404(
            Module,
            id=module_id,
            course__owner=request.user
        )
        return self.render_to_response({
            'module': module
        })
```

这是`ModuleContentListView`视图。这个视图用给定的`id`获得属于当前用户的`Module`对象，并用给定的单元渲染模板。

编辑`courses`应用的`urls.py`文件，并添加以下 URL 模式：

```py
url(r'^module/(?P<module_id>\d+)/$', 
    views.ModuleContentListView.as_view(), 
    name='module_content_list'),
```

在`templates/courses/manage/module/`目录中创建`content_list.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    Module {{ module.order|add:1 }}: {{ module.title }}
{% endblock title %}

{% block content %}
    {% with course=module.course %}
        <h1>Course: "{{ course.title }}"</h1>
        <div class="contents">
            <h3>Modules</h3>
            <ul id="modules">
                {% for m in course.modules.all %}
                    <li data-id="{{ m.id }}" {% if m == module %}class="selected"{% endif %}>
                        <a href="{% url "module_content_list" m.id %}">
                            <span>
                                Module <span class="order">{{ m.order|add:1 }}</span>
                            </span>
                            <br>
                            {{ m.title }}
                        </a>
                    </li>
                {% empty %}
                    <li>No modules yet.</li>
                {% endfor %}
            </ul>
            <p><a href="{% url "course_module_update" course.id %}">Edit modules</a></p>
        </div>
        <div class="module">
            <h2>Module {{ moudle.order|add:1 }}: {{ module.title }}</h2>
            <h3>Module contents:</h3>

            <div id="module-contents">
                {% for content in module.contents.all %}
                    <div data-id="{{ content.id }}">
                        {% with item=content.item %}
                            <p>{{ item }}</p>
                            <a href="#">Edit</a>
                            <form action="{% url "module_content_delete" content.id %}" method="post">
                                <input type="submit" value="Delete">
                                {% csrf_token %}
                            </form>
                        {% endwith %}
                    </div>
                {% empty %}
                    <p>This module has no contents yet.</p>
                {% endfor %}
            </div>
            <hr>
            <h3>Add new content:</h3>
            <ul class="content-types">
                <li><a href="{% url "module_content_create" module.id "text" %}">Text</a></li>
                <li><a href="{% url "module_content_create" module.id "image" %}">Image</a></li>
                <li><a href="{% url "module_content_create" module.id "video" %}">Video</a></li>
                <li><a href="{% url "module_content_create" module.id "file" %}">File</a></li>
            </ul>
        </div>
    {% endwith %}
{% endblock content %}
```

这个模板用于显示某个课程的所有单元，以及选定单元的内容。我们迭代课程单元，并在侧边栏显示它们。我们还迭代单元的内容，并访问`content.item`获得关联的`Text`，`Video`，`Image`或`File`对象。我们还包括一个用于创建新文本，视频，图片或文件内容的链接。

我们想知道每个对象的`item`对象的类型：`Text`，`Video`，`Image`或`File`。我们需要模型名构建编辑对象的 URL。除了这个，我们还根据内容的类型，在模板中显示每个不同的`item`。我们可以从模型的`Meta`类获得一个对象的模型（通过访问对象的`_meta`属性）。然而，Django 不允许在模板中访问下划线开头的变量或属性，来阻止访问私有数据或调到私有方法。我们可以编写一个自定义模板过滤器来解决这个问题。

在`courses`应用目录中创建以下文件结构：

```py
templatetags/
	__init__.py
	course.py
```

编辑`course.py`模块，并添加以下代码：

```py
from django import template

register = template.Library()

@register.filter
def model_name(obj):
    try:
        return obj._meta.model_name
    except AttributeError:
        return None
```

这是`model_name`模板过滤器。我们在模板中用`object|model_name`获得一个对象的模型名。

编辑`templates/courses/manage/module/content_list.html`模板，并在`{% extends %}`模板标签之后添加这一行代码：

```py
{% load course %}
```

这会加载`coursse`模板标签。然后找到以下代码：

```py
<p>{{ itme }}</p>
<a href="#">Edit</a>
```

替换为以下代码：

```py
<p>{{ itme }} ({{ item|model_name }})</p>
<a href="{% url "module_content_update" module.id item|model_name item.id %}">
    Edit
</a>
```

现在我们在模板中显示`item`模型，并用模型名构建链接来编辑对象。编辑`courses/manage/course/list.html`模板，并添加一个到`module_content_list` URL 的链接：

```py
<a href="{% url "course_module_update" course.id %}">Edit modules</a>
{% if course.modules.count > 0 %}
    <a href="{% url "module_content_list" course.modules.first.id %}">
        Manage contents
    </a>
{% endif %}
```

新链接允许用户访问课程第一个单元的内容（如果存在的话）。

在浏览器中打开`http://127.0.0.1:8000/course/mine/`，并点击至少包括一个单元的课程的`Manage contents`链接。你会看到如图所示的页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.11.png)

当你点击左边栏的单元，则会在主区域显示它的内容。模板还包括链接，用于添加文本，视频，图片或文件内容到显示的单元。添加一组不同的内容到单元中，并看一下眼结果。内容会在`Module contents`下面显示，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.12.png)

### 10.5.9 重新排序单元和内容

我们需要提供一种简单的方式对课程单元和它们的内容重新排序。我们将使用一个 JavaScript 拖放组件，让用户通过拖拽对课程的单元进行重新排序。当用户完成拖拽一个单元，我们会发起一个异步请求（AJAX）来存储新的单元序号。

我们需要一个视图接收用 JSON 编码的单元`id`的新顺序。编辑`courses`应用的`views.py`文件，并添加以下代码：

```py
from braces.views import CsrfExemptMixin
from braces.views import JsonRequestResponseMixin

class ModuleOrderView(CsrfExemptMixin, JsonRequestResponseMixin, View):
    def post(self, request):
        for id, order in self.request_json.items():
            Module.objects.filter(
                id=id,
                course__owner=request.user
            ).update(order=order)
        return self.render_json_response({
            'saved': 'OK'
        })
```

这是`ModuleOrderView`视图。我们使用了 django-braces 的以下 mixins：

- `CsrfExemptMixin`：避免在 POST 请求中检查 CSRF 令牌。我们需要它执行 AJAX POST 请求，而不用生成`csrf_token`。
- `JsonRequestResponseMixin`：解析数据为 JSON 格式，并序列化响应为 JSON，同时返回带`application/json`内容类型的 HTTP 响应。

我们可以构建一个类似的视图来排序单元的内容。在`views.py`文件中添加以下代码：

```py
class ContentOrderView(CsrfExemptMixin, JsonRequestResponseMixin, View):
    def post(self, request):
        for id, order in self.request_json.items():
            Content.objects.filter(
                id=id,
                module__course__owner=request.user
            ).update(order=order)
        return self.render_json_response({
            'saved': 'OK'
        })
```

现在编辑`courses`应用的`urls.py`文件，并添加以下 URL 模式：

```py
url(r'^module/order/$', views.ModuleOrderView.as_view(), name='module_order'),
url(r'^content/order/$', views.ContentOrderView.as_view(), name='content_order'),
```

最后，我们需要在模板中实现拖放功能。我们将使用 jQuery UI 库实现这个功能。jQuery UI 构建在 jQuery 之上，它提供了一组界面交互，效果和组件。我们将使用它的`sortable`元素。首先，我们需要在基础模板中加载 jQuery UI。打开`courses`应用中`templates`目录的`base.html`文件，在加载 jQuery 下面加载 jQuery UI：

```py
<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>
```

我们在 jQuery 框架之后加载 jQuery UI 库。现在编辑`courses/manage/module/content_list.html`模板，在底部添加以下代码：

```py
{% block domready %}
    $('#modules').sortable({
        stop: function(event, ui) {
            modules_order = {};
            $('#modules').children().each(function() {
                // update the order field
                $(this).find('.order').text($(this).index() + 1);
                // associate the module's id with its order
                modules_order[$(this).data('id')] = $(this).index();
            });
            $.ajax({
                type: 'POST',
                url: '{% url "module_order" %}',
                contentType: 'application/json; charset=utf-8',
                dataType: 'json',
                data: JSON.stringify(modules_order)
            });
        }
    });

    $('#module-contents').sortable({
        stop: function(event, ui) {
            contents_order = {};
            $('#module-contents').children().each(function() {
                // associate the module's id with its order
                contents_order[$(this).data('id')] = $(this).index();
            });

            $.ajax({
                type: 'POST',
                url: '{% url "content_order" %}',
                contentType: 'application/json; charset=utf-8',
                dataType: 'json',
                data: JSON.stringify(content_order),
            });
        }
    });
{% endblock domready %}
```

这段 JavaScript 代码在`{% block domready %}`块中，因此它会包括在 jQuery 的`$(document).ready()`事件中，这个事件在`base.html`模板中定义。这确保了一旦页面加载完成，就会执行我们的 JavaScript 代码。我们为侧边栏中的单元列表和单元的内容列表定义了两个不同的`sortable`元素。它们以同样的方式工作。在这段代码中，我们执行了以下任务：

1. 首先，我们为`modules`元素定义了一个`sortable`元素。记住，因为 jQuery 选择器使用 CSS 语法，所以我们使用了`#modules`。
2. 我们为`stop`事件指定了一个函数。每次用户完成对一个元素排序，会触发这个事件。
3. 我们创建了一个空的`modules_order`字典。这个字段的`key`是单元的`id`，值是分配给每个单元的序号。
4. 我们迭代`#modules`的子元素。我们重新计算每个单元的显示序号，并获得它的`data-id`属性，其中包括了单元的`id`。我们添加`id`为`modules_order`字段的`key`，单元的新索引作为值。
5. 我们发起一个 AJAX POST 请求到`content_order` URL，在请求中包括`modules_order`序列化后的 JSON 数据。相应的`ModuleOrderView`负责更新单元序号。

对内容进行排序的`sortable`元素跟它很类似。回到浏览器中，重新加载页面。现在你可以点击和拖拽单元和内容，对它们进行排序，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE10.13.png)

非常棒！你现在可以对课程单元和单元内容重新排序了。

## 10.6 总结

在本章中，你学习了如果创建一个多功能的内容管理系统。你使用了模型继承，并创建自定义模型字段。你还使用了基于类的视图和 mixins。你创建了表单集和一个系统，来管理不同类型的内容。

下一章中，你会创建一个学生注册系统。你还会渲染不同类型的内容，并学习如何使用 Django 的缓存框架。