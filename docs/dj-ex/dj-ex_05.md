# 第五章：分享内容到你的网站

上一章中，你在网站中构建了用户注册和认证。你学会了如何为用户创建自定义的个人资料模型，并添加了主流社交网站的社交认证。

在这一章中，你会学习如何创建 JavaScript 书签工具，来从其它网站分享内容到你的网站，你还会使用 jQuery 和 Django 实现 AJAX 特性。

本章会覆盖以下知识点：

- 创建多对多的关系
- 定制表单行为
- 在 Django 中使用 jQuery
- 构建 jQuery 书签工具
- 使用 sorl-thumbnail 生成图片缩略图
- 实现 AJAX 视图，并与 jQuery 集成
- 为视图创建自定义装饰器
- 构建 AJAX 分页

## 5.1 创建图片标记网站

我们将允许用户在其他网站上标记和分享他们发现的图片，并将其分享到我们的网站。为了实现这个功能，我们需要完成以下任务：

1. 定义一个存储图片和图片信息的模型。
2. 创建处理图片上传的表单和视图。
3. 为用户构建一个系统，让用户可以上传在其它网站找到的图片。

首先在`bookmarks`项目目录中，使用以下命令创建一个新的应用：

```py
django-admin startapp images
```

在`settings.py`文件的`INSTALLED_APPS`设置中添加`images`：

```py
INSTALLED_APPS = (
	# ...
	'images',
)
```

现在 Django 知道新应用已经激活了。

### 5.1.1 创建图片模型

编辑`images`应用的`models.py`文件，添加以下代码：

```py
from django.db import models
from django.conf import settings

class Image(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='images_created')
    title = models.CharField(max_length=200)
    slug = models.CharField(max_length=200, blank=True)
    url = models.URLField()
    image = models.ImageField(upload_to='/images/%Y/%m/%d')
    description = models.TextField(blank=True)
    created = models.DateField(auto_now_add=True, db_index=True)

    def __str__(self):
        return self.title
```

我们将使用这个模型存储来自不同网站的被标记的图片。让我们看看这个模型中的字段：

- `user`：标记这张图片的`User`对象。这是一个`ForeignKey`字段，它指定了一对多的关系：一个用户可以上传多张图片，但一张图片只能由一个用户上传。
- `title`：图片的标题。
- `slug`：只包括字母，数据，下划线或连字符的短标签，用于构建搜索引擎友好的 URL。
- `url`：图片的原始 URL。
- `image`：图片文件。
- `description`：一个可选的图片描述。
- `created`：在数据库中创建对象的时间。因为我们使用了`auto_now_add`，所以创建对象时会自动设置时间。我们使用了`db_index=True`，所以 Django 会在数据库中为该字段创建一个索引。

> 数据库索引会提高查询效率。考虑为经常使用`filter()`，`exclude()`或`order_by()`查询的字段设置`db_index=True`。`ForeignKey`字段或带`unique=True`的字段隐式的创建了索引。你也可以使用`Meta.index_together`为多个字段创建索引。

我们会覆写`Image`模型的`save()`方法，根据`title`字段的值自动生成`slug`字段。在`Image`模型中导入`slugify()`函数，并添加`save()`方法，如下所示：

```py
from django.utils.text import slugify

class Image(models.Model):
    # ...
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
            super().save(*args, **kwargs)
```

没有提供别名（slug）时，我们根据给定的标题，使用 Django 提供的`slufigy()`函数自动生成图片的`slug`字段。然后保存对象。我们为图片自动生成别名，所以用户不需要为每张图片输入`slug`字段。

### 5.1.2 创建多对多的关系

我们将会在`Image`模型中添加另一个字段，用于存储喜欢这张图片的用户。这种情况下，我们需要一个多对多的关系，因为一个用户可能喜欢多张图片，每张图片也可能被多个用户喜欢。

添加以下代码到`Image`模型中：

```py
users_like = models.ManyToManyField(settings.AUTH_USER_MODEL,
                                    related_name='images_liked',
                                    blank=True)
```

当你定义一个`ManyToManyFeild`时，Django 会使用两个模型的主键创建一张中介连接表。`ManyToManyFeild`可以在两个关联模型的任何一个中定义。

与`ForeignKey`字段一样，`ManyToManyFeild`允许我们命名从关联对象到这个对象的逆向关系。`ManyToManyFeild`字段提供了一个多对多管理器，允许我们检索关联的对象，比如：`image.users_like.all()`，或者从`user`对象检索：`user.images_liked.all()`。

打开命令行，执行以下命令创建初始数据库迁移：

```py
python manage.py makemigrations images
```

你会看到以下输出：

```py
Migrations for 'images':
  images/migrations/0001_initial.py
    - Create model Image
```

现在运行这条命令，让迁移生效：

```py
python manage.py migrate images
```

你会看到包括这一行的输出：

```py
Applying images.0001_initial... OK
```

现在`Image`模型已经同步到数据库中。

### 5.1.3 在管理站点注册图片模型

编辑`images`应用的`admin.py`文件，在管理站点注册`Image`模型，如下所示：

```py
from django.contrib import admin
from .models import Image

class ImageAdmin(admin.ModelAdmin):
    list_display = ('title', 'slug', 'image', 'created')
    list_filter = ['created']

admin.site.register(Image, ImageAdmin)
```

执行`python manage.py runserver`命令启动开发服务器。在浏览器中打开`http://127.0.0.1:8000/amdin/`，可以看到`Image`模型已经在管理站点注册，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.1.png)

## 5.2 从其它网站上传内容

我们将允许用户标记从其它网站找到的图片。用户将提供图片的 URL，一个标题和一个可选的描述。我们的应用会下载图片，并在数据库中创建一个新的`Image`对象。

我们从构建一个提交新图片的表单开始。在`images`应用目录中创建`forms.py`文件，并添加以下代码：

```py
from django import forms
from .models import Image

class ImageCreateForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('title', 'url', 'description')
        widgets = {
            'url': forms.HiddenInput,
        }
```

正如你所看到的，这是一个从`Image`模型创建的`ModelForm`表单，只包括`title`，`url`和`description`字段。用户不会直接在表单中输入图片 URL。而是使用一个 JavaScript 工具，从其它网站选择一张图片，我们的表单接收这张图片的 URL 作为参数。我们用`HiddenInput`组件覆盖了`url`字段的默认组件。这个组件渲染为带有`type="hidden"`属性的 HTML 输入元素。使用这个组件是因为我们不想用户看见这个字段。

### 5.2.1 清理表单字段

为了确认提供的图片 URL 是有效的，我们会检查文件名是否以`.jpg`或`.jpeg`扩展名结尾，只允许 JPG 文件。Django 允许你通过形如`clean_<filedname>()`的方法，定义表单方法来清理指定字段。如果存在这个方法，它会在调用表单实例的`is_valid()`方法时执行。在清理方法中，你可以修改字段的值，或者需要时，为这个字段抛出任何验证错误。在`ImageCreateForm`中添加以下方法：

```py
def clean_url(self):
    url = self.cleaned_data['url']
    valid_extensions = ['jpg', 'jpeg']
    extension = url.rsplit('.', 1)[1].lower()
    if extension not in valid_extensions:
        raise forms.ValidationError('The given URL does not match valid image extensions.')
    return url
```

我们在这段代码中定义了`clean_url()`方法来清理`url`字段。它是这样工作的：

1. 从表单示例的`cleaned_data`字典中获得`url`字段的值。
2. 通过分割 URL 获得文件扩展名，并检查是否为合法的扩展名。如果不是，抛出`ValidationError`，表单实例不会通过验证。我们执行了一个非常简单的验证。你可以使用更好的方法验证给定的 URL 是否提供了有效的图片。

除了验证给定的 URL，我们还需要下载并保存图片。比如，我们可以用处理这个表单的视图来下载图片文件。不过我们会使用更通用的方式：覆写模型表单的`save()`方法，在每次保存表单时执行这个任务。

### 5.2.2 覆写 ModelForm 的 save()方法

你知道，`ModelForm`提供了`save()`方法，用于把当前模型的实例保存到数据库中，并返回该对象。这个方法接收一个`commit`布尔参数，允许你指定是否把该对象存储到数据库中。如果`commit`为`False`，`save()`方法会返回模型的实例，但不会保存到数据库中。我们会覆写表单的`save()`方法下载指定的图片，然后保存。

在`forms.py`文件顶部添加以下导入：

```py
from urllib import request
from django.core.files.base import ContentFile
from django.utils.text import slugify
```

接着在`ImageCreateForm`中添加`save()`方法：

```py
def save(self, force_insert=False, force_update=False, commit=True):
    image = super().save(commit=False)
    image_url = self.cleaned_data['url']
    image_name = '{}.{}'.format(slugify(image.title), image_url.rsplit('.', 1)[1].lower())

    #download image from the given URL
    response = request.urlopen(image_url)
    image.image.save(image_name, ContentFile(response.read()), save=False)

    if commit:
        image.save()
    return image
```

我们覆写了`save()`方法，保留了`ModelForm`必需的参数。这段代码完成以下操作：

1. 我们用`commit=False`调用表单的`save()`方法，创建了一个新的`image`实例。
2. 我们从表单的`cleaned_data`字典中获得 URL。
3. 我们用`image`的标题别名和原始文件扩展名的组合生成图片名。
4. 我们使用`urllib`模块下载图片，然后调用`image`字段的`save()`方法，并传递一个`ContentFile`对象，这个对象由下载的文件内容实例化。这样就把文件保存到项目的`media`目录中了。我们还传递了`save=False`参数，避免把对象保存到数据库中。
5. 为了与被我们覆写的`save()`方法保持一致的行为，只有在`commit`参数为`True`时，才把表单保存到数据库中。

现在我们需要一个处理表单的视图。编辑`images`应用的`views.py`文件，添加以下代码：

```py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import message
from .forms import ImageCreateForm

@login_required
def image_create(request):
    if request.method == 'POST':
        # form is sent
        form = ImageCreateForm(data=request.POST)
        if form.is_valid():
            # form data is valid
            cd = form.cleaned_data
            new_item = form.save(commit=False)

            # assign current user to the item
            new_item.user = request.user
            new_item.save()
            message.success(request, 'Image added successfully')

            # redirect to new created item detail view
            return redirect(new_item.get_absolute_url())
    else:
        # build form with data provided by the bookmarklet via GET
        form = ImageCreateForm(data=request.GET)

    return render(request, 'images/image/create.html', {'section': 'images', 'form': form})
```

为了阻止未认证用户访问，我们在`image_create`视图上添加了`login_required`装饰器。这个视图是这样工作的：

1. 我们期望通过`GET`请求获得创建表单实例的初始数据。数据由其它网站的图片`url`和`title`属性组成，这个数据由我们之后会创建的 JavaScript 工具提供。现在我们假设初始的时候有数据。
2. 如果提交了表单，我们检查表单是否有效。如果有效，我们创建一个新的`Image`实例，但我们通过传递`commit=False`来阻止对象保存到数据库中。
3. 我们把当前对象赋值给新的`image`对象。这样就知道每张图片是谁上传的。
4. 我们把图片对象保存到数据库中。
5. 最后，我们用 Django 消息框架创建一条成功消息，并重定向到新图片的标准 URL。我们还没有实现`Image`模型的`get_absolute_url()`方法，我们会马上完成这个工作。

在`images`应用中创建`urls.py`文件，添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^create/$', views.image_create, name='create'),
]
```

编辑项目的主`urls.py`文件，引入我们刚创建的`images`应用的模式：

```py
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^account/', include('account.urls')),
    url(r'^images/', include('images.urls', namespace='images')),
]
```

最近，你需要创建模板来渲染表单。在`images`应用目录中创建以下目录结构：

```py
templates/
	images/
		image/
			create.html
```

编辑`create.hmtl`文件，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Bookmark an image{% endblock %}

{% block content %}
    <h1>Bookmark an image</h1>
    <img src="{{ request.GET.url }}" class="image-preview">
    <form action="." method=POST>
        {{ form.as_p }}
        {% csrf_token %}
        <input type="submit" value="Bookmark it!">
    </form>
{% endblock %}
```

现在在浏览器中打开`http://127.0.0.1:8000/images/create/?title=...&url=...`，其中包括`title`和`url`参数，后者是现有的 JPG 图片的 URL。

例如，你可以使用以下 URL：

```py
http://127.0.0.1:8000/images/create/?title=%20Django%20and%20Duke&url=http%3A%2F%2Fmvimg2.meitudata.com%2F56d7967dd02951453.jpg
```

你会看到带一张预览图片的表单，如下所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.2.png)

添加描述并点击`Bookmark it!`按钮。一个新的`Image`对象会保存到数据库中。你会得到一个错误，显示`Image`对象没有`get_absolute_url()`方法。现在不用担心，我们之后会添加这个方法。在浏览器打开`http://127.0.0.1:8000/admin/images/image/`，确认新的图片对象已经保存了。

### 5.2.3 用 jQuery 构建书签工具

书签工具是保存在 web 浏览器中的书签，其中包括 JavaScript 代码，可以扩展浏览器的功能。当你点击书签，JavaScript 代码会在浏览器正在显示的网页中执行。这对构建与其它网站交互的工具非常有用。

某些在线服务（比如 Pinterest）实现了自己的书签工具，让用户可以在自己的平台分享其它网站的内容。我们会创建一个书签工具，让用户以类似的方式在我们的网站中分享其它网站的图片。

我们将使用 jQuery 构建书签工具。jQuery 是一个非常流行的 JavaScript 框架，可以快速开发客户端的功能。你可以在[官方网站](http://jquery.com/)进一步了解 jQuery。

以下是用户如何在浏览器中添加书签工具，并使用它：

1. 用户从你的网站中拖拽一个链接到浏览器的书签中。该链接在`href`属性中包含 JavaScript 代码。这段代码会存储在书签中。
2. 用户导航到任意网站，并点击该书签。该书签的 JavaScript 代码会执行。

因为 JavaScript 代码会以书签的形式存储，所以之后你不能更新它。这是一个显著的缺点，但你可以实现一个简单的启动脚本解决这个问题。该脚本从 URL 中加载实际的 JavaScript 书签工具。用户会以书签的形式保存启动脚本，这样你就可以在任何时候更新书签工具的代码了。我们将采用这种方式构建书签工具。让我们开始吧。

在`images/templates/`中创建一个`bookmarklet_launcher.js`模板。这是启动脚本，并添加以下代码：

```py
(function() {
    if (window.myBookmarklet !== underfined) {
        myBookmarklet();
    }
    else {
        document.body.appendChild(document.createElement('script'))
            .src='http://127.0.0.1:8000/static/js/bookmarklet.js?r='+
            Math.floor(Math.random()*99999999999999999999);
    }
})();
```

这个脚本检查是否定义`myBookmarklet`变量，来判断书签工具是否加载。这样我们就避免了用户重复点击书签时多次加载它。如果没有定义`myBookmarklet`，我们通过在文档中添加`<script>`元素，来加载另一个 JavaScript 文件。这个`scrip`标签加载`bookmarklet.js`脚本，并用一个随机参数作为变量，防止从浏览器缓存中加载文件。

真正的书签工具代码位于静态文件`bookmarklet.js`中。这样就能更新我们的书签工具代码，而不用要求用户更新之前在浏览器中添加的书签。让我们把书签启动器添加到仪表盘页面，这样用户就可以拷贝到他们的书签中。

编辑`account`应用中的`account/dashboard.html`模板，如下所示：

```py
{% extends "base.html" %}

{% block title %}Dashboard{% endblock %}

{% block content %}
    <h1>Dashboard</h1>

    {% with total_images_created=request.user.images_created.count %}
        <p>
            Welcome to your dashboard. 
            You have bookmarked {{ total_images_created }} image{{ total_images_created|pluralize }}.
        </p>
    {% endwith %} 

    <p>
        Drag the following button to your bookmarks toolbar to bookmark images from other websites
        → <a href="javascript:{% include "bookmarklet_launcher.js" %}" class="button">Bookmark it</a>
    </p>

    <p>
        You can also <a href="{% url "edit" %}">edit your profile</a> 
        or <a href="{% url "password_change" %}">change your password</a>.
    <p>
{% endblock %}
```

仪表盘显示用户现在添加书签的图片总数。我们使用`{% with %}`模板标签设置当前用户添加书签的图片总数为一个变量。我们还包括了一个带`href`属性的链接，该属性指向书签工具启动脚本。我们从`bookmarklet_launcher.js`模板中引入这段 JavaScript 代码。

在浏览器中打开`http://127.0.0.1:8000/account/`，你会看下如下所示的页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.3.png)

拖拽`Bookmark it!`链接到浏览器的书签工具栏中。

现在，在`images`应用目录中创建以下目录和文件：

```py
static/
	js/
		bookmarklet.js
```

你在本章示例代码的`images`应用目录下可以找到`static/css`目录。拷贝`css/`目录到你代码的`static/`目录中。`css/bookmarklet.css`文件为我们的 JavaScript 书签工具提供了样式。

编辑`bookmarklet.js`静态文件，添加以下 JavaScript 代码：

```py
(function() {
    var jquery_version = '2.1.4';
    var site_url = 'http://127.0.0.1:8000/';
    var static_url = site_url + 'static/';
    var min_width = 100;
    var min_height = 100;

    function bookmarklet(msg) {
        // Here goes our bookmarklet code
    };

    // Check if jQuery is loaded
    if(typeof window.jQuery != 'undefined') {
        bookmarklet();
    } else {
        // Check for conflicts
        var conflict = typeof window.$ != 'undefined';
        // Create the script and point to Google API
        var script = document.createElement('script');
        script.setAttribute('src', 
            'http://ajax.googleapis.com/ajax/libs/jquery/' + 
            jquery_version + '/jquery.min.js');
        // Add the script to the 'head' for processing
        document.getElementsByTagName('head')[0].appendChild(script);
        // Create a way to wait until script loading
        var attempts = 15;
        (function(){
            // Check again if jQuery is undefined
            if (typeof window.jQuery == 'undefined') {
                if(--attempts > 0) {
                    // Calls himself in a few milliseconds
                    window.setTimeout(arguments.callee, 250);
                } else {
                    // Too much attempts to load, send error
                    alert('An error ocurred while loading jQuery')
                }
            } else {
                bookmarklet();
            }
        })();
    }
})()
```

这是主要的 jQuery 加载脚本。如果当前网站已经加载了 jQuery，那么它会使用 jQuery；否则会从 Google CDN 中加载 jQuery。加载 jQuery 后，它会执行包含书签工具代码的`bookmarklet()`函数。我们还在文件顶部设置了几个变量：

- `jquery_version`：要加载的 jQuery 版本
- `site_url`和`static_url`：我们网站的主 URL 和各个静态文件的主 URL
- `min_width`和`min_height`：我们的书签工具在网站中查找的图片的最小宽度和高度（单位是像素）

现在让我们实现`bookmarklet()`函数，如下所示：

```py
function bookmarklet(msg) {
    // load CSS
    var css = jQuery('<link>');
    css.attr({
        rel: 'stylesheet',
        type: 'text/css',
        href: static_url + 'css/bookmarklet.css?r=' + Math.floor(Math.random()*99999999999999999999)
    });
    jQuery('head').append(css);

    // load HTML
    box_html = '<div id="bookmarklet"><a href="#" id="close">&times;</a><h1>Select an image to bookmark:</h1><div class="images"></div></div>';
    jQuery('body').append(box_html);

    // close event
    jQuery('#bookmarklet #close').click(function() {
        jQuery('#bookmarklet').remove();
    });
};
```

这段代码是这样工作的：

1. 为了避免浏览器缓存，我们使用一个随机数作为参数，来加载`bookmarklet.css`样式表。
2. 我们添加了定制的 HTML 到当前网站的`<body>`元素中。它由一个`<div>`元素组成，里面会包括在当前网页中找到的图片。
3. 我们添加了一个事件。当用户点击我们的 HTML 块中的关闭链接时，我们会从文档中移除我们的 HTML。我们使用`#bookmarklet #close`选择器查找 ID 为`close`，父元素 ID 为`bookmarklet`的 HTML 元素。一个 jQuery 选择器允许你查找多个 HTML 元素。一个 jQuery 选择器返回指定 CSS 选择器找到的所有元素。你可以在[这里](http://api.jquery.com/category/selectors/)找到 jQuery 选择器列表。

加载 CSS 样式和书签工具需要的 HTML 代码后，我们需要找到网站中的图片。在`bookmarklet()`函数底部添加以下 JavaScript 代码：

```py
// find images and display them
jQuery.each(jQuery('img[src$="jpg"]'), function(index, image) {
    if (jQuery(image).width() >= min_width && jQuery(image).height() >= min_height) {
        image_url = jQuery(image).attr('src');
        jQuery('#bookmarklet .images').append('<a href="#"><img src="' + image_url + '" /></a>');
    }
});
```

这段代码使用`img[src$="jpg"]`选择器查找所有`src`属性以`jpg`结尾的`<img>`元素。这意味着我们查找当前网页中显示的所有 JPG 图片。我们使用 jQuery 的`each()`方法迭代结果。我们把尺寸大于`min_width`和`min_height`变量的图片添加到`<div class="images">`容器中。

现在，HTML 容器中包括可以添加标签图片。我们希望用户点击需要的图片，并为它添加标签。在`bookmarklet()`函数底部添加以下代码：

```py
// when an image is selected open URL with it
jQuery('#bookmarklet .images a').click(function(e) {
    selected_image = jQuery(this).children('img').attr('src');
    // hide bookmarklet
    jQuery('#bookmarklet').hide();
    // open new window to submit the image
    window.open(site_url + 'images/create/?url=' 
        + encodeURIComponent(selected_image)
        + '&title='
        + encodeURIComponent(jQuery('title').text()),
        '_blank');
});
```

这段代码完成以下工作：

1. 我们绑定一个`click()`事件到图片的链接元素。
2. 当用户点击一张图片时，我们设置一个新变量——`selected_image`，其中包含了选中图片的 URL。
3. 我们隐藏书签工具，并在我们网站中打开一个新的浏览器窗口为新图片添加标签。传递网站的`<title>`元素和选中的图片 URL 作为`GET`参数。

在浏览器中随便打开一个网址（比如`http://z.cn`），并点击你的书签工具。你会看到一个新的白色框出现在网页上，其中显示所有尺寸大于 100*100px 的 JPG 图片，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.4.png)

因为我们使用的是 Django 开发服务器，它通过 HTTP 提供页面，所以出于浏览器安全限制，书签工具不能在 HTTPS 网站上工作。

如果你点击一张图片，会重定向到图片创建页面，并传递网站标题和选中图片的 URL 作为`GET`参数：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.5.png)

恭喜你！这是你的第一个 JavaScript 书签工具，并且完全集成到你的 Django 项目中了。

## 5.3 为图片创建详情视图

我们将创建一个简单的详情视图，用于显示一张保存在我们网站的图片。打开`images`应用的`views.py`文件，添加以下代码：

```py
from django.shortcuts import get_object_or_404
from .models import Image

def image_detail(request, id, slug):
    image = get_object_or_404(Image, id=id, slug=slug)
    return render(request, 'images/image/detail.html', {'section': 'images', 'image': image})
```

这是显示一张图片的简单视图。编辑`images`应用的`urls.py`文件，添加以下 URL 模式：

```py
url(r'^detail/(?P<id>\d+)/(?P<slug>[-\w]+)/$', views.image_detail, name='detail'),
```

编辑`images`应用的`models.py`文件，在`Image`模型中添加`get_absolute_url()`方法，如下所示：

```py
from django.core.urlresolvers import reverse

class Image(models.Model):
	# ...
	def get_absolute_url(self):
		return reverse('image:detail', args=[self.id, self.slug])
```

记住，为对象提供标准 URL 的通用方式是在模型中添加`get_absolute_url()`方法。

最后，在`images`应用的`/images/image/`模板目录中创建`detail.html`模板，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}{{ image.title }}{% endblock %}

{% block content %}
    <h1>{{ image.title }}</h1>
    <img src="{{ image.image.url }}" class="image-detail">
    {% with total_likes=image.users_like.count %}
        <div class="image-info">
            <div>
                <span class="count">
                    {{ total_likes }} like{{ total_likes|pluralize }}
                </span>
            </div>
            {{ image.description|linebreaks }}
        </div>
        <div class="image-likes">
            {% for user in image.users_like.all %}
                <div>
                    <img src="{{ user.profile.photo.url }}">
                    <p>{{ user.first.name }}</p>
                </div>
            {% empty %}
                Nobody likes this image yet.
            {% endfor %}
        </div>
    {% endwith %}
{% endblock  %}
```

这是显示一张添加了标签的图片的详情模板。我们使用`{% with %}`标签存储`QuerySet`的结果，这个`QuerySet`在`total_likes`变量中统计所有喜欢这张图片的用户。这样就能避免计算同一个`QuerySet`两次。我们还包括了图片的描述，并迭代`image.users_like.all()`来显示所有喜欢这张图片的用户。

> 使用`{% width %}`模板标签可以有效地阻止 Django 多次计算`QuerySet`。

现在用书签工具标记一张新图片。当你上传图片后，会重定向到图片详情页面。该页面会包括一条成功消息，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.6.png)

## 5.4 使用 sorl-thumbnail 创建缩略图

现在，我们在详情页面显示原图，但是不同图片的尺寸各不相同。同时，有些图片的源文件可能很大，需要很长时间才能加载。用统一的方式显示优化图像的最好方法是生成缩略图。因此我们将使用一个名为`sorl-thumbnail`的 Django 应用。

打开终端，执行以下命令安装`sorl-thumbnail`：

```py
pip install sorl-thumbnail
```

编辑`bookmarks`项目的`settings.py`文件，把`sorl.thumbnail`添加到`INSTALLED_APPS`设置中：

接着执行以下命令同步应用和数据库：

```py
python manage.py makemigrations thumbnail
python manage.py migrate
```

`sorl-thumbnail`应用提供了多种定义图片缩略图的方式。它提供了`{% thumbnail %}`模板标签，可以在模板中生成缩略图；如果你想在模型中定义缩略图，还提供自定义的`ImageField`。我们将使用模板标签的方式。编辑`images/image/detail.html`模板，把这一行代码：

```py
<img src="{{ image.image.url }}" class="image-detail">
```

替换为：

```py
{% load thumbnail %}
{% thumbnail image.image "300" as im %}
    <a href="{{ image.image.url }}">
        <img src="{{ im.url }}" class="image-detail">
    </a>
{% endthumbnail %}
```

我们在这里定义了一张固定宽度为 300 像素的缩略图。用户第一次加载这个页面时，会创建一张缩略图。之后的请求会使用生成的缩略图。用`python manage.py runserver`启动开发服务器后，访问一张已存在的图片详情页。此时会生成一张缩略图并显示。

`sorl-thumbmail`应用提供了一些选项来定制缩略图，包括裁剪算法和不同的效果。如果你在生成缩略图时遇到问题，可以在设置中添加`THUMBNAIL_DEBUG=True`，就能查看调试信息。你可以在[这里](http://sorl-thumbnail.readthedocs.org/)阅读`sorl-thumbnail`应用的完整文档。

## 5.5 使用 JQuery 添加 AJAX 操作

现在我们将向应用中添加 AJAX 操作。AJAX 是`Asynchronous JavaScript and XML`的缩写。这个术语包括一组异步 HTTP 请求技术。它包括从服务器异步发送和接收数据，而不用加载整个页面。尽管名字中有`XML`，但它不是必需的。你可以使用其它格式发送或接收数据，比如 JSON，HTML 或者普通文本。

我们将会在图片详情页面添加一个链接，用户点击链接表示喜欢这张图片。我们会用 AJAX 执行这个操作，避免加载整个页面。首先，我们需要创建一个视图，让用户喜欢或不喜欢图片。编辑`images`应用的`views.py`文件，添加以下代码：

```py
from django.http import JsonResponse
from django.views.decorators.http import require_POST

@login_required
@require_POST
def image_like(request):
    image_id = request.POST.get('id')
    action = request.POST.get('action')
    if image_id and action:
        try:
            image = Image.objects.get(id=image_id)
            if action == 'like':
                image.users_like.add(request.user)
            else:
                image.users_like.remove(request.user)
            return JsonResponse({'status': 'ok'})
        except:
            pass
    return JsonResponse({'status': 'ko'})
```

我们在这个视图上使用了两个装饰器。`login_required`装饰阻止没有登录的用户访问这个视图；如果 HTTP 请求不是通过`POST`完成，`required_ POST`装饰器返回一个`HttpResponseNotAllowed`对象（状态码为 405）。这样就只允许`POST`请求访问这个视图。Django 还提供了`required_GET`装饰器，只允许`GET`请求，以及`required_http_methods`装饰器，你可以把允许的方法列表作为参数传递。

我们在这个视图中使用了两个`POST`参数：

- `image_id`：用户执行操作的图片对象的 ID。
- `action`：用户希望执行的操作，我们假设为`like`或`unlike`字符串。

我们使用 Django 为`Image`模型的`users_like`多对多字段提供的管理器的`add()`或`remove()`方法从关系中添加或移除对象。调用`add()`方法时，如果传递一个已经存在关联对象集中的对象，不会重复添加这个对象；同样，调用`remove()`方法时，如果传递一个不存在关联对象集中的对象，不会执行任何操作。另一个多对多管理器方法是`clear()`，会从关联对象集中移除所有对象。

最后，我们使用 Django 提供的`JsonResponse`类返回一个 HTTP 响应，其中内容类型为`application/json`，它会把给定对象转换为 JSON 输出。

编辑`images`应用的`urls.py`文件，添加以下 URL 模式：

```py
url(r'^like/$', views.image_like, name='like'),
```

### 5.5.1 加载 jQuery

我们需要添加 AJAX 功能到图片详情模板中。为了在模板中使用 jQuery，首先在项目的`base.html`模板中引入它。编辑`account`应用的`base.html`模板，在`</body>`标签之前添加以下代码：

```py
<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
<script>
    $(document).ready(function() {
        {% block domready %}
        {% endblock %}
    });
</script>
```

我们从 Google 加载 jQuery 框架，Google 在高速内容分发网络中托管了流行的 JavaScript 框架。你也可以从`http://jquery.com/`下载 jQuery，然后把它添加到应用的`static`目录中。

我们添加一个`<script>`标签来包括 JavaScript 代码。`$(document).ready()`是一个 jQuery 函数，参数是一个处理函数，当 DOM 层次构造完成后，会执行这个处理函数。DOM 是`Document Object Model`的缩写。DOM 是网页加载时浏览器创建的一个树对象。在这个函数中包括我们的代码，可以确保我们要交互的 HTML 元素都已经在 DOM 中加载完成。我们的代码只有在 DOM 准备就绪后才执行。

在文档准备就绪后的处理函数中，我们包括了一个名为`domready`的 Django 模板块，在扩展了基础模板的模板中可以包括特定的 JavaScript。

不要将 JavaScript 代码和 Django 模板标签搞混了。Django 模板语言在服务端渲染，输出为最终的 HTML 文档；JavaScript 在客户端执行。某些情况下，使用 Django 动态生成 JavaScript 非常有用。

> 本章的示例中，我们在 Django 模板中引入了 JavaScript 代码。引入 JavaScript 代码更好的方式是加载作为静态文件的`.js`文件，尤其当它们是代码量很大的脚本时。

### 5.5.2 AJAX 请求的跨站请求伪造

你已经在第二章中学习了跨站点请求伪造。在激活了 CSRF 保护的情况下，Django 会检查所有 POST 请求的 CSRF 令牌。当你提交表单时，可以使用`{% csrf_token %}`模板标签发送带令牌的表单。但是，对于每个 POST 请求，AJAX 请求都将 CSRF 令牌作为 POST 数据传递是不方便的。因此，Django 允许你在 AJAX 请求中，用 CSRF 令牌的值设置一个自定义的`X-CSRFToken`头。这允许你用 jQuery 或其它任何 JavaScript 库，在每次请求中自动设置`X-CSRFToken`头。

要在所有请求中包括令牌，你需要：

1. 从`csrftoken` cookie 中检索 CSRF 令牌，如果激活了 CSRF，它就会被设置。
2. 在 AJAX 请求中，使用`X-CSRFToken`头发送令牌。

你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/csrf/#ajax)找到更多关于 CSRF 保护和 AJAX 的信息。

编辑你在`base.html`中最后引入的代码，修改为以下代码：

```py
<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
<script src="http://cdn.jsdelivr.net/jquery.cookie/1.4.1/jquery.cookie.min.js"></script>
<script>
    var csrftoken = $.cookie('csrftoken');
    function csrfSafeMethod(method) {
        // these HTTP methods do not required CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
    $(document).ready(function() {
        {% block domready %}
        {% endblock %}
    });
</script>
```

这段代码完成以下工作：

1. 我们从一个公有 CDN 加载 jQuery Cookie 插件，因此我们可以与 cookies 交互。
2. 我们读取`csrftoken` cookie 中的值。
3. 我们定义`csrfSafeMethod()`函数，检查 HTTP 方法是否安全。安全的方法不需要 CSRF 保护，包括`GET`，`HEAD`，`OPTIONS`和`TRACE`。
4. 我们使用`$.ajaxSetup()`设置 jQuery AJAX 请求。每个 AJAX 请求执行之前，我们检查请求方法是否安全，以及当前请求是否跨域。如果请求不安全，我们用从 cookie 中获取的值设置`X-CSRFToken`头。这个设置会应用到 jQuery 执行的所有 AJAX 请求。

CSRF 令牌会在所有使用不安全的 HTTP 方法的 AJAX 请求中引入，比如`POST`或`PUT`。

### 5.5.3 使用 jQuery 执行 AJAX 请求

编辑`images`应用的`images/image/detail.htmlt`模板，把这一行代码：

```py
{% with total_likes=image.users_like.count %}
```

替换为下面这行：

```py
{% with total_likes=image.users_like.count users_like=image.users_like.all %}
```

然后修改`class`为`image-info`的`<div>`元素，如下所示：

```py
<div class="image-info">
    <div>
        <span class="count">
            <span class="total">{{ total_likes }}</span>
            like{{ total_likes|pluralize }}
        </span>
        <a href="#" data-id="{{ image.id }}" 
            data-action="{% if request.user in users_like %}un{% endif %}like" class="like button">
            {% if request.user not in users_like %}
                Like
            {% else %}
                Unlike
            {% endif %}
        </a>
    </div>
    {{ image.description|linebreaks }}
</div>
```

首先，我们添加了另一个变量到`{% with %}`模板标签中，用于存储`image.users_like.all`的查询结果，避免执行两次查询。我们显示喜欢这张图片的用户总数，以及一个`like/unlike`链接：我们检查用户是否在`users_like`关联对象集中，根据当前用户跟这样图片的关系显示`like`或`unlike`。我们在`<a>`元素中添加了以下属性：

- `data-id`：显示的图片的 ID。
- `data-action`：用户点击链接时执行的操作。可能是`like`或`unlike`。

我们将会在 AJAX 请求发送这两个属性的值给`image_like`视图。当用户点击`like/unlike`链接时，我们需要在客户端执行以下操作：

1. 调用 AJAX 视图，并传递图片 ID 和 action 参数。
2. 如果 AJAX 请求成功，用相反操作（`like/unlike`）更新`<a>`元素的`data-action`属性，并相应的修改显示文本。
3. 更新显示的喜欢总数。

在`images/image/detail.html`模板底部添加包括以下代码的`domready`块：

```py
{% block domready %}
    $('a.like').click(function(e) {
        e.preventDefault();
        $.post('{% url "images:like" %}', 
            {
                id: $(this).data('id'),
                action: $(this).data('action')
            },
            function(data) {
                if (data['status'] == 'ok') {
                    var previous_action = $('a.like').data('action');

                    // toggle data-action
                    $('a.like').data('action', previous_action == 'like' ? 'unlike' : 'like');
                    // toggle link text
                    $('a.like').text(previous_action == 'like' ? 'Unlike' : 'Like');
                    // update total likes
                    var previous_likes = parseInt($('span.count .total').text());
                    $('span.count .total').text(previous_action == 'like' ? previous_likes+1 : previous_likes-1);
                }
            }
        );
    });
{% endblock %}
```

这段代码是这样工作的：

1. 我们使用`$('a.like')` jQuery 选择器查找 HTML 文档中`class`是`like`的`<a>`元素。
2. 我们为`click`事件定义了一个处理函数。用户每次点击`like/unlike`链接时，会执行这个函数。
3. 在处理函数内部，我们使用`e.preventDefault()`阻止`<a>`元素的默认行为。这会阻止链接调转到其它地方。
4. 我们使用`$.post()`向服务器执行一个异步请求。jQuery 还提供了执行`GET`请求的`$.get()`方法，以及一个底层的`$.ajax()`方法。
5. 我们使用 Django 的`{% url %}`模板标签为 AJAX 请求构建 URL。
6. 我们构建在请求中发送的`POST`参数字典。我们的 Django 视图需要`id`和`action`参数。我们从`<a>`元素的`<data-id>`和`<data-action>`属性中获得这两个值。
7. 我们定义了回调函数，当收到 HTTP 响应时，会执行这个函数。它的`data`属性包括响应的内容。
8. 我们访问收到的`data`的`status`属性，检查它是否等于`ok`。如果返回的`data`是期望的那样，我们切换链接的`data-action`属性和文本。这样就允许用户取消这个操作。
9. 根据执行的操作，我们在喜欢的总数上加 1 或减 1。

在浏览器中打开之前上传的图片详情页面。你会看到以下初始的喜欢总数和`LIKE`按钮：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.7.png)

点击`LIKE`按钮。你会看到喜欢总数加 1，并且按钮的文本变为`UNLIKE`：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.8.png)

当你点击`UNLIKE`按钮时，会执行这个操作，按钮的文本变回`LIKE`，总数也会相应的改变。

编写 JavaScript 代码时，尤其是执行 AJAX 请求时，推荐使用 Firebug 等调试工具。Firebug 是一个 Firefox 插件，允许你调试 JavaScript 代码，并监控 CSS 和 HTML 的变化。你可以从[这里](http://getfirebug.com/)下载 Firebug。其它浏览器，比如 Chrome 或 Safari，也有调试 JavaScript 的开发者工具。在这些浏览器中，右键网页中的任何一个地方，然后点击`Inspect element`访问开发者工具。

## 5.6 为视图创建自定义装饰器

我们将限制 AJAX 视图只允许由 AJAX 发起的请求。Django 的`Request`对象提供一个`is_ajax()`方法，用于检查请求是否带有`XMLHttpRequest`，也就是说是否是一个 AJAX 请求。这个值在 HTTP 头的`HTTP_X_REQUESTED_WITH`中设置，绝大部分 JavaScript 库都会在 AJAX 请求中包括它。

我们将创建一个装饰器，用于在视图中检查`HTTP_X_REQUESTED_WITH`头。装饰器是一个接收另一个函数为参数的函数，并且不需要显式修改作为参数的函数，就能扩展它的行为。如果你忘了装饰器的概念，你可能需要先阅读[这里](https://www.python.org/dev/peps/pep-0318/)。

因为这是一个通用的装饰器，可以作用于任何视图，所以我们会在项目中创建一个`common`包。在`bookmarks`项目目录中创建以下目录和文件：

```py
common/
	__init__.py
	decrorators.py
```

编辑`decrorators.py`文件，添加以下代码：

```py
from django.http import HttpResponseBadRequest

def ajax_required(f):
    def wrap(request, *args, **kwargs):
        if not request.is_ajax():
            return HttpResponseBadRequest()
        return f(request, *args, **kwargs)
    wrap.__doc__ = f.__doc__
    wrap.__name__ = f.__name__
    return wrap
```

这是我们自定义的`ajax_required`装饰器。它定义了一个`wrap`函数，如果不是 AJAX 请求，则返回`HttpResponseBadRequest`对象（HTTP 400）。否则返回被装饰的函数。

现在编辑`images`应用的`views.py`文件，添加这个装饰器到`image_like`视图中：

```py
from common.decrorators import ajax_required

@ajax_required
@login_required
@require_POST
def image_like(request):
	# ...
```

如果你直接在浏览器中访问`http://127.0.0.1:8000/images/like/`，你会得到一个 HTTP 400 的响应。

> 如果你在多个视图中重复同样的验证，则可以为视图构建自定义装饰器。

## 5.7 为列表视图创建 AJAX 分页

我们需要在网站中列出所有标记过的图片。我们将使用 AJAX 分页构建一个无限滚动功能。当用户滚动到页面底部时，通过自动加载下一页的结果实现无限滚动。

我们将实现一个图片列表视图，同时处理标准浏览器请求和包括分页的 AJAX 请求。当用户首次加载图片列表页面，我们显示第一页的图片。当用户滚动到页面底部，我们通过 AJAX 加载下一页的项，并添加到主页面的底部。

我们用同一个视图处理标准和 AJAX 分页。编辑`images`应用的`views.py`文件，添加以下代码：

```py
from django.http import HttpResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

@login_required
def image_list(request):
    images = Image.objects.all()
    paginator = Paginator(images, 8)
    page = request.GET.get('page')
    try:
        images = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer deliver first page
        images = paginator.page(1)
    except EmptyPage:
        if request.is_ajax():
            # If the request is AJAX an the page is out of range
            # return an empty page
            return HttpResponse('')
        # If page is out of range deliver last page of results
        images = paginator.page(paginator.num_pages)

    if request.is_ajax():
        return render(request, 'images/image/list_ajax.html', {'section': 'images', 'images': images})
    
    return render(request, 'images/image/list.html', {'section': 'images', 'images': images})
```

我们在这个视图中创建了一个返回数据库中所有图片的`QuerySet`。然后我们构造了一个`Paginator`对象来分页查询结果，每页显示八张图片。如果请求的页数超出范围，则抛出`EmptyPage`异常。这种情况下，如果是通过 AJAX 发送请求，则返回一个空的`HttpResponse`对象，帮助我们在客户端停止 AJAX 分页。我们把结果渲染到两个不同的模板中：

- 对于 AJAX 请求，我们渲染`list_ajax.html`模板。该模板只包括请求页的图片。
- 对于标准请求，我们渲染`list.html`模板。该模板继承自`base.html`模板，并显示整个页面，同时还包括`list_ajax.html`模板，用来引入图片列表。

编辑`images`应用的`urls.py`文件，添加以下 URL 模式：

```py
url(r'^$', views.image_list, name='list'),
```

最后，我们需要创建上面提到的模板。在`images/image/`模板目录中创建`list_ajax.html`模板，添加以下代码：

```py
{% load thumbnail %}

{% for image in images %}
    <div class="image">
        <a href="{{ image.get_absolute_url }}">
            {% thumbnail image.image "300*300" crop="100%" as im %}
                <a href="{{ image.get_absolute url }}">
                    <img src="{{ im.url }}">
                </a>
            {% endthumbnail %}
        </a>
        <div class="info">
            <a href="{{ image.get_absolute_url }}" class="title">
                {{ image.title }}
            </a>
        </div>
    </div>s
{% endfor %}
```

这个模板显示图片的列表。我们将用它返回 AJAX 请求的结果。在同一个目录下创建`list.html`模板，添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Images bookmarked{% endblock %}

{% block content %}
    <h1>Images bookmarked</h1>
    <div id="image-list">
        {% include "images/image/list_ajax.html" %}
    </div>
{% endblock %}
```

列表模板继承自`base.html`模板。为了避免重复代码，我们引入了`list_ajax.html`模板来显示图片。`list.html`模板会包括 JavaScript 代码，当用户滚动到页面底部时，负责加载额外的页面。

在`list.html`模板中添加以下代码：

```py
{% block domready %}
    var page = 1;
    var empty_page = false;
    var block_request = false;

    $(window).scroll(function() {
        var margin = $(document).height() - $(window).height() - 200;
        if ($(window).scrollTop() > margin && empty_page == false && block_request == false) {
            block_request = true;
            page += 1;
            $.get('?page=' + page, function(data) {
                if (data == '') {
                    empty_page = true;
                } else {
                    block_request = false;
                    $('#image-list').append(data);
                }
            });
        }
    });
{% endblock %}
```

这段代码提供了无限滚动功能。我们在`base.html`模板中定义的`domready`块中引入了 JavaScript 代码。这段代码是这样工作的：

1. 我们定义了以下变量：
 - `page`：存储当前页码。
 - `empty_page`：让我们知道是否到了最后一页，如果是则接收一个空的页面。只要我们得到一个空的页面，就会停止发送额外的 AJAX 请求，因为我们假设此时没有结果了。
 - `block_request`：正在处理 AJAX 请求时，阻止发送另一个请求。
2. 我们使用`$(window).scroll()`捕获滚动事件，并为它定义一个处理函数。
3. 我们计算边距变量来获得文档总高度和窗口高度之间的差值，这是用户滚动的剩余内容的高度。我们从结果中减去 200，因此，当用户距离页面底部小于 200 像素时，我们会加载下一页。
4. 如果没有执行其它 AJAX 请求（`block_request`必须为`false`），并且用户没有获得最后一页的结果时（`empty_page`也为`false`），我们才发送 AJAX 请求。
5. 我们设置`block_request`为`true`，避免滚动事件触发额外的 AJAX 请求，同时给`page`计算器加 1 来获取下一页。
6. 我们使用`$.get()`执行 AJAX GET 请求，然后在名为`data`的变量中接收 HTML 响应。这里有两种情况：
 - 响应不包括内容：我们已经到了结果的末尾，没有更多页面需要加载。我们设置`empty_page`为`true`阻止额外的 AJAX 请求。
 - 响应包括内容：我们把数据添加到 id 为`image-list`的 HTML 元素中。当用户接近页面底部时，页面内容会垂直扩展附加的结果。

在浏览器中打开`http://127.0.0.1:8000/images/`，你会看到目前已经标记过的图片列表，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE5.9.png)

滚动到页面底部来加载下一页。确保你用书签工具标记了八张以上图片，因为我们每页显示八张图片。记住，你可以使用 Firebug 或类似工具追踪 AJAX 请求和调试 JavaScript 代码。

最后，编辑`account`应用的`base.html`模板，为主菜单的`Images`项添加 URL：

```py
<li {% if section == "images" %}class="selected"{% endif %}>
	<a href="{% url "images:list" %}">Images</a>
</li>
```

现在你可以从主菜单中访问图片列表。

## 5.8 总结

在本章中，我们构建了一个 JavaScript 书签工具，可以分享其它网站的图片到我们的网站中。你用 jQuery 实现了 AJAX 视图，并添加了 AJAX 分页。

下一章会教你如何构建关注系统和活动流。你会使用通用关系（generic relations），信号（signals）和反规范化（denormalization）。你还会学习如何在 Django 中使用 Redis。