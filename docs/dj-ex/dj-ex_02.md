# 第二章：为博客添加高级功能

上一章中，你创建了一个基础的博客应用。现在，利用一些高级特性，你要把它打造成一个功能完整的博客，比如通过邮件分享帖子，添加评论，为帖子打上标签，以及通过相似度检索帖子。在这一章中，你会学习以下主题：

- 使用 Django 发送邮件
- 在视图中创建和处理表单
- 通过模型创建表单
- 集成第三方应用
- 构造复杂的`QuerySet`。

## 2.1 通过邮件分享帖子

首先，我们将会允许用户通过邮件分享帖子。花一点时间想想，通过上一章学到的知识，你会如何使用视图，URL 和模板来完成这个功能。现在核对一下，允许用户通过邮件发送帖子需要完成哪些操作：

- 为用户创建一个填写名字，邮箱，收件人和评论（可选的）的表单
- 在`views.py`中创建一个视图，用于处理`post`数据和发送邮件
- 在`blog`应用的`urls.py`文件中，为新视图添加 URL 模式
- 创建一个显示表单的模板

### 2.1.1 使用 Django 创建表单

让我们从创建分享帖子的表单开始。Django 有一个内置的表单框架，让你很容易的创建表单。表单框架允许你定义表单的字段，指定它们的显示方式，以及如何验证输入的数据。Django 的表单框架还提供了一种灵活的方式，来渲染表单和处理数据。

Django 有两个创建表单的基础类：

- `Form`：允许你创建标准的表单
- `ModelForm`：允许你通过创建表单来创建或更新模型实例

首先，在`blog`应用目录中创建`forms.py`文件，添加以下代码：

```py
from django import forms

class EmailPostForm(forms.Form):
	name = forms.CharField(max_length=25)
	email = forms.EmailField()
	to = forms.EmailField()
	comments = forms.CharField(required=False, 
	                           widget=forms.Textarea)
```

这是你的第一个 Django 表单。这段代码通过继承基类`Form`创建了一个表单。我们使用不同的字段类型，Django 可以相应的验证字段。

> 表单可以放在 Django 项目的任何地方，但惯例是放在每个应用的`forms.py`文件中。

`name`字段是一个`CharField`。这种字段的类型渲染为`<input type="text">` HTML 元素。每种字段类型都有一个默认组件，决定了该字段如何在 HTML 中显示。可以使用`widget`属性覆盖默认组件。在`comments`字段中，我们使用`Textarea`组件显示为`<textarea>` HTML 元素，而不是默认的`<input>`元素。

字段的验证也依赖于字段类型。例如，`email`和`to`字段是`EmailField`。这两个字段都要求一个有效的邮箱地址，否则字段验证会抛出`forms.ValidationError`异常，导致表单无效。表单验证时，还会考虑其它参数：我们定义`name`字段的最大长度为 25 个字符，并使用`required=False`让`comments`字段是可选的。字段验证时，这些所有因素都会考虑进去。这个表单中使用的字段类型只是 Django 表单字段的一部分。在[这里](https://docs.djangoproject.com/en/1.11/ref/forms/fields/)查看所有可用的表单字段列表。

### 2.1.2 在视图中处理表单

你需要创建一个新视图，用于处理表单，以及提交成功后发送一封邮件。编辑`blog`应用的`views.py`文件，添加以下代码：

```py
from .forms import EmailPostForm

def post_share(request, post_id):
	# Retrieve post by id
	post = get_object_or_404(Post, id=post_id, status='published')
	
	if request.method == 'POST':
		# Form was submitted
		form = EmailPostForm(request.POST)
		if form.is_valid():
			# Form fields passed validation
			cd = form.cleaned_data
			# ... send email
	else:
		form = EmailPostForm()
	return render(request, 
					'blog/post/share.html', 
					{'post': post, 'form': form})
```

该视图是这样工作的：

- 我们定义了`post_share`视图，接收`request`对象和`post_id`作为参数。
- 我们通过 ID，使用`get_object_or_404()`快捷方法检索状态为`published`的帖子。
- 我们使用同一个视图=显示初始表单和处理提交的数据。根据`request.method`区分表单是否提交。我们将使用`POST`提交表单。如果我们获得一个`GET`请求，需要显示一个空的表单；如果获得一个`POST`请求，表单会被提交，并且需要处理它。因此，我们使用`request.method == 'POST'`来区分这两种场景。

以下是显示和处理表单的过程：

1. 当使用`GET`请求初始加载视图时，我们创建了一个新的表单实例，用于在模板中显示空表单。

 `form = EmailPostForm()`

2. 用户填写表单，并通过`POST`提交。接着，我们使用提交的数据创建一个表单实例，提交的数据包括在`request.POST`中：
 ```py
 if request.POST == 'POST':
     # Form was submitted
     form = EmailPostForm(request.POST)
 ```

3. 接着，我们使用表单的`is_valid()`方法验证提交的数据。该方法会验证表单中的数据，如果所有字段都是有效数据，则返回`True`。如果任何字段包含无效数据，则返回`False`。你可以访问`form.errors`查看验证错误列表。
4. 如果表单无效，我们使用提交的数据在模板中再次渲染表单。我们将会在模板中显示验证错误。
5. 如果表单有效，我们访问`form.cleaned_data`获得有效的数据。该属性是表单字段和值的字典。

> 如果你的表单数据无效，`cleaned_data`只会包括有效的字段。

现在，你需要学习如何使用 Django 发送邮件，把所有功能串起来。

### 2.1.3 使用 Django 发送邮件

使用 Django 发送邮件非常简单。首先，你需要一个本地 SMTP 服务，或者在项目的`settings.py`文件中添加以下设置，定义一个外部 SMTP 服务的配置：

- `EMAIL_HOST`：SMTP 服务器地址。默认是`localhost`。
- `EMAIL_PORT`：SMTP 服务器端口，默认 25。
- `EMAIL_HOST_USER`：SMTP 服务器的用户名。
- `EMAIL_HOST_PASSWORD`：SMTP 服务器的密码。
- `EMAIL_USE_TLS`：是否使用 TLS 加密连接。
- `EMAIL_USE_SSL`：是否使用隐式 TLS 加密连接。

如果你没有本地 SMTP 服务，可以使用你的邮箱提供商的 SMTP 服务。下面这个例子中的配置使用 Google 账户发送邮件：

```py
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_HOST_USER = 'your_account@gmail.com'
EMAIL_HOST_PASSWORD = 'your_password'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
```

运行`python manage.py shell`命令打开 Python 终端，如下发送邮件：

```py
>>> from django.core.mail import send_mail
>>> send_mail('Django mail', 'This e-mail was sent with Django',
'your_account@gmail.com', ['your_account@gmail.com'], 
fail_silently=False)
```

`send_mail()`的必填参数有：主题，内容，发送人，以及接收人列表。通过设置可选参数`fail_silently=False`，如果邮件不能正确发送，就会抛出异常。如果看到输出`1`，则表示邮件发送成功。如果你使用前面配置的 Gmail 发送邮件，你可能需要在[这里](https://www.google.com/settings/security/lesssecureapps)启用低安全级别应用访问权限。

现在，我们把它添加到视图中。编辑`blog`应用中`views.py`文件的`post_share`视图，如下所示：

```py
from django.core.mail import send_mail

def post_share(request, post_id):
	# Retrieve post by id
	post = get_object_or_404(Post, id=post_id, status='published')
	sent = False
	
	if request.method == 'POST':
		# Form was submitted
		form = EmailPostForm(request.POST)
		if form.is_valid():
			# Form fields passed validation
			cd = form.cleaned_data
			post_url = request.build_absolute_uri(post.get_absolute_url())
			subject = '{} ({}) recommends you reading "{}"'.format(cd['name'], cd['email'], post.title)
			message = 'Read "{}" at {}\n\n{}\'s comments: {}'.format(post.title, post_url, cd['name'], cd['comments'])
			send_mail(subject, message, 'admin@blog.com', [cd['to']])
			sent = True
	else:
		form = EmailPostForm()
	return render(request, 
		           'blog/post/share.html', 
		           {'post': post, 'form': form, 'sent': sent}) 
```

注意，我们声明了一个`sent`变量，当帖子发送后，设置为`True`。当表单提交成功后，我们用该变量在模板中显示一条成功的消息。因为我们需要在邮件中包含帖子的链接，所以使用了`get_absolute_url()`方法检索帖子的绝对路径。我们把这个路径作为`request.build_absolute_uri()`的输入，构造一个包括 HTTP 模式（schema）和主机名的完整 URL。我们使用验证后的表单数据构造邮件的主题和内容，最后发送邮件到表单`to`字段中的邮件地址。

现在，视图的开发工作已经完成，记得为它添加新的 URL 模式。打开`blog`应用的`urls.py`文件，添加`post_share`的 URL 模式：

```py
urlpatterns = [
	# ...
	url(r'^(?P<post_id>\d+)/share/$', views.post_share, name='post_share'),
]
```

### 2.1.4 在模板中渲染表单

完成创建表单，编写视图和添加 URL 模式后，我们只缺少该视图的模板了。在`blog/templates/blog/post/`目录中创建`share.html`文件，添加以下代码：

```py
{% extends "blog/base.html" %}

{% block title %}Share a post{% endblock %}

{% block content %}
	{% if sent %}
		<h1>E-mail successfully sent</h1>
		<p>
			"{{ post.title }}" was successfully sent to {{ cd.to }}.
		</p>
	{% else %}
		<h1>Share "{{ post.title }}" by e-mail</h1>
		<form action="." method="post">
			{{ form.as_p }}
			{% csrf_token %}
			<input type="submit" value="Send e-mail">
		</form>
	{% endif %}
{% endblock %}
```

这个模板用于显示表单，或者表单发送后的一条成功消息。正如你所看到的，我们创建了一个 HTML 表单元素，指定它需要使用`POST`方法提交：

```py
<form action="." method="post">
```

然后，我们包括了实际的表单实例。我们告诉 Django 使用`as_p`方法，在 HTML 的`<p>`元素中渲染表单的字段。我们也可以使用`as_ul`把表单渲染为一个无序列表，或者使用`as_table`渲染为 HTML 表格。如果你想渲染每一个字段，我们可以这样迭代字段：

```py
{% for field in form %}
	<div>
		{{ field.errors }}
		{{ field.label_tag }} {{ field }}
	</div>
{% endfor %}
```

模板标签`{% csrf_token %}`使用自动生成的令牌引入一个隐藏字段，以避免跨站点请求伪造（CSRF）的攻击。这些攻击包含恶意网站或程序，对你网站上的用户执行恶意操作。你可以在[这里](https://en.wikipedia.org/wiki/Cross-site_request_forgery)找到更多相关的信息。

上述标签生成一个类似这样的隐藏字段：

```py
<input type="hidden" name="csrfmiddlewaretoken" value="26JjKo2lcEtYkGoV9z4XmJIEHLXN5LDR" />
```

> 默认情况下，Django 会检查所有`POST`请求中的 CSRF 令牌。记得在所有通过`POST`提交的表单中包括`csrf_token`标签。

编辑`blog/post/detail.html`模板，在`{{ post.body|linebreaks }}`变量之后添加链接，用于分享帖子的 URL：

```py
<p>
	<a href="{% url "blog:post_share" post.id %}">
		Share this post
	</a>
</p>
```

记住，我们使用 Django 提供的`{% url %}`模板标签，动态生成 URL。我们使用名为`blog`命名空间和名为`post_share`的 URL，并传递帖子 ID 作为参数来构造绝对路径的 URL。

现在，使用`python manage.py runserver`命令启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/blog/`。点击任何一篇帖子的标题，打开详情页面。在帖子正文下面，你会看到我们刚添加的链接，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.1.png)

点击`Share this post`，你会看到一个包含表单的页面，该页面可以通过邮件分享帖子。如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.2.png)

该表单的 CSS 样式在`static/css/blog.css`文件中。当你点击`Send e-mail`按钮时，该表单会被提交和验证。如果所有字段都是有效数据，你会看到一条成功消息，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.3.png)

如果你输入了无效数据，会再次渲染表单，其中包括了所有验证错误：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.4.png)

> **译者注：**不知道是因为浏览器不同，还是 Django 的版本不同，这里显示的验证错误跟原书中不一样。我用的是 Chrome 浏览器。

## 2.2 创建评论系统

现在，我们开始为博客构建评论系统，让用户可以评论帖子。要构建评论系统，你需要完成以下工作：

- 创建一个保存评论的模型
- 创建一个提交表单和验证输入数据的表单
- 添加一个视图，处理表单和保存新评论到数据库中
- 编辑帖子详情模板，显示评论列表和添加新评论的表单

首先，我们创建一个模型存储评论。打开`blog`应用的`models.py`文件，添加以下代码：

```py
class Comment(models.Model):
	post = models.ForeignKey(Post, related_name='comments')
	name = models.CharField(max_length=80)
	email = models.EmailField()
	body = models.TextField()
	created = models.DateTimeField(auto_now_add=True)
	updated = models.DateTimeField(auto_now=True)
	active = models.BooleanField(default=True)
	
	class Meta:
		ordering = ('created', )
		
	def __str__(self):
		return 'Comment by {} on {}'.format(self.name, self.post)
```

这就是我们的`Comment`模型。它包含一个外键，把评论与单篇帖子关联在一起。这个多对一的关系在`Comment`模型中定义，因为每条评论对应一篇帖子，而每篇帖子可能有多条评论。从关联对象反向到该对象的关系由`related_name`属性命名。定义这个属性后，我们可以使用`comment.post`检索评论对象的帖子，使用`post.comments.all()`检索帖子的所有评论。如果你没有定义`related_name`属性，Django 会使用模型名加`_set`（即`comment_set`）命名关联对象反向到该对象的管理器。

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/db/examples/many_to_one/)学习更多关于多对一的关系。

我们使用了`active`布尔字段，用于手动禁用不合适的评论。我们使用`created`字段排序评论，默认按时间排序。

刚创建的`Comment`模型还没有同步到数据库。运行以下命令，生成一个新的数据库迁移，反射创建的新模型：

```py
python manage.py makemigrations blog
```

你会看到以下输出：

```py
Migrations for 'blog'
  0002_comment.py:
    - Create model Comment
```

Django 在`blog`应用的`migrations/`目录中生成了`0002_comment.py`文件。现在，你需要创建一个相关的数据库架构，并把这些改变应用到数据库中。运行以下命令，让已存在的数据库迁移生效：

```py
python manage.py migrate
```

你会得到一个包括下面这一行的输出：

```py
Apply blog.0002_comment... OK
```

我们刚创建的数据库迁移已经生效，数据库中已经存在一张新的`blog_comment`表。

现在我们可以添加新的模型到管理站点，以便通过简单的界面管理评论。打开`blog`应用的`admin.py`文件，导入`Comment`模型，并增加`CommentAdmin`类：

```py
from .models import Post, Comment

class CommentAdmin(admin.ModelAdmin):
	list_display = ('name', 'email', 'post', 'created', 'active')
	list_filter = ('active', 'created', 'updated')
	search_fields = ('name', 'email', 'body')
admin.site.register(Comment, CommentAdmin)
```

使用`python manage.py runserver`命令启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/admin/`。你会在`Blog`中看到新的模型，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.5.png)

我们的模型已经在管理站点注册，并且可以使用简单的界面管理`Comment`实例。

### 2.2.1 通过模型创建表单

我们仍然需要创建一个表单，让用户可以评论博客的帖子。记住，Django 有两个基础类用来创建表单：`Form`和`ModelForm`。之前你使用了第一个，让用户可以通过邮件分享帖子。在这里，你需要使用`ModelForm`，因为你需要从`Comment`模型中动态的创建表单。编辑`blog`应用的`forms.py`文件，添加以下代码：

```py
from .models import Comment

class CommentForm(forms.ModelForm):
	class Meta:
		model = Comment
		fields = ('name', 'email', 'body')
```

要通过模型创建表单，我们只需要在表单的`Meta`类中指定，使用哪个模型构造表单。Django 自省模型，并动态的为我们创建表单。每种模型字段类型都有相应的默认表单字段类型。我们定义模型字段的方式考虑了表单的验证。默认情况下，Django 为模型中的每个字段创建一个表单字段。但是，你可以使用`fields`列表明确告诉框架，你想在表单中包含哪些字段，或者使用`exclude`列表定义你想排除哪些字段。对应`CommentForm`，我们只使用`name`，`email`，和`body`字段，因为用户只可能填写这些字段。

### 2.2.2 在视图中处理 ModelForm

为了简单，我们将会使用帖子详情页面实例化表单，并处理它。编辑`views.py`文件，导入`Comment`模型和`CommentForm`表单，并修改`post_detail`视图，如下所示：

> **译者注：**原书中是编辑`models.py`文件，应该是作者的笔误。

```py
from .models import Post, Comment
from .forms import EmailPostForm, CommentForm

def post_detail(request, year, month, day, post):
	post = get_object_or_404(Post, slug=post,
										 status='published',
										 publish__year=year,
										 publish__month=month,
										 publish__day=day)
	# List of active comments for this post
	comments = post.comments.filter(active=True)
	new_comment = None
	
	if request.method == 'POST':
		# A comment was posted
		comment_form = CommentForm(data=request.POST)
		if comment_form.is_valid():
			# Create Comment object but don't save to database yet
			new_comment = comment_form.save(commit=False)
			# Assign the current post to comment
			new_comment.post = post
			# Save the comment to the database
			new_comment.save()
	else:
		comment_form = CommentForm()
	return render(request, 
					 'blog/post/detail.html',
					 {'post': post,
					  'comments': comments,
					  'new_comment': new_comment,
					  'comment_form': comment_form})
```

让我们回顾一下，我们往视图里添加了什么。我们使用`post_detail`视图显示帖子和它的评论。我们添加了一个`QuerySet`，用于检索该帖子所有有效的评论：

```py
comments = post.comments.filter(active=True)
```

我们从`post`对象开始创建这个`QuerySet`。我们在`Comment`模型中使用`related_name`属性，定义了关联对象的管理器为`comments`。这里使用了这个管理器。

同时，我们使用同一个视图让用户添加新评论。因此，如果视图通过`GET`调用，我们使用`comment_form = CommentForm()`创建一个表单实例。如果是`POST`请求，我们使用提交的数据实例化表单，并使用`is_valid()`方法验证。如果表单无效，我们渲染带有验证错误的模板。如果表单有效，我们完成以下操作：

1. 通过调用表单的`save()`方法，我们创建一个新的`Comment`对象：

 `new_comment = comment_form.save(commit=False)`

 `save()`方法创建了一个链接到表单模型的实例，并把它存到数据库中。如果使用`commit=False`调用，则只会创建模型实例，而不会存到数据库中。当你想在存储之前修改对象的时候，会非常方便，之后我们就是这么做的。`save()`只对`ModelForm`实例有效，对`Form`实例无效，因为它们没有链接到任何模型。

2. 我们把当前的帖子赋值给刚创建的评论：

 `new_comment.post = post `

 通过这个步骤，我们指定新评论属于给定的帖子。

3. 最后，使用下面的代码，把新评论存到数据库中：

 `new_comment.save()`

现在，我们的视图已经准备好了，可以显示和处理新评论了。

### 2.2.3 在帖子详情模板中添加评论

我们已经为帖子创建了管理评论的功能。现在我们需要修改`blog/post/detail.html`模板，完成以下工作：

- 为帖子显示评论总数
- 显示评论列表
- 显示一个表单，用户增加评论

首先，我们会添加总评论数。打开`detail.html`模板，在`content`块中添加以下代码：

```py
{% with comments.count as total_comments %}
	<h2>
		{{ total_comments }} comment{{ total_comments|pluralize }}
	</h2>
{% endwith %}
```

我们在模板中使用 Django ORM 执行`comments.count()`这个`QuerySet`。注意，Django 模板语言调用方法时不带括号。`{% with %}`标签允许我们把值赋给一个变量，我们可以在`{% endwith %}`标签之前一直使用它。

> `{% with %}`模板标签非常有用，它可以避免直接操作数据库，或者多次调用昂贵的方法。

我们使用了`pluralize`模板过滤器，根据`total_comments`的值决定是否显示单词`comment`的复数形式。模板过滤器把它们起作用变量的值作为输入，并返回一个计算后的值。我们会在第三章讨论模板过滤器。

如果值不是 1，`pluralize`模板过滤器会显示一个“s”。上面的文本会渲染为`0 comments`，`1 comment`，或者`N comments`。Django 包括大量的模板标签和过滤器，可以帮助你以希望的方式显示信息。

现在，让我们添加评论列表。在上面代码后面添加以下代码：

```py
{% for comment in comments %}
	<div class="comment">
		<p class="info">
			Comment {{ forloop.counter }} by {{ comment.name }}
			{{ comment.created }}
		</p>
		{{ comment.body|linebreaks }}
	</div>
{% empty %}
	<p>There are no comments yet.</p>
{% endfor %}
```

我们使用`{% for %}`模板标签循环所有评论。如果`comments`列表为空，显示一个默认消息，告诉用户该帖子还没有评论。我们使用`{{ forloop.counter }}`变量枚举评论，它包括每次迭代中循环的次数。然后我们显示提交评论的用户名，日期和评论的内容。

最后，当表单成功提交后，我们需要渲染表单，或者显示一条成功消息。在上面的代码之后添加以下代码：

```py
{% if new_comment %}
	<h2>Your comment has been added.</h2>
{% else %}
	<h2>Add a new comment</h2>
	<form action="." method="post">
		{{ comment_form.as_p }}
		{% csrf_token %}
		<p><input type="submit" value="Add comment"></p>
	</form>
{% endif %}
```

代码非常简单：如果`new_comment`对象存在，则显示一条成功消息，因为已经创建评论成功。否则渲染表单，每个字段使用一个`<p>`元素，以及`POST`请求必需的 CSRF 令牌。在浏览器中打开`http://127.0.0.1:8000/blog/`，点击一条帖子标题，打开详情页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.6.png)

使用表单添加两条评论，它们会按时间顺序显示在帖子下方，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.7.png)

在浏览器中打开`http://127.0.0.1:8000/admin/blog/comment/`，你会看到带有刚创建的评论列表的管理页面。点击某一条编辑，不选中`Active`选择框，然后点击`Save`按钮。你会再次被重定向到评论列表，该评论的`Active`列会显示一个禁用图标。类似下图的第一条评论：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.8.png)

如果你回到帖子详情页面，会发现被删除的评论没有显示；同时也没有算在评论总数中。多亏了`active`字段，你可以禁用不合适的评论，避免它们在帖子中显示。

## 2.3 增加标签功能

实现评论系统之后，我们准备为帖子添加标签。我们通过在项目中集成一个第三方的 Django 标签应用，来实现这个功能。`django-taggit`是一个可复用的应用，主要提供了一个`Tag`模型和一个管理器，可以很容易的为任何模型添加标签。你可以在[这里](https://github.com/alex/django-taggit)查看它的源码。

首先，你需要通过`pip`安装`django-taggit`，运行以下命令：

```py
pip install django-taggit
```

然后打开`mysite`项目的`settings.py`文件，添加`taggit`到`INSTALLED_APPS`设置中：

```py
INSTALLED_APPS = (
	# ...
	'blog',
	'taggit',
)
```

打开`blog`应用的`models.py`文件，添加`django-taggit`提供的`TaggableManager`管理器到`Post`模型：

```py
from taggit.managers import TaggableManager

class Post(models.Model):
	# ...
	tags = TaggableManager()
```

`tags`管理器允许你从`Post`对象中添加，检索和移除标签。

运行以下命令，为模型改变创建一个数据库迁移：

```py
python manage.py makemigrations blog
```

你会看下以下输出：

```py
Migrations for 'blog'
  0003_post_tags.py:
    - Add field tags to post
```

现在，运行以下命令创建`django-taggit`模型需要的数据库表，并同步模型的变化：

```py
python manage.py migrate
```

你会看到迁移数据库生效的输入，如下所示：

```py
Applying taggit.0001_initial... OK
Applying taggit.0002_auto_20150616_2121... OK
Applying blog.0003_post_tags... OK
```

你的数据库已经为使用`django-taggit`模型做好准备了。使用`python manage.py shell`打开终端，学习如何使用`tags`管理器。

首先，我检索其中一个帖子（ID 为 3 的帖子）：

```py
>>> from blog.models import Post
>>> post = Post.objects.get(id=3)
```

接着给它添加标签，并检索它的标签，检查是否添加成功：

```py
>>> post.tags.add('music', 'jazz', 'django')
>>> post.tags.all()
[<Tag: jazz>, <Tag: django>, <Tag: music>]
```

最后，移除一个标签，并再次检查标签列表：

```py
>>> post.tags.remove('django')
>>> post.tags.all()
[<Tag: jazz>, <Tag: music>]
```

这很容易，对吧？运行`python manage.py runserver`，再次启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/admin/taggit/tag/`。你会看到`taggit`应用管理站点，其中包括`Tag`对象的列表：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.9.png)

导航到`http://127.0.0.1:8000/admin/blog/post/`，点击一条帖子编辑。你会看到，现在帖子包括一个新的`Tags`字段，如下图所示，你可以很方便的编辑标签：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.10.png)

现在，我们将会编辑博客帖子，来显示标签。打开`blog/post/list.html`模板，在帖子标题下面添加以下代码：

```py
<p class="tags">Tags: {{ post.tags.all|join:", " }}</p>
```

模板过滤器`join`与 Python 字符串的`join()`方法类似，用指定的字符串连接元素。在浏览器中打开`http://127.0.0.1:8000/blog/`。你会看到每篇帖子标题下方有标签列表：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.11.png)

现在，我们将要编辑`post_list`视图，为用户列出具有指定标签的所有帖子。打开`blog`应用的`views.py`文件，从`django-taggit`导入`Tag`模型，并修改`post_list`视图，可选的通过标签过滤帖子：

```py
from taggit.models import Tag

def post_list(request, tag_slug=None):
	object_list = Post.published.all()
	tag = None
	
	if tag_slug:
		tag = get_object_or_404(Tag, slug=tag_slug)
		object_list = object_list.filter(tags__in=[tag])
		# ...
```

该视图是这样工作的：

1. 该视图接收一个默认值为`None`的可选参数`tag_slug`。该参数会在 URL 中。
2. 在视图中，我们创建了初始的`QuerySet`，检索所有已发布的帖子，如果给定了标签别名，我们使用`get_object_or_404()`快捷方法获得给定别名的`Tag`对象。
3. 然后，我们过滤包括给定标签的帖子列表。因为这是一个多对多的关系，所以我们需要把过滤的标签放在指定列表中，在这个例子中只包含一个元素。

记住，`QeurySet`是懒惰的。这个`QuerySet`只有在渲染模板时，循环帖子列表时才会计算。

最后，修改视图底部的`render()`函数，传递`tag`变量到模板中。视图最终是这样的：

```py
def post_list(request, tag_slug=None):
	object_list = Post.published.all()
	tag = None
	
	if tag_slug:
		tag = get_object_or_404(Tag, slug=tag_slug)
		object_list = object_list.filter(tags__in=[tag])
		
	paginator = Paginator(object_list, 3)
	page = request.GET.get('page')
	try:
		posts = paginator.page(page)
	except PageNotAnInteger:
		posts = paginator.page(1)
	excpet EmptyPage:
		posts = paginator.page(paginator.num_pages)
	return render(request,
					 'blog/post/list.html',
					 {'page': page,
					  'posts': posts,
					  'tag': tag})
```

打开`blog`应用的`urls.py`文件，注释掉基于类`PostListView`的 URL 模式，取消`post_list`视图的注释：

```py
url(r'^$', views.post_list, name='post_list'),
# url(r'^$', views.PostListView.as_view(), name='post_list'),
```

添加以下 URL 模式，通过标签列出帖子：

```py
url(r'^tag/(?P<tag_slug>[-\w]+)/$', views.post_list,
    name='post_list_by_tag'),
```

正如你所看到的，两个模式指向同一个视图，但是名称不一样。第一个模式不带任何可选参数调用`post_list`视图，第二个模式使用`tag_slug`参数调用视图。

因为我们使用的是`post_list`视图，所以需要编辑`blog/post/list.hmlt`模板，修改`pagination`使用`posts`参数：

```py
{% include "pagination.html" with page=posts %}
```

在`{% for %}`循环上面添加以下代码：

```py
{% if tag %}
	<h2>Posts tagged with "{{ tag.name }}"</h2>
{% endif %}
```

如果用户正在访问博客，他会看到所有帖子列表。如果他通过指定标签过滤帖子，就会看到这个信息。现在，修改标签的显示方式：

```py
<p class="tag">
	Tags:
	{% for tag in post.tags.all %}
		<a href="{% url "blog:post_list_by_tag" tag.slug %}">
			{{ tag.name }}
		</a>
	{% if not forloop.last %}, {% endif %}
	{% endfor %}
</p>
```

现在，我们循环一篇帖子的所有标签，显示一个自定义链接到 URL，以便使用该便签过滤帖子。我们用`{% url "blog:post_list_by_tag" tag.slug %}`构造 URL，把 URL 名和标签的别名作为参数。我们用逗号分隔标签。

在浏览器中打开`http://127.0.0.1:8000/blog/`，点击某一个标签链接。你会看到由该标签过滤的帖子列表：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.12.png)

## 2.4 通过相似度检索帖子

现在，我们已经为博客帖子添加了标签，我们还可以用标签做更多有趣的事。通过便签，我们可以很好的把帖子分类。主题类似的帖子会有几个共同的标签。我们准备增加一个功能：通过帖子共享的标签数量来显示类似的帖子。在这种情况下，当用户阅读一篇帖子的时候，我们可以建议他阅读其它相关帖子。

为某个帖子检索相似的帖子，我们需要：

- 检索当前帖子的所有标签。
- 获得所有带这些便签中任何一个的帖子。
- 从列表中排除当前帖子，避免推荐同一篇帖子。
- 通过和当前帖子共享的标签数量来排序结果。
- 如果两篇或以上的帖子有相同的标签数量，推荐最近发布的帖子。
- 限制我们想要推荐的帖子数量。

这些步骤转换为一个复杂的`QuerySet`，我们需要在`post_detail`视图中包含它。打开`blog`应用的`views.py`文件，在顶部添加以下导入：

```py
from django.db.models import Count
```

这是 Django ORM 的`Count`汇总函数。此函数允许我们执行汇总计数。然后在`post_detail`视图的`render()`函数之前添加以下代码：

```py
# List of similar posts
post_tags_ids = post.tags.values_list('id', flat=True)
similar_posts = Post.published.filter(tags__in=post_tags_ids)\
									.exclude(id=post.id)
similar_posts = similar_posts.annotate(same_tags=Count('tags'))\
                             .order_by('-same_tags', '-publish')[:4]
```

这段代码完成以下操作：

1. 我们获得一个包含当前帖子所有标签的 ID 列表。`values_list()`这个`QuerySet`返回指定字段值的元组。我们传递`flat=True`给它，获得一个`[1, 2, 3, ...]`的列表。
2. 我们获得包含这些标签中任何一个的所有帖子，除了当前帖子本身。
3. 我们使用`Count`汇总函数生成一个计算后的字段`same_tags`，它包含与所有查询标签共享的标签数量。
4. 我们通过共享的标签数量排序结果（降序），共享的标签数量相等时，用`publish`优先显示最近发布的帖子。我们对结果进行切片，只获取前四篇帖子。

为`render()`函数添加`similar_posts`对象到上下文字典中：

```py
return render(request,
              'blog/post/detail.html',
              {'post': post,
               'comments': comments,
               'new_comment':new_comment,
               'comment_form': comment_form,
               'similar_posts': similar_posts})
```

现在，编辑`blog/post/detail.html`模板，在帖子的评论列表前添加以下代码：

```py
<h2>Similar posts</h2>
{% for post in similar_posts %}
	<p>
		<a href="{{ post.get_absolute_url }}">{{ post.title }}</a>
	</p>
{% empty %}
	There are no similar post yet.
{% endfor %}
```

推荐你在帖子详情模板中也添加标签列表，就跟我们在帖子列表模板中所做的那样。现在，你的帖子详情页面应该看起来是这样的：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE2.13.png)

> **译者注：**需要给其它帖子添加标签，才能看到上图所示的相似的帖子。

你已经成功的推荐了相似的帖子给用户。`django-taggit`也包含一个`similar_objects()`管理器，可以用来检索共享的标签。你可以在[这里](http://django-taggit.readthedocs.org/en/latest/api.html)查看所有`django-taggit`管理器。
    
## 2.5 总结

在这一章中，你学习了如何使用 Django 表单和模型表单。你创建了一个可以通过邮件分享网站内容的系统，还为博客创建了评论系统。你为帖子添加了标签，集成了一个可复用的应用，并创建了一个复杂的`QuerySet`，通过相似度检索对象。

下一章中，你会学习如何创建自定义模板标签和过滤器。你还会构建一个自定义的站点地图和帖子的 RSS 源，并在应用中集成一个高级的搜索引擎。























