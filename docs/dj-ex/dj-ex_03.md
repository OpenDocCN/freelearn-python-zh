# 第三章：扩展你的博客应用

上一章介绍了标签的基础知识，你学会了如何在项目中集成第三方应用。本章将会涉及以下知识点：

- 创建自定义模板标签和过滤器
- 添加站点地图和帖子订阅
- 使用 Solr 和 Haystack 构建搜索引擎

## 3.1 创建自定义模板标签和过滤器

Django 提供了大量内置的模板标签，比如`{% if %}`，`{% block %}`。你已经在模板中使用过几个了。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/templates/builtins/)找到所有内置的模板标签和过滤器。

当然，Django 也允许你创建自定义模板标签来执行操作。当你需要在模板中添加功能，而 Django 模板标签不满足需求时，自定义模板标签会非常方便。

### 3.1.1 创建自定义模板标签

Django 提供了以下帮助函数，让你很容易的创建自定义模板标签：

- `simple_tag`：处理数据，并返回一个字符串。
- `inclusion_tag`：处理数据，并返回一个渲染后的模板。
- `assignment_tag`：处理数据，并在上下文中设置一个变量。

模板标签必须存在 Django 应用中。

在`blog`应用目录中，创建`templatetags`目录，并在其中添加一个空的`__init__.py`文件和一个`blog_tags.py`文件。博客应用的目录看起来是这样的：

```py
blog/
	__init__.py
	models.py
	...
	templatetags/
		__init__.py
		blog_tags.py
```

文件名非常重要。你将会在模板中使用该模块名来加载你的标签。

我们从创建一个`simple_tag`标签开始，该标签检索博客中已发布的帖子总数。编辑刚创建的`blog_tags.py`文件，添加以下代码：

```py
from django import template

register = template.Library()

from ..models import Post

@register.simple_tag
def total_posts():
	return Post.published.count()
```

我们创建了一个简单的模板标签，它返回已发布的帖子数量。每一个模板标签模块想要作为一个有效的标签库，都需要包含一个名为`register`的变量。该变量是一个`template.Library`的实例，用于注册你自己的模板标签和过滤器。然后我们使用 Python 函数定义了一个名为`total_posts`的标签，并使用`@register.simple_tag`定义该函数为一个`simple_tag`，并注册它。Django 将会使用函数名作为标签名。如果你想注册为另外一个名字，可以通过`name`属性指定，比如`@register.simple_tag(name='my_tag')`。

> 添加新模板标签模块之后，你需要重启开发服务器，才能使用新的模板标签和过滤器。

使用自定义模板标签之前，你必须使用`{% load %}`标签让它们在模板中生效。像之前提到的，你需要使用包含模板标签和过滤器的 Python 模块名。打开`blog/base.html`模板，在顶部添加`{% load blog_tags %}`，来加载你的模板标签模块。然后使用创建的标签显示帖子总数。只需要在模板中添加`{% total_posts %}`。最终，该模板看起来是这样的：

```py
{% load blog_tags %}
{% load staticfiles %}
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
    <link href="{% static "css/blog.css" %}" rel="stylesheet">
</head>
<body>
    <div id="content">
        {% block content %}
        {% endblock %}
    </div>
    <div id="sidebar">
        <h2>My blog</h2>
        <p>This is my blog. I've written {% total_posts %} posts so far.</p>
    </div>
</body>
</html>
```

因为在项目中添加了新文件，所以需要重启开发服务器。运行`python manage.py runserver`启动开发服务器。在浏览器中打开`http://127.0.0.1:8000/blog/`。你会在侧边栏中看到帖子的总数量，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.1.png)

自定义模板标签的强大之处在于，你可以处理任意数据，并把它添加到任意模板中，不用管视图如何执行。你可以执行`QuerySet`或处理任意数据，并在模板中显示结果。

现在，我们准备创建另一个标签，在博客侧边栏中显示最近发布的帖子。这次我们使用`inclusion_tag`标签。使用该标签，可以利用你的模板标签返回的上下文变量渲染模板。编辑`blog_tags.py`文件，添加以下代码：

```py
@register.inclusion_tag('blog/post/latest_posts.html')
def show_latest_posts(count=5):
	latest_posts = Post.published.order_by('-publish')[:count]
	return {'latest_posts': latest_posts}
```

在这段代码中，我们使用`@register.inclusion_tag`注册模板标签，并用该模板标签的返回值渲染`blog/post/latest_posts.html`模板。我们的模板标签接收一个默认值为 5 的可选参数`count`，用于指定想要显示的帖子数量。我们用该变量限制`Post.published.order_by('-publish')[:count]`查询返回的结果。注意，该函数返回一个字典变量，而不是一个简单的值。`Inclusion`标签必须返回一个字典值作为上下文变量，来渲染指定的模板。`Inclusion`标签返回一个字典。我们刚创建的模板标签可以传递一个显示帖子数量的可选参数，比如`{% show_latest_posts 3 %}`。

现在，在`blog/post/`目录下新建一个`latest_posts.html`文件，添加以下代码：

```py
<ul>
{% for post in latest_posts %}
	<li>
		<a href="{{ post.get_absolute_url }}">{{ post.title }}</a>
	</li>
{% endfor %}
</ul>
```

在这里，我们用模板标签返回的`latest_posts`变量显示一个帖子的无序列表。现在，编辑`blog/base.html`模板，添加新的模板标签，显示最近 3 篇帖子，如下所示：

```py
<div id="sidebar">
	<h2>My blog</h2>
	<p>This is my blog. I've written {% total_posts %} posts so far.</p>
	<h3>Latest posts</h3>
	{% show_latest_posts 3 %}
</div>
```

通过传递显示的帖子数量调用模板标签，并用给定的上下文在原地渲染模板。

现在回到浏览器，并刷新页面。侧边栏现在看起来是这样的：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.2.png)

最后，我们准备创建一个`assignment`标签。`Assignment`标签跟`simple`标签很像，它们把结果存在指定的变量中。我们会创建一个`assignment`标签，用于显示评论最多的帖子。编辑`blog_tags.py`文件，添加以下导入和模板标签：

```py
from django.db.models import Count

@register.assignment_tag
def get_most_commented_posts(count=5):
	return Post.published.annotate(total_comments=Count('comments')).order_by('-total_comments')[:count]
```

这个`QuerySet`使用了`annotate()`函数，调用`Count`汇总函数进行汇总查询。我们构造了一个`QuerySet`，在`totaol_comments`字段中汇总每篇帖子的评论数，并用该字段对`QeurySet`排序。我们还提供了一个可选变量`count`，限制返回的对象数量。

除了`Count`，Django 还提供了`Avg`，`Max`，`Min`，`Sum`汇总函数。你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/db/aggregation/)阅读更多关于汇总函数的信息。

编辑`blog/base.html`模板，在侧边栏的`<div>`元素中添加以下代码：

```py
<h3>Most commented posts</h3>
{% get_most_commented_posts as most_commented_posts %}
<ul>
{% for post in most_commented_posts %}
	<li>
		<a href="{{ post.get_absolute_url }}">{{ post.title }}</a>
	</li>
{% endfor %}
</ul>
```

`Assignment`模板标签的语法是`{% template_tag as variable %}`。对于我们这个模板标签，我们使用`{% get_most_commented_posts as most_commented_posts %}`。这样，我们就在名为`most_commented_posts`的变量中存储了模板标签的结果。接着，我们用无序列表显示返回的帖子。

现在，打开浏览器，并刷新页面查看最终的结果，如下所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.3.png)

你可以在[这里](https://docs.djangoproject.com/en/1.11/howto/custom-template-tags/)阅读更多关于自定义模板标签的信息。

### 3.1.2 创建自定义模板过滤器

Django 内置了各种模板过滤器，可以在模板中修改变量。过滤器就是接收一个或两个参数的 Python 函数——一个是它要应用的变量的值，以及一个可选参数。它们返回的值可用于显示，或者被另一个过滤器处理。一个过滤器看起来是这样的：`{{ variable|my_filter }}`，或者传递一个参数：`{{ variable|my_filter:"foo" }}`。你可以在一个变量上应用任意多个过滤器：`{{ variable|filter1|filter2 }}`，每个过滤器作用于前一个过滤器产生的输出。

我们将创建一个自定义过滤器，可以在博客帖子中使用`markdown`语法，然后在模板中把帖子内容转换为 HTML。`Markdown`是一种纯文本格式化语法，使用起来非常简单，并且可以转换为 HTML。你可以在[这里](http://daringfireball.net/projects/markdown/basics)学习该格式的基本语法。

首先，使用下面的命令安装 Python 的`markdown`模块：

```py
pip install Markdown
```

接着，编辑`blog_tags.py`文件，添加以下代码：

```py
from django.utils.safestring import mark_safe
import markdown

@register.filter(name='markdown')
def markdown_format(text):
	return mark_safe(markdown.markdown(text))
```

我们用与模板标签同样的方式注册模板过滤器。为了避免函数名和`markdown`模块名的冲突，我们将函数命名为`markdown_format`，把过滤器命名为`markdown`，在模板中这样使用：`{{ variable|markdown }}`。Django 会把过滤器生成的 HTML 代码转义。我们使用 Django 提供的`mark_safe`函数，把要在模板中渲染的结果标记为安全的 HTML 代码。默认情况下，Django 不会信任任何 HTML 代码，并且会在输出结果之前进行转义。唯一的例外是标记为安全的转义变量。这种行为可以阻止 Django 输出有潜在危险的 HTML 代码；同时，当你知道返回的是安全的 HTML 代码时，允许这种例外情况发生。

现在，在帖子列表和详情模板中加载你的模板标签。在`post/list.html`和`post/detail.html`模板的`{% extends %}`标签之后添加下面这行代码：

```py
{% load blog_tags %}
```

在`post/detail.html`模板中，把这一行：

```py
{{ post.body|linebreaks }}
```

替换为：

```py
{{ post.body|markdown }}
```

接着，在`post/list.html`模板中，把这一行：

```py
{{ post.body|truncatewords:30|linebreaks }}
```

替换为：

```py
{{ post.body|markdown|truncatewords_html:30 }}
```

过滤器`truncatewords_html`会在指定数量的单词之后截断字符串，并避免没有闭合的 HTML 标签。

现在，在浏览器中打开`http://127.0.0.1:8000/admin/blog/post/add/`，并用下面的正文添加一篇帖子：

```py
This is a post formatted with markdown
--------------------------------------

*This is emphasized* and **this is more emphasized**.

Here is a list:

* One
* Two
* Three

And a [link to the Django website](https://www.djangoproject.com/)
```

打开浏览器，看看帖子是如何渲染的，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.4.png)

正如你所看到的，自定义模板过滤器对自定义格式非常有用。你可以在[这里](https://docs.djangoproject.com/en/1.11/howto/custom-template-tags/#writing-custom-template-filters)查看更多自定义过滤器的信息。

## 3.2 为站点添加站点地图

Django 自带一个站点地图框架，可以为站点动态生成站点地图。站点地图是一个 XML 文件，告诉搜索引擎你的网站有哪些页面，它们之间的关联性，以及更新频率。使用站点地图，可以帮助网络爬虫索引网站的内容。

Django 的站点地图框架依赖`django.contrib.sites`，它允许你将对象关联到在项目中运行的指定网站。当你用单个 Django 项目运行多个站点时，会变得非常方便。要安装站点地图框架，我们需要在项目中启用`sites`和`sitemap`两个应用。编辑项目的`settings.py`文件，并在`INSTALLED_APPS`设置中添加`django.contrib.sites`和`django.contrib.sitemaps`。同时为站点 ID 定义一个新的设置，如下所示：

```py
SITE_ID = 1

INSTALLED_APPS = [
	# ...
	'django.contrib.sites',
	'django.contrib.sitemaps',
]
```

现在，运行以下命令，在数据库中创建`sites`应用的数据库表：

```py
python manage.py migrate
```

你会看到包含这一行的输出：

```py
Applying sites.0001_initial... OK
```

现在，`sites`应用与数据库同步了。在`blog`应用目录中创建`sitemaps.py`文件，添加以下代码：

```py
from django.contrib.sitemaps import Sitemap
from .models import Post

class PostSitemap(Sitemap):
	changefreq = 'weekly'
	priority = 0.9
	
	def items(self):
		return Post.published.all()
		
	def lastmod(self, obj):
		return obj.publish
```

我们创建了一个自定义的站点地图，它继承自`sitemaps`模块的`Sitemap`类。`changefreq`和`priority`属性表示帖子页面的更新频率和它们在网站中的关联性（最大值为 1）。`items()`方法返回这个站点地图中包括的对象的`QuerySet`。默认情况下，Django 调用每个对象的`get_absolute_url()`方法获得它的 URL。记住，我们在第一章创建了该方法，用于获得帖子的标准 URL。如果你希望为每个对象指定 URL，可以在站点地图类中添加`location`方法。`lastmod`方法接收`items()`返回的每个对象，并返回该对象的最后修改时间。`changefreq`和`priority`既可以是方法，也可以是属性。你可以在[官方文档](https://docs.djangoproject.com/en/1.11/ref/contrib/sitemaps/)中查看完整的站点地图参考。

最后，我们只需要添加站点地图的 URL。编辑项目的`urls.py`文件，添加站点地图：

```py
from django.conf.urls import include, url
from django.contrib import admin
from django.contrib.sitemaps.views import sitemap 
from blog.sitemaps import PostSitemap

sitemaps = {
	'posts': PostSitemap,
}

urlpatterns = [
	url(r'^admin/', include(admin.site.urls)),
	url(r'^blog/', 
		include('blog.urls'namespace='blog', app_name='blog')),
	url(r'^sitemap\.xml$', sitemap, {'sitemaps': sitemaps},
		name='django.contrib.sitemaps.views.sitemap'),
]
```

我们在这里包括了必需的导入，并定义了一个站点地图的字典。我们定义了一个匹配`sitemap.xml`的 URL 模式，并使用`sitemap`视图。把`sitemaps`字典传递给`sitemap`视图。在浏览器中打开`http://127.0.0.1:8000/sitemap.xml`，你会看到类似这样的 XML 代码：

```py
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
	<url>
		<loc>http://example.com/blog/2017/04/28/markdown-post/</loc>
		<lastmod>2017-04-28</lastmod>
		<changefreq>weekly</changefreq>
		<priority>0.9</priority>
	</url>
	<url>
		<loc>http://example.com/blog/2017/04/25/one-more-again/</loc>
		<lastmod>2017-04-25</lastmod>
		<changefreq>weekly</changefreq>
		<priority>0.9</priority>
	</url>
	...
</urlset>
```

通过调用`get_absolute_url()`方法，为每篇帖子构造了 URL。我们在站点地图中指定了，`lastmod`属性对应帖子的`publish`字段，`changefreq`和`priority`属性也是从`PostSitemap`中带过来的。你可以看到，用于构造 URL 的域名是`example.com`。该域名来自数据库中的`Site`对象。这个默认对象是在我们同步`sites`框架数据库时创建的。在浏览器中打开`http://127.0.0.1/8000/admin/sites/site/`，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.5.png)

这是`sites`框架的管理视图显示的列表。你可以在这里设置`sites`框架使用的域名或主机，以及依赖它的应用。为了生成存在本机环境中的 URL，需要把域名修改为`127.0.0.1:8000`，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.6.png)

为了方便开发，我们指向了本机。在生产环境中，你需要为`sites`框架使用自己的域名。

## 3.3 为博客帖子创建订阅

Django 内置一个聚合订阅（syndication feed）框架，可以用来动态生成 RSS 或 Atom 订阅，与用`sites`框架创建站点地图的方式类似。

在`blog`应用目录下创建一个`feeds.py`文件，添加以下代码：

```py
from django.contrib.syndication.views import Feed
from django.template.defaultfilters import truncatewords
from .models import Post

class LatestPostsFeed(Feed):
	title = 'My blog'
	link = '/blog/'
	description = 'New posts of my blog.'
	
	def items(self):
		return Post.published.all()[:5]
		
	def item_title(self, item):
		return item.title
		
	def item_description(self, item):
		return truncatewords(item.body, 30)
```

首先，我们从`syndication`框架的`Feed`类继承。`title`，`link`，`description`属性分别对应 RSS 的`<title>`，`<link>`，`<description>`元素。

`items()`方法获得包括在订阅中的对象。我们只检索最近发布的五篇帖子。`item_title()`和`item_description()`方法接收`items()`返回的每一个对象，并返回每一项的标题和描述。我们用内置的`truncatewords`模板过滤器截取前 30 个单词，用于构造博客帖子的描述。

现在编辑`blog`应用的`urls.py`文件，导入刚创建的`LatestPostsFeed`，并在新的 URL 模式中实例化：

```py
from .feeds import LatestPostsFeed

urlpatterns = [
	# ...
	url(r'^feed/$', LatestPostsFeed(), name='post_feed'),
]
```

在浏览器中打开`http://127.0.0.1:8000/blog/feed/`，你会看到 RSS 订阅包括了最近五篇博客帖子：

```py
<?xml version="1.0" encoding="utf-8"?>
<rss xmlns:atom="http://www.w3.org/2005/Atom" version="2.0">
	<channel>
		<title>My blog</title>
		<link>http://127.0.0.1:8000/blog/</link>
		<description>New posts of my blog</description>
		<atom:link href="http://127.0.0.1:8000/blog/feed/" rel="self"></atom:link>
		<language>en-us</language>
		<lastBuildDate>Fri, 28 Apr 2017 05:44:43 +0000</lastBuildDate>
		<item>
			<title>One More Again</title>
			<link>http://127.0.0.1:8000/blog/2017/04/25/one-more-again/</link>
			<description>Post body.</description>
			<guid>http://127.0.0.1:8000/blog/2017/04/25/one-more-again/</guid>
		</item>
		<item>
			<title>Another Post More</title>
			<link>http://127.0.0.1:8000/blog/2017/04/25/another-post-more/</link>
			<description>Post body.</description>
			<guid>http://127.0.0.1:8000/blog/2017/04/25/another-post-more/</guid>
		</item>
		...
	</channel>
</rss>
```

如果你在 RSS 客户端中打开这个 URL，你会看到一个用户界面友好的订阅。

最后一步是在博客的侧边栏添加一个订阅链接。打开`blog/base.html`模板，在侧边栏`<div>`的帖子总数后面添加这一行代码：

```py
<p><a href="{% url "blog:post_feed" %}">Subscribe to my RSS feed</a></p>
```

现在，在浏览器中打开`http://127.0.0.1:8000/blog/`，你会看到如下图所示的侧边栏：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE3.7.png)

## 3.4 使用 Solr 和 Haystack 添加搜索引擎

> **译者注：**暂时跳过这一节的翻译，对于一般的博客，实在是用不上搜索引擎。

## 3.5 总结

在这一章中，你学习了如何创建自定义 Django 模板标签和过滤器，为模板提供自定义的功能。你还创建了站点地图，便于搜索引擎爬取你的网站，以及一个 RSS 订阅，便于用户订阅。同时，你在项目中使用 Haystack 集成了 Solr，为博客构建了一个搜索引擎。

在下一章，你会学习如何使用 Django 的`authentication`构建社交网站，创建自定义的用户资料，以及社交认证。



























