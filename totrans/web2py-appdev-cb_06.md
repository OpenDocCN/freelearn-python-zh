# 第六章 使用第三方库

在本章中，我们将涵盖以下配方：

+   定制日志

+   聚合源

+   显示推文

+   使用 matplotlib 绘图

+   使用 RSS 小部件扩展 PluginWiki

# 简介

Python 的力量来自于可用的众多**第三方库**。本章的目标不是讨论这些第三方库的 API，因为这个任务将是巨大的。相反，目标是通过定制日志、检测可能的问题、在模型文件中创建自己的 API 以及将新接口打包为插件来展示如何正确地完成这项工作。

# 定制日志

Python 的日志功能强大且灵活，但实现起来可能很复杂。此外，web2py 中的日志记录引入了一组新的问题。这个配方提供了一个在 web2py 中有效日志记录的方法，利用 Python 的本地日志功能。

Python 的本地日志框架使用一个 logger 与 handler 的组合，其中一个或多个 logger 将日志记录到一个或多个 handler。日志框架使用单例模型来管理其 logger，因此以下代码行通过该名称返回一个全局`Logger`实例，仅在首次访问时实例化：

```py
logging.getLogger('name')

```

默认情况下，Python 进程以单个 root，`logger (name == ")`开始，只有一个 handler 将日志记录到`stdout`。

## 如何做到...

在 web2py 中的日志记录涉及一些新问题，如下所述：

+   在应用级别配置和控制日志

+   只配置一次 logger

+   实现简单的日志记录语法

Python 的本地日志框架已经为每个进程维护了一个全局的命名 logger 集合。但在 web2py 中，由于应用在同一个进程中运行，logger 是跨应用共享的。如果我们想根据应用特定地配置和控制 logger，我们需要一个不同的解决方案。

创建特定于应用的 logger 的一个简单方法是在 logger 的名称中包含应用名称。

```py
logging.getLogger(request.application)

```

这可以在模型文件中完成。现在，跨多个应用使用的相同代码将为每个应用返回不同的 logger。

我们希望在启动时只配置一次 logger。然而，当访问一个命名 logger 时，Python 没有提供检查 logger 是否已经存在的方法。

确保 logger 只配置一次的最简单方法，是检查它是否有任何 handler，如下所示：

```py
def get_configured_logger(name):
	logger = logging.getLogger(name)
	if len(logger.handlers) == 0:
		# This logger has no handlers, so we can assume
		# it hasn't yet been configured.
		# (Configure logger)
	return logger

```

注意，如果`loggername`为空，你需要检索 Python 的 root logger。默认的 root logger 已经关联了一个 handler，所以你会检查 handler 的数量为`1`。root logger 不能被设置为特定于应用。

当然，我们不希望每次记录日志时都要调用`get_configured_logger`。相反，我们可以在模型中一次性进行全局赋值，并在整个应用程序中使用它。赋值将在每次你在控制器中使用记录器时执行，但实例化和配置只会在第一次访问时发生。

所以最后，只需将此代码放置在模型中：

```py
import logging, logging.handlers
def get_configured_logger(name):
	logger = logging.getLogger(name)
	if (len(logger.handlers) == 0):
		# This logger has no handlers, so we can assume
		# it hasn't yet been configured
		# (Configure logger)
		pass
	return logger
logger = get_configured_logger(request.application)

```

在以下示例中，在你的控制器中使用它：

```py
logger.debug('debug message')
logger.warn('warning message')
logger.info('information message')
logger.error('error message')

```

## 更多...

我们可以用自定义的应用程序级记录器做什么？例如，我们可以重新编程 Google App Engine 上的记录，让消息进入数据存储表。以下是我们可以这样做的步骤：

```py
import logging, logging.handlers

class GAEHandler(logging.Handler):
	"""
	Logging handler for GAE DataStore
	"""
	def emit(self, record):
		from google.appengine.ext import db
		class Log(db.Model):
		name = db.StringProperty()
		level = db.StringProperty()
		module = db.StringProperty()
		func_name = db.StringProperty()
		line_no = db.IntegerProperty()
		thread = db.IntegerProperty()
		thread_name = db.StringProperty()
		process = db.IntegerProperty()
		message = db.StringProperty(multiline=True)
		args = db.StringProperty(multiline=True)
		date = db.DateTimeProperty(auto_now_add=True)
	log = Log()
	log.name = record.name
	log.level = record.levelname
	log.module = record.module
	log.func_name = record.funcName
	log.line_no = record.lineno
	log.thread = record.thread
	log.thread_name = record.threadName
	log.process = record.process
	log.message = record.msg
	log.args = str(record.args)
	log.put()

def get_configured_logger(name):
	logger = logging.getLogger(name)
	if len(logger.handlers) == 0:
		if request.env.web2py_runtime_gae:
			# Create GAEHandler
			handler = GAEHandler()
		else:
			# Create RotatingFileHandler
			import os
			formatter = "%(asctime)s %(levelname)s " + \
				"%(process)s %(thread)s "+ \
				"%(funcName)s():%(lineno)d %(message)s"
			handler = logging.handlers.RotatingFileHandler(
				os.path.join(request.folder,'private/app.log'),
				maxBytes=1024,backupCount=2)
			handler.setFormatter(logging.Formatter(formatter))
			handler.setLevel(logging.DEBUG)
			logger.addHandler(handler)
			logger.setLevel(logging.DEBUG)
			logger.debug(name + ' logger created') # Test entry
		else:
			logger.debug(name + ' already exists') # Test entry
	return logger

#### Assign application logger to a global var
logger = get_configured_logger(request.application)

```

你可以在以下网址了解更多关于这个主题的信息：

+   [`docs.python.org/library/logging.html`](http://docs.python.org/library/logging.html)

+   [`github.com/apptactic/apptactic-python/blob/master/logging/custom_handlers.py`](http://github.com/apptactic/apptactic-python/blob/master/logging/custom_handlers.py)

# 汇总源

在这个菜谱中，我们将使用**feedparser**和**rss2**构建一个 RSS 源聚合器。我们称之为**Planet Web2py**，因为它将基于字符串`web2py`过滤 rss 条目。

## 如何做到...

1.  创建一个`models/db_feed.py`，内容如下：

    ```py
    db.define_table("feed",
    	Field("name"),
    	Field("author"),
    	Field("email", requires=IS_EMAIL()),
    	Field("url", requires=IS_URL(), comment="RSS/Atom feed"),
    	Field("link", requires=IS_URL(), comment="Blog href"),
    	Field("general", "boolean", comment="Many categories (needs filters)"),
    )

    ```

1.  然后在`controllers/default.py`中添加一个`planet`函数，通过使用`feedparser:`获取所有源来渲染一个基本页面

    ```py
    def planet():
    	FILTER = 'web2py'
    	import datetime
    	import re
    	import gluon.contrib.rss2 as rss2
    	import gluon.contrib.feedparser as feedparser

    	# filter for general (not categorized) feeds
    	regex = re.compile(FILTER,re.I)
    	# select all feeds
    	feeds = db(db.feed).select()
    	entries = []

    	for feed in feeds:
    		# fetch and parse feeds
    		d = feedparser.parse(feed.url)
    		for entry in d.entries:
    			# filter feed entries
    			if not feed.general or regex.search(entry.description):
    				# extract entry attributes
    				entries.append({
    					'feed': {'author':feed.author,
    							'link':feed.link,
    							'url':feed.url,
    							'name':feed.name},
    							'title': entry.title,
    							'link': entry.link,
    							'description': entry.description,
    							'author': hasattr(entry, 'author_detail') \
    							and entry.author_detail.name \
    							or feed.author,
    							'date': datetime.datetime(*entry.date_parsed[:6])
    							})

    	# sort entries by date, descending
    	entries.sort(key=lambda x: x['date'],reverse=True)
    	now = datetime.datetime.now()

    	# aggregate rss2 feed with parsed entries
    	rss = rss2.RSS2(title="Planet web2py",
    	link = URL("planet").encode("utf8"),
    	description = "planet author",
    		lastBuildDate = now,
    		items = [rss2.RSSItem(
    				title = entry['title'],
    				link = entry['link'],
    				description = entry['description'],
    				author = entry['author'],
    				# guid = rss2.Guid('unknown'),
    			pubDate = entry['date']) for entry in entries]
    		)
    	# return new rss feed xml
    	response.headers['Content-Type']='application/rss+xml'
    	return rss2.dumps(rss)

    ```

在你能够使用这个函数之前，你需要在`db.feed`中添加一些源网址，例如，使用`appadmin`。

以下是一些关于 web2py 的示例 RSS 源：

+   [`reingart.blogspot.com/feeds/posts/default/-/web2py`](http://reingart.blogspot.com/feeds/posts/default/-/web2py)

+   [`web2py.wordpress.com/feed/`](http://web2py.wordpress.com/feed/)

+   [`www.web2pyslices.com/main/slices/get_latest.rss`](http://www.web2pyslices.com/main/slices/get_latest.rss)

+   [`martin.tecnodoc.com.ar/myblog/default/feed_articles.rss`](http://martin.tecnodoc.com.ar/myblog/default/feed_articles.rss)

## 更多...

可以在以下网址找到 web2py 示例 planet 的工作示例：

[`www.web2py.com.ar/planet/`](http://www.web2py.com.ar/planet/)

完整示例的完整源代码（planet-web2py）发布在 Google 代码项目中，可在以下网址找到：

[`code.google.com/p/planet-web2py/`](http://code.google.com/p/planet-web2py/)

该应用程序存储`rss`源条目，以加快聚合，并定期刷新源。

# 显示推文

在这个菜谱中，我们将展示如何使用`simplejson`显示最近的推文，并使用 web2py 包含的工具获取它。

## 如何做到...

1.  首先，创建一个`models/0.py`文件来存储基本配置，如下所示：

    ```py
    TWITTER_HASH = "web2py"

    ```

1.  在`controllers/default.py`中添加一个 Twitter 函数，通过使用获取工具获取所有推文并使用`simplejson:`解析它来渲染一个基本页面部分

    ```py
    @cache(request.env.path_info,time_expire=60*15,
    	cache_model=cache.r
    	am)
    def twitter():
    	session.forget()
    	session._unlock(response)
    	import gluon.tools
    	import gluon.contrib.simplejson as sj
    	try:
    			page = gluon.tools.fetch(' http://search.twitter.com/search.
    	json?q=%%40%s'
    				% TWITTER_HASH)
    			data = sj.loads(page, encoding="utf-8")['results']
    			d = dict()
    			for e in data:
    				d[e["id"]] = e
    			r = reversed(sorted(d))
    			return dict(tweets = [d[k] for k in r])
    		else:
    			return 'disabled'
    	except Exception, e:
    		return DIV(T('Unable to download because:'),BR(),str(e))

    ```

1.  在`views/default/twitter.load`中创建一个用于 twitter 组件的视图，我们将渲染每条推文：

    ```py
    <OL>
    {{ for t in tweets: }}
    	<LI>
    	{{ =DIV(H5(t["from_user_name"])) }}
    	{{ =DIV(t["text"]) }}
    	</LI>
    {{ pass }}
    </OL>

    ```

1.  然后，在 `default/index.html` 中，添加使用 LOAD（jQuery）加载推文的部分：

    ```py
    {{if TWITTER_HASH:}}
    	<div class="box">
    		<h3>{{=T("%s Recent Tweets") % TWITTER_HASH}}</h3>
    		<div id="tweets"> {{=LOAD('default','twitter.
    load',ajax=True)}}</div>
    	</div>{{pass}}

    ```

### 更多内容...

您可以使用 CSS 样式来增强推文部分。创建一个 `static/css/tweets.css` 文件，包含以下代码：

```py
/* Tweets */

#tweets ol {
	margin: 1em 0;
}

#tweets ol li {
	background: #d3e5ff;
	list-style: none;
	-moz-border-radius: 0.5em;
	border-radius: 0.5em;
	padding: 0.5em;
	margin: 1em 0;
	border: 1px solid #aaa;
}

#tweets .entry-date {
	font-weight: bold;
	display: block;
}

```

然后，将 CSS 文件添加到响应中：

```py
def index():
	response.files.append(URL("static","css/tweets.css"))
	response.flash = T('You are successfully running web2py.')
	return dict(message=T('Hello World'))

```

您可以使用以下属性进一步自定义此配方，这是该推文 API 为每条推文返回的：

+   `iso_language_code`

+   `to_user_name`

+   `to_user_id_str`

+   `profile_image_url_https`

+   `from_user_id_str`

+   `text`

+   `from_user_name`

+   `in_reply_to_status_id_str`

+   `profile_image_url`

+   `id'`

+   `to_user`

+   `source`

+   `in_reply_to_status_id`

+   `id_str'`

+   `from_user`

+   `from_user_id`

+   `to_user_id`

+   `geo`

+   `created_at`

+   `metadata`

记住，在这个配方中，我们使用缓存来加速页面加载（15 分钟=60*15）。如果您需要更改它，请修改 @cache(…,time_expire=…)

# 使用 matplotlib 绘图

**Matplotlib** 是 Python 中最先进的绘图库。一些示例可以在以下 URL 中找到：

[`matplotlib.sourceforge.net/gallery.html`](http://matplotlib.sourceforge.net/gallery.html)

Matplotlib 可以用于以下两种模型：

+   PyLab（一个 Matlab 兼容模式）

+   更多的 Pythonic API

大多数文档使用 PyLab，这是一个问题，因为 PyLab 共享全局状态，并且与 Web 应用程序不兼容。我们需要使用更 Pythonic 的 API。

## 如何做到这一点...

Matplotlib 有许多后端可以用于在 GUI 中打印或打印到文件。

为了在 Web 应用中使用 matplotlib，我们需要指导它实时生成图形，将其打印到内存映射文件中，并将文件内容流式传输到页面访问者。

在这里，我们展示了一个实用函数来绘制以下形式的数据集：

```py
name = [(x0,y0),(x1,y1),...(xn,yn)]

```

1.  创建一个 `models/matplotlib.py` 文件，包含以下代码：

    ```py
    from matplotlib.backends.backend_agg import FigureCanvasAgg as
    	FigureCanvas
    from matplotlib.figure import Figure
    import cStringIO

    def myplot(title='title',xlab='x',ylab='y',mode='plot',
    	data={'xxx':[(0,0),(1,1),(1,2),(3,3)],
    		'yyy':[(0,0,.2,.2),(2,1,0.2,0.2),(2,2,0.2,0.2),
    			(3,3,0.2,0.3)]}):
    	fig=Figure()
    	fig.set_facecolor('white')
    	ax=fig.add_subplot(111)
    	if title: ax.set_title(title)
    	if xlab: ax.set_xlabel(xlab)
    	if ylab: ax.set_ylabel(ylab)
    	legend=[]
    	keys=sorted(data)
    	for key in keys:
    		stream = data[key]
    		(x,y)=([],[])
    	for point in stream:
    		x.append(point[0])
    		y.append(point[1])
    	if mode=='plot':
    		ell=ax.plot(x, y)
    		legend.append((ell,key))
    	if mode=='hist':
    		ell=ax.hist(y,20)
    	if legend:
    		ax.legend([x for (x,y) in legend], [y for (x,y) in
    			legend],
    			'upper right', shadow=True)
    	canvas=FigureCanvas(fig)
    	stream=cStringIO.StringIO()
    	canvas.print_png(stream)
    return stream.getvalue()

    ```

1.  您现在可以尝试它，在您的控制器中使用以下操作：

    ```py
    def test_images():
    	return HTML(BODY(
    		IMG(_src=URL('a_plot')),
    		IMG(_src=URL('a_histogram'))))

    def a_plot():
    		response.headers['Content-Type']='image/png'
    		return myplot(data={'data':[(0,0),(1,1),(2,4),(3,9),(4,16)]})

    def a_histogram():
    		response.headers['Content-Type']='image/png'
    		return myplot(data={'data':[(0,0),(1,1),(2,4),(3,9),(4,16)]},
    			mode='hist')

    ```

    +   `http://.../test_images`

    +   `http://.../a_plot.png`

    +   `http://.../a_histogram.png`

## 它是如何工作的...

当您访问 `test_images` 时，它会生成一个包含图形的 HTML：

```py
<img src="img/a_plot.png"/>
<img src="img/a_histogram.png"/>

```

每个这些 URL 都调用 `models/matplotlib.py` 中的 myplot 函数。绘图函数生成一个包含一个子图的图形（一组 X-Y 轴）。然后在该子图 `ax` 上绘制（当 `mode="plot"` 时连接点，当 `mode="hist"` 时绘制直方图），并将图形打印到一个名为 **stream** 的内存映射画布上。然后它从流中读取二进制数据并返回。

## 更多内容...

在示例中，关键函数是 `ax.plot` 和 `ax.hist`，它们在子图的轴上绘制。您现在可以通过复制提供的 `myplot` 函数，重命名它，并用其他函数替换 `ax.plot` 或 `ax.hist`（例如散点图、误差线等）来创建更多的绘图函数。现在，您应该可以直接从 matplotlib 文档中找到。

# 使用 RSS 小部件扩展 PluginWiki

**PluginWiki**是 web2py 插件中最复杂的。它添加了许多功能；特别是，它为您的应用程序添加了一个 CMS，并定义了可以嵌入 CMS 页面以及您自己的视图的小部件。此插件可以扩展，这里我们向您展示如何添加一个新的小部件。

有关插件-wiki 的更多信息，请参阅：

[`web2py.com/examples/default/download`](http://web2py.com/examples/default/download)

## 如何操作...

1.  创建一个名为`models/plugin_wiki_rss.py`的文件，并将以下代码添加到其中：

    ```py
    class PluginWikiWidgets(PluginWikiWidgets):
    	@staticmethod
    	def aggregator(feed, max_entries=5):
    		import gluon.contrib.feedparser as feedparser
    		d = feedparser.parse(feed)
    		title = d.channel.title
    		link = d.channel.link
    		description = d.channel.description
    		div = DIV(A(B(title[0], _href=link[0])))
    		created_on = request.now
    		for entry in d.entries[0:max_entries]:
    			div.append(A(entry.title,' - ', entry.updated,
    				_href=entry.link))
    			div.append(DIV(description))
    		return div

    ```

1.  现在，您可以使用以下语法将此小部件包含在 PluginWiki CMS 页面上：

    ```py
    name:aggregator
    feed:http://rss.cbc.ca/lineup/topstories.xml
    max_entries:4

    ```

    您也可以使用以下语法将其包含在任何 web2py 页面上：

    ```py
    {{=plugin_wiki.widget('aggregator',max_entries=4,
    	feed='http://rss.cbc.ca/lineup/topstories.xml')}}

    ```

### 还有更多...

web2py 用户**博格丹**通过使用随 PluginWiki 一起提供的 jQuery UI 对此插件进行了一些修改，使其更加流畅。以下是改进后的插件：

```py
class PluginWikiWidgets(PluginWikiWidgets):
	@staticmethod
	def aggregator(feeds, max_entries=5):
		import gluon.contrib.feedparser as feedparser
		lfeeds = feeds.split(",")
		strg='''
			<script>
				var divDia = document.createElement("div");
				divDia.id ="dialog";
				document.body.appendChild(divDia);
				var jQuerydialog=jQuery("#dialog").dialog({
				autoOpen: false,
				draggable: false,
				resizable: false,
				width: 500
				});
			</script>
			'''

	for feed in lfeeds:
		d = feedparser.parse(feed)
		title=d.channel.title
		link = d.channel.link
		description = d.channel.description
		created_on = request.now
		strg+='<a class="feed_title" href="%s">%s</a>' % \
		(link[0],title[0])
	for entry in d.entries[0:max_entries]:
	strg+='''
		<div class="feed_entry">
		<a rel="%(description)s" href="%(link)s">
		%(title)s - %(updated)s</a>
		<script>
			jQuery("a").mouseover(function () {
			var msg = jQuery(this).attr("rel");
			if (msg) {
				jQuerydialog[0].innerHTML = msg;
				jQuerydialog.dialog("open");
				jQuery(".ui-dialog-titlebar").hide();
				}
			}).mousemove(function(event) {
			jQuerydialog.dialog("option", "position", {
				my: "left top",
				at: "right bottom",
				of: event,
				offset: "10 10"
				});
			}).mouseout(function(){
				jQuerydialog.dialog("close");
			});
			</script></div>''' % entry

return XML(strg)

```

此脚本的修改版本不使用辅助工具，而是使用原始 HTML 以提高速度，对 CSS 友好，并使用对话框弹出窗口来输入详细信息。
