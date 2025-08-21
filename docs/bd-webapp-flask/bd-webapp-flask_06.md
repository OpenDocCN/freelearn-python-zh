# 第六章：但是我现在想休息妈妈！

REST 是一种架构风格，由于其许多特性和架构约束（如可缓存性、无状态行为和其接口要求），近年来一直在获得动力。

### 提示

有关 REST 架构的概述，请参阅[`www.drdobbs.com/Web-development/restful-Web-services-a-tutorial/240169069`](http://www.drdobbs.com/Web-development/restful-Web-services-a-tutorial/240169069)和[`en.wikipedia.org/wiki/Representational_state_transfer`](http://en.wikipedia.org/wiki/Representational_state_transfer)。

本章我们将专注于 RESTful Web 服务和 API——即遵循 REST 架构的 Web 服务和 Web API。让我们从开始说起：什么是 Web 服务？

Web 服务是一个可以被你的应用程序查询的 Web 应用程序，就像它是一个 API 一样，提高了用户体验。如果你的 RESTful Web 服务不需要从传统的 UI 界面调用，并且可以独立使用，那么你拥有的是一个**RESTful Web 服务 API**，简称“RESTful API”，它的工作方式就像一个常规 API，但通过 Web 服务器。

对 Web 服务的调用可能会启动批处理过程、更新数据库或只是检索一些数据。对服务可能执行的操作没有限制。

RESTful Web 服务应该通过**URI**（类似于 URL）访问，并且可以通过任何 Web 协议访问，尽管**HTTP**在这里是王者。因此，我们将专注于**HTTP**。我们的 Web 服务响应，也称为资源，可以具有任何所需的格式；如 TXT、XML 或 JSON，但最常见的格式是 JSON，因为它非常简单易用。我们还将专注于 JSON。在使用 HTTP 与 Web 服务时，一种常见的做法是使用 HTTP 默认方法（`GET`、`POST`、`PUT`、`DELETE`和`OPTIONS`）向服务器提供关于我们想要实现的更多信息。这种技术允许我们在同一个服务中拥有不同的功能。

对`http://localhost:5000/age`的服务调用可以通过`GET`请求返回用户的年龄，或通过`DELETE`请求删除其值。

让我们看看每个*通常使用*的方法通常用于什么：

+   `GET`：这用于检索资源。你想要信息？不需要更新数据库？使用 GET！

+   `POST`：这用于将新数据插入服务器，比如在数据库中添加新员工。

+   `PUT`：这用于更新服务器上的数据。你有一个员工决定在系统中更改他的昵称？使用`PUT`来做到这一点！

+   `DELETE`：这是你在服务器上删除数据的最佳方法！

+   `OPTIONS`：这用于询问服务支持哪些方法。

到目前为止，有很多理论；让我们通过一个基于 Flask 的 REST Web 服务示例来实践。

首先，安装示例所需的库：

```py
pip install marshmallow

```

现在，让我们来看一个例子：

```py
# coding:utf-8

from flask import Flask, jsonify
from flask.ext.sqlalchemy import SQLAlchemy

from marshmallow import Schema

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/articles.sqlite'

db = SQLAlchemy(app)

class Article(db.Model):
    __tablename__ = 'articles'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text(), nullable=False)

    def __unicode__(self):
        return self.content

# we use marshmallow Schema to serialize our articles
class ArticleSchema(Schema):
    """
    Article dict serializer
    """
    class Meta:
        # which fields should be serialized?
        fields = ('id', 'title', 'content')

article_schema = ArticleSchema()
# many -> allow for object list dump
articles_schema = ArticleSchema(many=True)

@app.route("/articles/", methods=["GET"])
@app.route("/articles/<article_id>", methods=["GET"])
def articles(article_id=None):
    if article_id:
        article = Article.query.get(article_id)

        if article is None:
            return jsonify({"msgs": ["the article you're looking for could not be found"]}), 404

        result = article_schema.dump(article)
        return jsonify({'article': result})
    else:
        # never return the whole set! As it would be very slow
        queryset = Article.query.limit(10)
        result = articles_schema.dump(queryset)

        # jsonify serializes our dict into a proper flask response
        return jsonify({"articles": result.data})

db.create_all()

# let's populate our database with some data; empty examples are not that cool
if Article.query.count() == 0:
    article_a = Article(title='some title', content='some content')
    article_b = Article(title='other title', content='other content')

    db.session.add(article_a)
    db.session.add(article_b)
    db.session.commit()

if __name__ == '__main__':
    # we define the debug environment only if running through command line
    app.config['SQLALCHEMY_ECHO'] = True
    app.debug = True
    app.run()
```

在前面的示例中，我们创建了一个 Web 服务，使用 GET 请求来查询文章。引入了`jsonify`函数，因为它用于将 Python 对象序列化为 Flask JSON 响应。我们还使用 marshmallow 库将 SQLAlchemy 结果序列化为 Python 字典，因为没有原生 API 可以做到这一点。

让我们逐步讨论这个例子：

首先，我们创建我们的应用程序并配置我们的 SQLAlchemy 扩展。然后定义`Article`模型，它将保存我们的文章数据，以及一个 ArticleSchema，它允许 marshmallow 将我们的文章序列化。我们必须在 Schema Meta 中定义应该序列化的字段。`article_schema`是我们用于序列化单篇文章的模式实例，而`articles_schema`序列化文章集合。

我们的文章视图有两个定义的路由，一个用于文章列表，另一个用于文章详情，返回单篇文章。

在其中，如果提供了`article_id`，我们将序列化并返回请求的文章。如果数据库中没有与`article_id`对应的记录，我们将返回一个带有给定错误和 HTTP 代码 404 的消息，表示“未找到”状态。如果`article_id`为`None`，我们将序列化并返回 10 篇文章。您可能会问，为什么不返回数据库中的所有文章？如果我们在数据库中有 10,000 篇文章并尝试返回那么多，我们的服务器肯定会出问题；因此，避免返回数据库中的所有内容。

这种类型的服务通常由使用 JavaScript（如 jQuery 或 PrototypeJS）的 Ajax 请求来消耗。在发送 Ajax 请求时，这些库会添加一个特殊的标头，使我们能够识别给定请求是否实际上是 Ajax 请求。在我们的前面的例子中，我们为所有 GET 请求提供 JSON 响应。

### 提示

不懂 Ajax？访问[`www.w3schools.com/Ajax/ajax_intro.asp`](http://www.w3schools.com/Ajax/ajax_intro.asp)。

我们可以更加选择性，只对 Ajax 请求发送 JSON 响应。常规请求将收到纯 HTML 响应。要做到这一点，我们需要对视图进行轻微更改，如下所示：

```py
from flask import request
…

@app.route("/articles/", methods=["GET"])
@app.route("/articles/<article_id>", methods=["GET"])
def articles(article_id=None):
    if article_id:
        article = Article.query.get(article_id)

        if request.is_xhr:
            if article is None:
                return jsonify({"msgs": ["the article you're looking for could not be found"]}), 404

            result = article_schema.dump(article)
            return jsonify({'article': result})
        else:
            if article is None:
                abort(404)

            return render_template('article.html', article=article)
    else:
        queryset = Article.query.limit(10)

        if request.is_xhr:
            # never return the whole set! As it would be very slow
            result = articles_schema.dump(queryset)

            # jsonify serializes our dict into a proper flask response
            return jsonify({"articles": result.data})
        else:
            return render_template('articles.html', articles=queryset)
```

`request`对象有一个名为`is_xhr`的属性，您可以检查该属性以查看请求是否实际上是 Ajax 请求。如果我们将前面的代码拆分成几个函数，例如一个用于响应 Ajax 请求，另一个用于响应纯 HTTP 请求，那么我们的前面的代码可能会更好。为什么不尝试重构代码呢？

我们的最后一个示例也可以采用不同的方法；我们可以通过 Ajax 请求加载所有数据，而不向其添加上下文变量来呈现 HTML 模板。在这种情况下，需要对代码进行以下更改：

```py
from marshmallow import Schema, fields
class ArticleSchema(Schema):
    """
      Article dict serializer
      """
      url = fields.Method("article_url")
      def article_url(self, article):
          return article.url()

      class Meta:
          # which fields should be serialized?
          fields = ('id', 'title', 'content', 'url')

@app.route("/articles/", methods=["GET"])
@app.route("/articles/<article_id>", methods=["GET"])
def articles(article_id=None):
    if article_id:
        if request.is_xhr:
            article = Article.query.get(article_id)
            if article is None:
                return jsonify({"msgs": ["the article you're looking for could not be found"]}), 404

            result = article_schema.dump(article)
            return jsonify({'article': result})
        else:
            return render_template('article.html')
    else:
        if request.is_xhr:
            queryset = Article.query.limit(10)
            # never return the whole set! As it would be very slow
            result = articles_schema.dump(queryset)

            # jsonify serializes our dict into a proper flask response
            return jsonify({"articles": result.data})
        else:
            return render_template('articles.html')
```

我们在模式中添加了一个新字段`url`，以便从 JavaScript 代码中访问文章页面的路径，因为我们返回的是一个 JSON 文档而不是 SQLAlchemy 对象，因此无法访问模型方法。

`articles.html`文件将如下所示：

```py
<!doctype html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Articles</title>
</head>
<body>
<ul id="articles">
</ul>

<script type="text/javascript" src="img/jquery-2.1.3.min.js"></script>
<script type="text/javascript">
  // only execute after loading the whole HTML
  $(document).ready(function(){
    $.ajax({
      url:"{{ url_for('.articles') }}",
      success: function(data, textStatus, xhr){
        $(data['articles']).each(function(i, el){
          var link = "<a href='"+ el['url'] +"'>" + el['title'] + "</a>";
          $("#articles").append("<li>" + link + "</li>");
        });}});}); // don't do this in live code
</script>
</body>
</html>
```

在我们的模板中，文章列表是空的；然后在使用 Ajax 调用我们的服务后进行填充。如果您测试完整的示例，Ajax 请求非常快，您甚至可能都没有注意到页面在填充 Ajax 之前是空的。

# 超越 GET

到目前为止，我们已经有了一些舒适的 Ajax 和 RESTful Web 服务的示例，但我们还没有使用服务将数据记录到我们的数据库中。现在试试吧？

使用 Web 服务记录到数据库与我们在上一章中所做的并没有太大的不同。我们将从 Ajax 请求中接收数据，然后检查使用了哪种 HTTP 方法以决定要做什么，然后我们将验证发送的数据并保存所有数据（如果没有发现错误）。在第四章*请填写这张表格，夫人*中，我们谈到了 CSRF 保护及其重要性。我们将继续使用我们的 Web 服务对数据进行 CSRF 验证。诀窍是将 CSRF 令牌添加到要提交的表单数据中。有关示例 HTML，请参见随附的电子书代码。

这是我们的视图支持`POST`，`PUT`和`REMOVE`方法：

```py
@app.route("/articles/", methods=["GET", "POST"])
@app.route("/articles/<int:article_id>", methods=["GET", "PUT", "DELETE"])
def articles(article_id=None):
    if request.method == "GET":
        if article_id:
            article = Article.query.get(article_id)

            if request.is_xhr:
                if article is None:
                    return jsonify({"msgs": ["the article you're looking for could not be found"]}), 404

                result = article_schema.dump(article)
                return jsonify({': result.data})

            return render_template('article.html', article=article, form=ArticleForm(obj=article))
        else:
            if request.is_xhr:
                # never return the whole set! As it would be very slow
                queryset = Article.query.limit(10)
                result = articles_schema.dump(queryset)

                # jsonify serializes our dict into a proper flask response
                return jsonify({"articles": result.data})
    elif request.method == "POST" and request.is_xhr:
        form = ArticleForm(request.form)

        if form.validate():
            article = Article()
            form.populate_obj(article)
            db.session.add(article)
            db.session.commit()
            return jsonify({"msgs": ["article created"]})
        else:
            return jsonify({"msgs": ["the sent data is not valid"]}), 400

    elif request.method == "PUT" and request.is_xhr:
        article = Article.query.get(article_id)

        if article is None:
            return jsonify({"msgs": ["the article you're looking for could not be found"]}), 404

        form = ArticleForm(request.form, obj=article)

        if form.validate():
            form.populate_obj(article)
            db.session.add(article)
            db.session.commit()
            return jsonify({"msgs": ["article updated"]})
        else:
            return jsonify({"msgs": ["the sent data was not valid"]}), 400
    elif request.method == "DELETE" and request.is_xhr:
        article = Article.query.get(article_id)

        if article is None:
            return jsonify({"msgs": ["the article you're looking for could not be found"]}), 404

        db.session.delete(article)
        db.session.commit()
        return jsonify({"msgs": ["article removed"]})

    return render_template('articles.html', form=ArticleForm())
```

好吧，事实就是这样，我们再也不能隐藏了；在同一页中处理 Web 服务和纯 HTML 渲染可能有点混乱，就像前面的例子所示。即使您将函数按方法分割到其他函数中，事情可能看起来也不那么好。通常的模式是有一个视图用于处理 Ajax 请求，另一个用于处理“正常”请求。只有在方便的情况下才会混合使用两者。

# Flask-Restless

Flask-Restless 是一个扩展，能够自动生成整个 RESTful API，支持`GET`、`POST`、`PUT`和`DELETE`，用于你的 SQLAlchemy 模型。大多数 Web 服务不需要更多。使用 Flask-Restless 的另一个优势是可以扩展自动生成的方法，进行身份验证验证、自定义行为和自定义查询。这是一个必学的扩展！

让我们看看我们的 Web 服务在 Flask-Restless 下会是什么样子。我们还需要为这个示例安装一个新的库：

```py
pip install Flask-Restless

```

然后：

```py
# coding:utf-8

from flask import Flask, url_for
from flask.ext.restless import APIManager
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/employees.sqlite'

db = SQLAlchemy(app)

class Article(db.Model):
    __tablename__ = 'articles'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(255), nullable=False)

    def __unicode__(self):
        return self.content

    def url(self):
        return url_for('.articles', article_id=self.id)

# create the Flask-Restless API manager
manager = APIManager(app, flask_sqlalchemy_db=db)

# create our Article API at /api/articles
manager.create_api(Article, collection_name='articles', methods=['GET', 'POST', 'PUT', 'DELETE'])

db.create_all()

if __name__ == '__main__':
    # we define the debug environment only if running through command line
    app.config['SQLALCHEMY_ECHO'] = True
    app.debug = True
    app.run()
```

在前面的示例中，我们创建了我们的模型，然后创建了一个 Flask-Restless API 来保存所有我们的模型 API；然后我们为`Article`创建了一个带有前缀`articles`的 Web 服务 API，并支持`GET`、`POST`、`PUT`和`DELETE`方法，每个方法都有预期的行为：`GET`用于查询，`POST`用于新记录，`PUT`用于更新，`DELETE`用于删除。

在控制台中，输入以下命令发送 GET 请求到 API，并测试您的示例是否正常工作：

```py
curl http://127.0.0.1:5000/api/articles

```

由于 Flask-Restless API 非常广泛，我们将简要讨论一些对大多数项目非常有用的常见选项。

`create_api`的`serializer`/`deserializer`参数在您需要为模型进行自定义序列化/反序列化时非常有用。使用方法很简单：

```py
manager.create_api(Model, methods=METHODS,
                   serializer=my_serializer,
                   deserializer=my_deserializer)
def my_serializer(instance):
    return some_schema.dump(instance).data

def my_deserializer(data):
    return some_schema.load(data).data
```

您可以使用 marshmallow 生成模式，就像前面的示例一样。

`create_api`的另一个有用的选项是`include_columns`和`exclude_columns`。它们允许您控制 API 返回多少数据，并防止返回敏感数据。当设置`include_columns`时，只有其中定义的字段才会被 GET 请求返回。当设置`exclude_columns`时，只有其中未定义的字段才会被 GET 请求返回。例如：

```py
# both the statements below are equivalents
manager.create_api(Article, methods=['GET'], include_columns=['id', 'title'])
manager.create_api(Article, methods=['GET'], exclude_columns=['content'])
```

# 总结

在本章中，我们学习了 REST 是什么，它的优势，如何创建 Flask RESTful Web 服务和 API，以及如何使用 Flask-Restless 使整个过程顺利运行。我们还概述了 jQuery 是什么，以及如何使用它发送 Ajax 请求来查询我们的服务。这些章节示例非常深入。尝试自己编写示例代码，以更好地吸收它们。

在下一章中，我们将讨论确保软件质量的一种方式：测试！我们将学习如何以各种方式测试我们的 Web 应用程序，以及如何将这些测试集成到我们的编码例程中。到时见！
