# 第六章：跟踪用户动作

在上一章中，你用 jQuery 实现了 AJAX 视图，并构建了一个分享其它网站内容的 JavaScript 书签工具。

本章中，你将学习如何构建关注系统和用户活动流。你会了解 Django 的信号（signals）如何工作，并在项目中集成 Redis 快速 I/O 存储，用于存储项视图。

本章将会覆盖以下知识点：

- 用中介模型创建多对多关系
- 构建 AJAX 视图
- 创建活动流应用
- 为模型添加通用关系
- 优化关联对象的`QuerySet`
- 使用信号进行反规范化计数
- 在 Redis 中存储项的浏览次数

## 6.1 构建关注系统

我们将在项目中构建关注系统。用户可以相互关注，并跟踪其他用户在平台分享的内容。用户之间是多对多的关系，一个用户可以关注多个用户，也可以被多个用户关注。

### 6.1.1 用中介模型创建多对多关系

在上一章中，通过在一个关联模型中添加`ManyToManyField`，我们创建了多对多的关系，并让 Django 为这种关系创建了一张数据库表。这种方式适用于大部分情况，但有时候你需要为这种关系创建一个中介模型。当你希望存储这种关系的额外信息（比如关系创建的时间，或者描述关系类型的字段）时，你需要创建中介模型。

我们将创建一个中介模型用于构建用户之间的关系。我们使用中介模型有两个原因：

- 我们使用的是 Django 提供的`User`模型，不想修改它。
- 我们想要存储关系创建的时间。

编辑`account`应用的`models.py`文件，添加以下代码：

```py
from django.contrib.auth.models import User

class Contact(models.Model):
    user_from = models.ForeignKey(User, related_name='rel_from_set')
    user_to = models.ForeignKey(User, related_name='rel_to_set')
    created = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ('-created', )

    def __str__(self):
        return '{} follows {}'.format(self.user_from, self.user_to)
```

我们将把`Contact`模型用于用户关系。它包括以下字段：

- `user_from`：指向创建关系的用户的`ForeignKey`
- `user_to`：指向被关注用户的`ForeignKey`
- `created`：带`auto_new_add=True`的`DateTimeField`字段，存储创建关系的时间

数据库会自动在`ForeignKey`字段上创建索引。我们在`created`字段上用`db_index=True`创建数据库索引。当用这个字段排序`QuerySet`时，可以提高查询效率。

通过 ORM，我们可以创建用户`user1`关注用户`user2`的关系，如下所示：

```py
user1 = User.objects.get(id=1)
user2 = User.objects.get(id=2)
Contact.objects.create(user_from=user1, user_to=user2)
```

关联管理器`rel_from_set`和`rel_to_set`会返回`Contact`模型的`QuerySet`。为了从`User`模型访问关系的另一端，我们希望`User`模型包括一个`ManyToManyField`，如下所示：

```py
following = models.ManyToManyField(
	'self',
	through=Contact,
	related_name='followers',
	symmetrical=False)
```

这个例子中，通过在`ManyToManyField`字段中添加`through=Contact`，我们告诉 Django 使用自定义的中介模型。这是从`User`模型到它自身的多对多关系：我们在`ManyToManyField`字段中引用`'self'`来创建到同一个模型的关系。

> 当你在多对多的关系中需要额外字段时，可以在关系两端创建带`ForeignKey`的自定义模型。在其中一个关联模型中添加`ForeignKey`，并通过`through`参数指向中介模型，让 Django 使用该中介模型。

如果`User`模型属于我们的应用，我们就可以把上面这个字段添加到模型中。但是我们不能直接修改它，因为它属于`django.contrib.auth`应用。我们将采用略微不同的方法：动态的添加该字段到`User`模型中。编辑`account`应用的`models.py`文件，添加以下代码：

```py
User.add_to_class('following', 
    models.ManyToManyField('self', 
    	through=Contact, 
    	related_name='followers', 
    	symmetrical=False))
```

在这段代码中，我们使用 Django 模型的`add_to_class()`方法添加`monkey-patch`到`User`模型中。不推荐使用`add_to_class()`为模型添加字段。但是，我们在这里使用这种方法有以下几个原因：

- 通过 Django ORM 的`user.followers.all()`和`user.following.all()`，可以简化检索关联对象。我们使用`Contact`中介模型，避免涉及数据库连接（join）的复杂查询。如果我们在`Profile`模型中定义关系，则需要使用复杂查询。
- 这个多对多关系的数据库表会使用`Contact`模型创建。因此，动态添加的`ManyToManyField`不会对 Django 的`User`模型数据库做任何修改。
- 我们避免创建自定义的用户模型，充分利用 Django 内置的`User`模型。

记住，在大部分情况下都推荐使用添加字段到我们之前创建的`Profile`模型，而不是添加`monkey-patch`到`User`模型。Django 也允许你使用自定义的用户模型。如果你想使用自定义的用户模型，请参考[文档](https://docs.djangoproject.com/en/1.11/topics/auth/customizing/#specifying-a-custom-user-model)。

你可以看到，关系中包括`symmetrical=False`。当你定义`ManyToManyField`到模型自身时，Django 强制关系是对称的。在这里，我们设置`symmetrical=False`定义一个非对称关系。也就是说，如果我关注了你，你不会自动关注我。

> 当使用中介模型定义多对多关系时，一些关系管理器的方法将不可用，比如`add()`，`create()`，`remove()`。你需要创建或删除中介模型来代替。

执行以下命令为`account`应用生成初始数据库迁移：

```py
python manage.py makemigrations account
```

你会看到以下输出：

```py
Migrations for 'account':
  account/migrations/0002_contact.py
    - Create model Contact
```

现在执行以下命令同步数据库和应用：

```py
python manage.py migrate account
```

你会看到包括下面这一行的输出：

```py
Applying account.0002_contact... OK
```

现在`Contact`模型已经同步到数据库中，我们可以在用户之间创建关系了。但是我们的网站还不能浏览用户，或者查看某个用户的个人资料。让我们为`User`模型创建列表和详情视图。

### 6.1.2 为用户资料创建列表和详情视图

打开`account`应用的`views.py`文件，添加以下代码：

```py
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User

@login_required
def user_list(request):
    users = User.objects.filter(is_active=True)
    return render(request, 'account/user/list.html', {'section': 'people', 'users': users})

@login_required
def user_detail(request, username):
    user = get_object_or_404(User, username=username, is_active=True)
    return render(request, 'account/user/detail.html', {'section': 'people', 'user': user})
```

这是`User`对象简单的列表和详情视图。`user_list`视图获得所有激活的用户。Django 的`User`模型包括一个`is_active`标记，表示用户账户是否激活。我们通过`is_active=True`过滤查询，只返回激活的用户。这个视图返回了所有结果，你可以跟`image_list`视图那样，为它添加分页。

`user_detail`视图使用`get_object_or_404()`快捷方法，检索指定用户名的激活用户。如果没有找到指定用户名的激活用户，该视图返回 HTTP 404 响应。

编辑`account`应用的`urls.py`文件，为每个视图添加 URL 模式，如下所示：

```py
urlpatterns= [
	# ...
    url(r'^users/$', views.user_list, name='user_list'),
    url(r'^users/(?P<username>[-\w]+)/$', views.user_detail, name='user_detail'),
]
```

我们将使用`user_detail` URL 模式为用户生成标准 URL。你已经在模型中定义过`get_absolute_url()`方法，为每个对象返回标准 URL。另一种方式是在项目中添加`ABSOLUTE_URL_OVERRIDES`设置。

编辑项目的`settings.py`文件，添加以下代码：

```py
ABSOLUTE_URL_OVERRIDES = {
    'auth.user': lambda u: reverse_lazy('user_detail', args=[u.username])
}
```

Django 为`ABSOLUTE_URL_OVERRIDES`设置中的所有模型动态添加`get_absolute_url()`方法。这个方法返回给定模型的对应 URL。我们为给定用户返回`user_detail` URL。现在你可以在`User`实例上调用`get_absolute_url()`方法获得相应的 URL。用`python manage.py shell`打开 Python 终端，执行以下命令测试：

```py
>>> from django.contrib.auth.models import User
>>> user = User.objects.latest('id')
>>> str(user.get_absolute_url())
'/account/users/Antonio/'
```

返回的结果是期望的 URL。我们需要为刚创建的视图创建模板。在`account`应用的`templates/account/`目录中添加以下目录和文件：

```py
user/
	detail.html
	list.html
```

编辑`account/user/list.html`模板，添加以下代码：

```py
{% extends "base.html" %}
{% load thumbnail %}

{% block title %}People{% endblock %}

{% block content %}
    <h1>People</h1>
    <div id="people-list">
        {% for user in users %}
            <div class="user">
                <a href="{{ user.get_absolute_url }}">
                    {% thumbnail user.profile.photo "180x180" crop="100%" as im %}
                        <img src="{{ im.url }}">
                    {% endthumbnail %}
                </a>
                <div class="info">
                    <a href="{{ user.get_absolute_url }}" class="title">
                        {{ user.get_full_name }}
                    </a>
                </div>
            </div>
        {% endfor %}
    </div>
{% endblock %}
```

该模板列出网站中所有激活的用户。我们迭代给定的用户，使用`sorl-thumbnail`的`{% thumbnail %}`模板标签生成个人资料的图片缩略图。

打开项目的`base.html`文件，在以下菜单项的`href`属性中包括`user_list` URL：

```py
<li {% if section == "people" %}class="selected"{% endif %}>
	<a href="{% url "user_list" %}">People</a>
</li>
```

执行`python manage.py runserver`命令启动开发服务器，然后在浏览器中打开`http://127.0.0.1/8000/account/users/`。你会看到用户列表，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.1.png)

编辑`account`应用的`account/user/detail.html`模板，添加以下代码：

```py
{% extends "base.html" %}
{% load thumbnail %}

{% block title %}{{ user.get_full_name }}{% endblock %}

{% block content %}
    <h1>{{ user.get_full_name }}</h1>
    <div class="profile-info">
        {% thumbnail user.profile.photo "180x180" crop="100%" as im %}
            <img src="{{ im.url }}" class="user-detail">
        {% endthumbnail %}
    </div>
    {% with total_followers=user.followers.count %}
        <span class="count">
            <span class="total">{{ total_followers }}</span>
            follower{{ total_followers|pluralize }}
        </span>
        <a href="#" data-id="{{ user.id }}" 
            data-action="{% if request.user in user.followers.all %}un{% endif %}follow" 
            class="follow button">
            {% if request.user not in user.followers.all %}
                Follow
            {% else %}
                Unfollow
            {% endif %}
        </a>
        <div id="image-list" class="image-container">
            {% include "images/image/list_ajax.html" with images=user.images_created.all %}
        </div>
    {% endwith %}
{% endblock %}
```

我们在详情模板中显示用户个人资料，并使用`{% thumbnail %}`模板标签显示个人资料图片。我们显示关注者总数和一个用于`follow/unfollow`的链接。如果用户正在查看自己的个人资料，我们会隐藏该链接，防止用户关注自己。我们将执行 AJAX 请求来`follow/unfollow`指定用户。我们在`<a>`元素中添加`data-id`和`data-action`属性，其中分别包括用户 ID 和点击链接时执行的操作（关注或取消关注），这取决于请求该页面的用户是否已经关注了这个用户。我们用`list_ajax.html`模板显示这个用户标记过的图片。

再次打开浏览器，点击标记过一些图片的用户。你会看到个人资料详情，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.2.png)

### 6.1.3 构建关注用户的 AJAX 视图

我们将使用 AJAX 创建一个简单视图，用于关注或取消关注用户。编辑`account`用于的`views.py`文件，添加以下代码：

```py
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from common.decrorators import ajax_required
from .models import Contact

@ajax_required
@require_POST
@login_required
def user_follow(request):
    user_id = request.POST.get('id')
    action = request.POST.get('action')
    if user_id and action:
        try:
            user = User.objects.get(id=user_id)
            if action == 'follow':
                Contact.objects.get_or_create(user_from=request.user, user_to=user)
            else:
                Contact.objects.filter(user_from=request.user, user_to=user).delete()
            return JsonResponse({'status': 'ok'})
        except User.DoesNotExist:
            return JsonResponse({'status': 'ko'})
    
    return JsonResponse({'status': 'ko'})
```

`user_follow`视图跟我们之前创建的`image_like`视图很像。因为我们为用户的多对多关系使用了自定义的中介模型，所以`ManyToManyField`自动生成的管理器的默认`add()`和`remove()`方法不可用了。我们使用`Contact`中介模型创建或删除用户关系。

在`account`应用的`urls.py`文件中导入你刚创建的视图，然后添加以下 URL 模式：

```py
url(r'^users/follow/$', views.user_follow, name='user_follow'),
```

确保你把这个模式放在`user_detail`模式之前。否则任何到`/users/follow/`的请求都会匹配`user_detail`模式的正则表达式，然后执行`user_detail`视图。记住，每一个 HTTP 请求时，Django 会按每个模式出现的先后顺序匹配请求的 URL，并在第一次匹配成功后停止。

编辑`account`应用的`user/detail.html`模板，添加以下代码：

```py
{% block domready %}
    $('a.follow').click(function(e){
        e.preventDefault();
        $.post('{% url "user_follow" %}', {
            id: $(this).data('id'),
            action: $(this).data('action')
        },
        function(data){
            if (data['status'] == 'ok') {
                var previous_action = $('a.follow').data('action');

                // toggle data-action
                $('a.follow').data('action', previous_action == 'follow' ? 'unfollow' : 'follow');
                // toggle link text
                $('a.follow').text(previous_action == 'follow' ? 'Unfollow' : 'Follow');

                // update total followers
                var previous_followers = parseInt($('span.count .total').text())
                $('span.count .total').text(previous_action == 'follow' ? previous_followers+1 : previous_followers - 1);
            }
        });
    });
{% endblock %}
```

这段 JavaScript 代码执行关注或取消关注指定用户的 AJAX 请求，同时切换`follow/unfollow`链接。我们用 jQuery 执行 AJAX 请求，并根据之前的值设置`data-action`属性和`<a>`元素的文本。AJAX 操作执行完成后，我们更新页面显示的关注总数。打开一个已存在用户的详情页面，点击`FOLLOW`链接，测试我们刚添加的功能。

## 6.2 构建通用的活动流应用

很多社交网站都会给用户显示活动流，让用户可以跟踪其他用户在平台上做了什么。活动流是一个或一组用户最近执行的活动列表。比如，Facebook 的 News Feed 就是一个活动流。又或者用户 X 标记了图片 Y，或者用户 X 不再关注用户 Y。我们将构造一个活动流应用，让每个用户都可以看到他关注的用户最近的操作。要实现这个功能，我们需要一个模型，存储用户在网站中执行的操作，并提供简单的添加操作的方式。

用以下命令在项目中创建`actions`应用：

```py
django-admin startapp actions
```

在项目的`settings.py`文件的`INSTALLED_APPS`中添加`actions`，让 Django 知道新应用已经激活：

```py
INSTALLED_APPS = (
	# ...
	'actions',
)
```

编辑`actions`应用的`models.py`文件，添加以下代码：

```py
from django.db import models
from django.contrib.auth.models import User

class Action(models.Model):
    user = models.ForeignKey(User, related_name='actions', db_index=True)
    verb = models.CharField(max_length=255)
    created = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ('-created', )
```

这是`Actioin`模型，用于存储用户活动。该模型的字段有：

- `user`：执行这个操作的用户。这是一个指向 Django 的`User`模型的`ForeignKey`。
- `verb`：描述用户执行的操作。
- `created`：该操作创建的日期和时间。我们使用`auto_now_add=True`自动设置为对象第一次在数据库中保存的时间。

通过这个基础模型，我们只能存储类似用户 X 做了某些事情的操作。我们需要一个额外的`ForeignKey`字段，存储涉及目标对象的操作，比如用户 X 标记了图片 Y，或者用户 X 关注了用户 Y。你已经知道，一个普通的`ForeignKey`字段只能指向另一个模型。但是我们需要一种方式，让操作的目标对象可以是任何一个已经存在的模型的实例。这就是 Django 的`contenttypes`框架的作用。

### 6.2.1 使用 contenttypes 框架

Django 的`contenttypes`框架位于`django.contrib.contenttypes`中。这个应用可以跟踪项目中安装的所有模型，并提供一个通用的接口与模型交互。

当你使用`startproject`命令创建新项目时，`django.contrib.contenttypes`已经包括在`INSTALLED_APPS`设置中。它被其它`contrib`包（比如`authentication`框架和`admin`应用）使用。

`contenttypes`应用包括一个`ContentType`模型。这个模型的实例代表你的应用中的真实模型，当你的项目中安装了一个新模型时，会自动创建一个新的`ContentType`实例。`ContentType`模型包括以下字段：

- `app_label`：模型所属应用的名字。它会自动从模型`Meta`选项的`app_label`属性中获得。例如，我们的`Image`模型属于`images`应用。
- `model`：模型的类名。
- `name`：模型的人性化名字。它自动从模型`Meta`选项的`verbose_name`属性中获得。

让我们看下如何与`ContentType`对象交互。使用`python manage.py shell`命令打开 Python 终端。通过执行带`label_name`和`model`属性的查询，你可以获得指定模型对应的`ContentType`对象，比如：

```py
>>> from django.contrib.contenttypes.models import ContentType
>>> image_type = ContentType.objects.get(app_label='images',model='image')
>>> image_type
<ContentType: image>
```

你也可以通过调用`ContentType`对象的`model_class()`方法，反向查询模型类：

```py
>>> image_type.model_class()
<class 'images.models.Image'>
```

从指定的模型类获得`ContentType`对象操作也很常见：

```py
>>> from images.models import Image
>>> ContentType.objects.get_for_model(Image)
<ContentType: image>
```

这些只是使用`contenttypes`的几个示例。Django 提供了更多使用它们的方式。你可以在[官方文档](https://docs.djangoproject.com/en/1.11/ref/contrib/contenttypes/)学习`contenttypes`框架。

### 6.2.2 在模型中添加通用关系

在通用关系中，`ContentType`对象指向关系中使用的模型。在模型中设置通用关系，你需要三个字段：

- 一个`ForeignKey`字段指向`ContentType`。这会告诉我们关系中的模型。
- 一个存储关联对象主键的字段。通常这是一个`PositiveIntegerField`，来匹配 Django 自动生成的主键字段。
- 一个使用上面两个字段定义和管理通用关系的字段。`contenttypes`框架为此定义了`GenericForeignKey`字段。

编辑`actions`应用的`models.py`文件，如下所示：

```py
from django.db import models
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey

class Action(models.Model):
    user = models.ForeignKey(User, related_name='actions', db_index=True)
    verb = models.CharField(max_length=255)

    target_ct = models.ForeignKey(ContentType, blank=True, null=True, related_name='target_obj')
    target_id = models.PositiveIntegerField(null=True, blank=True, db_index=True)
    target = GenericForeignKey('target_ct', 'target_id')
    created = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ('-created', )
```

我们在`Action`模型中添加了以下字段：

- `target_ct`：一个指向`ContentType`模型的`ForeignKey`字段。
- `target_id`：一个用于存储关联对象主键的`PositiveIntegerField`。
- `target`：一个指向由前两个字段组合的关联对象的`GenericForeignKey`字段。

Django 不会在数据库中为`GenericForeignKey`字段创建任何字段。只有`target_ct`和`target_id`字段会映射到数据库的字段。因为这两个字段都有`blank=True`和`null=True`属性，所以保存`Action`对象时`target`对象不是必需的。

> 使用通用关系有意义的时候，你可以使用它代替外键，让应用更灵活。

执行以下命令为这个应用创建初始的数据库迁移：

```py
python manage.py makemigrations actions
```

你会看到以下输出：

```py
Migrations for 'actions':
  actions/migrations/0001_initial.py
    - Create model Action
```

接着执行以下命令同步应用和数据库：

```py
python manage.py migrate
```

这个命令的输入表示新的数据库迁移已经生效：

```py
Applying actions.0001_initial... OK
```

当我们把`Action`模型添加到管理站点。编辑`actions`应用的`admin.py`文件，添加以下代码：

```py
from django.contrib import admin
from .models import Action

class ActionAdmin(admin.ModelAdmin):
    list_display = ('user', 'verb', 'target', 'created')
    list_filter = ('created', )
    search_fields = ('verb', )

admin.site.register(Action, ActionAdmin)
```

你刚刚在管理站点注册了`Action`模型。执行`python manage.py runserver`命令启动开服务器，然后在浏览器中打开`http://127.0.0.1:8000/actions/action/add/`。你会看到创建一个新的`Action`对象的页面，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.3.png)

正如你所看到的，只有`target_ct`和`target_id`字段映射到实际的数据库字段，而`GenericForeignKey`没有在这里出现。`target_ct`允许你选择在 Django 项目中注册的任何模型。使用`target_ct`字段的`limit_choices_to`属性，可以让`contenttypes`从一个限制的模型集合中选择：`limit_choices_to`属性允许你限制`ForeignKey`字段的内容为一组指定的值。

在`actions`应用目录中创建一个`utils.py`文件。我们将定义一些快捷方法，快速创建`Action`对象。编辑这个新文件，添加以下代码：

```py
from django.contrib.contenttypes.models import ContentType
from .models import Action

def create_action(user, verb, target=None):
    action = Action(user=user, verb=verb, target=target)
    action.save()
```

`create_action()`方法允许我们创建`Action`对象，其中包括一个可选的`target`对象。我们可以在任何地方使用这个函数添加新操作到活动流中。

### 6.2.3 避免活动流中的重复操作

有时候用户可能执行一个操作多次。他们可能在很短的时间内多次点击`like/unlike`按钮，或者执行同一个操作多次。最终会让你存储和显示重复操作。为了避免这种情况，我们会完善`create_acion()`函数，避免大部分重复操作。

编辑`actions`应用的`utils.py`文件，如下所示：

```py
import datetime
from django.utils import timezone
from django.contrib.contenttypes.models import ContentType
from .models import Action

def create_action(user, verb, target=None):
    # check for any similar action made in the last minute
    now = timezone.now()
    last_minute = now - datetime.timedelta(seconds=60)
    similar_actions = Action.objects.filter(user_id=user.id, verb=verb, created__gte=last_minute)
    if target:
        target_ct = ContentType.objects.get_for_model(target)
        similar_actions = similar_actions.filter(target_ct=target_ct, targt_id=target.id)

    if not similar_actions:
        # no existing actions found
        action = Action(user=user, verb=verb, target=target)
        action.save()
        return True
    return False
```

我们修改了`create_action()`函数，避免保存重复操作，并返回一个布尔值，表示操作是否保存。我们是这样避免重复的：

- 首先使用 Django 提供的`timezone.now()`方法获得当前时间。这个函数的作用与`datetime.datetime.now()`相同，但它返回一个`timezone-aware`对象。Django 提供了一个`USE_TZ`设置，用于启用或禁止时区支持。使用`startproject`命令创建的默认`settings.py`文件中，包括`USE_TZ=True`。
- 我们使用`last_minute`变量存储一分钟之前的时间，然后我们检索用户从那之后执行的所有相同操作。
- 如果最后一分钟没有相同的操作，则创建一个`Action`对象。如果创建了`Action`对象，则返回`True`，否则返回`False`。

### 6.2.4 添加用户操作到活动流中

是时候为用户添加一些操作到视图中，来创建活动流了。我们将为以下几种交互存储操作：

- 用户标记图片
- 用户喜欢或不喜欢一张图片
- 用户创建账户
- 用户关注或取消关注其它用户

编辑`images`应用的`views.py`文件，添加以下导入：

```py
from actions.utils import create_action
```

在`image_create`视图中，在保存图片之后添加`create_action()`：

```py
new_item.save()
create_action(request.user, 'bookmarked image', new_item)
```

在`image_like`视图中，在添加用户到`users_like`关系之后添加`create_action()`：

```py
image.users_like.add(request.user)
create_action(request.user, 'likes', image)
```

现在编辑`account`应用的`views.py`文件，添加以下导入：

```py
from actions.utils import create_action
```

在`register`视图中，在创建`Profile`对象之后添加`create_action()`：

```py
new_user.save()
profile = Profile.objects.create(user=new_user)
create_action(new_user, 'has created an account')
```

在`user_follow`视图中，添加`create_action()`：

```py
Contact.objects.get_or_create(user_from=request.user, user_to=user)
create_action(request.user, 'is following', user)
```

正如你所看到的，多亏了`Action`模型和帮助函数，让我们很容易的在活动流中保存新操作。

### 6.2.5 显示活动流

最后，我们需要为每个用户显示活动流。我们将把它包括在用户的仪表盘中。编辑`account`应用的`views.py`文件。导入`Action`模型，并修改`dashboard`视图：

```py
from actions.models import Action

@login_required
def dashboard(request):
    # Display all actions by default
    actions = Action.objects.exclude(user=request.user)
    following_ids = request.user.following.values_list('id', flat=True)
    if following_ids:
        # If user is following others, retrieve only their actions
        actions = actions.filter(user_id__in=following_ids)
    actions = actions[:10]

    return render(request,
                  'account/dashboard.html',
                  {'section': 'dashboard', 'actions': actions})
```

在这个视图中，我们从数据库中检索当前用户之外的所有用户执行的操作。如果用户还没有关注任何人，我们显示其它用户最近的操作。这是用户没有关注其他用户时的默认行为。如果用户关注了其他用户，我们限制查询只显示他关注的用户执行的操作。最后，我们限制只返回前 10 个操作。在这里没有使用`order_by()`进行排序，因为我们使用`Action`模型的`Meta`选项提供的默认排序。因为我们在`Action`模型中设置了`ordering = ('-created',)`，所以会先返回最新的操作。

### 6.2.6 优化涉及关联对象的 QuerySet

每次检索一个`Action`对象时，你可能需要访问与它关联的`User`对象，以及该用户关联的`Profile`对象。Django ORM 提供了一种方式，可以一次检索关联对象，避免额外的数据库查询。

#### 6.2.6.1 使用 select_related

Django 提供了一个`select_related`方法，允许你检索一对多关系的关联对象。它会转换为单个更复杂的`QuerySet`，但是访问关联对象时，可以避免额外的查询。`select_related`方法用于`ForeignKey`和`OneToOne`字段。它在`SELECT`语句中执行`SQL JOIN`，并且包括了关联对象的字段。

要使用`select_related()`，需要编辑之前代码的这一行：

```py
actions = actions.filter(user_id__in=following_ids)
```

并在你会使用的字段上添加`select_related`：

```py
actions = actions.filter(user_id__in=following_ids)\
	.select_related('user', 'user__profile')
```

我们用`user__profile`在单条 SQL 查询中连接了`Profile`表。如果调用`select_related()`时没有传递参数，那么它会从所有`ForeignKey`关系中检索对象。总是将之后会访问的关系限制为`select_related()`。

> 仔细使用`select_related()`可以大大减少执行时间。

#### 6.2.6.2 使用 prefetch_related

正如你所看到的，在一对多关系中检索关联对象时，`select_related()`会提高执行效率。但是`select_related()`不能用于多对多或者多对一关系。Django 提供了一个名为`prefetch_related`的`QuerySet`方法，除了`select_related()`支持的关系之外，还可以用于多对多和多对一关系。`prefetch_related()`方法为每个关系执行独立的查询，然后用 Python 连接结果。该方法还支持`GenericRelation`和`GenericForeignKey`的预读取。

为`GenericForeignKey`字段`target`添加`prefetch_related()`，完成这个查询：

```py
actions = actions.filter(user_id__in=following_ids)\
	.select_related('user', 'user__profile')\
	.prefetch_related('target')
```

现在查询已经优化，用于检索包括关联对象的用户操作。

### 6.2.7 为操作创建模板

我们将创建模板用于显示特定的`Action`对象。在`actions`应用目录下创建`templates`目录，并添加以下文件结构：

```py
actions/
	action/
		detail.html
```

编辑`actions/action/detail.html`目录文件，并添加以下代码：

```py
{% load thumbnail %}

{% with user=action.user profile=action.user.profile %}
    <div class="action">
        <div class="images">
            {% if profile.photo %}
                {% thumbnail user.profile.photo "80x80" crop="100%" as im %}
                    <a href="{{ user.get_absolute_url }}">
                        <img src="{{ im.url }}" alt="{{ user.get_full_nam }}" class="item-img">
                    </a>
                {% endthumbnail %}
            {% endif %}

            {% if action.target %}
                {% with target=action.target %}
                    {% if target.image %}
                        {% thumbnail target.image "80x80" crop="100%" as im %}
                            <a href="{{ target.get_absolute_url }}">
                                <img src="{{ im.url }}" class="item-img">
                            </a>
                        {% endthumbnail %}
                    {% endif %}
                {% endwith %}
            {% endif %}
        </div>
        <div class="info">
            <p>
                <span class="date">{{ action.created|timesince }} age</span>
                <br />
                <a href="{{ user.get_absolute_url }}">
                    {{ user.first_name }}
                </a>
                {{ action.verb }}
                {% if action.target %}
                    {% with target=action.target %}
                        <a href="{{ target.get_absolute_url }}">{{ target }}</a>
                    {% endwith %}
                {% endif %}
            </p>
        </div>
    </div>
{% endwith %}
```

这是显示一个`Action`对象的模板。首先，我们使用`{% with %}`模板标签检索执行操作的用户和他们的个人资料。接着，如果`Action`对象有关联的`target`对象，则显示`target`对象的图片。最后，我们显示执行操作的用户链接，描述，以及`target`对象（如果有的话）。

现在编辑`account/dashboard.html`模板，在`content`块底部添加以下代码：

```py
<h2>What's happening</h2>
<div id="action-list">
	{% for action in actions %}
		{% include "actions/action/detail.html" %}
	{% endfor %}
</div>
```

在浏览器中打开`http://127.0.0.1:8000/account/`。用已存在的用户登录，并执行一些操作存储在数据库中。接着用另一个用户登录，并关注之前那个用户，然后在仪表盘页面查看生成的活动流，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.4.png)

我们为用户创建了一个完整的活动流，并且能很容易的添加新的用户操作。你还可以通过 AJAX 分页，在活动流中添加无限滚动，就像我们在`image_list`视图中那样。

## 6.3 使用信号进行反规范化计数

某些情况下你希望对数据进行反规范化处理。反规范化（denormalization）是在一定程度上制造一些冗余数据，从而优化读取性能。你必须小心使用反规范化，只有当你真的需要的时候才使用。反规范化最大的问题是很难保持数据的更新。

我们将通过一个例子解释如何通过反规范化计数来改善查询。缺点是我们必须保持冗余数据的更新。我们将在`Image`模型中使用反规范数据，并使用 Django 的信号来保持数据的更新。

### 6.3.1 使用信号

Django 自带一个信号调度程序，当特定动作发生时，允许接收函数获取通知。当某些事情发生时，你的代码需要完成某些工作，信号非常有用。你也可以创建自己的信号，当事件发生时，其他人可以获得通知。

Django 在`django.db.models.signals`中为模型提供了几种信号，其中包括：

- `pre_save`和`post_save`：调用模型的`save()`方法之前或之后发送
- `pre_delete`和`post_delete`：调用模型或`QuerySet`的`delete()`方法之前或之后发送
- `m2m_changed`：当模型的`ManyToManyField`改变时发送

这只是 Django 提供了部分信号。你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/signals/)查看 Django 的所有内置信号。

我们假设你想获取热门图片。你可以使用 Django 聚合函数，按用户喜欢数量进行排序。记住你已经在第三章中使用了聚合函数。以下代码按喜欢数量查询图片：

```py
from django.db.models import Count
from images.models import Image

images_by_popularity = Image.objects.annotate(total_likes=Count('users_like')).order_by('-total_likes')
```

但是，通过统计图片的喜欢数量比直接使用一个存储喜欢数量的字段更费时。你可以在`Image`模型中添加一个字段，用来反规范化喜欢数量，从而提高涉及这个字段的查询性能。如何保持这个字段的更新呢？

编辑`images`应用的`models.py`文件，为`Image`模型添加以下字段：

```py
total_likes = models.PositiveIntegerField(db_index=True, default=0)
```

`total_likes`字段允许我们存储每张图片被用户喜欢的数量。当你希望过滤或者排序`QuerySet`时，反规范计数非常有用。

> 在使用反规范字段之前，你必须考虑其它提升性能的方式。比如数据库索引，查询优化和缓存。

执行以下命令为新添加的字段创建数据库迁移：

```py
python manage.py makemigrations images
```

你会看到以下输出：

```py
Migrations for 'images':
  images/migrations/0002_image_total_likes.py
    - Add field total_likes to image
```

接着执行以下命令让迁移生效：

```py
python manage.py migrate images
```

输出中会包括这一行：

```py
Applying images.0002_image_total_likes... OK
```

我们将会为`m2m_changed`信号附加一个`receiver`函数。在`images`应用目录下创建一个`signals.py`文件，添加以下代码：

```py
from django.db.models.signals import m2m_changed
from django.dispatch import receiver
from .models import Image

@receiver(m2m_changed, sender=Image.users_like.through)
def users_like_changed(sender, instance, **kwargs):
    instance.total_likes = instance.users_like.count()
    instance.save()
```

首先，我们使用`receiver()`装饰器注册`users_like_changed`函数为`receiver()`函数，并把它附加给`m2m_changed`信号。我们把函数连接到`Image.users_like.throuth`，只有这个发送者发起`m2m_changed`信号时，这个方法才会被调用。还可以使用`Signal`对象的`connect()`方法来注册`receiver()`函数。

> Django 信号是同步和阻塞的。不要用异步任务导致信号混乱。但是，当你的代码从信号中获得通知时，你可以组合两者来启动异步任务。

你必须把接收器函数连接到一个信号，这样每次发送信号时，接收器函数才会调用。注册信号的推荐方式是在应用配置类的`ready()`函数中导入它们。Django 提供了一个应用注册表，用于配置和内省应用。

### 6.3.2 定义应用配置类

Django 允许你为应用指定配置类。要为应用提供一个自定义配置，你需要创建一个自定义类，它继承自位于`django.apps`中的`AppConfig`类。应用配置类允许为应用存储元数据和配置，并提供内省。

你可以在[这里](https://docs.djangoproject.com/en/1.11/ref/applications/)阅读更多关于应用配置的信息。

为了注册你的信号接收函数，当你使用`receiver()`装饰器时，你只需要在`AppConfig`类的`ready()`方法中导入应用的信号模块。一旦应用注册表完全填充，就会调用这个方法。这个方法中应该包括应用的所有初始化工作。

在`images`应用目录下创建`apps.py`文件，并添加以下代码：

```py
from django.apps import AppConfig


class ImagesConfig(AppConfig):
    name = 'images'
    verbose_name = 'Image bookmarks'

    def ready(self):
        # import signal handlers
        import images.signals
```

> **译者注：**Django 1.11 版本中，默认已经生成了`apps.py`文件，只需要在其中添加`ready()`方法。

其中，`name`属性定义应用的完整 Python 路径；`verbose_name`属性设置应用的可读名字。它会在管理站点中显示。我们在`ready()`方法中导入该应用的信号。

现在我们需要告诉 Django 应用配置的位置。编辑`images`应用目录的`__init__.py`文件，添加这一行代码：

```py
default_app_config = 'images.apps.ImagesConfig'
```

在浏览器中查看图片详情页面，并点击`like`按钮。然后回到管理站点查看`total_likes`属性。你会看到`total_likes`已经更新，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.5.png)

现在你可以使用`total_likes`属性按热门排序图片，或者在任何地方显示它，避免了用复杂的查询来计算。以下按图片被喜欢的总数量排序的查询：

```py
images_by_popularity = Image.objects.annotate(likes=Count('users_like')).order_by('-likes')
```

可以变为这样：

```py
images_by_popularity = Image.objects.order_by('-total_likes')
```

通过更快的 SQL 查询就返回了这个结果。这只是使用 Django 信号的一个示例。

> 小心使用信号，因为它会让控制流更难理解。如果你知道会通知哪个接收器，很多情况下就能避免使用信号。

你需要设置初始计数，来匹配数据库的当前状态。使用`python manage.py shell`命令打开终端，执行以下命令：

```py
from images.models import Image
for image in Image.objects.all():
	image.total_likes = image.users_like.count()
	image.save()
```

现在每张图片被喜欢的总数量已经更新了。

## 6.4 用 Redis 存储项视图

Redis 是一个高级的键值对数据库，允许你存储不同类型的数据，并且能进行非常快速的 I/O 操作。Redis 在内存中存储所有数据，但数据集可以一次性持久化到硬盘中，或者添加每条命令到日志中。与其它键值对存储相比，Redis 更通用：它提供了一组功能强大的命令，并支持各种各样的数据结构，比如`strings`，`hashes`，`lists`，`sets`，`ordered sets`，甚至`bitmaps`或`HyperLogLogs`。

SQL 最适合于模式定义的持久化数据存储，而当处理快速变化的数据，短暂的存储，或者快速缓存时，Redis 有更多的优势。让我们看看如何使用 Redis 为我们的项目添加新功能。

### 6.4.1 安装 Redis

从[这里](https://redis.io/download)下载最新的 Redis 版本。解压`tar.gz`文件，进入`redis`目录，使用`make`命令编译 Redis：

```py
cd redis-3.2.8
make
```

安装完成后，使用以下命令初始化 Redis 服务器：

```py
src/redis-server
```

你会看到结尾的输出为：

```py
19608:M 08 May 17:04:38.217 # Server started, Redis version 3.2.8
19608:M 08 May 17:04:38.217 * The server is now ready to accept connections on port 6379
```

默认情况下，Redis 在 6379 端口运行，但你可以使用`--port`之指定自定义端口，比如：`redis-server --port 6655`。服务器就绪后，使用以下命令在另一个终端打开 Redis 客户端：

```py
src/redis-cli
```

你会看到 Redis 客户端终端：

```py
127.0.0.1:6379>
```

你可以直接在 Redis 客户端执行 Redis 命令。让我们尝试一下。在 Redis 终端输入`SET`命令，在键中存储一个值：

```py
127.0.0.1:6379> SET name "Peter"
OK
```

以上命令在 Redis 数据库中创建了一个字符串值为`Peter`的`name`键。输出`OK`表示键已经成功保存。接收，使用`GET`命令查询值：

```py
127.0.0.1:6379> GET name
"Peter"
```

我们也可以使用`EXISTS`命令检查一个叫键是否存在。如果存在返回`1`，否则返回`0`：

```py
127.0.0.1:6379> EXISTS name
(integer) 1
```

你可以使用`EXPIRE`命令为键设置过期时间，这个命令允许你设置键的存活秒数。另一个选项是使用`EXPIREAT`命令，它接收一个 Unix 时间戳。把 Redis 作为缓存，或者存储临时数据时，键过期非常有用：

```py
127.0.0.1:6379> EXPIRE name 2
(integer) 1
```

等待 2 秒，再次获取同样的键：

```py
127.0.0.1:6379> GET name
(nil)
```

返回值`(nil)`是一个空返回，表示没有找到键。你也可以使用`DEL`命令删除键：

```py
127.0.0.1:6379> SET total 1
OK
127.0.0.1:6379> DEL total
(integer) 1
127.0.0.1:6379> GET total
(nil)
```

这只是键操作的基本命令。Redis 为每种数据类型（比如`strings`，`hashes`，`lists`，`sets`，`ordered sets`等等）提供了大量的命令。你可以在[这里](https://redis.io/commands)查看所有 Redis 命令，在[这里](https://redis.io/topics/data-types)查看所有 Redis 数据类型。

### 6.4.2 在 Python 中使用 Redis

我们需要为 Redis 绑定 Python。通过`pip`安装`redis-py`：

```py
pip install redis
```

你可以在[这里](http://redis-py.readthedocs.org/)查看`redis-py`的文档。

`redis-py`提供了两个类用于与 Redis 交互：`StricRedis`和`Redis`。两个类提供了相同的功能。`StricRedis`类视图遵守官方 Redis 命令语法。`Redis`类继承自`StricRedis`，覆写了一些方法，提供向后的兼容性。我们将使用`StrictRedis`类，因为它遵循 Redis 命令语法。打开 Python 终端，执行以下命令：

```py
>>> import redis
>>> r = redis.StrictRedis(host='localhost', port=6379, db=0)
```

这段代码创建了一个 Redis 数据连接。在 Redis 中，数据由整数索引区分，而不是数据库名。默认情况下，客户端连接到数据库 0。Redis 数据库有效的数字到 16，但你可以在`redis.conf`文件中修改这个值。

现在使用 Python 终端设置一个键：

```py
>>> r.set('foo', 'bar')
True
```

命令返回`True`表示键创建成功。现在你可以使用`get()`命令查询键：

```py
>>> r.get('foo')
b'bar'
```

正如你锁看到的，`StrictRedis`方法遵循 Redis 命令语法。

让我们在项目中集成 Redis。编辑`bookmarks`项目的`settings.py`文件，添加以下设置：

```py
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
```

以上设置了 Redis 服务器和我们在项目中使用的数据库。

### 6.4.3 在 Redis 中存储项的浏览次数

让我们存储一张图片被查看的总次数。如果我们使用 Django ORM，则每次显示图片后，都会涉及`UPDATE`语句。如果使用 Redis，我们只需要增加内存中的计数，从而获得更好的性能。

编辑`images`应用的`views.py`文件，添加以下代码：

```py
import redis
from django.conf import settings

# connect to redis
r = redis.StrictRedis(host=settings.REDIS_HOST,
                      port=settings.REDIS_PORT,
                      db=settings.REDIS_DB)
```

我们建立了 Redis 连接，以便在视图中使用。修改`image_detail`视图，如下所示：

```py
def image_detail(request, id, slug):
    image = get_object_or_404(Image, id=id, slug=slug)
    # increament total image views by 1
    total_views = r.incr('image:{}:views'.format(image.id))
    return render(request, 
                  'images/image/detail.html', 
                  {'section': 'images', 'image': image, 'total_views': total_views})
```

在这个视图中，我们使用`INCR`命令把一个键的值加 1，如果键不存在，则在执行操作之前设置值为 0。`incr()`方法返回执行操作之后键的值，我们把它存在`total_views`变量中。我们用`object-type:id:field`（比如`image:33:id:views`）构建 Redis 键。

> Redis 键的命名惯例是使用冒号分割，来创建带命名空间的键。这样键名会很详细，并且相关的键共享部分相同的模式。

编辑`image/detail.html`模板，在`<span class="count">`元素之后添加以下代码：

```py
<span class="count">
	<span class="total">{{ total_views }}</span>
	view{{ total_views|pluralize }}
</span>
```

现在在浏览器中打开图片详情页面，加载多次。你会看到每次执行视图，显示的浏览总数都会加 1，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.6.png)

你已经成功的在项目集成了 Redis，来存储项的浏览次数。

### 6.4.4 在 Redis 中存储排名

让我们用 Redis 构建更多功能。我们将创建浏览次数最多的图片排名。我们将使用 Redis 的`sorted set`来构建排名。一个`sorted set`是一个不重复的字符串集合，每个成员关联一个分数。项通过它们的分数存储。

编辑`images`应用的`views.py`文件，修改`image_detail`视图，如下所示：

```py
def image_detail(request, id, slug):
    image = get_object_or_404(Image, id=id, slug=slug)
    # increament total image views by 1
    total_views = r.incr('image:{}:views'.format(image.id))
    # increament image ranking by 1
    r.zincrby('image_ranking', image.id, 1)
    return render(request, 
                  'images/image/detail.html', 
                  {'section': 'images', 'image': image, 'total_views': total_views})
```

我们用`zincrby()`命令在`sorted set`中存储图片浏览次数，其中键为`image_ranking`。我们存储图片 id，分数 1 会被加到`sorted set`中这个元素的总分上。这样就可以全局追踪所有图片的浏览次数，并且有一个按浏览次数排序的`sorted set`。

现在创建一个新视图，用于显示浏览次数最多的图片排名。在`views.py`文件中添加以下代码：

```py
@login_required
def image_ranking(request):
    # get image ranking dictinary
    image_ranking = r.zrange('image_ranking', 0, -1, desc=True)[:10]
    image_ranking_ids = [int(id) for id in image_ranking]
    # get most viewed images
    most_viewed = list(Image.objects.filter(id__in=image_ranking_ids))
    most_viewed.sort(key=lambda x: image_ranking_ids.index(x.id))
    return render(request, 'images/image/ranking.html', {'section': 'images', 'most_viewed': most_viewed})
```

这是`image_ranking`视图。我们用`zrange()`命令获得`sorted set`中的元素。这个命令通过最低和最高分指定自定义范围。通过 0 作为最低，-1 作为最高分，我们告诉 Redis 返回`sorted set`中的所有元素。我们还指定`desc=True`，按分数的降序排列返回元素。最后，我们用`[:10]`切片操作返回分数最高的前 10 个元素。我们构建了一个返回的图片 ID 列表，并作为整数列表存在`image_ranking_ids`变量中。我们迭代这些 ID 的`Image`对象，并使用`list()`函数强制执行查询。强制`QuerySet`执行很重要，因为之后我们要调用列表的`sort()`方法（此时我们需要一组对象，而不是一个`QuerySet`）。我们通过`Image`对象在图片排名中的索引进行排序。现在我们可以在模板中使用`most_viewed`列表显示浏览次数最多的前 10 张图片。

创建`image/ranking.html`模板文件，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}Images ranking{% endblock %}

{% block content %}
    <h1>Images ranking</h1>
    <ol>
        {% for image in most_viewed %}
            <li>
                <a href="{{ image.get_absolute_url }}">
                    {{ image.title }}
                </a>
            </li>
        {% endfor %}
    </ol>
{% endblock %}
```

这个模板非常简单，我们迭代`most_viewed`列表中的`Image`对象。

最后为新视图创建 URL 模式。编辑`images`应用的`urls.py`文件，添加以下模式：

```py
url(r'^/ranking/$', views.image_ranking, name='ranking')
```

在浏览器中打开`http://127.0.0.1:8000/images/ranking/`，你会看到图片排名，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE6.7.png)

### 6.4.5 Redis 的后续功能

Redis 不是 SQL 数据库的替代者，而是更适用于特定任务的快速的内存存储。当你真的需要时可以使用它。Redis 非常适合以下场景：

- 计数：正如你所看到的，使用 Redis 管理计算非常简单。你可以使用`incr()`和`incrby()`计数。
- 存储最近的项：你可以使用`lpush()`和`rpush()`在列表开头或结尾添加项。使用`lpop()`或`rpop()`移除并返回第一或最后一项。你可以使用`ltrim()`截断列表长度。
- 队列：除了`push`和`pop`命令，Redis 还提供了阻塞队列的命令。
- 缓存：使用`expire()`和`expireat()`允许你把 Redis 当做缓存。你还可以找到 Django 的第三方 Redis 缓存后台。
- 订阅/发布：Redis 还为订阅/取消订阅，以及发送消息给频道提供了命令。
- 排名和排行榜：Redis 的`sorted set`可以很容易创建排行榜。
- 实时跟踪：Redis 的快速 I/O 非常适合实时场景。

## 6.5 总结

这一章中，你构建了关注系统和用户活动流。你学习了 Django 信号是如何工作的，并在项目中集成了 Redis。

下一章中，你会学习如何构建一个在线商店。你将创建一个产品目录，并使用会话构建购物车。你还讲学习如何使用 Celery 启动异步任务。