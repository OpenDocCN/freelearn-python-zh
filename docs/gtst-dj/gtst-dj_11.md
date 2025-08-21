# 第十一章：在 Django 中使用 AJAX

AJAX 是异步 JavaScript 和 XML 的缩写。这项技术允许浏览器使用 JavaScript 与服务器异步通信。不一定需要刷新网页来执行服务器上的操作。

已发布许多基于 AJAX 的 Web 应用程序。Web 应用程序通常被描述为只包含一个页面的网站，并且使用 AJAX 服务器执行所有操作。

如果不使用库，使用 AJAX 需要大量代码行才能与多个浏览器兼容。包含 jQuery 后，可以轻松进行 AJAX 请求，同时与许多浏览器兼容。

在本章中，我们将涵盖：

+   使用 JQuery

+   JQuery 基础

+   在任务管理器中使用 AJAX

# 使用 jQuery

jQuery 是一个旨在有效操作 HTML 页面的 DOM 的 JavaScript 库。**DOM**（**文档对象模型**）是 HTML 代码的内部结构，jQuery 极大地简化了处理过程。

以下是 jQuery 的一些优点：

+   DOM 操作可以使用 CSS 1-3 选择器

+   它集成了 AJAX

+   可以使用视觉效果来使页面动画化

+   良好的文档，有许多示例

+   围绕 jQuery 创建了许多库

# jQuery 基础

在本章中，我们使用 jQuery 进行 AJAX 请求。在使用 jQuery 之前，让我们先了解其基础知识。

## jQuery 中的 CSS 选择器

在样式表中使用的 CSS 选择器可以有效地检索具有非常少代码的项目。这是一个非常有趣的功能，它以以下语法实现在 HTML5 选择器 API 中：

```py
item = document.querySelector('tag#id_content');
```

jQuery 还允许我们使用 CSS 选择器。要使用 jQuery 执行相同的操作，必须使用以下语法：

```py
item = $('tag#id_content');
```

目前，最好使用 jQuery 而不是选择器 API，因为 jQuery 1.x.x 保证与旧版浏览器的兼容性很好。

## 获取 HTML 内容

可以使用`html()`方法获取两个标签之间的 HTML 代码：

```py
alert($('div#div_1').html());
```

这行将显示一个警报，其中包含`<div id="div_1">`标签的 HTML 内容。关于输入和文本区域标签，可以以与`val()`方法相同的方式恢复它们的内容。

## 在元素中设置 HTML 内容

更改标签的内容非常简单，因为我们使用了与恢复相同的方法。两者之间的主要区别在于我们将一个参数发送到方法。

因此，以下指令将在 div 标签中添加一个按钮：

```py
$('div#div_1').html($('div#div_1').html()+'<button>JQuery</button>');
```

## 循环元素

jQuery 还允许我们循环所有与选择器匹配的元素。为此，您必须使用`each()`方法，如下例所示：

```py
var cases = $('nav ul li').each(function() {
  $(this).addClass("nav_item");
});
```

## 导入 jQuery 库

要使用 jQuery，必须首先导入库。将 jQuery 添加到网页有两种方法。每种方法都有其自己的优势，如下所述：

+   下载 jQuery 并从我们的 Web 服务器导入。使用此方法，我们可以控制库，并确保文件在我们自己的网站上也是可访问的。

+   使用 Google 托管书店的托管库，可从任何网站访问。优点是我们避免向我们的服务器发出 HTTP 请求，从而节省了一些功率。

在本章中，我们将在我们的 Web 服务器上托管 jQuery，以免受主机的限制。

我们将在应用程序的所有页面中导入 jQuery，因为我们可能需要多个页面。此外，浏览器的缓存将保留 jQuery 一段时间，以免频繁下载。为此，我们将下载 jQuery 1.11.0 并保存在`TasksManager/static/javascript/lib/jquery-1.11.0.js`文件中。

然后，您必须在`base.html`文件的 head 标签中添加以下行：

```py
<script src="img/jquery-1.11.0.js' %}"></script>
{% block head %}{% endblock %}
```

通过这些更改，我们可以在网站的所有页面中使用 jQuery，并且可以在扩展`base.html`的模板中的`head`块中添加行。

# 在任务管理器中使用 AJAX

在这一部分，我们将修改显示任务列表的页面，以便在 AJAX 中执行删除任务。为此，我们将执行以下步骤：

1.  在`task_list`页面上添加一个`删除`按钮。

1.  创建一个 JavaScript 文件，其中包含 AJAX 代码和处理 AJAX 请求返回值的函数。

1.  创建一个 Django 视图来删除任务。

我们将通过修改`tasks_list.html`模板来添加删除按钮。为此，您必须将`tasks_list`中的`for`任务循环更改为以下内容：

```py
{% for task in tasks_list %}
  <tr id="task_{{ task.id }}">
    <td><a href="{% url "task_detail" task.id %}">{{ task.title }}</a></td>
    <td>{{ task.description|truncatechars:25 }}</td>
    <td><a href="{% url "update_task" task.id %}">Edit</a></td>
    <td><button onclick="javascript:task_delete({{ task.id }}, '{% url "task
_delete_ajax" %}');">Delete</button></td>
  </tr>
{% endfor %}
```

在上面的代码中，我们向`<tr>`标签添加了一个`id`属性。这个属性将在 JavaScript 代码中很有用，当页面接收到 AJAX 响应时，它将删除任务行。我们还用一个执行 JavaScript `task_delete()` 函数的**删除**按钮替换了**删除**链接。新按钮将调用`task_delete()`函数来执行 AJAX 请求。这个函数接受两个参数：

+   任务的标识符

+   AJAX 请求的 URL

我们将在`static/javascript/task.js`文件中创建这个函数，添加以下代码：

```py
function task_delete(id, url){
  $.ajax({
    type: 'POST', 
    // Here, we define the used method to send data to the Django views. Other values are possible as POST, GET, and other HTTP request methods.
    url: url, 
    // This line is used to specify the URL that will process the request.
    data: {task: id}, 
    // The data property is used to define the data that will be sent with the AJAX request.
    dataType:'json', 
    // This line defines the type of data that we are expecting back from the server. We do not necessarily need JSON in this example, but when the response is more complete, we use this kind of data type.
    success: task_delete_confirm,
    // The success property allows us to define a function that will be executed when the AJAX request works. This function receives as a parameter the AJAX response.
    error: function () {alert('AJAX error.');} 
    // The error property can define a function when the AJAX request does not work. We defined in the previous code an anonymous function that displays an AJAX error to the user.
  });
}
function task_delete_confirm(response) {
  task_id = JSON.parse(response); 
  // This line is in the function that receives the AJAX response when the request was successful. This line allows deserializing the JSON response returned by Django views.
  if (task_id>0) {
    $('#task_'+task_id).remove(); 
    // This line will delete the <tr> tag containing the task we have just removed
  }
  else {
    alert('Error');
  }
}
```

我们必须在`tasks_list.html`模板中的`title_html`块之后添加以下行，以在模板中导入`task.js`：

```py
{% load static %}
{% block head %}
  <script src="img/task.js' %}"></script>
{% endblock %}
```

我们必须在`urls.py`文件中添加以下 URL：

```py
  url(r'^task-delete-ajax$', 'TasksManager.views.ajax.task_delete_ajax.page', name="task_delete_ajax"),
```

这个 URL 将使用`view/ajax/task_delete_ajax.py`文件中包含的视图。我们必须创建带有`__init__.py`文件的 AJAX 模块，以及我们的`task_delete_ajax.py`文件，内容如下：

```py
from TasksManager.models import Task
from django.http import HttpResponse
from django import forms
from django.views.decorators.csrf import csrf_exempt
# We import the csrf_exempt decorator that we will use to line 4.
import json
# We import the json module we use to line 8.
class Form_task_delete(forms.Form):
# We create a form with a task field that contains the identifier of the task. When we create a form it allows us to use the Django validators to check the contents of the data sent by AJAX. Indeed, we are not immune that the user sends data to hack our server.
  task       = forms.IntegerField()
@csrf_exempt
# This line allows us to not verify the CSRF token for this view. Indeed, with AJAX we cannot reliably use the CSRF protection.
def page(request):
  return_value="0"
  # We create a variable named return_value that will contain a code returned to our JavaScript function. We initialize the value 0 to the variable.
  if len(request.POST) > 0:
    form = Form_task_delete(request.POST)
    if form.is_valid():
    # This line allows us to verify the validity of the value sent by the AJAX request.
      id_task = form.cleaned_data['task']
      task_record = Task.objects.get(id = id_task)
      task_record.delete()
      return_value=id_task
      # If the task been found, the return_value variable will contain the value of the id property after removing the task. This value will be returned to the JavaScript function and will be useful to remove the corresponding row in the HTML table.
  # The following line contains two significant items. The json.dumps() function will return a serialized JSON object. Serialization allows encoding an object sequence of characters. This technique allows different languages to share objects transparently. We also define a content_type to specify the type of data returned by the view.
  return HttpResponse(json.dumps(return_value), content_type = "application/json")
```

# 总结

在本章中，我们学习了如何使用 jQuery。我们看到了如何使用这个库轻松访问 DOM。我们还在我们的`TasksManager`应用程序上创建了一个 AJAX 请求，并编写了处理这个请求的视图。

在下一章中，我们将学习如何部署基于 Nginx 和 PostgreSQL 服务器的 Django 项目。我们将逐步看到并讨论安装步骤。
