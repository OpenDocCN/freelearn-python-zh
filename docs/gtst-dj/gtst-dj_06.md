# 第六章：使用 Querysets 获取模型的数据

**Querysets** 用于数据检索，而不是直接构建 SQL 查询。它们是 Django 使用的 ORM 的一部分。ORM 用于通过抽象层将视图和控制器连接起来。开发人员可以使用对象模型类型，而无需编写 SQL 查询。我们将使用 querysets 来检索我们通过模型存储在数据库中的数据。这四个操作通常被 **CRUD** (**创建**，**读取**，**更新** 和 **删除**) 所总结。

本章中讨论的示例旨在向您展示查询集的工作原理。下一章将向您展示如何使用表单，以及如何将来自客户端的数据保存在模型中。

在本章结束时，我们将知道如何：

+   在数据库中保存数据

+   从数据库中检索数据

+   更新数据库中的数据

# 在数据库上持久化模型的数据

使用 Django 进行数据存储很简单。我们只需要在模型中填充数据，并使用方法将它们存储在数据库中。Django 处理所有的 SQL 查询；开发人员不需要编写任何查询。

## 填充模型并将其保存在数据库中

在将模型实例的数据保存到数据库之前，我们需要定义模型所需字段的所有值。我们可以在我们的视图索引中显示示例。

以下示例显示了如何保存模型：

```py
from TasksManager.models import Project # line 1
from django.shortcuts import render
def page(request):
  new_project = Project(title="Tasks Manager with Django", description="Django project to getting start with Django easily.", client_name="Me") # line 2
  new_project.save() # line 3
  return render(request, 'en/public/index.html', {'action':'Save datas of model'})
```

我们将解释我们视图的新行：

+   我们导入我们的 `models.py` 文件；这是我们将在视图中使用的模型

+   然后，我们创建我们的 `Project` 模型的一个实例，并用数据填充它

+   最后，我们执行 `save()` 方法，将当前数据保存在实例中

我们将通过启动开发服务器（或 runserver）来测试此代码，然后转到我们的 URL。在 `render()` 方法中，我们定义的 `action` 变量的值将被显示。要检查查询是否执行，我们可以使用管理模块。还有用于管理数据库的软件。

我们需要通过更改 `line 2` 中的值来添加更多记录。要了解如何做到这一点，我们需要阅读本章。

# 从数据库中获取数据

在使用 Django 从数据库中检索数据之前，我们使用 SQL 查询来检索包含结果的对象。使用 Django，根据我们是要获取一个记录还是多个记录，有两种检索记录的方式。

## 获取多条记录

要从模型中检索记录，我们必须首先将模型导入视图，就像我们之前保存数据到模型中一样。

我们可以按以下方式检索和显示 `Project` 模型中的所有记录：

```py
from TasksManager.models import Project
from django.shortcuts import render
def page(request):
  all_projects = Project.objects.all()
  return render(request, 'en/public/index.html', {'action': "Display all project", 'all_projects': all_projects})
```

显示项目的代码模板如下：

```py
{% extends "base.html" %}
{% block title_html %}
  Projects list
{% endblock %}
{% block h1 %}
  Projects list
{% endblock %}
{% block article_content %}
  <h3>{{ action }}</h3>
  {% if all_projects|length > 0 %}
  <table>
    <thead>
      <tr>
        <td>ID</td>
        <td>Title</td>
      </tr>
    </thead>
    <tbody>
    {% for project in all_projects %}
      <tr>
        <td>{{ project.id }}</td>
        <td>{{ project.title }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <span>No project.</span>
  {% endif %}
{% endblock %}
```

`all()` 方法可以链接到 SQL `SELECT * FROM` 查询。现在，我们将使用 `filter()` 方法来过滤我们的结果，并进行等效于 `SELECT * FROM Project WHERE field = value` 查询。

以下是筛选模型记录的代码：

```py
from TasksManager.models import Project
from django.shortcuts import render
def page(request):
  action='Display project with client name = "Me"'
  projects_to_me = Project.objects.filter(client_name="Me")
  return render(request, 'en/public/index.html', locals())
```

我们使用了一种新的语法将变量发送到模板。`locals()` 函数将所有本地变量发送到模板，这简化了渲染行。

### 提示

最佳实践建议您逐个传递变量，并且只发送必要的变量。

`filter()` 方法中的每个参数都定义了查询的过滤器。实际上，如果我们想要进行两个过滤，我们将编写以下代码行：

```py
projects_to_me = Project.objects.filter(client_name="Me", title="Project test")
```

这行代码等同于以下内容：

```py
projects_to_me = Project.objects.filter(client_name="Me")
projects_to_me = projects_to_me.filter(title="Project test") 
```

第一行可以分成两行，因为 querysets 是可链接的。可链接方法是返回查询集的方法，因此可以使用其他查询集方法。

使用 `all()` 和 `filter()` 方法获得的响应是查询集类型。查询集是可以迭代的模型实例集合。

## 仅获取一条记录

我们将在本章中看到的方法返回 `Model` 类型的对象，这些对象将用于记录关系或修改恢复的模型实例。

要使用查询集检索单个记录，我们应该像下面这行代码一样使用`get()`方法：

```py
first_project = Project.objects.get(id="1")
```

`get()`方法在作为`filter()`方法使用时接受过滤参数。但是，设置检索单个记录的过滤器时要小心。

如果`get()`的参数是`client_name = "Me"`，如果我们有超过两条记录与`client_name`对应，它将生成错误。

## 从查询集实例中获取模型实例

我们说过只有`get()`方法才能检索模型的实例。这是正确的，但有时从查询集中检索模型的实例也是有用的。

例如，如果我们想要获取客户`Me`的第一条记录，我们将写：

```py
queryset_project = Project.objects.filter(client_name="Me").order_by("id")
# This line returns a queryset in which there are as many elements as there are projects for the Me customer

first_item_queryset = queryset_project[:1]
# This line sends us only the first element of this queryset, but this element is not an instance of a model

project = first_item_queryset.get()
# This line retrieves the instance of the model that corresponds to the first element of queryset
```

这些方法是可链接的，所以我们可以写下面的一行代码，而不是前面的三行代码：

```py
project = Project.objects.filter(client_name="Me").order_by("id")[:1].get()
```

# 使用 get 参数

现在我们已经学会了如何检索记录，也知道如何使用 URL，我们将创建一个页面，用于显示项目的记录。为此，我们将看到一个新的 URL 语法：

```py
url(r'^project-detail-(?P<pk>\d+)$', 'TasksManager.views.project_detail.page', name="project_detail"),
```

这个 URL 包含一个新的字符串，`(?P<pk>\d+)`。它允许具有十进制参数的 URL 是有效的，因为它以`\d`结尾。结尾处的`+`字符表示参数不是可选的。`<pk>`字符串表示参数的名称是`pk`。

Django 的系统路由将直接将此参数发送到我们的视图。要使用它，只需将其添加到我们的`page()`函数的参数中。我们的视图变成了以下内容：

```py
from TasksManager.models import Project
from django.shortcuts import render
def page(request, pk):
  project = Project.objects.get(id=pk)
  return render(request, 'en/public/project_detail.html', {'project' : project})
```

然后，我们将创建我们的`en/public/project_detail.html`模板，从`base.html`扩展，并在`article_content`块中添加以下代码：

```py
<h3>{{ project.title }}</h3>
<h4>Client : {{ project.client_name }}</h4>
<p>
  {{ project.description }}
</p>
```

我们刚刚编写了我们的第一个包含参数的 URL。我们以后会用到这个，特别是在关于基于类的视图的章节中。

# 保存外键

我们已经从模型中记录了数据，但到目前为止，我们从未在关系数据库中记录过。以下是一个我们将在本章后面解释的关系记录的例子：

```py
from TasksManager.models import Project, Task, Supervisor, Developer
from django.shortcuts import render
from django.utils import timezone
def page(request):
  # Saving a new supervisor
  new_supervisor = Supervisor(name="Guido van Rossum", login="python", password="password", last_connection=timezone.now(), email="python@python.com", specialisation="Python") # line 1
  new_supervisor.save()
  # Saving a new developer
  new_developer = Developer(name="Me", login="me", password="pass", last_connection=timezone.now(), email="me@python.com", supervisor=new_supervisor)
  new_developer.save()
  # Saving a new task
  project_to_link = Project.objects.get(id = 1) # line 2
  new_task = Task(title="Adding relation", description="Example of adding relation and save it", time_elapsed=2, importance=0, project=project_to_link, developer=new_developer) # line 3
  new_task.save()
  return render(request, 'en/public/index.html', {'action' : 'Save relationship'})
```

在这个例子中，我们加载了四个模型。这四个模型用于创建我们的第一个任务。实际上，一个职位与一个项目和开发人员相关联。开发人员附属于监督者。

根据这种架构，我们必须首先创建一个监督者来添加一个开发人员。以下列表解释了这一点：

+   我们创建了一个新的监督者。请注意，扩展模型无需额外的步骤来记录。在`Supervisor`模型中，我们定义了`App_user`模型的字段，没有任何困难。在这里，我们使用`timezone`来记录当天的日期。

+   我们寻找第一个记录的项目。这行代码的结果将在`project_to_link`变量中记录`Model`类实例的遗留。只有`get()`方法才能给出模型的实例。因此，我们不应该使用`filter()`方法。

+   我们创建了一个新的任务，并将其分配给代码开头创建的项目和刚刚记录的开发人员。

这个例子非常全面，结合了我们从一开始学习的许多元素。我们必须理解它，才能继续在 Django 中编程。

# 更新数据库中的记录

Django 中有两种机制可以更新数据。实际上，有一种机制可以更新一条记录，另一种机制可以更新多条记录。

## 更新模型实例

更新现有数据非常简单。我们已经看到了如何做到这一点。以下是一个修改第一个任务的例子：

```py
from TasksManager.models import Project, Task
from django.shortcuts import render
def page(request):
  new_project = Project(title = "Other project", description="Try to update models.", client_name="People")
  new_project.save()
  task = Task.objects.get(id = 1)
  task.description = "New description"
  task.project = new_project
  task.save()
  return render(request, 'en/public/index.html', {'action' : 'Update model'})
```

在这个例子中，我们创建了一个新项目并保存了它。我们搜索了我们的任务，找到了`id = 1`。我们修改了描述和项目，使其与任务相关联。最后，我们保存了这个任务。

## 更新多条记录

要一次编辑多条记录，必须使用带有查询集对象类型的`update()`方法。例如，我们的`People`客户被名为`Nobody`的公司购买，因此我们需要更改所有`client_name`属性等于`People`的项目：

```py
from TasksManager.models import Project
from django.shortcuts import render
def page(request):
  task = Project.objects.filter(client_name = "people").update(client_name="Nobody")
  return render(request, 'en/public/index.html', {'action' : 'Update for many model'})
```

查询集的`update()`方法可以更改与该查询集相关的所有记录。这个方法不能用于模型的实例。

# 删除记录

要删除数据库中的记录，我们必须使用`delete()`方法。删除项目比更改项目更容易，因为该方法对查询集和模型实例都是相同的。一个例子如下：

```py
from TasksManager.models import Task
from django.shortcuts import render
def page(request):
  one_task = Task.objects.get(id = 1)
  one_task.delete() # line 1
  all_tasks = Task.objects.all()
  all_tasks.delete() # line 2
  return render(request, 'en/public/index.html', {'action' : 'Delete tasks'})
```

在这个例子中，`第 1 行`删除了`id = 1`的污渍。然后，`第 2 行`删除了数据库中所有现有的任务。

要小心，因为即使我们使用了一个 Web 框架，我们仍然掌握着数据。在这个例子中不需要确认，也没有进行备份。默认情况下，具有`ForeignKey`的模型删除规则是`CASCADE`值。这个规则意味着如果我们删除一个模板实例，那么对这个模型有外键的记录也将被删除。

# 获取关联记录

我们现在知道如何在数据库中创建、读取、更新和删除当前记录，但我们还没有恢复相关的对象。在我们的`TasksManager`应用程序中，检索项目中的所有任务将是有趣的。例如，由于我们刚刚删除了数据库中所有现有的任务，我们需要创建其他任务。我们特别需要在本章的其余部分为项目数据库创建任务。

使用 Python 及其面向对象模型的全面实现，访问相关模型是直观的。例如，当`login = 1`时，我们将检索所有项目任务：

```py
from TasksManager.models import Task, Project
from django.shortcuts import render
def page(request):
  project = Project.objects.get(id = 1)
  tasks = Task.objects.filter(project = project)
  return render(request, 'en/public/index.html', {'action' : 'Tasks for project', 'tasks':tasks})
```

现在我们将查找`id = 1`时的项目任务：

```py
from TasksManager.models import Task, Project
from django.shortcuts import render
def page(request):
  task = Task.objects.get(id = 1)
  project = task.project
  return render(request, 'en/public/index.html', {'action' : 'Project for task', 'project':project})
```

现在我们将使用关系来访问项目任务。

# 查询集的高级用法

我们学习了允许您与数据交互的查询集的基础知识。在特定情况下，需要对数据执行更复杂的操作。

## 在查询集中使用 OR 运算符

在查询集过滤器中，我们使用逗号来分隔过滤器。这一点隐含地意味着逻辑运算符`AND`。当应用`OR`运算符时，我们被迫使用`Q`对象。

这个`Q`对象允许您在模型上设置复杂的查询。例如，要选择客户`Me`和`Nobody`的项目，我们必须在视图中添加以下行：

```py
from TasksManager.models import Task, Project
from django.shortcuts import render
from django.db.models import Q
def page(request):
  projects_list = Project.objects.filter(Q(client_name="Me") | Q(client_name="Nobody"))
  return render(request, 'en/public/index.html', {'action' : 'Project with OR operator', 'projects_list':projects_list})
```

## 使用小于和大于的查找

使用 Django 查询集，我们不能使用`<`和`>`运算符来检查一个参数是否大于或小于另一个参数。

您必须使用以下字段查找：

+   `__gte`：这相当于 SQL 的大于或等于运算符，`>=`

+   `__gt`：这相当于 SQL 的大于运算符，`>`

+   `__lt`：这相当于 SQL 的小于运算符，`<`

+   `__lte`：这相当于 SQL 的小于或等于运算符，`<=`

例如，我们将编写一个查询集，可以返回持续时间大于或等于四小时的所有任务：

```py
tasks_list = Task.objects.filter(time_elapsed__gte=4)
```

## 执行排除查询

在网站的上下文中，排除查询可能很有用。例如，我们想要获取持续时间不超过四小时的项目列表：

```py
from TasksManager.models import Task, Project
from django.shortcuts import renderdef page(request):
  tasks_list = Task.objects.filter(time_elapsed__gt=4)
  array_projects = tasks_list.values_list('project', flat=True).distinct()
  projects_list = Project.objects.all()
  projects_list_lt4 = projects_list.exclude(id__in=array_projects)
  return render(request, 'en/public/index.html', {'action' : 'NOT IN SQL equivalent', 'projects_list_lt4':projects_list_lt4})
```

```py
In the first queryset, we first retrieve the list of all the tasks for which `time_elapsed` is greater than `4`In the second queryset, we got the list of all the related projects in these tasksIn the third queryset, we got all the projectsIn the fourth queryset, we excluded all the projects with tasks that last for more than `4` hours
```

## 进行原始 SQL 查询

有时，开发人员可能需要执行原始的 SQL 查询。为此，我们可以使用`raw()`方法，将 SQL 查询定义为参数。以下是一个检索第一个任务的示例：

```py
first_task = Project.objects.raw("SELECT * FROM TasksManager_project")[0]
```

要访问第一个任务的名称，只需使用以下语法：

```py
first_task.title
```

# 总结

在本章中，我们学习了如何通过 Django ORM 处理数据库。确实，借助 ORM，开发人员不需要编写 SQL 查询。在下一章中，我们将学习如何使用 Django 创建表单。
