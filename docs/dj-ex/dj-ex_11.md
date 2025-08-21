# 第十一章：缓存内容

在上一章中，你使用模型继承和通用关系来创建灵活的课程内容模型。你还使用基于类的视图，表单集和 AJAX 排序内容创建了一个课程管理系统。在本章中，你会学习学习以下内容：

- 创建显示课程信息的公开视图
- 构建一个学生注册系统
- 在课程中管理学生报名
- 渲染不同的课程内容
- 使用缓存框架缓存内容

我们从创建课程目录开始，让学生可以浏览已存在的课程，并且可以报名参加。

## 11.1 显示课程

对于我们的课程目录，我们需要构建以下功能：

- 列出所有可用课程，可用通过主题过滤
- 显示单个课程的概述

编辑`courses`应用的`views.py`文件，添加以下代码：

```py
from .models import Subject
from django.db.models import Count

class CourseListView(TemplateResponseMixin, View):
    model = Course
    template_name = 'courses/course/list.html'

    def get(self, request, subject=None):
        subjects = Subject.objects.annotate(
            total_courses=Count('courses')
        )
        courses = Course.objects.annotate(
            total_modules=Count('modules')
        )
        if subject:
            subject = get_object_or_404(Subject, slug=subject)
            courses = courses.filter(subject=subject)
        return self.render_to_response({
            'subjects': subjects,
            'subject': subject,
            'courses': courses
        })
```

这是`CourseListView`视图。它从`TemplateResponseMixin`和`View`继承。在这个视图中，我们执行以下任务：

1. 我们检索所有主题，包括每个主题的课程总数。我们在 ORM 的`annotate()`方法中使用`Count()`聚合函数完成这个功能。
2. 我们检索所有可用的课程，包括每个课程的单元总数。
3. 如果给定了一个主题的`slug` URL 参数，我们检索相应的主题对象，并限制查询属于这个主题的课程。
4. 我们使用`TemplateResponseMixin`提供的`render_to_response()`方法在模板中渲染对象，并返回一个 HTTP 响应。

让我们创建一个详情视图，显示单个课程的概述。在`views.py`文件中添加以下代码：

```py
from django.views.generic.detail import DetailView

class CourseDetailView(DetailView):
    model = Course
    template_name = 'courses/course/detail.html'
```

这个视图从 Django 提供的通用`DetailView`视图继承。我们指定了`model`和`template_name`属性。Django 的`DetailView`期望一个主键（pk）或者`slug` URL 参数，来检索给定模型的单个对象。然后它渲染`template_name`中指定的模板，并在上下文中包括`object`对象。

编辑`educa`项目的主`urls.py`文件，并添加以下代码：

```py
from courses.views import CourseListView

urlpatterns = [
	# ...
	url(r'^$', CourseListView.as_view(), name='course_list'),
]
```

因为我们想在`http://127.0.0.1:8000/`显示课程列表，而`courses`应用的其它所有 URL 带`/course/`前缀，所以我们在项目的主`urls.py`文件中添加`course_list` URL 模式。

编辑`courses`应用的`urls.py`文件，添加以下 URL 模式：

```py
url(r'^subject/(?P<subject>[\w-]+)/$', 
    views.CourseListView.as_view(), 
    name='course_list_subject'),
url(r'^(?P<slug>[\w-]+)/$', 
    views.CourseDetailView.as_view(),
    name='course_detail'),
```

我们定义了以下 URL 模式：

- `course_list_subject`：显示一个主题的所有课程
- `course_detail_subject`：显示单个课程的概述

让我们为`CourseListView`和`CourseDetailView`视图构建模板。在`courses`应用的`templates/courses/`目录中创建以下文件结构：

```py
course/
	list.html
	detail.html
```

编辑`courses/course/list.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    {% if subject %}
        {{ subject.title }} courses
    {% else %}
        All courses
    {% endif %}
{% endblock title %}

{% block content %}
    <h1>
        {% if subject %}
            {{ subject.title }} courses
        {% else %}
            All courses
        {% endif %}
    </h1>
    <div class="contents">
        <h3>Subjects</h3>
        <ul id="modules">
            <li {% if not subject %}class="selected"{% endif %}>
                <a href="{% url "course_list" %}">All</a>
            </li>
            {% for s in subjects %}
                <li {% if subject == s %}class="selected"{% endif %}>
                    <a href="{% url "course_list_subject" s.slug %}">
                        {{ s.title }}
                        <br><span>{{ s.total_courses }} courses</span>
                    </a>
                </li>
            {% endfor %}
        </ul>
    </div>
    <div class="module">
        {% for course in courses %}
            {% with subject=course.subject %}
                <h3><a href="{% url "course_detail" course.slug %}">{{ course.title }}</a></h3>
                <p>
                    <a href="{% url "course_list_subject" subject.slug %}">{{ subject }}</a>
                    {{ coursse.total_modules }} modules.
                    Instructor: {{ course.owner.get_full_name }}
                </p>
            {% endwith %}
        {% endfor %}
    </div>
{% endblock content %}
```

这是显示可用课程列表的模板。我们创建了一个 HTML 列表显示所有`Subject`对象，并为每个`Subject`对象构建一个到`course_list_subject` URL 的链接。我们点击了`selected`类高亮显示当前主题（如果存在的话）。我们迭代每个`Course`对象，显示单元总数和教师姓名。

使用`python manage.py runserver`命令启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/`。你会看到类似以下的页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.1.png)

左侧边栏包括所有主题，已经每个主题的课程总数。你可以点击任何一个主题来过滤显示的课程。

编辑`courses/course/detail.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    {{ object.title }}
{% endblock title %}

{% block content %}
    {% with subject=course.subject %}
        <h1>
            {{ object.title }}
        </h1>
        <div class="module">
            <h2>Overview</h2>
            <p>
                <a href="{% url "course_list_subject" subject.slug %}">{{ subject.title }}</a>
                {{ course.modules.count }} modules.
                Instructors: {{ course.owner.get_full_name }}
            </p>
            {{ object.overview|linebreaks }}
        </div>
    {% endwith %}
{% endblock content %}
```

在这个模板中，我们显示单个课程的概述和详情。在浏览器中打开`http://127.0.0.1:8000/`，然后点击其中一个课程。你会看到以下结构的页面：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.2.png)

我们已经创建了展示课程的公开区域。接下来，让我们允许用户注册为学生，并报名参加课程。

## 11.2 添加学生注册

使用以下命令创建一个新应用：

```py
python manage.py startapp students
```

编辑`educa`项目的`settings.py`文件，把`students`添加到`INSTALLED_APPS`设置中：

```py
INSTALLED_APPS = [
	# ...
	'students',
]
```

### 11.2.1 创建学生注册视图

编辑`students`应用的`views.py`文件，并添加以下代码：

```py
from django.core.urlresolvers import reverse_lazy
from django.views.generic.edit import CreateView
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login

class StudentRegistrationView(CreateView):
    template_name = 'students/student/registration.html'
    form_class = UserCreationForm
    success_url = reverse_lazy('student_course_list')

    def form_valid(self, form):
        result = super().form_valid(form)
        cd = form.cleaned_data
        user = authenticate(
            username=cd['username'],
            password=cd['password']
        )
        login(self.request, user)
        return result
```

这个视图允许学生在我们网站上注册。我们使用通用的`CreateView`，它提供了创建模型对象的功能。这个视图需要以下属性：

- `template_name`：用于这个视图的模板的路径。
- `form_class`：创建对象的表单，必须是一个`ModelForm`。我们使用 Django 的`UserCreationForm`作为注册表单，来创建`User`对象。
- `success_url`：当表单提交成功后，重定向用户的 URL。我们逆向之后会创建的`student_course_list` URL，列出学生报名参加的课程。

当提交了有效的表单数据后，会执行`form_valid()`方法。它必须返回一个 HTTP 响应。当用户注册成功后，我们覆写这个方法来登录用户。

在`students`应用目录中创建`urls.py`文件，并添加以下代码：

```py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^register/$', views.StudentRegistrationView.as_view(), name='student_registration'),
]
```

编辑`educa`项目的主`urls.py`文件，把`students`应用的 URL 添加到 URL 配置中：

```py
url(r'^students/', include('students.urls')),
```

在`students`应用中创建以下文件结构：

```py
templates/
	students/
		student/
			registration.html
```

编辑`students/student/registration.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    Sign up
{% endblock title %}

{% block content %}
    <h1>
        Sign up
    </h1>
    <div class="module">
        <p>Enter your details to create an account:</p>
        <form action="" method="post">
            {{ form.as_p }}
            {% csrf_token %}
            <p><input type="submit" value="Create my account"></p>
        </form>
    </div>
{% endblock content %}
```

最后，编辑`educa`项目的`settings.py`文件，并添加以下代码：

```py
from django.core.urlresolvers import reverse_lazy
LOGIN_REDIRECT_URL = reverse_lazy('student_course_list')
```

一次成功登录后，如果请求中没有`next`参数，则`auth`模块用这个设置重定义用户。

启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/students/register/`。你会看到以下注册表单：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.3.png)

### 11.2.2 报名参加课程

用户创建账户之后，他们应该可以报名参加课程。为了存储花名册，我们需要在`Course`和`User`模型之间创建多对多关系。编辑`courses`应用的`models.py`文件，并在`Course`模型中添加以下字段：

```py
students = models.ManyToManyField(User, related_name='courses_joined', blank=True)
```

在终端中执行以下命令，为这个修改创建一个数据库迁移：

```py
python manage.py makemigrations
```

你会看到类似这样的输出：

```py
Migrations for 'courses':
  courses/migrations/0004_course_students.py
    - Add field students to course
```

然后执行以下命令，应用数据库迁移：

```py
python manage.py migrate
```

你会看到以下输出：

```py
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, courses, sessions
Running migrations:
  Applying courses.0004_course_students... OK
```

现在，我们可以用学生参加的课程关联到学生。让我们创建学生参加课程的功能。

在`students`应用目录中创建`forms.py`文件，并添加以下代码：

```py
from django import forms
from courses.models import Course

class CourseEnrollForm(forms.Form):
    course = forms.ModelChoiceField(
        queryset=Course.objects.all(),
        widget=forms.HiddenInput
    )
```

我们将把这个表单用于学生报名。`course`字段用于学生报名的课程。因此它是一个`ModelChoiceField`。因为我们不会显示这个字段，所以使用了`HiddenInput`组件。我们将在`CourseDetailView`视图中使用这个表单，来显示一个报名按钮。

编辑`students`应用的`views.py`文件，并添加以下代码：

```py
from django.views.generic.edit import FormView
from braces.views import LoginRequiredMixin
from .forms import CourseEnrollForm

class StudentEnrollCourseView(LoginRequiredMixin, FormView):
    course = None
    form_class = CourseEnrollForm

    def form_valid(self, form):
        self.course = form.cleaned_data['course']
        self.course.students.add(self.request.user)
        return super().form_valid(form)
        
    def get_success_url(self):
        return reverse_lazy(
            'student_course_detail',
            args=[self.course.id]
        )
```

这是`StudentEnrollCourseView`视图。它处理学生报名课程。这个视图从`LoginRequiredMixin`继承，所以只有登录的用户才可以访问这个视图。因为我们需要处理表单的提交，所以它还从 Django 的`FormView`继承。我们为`form_class`属性使用`CourseEnrollForm`表单，还定义了存储给定`Course`对象的`course`属性。当表单有效时，我们添加当前用户到课程的注册学生中。

`get_success_url()`方法返回一个 URL，如果表单提交成功，则重定向用户到这个 URL。这个方法等同于`success_url`属性。我们逆向之后会创建的`student_course_detail` URL，用来显示课程内容。

编辑`students`应用的`urls.py`文件，添加以下 URL 模型：

```py
url(r'^enroll-course/$', views.StudentEnrollCourseView.as_view(), name='student_enroll_course'),
```

让我们在课程概述页面添加报名按钮表单。编辑`courses`应用的`views.py`文件，并修改`CourseDetailView`：

```py
from students.forms import CourseEnrollForm

class CourseDetailView(DetailView):
    model = Course
    template_name = 'courses/course/detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['enroll_form'] = CourseEnrollForm(
            initial={'course': self.object}
        )
        return context
```

我们使用`get_context_data()`方法在上下文中包括报名表单，用于渲染模板。我们使用当前`Course`对象初始化表单的隐藏课程字段，因此可以直接提交这个字段。编辑`courses/course/detail.html`模板，找到这行代码：

```py
{{ object.overview|linebreaks }}
```

替换为以下代码：

```py
{{ object.overview|linebreaks }}
{% if request.user.is_authenticated %}
    <form action="{% url "student_enroll_course" %}" method="post">
        {{ enroll_form }}
        {% csrf_token %}
        <input type="submit" class="button" value="Enroll now">
    </form>
{% else %}
    <a href="{% url "student_registration" %}" class="button">
        Register to enroll
    </a>
{% endif %}
```

这是报名参加课程的按钮。如果用户已认证，则显示报名按钮和指向`student_enroll_course` URL 的隐藏表单。如果用户未认证，则显示在网站注册的链接。

确保开发服务器正在运行，在浏览器中打开`http://127.0.0.1:8000/`，然后点击一个课程。如果你已经登录，你会在课程概述下面看到`ENROLL NOW`按钮，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.4.png)

如果你没有登录，则会看到`Register to enroll`按钮。

## 11.3 访问课程内容

我们需要一个视图显示学生参加的课程，以及一个访问实际课程内容的视图。编辑`students`应用的`views.py`文件，并添加以下代码：

```py
from django.views.generic.list import ListView
from courses.models import Course

class StudentCourseListView(LoginRequiredMixin, ListView):
    model = Course
    template_name = 'students/course/list.html'

    def get_queryset(self):
        qs = super().get_queryset()
        return qs.filter(students__in=[self.request.user])
```

这是视图列出学生报名参加的课程。它从`LoginRequiredMixin`继承，确保只有登录的用户才能访问这个视图。它还从通用的`ListView`继承，显示`Course`对象列表。我们覆写`get_queryset()`方法，只检索用户报名的课程：我们用`students`字段过滤 QuerySet。

然后在`views.py`文件添加以下代码：

```py
from django.views.generic.detail import DetailView

class StudentCourseDetailView(DetailView):
    model = Course
    template_name = 'students/course/detail.html'

    def get_queryset(self):
        qs = super().get_queryset()
        return qs.filter(students__in=[self.request.user])

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # get course object
        course = self.get_object()
        if 'module_id' in self.kwargs:
            # get current module
            context['module'] = course.modules.get(
                id=self.kwargs['module_id']
            )
        else:
            # get first module
            context['module'] = course.modules.all()[0]
        return context
```

这是`StudentCourseDetailView`视图。我们覆写了`get_queryset()`方法来限制 QuerySet 为用户报名的课程。我们还覆写了`get_context_data()`方法，如果给定了 URL 参数`module_id`，则在上下文中设置一个课程单元。否则我们设置课程的第一个单元。这样，学生可以在课程中浏览各个单元。

编辑`students`应用的`urls.py`文件，并添加以下 URL 模式：

```py
url(r'^courses/$', 
    views.StudentCourseListView.as_view(), 
    name='student_course_list'),
url(r'^course/(?P<pk>\d+)/$', 
    views.StudentCourseDetailView.as_view(), 
    name='student_course_detail'),
url(r'^course/(?P<pk>\d+)/(?P<module_id>\d+)/$', 
    views.StudentCourseDetailView.as_view(), 
    name='student_course_detail_module'),
```

在`students`应用的`templates/students/`目录中创建以下文件结构：

```py
course/
	detail.html
	list.html
```

编辑`students/course/list.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}My courses{% endblock title %}

{% block content %}
    <h1>My courses</h1>

    <div class="module">
        {% for course in object_list %}
            <div class="course-info">
                <h3>{{ course.title }}</h3>
                <p><a href="url "student_course_detail" course.id %}">Access contents</a></p>
            </div>
        {% empty %}
            <p>
                You are not enrolled in any courses yet.
                <a href="{% url "course_list" %}">Browse courses</a>
                to enroll in a course.
            </p>
        {% endfor %}
    </div>
{% endblock content %}
```

这个模板显示用户报名的课程。编辑`students/course/detail.html`模板，并添加以下代码：

```py
{% extends "base.html" %}

{% block title %}
    {{ object.title }}
{% endblock title %}

{% block content %}
    <h1>
        {{ module.title }}
    </h1>
    <div class="contents">
        <h3>Modules</h3>
        <ul id="modules">
            {% for m in object.modules.all %}
                <li data-id="{{ m.id }}" {% if m == module %}class="selected"{% endif %}>
                    <a href="{% url "student_course_detail_module" object.id m.id %}">
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
    </div>
    <div class="module">
        {% for content in module.contents.all %}
            {% with item=content.item %}
                <h2>{{ item.title }}</h2>
                {{ item.render }}
            {% endwith %}
        {% endfor %}
    </div>
{% endblock content %}
```

报名的学生通过这个模板访问一个课程的内容。首先，我们构建了一个包括所有课程单元的 HTML 列表，并高亮显示当前单元。然后我们迭代当前单元内容，并用`{{ item.render }}`访问每个内容项来显示它。接下来我们会添加`render()`方法到内容模型中。该方法负责适当的渲染内容。

### 11.3.1 渲染不同类型的内容

我们需要提供一种方式来渲染每种类型的内容。编辑`courses`应用的`models.py`文件，并在`ItemBase`模型中添加`render()`方法：

```py
from django.template.loader import render_to_string
from django.utils.safestring import mark_safe

class ItemBase(models.Model):
    # ...
    def render(self):
        return render_to_string(
            'courses/content/{}.html'.format(self._meta.model_name),
            {'item': self}
        )
```

这个方法使用`render_to_string()`方法渲染模板，并返回一个字符串作为被渲染的内容。每种类型的内容使用内容模型的模板名称渲染。我们用`self._meta.model_name`构建适当的模板名称。`render()`方法为渲染各种内容提供了通用的接口。

在`courses`应用的`templates/courses/`目录中创建以下文件结构：

```py
content/
	text.html
	file.html
	image.html
	video.html
```

编辑`courses/content/text.html`模板，并添加以下代码：

```py
{{ item.content|linebreaks|safe}}
```

编辑`courses/content/file.html`模板， 并添加以下代码：

```py
<p><a href="{{ item.file.url }}" class="button">Download file</a></p>
```

编辑`courses/content/image.html`模板，并添加以下代码：

```py
<p><img src="{{ item.file.url }}"></p>
```

要使用`ImageField`和`FileField`上传文件，我们需要设置项目用开发服务器提供多媒体文件服务。编辑项目的`settings.py`文件，并添加以下代码：

```py
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')
```

记住，`MEDIA_URL`是提供多媒体文件上传服务的基础 URL，而`MEDIA_ROOT`是存储文件的本地路径。

编辑项目的主`urls.py`文件，添加以下导入：

```py
from django.conf import settings
from django.conf.urls.static import static
```

然后在文件结尾添加以下代码：

```py
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

现在，你的项目可用使用开发服务器上传和保存多媒体文件了。记住，开发服务器不适合生产环境使用。我们会在下一章学习如何配置生产环境。

我们还需要一个模板渲染`Video`对象。我们将使用 django-embed-video 嵌入视频内容。Django-embed-video 是一个第三方 Django 应用，允许你通过提供一个视频的公开 URL（比如从 YouTube 或 Vimeo 源），在模板中嵌入视频。

使用以下命令安装这个包：

```py
pip install django-embed-video
```

然后编辑项目的`settings.py`文件，把`embed_video`添加到`INSTALLED_APPS`设置中。你可以在[这里](http://django-embed-video.readthedocs.io/en/latest/)查看 django-embed-video 文档。

编辑`courses/content/video.html`模板，并添加以下代码：

```py
{% load embed_video_tags %}
{% video item.url 'small' %}
```

现在启动开发服务器，并在浏览器中访问`http://127.0.0.1:8000/course/mine/`。使用超级用户或者属于教师组的用户登录网站，然后在一个课程中添加多种内容。对于视频内容，你可以拷贝任何 YouTube 的 URL（比如 https://www.youtube.com/watch?v=bgV39DlmZ2U）到表单的`url`字段。添加内容到课程后，打开`http://127.0.0.1:8000/`，点击课程，然后点击`ENROLL NOW`按钮。你会报名参加课程，并重定向到`student_course_detail` URL。如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.5.png)

非常棒！你已经为渲染课程内容创建了一个通用接口，每种课程内容都已特定方式渲染。

## 11.4 使用缓存框架

对 Web 应用的 HTTP 请求通常需要数据库访问，数据处理和模板渲染。在处理方法，它们的开销比静态网站大多了。

当网站的流量变得越来越大，有些请求的开销可能会很大。此时缓存变得很有意义。通过在 HTTP 请求中缓存查询，计算结果或者渲染的内容，你会之后的请求中避免昂贵的操作。这意味着服务端更短的响应时间和更少的处理。

Django 包括一个健壮的缓存系统，允许你用不同级别的粒度缓存数据。你可以缓存单个查询，一个特定视图的输出，部分被渲染模板的内容，或者整个网站。内容会在默认时间内存储在缓存系统中。你可以为缓存的数据指定默认的超时时间。

当你的应用收到一个 HTTP 请求时，通常会这样使用缓存框架：

1. 尝试在缓存中查找请求的数据。
2. 如果找到，则返回缓存数据。
3. 如果没有找到，则执行以下步骤：
 - 执行获取数据所需的查询或处理。
 - 在缓存中保存生成的数据。
 - 返回数据

你可以在[这里](https://docs.djangoproject.com/en/1.11/topics/cache/)阅读 Django 缓存系统的详细信息。

### 11.4.1 可用的缓存后台

Django 自带数个缓存后台，分别是：

- `backends.memcached.MemcachedCache`或`backends.memcached.PyLibMCCache`：一个 Memcached 后台。Memcached 是一个快速和高效的基于内存的缓存服务。使用的后台取决于你选择的 Python 绑定的 Memcached。
- `backends.db.DatabaseCache`：使用数据库作为缓存系统。
- `backends.filebased.FileBasedCache`：使用文件存储系统。在单个文件中序列号和存储每个缓存值。
- `backends.locmem.LocMemCache`：本地内存缓存后台。这是默认的缓存后台。
- `backends.dummy.DummyCache`：一个只适用于开发的缓存后台。它实现了缓存接口，但不会真正缓存任何数据。这个缓存是预处理和线程安全的。

> 使用基于内存的缓存后台，比如`Memcached`，会有最优的性能。

### 11.4.2 安装 Memcached

我们将使用 Memcached 后台。Memcached 在内存中运行，并分配了一定数量的 RAM。当分配的 RAM 满了之后，Memcached 会移除最旧的数据。

从[这里](http://memcached.org/downloads)下载 Memcached。如果你使用的是 Linux，你可以使用以下命令安装 Memcached：

```py
./configure && make && make test && sudo make install
```

如果你使用的是 Mac OS X，你可以使用`brew install Memcached`命令安装。你可以在[这里](https://brew.sh/)下载 Homebrew。

如果你使用的是 Windows，你可以在[这里](http://code.jellycan.com/memcached/)找到 Windows 二进制版本。

> **译者注：**Windows 版本的链接已经失效。

安装 Memcached 后，打开终端，并使用以下命令启动：

```py
memcached -l 127.0.0.1:11211
```

Memcached 默认在 11211 端口运行。但你可以使用`-l`选项指定主机和端口。你可以在[这里](http://memcached.org/)查看更多关于 Memcached 的信息。

安装 Memcached 后，你需要安装它的 Python 绑定。使用以下命令完成安装：

```py
pip install python3-memcached
```

### 11.4.3 缓存设置

Django 提供了以下缓存设置：

- `CACHES`：包括项目中所有可用缓存的字典。
- `CACHE_MIDDLEWARE_ALIAS`：用于存储的缓存别名。
- `CACHE_MIDDLEWARE_KEY_PREFIX`：用于缓存键的前缀。如果你在数个网站中共享同一个缓存，可用设置前缀来避免键冲突。
- `CACHE_MIDDLEWARE_SECONDS`：缓存页面的默认秒数。

可用使用`CACHES`设置来配置项目的缓存系统。这个设置是一个字典，允许你配置多个缓存。`CACHES`字典中每个缓存可用指定以下数据：

- `BACKEND`：使用的缓存后台。
- `KEY_FUNCTION`：一个包括点号路径的可调用对象的字符串，可调用对象接收`prefix`，`version`和`key`作为参数，返回最终的缓存键。
- `KEY_PREFIX`：所有缓存键的字符串前缀，避免冲突。
- `LOCATION`：缓存的位置。根据缓存后台，它可能是一个目录，一个主机和端口，或者内存后台的命名。
- `OPTIONS`：传递给缓存后台的任何额外参数。
- `TIMEOUT`：存储缓存键的默认超时时间（单位秒）。默认是 300 秒，也就是 5 分钟。如果设置为 None，缓存键永远不会过期。
- `VERSION`：缓存键的默认版本号。用于缓存的版本控制。

### 11.4.4 添加 Memcached 到项目中

让我们为项目配置缓存。编辑`educa`项目的`settings.py`文件，并添加以下代码：

```py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache',
        'LOCATION': '127.0.0.1:11211',
    }
}
```

我们使用`MemcachedCache`后台。我们使用`address:port`语法之定义它的位置。如果你有多个 Memcached 实例，你可以为`LOCATION`使用列表。

#### 11.4.4.1 监控 Memcached

有一个`django-memcache-status`第三方包，可以在管理站点显示 Memcached 实例的统计信息。为了兼容 Python3，使用以下命令从分支中安装：

```py
pip install git+git://github.com/zenx/django-memcache-status.git
```

> **译者注：**最新版已经兼容 Python3，可以直接用 pip 命令安装。

编辑项目的`settings.py`文件，把`memcache_status`添加到`INSTALLED_APPS`设置中。确保 Memcached 正在运行中，并在另一个终端打开开发服务器，然后在浏览器中打开`http://127.0.0.1/8000/admin/`。使用超级用户登录管理站点，你会看到以下区域块：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.6.png)

这张图片显示了缓存的使用情况。绿色代表空闲缓存，红色代表已使用的空间。如果你点击标题，则会显示 Memcached 实例的详细统计。

我们已经为项目设置了 Memcached，并且可以监控它。让我们开始缓存数据！

### 11.4.5 缓存级别

Django 提供了以下缓存级别，按粒度的升序排列：

- `Low-level cache API`：提供了最细的粒度。允许你缓存具体的查询或计算。
- `Pre-view cache`：为单个视图提供缓存。
- `Template cache`：允许你缓存模板片段。
- `Pre-site cache`：最高级别缓存。它缓存整个网站。

> 实现缓存之前，请仔细考虑你的缓存策略。首先考虑费时的查询，或者不是基于单个用户的计算。

### 11.4.6 使用低级别的缓存 API

低级别缓存 API 允许你在缓存中保存任何粒度的对象。它位于`django.core.cache`中。你可以这样导入它：

```py
from django.core.cache import cache
```

这是使用默认缓存。它等价于`caches['default']`。也可以通过别名访问指定缓存：

```py
from django.core.cache import caches
my_cache = caches['alias']
```

让我们看看缓存 API 是如何工作的。使用`python manage.py shell`打开终端，然后执行以下代码：

```py
>>> from django.core.cache import cache
>>> cache.set('musician', 'Django Reinhardt', 20)
```

我们访问默认缓存后台，并使用`set(key, value, timeout)`存储`misician`键 20 秒，它的值是字符串`Django Reinhardt`。如果我们每页指定超时，则 Django 会使用`CACHE`设置中为缓存后台指定的默认超时。现在执行以下代码：

```py
>>> cache.get('musician')
'Django Reinhardt'
```

我们从缓存中检索键。等待 20 秒，然后执行同样的代码：

```py
>>> cache.get('musician')
None
```

`musician`缓存键已经过期，`get()`函数返回 None，因为键已经不再缓存中。

> 避免在缓存键中存储`None`，因为你不能区分实际值和缓存丢失。

让我们缓存一个 QuerySet：

```py
>>> from courses.models import Subject
>>> subjects = Subject.objects.all()
>>> cache.set('all_subjects', subjects)
```

我们在`Subject`模型上执行了一个 QuerySet，并在`all_subjects`键中存储返回的对象。让我们检索缓存的数据：

```py
>>> cache.get('all_subjects')
[<Subject: Mathematics>, <Subject: Music>, <Subject: Physics>, <Subject: Programming>]
```

我们将在视图中缓存一些查询。编辑`courses`应用的`views.py`文件，添加以下导入：

```py
from django.core.cache import cache
```

在`CourseListView`的`get()`方法中，找到这一行代码：

```py
subjects = Subject.objects.annotate(
	total_courses=Count('courses')
)
```

替换为以下代码：

```py
subjects = cache.get('all_subjects')
if not subjects:
    subjects = Subject.objects.annotate(
        total_courses=Count('courses')
    )
    cache.set('all_subjects', subjects)
```

在这段代码中，我们首先尝试使用`cache.get()`从缓存获得`all_subjects`键。如果没有找到给定的键则返回 None。如果没有找到键（还没有缓存，或者缓存了，但是超时了），我们执行查询检索所有`Subject`对象和它们的课程数量，然后使用`cache.set()`缓存结果。

启动开发服务器，并在浏览器中打开`http://127.0.0.1:8000/`。当执行视图时，没有找到缓存键，并且会执行 QuerySet。在浏览器中打开`http://127.0.0.1:8000/admin/`，并展开 Memcached 统计。你会看到缓存使用数据，如下图所示：

![](http://ooyedgh9k.bkt.clouddn.com/%E5%9B%BE11.7.png)

看一眼`Curr Items`，现在是 1。它表示当前缓存中存储了一条记录。`Get Hits`表示成功执行了多少次`get`命令，`Get Misses`表示键的多少次`get`请求丢失了。`Miss Ratio`是这两项值计算出来的结果。

现在回到`http://127.0.0.1:8000/`，然后重新加载几次页面。如果你现在看一眼缓存统计，会看到更多的读取（`Get Hits`和`Cmd Get`增加了）。

#### 11.4.6.1 基于动态数据缓存

很多时候，你会想要基于动态数据缓存一些东西。这种情况下，你需要动态构建包含所有信息的键，来唯一识别缓存的数据。编辑`courses`应用的`views.py`文件，并修改`CourseListView`视图：

```py
class CourseListView(TemplateResponseMixin, View):
    model = Course
    template_name = 'courses/course/list.html'

    def get(self, request, subject=None):
        subjects = cache.get('all_subjects')
        if not subjects:
            subjects = Subject.objects.annotate(
                total_courses=Count('courses')
            )
            cache.set('all_subjects', subjects)
        all_courses = Course.objects.annotate(
            total_modules=Count('modules')
        )
        if subject:
            subject = get_object_or_404(Subject, slug=subject)
            key = 'subject_{}_courses'.format(subject.id)
            courses = cache.get(key)
            if not courses:
                courses = all_courses.filter(subject=subject)
                cache.set(key, courses)
        else:
            courses = cache.get('all_courses')
            if not courses:
                courses = all_courses
                cache.set('all_courses', courses)
        return self.render_to_response({
            'subjects': subjects,
            'subject': subject,
            'courses': courses
        })
```
这段代码中，我们缓存了所有课程和过滤主题的课程。如果没有给定主题，则使用`all_courses`缓存键存储所有课程。如果给定了主题了，则使用`'subject_{}_courses'.format(subject.id)`动态构建键。

请注意，我们不能使用缓存的 QuerySet 来构建另一个 QuerySet，因为我们缓存的是 QuerySet 的实际结果。所以我们不能这么做：

```py
courses = cache.get('all_courses')
courses.filter(subject=subject)
```

相反，我们必须创建一个基本的 QuerySet：`Course.objects.annotate(total_modules=Count('modules'))`，它在你强制执行之前不会执行。当在缓存中没有找到数据时，使用`all_courses.filter(subject=subject)`进一步限制 QuerySet。

### 11.4.7 缓存模板片段

缓存模板片段是一个更高级别的方法。你需要在模板中使用`{% load cache %}`加载缓存模板标签。然后你才可以使用`{% cache %}`模板标签缓存指定的模板片段。通常你会这样使用模板标签：

```py
{% cache 300 fragment_name %}
	...
{% endcache %}
```

`{% cache %}`有两个必需的参数：超时时间（单位秒）和片段的名称。如果你需要缓存基于动态数据的内容，你可以传递额外的参数给`{% cache %}`模板标签，来唯一识别片段。

编辑`students`应用的`/students/course/detail.html`模板。在`{% extend %}`标签之后添加以下代码：

```py
{% load cache %}
```

然后找到以下代码：

```py
{% for content in module.contents.all %}
    {% with item=content.item %}
        <h2>{{ item.title }}</h2>
        {{ item.render }}
    {% endwith %}
{% endfor %}
```

替换为下面的代码：

```py
{% cache 600 module_contents module %}
    {% for content in module.contents.all %}
        {% with item=content.item %}
            <h2>{{ item.title }}</h2>
            {{ item.render }}
        {% endwith %}
    {% endfor %}
{% endcache %}
```

我们使用`module_contents`名字缓存这个模板片段，并把当前的`Module`对象传递给它。因此，我们可以唯一识别这个片段。当请求不同的单元时，这对于避免提供错误的内容很重要。

> 如果`USE_I18N`设置为`True`，那么整个网站中间件缓存会遵循激活的语言。如果你使用`{% cache %}`模板标签，则可以使用模板中可用的某个转换特定变量来实现同样的结果，比如`{% cache 600 name request.LANGUAGE_CODE %}`。

### 11.4.8 缓存视图

你可以使用`cache_page`装饰器缓存单个视图的输出，它位于`django.views.decorators.cache`中。该装饰器需要一个超时参数（单位秒）。

让我们在视图中使用它。编辑`students`应用的`urls.py`文件，添加以下导入：

```py
from django.views.decorators.cache import cache_page
```

然后在`student_course_detail`和`student_course_detail_module`模式上应用`cache_page`装饰器，如下所示：

```py
url(r'^course/(?P<pk>\d+)/$', 
    cache_page(60 * 15)(views.StudentCourseDetailView.as_view()), 
    name='student_course_detail'),
url(r'^course/(?P<pk>\d+)/(?P<module_id>\d+)/$', 
    cache_page(60 * 15)(views.StudentCourseDetailView.as_view()), 
    name='student_course_detail_module'),
```

现在`StudentCourseDetailView`的结果会缓存 15 分钟。

> 单个视图缓存使用 URL 构建缓存键。多个指向同一个视图的 URL 会分别缓存。

#### 11.4.81 使用整个站点缓存

这是最高级别的缓存。它允许你缓存整个站点。

编辑项目的`settings.py`文件，在`MIDDLEWARE`设置中添加`UpdateCacheMiddleware`和`FetchFromCacheMiddleware`类，来启用整个站点缓存：

```py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.cache.UpdateCacheMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.cache.FetchFromCacheMiddleware',
    # ...
]
```

记住，在解析请求的过程中，中间件按给定的顺序执行；而在解析响应的过程中，则是逆序执行。`UpdateCacheMiddleware`放在`CommonMiddleware`之前是因为它在响应的时候执行，此时中间件是逆序执行的。`FetchFromCacheMiddleware`特意放在`CommonMiddleware`之后，是因为它需要访问后者设置的请求数据。

然后，在`settings.py`文件中添加以下设置：

```py
CACHE_MIDDLEWARE_ALIAS = 'default'
CACEH_MIDDLEWARE_SECONDS = 60 * 15 # 15 minutes
CACHE_MIDDLEWARE_KEY_PRIFIX = 'educa'
```

在这些设置中，我们为缓存中间件使用默认缓存，并设置全局缓存超时为 15 分钟。我们还为所有缓存键指定一个前缀，来避免在多个项目中使用同一个 Memcached 后台时的冲突。现在我们的网站会缓存数据，并为所有 GET 请求返回缓存的数据。

我们已经测试了整个站点缓存功能。但是，整个站点缓存不适合我们，因为课程管理视图需要立即显示更新的数据。我们项目中最好的方式是缓存模板，或者用于显示课程内容的视图。

我们已经学习了 Django 提供的缓存数据的方法。你应该明智的定义缓存策略，优先考虑开销最大的 QuerySet 或者计算。

## 11.5 总结

在这章中，我们为课程创建了公开的视图，并构建了一个学生注册和报名课程的系统。我们安装了 Memcached，并实现了不同的缓存级别。

下一章我们会为项目构建 RESTful API。