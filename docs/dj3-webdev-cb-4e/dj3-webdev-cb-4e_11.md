# 第十一章：测试

在本章中，我们将涵盖以下主题：

+   使用 mock 测试视图

+   使用 Selenium 测试用户界面

+   使用 Django REST 框架创建 API 的测试

+   确保测试覆盖率

# 介绍

为了确保代码的质量和正确性，您应该进行自动化软件测试。 Django 为您提供了编写网站测试套件的工具。 测试套件会自动检查您的网站及其组件，以确保一切正常运行。 当您修改代码时，可以运行测试以检查您的更改是否对应用程序的行为产生了负面影响。

自动化软件测试领域有各种划分和术语。 为了本书的目的，我们将测试划分为以下类别：

+   **单元测试**指的是严格针对代码的单个部分或单元的测试。 最常见的情况是，一个单元对应于单个文件或模块，单元测试会尽力验证逻辑和行为是否符合预期。

+   **集成测试**进一步进行，处理两个或多个单元彼此协作的方式。 这种测试不像单元测试那样细粒度，并且通常是在假设所有单元测试都已通过的情况下编写的。 因此，集成测试仅涵盖了必须对单元正确地彼此协作的行为集。

+   **组件接口测试**是集成测试的一种高阶形式，其中单个组件从头到尾进行验证。 这种测试以一种对提供组件行为的基础逻辑无知的方式编写，因此逻辑可以更改而不修改行为，测试仍将通过。

+   系统测试验证了构成系统的所有组件的端到端集成，通常对应于完整的用户流程。

+   **操作接受测试**检查系统的所有非功能方面是否正常运行。 验收测试检查业务逻辑，以找出项目是否按照最终用户的观点正常工作。

# 技术要求

要使用本章中的代码，您需要最新稳定版本的 Python，一个 MySQL 或 PostgreSQL 数据库，以及一个带有虚拟环境的 Django 项目。

您可以在 GitHub 存储库的`ch11`目录中找到本章的所有代码：[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 使用 mock 测试视图

在本示例中，我们将看看如何编写单元测试。 单元测试检查单个函数或方法是否返回正确的结果。 我们将查看`likes`应用程序，并编写测试，检查对`json_set_like()`视图的发布是否对未经身份验证的用户返回失败响应，并对经过身份验证的用户返回成功结果。 我们将使用`Mock`对象来模拟`HttpRequest`和`AnonymousUser`对象。

# 准备工作

让我们从*在第四章*的*实现点赞小部件*食谱中的`locations`和`likes`应用程序开始。

我们将使用`mock`库，自 Python 3.3 以来一直作为内置的`unittest.mock`可用。

# 如何操作...

我们将通过以下步骤使用`mock`测试点赞操作：

1.  在`likes`应用中创建`tests`模块

1.  在本模块中，创建一个名为`test_views.py`的文件，内容如下：

```py
# myproject/apps/likes/tests/test_views.py
import json
from unittest import mock
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase
from myproject.apps.locations.models import Location

class JSSetLikeViewTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super(JSSetLikeViewTest, cls).setUpClass()

        cls.location = Location.objects.create(
            name="Park Güell",
            description="If you want to see something spectacular, 
            come to Barcelona, Catalonia, Spain and visit Park 
            Güell. Located on a hill, Park Güell is a public 
            park with beautiful gardens and organic 
            architectural elements.",
            picture="locations/2020/01/20200101012345.jpg",  
            # dummy path
        )
        cls.content_type = 
         ContentType.objects.get_for_model(Location)
        cls.superuser = User.objects.create_superuser(
            username="admin", password="admin", 
             email="admin@example.com"
        )

    @classmethod
    def tearDownClass(cls):
        super(JSSetLikeViewTest, cls).tearDownClass()
        cls.location.delete()
        cls.superuser.delete()

    def test_authenticated_json_set_like(self):
        from ..views import json_set_like

        mock_request = mock.Mock()
        mock_request.user = self.superuser
        mock_request.method = "POST"

        response = json_set_like(mock_request, 
         self.content_type.pk, self.location.pk)
        expected_result = json.dumps(
            {"success": True, "action": "add", "count": 
             Location.objects.count()}
        )
        self.assertJSONEqual(response.content, expected_result)

    @mock.patch("django.contrib.auth.models.User")
    def test_anonymous_json_set_like(self, MockUser):
        from ..views import json_set_like

        anonymous_user = MockUser()
        anonymous_user.is_authenticated = False

        mock_request = mock.Mock()
        mock_request.user = anonymous_user
        mock_request.method = "POST"

        response = json_set_like(mock_request, 
        self.content_type.pk, self.location.pk)
        expected_result = json.dumps({"success": False})
        self.assertJSONEqual(response.content, expected_result)
```

1.  运行`likes`应用的测试，如下所示：

```py
(env)$ python manage.py test myproject.apps.likes --settings=myproject.settings.test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
..
----------------------------------------------------------------------
Ran 2 tests in 0.268s
OK
Destroying test database for alias 'default'...
```

# 工作原理...

当您运行`likes`应用的测试时，首先会创建一个临时测试数据库。然后，会调用`setUpClass()`方法。稍后，将执行以`test`开头的方法，最后会调用`tearDownClass()`方法。对于每个通过的测试，您将在命令行工具中看到一个点（.），对于每个失败的测试，将会有一个字母 F，对于测试中的每个错误，您将看到字母 E。最后，您将看到有关失败和错误测试的提示。因为我们目前在`likes`应用的套件中只有两个测试，所以您将在结果中看到两个点。

在`setUpClass()`中，我们创建一个位置和一个超级用户。此外，我们找出`Location`模型的`ContentType`对象。我们将需要它用于`json_set_like()`视图，该视图为不同对象设置或移除喜欢。作为提醒，该视图看起来类似于以下内容，并返回一个 JSON 字符串作为结果：

```py
def json_set_like(request, content_type_id, object_id):
    # all the view logic goes here…
    return JsonResponse(result)
```

在`test_authenticated_json_set_like()`和`test_anonymous_json_set_like()`方法中，我们使用`Mock`对象。这些对象可以具有任何属性或方法。`Mock`对象的每个未定义属性或方法都是另一个`Mock`对象。因此，在 shell 中，您可以尝试链接属性，如下所示：

```py
>>> from unittest import mock
>>> m = mock.Mock()
>>> m.whatever.anything().whatsoever
<Mock name='mock.whatever.anything().whatsoever' id='4320988368'>
```

在我们的测试中，我们使用`Mock`对象来模拟`HttpRequest`对象。对于匿名用户，`MockUser`被生成为标准 Django `User`对象的一个补丁，通过`@mock.patch()`装饰器。对于经过身份验证的用户，我们仍然需要真实的`User`对象，因为视图使用用户的 ID 来获取`Like`对象。

因此，我们调用`json_set_like()`函数，并检查返回的 JSON 响应是否正确：

+   如果访问者未经身份验证，则响应中返回`{"success": false}`

+   对于经过身份验证的用户，它返回类似`{"action": "add", "count": 1, "success": true}`的内容

最后，调用`tearDownClass()`类方法，从测试数据库中删除位置和超级用户。

# 还有更多...

要测试使用`HttpRequest`对象的内容，您还可以使用 Django 请求工厂。您可以在[`docs.djangoproject.com/en/3.0/topics/testing/advanced/#the-request-factory`](https://docs.djangoproject.com/en/3.0/topics/testing/advanced/#the-request-factory)上阅读如何使用它。

# 另请参阅

+   在第四章*，模板和 JavaScript*中的*实现“喜欢”小部件*食谱中

+   *使用 Selenium 测试用户界面*食谱

+   *使用 Django REST 框架创建 API 的测试*食谱

+   *确保测试覆盖*食谱

# 使用 Selenium 测试用户界面

**操作接受测试**检查业务逻辑，以了解项目是否按预期工作。在这个食谱中，您将学习如何使用**Selenium**编写接受测试，它允许您模拟前端的活动，如填写表单或在浏览器中单击特定的 DOM 元素。

# 准备工作

让我们从第四章*，模板和 JavaScript*中的*实现“喜欢”小部件*食谱中的`locations`和`likes`应用开始。

对于这个食谱，我们将使用 Selenium 库与**Chrome**浏览器和**ChromeDriver**来控制它。让我们准备一下：

1.  从[`www.google.com/chrome/`](https://www.google.com/chrome/)下载并安装 Chrome 浏览器。

1.  在 Django 项目中创建一个`drivers`目录。从[`sites.google.com/a/chromium.org/chromedriver/`](https://sites.google.com/a/chromium.org/chromedriver/)下载 ChromeDriver 的最新稳定版本，解压缩并将其放入新创建的`drivers`目录中。

1.  在虚拟环境中安装 Selenium，如下所示：

```py
(env)$ pip install selenium
```

# 如何做...

我们将通过 Selenium 测试基于 Ajax 的点赞功能，执行以下步骤：

1.  在项目设置中，添加一个`TESTS_SHOW_BROWSER`设置：

```py
# myproject/settings/_base.py
TESTS_SHOW_BROWSER = True
```

1.  在您的`locations`应用中创建`tests`模块，并在其中添加一个`test_frontend.py`文件，内容如下：

```py
# myproject/apps/locations/tests/test_frontend.py
import os
from io import BytesIO
from time import sleep

from django.core.files.storage import default_storage
from django.test import LiveServerTestCase
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User
from django.conf import settings
from django.test import override_settings
from django.urls import reverse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from myproject.apps.likes.models import Like
from ..models import Location

SHOW_BROWSER = getattr(settings, "TESTS_SHOW_BROWSER", False)

@override_settings(DEBUG=True)
class LiveLocationTest(LiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super(LiveLocationTest, cls).setUpClass()
        driver_path = os.path.join(settings.BASE_DIR, "drivers", 
        "chromedriver")
        chrome_options = Options()
        if not SHOW_BROWSER:
 chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1200,800")

        cls.browser = webdriver.Chrome(
            executable_path=driver_path, options=chrome_options
        )
        cls.browser.delete_all_cookies()

        image_path = cls.save_test_image("test.jpg")
        cls.location = Location.objects.create(
            name="Park Güell",
            description="If you want to see something spectacular, 
             come to Barcelona, Catalonia, Spain and visit Park 
             Güell. Located on a hill, Park Güell is a public 
             park with beautiful gardens and organic 
             architectural elements.",
            picture=image_path,  # dummy path
        )
        cls.username = "admin"
        cls.password = "admin"
        cls.superuser = User.objects.create_superuser(
            username=cls.username, password=cls.password, 
             email="admin@example.com"
        )

    @classmethod
    def tearDownClass(cls):
        super(LiveLocationTest, cls).tearDownClass()
        cls.browser.quit()
        cls.location.delete()
        cls.superuser.delete()

    @classmethod
    def save_test_image(cls, filename):
        from PIL import Image

        image = Image.new("RGB", (1, 1), 0)
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG")
        path = f"tests/{filename}"
        default_storage.save(path, image_buffer)
        return path

    def wait_a_little(self):
        if SHOW_BROWSER:
 sleep(2)

    def test_login_and_like(self):
        # login
        login_path = reverse("admin:login")
        self.browser.get(
            f"{self.live_server_url}{login_path}?next=
          {self.location.get_url_path()}"
        )
        username_field = 
        self.browser.find_element_by_id("id_username")
        username_field.send_keys(self.username)
        password_field = 
        self.browser.find_element_by_id("id_password")
        password_field.send_keys(self.password)
        self.browser.find_element_by_css_selector
        ('input[type="submit"]').click()
        WebDriverWait(self.browser, timeout=10).until(
            lambda x: 
       self.browser.find_element_by_css_selector(".like-button")
        )
        # click on the "like" button
        like_button = 
       self.browser.find_element_by_css_selector(".like-button")
        is_initially_active = "active" in 
         like_button.get_attribute("class")
        initial_likes = int(
            self.browser.find_element_by_css_selector
             (".like-badge").text
        )

        self.assertFalse(is_initially_active)
        self.assertEqual(initial_likes, 0)

        self.wait_a_little()

        like_button.click()
        WebDriverWait(self.browser, timeout=10).until(
            lambda x:  
            int(self.browser.find_element_by_css_selector
             (".like-badge").text) != initial_likes
        )
        likes_in_html = int(
            self.browser.find_element_by_css_selector
             (".like-badge").text
        )
        likes_in_db = Like.objects.filter(

       content_type=ContentType.objects.get_for_model(Location),
            object_id=self.location.pk,
        ).count()
        self.assertEqual(likes_in_html, 1)
        self.assertEqual(likes_in_html, likes_in_db)

        self.wait_a_little()

        self.assertGreater(likes_in_html, initial_likes)

        # click on the "like" button again to switch back to the 
        # previous state
        like_button.click()
        WebDriverWait(self.browser, timeout=10).until(
            lambda x: int(self.browser.find_element_by_css_selector
            (".like-badge").text) == initial_likes
        )

        self.wait_a_little()
```

1.  运行`locations`应用的测试，如下所示：

```py
(env)$ python manage.py test myproject.apps.locations --settings=myproject.settings.test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
.
----------------------------------------------------------------------
Ran 1 test in 4.284s

OK
Destroying test database for alias 'default'...
```

# 它是如何工作的...

当我们运行这些测试时，我们将看到一个 Chrome 窗口打开，显示管理登录屏幕的 URL，例如

`http://localhost:63807/en/admin/login/?next=/en/locations/176255a9-9c07-4542-8324-83ac0d21b7c3/`。

用户名和密码字段将填写为 admin，然后您将被重定向到 Park Güell 位置的详细页面，URL 如下

`http://localhost:63807/en/locations/176255a9-9c07-4542-8324-83ac0d21b7c3/`。在那里，您将看到点赞按钮被点击两次，导致点赞和取消点赞操作。

如果我们将`TESTS_SHOW_BROWSER`设置为`False`（或将其全部删除）并再次运行测试，测试将以最小的等待时间在后台进行，而不会打开浏览器窗口。

让我们看看这在测试套件中是如何工作的。我们定义一个扩展`LiveServerTestCase`的类。这将创建一个测试套件，该测试套件将在一个随机未使用的端口（例如`63807`）下运行一个本地服务器。默认情况下，`LiveServerTestCase`以非 DEBUG 模式运行服务器。但是，我们使用`override_settings()`装饰器将其切换到 DEBUG 模式，以便使静态文件可访问而无需收集它们，并在任何页面上发生错误时显示错误回溯。`setUpClass()`类方法将在所有测试开始时执行，`tearDownClass()`类方法将在测试运行后执行。在中间，测试将执行所有以`test`开头的套件方法。

当我们开始测试时，会创建一个新的测试数据库。在`setUpClass()`中，我们创建一个浏览器对象，一个位置和一个超级用户。然后，执行`test_login_and_like()`方法，该方法打开管理登录页面，找到用户名字段，输入管理员的用户名，找到密码字段，输入管理员的密码，找到提交按钮，并点击它。然后，它等待最多 10 秒，直到页面上可以找到具有`.like-button` CSS 类的 DOM 元素。

正如您可能记得的*在第四章*中实现点赞小部件的教程，模板和 JavaScript，我们的小部件由两个元素组成：

+   一个点赞按钮

+   显示点赞总数的徽章

如果点击按钮，您的`Like`实例将通过 Ajax 调用添加或从数据库中删除。此外，徽章计数将更新以反映数据库中的点赞数。

在测试中，我们检查按钮的初始状态（是否具有`.active` CSS 类），检查初始点赞数，并模拟点击按钮。我们等待最多 10 秒，直到徽章中的计数发生变化。然后，我们检查徽章中的计数是否与数据库中位置的总点赞数匹配。我们还将检查徽章中的计数如何发生变化（增加）。最后，我们将再次模拟点击按钮，以切换回先前的状态。

最后，调用`tearDownClass()`方法，关闭浏览器并从测试数据库中删除位置和超级用户。

# 另请参阅

+   *在第四章*中实现点赞小部件的教程，模板和 JavaScript

+   *使用模拟测试视图*教程

+   *使用 Django REST 框架创建 API 的测试*教程

+   *确保测试覆盖率*教程

# 使用 Django REST 框架创建的 API 的测试

您应该已经了解如何编写单元测试和操作接受测试。在这个教程中，我们将介绍**RESTful API 的组件接口测试**，这是我们在本书中早些时候创建的。

如果您不熟悉 RESTful API 是什么以及 API 的用途，您可以在[`www.restapitutorial.com/`](http://www.restapitutorial.com/)上了解更多。

# 准备工作

让我们从第九章*中的*使用 Django REST 框架创建 API*配方中的`music`应用开始。

# 操作步骤...

要测试 RESTful API，请执行以下步骤：

1.  在`music`应用中创建一个`tests`模块。在`tests`模块中，创建一个名为`test_api.py`的文件，并创建`SongTests`类。该类将具有`setUpClass()`和`tearDownClass()`方法，如下所示：

```py
# myproject/apps/music/tests/test_api.py
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from ..models import Song

class SongTests(APITestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.superuser = User.objects.create_superuser(
            username="admin", password="admin", 
             email="admin@example.com"
        )

        cls.song = Song.objects.create(
            artist="Lana Del Rey",
            title="Video Games - Remastered",
            url="https://open.spotify.com/track/5UOo694cVvj
             cPFqLFiNWGU?si=maZ7JCJ7Rb6WzESLXg1Gdw",
        )

        cls.song_to_delete = Song.objects.create(
            artist="Milky Chance",
            title="Stolen Dance",
            url="https://open.spotify.com/track/3miMZ2IlJ
             iaeSWo1DohXlN?si=g-xMM4m9S_yScOm02C2MLQ",
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        cls.song.delete()
        cls.superuser.delete()
```

1.  添加一个 API 测试，检查列出歌曲：

```py
    def test_list_songs(self):
        url = reverse("rest_song_list")
        data = {}
        response = self.client.get(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], Song.objects.count())
```

1.  添加一个 API 测试，检查单个歌曲的详细信息：

```py
    def test_get_song(self):
        url = reverse("rest_song_detail", kwargs={"pk": self.song.pk})
        data = {}
        response = self.client.get(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["uuid"], str(self.song.pk))
        self.assertEqual(response.data["artist"], self.song.artist)
        self.assertEqual(response.data["title"], self.song.title)
        self.assertEqual(response.data["url"], self.song.url)
```

1.  添加一个 API 测试，检查成功创建新歌曲：

```py
 def test_create_song_allowed(self):
        # login
        self.client.force_authenticate(user=self.superuser)

        url = reverse("rest_song_list")
        data = {
            "artist": "Capital Cities",
            "title": "Safe And Sound",
            "url": "https://open.spotify.com/track/40Fs0YrUGu
              wLNQSaHGVfqT?si=2OUawusIT-evyZKonT5GgQ",
        }
        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, 
         status.HTTP_201_CREATED)

        song = Song.objects.filter(pk=response.data["uuid"])
        self.assertEqual(song.count(), 1)

        # logout
        self.client.force_authenticate(user=None)
```

1.  添加一个尝试在没有身份验证的情况下创建歌曲并因此失败的测试：

```py
 def test_create_song_restricted(self):
        # make sure the user is logged out
        self.client.force_authenticate(user=None)

        url = reverse("rest_song_list")
        data = {
            "artist": "Men I Trust",
            "title": "Tailwhip",
            "url": "https://open.spotify.com/track/2DoO0sn4S
              bUrz7Uay9ACTM?si=SC_MixNKSnuxNvQMf3yBBg",
        }
        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, 
         status.HTTP_403_FORBIDDEN)
```

1.  添加一个检查成功更改歌曲的测试：

```py
def test_change_song_allowed(self):
        # login
        self.client.force_authenticate(user=self.superuser)

        url = reverse("rest_song_detail", kwargs=
         {"pk": self.song.pk})

        # change only title
        data = {
            "artist": "Men I Trust",
            "title": "Tailwhip",
            "url": "https://open.spotify.com/track/2DoO0sn4S
              bUrz7Uay9ACTM?si=SC_MixNKSnuxNvQMf3yBBg",
        }
        response = self.client.put(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["uuid"], str(self.song.pk))
        self.assertEqual(response.data["artist"], data["artist"])
        self.assertEqual(response.data["title"], data["title"])
        self.assertEqual(response.data["url"], data["url"])

        # logout
        self.client.force_authenticate(user=None)
```

1.  添加一个检查由于缺少身份验证而导致更改失败的测试：

```py
def test_change_song_restricted(self):
        # make sure the user is logged out
        self.client.force_authenticate(user=None)

        url = reverse("rest_song_detail", kwargs=
         {"pk": self.song.pk})

        # change only title
        data = {
            "artist": "Capital Cities",
            "title": "Safe And Sound",
            "url": "https://open.spotify.com/track/40Fs0YrU
             GuwLNQSaHGVfqT?si=2OUawusIT-evyZKonT5GgQ",
        }
        response = self.client.put(url, data, format="json")

        self.assertEqual(response.status_code, 
         status.HTTP_403_FORBIDDEN)
```

1.  添加一个检查歌曲删除失败的测试：

```py
    def test_delete_song_restricted(self):
        # make sure the user is logged out
        self.client.force_authenticate(user=None)

        url = reverse("rest_song_detail", kwargs=
         {"pk": self.song_to_delete.pk})

        data = {}
        response = self.client.delete(url, data, format="json")

        self.assertEqual(response.status_code, 
         status.HTTP_403_FORBIDDEN)
```

1.  添加一个检查成功删除歌曲的测试：

```py
  def test_delete_song_allowed(self):
        # login
        self.client.force_authenticate(user=self.superuser)

        url = reverse("rest_song_detail", kwargs=
         {"pk": self.song_to_delete.pk})

        data = {}
        response = self.client.delete(url, data, format="json")

        self.assertEqual(response.status_code, 
         status.HTTP_204_NO_CONTENT)

        # logout
        self.client.force_authenticate(user=None)
```

1.  运行`music`应用的测试，如下所示：

```py
(env)$python manage.py test myproject.apps.music --settings=myproject.settings.test
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
........
----------------------------------------------------------------------
Ran 8 tests in 0.370s

OK
Destroying test database for alias 'default'...
```

# 它是如何工作的...

这个 RESTful API 测试套件扩展了`APITestCase`类。再次，我们有`setUpClass()`和`tearDownClass()`类方法，它们将在不同测试之前和之后执行。此外，测试套件具有`APIClient`类型的 client 属性，可用于模拟 API 调用。客户端提供所有标准 HTTP 调用的方法：`get()`，`post()`，`put()`，`patch()`，`delete()`，`head()`和`options()`。

在我们的测试中，我们使用`GET`，`POST`和`DELETE`请求。此外，客户端还具有根据登录凭据、令牌或`User`对象强制对用户进行身份验证的方法。在我们的测试中，我们正在进行第三种身份验证：直接将用户传递给`force_authenticate()`方法。

代码的其余部分是不言自明的。

# 另请参阅

+   第九章*中的*使用 Django REST 框架创建 API*配方，导入和导出数据

+   *使用模拟测试视图*配方

+   *使用 Selenium 测试用户界面*配方

+   *确保测试覆盖率*配方

# 确保测试覆盖率

Django 允许快速原型设计和从想法到实现的项目构建。但是，为了确保项目稳定且可用于生产，您应该尽可能多地对功能进行测试。通过测试覆盖率，您可以检查项目代码的测试覆盖率。让我们看看您可以如何做到这一点。

# 准备工作

为您的项目准备一些测试。

在您的虚拟环境中安装`coverage`实用程序：

```py
(env)$ pip install coverage~=5.0.1
```

# 操作步骤...

这是如何检查项目的测试覆盖率的：

1.  为覆盖率实用程序创建一个名为`setup.cfg`的配置文件，内容如下：

```py
# setup.cfg
[coverage:run]
source = .
omit =
    media/*
    static/*
    tmp/*
    drivers/*
    locale/*
    myproject/site_static/*
    myprojext/templates/*
```

1.  如果您使用 Git 版本控制，请确保在`.gitignore`文件中有这些行：

```py
# .gitignore
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
```

1.  创建一个名为`run_tests_with_coverage.sh`的 shell 脚本，其中包含运行测试并报告结果的命令：

```py
# run_tests_with_coverage.sh
#!/usr/bin/env bash
coverage erase
coverage run manage.py test --settings=myproject.settings.test
coverage report
```

1.  为该脚本添加执行权限：

```py
(env)$ chmod +x run_tests_with_coverage.sh
```

1.  运行脚本：

```py
(env)$ ./run_tests_with_coverage.sh 
Creating test database for alias 'default'...
System check identified no issues (0 silenced).
...........
----------------------------------------------------------------------
Ran 11 tests in 12.940s

OK
Destroying test database for alias 'default'...
Name Stmts Miss Cover
-----------------------------------------------------------------------------------------------
manage.py 12 2 83%
myproject/__init__.py 0 0 100%
myproject/apps/__init__.py 0 0 100%
myproject/apps/core/__init__.py 0 0 100%
myproject/apps/core/admin.py 16 10 38%
myproject/apps/core/context_processors.py 3 0 100%
myproject/apps/core/model_fields.py 48 48 0%
myproject/apps/core/models.py 87 29 67%
myproject/apps/core/templatetags/__init__.py 0 0 100%
myproject/apps/core/templatetags/utility_tags.py 171 135 21%

the statistics go on…

myproject/settings/test.py 5 0 100%
myproject/urls.py 10 0 100%
myproject/wsgi.py 4 4 0%
-----------------------------------------------------------------------------------------------
TOTAL 1363 712 48%
```

# 它是如何工作的...

覆盖率实用程序运行测试并检查有多少行代码被测试覆盖。在我们的示例中，我们编写的测试覆盖了 48%的代码。如果项目稳定性对您很重要，那么在有时间的时候，尽量接近 100%。

在覆盖配置中，我们跳过了静态资产、模板和其他非 Python 文件。

# 另请参阅

+   *使用模拟测试视图*配方

+   *使用 Selenium 测试用户界面*配方

+   *使用 Django REST 框架创建的 API 进行测试*配方
