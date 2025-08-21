# 第十九章：Django 中的安全性

确保您构建的网站是安全的对于专业的 Web 应用程序开发人员至关重要。

Django 框架现在非常成熟，大多数常见的安全问题都以某种方式得到了解决，但是没有安全措施是 100%保证的，而且新的威胁不断出现，因此作为 Web 开发人员，您需要确保您的网站和应用程序是安全的。

Web 安全是一个庞大的主题，无法在一本书的章节中深入讨论。本章概述了 Django 的安全功能，并提供了有关保护 Django 网站的建议，这将在 99%的时间内保护您的网站，但您需要随时了解 Web 安全的变化。

有关 Web 安全的更详细信息，请参阅 Django 的安全问题存档（有关更多信息，请访问[`docs.djangoproject.com/en/1.8/releases/security/`](https://docs.djangoproject.com/en/1.8/releases/security/)），以及维基百科的 Web 应用程序安全页面（[`en.wikipedia.org/wiki/web_application_security`](https://en.wikipedia.org/wiki/web_application_security)）。

# Django 内置的安全功能

## 跨站点脚本攻击（XSS）保护

**跨站点脚本**（**XSS**）攻击允许用户向其他用户的浏览器注入客户端脚本。

这通常是通过将恶意脚本存储在数据库中，然后检索并显示给其他用户，或者让用户点击一个链接，从而导致攻击者的 JavaScript 在用户的浏览器中执行。但是，XSS 攻击可能源自任何不受信任的数据源，例如 cookie 或 Web 服务，只要在包含在页面中之前未经充分净化。

使用 Django 模板可以保护您免受大多数 XSS 攻击。但是，重要的是要了解它提供的保护措施及其局限性。

Django 模板会转义对 HTML 特别危险的特定字符。虽然这可以保护用户免受大多数恶意输入，但并非绝对安全。例如，它无法保护以下内容：

```py
<style class={{ var }}>...</style> 

```

如果`var`设置为`'class1 onmouseover=javascript:func()'`，这可能导致未经授权的 JavaScript 执行，具体取决于浏览器如何呈现不完美的 HTML。（引用属性值将修复此情况）。

在使用自定义模板标记时，使用`is_safe`、`safe`模板标记、`mark_safe`以及关闭`autoescape`时要特别小心。

此外，如果您使用模板系统输出除 HTML 之外的内容，可能需要转义完全不同的字符和单词。

在存储 HTML 在数据库时，特别需要非常小心，特别是当检索和显示该 HTML 时。

## 跨站点请求伪造（CSRF）保护

**跨站点请求伪造**（**CSRF**）攻击允许恶意用户在不知情或未经同意的情况下使用另一个用户的凭据执行操作。

Django 内置了对大多数 CSRF 攻击的保护，只要您已启用并在适当的地方使用它。但是，与任何缓解技术一样，存在局限性。

例如，可以全局禁用 CSRF 模块或特定视图。只有在知道自己在做什么时才应该这样做。如果您的网站具有超出您控制范围的子域，还存在其他限制。

CSRF 保护通过检查每个`POST`请求中的一次性令牌来实现。这确保了恶意用户无法简单地重放表单`POST`到您的网站，并使另一个已登录的用户无意中提交该表单。恶意用户必须知道一次性令牌，这是用户特定的（使用 cookie）。

在使用 HTTPS 部署时，`CsrfViewMiddleware`将检查 HTTP 引用头是否设置为同一来源的 URL（包括子域和端口）。因为 HTTPS 提供了额外的安全性，所以必须确保连接在可用时使用 HTTPS，通过转发不安全的连接请求并为受支持的浏览器使用 HSTS。

非常小心地标记视图为`csrf_exempt`装饰器，除非绝对必要。

Django 的 CSRF 中间件和模板标签提供了易于使用的跨站请求伪造保护。

对抗 CSRF 攻击的第一道防线是确保`GET`请求（以及其他“安全”方法，如 9.1.1 安全方法，HTTP 1.1，RFC 2616 中定义的方法（有关更多信息，请访问[`tools.ietf.org/html/rfc2616.html#section-9.1.1`](https://tools.ietf.org/html/rfc2616.html#section-9.1.1)）是无副作用的。然后，通过以下步骤保护通过“不安全”方法（如`POST`，`PUT`和`DELETE`）的请求。

### 如何使用它

要在视图中利用 CSRF 保护，请按照以下步骤进行操作：

1.  CSRF 中间件在`MIDDLEWARE_CLASSES`设置中默认激活。如果您覆盖该设置，请记住`'django.middleware.csrf.CsrfViewMiddleware'`应该在任何假设已处理 CSRF 攻击的视图中间件之前。

1.  如果您禁用了它，这是不推荐的，您可以在要保护的特定视图上使用`csrf_protect()`（见下文）。

1.  在任何使用`POST`表单的模板中，如果表单用于内部 URL，请在`<form>`元素内使用`csrf_token`标签，例如：

```py
        <form action="." method="post">{% csrf_token %} 

```

1.  不应该对目标外部 URL 的`POST`表单执行此操作，因为这会导致 CSRF 令牌泄漏，从而导致漏洞。

1.  在相应的视图函数中，确保使用了`'django.template.context_processors.csrf'`上下文处理器。通常，可以通过以下两种方式之一完成：

1.  使用`RequestContext`，它始终使用`'django.template.context_processors.csrf'`（无论在`TEMPLATES`设置中配置了哪些模板上下文处理器）。如果您使用通用视图或贡献应用程序，则已经涵盖了，因为这些应用程序始终在整个`RequestContext`中使用。

1.  手动导入并使用处理器生成 CSRF 令牌，并将其添加到模板上下文中。例如：

```py
        from django.shortcuts import render_to_response 
        from django.template.context_processors import csrf 

        def my_view(request): 
            c = {} 
            c.update(csrf(request)) 
            # ... view code here 
            return render_to_response("a_template.html", c) 

```

1.  您可能希望编写自己的`render_to_response()`包装器，以便为您处理此步骤。

### AJAX

虽然上述方法可以用于 AJAX POST 请求，但它有一些不便之处：您必须记住在每个 POST 请求中将 CSRF 令牌作为 POST 数据传递。因此，有一种替代方法：在每个`XMLHttpRequest`上，将自定义的`X-CSRFToken`标头设置为 CSRF 令牌的值。这通常更容易，因为许多 JavaScript 框架提供了允许在每个请求上设置标头的钩子。

首先，您必须获取 CSRF 令牌本身。令牌的推荐来源是`csrftoken` cookie，如果您已经按上述方式为视图启用了 CSRF 保护，它将被设置。

CSRF 令牌 cookie 默认名为`csrftoken`，但您可以通过`CSRF_COOKIE_NAME`设置控制 cookie 名称。

获取令牌很简单：

```py
// using jQuery 
function getCookie(name) { 
    var cookieValue = null; 
    if (document.cookie && document.cookie != '') { 
        var cookies = document.cookie.split(';'); 
        for (var i = 0; i < cookies.length; i++) { 
            var cookie = jQuery.trim(cookies[i]); 
            // Does this cookie string begin with the name we want? 
            if (cookie.substring(0, name.length + 1) == (name + '=')) { 
                cookieValue =  decodeURIComponent(cookie.substring(name.length + 1)); 
                break; 
            } 
        } 
    } 
    return cookieValue; 
} 
var csrftoken = getCookie('csrftoken'); 

```

通过使用 jQuery cookie 插件（[`plugins.jquery.com/cookie/`](http://plugins.jquery.com/cookie/)）来替换`getCookie`，可以简化上述代码：

```py
var csrftoken = $.cookie('csrftoken'); 

```

### 注意

CSRF 令牌也存在于 DOM 中，但仅当在模板中明确包含`csrf_token`时才会存在。cookie 包含规范令牌；`CsrfViewMiddleware`将优先使用 cookie 而不是 DOM 中的令牌。无论如何，如果 DOM 中存在令牌，则保证会有 cookie，因此应该使用 cookie！

### 注意

如果您的视图没有呈现包含`csrf_token`模板标签的模板，则 Django 可能不会设置 CSRF 令牌 cookie。这在动态添加表单到页面的情况下很常见。为了解决这种情况，Django 提供了一个视图装饰器，强制设置 cookie：`ensure_csrf_cookie()`。

最后，您将需要在 AJAX 请求中实际设置标头，同时使用 jQuery 1.5.1 及更高版本中的`settings.crossDomain`保护 CSRF 令牌，以防止发送到其他域：

```py
function csrfSafeMethod(method) { 
    // these HTTP methods do not require CSRF protection 
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method)); 
} 
$.ajaxSetup({ 
    beforeSend: function(xhr, settings) { 
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) { 
            xhr.setRequestHeader("X-CSRFToken", csrftoken); 
        } 
    } 
}); 

```

### 其他模板引擎

当使用与 Django 内置引擎不同的模板引擎时，您可以在确保它在模板上下文中可用后，在表单中手动设置令牌。

例如，在 Jinja2 模板语言中，您的表单可以包含以下内容：

```py
<div style="display:none"> 
    <input type="hidden" name="csrfmiddlewaretoken" value="{{ csrf_token }}"> 
</div> 

```

您可以使用类似于上面的 AJAX 代码的 JavaScript 来获取 CSRF 令牌的值。

### 装饰器方法

您可以使用`csrf_protect`装饰器，而不是将`CsrfViewMiddleware`作为一种全面的保护措施，该装饰器具有完全相同的功能，用于需要保护的特定视图。它必须同时用于在输出中插入 CSRF 令牌的视图和接受`POST`表单数据的视图。（这些通常是相同的视图函数，但并非总是如此）。

不建议单独使用装饰器，因为如果您忘记使用它，将会有安全漏洞。同时使用两者的“双重保险”策略是可以的，并且会产生最小的开销。

`django.views.decorators.csrf.csrf_protect(view)`

提供对视图的`CsrfViewMiddleware`保护的装饰器。

用法：

```py
from django.views.decorators.csrf import csrf_protect 
from django.shortcuts import render 

@csrf_protect 
def my_view(request): 
    c = {} 
    # ... 
    return render(request, "a_template.html", c) 

```

如果您正在使用基于类的视图，可以参考装饰基于类的视图。

### 被拒绝的请求

默认情况下，如果传入请求未通过`CsrfViewMiddleware`执行的检查，则向用户发送*403 Forbidden*响应。通常只有在存在真正的跨站请求伪造或由于编程错误，CSRF 令牌未包含在`POST`表单中时才会看到这种情况。

然而，错误页面并不是很友好，因此您可能希望为处理此条件提供自己的视图。要做到这一点，只需设置`CSRF_FAILURE_VIEW`设置。

#### 工作原理

CSRF 保护基于以下几点：

+   设置为随机值的 CSRF cookie（称为会话独立 nonce），其他站点将无法访问。

+   这个 cookie 是由`CsrfViewMiddleware`设置的。它是永久性的，但由于没有办法设置永不过期的 cookie，因此它会随着每次调用`django.middleware.csrf.get_token()`（内部用于检索 CSRF 令牌的函数）的响应一起发送。

+   所有传出的 POST 表单中都有一个名为*csrfmiddlewaretoken*的隐藏表单字段。该字段的值是 CSRF cookie 的值。

+   这部分是由模板标签完成的。

+   对于所有不使用 HTTP `GET`，`HEAD`，`OPTIONS`或`TRACE`的传入请求，必须存在 CSRF cookie，并且必须存在并正确的*csrfmiddlewaretoken*字段。如果没有，用户将收到 403 错误。

+   这个检查是由`CsrfViewMiddleware`完成的。

+   此外，对于 HTTPS 请求，`CsrfViewMiddleware`会进行严格的引用检查。这是必要的，以解决在 HTTPS 下使用会话独立 nonce 时可能发生的中间人攻击，因为（不幸的是）客户端接受了对 HTTPS 站点进行通信的“Set-Cookie”标头。 （在 HTTP 请求下不进行引用检查，因为在 HTTP 下，引用标头的存在不够可靠。）

这确保只有来自您网站的表单才能用于将数据`POST`回来。

它故意忽略`GET`请求（以及 RFC 2616 定义为“安全”的其他请求）。这些请求不应该具有任何潜在的危险副作用，因此使用`GET`请求的 CSRF 攻击应该是无害的。RFC 2616 将`POST`、`PUT`和`DELETE`定义为“不安全”，并假定所有其他方法都是不安全的，以获得最大的保护。

### 缓存

如果模板使用`csrf_token`模板标签（或以其他方式调用`get_token`函数），`CsrfViewMiddleware`将向响应添加一个 cookie 和一个`Vary: Cookie`标头。这意味着如果按照指示使用缓存中间件（`UpdateCacheMiddleware`在所有其他中间件之前），中间件将与缓存中间件协同工作。

然而，如果您在单个视图上使用缓存装饰器，CSRF 中间件还没有能够设置`Vary`标头或 CSRF cookie，响应将被缓存而没有任何一个。

在这种情况下，对于任何需要插入 CSRF 令牌的视图，您应该首先使用`django.views.decorators.csrf.csrf_protect()`装饰器：

```py
from django.views.decorators.cache import cache_page 
from django.views.decorators.csrf import csrf_protect 

@cache_page(60 * 15) 
@csrf_protect 
def my_view(request): 
    ... 

```

如果您正在使用基于类的视图，可以参考 Django 文档中的装饰基于类的视图（[`docs.djangoproject.com/en/1.8/topics/class-based-views/intro/#decorating-class-based-views`](https://docs.djangoproject.com/en/1.8/topics/class-based-views/intro/#decorating-class-based-views)）。

### 测试

由于需要在每个`POST`请求中发送 CSRF 令牌，`CsrfViewMiddleware`通常会对测试视图函数造成很大的阻碍。因此，Django 的测试 HTTP 客户端已经修改，以在请求上设置一个标志，从而放宽中间件和`csrf_protect`装饰器，使其不再拒绝请求。在其他方面（例如发送 cookie 等），它们的行为是相同的。

如果出于某种原因，您希望测试客户端执行 CSRF 检查，您可以创建一个强制执行 CSRF 检查的测试客户端实例：

```py
>>> from django.test import Client 
>>> csrf_client = Client(enforce_csrf_checks=True) 

```

### 限制

站点内的子域将能够在整个域上为客户端设置 cookie。通过设置 cookie 并使用相应的令牌，子域将能够规避 CSRF 保护。避免这种情况的唯一方法是确保子域由受信任的用户控制（或者至少无法设置 cookie）。

请注意，即使没有 CSRF，也存在其他漏洞，例如会话固定，这使得将子域分配给不受信任的方可能不是一个好主意，而且这些漏洞在当前浏览器中不能轻易修复。

### 边缘情况

某些视图可能具有不符合此处正常模式的特殊要求。在这些情况下，一些实用程序可能会有用。它们可能需要的场景在下一节中描述。

### 实用程序

下面的示例假定您正在使用基于函数的视图。如果您正在使用基于类的视图，可以参考 Django 文档中的装饰基于类的视图。

#### django.views.decorators.csrf.csrf_exempt(view)

大多数视图需要 CSRF 保护，但有一些不需要。与其禁用中间件并将`csrf_protect`应用于所有需要它的视图，不如启用中间件并使用`csrf_exempt()`。

这个装饰器标记一个视图被中间件保护豁免。示例：

```py
from django.views.decorators.csrf import csrf_exempt 
from django.http import HttpResponse 

@csrf_exempt 
def my_view(request): 
    return HttpResponse('Hello world') 

```

#### django.views.decorators.csrf.requires_csrf_token(view)

有些情况下，`CsrfViewMiddleware.process_view`可能在您的视图运行之前没有运行-例如 404 和 500 处理程序-但您仍然需要表单中的 CSRF 令牌。

通常，如果`CsrfViewMiddleware.process_view`或类似`csrf_protect`没有运行，`csrf_token`模板标签将无法工作。视图装饰器`requires_csrf_token`可用于确保模板标签正常工作。这个装饰器的工作方式类似于`csrf_protect`，但从不拒绝传入的请求。

示例：

```py
from django.views.decorators.csrf import requires_csrf_token 
from django.shortcuts import render 

@requires_csrf_token 
def my_view(request): 
    c = {} 
    # ... 
    return render(request, "a_template.html", c) 

```

还可能有一些未受保护的视图已经被`csrf_exempt`豁免，但仍需要包含 CSRF 令牌。在这些情况下，使用`csrf_exempt()`后跟`requires_csrf_token()`。（即`requires_csrf_token`应该是最内层的装饰器）。

最后一个例子是，当视图仅在一组条件下需要 CSRF 保护，并且在其余时间不得具有保护时。解决方案是对整个视图函数使用`csrf_exempt()`，并对其中需要保护的路径使用`csrf_protect()`。

例如：

```py
from django.views.decorators.csrf import csrf_exempt, csrf_protect 

@csrf_exempt 
def my_view(request): 

    @csrf_protect 
    def protected_path(request): 
        do_something() 

    if some_condition(): 
       return protected_path(request) 
    else: 
       do_something_else() 

```

#### django.views.decorators.csrf.ensure_csrf_cookie(view)

这个装饰器强制视图发送 CSRF cookie。如果页面通过 AJAX 进行 POST 请求，并且页面没有带有`csrf_token`的 HTML 表单，这将导致所需的 CSRF cookie 被发送。解决方案是在发送页面的视图上使用`ensure_csrf_cookie()`。

### 贡献和可重用应用程序

由于开发人员可以关闭`CsrfViewMiddleware`，因此贡献应用程序中的所有相关视图都使用`csrf_protect`装饰器来确保这些应用程序对 CSRF 的安全性。建议其他希望获得相同保障的可重用应用程序的开发人员也在其视图上使用`csrf_protect`装饰器。

### CSRF 设置

可以用一些设置来控制 Django 的 CSRF 行为：

+   `CSRF_COOKIE_AGE`

+   `CSRF_COOKIE_DOMAIN`

+   `CSRF_COOKIE_HTTPONLY`

+   `CSRF_COOKIE_NAME`

+   `CSRF_COOKIE_PATH`

+   `CSRF_COOKIE_SECURE`

+   `CSRF_FAILURE_VIEW`

有关这些设置的更多信息，请参见附录 D，*设置*。

## SOL 注入保护

SQL 注入是一种攻击类型，恶意用户能够在数据库上执行任意的 SQL 代码。这可能导致记录被删除或数据泄露。

通过使用 Django 的查询集，生成的 SQL 将由底层数据库驱动程序正确转义。但是，Django 还赋予开发人员编写原始查询或执行自定义 SQL 的权力。这些功能应该谨慎使用，并且您应该始终小心地正确转义用户可以控制的任何参数。此外，在使用`extra()`时应谨慎。

## 点击劫持保护

点击劫持是一种攻击类型，恶意站点在框架中包裹另一个站点。当恶意站点欺骗用户点击他们在隐藏框架或 iframe 中加载的另一个站点的隐藏元素时，就会发生这种类型的攻击。

Django 包含防止点击劫持的保护，即`X-Frame-Options 中间件`，在支持的浏览器中可以防止网站在框架内呈现。可以在每个视图的基础上禁用保护，或配置发送的确切标头值。

强烈建议对于任何不需要其页面被第三方站点包裹在框架中的站点，或者只需要允许站点的一小部分进行包裹的站点使用中间件。

### 点击劫持的一个例子

假设一个在线商店有一个页面，用户可以在其中点击“立即购买”来购买商品。用户选择一直保持登录以方便使用。攻击者站点可能在其自己的页面上创建一个“我喜欢小马”按钮，并以透明的`iframe`加载商店的页面，使得“立即购买”按钮被隐形地覆盖在“我喜欢小马”按钮上。如果用户访问攻击者的站点，点击“我喜欢小马”将导致无意中点击“立即购买”按钮，并无意中购买商品。

### 防止点击劫持

现代浏览器遵守 X-Frame-Options（有关更多信息，请访问 [`developer.mozilla.org/en/The_X-FRAME-OPTIONS_response_header`](https://developer.mozilla.org/en/The_X-FRAME-OPTIONS_response_header)）HTTP 头部，该头部指示资源是否允许在框架或 iframe 中加载。如果响应包含带有 `SAMEORIGIN` 值的头部，则浏览器只会在请求源自同一站点时才在框架中加载资源。如果头部设置为 `DENY`，则浏览器将阻止资源在框架中加载，无论哪个站点发出了请求。

Django 提供了一些简单的方法来在您的站点的响应中包含这个头部：

+   一个简单的中间件，可以在所有响应中设置头部。

+   一组视图装饰器，可用于覆盖中间件或仅为特定视图设置头部。

### 如何使用它

#### 为所有响应设置 X-Frame-Options

要为站点中的所有响应设置相同的 `X-Frame-Options` 值，请将 `'django.middleware.clickjacking.XFrameOptionsMiddleware'` 放到 `MIDDLEWARE_CLASSES` 中：

```py
MIDDLEWARE_CLASSES = [ 
    # ... 
    'django.middleware.clickjacking.XFrameOptionsMiddleware', 
    # ... 
] 

```

此中间件在由 `startproject` 生成的设置文件中启用。

默认情况下，中间件将为每个传出的 `HttpResponse` 设置 `X-Frame-Options` 头部为 `SAMEORIGIN`。如果要改为 `DENY`，请设置 `X_FRAME_OPTIONS` 设置：

```py
X_FRAME_OPTIONS = 'DENY' 

```

在使用中间件时，可能存在一些视图，您不希望设置 `X-Frame-Options` 头部。对于这些情况，您可以使用视图装饰器告诉中间件不要设置头部：

```py
from django.http import HttpResponse 
from django.views.decorators.clickjacking import xframe_options_exempt 

@xframe_options_exempt 
def ok_to_load_in_a_frame(request): 
    return HttpResponse("This page is safe to load in a frame on any site.") 

```

#### 为每个视图设置 X-Frame-Options

要在每个视图基础上设置 `X-Frame-Options` 头部，Django 提供了这些装饰器：

```py
from django.http import HttpResponse 
from django.views.decorators.clickjacking import xframe_options_deny 
from django.views.decorators.clickjacking import  xframe_options_sameorigin 

@xframe_options_deny 
def view_one(request): 
    return HttpResponse("I won't display in any frame!") 

@xframe_options_sameorigin 
def view_two(request): 
    return HttpResponse("Display in a frame if it's from the same    
      origin as me.") 

```

请注意，您可以将装饰器与中间件一起使用。使用装饰器会覆盖中间件。

### 限制

`X-Frame-Options` 头部只会在现代浏览器中保护免受点击劫持攻击。旧版浏览器会悄悄地忽略这个头部，并需要其他点击劫持防护技术。

### 支持 X-Frame-Options 的浏览器

+   Internet Explorer 8+

+   Firefox 3.6.9+

+   Opera 10.5+

+   Safari 4+

+   Chrome 4.1+

## SSL/HTTPS

尽管在所有情况下部署站点在 HTTPS 后面对于安全性来说总是更好的，但并非在所有情况下都是实际可行的。如果没有这样做，恶意网络用户可能会窃取身份验证凭据或客户端和服务器之间传输的任何其他信息，并且在某些情况下，主动的网络攻击者可能会更改在任一方向上发送的数据。

如果您希望获得 HTTPS 提供的保护，并已在服务器上启用了它，则可能需要一些额外的步骤：

+   如有必要，请设置 `SECURE_PROXY_SSL_HEADER`，确保您已充分理解其中的警告。不这样做可能会导致 CSRF 漏洞，并且不正确地执行也可能很危险！

+   设置重定向，以便通过 HTTP 的请求被重定向到 HTTPS。

+   这可以通过使用自定义中间件来实现。请注意 `SECURE_PROXY_SSL_HEADER` 下的注意事项。对于反向代理的情况，配置主要的 Web 服务器来执行重定向到 HTTPS 可能更容易或更安全。

+   使用 *secure* cookies。如果浏览器最初通过 HTTP 连接，这是大多数浏览器的默认设置，现有的 cookies 可能会泄漏。因此，您应该将 `SESSION_COOKIE_SECURE` 和 `CSRF_COOKIE_SECURE` 设置为 `True`。这指示浏览器仅在 HTTPS 连接上发送这些 cookies。请注意，这意味着会话将无法在 HTTP 上工作，并且 CSRF 保护将阻止任何通过 HTTP 接受的 `POST` 数据（如果您将所有 HTTP 流量重定向到 HTTPS，则这将是可以接受的）。

+   使用 HTTP 严格传输安全（HSTS）。HSTS 是一个 HTTP 标头，通知浏览器所有未来连接到特定站点应始终使用 HTTPS（见下文）。结合将请求重定向到 HTTPS，这将确保连接始终享有 SSL 提供的额外安全性，只要成功连接一次。HSTS 通常在 Web 服务器上配置。

### HTTP 严格传输安全

对于应仅通过 HTTPS 访问的站点，您可以指示现代浏览器拒绝通过不安全连接（在一定时间内）连接到您的域名，方法是设置 Strict-Transport-Security 标头。这将减少您对某些 SSL 剥离中间人（MITM）攻击的风险。

如果将`SECURE_HSTS_SECONDS`设置为非零整数值，`SecurityMiddleware`将在所有 HTTPS 响应上为您设置此标头。

在启用 HSTS 时，最好首先使用一个小值进行测试，例如`SECURE_HSTS_SECONDS = 3600`表示一小时。每次 Web 浏览器从您的站点看到 HSTS 标头时，它将拒绝在给定时间内与您的域进行非安全通信（使用 HTTP）。

一旦确认您的站点上的所有资产都安全提供（即 HSTS 没有破坏任何内容），最好增加此值，以便偶尔访问者受到保护（31536000 秒，即 1 年，是常见的）。

此外，如果将`SECURE_HSTS_INCLUDE_SUBDOMAINS`设置为`True`，`SecurityMiddleware`将向`Strict-Transport-Security`标头添加`includeSubDomains`标记。这是推荐的（假设所有子域都仅使用 HTTPS 提供服务），否则您的站点仍可能通过不安全的连接对子域进行攻击。

### 注意

HSTS 策略适用于整个域，而不仅仅是您在响应上设置标头的 URL。因此，只有在整个域仅通过 HTTPS 提供服务时才应使用它。

浏览器正确尊重 HSTS 标头将拒绝允许用户绕过警告并连接到具有过期、自签名或其他无效 SSL 证书的站点。如果您使用 HSTS，请确保您的证书状况良好并保持良好！

如果您部署在负载均衡器或反向代理服务器后，并且未将`Strict-Transport-Security`标头添加到您的响应中，可能是因为 Django 没有意识到它处于安全连接中；您可能需要设置`SECURE_PROXY_SSL_HEADER`设置。

## 主机标头验证

Django 使用客户端提供的`Host`标头在某些情况下构建 URL。虽然这些值经过清理以防止跨站点脚本攻击，但可以使用虚假的`Host`值进行跨站点请求伪造、缓存污染攻击和电子邮件中的链接污染。因为即使看似安全的 Web 服务器配置也容易受到虚假的`Host`标头的影响，Django 会在`django.http.HttpRequest.get_host()`方法中针对`ALLOWED_HOSTS`设置验证`Host`标头。此验证仅适用于`get_host()`；如果您的代码直接从`request.META`访问`Host`标头，则会绕过此安全保护。

## 会话安全

与 CSRF 限制类似，要求站点部署在不受信任用户无法访问任何子域的情况下，`django.contrib.sessions`也有限制。有关详细信息，请参阅安全主题指南部分的会话主题。

### 用户上传的内容

### 注意

考虑从云服务或 CDN 提供静态文件以避免其中一些问题。

+   如果您的站点接受文件上传，强烈建议您在 Web 服务器配置中限制这些上传的大小，以防止拒绝服务（DOS）攻击。在 Apache 中，可以使用`LimitRequestBody`指令轻松设置这一点。

+   如果您正在提供自己的静态文件，请确保像 Apache 的`mod_php`这样的处理程序已被禁用，因为它会将静态文件作为代码执行。您不希望用户能够通过上传和请求特制文件来执行任意代码。

+   当媒体以不遵循安全最佳实践的方式提供时，Django 的媒体上传处理会存在一些漏洞。具体来说，如果 HTML 文件包含有效的 PNG 标头，后跟恶意 HTML，则可以将 HTML 文件上传为图像。这个文件将通过 Django 用于`ImageField`图像处理的库（Pillow）的验证。当此文件随后显示给用户时，根据您的 Web 服务器的类型和配置，它可能会显示为 HTML。

在框架级别没有防弹的技术解决方案可以安全地验证所有用户上传的文件内容，但是，您可以采取一些其他步骤来减轻这些攻击：

1.  一类攻击可以通过始终从不同的顶级或二级域名提供用户上传的内容来防止。这可以防止任何被同源策略（有关更多信息，请访问[`en.wikipedia.org/wiki/Same-origin_policy`](http://en.wikipedia.org/wiki/Same-origin_policy)）阻止的利用，例如跨站脚本。例如，如果您的站点运行在`example.com`上，您希望从类似`usercontent-example.com`的地方提供上传的内容（`MEDIA_URL`设置）。仅仅从子域名（如`usercontent.example.com`）提供内容是不够的。

1.  此外，应用程序可以选择为用户上传的文件定义一个允许的文件扩展名白名单，并配置 Web 服务器仅提供这些文件。

# 其他安全提示

+   尽管 Django 在开箱即用时提供了良好的安全保护，但仍然很重要正确部署应用程序并利用 Web 服务器、操作系统和其他组件的安全保护。

+   确保您的 Python 代码位于 Web 服务器的根目录之外。这将确保您的 Python 代码不会被意外地作为纯文本（或意外执行）提供。

+   小心处理任何用户上传的文件。

+   Django 不会限制对用户进行身份验证的请求。为了防止针对身份验证系统的暴力攻击，您可以考虑部署 Django 插件或 Web 服务器模块来限制这些请求。

+   保持您的`SECRET_KEY`是秘密的。

+   限制缓存系统和数据库的可访问性是一个好主意。

## 安全问题档案

Django 的开发团队坚决致力于负责任地报告和披露安全相关问题，如 Django 的安全政策所述。作为承诺的一部分，他们维护了一个已修复和披露的问题的历史列表。有关最新列表，请参阅安全问题档案（[`docs.djangoproject.com/en/1.8/releases/security/`](https://docs.djangoproject.com/en/1.8/releases/security/)）。

## 加密签名

Web 应用程序安全的黄金法则是永远不要相信来自不受信任来源的数据。有时通过不受信任的媒介传递数据可能是有用的。通过加密签名的值可以通过不受信任的渠道传递，以确保任何篡改都将被检测到。Django 提供了用于签名值的低级 API 和用于设置和读取签名 cookie 的高级 API，签名在 Web 应用程序中是最常见的用途之一。您可能还会发现签名对以下内容有用：

+   为失去密码的用户生成*找回我的账户*URL。

+   确保存储在隐藏表单字段中的数据没有被篡改。

+   为允许临时访问受保护资源（例如，用户已支付的可下载文件）生成一次性秘密 URL。

### 保护 SECRET_KEY

当您使用`startproject`创建一个新的 Django 项目时，`settings.py`文件会自动生成并获得一个随机的`SECRET_KEY`值。这个值是保护签名数据的关键-您必须保持它安全，否则攻击者可能会使用它来生成自己的签名值。

### 使用低级 API

Django 的签名方法位于`django.core.signing`模块中。要签名一个值，首先实例化一个`Signer`实例：

```py
>>> from django.core.signing import Signer
>>> signer = Signer()
>>> value = signer.sign('My string')
>>> value
'My string:GdMGD6HNQ_qdgxYP8yBZAdAIV1w'
```

签名附加到字符串的末尾，跟在冒号后面。您可以使用`unsign`方法检索原始值：

```py
>>> original = signer.unsign(value)
>>> original
'My string'
```

如果签名或值以任何方式被更改，将引发`django.core.signing.BadSignature`异常：

```py
>>> from django.core import signing
>>> value += 'm'
>>> try:
   ... original = signer.unsign(value)
   ... except signing.BadSignature:
   ... print("Tampering detected!")
```

默认情况下，`Signer`类使用`SECRET_KEY`设置生成签名。您可以通过将其传递给`Signer`构造函数来使用不同的密钥：

```py
>>> signer = Signer('my-other-secret')
>>> value = signer.sign('My string')
>>> value
'My string:EkfQJafvGyiofrdGnuthdxImIJw'
```

`django.core.signing.Signer`返回一个签名者，该签名者使用`key`生成签名，`sep`用于分隔值。`sep`不能在 URL 安全的 base64 字母表中。这个字母表包含字母数字字符、连字符和下划线。

### 使用盐参数

如果您不希望特定字符串的每次出现都具有相同的签名哈希，可以使用`Signer`类的可选`salt`参数。使用盐将使用盐和您的`SECRET_KEY`对签名哈希函数进行种子处理：

```py
>>> signer = Signer()
>>> signer.sign('My string')
'My string:GdMGD6HNQ_qdgxYP8yBZAdAIV1w'
>>> signer = Signer(salt='extra')
>>> signer.sign('My string')
'My string:Ee7vGi-ING6n02gkcJ-QLHg6vFw'
>>> signer.unsign('My string:Ee7vGi-ING6n02gkcJ-QLHg6vFw')
'My string'
```

以这种方式使用盐将不同的签名放入不同的命名空间。来自一个命名空间（特定盐值）的签名不能用于验证使用不同盐设置的不同命名空间中的相同纯文本字符串。结果是防止攻击者使用在代码中的一个地方生成的签名字符串作为输入到另一段使用不同盐生成（和验证）签名的代码。

与您的`SECRET_KEY`不同，您的盐参数不需要保密。

### 验证时间戳值

`TimestampSigner`是`Signer`的子类，它附加了一个签名的时间戳到值。这允许您确认签名值是在指定的时间段内创建的：

```py
>>> from datetime import timedelta
>>> from django.core.signing import TimestampSigner
>>> signer = TimestampSigner()
>>> value = signer.sign('hello')
>>> value 'hello:1NMg5H:oPVuCqlJWmChm1rA2lyTUtelC-c'
>>> signer.unsign(value)
'hello'
>>> signer.unsign(value, max_age=10)
...
SignatureExpired: Signature age 15.5289158821 > 10 seconds
>>> signer.unsign(value, max_age=20)
'hello'
>>> signer.unsign(value, max_age=timedelta(seconds=20))
'hello'
```

`sign(value)`签名`value`并附加当前时间戳。

`unsign(value, max_age=None)`检查`value`是否在`max_age`秒之内签名，否则会引发`SignatureExpired`。`max_age`参数可以接受整数或`datetime.timedelta`对象。

### 保护复杂的数据结构

如果您希望保护列表、元组或字典，可以使用签名模块的`dumps`和`loads`函数。这些函数模仿了 Python 的 pickle 模块，但在底层使用 JSON 序列化。JSON 确保即使您的`SECRET_KEY`被盗，攻击者也无法利用 pickle 格式执行任意命令：

```py
>>> from django.core import signing
>>> value = signing.dumps({"foo": "bar"})
>>> value 'eyJmb28iOiJiYXIifQ:1NMg1b:zGcDE4-TCkaeGzLeW9UQwZesciI'
>>> signing.loads(value) {'foo': 'bar'}
```

由于 JSON 的性质（没有本地区分列表和元组的区别），如果传入元组，您将从`signing.loads(object)`得到一个列表：

```py
>>> from django.core import signing
>>> value = signing.dumps(('a','b','c'))
>>> signing.loads(value)
['a', 'b', 'c']
```

`django.core.signing.dumps(obj, key=None, salt='django.core.signing', compress=False)`

返回 URL 安全的，经过 sha1 签名的 base64 压缩的 JSON 字符串。序列化对象使用`TimestampSigner`进行签名。

`django.core.signing.loads(string, key=None, salt='django.core.signing', max_age=None)`

`dumps()`的反向操作，如果签名失败则引发`BadSignature`。如果给定，检查`max_age`（以秒为单位）。

### 安全中间件

### 注意

如果您的部署情况允许，通常最好让前端 Web 服务器执行`SecurityMiddleware`提供的功能。这样，如果有一些请求不是由 Django 提供的（例如静态媒体或用户上传的文件），它们将具有与请求到您的 Django 应用程序相同的保护。

`django.middleware.security.SecurityMiddleware`为请求/响应周期提供了几个安全增强功能。每个功能都可以通过设置独立启用或禁用。

+   `SECURE_BROWSER_XSS_FILTER`

+   `SECURE_CONTENT_TYPE_NOSNIFF`

+   `SECURE_HSTS_INCLUDE_SUBDOMAINS`

+   `SECURE_HSTS_SECONDS`

+   `SECURE_REDIRECT_EXEMPT`

+   `SECURE_SSL_HOST`

+   `SECURE_SSL_REDIRECT`

有关安全标头和这些设置的更多信息，请参阅第十七章*Django 中间件*。

### 接下来是什么？

在下一章中，我们将扩展来自第一章的快速安装指南，*Django 简介和入门*，并查看 Django 的一些额外安装和配置选项。
