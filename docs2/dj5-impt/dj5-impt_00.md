# **前言**

**Django 是一个高级 Python Web 框架，鼓励快速开发和清晰、实用的设计**。Django 用于构建现代 Python Web 应用，它是免费且开源的。

学习 Django 可能是一项棘手且耗时的工作。有成百上千的教程、大量的文档和许多难以消化的解释。然而，这本书能让你在短短几天内学会并使用 Django。

在本书中，你将踏上愉快、动手实践且实用的旅程，学习 Django 全栈开发。你将在几分钟内开始构建你的第一个 Django 应用。你将获得简短的解释和实用的方法，涵盖一些最重要的 Django 特性，例如 Django 的结构、URL、视图、模板、模型、CSS 包含、图像存储、表单、会话、认证和授权，以及 Django 管理面板。你还将学习如何设计 Django 的**<st c="948">模型-视图-模板</st>** <st c="967">(</st>**<st c="969">MVT</st>**<st c="972">) 架构以及如何实现它们。此外，你将使用 Django 开发一个 Movies Store 应用并将其部署到</st> <st c="1102">互联网上。</st>

在本书结束时，你将能够构建和部署自己的 Django Web 应用。

# **本书面向对象**

**本书适用于任何水平的 Python 开发者，他们希望使用 Django 构建全栈 Python Web 应用**。本书适用于完全的 Django 初学者。

# **本书涵盖内容**

*<st c="1448">第一章</st>*<st c="1458">，*<st c="1460">安装 Python 和 Django，以及介绍 Movies Store 应用</st>*<st c="1534">，涵盖了 Python 和 Django 的安装，并介绍了 Movies Store 应用，展示了功能、类图和</st> <st c="1668">MVT 架构。</st>

*<st c="1685">第二章</st>*<st c="1695">，*<st c="1697">理解项目结构和创建我们的第一个应用</st>*<st c="1759">，探讨了 Django 的项目结构和应用创建，并演示了如何使用 Django 的 URL、视图和模板来</st> <st c="1883">创建页面。</st>

*<st c="1898">第三章</st>*<st c="1908">，*<st c="1910">设计基础模板</st>*<st c="1935">，探讨了如何使用 Django 基础模板来减少重复代码并改善 Movies</st> <st c="2054">Store 应用的外观和感觉。</st>

*<st c="2072">第四章</st>*<st c="2082">,</st> *<st c="2084">使用模拟数据创建电影应用</st>*<st c="2121">，使用</st> <st c="2177">模拟数据</st> 构建一个显示电影列表的电影应用。</st>

*<st c="2188">第五章</st>*<st c="2198">,</st> *<st c="2200">与模型一起工作</st>*<st c="2219">，讨论 Django 模型的基础知识以及如何与数据库一起工作。</st>

*<st c="2296">第六章</st>*<st c="2306">,</st> *<st c="2308">从数据库收集和显示数据</st>*<st c="2356">，讨论了如何从</st> <st c="2405">数据库</st> 收集和显示数据。</st>

*<st c="2418">第七章</st>*<st c="2428">,</st> *<st c="2430">理解数据库</st>*<st c="2456">，展示了如何检查数据库信息以及如何在数据库引擎之间切换。</st>

*<st c="2547">第八章</st>*<st c="2557">,</st> *<st c="2559">实现用户注册和登录</st>*<st c="2593">，讨论 Django 认证系统，并通过一些功能增强电影商店应用程序，允许用户注册和</st> <st c="2729">登录。</st>

*<st c="2736">第九章</st>*<st c="2746">,</st> *<st c="2748">允许用户创建、读取、更新和删除电影评论</st>*<st c="2808">，通过在评论上执行标准的</st> **<st c="2862">CRUD（创建、读取、更新、删除）</st>** <st c="2898">操作来增强电影商店应用程序。</st>

*<st c="2931">第十章</st>*<st c="2942">,</st> *<st c="2944">实现购物车系统</st>*<st c="2979">，涵盖了 Django 会话的使用，以及如何使用 Web 会话来实现购物</st> <st c="3073">车系统。</st>

*<st c="3085">第十一章</st>*<st c="3096">,</st> *<st c="3098">实现订单和项目模型</st>*<st c="3132">，探讨发票的工作原理，并创建订单和项目模型来管理购买信息。</st>

*<st c="3233">第十二章</st>*<st c="3244">,</st> *<st c="3246">实现购买和订单页面</st>*<st c="3288">，创建购买和订单页面，并以对电影商店架构的回顾结束。</st>

*<st c="3387">第十三章</st>*<st c="3398">,</st> *<st c="3400">将应用程序部署到云</st>*<st c="3438">，展示了如何将 Django 应用程序部署到</st> <st c="3483">云上。</st>

# <st c="3493">为了充分利用这本书</st>

<st c="3526">您需要安装 Python 3.10+，pip，以及一个好的代码编辑器，如 Visual Studio Code。</st> <st c="3621">最后一章需要使用 Git 将应用程序代码部署到云中。</st> <st c="3707">所有软件要求均适用于 Windows，macOS，</st> <st c="3771">和 Linux。</st>

| **<st c="3781">本书涵盖的软件/硬件</st>** **<st c="3811">** | **<st c="3819">操作系统要求</st>** **<st c="3830">** |
| --- | --- |
| <st c="3849">Python 3.10+</st> | <st c="3862">Windows, macOS,</st> <st c="3879">或 Linux</st> |
| <st c="3887">Pip</st> | <st c="3891">Windows, macOS,</st> <st c="3908">或 Linux</st> |
| <st c="3916">Visual</st> <st c="3924">Studio Code</st> | <st c="3935">Windows, macOS,</st> <st c="3952">或 Linux</st> |
| <st c="3960">Git</st> | <st c="3964">Windows, macOS,</st> <st c="3981">或 Linux</st> |

**<st c="3989">如果您正在使用本书的数字版，我们建议您亲自输入代码或访问</st>** **<st c="4091">本书的 GitHub 仓库中的代码（下一节将提供链接）。</st> <st c="4177">这样做将帮助您避免与代码复制粘贴相关的任何潜在错误。</st>** **<st c="4262">代码。</st>**

# <st c="4270">下载示例代码文件</st>

<st c="4302">您可以从 GitHub 下载本书的示例代码文件</st> [<st c="4372">https://github.com/PacktPublishing/Django-5-for-the-Impatient-Second-Edition</st>](https://github.com/PacktPublishing/Django-5-for-the-Impatient-Second-Edition)<st c="4448">。如果代码有更新，它将在 GitHub 仓库中更新。</st>

<st c="4528">我们还在我们的丰富图书和视频目录中提供了其他代码包，可在</st> [<st c="4616">https://github.com/PacktPublishing/</st>](https://github.com/PacktPublishing/)<st c="4651">找到。查看它们！</st>

# <st c="4668">《代码实战》</st>

<st c="4683">本书的《代码实战》视频可在</st> <st c="4738">[<st c="4741">https://packt.link/L3S8S</st>](https://packt.link/L3S8S)<st c="4765">查看。</st>

# <st c="4766">使用的约定</st>

<st c="4783">本书中使用了多种文本约定。</st> <st c="4839">。</st>

`<st c="4849">代码文本</st>`<st c="4862">：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。</st> <st c="5015">以下是一个示例：“db.sqlite3 文件是 Django 用于开发目的的默认 SQLite 数据库文件。”</st>

<st c="5135">代码块设置为以下格式：</st>

```py
 from django.contrib import admin
from django.urls import path
urlpatterns = [
    path('admin/', admin.site.urls),
]
```

<st c="5283">当我们希望您注意代码块中的特定部分时，相关的行或项目将被设置为粗体：</st>

```py
 from django.shortcuts import render <st c="5439">def index(request):</st>
 <st c="5458">return render(request, 'home/index.html')</st>
```

<st c="5500">任何命令行输入或输出都按照以下方式编写：</st>

```py
 python3 --version
```

**<st c="5574">粗体</st>**<st c="5579">：表示新术语、重要单词或屏幕上看到的单词。</st> <st c="5655">例如，菜单或对话框中的单词以</st> **<st c="5710">粗体</st>**<st c="5714">显示。以下是一个示例：“对于 Windows，您必须选择</st> **<st c="5770">将 python.exe 添加到</st>** **<st c="5788">PATH</st>** <st c="5792">选项。”</st>

<st c="5801">提示或重要注意事项</st>

<st c="5825">看起来像这样。</st>

# <st c="5843">联系我们</st>

<st c="5856">我们的读者反馈</st> <st c="5886">总是受欢迎的。</st>

**<st c="5901">一般反馈</st>**<st c="5918">：如果您对此书的任何方面有疑问，请通过电子邮件发送至</st> <st c="5986">customercare@packtpub.com</st> <st c="6011">，并在邮件主题中提及书名。</st>

**<st c="6070">勘误表</st>**<st c="6077">：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。</st> <st c="6173">如果您在此书中发现错误，如果您能向我们报告，我们将不胜感激。</st> <st c="6268">请访问</st> [<st c="6281">www.packtpub.com/support/errata</st>](http://www.packtpub.com/support/errata) <st c="6312">并填写</st> <st c="6325">表格。</st>

**<st c="6334">盗版</st>**<st c="6341">：如果您在互联网上以任何形式发现我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。</st> <st c="6512">请通过以下方式联系我们</st> <st c="6533">copyright@packt.com</st> <st c="6552">并提供</st> <st c="6568">材料的链接。</st>

**<st c="6581">如果您有兴趣成为作者</st>**<st c="6625">：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请</st> <st c="6750">访问</st> [<st c="6756">authors.packtpub.com</st>](http://authors.packtpub.com)<st c="6776">。</st>

# <st c="6777">分享您的想法</st>

<st c="6797">一旦您阅读了</st> *<st c="6815">《Django 5 for the Impatient》</st>*<st c="6841">，我们很乐意听听您的想法！</st> [<st c="6876">请点击此处直接进入此书的亚马逊评论页面并分享</st> <st c="6959">您的反馈</st>](https://packt.link/r/1835461557)<st c="6972">。</st>

<st c="6973">您的评论对我们和科技社区都很重要，并将帮助我们确保我们提供的是优秀的</st> <st c="7082">内容质量。</st>

# <st c="7098">下载此书的免费 PDF 副本</st>

<st c="7136">感谢购买</st> <st c="7159">此书！</st>

<st c="7169">您喜欢在路上阅读，但无法携带您的印刷</st> <st c="7235">书籍到处走吗？</st>

<st c="7252">您的电子书购买是否与您选择的设备不兼容？</st>

<st c="7322">不用担心，现在每购买一本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。</st> <st c="7409">无需付费。</st>

<st c="7417">在任何地方、任何设备上阅读。</st> <st c="7459">从您最喜欢的技术书籍中直接搜索、复制和粘贴代码到您的应用程序中。</st>

<st c="7555">优惠不仅限于此，您还可以每天在您的</st> <st c="7670">收件箱中获取独家折扣、时事通讯和丰富的免费内容</st>

<st c="7681">按照以下简单步骤获取</st> <st c="7715">福利：</st>

1.  <st c="7728">扫描二维码或访问以下</st> <st c="7759">链接</st>

![](img/B22457_QR_Free_PDF.jpg)

[<st c="7771">https://packt.link/free-ebook/9781835461556</st>](https://packt.link/free-ebook/9781835461556)

1.  <st c="7814">提交您的购买</st> <st c="7833">证明</st>

1.  <st c="7844">这就完了！</st> <st c="7856">我们将直接将您的免费 PDF 和其他福利发送到您的</st> <st c="7908">邮箱</st>
