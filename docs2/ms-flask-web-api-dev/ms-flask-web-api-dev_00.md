# 第一章：<st c="0">前言</st>

<st c="8">自 2009 年以来，我开始使用该框架进行软件开发项目以来，Flask 一直是一个强大、轻量级、无缝且易于使用的 Python 框架，用于 API 和 Web 应用程序开发。</st> <st c="217">这个非模板 WSGI 框架已经扩展了其支持，现在它有几个实用工具来支持不同的功能，甚至实现了</st> <st c="357">异步组件。</st>

<st c="381">根据我的经验，Flask 的灵活性使其成为构建各种应用程序的最佳工具，从小型电子商务到中型企业应用程序，以及许多需要 XLSX 和 CSV 自动化、报告和</st> <st c="556">图形生成的应用程序。</st> <st c="800">生成。</st>

<st c="817">本书展示了 Flask 3 及其如何使用最新功能将所有之前的软件开发规范与之前 Flask 版本进行翻译和升级。</st> <st c="989">我希望这本书能帮助你理解 Flask 3，并将其组件应用于创建解决方案和解决具有挑战性的</st> <st c="1121">现实世界问题。</st>

# <st c="1141">本书面向对象</st>

<st c="1162">本书面向那些寻求对 Flask 框架有更深入理解的熟练 Python 开发者，作为解决企业挑战的解决方案。</st> <st c="1313">它也是 Flask 熟练读者学习框架高级功能和</st> <st c="1433">新特性的绝佳资源。</st>

# <st c="1446">本书涵盖内容</st>

*<st c="1468">第一章</st>*<st c="1478">,</st> *<st c="1480">深入浅出 Flask 框架</st>*<st c="1516">，介绍了 Flask 作为一个简单且轻量级的 Python Web 框架，并展示了使用基本 Flask 功能（如视图函数、基于类的视图、数据库连接、内置 Werkzeug 服务器和库以及自定义</st> <st c="1881">环境变量）的非标准项目目录结构安装 Flask 3 以启动 Web 应用程序开发。</st>

*<st c="1903">第二章</st>*<st c="1913">,</st> *<st c="1915">添加高级核心功能</st>*<st c="1944">，提供了 Web 应用程序的 Flask 3 核心功能，如会话管理、使用</st> **<st c="2052">对象关系映射</st>** <st c="2077">(**<st c="2079">ORM</st>**) <st c="2082">进行数据管理、使用 Jinja2 模板进行视图渲染、消息闪现、错误处理、软件日志记录、添加静态内容以及将蓝图和应用工厂设计应用于</st> <st c="2258">项目结构。</st>

*<st c="2278">第三章</st>*<st c="2288">,</st> *<st c="2290">创建 REST Web 服务</st>*<st c="2316">，介绍了使用 Flask 3 进行 API 开发，包括请求和响应处理，实现 JSON 编码器和解码器以解析传入的请求体和输出的响应，使用</st> `<st c="2549">@before_request</st>` <st c="2564">和</st> `<st c="2569">@after_request</st>` <st c="2583">事件访问请求和应用程序上下文，异常处理，以及实现客户端应用程序以消费 REST 服务。</st>

*<st c="2683">第四章</st>*<st c="2693">,</st> *<st c="2695">利用 Flask 扩展</st>*<st c="2721">，讨论了如何通过使用有用的和高效的 Flask 模块来替代其底层等效模块来节省开发和努力时间，例如 Flask-Session 用于非浏览器基于的会话处理，Bootstrap-Flask 用于提供表示层，Flask-WTF 用于构建基于模型的 Web 表单，Flask-Caching 用于创建缓存，Flask-Mail 用于发送电子邮件，以及 Flask-Migrate 用于从数据模型构建数据库模式。</st>

*<st c="3160">第五章</st>*<st c="3170">,</st> *<st c="3172">构建异步事务</st>*<st c="3206">，解释了 Flask 3 的异步特性，包括创建异步视图和 API 端点函数，使用 SQLAlchemy 实现异步存储库层，使用 Celery 和 Redis 构建异步后台任务，实现 WebSocket 和</st> `<st c="3510">asyncio</st>` <st c="3517">实用工具，应用异步信号以触发事务，应用响应式编程，并介绍了 Quart 作为 Flask 3 的 ASGI 变体。</st>

*<st c="3675">第六章</st>*<st c="3685">,</st> *<st c="3687">开发计算和科学应用</st>*<st c="3739">，讨论了在构建科学应用中使用 Flask，包括使用 XLSX 和 CSV 上传以及使用流行的 Python 库（如</st> `<st c="3905">numpy</st>`<st c="3910">，《st c="3912">pandas</st>`<st c="3918">，《st c="3920">matplotlib</st>`<st c="3930">，《st c="3932">seaborn</st>`<st c="3939">，《st c="3941">scipy</st>`<st c="3946">，和</st> `<st c="3952">sympy</st>`<st c="3957">），JavaScript 库（如 Chart.js，Bokeh 和 Plotly），用于 PDF 生成的 LaTeX 工具，Celery 和 Redis 用于耗时的后台计算，以及其他科学工具，例如 Julia。</st>

*<st c="4152">第七章</st>*<st c="4162">,</st> *<st c="4164">使用非关系型数据存储</st>*<st c="4197">，解释了 Flask 如何使用流行的 NoSQL 数据库（如 Apache HBase/Hadoop、Apache Cassandra、Redis、MongoDB、Couchbase 和 Neo4J）来管理非关系型和大数据。</st>

*<st c="4372">第八章</st>*<st c="4382">,</st> *<st c="4384">使用 Flask 构建工作流程</st>*<st c="4413">，讨论了如何使用 Celery 和 Redis、SpiffWorkflow、Camunda 的 Zeebe 服务器、Airflow 2 和 Temporal.io 在 Flask 3 中实现非 BPMN 和 BPMN 工作流程。</st>

*<st c="4577">第九章</st>*<st c="4587">,</st> *<st c="4589">确保 Flask 应用程序安全</st>*<st c="4616">，提供了多种确保基于 Web 和 API 的 Flask 应用程序安全的方法，例如使用 HTTP Basic、Digest 和 Bearer-token 认证方案实现身份验证和授权机制，OAuth2 授权方案和 Flask-Login；利用编码/解码和加密/解密库来保护用户凭证；应用表单验证和数据清理以避免不同的 Web 攻击；用安全的 HTTPS 替换 HTTP 来运行应用程序；以及控制响应头以限制或限制</st> <st c="5156">用户访问。</st>

*<st c="5168">第十章</st>*<st c="5179">,</st> *<st c="5181">为 Flask 创建测试用例</st>*<st c="5210">，提供了使用 PyTest 框架测试 Flask 3 组件（如模型类、存储库事务、本地服务、视图和 API 端点函数、数据库连接、异步进程和 WebSockets）的技术，无论是否进行模拟。</st>

*<st c="5480">第十一章</st>*<st c="5491">,</st> *<st c="5493">部署 Flask 应用程序</st>*<st c="5521">，讨论了部署和运行 Web 或 API 应用程序的不同选项，包括使用 Gunicorn 为标准和非异步 Flask 应用程序、uWSGI、通过 Docker Compose 和 Kubernetes 部署的 Docker 平台以及 Apache</st> <st c="5772">HTTP 服务器。</st>

*<st c="5784">第十二章</st>*<st c="5795">,</st> *<st c="5797">将 Flask 与其他工具和框架集成</st>*<st c="5846">，提供了将 Flask 应用程序集成到不同流行工具的解决方案，例如 GraphQL、React 客户端应用程序和 Flutter 移动应用程序，以及使用 Flask 的互操作性功能在微服务应用程序中构建由 Django、FastAPI、Tornado 和 Flask 框架构建的子应用程序。</st>

# <st c="6174">为了充分利用本书</st>

<st c="6207">为了完全理解本书的前几章，您应该具备使用任何框架进行 Python 网络和 API 编程的背景，或者至少对 Flask 有一些了解。</st> <st c="6373">但是，如果您有使用标准 Python 编写脚本的背景，这也有助于您至少理解第一章，该章节介绍了如何使用 Python 语言和基本的 Flask 概念来构建网络应用程序。</st> <st c="6606">经验丰富的开发者可以使用本书进一步丰富他们的 Flask 经验，利用 Flask</st> <st c="6743">3.x 框架的新实用类和函数。</st>

| **<st c="6757">本书涵盖的软件/硬件</st>** **<st c="6787">操作系统要求</st>** | **<st c="6795">操作系统</st>** **<st c="6806">要求</st>** |
| --- | --- |
| <st c="6825">Python 3.11.x</st> | <st c="6839">Windows 10，至少</st> |
| <st c="6860">React 18.3.1</st> | <st c="6873">Ubuntu（使用 PowerShell</st> <st c="6899">和 WSL2）</st> |
| <st c="6908">Flutter 3.19.5</st> |  |
| <st c="6923">PostgreSQL 13.4</st> |  |
| <st c="6939">MongoDB 社区</st> <st c="6958">服务器 7.0.11</st> |  |
| <st c="6971">Redis 服务器</st> <st c="6985">7.2.3 (Ubuntu)</st> |  |
| <st c="6999">Redis 服务器</st> <st c="7013">3.0.504 (Windows)</st> |  |
| <st c="7030">HBase/Hadoop 2.5.5</st> |  |
| <st c="7049">Couchbase 7.2.0</st> |  |
| <st c="7065">Cassandra 4.1.5</st> |  |
| <st c="7081">Neo4J</st> <st c="7088">桌面版 1.5.8</st> |  |
| <st c="7101">Julia 1.9.2</st> |  |
| <st c="7113">Docker 25.0.3</st> |  |
| <st c="7127">Kubernetes</st> <st c="7139">5.0.4 (Docker 捆绑)</st> |  |
| <st c="7161">Zeebe</st> <st c="7168">1.1.0 (Docker)</st> |  |
| <st c="7182">Airflow 2.5</st> |  |
| <st c="7194">Temporal.io 服务器</st> <st c="7207">1.22.0</st> |  |
| <st c="7220">Apache HTTP</st> <st c="7233">服务器 2.4</st> |  |
| <st c="7243">Jaeger 1.5</st> |  |
| <st c="7254">VS</st> <st c="7258">Code 1.88.0</st> |  |

<st c="7269">可选地，如果您使用授权的 Microsoft Excel 打开 XLSX 和 CSV 文件，以及使用 Foxit PDF Reader 打开生成的</st> <st c="7414">PDF 文件，这将有所帮助。</st> 

**<st c="7424">如果您正在使用本书的数字版，我们建议您亲自输入代码或从本书的 GitHub 仓库（下一节中提供链接）获取代码。</st> <st c="7612">这样做将有助于您避免与代码复制和粘贴相关的任何潜在错误</st>** **<st c="7697">。</st>**

<st c="7705">我们建议您安装指定的 Python 版本，以避免与版本不兼容相关的意外错误。</st> <st c="7825">此外，在阅读章节的同时下载和阅读 GitHub 上的代码，以了解讨论内容，这也是一个明智的选择。</st> <st c="7952">GitHub 上的代码只是一个原型，可以作为构建您版本的应用程序的指南。</st>

# <st c="8059">下载示例代码文件</st>

<st c="8091">您可以从 GitHub 下载本书的示例代码文件，网址为</st> [<st c="8161">https://github.com/PacktPublishing/Mastering-Flask-Web-Development</st>](https://github.com/PacktPublishing/Mastering-Flask-Web-Development)<st c="8227">。如果代码有更新，它将在 GitHub 仓库中更新。</st>

<st c="8307">我们还有其他来自我们丰富的图书和视频目录的代码包可供下载，网址为</st> [<st c="8395">https://github.com/PacktPublishing/</st>](https://github.com/PacktPublishing/)<st c="8430">。查看它们吧！</st>

# <st c="8447">使用的约定</st>

<st c="8464">本书中使用了多种文本约定。</st>

`<st c="8530">文本中的代码</st>`<st c="8543">：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。</st> <st c="8696">以下是一个示例：“然而，要完全使用此功能，请使用以下</st> `<st c="8765">flask[async]</st>` <st c="8777">模块，使用以下</st> `<st c="8805">pip</st>` <st c="8808">命令安装：”</st>

<st c="8818">代码块按照以下方式设置：</st> <st c="8842">如下：</st>

```py
 from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
DB_URL = "postgresql://<username>:<password>@localhost:5433/sms"
```

<st c="9067">当我们希望您注意代码块中的特定部分时，相关的行或项目会被设置为</st> <st c="9178">粗体：</st>

```py
<st c="9186">engine = create_engine(DB_URL)</st>
<st c="9217">db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))</st> Base = declarative_base()
def init_db():
    import modules.model.db
```

<st c="9372">任何命令行输入或输出都按照以下方式编写：</st> <st c="9417">如下：</st>

```py
 pip install flask[async]
```

**<st c="9453">粗体</st>**<st c="9458">：表示新术语、重要单词或屏幕上看到的单词。</st> <st c="9534">例如，菜单或对话框中的单词会以</st> **<st c="9589">粗体</st>**<st c="9593">显示。以下是一个示例：“点击此选项将带您到</st> **<st c="9658">输入解释器路径…</st>** <st c="9681">菜单命令，最终到</st> **<st c="9719">查找…</st>** <st c="9724">选项。”</st>

<st c="9733">提示或重要注意事项</st>

<st c="9757">看起来像这样。</st>

# <st c="9775">联系我们</st>

<st c="9788">我们欢迎读者的反馈。</st>

**<st c="9833">一般反馈</st>**<st c="9850">：如果您对本书的任何方面有疑问，请通过电子邮件发送给我们，邮箱地址为</st> <st c="9918">customercare@packtpub.com</st> <st c="9943">，并在邮件主题中提及书名。</st>

**<st c="10002">勘误表</st>**<st c="10009">：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。</st> <st c="10105">如果您在这本书中发现了错误，我们非常感谢您向我们报告。</st> <st c="10200">请访问</st> [<st c="10213">www.packtpub.com/support/errata</st>](http://www.packtpub.com/support/errata) <st c="10244">并填写</st> <st c="10257">表格。</st>

**<st c="10266">盗版</st>**<st c="10273">：如果您在互联网上以任何形式发现我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。</st> <st c="10444">请通过以下邮箱联系我们</st> <st c="10465">copyright@packtpub.com</st> <st c="10487">并提供</st> <st c="10503">材料的链接。</st>

**<st c="10516">如果您有兴趣成为作者</st>**<st c="10560">：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请</st> <st c="10685">访问</st> [<st c="10691">authors.packtpub.com</st>](http://authors.packtpub.com)<st c="10711">。</st>

# <st c="10712">分享您的想法</st>

<st c="10732">一旦您阅读了</st> *<st c="10750">精通 Flask Web 和 API 开发</st>*<st c="10789">，我们非常乐意听听您的想法！</st> <st c="10824">请</st> [<st c="10831">点击此处直接进入此书的亚马逊评论页面</st>](https://packt.link/r/1-837-63322-3) <st c="10882">并分享</st> <st c="10907">您的反馈。</st>

<st c="10921">您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供优质</st> <st c="11030">的内容。</st>

# <st c="11046">免费下载此书的 PDF 副本</st>

<st c="11084">感谢您购买</st> <st c="11107">此书！</st>

<st c="11117">您喜欢在路上阅读，但无法携带您的印刷</st> <st c="11183">书籍到处走吗？</st>

<st c="11200">您的电子书购买是否与您选择的设备不兼容？</st> <st c="11259">的设备？</st>

<st c="11271">不用担心！现在，每本 Packt 书籍都免费提供该书的 DRM 免费 PDF 版本。</st> <st c="11360">无需付费。</st>

<st c="11368">在任何地方、任何地方、任何设备上阅读。</st> <st c="11410">从您最喜欢的技术书籍中直接搜索、复制和粘贴代码到您的应用程序中。</st>

<st c="11505">优惠远不止于此，您还可以每天在您的</st> <st c="11621">收件箱</st>中获得独家折扣、时事通讯和优质免费内容

<st c="11632">按照以下简单步骤获取</st> <st c="11666">以下好处：</st>

1.  <st c="11679">扫描二维码或访问以下链接：</st> <st c="11710">以下链接：</st>

![](img/B19383_QR_Free_PDF.jpg)

[<st c="11746">https://packt.link/free-ebook/9781837633227</st>](https://packt.link/free-ebook/9781837633227)

1.  <st c="11789">提交您的购买</st> <st c="11808">证明。</st>

1.  <st c="11820">这就完了！</st> <st c="11832">我们将直接将您的免费 PDF 和其他福利发送到您的</st> <st c="11884">电子邮件。</st>

# <st c="0">第一部分：学习 Flask 3.x 框架</st>

<st c="40">在本部分中，您将学习和理解实现 Flask 接受的 Web 和 API 应用程序的基本和核心组件，包括使用蓝图和应用工厂函数构建适当的 Flask 项目结构。</st> <st c="279">本部分还将教授您如何使用 psycopg2 和 asyncpg 驱动程序将 Flask 集成到 PostgreSQL 数据库中，并使用原生数据库驱动程序或</st> **<st c="497">对象关系映射</st>** <st c="522">(</st>**<st c="524">ORM</st>**<st c="527">)工具实现应用程序的存储库层。</st> <st c="536">此外，您还将学习如何使用外部 Flask 模块实现应用程序的功能，而无需花费太多时间和精力。</st>

<st c="680">本部分包括以下章节：</st> <st c="704">:</st>

+   *<st c="723">第一章</st>*<st c="733">,</st> *<st c="735">深入探索 Flask 框架</st>*

+   *<st c="771">第二章</st>*<st c="781">,</st> *<st c="783">添加高级核心功能</st>*

+   *<st c="812">第三章</st>*<st c="822">,</st> *<st c="824">创建 REST Web 服务</st>*

+   *<st c="850">第四章</st>*<st c="860">,</st> *<st c="862">利用 Flask 扩展</st>*
