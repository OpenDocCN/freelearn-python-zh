# 重要的 RESTful API 模式

本章提供了 RESTful API 常见和基本设计模式的概念和代码示例，以便您可以挑选这些示例并复制和增强它们的 RESTful API 服务。

由于设计模式提供了通用的、经过时间考验的、经过验证的和可重用的解决方案，以解决熟悉但反复出现的设计问题，因此 API 设计模式对于软件设计师在他们的 RESTful API 应用中学习和适应是必不可少的。API 设计模式提供了解决特定、反复出现的 API 设计问题的描述或模板，任何软件架构师和 API 设计师都希望在他们的 API 设计中采用。采用模式为开发者提供了很大的灵活性，并帮助他们专注于业务逻辑实现，以高质量交付服务。

作为本章的一部分，我们将学习以下常见且重要的 API 设计模式，以及一些代码示例。然而，请注意，以下模式没有特定的顺序，每个模式都针对 RESTful 约束。我们还需要确保这些基本模式在我们的 API 设计和实现模式中得到了考虑和内化：

+   无状态

+   内容协商

+   URI 模板

+   针对意图进行设计

+   分页

+   可发现性

+   错误和异常记录

+   Unicode

# 技术要求

由于我们将在本章中深入代码和示例，以及一些模式实现，因此读者应具备 Java 编程语言的知识并理解 Java 1.8 的基本概念。我们的示例使用 Spring Boot 实现，并提供了在任何地方下载和运行示例代码的说明。然而，对于那些想要执行和测试本章提供的代码示例的人来说，他们可能需要具备对数据格式（如 JSON 和 XML）的基本和必要的理解，以及对于 Maven 构建过程和客户端/服务器或 Web 服务开发的基本理解。

以下为本章的 GitHub 链接：[`github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices.git`](https://github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices.git).[ ](https://github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices.git)

要在您的计算机上运行代码，您需要 Java 8、Spring 4（1.4.4）和 Maven 3.x。请按照以下说明开始。以下为本章的先决条件：

+   Java 1.8

+   Maven 3.2

+   您喜欢的 IDE（可选）

+   命令行

+   Postman

首先，请从 GitHub 下载或克隆示例。如果有人需要帮助将示例下载到本地机器，有多种在线帮助资源可用。

# 从安装开始

如果您环境中还没有安装 JDK 或 Maven，您可能需要现在按照它们各自的安装说明进行安装。我们需要在我们的机器上安装 Maven，并且我们还可以使用 IDE 运行程序。以下说明涵盖了如何使用 Windows 命令行和 maven 命令行运行这些示例。

如果您需要 Maven 的安装说明，请按照安装文档中定义的链接和步骤进行操作。Maven 安装指南可在[`maven.apache.org/install.html`](https://maven.apache.org/install.html)找到。

以下部分提供了如何运行本章示例的说明，以及作者在其基于 Windows 的笔记本电脑上能够设置和运行的截图：

+   从[Hands-On-RESTful-API-Design-Patterns-and-Best-Practices/scripts/buildMyExamples.bat](https://github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices/blob/master/scripts/buildMyExamples.bat)下载构建脚本。

+   在您的 Windows 命令提示符中运行下载的`buildMyExample.bat`脚本，并观察您的输出，它应该与以下截图中的类似：

![图片](img/d40a8af4-84ea-4596-9bd5-16caaf99956e.png)

+   从[Hands-On-RESTful-API-Design-Patterns-and-Best-Practices/scripts/runMyExamples.bat](https://github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices/blob/master/scripts/runMyExamples.bat)下载运行脚本。观察以下截图，并将其与您的 Windows 命令行输出进行匹配：

![图片](img/28f94f2a-1a74-4c7c-a51b-2d2bee315512.png)

+   一旦您的本地服务器启动，您可以从以下 GitHub 链接下载本书的示例 Postman 集合：[`github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices/blob/master/scripts/book-examples.postman_collection.json`](https://github.com/PacktPublishing/Hands-On-RESTful-API-Design-Patterns-and-Best-Practices/blob/master/scripts/book-examples.postman_collection.json)。将集合导入到您的本地 Postman 工具中，然后运行本章中提到的示例，并像以下截图所示在您的 Postman 中查看结果：

![图片](img/41d14f24-6cec-45f9-b738-1bc4dc5363d2.png)

如果您需要安装 Postman 的帮助，您可能会发现此链接很有用：[`learning.getpostman.com/docs/postman/launching_postman/installation_and_updates/`](https://learning.getpostman.com/docs/postman/launching_postman/installation_and_updates/)。如果您需要帮助将 Postman 集合导入到您的本地 Postman 安装中，您可能会发现此链接很有用：[`learning.getpostman.com/docs/postman/collections/data_formats/#exporting-and-importing-postman-data`](https://learning.getpostman.com/docs/postman/collections/data_formats/#exporting-and-importing-postman-data)。

# 从 RESTful API 模式开始 - 第一部分

本章涵盖了最常见和必要的 REST 服务设计模式，以帮助 API 设计师和开发者在各个领域的 API 开发中。本节包含的设计模式如下：

+   无状态

+   内容协商

+   URI 模板

+   设计意图

+   分页

+   可发现性

+   错误和异常记录

+   Unicode

# 无状态

无状态指的是服务器摆脱应用状态，即存储在服务器端并帮助识别客户端请求、客户端最后交互细节及其当前上下文信息的状态。

REST 架构强制服务器**不**在服务器端维护任何客户端状态，并坚持服务器与调用客户端之间的无状态性。任何 API 开发者当然不希望在应用服务器端存储状态信息。因此，应用服务器应该始终设计为无状态（在大多数情况下）。

让我们观察客户端和服务器的一些责任，以便我们实现无状态性：

| **客户端** | **服务器** |
| --- | --- |
| 客户端应将其请求中所有必要的信息提供给服务器。 | 服务器理解客户端的请求，并应在响应中包含客户端创建会话所需的所有必要信息。 |
| 会话状态应完全由客户端管理和保持。 | 服务器不存储客户端状态，也不依赖于其存储的上下文。 |
| 客户端负责存储和处理所有其状态，并在需要时将状态信息发送到服务器。 | 服务器上不维护任何会话亲和性或会话粘性。 |

为了遵守 RESTful 约束，并且为了服务真正实现无状态，服务器甚至不存储客户端的认证/授权信息，并要求客户端在请求中提供凭证。因此，服务器会单独理解每个请求，并且之前的请求对当前请求没有影响。

在我们章节的代码示例中，我们可以观察到，我们的所有请求和响应都与/携带任何状态信息无关，它们是完全独立的。即使在后面的章节中，当我们开发认证示例时，我们的代码仍然会发展和保持在其生命周期中的无状态性。

对于无状态约束，我们如何编码？Spring Boot 的 REST API 框架提供了现成的实现；作为开发者，我们的责任是确保我们遵循 URI 约束，并为 URI 提供必要的实现。在以下代码片段中，来自我们的示例，`InvestorController.java`，我们定义了一个 URI (`/investors/{investorId}/stocks` )，用于通过投资者 ID 获取投资者的股票；仅此而已——我们没有任何特定的实现，会话验证等：

```py
@GetMapping(path = "/investors/{investorId}/stocks")
public List<Stock> fetchStocksByInvestorId(@PathVariable String investorId,
  @RequestParam(value = "offset", defaultValue = "0") int offset,
  @RequestParam(value = "limit", defaultValue = "5") int limit) {
    return investorService.fetchStocksByInvestorId(investorId, offset, limit);
}
```

由于路径元素没有任何状态指示，并且代码期望一个`investorId`以及一些其他参数，我们的实现满足了无状态约束。这将在我们处理下一章中的身份验证（也是无状态的）时更有趣。

让我们通过以下表格观察强制 RESTful 服务无状态化的必要性和一些优点：

| **优点** | **细节** |
| --- | --- |
| 可扩展性 | 通过将服务器代码部署到多个服务器上，实现可扩展性，这样任何服务器都可以处理任何请求，因为服务器不需要维护会话粘性/亲和力。这是因为客户端的请求将包含服务器管理所需的所有必要信息。 |
| 简化复杂性 | 由于服务器可以摆脱服务器端状态同步逻辑和实现（导致简化了应用程序设计），因此复杂性降低。 |
| 易于缓存，因此性能提升 | 中间件软件可以通过查看客户端请求来缓存特定 HTTP 请求的结果。此外，关于先前请求的状态信息没有不确定性和担忧。 |
| 可追踪性 | 由于客户端请求本身包含所有必要信息，服务器永远不会失去对每个客户端在应用程序中位置的跟踪。 |
| 最佳使用 HTTP/HTTPS | 由于 HTTP 本身是一个无状态协议，REST 实现与 HTTP 协议无缝结合（它很好地遵循 REST 约束）。 |

# 内容协商

RESTful API 中的资源需要处理不同类型的表示，不仅仅是 XML 或**JavaScript 对象表示法**（**JSON**），因为不同的客户端可能需要不同的表示。实际上，随着我们构建复杂 API，我们可能会发现 XML/JSON 过于限制，我们可能需要转向另一种完全不同格式的内容（Instagram 和 Flickr 使用 JPEG 和 PNG 图像，而媒体机构使用 MP3/MP4），这就是我们如何进行内容协商的。

**内容协商**是一种机制或过程，服务和客户端可以在它们通常的通信过程中选择作为它们资源表示格式的通信和握手。

正如我们在第二章，“设计策略、指南和最佳实践”，在*HTTP 头*部分所看到的，HTTP 规范提出了一组标准头，客户端可以通过这些头获取有关请求资源的信息，并携带指示其表示的消息。

因此，对于内容协商，REST 服务需要使用 HTTP 头；也就是说，当客户端发起请求时，它包括接受头，客户端和服务器可以处理的文件类型列表，无需对客户端请求、服务器处理和回复采取额外步骤。

如果响应表示的选择是在服务器端确定的，那么它是服务器驱动的协商或使用请求头的主动协商。如果同样的选择是在客户端，那么它是代理驱动的内容协商或使用不同 URI 的反应式协商。对于大多数实际用途，服务器端协商更为复杂，并且需要对客户端需求做出许多假设。因此，大多数 REST API 实现遵循代理驱动的内容协商，这依赖于 HTTP 请求头或资源 URI 模式。

让我们通过以下链接快速查看一个内容协商的实时例子：[`www.w3.org/StyleSheets/TR/logo-REC`](http://www.w3.org/StyleSheets/TR/logo-REC)。观察一下，logo-REC 是一个图像文件；然而，它没有任何文件扩展名（即，w3.org 在没有任何文件后缀的情况下提供图像），而 log-REC 不仅仅是一个文件，而是两个——`loge-REC.gif`和`logo-REC.png`。因此，通过内容协商，w3.org 服务器提供了两个不同的文件。以下截图解释了更多关于相同请求和响应头的信息：

![](img/d792f267-caa2-472d-b23c-e5e38478daf2.png)

请注意前一个截图中突出显示的部分。URL 路径，[`www.w3.org/StyleSheets/TR/logo-REC`](http://www.w3.org/StyleSheets/TR/logo-REC)，没有说明任何文件扩展名；然而，在响应头中，观察**content-location**和**content-type**为`image/png`。突出显示的矩形是一些内容协商的快速示例。同时，请观察通过请求头 accept 进行的反应式协商。

让我们接下来详细讨论内容协商以及服务/客户端如何使用它们。

# 使用 HTTP 头进行内容协商

通过我们之前的例子，我们已经看到服务器理解了传入请求和具有 HTTP 请求头内容类型的实体的内容类型。

在我们的代码示例中，我们实现了以下内容类型，默认为`application/JSON`，并且为了表示客户端希望获取的内容类型，我们使用`application/JSON`作为 Accept 头。

请注意，如果请求中没有 Accept 头，服务器将发送预先配置的默认表示类型。在我们的例子中，它始终是`application/JSON`，如下面的截图所示：

![](img/e8cc437e-5741-4487-91eb-e2c1aa9fda2a.png)

上述截图展示了我们例子中的 Content-Type 和 Accept 头。

如果我们想在我们的投资者服务应用程序中以类似的方式实现我们之前提到的 w3c 图像示例，我们只需要在`pom.xml`中添加以下依赖项：

```py
<dependency> 
  <groupId>com.fasterxml.jackson.dataformat</groupId> 
  <artifactId>jackson-dataformat-xml</artifactId> 
</dependency> 
```

按照以下方式修改控制器类中的`@GetMapping`注解：

```py
@GetMapping(path="/investors/{investorId}/stocks/{symbol}", produces={MediaType.APPLICATION_JSON_VALUE, MediaType.APPLICATION_XML_VALUE})
```

因此，通过使用 accept 请求头，客户端要么以 XML 形式，要么以 JSON 形式获取响应。酷，不是吗？

使用 Postman，我们将根据`application/XML`或`application/JSON`的`Accept`头信息值获取 XML 或 JSON 响应：

![](img/9587bcd8-1b89-49d5-b386-e154834fbaa8.png)

我们可以将内容类型信息传递给服务器的两种其他方式如下：

+   在资源 URI 中使用特定扩展：

    +   [`xxx.api.com/v1/students/course.xml`](https://xxx.api.com/v1/students/course.xml)

    +   [`xxx.api.com/v1/students/courses.json`](https://xxx.api.com/v1/students/courses.json)

+   使用参数表示扩展：

    +   [`xxx.api.com/v1/students/course?forrmat=xml`](https://xxx.api.com/v1/students/course?forrmat=xml)

    +   [`xxx.api.com/v1/students/courses?forrmat=json`](https://xxx.api.com/v1/students/courses?forrmat=json)

以下表格提供了一个快速参考指南，其中包含一些其他内容协商示例，因为服务器和客户端可能需要处理许多其他内容协商方面：

| **内容协商** | **实现** |
| --- | --- |
| 表明客户端偏好——允许客户端表明其能力（如媒体类型和语言） | `Accept:application/json,application/xml;q=0.9,*/*;q=0.8` (支持 JSON 或 XML，q -> 偏好顺序) |
| 实现媒体类型——如何决定在响应中表示中应使用哪种媒体类型 | 根据客户端的`Accept`头信息（Accept: `application/atom+xml`），服务器可以响应一个内容类型，即`Content-Type: application/atom+xml`; `charset=UTF-8` |
| 实现字符编码——了解在响应中的文本表示应使用哪种字符编码 | `Content-Type: charset=UTF-8` (使用 content-type charset) |
| 支持压缩——了解何时启用表示的压缩 | # 请求`Accept-Encoding: gzip`# 响应`Content-Encoding: gzip Vary: Accept-Encoding` |
| 发送 Vary 头信息——了解如何使用 Vary 头向客户端指示服务器如何选择特定的表示 | # 响应`Content-Language: en Vary: Accept-Language` |
| 处理谈判失败——了解何时提供默认表示或返回错误 | # 请求`Accept: application/json,*/*;q=0.0` (客户端只能处理 JSON 格式的内容)# 响应`406 Not Acceptable` (服务器返回错误代码，因为它不支持 JSON。`@GetMapping`我们的章节示例在客户端期望只有 XML 响应时抛出此错误，因为我们的示例提供的是 JSON 而不是 XML，除了一个`GET`映射)链接：`<http://www.example.org/errors/mediatypes.html>`;`rel="help"`{`"message": "此服务器不支持 JSON。请参阅帮助了解替代方案。"}` (额外帮助) |

# URI 模板

如我们在第一章，“RESTful 架构基础介绍”中看到的，**统一资源标识符**（**URI**）通常用于在类似资源的公共空间中标识特定的资源。例如，如果我们从第一章“RESTful 架构基础介绍”中选取星球大战 API 示例，电影资源由以下 URI 表示：[`swapi.co/api/films/2`](https://swapi.co/api/films/2)，[`swapi.co/api/films/3`](https://swapi.co/api/films/3)，等等。客户端通常需要在他们的请求中包含一些附加信息，以及服务器如何让客户端在 URI 中包含有关资源的信息。服务器端开发者需要能够描述其服务将响应的 URI 布局。

这个问题的答案是 URI 模板。URI 模板提供了一种描述一组资源作为变量的方式。因此，在我们继续前进并查看这一章的代码示例之前，让我们观察以下表格，其中包含我们的更通用示例：

| **资源** | **URI 模板** |
| --- | --- |
| 人物：[`swapi.co/api/people/`](https://swapi.co/api/people/) 行星：[`swapi.co/api/planets/`](https://swapi.co/api/planets/) 电影：[`swapi.co/api/films/`](https://swapi.co/api/films/) 物种：[`swapi.co/api/species/`](https://swapi.co/api/species/) | `https://swapi.co/api/{resource_id}/` |
| [`swapi.co/api/films/2/`](https://swapi.co/api/films/2/)[`swapi.co/api/films/6/`](https://swapi.co/api/films/6/) | `https://swapi.co/api/{resource_id1}/{resource_id2}` |

因此，很明显，如果我们需要为上述列表定义带有变量或资源标识符的 URI 模板，我们需要在花括号内提供这些变量。

现在，我们将看到来自本章代码实现的示例，这些示例是从投资者服务控制器类中挑选出来的：

| `@GetMapping("/investors/{investorId}/stocks")` `@GetMapping("/investors/{investorId}/stocks/{symbol}")` | 客户需要发送一个投资者 ID 和一个股票符号以进行`GET`操作 |
| --- | --- |
| `@DeleteMapping("/investors/{investorId}/stocks/{symbol}")` | 客户需要发送一个投资者 ID 和一个股票符号以进行`Delete`操作 |

Spring Boot 的`@PathVariable`注解为客户端需要的资源应用 URI 模板的魔法。请注意以下代码片段（来自我们的代码`InvestorController.java`）作为示例：

```py
........ 
.......... 
public ResponseEntity<Void> updateAStockOfTheInvestorPortfolio( 
    @PathVariable String investorId,  
    @PathVariable String symbol,  
    @RequestBody Stock stockTobeUpdated)  
{ 
          Stock updatedStock = investorService.updateAStockByInvestorIdAndStock(investorId, symbol, stockTobeUpdated); 
.......... 
........ 

```

Spring Boot 提供的`@PathVariable`注解帮助我们无缝地实现 URI 模板模式。正如我们所见，`investorId`和符号变量被我们的方法作为参数提取。

请注意，还有其他 REST 框架做得很好，并提供开箱即用的 URI 模板功能。

# 设计意图

想象一下我们的汽车在没有送到汽车服务中心修理的情况下被修理。几年前，汽车制造公司特斯拉在向其汽车发布空中修复时，就著名地做到了这一点。特斯拉的软件更新以检测充电问题并帮助降低充电速率（级联效应），以避免过热并因此避免发动机火灾。

意图设计是一个在结构和汽车领域用于参数化更改的术语，我们将学习它如何使 REST API 服务受益。

意图设计是一种表达对象之间不同关系的方法，以便一个对象的变化会自动传播到其他对象。在特斯拉的案例中，充电周期数量的减少（级联效应）有助于避免发动机过热。

在 RESTful API 世界中，API 应该被开发以确保它们满足用户提供的和面临的使用案例的要求，但又不暴露内部业务对象。意图设计是一种旨在影响或导致特定和额外用户行为的战略设计模式。

为了将我们投资者服务示例中的意图实现设计联系起来，我们应该提供一种机制，以便在新的（类型）股票被添加到投资者对象时自动更新投资者的投资组合：

![图片](img/701f6963-a5cf-45ae-aa21-dd2f7e67d290.png)

正如您在上述图中看到的，API 的意图是添加一个新的股票；然而，它也应该对投资者的投资组合产生级联效应——也许应该添加新的股票类型，需要更新股票的总数，等等。这抽象于应用开发者。当删除股票时也是如此。

请回忆一下第二章中关于 API 粒度的讨论，*设计策略、指南和最佳实践*，因为选择正确的 API 粒度在意图设计策略中起着至关重要的作用。在本章的示例中，我们提供了一个简单的更新方法，用于向投资者的投资组合添加股票类型，并鼓励读者开发更多功能，因为这种模式确保了具有正确的 API 粒度（粗粒度与细粒度）来更新和删除投资者投资组合中的股票类型。考虑以下代码：

```py
........ 
public Stock addNewStockToTheInvestorPortfolio(....) { 
  if (......)) { 
      designForIntentCascadePortfolioAdd(investorId); 
      return ......; 
  } 
...... 
} 
......... 
......... 
public boolean deleteStockFromTheInvestorPortfolio(....) { 
  ...... 
  if (....) { 
     .... 
     designForIntentCascadePortfolioDelete(investorId, deletedStatus); 
    return .......; 
      } 
......... 

private void designForIntentCascadePortfolioAdd(...) { 
            ..... 
      } 
private void designForIntentCascadePortfolioDelete(...) { 
            ........; 
      } 
} 
```

上述代码的输出如下：

![图片](img/74b87bb0-0108-4c6d-adbc-5cc5fb182bca.png)

上述截图和代码是我们在`InvestorService.java`中可以看到的示例实现（代码片段）。这也显示了当使用 Postman 工具调用`Delete`和`Add` API 时，控制台消息的情况。

# 分页

当客户端尝试获取一个分页的对象列表时，我们必须考虑服务器如何在不损害其性能的情况下管理向客户端提供如此巨大的响应。

**分页**是一个概念，它有助于仅作为响应提供部分数据，然而，它提供了如何从服务器逐页访问所有数据的信息，从而减少了服务器提供全部数据的负载和计算量。

我们是否应该将页面（的结果）视为资源或仅仅是资源的表示？将页面视为表示而不是资源是我们将在本节中讨论的内容。

由于我们决定分页是资源表示的一部分，我们将分页信息包含在 URI 查询中，即`xxx.api.com/stocks?page=2`。

请注意，作为 URI 路径一部分的分页不是一个选项（因为我们认为它不是一个资源，而是资源表示），即`xxx.api.com/stocks/page/2`，因为我们可能无法在调用之间唯一地找到资源。

在处理 URI 查询的分页时，我们需要解决的一个问题是编码，我们可以使用标准的编码方式来编码分页信息。

在我们深入代码之前，让我们看看一些行业中的更好的 API 分页示例以及一些分页类型。

Facebook 的 API 使用偏移量和限制（`fields=id`，`message& amp;limit=5`），LinkedIn 使用起始和计数（`.../{service}?start=10&count=10`），而 Twitter 使用每页记录数或计数（`api.twitter.com/2/accounts/abc1/campaigns?cursor=c-3yvu1pzhd3i7&count=50`）。

分页的资源表示方式有三种变体，具体如下：

+   **基于偏移量的**: 当客户端需要基于页数和页码的响应时。例如，`GET /investor/{id}/stocks?offset=2&limit=5 (返回 2 到 7 的股票)`**。**

+   **基于时间的**: 当客户端需要在指定的时间范围内获得响应，并且可以有一个限制，以及作为参数的一部分来表示每页的最大结果数。例如，`GET /investor/{id}/stocks?since=xxxxxx&until=yyyyy` `(返回给定日期之间的股票)`*.*

+   **基于游标的**: 一种技术，其中指针（一个带有面包屑的内置书签）引用剩余数据，作为响应提供特定子集的数据，然后释放。然而，直到游标到达记录的末尾，剩余的数据仍然需要用于后续请求。例如，`GET slack.com/api/users.list?limit=2&token=xoxp-1234-5678-90123`。以下代码解释了这一点：

```py
@GetMapping(path = "/investors/{investorId}/stocks") 
  public List<Stock> fetchStocksByInvestorId( 
      @PathVariable String investorId,  
      @RequestParam 
      (value = "offset", defaultValue = "0") int offset, 
      @RequestParam 
      (value = "limit", defaultValue = "5") int limit) { 

  return investorService.fetchStocksByInvestorId(investorId, offset, limit); 
      } 
```

上述代码的输出如下：

![](img/f0047d1f-8cca-42d8-b2cd-97129061ee0e.png)

前面的代码块展示了在我们投资者服务代码中的基于偏移量的分页实现。我们将在下一章讨论版本化和涉及数据库的其他模式时，实现和讨论其他分页方法，包括排序和过滤。

# 可发现性

正如我们在前面的章节中讨论的那样，API 开发者是 API 存在的理由。帮助他们找到合适的 API，并帮助他们通过编程方式确定正在访问的网站是否启用了 API，将是 API 最关键的责任。

与网站建立连接的主要步骤是通过使用作为用户输入的简单 URL 来找出该网站是否启用了 API。它们还帮助我们了解如何访问它们。

API 的可发现性完全取决于服务器描述其 API 使用方式的描述能力。

让我们在下一节中查看两种类型的可发现性及其在我们代码示例（截图）中的实现：

1.  **通过有效的 HTTP 方法**：当客户端使用无效的 HTTP 方法调用 REST 服务时，该请求的响应应最终以`405`HTTP 错误代码结束；即`405 方法不允许`。除了错误代码外，响应头还应提供灵活性，使客户端能够找到其响应中允许的允许的方法。此代码如下：

```py
@DeleteMapping("/investors/{investorId}/stocks/{symbol}") 
public ResponseEntity<Void> deleteAStockFromTheInvestorPortfolio( 
  @PathVariable String investor, 
  @PathVariable String symbol) { 
    if (investorService.deleteStockFromTheInvestorPortfolio(investorId, symbol)) { 
      return ResponseEntity.noContent().build(); 
    } 
    return ResponseEntity.ok(null); 
} 
```

上述代码的输出如下：

![图片](img/db97037f-37e9-4eca-ba46-7b0a3ab7032d.png)

1.  **通过提供新创建资源的 URI**：将 URI 作为部分位置头作为新创建资源的响应是另一种可发现性的方法。返回的 URI 作为位置将通过`GET`可用。此代码如下：

```py
@PostMapping("/investors/{investorId}/stocks") 
public ResponseEntity<Void......) { 
  Stock insertedStock = investorService.addNewSto...; 
  if (insertedStock == null) { 
    return ResponseEntity.noContent().build(); 
  } 
  URI location = ServletUriComponentsBuilder..... 
  return ResponseEntity.created(location).build(); 
} 
```

上述代码的输出如下：

![图片](img/a65b4b72-7c56-46a8-8bff-b5a1af3e4a06.png)

请注意，由于我们的示例使用的是 Spring Boot，我们正在利用 Spring Boot 无缝和开箱即用的可发现性实现能力（使用`@GetMapping`、servlets URI 组件构建器等）。

另一种类型的可发现性（尽管尚未标准化）可以通过有效的链接头实现。当通过`GET`响应客户端的特定资源时，你必须向客户端提供有关他们下一步可以做什么的线索；也就是说，通过链接头提供所有可用资源列表：

![图片](img/eacff5f9-49e7-4ca5-ac31-51f758a4e1d1.png)

作为可发现性的一个实时示例，前面的截图展示了谷歌的一个 API。

# 错误和异常日志记录

当我们讨论 API 开发者对我们服务消费的重要性时，我们一直在强调，错误处理和异常处理应无妥协地得到妥善处理。服务对 API 开发者来说是黑盒子，因此提供错误和异常的服务可以为客户端提供清晰的上下文和可见性，以便使用我们的 API。

让我们看看一些真实世界且非常流行的 API，以及它们在样本错误场景中是如何处理这些错误的。这将在以下表格中展示。我们还将查看我们在投资者服务代码中处理这些错误和异常的方式：

| **APIs** | **HTTP 代码** | **样本消息** | **备注** |
| --- | --- | --- | --- |
| Facebook | `200` | `{"type":"OAUTH exception","message":"...."}` | 注意，`200`表示成功。然而，API 决定发送一个错误消息作为响应，同时仍然为 API 调用提供一个成功（`200`）代码。 |
| Twilio | `401` | `{"status":401,"message":"Authenticate","code":"...."..}` | 利用现有的 HTTP 代码。 |
| 投资者服务 | `405` | `{"Status":405,"error":"Method Not Allowed"....` | 利用现有的 HTTP 代码。 |

在我们的样本仓库中，我们还实现了一个名为`InvestorNotFoundException`的自定义异常类的示例。以下是从 Postman 获取的代码片段和输出样本：

```py
public class InvestorNotFoundException extends RuntimeException{ 

      ........ 
      public InvestorNotFoundException(String exception) 
      { 
            super(exception); 
      } 
} 
```

上述代码的输出如下：

![图片](img/c1fe1000-aba7-4237-a270-23baef5c0a6b.png)

从我们之前的例子中，我们可以观察到我们在使用现有的 HTTP 标准错误代码管理错误和异常方面做得相当不错。然而，我们可以更进一步，通过向调用者提供更多定制化的错误和消息来做到这一点；例如，调用者收到一个`404`错误而不是`500`错误可能更为恰当。也许我们将在下一章构建更多模式的同时实现一些自定义错误消息。

# Unicode

使我们的 API 支持多种语言的一个简单而强大的方法是使 API 能够支持 Unicode。

**Unicode**是一种编码标准，它支持国际字符集。它为包括中文、韩文和阿拉伯文及其脚本在内的多种语言中的每个字符都有一个唯一的数字。这个唯一的数字使得几乎所有的字符都可以在平台、程序和设备之间识别和访问。

因此，简而言之，我们可以通过支持 Unicode 作为它们头部的部分来简化支持多语言的 REST API。以下代码展示了这一点：

```py
@GetMapping(value="/investors/welcome", produces="text/plain;charset=UTF-8") 
  public String responseProducesConditionCharset() {    return " (\"Welcome Investor!\" in Japanese)"; 
} 
```

其输出如下：

![图片](img/68bf55e0-448a-48e2-a53c-b31382c485bd.png)

以下图表显示了(`InvestorController.java`)头部中的 accept-encoding 字符集代码片段以及 Postman 的结果。

# 摘要

在了解了处理无状态、内容协商实践、URI 模板定义、服务设计以实现意图、可发现性和一种分页类型等真实代码示例之后，我们详细讨论了错误和异常处理，最后以 Unicode 国际化实现（支持我们的服务中的多种语言）作为结论。

本章应为任何想要掌握 RESTful 服务的人提供一个出色的起点；不仅包括基础知识，还包括核心模式。在下一章中，我们将看到更多高级模式的实现和示例，以便我们的读者能够越来越多地利用最佳实践和实现。
