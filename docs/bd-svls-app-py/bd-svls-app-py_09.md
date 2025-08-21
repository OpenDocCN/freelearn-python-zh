# 第九章：Microsoft Azure Functions 简介

到目前为止，我们已经学习了如何在 AWS 环境中使用 Python 构建无服务器函数和无服务器架构。我们还详细了解了 AWS Lambda 工具的设置和环境。现在，我们将学习并探索其在 Microsoft Azure Functions 中的对应物。

在本章中，您将学习 Microsoft Azure Functions 的工作原理，Microsoft Azure Functions 控制台的外观，以及如何理解控制台中的设置。本章分为以下几个部分：

+   Microsoft Azure Functions 简介

+   创建你的第一个 Azure 函数

+   理解触发器

+   理解日志记录和监控

+   编写 Microsoft Azure Functions 的最佳实践

# Microsoft Azure Functions 简介

Microsoft Azure Functions 是 AWS Lambda 服务的 Azure 对应物。在本节中，我们将学习如何定位和浏览 Microsoft Azure Functions 控制台。因此，让我们开始执行以下步骤：

1.  您可以通过导航到左侧菜单上的“所有服务”选项卡并键入函数过滤器来找到 Azure Functions 应用。您现在会注意到 Microsoft Azure Function 的服务名称下有“函数应用”：

![](img/024eb3c7-fccc-499b-909b-1686c042d011.png)

1.  点击后，您将被重定向到函数应用控制台。如果您还没有创建任何函数，现在它将是空的。控制台的外观将类似于这样：

![](img/f99b9643-401e-4697-b00c-e364285f5b7b.png)

1.  现在，让我们开始创建一个 Azure 函数。为此，我们需要在左侧菜单中点击“创建资源”选项，然后从列表中点击“计算”选项，然后从随后的选项列表中选择“函数应用”选项：

![](img/016eb2f1-aa5d-45f8-a10f-816728d6c56c.png)

Microsoft Azure Functions 列在仪表板上的**计算**资源列表中。在接下来的部分中，我们将学习如何创建 Microsoft Azure Functions，还将了解不同类型的触发器以及它们的工作原理。

# 创建你的第一个 Azure 函数

在本节中，我们将学习如何创建和部署 Azure 函数。我们将逐步了解 Azure 函数的每个部分是如何工作的：

1.  当您在菜单中点击“函数应用”时，您将被重定向到“函数应用”创建向导，如下截图所示：

![](img/a8b9e92c-4db0-4aae-864a-647385c1b99e.png)

1.  根据向导中的要求添加所需信息。选择 Linux（预览）作为操作系统。然后，点击向导底部的蓝色“创建”按钮：

![](img/ef0f8be0-9278-4e33-b642-55d248868cdf.png)

1.  点击底部的“自动化”选项将打开一个用于自动化函数部署的验证屏幕。本章不需要这个。这只是验证您的 Azure 函数：

![](img/bb8349b8-d04d-477c-b3ab-b4ad154d6d82.png)

1.  点击创建后，你将在“通知”菜单下看到部署正在进行中：

![](img/9f169501-1b18-4a1a-980b-1a745ddda737.png)

1.  成功创建后，它将在通知列表中以绿色通知的形式反映出来：

![](img/88794804-d64a-448e-9751-ead6d85dc3ee.png)

1.  点击“转到资源”将带您到新创建的 Azure 函数。函数控制台将如下所示：

![](img/f629720e-9238-4ffc-a5b0-e46d24b2ccf4.png)

我们已成功创建了一个 Azure 函数。在本章的后续部分中，我们将更详细地介绍触发器、监控和安全性。

# 理解触发器

在本节中，我们将了解 Azures 函数应用中触发器的工作原理。我们还将学习不同类型的触发器及其目的。执行以下步骤：

1.  在左侧菜单中，点击“函数”选项旁边的(+)符号，以添加、删除或编辑触发器：

![](img/9dac12f7-901a-4b32-9af5-378901345898.png)

1.  您将被带到函数创建控制台，看起来像这样：

![](img/79967d0d-18e8-47da-84e7-03dc1ba59ba6.png)

1.  Azure 对 Python 的支持并不多。因此，在这个控制台中，让我们选择自定义函数。在底部的 Get Started on your own 选项下，单击 Custom function：

![](img/14f14139-bed6-4ed0-b1bf-216b9e38d985.png)

1.  在函数创建向导中，启用右侧菜单中的实验性语言选项。现在，您将能够在可用语言中看到 Python 选项：

![](img/e5f630d7-ba46-477e-9267-5c2a5b603112.png)

1.  Python 语言有两个可用的触发器。一个是 HTTP 触发器，另一个是队列触发器，如下面的屏幕截图所示：

![](img/c47cbde1-accd-4af0-8d24-917bb8f2eb1c.png)

1.  HTTP 触发器将在收到 HTTP 请求时触发函数。当您点击它时，您将注意到添加不同的与 HTTP 相关的设置的选项，例如授权和名称：

![](img/eeb2b99c-509e-4345-8dc5-7458d2efea43.png)

1.  下一个触发器是队列触发器。这将在消息添加到队列时触发函数。我们在之前的章节中也在 AWS Lambda 中做过同样的事情：

![](img/08ae9d33-259b-4b40-8e90-92066801f03d.png)

# 了解 Azures Functions 中的日志记录和监视

在本节中，我们将学习并了解 Microsoft Azure Functions 中用户可用的监视和日志记录机制。执行以下步骤：

1.  通过单击函数下的 Monitor 选项，我们可以访问特定 Azure 函数的监视套件：

![](img/005266aa-b166-4bad-8cd9-d86bfb2ff1ba.png)

1.  我们创建的函数的监视套件如下所示：

![](img/7c2ede0a-6906-4060-ba71-dded4e337498.png)

1.  现在，单击菜单顶部的 Open Application Insights 选项。这将带您到详细的监视页面：

![](img/c447d1d8-b10c-48c6-85c8-948f4f39bc97.png)

1.  如果您向下滚动，您将看到特定于函数的指标，例如服务器响应时间和请求性能。这非常有用，因为这意味着我们不需要单独的仪表板来监视所有这些统计数据：

![](img/1394c021-d657-44c7-a04d-de4045b085cc.png)

现在我们已经了解了 Microsoft Azure Functions 的日志记录和监视，让我们看看一些最佳实践。

# 编写 Azure Functions 的最佳实践

我们已经学会了如何创建、配置和部署 Microsoft Azure Functions。现在我们将学习如何使用它们的最佳实践：

+   Microsoft Azure Functions 对 Python 的支持不像 AWS Lambda 那样广泛。它们有一组非常有限的基于 Python 的触发器。因此，在决定使用 Microsoft Azure Functions 之前，开发人员需要牢记这一点。Microsoft Azure Functions 支持的语言有 C＃、F＃和 JavaScript：

![](img/7c83b144-9f29-4aae-9d51-51e05aa526da.png)

+   Microsoft Azure Functions 支持的实验性语言包括 Bash、Batch、PHP、TypeScript、Python 和 PowerShell：

![](img/4c86dcb8-d7e3-405c-b8fc-539eb14fcad2.png)

+   确保正确使用安全设置来保护您的函数。您可以在平台功能选项中找到所有您需要的设置：

![](img/a34cb76e-01c0-439c-9671-00a004923d6d.png)

+   最后，尽可能多地使用监视，因为对于记录和监视无服务器函数至关重要。我们已经了解了监视细节和相应的设置。

# 总结

在本章中，我们学习了关于 Microsoft Azure Functions 以及如何构建它们的知识。我们了解了可用的各种功能，以及 Python 运行时的可用触发器。我们还学习并尝试了 Microsoft Azure Functions 的日志记录和监控功能，以及理解并尝试了 Azure 的实验性功能，例如除了标准语言集之外提供的额外运行时。
