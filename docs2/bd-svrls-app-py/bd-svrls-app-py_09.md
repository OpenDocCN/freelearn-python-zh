# Microsoft Azure Functions 简介

到目前为止，我们已经学习了如何在 AWS 环境中使用 Python 构建 serverless 函数和 serverless 架构。我们还详细学习了 AWS Lambda 工具的设置和环境。现在，我们将学习和探索其 Microsoft Azure Functions 对应版本。

在本章中，您将学习 Microsoft Azure Functions 的工作原理，Azure Functions 控制台的外观，以及如何了解控制台中的设置。本章分为以下部分：

+   Microsoft Azure Functions 简介

+   创建您的第一个 Azure Function

+   理解触发器

+   理解日志记录和监控

+   编写 Microsoft Azure Functions 的最佳实践

# Microsoft Azure Functions 简介

Microsoft Azure Functions 是 AWS Lambda 服务的 Azure 对应服务。在本节中，我们将学习如何定位和导航 Microsoft Azure Functions 控制台。因此，让我们先执行以下步骤：

1.  您可以通过导航到左侧菜单中的“所有服务”标签并输入函数过滤器来定位 Azure Functions 应用。现在，您将注意到名为“Function Apps”下的 Microsoft Azure Function 服务：

![图片](img/00280.jpeg)

1.  一旦点击，您将被重定向到 Function Apps 控制台。目前，如果您还没有创建任何函数，它将是空的。控制台看起来可能如下所示：

![图片](img/00281.jpeg)

1.  现在，让我们开始创建一个 Azure Function。为此，我们需要点击左侧菜单中的“创建资源”选项，然后从该列表中点击“计算”选项，然后从随后的选项列表中选择“Function App”选项：

![图片](img/00282.jpeg)

Microsoft Azure Functions 在仪表板上的“计算”资源列表中。在以下部分，我们将学习如何创建 Microsoft Azure Functions，并了解不同的触发器及其工作方式。

# 创建您的第一个 Azure Function

在本节中，我们将学习如何创建和部署 Azure Function。我们将逐步进行，以便了解 Azure Function 的每个部分是如何工作的：

1.  当您在菜单中点击 Functions App 时，您将被重定向到 Function App 创建向导，如下截图所示：

![图片](img/00283.jpeg)

1.  根据向导中的要求添加所需信息。选择 Linux（预览）作为操作系统。然后，点击向导底部的蓝色“创建”按钮：

![图片](img/00284.jpeg)

1.  点击底部的自动化选项将打开自动化 Function 部署的验证屏幕。这对于本章不是必需的。这只会验证您的 Azure Function：

![图片](img/00285.jpeg)

1.  一旦点击创建，您将在“通知”菜单下看到正在进行的部署：

![图片](img/00286.jpeg)

1.  一旦成功创建，它将以绿色通知的形式反映在您的通知列表中：

![](img/00287.jpeg)

1.  点击“转到资源”将带您进入新创建的 Azure 函数。函数控制台将看起来像这样：

![](img/00288.jpeg)

我们已成功创建了一个 Azure 函数。我们将在本章的后续部分更详细地介绍触发器、监控和安全。

# 理解触发器

在本节中，我们将探讨 Azure 函数应用程序中触发器的工作原理。我们还将了解不同类型的触发器及其用途。执行以下步骤：

1.  在左侧菜单中，点击“函数”选项旁边的 (+) 符号以添加、删除或编辑触发器：

![](img/00289.jpeg)

1.  您将被带到函数创建控制台，其外观如下：

![](img/00290.jpeg)

1.  Azure 对 Python 的支持并不多。因此，在这个控制台中，让我们选择我们自己的自定义函数。在底部“自己开始”选项下点击“自定义函数”：

![](img/00291.jpeg)

1.  在函数创建向导中，在右侧菜单中启用“实验性语言”选项。现在，您将能够在可用语言中看到 Python 选项：

![](img/00292.jpeg)

1.  对于 Python 语言，有两个可用的触发器。一个是 HTTP 触发器，另一个是队列触发器**，如以下截图所示**：

![](img/00293.jpeg)

1.  HTTP 触发器会在接收到 HTTP 请求时触发函数。当您点击它时，您会注意到添加不同 HTTP 相关设置（如授权和名称）的选项：

![](img/00294.jpeg)

1.  下一个触发器是队列触发器。当队列中添加消息时，它将触发函数。我们也在我们之前的一个章节中在 AWS Lambda 中做了同样的事情：

![](img/00295.jpeg)

# 理解 Azure 函数中的日志记录和监控

在本节中，我们将学习和理解用户在 Microsoft Azure 函数中可用的监控和日志记录机制。执行以下步骤：

1.  通过在函数下点击“监控”选项，我们可以访问特定 Azure 函数的监控套件：

![](img/00296.jpeg)

1.  我们创建的函数的监控套件看起来如下：

![](img/00297.jpeg)

1.  现在，点击菜单顶部的“打开应用洞察”选项。这将带您进入详细监控页面：

![](img/00298.jpeg)

1.  如果您向下滚动，您将看到特定于函数的指标，例如服务器响应时间和请求性能。这非常有用，因为它意味着我们不需要为监控所有这些统计数据而设置单独的仪表板：

![](img/00299.jpeg)

既然我们已经了解了 Microsoft Azure 函数的日志记录和监控，让我们来探讨一些最佳实践。

# 编写 Azure 函数的最佳实践

我们已经学习了如何创建、配置和部署 Microsoft Azure Functions。现在我们将学习使用它们的最佳实践：

+   与 AWS Lambda 相比，Microsoft Azure Functions 对 Python 的支持并不广泛。它们有一组非常有限的基于 Python 的触发器。因此，您需要为大多数功能编写自定义函数。开发者在做出使用 Microsoft Azure Functions 的决定之前需要牢记这一点。Microsoft Azure Functions 支持的语言有 C#、F# 和 JavaScript：

![](img/00300.jpeg)

+   Microsoft Azure Functions 支持的实验性语言包括 Bash、Batch、PHP、TypeScript、Python 和 PowerShell：

![](img/00301.jpeg)

+   确保您正确使用安全设置以保护您的功能。您可以在平台功能选项中找到您需要的所有设置：

![](img/00302.jpeg)

+   最后，尽可能多地使用监控，因为它对于记录和监控无服务器功能至关重要。我们已经了解了监控细节和相应的设置。

# 摘要

在本章中，我们学习了 Microsoft Azure Functions 以及如何构建它们。我们了解了可用的各种功能，以及 Python 运行时的可用触发器。我们还学习了 Microsoft Azure Functions 的日志记录和监控功能，并了解了 Azure 的实验性功能，例如除了它提供的标准语言集之外的附加运行时。我们进行了实验：
