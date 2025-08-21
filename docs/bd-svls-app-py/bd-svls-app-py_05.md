# 第五章：日志记录和监视

我们已经了解了无服务器架构的概念，并了解了 AWS 的无服务器服务 AWS Lambda 的基础知识和内部工作原理。我们还创建了一些示例无服务器项目，以更好地理解这些概念。在学习过程中，我们还学习了其他几个 AWS 服务的基础知识，例如警报、SNS、SQS、S3 存储桶和 CloudWatch。

在本章中，我们将学习如何为我们构建的无服务器系统进行日志记录和监视。日志记录和监视软件代码和系统非常重要，因为它们帮助我们进行遥测和灾难恢复。日志记录是一个过程，我们在其中存储代码或整体架构发出的日志。监视是一个过程，我们密切监视代码或架构中组件和进程的活动、状态和健康状况。

因此，您将学习如何设置和了解 AWS Lambda 的监视套件，它与 AWS 的监视服务 CloudWatch 仪表板紧密集成。我们还将学习 AWS 的日志记录服务 CloudWatch Logs 服务。最后，我们还将学习和了解 AWS 的分布式跟踪和监视服务 CloudTrail 服务。

本章涵盖以下主题：

+   了解 CloudWatch

+   了解 CloudTrail

+   CloudWatch 中的 Lambda 指标

+   CloudWatch 中的 Lambda 日志

+   Lambda 中的日志记录

# 了解 CloudWatch

如前所述，CloudWatch 是 AWS 的日志记录和监视服务。我们已经了解并学习了 CloudWatch 警报，这是 CloudWatch 的一个子功能。现在我们将学习该服务的图形套件。AWS 环境中几乎每个服务都有一种方法将其日志和指标发送到 CloudWatch 进行日志记录和监视。每个服务可能有多个可以监视的指标，具体取决于其功能。

同样，AWS Lambda 也有一些指标，例如调用次数、调用运行时间等，它会发送到 CloudWatch。值得注意的是开发人员也可以将自定义指标发送到 CloudWatch。因此，在接下来的步骤中，我们将学习与 AWS Lambda 对应的 AWS CloudWatch 的不同部分和功能：

1.  首先，让我们看看 CloudWatch 控制台的外观，并通过浏览控制台来感受一下。浏览至[console.aws.amazon.com/cloudwatch/](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fconsole.aws.amazon.com%2Fcloudwatch%2F%3Fstate%3DhashArgs%2523%26isauthcode%3Dtrue&client_id=arn%3Aaws%3Aiam%3A%3A015428540659%3Auser%2Fcloudwatch&forceMobileApp=0)：

![](img/1046303e-63ab-4f58-8dd5-3757559e8d88.png)

1.  正如我们所看到的，CloudWatch 控制台中有大量信息。因此，我们现在将尝试逐个了解每个组件。在左侧，我们可以看到一系列选项，包括仪表板、警报、计费等。我们将尝试了解它们以及它们作为了解 CloudWatch 控制台的一部分的功能。

1.  这里的仪表板是用户可以配置的 CloudWatch 指标面板。例如，用户可能希望在一个地方拥有一组特定的服务器（EC2）指标，以便更好地监视它们。这就是 AWS CloudWatch 仪表板发挥作用的地方。当您点击左侧的“仪表板”选项时，您可以看到仪表板控制台，它看起来像这样：

![](img/8a4bc481-d9a3-4ff4-96be-c025773752b8.png)

1.  让我们继续点击控制台左上角的蓝色“创建仪表板”按钮，创建一个新的仪表板。将出现以下框：

![](img/22ac5911-7102-4b57-b57c-67f86a992261.png)

1.  这将带您进入下一步，您将被要求为仪表板选择一个小部件类型。目前有四种类型的小部件可用。小部件选择屏幕如下所示：

![](img/98a3e0c5-f7bb-465a-8b7d-d719c5067d18.png)

1.  出于本教程的目的，我选择了线条样式小部件。您可以选择适合您的图表样式和所需监视的任何小部件。一旦您选择了小部件样式并单击蓝色的“配置”按钮，您将被重定向到一个向导，在那里您将被要求添加一个度量，如下面的屏幕截图所示：

![](img/03495b0c-d00c-4397-bf4f-0435e606b263.png)

1.  在底部选择一个可用的度量，并将其添加到小部件中。一旦您选择了度量标准，点击页面右下角的蓝色“创建小部件”按钮，如下面的屏幕截图所示：

![](img/49439599-8f9f-4604-8795-a45411ba4eeb.png)

1.  现在，您可以在“仪表板”部分看到您刚刚创建的仪表板：

![](img/0d9b4553-1e12-4e4f-bb4f-e0ee8f81352b.png)

1.  我们已经成功学习并创建了 AWS CloudWatch 仪表板。现在我们将继续学习 CloudWatch 事件。在前几章中，我们已经了解了 CloudWatch 警报，查看了它们的功能以及如何创建和使用它们。

1.  在 CloudWatch 菜单的左侧单击“事件”链接。您将被重定向到 CloudWatch 事件页面，如下面的屏幕截图所示：

![](img/db25848e-e62f-43ab-ba37-ccb0fa4fb150.png)

1.  一旦您单击蓝色的“创建规则”按钮，您将被重定向到事件创建向导，它看起来像这样：

![](img/cb2ad2e0-c2b9-4499-9e65-55274aa727ce.png)

1.  可以有两种类型的事件，即事件模式和计划，它们各自有不同的目的。在这里，我们只会了解计划类型，因为它对于调度 Lambda 函数非常方便：

![](img/52498f0f-822b-47b8-a1cf-5d9f9072ffcd.png)

1.  速率可以以分钟、小时或天为单位设置，也可以设置为 cron 表达式，无论您喜欢哪种方式。现在，需要选择目标。目标可以是任何有效的 Lambda 函数，如下拉菜单所示：

![](img/a0c2d995-161b-4f16-b852-448fbcc6b48f.png)

1.  一旦您选择了函数，您可以在底部单击蓝色的“配置详细信息”。这将带您到配置规则详细信息页面，如下面的屏幕截图所示：

![](img/c7c5f05a-3898-404f-9493-e6565f19fbe7.png)

1.  一旦您输入要创建的规则的名称和描述，您可以单击底部的蓝色“创建规则”按钮。这将成功创建一个事件，并将在您的 CloudWatch 控制台中反映出来：

![](img/2f488395-182b-46db-ab1b-bd33a090ef0f.png)

我们已成功为 Lambda 函数添加了一个 cron 事件，这意味着 Lambda 将按照用户在事件设置中指定的间隔定期调用。

1.  现在，我们将尝试了解 AWS CloudWatch 的日志功能。这是 Lambda 函数存储其日志的地方。您可以单击左侧菜单中的“日志”链接，以访问 CloudWatch 日志的控制台：

![](img/d2d37382-f5b9-4a68-a770-a3f74538479d.png)

1.  我们可以看到我们在整本书中创建的所有 Lambda 函数的完整日志列表。当您单击日志组时，您可以找到有关它的更多详细信息，以及自定义选项。每个日志流都是与 Lambda 函数相关联的调用：

![](img/eef02958-3237-4aeb-84d1-4724fdf4f25d.png)

1.  您还可以利用 CloudWatch 提供的附加功能来处理日志数据，这可以在“日志组”中的下拉“操作”菜单中看到：

![](img/922f928e-2eb2-4130-a62f-abbddb68aa90.png)

1.  最后，我们将通过探索和学习 CloudWatch 指标来结束。可以通过单击 CloudWatch 控制台左侧的“指标”选项来访问指标控制台：

![](img/8841372c-f770-4656-b97b-c4042c14eb8d.png)

1.  您可以在底部菜单中选择任何选项来绘制指标。在本教程中，我已添加了一个 Lambda 指标，即函数`serverless-api`中的错误数量：

![](img/159fc81d-ca6b-4b58-acba-becd2c391461.png)

# 了解 CloudTrail

CloudTrail 是 AWS 的另一个监控服务，您可以查看在您的 AWS 帐户中发生的所有事件和路径。该服务比 CloudWatch 服务更详细，因为它记录和存储事件和路径的方式更详细。

因此，我们将在以下步骤中探索和学习有关此服务的信息：

1.  可以在[console.aws.amazon.com/cloudtrail/](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fconsole.aws.amazon.com%2Fcloudtrail%2Fhome%3Fstate%3DhashArgs%2523%26isauthcode%3Dtrue&client_id=arn%3Aaws%3Aiam%3A%3A015428540659%3Auser%2Fcloudtrail&forceMobileApp=0)访问 AWS CloudTrail 的仪表板：

![](img/ce2500b6-f225-4255-8023-d255bbee795b.png)

1.  当您单击“事件历史记录”按钮时，您可以在 CloudTrail 菜单的左侧看到 AWS 帐户中的事件列表。事件历史记录页面如下所示：

![](img/6b2bab2d-c948-45e8-9594-0b29c27bd9db.png)

1.  CloudTrail 的第三个功能是路径。用户可以为他们的 AWS 服务设置路径，例如 Lambda。已设置的路径可以在路径仪表板上找到。这可以通过单击左侧菜单中的“路径”选项来访问：

![](img/623c5cb7-de80-48fd-b9ac-7d36933a3b27.png)

1.  现在，让我们了解如何在 CloudTrail 仪表板中创建路径。您可以转到 CloudTrail 的主仪表板，然后单击蓝色的“创建路径”按钮。这将带您进入路径创建向导：

![](img/46c4272b-7b8d-47f9-818a-f1beeaa44072.png)

1.  您可以在此处输入您的路径的详细信息。您可以将默认选项保留为“将路径应用于所有区域”和“管理事件”选项：

![](img/1617ea6b-b03a-4308-ad6e-894561bd5025.png)

1.  现在，继续下一个设置，选择 Lambda 选项，然后单击选项列表中的“记录所有当前和未来的函数”。这将确保我们所有的 Lambda 函数都能够正确记录在 CloudTrail 中：

![](img/9c69f7dd-45b8-40d7-997f-157529678aa4.png)

1.  现在，在最终的“存储位置”选项中，选择一个 S3 存储桶来存储 CloudTrail 日志。这可以是已经存在的存储桶，或者您也可以要求 CloudTrail 为此创建一个新的存储桶。我正在使用一个现有的存储桶：

![](img/d39c4224-9ae5-41d8-9ac0-6f220f628439.png)

1.  在所有详细信息和设置都已相应配置后，您可以单击蓝色的“创建路径”按钮来创建路径。现在，您可以在 CloudTrail 仪表板中看到您刚刚创建的路径，如下面的屏幕截图所示：

![](img/93727858-8059-47b2-a5be-3975204b45d8.png)

1.  现在，当您单击刚刚创建的路径时，您可以看到所有配置详细信息，如下面的屏幕截图所示：

![](img/b8ecaae1-742f-4432-89f3-9eb34c7bc33e.png)

1.  您还可以注意到一个非常有趣的选项，它使您能够配置 CloudWatch 日志以及 SNS，以通知您任何特定的活动，例如当 Lambda 函数出现错误时：

![](img/70560b9e-6ef6-4e5f-af54-6692315440a6.png)

1.  最后，您还可以像其他 AWS 服务一样为路径添加标记：

![](img/1b11d26e-56a2-4116-b352-2001c689eaf2.png)

1.  此外，让我们了解如何为我们的路径配置 CloudWatch 日志。因此，您需要单击标记部分上方的蓝色“配置”按钮：

![](img/eaea7387-216c-46d9-b4f3-70f3d8c5531d.png)

1.  单击“继续”将带您到创建向导，您需要根据 IAM 角色设置相应地配置权限。在本教程中，我已选择了“创建新的 IAM 角色”选项，如下截图所示：

![](img/f6d4dba6-f006-463b-81a7-1ba5f14f6833.png)

1.  完成 IAM 角色设置配置后，您可以单击底部的蓝色“允许”按钮。经过几秒钟的验证后，CloudWatch 日志将被配置，您可以在此处的同一 CloudWatch 日志部分中看到：

![](img/8129a3da-2968-47ea-b627-f3db726c986e.png)

# Lambda 在 CloudWatch 中的指标

由于我们已经学习和了解了 CloudWatch 和 CloudTrail 服务在日志记录和监视方面的工作原理，我们将继续尝试为我们的 Lambda 函数实现它们。在本节中，您将了解 Lambda 拥有的 CloudWatch 监控的指标类型，并学习如何创建包含所有这些指标的仪表板。

与本章和本书中的先前部分类似，我们将尝试以以下步骤的形式理解概念：

1.  当您导航到 AWS Lambda 控制台时，您将看到您已经创建的 Lambda 函数在可用函数列表中：

![](img/b2c86ede-f9c1-4044-b7bf-bc4e5627be16.png)

1.  当您单击函数时，您将在顶部看到两个可用选项：配置和监视。导航到监视部分。您将看到包含以下内容的指标仪表板：

+   调用

+   持续时间

+   错误

+   节流

+   迭代器年龄

+   DLQ 错误

![](img/45c11785-1c42-40f6-9a4d-5dab9bdadb50.png)

调用和持续时间

![](img/6ec83c08-9292-429f-b6c1-dcd95e98e439.png)

错误和节流

![](img/f494d4c8-8651-4c69-8d14-7d47ca414b94.png)

迭代器年龄和 DLQ 错误

1.  让我们逐一详细了解每一个。第一个指标是调用指标，*x*轴上是时间，*y*轴上是 Lambda 函数的调用次数。该指标帮助我们了解 Lambda 函数何时以及多少次被调用：

![](img/97cc9acf-9a68-4bbe-99dc-17d3ef287af2.png)

单击“跳转到日志”将带您到 Lambda 调用的 CloudWatch 日志控制台，看起来像这样：

![](img/7ce781ee-0bf6-48a6-b992-eac28a1924db.png)

当您单击“跳转到指标”选项时，它将带您到该特定指标的 CloudWatch 指标仪表板，该仪表板为您提供了同一指标的更加定制和细粒度的图表，看起来像这样：

![](img/87642e50-d03f-4d24-abfe-78bba3540f4f.png)

1.  Lambda 监控仪表板中的第二个指标是持续时间指标，它告诉您每次调用 Lambda 函数的持续时间。它还将时间作为*X*轴，并以毫秒为单位在*Y*轴上显示持续时间。它还告诉您在一段时间内 Lambda 函数的最大、平均和最小持续时间：

![](img/2d2da06d-39b3-420f-b81d-735af4e8f75a.png)

1.  再次单击“跳转到日志”按钮将带您到与先前指标相同的页面。单击“跳转到指标”按钮将带您到持续时间指标的 CloudWatch 指标页面，看起来像这样：

![](img/b79022ab-c988-4ba6-80d1-dd7a5c4e69ce.png)

1.  第三个指标是错误指标，它帮助我们查看 Lambda 函数调用中的错误。*Y*轴是错误数量，*X*轴是时间轴：

![](img/8973a822-31e5-4b64-bbd7-93dac36607b5.png)

1.  单击“跳转到指标”链接，可以看到相同指标的 CloudWatch 仪表板：

![](img/25fa346e-6462-4fe2-a095-99dc0b74e996.png)

1.  第四个指标是节流。这个指标计算了您的 Lambda 函数被节流的次数，也就是函数的并发执行次数超过了每个区域的设定限制 1,000 次的次数。我们不会经常遇到这个指标，因为我们在本书中构建的 Lambda 函数示例都远远低于并发限制：

![](img/531b1e5a-cf88-4c3f-b9f8-f9bab88de0ef.png)

1.  通过单击跳转到指标链接，我们还可以在我们的 CloudWatch 指标仪表板中看到这个指标：

![](img/c45abd31-7c67-47cb-943d-41607e64ac3f.png)

1.  第五个指标是迭代器年龄。这仅对由 DynamoDB 流或 Kinesis 流触发的函数有效。它给出了函数处理的最后一条记录的年龄：

![](img/1d65a61c-c6c5-49bd-a23c-dd9a014a0b3e.png)

跳转到指标链接将带您到此指标的 CloudWatch 指标仪表板：

![](img/335da45d-9628-435e-b9db-b7c9a5c6d562.png)

1.  第六个也是最后一个指标是 DLQ 错误指标。这给出了在将消息（失败的事件负载）发送到死信队列时发生的错误数量。大多数情况下，错误是由于故障的权限配置和超时引起的：

![](img/089213e4-6e0e-4b56-aa5e-56af3eb17f52.png)

单击跳转到指标链接将带您到相同指标的 CloudWatch 指标仪表板：

![](img/062ee619-3b95-4949-85b3-054c50481d1e.png)

# CloudWatch 中的 Lambda 日志

到目前为止，我们已经非常详细地了解了 AWS Lambda 的指标。现在，我们将继续了解 Lambda 函数的日志。与往常一样，我们将尝试通过以下步骤来理解它们：

1.  AWS Lambda 函数的日志存储在 CloudWatch 的日志服务中。您可以通过单击主 CloudWatch 仪表板上的日志仪表板来访问 CloudWatch 日志服务。

1.  当您单击服务器端 API 的日志，/aws/lambda/serverless-api，在列表中，我们转到无服务器 API 的日志流，它看起来像这样：

![](img/40d6b9b7-3963-428a-8659-c65aace1b751.png)

1.  这里的每个日志流都是一个 Lambda 调用。因此，每当您的 Lambda 函数被调用时，它都会在这里创建一个新的日志流。如果调用是 Lambda 的重试过程的一部分，那么该特定调用的日志将被写入最近的日志流下。单个日志流可以包含多个细节。但首先，让我们看看特定的日志流是什么样子的：

![](img/9b532fda-5179-4224-bb3b-8b3368ae8fba.png)

1.  此外，如果您仔细观察，您会发现 Lambda 的日志还提供有关 Lambda 函数调用的持续时间、计费持续时间以及函数使用的内存的信息。这些指标有助于更好地了解我们函数的性能，并进行进一步的优化和微调：

![](img/5737c445-d61e-4dfe-9c50-06eb9be58965.png)

1.  CloudWatch 日志中有几列可供选择，这些列在前面的截图中没有显示。这些是可用选项：

![](img/b8580207-9fe8-444e-aa6b-7b34ee5e997e.png)

因此，当您选择更多的选项时，您将在仪表板中看到它们作为列。当您对我们的 Lambda 函数进行更精细的调试时，这些选项非常有用：

![](img/613ad444-0922-42a9-bf3b-43c226ad3014.png)

# Lambda 中的日志记录语句

清楚地记录您的评论和错误始终是一个良好的软件实践。因此，我们现在将了解如何从 Lambda 函数内部记录日志。在 Lambda 函数内部记录日志有两种广泛的方法。我们现在将通过以下步骤的示例来学习和理解它们：

1.  第一种方法是使用 Python 的`logging`库。这在 Python 脚本中作为标准的日志记录实践被广泛使用。我们将编辑之前为无服务器 API 编写的代码，并在其中添加日志记录语句。代码将如下所示：

![](img/efddfce9-0f57-47c1-b818-f2e354229456.png)

在前面的截图中的代码如下：

```py
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
def lambda_handler(event, context):
 mobs = {
 "Sea": ["GoldFish", "Turtle", "Tortoise", "Dolphin", "Seal"],
 "Land": ["Labrador", "Cat", "Dalmatian", "German Shepherd",
 "Beagle", "Golden Retriever"],
 "Exotic": ["Iguana", "Rock Python"]
 }

 logger.info('got event{}'.format(event))
 logger.error('something went wrong')

 return 'Hello from Lambda!'
 #return {"type": mobs[event['type']]}
```

1.  现在，当您保存后运行 Lambda 函数，您可以看到一个绿色的成功执行语句，看起来像这样：

![](img/947377e1-12e7-4197-844a-9f7433a0a6b6.png)

1.  当您点击“详细”选项时，您可以清楚地看到执行日志语句：

![](img/dac4ea52-e0f2-4246-8c51-aaedc2db9558.png)

1.  记录语句的下一种方式是简单地在 Python 中使用`print`语句。这是在 Python 脚本中打印日志语句的最常见方式。因此，我们将在我们的函数代码中添加一个`Hello from Lambda`的打印语句，看看我们是否在 Lambda 执行中获得日志：

![](img/bfdb14b4-bbaf-4f5f-b5e1-ddef710ef470.png)

此 Lambda 函数的代码如下：

```py
 def lambda_handler(event, context):
 mobs = {
     "Sea": ["GoldFish", "Turtle", "Tortoise", "Dolphin", "Seal"],
     "Land": ["Labrador", "Cat", "Dalmatian", "German Shepherd",
     "Beagle", "Golden Retriever"],
     "Exotic": ["Iguana", "Rock Python"]
}
print 'Hello from Lambda!'
return 1
#return {"type": mobs[event['type']]}
```

1.  当我们点击“测试”来执行代码时，我们应该看到一个绿色的消息，表示成功执行：

![](img/935ec9df-157e-478f-8eae-ca699d5d7b0c.png)

1.  同样，就像之前所做的那样，点击“详细”切换将给您完整的执行日志：

![](img/912656a0-9fa1-4946-a70d-0989ce6f2563.png)

1.  我们也可以看到`Hello from Lambda`的消息。对于我们的 Lambda 函数有两种可用的日志记录选项，始终最好使用第一种选项，即通过 Python 的日志记录模块。这是因为该模块提供了更大的灵活性，并帮助您区分信息、错误和调试日志。

# 摘要

在本章中，我们已经了解了 AWS 的监控和日志记录功能。我们还了解了 AWS 环境中可用的监控和日志记录工具。我们还学习了如何监控我们的 Lambda 函数以及如何为我们的 Lambda 函数设置日志记录。

我们已经了解了行业遵循的日志记录和监控实践，以及在 Lambda 函数内部记录语句的各种方式。

在下一章中，我们将学习如何扩展我们的无服务器架构，使其变得分布式，并能够处理大规模的工作负载，同时仍然保留无服务器设置的优点。
