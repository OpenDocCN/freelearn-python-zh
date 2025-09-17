# 第十章. 使用基于云的物联网分析分析大量数据

在本章中，我们将使用英特尔物联网分析，利用这个强大的基于云的服务来分析大量数据。我们将：

+   理解物联网与大数据之间的关系

+   学习英特尔物联网分析的结构

+   在英特尔物联网分析中设置设备

+   在英特尔物联网分析中配置组件

+   使用英特尔物联网分析收集传感器数据

+   使用英特尔物联网分析分析传感器数据

+   在英特尔物联网分析中使用规则触发警报

# 理解物联网与大数据之间的关系

大数据在监视我们。每次我们执行一个动作，即使我们不知道，我们都在生成有价值的数据。每次我们轻触、点击、发推文、在红灯信号前停下、乘坐公交车，或者在世界任何城市的数百万个实时传感器捕捉到的任何动作，我们都在生成有价值的数据。我们与具有传感器的物联网设备互动，收集数据并将其发布到云端。为了分析和处理大数据，管理者、架构师、开发人员和系统管理员需要许多在处理较小数据集的应用程序中并不必要的技能。

我们一直在使用示例，通过传感器从现实世界收集数据并将其发布到云端。我们还发布了包含传感器数据和必须由物联网设备上运行的代码处理的命令的消息。有时，我们每秒从传感器中检索数据。因此，很容易意识到我们正在生成大量数据，因此，学习与大数据相关的大量知识非常重要。物联网包含大数据。

假设我们编写了在英特尔 Galileo Gen 2 板上运行的 Python 代码，并且每秒执行以下操作：

+   从温度和湿度传感器中读取测量的环境温度

+   从温度和湿度传感器中读取测量的环境湿度水平

+   从测量值位于不同位置的十个土壤湿度传感器中读取测量的体积水含量

+   发布包含环境温度、环境湿度和十个体积水含量的消息

可能首先出现在我们脑海中的是我们需要连接到板上的传感器数量。让我们假设所有的传感器都是数字传感器，并且我们必须将它们连接到 I²C 总线上。只要所有传感器都有不同的 I²C 总线地址，我们就可以将数字温度和湿度传感器以及十个土壤湿度传感器连接到 I²C 总线上。我们只需要确保我们可以配置土壤湿度传感器的 I²C 总线地址，并且我们可以为这些传感器中的每一个分配不同的 I²C 地址。

Catnip Electronics 设计了一种数字土壤湿度传感器，它提供了一个 I²C 接口，其一个特性是它允许更改 I²C 地址。该传感器的默认 I²C 地址为 0x20（十六进制 20），但我们可以轻松地更改它。我们只需将每个传感器连接到 I²C 总线，将新地址写入寄存器，并在重置传感器后，新地址将生效。我们只需将 6 写入传感器的 I²C 地址即可重置传感器。我们可以对所有的传感器执行相同的程序，并为它们分配不同的 I²C 地址。您可以在以下网页上了解更多关于数字土壤湿度传感器的信息：[`www.tindie.com/products/miceuz/i2c-soil-moisture-sensor`](http://www.tindie.com/products/miceuz/i2c-soil-moisture-sensor)。

我们希望分析每小时、每天、每月、每季度和每年的数据。然而，我们确实需要每秒测量一次，而不是每天测量一次，因为分析数据每秒的变化非常重要。我们将收集以下数据：

+   每分钟对所有变量进行 60 次测量

+   每小时 3,600（60 * 60）次测量

+   每天进行 86,400（3,600 x 24）次测量

+   每年 31,536,000（86,400 * 365）次测量（考虑到我们不是在谈论闰年）

我们不会只有一个物联网设备收集数据并发布它。我们将有 3,000 个物联网设备运行相同的代码，并且它们每年将生成 94,608,000,000（31,356,300 * 3,000），即九百四十亿六千八百万次测量。此外，我们还有其他必须分析的数据来源：所有关于传感器捕获数据位置天气相关问题的推文。因此，我们拥有大量结构化和非结构化数据，我们希望对其进行计算分析，以揭示模式和关联。我们无疑是在谈论大数据实践。

样本数量有助于理解大数据和物联网之间的关系。在下一个示例中，我们不会部署 3,000 块板，也不会涵盖与物联网分析和大数据相关的所有主题，因为这超出了本书的范围。然而，我们将使用基于云的分析系统，该系统与我们在第二章中使用的英特尔物联网开发套件镜像中的组件一起工作，以使用 Yocto Linux 元分布启动板。

# 理解英特尔物联网分析结构

假设我们需要收集和分析 3,000 个物联网设备的数据，即 3,000 个运行与传感器交互的 Python 代码的 Intel Galileo Gen 2 板。我们需要投资存储和处理能力来处理如此大量的物联网分析数据。每当我们有类似的需求时，我们可以利用基于云的解决方案。Intel IoT Analytics 就是其中之一，它与 Intel Galileo Gen 2 板和 Python 非常兼容。

Intel IoT Analytics 要求我们注册，使用有效的电子邮件地址和密码创建账户，并在发布传感器数据之前点击确认电子邮件的激活链接。我们不需要输入任何信用卡或支付信息。如果您已经在 Intel IoT Analytics 上有账户，您可以跳过此步骤。您也可以使用现有的 Facebook、Google+ 或 GitHub 账户登录。以下是 Intel IoT Analytics 网站的主页：[`dashboard.us.enableiot.com`](https://dashboard.us.enableiot.com)。在使用此基于云的服务处理敏感数据之前，请务必查看条款和条件。

一旦您在 Intel IoT Analytics 上创建账户并首次登录，网站将显示**创建新账户**页面。输入您希望用于识别账户的名称，即您的分析项目在**账户名称**中。以我们的示例为例，输入`温度和湿度`，并保留**传感器健康报告**的默认选项。然后，点击**创建**，网站将显示为最近创建的账户的**我的仪表板**页面。每个账户代表一个独立的工作空间，拥有自己的传感器和相关数据集。网站允许我们创建多个账户，并轻松地在它们之间切换。以下截图显示了创建新账户后**我的仪表板**页面的初始视图：

![理解 Intel IoT Analytics 结构](img/B05042_10_01.jpg)

**我的仪表板**页面表明我们尚未注册任何设备，因此我们也没有传输设备或观测数据。每次我们从注册设备向 Intel IoT Analytics 发布数据时，我们都会为该设备创建一个观测数据。因此，**我的仪表板**页面提供了特定时间段内最后观测数据的数量。默认情况下，页面显示所有注册设备过去一小时的观测数据总和。请保持您的网页浏览器打开，因为我们稍后将继续使用它。

作为用户，我们可以与许多账户一起工作。每个账户可以包含许多设备，有一个名称和一个称为`accountId`的标识符。每个设备都有一个称为`deviceId`的全局唯一标识符。因此，每个包含传感器的英特尔 Galileo Gen 2 板将为我们创建的账户成为一台设备。在我们的情况下，我们只需与一个英特尔 Galileo Gen 2 板一起工作。然而，请记住，我们的目标是展示我们如何与单个账户处理的 3,000 个物联网设备一起工作。

我们可以将每个设备视为一个端点，该端点包含一个或多个组件，这些组件可以在英特尔物联网分析中提供以下之一：

+   **执行器**：可以在设备上修改的设置。例如，旋转伺服轴的角度或打开 LED。

+   **时间序列**：从传感器捕获的一系列值，即观测值的集合。例如，使用温度和湿度传感器检索的环境温度值集合，以华氏度表示，包括时间戳。

在我们的情况下，我们需要一个设备来使用以下组件，这些组件将从连接到我们板上的数字温度和湿度传感器中检索值：

+   一个以华氏度（ºF）表示的环境温度观测的时间序列

+   一个以摄氏度（ºC）表示的环境温度观测的时间序列

+   一个以百分比表示的环境湿度水平观测的时间序列

首先，我们将结合使用英特尔物联网分析网站提供的 UI 和`iotkit-admin`实用程序来设置设备、激活它并注册之前列表中包含的三个组件。这样，我们将学会与英特尔物联网分析所需的结构一起工作。然后，我们将编写 Python 代码，使用 REST API 为属于我们最近创建的账户中已激活设备的定义组件创建观测值。

我们也可以通过编写 Python 代码来使用 REST API 执行之前解释的设置任务。如果我们必须与超过一打设备一起工作，我们不会想通过使用英特尔物联网分析网站提供的 UI 来执行设置任务，我们肯定会想编写自动化设置任务的代码。

# 在英特尔物联网分析中设置设备

我们用来启动英特尔 Galileo Gen 2 板的映像中预装了英特尔物联网分析的本地代理。除非我们对 Yocto Linux meta 分布进行了特定更改以禁用特定组件，否则代理将以守护进程的形式在设备上运行。代理包括`iotkit-admin`命令行实用程序，允许我们与英特尔物联网分析进行特定交互。我们将使用此命令行实用程序执行以下任务：

+   测试与英特尔物联网分析的正确通信

+   获取设备 ID

+   激活设备

+   为设备注册三个时间序列组件。

+   发送测试观测值

首先，我们将检查`iotkit-admin`命令行工具是否能够与英特尔物联网分析服务建立适当的通信。我们只需在 SSH 终端中运行以下命令：

```py
iotkit-admin test
```

如果连接成功，我们将看到类似以下的一些行。最后一行提供了有关构建的信息，即版本。

```py
2016-04-05T02:17:49.573Z - info: Trying to connect to host ...
2016-04-05T02:17:56.780Z - info: Connected to dashboard.us.enableiot.com
2016-04-05T02:17:56.799Z - info: Environment: prod
2016-04-05T02:17:56.807Z - info: Build: 0.14.5
```

现在，在 SSH 终端中运行以下命令以获取设备 ID，也称为`deviceId`：

```py
iotkit-admin device-id
```

上一条命令将生成类似以下行的输出行，其中包含设备 ID。默认情况下，设备 ID 等于网络接口卡的 MAC 地址。

```py
2016-04-05T02:23:23.170Z - info: Device ID: 98-4F-EE-01-75-72
```

您可以使用以下命令将设备 ID 更改为不同的一个：`iotkit-admin set-device-id new-device-id`。您只需将`new-device-id`替换为您为设备设置的新的设备 ID。然而，请注意，新的设备 ID 必须是一个全局唯一标识符。

在这种情况下，我们将使用`kansas-temperature-humidity-01`作为我们所有示例的设备 ID。您必须在所有命令中替换它，然后包括您检索到的设备名称或分配给设备的新的设备 ID。

在 SSH 终端中运行的以下命令将重命名设备：

```py
iotkit-admin set-device-id kansas-temperature-humidity-01
```

以下行显示了上一条命令的输出：

```py
2016-04-08T17:56:15.355Z - info: Device ID set to: kansas-temperature-humidity
```

前往您正在使用的英特尔物联网分析仪表板的网页浏览器，点击菜单图标（位于左上角的一个有三个横线的按钮）。选择**账户**，网站将显示我们之前创建的账户的**我的账户**页面，其中包含详细的账户信息。

初始视图将显示**详细信息**选项卡。如果**激活码**包含**（代码已过期）**文本，则意味着激活码已不再有效，并且有必要点击位于**激活码**文本框右侧的刷新图标（第二个带有两个箭头的图标）。我们必须确保激活码尚未过期，以便成功激活设备。以下截图显示了**我的账户**页面的初始视图，对于**温度和湿度**账户，激活码已过期：

![在英特尔物联网分析中设置设备](img/B05042_10_02.jpg)

一旦您通过点击刷新按钮刷新激活码，倒计时计时器将指示激活码到期前的剩余时间。您在点击刷新按钮后将有一个小时。点击眼睛图标以查看隐藏的激活码并复制它。我们将使用`01aCti0e`作为我们的示例激活码，您必须将其替换为您自己的激活码。

现在，在 SSH 终端中运行以下命令以使用之前生成的激活码激活设备。将`01aCti0e`替换为您的激活码。

```py
iotkit-admin activate 01aCti0e
```

上一条命令将生成类似以下行的输出：

```py
2016-04-05T02:24:46.449Z - info: Activating ...
2016-04-05T02:24:49.817Z - info: Saving device token...
2016-04-05T02:24:50.646Z - info: Updating metadata...
2016-04-05T02:24:50.691Z - info: Metadata updated.
```

我们现在将英特尔 Galileo Gen 2 板，即设备，与提供激活代码并生成必要安全凭证（即设备令牌）的**温度和湿度**账户关联起来。

打开您正在使用英特尔物联网分析仪表板的浏览器，点击菜单图标（位于左上角的一个有三个水平线的按钮）。选择**设备**，网站将显示包含我们为当前账户激活的所有设备的**我的设备**页面。之前激活的 `kansas-temperature-humidity-01` 设备将出现在列表中，其**名称**列为**Kansas-temperature-humidity-01-NAME**，**状态**列为**active**。以下截图显示了**我的设备**页面中的设备列表：

![在英特尔物联网分析中设置设备](img/B05042_10_03.jpg)

点击前一个列表中的设备**ID**（**kansas-temperature-humidity-01**）以查看和编辑设备详细信息。您可以添加标签和属性，以便更容易过滤前一个列表中的设备。当我们需要处理超过一打设备时，这些功能非常有用，因为它们使我们能够轻松过滤列表中的设备。

# 在英特尔物联网分析中设置组件

打开您正在使用英特尔物联网分析仪表板的浏览器，点击菜单图标，选择**账户**，网站将显示**我的账户**页面。然后，点击**目录**选项卡，网站将显示按以下三个类别分组的目录中注册的组件：

+   湿度

+   Powerswitch

+   温度

确保已展开**湿度**组件面板，然后点击**humidity.v1.0**。网站将显示**humidity.v1.0**组件的**组件定义**对话框，即版本为 `1.0` 的名为**humidity**的组件。以下截图显示了组件定义中不同字段的值：

![在英特尔物联网分析中设置组件](img/B05042_10_04.jpg)

版本为 `1.0` 的 `humidity` 组件表示一个以百分比表示的环境湿度水平的时间序列。**数据类型**是**数字**，**计量单位**是**百分比 (%)**，**格式**是**浮点数**，**显示**为**时间序列**。我们可以使用此组件来观察我们的环境湿度水平。

点击**关闭**，确保**温度**组件面板已展开，然后点击**temperature.v1.0**。网站将显示**temperature.v1.0**组件的**组件定义**对话框，即版本为 `1.0` 的名为**temperature**的组件。以下截图显示了组件定义中不同字段的值：

![在英特尔物联网分析中设置组件](img/B05042_10_05.jpg)

`temperature`组件版本**1.0**代表一个以摄氏度表示的温度的时间序列。**数据类型**是**数字**，**测量单位**是**摄氏度**，**格式**是**浮点数**，**显示**是**时间序列**。我们可以使用此组件来表示以摄氏度表示的环境温度观测。

点击**关闭**并确保**温度**组件面板已展开。没有其他温度组件，因此，我们必须为以华氏度表示的环境温度观测创建一个新的组件。

在页面底部点击**添加新目录项**，网站将显示包含所有字段为空的**组件定义**对话框，除了版本号将具有固定的`1.0`值。我们正在创建新目录项的第一个版本。输入并选择以下值：

+   在**组件名称**中输入**temperaturef**

+   在**类型**中选择**传感器**

+   在**数据类型**中选择**数量**

+   在**测量单位**中输入**华氏度**

+   在**格式**中选择**浮点数**

+   在**显示**中选择**时间序列**

最后，点击**保存**，网站将在列表底部添加名为`temperaturef.v.1.0`的新组件定义。

现在我们确信目录中包含所有必需的组件定义，我们必须注册设备将用于创建观测的组件。我们必须为每个注册的组件提供一个名称或别名，并必须指定来自上一目录的组件类型和版本。以下表格总结了我们将为设备注册的组件：

| 组件名称或别名 | 组件类型 | 描述 |
| --- | --- | --- |
| `temperaturec` | temperature.v1.0 | 一个以摄氏度（ºC）表示的环境温度观测的时间序列 |
| `temperaturef` | temperaturef.v1.0 | 一个以华氏度（ºF）表示的环境温度观测的时间序列 |
| `humidity` | humidity.v1.0 | 一个以百分比表示的环境湿度水平观测的时间序列 |

我们可以使用以下命令来注册每个组件：`iotkit-admin register component-name component-type`。我们只需要将`component-name`替换为将标识组件的名称，将`component-type`替换为标识组件类型（包括版本号）的名称。

在 SSH 终端中的以下命令将注册来自上一表的`temperaturec`组件：

```py
iotkit-admin register temperaturec temperature.v1.0
```

以下行显示了上一条命令的输出。

```py
2016-04-08T22:40:04.581Z - info: Starting registration ...
2016-04-08T22:40:04.711Z - info: Device has already been activated. Updating ...
2016-04-08T22:40:04.739Z - info: Updating metadata...
2016-04-08T22:40:04.920Z - info: Metadata updated.
Attributes sent
2016-04-08T22:40:10.167Z - info: Component registered name=temperaturec, type=temperature.v1.0, cid=c37cb57d-002c-4a66-866e-ce66bc3b2340, d_id=kansas-temperature-humidity-01

```

最后一行为我们提供了组件 ID，即`cid=`之后和下一个逗号（`,`）之前的值。在上一个输出中，组件 ID 是`c37cb57d-002c-4a66-866e-ce66bc3b2340`。我们必须保存每个组件 ID，因为我们稍后需要它来编写使用 REST API 创建观测的代码。

在 SSH 终端中，以下命令将注册来自上一张表的`temperaturef`组件：

```py
iotkit-admin register temperaturef temperaturef.v1.0
```

以下几行显示了上一条命令的输出结果：

```py
2016-04-08T22:40:20.510Z - info: Starting registration ...
2016-04-08T22:40:20.641Z - info: Device has already been activated. Updating ...
2016-04-08T22:40:20.669Z - info: Updating metadata...
2016-04-08T22:40:20.849Z - info: Metadata updated.
Attributes sent
2016-04-08T22:40:26.156Z - info: Component registered name=temperaturef, type=temperaturef.v1.0, cid=0f3b3aae-ce40-4fb4-a939-e7c705915f0c, d_id=kansas-temperature-humidity-01

```

与其他命令一样，最后一行为我们提供了组件 ID，即`cid=`之后和下一个逗号（`,`）之前的值。在上一个输出中，组件 ID 是`0f3b3aae-ce40-4fb4-a939-e7c705915f0c`。我们必须保存这个 ID，以便在代码中的后续使用。

在 SSH 终端中，以下命令将注册来自上一张表的`humidity`组件：

```py
iotkit-admin register humidity humidity.v1.0
```

以下几行显示了上一条命令的输出结果，最后一行包括组件 ID。

```py
2016-04-08T22:40:36.512Z - info: Starting registration ...
2016-04-08T22:40:36.643Z - info: Device has already been activated. Updating ...
2016-04-08T22:40:36.670Z - info: Updating metadata...
2016-04-08T22:40:36.849Z - info: Metadata updated.
Attributes sent
2016-04-08T22:40:43.003Z - info: Component registered name=humidity, type=humidity.v1.0, cid=71aba984-c485-4ced-bf19-c0f32649bcee, d_id=kansas-temperature-humidity-01

```

### 小贴士

组件 ID 将与之前输出中指示的值不同，您需要记录由之前命令生成的每个组件 ID。

访问您正在使用的 Intel IoT Analytics 仪表板的网页浏览器，点击菜单图标。选择**设备**，站点将显示**我的设备**页面。点击之前列表中的设备**ID**（**kansas-temperature-humidity-01**）以查看和编辑设备详细信息。点击**+组件**以展开为设备注册的组件，您将看到一个包含以下三个组件的列表：

+   **temperaturec**

+   **temperaturef**

+   **humidity**

以下截图显示了为所选设备注册的三个组件：

![在 Intel IoT Analytics 中设置组件](img/B05042_10_06.jpg)

我们可以点击这三个组件中的任何一个，并检查注册组件的详细信息。如果我们丢失了组件 ID，我们可以通过点击组件并显示**组件定义**对话框来检索它，组件 ID 将显示在组件类型描述下方。以下截图显示了`temperaturef`组件的组件定义。组件 ID **0f3b3aae-ce40-4fb4-a939-e7c705915f0c**出现在右侧的**自定义组件**标签下方。

![在 Intel IoT Analytics 中设置组件](img/B05042_10_07.jpg)

不幸的是，我们没有方法检索在激活设备时生成的设备令牌，该设备具有站点中包含的功能。我们需要设备令牌来为注册的组件创建观测值。Intel IoT Analytics 代理将设备令牌与其他配置值一起保存在`device.json`文件中，其默认路径是`/usr/lib/node_modules/iotkit-agent/data/device.json`。正如文件名所暗示的，该文件包含 JSON 代码。我们只需在 SSH 终端中运行以下命令，即可显示上一个文件的文本内容，并允许我们检索设备令牌：

```py
cat /usr/lib/node_modules/iotkit-agent/data/device.json
```

以下几行显示了上一条命令的输出结果，其中包括我们迄今为止为设备所做的所有配置。定义设备令牌值的行被突出显示。

```py
{
    "activation_retries": 10,
    "activation_code": null,
    "device_id": "kansas-temperature-humidity-01",
    "device_name": false,
    "device_loc": [
        88.34,
        64.22047,
        0
    ],
    "gateway_id": false,
 "device_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJqdGkiOiJjOTNmMTJhMy05MWZlLTQ3MWYtODM4OS02OGM1NDYxNDIxMDUiLCJpc3MiOiJodHRwOi8vZW5hYmxlaW90LmNvbSIsInN1YiI6ImthbnNhcy10ZW1wZXJhdHVyZS1odW1pZGl0eS0wMSIsImV4cCI6IjIwMjYtMDQtMDZUMTk6MDA6MTkuNzA0WiJ9.PH5yQas2FiQvUSR9V2pa3n3kIYZvmSe_xXY7QkFjlXUVUcyy9Sk_eVF4AL6qpZlBC9vjtd0L-VMZiULC9YXxAVl9s5Cl8ZqpQs36E1ssv_1H9CBFXKiiPArplzaWXVzvIRBVVzwfQrGrMoD_l4DcHlH2zgn5UGxhZ3RMPUvqgeneG3P-hSbPScPQL1pW85VT2IHT3seWyW1c637I_MDpHbJJCbkytPVpJpwKBxrCiKlGhvsh5pl4eLUXYUPlQAzB9QzC_ohujG23b-ApfHZugYD7zJa-05u0lkt93EEnuCk39o5SmPmIiuBup-k_mLn_VMde5fUvbxDt_SMI0XY3_Q",
    "account_id": "22612154-0f71-4f64-a68e-e116771115d5",
    "sensor_list": [
        {
            "name": "temperaturec",
            "type": "temperature.v1.0",
            "cid": "c37cb57d-002c-4a66-866e-ce66bc3b2340",
            "d_id": "kansas-temperature-humidity-01"
        },
        {
            "name": "temperaturef",
            "type": "temperaturef.v1.0",
            "cid": "0f3b3aae-ce40-4fb4-a939-e7c705915f0c",
            "d_id": "kansas-temperature-humidity-01"
        },
        {
            "name": "humidity",
            "type": "humidity.v1.0",
            "cid": "71aba984-c485-4ced-bf19-c0f32649bcee",
            "d_id": "kansas-temperature-humidity-01"
        }
    ]
}
```

前几行还显示了我们所注册的每个组件的组件 ID。因此，我们只需在一个地方就可以找到所有必要的配置值，这些值将用于我们的代码中。在这种情况下，设备令牌如下，即`"device_token"`键的字符串值。然而，您将检索到的值将不同。

```py
"eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJqdGkiOiJjOTNmMTJhMy05MWZlLTQ3MWYtODM4OS02OGM1NDYxNDIxMDUiLCJpc3MiOiJodHRwOi8vZW5hYmxlaW90LmNvbSIsInN1YiI6ImthbnNhcy10ZW1wZXJhdHVyZS1odW1pZGl0eS0wMSIsImV4cCI6IjIwMjYtMDQtMDZUMTk6MDA6MTkuNzA0WiJ9.PH5yQas2FiQvUSR9V2pa3n3kIYZvmSe_xXY7QkFjlXUVUcyy9Sk_eVF4AL6qpZlBC9vjtd0L-VMZiULC9YXxAVl9s5Cl8ZqpQs36E1ssv_1H9CBFXKiiPArplzaWXVzvIRBVVzwfQrGrMoD_l4DcHlH2zgn5UGxhZ3RMPUvqgeneG3P-hSbPScPQL1pW85VT2IHT3seWyW1c637I_MDpHbJJCbkytPVpJpwKBxrCiKlGhvsh5pl4eLUXYUPlQAzB9QzC_ohujG23b-ApfHZugYD7zJa-05u0lkt93EEnuCk39o5SmPmIiuBup-k_mLn_VMde5fUvbxDt_SMI0XY3_Q"
```

# 使用英特尔物联网分析收集传感器数据

我们将使用我们在第八章中编写的代码，*显示信息和执行操作*，当我们从传感器读取温度和湿度值时，我们在 OLED 矩阵中打印了这些值，并旋转了伺服轴以显示以华氏度表示的测量温度。该示例的代码文件为`iot_python_chapter_08_03.py`。我们将使用此代码作为基准，添加新功能，使我们能够为我们激活的设备注册的三个组件创建观测值。

在第二章，*在英特尔 Galileo Gen 2 上使用 Python*中，我们确保了`pip`安装程序在板上运行的 Yocto Linux 中可用，以便安装额外的 Python 2.7.3 包。现在，我们将使用`pip`安装程序确保安装了`requests`包。这个包是一个非常流行的 Python HTTP 库，它允许我们使用极其容易理解的语法轻松构建和发送 HTTP 请求。

如果您已经处理了上一章的示例，您将已经安装了此包。然而，如果您刚刚跳到本章，可能需要安装它。我们只需在 SSH 终端中运行以下命令即可安装该包。请注意，安装可能需要几分钟时间。

```py
pip install requests
```

如果您看到以下输出，则表示`requests`包已经安装，您可以继续下一步。

```py
Requirement already satisfied (use --upgrade to upgrade): requests in /usr/lib/python2.7/site-packages
```

我们将创建一个`IntelIotAnalytics`类来表示英特尔物联网分析接口，并使我们能够轻松地为三个组件发布观测值。然而，在我们编写类代码之前，我们必须确保我们可以替换与我们的账户、组件和设备相关的重要值的相关类属性的内容。您必须将以下类属性指定的字符串替换为适当的值：

+   `account_name`: **账户名称**字段在**我的账户**页面中的值。在我们的例子中，我们为账户名称使用了`"Temperature and humidity"`。

+   `account_id`: **账户 ID**字段在**我的账户**页面中的值。在我们的例子中，我们为账户 ID 使用了`"22612154-0f71-4f64-a68e-e116771115d5"`。我们还可以通过读取`device.json`文件中指定的`"account_id"`键的字符串值来检索账户 ID 值。

+   `device_id`: 在我们点击**我的设备**页面中显示的列表中的设备名称时，**添加/编辑设备**页面中**ID**字段的值。在我们的例子中，我们使用`"kansas-temperature-humidity-01"`作为我们的设备 ID。我们也可以通过在 SSH 终端中运行以下命令来检索`device_id`：`iotkit-admin device-id`，或者通过读取`device.json`文件中指定的`"device_id"`键的字符串值。

+   `device_token`: 当我们激活设备时生成的设备令牌的值。如前所述，我们可以通过读取`device.json`文件中指定的`"device_token"`键的字符串值来检索设备令牌。

+   `component_id_temperature_fahrenheit`: 当我们注册了`temperaturef`组件时生成的组件 ID 的值。组件 ID 显示在**组件定义**对话框中组件类型下方。在我们的例子中，我们使用`"0f3b3aae-ce40-4fb4-a939-e7c705915f0c"`作为这个值。我们也可以通过读取`device.json`文件中声明`"name": "temperaturef"`键值对同一块中指定的`"cid"`键的字符串值来检索组件 ID 值。

+   `component_id_temperature_celsius`: 当我们注册了`temperaturec`组件时生成的组件 ID 的值。在我们的例子中，我们使用`"c37cb57d-002c-4a66-866e-ce66bc3b2340"`作为这个值。

+   `component_id_humidity_level_percentage`: 当我们注册了`humidity`组件时生成的组件 ID 的值。在我们的例子中，我们使用`"71aba984-c485-4ced-bf19-c0f32649bcee"`作为这个值。

样本的代码文件是`iot_python_chapter_10_01.py`。请记住，我们使用代码文件`iot_python_chapter_08_03.py`作为基线，因此，我们将把`IntelIotAnalytics`类添加到该文件中的现有代码中，并创建一个新的 Python 文件。以下代码展示了允许我们通过 REST API 发布`temperaturef`、`temperaturec`和`humidity`组件观察结果的`IntelIotAnalytics`类。

```py
import time
import json
import requests

class IntelIotAnalytics:
    base_url = "https://dashboard.us.enableiot.com/v1/api"
    # You can retrieve the following information from the My Account page
    account_name = "Temperature and humidity"
    account_id = "22612154-0f71-4f64-a68e-e116771115d5"
    # You can retrieve the device token with the following command:
    # cat /usr/lib/node_modules/iotkit-agent/data/device.json
    device_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJqdGkiOiJjOTNmMTJhMy05MWZlLTQ3MWYtODM4OS02OGM1NDYxNDIxMDUiLCJpc3MiOiJodHRwOi8vZW5hYmxlaW90LmNvbSIsInN1YiI6ImthbnNhcy10ZW1wZXJhdHVyZS1odW1pZGl0eS0wMSIsImV4cCI6IjIwMjYtMDQtMDZUMTk6MDA6MTkuNzA0WiJ9.PH5yQas2FiQvUSR9V2pa3n3kIYZvmSe_xXY7QkFjlXUVUcyy9Sk_eVF4AL6qpZlBC9vjtd0L-VMZiULC9YXxAVl9s5Cl8ZqpQs36E1ssv_1H9CBFXKiiPArplzaWXVzvIRBVVzwfQrGrMoD_l4DcHlH2zgn5UGxhZ3RMPUvqgeneG3P-hSbPScPQL1pW85VT2IHT3seWyW1c637I_MDpHbJJCbkytPVpJpwKBxrCiKlGhvsh5pl4eLUXYUPlQAzB9QzC_ohujG23b-ApfHZugYD7zJa-05u0lkt93EEnuCk39o5SmPmIiuBup-k_mLn_VMde5fUvbxDt_SMI0XY3_Q"
    device_id = "kansas-temperature-humidity-01"
    component_id_temperature_fahrenheit = "0f3b3aae-ce40-4fb4-a939-e7c705915f0c"
    component_id_temperature_celsius = "c37cb57d-002c-4a66-866e-ce66bc3b2340"
    component_id_humidity_level_percentage = "71aba984-c485-4ced-bf19-c0f32649bcee"

    def publish_observation(self,
                            temperature_fahrenheit,
                            temperature_celsius,
                            humidity_level):
        url = "{0}/data/{1}".\
            format(self.__class__.base_url, self.__class__.device_id)
        now = int(time.time()) * 1000
        body = {
            "on": now,
            "accountId": self.__class__.account_id,
            "data": []
        }
        temperature_celsius_data = {
            "componentId": self.__class__.component_id_temperature_celsius,
            "on": now,
            "value": str(temperature_celsius)
        }
        temperature_fahrenheit_data = {
            "componentId": self.__class__.component_id_temperature_fahrenheit,
            "on": now,
            "value": str(temperature_fahrenheit)
        }
        humidity_level_percentage_data = {
            "componentId": self.__class__.component_id_humidity_level_percentage,
            "on": now,
            "value": str(humidity_level)
        }
        body["data"].append(temperature_celsius_data)
        body["data"].append(temperature_fahrenheit_data)
        body["data"].append(humidity_level_percentage_data)
        data = json.dumps(body)
        headers = {
            'Authorization': 'Bearer ' + self.__class__.device_token,
            'content-type': 'application/json'
        }
        response = requests.post(url, data=data, headers=headers, proxies={}, verify=True)
        if response.status_code != 201:
            print "The request failed. Status code: {0}. Response text: {1}.".\
                format(response.status_code, response.text)
```

`IntelIotAnalytics`类声明了许多我们之前解释过的类属性，您需要用您自己的字符串值替换它们：`account_name`、`account_id`、`device_token`、`device_id`、`component_id_temperature_fahrenheit`、`component_id_temperature_celsius`和`component_id_humidity_level_percentage`。`base_url`类属性定义了访问 REST API 的基本 URL：`https://dashboard.us.enableiot.com/v1/api`。我们将使用这个值与`data`路径和`device_id`类属性结合来构建我们将发送 HTTP 请求以发布观察结果的 URL。

类声明了`publish_observation`方法，该方法接收华氏度表示的温度、摄氏度表示的温度和湿度百分比，参数为`temperature_fahrenheit`、`temperature_celsius`和`humidity_level`。该方法构建我们将发送 HTTP 请求以创建设备及其三个组件观察值的 URL。URL 由`base_url`类属性、`/data/`和`device_id`类属性组成。正如许多 REST API 所发生的那样，`base_url`类属性指定了 API 的版本号。这样，我们确保我们始终在特定版本上工作，并且我们的请求与该版本兼容。代码将构建的 URL 的值保存到`url`局部变量中。

然后，代码将板当前时间（以秒为单位乘以 1000）保存到`now`局部变量中。代码创建一个`body`字典，该字典代表请求的正文，并包含以下键值对：

+   `"on"`: 存储在`now`局部变量中的值，即板的当前时间。它是观察的时间。

+   `"accountId"`: 存储在`accountId`类属性中的值，即我们将发布观察值的 Intel IoT Analytics 账户。

+   `"data"`: 一个空数组，我们将在稍后用每个组件的一个观察值填充。

然后，代码创建三个字典，包含以下键值对，代表特定组件的观察值：

+   `"componentId"`: 存储在指定要发布观察值的组件 ID 的类属性中的值。

+   `"on"`: 存储在`now`局部变量中的值，即板的当前时间。它是观察的时间。我们使用相同的变量对所有观察值进行操作，因此它们以相同的时间注册。

+   `"value"`: 方法中作为参数接收的值的字符串表示。

然后，代码调用`append`方法将三个字典添加到`body`字典中的`data`键。这样，`data`键将有一个包含三个字典的数组作为其值。代码调用`json.dumps`函数将`body`字典序列化为 JSON 格式的字符串，并将其保存到`data`局部变量中。

下一行创建一个`headers`字典，包含以下键值对，代表 HTTP 请求的头部：

+   `"Authorization"`: 由`"Bearer"`和保存在`device_token`类属性中的设备令牌连接而成的授权字符串

+   `"content-type"`: 声明内容类型为 JSON: `"application/json"`

在这个阶段，代码已经构建了用于发布观测数据到英特尔物联网分析服务的 HTTP 请求的头部和主体。下一行调用`requests.post`函数，向由`url`局部变量指定的 URL 发送一个 HTTP POST 请求，其中`data`字典作为 JSON 主体数据，`headers`字典作为头部。

`requests.post`方法返回一个保存在`response`局部变量的响应，代码评估响应的 code 属性是否不等于 201。如果 code 与 201 不同，这意味着观测数据没有成功发布，即出了问题。在这种情况下，代码将响应的`status_code`和`text`属性的值打印到控制台输出，以便我们了解出了什么问题。如果我们使用错误的设备令牌或账户、设备或组件的 id，我们将收到错误。

现在，我们将使用之前编写的`IntelIoTAnalytics`类来创建一个新的`__main__`方法版本，该方法每 5 秒向英特尔物联网分析服务发布一次观测数据。以下行显示了新的`__main__`方法。示例的代码文件为`iot_python_chapter_10_01.py`。

```py
if __name__ == "__main__":
    temperature_and_humidity_sensor = \
        TemperatureAndHumiditySensor(0)
    oled = TemperatureAndHumidityOled(0)
 intel_iot_analytics = IntelIotAnalytics()
    while True:
        temperature_and_humidity_sensor.\
            measure_temperature_and_humidity()
        oled.print_temperature(
            temperature_and_humidity_sensor.temperature_fahrenheit,
            temperature_and_humidity_sensor.temperature_celsius)
        oled.print_humidity(
            temperature_and_humidity_sensor.humidity)
        print("Ambient temperature in degrees Celsius: {0}".
              format(temperature_and_humidity_sensor.temperature_celsius))
        print("Ambient temperature in degrees Fahrenheit: {0}".
              format(temperature_and_humidity_sensor.temperature_fahrenheit))
        print("Ambient humidity: {0}".
              format(temperature_and_humidity_sensor.humidity))
 intel_iot_analytics.publish_observation(
 temperature_and_humidity_sensor.temperature_fahrenheit,
 temperature_and_humidity_sensor.temperature_celsius,
 temperature_and_humidity_sensor.humidity
 )
 # Sleep 5 seconds (5000 milliseconds)
 time.sleep(5)

```

突出的行显示了创建之前创建的`IntelIoTAnalytics`类实例并将其引用保存到`intel_iot_analytics`局部变量的代码。然后，在每 5 秒运行一次的循环中的代码使用从温度和湿度传感器检索的温度和湿度值作为参数调用`publish_observation`方法。

以下行将启动示例：

```py
python iot_python_chapter_10_01.py
```

运行示例后，打开空调或加热系统，以产生环境温度和湿度的变化。这样，我们将注意到每 5 秒发布的数据中的变化。在我们探索英特尔物联网分析服务中包含的不同功能时，请保持代码运行。

前往您正在使用的英特尔物联网分析仪表板的网页浏览器，点击菜单图标并选择**仪表板**。网站将显示**我的仪表板**页面，它将指示您有一个活动设备，并且当它从板子上接收观测数据时，它将更新过去一小时发布的观测数据数量。以下图片显示了包含活动设备和包含过去一小时发布`945`次观测数据的计数器的仪表板：

![使用英特尔物联网分析服务收集传感器数据](img/B05042_10_08.jpg)

保持浏览器打开并显示仪表板视图，您会注意到在过去一小时中观察值增加，因为代码在板上继续运行。您可以在显示观察数量和上下文菜单的面板右上角点击配置图标，上下文菜单允许您配置在此面板中查看的观察周期。例如，您可以将**上一小时**更改为**上一周**以显示设备在过去一周内注册的观察数量。

# 使用英特尔物联网分析工具分析传感器数据

英特尔物联网分析工具允许我们为具有特定设备观察数据的每个组件生成图表。首先，我们必须选择设备，然后我们必须选择一个或多个组件以生成带有历史时间序列或由板上运行的代码生成的时间序列的图表，即组件的实时数据。

打开您正在使用英特尔物联网分析仪表板的浏览器，点击菜单图标并选择**图表**。网站将显示**我的图表**页面，允许您使用许多搜索条件搜索设备，例如设备名称、相关标签及其属性。

在这种情况下，我们只有一个激活的设备，因此我们可以从网站在**选择设备**部分下方显示的设备列表中选择设备。此部分在复选框的右侧显示设备名称的首个字符，并在文本的右侧显示已为此设备注册的组件数量。以下图片显示了**选择设备**部分，其中**kansas-temp…**代表`kansas-temperature-humidity-01`设备。如果您将鼠标悬停在复选框上或轻触文本，网站将显示一个弹出窗口，显示设备的完整名称和已注册组件的类型。以下截图显示了包含此信息的弹出窗口，显示在**kansas-temp…**复选框上。

![使用英特尔物联网分析工具分析传感器数据](img/B05042_10_09.jpg)

选择**kansas-temp…**复选框，网站将显示所选设备的三个注册组件。在这种情况下，网站显示组件名称（`temperaturec`、`temperaturef`和`humidity`），而在之前解释的弹出窗口中，网站显示了组件类型（`temperature.v1.0`、`temperaturef.v1.0`和`humidity.v1.0`）。

选择`temperaturef`复选框，网站将显示过去一小时测量的环境温度图表，单位为华氏度。默认情况下，图表使用线条并生成过去一小时记录的时间序列值图表。默认情况下，图表的刷新率为 30 秒，因此图表将每 30 秒更新一次，并显示在此期间通过 REST API 由板发布的所有新观测值。

![使用英特尔物联网分析分析传感器数据](img/B05042_10_10.jpg)

我们可以使用图表顶部的不同按钮来更改图表类型，并选择我们想要在图表中显示的时间范围。我们还可以将刷新率更改为低至 5 秒或高达 60 秒。如果我们将图表保存为收藏夹，网站将在**我的仪表板**中将其显示为仪表板的一部分。

点击位于**图表**按钮（一个带有山脉的图片图标）右侧的**原始数据**按钮（一个子弹图标）。网站将显示一个列表，其中包含用于构建时间序列的原始数据，即所选组件接收到的所有观测值。以下截图显示了过去一小时`temperaturef`组件的原始数据视图的第一页。

![使用英特尔物联网分析分析传感器数据](img/B05042_10_11.jpg)

在这个例子中，生成一个包含温度和湿度水平的图表非常有用。通过点击**图表**按钮（一个带有山脉的图片图标）返回图表视图，并勾选湿度复选框。这样，网站将生成一个图表，结合华氏度表示的温度和百分比表示的湿度水平。以下截图显示了当**temperaturef**和**humidity**都勾选时生成的图表：

![使用英特尔物联网分析分析传感器数据](img/B05042_10_12.jpg)

# 使用英特尔物联网分析中的规则触发警报

英特尔物联网分析允许我们定义可以触发以下任何通知类型的规则：

+   电子邮件

+   HTTP 端点

+   动作

打开你正在使用的英特尔物联网分析仪表板的网页浏览器，点击菜单图标并选择**规则**。网站将显示**我的规则**页面，允许你为激活的设备定义规则。在这种情况下，我们将定义一个规则，当湿度水平低于 10%时，将发送电子邮件给我们。

点击**添加规则**，网站将显示一个表单，允许我们输入新规则的详细信息。在**规则名称**中输入`非常低的湿度水平`，在**优先级**中选择`低`，在**通知类型**中选择`电子邮件`。在**通知对象**面板的下拉菜单中选择你想要接收通知的电子邮件地址。

点击**下一步**，网站将要求我们选择要应用新规则的设备。在这种情况下，我们只有一个激活的设备，因此我们可以从网站在**选择设备**部分下方显示的设备列表中选择设备。如前述设备选择页面所示，此部分在复选框的右侧显示设备名称的第一个字符，在文本的右侧显示已为此设备注册的组件数量。勾选**kansas-temp…**复选框，名称将出现在**已选设备**列表中。

点击**下一步**，网站将要求我们指定新规则的条件。不要勾选**启用自动重置**复选框，因为我们希望规则在每次警报后变为不活动状态，直到被确认。这样，在收到警报后，我们只有在确认了第一个生成的警报后才会收到额外的警报。

在**监控度量**中选择`湿度（数字）`，在**触发条件**中选择**基本条件**。然后，在出现的附加下拉菜单中选择`<`，并在**输入值**文本框中输入`10`。这样，我们正在创建一个当湿度观测值低于 10（`湿度 < 10`）时将触发的规则。以下截图显示了定义的条件：

![使用 Intel IoT Analytics 中的规则触发警报](img/B05042_10_13.jpg)

点击**完成**，规则将被添加到**我的规则**中显示的列表中。以下截图显示了定义后包含在此列表中的规则定义：

![使用 Intel IoT Analytics 中的规则触发警报](img/B05042_10_14.jpg)

当湿度低于 10%时，将触发警报，我们将在警报图标（铃铛）中看到一个数字 1。点击图标后，网站将显示我们所有未读的警报。以下截图显示了包含一个未读警报的我的仪表盘页面：

![使用 Intel IoT Analytics 中的规则触发警报](img/B05042_10_15.jpg)

如果我们点击警报，网站将显示触发警报的情况的详细信息。我们还可以在菜单中选择**警报**，查看接收到的警报列表。以下截图显示了包含在接收到的警报列表中的警报：

![使用 Intel IoT Analytics 中的规则触发警报](img/B05042_10_16.jpg)

如果我们点击**警报**编号，网站将显示警报的详细信息，包括触发警报的规则中定义的条件和测量的值。在这种情况下，测量的值是`7.99`。可以向警报添加注释。以下截图显示了警报的详细信息：

![使用 Intel IoT Analytics 中的规则触发警报](img/B05042_10_17.jpg)

此外，我们还会收到一封包含以下文本的电子邮件：

```py
Alert Monitor has received an alert. Alert data: 

- Alert Id: 1 
- Device: kansas-temperature-humidity-01 
- Reason: humidity < 10 
- Priority: Low 

Alert Data
Component Name	Values
humidity	7.99; 

You can go here to check it on Dashboard 
Regards
```

在这种情况下，我们在规则中定义了一个非常简单的条件。然而，我们可以定义一个更复杂的条件，该条件可以包括以下任何条件：

+   基于时间的条件

+   基于统计的条件

+   单传感器变化检测

+   多感官变化检测

我们可以玩转不同的选项，为具有多个传感器和大量数据的众多设备触发警报。英特尔物联网分析最有趣的功能之一是我们可以用图表、规则和警报轻松地处理大量数据。

# 测试你的知识

1.  英特尔物联网分析中每个设备的组件可以是：

    1.  执行器或时间序列。

    1.  账户、执行器或时间序列。

    1.  代理、账户、执行器或时间序列。

1.  每次我们将从注册设备发布数据到英特尔物联网分析时，我们都会创建：

    1.  执行器。

    1.  账户。

    1.  观察结果。

1.  时间序列是：

    1.  执行器执行的一系列动作，即动作的集合。

    1.  从传感器捕获的一系列值，即观察的集合。

    1.  一系列触发的警报，即警报的集合。

1.  我们可以使用以下命令行工具激活我们的板作为英特尔物联网分析账户中的一个设备：

    1.  iotkit-admin

    1.  iotkit-configure

    1.  iotkit-setup

1.  为了使用英特尔物联网分析提供的 REST API 从设备发送观察结果，我们需要以下令牌：

    1.  传感器令牌。

    1.  观察令牌。

    1.  设备令牌。

# 摘要

在本章中，我们了解了物联网与大数据之间的紧密关系。我们与一个基于云的服务合作，该服务允许我们组织由多个设备和其传感器收集的大量数据。我们利用`requests`包编写了几行 Python 代码，可以与英特尔物联网分析 REST API 交互。

我们使用英特尔物联网分析网站来设置设备和其组件。然后，我们对我们的一个示例进行了修改，以从传感器收集数据并将观察结果发布到英特尔物联网分析。然后，我们了解了英特尔物联网分析为我们提供的不同选项，以分析大量数据。最后，我们定义了触发警报的规则。现在，我们能够利用英特尔物联网分析来分析大量数据，我们准备部署成千上万的物联网设备，这些设备从多个传感器收集数据。

我们学会了使用 Python 和英特尔 Galileo Gen 2 板来创建低成本设备，这些设备可以收集大量数据，相互交互，并利用云服务和基于云的存储。我们可以使用 Python 2.7.3 及其库和工具从硬件选择到所有必要的堆栈开发物联网原型。如果我们需要更小的板或不同的替代品，我们可以切换到任何兼容的英特尔 Edison 板，因此，如果我们需要，我们可以切换到这块板。

我们能够利用现有的 Python 知识从现实世界中捕获数据，与物理对象交互，开发 API 并使用不同的物联网协议。我们学会了使用特定的库来处理底层硬件、传感器、执行器、总线和显示器。我们准备好成为创造者，并成为激动人心的物联网世界的一部分。

我们可以开始着手于将日常物体转变为带有传感器和执行器的智能设备的项目。我们准备好开始构建由数千个物联网设备组成的生态系统，Python 作为我们的主要编程语言。

# 附录 A. 练习答案

# 第一章，理解和设置基础物联网硬件

| Q1 | 2 |
| --- | --- |
| Q2 | 1 |
| Q3 | 2 |
| Q4 | 3 |
| Q5 | 1 |

# 第二章，在英特尔 Galileo Gen 2 上使用 Python

| Q1 | 2 |
| --- | --- |
| Q2 | 1 |
| Q3 | 2 |
| Q4 | 3 |
| Q5 | 1 |

# 第三章，使用 Python 与数字输出交互

| Q1 | 3 |
| --- | --- |
| Q2 | 1 |
| Q3 | 1 |
| Q4 | 2 |
| Q5 | 2 |

# 第四章，使用 RESTful API 和脉宽调制进行工作

| Q1 | 3 |
| --- | --- |
| Q2 | 3 |
| Q3 | 2 |
| Q4 | 1 |
| Q5 | 2 |

# 第五章，使用数字输入、轮询和中断

| Q1 | 1 |
| --- | --- |
| Q2 | 2 |
| Q3 | 1 |
| Q4 | 2 |
| Q5 | 3 |

# 第六章，使用模拟输入和本地存储

| Q1 | 3 |
| --- | --- |
| Q2 | 1 |
| Q3 | 2 |
| Q4 | 1 |
| Q5 | 3 |

# 第七章，使用传感器从现实世界获取数据

| Q1 | 2 |
| --- | --- |
| Q2 | 1 |
| Q3 | 2 |
| Q4 | 3 |
| Q5 | 1 |

# 第八章，显示信息和执行操作

| Q1 | 1 |
| --- | --- |
| Q2 | 1 |
| Q3 | 3 |
| Q4 | 3 |
| Q5 | 2 |

# 第九章，使用云服务

| Q1 | 2 |
| --- | --- |
| Q2 | 1 |
| Q3 | 3 |
| Q4 | 2 |
| Q5 | 1 |

# 第十章，使用基于云的物联网分析分析大量数据

| Q1 | 1 |
| --- | --- |
| Q2 | 3 |
| Q3 | 2 |
| Q4 | 1 |
| Q5 | 3 |
