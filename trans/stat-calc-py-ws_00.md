# 前言

# 关于本书

你是否想要开始开发人工智能应用程序？您是否需要对关键数学概念进行复习？充满有趣的实践练习，《Python 统计学和微积分工作坊》将向您展示如何在 Python 环境中应用您对高级数学的理解。

本书首先概述了在使用 Python 进行统计时将使用的库。随着学习的深入，您将使用 Python 编程语言执行各种数学任务，例如使用 Python 解决代数函数，从基本函数开始，然后进行变换和解方程。本书的后几章将涵盖统计学和微积分概念，以及如何使用它们来解决问题并获得有用的见解。最后，您将学习重点是数值方法的微分方程，并了解直接计算函数值的算法。

通过本书，您将学会如何将基本统计学和微积分概念应用于开发解决业务挑战的强大 Python 应用程序。

## 受众

如果您是一名希望开发解决具有挑战性的业务问题的智能解决方案的 Python 程序员，那么本书适合您。为了更好地理解本书中解释的概念，您必须对高级数学概念有透彻的理解，例如马尔可夫链、欧拉公式和龙格-库塔方法，因为本书只解释了这些技术和概念如何在 Python 中实现。

## 关于章节

*第一章*《Python 基础》介绍了 Python 语言。您将学习如何使用 Python 最基本的数据结构和控制流程，以及掌握针对编程特定任务的最佳实践，如调试、测试和版本控制。

*第二章*《Python 统计学的主要工具》介绍了 Python 中科学计算和可视化的生态系统。这些讨论将围绕着促进这些任务的特定 Python 库展开，如 NumPy、pandas 和 Matplotlib。动手练习将帮助您练习它们的使用。

*第三章*《Python 统计工具箱》描述了统计分析的理论基础。您将了解统计学领域的基本组成部分，即各种类型的统计和统计变量。本章还包括对各种不同 Python 库和工具的简要概述，这些库和工具可以帮助简化专门任务，如 SymPy、PyMC3 和 Bokeh。

*第四章*《Python 函数和代数》讨论了数学函数和代数方程的理论基础。这些讨论还伴随着交互式练习，展示了 Python 中相应的工具，可以简化和/或自动化各种过程，如绘制函数图形和解方程组。

*第五章*《Python 更多数学知识》教授您序列、级数、三角学和复数的基础知识。虽然这些可能是具有挑战性的理论主题，但我们将从不同的实际角度考虑它们，特别是通过实际应用，如财务分析和 401(k)/退休计算。

*第六章*《Python 矩阵和马尔可夫链》介绍了矩阵和马尔可夫链的概念。这些是数学对象，在人工智能和机器学习等一些最流行的数学应用中常用。本章配有动手实践活动，开发一个单词预测器。

第七章《Python 基础统计学》标志着本书重点讨论统计和统计分析的部分的开始。本章介绍了探索性数据分析的过程，以及一般使用简单的统计技术来解释数据集。

第八章《Python 基础概率概念及其应用》深入探讨了复杂的统计概念，如随机性，随机变量以及使用模拟作为分析随机性的技术。本章将帮助您更加熟练地处理涉及随机性的统计问题。

第九章《Python 中级统计学》总结了统计学的主题，重点介绍了该领域中最重要的理论，如大数定律和中心极限定理，以及常用的技术，包括置信区间，假设检验和线性回归。通过本章获得的知识，您将能够使用 Python 解决许多现实生活中的统计问题。

第十章《Python 基础微积分》开始讨论微积分的主题，包括更多涉及的概念，如函数的斜率，曲线下的面积，优化和旋转体。这些通常被认为是数学中复杂的问题，但本书通过 Python 以直观和实用的方式解释这些概念。

第十一章《Python 更多微积分》涉及微积分中更复杂的主题，包括弧长和表面积的计算，偏导数和级数展开。再次，我们将看到 Python 在帮助我们处理这些高级主题方面的强大力量，这些通常对许多学生来说可能非常具有挑战性。

第十二章《Python 中级微积分》总结了本书中最有趣的微积分主题，如微分方程，欧拉方法和 Runge-Kutta 方法。这些方法提供了解微分方程的算法方法，特别适用于 Python 作为计算工具。

## 约定

文本中的代码单词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄显示如下：

“为此，我们可以使用`with`关键字和`open()`函数与文本文件交互。”

代码块设置如下：

```py
if x % 6 == 0:
    print('x is divisible by 6')
```

在某些情况下，一行代码紧接着它的输出。这些情况如下所示：

```py
>>> find_sum([1, 2, 3]) 
6 
```

在此示例中，执行的代码是以`>>>`开头的行，输出是第二行（`6`）。

在其他情况下，输出与代码块分开显示，以便阅读。

屏幕上显示的单词，例如菜单或对话框中的单词，也会出现在文本中，如：“当您单击`获取图像`按钮时，图像将显示作者的名称。”

新术语和重要单词显示如下：“将返回的列表以相同的**逗号分隔值**（**CSV**）格式写入同一输入文件的新行中”。

## 代码演示

跨多行的代码使用反斜杠（`\`）进行分割。当代码执行时，Python 将忽略反斜杠，并将下一行的代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                   validation_split=0.2, shuffle=False)
```

代码中添加了注释以帮助解释特定的逻辑部分。单行注释使用`#`符号表示，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释使用三重引号括起来，如下所示：

```py
"""
Define a seed for the random number generator to ensure the 
result will be reproducible
"""
seed = 1
np.random.seed(seed)
random.set_seed(seed)
```

## 设置您的环境

在我们详细探讨本书之前，我们需要设置特定的软件和工具。在接下来的部分中，我们将看到如何做到这一点。

## 软件要求

您还需要预先安装以下软件：

+   操作系统：Windows 7 SP1 64 位，Windows 8.1 64 位或 Windows 10 64 位，macOS 或 Linux

+   浏览器：最新版本的 Google Chrome，Firefox 或 Microsoft Edge

+   Python 3.7

+   Jupyter 笔记本

## 安装和设置

在开始本书之前，您需要安装 Python（3.7 或更高版本）和 Jupyter，这是我们将在整个章节中使用的主要工具。

## 安装 Python

安装 Python 的最佳方法是通过环境管理器 Anaconda，可以从[`docs.anaconda.com/anaconda/install/`](https://docs.anaconda.com/anaconda/install/)下载。一旦成功安装了 Anaconda，您可以按照[`docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)上的说明创建一个虚拟环境，其中可以运行 Python。

与其他安装 Python 的方法不同，Anaconda 提供了一个易于导航的界面，当安装 Python 及其库时，它还负责大部分低级过程。

按照上述说明，您可以使用命令`conda create -n workshop python=3.7`创建一个名为`workshop`的新环境。要激活新环境，请运行`conda activate workshop`。在接下来的步骤中，您需要每次需要测试代码时激活此环境。

在本研讨会中，每次使用尚未安装的新库时，可以使用`pip install [library_name]`或`conda install [library_name]`命令来安装该库。

## Jupyter 项目

Jupyter 项目是开源的免费软件，它使您能够从特殊的笔记本中以交互方式运行用 Python 和其他一些语言编写的代码，类似于浏览器界面。它诞生于 2014 年的**IPython**项目，自那时起就成为整个数据科学工作人员的默认选择。

要在`workshop`环境中安装 Jupyter Notebook，只需运行`conda install -c conda-forge notebook`。有关 Jupyter 安装的更多信息，请访问：[`jupyter.org/install`](https://jupyter.org/install)。

在[`jupyterlab.readthedocs.io/en/stable/getting_started/starting.html`](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html)上，您将找到所有关于如何启动 Jupyter Notebook 服务器的详细信息。在本书中，我们使用经典的笔记本界面。

通常，我们从 Anaconda Prompt 使用`jupyter notebook`命令启动笔记本。

从您选择下载代码文件的目录开始笔记本，参见*安装代码包*部分。

例如，如果您已将文件安装在 macOS 目录`/Users/YourUserName/Documents/` `The-Statistics-and-Calculus-with-Python-Workshop`中，那么在 CLI 中，您可以输入`cd /Users/YourUserName/Documents/The-Statistics-and-Calculus-with-Python-Workshop`并运行`jupyter notebook`命令。Jupyter 服务器将启动，您将看到 Jupyter 浏览器控制台：

![图 0.1：Jupyter 浏览器控制台](img/B15968_Preface_01.jpg)

图 0.1：Jupyter 浏览器控制台

一旦您运行了 Jupyter 服务器，点击`New`，选择`Python 3`。一个新的浏览器标签页将打开一个新的空白笔记本。重命名 Jupyter 文件：

![图 0.2：Jupyter 服务器界面](img/B15968_Preface_02.jpg)

图 0.2：Jupyter 服务器界面

Jupyter 笔记本的主要构建模块是单元格。有两种类型的单元格：`In`（输入的缩写）和`Out`（输出的缩写）。您可以在`In`单元格中编写代码、普通文本和 Markdown，按*Shift* + *Enter*（或*Shift* + *Return*），那个特定`In`单元格中编写的代码将被执行。结果将显示在`Out`单元格中，然后您将进入一个新的`In`单元格，准备好下一个代码块。一旦您习惯了这个界面，您将慢慢发现它提供的强大和灵活性。

当您开始一个新的单元格时，默认情况下假定您将在其中编写代码。但是，如果您想要编写文本，那么您必须更改类型。您可以使用以下键序列来执行此操作：*Esc* | *M* | *Enter*。这将将所选单元格转换为**Markdown**（**M**）单元格类型：

![图 0.3：Jupyter Notebook](img/B15968_Preface_03.jpg)

图 0.3：Jupyter Notebook

当您完成编写一些文本时，请使用*Shift* + *Enter*执行它。与代码单元格不同，编译后的 Markdown 的结果将显示在与`In`单元格相同的位置。

要获取 Jupyter 中所有方便的快捷键的*备忘单*，请访问[`packt.live/33sJuB6`](https://packt.live/33sJuB6)。通过这个基本介绍，我们准备开始一段激动人心和启发人心的旅程。

## 安装库

`pip`已经预装在 Anaconda 中。一旦在您的计算机上安装了 Anaconda，所有所需的库都可以使用`pip`安装，例如`pip install numpy`。或者，您可以使用`pip install –r requirements.txt`安装所有所需的库。您可以在[`packt.live/3gv0zhb`](https://packt.live/3gv0zhb)找到`requirements.txt`文件。

练习和活动将在 Jupyter 笔记本中执行。Jupyter 是一个 Python 库，可以像其他 Python 库一样安装-即使用`pip install jupyter`，但幸运的是，它已经预装在 Anaconda 中。要打开笔记本，只需在终端或命令提示符中运行`jupyter notebook`命令。

## 访问代码文件

您可以在[`packt.live/3kcWZe6`](https://packt.live/3kcWZe6)找到本书的完整代码文件。您还可以通过使用[`packt.live/2PpqDOX`](https://packt.live/2PpqDOX)上的交互式实验室环境直接在 Web 浏览器中运行许多活动和练习。

我们已经尝试支持所有活动和练习的交互式版本，但我们也建议在不支持此功能的情况下进行本地安装。

如果您在安装过程中遇到任何问题或有任何问题，请发送电子邮件至`workshops@packt.com`与我们联系。
