# 10

# 测试和 TDD

无论开发者多么优秀，他们都会编写出不一定总是正确执行的代码。这是不可避免的，因为没有任何开发者是完美的。但这也因为预期的结果有时并不是在编码过程中会想到的。

设计很少按预期进行，在实施过程中总会有来回讨论，直到它们被精炼并变得正确。

> 每个人都有自己的计划，直到他们被打得满嘴是血。 —— 迈克·泰森

编写软件因其极端的灵活性而闻名地困难，但与此同时，我们可以使用软件来双重检查代码是否正在执行其应有的操作。

请注意，就像任何其他代码一样，测试也可能有错误。

编写测试可以在代码新鲜时检测到问题，并带有一些合理的怀疑来验证预期的结果是否是实际的结果。在本章中，我们将看到如何轻松编写测试，以及不同的策略来编写不同类型的测试，以捕获不同种类的问题。

我们将描述如何在 TDD（测试驱动开发）下工作，这是一种通过首先定义测试来确保验证尽可能独立于实际代码实现的方法。

我们还将展示如何使用常见的单元测试框架、标准`unittest`模块以及更高级和强大的`pytest`在 Python 中创建测试。

注意，本章比其他章节要长一些，主要是因为需要展示示例代码。

在本章中，我们将涵盖以下主题：

+   测试代码

+   不同的测试级别

+   测试哲学

+   测试驱动开发

+   Python 单元测试简介

+   测试外部依赖

+   高级 pytest

让我们从一些关于测试的基本概念开始。

# 测试代码

讨论代码测试时的第一个问题很简单：我们究竟指的是什么测试代码？

虽然对此有多种答案，但从最广泛的意义上讲，答案可能是“*任何在最终客户到达之前检查应用程序是否正确工作的程序。*”在这个意义上，任何正式或非正式的测试程序都将满足定义。

最轻松的方法，有时在只有一个或两个开发者的小型应用程序中可以看到，就是不去创建特定的测试，而是进行非正式的“完整应用程序运行”，检查新实现的功能是否按预期工作。

这种方法可能适用于小型、简单的应用程序，但主要问题是确保旧功能保持稳定。

但是，对于大型且复杂的、高质量的软件，我们需要对测试更加小心。所以，让我们尝试给出一个更精确的测试定义：*测试是任何记录在案的程序，最好是自动化的，它从一个已知的设置出发，检查应用程序的不同元素在到达最终客户之前是否正确工作。*

如果我们将与之前定义的差异进行检查，有几个关键词。让我们逐一检查它们，看看不同的细节：

+   **已记录的**：与之前的版本相比，目标应该是测试应该被记录下来。这样，如果需要的话，你可以精确地重新执行它们，并且可以比较它们以发现盲点。

    测试可以有多种记录方式，要么指定要运行的步骤和预期结果列表，要么创建运行测试的代码。主要思想是测试可以被分析，可以被不同的人多次运行，如果需要的话可以更改，并且有一个清晰的设计和结果。

+   **最好是自动化的**：测试应该能够自动运行，尽可能减少人为干预。这允许你触发持续集成技术，反复运行许多测试，创建一个“安全网”，能够尽早捕捉到意外错误。我们说“最好是”是因为也许有些测试是完全不可能或非常昂贵的自动化。无论如何，目标应该是让绝大多数测试自动化，让计算机承担繁重的工作，节省宝贵的人类时间。也有多种软件工具可以帮助你运行测试。

+   **从已知设置开始**：为了能够独立运行测试，我们需要知道在运行测试之前系统的状态应该是什么。这确保了测试的结果不会创建一个可能会干扰下一个测试的特定状态。在测试前后，可能需要进行一些清理工作。

    与不考虑初始或最终状态相比，这可能会使批量运行测试的速度变慢，但它将创建一个坚实的基础，以避免问题。

作为一般规则，尤其是在自动化测试中，测试执行的顺序应该是无关紧要的，以避免交叉污染。这说起来容易做起来难，在某些情况下，测试的顺序可能会造成问题。例如，测试 A 创建了一个条目，测试 B 读取。如果测试 B 单独运行，它将失败，因为它期望 A 创建的条目。这些情况应该得到修复，因为它们可能会极大地复杂化调试。此外，能够独立运行测试也允许它们并行化。

+   **应用程序的不同元素**：大多数测试不应该针对整个应用程序，而应该针对其较小的部分。我们稍后会更多地讨论不同的测试级别，但测试应该具体说明它们正在测试什么，并覆盖不同的元素，因为覆盖更多范围的测试将更昂贵。

测试的一个关键要素是获得良好的投资回报率。设计和运行测试需要时间，这些时间需要被充分利用。任何测试都需要维护，这应该是值得的。在整个章节中，我们将讨论测试的这个重要方面。

有一种重要的测试类型我们没有在这个定义中涵盖，这被称为*探索性测试*。这些测试通常由质量保证工程师运行，他们使用最终应用程序而没有明确的先入之见，但试图预先发现问题。如果应用程序有一个面向客户的用户界面，这种测试风格在检测设计阶段未检测到的不一致和问题方面可能非常有价值。

例如，一个优秀的质量保证工程师能够说出页面 X 上的按钮颜色与页面 Y 上的按钮颜色不同，或者按钮不够明显以至于无法执行操作，或者执行某个特定操作有一个不明显或在新界面中不可能的前提条件。任何**用户体验**（**UX**）检查都可能属于这一类别。

从本质上讲，这种测试不能“设计”或“文档化”，因为它最终取决于解释和良好的洞察力来理解应用程序是否“感觉正确”。一旦发现问题，就可以记录下来以避免。

虽然这确实很有用且被推荐，但这种测试风格更多的是一种艺术而不是工程实践，我们不会详细讨论它。

这个一般定义有助于开始讨论，但我们可以更具体地讨论通过测试时系统受测试的部分来定义的不同测试。

# 不同的测试级别

正如我们之前所描述的，测试应该覆盖系统的不同元素。这意味着一个测试可以针对系统的小部分或大部分（或整个系统），试图缩小其作用范围。

当测试系统的小部分时，我们减少了测试的复杂性和范围。我们只需要调用系统的那一小部分，并且设置更容易开始。一般来说，要测试的元素越小，测试的速度越快，越容易。

我们将定义三个不同级别或种类的测试，从小范围到大范围：

+   **单元测试**，用于检查服务的一部分

+   **集成测试**，用于检查单个服务作为一个整体

+   **系统测试**，用于检查多个服务协同工作的情况

名称可能实际上有很大差异。在这本书中，我们不会对定义过于严格，而是定义软限制，并建议找到适合您特定项目的平衡点。不要害羞，对每个测试的适当级别做出决定，并定义自己的命名法，同时始终牢记创建测试所需的努力，以确保它们总是值得的。

级别的定义可能有些模糊。例如，集成测试和单元测试可以并列定义，在这种情况下，它们之间的区别可能更多地体现在学术上。

让我们更详细地描述每个级别的细节。

## 单元测试

最小类型的测试也是通常投入最多努力的测试，即*单元测试*。这种测试检查的是一小段代码的行为，而不是整个系统。这个代码单元可能小到只是一个单个函数，或者测试一个单个 API 端点，等等。

正如我们上面所说的，关于单元测试应该有多大，基于“单元”是什么以及它是否实际上是一个单元，存在很多争议。例如，在某些情况下，人们只有在测试涉及单个函数或类时才会将其称为单元测试。

由于单元测试检查的是功能的一个小部分，因此它可以非常容易地设置并快速运行。因此，创建新的单元测试既快又能够彻底测试系统，确保构成整个系统的小部分能够按预期工作。

单元测试的目的是深入检查服务中定义的功能的行为。任何外部请求或元素都应该被模拟，这意味着它们被定义为测试的一部分。我们将在本章的后面更详细地介绍单元测试，因为它们是 TDD 方法的关键元素。

## 集成测试

下一个层次是集成测试。这是检查一个服务或几个服务的整体行为。

集成测试的主要目标是确保不同的服务或同一服务内的不同模块可以相互工作。在单元测试中，外部请求是模拟的，而集成测试使用的是真实的服务。

可能仍然需要模拟外部 API。例如，模拟外部支付提供者进行测试。但，总的来说，尽可能多地使用真实服务进行集成测试，因为测试的目的在于测试不同的服务是否能够协同工作。

需要注意的是，通常情况下，不同的服务将由不同的开发者或甚至不同的团队开发，他们对于特定 API 实现的理解可能会有所不同，即使在定义良好的规范下也是如此。

集成测试的设置比单元测试更复杂，因为需要正确设置更多的元素。这使得集成测试比单元测试更慢、更昂贵。

集成测试非常适合检查不同的服务是否能够协同工作，但也有一些局限性。

集成测试通常不如单元测试彻底，它们主要关注检查基本功能并遵循*快乐路径*。快乐路径是测试中的一个概念，意味着测试用例应该不会产生错误或异常。

预期错误和异常通常在单元测试中进行测试，因为它们也是可能失败的因素。但这并不意味着每个集成测试都应该遵循快乐路径；一些集成错误可能值得检查，但总的来说，快乐路径测试的是预期的通用行为。它们将构成大部分的集成测试。

## 系统测试

最后一级是系统级。系统测试检查所有不同的服务是否能够正确协同工作。

进行这类测试的一个要求是系统中实际上存在多个服务。如果不是这样，它们与低级别的测试没有区别。这些测试的主要目标是检查不同的服务能否协同工作，并且配置是否正确。

系统测试缓慢且难以实施。它们需要整个系统设置好，所有不同的服务都正确配置。创建这样的环境可能很复杂。有时，这太难了，唯一实际执行系统测试的方法是在实时环境中运行它们。

环境配置是这些测试检查的重要部分。这可能会使它们在测试每个环境时都变得很重要，包括实时环境。

虽然这并不理想，但有时这是不可避免的，并且有助于在部署后提高信心，确保新代码能够正确运行。在这种情况下，考虑到限制条件，应该只运行最少量的测试，因为实时环境至关重要。要运行的测试还应该测试最大量的常用功能和服务，以便尽可能快地检测到任何关键问题。这组测试有时被称为**验收测试**或**冒烟测试**。它们可以手动运行，作为一种确保一切看起来正确的手段。

当然，冒烟测试不仅可以在实时环境中运行，还可以作为一种确保其他环境正常工作的手段。

冒烟测试应该非常清晰、有良好的文档记录，并且精心设计以涵盖整个系统的最关键部分。理想情况下，它们还应该是只读的，这样它们在执行后不会留下无用的数据。

# 测试哲学

与测试相关的所有事情中的一个关键问题是：**为什么要测试？**我们试图通过它达到什么目标？

正如我们所看到的，测试是一种确保代码行为符合预期的方式。测试的目标是在代码发布和真实用户使用之前检测可能的问题（有时称为**缺陷**）。

**缺陷**和**错误**之间有一个细微的差别。错误是一种缺陷，指的是软件的行为不符合预期。例如，某些输入会产生意外的错误。缺陷更为普遍。一个缺陷可能是指按钮不够明显，或者页面上显示的标志不是正确的。一般来说，测试在检测错误方面比检测其他缺陷更有效，但请记住我们之前提到的探索性测试。

一个未被发现的缺陷被部署到实际系统中去修复的成本相当高。首先，它需要被发现。在一个活动频繁的实际应用中，发现问题可能很困难（尽管我们将在*第十六章*，*持续架构*中讨论），但更糟糕的是，它通常是由使用该应用的用户检测到的。用户可能无法正确地传达问题，因此问题仍然存在，造成问题或限制活动。检测问题的用户可能会放弃系统，或者至少他们对系统的信心会下降。

任何声誉损失都将是不利的，但同时也可能很难从用户那里提取足够的信息来确切了解发生了什么以及如何修复它。这使得从发现问题到修复问题的周期变得很长。

任何测试系统都会提高早期修复缺陷的能力。我们不仅可以创建一个模拟确切相同问题的特定测试，还可以创建一个定期执行测试的框架，以明确的方法来检测和修复问题。

不同的测试级别对这种成本有不同的影响。一般来说，任何可以在单元测试级别检测到的问题，在那里修复的成本都会更低，成本从那里开始增加。设计和运行单元测试比集成测试更容易、更快，而集成测试的成本比系统测试低。

不同的测试级别可以理解为不同的层级，它们捕捉可能的问题。如果问题出现，每个层级都会捕捉不同的问题。越接近过程的开始（编码时的设计和单元测试），创建一个能够检测和警告问题的密集网络就越便宜。问题修复的成本随着它远离过程开始时的控制环境而增加。

![图解自动生成的描述](img/B17580_10_01.png)

图 10.1：修复缺陷的成本随着发现时间的延迟而增加

一些缺陷在单元测试级别是无法检测到的，比如不同部分的集成。这就是下一个级别发挥作用的地方。正如我们所见，最糟糕的情况是没有发现问题，它影响了实际系统上的真实用户。

但拥有测试不仅是一次捕捉问题的好方法。因为测试可以保留下来，并在新的代码更改上运行，它还在开发过程中提供了一个安全网，以确保创建新代码或修改代码不会影响旧功能。

这是最有力的论据之一，即按照持续集成实践自动和持续地运行测试。开发者可以专注于正在开发的功能，而持续集成工具将运行每个测试，如果某个测试出现问题，会提前警告。之前引入的功能出现问题而失败的称为*回归*。

回归问题相当常见，因此拥有良好的测试覆盖率以防止它们未被检测到是非常好的。可以引入覆盖先前功能的特定测试，以确保它按预期运行。这些是回归测试，有时在检测到回归问题后添加。

拥有良好的测试来检查系统的行为的好处之一是，代码本身可以大量更改，因为我们知道行为将保持不变。这些更改可以用来重构代码、清理它，并在一般情况下改进它。这些更改被称为*重构*代码，即在不改变代码预期行为的情况下改变代码的编写方式。

现在，我们应该回答“什么是好的测试？”这个问题。正如我们讨论的，编写测试不是免费的，需要付出努力，我们需要确保它是值得的。我们如何创建好的测试？

## 如何设计一个优秀的测试

设计良好的测试需要一定的思维模式。在设计覆盖特定功能的代码时，目标是使代码实现该功能，同时保持高效，编写清晰、甚至可以说是优雅的代码。

测试的目标是确保功能符合预期行为，并且所有可能出现的不同问题都会产生有意义的输出。

现在，为了真正测试功能，心态应该是尽可能多地压榨代码。例如，让我们想象一个函数 `divide(A, B)`，它将两个介于 -100 和 100 之间的整数 A 除以 B：`A` 介于 `B` 之间。

在接近测试时，我们需要检查这个测试的极限是什么，试图检查函数是否以预期的行为正确执行。例如，可以创建以下测试：

| 操作 | 预期行为 | 备注 |
| --- | --- | --- |
| `divide(10, 2)` | `return 5` | 基本情况 |
| `divide(-20, 4)` | `return -5` | 除以一个负整数和一个正整数 |
| `divide(-10, -5)` | `return 2` | 除以两个负整数 |
| `divide(12, 2)` | `return 5` | 非精确除法 |
| `divide(100, 50)` | `return 2` | A 的最大值 |
| `divide(101, 50)` | `产生输入错误` | A 的值超过最大值 |
| `divide(50, 100)` | `return 0` | B 的最大值 |
| `divide(50, 101)` | `产生输入错误` | B 的值超过最大值 |
| `divide(10, 0)` | `产生异常` | 除以零 |
| `divide('10', 2)` | `产生输入错误` | 参数 A 的格式无效 |
| `divide(10, '2')` | `产生输入错误` | 参数 B 的格式无效 |

注意我们如何测试不同的可能性：

+   所有参数的常规行为都是正确的，除法操作也是正确的。这包括正数和负数，精确除法和非精确除法。

+   在最大值和最小值之间的值：我们检查最大值是否被正确击中，并且下一个值被正确检测。

+   除以零：功能上已知的一个限制，应该产生一个预定的响应（异常）。

+   错误的输入格式。

我们可以为简单的功能真正创建很多测试用例！请注意，所有这些情况都可以扩展。例如，我们可以添加`divide(-100, 50)`和`divide(100, -50)`的情况。在这些情况下，问题是相同的：这些测试是否增加了对问题的更好检测？ 

最好的测试是真正对代码施加压力并确保其按预期工作的测试，尽力覆盖最困难的用例。让测试向被测试的代码提出困难的问题，是让你的代码为实际操作做好准备的最佳方式。在负载下的系统将看到各种组合，因此最好的准备是创建尽可能努力寻找问题的测试，以便在进入下一阶段之前解决这些问题。

这类似于足球训练，一系列非常苛刻的练习被提出，以确保受训者能够在比赛中进行表现。确保你的训练计划足够艰难，以便为高强度的比赛做好准备！

在测试数量和避免重复测试已由现有测试覆盖的功能（例如，创建一个大的表格，用很多除法来划分数字）之间找到适当的平衡，可能很大程度上取决于被测试的代码和你们组织中的实践。某些关键区域可能需要更彻底的测试，因为那里的失败可能更重要。

例如，任何外部 API 都应该仔细测试任何输入，并对此进行真正的防御，因为外部用户可能会滥用外部 API。例如，测试当在整数字段中输入字符串时会发生什么，添加了无穷大或`NaN`（非数字）值，超出了有效负载限制，列表或页面的最大大小被超过等情况。

相比之下，大部分内部接口将需要较少的测试，因为内部代码不太可能滥用 API。例如，如果`divide`函数仅是内部的，可能不需要测试输入格式是否错误，只需检查是否尊重了限制。

注意，测试是在代码实现独立进行的。测试定义纯粹是从外部视角对要测试的函数进行，而不需要了解其内部结构。这被称为*黑盒测试*。一个健康的测试套件总是从这种方法开始。

作为编写测试的开发者，一个关键的能力是脱离对代码本身的知识，独立地对待测试。

测试可以如此独立，以至于可能需要独立的人员来创建测试，就像一个 QA 团队进行测试一样。不幸的是，这种方法对于单元测试来说是不可能的，单元测试很可能会由编写代码的同一开发者创建。

在某些情况下，这种外部方法可能不足以。如果开发者知道存在一些可能存在问题的特定区域，那么补充一些检查功能性的测试可能是个好主意，这些功能性从外部视角看可能不明显。

例如，一个基于某些输入计算结果的函数可能有一个内部点，其中算法改变以使用不同的模型来计算。外部用户不需要知道这些信息，但添加一些检查以验证转换是否正确将是有益的。

这种测试被称为*白盒测试*，与之前讨论的黑盒方法相对。

重要的是要记住，在测试套件中，白盒测试应该始终是次要的黑盒测试。主要目标是测试从外部视角的功能性。白盒测试可能是一个很好的补充，特别是在某些方面，但它应该有较低的优先级。

发展能够创建良好黑盒测试的能力非常重要，并且应该传达给团队。

黑盒测试试图避免一个常见问题，即同一个开发者编写了代码和测试，然后检查代码中实现的功能的解释是否按预期工作，而不是检查从外部端点看它是否按预期工作。我们稍后会看看 TDD，它试图通过在编写代码之前编写测试来确保测试的创建不考虑实现。

## 结构化测试

在结构方面，特别是对于单元测试，使用**安排-行动-断言**（**AAA**）模式来结构化测试是一个很好的方法。

这种模式意味着测试分为三个不同的阶段：

+   **安排**：为测试准备环境。这包括所有设置，以确保在执行下一步之前系统处于稳定状态。

+   **行动**：执行测试的目标行为。

+   **断言**：检查行为的结果是预期的。

测试被结构化为一个句子，如下所示：

**给定**（安排）一个已知的环境，**行动**（行动）产生指定的**结果**（断言）

这种模式有时也被称为*给定*、*当*、*然后*，因为每个步骤都可以用这些术语来描述。

注意，这种结构旨在使所有测试都独立，并且每个测试都测试一个单独的东西。

一种常见的不同模式是在测试中分组行动步骤，在一个测试中测试多个功能。例如，测试写入值是否正确，然后检查搜索该值是否返回正确的值。这不会遵循 AAA 模式。相反，为了遵循 AAA 模式，应该创建两个测试，第一个测试验证写入是否正确，第二个测试在搜索之前，将值作为在安排步骤中的设置部分创建。

注意，无论测试是通过代码执行还是手动运行，这种结构都可以使用，尽管它们更多地用于自动化测试。当手动运行时，Arrange 阶段可能需要花费很长时间才能为每个测试生成，导致在该阶段花费大量时间。相反，手动测试通常按照我们上面描述的模式分组，执行一系列的 Act 和 Assert，并使用前一个阶段的输入作为下一个阶段的设置。这创建了一个依赖性，需要按照特定的顺序运行测试，这对单元测试套件来说不是很好，但可能对烟雾测试或其他环境中 Arrange 步骤非常昂贵的情况更好。

同样，如果要测试的代码是纯函数式的（意味着只有输入参数决定了其状态，如上面的`divide`示例），则不需要 Arrange 步骤。

让我们看看使用这种结构创建的代码示例。假设我们有一个想要测试的方法，称为`method_to_test`。该方法属于名为`ClassToTest`的类。

```py
 def test_example():

    # Arrange step

    # Create the instance of the class to test

    object_to_test = ClassToTest(paramA='some init param', 

                                 paramB='another init param')

    # Act step

    response = object_to_test.method_to_test(param='execution_param')

    # Assert step

    assert response == 'expected result' 
```

每个步骤都非常清晰地定义。第一个步骤是准备，在这种情况下，是我们想要测试的类中的一个对象。请注意，我们可能需要添加一些参数或一些准备，以便对象处于一个已知的起始点，以便下一个步骤按预期工作。

Act 步骤仅生成要测试的动作。在这种情况下，使用适当的参数调用准备好的对象的`method_to_test`方法。

最后，Assert 步骤非常直接，只是检查响应是否是预期的。

通常，Act 和 Assert 步骤都很容易定义和编写。Arrange 步骤通常是测试中需要的大部分努力所在。

使用 AAA 模式进行测试时出现的另一个常见模式是在 Arrange 步骤中创建用于测试的通用函数。例如，创建一个基本环境，这可能需要复杂的设置，然后有多个副本，其中 Act 和 Assert 步骤不同。这减少了代码的重复。

例如：

```py
def create_basic_environment():

    object_to_test = ClassToTest(paramA='some init param', 

                                 paramB='another init param')

    # This code may be much more complex and perhaps have

    # 100 more lines of code, because the basic environment

    # to test requires a lot of things to set up

    return object_to_test

def test_exampleA():

    # Arrange

    object_to_test = create_basic_environment()

    # Act

    response = object_to_test.method_to_test(param='execution_param')

    # Assert

    assert response == 'expected result B'

def test_exampleB():

    # Arrange

    object_to_test = create_basic_environment()

    # Act

    response = object_to_test.method_to_test(param='execution_param')

    # Assert

    assert response == 'expected result B' 
```

我们稍后会看到如何结构化多个非常相似的测试，以避免重复，这在拥有大型测试套件时是一个问题。拥有大型测试套件对于创建良好的测试覆盖率很重要，如我们上面所看到的。

在测试中，重复在某种程度上是不可避免的，甚至在某种程度上是有益的。当因为某些更改而更改代码的一部分的行为时，测试需要相应地更改以适应这些更改。这种更改有助于衡量更改的大小，并避免轻率地做出大的更改，因为测试将作为受影响功能的一个提醒。

然而，无意义的重复并不好，我们稍后会看到一些减少重复代码量的选项。

# 测试驱动开发

一种非常流行的编程方法是**测试驱动开发**或**TDD**。TDD 包括将测试置于开发体验的中心。

这基于我们在本章前面提出的一些想法，尽管是以更一致的观点来工作的。

TDD 开发软件的流程如下：

1.  决定将新功能添加到代码中。

1.  编写一个新的测试来定义新的功能。注意，这是在代码之前完成的。

1.  运行测试套件以显示它正在失败。

1.  然后将新功能添加到主代码中，重点是简洁性。应该只添加所需的功能，而不添加额外的细节。

1.  运行测试套件以显示新测试正在工作。这可能需要多次进行，直到代码准备就绪。

1.  新功能已准备就绪！现在可以重构代码以改进它，避免重复，重新排列元素，将其与先前存在的代码分组等。

对于任何新的功能，循环可以再次开始。

如您所见，TDD 基于三个主要思想：

+   **在编写代码之前编写测试**：这防止了创建与当前实现过于紧密耦合的测试的问题，迫使开发者在编写之前先思考测试和功能。这也迫使开发者在编写功能之前检查测试是否实际失败，确保以后发现的问题。这与我们在“如何设计一个优秀的测试”部分中描述的黑盒测试方法类似。

+   **持续运行测试**：过程的关键部分是运行整个测试套件，以确保系统中的所有功能都是正确的。这需要反复进行，每次创建新测试时都要这样做，而且在功能编写过程中也是如此。在 TDD（测试驱动开发）中，运行测试是开发的一个基本部分。这确保了所有功能始终得到检查，并且代码在所有时间都能按预期工作，因此任何错误或不一致都可以迅速解决。

+   **以非常小的增量工作**：专注于手头的任务，这样每一步都会构建和扩展一个大的测试套件，深入覆盖代码的所有功能。

这个大的测试套件创建了一个安全网，允许你经常进行代码的重构，无论是大重构还是小重构，因此可以不断改进代码。小的增量意味着具体的测试，在添加代码之前需要思考。

这个想法的一个扩展是只编写完成任务所需的代码，而不是更多。这有时被称为**YAGNI**原则（**You Ain't Gonna Need It**）。这个原则的目的是防止过度设计或为“可预见的未来请求”编写代码，实际上这些请求有很大概率永远不会实现，而且更糟糕的是，这会使代码在其他方向上更难以更改。鉴于软件开发在事先规划上众所周知地困难，重点应该放在保持事物的小规模上，不要过于超越自己。

这三个想法在开发周期中不断相互作用，并使测试始终处于开发过程的核心，因此这种实践被称为“测试驱动开发”。

TDD 的另一个重要优势是，如此重视测试意味着从一开始就会考虑代码的测试方式，这有助于设计易于测试的代码。此外，减少要编写的代码量，专注于它是否严格必要以通过测试，这降低了过度设计的发生概率。创建小型测试并在增量中工作的要求也倾向于生成模块化代码，这些代码以小单元组合在一起，但能够独立进行测试。

一般的流程是持续地与新的失败测试一起工作，让它们通过，然后进行重构，这有时被称为“*红/绿/重构*”模式：当测试失败时为红色，当所有测试都通过时为绿色。

重构是 TDD 过程中的一个关键方面。强烈鼓励不断改进现有代码的质量。这种方式工作的最佳结果之一是生成非常广泛的测试套件，覆盖代码功能的所有细节，这意味着重构代码时可以知道有一个坚实的基础，可以捕捉到代码更改引入的任何问题，并添加错误。

通过重构来提高代码的可读性、可用性等，这在提高开发者的士气和加快引入变更的速度方面有着良好的影响，因为这样可以保持代码的良好状态。

通常来说，不仅在 TDD 中，留出时间清理旧代码并改进它对于保持良好的变更节奏至关重要。陈旧的旧代码往往越来越难以处理，随着时间的推移，将其更改以进行更多变更将需要更多的努力。鼓励健康的习惯，关注代码的当前状态，并留出时间进行维护性改进，对于任何软件系统的长期可持续性至关重要。

TDD 的另一个重要方面是快速测试的要求。由于测试始终在 TDD 实践中运行，因此总执行时间非常重要。每个测试所需的时间应仔细考虑，因为测试套件的增长将使运行时间更长。

存在一个普遍的阈值，超过这个阈值注意力就会分散，因此运行时间超过大约 10 秒的测试将使它们不再是“同一操作的一部分”，这会风险开发者去想其他事情。

显然，在 10 秒内运行整个测试套件将非常困难，尤其是在测试数量增加的情况下。一个复杂应用的完整单元测试套件可能包含 10,000 个测试或更多！在现实生活中，有多种策略可以帮助缓解这一事实。

整个测试套件并不需要一直运行。相反，任何测试运行器都应该允许你选择要运行的测试范围，允许你在开发功能时减少每次运行要运行的测试数量。这意味着只运行与同一模块相关的测试，例如。在某些情况下，甚至可以运行单个测试来加快结果。

当然，在某个时候，应该运行整个测试套件。TDD 实际上与持续集成相一致，因为它也基于运行测试，这次是在代码被检出到仓库后自动运行。能够在本地运行一些测试以确保开发过程中一切正常，同时在将代码提交到仓库后，在后台运行整个测试套件，这种组合是非常好的。

无论如何，在 TDD 中，运行测试所需的时间很重要，观察测试的持续时间很重要，生成可以快速运行的测试是能够以 TDD 方式工作的关键。这主要通过创建覆盖代码小部分的测试来实现，因此可以保持设置时间在可控范围内。

TDD 实践与单元测试配合得最好。集成和系统测试可能需要一个大的设置，这与 TDD 工作所需的速度和紧密的反馈循环不兼容。

幸运的是，正如我们之前所看到的，单元测试通常是大多数项目中测试的重点。

## 将 TDD 引入新团队

在一个组织中引入 TDD 实践可能很棘手，因为它们改变了执行基本操作的方式，并且与通常的工作方式（在写代码后编写测试）有些相悖。

当考虑将 TDD 引入一个团队时，有一个可以充当团队其他成员的联系人并解决可能通过创建测试而出现的问题的倡导者是很好的。

TDD 在结对编程也普遍的环境中非常受欢迎，因此，在培训其他开发人员并引入这一实践的同时，让某人驱动一个会话是另一种可能性。

记住，TDD 的关键要素是迫使开发者首先思考如何测试特定的功能，然后再开始考虑实现。这种心态不是自然而然产生的，需要训练和实践。

在现有代码中应用 TDD 技术可能具有挑战性，因为在这种配置下，预存的代码可能难以测试，尤其是如果开发者对这种实践不熟悉。然而，对于新项目来说，TDD 效果很好，因为新代码的测试套件将与代码同时创建。在现有项目中启动一个新模块的混合方法，因此大部分代码都是新的，可以使用 TDD 技术进行设计，这减少了处理遗留代码的问题。

如果你想看看 TDD 是否对新代码有效，尝试从小处开始，使用一些小型项目和小型团队来确保它不会过于破坏性，并且原则可以正确消化和应用。有些开发者非常喜欢使用 TDD 原则，因为它符合他们的个性和他们处理开发过程的方式。记住，这并不一定适合每个人，并且开始这些实践需要时间，也许不可能 100%地应用它们，因为之前的代码可能会限制它。

## 问题与局限性

TDD 实践在业界非常流行且被广泛遵循，尽管它们有其局限性。其中一个是运行时间过长的大测试问题。在某些情况下，这些测试可能是不可避免的。

另一个问题是，如果这不是从一开始就做的，那么完全采用这种方法可能会有困难，因为代码的部分已经编写，可能还需要添加新的测试，违反了在编写代码之前创建测试的规则。

另一个问题是在要实现的功能不流动且未完全定义的情况下设计新代码。这需要实验，例如，设计一个函数来返回与输入颜色形成对比的颜色，例如，根据用户可选择的主题呈现对比颜色。这个函数可能需要检查它是否“看起来合适”，这可能需要调整，而使用预配置的单元测试很难实现。

这不是 TDD 特有的问题，但需要注意的一点是，要记住避免测试之间的依赖关系。这可能在任何测试套件中发生，但鉴于对创建新测试的关注，如果团队从 TDD 实践开始，这很可能是一个问题。依赖关系可能通过要求测试以特定顺序运行而引入，因为测试可能会污染环境。这通常不是故意为之，但在编写多个测试时无意中发生。

典型的效果是，如果独立运行，某些测试可能会失败，因为在这种情况下没有运行它们的依赖关系。

无论如何，请记住，TDD 并不一定是一切或无的东西，而是一套可以帮助你设计经过良好测试和高质量代码的想法和实践。系统中并非每个测试都需要使用 TDD 来设计，但其中很多可以。

## TDD 流程的示例

让我们想象我们需要创建一个函数，它：

+   对于小于 0 的值，返回零

+   对于大于 10 的值，返回 100

+   对于介于两者之间的值，它返回该值的 2 的幂。注意，对于边缘，它返回输入的 2 的幂（0 对应于 0，100 对应于 10）

要以全 TDD 风格编写代码，我们从可能的最小测试开始。让我们创建最小的骨架和第一个测试。

```py
def parameter_tdd(value):

    pass

assert parameter_tdd(5) == 25 
```

我们运行测试，并且测试失败时出现错误。现在，我们将使用纯 Python 代码，但稍后在本章中，我们将看到如何更有效地运行测试。

```py
$ python3 tdd_example.py

Traceback (most recent call last):

  File ".../tdd_example.py", line 6, in <module>

    assert parameter_tdd(5) == 25

AssertionError 
```

用例的实现相当直接。

```py
def parameter_tdd(value):

    return 25 
```

是的，我们实际上返回了一个硬编码的值，但这确实是通过第一个测试所必需的全部。现在让我们运行测试，你将看到没有错误。

```py
$ python3 tdd_example.py 
```

但现在我们添加了对较低边缘的测试。虽然这两行，但它们可以被认为是同一个测试，因为它们检查边缘是否正确。

```py
assert parameter_tdd(-1) == 0

assert parameter_tdd(0) == 0

assert parameter_tdd(5) == 25 
```

让我们再次运行测试。

```py
$ python3 tdd_example.py

Traceback (most recent call last):

  File ".../tdd_example.py", line 6, in <module>

    assert parameter_tdd(-1) == 0

AssertionError 
```

我们需要添加代码来处理较低的边缘。

```py
def parameter_tdd(value):

    if value <= 0:

        return 0

    return 25 
```

当运行测试时，我们看到它正在正确地运行测试。现在让我们添加参数来处理上边缘。

```py
assert parameter_tdd(-1) == 0

assert parameter_tdd(0) == 0

assert parameter_tdd(5) == 25

assert parameter_tdd(10) == 100

assert parameter_tdd(11) == 100 
```

这触发了相应的错误。

```py
$ python3 tdd_example.py

Traceback (most recent call last):

  File "…/tdd_example.py", line 12, in <module>

    assert parameter_tdd(10) == 100

AssertionError 
```

让我们添加更高的边缘。

```py
def parameter_tdd(value):

    if value <= 0:

        return 0

    if value >= 10:

        return 100

    return 25 
```

这运行正确。我们并不确信所有代码都正确，我们真的想确保中间部分是正确的，所以我们添加了另一个测试。

```py
assert parameter_tdd(-1) == 0

assert parameter_tdd(0) == 0

assert parameter_tdd(5) == 25

assert parameter_tdd(7) == 49

assert parameter_tdd(10) == 100

assert parameter_tdd(11) == 100 
```

哎呀！现在显示了一个错误，这是由于初始的硬编码造成的。

```py
$ python3 tdd_example.py

Traceback (most recent call last):

  File "/…/tdd_example.py", line 15, in <module>

    assert parameter_tdd(7) == 49

AssertionError 
```

所以让我们修复它。

```py
def parameter_tdd(value):

    if value <= 0:

        return 0

    if value >= 10:

        return 100

    return value ** 2 
```

这运行了所有的测试并且正确无误。现在，有了测试的安全网，我们认为我们可以稍微重构一下代码来清理它。

```py
def parameter_tdd(value):

    if value < 0:

        return 0

    if value < 10:

        return value ** 2

    return 100 
```

我们可以在整个过程中运行测试，并确保代码是正确的。最终结果可能基于团队认为的好代码或更明确的内容而有所不同，但我们有我们的测试套件，这将确保测试是一致的，行为是正确的。

这里的函数相当小，但这显示了以 TDD 风格编写代码时的流程。

# Python 单元测试简介

在 Python 中运行测试有多种方式。一种，如我们上面所见，有点粗糙，是执行带有多个断言的代码。一种常见的方式是标准库 `unittest`。

## Python unittest

`unittest` 是 Python 标准库中的一个模块。它基于创建一个测试类来分组多个测试方法的概念。让我们写一个新文件，用适当的格式编写测试，命名为 `test_unittest_example.py`。

```py
import unittest

from tdd_example import parameter_tdd

class TestTDDExample(unittest.TestCase):

    def test_negative(self):

        self.assertEqual(parameter_tdd(-1), 0)

    def test_zero(self):

        self.assertEqual(parameter_tdd(0), 0)

    def test_five(self):

        self.assertEqual(parameter_tdd(5), 25)

    def test_seven(self):

        # Note this test is incorrect

        self.assertEqual(parameter_tdd(7), 0)

    def test_ten(self):

        self.assertEqual(parameter_tdd(10), 100)

    def test_eleven(self):

        self.assertEqual(parameter_tdd(11), 100)

if __name__ == '__main__':

    unittest.main() 
```

让我们分析不同的元素。首先是顶部的导入。

```py
import unittest

from tdd_example import parameter_tdd 
```

我们导入 `unittest` 模块和要测试的函数。最重要的部分接下来，它定义了测试。

```py
class TestTDDExample(unittest.TestCase):

    def test_negative(self):

        self.assertEqual(parameter_tdd(-1), 0) 
```

`TestTDDExample`类将不同的测试分组。请注意，它继承自`unittest.TestCase`。然后，以`test_`开头的方法将产生独立的测试。在这里，我们将展示一个。内部，它调用函数并使用`self.assertEqual`函数将结果与 0 进行比较。

注意，`test_seven`定义不正确。我们这样做是为了在运行时产生错误。

最后，我们添加这段代码。

```py
if __name__ == '__main__':

    unittest.main() 
```

如果我们运行文件，它会自动运行测试。所以，让我们运行这个文件：

```py
$ python3 test_unittest_example.py

...F..

======================================================================

FAIL: test_seven (__main__.TestTDDExample)

----------------------------------------------------------------------

Traceback (most recent call last):

  File ".../unittest_example.py", line 17, in test_seven

    self.assertEqual(parameter_tdd(7), 0)

AssertionError: 49 != 0

----------------------------------------------------------------------

Ran 6 tests in 0.001s

FAILED (failures=1) 
```

如您所见，它已运行所有六个测试，并显示了任何错误。在这里，我们可以清楚地看到问题。如果我们需要更多细节，我们可以使用`-v showing`选项运行，显示正在运行的每个测试：

```py
$ python3 test_unittest_example.py -v

test_eleven (__main__.TestTDDExample) ... ok

test_five (__main__.TestTDDExample) ... ok

test_negative (__main__.TestTDDExample) ... ok

test_seven (__main__.TestTDDExample) ... FAIL

test_ten (__main__.TestTDDExample) ... ok

test_zero (__main__.TestTDDExample) ... ok

======================================================================

FAIL: test_seven (__main__.TestTDDExample)

----------------------------------------------------------------------

Traceback (most recent call last):

  File ".../unittest_example.py", line 17, in test_seven

    self.assertEqual(parameter_tdd(7), 0)

AssertionError: 49 != 0

----------------------------------------------------------------------

Ran 6 tests in 0.001s

FAILED (failures=1) 
```

您也可以使用`-k`选项运行单个测试或它们的组合，该选项会搜索匹配的测试。

```py
$ python3 test_unittest_example.py -v -k test_ten

test_ten (__main__.TestTDDExample) ... ok

----------------------------------------------------------------------

Ran 1 test in 0.000s

OK 
```

`unittest`非常受欢迎，可以接受很多选项，并且与 Python 中的几乎每个框架都兼容。它在测试方式上也非常灵活。例如，有多种方法可以比较值，如`assertNotEqual`和`assertGreater`。

有一个特定的断言函数工作方式不同，即`assertRaises`，用于检测代码生成异常的情况。我们将在测试模拟外部调用时稍后查看它。

它还具有`setUp`和`tearDown`方法，用于在每个测试执行前后执行代码。

请确保查看官方文档：[`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)。

虽然`unittest`可能是最受欢迎的测试框架，但它并不是最强大的。让我们来看看它。

## Pytest

Pytest 进一步简化了编写测试的过程。关于`unittest`的一个常见抱怨是它强制您设置许多不是显而易见的`assertCompare`调用。它还需要对测试进行结构化，添加一些样板代码，如`test`类。其他问题可能不那么明显，但创建大型测试套件时，不同测试的设置可能会变得复杂。

一个常见的模式是创建继承自其他测试类的类。随着时间的推移，这可能会发展出自己的特性。

Pytest 简化了测试的运行和定义，并使用更易于阅读和识别的标准`assert`语句捕获所有相关信息。

在本节中，我们将以最简单的方式使用`pytest`。在章节的后面部分，我们将介绍更多有趣的案例。

请确保通过 pip 在您的环境中安装`pytest`。

```py
$ pip3 install pytest 
```

让我们看看如何在`test_pytest_example.py`文件中运行定义的测试。

```py
from tdd_example import parameter_tdd

def test_negative():

    assert parameter_tdd(-1) == 0

def test_zero():

    assert parameter_tdd(0) == 0

def test_five():

    assert parameter_tdd(5) == 25

def test_seven():

    # Note this test is deliberatly set to fail

    assert parameter_tdd(7) == 0

def test_ten():

    assert parameter_tdd(10) == 100

def test_eleven():

    assert parameter_tdd(11) == 100 
```

如果您将其与`test_unittest_example.py`中的等效代码进行比较，代码会显著更简洁。当使用`pytest`运行时，它也会显示更详细、带颜色的信息。

```py
$ pytest test_unittest_example.py

================= test session starts =================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1

collected 6 items

test_unittest_example.py ...F..                 [100%]

====================== FAILURES =======================

______________ TestTDDExample.test_seven ______________

self = <test_unittest_example.TestTDDExample testMethod=test_seven>

    def test_seven(self):

>       self.assertEqual(parameter_tdd(7), 0)

E       AssertionError: 49 != 0

test_unittest_example.py:17: AssertionError

=============== short test summary info ===============

FAILED test_unittest_example.py::TestTDDExample::test_seven

============= 1 failed, 5 passed in 0.10s ============= 
```

与`unittest`一样，我们可以使用`-v`选项看到更多信息，并使用`-k`选项运行测试的选择。

```py
$ pytest -v test_unittest_example.py

========================= test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

cachedir: .pytest_cache

collected 6 items

test_unittest_example.py::TestTDDExample::test_eleven PASSED      [16%]

test_unittest_example.py::TestTDDExample::test_five PASSED        [33%]

test_unittest_example.py::TestTDDExample::test_negative PASSED    [50%]

test_unittest_example.py::TestTDDExample::test_seven FAILED       [66%]

test_unittest_example.py::TestTDDExample::test_ten PASSED         [83%]

test_unittest_example.py::TestTDDExample::test_zero PASSED        [100%]

============================== FAILURES ===============================

______________________ TestTDDExample.test_seven ______________________

self = <test_unittest_example.TestTDDExample testMethod=test_seven>

    def test_seven(self):

>       self.assertEqual(parameter_tdd(7), 0)

E       AssertionError: 49 != 0

test_unittest_example.py:17: AssertionError

======================= short test summary info =======================

FAILED test_unittest_example.py::TestTDDExample::test_seven - AssertionErr...

===================== 1 failed, 5 passed in 0.08s =====================

$ pytest test_pytest_example.py -v -k test_ten

========================= test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

cachedir: .pytest_cache

collected 6 items / 5 deselected / 1 selected

test_pytest_example.py::test_ten PASSED                           [100%]

=================== 1 passed, 5 deselected in 0.02s =================== 
```

它与`unittest`定义的测试完全兼容，这允许你结合两种风格或迁移它们。

```py
$ pytest test_unittest_example.py

========================= test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1

collected 6 items

test_unittest_example.py ...F..                                   [100%]

============================== FAILURES ===============================

______________________ TestTDDExample.test_seven ______________________

self = <test_unittest_example.TestTDDExample testMethod=test_seven>

    def test_seven(self):

>       self.assertEqual(parameter_tdd(7), 0)

E       AssertionError: 49 != 0

test_unittest_example.py:17: AssertionError

======================= short test summary info =======================

FAILED test_unittest_example.py::TestTDDExample::test_seven - AssertionErr...

===================== 1 failed, 5 passed in 0.08s ===================== 
```

`pytest`的另一个出色功能是易于自动发现以查找以`test_`开头的文件，并在所有测试中运行。如果我们尝试它，指向当前目录，我们可以看到它运行了`test_unittest_example.py`和`test_pytest_example.py`。

```py
$ pytest .

========================= test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1

collected 12 items

test_pytest_example.py ...F..                                    [50%]

test_unittest_example.py ...F..                                  [100%]

============================== FAILURES ===============================

_____________________________ test_seven ______________________________

    def test_seven():

        # Note this test is deliberatly set to fail

>       assert parameter_tdd(7) == 0

E       assert 49 == 0

E        +  where 49 = parameter_tdd(7)

test_pytest_example.py:18: AssertionError

______________________ TestTDDExample.test_seven ______________________

self = <test_unittest_example.TestTDDExample testMethod=test_seven>

    def test_seven(self):

>       self.assertEqual(parameter_tdd(7), 0)

E       AssertionError: 49 != 0

test_unittest_example.py:17: AssertionError

======================= short test summary info =======================

FAILED test_pytest_example.py::test_seven - assert 49 == 0

FAILED test_unittest_example.py::TestTDDExample::test_seven - AssertionErr...

==================== 2 failed, 10 passed in 0.23s ===================== 
```

在本章中，我们将继续讨论`pytest`的更多功能，但首先，我们需要回到如何定义代码有依赖时的测试。

# 测试外部依赖

在构建单元测试时，我们讨论了它是如何围绕将代码中的单元隔离以独立测试的概念。

这种隔离概念是关键，因为我们想专注于代码的小部分来创建小而清晰的测试。创建小测试也有助于保持测试的快速。

在我们上面的例子中，我们测试了一个没有依赖的纯功能函数`parameter_tdd`。它没有使用任何外部库或任何其他函数。但不可避免的是，在某个时候，你需要测试依赖于其他东西的东西。

在这种情况下的问题是*其他组件是否应该包含在测试中？*

这是一个不容易回答的问题。一些开发者认为所有单元测试都应该纯粹关于一个函数或方法，因此任何依赖都不应该包含在测试中。但是，在更实际的水平上，有时有一些代码片段形成一个单元，联合测试比单独测试更容易。

例如，考虑一个函数：

+   对于小于 0 的值，返回 0。

+   对于大于 100 的值，返回 10。

+   对于介于两者之间的值，它返回值的平方根。注意，对于边缘值，它返回它们的平方根（0 对于 0 和 10 对于 100）。

这与上一个函数`parameter_tdd`非常相似，但这次我们需要外部库的帮助来计算数字的平方根。让我们看看代码。

它分为两个文件。`dependent.py`包含函数的定义。

```py
import math

def parameter_dependent(value):

    if value < 0:

        return 0

    if value <= 100:

        return math.sqrt(value)

    return 10 
```

代码与`parameter_tdd`示例中的代码非常相似。模块`math.sqrt`返回数字的平方根。

测试文件位于`test_dependent.py`。

```py
from dependent import parameter_dependent

def test_negative():

    assert parameter_dependent(-1) == 0

def test_zero():

    assert parameter_dependent(0) == 0

def test_twenty_five():

    assert parameter_dependent(25) == 5

def test_hundred():

    assert parameter_dependent(100) == 10

def test_hundred_and_one():

    assert parameter_dependent(101) == 10 
```

在这种情况下，我们完全使用外部库，并在测试我们的代码的同时测试它。对于这个简单的例子，这是一个完全有效的选项，尽管对于其他情况可能并非如此。

代码可在 GitHub 上找到：[`github.com/PacktPublishing/Python-Architecture-Patterns/tree/main/chapter_10_testing_and_tdd`](https://github.com/PacktPublishing/Python-Architecture-Patterns/tree/main/chapter_10_testing_and_tdd)。

例如，外部依赖可能是需要捕获的外部 HTTP 调用，以防止在运行测试时执行它们，并控制返回的值，或者应该单独测试的其他大型功能功能。

要将函数与其依赖项分离，有两种不同的方法。我们将使用 `parameter_dependent` 作为基准来展示它们。

再次强调，在这种情况下，包含依赖项时测试工作得非常完美，因为它简单且不会产生像外部调用等副作用。

我们将在下一节中看到如何模拟外部调用。

## 模拟

模拟是一种内部替换依赖项的实践，用假的调用替换它们，由测试本身控制。这样，我们可以为任何外部依赖项引入已知的响应，而不调用实际的代码。

在内部，模拟是通过所谓的 *monkey-patching* 实现的，即用替代品动态替换现有库。虽然这可以在不同的编程语言中以不同的方式实现，但在像 Python 或 Ruby 这样的动态语言中特别流行。虽然模拟可以用于测试之外的其他目的，但应该谨慎使用，因为它可以改变库的行为，并且对于调试来说可能会相当令人不安。

为了能够模拟代码，在我们的测试代码中，我们需要在 Arrange 步骤中准备模拟。有不同库可以模拟调用，但最简单的是使用标准库中包含的 `unittest.mock` 库。

`mock` 的最简单用法是修补一个外部库：

```py
from unittest.mock import patch

from dependent import parameter_dependent

@patch('math.sqrt')

def test_twenty_five(mock_sqrt):

    mock_sqrt.return_value = 5

    assert parameter_dependent(25) == 5

    mock_sqrt.assert_called_once_with(25) 
```

`patch` 装饰器拦截对定义的库 `math.sqrt` 的调用，并用一个 `mock` 对象替换它，这里称为 `mock_sqrt`。

这个对象有点特殊。它基本上允许任何调用，几乎可以访问任何方法或属性（除了预定义的），并且持续返回一个模拟对象。这使得模拟对象非常灵活，可以适应周围的任何代码。当需要时，可以通过调用 `.return_value` 来设置返回值，就像我们在第一行中展示的那样。

从本质上讲，我们是在说对 `mock_sqrt` 的调用将返回值 5。因此，我们正在准备外部调用的输出，以便我们可以控制它。

最后，我们检查我们只调用了一次模拟 `mock_sqrt`，使用输入（`25`）并通过 `assert_called_once_with` 方法。

从本质上讲，我们是在：

+   准备模拟以替换 `math.sqrt`

+   设置它在被调用时将返回的值

+   检查调用是否按预期工作

+   再次确认模拟是否以正确的值被调用

对于其他测试，例如，我们可以检查模拟没有被调用，这表明外部依赖没有被调用。

```py
@patch('math.sqrt')

def test_hundred_and_one(mock_sqrt):

    assert parameter_dependent(101) == 10

    mock_sqrt.assert_not_called() 
```

有多个 `assert` 函数允许你检测模拟是如何被使用的。以下是一些示例：

+   `called` 属性根据模拟是否被调用返回 `True` 或 `False`，允许你编写：

    ```py
    `assert mock_sqrt.called is True` 
    ```

+   `call_count` 属性返回模拟被调用的次数。

+   `assert_called_with()` 方法用于检查它被调用的次数。如果最后的调用不是以指定的方式产生的，它将引发异常。

+   `assert_any_call()` 方法用于检查是否以指定方式产生了任何调用。

根据这些信息，完整的测试文件 `test_dependent_mocked_test.py` 将如下所示。

```py
from unittest.mock import patch

from dependent import parameter_dependent

@patch('math.sqrt')

def test_negative(mock_sqrt):

    assert parameter_dependent(-1) == 0

    mock_sqrt.assert_not_called()

@patch('math.sqrt')

def test_zero(mock_sqrt):

    mock_sqrt.return_value = 0

    assert parameter_dependent(0) == 0

    mock_sqrt.assert_called_once_with(0)

@patch('math.sqrt')

def test_twenty_five(mock_sqrt):

    mock_sqrt.return_value = 5

    assert parameter_dependent(25) == 5

    mock_sqrt.assert_called_with(25)

@patch('math.sqrt')

def test_hundred(mock_sqrt):

    mock_sqrt.return_value = 10

    assert parameter_dependent(100) == 10

    mock_sqrt.assert_called_with(100)

@patch('math.sqrt')

def test_hundred_and_one(mock_sqrt):

    assert parameter_dependent(101) == 10

    mock_sqrt.assert_not_called() 
```

如果模拟需要返回不同的值，你可以将模拟的 `side_effect` 属性定义为列表或元组。`side_effect` 与 `return_value` 类似，但它有一些区别，我们将在下面看到。

```py
@patch('math.sqrt')

def test_multiple_returns_mock(mock_sqrt):

    mock_sqrt.side_effect = (5, 10)

    assert parameter_dependent(25) == 5

    assert parameter_dependent(100) == 10 
```

如果需要，`side_effect` 也可以用来产生异常。

```py
import pytest

from unittest.mock import patch

from dependent import parameter_dependent

@patch('math.sqrt')

def test_exception_raised_mock(mock_sqrt):

    mock_sqrt.side_effect = ValueError('Error on the external library')

    with pytest.raises(ValueError):

        parameter_dependent(25) 
```

`with` 块断言在块中引发了预期的 `Exception`。如果没有，它将显示错误。

在 `unittest` 中，可以使用类似的 `with` 块来检查引发的异常。

`with self.assertRaises(ValueError):`

`parameter_dependent(25)`

Mocking 不是处理测试依赖关系的唯一方法。我们将在下一个示例中看到不同的方法。

## 依赖注入

当通过外部修补来替换依赖关系时，mocking 不会让原始代码察觉到，而依赖注入是一种在调用测试函数时使该依赖关系明确的技术，这样就可以用测试替身来替换它。

本质上，这是一种设计代码的方式，通过要求它们作为输入参数来明确依赖关系。

虽然依赖注入对测试很有用，但它并不仅限于此。通过显式添加依赖关系，它还减少了函数需要知道如何初始化特定依赖关系的需要，而是依赖于依赖关系的接口。它创建了“初始化”依赖关系（这应该由外部处理）和“使用”它（这是依赖代码唯一会做的部分）之间的分离。这种区分将在我们看到面向对象示例时变得更加清晰。

让我们看看这如何改变测试中的代码。

```py
def parameter_dependent(value, sqrt_func):

    if value < 0:

        return 0

    if value <= 100:

        return sqrt_func(value)

    return 10 
```

注意现在 `sqrt` 函数是一个输入参数。

如果我们想在正常场景中使用 `parameter_dependent` 函数，我们必须产生依赖关系，例如。

```py
import math

def test_good_dependency():

    assert parameter_dependent(25, math.sqrt) == 5 
```

如果我们想进行测试，我们可以通过用特定的函数替换 `math.sqrt` 函数，然后使用它来完成。例如：

```py
def test_twenty_five():

    def good_dependency(number):

        return 5

    assert parameter_dependent(25, good_dependency) == 5 
```

我们也可以通过调用依赖关系来引发错误，以确保在某些测试中依赖关系没有被使用，例如。

```py
def test_negative():

    def bad_dependency(number):

        raise Exception('Function called')

    assert parameter_dependent(-1, bad_dependency) == 0 
```

注意这种方法比 mocking 更明确。要测试的代码在本质上完全功能，因为它没有外部依赖。

## 面向对象编程中的依赖注入

依赖注入也可以与面向对象编程（OOP）一起使用。在这种情况下，我们可以从类似这样的代码开始。

```py
class Writer:

    def __init__(self):

        self.path = settings.WRITER_PATH

    def write(self, filename, data):

        with open(self.path + filename, 'w') as fp:

            fp.write(data)

class Model:

    def __init__(self, data):

        self.data = data

        self.filename = settings.MODEL_FILE

        self.writer = Writer()

    def save(self):

        self.writer.write(self.filename, self.data) 
```

如我们所见，`settings` 类存储了在数据将存储的位置所需的不同元素。模型接收一些数据然后保存。正在运行的代码将需要最少的初始化。

```py
 model = Model('test')

    model.save() 
```

模型接收一些数据然后保存它。运行中的代码需要最小的初始化，但与此同时，它并不明确。

要使用依赖注入原则，代码需要以这种方式编写：

```py
class WriterInjection:

    def __init__(self, path):

        self.path = path

    def write(self, filename, data):

        with open(self.path + filename, 'w') as fp:

            fp.write(data)

class ModelInjection:

    def __init__(self, data, filename, writer):

        self.data = data

        self.filename = filename

        self.writer = writer

    def save(self):

        self.writer.write(self.filename, self.data) 
```

在这种情况下，每个作为依赖的值都是明确提供的。在代码的定义中，`settings`模块根本不存在，而是在类实例化时指定。现在代码需要直接定义配置。

```py
 writer = WriterInjection('./')

    model = ModelInjection('test', 'model_injection.txt', writer)

    model.save() 
```

我们可以比较如何测试这两种情况，如文件`test_dependency_injection_test.py`中所示。第一个测试是模拟，正如我们之前所见，模拟`Writer`类的`write`方法以断言它已被正确调用。

```py
@patch('class_injection.Writer.write')

def test_model(mock_write):

    model = Model('test_model')

    model.save()

    mock_write.assert_called_with('model.txt', 'test_model') 
```

与之相比，依赖注入的示例不需要通过猴子补丁进行模拟。它只是创建了自己的`Writer`来模拟接口。

```py
def test_modelinjection():

    EXPECTED_DATA = 'test_modelinjection'

    EXPECTED_FILENAME = 'model_injection.txt'

    class MockWriter:

        def write(self, filename, data):

            self.filename = filename

            self.data = data

    writer = MockWriter()

    model = ModelInjection(EXPECTED_DATA, EXPECTED_FILENAME,

                           writer)

    model.save()

    assert writer.data == EXPECTED_DATA

    assert writer.filename == EXPECTED_FILENAME 
```

第二种风格更冗长，但它展示了以这种方式编写代码时的一些差异：

+   不需要猴子补丁模拟。猴子补丁可能会相当脆弱，因为它是在干预不应该暴露的内部代码。虽然在进行测试时这种干预与为常规代码运行时所做的干预不同，但它仍然可能造成混乱并产生意外的效果，尤其是如果内部代码以某种不可预见的方式发生变化。

    请记住，模拟可能在某个时候涉及到与二级依赖相关的内容，这可能会产生奇怪或复杂的效果，需要你花费时间处理额外的复杂性。

+   编写代码的方式本身也有所不同。使用依赖注入产生的代码，正如我们所见，更模块化，由更小的元素组成。这往往会产生更小、更可组合的模块，它们可以协同工作，因为它们的依赖关系总是明确的。

+   虽然如此，但请注意，这需要一定的纪律和心智框架来产生真正松散耦合的模块。如果在设计接口时没有考虑到这一点，生成的代码将人为地分割，导致不同模块之间紧密耦合。培养这种纪律需要一定的训练；不要期望所有开发者都能自然而然地做到。

+   有时代码可能更难调试，因为配置将与代码的其他部分分离，有时这会使理解代码流程变得困难。复杂性可能在类之间的交互中产生，这可能更难以理解和测试。通常，以这种方式开发代码的前期工作也会更多一些。

依赖注入是某些软件领域和编程语言中非常流行的一种技术。在比 Python 更静态的语言中，模拟会更困难，而且不同的编程语言都有自己的代码结构理念。例如，依赖注入在 Java 中非常流行，那里有特定的工具来支持这种风格。

# 高级 pytest

虽然我们已经描述了`pytest`的基本功能，但在展示其帮助生成测试代码的可能性方面，我们只是触及了表面。

Pytest 是一个庞大且全面的工具。学习如何使用它是值得的。在这里，我们只会触及表面。请务必查看官方文档[`docs.pytest.org/`](https://docs.pytest.org/)。

不一一列举，我们将看到这个工具的一些有用可能性。

## 分组测试

有时候将测试分组在一起是有用的，这样它们就与特定的事物相关联，比如模块，或者一起运行。将测试分组在一起的最简单方法是将它们组合成一个单独的类。

例如，回到之前的测试示例，我们可以将测试结构化为两个类，就像我们在`test_group_classes.py`中看到的那样。

```py
from tdd_example import parameter_tdd

class TestEdgesCases():

    def test_negative(self):

        assert parameter_tdd(-1) == 0

    def test_zero(self):

        assert parameter_tdd(0) == 0

    def test_ten(self):

        assert parameter_tdd(10) == 100

    def test_eleven(self):

        assert parameter_tdd(11) == 100

class TestRegularCases():

    def test_five(self):

        assert parameter_tdd(5) == 25

    def test_seven(self):

        assert parameter_tdd(7) == 49 
```

这是一个将测试分割开来的简单方法，允许你独立运行它们：

```py
$ pytest -v test_group_classes.py

======================== test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

collected 6 items

test_group_classes.py::TestEdgesCases::test_negative PASSED      [16%]

test_group_classes.py::TestEdgesCases::test_zero PASSED          [33%]

test_group_classes.py::TestEdgesCases::test_ten PASSED           [50%]

test_group_classes.py::TestEdgesCases::test_eleven PASSED        [66%]

test_group_classes.py::TestRegularCases::test_five PASSED        [83%]

test_group_classes.py::TestRegularCases::test_seven PASSED       [100%]

========================= 6 passed in 0.02s ==========================

$ pytest -k TestRegularCases -v test_group_classes.py

========================= test session starts ========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

collected 6 items / 4 deselected / 2 selected

test_group_classes.py::TestRegularCases::test_five PASSED        [50%]

test_group_classes.py::TestRegularCases::test_seven PASSED       [100%]

================== 2 passed, 4 deselected in 0.02s ===================

$ pytest -v test_group_classes.py::TestRegularCases

========================= test session starts ========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

cachedir: .pytest_cache

rootdir: /Users/jaime/Dropbox/Packt/architecture_book/chapter_09_testing_and_tdd/advanced_pytest

plugins: celery-4.4.7

collected 2 items

test_group_classes.py::TestRegularCases::test_five PASSED        [50%]

test_group_classes.py::TestRegularCases::test_seven PASSED       [100%]

========================== 2 passed in 0.02s ========================= 
```

另一种可能性是使用标记。标记是可以通过测试中的装饰器添加的指示器，例如，在`test_markers.py`中。

```py
import pytest

from tdd_example import parameter_tdd

@pytest.mark.edge

def test_negative():

    assert parameter_tdd(-1) == 0

@pytest.mark.edge

def test_zero():

    assert parameter_tdd(0) == 0

def test_five():

    assert parameter_tdd(5) == 25

def test_seven():

    assert parameter_tdd(7) == 49

@pytest.mark.edge

def test_ten():

    assert parameter_tdd(10) == 100

@pytest.mark.edge

def test_eleven():

    assert parameter_tdd(11) == 100 
```

注意我们正在定义一个装饰器，`@pytest.mark.edge`，用于检查所有测试的值边界。

如果我们执行测试，可以使用参数 `-m` 来运行具有特定标签的测试。

```py
 $ pytest -m edge -v test_markers.py

========================= test session starts ========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

collected 6 items / 2 deselected / 4 selected

test_markers.py::test_negative PASSED                            [25%]

test_markers.py::test_zero PASSED                                [50%]

test_markers.py::test_ten PASSED                                 [75%]

test_markers.py::test_eleven PASSED                              [100%]

========================== warnings summary ==========================

test_markers.py:5

  test_markers.py:5: PytestUnknownMarkWarning: Unknown pytest.mark.edge - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/mark.html

    @pytest.mark.edge

test_markers.py:10

...

-- Docs: https://docs.pytest.org/en/stable/warnings.html

============ 4 passed, 2 deselected, 4 warnings in 0.02s ============= 
```

如果标记`edge`未注册，将产生警告`PytestUnknownMarkWarning: Unknown pytest.mark.edge`。

注意 GitHub 代码中包含了`pytest.ini`代码。如果存在`pytest.ini`文件，例如，如果你克隆了整个仓库，你将不会看到警告。

这对于查找错误非常有用，比如不小心写成`egde`或类似的错误。为了避免这种警告，你需要添加一个`pytest.ini`配置文件，其中包含标记的定义，如下所示。

```py
[pytest]

markers =

       edge: tests related to edges in intervals 
```

现在，运行测试不再显示警告。

```py
$ pytest -m edge -v test_markers.py

========================= test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

cachedir: .pytest_cache

rootdir: /Users/jaime/Dropbox/Packt/architecture_book/chapter_09_testing_and_tdd/advanced_pytest, configfile: pytest.ini

plugins: celery-4.4.7

collected 6 items / 2 deselected / 4 selected

test_markers.py::test_negative PASSED                            [25%]

test_markers.py::test_zero PASSED                                [50%]

test_markers.py::test_ten PASSED                                 [75%]

test_markers.py::test_eleven PASSED                              [100%]

=================== 4 passed, 2 deselected in 0.02s =================== 
```

注意标记可以在整个测试套件中使用，包括多个文件。这允许标记识别测试中的常见模式，例如，创建一个带有标记`basic`的快速测试套件，以运行最重要的测试。

此外，还有一些预定义的标记，具有一些内置功能。最常见的是`skip`（将跳过测试）和`xfail`（将反转测试，意味着它期望它失败）。

## 使用 fixtures

在`pytest`中，使用 fixtures 是设置测试的首选方式。本质上，fixture 是为了设置测试而创建的上下文。

Fixtures 被用作测试函数的输入，因此它们可以设置并创建特定的测试环境。

例如，让我们看看一个简单的函数，它计算字符串中字符出现的次数。

```py
def count_characters(char_to_count, string_to_count):

    number = 0

    for char in string_to_count:

        if char == char_to_count:

            number += 1

    return number 
```

这是一个相当简单的循环，它遍历字符串并计数匹配的字符。

这相当于使用字符串的`.count()`函数，但这是为了展示一个工作函数。之后可以对其进行重构！

一个常规测试来覆盖功能可能如下。

```py
def test_counting():

    assert count_characters('a', 'Barbara Ann') == 3 
```

非常直接。现在让我们看看我们如何定义一个固定装置来定义一个设置，以防我们想要复制它。

```py
import pytest

@pytest.fixture()

def prepare_string():

    # Setup the values to return

    prepared_string = 'Ba, ba, ba, Barbara Ann'

    # Return the value

    yield prepared_string

    # Teardown any value

    del prepared_string 
```

首先，固定装置被装饰为`pytest.fixture`以标记它。固定装置分为三个步骤：

+   **设置**：在这里，我们只是定义了一个字符串，但这可能是最大的部分，其中准备值。

+   **返回值**：如果我们使用`yield`功能，我们将能够进入下一步；如果不使用，固定装置将在这里结束。

+   **拆卸和清理值**：在这里，我们只是简单地删除变量作为示例，尽管这将在稍后自动发生。

    之后，我们将看到一个更复杂的固定装置。在这里，我们只是展示这个概念。

以这种方式定义固定装置将使我们能够轻松地在不同的测试函数中重用它，只需使用名称作为输入参数。

```py
def test_counting_fixture(prepare_string):

    assert count_characters('a', prepare_string) == 6

def test_counting_fixture2(prepare_string):

    assert count_characters('r', prepare_string) == 2 
```

注意`prepare_string`参数是如何自动提供我们使用`yield`定义的值的。如果我们运行测试，我们可以看到效果。甚至更多，我们可以使用参数`--setup-show`来查看设置和拆卸所有固定装置。

```py
$ pytest -v test_fixtures.py -k counting_fixture --setup-show

======================== test session starts ========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

plugins: celery-4.4.7

collected 3 items / 1 deselected / 2 selected

test_fixtures.py::test_counting_fixture

        SETUP    F prepare_string

        test_fixtures.py::test_counting_fixture (fixtures used: prepare_string)PASSED

        TEARDOWN F prepare_string

test_fixtures.py::test_counting_fixture2

        SETUP    F prepare_string

        test_fixtures.py::test_counting_fixture2 (fixtures used: prepare_string)PASSED

        TEARDOWN F prepare_string

=================== 2 passed, 1 deselected in 0.02s =================== 
```

这个固定装置非常简单，并没有做任何不能通过定义字符串来完成的事情，但固定装置可以用来连接数据库或准备文件，同时考虑到它们可以在结束时清理它们。

例如，稍微复杂化同一个例子，而不是从字符串中计数，它应该从文件中计数，因此函数需要打开文件，读取它，并计数字符。函数将如下所示。

```py
def count_characters_from_file(char_to_count, file_to_count):

    '''

    Open a file and count the characters in the text contained

    in the file

    '''

    number = 0

    with open(file_to_count) as fp:

        for line in fp:

            for char in line:

                if char == char_to_count:

                    number += 1

    return number 
```

固定装置应该创建一个文件，返回它，然后在拆卸过程中删除它。让我们看看它。

```py
import os

import time

import pytest

@pytest.fixture()

def prepare_file():

    data = [

        'Ba, ba, ba, Barbara Ann',

        'Ba, ba, ba, Barbara Ann',

        'Barbara Ann',

        'take my hand',

    ]

    filename = f'./test_file_{time.time()}.txt'

    # Setup the values to return

    with open(filename, 'w') as fp:

        for line in data:

            fp.write(line)

    # Return the value

    yield filename

    # Delete the file as teardown

    os.remove(filename) 
```

注意，在文件名中，我们定义了名称，并在生成时添加时间戳。这意味着将由该固定装置生成的每个文件都是唯一的。

```py
 filename = f'./test_file_{time.time()}.txt' 
```

然后，文件被创建，数据被写入。

```py
 with open(filename, 'w') as fp:

        for line in data:

            fp.write(line) 
```

文件名，正如我们所看到的，是唯一的，被返回。最后，在拆卸过程中删除该文件。

测试与之前的类似，因为大部分复杂性都存储在固定装置中。

```py
def test_counting_fixture(prepare_file):

    assert count_characters_from_file('a', prepare_file) == 17

def test_counting_fixture2(prepare_file):

    assert count_characters_from_file('r', prepare_file) == 6 
```

当运行它时，我们看到它按预期工作，并且我们可以检查在每次测试后拆卸步骤会删除测试文件。

```py
$ pytest -v test_fixtures2.py

========================= test session starts =========================

platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /usr/local/opt/python@3.9/bin/python3.9

collected 2 items

test_fixtures2.py::test_counting_fixture PASSED                  [50%]

test_fixtures2.py::test_counting_fixture2 PASSED                 [100%]

========================== 2 passed in 0.02s ========================== 
```

固定装置不需要定义在同一个文件中。它们也可以存储在一个名为`conftest.py`的特殊文件中，该文件将由`pytest`自动在所有测试中共享。

固定装置也可以组合，可以设置为自动使用，并且已经内置了用于处理时间数据和目录或捕获输出的固定装置。PyPI 上还有很多用于有用固定装置的插件，可以作为第三方模块安装，包括连接数据库或与其他外部资源交互等功能。务必检查 Pytest 文档，并在实现自己的固定装置之前进行搜索，以查看是否可以利用现有的模块：[`docs.pytest.org/en/latest/explanation/fixtures.html#about-fixtures`](https://docs.pytest.org/en/latest/explanation/fixtures.html#about-fixtures)。

在本章中，我们只是对`pytest`的可能性进行了初步探索。这是一个出色的工具，我鼓励你们去了解它。高效地运行测试并以最佳方式设计测试将带来巨大的回报。测试是项目的一个关键部分，也是开发者花费大部分时间的一个开发阶段。

# 摘要

在本章中，我们探讨了测试的为什么和怎么做，以描述一个好的测试策略对于生产高质量的软件和防止代码在使用过程中出现问题的必要性。

我们首先描述了测试背后的基本原则，如何编写比成本更高的测试，以及确保这一点的不同测试级别。我们看到了三个主要的测试级别，我们称之为单元测试（单个组件的部分）、系统测试（整个系统）和中间的集成测试（一个或多个组件的整体，但不是全部）。

我们继续描述了确保我们的测试是优秀测试的不同策略，以及如何使用 Arrange-Act-Assert 模式来构建它们，以便在编写后易于编写和理解。

之后，我们详细描述了测试驱动开发背后的原则，这是一种将测试置于开发中心的技术，要求在编写代码之前编写测试，以小步增量工作，并反复运行测试以创建一个良好的测试套件，以防止意外行为。我们还分析了以 TDD 方式工作的局限性和注意事项，并提供了流程的示例。

我们继续通过展示在 Python 中创建单元测试的方法来介绍，包括使用标准的`unittest`模块和引入更强大的`pytest`。我们还提供了一个关于`pytest`高级使用的部分，以展示这个优秀的第三方模块的能力。

我们描述了如何测试外部依赖，这在编写单元测试以隔离功能时至关重要。我们还描述了如何模拟依赖关系以及如何在依赖注入原则下工作。
