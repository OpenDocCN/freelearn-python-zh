# 第八章

项目 2.5：模式和元数据

将数据模式与共享该模式的各个应用程序保持分离是有帮助的。实现这一目标的一种方法是为套件中的所有应用程序创建一个具有类定义的单独模块。虽然这对简单项目有帮助，但在更广泛地共享数据模式时可能会有些尴尬。Python 语言模块在共享 Python 环境之外的数据时尤其困难。

本项目将使用 JSON Schema 语法定义一个模式，首先通过构建 `pydantic` 类定义，然后从类定义中提取 JSON。这将允许您发布正在创建的数据的正式定义。该模式可以被各种工具用于验证数据文件，并确保数据适合进一步的分析使用。

模式对于诊断数据源的问题也很有用。像 `jsonschema` 这样的验证工具可以提供详细的错误报告，有助于识别源数据中的更改，这些更改可能是由于错误修复或软件更新。

本章将涵盖与数据检查技术相关的多个技能：

+   使用 **Pydantic** 模块进行清晰、完整的定义

+   使用 JSON Schema 创建一个可导出且语言无关的定义，任何人都可以使用

+   创建用于使用正式模式定义的测试场景

我们将首先探讨正式模式为什么有帮助的原因。

## 8.1 描述

在应用程序之间移动数据时，数据验证是一个常见的要求。拥有一个明确的数据有效性的定义非常有帮助。当这个定义存在于特定的编程语言或平台之外时，帮助就更大了。

我们可以使用 JSON Schema ([`json-schema.org`](https://json-schema.org)) 来定义一个适用于由获取过程创建的中间文档的模式。使用 JSON Schema 可以使 JSON 数据格式的使用更加自信和可靠。

JSON Schema 定义可以在不同的 Python 项目和非 Python 环境中共享和重用。它允许我们将数据质量检查集成到获取管道中，以积极确认数据确实符合分析和处理的要求。

使用模式提供的附加元数据通常包括数据的来源和属性值是如何导出的详细信息。这不是 JSON 模式的一部分，但我们可以向 JSON 模式文档中添加一些包含来源和处理描述的详细信息。

随后的数据清洗项目应使用源架构验证输入文档。从第九章*项目 3.1：数据清洗基础应用*开始，应用应使用目标分析架构验证其输出。一个应用既创建样本记录又验证这些记录是否符合架构可能看起来有些荒谬。重要的是架构将是共享的，并随着数据消费者的需求而发展。另一方面，数据采集和清洗操作随着数据源的发展而发展。一个临时的数据问题解决方案看起来很好，但可能创建无效数据的情况非常普遍。

验证输入和输出是否符合可见的、达成一致的架构很少会创建新的问题。验证操作会有一些开销，但大部分处理成本是由输入和输出的时间决定的，而不是数据验证。

展望第十二章*项目 3.8：集成数据采集 Web 服务*，我们将看到正式定义的架构的更多用途。我们还将揭示使用 JSON Schema 描述 ND JSON 文档时存在的一个小问题。目前，我们将专注于使用 JSON Schema 描述数据的需求。

我们将首先添加一些模块，以便更容易创建 JSON Schema 文档。

## 8.2 方法

首先，我们需要一些额外的模块。`jsonschema`模块定义了一个验证器，可以用来确认文档是否符合定义的架构。

此外，**Pydantic**模块提供了一种创建可以发出 JSON Schema 定义的类定义的方法，这样我们就不必手动创建架构。在大多数情况下，手动创建架构并不特别困难。然而，在某些情况下，架构和验证规则可能难以直接编写，并且有 Python 类定义可用可以简化这个过程。

这需要添加到`requirements-dev.txt`文件中，以便其他开发者知道安装它。

当使用**conda**管理虚拟环境时，命令可能如下所示：

```py
% conda install jsonschema pydantic
```

当使用其他工具管理虚拟环境时，命令可能如下所示：

```py
% python -m pip install jupyterlab
```

JSON Schema 包需要一些补充的类型存根。这些由**mypy**工具使用，以确认应用程序正在一致地使用类型。使用以下命令添加存根：

```py
% mypy --install-types
```

此外，`pydantic`包包括一个**mypy**插件，它将扩展**mypy**的类型检查功能。这将发现使用`pydantic`定义的类中更多细微的潜在问题。

要启用插件，将`pydantic.mypy`添加到**mypy**配置文件`mypy.ini`中插件列表。`mypy.ini`文件应如下所示：

```py
[mypy]
plugins = pydantic.mypy
```

（此文件应放在项目目录的根目录下。）

此插件是 **pydantic** 下载的一部分，并且与从 0.910 版本开始的 **mypy** 兼容。

使用这两个包，我们可以定义具有详细信息的类，这些信息可用于创建 JSON Schema 文件。一旦我们有了 JSON Schema 文件，我们就可以使用该模式定义来确认样本数据的有效性。

有关 **Pydantic** 的更多信息，请参阅 [`docs.pydantic.dev`](https://docs.pydantic.dev)。

核心概念是使用 **Pydantic** 来定义具有详细字段定义的数据类。这些定义可以用于 Python 中的数据验证。定义也可以用来生成一个 JSON Schema 文档，以便与其他项目共享。

模式定义对于定义 OpenAPI 规范也很有用。在 *第十二章*，*项目 3.8：集成数据采集网络服务* 中，我们将转向创建提供数据的网络服务。此服务的 OpenAPI 规范将包括来自此项目的模式定义。

使用 **Pydantic** 不是必需的。然而，它对于创建可以通过 JSON Schema 描述的模式来说非常方便。它节省了大量与 JSON 语法细节的纠缠。

我们将开始使用 **Pydantic** 创建一个有用的数据模型模块。这将扩展早期章节中为项目构建的数据模型。

### 8.2.1 定义 Pydantic 类并生成 JSON Schema

我们将从对早期章节中使用的数据模型定义进行两个深刻的修改开始。一个变化是将从 `dataclasses` 模块切换到 `pydantic.dataclasses` 模块。这样做需要显式使用 `dataclasses.field` 进行单个字段定义。这通常是对 `import` 语句的一个小改动，使用 `from pydantic.dataclasses import dataclass`。数据类的 `field()` 函数也需要一些更改，以添加 **pydantic** 所使用的额外细节。这些更改对现有应用程序应该是完全透明的；所有测试在更改后都将通过。

第二个变化是为类添加一些重要的元数据。在 `dataclasses.field(...)` 定义中使用的地方，可以添加 `metadata={}` 属性，以包含一个包含 JSON Schema 属性的字典，如描述、标题、示例、值的有效范围等。对于其他字段，必须使用 `pydantic.Field()` 函数来提供标题、描述和其他字段约束。这将为我们生成大量的元数据。

有关可用的各种字段定义细节，请参阅 [`docs.pydantic.dev/usage/schema/#field-customization`](https://docs.pydantic.dev/usage/schema/#field-customization)。

```py
from pydantic import Field
from pydantic.dataclasses import dataclass

@dataclass
class SeriesSample:
    """
    An individual sample value.
    """
    x: float = Field(title="The x attribute", ge=0.0)
    y: float = Field(title="The y attribute", ge=0.0)

@dataclass
class Series:
    """
    A named series with a collection of values.
    """
    name: str = Field(title="Series name")
    samples: list[SeriesSample] = Field(title="Sequence of samples
      in this series")
```

我们在这个模型定义模块中提供了几个额外的细节。这些细节包括：

+   每个类的文档字符串。这些将成为 JSON Schema 中的描述。

+   每个属性的字段。这些字段也成为了 JSON Schema 中的描述。

+   对于`SeriesSample`类定义的`x`和`y`属性，我们添加了一个`ge`值。这是一个范围规范，要求值大于或等于零。

我们还对模型进行了极其深刻的变化：我们从源数据描述——即多个`str`值——转变为目标数据描述，使用`float`值。

在这里的核心是，我们对每个模型有两种变体：

+   **获取**：这是我们在“野外”找到的数据。在本书的例子中，一些源数据变体是纯文本，迫使我们使用`str`作为通用类型。一些数据源将包含更有用的 Python 对象，允许使用除`str`之外的其他类型。

+   **分析**：这是用于进一步分析的数据。这些数据集可以使用原生 Python 对象。大部分时间，我们将关注那些容易序列化为 JSON 的对象。例外的是日期时间值，它们不能直接序列化为 JSON，但需要从标准的 ISO 文本格式进行一些额外的转换。

上面的类示例并不*替换*我们应用程序中的`model`模块。它们形成了一个更有用数据的第二个模型。建议的方法是将初始获取模型的模块名称从`model`更改为`acquisition_model`（或者可能是更短的`source_model`）。这个属性主要用字符串值描述模型。这个第二个模型是`analysis_model`。

对数据的初步调查结果可以为分析模型类定义提供更窄和更严格的约束。参见*第七章*，*数据检查功能*中的一些检查，这些检查有助于揭示属性值的预期最小值和最大值。

**Pydantic**库附带了许多自定义数据类型，可以用来描述数据值。请参阅[`docs.pydantic.dev/usage/types/`](https://docs.pydantic.dev/usage/types/)以获取文档。使用`pydantic`类型可能比将属性定义为字符串并尝试创建有效值的正则表达式要简单。

注意，源值验证不是**Pydantic**的核心。当提供 Python 对象时，**Pydantic**模块完全有可能执行成功的数据转换，而我们在希望抛出异常的地方可能没有。一个具体的例子是将 Python `float`对象提供给需要`int`值的字段。`float`对象将被转换；不会抛出异常。如果需要这种非常严格的 Python 对象验证，则需要一些额外的编程。

在下一节中，我们将创建我们模型的 JSON Schema 定义。我们可以从类定义中导出定义，或者我们可以手动构建 JSON。

### 8.2.2 使用 JSON Schema 表示法定义预期数据域

一旦我们有了类定义，我们就可以导出一个描述类的模式。请注意，**Pydantic**数据类是一个围绕底层`pydantic.BaseModel`子类定义的包装器。

我们可以通过在模块底部添加以下行来创建一个 JSON Schema 文档：

```py
from pydantic import schema_of
import json

if __name__ == "__main__":
    schema = schema_of(Series)
    print(json.dumps(schema, indent=2))
```

这些行将数据定义模块转换为一个脚本，该脚本将 JSON Schema 定义写入标准输出文件。

`schema_of()`函数将从上一节创建的数据类中提取一个模式。（参见*定义 Pydantic 类并生成 JSON Schema*。）底层的`pydantic.BaseModel`子类还有一个`schema()`方法，它将类定义转换为一个详细丰富的 JSON Schema 定义。当与**pydantic**数据类一起工作时，`pydantic.BaseModel`不可直接使用，必须使用`schema_of()`函数。

当执行终端命令`python src/analysis_model.py`时，将显示模式。

输出开始如下：

```py
{
  "title": "Series",
  "description": "A named series with a collection of values.",
  "type": "object",
  "properties": {
    "name": {
      "title": "Series name",
      "type": "string"
    },
    "samples": {
      "title": "Sequence of samples in this series",
      "type": "array",
      "items": {
        "\$ref": "#/definitions/SeriesSample"
      }
    }
  },
  "required": [
    "name",
    "samples"
  ],
  ...
}
```

我们可以看到标题与类名匹配。描述与文档字符串匹配。属性集合与类中的属性名称匹配。每个属性定义都提供了数据类中的类型信息。

`$ref`项是对 JSON Schema 中稍后提供的另一个定义的引用。这种引用的使用确保其他类定义是单独可见的，并且可用于支持此模式定义。

一个非常复杂的模型可能有多个定义，这些定义在多个地方共享。这种`$ref`技术使结构标准化，因此只提供一个定义。对单个定义的多次引用确保了类定义的正确重用。

JSON 结构乍一看可能看起来不寻常，但它并不令人畏惧地复杂。查看[`json-schema.org`](https://json-schema.org)将提供有关如何在不使用**Pydantic**模块的情况下最佳创建 JSON Schema 定义的信息。

### 8.2.3 使用 JSON Schema 验证中间文件

一旦我们有了 JSON Schema 定义，我们可以将其提供给其他利益相关者，以确保他们理解所需或提供的数据。我们还可以使用 JSON Schema 创建一个验证器，该验证器可以检查 JSON 文档，并确定该文档是否真的符合模式。

我们可以使用`pydantic`类定义来做这件事。有一个`parse_obj()`方法，它将检查字典以创建给定`pydantic`类的实例。`parse_raw()`方法可以将字符串或字节对象解析为给定类的实例。

我们也可以使用`jsonschema`模块来做这件事。我们将将其视为`pydantic`的替代方案，以展示共享 JSON Schema 如何允许其他应用程序与分析模型的正式定义一起工作。

首先，我们需要从模式中创建一个验证器。我们可以将 JSON 数据导出到一个文件中，然后从文件中重新加载 JSON 数据。我们还可以通过直接从由 **Pydantic** 创建的 JSON 模式创建验证器来省略一个步骤。以下是简短版本：

```py
from pydantic import schema_of
from jsonschema.validators import Draft202012Validator
from analysis_model import *

schema = schema_of(SeriesSample)
validator = Draft202012Validator(schema)
```

这将使用最新的 JSON 模式版本，即 2020 草案。 (该项目正在成为标准，并且随着其成熟已经通过了多个草案。)

这是我们可能编写一个函数来扫描文件以确保 NDJSON 文档都正确符合定义的模式的示例：

```py
def validate_ndjson_file(
        validator: Draft202012Validator,
        source_file: TextIO
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in source_file:
        document = json.loads(row)
        if not validator.is_valid(document):
            errors = list(validator.iter_errors(document))
            print(document, errors)
            counts[’faulty’] += 1
        else:
            counts[’good’] += 1
    return counts
```

此函数将读取给定源文件中的每个 NDJSON 文档。它将使用给定的验证器来检查文档是否存在问题或是否有效。对于有问题的文档，它将打印文档和整个验证错误列表。

这种类型的函数可以嵌入到单独的脚本中以检查文件。

类似地，我们可以为源模型创建模式，并使用 JSON 模式（或 **Pydantic**）在尝试处理源文件之前对其进行验证。

我们将转向更完整的验证和清理解决方案，见 *第九章*，*项目 3.1：数据清理基础应用*。该项目是更完整解决方案的基础组件之一。

我们将在下一节中查看这个项目的交付成果。

## 8.3 交付成果

本项目有以下交付成果：

+   一个 `requirements.txt` 文件，用于标识使用的工具，通常是 `pydantic==1.10.2` 和 `jsonschema==4.16.0`。

+   `docs` 文件夹中的文档。

+   包含源和分析模式的 JSON 格式文件。建议将这些文件放在单独的 `schema` 目录中。

+   模式的接受测试。

我们将详细查看模式接受测试。然后我们将查看如何使用模式扩展其他接受测试。

### 8.3.1 模式接受测试

要知道模式是否有用，必须要有接受测试用例。随着新的数据源集成到应用程序中，以及旧的数据源通过常规的错误修复和升级而发生变化，文件将发生变化。新文件通常会引发问题，问题的根本原因将是意外的文件格式变化。

一旦确定文件格式发生变化，最小的相关示例需要转换为接受测试。当然，测试将失败。现在，数据获取管道可以修复，因为有一个精确的完成定义。

首先，接受测试套件应该有一个有效的示例文件和一个无效的示例文件。

正如我们在 *第四章*，*数据获取功能：Web API 和抓取* 中所提到的，我们可以将一大块文本作为 Gherkin 场景的一部分提供。我们可以考虑以下场景：

```py
Scenario: Valid file is recognized.
    Given a file "example_1.ndjson" with the following content
        """
        {"x": 1.2, "y": 3.4}
        {"x": 5.6, "y": 7.8}
        """
    When the schema validation tool is run with the analysis schema
    Then the output shows 2 good records
    And the output shows 0 faulty records
```

这使我们能够提供 NDJSON 文件的内容。HTML 提取命令相当长。内容作为步骤定义函数的 `context.text` 参数提供。参见*验收测试*以获取更多如何编写步骤定义以创建用于此测试用例的临时文件的示例。

当然，故障记录的场景也是必不可少的。确保模式定义能够拒绝无效数据是很重要的。

### 8.3.2 扩展验收测试

在第三章、第四章和第五章中，我们编写了验收测试，通常查看应用程序活动的日志摘要，以确保它正确获取了源数据。我们没有编写专门查看数据的验收测试。

使用模式定义进行测试允许对文件中的每个字段和记录进行完整分析。这种检查的完整性具有极大的价值。

这意味着我们可以为现有场景添加一些额外的“然后”步骤。它们可能看起来像以下这样：

```py
    # Given (shown earlier)...
    # When (shown earlier)...
    Then the log has an INFO line with "header: [’x’, ’y’]"
    And log has INFO line with "Series_1 count: 11"
    And log has INFO line with "Series_2 count: 11"
    And log has INFO line with "Series_3 count: 11"
    And log has INFO line with "Series_4 count: 11"
    And the output directory files are valid
        using the "schema/Anscombe_Source.json" schema
```

额外的“然后输出目录文件有效...”行需要一个步骤定义，该定义必须执行以下操作：

1.  加载命名的 JSON 模式文件并构建一个 `Validator`。

1.  使用 `Validator` 对象检查 ND JSON 文件的每一行，以确保它们是有效的。

将模式作为验收测试套件的一部分使用，将并行于数据供应商和数据消费者如何使用模式来确保数据文件有效的方式。

需要注意的是，本章前面给出的模式定义（在*定义 Pydantic 类并生成 JSON 模式*中）是从未来项目的数据清理步骤中输出的。该示例中显示的模式不是之前数据获取应用的输出。

要验证数据获取的输出，您需要使用第三章、第四章和第五章中各种数据获取项目的模型。这将与本章前面显示的示例非常相似。虽然相似，但它将在本质上有所不同：它将使用 `str` 而不是 `float` 作为序列样本属性值。

## 8.4 概述

本章的项目展示了数据获取应用以下功能的一些示例：

+   使用 Pydantic 模块进行清晰、完整的定义

+   使用 JSON 模式创建一个可导出且语言无关的定义，任何人都可以使用

+   创建测试场景以使用正式的模式定义

通过正式化模式定义，可以记录有关数据处理应用程序以及应用于数据的转换的更多详细信息。

类定义的文档字符串成为模式中的描述。这允许记录有关数据来源和转换的详细信息，这些信息对所有数据用户都是可见的。

JSON Schema 标准允许记录值示例。**Pydantic**包有方法在字段定义和类配置对象中包含此元数据，这有助于解释奇怪或不寻常的数据编码。

此外，对于文本字段，JSONSchema 允许包含一个格式属性，该属性可以提供用于验证文本的正则表达式。**Pydantic**包对文本字段的这种额外验证提供了第一级支持。

我们将在*第九章*、*项目 3.1：数据清洗基础应用*和*第十章*、*数据清洗功能*的细节中返回数据验证的细节。在这些章节中，我们将更深入地探讨**Pydantic**的各种验证功能。

## 8.5 额外内容

这里有一些想法供您添加到这个项目中。

### 8.5.1 修订所有之前的章节模型以使用 Pydantic

前几章使用了`dataclasses`模块中的`dataclass`定义。这些可以转换为使用`pydantic.dataclasses`模块。这应该对之前的项目影响最小。

我们也可以将所有之前的验收测试套件转换为使用正式的源数据模式定义。

### 8.5.2 使用 ORM 层

对于 SQL 提取，ORM 非常有用。`pydantic`模块允许应用程序从中间 ORM 对象创建 Python 对象。这种双层处理似乎很复杂，但允许在**Pydantic**对象中进行详细的验证，这些对象不受数据库处理。

例如，一个数据库可能有一个没有提供任何范围的数值列。**Pydantic**类定义可以提供一个带有`ge`和`le`属性的字段定义来定义一个范围。此外，**Pydantic**允许定义具有独特验证规则的唯一数据类型，这些规则可以应用于数据库提取值。

首先，查看[`docs.sqlalchemy.org/en/20/orm/`](https://docs.sqlalchemy.org/en/20/orm/)以获取关于 SQLAlchemy ORM 层的详细信息。这提供了一个类定义，从中可以派生出 SQL 语句，如`CREATE TABLE`、`SELECT`和`INSERT`。

然后，查看**Pydantic**文档中的[`docs.pydantic.dev/usage/models/#orm-mode-aka-arbitrary-class-instances`](https://docs.pydantic.dev/usage/models/#orm-mode-aka-arbitrary-class-instances) “ORM 模式（也称为任意类实例）”部分，了解如何将更有用的类映射到中间 ORM 类。

对于一个古怪、设计不佳的数据库中的旧数据，这可能会成为一个问题。另一方面，对于从一开始就设计有 ORM 层的数据库，这可以简化 SQL。
