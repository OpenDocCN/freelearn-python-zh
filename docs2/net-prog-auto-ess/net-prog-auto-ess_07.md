# 7

# 错误处理和日志记录

在上一章中，我们描述了 Python 和 Go 的运行方式以及它们如何访问网络；然而，在构建我们的网络自动化解决方案时，我们遗漏了两个重要的点：如何报告程序执行事件以及如何处理错误。

这两个主题并不像它们看起来那么简单，而且它们大多数时候在系统中都实现得不好。一些网络开发者可能由于知识不足而没有正确地做到这一点，但也有一些开发者由于时间限制和编码所需额外时间而没有正确地做到这一点。

但这些活动真的重要吗？让我们在本章中探讨这些问题。首先，让我们研究我们如何以及为什么处理错误，然后研究我们为什么以及如何进行事件记录。

本章我们将涵盖以下主题：

+   编写错误处理代码

+   记录事件

+   在你的代码中添加日志记录

在阅读本章之后，你将能够有效地添加代码来处理错误并在你的网络开发中记录事件。

# 技术要求

本章中描述的源代码存储在 GitHub 仓库[`github.com/PacktPublishing/Network-Programming-and-Automation-Essentials/tree/main/Chapter07`](https://github.com/PacktPublishing/Network-Programming-and-Automation-Essentials/tree/main/Chapter07)中。

# 编写错误处理代码

要了解处理错误的重要性，我们必须将我们的系统作为一个整体来考虑，包括输入和输出。我们的代码本身可能永远不会遇到错误；然而，当与其他系统集成时，它可能会产生不可预测的输出，或者可能只是崩溃并停止工作。

因此，处理错误对于应对输入的不确定性以及保护你的代码以避免错误输出或崩溃至关重要。但我们如何做到这一点呢？

首先，我们需要确定我们的输入，然后我们创建一系列不同的值组合，这些值被发送到我们的输入。然后通过运行我们的代码来评估这些输入组合的行为。对于一个函数，我们通过添加单元测试来实现，正如我们在*第五章*中讨论的那样。对于系统，我们添加集成测试和端到端测试。在*第五章*中也讨论了其他一些技术。

但编写处理错误的代码的正确方法是什么？这取决于语言。

在 Go 中编写处理错误的代码与 Python 相比有很大不同。让我们看看我们如何在 Go 中有效地做到这一点，然后再在 Python 中做到这一点。

## 在 Go 中添加错误处理

Go 语言的设计要求在发生错误时显式检查错误，这与 Python 中抛出异常然后捕获它们的方式不同。在 Go 中，错误只是函数返回的值，这使得 Go 编码更加冗长，也许更加重复。在 Python 中，你不需要检查错误，因为会抛出异常，但在 Go 中，你必须检查错误。但另一方面，与 Python 相比，Go 的错误处理要简单得多。

Go 中的错误是通过使用`error`类型接口创建的，如下所示：

```py
type error interface {
    Error() string
}
```

如前述代码所示，Go 中的错误实现通过使用名为`Error()`的方法来返回错误消息作为字符串，相当简单。

在你的代码中构建错误的正确方式是使用`errors`或`fmt`标准库。

下面是两个使用每个函数进行除以零函数的示例。

使用`errors`库的示例如下：

```py
func divide(q int, d int) (int, error) {
    if d == 0 {
        return 0, errors.New("division by zero not valid")
    }
    return q / d, nil
}
```

使用`fmt`库的示例如下：

```py
func divide(q int, d int) (int, error) {
    if d == 0 {
        return 0, fmt.Errorf("divided by zero not valid")
    }
    return q / d, nil
}
```

前两个示例产生相同的结果。例如，如果你调用这两个函数中的任何一个并使用`fmt.Println(divide(10, 0))`，它将打印以下输出：`0 divided by zero` `not valid`。

`fmt.Errorf`和`errors.New`之间的主要区别在于格式化字符串和添加值的可能性。另一个点是`errors.New`更快，因为它没有调用格式化器。

如果你想要创建自定义错误、堆栈跟踪和更高级的错误功能，请考虑使用`errors`库或第三方库，如流行的`pkg/errors` ([`pkg.go.dev/github.com/pkg/errors`](https://pkg.go.dev/github.com/pkg/errors)) 或 `golang.org/x/xerrors` ([`pkg.go.dev/golang.org/x/xerrors`](https://pkg.go.dev/golang.org/x/xerrors))。

让我们现在关注编写 Go 中错误处理代码的最佳实践。以下是一些最佳实践。

### 最后返回错误，并将值设为 0

当创建一个返回多个值的函数时，错误应该放在返回值的最后一个参数。当返回带有错误的值时，如果是数字则使用 0，如果是字符串则使用`empty string`，如下例所示：

```py
func findNameCount(text string) (string, int, error) {
    if len(text) < 5 {
        return "", 0, fmt.Errorf("text too small")
    }
    . . .
}
```

在前面的例子中，返回的字符串值为空，返回的`int`值为 0。这些值只是建议，因为当返回错误时，调用语句会首先检查是否存在错误，然后再分配返回的变量。因此，带有错误的返回值是不相关的。

### 只添加调用者没有的信息

在创建错误消息时，不要添加调用者已经知道的信息。以下示例说明了这个问题：

```py
func divide(q int, d int) (int, error) {
    if d == 0 {
        return 0, fmt.Errorf("%d can't be divided by zero", q)
    }
    return q / d, nil
}
```

如前述示例所示，`q`的值被返回在错误消息中。但这不是必要的，因为`divide`函数的调用者已经有了这个值。

在您的函数中创建要返回的错误时，不要包含任何传递给函数的参数，因为这是调用者所知道的。这可能导致信息重复。

### 使用小写且不要以标点符号结尾

总是使用小写，因为错误信息将在返回时与其他消息连接。大多数时候，您也不应使用任何标点符号，因为错误信息可能会链接在一起，标点符号最终会在错误信息中间看起来很奇怪。

小写规则的一个例外是当您引用已经具有大写字母的函数或方法名称时。

### 在错误信息中添加冒号

冒号（`:`）用于您想要添加来自代码内部调用产生的另一个错误信息的任何信息时。以下代码将作为示例：

```py
func connect(host string, conf ssh.ClientConfig) error {
    conn, err := ssh.Dial("tcp", host+":22", conf)
    if err != nil {
        return fmt.Errorf("ssh.Dial: %v", err)
    }
    . . .
```

在前例中，`connect` 函数封装了对 `ssh.Dial` 的调用。我们可以通过添加调用名称或有关 `ssh.Dial` 的信息来将错误上下文添加到错误信息中，如果需要，使用冒号分隔。请注意，`config` 和 `host` 参数由 `connect` 函数的调用者知道，因此不应添加到错误信息中。

### 使用 defer、panic 和 recover

Go 有重要的机制来控制程序在错误发生时的流程。这主要用于使用 goroutines，因为一个错误可能会导致程序崩溃，您可能需要格外小心以避免未关闭的软件管道和软件缓存；以及避免未释放的内存和未关闭的文件描述符。

#### defer

Go 的 `defer` 用于将执行推送到一个列表，该列表仅在周围函数返回或崩溃后执行。`defer` 的主要目的是执行清理。考虑以下示例，该示例将数据从文件复制到另一个文件，然后删除它：

```py
func moveFile(srcFile, dstFile string) error {
    src, err := os.Open(srcFile)
    if err != nil {
        return fmt.Errorf("os.Open: %v", err)
    }
    dst, err := os.Create(dstFile)
    if err != nil {
        return fmt.Errorf("os.Create: %v", err)
    }
    _, err = io.Copy(dst, src)
    if err != nil {
        return fmt.Errorf("io.Copy: %v", err)
    }
    dst.Close()
    src.Close()
    err = os.Remove(srcFile)
    if err != nil {
        return fmt.Errorf("os.Remove: %v", err)
    }
    return nil
}
```

在前例中，如果 `os.Create` 发生错误，函数会在调用 `src.Close()` 之前返回，这意味着文件没有被正确关闭。

避免在代码中重复添加 `close` 语句的方法是使用 `defer`，如下例所示：

```py
func moveFile(srcFile, dstFile string) error {
    src, err := os.Open(srcFile)
    if err != nil {
        return fmt.Errorf("os.Open: %v", err)
    }
    defer src.Close()
    dst, err := os.Create(dstFile)
    if err != nil {
        return fmt.Errorf("os.Create: %v", err)
    }
    defer dst.Close()
    _, err = io.Copy(dst, src)
    if err != nil {
        return fmt.Errorf("io.Copy: %v", err)
    }
    err = os.Remove(srcFile)
    if err != nil {
        return fmt.Errorf("os.Remove: %v", err)
    }
    return nil
}
```

如前例所示，`defer` 在成功的 `os.Open` 和成功的 `os.Create` 之后使用。因此，如果发生错误或函数结束，它将首先调用 `dst.Close()`，然后以相反的顺序调用 `src.Close()`，就像一个 **后进先出**（**LIFO**）队列一样。

现在我们来看看如何使用 `panic`。

#### panic

在编写代码时，如果您不想处理错误，可以使用 `panic` 来立即停止。在 Go 中，可以通过显式编写来调用 `panic`，但在运行时如果发生错误，它也会自动调用。以下是可以发生的重大运行时错误列表：

+   越界内存访问，包括数组

+   错误类型断言

+   尝试使用`nil`指针调用函数

+   向已关闭的通道或文件描述符发送数据

+   零除

因此，`panic`仅在你不打算处理错误或处理尚未理解的错误时才用于你的代码中。

重要的是要注意，在退出函数并传递`panic`消息之前，程序仍然会运行在函数中之前堆叠的所有`defer`语句。

这里是一个使用`panic`在接收到负值作为参数后退出程序示例：

```py
import (
    "fmt"
    "math"
)
func squareRoot(value float64) float64 {
    if value < 0 {
        panic("negative values are not allowed")
    }
    return math.Sqrt(value)
}
func main() {
    fmt.Println(squareRoot(-2))
    fmt.Println("done")
}
```

让我们运行这个程序并检查输出：

```py
$ go run panic-example.go
panic: negative values are not allowed
goroutine 1 [running]:
main.squareRoot(...)
    Dev/Chapter07/Go/panic-example.go:10
main.main()
    Dev/Chapter07/Go/panic-example.go:17 +0x45
exit status 2
```

注意，输出没有打印`done`，因为`panic`是在`squareRoot`函数中调用的，在打印指令之前。

假设我们按照以下方式将`defer`添加到函数中：

```py
func squareRoot(value float64) float64 {
    defer fmt.Println("ending the function")
    if value < 0 {
        panic("negative values are not allowed")
    }
    return math.Sqrt(value)
}
```

输出将如下所示：

```py
$ go run panic-example.go
ending the function
panic: negative values are not allowed
goroutine 1 [running]:
main.squareRoot(…)
    Dev/Chapter07/Go/panic-example.go:10
main.main()
    Dev/Chapter07/Go/panic-example.go:17 +0x45
exit status 2
```

注意，`ending the function`打印语句是在发送`panic`消息之前放置的。这是因为，正如我们解释的，`defer`栈在`panic`返回函数之前执行。

现在让我们看看我们如何使用`recover`。

#### recover

在 Go 中，`recover`是处理错误所需的最后一块错误流控制。它用于处理`panic`情况并恢复控制。它应该只在使用`defer`函数调用时使用。在正常调用中，`recover`将返回一个`nil`值，但在`panic`情况下，它将返回`panic`给出的值。

例如，让我们考虑以下程序：

```py
import "fmt"
func divide(q, d int) int {
    fmt.Println("Dividing it now")
    return q / d
}
func main() {
    fmt.Println("the division is:", divide(4, 0))
}
```

如果你运行前面的程序，你会得到以下`panic`消息：

```py
$ go run division-by-zero-panic.go
Dividing it now
panic: runtime error: integer divide by zero
goroutine 1 [running]:
main.divide(...)
    Dev/Chapter07/Go/division-by-zero-panic.go:7
main.main()
    Dev/Chapter07/Go/division-by-zero-panic.go:11 +0x85
exit status 2
```

如从这个输出中可以看出，你对`panic`情况没有控制权。它基本上会崩溃程序，没有机会正确处理错误。这在大多数生产软件中是不理想的，尤其是在使用多个 goroutine 时。

因此，为了正确处理`panic`情况，你应该添加一个`defer`函数来测试是否是`panic`情况，使用`recover`，如下例所示：

```py
import "fmt"
func divide(q, d int) int {
    fmt.Println("Dividing it now")
    return q / d
}
func main() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Got a panic:", r)
        }
    }()
    fmt.Println("the division is:", divide(4, 0))
}
```

在添加了与前面示例相同的`defer`函数之后，输出将如下所示：

```py
$ go run division-by-zero-panic-recover.go
Dividing it now
Got a panic: runtime error: integer divide by zero
```

如你所见，在`defer`函数内添加一个`recover`测试将允许你处理意外的`panic`情况，避免程序意外崩溃，没有进行适当的清理或修复错误。

现在我们已经研究了如何处理 Go 错误，让我们看看 Python 的错误处理。

## 在 Python 中添加错误处理

Python 处理错误的方式与 Go 不同。Python 不需要你的函数返回错误值。在 Python 中，错误在运行时抛出，它们被称为异常。为了处理异常，你的代码必须正确捕获它们并避免引发它们。

### 捕获异常

在 Python 中，有许多运行时错误会引发内置异常。内置异常的列表相当长，可以在以下位置找到：[`docs.python.org/3/library/exceptions.html`](https://docs.python.org/3/library/exceptions.html)。例如，除以零错误被称为`ZeroDivisionError`异常。

为了处理错误，您需要捕获异常，然后使用`try`、`except`、`else`和`finally` Python 语句来处理它。为了创建一个处理除以零异常的示例，让我们首先运行以下程序而不捕获异常：

```py
def division(q, d):
    return q/d
print(division(1, 0))
```

如果您运行前面的程序，它将生成以下输出：

```py
$ python catching-division-by-zero-exception.py
Traceback (most recent call last):
  File "Chapter07/Python/catching-division-by-zero-exception.py", line 7, in <module>
    print(division(1, 0))
  File "Chapter07/Python/catching-division-by-zero-exception.py", line 4, in division
    return q/d
ZeroDivisionError: division by zero
```

如您所见，程序崩溃并在屏幕上显示错误消息作为`Traceback`，其中包含错误发生的位置和异常名称的详细信息，在这种情况下是`ZeroDivisionError`。

现在，让我们更新 Python 代码以捕获此异常并更优雅地处理错误，如下所示：

```py
def division(q, d):
    return q/d
try:
    print(division(1, 0))
except ZeroDivisionError:
    print("Error: We should not divide by zero")
```

现在，如果您运行程序，它将优雅地打印错误而不会崩溃，如下所示：

```py
$ python catching-division-by-zero-exception.py
Error: We should not divide by zero
```

因此，每当您认为函数可能会通过错误引发异常时，请使用`try`和`except`语句，正如前一个示例所示。

除了`try`和`except`语句之外，Python 还允许使用`else`和`finally`语句来添加更多的错误处理流程控制。它们不是强制的，因为流程可以在`try`/`except`语句之外控制，但有时它们很有用。以下是在相同示例中添加`else`和`finally`语句的代码：

```py
def division(q, d):
    return q/d
try:
    result = division(10, 1)
except ZeroDivisionError:
    print("Error: We should not divide by zero")
else:
    print("Division succeded, result is:", result)
finally:
    print("done")
```

如果您运行此程序，它将生成以下输出：

```py
$ python catch-else-finally-division-by-zero.py
Division succeded, result is: 10.0
done
```

注意，只有当`try`子句中没有引发异常时，`else`语句才会执行。`finally`语句总是执行，无论`try`子句中是否引发了异常。

现在我们已经看到了如何在 Python 中捕获异常，让我们讨论如何选择我们想要捕获的异常。

### 选择更具体的异常

在 Python 中，异常是有层次的，并且始终以名为`BaseException`的异常开始。例如，除以零展示了以下层次结构：

```py
BaseException -> Exception -> ArithmeticError-> ZeroDivisionError
```

异常层次结构非常有用，因为您的代码可以捕获更高层次的异常或更具体的异常。对于除以零的情况，您可以捕获`ArithmeticError`异常而不是`ZeroDivisionError`。然而，有时捕获更具体的异常而不是更高层次的异常是一种好的实践。

更具体的异常更希望在函数和库内部捕获，因为如果您在函数内部捕获通用异常，那么当代码的另一个部分调用您的函数时，可能会掩盖问题。因此，这取决于您在哪里捕获以及如何处理它。

我们现在对如何在 Go 和 Python 中处理错误有了很好的了解。让我们讨论如何将日志记录添加到我们的代码中。

# 记录事件

在计算机软件中，日志记录是一种众所周知的技巧，用于帮助调试问题、记录里程碑、理解行为、检索信息和检查历史事件，以及其他有用的操作。尽管有这些优势，但许多开发人员并没有在他们的代码中添加适当的日志记录。事实上，一些开发人员什么都不做，只有在程序出现问题时才添加日志记录以进行调试。

在网络自动化中，日志记录甚至更为重要，因为网络元素通常是分布式的，并且严重依赖日志记录以便在出现问题时或需要改进时进行审计。将日志添加到您的代码中是一种良好的实践，将受到多个工程级别的赞赏，例如网络操作员、网络规划师、网络安全和网络设计师等。

但这里有一个重要观点必须注意，那就是网络元素之间的时间同步是强制性的，以便使日志变得有用。必须在整个网络中使用诸如**网络时间协议**（**NTP**）或**精确时间协议**（**PTP**）之类的协议。

使用日志的一个良好实践是使用名为`syslog`的 Unix 日志参考，它最初作为 RFC3164 的信息性 RFC 发布，后来作为 RFC5424 的标准文档。([`www.rfc-editor.org/rfc/rfc3164`](https://www.rfc-editor.org/rfc/rfc3164))

对于我们的网络自动化代码，我们不需要遵循`syslog`协议标准的所有细节，但我们将根据严重程度级别将其用作记录有用信息的指南。

让我们谈谈在记录事件时我们希望记录的一些信息级别。

## 严重程度级别

RFC5424 的`syslog`协议定义了八个严重程度级别，这些级别在 RFC5424 的*第 6.2.1 节*中描述。以下列表中提到了这些级别，并简要说明了打算添加到每个级别的信息消息类型：

+   `紧急`: 系统无法运行，且无法恢复。

+   `警报`: 需要立即关注。

+   `关键`: 发生了不好的事情，需要快速关注以修复它。

+   `错误`: 正在发生故障，但不需要紧急关注。

+   `警告或 Warn`: 表明有问题，可能会在未来导致错误，例如软件未更新。

+   `通知`: 已达到一个重要里程碑，可能表明未来的警告，例如配置未保存或资源利用率限制未设置。

+   `信息性或 Info`: 正常操作里程碑消息。用于以后的审计和调查。

+   `调试`: 由开发人员用于调试问题或调查可能的改进。

虽然这八个级别在 `syslog` 协议中定义，但它们相当模糊，容易产生不同的解释。例如，`Alert` 和 `Emergency` 对于不同的开发者在编写代码时可能会有所不同，其他级别如 `Notice` 和 `Informational` 也是如此。因此，一些网络开发者更喜欢使用较少的级别，这些级别更容易理解。具体数量将取决于网络的运行方式，但通常在三个到五个级别之间。对于 Go 和 Python，级别的数量将取决于你用来创建日志消息的库。有些库可能比其他库有更多的级别。

现在，让我们研究如何使用 Go 和 Python 将日志事件添加到你的代码中。

# 在你的代码中添加日志

在你的代码中添加事件日志在 Go 和 Python 中会有所不同，并且会根据你代码中使用的库而变化。但两种语言的想法都是将信息划分为严重性级别，就像在 `syslog` 中做的那样。

严重性日志级别也会根据所使用的库而有所不同。Python 和 Go 都有标准的日志库，但你也可以使用第三方库来在两种语言中记录事件。

这里的一个重要点是，在编写代码时，你将决定是否需要添加一个日志事件行。添加的日志行必须携带一些信息，这将向程序发出信号，表明消息的严重性级别。因此，像失败这样重要的消息将比像调试这样不太重要的消息有更高的优先级。理想情况下，关于应该公开哪个日志级别的决定通常是通过向程序添加允许设置日志级别的输入参数来做出的。所以，如果你在调试程序运行时，它将生成比正常操作多得多的信息。

现在我们来看看如何在 Go 代码中添加日志事件，然后我们检查如何在 Python 中实现。

## 在 Go 中添加事件日志

Go 语言有一个标准的日志库，它随 Go 安装一起提供，但它相当有限。如果你想在 Go 中实现更高级的日志功能，你可能需要使用第三方日志库。

让我们看看我们如何使用标准库，然后检查其他流行的第三方库。

### 使用标准的 Go 日志

Go 的标准日志库可以通过在 `import` 语句中使用 `log` 来导入。默认情况下，Go 标准日志不提供任何严重性级别，但它有一些辅助函数可以帮助创建日志。辅助函数在此列出：

+   `Print`、`Printf` 和 `Println`：这些函数使用 `stderr` 在终端打印传递给它们的消息

+   `Panic`、`Panicf` 和 `Panicln`：这些像 `Print` 一样工作，但在打印日志消息后会调用 `Panic`

+   `Fatal`、`Fatalf` 和 `Fatalln`：这些也像 `Print` 一样工作，但在打印日志消息后会调用 `os.Exit(1)`

以下是一个使用标准 Go 日志库的简单示例：

```py
import (
    "log"
    "os/user"
)
func main() {
    user, err := user.Current()
    if err != nil {
        log.Fatalf("Failed with error: %v", err)
    }
    log.Printf("Current user is %s", user.Username)
}
```

运行此程序将无错误地打印以下输出：

```py
% go run standard-logging.go
2022/11/08 18:53:24 Current user is claus
```

如果由于任何原因无法检索当前用户，它将调用 `Fatalf`，在打印失败信息后，将调用 `os.Exit(1)`。

现在，让我们展示一个更复杂的示例，说明如何使用标准日志库创建严重性级别并将其保存到文件中：

```py
import (
    "log"
    "os"
)
var criticalLog, errorLog, warnLog, infoLog, debugLog *log.Logger
func init() {
    file, err := os.Create("log-file.txt")
    if err != nil {
        log.Fatal(err)
    }
    flags := log.Ldate | log.Ltime
    criticalLog = log.New(file, "CRITICAL: ", flags)
    errorLog = log.New(file, "ERROR: ", flags)
    warnLog = log.New(file, "WARNING: ", flags)
    infoLog = log.New(file, "INFO: ", flags)
    debugLog = log.New(file, "DEBUG: ", flags)
}
func main() {
    infoLog.Print("That is a milestone")
    errorLog.Print("Got an error here")
    debugLog.Print("Extra information for a debug")
    warnLog.Print("You should be warned about this")
}
```

在前面的示例中，我们创建了五个严重性级别，可以根据需要将它们写入文件。请注意，在 Go 中，`init()` 函数在 `main()` 函数之前执行。如果您想在其他包中使用这些日志定义，请记住使用变量的大写；否则，变量将仅限于此包；例如，`errorLog` 应该是 `ErrorLog`。

此外，如果您想设置日志级别以避免 `Debug` 或 `Info` 消息，您必须向程序传递一个参数，并根据设置的级别抑制较低级别的严重性。使用 Go 标准日志库，您必须自己这样做。

现在，让我们调查一个在 Go 开发者中非常受欢迎的第三方日志库。

### 使用 logrus

也许在 Go 中最受欢迎的日志库之一是 `logrus`，它是一个具有多个日志功能的结构化日志库。`logrus` 有七个日志级别，并且与标准日志库兼容。默认情况下，该库允许您设置日志级别，因此如果您不想看到调试信息，它不会产生噪音。

这里是一个使用 `logrus` 并将日志级别设置为 `Error` 的简单示例，这意味着不会显示低级别的日志，例如 `Warning`、`Info` 或 `Debug`：

```py
import (
    log "github.com/sirupsen/logrus"
)
func init() {
    log.SetFormatter(&log.TextFormatter{
        DisableColors: true,
        FullTimestamp: true,
    })
    log.SetLevel(log.ErrorLevel)
}
func main() {
    log.Debug("Debug is suppressed in error level")
    log.Info("This info won't show in error level")
    log.Error("Got an error here")
}
```

运行前面的示例将在终端中仅显示以下输出：

```py
% go run logrus-logging.go
time="2022-11-09T11:16:48-03:00" level=error msg="Got an error here"
```

由于严重性级别设置为 `ErrorLevel`，不会显示任何不太重要的日志消息——在示例中，对 `log.Info` 和 `log.Debug` 的调用。

`logrus` 非常灵活且功能强大，互联网上有许多使用示例。有关 `logrus` 的更多详细信息，请参阅 [`github.com/sirupsen/logrus`](https://github.com/sirupsen/logrus)。

如果您想在 Go 中使用更多日志库，这里有一个第三方日志库的编译列表：[`awesome-go.com/logging/`](https://awesome-go.com/logging/)。

现在，让我们检查如何使用 Python 将日志添加到我们的代码中。

## 在 Python 中添加事件日志

与 Go 相比，Python 为标准日志库添加了许多更多功能。尽管支持更好，但 Python 社区也开发了多个第三方日志库。

让我们看看 Python 的标准库和流行的第三方库。

### 使用标准日志库进行 Python

标准日志库包含五个严重级别和一个额外的级别，用于指示级别未由记录器设置。每个级别都与一个数字相关联，可以用来解释优先级级别，其中数字越小，优先级越低。级别包括 `CRITICAL`（50）、`ERROR`（40）、`WARNING`（30）、`INFO`（20）、`DEBUG`（10）和 `NOTSET`（0）。

`NOTSET` 级别在使用日志层次结构时很有用，允许非根记录器将级别委派给其父记录器。

以下是一个使用 Python 标准日志的示例：

```py
import logging
    logging.basicConfig(
        filename='file-log.txt',
        level=logging.ERROR,
        format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
)
logging.debug("This won't show, level is set to info")
logging.info("Info is not that important as well")
logging.warning("Warning will not show as well")
logging.error("This is an error")
```

运行前面的程序将在名为 `file-log.txt` 的输出文件中产生以下行：

```py
2022-11-09 14:48:52.920 ERROR: This is an error
```

如前述代码所示，将级别设置为 `logging.ERROR` 将不允许在文件中写入低级别的日志消息。程序只是忽略了 `logging.debug()`、`logging.info()` 和 `logging.warning()` 调用。

另一个重要点是展示在 Python 中使用标准日志的简便性。前述示例表明，您只需调用一次 `logging.basicConfig` 就可以设置几乎您需要的所有内容，从格式化程序到严重级别。

除了易于使用之外，Python 社区还为标准日志库创建了优秀的教程和文档。以下是文档和高级使用信息的三个主要参考：

+   [`docs.python.org/3/library/logging`](https://docs.python.org/3/library/logging)

+   [`docs.python.org/3/howto/logging.html`](https://docs.python.org/3/howto/logging.html)

+   [`docs.python.org/3/howto/logging-cookbook.html`](https://docs.python.org/3/howto/logging-cookbook.html)

从本质上讲，Python 标准日志库非常完整，您在大多数工作中不需要使用第三方库。然而，有一个名为 `loguru` 的流行第三方库提供了许多有趣和有用的功能。让我们看看如何使用它。

### 使用 Python loguru

Python `loguru` 提供了比标准 Python 日志库更多的功能，并旨在使其更容易使用和配置。例如，使用 `loguru`，您将能够设置日志文件的文件轮换，使用更高级的字符串格式化程序，并使用装饰器在函数上捕获异常，并且它是线程和进程安全的。

它还具有一些有趣的功能，允许您通过使用 `patch` 方法（更多关于 `patch` 方法的信息请参阅 [`loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.patch`](https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.patch)）来添加额外的日志信息。

以下是一个使用 `loguru` 的简单示例：

```py
from loguru import logger
logger.add(
    "file-log-{time}.txt",
    rotation="1 MB",
    colorize=False,
    level="ERROR",
)
logger.debug("That's not going to show")
logger.warning("This will not show")
logger.error("Got an error")
```

运行前面的示例将创建一个包含日期和时间的文件，其中包含日志消息，如果文件大小达到 1 MB，则将进行轮换。文件中的输出将如下所示：

```py
% cat file-log-2022-11-09_15-53-58_056790.txt
2022-11-09 15:53:58.063 | ERROR    | __main__:<module>:13 - Got an error
```

更详细的文档可以在[`loguru.readthedocs.io`](https://loguru.readthedocs.io)找到，源代码在[`github.com/Delgan/loguru`](https://github.com/Delgan/loguru)。

# 摘要

在阅读本章之后，你可能更加清楚为什么我们需要处理错误以及为什么我们需要创建适当的事件记录。你也应该更加熟悉 Go 和 Python 在处理错误方面的差异。此外，你还看到了在使用标准库和第三方库进行事件记录方面的差异。从现在开始，你的网络自动化代码设计将包含一个专门关于日志记录和错误处理的章节。

在下一章中，我们将讨论如何扩展我们的代码以及我们的网络自动化解决方案如何与大型网络交互。
