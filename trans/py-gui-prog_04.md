# 通过验证和自动化减少用户错误

我们的表单有效，主管和数据输入人员都对表单设计感到满意，但我们还没有准备好投入生产！我们的表单还没有履行承诺的任务，即防止或阻止用户错误。数字框仍然允许字母，组合框不限于给定的选择，日期必须手动填写。在本章中，我们将涵盖以下主题：

+   决定验证用户输入的最佳方法

+   学习如何使用Tkinter的验证系统

+   为我们的表单创建自定义小部件，验证输入的数据

+   在我们的表单中适当的情况下自动化默认值

让我们开始吧！

# 验证用户输入

乍一看，Tkinter的输入小部件选择似乎有点令人失望。它没有给我们一个真正的数字输入，只允许数字，也没有一个真正的下拉选择器，只允许从下拉列表中选择项目。我们没有日期输入、电子邮件输入或其他特殊格式的输入小部件。

但这些弱点可以成为优势。因为这些小部件什么都不假设，我们可以使它们以适合我们特定需求的方式行为，而不是以可能或可能不会最佳地工作的通用方式。例如，字母在数字输入中可能看起来不合适，但它们呢？在Python中，诸如`NaN`和`Infinity`之类的字符串是有效的浮点值；拥有一个既可以增加数字又可以处理这些字符串值的框在某些应用中可能非常有用。

我们将学习如何根据需要调整我们的小部件，但在学习如何控制这种行为之前，让我们考虑一下我们想要做什么。

# 防止数据错误的策略

对于小部件如何响应用户尝试输入错误数据，没有通用答案。各种图形工具包中的验证逻辑可能大不相同；当输入错误数据时，输入小部件可能会验证用户输入如下：

+   防止无效的按键注册

+   接受输入，但在提交表单时返回错误或错误列表

+   当用户离开输入字段时显示错误，可能会禁用表单提交，直到它被纠正

+   将用户锁定在输入字段中，直到输入有效数据

+   使用最佳猜测算法悄悄地纠正错误的数据

数据输入表单中的正确行为（每天由甚至可能根本不看它的用户填写数百次）可能与仪器控制面板（值绝对必须正确以避免灾难）或在线用户注册表单（用户以前从未见过的情况下填写一次）不同。我们需要向自己和用户询问哪种行为将最大程度地减少错误。

与数据输入人员讨论后，您得出以下一组指南：

+   尽可能忽略无意义的按键（例如数字字段中的字母）

+   空字段应该注册一个错误（所有字段都是必填的），但`Notes`除外

+   包含错误数据的字段应以某种可见的方式标记，并描述问题

+   如果存在错误字段，则应禁用表单提交

让我们在继续之前，将以下要求添加到我们的规范中。在“必要功能”部分，更新硬性要求如下：

```py
The program must:
...
* have inputs that:
  - ignore meaningless keystrokes
  - require a value for all fields, except Notes
  - get marked with an error if the value is invalid on focusout
* prevent saving the record when errors are present

```

那么，我们如何实现这一点呢？

# Tkinter中的验证

Tkinter的验证系统是工具包中不太直观的部分之一。它依赖于以下三个配置选项，我们可以将其传递到任何输入小部件中：

+   `validate`：此选项确定哪种类型的事件将触发验证回调

+   `validatecommand`：此选项接受将确定数据是否有效的命令

+   `invalidcommand`：此选项接受一个命令，如果`validatecommand`返回`False`，则运行该命令

这似乎很简单，但有一些意想不到的曲线。

我们可以传递给`validate`的值如下：

| **验证字符串** | **触发时** |
| --- | --- |
| `none` | 它是关闭验证的无 |
| `focusin` | 用户输入或选择小部件 |
| `unfocus` | 用户离开小部件 |
| `focus` | `focusin`或`focusout` |
| `key` | 用户在小部件中输入文本 |
| `all` | `focusin`，`focusout`和`key` |

`validatecommand`参数是事情变得棘手的地方。您可能会认为这需要Python函数或方法的名称，但事实并非如此。相反，我们需要给它一个包含对Tcl/`Tk`函数的引用的元组，并且可以选择一些**替换代码**，这些代码指定我们要传递到函数中的触发事件的信息。

我们如何获得对Tcl/`Tk`函数的引用？幸运的是，这并不太难；我们只需将Python可调用对象传递给任何Tkinter小部件的`.register（）`方法。这将返回一个字符串，我们可以在`validatecommand`中使用。

当然，除非我们传入要验证的数据，否则验证函数没有什么用。为此，我们向我们的`validatecommand`元组添加一个或多个替换代码。

这些代码如下：

| **代码** | **传递的值** |
| --- | --- |
| “％d” | 指示正在尝试的操作的代码：`0`表示`delete`，`1`表示插入，`-1`表示其他事件。请注意，这是作为字符串而不是整数传递的。 |
| “％P” | 更改后字段将具有的建议值（仅限键事件）。 |
| “％s” | 字段中当前的值（仅限键事件）。 |
| “％i” | 在键事件上插入或删除的文本的索引（从`0`开始），或在非键事件上为`-1`。请注意，这是作为字符串而不是整数传递的。 |
| “％S” | 对于插入或删除，正在插入或删除的文本（仅限键事件）。 |
| “％v” | 小部件的“验证”值。 |
| “％V” | 触发验证的事件：`focusin`，`focusout`，`key`或`forced`（表示文本变量已更改）。 |
| “％W” | Tcl/`Tk`中小部件的名称，作为字符串。 |

`invalidcommand`选项的工作方式完全相同，需要使用`.register（）`方法和替换代码。

要查看这些内容是什么样子，请考虑以下代码，用于仅接受五个字符的`Entry`小部件：

```py
def has_five_or_less_chars(string):
    return len(string) <= 5

wrapped_function = root.register(has_five_or_less_chars)
vcmd = (wrapped_function, '%P')
five_char_input = ttk.Entry(root, validate='key', validatecommand=vcmd)
```

在这里，我们创建了一个简单的函数，它只返回字符串的长度是否小于或等于五个字符。然后，我们使用“register（）”方法将此函数注册到`Tk`，将其引用字符串保存为`wrapped_function`。接下来，我们使用引用字符串和“'％P'”替换代码构建我们的`validatecommand`元组，该替换代码表示建议的值（如果接受键事件，则输入将具有的值）。

您可以传入任意数量的替换代码，并且可以按任何顺序，只要您的函数是写入接受这些参数的。最后，我们将创建我们的`Entry`小部件，将验证类型设置为`key`，并传入我们的验证命令元组。

请注意，在这种情况下，我们没有定义`invalidcommand`方法；当通过按键触发验证时，从`validate`命令返回`False`将导致忽略按键。当通过焦点或其他事件类型触发验证时，情况并非如此；在这种情况下，没有定义默认行为，需要`invalidcommand`方法。

考虑以下`FiveCharEntry`的替代基于类的版本，它允许您输入任意数量的文本，但在离开字段时会截断您的文本：

```py
class FiveCharEntry2(ttk.Entry):
    """An Entry that truncates to five characters on exit."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config(
            validate='focusout',
            validatecommand=(self.register(self._validate), '%P'),
            invalidcommand=(self.register(self._on_invalid),)
        )

    def _validate(self, proposed_value):
        return len(proposed_value) <= 5

    def _on_invalid(self):
        self.delete(5, tk.END)
```

这一次，我们通过对`Entry`进行子类化并在方法中定义我们的验证逻辑来实现验证，而不是在外部函数中。这简化了我们在验证方法中访问小部件。

`_validate()`和`_on_invalid()`开头的下划线表示这些是内部方法，只能在类内部访问。虽然这并不是必要的，而且Python并不会将其与普通方法区别对待，但它让其他程序员知道这些方法是供内部使用的，不应该在类外部调用。

我们还将`validate`参数更改为`focusout`，并添加了一个`_on_invalid()`方法，该方法将使用`Entry`小部件的`delete()`方法截断值。每当小部件失去焦点时，将调用`_validate()`方法并传入输入的文本。如果失败，将调用`_on_invalid()`，导致内容被截断。

# 创建一个DateEntry小部件

让我们尝试创建一个验证版本的`Date`字段。我们将创建一个`DateEntry`小部件，它可以阻止大多数错误的按键，并在`focusout`时检查日期的有效性。如果日期无效，我们将以某种方式标记该字段并显示错误。让我们执行以下步骤来完成相同的操作：

1.  打开一个名为`DateEntry.py`的新文件，并从以下代码开始：

```py
from datetime import datetime

class DateEntry(ttk.Entry):
    """An Entry for ISO-style dates (Year-month-day)"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config(
            validate='all',
            validatecommand=(
                self.register(self._validate),
                '%S', '%i', '%V', '%d'
            ),
        invalidcommand=(self.register(self._on_invalid), '%V')
    )
    self.error = tk.StringVar()
```

1.  由于我们需要在验证方法中使用`datetime`，所以我们在这里导入它。

1.  我们对`ttk.Entry`进行子类化，然后在构造方法中开始调用`super().__init__()`，就像往常一样。

1.  接下来，我们使用`self.config()`来更改小部件的配置。你可能会想知道为什么我们没有将这些参数传递给`super().__init__()`调用；原因是直到底层的`Entry`小部件被初始化之后，`self.register()`方法才存在。

1.  我们注册以下两种方法：`self._validate`和`self._on_invalid`，我们将很快编写：

+   `_validate()`：这个方法将获取插入的文本（`%S`），插入的索引（`%i`），事件类型（`%V`）和执行的操作（`%d`）。

+   `_on_invalid()`：这个方法只会获取事件类型。由于我们希望在按键和`focusout`时进行验证，所以我们将`validate`设置为`all`。我们的验证方法可以通过查看事件类型（`%V`）来确定正在发生的事件。

1.  最后，我们创建`StringVar`来保存我们的错误文本；这将在类外部访问，所以我们不在其名称中使用前导下划线。

1.  我们创建的下一个方法是`_toggle_error()`，如下所示：

```py
def _toggle_error(self, error=''):
    self.error.set(error)
    if error:
        self.config(foreground='red')
    else:
        self.config(foreground='black')
```

1.  我们使用这种方法来在出现错误的情况下整合小部件的行为。它首先将我们的`error`变量设置为提供的字符串。如果字符串不为空，我们会打开错误标记（在这种情况下，将文本变为红色）；如果为空，我们会关闭错误标记。`_validate()`方法如下：

```py
    def _validate(self, char, index, event, action):

        # reset error state
        self._toggle_error()
        valid = True

        # ISO dates, YYYY-MM-DD, only need digits and hyphens
        if event == 'key':
            if action == '0':  # A delete event should always validate
                valid = True
            elif index in ('0', '1', '2', '3',
                           '5', '6', '8', '9'):
                valid = char.isdigit()
            elif index in ('4', '7'):
                valid = char == '-'
            else:
                valid = False
```

1.  我们要做的第一件事是切换关闭我们的错误状态，并将`valid`标志设置为`True`。我们的输入将是“无罪直到被证明有罪”。

1.  然后，我们将查看按键事件。`if action == '0':`告诉我们用户是否尝试删除字符。我们总是希望允许这样做，以便用户可以编辑字段。

ISO日期的基本格式是：四位数字，一个破折号，两位数字，一个破折号，和两位数字。我们可以通过检查插入的字符是否与我们在插入的`index`位置的期望相匹配来测试用户是否遵循这种格式。例如，`index in ('0', '1', '2', '3', '5', '6', '8', '9')`将告诉我们插入的字符是否是需要数字的位置之一，如果是，我们检查该字符是否是数字。索引为`4`或`7`应该是一个破折号。任何其他按键都是无效的。

尽管你可能期望它们是整数，但Tkinter将动作代码传递为字符串并将其索引化。在编写比较时要记住这一点。

虽然这是一个对于正确日期的幼稚的启发式方法，因为它允许完全无意义的日期，比如`0000-97-46`，或者看起来正确但仍然错误的日期，比如`2000-02-29`，但至少它强制执行了基本格式并消除了大量无效的按键。一个完全准确的部分日期分析器是一个单独的项目，所以现在这样做就可以了。

在`focusout`上检查我们的日期是否正确更简单，也更可靠，如下所示：

```py
        elif event == 'focusout':
            try:
                datetime.strptime(self.get(), '%Y-%m-%d')
            except ValueError:
                valid = False
        return valid
```

由于我们在这一点上可以访问用户打算输入的最终值，我们可以使用`datetime.strptime()`来尝试使用格式`%Y-%m-%d`将字符串转换为Python的`datetime`。如果失败，我们就知道日期是无效的。

结束方法时，我们返回我们的`valid`标志。

验证方法必须始终返回一个布尔值。如果由于某种原因，您的验证方法没有返回值（或返回`None`），您的验证将在没有任何错误的情况下悄悄中断。请务必确保您的方法始终返回一个布尔值，特别是如果您使用多个`return`语句。

正如您之前看到的，对于无效的按键，只需返回`False`并阻止插入字符就足够了，但对于焦点事件上的错误，我们需要以某种方式做出响应。

看一下以下代码中的`_on_invalid()`方法：

```py
    def _on_invalid(self, event):
        if event != 'key':
            self._toggle_error('Not a valid date')
```

我们只将事件类型传递给这个方法，我们将使用它来忽略按键事件（它们已经被默认行为充分处理）。对于任何其他事件类型，我们将使用我们的`_toggle_error()`方法来显示错误。

要测试我们的`DateEntry`类，请将以下测试代码添加到文件的底部：

```py
if __name__ == '__main__':
    root = tk.Tk()
    entry = DateEntry(root)
    entry.pack()
    tk.Label(textvariable=entry.error).pack()

    # add this so we can unfocus the DateEntry
    tk.Entry(root).pack()
    root.mainloop()
```

保存文件并运行它以尝试新的`DateEntry`类。尝试输入各种错误的日期或无效的按键，并看看会发生什么。

# 在我们的表单中实现验证小部件

现在您知道如何验证您的小部件，您有很多工作要做！我们有16个输入小部件，您将不得不为所有这些编写代码，以获得我们需要的行为。在这个过程中，您需要确保小部件对错误的响应是一致的，并向应用程序提供一致的API。

如果这听起来像是你想无限期推迟的事情，我不怪你。也许有一种方法可以减少我们需要编写的代码量。

# 利用多重继承的力量

到目前为止，我们已经了解到Python允许我们通过子类化创建新的类，从超类继承特性，并只添加或更改新类的不同之处。Python还支持**多重继承**，其中子类可以从多个超类继承。我们可以利用这个特性来为我们带来好处，创建所谓的**混合**类。

混合类只包含我们想要能够与其他类混合以组成新类的特定功能集。

看一下以下示例代码：

```py
class Displayer():

    def display(self, message):
        print(message)

class LoggerMixin():

    def log(self, message, filename='logfile.txt'):
        with open(filename, 'a') as fh:
            fh.write(message)

    def display(self, message):
        super().display(message)
        self.log(message)

class MySubClass(LoggerMixin, Displayer):

    def log(self, message):
        super().log(message, filename='subclasslog.txt')

subclass = MySubClass()
subclass.display("This string will be shown and logged in subclasslog.txt.")
```

我们实现了一个名为`Displayer`的基本类，其中包含一个`display()`方法，用于打印消息。然后，我们创建了一个名为`LoggerMixin`的混合类，它添加了一个`log()`方法来将消息写入文本文件，并覆盖了`display()`方法以调用`log()`。最后，我们通过同时继承`LoggerMixin`和`Displayer`来创建一个子类。子类然后覆盖了`log()`方法并设置了不同的文件名。

当我们创建一个使用多重继承的类时，我们指定的最右边的类称为**基类**，混合类应该在它之前指定。对于混合类与任何其他类没有特殊的语法，但要注意混合类的`display()`方法中使用`super()`。从技术上讲，`LoggerMixin`继承自Python内置的`object`类，该类没有`display()`方法。那么，我们如何在这里调用`super().display()`呢？

在多重继承的情况下，`super()`做的事情比仅仅代表超类要复杂一些。它使用一种叫做**方法解析顺序**的东西来查找继承链，并确定定义我们调用的方法的最近的类。因此，当我们调用`MySubclass.display()`时，会发生一系列的方法解析，如下所示：

+   `MySubClass.display()`被解析为`LoggerMixin.display()`。

+   `LoggerMixin.display()`调用`super().display()`，解析为`Displayer.display()`。

+   它还调用`self.log()`。在这种情况下，`self`是一个`MySubClass`实例，所以它解析为`MySubClass.log()`。

+   `MySubClass.log()`调用`super().log()`，解析回`LoggerMixin.log()`。

如果这看起来令人困惑，只需记住`self.method()`将首先在当前类中查找`method()`，然后按照从左到右的继承类列表查找方法。`super().method()`也会这样做，只是它会跳过当前类。

类的方法解析顺序存储在它的`__mro__`属性中；如果你在Python shell或调试器中遇到继承方法的问题，你可以检查这个方法。

请注意，`LoggerMixin`不能单独使用：它只在与具有`display()`方法的类结合时起作用。这就是为什么它是一个mixin类，因为它的目的是混合到其他类中以增强它们。

# 一个验证mixin类

让我们运用我们对多重继承的知识来构建一个mixin，通过执行以下步骤来给我们一些样板验证逻辑：

1.  打开`data_entry_app.py`并在`Application`类定义之前开始这个类：

```py
class ValidatedMixin:
    """Adds a validation functionality to an input widget"""

    def __init__(self, *args, error_var=None, **kwargs):
        self.error = error_var or tk.StringVar()
        super().__init__(*args, **kwargs)

```

1.  我们像往常一样开始这节课，尽管这次我们不会再继承任何东西。构造函数还有一个额外的参数叫做`error_var`。这将允许我们传入一个变量来用于错误消息；如果我们不这样做，类会创建自己的变量。调用`super().__init__()`将导致我们混合的基类执行它的构造函数。

1.  接下来，我们进行验证，如下所示：

```py
        vcmd = self.register(self._validate)
        invcmd = self.register(self._invalid)

        self.config(
            validate='all',
            validatecommand=(vcmd, '%P', '%s', '%S', '%V', '%i', '%d'),
            invalidcommand=(invcmd, '%P', '%s', '%S', '%V', '%i', '%d')
        )
```

1.  我们在这里设置了我们的`validate`和`invalid`方法。我们将继续传入所有的替换代码（除了`'%w'`，因为在类上下文中它几乎没有用）。我们对所有条件进行验证，所以我们可以捕获焦点和按键事件。

1.  现在，我们将定义我们的错误条件处理程序：

```py
    def _toggle_error(self, on=False):
        self.config(foreground=('red' if on else 'black'))
```

1.  如果有错误，这将只是将文本颜色更改为红色，否则更改为黑色。我们不在这个函数中设置错误，因为我们将希望在验证方法中设置实际的错误文本，如下所示：

```py
  def _validate(self, proposed, current, char, event, index, 
  action):
        self._toggle_error(False)
        self.error.set('')
        valid = True
        if event == 'focusout':
            valid = self._focusout_validate(event=event)
        elif event == 'key':
            valid = self._key_validate(proposed=proposed,
                current=current, char=char, event=event,
                index=index, action=action)
        return valid

    def _focusout_validate(self, **kwargs):
        return True

    def _key_validate(self, **kwargs):
        return True 
```

我们的`_validate()`方法只处理一些设置工作，比如关闭错误和清除错误消息。然后，它运行一个特定于事件的验证方法，取决于传入的事件类型。我们现在只关心`key`和`focusout`事件，所以任何其他事件都会返回`True`。

请注意，我们使用关键字调用各个方法；当我们创建我们的子类时，我们将覆盖这些方法。通过使用关键字参数，我们覆盖的函数只需指定所需的关键字或从`**kwargs`中提取单个参数，而不必按正确的顺序获取所有参数。还要注意，所有参数都传递给`_key_validate()`，但只有`event`传递给`_focusout_validate()`。焦点事件对于其他参数都没有有用的返回值，所以将它们传递下去没有意义。

1.  这里的最终想法是，我们的子类只需要覆盖我们关心的小部件的验证方法或方法。如果我们不覆盖它们，它们就会返回`True`，所以验证通过。现在，我们需要处理一个无效的事件：

```py
   def _invalid(self, proposed, current, char, event, index, 
   action):
        if event == 'focusout':
            self._focusout_invalid(event=event)
        elif event == 'key':
            self._key_invalid(proposed=proposed,
                current=current, char=char, event=event,
                index=index, action=action)

    def _focusout_invalid(self, **kwargs):
        self._toggle_error(True)

    def _key_invalid(self, **kwargs):
        pass

```

1.  我们对这些方法采取相同的方法。不像验证方法，我们的无效数据处理程序不需要返回任何内容。对于无效的键，默认情况下我们什么也不做，对于`focusout`上的无效数据，我们切换错误状态。

1.  按键验证只在输入键的情况下才有意义，但有时我们可能希望手动运行`focusout`检查，因为它有效地检查完全输入的值。因此，我们将实现以下方法：

```py
   def trigger_focusout_validation(self):
        valid = self._validate('', '', '', 'focusout', '', '')
        if not valid:
            self._focusout_invalid(event='focusout')
        return valid
```

1.  我们只是复制了`focusout`事件发生时发生的逻辑：运行验证函数，如果失败，则运行无效处理程序。这就是我们对`ValidatedMixin`所需的全部内容，所以让我们开始将其应用于一些小部件，看看它是如何工作的。

# 构建我们的小部件

让我们仔细考虑我们需要使用新的`ValidatedMixin`类实现哪些类，如下所示：

+   除了`Notes`之外，我们所有的字段都是必需的，因此我们需要一个基本的`Entry`小部件，如果没有输入，则会注册错误。

+   我们有一个`Date`字段，因此我们需要一个强制有效日期字符串的`Entry`小部件。

+   我们有一些用于十进制或整数输入的`Spinbox`小部件。我们需要确保这些只接受有效的数字字符串。

+   我们有一些`Combobox`小部件的行为不太符合我们的期望。

让我们开始吧！

# 需要数据

我们所有的字段都是必需的，所以让我们从一个需要数据的基本`Entry`小部件开始。我们可以将这些用于字段：`Technician`和`Seed sample`。

在`ValidatedMixin`类下添加以下代码：

```py
class RequiredEntry(ValidatedMixin, ttk.Entry):

    def _focusout_validate(self, event):
        valid = True
        if not self.get():
            valid = False
            self.error.set('A value is required')
        return valid
```

这里没有按键验证要做，所以我们只需要创建`_focusout_validate()`。如果输入的值为空，我们只需设置一个错误字符串并返回`False`。

就是这样了！

# 日期小部件

现在，让我们将mixin类应用于之前制作的`DateEntry`类，保持相同的验证算法如下：

```py
class DateEntry(ValidatedMixin, ttk.Entry):

    def _key_validate(self, action, index, char, **kwargs):
        valid = True

        if action == '0':
            valid = True
        elif index in ('0', '1', '2', '3', '5', '6', '8', '9'):
            valid = char.isdigit()
        elif index in ('4', '7'):
            valid = char == '-'
        else:
            valid = False
        return valid

    def _focusout_validate(self, event):
        valid = True
        if not self.get():
            self.error.set('A value is required')
            valid = False
        try:
            datetime.strptime(self.get(), '%Y-%m-%d')
        except ValueError:
            self.error.set('Invalid date')
            valid = False
        return valid
```

同样，非常简单，我们只需要指定验证逻辑。我们还添加了来自我们的`RequiredEntry`类的逻辑，因为`Date`值是必需的。

让我们继续进行一些更复杂的工作。

# 更好的Combobox小部件

不同工具包中的下拉式小部件在鼠标操作时表现相当一致，但对按键的响应有所不同，如下所示：

+   有些什么都不做

+   有些需要使用箭头键来选择项目

+   有些移动到按下任意键开始的第一个条目，并在后续按键开始的条目之间循环

+   有些会缩小列表以匹配所键入的内容

我们需要考虑我们的`Combobox`小部件应该具有什么行为。由于我们的用户习惯于使用键盘进行数据输入，有些人使用鼠标有困难，小部件需要与键盘配合使用。让他们重复按键来选择选项也不是很直观。与数据输入人员讨论后，您决定采用以下行为：

+   如果建议的文本与任何条目都不匹配，它将被忽略。

+   当建议的文本与单个条目匹配时，小部件将设置为该值

+   删除或退格会清除整个框

在`DateEntry`代码下添加此代码：

```py
class ValidatedCombobox(ValidatedMixin, ttk.Combobox):

    def _key_validate(self, proposed, action, **kwargs):
        valid = True
        # if the user tries to delete, just clear the field
        if action == '0':
            self.set('')
            return True
```

`_key_validate()`方法首先设置一个`valid`标志，并快速检查是否是删除操作。如果是，我们将值设置为空字符串并返回`True`。

现在，我们将添加逻辑来匹配建议的文本与我们的值：

```py
       # get our values list
        values = self.cget('values')
        # Do a case-insensitive match against the entered text
        matching = [
            x for x in values
            if x.lower().startswith(proposed.lower())
        ]
        if len(matching) == 0:
            valid = False
        elif len(matching) == 1:
            self.set(matching[0])
            self.icursor(tk.END)
            valid = False
        return valid
```

使用其`.cget()`方法检索小部件值列表的副本。然后，我们使用列表推导来将此列表减少到仅与建议的文本匹配的条目，对列表项和建议的文本的值调用`lower()`，以便我们的匹配不区分大小写。

每个Tkinter小部件都支持`.cget()`方法。它可以用来按名称检索小部件的任何配置值。

如果匹配列表的长度为`0`，我们拒绝按键。如果为`1`，我们找到了匹配，所以我们将变量设置为该值。如果是其他任何值，我们需要让用户继续输入。作为最后的修饰，如果找到匹配，我们将使用`.icursor()`方法将光标发送到字段的末尾。这并不是严格必要的，但比将光标留在文本中间看起来更好。现在，我们将添加`focusout`验证器，如下所示：

```py
    def _focusout_validate(self, **kwargs):
        valid = True
        if not self.get():
            valid = False
            self.error.set('A value is required')
        return valid
```

这里我们不需要做太多，因为关键验证方法确保唯一可能的值是空字段或值列表中的项目，但由于所有字段都需要有一个值，我们将从`RequiredEntry`复制验证。

这就处理了我们的`Combobox`小部件。接下来，我们将处理`Spinbox`小部件。

# 范围限制的Spinbox小部件

数字输入似乎不应该太复杂，但有许多微妙之处需要解决，以使其牢固。除了将字段限制为有效的数字值之外，您还希望将`from`、`to`和`increment`参数分别强制为输入的最小、最大和精度。

算法需要实现以下规则：

+   删除始终允许

+   数字始终允许

+   如果`from`小于`0`，则允许减号作为第一个字符

+   如果`increment`有小数部分，则允许一个点

+   如果建议的值大于`to`值，则忽略按键

+   如果建议的值需要比`increment`更高的精度，则忽略按键

+   在`focusout`时，确保值是有效的数字字符串

+   同样在`focusout`时，确保值大于`from`值

看一下以下步骤：

1.  以下是我们将如何编码，关于前面的规则：

```py
class ValidatedSpinbox(ValidatedMixin, tk.Spinbox):

    def __init__(self, *args, min_var=None, max_var=None,
                 focus_update_var=None, from_='-Infinity',    
                 to='Infinity', **kwargs):
        super().__init__(*args, from_=from_, to=to, **kwargs)
        self.resolution = Decimal(str(kwargs.get('increment',  
        '1.0')))
        self.precision = (
            self.resolution
            .normalize()
```

```py
            .as_tuple()
            .exponent
        )
```

1.  我们将首先重写`__init__()`方法，以便我们可以指定一些默认值，并从构造函数参数中获取`increment`值以进行处理。

1.  `Spinbox`参数可以作为浮点数、整数或字符串传递。无论如何传递，Tkinter都会将它们转换为浮点数。确定浮点数的精度是有问题的，因为浮点误差的原因，所以我们希望在它变成浮点数之前将其转换为Python `Decimal`。

浮点数尝试以二进制形式表示十进制数。打开Python shell并输入`1.2 / .2`。您可能会惊讶地发现答案是`5.999999999999999`而不是`6`。这被称为**浮点误差**，几乎在每种编程语言中都是计算错误的来源。Python为我们提供了`Decimal`类，它接受一个数字字符串并以一种使数学运算免受浮点误差的方式存储它。

1.  在我们使用`Decimal`之前，我们需要导入它。在文件顶部的导入中添加以下代码：

```py
from decimal import Decimal, InvalidOperation
```

1.  `InvalidOperation`是当`Decimal`得到一个它无法解释的字符串时抛出的异常。我们稍后会用到它。

请注意，在将其传递给`Decimal`之前，我们将`increment`转换为`str`。理想情况下，我们应该将`increment`作为字符串传递，以确保它将被正确解释，但以防我们因某种原因需要传递一个浮点数，`str`将首先进行一些明智的四舍五入。

1.  我们还为`to`和`from_`设置了默认值：`-Infinity`和`Infinity`。`float`和`Decimal`都会愉快地接受这些值，并将它们视为您期望的那样处理。`Tkinter.Spinbox`的默认`to`和`from_`值为`0`；如果它们保留在那里，Tkinter会将其视为无限制，但如果我们指定一个而不是另一个，这就会产生问题。

1.  我们提取`resolution`值的`precision`作为最小有效小数位的指数。我们将在验证类中使用这个值。

1.  我们的构造函数已经确定，所以让我们编写验证方法。关键验证方法有点棘手，所以我们将一步一步地走过它。首先，我们开始这个方法：

```py
    def _key_validate(self, char, index, current,
                      proposed, action, **kwargs):
        valid = True
        min_val = self.cget('from')
        max_val = self.cget('to')
        no_negative = min_val >= 0
        no_decimal = self.precision >= 0
```

1.  首先，我们检索`from`和`to`值，然后分配标志变量以指示是否应允许负数和小数，如下所示：

```py
        if action == '0':
            return True
```

删除应该总是有效的，所以如果是删除，返回`True`。

我们在这里打破了不要多次返回的准则，因为只有一个`return`的相同逻辑会嵌套得非常深。在尝试编写可读性好、易于维护的代码时，有时不得不选择两害相权取其轻。

1.  接下来，我们测试按键是否是有效字符，如下所示：

```py
      # First, filter out obviously invalid keystrokes
        if any([
                (char not in ('-1234567890.')),
                (char == '-' and (no_negative or index != '0')),
                (char == '.' and (no_decimal or '.' in current))
        ]):
            return False
```

有效字符是数字加上`-`和`.`。减号只在索引`0`处有效，点只能出现一次。其他任何字符都返回`False`。

内置的`any`函数接受一个表达式列表，并在列表中的任何一个表达式为真时返回`True`。还有一个`all`函数，如果所有表达式都为真，则返回`True`。这些函数允许您压缩一长串布尔表达式。

在这一点上，我们几乎可以保证有一个有效的`Decimal`字符串，但还不够；我们可能只有`-`、`.`或`-.`字符。

1.  以下是有效的部分条目，因此我们只需为它们返回`True`：

```py
        # At this point, proposed is either '-', '.', '-.',
        # or a valid Decimal string
        if proposed in '-.':
            return True
```

1.  此时，建议的文本只能是有效的`Decimal`字符串，因此我们将从中制作一个`Decimal`并进行更多的测试：

```py
        # Proposed is a valid Decimal string
        # convert to Decimal and check more:
        proposed = Decimal(proposed)
        proposed_precision = proposed.as_tuple().exponent

        if any([
            (proposed > max_val),
            (proposed_precision < self.precision)
        ]):
            return False

        return valid
```

1.  我们最后两个测试检查建议的文本是否大于我们的最大值，或者比我们指定的“增量”具有更多的精度（我们在这里使用`<`运算符的原因是因为“精度”给出为小数位的负值）。如果还没有返回任何内容，我们将返回`valid`值作为保障。我们的`focusout`验证器要简单得多，如下所示：

```py
    def _focusout_validate(self, **kwargs):
        valid = True
        value = self.get()
        min_val = self.cget('from')

        try:
            value = Decimal(value)
        except InvalidOperation:
            self.error.set('Invalid number string: {}'.format(value))
            return False

        if value < min_val:
            self.error.set('Value is too low (min {})'.format(min_val))
            valid = False
        return valid
```

1.  有了整个预期值，我们只需要确保它是有效的`Decimal`字符串并且大于最小值。

有了这个，我们的`ValidatedSpinbox`已经准备就绪。

# 动态调整Spinbox范围

我们的`ValidatedSpinbox`方法似乎对我们的大多数字段都足够了。但是考虑一下`Height`字段。`Mini height`值大于`Max height`值或`Median height`值不在它们之间是没有意义的。有没有办法将这种相互依赖的行为融入到我们的类中？

我们可以！为此，我们将依赖Tkinter变量的**跟踪**功能。跟踪本质上是对变量的`.get()`和`.set()`方法的钩子，允许您在读取或更改变量时触发任何Python函数或方法。

语法如下：

```py
sv = tk.StringVar()
sv.trace('w', some_function_or_method)
```

`.trace()`的第一个参数表示我们要跟踪的事件。这里，`w`表示写（`.set()`），`r`表示读（`.get()`），`u`表示未定义的变量或删除变量。

我们的策略是允许可选的`min_var`和`max_var`变量进入`ValidatedSpinbox`方法，并在这些变量上设置一个跟踪，以便在更改此变量时更新`ValidatedSpinbox`方法的最小或最大值。我们还将有一个`focus_update_var`变量，它将在`focusout`时间更新为`Spinbox`小部件值。

让我们看看以下步骤：

1.  首先，我们将更新我们的`ValidatedSpinbox`构造函数如下：

```py
    def __init__(self, *args, min_var=None, max_var=None,
        focus_update_var=None, from_='-Infinity', to='Infinity', 
    **kwargs
    ):
        super().__init__(*args, from_=from_, to=to, **kwargs)
        self.resolution = Decimal(str(kwargs.get('increment', '1.0')))
        self.precision = (
            self.resolution
            .normalize()
            .as_tuple()
            .exponent
        )
        # there should always be a variable,
        # or some of our code will fail
        self.variable = kwargs.get('textvariable') or tk.DoubleVar()

        if min_var:
            self.min_var = min_var
            self.min_var.trace('w', self._set_minimum)
        if max_var:
            self.max_var = max_var
            self.max_var.trace('w', self._set_maximum)
        self.focus_update_var = focus_update_var
        self.bind('<FocusOut>', self._set_focus_update_var)
```

1.  首先，请注意我们已经添加了一行来将变量存储在`self.variable`中，如果程序没有明确传入变量，我们将创建一个变量。我们需要编写的一些代码将取决于文本变量的存在，因此我们将强制执行这一点，以防万一。

1.  如果我们传入`min_var`或`max_var`参数，该值将被存储，并配置一个跟踪。`trace()`方法指向一个适当命名的方法。

1.  我们还存储了对`focus_update_var`参数的引用，并将`<FocusOut>`事件绑定到一个方法，该方法将用于更新它。

`bind()`方法可以在任何Tkinter小部件上调用，它用于将小部件事件连接到Python可调用函数。事件可以是按键、鼠标移动或点击、焦点事件、窗口管理事件等等。

1.  现在，我们需要为我们的`trace()`和`bind()`命令添加回调方法。首先从`_set_focus_update_var()`开始，如下所示：

```py
def _set_focus_update_var(self, event):
        value = self.get()
        if self.focus_update_var and not self.error.get():
            self.focus_update_var.set(value)
```

这个方法只是简单地获取小部件的当前值，并且如果实例中存在`focus_update_var`参数，则将其设置为相同的值。请注意，如果小部件当前存在错误，我们不会设置值。将值更新为无效值是没有意义的。

当Tkinter调用`bind`回调时，它传递一个包含有关触发回调的事件的信息的事件对象。即使您不打算使用这些信息，您的函数或方法也需要能够接受此参数。

1.  现在，让我们创建设置最小值的回调，如下所示：

```py
    def _set_minimum(self, *args):
        current = self.get()
        try:
            new_min = self.min_var.get()
            self.config(from_=new_min)
        except (tk.TclError, ValueError):
            pass
        if not current:
            self.delete(0, tk.END)
        else:
            self.variable.set(current)
        self.trigger_focusout_validation()
```

1.  我们要做的第一件事是检索当前值。`Tkinter.Spinbox`在更改`to`或`from`值时有稍微让人讨厌的行为，将太低的值移动到`from`值，将太高的值移动到`to`值。这种悄悄的自动校正可能会逃过我们用户的注意，导致坏数据被保存。我们希望的是将值留在范围之外，并将其标记为错误；因此，为了解决Tkinter的问题，我们将保存当前值，更改配置，然后将原始值放回字段中。

1.  保存当前值后，我们尝试获取`min_var`的值，并从中设置我们的小部件的`from_`值。这里可能会出现几种问题，例如控制我们的最小和最大变量的字段中有空白或无效值，所有这些都应该引发`tk.TclError`或`ValueError`。在任何一种情况下，我们都不会做任何事情。

通常情况下，只是消除异常是一个坏主意；然而，在这种情况下，如果变量有问题，我们无法合理地做任何事情，除了忽略它。

1.  现在，我们只需要将我们保存的当前值写回字段。如果为空，我们只需删除字段；否则，我们设置输入的变量。该方法以调用`trigger_focusout_validation()`方法结束，以重新检查字段中的值与新最小值的匹配情况。

1.  `_set_maximum()`方法将与此方法相同，只是它将使用`max_var`来更新`to`值。您可以自己编写它，或者查看本书附带的示例代码。

1.  我们需要对我们的`ValidatedSpinbox`类进行最后一个更改。由于我们的最大值可能在输入后更改，并且我们依赖于我们的`focusout`验证来检测它，我们需要添加一些条件来检查最大值。

1.  我们需要将这个添加到`_focusout_validate()`方法中：

```py
        max_val = self.cget('to')
        if value > max_val:
            self.error.set('Value is too high (max {})'.format(max_val))
```

1.  在`return`语句之前添加这些行以检查最大值并根据需要设置错误。

# 更新我们的表单

现在我们所有的小部件都已经制作好了，是时候通过执行以下步骤让表单使用它们了：

1.  向下滚动到`DataRecordForm`类构造函数，并且我们将逐行更新我们的小部件。第1行非常简单：

```py
        self.inputs['Date'] = LabelInput(
            recordinfo, "Date",
            input_class=DateEntry,
            input_var=tk.StringVar())
        self.inputs['Date'].grid(row=0, column=0)
        self.inputs['Time'] = LabelInput(
            recordinfo, "Time",
            input_class=ValidatedCombobox,
            input_var=tk.StringVar(),
            input_args={"values": ["8:00", "12:00", "16:00", "20:00"]})
        self.inputs['Time'].grid(row=0, column=1)
        self.inputs['Technician'] = LabelInput(
            recordinfo, "Technician",
            input_class=RequiredEntry,
            input_var=tk.StringVar())
        self.inputs['Technician'].grid(row=0, column=2)
```

1.  将`LabelInput`中的`input_class`值替换为我们的新类就像交换一样简单。继续运行你的应用程序并尝试小部件。尝试一些不同的有效和无效日期，并查看`Combobox`小部件的工作方式（`RequiredEntry`在这一点上不会有太多作用，因为唯一可见的指示是红色文本，如果为空，就没有文本标记为红色；我们稍后会解决这个问题）。现在，转到第2行，首先添加`Lab`小部件，如下所示：

```py
        self.inputs['Lab'] = LabelInput(
            recordinfo, "Lab",
            input_class=ValidatedCombobox,
            input_var=tk.StringVar(),
            input_args={"values": ["A", "B", "C", "D", "E"]})
```

1.  接下来，添加`Plot`小部件，如下所示：

```py
        self.inputs['Plot'] = LabelInput(
            recordinfo, "Plot",
            input_class=ValidatedCombobox,
            input_var=tk.IntVar(),
            input_args={"values": list(range(1, 21))})
```

再次相当简单，但如果您运行它，您会发现`Plot`存在问题。事实证明，当值为整数时，我们的`ValidatedComobox`方法无法正常工作，因为用户键入的字符始终是字符串（即使它们是数字）；我们无法比较字符串和整数。

1.  如果您考虑一下，`Plot`实际上不应该是一个整数值。是的，这些值在技术上是整数，但正如我们在第3章*使用Tkinter和ttk小部件创建基本表单*中决定的那样，它们也可以是字母或符号；您不会在一个图表号上进行数学运算。因此，我们将更改`Plot`以使用`StringVar`变量，并将小部件的值也更改为字符串。更改`Plot`小部件的创建如下所示：

```py
       self.inputs['Plot'] = LabelInput(
            recordinfo, "Plot",
            input_class=ValidatedCombobox,
            input_var=tk.StringVar(),
            input_args={"values": [str(x) for x in range(1, 21)]})
```

1.  在这里，我们只是将`input_var`更改为`StringVar`，并使用列表推导将每个`values`项转换为字符串。现在，`Plot`的工作正常了。

1.  继续通过表单，用新验证的版本替换默认的`ttk`小部件。对于`Spinbox`小部件，请确保将`to`、`from_`和`increment`值作为字符串而不是整数传递。例如，`Humidity`小部件应该如下所示：

```py
        self.inputs['Humidity'] = LabelInput(
            environmentinfo, "Humidity (g/m³)",
            input_class=ValidatedSpinbox,
            input_var=tk.DoubleVar(),
            input_args={"from_": '0.5', "to": '52.0', "increment": 
            '.01'})
```

1.  当我们到达`Height`框时，是时候测试我们的`min_var`和`max_var`功能了。首先，我们需要设置变量来存储最小和最大高度，如下所示：

```py
        # Height data
        # create variables to be updated for min/max height
        # they can be referenced for min/max variables
        min_height_var = tk.DoubleVar(value='-infinity')
        max_height_var = tk.DoubleVar(value='infinity')
```

我们创建两个新的`DoubleVar`对象来保存当前的最小和最大高度，将它们设置为无限值。这确保一开始实际上没有最小或最大高度。

请注意，我们的小部件直到它们实际更改才会受到这些值的影响，因此它们不会使传入的原始`to`和`from_`值无效。

1.  现在，我们创建`Min Height`小部件，如下所示：

```py
        self.inputs['Min Height'] = LabelInput(
            plantinfo, "Min Height (cm)",
            input_class=ValidatedSpinbox,
            input_var=tk.DoubleVar(),
            input_args={
                "from_": '0', "to": '1000', "increment": '.01',
                "max_var": max_height_var, "focus_update_var": 
                 min_height_var})
```

1.  我们将使用`max_height_var`在此处设置最大值，确保我们的最小值永远不会超过最大值，并将`focus_update_var`设置为`min_height_var`的值，以便在更改此字段时它将被更新。现在，`Max Height`小部件如下所示：

```py
        self.inputs['Max Height'] = LabelInput(
            plantinfo, "Max Height (cm)",
            input_class=ValidatedSpinbox,
            input_var=tk.DoubleVar(),
            input_args={
                "from_": 0, "to": 1000, "increment": .01,
                "min_var": min_height_var, "focus_update_var":  
                max_height_var})
```

1.  这一次，我们使用我们的`min_height_var`变量来设置小部件的最小值，并从小部件的当前值更新`max_height_var`。最后，`Median Height`字段如下所示：

```py
        self.inputs['Median Height'] = LabelInput(
            plantinfo, "Median Height (cm)",
            input_class=ValidatedSpinbox,
            input_var=tk.DoubleVar(),
            input_args={
                "from_": 0, "to": 1000, "increment": .01,
                "min_var": min_height_var, "max_var": max_height_var})
```

1.  在这里，我们分别从`min_height_var`和`max_height_var`变量设置字段的最小和最大值。我们不会更新任何来自`Median Height`字段的变量，尽管我们可以在这里添加额外的变量和代码，以确保`Min Height`不会超过它，或者`Max Height`不会低于它。在大多数情况下，如果用户按顺序输入数据，`Median Height`就不重要了。

1.  您可能会想知道为什么我们不直接使用`Min Height`和`Max Height`中的`input_var`变量来保存这些值。如果您尝试这样做，您会发现原因：`input_var`会随着您的输入而更新，这意味着您的部分值立即成为新的最大值或最小值。我们宁愿等到用户提交值后再分配这个值，因此我们创建了一个只在`focusout`时更新的单独变量。

# 显示错误

如果您运行应用程序，您可能会注意到，虽然`focusout`错误的字段变红，但我们无法看到实际的错误。我们需要通过执行以下步骤来解决这个问题：

1.  找到您的`LabelInput`类，并将以下代码添加到构造方法的末尾：

```py
        self.error = getattr(self.input, 'error', tk.StringVar())
        self.error_label = ttk.Label(self, textvariable=self.error)
        self.error_label.grid(row=2, column=0, sticky=(tk.W + tk.E))
```

1.  在这里，我们检查我们的输入是否有错误变量，如果没有，我们就创建一个。我们将它保存为`self.error`的引用，然后创建一个带有错误的`textvariable`的`Label`。

1.  最后，我们将这个放在输入小部件下面。

1.  现在，当您尝试应用程序时，您应该能够看到字段错误。

# 防止表单在出现错误时提交

阻止错误进入CSV文件的最后一步是，如果表单存在已知错误，则停止应用程序保存。让我们执行以下步骤来做到这一点：

1.  实施这一步的第一步是为`Application`对象（负责保存数据）提供一种从`DataRecordForm`对象检索错误状态的方法。

1.  在`DataRecordForm`类的末尾，添加以下方法：

```py
    def get_errors(self):
        """Get a list of field errors in the form"""

        errors = {}
        for key, widget in self.inputs.items():
            if hasattr(widget.input, 'trigger_focusout_validation'):
                widget.input.trigger_focusout_validation()
            if widget.error.get():
                errors[key] = widget.error.get()

        return errors
```

1.  与我们处理数据的方式类似，我们只需循环遍历`LabelFrame`小部件。我们寻找具有`trigger_focusout_validation`方法的输入，并调用它，以确保所有值都已经被检查。然后，如果小部件的`error`变量有任何值，我们将其添加到一个`errors`字典中。这样，我们可以检索每个字段的字段名称和错误的字典。

1.  现在，我们需要将此行为添加到`Application`类的保存逻辑中。

1.  在`on_save()`的开头添加以下代码，在`docstring`下面：

```py
        # Check for errors first

        errors = self.recordform.get_errors()
        if errors:
            self.status.set(
                "Cannot save, error in fields: {}"
                .format(', '.join(errors.keys()))
            )
            return False
```

这个逻辑很简单：获取错误，如果我们找到任何错误，就在状态区域警告用户并从函数返回（因此不保存任何内容）。

1.  启动应用程序并尝试保存一个空白表单。您应该在所有字段中收到错误消息，并在底部收到一个消息，告诉您哪些字段有错误。

# 自动化输入

防止用户输入错误数据是帮助用户输入更好数据的一种方式；另一种方法是自动化。利用我们对表单可能如何填写的理解，我们可以插入对于某些字段非常可能是正确的值。

请记住[第2章](3ec510a4-0919-4f25-9c34-f7bbd4199912.xhtml)中提到的，*使用Tkinter设计GUI应用程序*，表单几乎总是在填写当天录入，并且按顺序从`Plot` 1到`Plot` 20依次填写。还要记住，`Date`，`Lab`和`Technician`的值对每个填写的表单保持不变。让我们为我们的用户自动化这个过程。

# 插入日期

插入当前日期是一个简单的开始地方。这个地方是在`DataRecordForm.reset()`方法中，该方法设置了输入新记录的表单。

按照以下方式更新该方法：

```py
    def reset(self):
        """Resets the form entries"""

        # clear all values
        for widget in self.inputs.values():
            widget.set('')

        current_date = datetime.today().strftime('%Y-%m-%d')
        self.inputs['Date'].set(current_date)
```

就像我们在`Application.save()`方法中所做的那样，我们从`datetime.today()`获取当前日期并将其格式化为ISO日期。然后，我们将`Date`小部件的输入设置为该值。

# 自动化Lab，Time和Technician

稍微复杂一些的是我们对`Lab`，`Time`和`Technician`的处理。让我们按照以下逻辑进行审查：

1.  在清除数据之前，保存`Lab`，`Time`和`Technician`的值。

1.  如果`Plot`小于最后一个值（`20`），我们将在清除所有字段后将这些值放回，然后增加到下一个`Plot`值。

1.  如果`Plot`是最后一个值或没有值，则将这些字段留空。代码如下：

```py
   def reset(self):
        """Resets the form entries"""

        # gather the values to keep for each lab
        lab = self.inputs['Lab'].get()
        time = self.inputs['Time'].get()
        technician = self.inputs['Technician'].get()
        plot = self.inputs['Plot'].get()
        plot_values = self.inputs['Plot'].input.cget('values')

        # clear all values
        for widget in self.inputs.values():
            widget.set('')

        current_date = datetime.today().strftime('%Y-%m-%d')
        self.inputs['Date'].set(current_date)
        self.inputs['Time'].input.focus()

        # check if we need to put our values back, then do it.
        if plot not in ('', plot_values[-1]):
            self.inputs['Lab'].set(lab)
            self.inputs['Time'].set(time)
            self.inputs['Technician'].set(technician)
            next_plot_index = plot_values.index(plot) + 1
            self.inputs['Plot'].set(plot_values[next_plot_index])
            self.inputs['Seed sample'].input.focus()
```

因为`Plot`看起来像一个整数，可能会诱人像增加一个整数一样增加它，但最好将其视为非整数。我们使用值列表的索引。

1.  最后一个微调，表单的焦点始终从第一个字段开始，但这意味着用户必须通过已经填写的字段进行标签。如果下一个空输入从一开始就聚焦，那将是很好的。Tkinter输入有一个`focus()`方法，它可以给它们键盘焦点。根据我们填写的字段，这要么是`Time`，要么是`Seed sample`。在设置`Date`值的下一行下面，添加以下代码行：

```py
self.inputs['Time'].input.focus()
```

1.  在设置`Plot`值的行下面，在条件块内，添加以下代码行：

```py
self.inputs['Seed sample'].input.focus()
```

我们的表单现在已经准备好与用户进行试运行。在这一点上，它绝对比CSV输入有所改进，并将帮助数据输入快速完成这些表单。

# 总结

应用程序已经取得了长足的进步。在本章中，我们学习了Tkinter验证，创建了一个验证混合类，并用它来创建`Entry`，`Combobox`和`Spinbox`小部件的验证版本。我们在按键和焦点事件上验证了不同类型的数据，并创建了根据相关字段的值动态更新其约束的字段。

在下一章中，我们将准备我们的代码基础以便扩展，并学习如何组织一个大型应用程序以便更容易维护。更具体地说，我们将学习MVC模式以及如何将我们的代码结构化为多个文件，以便更简单地进行维护。我们还将更多地了解RST和版本控制软件。
