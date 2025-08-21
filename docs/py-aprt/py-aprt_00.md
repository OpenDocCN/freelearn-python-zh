## 前言

这本书是通过迂回的方式产生的。2013 年，当我们成立了位于挪威的软件咨询和培训公司 *Sixty North* 时，我们受到了在线视频培训材料出版商 *Pluralsight* 的追捧，他们希望我们为迅速增长的大规模在线开放课程（MOOC）市场制作 Python 培训视频。当时，我们没有制作视频培训材料的经验，但我们确定希望仔细构建我们的 Python 入门内容，以尊重某些限制。例如，我们希望最少使用前向引用，因为这对我们的观众来说非常不方便。我们都是言辞之人，遵循图灵奖得主莱斯利·兰波特的格言 *“如果你在不写作的情况下思考，你只是以为自己在思考”*，因此，我们首先通过撰写脚本来攻击视频课程制作。

很快，我们的在线视频课程被 *Pluralsight* 以 [Python 基础知识](https://www.pluralsight.com/courses/python-fundamentals) 的形式写成、录制并发布，受到了极其积极的反响，这种反响已经持续了几年。从最早的日子起，我们就想到这个脚本可以成为一本书的基础，尽管可以说我们低估了将内容从一个好的脚本转化为一本更好的书所需的努力。

*Python 学徒* 就是这种转变的结果。它可以作为独立的 Python 教程，也可以作为我们视频课程的配套教材，具体取决于哪种学习方式更适合您。*Python 学徒* 是三本书中的第一本，另外两本分别是 [*Python 熟练者*](https://leanpub.com/python-journeyman) 和 [*Python 大师*](https://leanpub.com/python-master)。后两本书对应于我们随后的 *Pluralsight* 课程 [*Python - 进阶*](https://app.pluralsight.com/library/courses/python-beyond-basics/) 和 [*高级 Python*](https://app.pluralsight.com/library/courses/advanced-python/)。

### 勘误和建议

本书中的所有材料都经过了彻底的审查和测试；然而，不可避免地会出现一些错误。如果您发现了错误，我们会非常感激您通过 *Leanpub* [Python 学徒讨论](https://leanpub.com/python-apprentice/feedback) 页面让我们知道，这样我们就可以进行修正并部署新版本。

### 本书中使用的约定

本书中的代码示例显示为带有语法高亮的固定宽度文本：

```py
>>> def square(x):
...     return x * x
...

```

我们的一些示例显示了保存在文件中的代码，而其他一些示例（如上面的示例）来自交互式 Python 会话。在这种交互式情况下，我们包括 Python 会话中的提示符，如三角形箭头（`>>>`）和三个点（`...`）提示符。您不需要输入这些箭头或点。同样，对于操作系统的 shell 命令，我们将使用 Linux、macOS 和其他 Unix 系统的美元提示符（`$`），或者在特定操作系统对于当前任务无关紧要的情况下使用。

```py
$ python3 words.py

```

在这种情况下，您不需要输入 `$` 字符。

对于特定于 Windows 的命令，我们将使用一个前导大于提示符：

```py
> python words.py

```

同样，无需输入 `>` 字符。

对于需要放置在文件中而不是交互输入的代码块，我们显示的代码没有任何前导提示符：

```py
def write_sequence(filename, num):
    """Write Recaman's sequence to a text file."""
    with open(filename, mode='wt', encoding='utf-8') as f:
        f.writelines("{0}\n".format(r)
                     for r in islice(sequence(), num + 1))

```

我们努力确保我们的代码行足够短，以便每一行逻辑代码对应于您的书中的一行物理代码。然而，电子书发布到不同设备的变化和偶尔需要长行代码的真正需求意味着我们无法保证行不会换行。然而，我们可以保证，如果一行换行，出版商已经在最后一列插入了一个反斜杠字符 `\`。您需要自行判断这个字符是否是代码的合法部分，还是由电子书平台添加的。

```py
>>> print("This is a single line of code which is very long. Too long, in fact, to fi\
t on a single physical line of code in the book.")

```

如果您在上述引用的字符串中看到一条反斜杠，那么它*不*是代码的一部分，不应该输入。

偶尔，我们会对代码行进行编号，这样我们就可以很容易地从下一个叙述中引用它们。这些行号不应该作为代码的一部分输入。编号的代码块看起来像这样：

```py
 1 def write_grayscale(filename, pixels):
 2    height = len(pixels)
 3    width = len(pixels[0])
 4 
 5    with open(filename, 'wb') as bmp:
 6        # BMP Header
 7        bmp.write(b'BM')
 8 
 9        # The next four bytes hold the filesize as a 32-bit
10         # little-endian integer. Zero placeholder for now.
11         size_bookmark = bmp.tell()
12         bmp.write(b'\x00\x00\x00\x00')

```

有时我们需要呈现不完整的代码片段。通常这是为了简洁起见，我们要向现有的代码块添加代码，并且我们希望清楚地了解代码块的结构，而不重复所有现有的代码块内容。在这种情况下，我们使用包含三个点的 Python 注释`# ...`来指示省略的代码：

```py
class Flight:

    # ...

    def make_boarding_cards(self, card_printer):
        for passenger, seat in sorted(self._passenger_seats()):
            card_printer(passenger, seat, self.number(), self.aircraft_model())

```

这里暗示了在`Flight`类块中的`make_boarding_cards()`函数之前已经存在一些其他代码。

最后，在书的文本中，当我们提到一个既是标识符又是函数的标识符时，我们将使用带有空括号的标识符，就像我们在前面一段中使用`make_boarding_cards()`一样。
