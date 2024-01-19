# 第十四章：文件 I/O 和 Python 工具

在本章中，我们将详细讨论文件 I/O，即读取、写入和追加文件。我们还将讨论 Python 工具，这些工具使得操作文件和与操作系统交互成为可能。每个主题都有不同的复杂程度，我们将通过一个例子来讨论。让我们开始吧！

# 文件 I/O

我们讨论文件 I/O 有两个原因：

+   在 Linux 操作系统的世界中，一切都是文件。与树莓派上的外围设备交互类似于读取/写入文件。例如：在第十二章中，*通信接口*，我们讨论了串口通信。您应该能够观察到串口通信类似于文件读写操作。

+   我们在每个项目中以某种形式使用文件 I/O。例如：将传感器数据写入 CSV 文件，或者读取 Web 服务器的预配置选项等。

因此，我们认为讨论 Python 中的文件 I/O 作为一个单独的章节会很有用（详细文档请参阅：[`docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files`](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)），并讨论它在开发树莓派 Zero 应用程序时可能发挥作用的示例。

# 从文件中读取

让我们创建一个简单的文本文件`read_file.txt`，其中包含以下文本：`我正在使用树莓派 Zero 学习 Python 编程`，并将其保存到代码示例目录（或您选择的任何位置）。

要从文件中读取，我们需要使用 Python 的内置函数：`open`来打开文件。让我们快速看一下一个代码片段，演示如何打开一个文本文件以读取其内容并将其打印到屏幕上：

```py
if __name__ == "__main__":
    # open text file to read
    file = open('read_line.txt', 'r')
    # read from file and store it to data
    data = file.read()
    print(data)
    file.close()
```

让我们详细讨论这段代码片段：

1.  读取文本文件内容的第一步是使用内置函数`open`打开文件。需要将所需的文件作为参数传递，并且还需要一个标志`r`，表示我们打开文件以读取内容（随着我们讨论每个读取/写入文件时，我们将讨论其他标志选项）。

1.  打开文件时，`open`函数返回一个指针（文件对象的地址），并将其存储在`file`变量中。

```py
       file = open('read_line.txt', 'r')
```

1.  这个文件指针用于读取文件的内容并将其打印到屏幕上：

```py
       data = file.read() 
       print(data)
```

1.  读取文件的内容后，通过调用`close()`函数关闭文件。

运行前面的代码片段（可与本章一起下载的`read_from_file.py`）使用 IDLE3 或命令行终端。文本文件的内容将如下打印到屏幕上：

```py
    I am learning Python Programming using the Raspberry Pi Zero
```

# 读取行

有时，有必要逐行读取文件的内容。在 Python 中，有两种选项可以做到这一点：`readline()`和`readlines()`：

+   `readline()`: 正如其名称所示，这个内置函数使得逐行读取成为可能。让我们通过一个例子来复习一下：

```py
       if __name__ == "__main__": 
          # open text file to read
          file = open('read_line.txt', 'r') 

          # read a line from the file
          data = file.readline() 
          print(data) 

          # read another line from the file 
          data = file.readline() 
          print(data) 

          file.close()
```

当执行前面的代码片段（可与本章一起下载，文件名为`read_line_from_file.py`）时，`read_line.txt`文件被打开，并且`readline()`函数返回一行。这一行被存储在变量 data 中。由于该函数在程序中被调用两次，输出如下：

```py
 I am learning Python Programming using the Raspberry Pi Zero. 

 This is the second line.
```

每次调用`readline`函数时都会返回一个新行，并且当到达文件结尾时会返回一个空字符串。

+   `readlines()`: 这个函数逐行读取文件的全部内容，并将每一行存储到一个列表中：

```py
       if __name__ == "__main__": 
           # open text file to read
           file = open('read_lines.txt', 'r') 

           # read a line from the file
           data = file.readlines() 
           for line in data: 
               print(line) 

           file.close()
```

由于文件的行被存储为一个列表，可以通过对列表进行迭代来检索它：

```py
       data = file.readlines() 
           for line in data: 
               print(line)
```

前面的代码片段可与本章一起下载，文件名为`read_lines_from_file.py`。

# 写入文件

按照以下步骤进行写入文件：

1.  写入文件的第一步是使用写入标志`w`打开文件。如果作为参数传递的文件名不存在，将创建一个新文件：

```py
      file = open('write_file.txt', 'w')
```

1.  文件打开后，下一步是将要写入的字符串作为参数传递给`write()`函数：

```py
      file.write('I am excited to learn Python using
      Raspberry Pi Zero')
```

1.  让我们将代码放在一起，我们将一个字符串写入文本文件，关闭它，重新打开文件并将文件的内容打印到屏幕上：

```py
       if __name__ == "__main__": 
          # open text file to write
          file = open('write_file.txt', 'w') 
          # write a line from the file
          file.write('I am excited to learn Python using
          Raspberry Pi Zero \n') 
          file.close() 

          file = open('write_file.txt', 'r') 
          data = file.read() 
          print(data) 
          file.close()
```

1.  前面的代码片段可与本章一起下载（`write_to_file.py`）。

1.  当执行前面的代码片段时，输出如下所示：

```py
       I am excited to learn Python using Raspberry Pi Zero
```

# 追加到文件

每当使用写入标志`w`打开文件时，文件的内容都会被删除，并重新打开以写入数据。还有一个叫做`a`的替代标志，它使得可以将数据追加到文件的末尾。如果打开的文件（作为打开的参数）不存在，这个标志也会创建一个新文件。让我们考虑下面的代码片段，我们将一行追加到上一节中的文本文件`write_file.txt`中：

```py
if __name__ == "__main__": 
   # open text file to append
   file = open('write_file.txt', 'a') 
   # append a line from the file
   file.write('This is a line appended to the file\n') 
   file.close() 

   file = open('write_file.txt', 'r') 
   data = file.read() 
   print(data) 
   file.close()
```

当执行前面的代码片段（可与本章一起下载的`append_to_file.py`）时，字符串`This is a line appended to the file`将被追加到文件的文本末尾。文件的内容将包括以下内容：

```py
    I am excited to learn Python using Raspberry Pi Zero
 This is a line appended to the file
```

# 寻找

一旦文件被打开，文件 I/O 中使用的文件指针会从文件的开头移动到文件的末尾。可以将指针移动到特定位置并从该位置读取数据。当我们对文件的特定行感兴趣时，这是非常有用的。让我们考虑上一个例子中的文本文件`write_file.txt`。文件的内容包括：

```py
    I am excited to learn Python using Raspberry Pi Zero
 This is a line appended to the file
```

让我们尝试跳过第一行，只读取第二行，使用`seek`：

```py
if __name__ == "__main__": 
   # open text file to read

   file = open('write_file.txt', 'r') 

   # read the second line from the file
   file.seek(53) 

   data = file.read() 
   print(data) 
   file.close()
```

在前面的例子中（可与本章一起下载的`seek_in_file.py`），`seek`函数用于将指针移动到字节`53`，即第一行的末尾。然后文件的内容被读取并存储到变量中。当执行这个代码片段时，输出如下所示：

```py
    This is a line appended to the file
```

因此，seek 使得移动文件指针到特定位置成为可能。

# 读取 n 个字节

`seek`函数使得将指针移动到特定位置并从该位置读取一个字节或`n`个字节成为可能。让我们重新阅读`write_file.txt`，并尝试读取句子`I am excited to learn Python using Raspberry Pi Zero`中的单词`excited`。

```py
if __name__ == "__main__": 
   # open text file to read and write 
   file = open('write_file.txt', 'r') 

   # set the pointer to the desired position 
   file.seek(5) 
   data = file.read(1) 
   print(data) 

   # rewind the pointer
   file.seek(5) 
   data = file.read(7) 
   print(data) 
   file.close()
```

前面的代码可以通过以下步骤来解释：

1.  第一步，使用`read`标志打开文件，并将文件指针设置为第五个字节（使用`seek`）——文本文件内容中字母`e`的位置。

1.  现在，我们通过将文件作为参数传递给`read`函数来从文件中读取一个字节。当整数作为参数传递时，`read`函数会从文件中返回相应数量的字节。当没有传递参数时，它会读取整个文件。如果文件为空，`read`函数会返回一个空字符串：

```py
       file.seek(5) 
       data = file.read(1) 
       print(data)
```

1.  在第二部分中，我们尝试从文本文件中读取单词`excited`。我们将指针的位置倒回到第五个字节。然后我们从文件中读取七个字节（单词`excited`的长度）。

1.  当执行代码片段时（可与本章一起下载的`seek_to_read.py`），程序应该打印字母`e`和单词`excited`：

```py
       file.seek(5) 
       data = file.read(7) 
       print(data)
```

# r+

我们讨论了使用`r`和`w`标志读取和写入文件。还有另一个叫做`r+`的标志。这个标志使得可以对文件进行读取和写入。让我们回顾一个例子，以便理解这个标志。

让我们再次回顾`write_file.txt`的内容：

```py
    I am excited to learn Python using Raspberry Pi Zero
 This is a line appended to the file
```

让我们修改第二行，改为：`This is a line that was modified`。代码示例可与本章一起下载（`seek_to_write.py`）。

```py
if __name__ == "__main__": 
   # open text file to read and write 
   file = open('write_file.txt', 'r+') 

   # set the pointer to the desired position 
   file.seek(68) 
   file.write('that was modified \n') 

   # rewind the pointer to the beginning of the file
   file.seek(0) 
   data = file.read() 
   print(data) 
   file.close()
```

让我们回顾一下这个例子是如何工作的：

1.  这个例子的第一步是使用`r+`标志打开文件。这使得可以对文件进行读取和写入。

1.  接下来是移动到文件的第 68 个字节

1.  在这个位置将`that was modified`字符串写入文件。字符串末尾的空格用于覆盖第二句原始内容。

1.  现在，文件指针已设置到文件的开头，并读取其内容。

1.  当执行前面的代码片段时，修改后的文件内容将打印到屏幕上，如下所示：

```py
       I am excited to learn Python using Raspberry Pi Zero
 This is a line that was modified
```

还有另一个`a+`标志，它可以使数据追加到文件末尾并同时进行读取。我们将留给读者使用到目前为止讨论的示例来弄清楚这一点。

我们已经讨论了 Python 中读取和写入文件的不同示例。如果没有足够的编程经验，可能会感到不知所措。我们强烈建议通过本章提供的不同代码示例进行实际操作。

# 读者的挑战

使用`a+`标志打开`write_file.txt`文件（在不同的示例中讨论），并向文件追加一行。使用`seek`设置文件指针并打印其内容。您可以在程序中只打开文件一次。

# 使用`with`关键字

到目前为止，我们讨论了可以用于以不同模式打开文件的不同标志。我们讨论的示例遵循一个常见模式——打开文件，执行读/写操作，然后关闭文件。有一种优雅的方式可以使用`with`关键字与文件交互。

如果在与文件交互的代码块执行过程中出现任何错误，`with`关键字会确保在退出代码块时关闭文件并清理相关资源。让我们通过一个示例来回顾`with`关键字：

```py
if __name__ == "__main__": 
   with open('write_file.txt', 'r+') as file: 
         # read the contents of the file and print to the screen 
         print(file.read()) 
         file.write("This is a line appended to the file") 

         #rewind the file and read its contents 
         file.seek(0) 
         print(file.read()) 
   # the file is automatically closed at this point 
   print("Exited the with keyword code block")
```

在前面的示例（`with_keyword_example`）中，我们跳过了关闭文件，因为`with`关键字在缩进的代码块执行完毕后会自动关闭文件。`with`关键字还会在由于错误离开代码块时关闭文件。这确保了资源在任何情况下都能得到适当的清理。接下来，我们将使用`with`关键字进行文件 I/O。

# configparser

让我们讨论一些在使用树莓派开发应用程序时特别有用的 Python 编程方面。其中一个工具是 Python 中提供的`configparser`。`configparser`模块（[`docs.python.org/3.4/library/configparser.html`](https://docs.python.org/3.4/library/configparser.html)）用于读取/写入应用程序的配置文件。

在软件开发中，配置文件通常用于存储常量，如访问凭据、设备 ID 等。在树莓派的上下文中，`configparser`可以用于存储所有使用的 GPIO 引脚列表，通过 I²C 接口接口的传感器地址等。让我们讨论三个示例，学习如何使用`configparser`模块。在第一个示例中，我们将使用`configparser`创建一个`config`文件。

在第二个示例中，我们将使用`configparser`来读取配置值，在第三个示例中，我们将讨论修改配置文件的最终示例。

**示例 1**：

在第一个示例中，让我们创建一个配置文件，其中包括设备 ID、使用的 GPIO 引脚、传感器接口地址、调试开关和访问凭据等信息：

```py
import configparser 

if __name__ == "__main__": 
   # initialize ConfigParser 
   config_parser = configparser.ConfigParser() 

   # Let's create a config file 
   with open('raspi.cfg', 'w') as config_file: 
         #Let's add a section called ApplicationInfo 
         config_parser.add_section('AppInfo') 

         #let's add config information under this section 
         config_parser.set('AppInfo', 'id', '123') 
         config_parser.set('AppInfo', 'gpio', '2') 
         config_parser.set('AppInfo', 'debug_switch', 'True') 
         config_parser.set('AppInfo', 'sensor_address', '0x62') 

         #Let's add another section for credentials 
         config_parser.add_section('Credentials') 
         config_parser.set('Credentials', 'token', 'abcxyz123') 
         config_parser.write(config_file) 
   print("Config File Creation Complete")
```

让我们详细讨论前面的代码示例（可与本章一起下载作为`config_parser_write.py`）：

1.  第一步是导入`configparser`模块并创建`ConfigParser`类的实例。这个实例将被称为`config_parser`：

```py
       config_parser = configparser.ConfigParser()
```

1.  现在，我们使用`with`关键字打开名为`raspi.cfg`的配置文件。由于文件不存在，将创建一个新的配置文件。

1.  配置文件将包括两个部分，即`AppInfo`和`Credentials`。

1.  可以使用`add_section`方法创建两个部分，如下所示：

```py
       config_parser.add_section('AppInfo') 
       config_parser.add_section('Credentials')
```

1.  每个部分将包含不同的常量集。可以使用`set`方法将每个常量添加到相关部分。`set`方法的必需参数包括参数/常量将位于的部分名称，参数/常量的名称及其对应的值。例如：`id`参数可以添加到`AppInfo`部分，并分配值`123`如下：

```py
       config_parser.set('AppInfo', 'id', '123')
```

1.  最后一步是将这些配置值保存到文件中。这是使用`config_parser`方法`write`完成的。一旦程序退出`with`关键字下的缩进块，文件就会关闭：

```py
       config_parser.write(config_file)
```

我们强烈建议尝试自己尝试代码片段，并将这些片段用作参考。通过犯错误，您将学到很多，并可能得出比这里讨论的更好的解决方案。

执行上述代码片段时，将创建一个名为`raspi.cfg`的配置文件。配置文件的内容将包括以下内容所示的内容：

```py
[AppInfo] 
id = 123 
gpio = 2 
debug_switch = True 
sensor_address = 0x62 

[Credentials] 
token = abcxyz123
```

**示例 2**：

让我们讨论一个示例，我们从先前示例中创建的配置文件中读取配置参数：

```py
import configparser 

if __name__ == "__main__": 
   # initialize ConfigParser 
   config_parser = configparser.ConfigParser() 

   # Let's read the config file 
   config_parser.read('raspi.cfg') 

   # Read config variables 
   device_id = config_parser.get('AppInfo', 'id') 
   debug_switch = config_parser.get('AppInfo', 'debug_switch') 
   sensor_address = config_parser.get('AppInfo', 'sensor_address') 

   # execute the code if the debug switch is true 
   if debug_switch == "True":
         print("The device id is " + device_id) 
         print("The sensor_address is " + sensor_address)
```

如果配置文件以所示格式创建，`ConfigParser`类应该能够解析它。实际上并不一定要使用 Python 程序创建配置文件。我们只是想展示以编程方式同时为多个设备创建配置文件更容易。

上述示例可与本章一起下载（`config_parser_read.py`）。让我们讨论一下这个代码示例是如何工作的：

1.  第一步是初始化名为`config_parser`的`ConfigParser`类的实例。

1.  第二步是使用实例方法`read`加载和读取配置文件。

1.  由于我们知道配置文件的结构，让我们继续阅读位于`AppInfo`部分下可用的一些常量。可以使用`get`方法读取配置文件参数。必需的参数包括配置参数所在的部分以及参数的名称。例如：配置`id`参数位于`AppInfo`部分下。因此，该方法的必需参数包括`AppInfo`和`id`：

```py
      device_id = config_parser.get('AppInfo', 'id')
```

1.  现在配置参数已读入变量中，让我们在程序中使用它。例如：让我们测试`debug_switch`变量（用于确定程序是否处于调试模式）并打印从文件中检索到的其他配置参数：

```py
       if debug_switch == "True":
           print("The device id is " + device_id) 
           print("The sensor_address is " + sensor_address)
```

**示例 3**：

让我们讨论一个示例，我们想要修改现有的配置文件。这在需要在执行固件更新后更新配置文件中的固件版本号时特别有用。

以下代码片段可与本章一起下载，文件名为`config_parser_modify.py`：

```py
import configparser 

if __name__ == "__main__": 
   # initialize ConfigParser 
   config_parser = configparser.ConfigParser() 

   # Let's read the config file 
   config_parser.read('raspi.cfg') 

   # Set firmware version 
   config_parser.set('AppInfo', 'fw_version', 'A3') 

   # write the updated config to the config file 
   with open('raspi.cfg', 'w') as config_file: 
       config_parser.write(config_file)
```

让我们讨论一下这是如何工作的：

1.  与往常一样，第一步是初始化`ConfigParser`类的实例。使用`read`方法加载配置文件：

```py
       # initialize ConfigParser 
       config_parser = configparser.ConfigParser() 

       # Let's read the config file 
       config_parser.read('raspi.cfg')
```

1.  使用`set`方法更新必需参数（在先前的示例中讨论）：

```py
       # Set firmware version 
       config_parser.set('AppInfo', 'fw_version', 'A3')
```

1.  使用`write`方法将更新后的配置保存到配置文件中：

```py
       with open('raspi.cfg', 'w') as config_file: 
          config_parser.write(config_file)
```

# 读者的挑战

使用示例 3 作为参考，将配置参数`debug_switch`更新为值`False`。重复示例 2，看看会发生什么。

# 读取/写入 CSV 文件

在本节中，我们将讨论读取/写入 CSV 文件。这个模块（[`docs.python.org/3.4/library/csv.html`](https://docs.python.org/3.4/library/csv.html)）在数据记录应用程序中非常有用。由于我们将在下一章讨论数据记录，让我们回顾一下读取/写入 CSV 文件。

# 写入 CSV 文件

让我们考虑一个场景，我们正在从不同的传感器读取数据。这些数据需要记录到一个 CSV 文件中，其中每一列对应于来自特定传感器的读数。我们将讨论一个例子，其中我们在 CSV 文件的第一行记录值`123`、`456`和`789`，第二行将包括值`Red`、`Green`和`Blue`：

1.  写入 CSV 文件的第一步是使用`with`关键字打开 CSV 文件：

```py
       with open("csv_example.csv", 'w') as csv_file:
```

1.  下一步是初始化 CSV 模块的`writer`类的实例：

```py
       csv_writer = csv.writer(csv_file)
```

1.  现在，通过创建一个包含需要添加到行中的所有元素的列表，将每一行添加到文件中。例如：第一行可以按如下方式添加到列表中：

```py
       csv_writer.writerow([123, 456, 789])
```

1.  将所有内容放在一起，我们有：

```py
       import csv 
       if __name__ == "__main__": 
          # initialize csv writer 
          with open("csv_example.csv", 'w') as csv_file: 
                csv_writer = csv.writer(csv_file) 
                csv_writer.writerow([123, 456, 789]) 
                csv_writer.writerow(["Red", "Green", "Blue"])
```

1.  当执行上述代码片段（与本章一起提供的`csv_write.py`可下载）时，在本地目录中创建了一个 CSV 文件，其中包含以下内容：

```py
 123,456,789
 Red,Green,Blue
```

# 从 CSV 文件中读取

让我们讨论一个例子，我们读取上一节中创建的 CSV 文件的内容：

1.  读取 CSV 文件的第一步是以读模式打开它：

```py
       with open("csv_example.csv", 'r') as csv_file:
```

1.  接下来，我们初始化 CSV 模块的`reader`类的实例。CSV 文件的内容被加载到对象`csv_reader`中：

```py
       csv_reader = csv.reader(csv_file)
```

1.  现在 CSV 文件的内容已加载，可以按如下方式检索 CSV 文件的每一行：

```py
       for row in csv_reader: 
           print(row)
```

1.  将所有内容放在一起：

```py
       import csv 

       if __name__ == "__main__": 
          # initialize csv writer 
          with open("csv_example.csv", 'r') as csv_file: 
                csv_reader = csv.reader(csv_file) 

                for row in csv_reader: 
                      print(row)
```

1.  当执行上述代码片段（与本章一起提供的`csv_read.py`可下载）时，文件的内容将逐行打印，其中每一行都是一个包含逗号分隔值的列表：

```py
       ['123', '456', '789']
 ['Red', 'Green', 'Blue']
```

# Python 实用程序

Python 带有几个实用程序，可以与其他文件和操作系统本身进行交互。我们已经确定了我们在过去项目中使用过的所有这些 Python 实用程序。让我们讨论不同的模块及其用途，因为我们可能会在本书的最终项目中使用它们。

# os 模块

正如其名称所示，这个模块（[`docs.python.org/3.1/library/os.html`](https://docs.python.org/3.1/library/os.html)）可以与操作系统进行交互。让我们通过示例讨论一些应用。

# 检查文件是否存在

`os`模块可用于检查特定目录中是否存在文件。例如：我们广泛使用了`write_file.txt`文件。在打开此文件进行读取或写入之前，我们可以检查文件是否存在：

```py
import os
if __name__ == "__main__":
    # Check if file exists
    if os.path.isfile('/home/pi/Desktop/code_samples/write_file.txt'):
        print('The file exists!')
    else:
        print('The file does not exist!')
```

在上述代码片段中，我们使用了`os.path`模块中提供的`isfile()`函数。当文件位置作为函数的参数传递时，如果文件存在于该位置，则返回`True`。在这个例子中，由于文件`write_file.txt`存在于代码示例目录中，该函数返回`True`。因此屏幕上打印出消息`文件存在`：

```py
if os.path.isfile('/home/pi/Desktop/code_samples/write_file.txt'): 
    print('The file exists!') 
else: 
    print('The file does not exist!')
```

# 检查文件夹是否存在

与`os.path.isfile()`类似，还有另一个名为`os.path.isdir()`的函数。如果特定位置存在文件夹，则返回`True`。我们一直在查看位于树莓派桌面上的名为`code_samples`的文件夹中的所有代码示例。可以通过以下方式确认其存在：

```py
# Confirm code_samples' existence 
if os.path.isdir('/home/pi/Desktop/code_samples'): 
    print('The directory exists!') 
else: 
    print('The directory does not exist!')
```

# 删除文件

`os`模块还可以使用`remove()`函数删除文件。将任何文件作为函数的参数传递即可删除该文件。在*文件 I/O*部分，我们讨论了使用文本文件`read_file.txt`从文件中读取。让我们通过将其作为`remove()`函数的参数来删除该文件：

```py
os.remove('/home/pi/Desktop/code_samples/read_file.txt')
```

# 终止进程

可以通过将进程`pid`传递给`kill()`函数来终止在树莓派上运行的应用程序。在上一章中，我们讨论了在树莓派上作为后台进程运行的`light_scheduler`示例。为了演示终止进程，我们将尝试终止该进程。我们需要确定`light_scheduler`进程的进程`pid`（您可以选择由您作为用户启动的应用程序，不要触及根进程）。可以使用以下命令从命令行终端检索进程`pid`：

```py
 ps aux
```

它会显示当前在树莓派上运行的进程（如下图所示）。`light_scheduler`应用程序的进程`pid`为 1815：

![](img/d74763d0-d5d1-4183-bfba-b820bf9e0784.png)light_scheduler 守护程序的 PID

假设我们知道需要终止的应用程序的进程`pid`，让我们回顾使用`kill()`函数终止该函数。终止函数所需的参数包括进程`pid`和需要发送到进程以终止应用程序的信号（`signal.SIGKILL`）：

```py
import os
import signal
if __name__ == "__main__":
    #kill the application
    try:
        os.kill(1815, signal.SIGKILL)
    except OSError as error:
        print("OS Error " + str(error))
```

`signal`模块（[`docs.python.org/3/library/signal.html)`](https://docs.python.org/2/library/signal.html)）包含表示可用于停止应用程序的信号的常量。在此代码片段中，我们使用了`SIGKILL`信号。尝试运行`ps`命令（`ps aux`），您会注意到`light_scheduler`应用程序已被终止。

# 监控一个进程

在前面的示例中，我们讨论了使用`kill()`函数终止应用程序。您可能已经注意到，我们使用了称为`try`/`except`关键字来尝试终止应用程序。我们将在下一章详细讨论这些关键字。

还可以使用`try`/`except`关键字使用`kill()`函数来监视应用程序是否正在运行。在介绍使用`try`/`except`关键字捕获异常的概念后，我们将讨论使用`kill()`函数监视进程。

`os`模块中讨论的所有示例都可以与本章一起下载，文件名为`os_utils.py`。

# glob 模块

`glob`模块（[`docs.python.org/3/library/glob.html`](https://docs.python.org/3/library/glob.html)）使得能够识别具有特定扩展名或特定模式的文件。例如，可以列出文件夹中的所有 Python 文件如下：

```py
# List all files
for file in glob.glob('*.py'):
    print(file)
```

`glob()`函数返回一个包含`.py`扩展名的文件列表。使用`for`循环来遍历列表并打印每个文件。当执行前面的代码片段时，输出包含属于本章的所有代码示例的列表（输出被截断以表示）：

```py
read_from_file.py
config_parser_read.py
append_to_file.py
read_line_from_file.py
config_parser_modify.py
python_utils.py
config_parser_write.py
csv_write.py
```

这个模块在列出具有特定模式的文件时特别有帮助。例如：让我们考虑这样一个场景，您想要上传来自实验不同试验的文件。您只对以下格式的文件感兴趣：`file1xx.txt`，其中`x`代表`0`到`9`之间的任意数字。这些文件可以按以下方式排序和列出：

```py
# List all files of the format 1xx.txt
for file in glob.glob('txt_files/file1[0-9][0-9].txt'):
    print(file)
```

在前面的示例中，`[0-9]`表示文件名可以包含`0`到`9`之间的任意数字。由于我们正在寻找`file1xx.txt`格式的文件，因此作为参数传递给`glob()`函数的搜索模式是`file1[0-9][0-9].txt`。

当执行前面的代码片段时，输出包含指定格式的所有文本文件：

```py
txt_files/file126.txt
txt_files/file125.txt
txt_files/file124.txt
txt_files/file123.txt
txt_files/file127.txt
```

我们找到了一篇解释使用表达式对文件进行排序的文章：[`www.linuxjournal.com/content/bash-extended-globbing`](http://www.linuxjournal.com/content/bash-extended-globbing)。相同的概念可以扩展到使用`glob`模块搜索文件。

# 读者的挑战

使用`glob`模块讨论的例子可以与本章一起下载，文件名为`glob_example.py`。在其中一个例子中，我们讨论了列出特定格式的文件。你将如何列出以下格式的文件：`filexxxx.*`？（这里的`x`代表`0`到`9`之间的任意数字。`*`代表任何文件扩展名。）

# shutil 模块

`shutil`模块（[`docs.python.org/3/library/shutil.html`](https://docs.python.org/3/library/shutil.html)）使得可以使用`move()`和`copy()`方法在文件夹之间移动和复制文件。在上一节中，我们列出了文件夹`txt_files`中的所有文本文件。让我们使用`move()`将这些文件移动到当前目录（代码执行的位置），再次在`txt_files`中复制这些文件，最后从当前目录中删除这些文本文件：

```py
import glob
import shutil
import os
if __name__ == "__main__":
    # move files to the current directory
    for file in glob.glob('txt_files/file1[0-9][0-9].txt'):
        shutil.move(file, '.')
    # make a copy of files in the folder 'txt_files' and delete them
    for file in glob.glob('file1[0-9][0-9].txt'):
        shutil.copy(file, 'txt_files')
        os.remove(file)
```

在前面的例子中（可以与本章一起下载，文件名为`shutil_example.py`），文件被移动和复制，源和目的地分别作为第一个和第二个参数指定。

使用`glob`模块识别要移动（或复制）的文件，然后使用它们对应的方法移动或复制每个文件。

# subprocess 模块

我们在上一章简要讨论了这个模块。`subprocess`模块（[`docs.python.org/3.2/library/subprocess.html`](https://docs.python.org/3.2/library/subprocess.html)）使得可以在 Python 程序内部启动另一个程序。`subprocess`模块中常用的函数之一是`Popen`。需要在程序内部启动的任何进程都需要作为列表参数传递给`Popen`函数：

```py
import subprocess
if __name__ == "__main__":
    subprocess.Popen(['aplay', 'tone.wav'])
```

在前面的例子中，`tone.wav`（需要播放的 WAVE 文件）和需要运行的命令作为列表参数传递给函数。`subprocess`模块中还有其他几个类似用途的命令。我们留给你去探索。

# sys 模块

`sys`模块（[`docs.python.org/3/library/sys.html`](https://docs.python.org/3/library/sys.html)）允许与 Python 运行时解释器进行交互。`sys`模块的一个功能是解析作为程序输入提供的命令行参数。让我们编写一个程序，读取并打印作为程序参数传递的文件的内容：

```py
import sys
if __name__ == "__main__":
    with open(sys.argv[1], 'r') as read_file:
        print(read_file.read())
```

尝试按以下方式运行前面的例子：

```py
python3 sys_example.py read_lines.txt
```

前面的例子可以与本章一起下载，文件名为`sys_example.py`。在运行程序时传递的命令行参数列表可以在`sys`模块的`argv`列表中找到。`argv[0]`通常是 Python 程序的名称，`argv[1]`通常是传递给函数的第一个参数。

当以`read_lines.txt`作为参数执行`sys_example.py`时，程序应该打印文本文件的内容：

```py
I am learning Python Programming using the Raspberry Pi Zero.
This is the second line.
Line 3.
Line 4.
Line 5.
Line 6.
Line 7.
```

# 总结

在本章中，我们讨论了文件 I/O - 读取和写入文件，以及用于读取、写入和追加文件的不同标志。我们谈到了将文件指针移动到文件的不同位置以检索特定内容或在特定位置覆盖文件内容。我们讨论了 Python 中的`ConfigParser`模块及其在存储/检索应用程序配置参数以及读写 CSV 文件中的应用。

最后，我们讨论了在我们的项目中潜在使用的不同 Python 工具。我们将广泛使用文件 I/O 和在本书中讨论的 Python 工具。我们强烈建议在进入本书中讨论的最终项目之前，熟悉本章讨论的概念。

在接下来的章节中，我们将讨论将存储在 CSV 文件中的传感器数据上传到云端，以及记录应用程序执行过程中遇到的错误。下一章见！
