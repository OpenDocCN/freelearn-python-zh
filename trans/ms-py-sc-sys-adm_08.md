# 第八章：文档和报告

在本章中，您将学习如何使用 Python 记录和报告信息。您还将学习如何使用 Python 脚本获取输入以及如何打印输出。在 Python 中编写接收电子邮件的脚本更容易。您将学习如何格式化信息。

在本章中，您将学习以下内容：

+   标准输入和输出

+   信息格式化

+   发送电子邮件

# 标准输入和输出

在本节中，我们将学习 Python 中的输入和输出。我们将学习`stdin`和`stdout`，以及`input()`函数。

`stdin`和`stdout`是类似文件的对象。这些对象由操作系统提供。每当用户在交互会话中运行程序时，`stdin`充当输入，`stdout`将是用户的终端。由于`stdin`是类似文件的对象，我们必须从`stdin`读取数据而不是在运行时读取数据。`stdout`用于输出。它用作表达式和`print()`函数的输出，以及`input()`函数的提示。

现在，我们将看一个`stdin`和`stdout`的例子。为此，请创建一个名为`stdin_stdout_example.py`的脚本，并在其中写入以下内容：

```py
import sys print("Enter number1: ") a = int(sys.stdin.readline()) print("Enter number2: ") b = int(sys.stdin.readline()) c = a + b sys.stdout.write("Result: %d " % c)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 stdin_stdout_example.py Enter number1: 10 Enter number2: 20 Result: 30
```

在上面的例子中，我们使用了`stdin`和`stdout`来获取输入和显示输出。`sys.stdin.readline()`将从`stdin`读取数据。将写入数据。

现在，我们将学习`input()`和`print()`函数。`input()`函数用于从用户那里获取输入。该函数有一个可选参数：提示字符串。

语法：

```py
 input(prompt)
```

`input()`函数返回一个字符串值。如果您想要一个数字值，只需在`input()`之前写入`int`关键字。您可以这样做：

```py
 int(input(prompt))
```

同样，您可以为浮点值写入`float`。现在，我们将看一个例子。创建一个`input_example.py`脚本，并在其中写入以下代码：

```py
str1 = input("Enter a string: ") print("Entered string is: ", str1) print() a = int(input("Enter the value of a: ")) b = int(input("Enter the value of b: ")) c = a + b print("Value of c is: ", c) print() num1 = float(input("Enter num 1: ")) num2 = float(input("Enter num 2: ")) num3 = num1/num2 print("Value of num 3 is: ", num3)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 input_example.py Output: Enter a string: Hello Entered string is:  Hello Enter the value of a: 10 Enter the value of b: 20 Value of c is:  30Enter num 1: 10.50 Enter num 2: 2.0 Value of num 3 is:  5.25
```

在上面的例子中，我们使用`input()`函数获取了三个不同的值。首先是字符串，第二个是整数值，第三个是`float`值。要将`input()`用于整数和浮点数，我们必须使用`int()`和`float()`类型转换函数将接收到的字符串转换为整数和浮点数。

现在，`print()`函数用于输出数据。我们必须输入一个以逗号分隔的参数列表。在`input_example.py`中，我们使用了`print()`函数来获取输出。使用`print()`函数，您可以通过将数据括在`""`或`''`中简单地将数据写入屏幕上。要仅访问值，只需在`print()`函数中写入变量名。如果您想在同一个`print()`函数中写一些文本并访问一个值，那么请用逗号将这两者分开。

我们将看一个`print()`函数的简单例子。创建一个`print_example.py`脚本，并在其中写入以下内容：

```py
# printing a simple string on the screen. print("Hello Python") # Accessing only a value. a = 80 print(a)  # printing a string on screen as well as accessing a value. a = 50 b = 30 c = a/b print("The value of c is: ", c)
```

运行脚本，您将得到以下输出：

```py
student@ubuntu:~/work$ python3 print_example.py Hello Python 80 The value of c is:  1.6666666666666667
```

在上面的例子中，首先我们简单地在屏幕上打印了一个字符串。接下来，我们只是访问了`a`的值并将其打印在屏幕上。最后，我们输入了`a`和`b`的值，然后将它们相加并将结果存储在变量`c`中，然后我们打印了一个语句并从同一个`print()`函数中访问了一个值。

# 信息格式化

在本节中，我们将学习字符串格式化。我们将学习两种格式化信息的方法：一种是使用字符串`format()`方法，另一种是使用`%`运算符。

首先，我们将学习使用字符串`format()`方法进行字符串格式化。`string`类的这种方法允许我们进行值格式化。它还允许我们进行变量替换。这将通过位置参数连接元素。

现在，我们将学习如何使用格式化程序进行格式化。调用此方法的字符串可以包含文字或由大括号`{}`分隔的替换字段。在格式化字符串时可以使用多对`{}`。此替换字段包含参数的索引或参数的名称。结果，您将获得一个字符串副本，其中每个替换字段都替换为参数的字符串值。

现在，我们将看一个字符串格式化的例子。

创建一个`format_example.py`脚本，并在其中写入以下内容：

```py
# Using single formatter print("{}, My name is John".format("Hi")) str1 = "This is John. I am learning {} scripting language." print(str1.format("Python")) print("Hi, My name is Sara and I am {} years old !!".format(26)) # Using multiple formatters str2 = "This is Mary {}. I work at {} Resource department. I am {} years old !!" print(str2.format("Jacobs", "Human", 30)) print("Hello {}, Nice to meet you. I am {}.".format("Emily", "Jennifer"))
```

按以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 format_example.py Output: Hi, My name is John This is John. I am learning Python scripting language. Hi, My name is Sara and I am 26 years old !! This is Mary Jacobs. I work at Human Resource department. I am 30 years old !! Hello Emily, Nice to meet you. I am Jennifer.
```

在前面的例子中，我们使用`string`类的`format()`方法进行了字符串格式化，使用了单个和多个格式化程序。

现在，我们将学习如何使用`％`运算符进行字符串格式化。`％`运算符与格式符一起使用。以下是一些常用的符号：

+   `％d`：十进制整数

+   `%s`：字符串

+   `％f`：浮点数

+   `％c`：字符

现在，我们将看一个例子。创建一个`string_formatting.py`脚本，并在其中写入以下内容：

```py
# Basic formatting a = 10 b = 30 print("The values of a and b are %d %d" % (a, b)) c = a + b print("The value of c is %d" % c) str1 = 'John' print("My name is %s" % str1)  x = 10.5 y = 33.5 z = x * y print("The value of z is %f" % z) print() # aligning name = 'Mary' print("Normal: Hello, I am %s !!" % name) print("Right aligned: Hello, I am %10s !!" % name) print("Left aligned: Hello, I am %-10s !!" % name) print() # truncating print("The truncated string is %.4s" % ('Examination')) print() # formatting placeholders students = {'Name' : 'John', 'Address' : 'New York'} print("Student details: Name:%(Name)s Address:%(Address)s" % students) 
```

运行脚本，您将获得以下输出：

```py
student@ubuntu:~/work$ python3 string_formatting.py The values of a and b are 10 30 The value of c is 40 My name is John The value of z is 351.750000Normal: Hello, I am Mary !! Right aligned: Hello, I am       Mary !! Left aligned: Hello, I am Mary       !!
The truncated string is Exam
Student details: Name:John Address:New York
```

在前面的例子中，我们使用`％`运算符来格式化字符串：`％d`表示数字，`％s`表示字符串，`％f`表示浮点数。然后，我们将字符串左对齐和右对齐。我们还学会了如何使用`％`运算符截断字符串。`％.4s`将仅显示前四个字符。接下来，我们创建了一个名为`students`的字典，并输入了`Name`和`Address`键值对。然后，我们在`％`运算符后放置了我们的键名以获取字符串。

# 发送电子邮件

在本节中，我们将学习如何通过 Python 脚本从 Gmail 发送电子邮件。为此，Python 有一个名为`smtplib`的模块。Python 中的`smtplib`模块提供了用于向具有 SMTP 侦听器的任何互联网机器发送电子邮件的 SMTP 客户端会话对象。

我们将看一个例子。在这个例子中，我们将从 Gmail 向接收者发送包含简单文本的电子邮件。

创建一个`send_email.py`脚本，并在其中写入以下内容：

```py
import smtplib from email.mime.text import MIMEText import getpass host_name = 'smtp.gmail.com' port = 465 u_name = 'username/emailid' password = getpass.getpass() sender = 'sender_name' receivers = ['receiver1_email_address', 'receiver2_email_address'] text = MIMEText('Test mail') text['Subject'] = 'Test' text['From'] = sender text['To'] = ', '.join(receivers) s_obj = smtplib.SMTP_SSL(host_name, port) s_obj.login(u_name, password) s_obj.sendmail(sender, receivers, text.as_string()) s_obj.quit() print("Mail sent successfully")
```

按以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 send_text.py
```

输出：

```py
Password: Mail sent successfully
```

在前面的例子中，我们从我们的 Gmail ID 向接收者发送了一封电子邮件。用户名变量将存储您的电子邮件 ID。在密码变量中，您可以输入密码，或者您可以使用`getpass`模块提示密码。在这里，我们提示输入密码。接下来，发件人变量将有您的名字。现在，我们将向多个接收者发送此电子邮件。然后，我们为该电子邮件包括了主题，发件人和收件人。然后在`login()`中，我们提到了我们的用户名和密码变量。接下来，在`sendmail()`中，我们提到了发件人，接收者和文本变量。因此，使用此过程，我们成功发送了电子邮件。

现在，我们将看一个发送带附件的电子邮件的例子。在这个例子中，我们将向收件人发送一张图片。我们将通过 Gmail 发送此邮件。创建一个`send_email_attachment.py`脚本，并在其中写入以下内容：

```py
import os import smtplib from email.mime.text import MIMEText from email.mime.image import MIMEImage from email.mime.multipart import MIMEMultipart import getpass host_name = 'smtp.gmail.com' port = 465  u_name = 'username/emailid' password = getpass.getpass() sender = 'sender_name' receivers = ['receiver1_email_address', 'receiver2_email_address'] text = MIMEMultipart() text['Subject'] = 'Test Attachment' text['From'] = sender text['To'] = ', '.join(receivers) txt = MIMEText('Sending a sample image.') text.attach(txt) f_path = '/home/student/Desktop/mountain.jpg' with open(f_path, 'rb') as f:
 img = MIMEImage(f.read()) img.add_header('Content-Disposition',
 'attachment', filename=os.path.basename(f_path)) text.attach(img) server = smtplib.SMTP_SSL(host_name, port) server.login(u_name, password) server.sendmail(sender, receivers, text.as_string()) print("Email with attachment sent successfully !!")server.quit()
```

按以下方式运行脚本：

```py
student@ubuntu:~/work$ python3 send_email_attachment.py
```

输出：

```py
Password: Email with attachment sent successfully!!
```

在前面的例子中，我们将图像作为附件发送给接收者。我们提到了发件人和收件人的电子邮件 ID。接下来，在`f_path`中，我们提到了我们发送为附件的图像的路径。接下来，我们将该图像作为附件发送给接收者。

在前面的两个例子——`send_text.py`和`send_email_attachment.py`——我们通过 Gmail 发送了电子邮件。您可以通过任何其他电子邮件提供商发送。要使用任何其他电子邮件提供商，只需在`host_name`中写入该提供商名称。不要忘记在其前面添加`smtp`。在这些示例中，我们使用了`smtp.gmail.com`；对于 Yahoo！您可以使用`smtp.mail.yahoo.com`。因此，您可以根据您的电子邮件提供商更改主机名以及端口。

# 摘要

在本章中，我们学习了标准输入和输出。我们了解了`stdin`和`stdout`分别作为键盘输入和用户终端。我们还学习了`input()`和`print()`函数。除此之外，我们还学习了如何从 Gmail 发送电子邮件给接收者。我们发送了一封包含简单文本的电子邮件，还发送了附件。此外，我们还学习了使用`format()`方法和`%`运算符进行字符串格式化。

在下一章中，您将学习如何处理不同类型的文件，如 PDF、Excel 和“csv”。

# 问题

1.  `stdin`和输入之间有什么区别？

1.  SMTP 是什么？

1.  以下内容的输出将是什么？

```py
>>> name = "Eric"
>>> profession = "comedian"
>>> affiliation = "Monty Python"
>>> age = 25
>>> message = (
...     f"Hi {name}. "
...     f"You are a {profession}. "
...     f"You were in {affiliation}."
... )
>>> message
```

1.  以下内容的输出将是什么？

```py
str1 = 'Hello'
str2 ='World!'
print('str1 + str2 = ', str1 + str2)
print('str1 * 3 =', str1 * 3)
```

# 进一步阅读

1.  `string`文档：[`docs.python.org/3.1/library/string.html`](https://docs.python.org/3.1/library/string.html)

1.  `smptplib`文档：[`docs.python.org/3/library/smtplib.html`](https://docs.python.org/3/library/smtplib.html)
