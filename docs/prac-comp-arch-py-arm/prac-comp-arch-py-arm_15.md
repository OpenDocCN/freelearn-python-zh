# 附录 – 关键概念摘要

在这些附录中，我们将简要总结本书中介绍的一些方面。我们将从介绍 IDLE 开始，这是 Python 解释器，它允许你快速开发程序并测试 Python 的功能。

第二个附录简要总结了在 Raspberry Pi 上开发 ARM 汇编语言程序时可能需要的某些 Linux 命令。

第三个附录提供了一个 ARM 汇编程序的运行和调试演示。这个示例的目的是将调试程序所需的所有步骤集中在一个地方。

第四个附录涵盖了可能让学生感到困惑的一些概念，例如计算机对“上”和“下”术语的使用，这些术语有时与“上”和“下”的正常含义不同。例如，向计算机堆栈添加内容会导致计算机堆栈向上增长到较低的地址。

最后一章附录定义了我们讨论计算机语言（如 Python）时使用的一些概念。

# 使用 IDLE

本书中的 Python 程序是用 Python 编写的，保存为`.py`文件，然后在集成开发环境中执行。然而，还有另一种执行 Python 的方法，你会在许多文本中看到提到。这是 Python IDLE 环境（包含在 Python 包中），它允许你逐行执行 Python 代码。

IDLE 是一个解释器，它读取输入的 Python 代码行，然后执行它。如果你只想测试几行代码而不想麻烦地创建源程序，这将非常有帮助。

考虑以下示例，其中粗体字体的文本是我的输入：

```py

Python 3.9.7 (tags/v3.9.7:1016ef3, Aug 30 2021, 20:19:38) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> x = 4
>>> y = 5
>>> print('Sum =', x+y)
Sum = 9
```

```py
>>>
```

当你运行编译后的 Python 程序时，输出将在运行窗口中显示。在这里，正如你所看到的，每个输入行在`>>>`提示符之后都被读取和解释，然后打印结果。

这个窗口实际上是 IDLE 环境的一部分。这意味着如果你的程序崩溃了，你可以在崩溃后检查变量。考虑以下示例，其中我们创建并运行了一个包含错误的程序：

```py

# An error example
x = 5
y = input('Type a number = ')
z = x + y
print('x + y =',z)
```

如果我们运行这个程序，执行窗口将显示以下消息：

```py

Type a number = 3
Traceback (most recent call last):
  File "E:/ArchitectureWithPython/IDE_Test.py", line 4, in <module>
    z = x + y
TypeError: unsupported operand type(s) for +: 'int' and 'str'
>>>
```

Python 解释器指示存在*类型错误*，因为我们输入了一个字符串并尝试将其添加到整数中。我们可以在显示窗口中继续操作，查看`x`和`y`变量，然后按如下方式修改代码。所有键盘输入都是粗体的：

```py

>>> x
5
>>> y
'3'
>>> y = int(y)
>>> z = x + y
>>> z
8
>>>
```

我们现在已经定位并修复了问题。当然，编辑原始的 Python 程序以纠正源代码是必要的。

由于 IDLE 一次执行一条语句，因此看起来无法执行循环，因为那需要多行代码。有一种方法。IDLE 自动缩进循环中的指令，这允许多个语句。为了完成（关闭）循环，你必须输入两个回车。考虑以下示例：

```py

>>> i = 0
>>> for j in range(0,5):  @ Here we've started a loop
      i = i + j*j         @ Add a statement. Note the automatic indentation
                          @ Now hit the enter key twice.
>>> print(i)               @ We have exited the loop and added a new statement
30                         @ And here is the result
>>>
```

# 指令和命令

此附录列出了您在 Raspberry Pi 上运行程序时将使用的一些常用命令：

Linux

```py

cd ..                     @ Change dictionary to parent
mkdir /home/pi/testProg   @ Create new file called testProg in folder pi
ls /home/pi               @ List files in folder pi
as -g -0 file.o file.s    @ Assemble source file file.s to create object file file.o
ld -0 file file.o         @ Link object file file.o
gdb file                  @ Call debugger to debug file
sudo apt-get update       @ Download packages in your configuration source files
sudo apt-get upgrade      @ Updates all installed packages
```

汇编指令

```py

.text                     @ This is a code section
.global _start            @ _start is a label (first instruction)
.word                     @ Bind 32-bit value to label and store in memory
.byte                     @ Bind 8-bit value to label and store in memory
.equ                      @ .equ x,7 binds or equates 7 to the name x
.asciz                    @ Bind ASCII string to label and store (terminated by 0)
.balign                   @ .balign 4 locates instruction/data is on a word boundary
```

gdb 调试器

```py

file toDebug              @ Load code file toDebug for debugging
b address                 @ Insert breakpoint at <address> (maybe line number or label)
x/4xw <address>           @ Display memory: four 32-bit words in hexadecimal format
x/7db <address>           @ Display memory: seven bytes in decimal format
r                         @ Run program (to a breakpoint or its termination)
s                         @ Step (execute) an instruction
n                         @ Same as step an instruction
i r                       @ Display registers
i b                       @ Display breakpoints
```

```py
c                         @ Continue from breakpoint
```

ARM 汇编语言程序的模板

```py

                     .text          @ Indicate this is code
        .global _start              @ Provide entry point
_start: mov   r0,#0                 @ Start of the code
        mov   r0,#0                 @ Exit parameter (optional)
        mov   r7,#goHome            @ Set up leave command
        svc   #0                    @ Call operating system to exit this code
test:   .word  0xABCD1234           @ Store a word in memory with label 'test'
        .equ   goHome, 1            @ Equate name to value
```

# 运行 ARM 程序

在这里，我们整理了所有您需要运行和调试 Raspberry Pi 上程序的信息。我们将从*第十一章*中的字符串复制示例开始，更详细地讲解，以提供一个程序开发的模板。此程序接受一个 ASCII 字符串并将其反转。在这种情况下，字符串是`"Hello!!!"`。我们将其设置为 8 个字符长，以便它适合两个连续的单词（8 * 8 位 = 64 位 = 2 个单词）。

我们在程序的`.text`部分找到了源字符串`string1`，因为它只被读取，从未被写入。

将接收反转字符串的目标`str2`位于`.data`段的读/写内存中。因此，我们必须使用间接指针技术——也就是说，.text 部分在`adr_str2`处有一个指针，它包含实际字符串`str2`的地址。

程序包含一些代码未访问的标签（例如`preLoop`和`Wait`）。这些标签的目的是在调试时通过给它们命名来简化断点的使用。

一个最终特性是使用*标记器*。我们在内存中插入了两个字符串之后的标记器——即`0xAAFFFFBB`和`0xCCFFFFCC`。这些标记使得在查看内存时更容易定位数据，因为它们很突出。

此程序测试基于指针的寻址、字节加载和存储，以及指针寄存器的自动递增和递减。我们将使用`gdb`的功能逐步执行此程序的执行：

```py

          .equ    len,8             @ Length of string to reverse (8 bytes/chars)
          .text                     @ Program (code) area
          .global _start            @
_start:   mov     r1,#len           @ Number of characters to move in r1
          adr     r2,string1        @ r2 points at source string1 in this section
          adr     r3,adr_str2       @ r3 points at dest string str2 address in this section
          ldr     r4,[r3]           @ r4 points to dest str2 in data section
preLoop:  add     r5,r4,#len-1      @ r5 points to bottom of dest str2
Loop:     ldrb    r6,[r2],#1        @ Get byte char from source in r6 inc pointer
          strb    r6,[r5],#-1       @ Store char in destination, decrement pointer
          subs    r1,r1,#1          @ Decrement char count
          bne     Loop              @ REPEAT until all done
Wait:     nop                       @ Stop here for testing

Exit:     mov     r0,#0             @ Stop here
          mov     r7,#1             @ Exit parameter required by svc
          svc     0                 @ Call operating system to exit program

string1:  .ascii  "Hello!!!"        @ The source string
marker:   .word   0xAAFFFFBB        @ Marker for testing

adr_str2: .word   str2              @ POINTER to source string2 in data area

          .data                     @ The read/write data area
str2:     .byte   0,0,0,0,0,0,0,0   @ Clear destination string
          .word   0xCCFFFFCC        @ Marker and terminator
          .end
```

程序被加载到`gdb`中进行调试，以下是调试步骤。注意，我的输入以粗体显示：

`alan@raspberrypi:~/Desktop $` gdb pLoop

第一步是在标签上放置三个断点，这样我们就可以执行代码直到这些点，然后检查寄存器或内存：

```py

(gdb) b _start
Breakpoint 1 at 0x10074: file pLoop.s, line 5.
(gdb) b preLoop
Breakpoint 2 at 0x10084: file pLoop.s, line 9.
(gdb) b Wait
Breakpoint 3 at 0x10098: file pLoop.s, line 14.
```

我们使用了`b <label>`三次来设置三个断点。我们可以使用`info b`命令来检查这些断点，该命令显示断点的状态：

```py

(gdb) info b
Num     Type           Disp Enb Address    What
1       breakpoint     keep y   0x00010074 pLoop.s:5
2       breakpoint     keep y   0x00010084 pLoop.s:9
```

```py
3       breakpoint     keep y   0x00010098 pLoop.s:14
```

下一步是运行程序直到第一条指令：

```py

(gdb) r
Starting program: /home/alan/Desktop/pLoop
Breakpoint 1, _start () at pLoop.s:5
5 _start:   mov     r1,#len            @ Number of characters to move in r1
(gdb) c
Continuing.
```

这里没有太多可看的内容。因此，我们按`c`继续到下一个断点，然后输入`i r`来显示寄存器。注意，我们没有显示未访问过的寄存器：

```py

Breakpoint 2, preLoop () at pLoop.s:9
9 preLoop:  add     r5,r4,#len-1       @ r5 points to bottom of dest str2
(gdb) i r
r0             0x0                 0
r1             0x8                 8
r2             0x100a8             65704      Pointer to string1
r3             0x100b4             65716      Pointer to str2 address
r4             0x200b8             131256     Pointer to str2 value
sp             0x7efff360          0x7efff360
lr             0x0                 0
pc             0x10084             0x10084 <preLoop>
```

让我们来看看代码中的数据部分。寄存器`r2`指向这个区域，命令表示从`0x100A8`开始显示以十六进制形式表示的四个内存字：

```py

(gdb) x/4wx 0x100a8
0x100a8 <string1>: 0x6c6c6548 0x2121216f 0xaaffffbb 0x000200b8
```

三个突出显示的值表示字符串`"Hello!!!"`和标记`0xCCFFFFCC`。注意这些值是如何从后往前出现的。这是小端字节序模式的结果。最不重要的字节位于单词的最不重要的末端。从 ASCII 字符的角度来看，这些是`lleH !!!o`。

我们接下来执行一个步骤，并显示数据区域的内存。在这个阶段，代码尚未完全执行，这个区域应该与最初设置的一样：

```py

(gdb) si 1
Loop () at pLoop.s:10
10 Loop:     ldrb    r6,[r2],#1        @ Get byte char from source in r6 inc pointer
(gdb) x/4wx 0x200b8
```

```py
0x200b8: 0x00000000 0x00000000 0xccffffcc 0x00001141
```

在这里，你可以看到加载在字节上的零和随后的标记。然后我们再次输入`c`，继续到`Wait`断点，此时代码应该已经完成。最后，我们查看寄存器和数据内存：

```py

(gdb) c
Continuing.
Breakpoint 3, Wait () at pLoop.s:14
14 Wait:     nop                       @ Stop here for testing

(gdb) i r
r0             0x0                 0
r1             0x0                 0
r2             0x100b0             65712
r3             0x100b4             65716
r4             0x200b8             131256
r5             0x200b7             131255
r6             0x21                33
sp             0x7efff360          0x7efff360
lr             0x0                 0
pc             0x10098             0x10098 <Wait>
(gdb) x/4wx 0x200b8
0x200b8: 0x6f212121 0x48656c6c 0xccffffcc 0x00001141
```

注意数据已经改变。正如你所看到的，顺序已经颠倒。再次注意小端字节序对单词内字节顺序的影响。现在数据的顺序是`o!!! Hell`。最后，我们再次输入`c`，程序完成：

```py

(gdb) c
Continuing.
[Inferior 1 (process 11670) exited normally]
(gdb)
```

# 常见混淆

从 20 世纪 60 年代到今天，计算机的发展迅速且混乱。这种混乱是由于技术发展得太快，以至于系统几个月内就过时了，这意味着大部分设计已经过时，但已经被纳入了现在正被其拖累的系统。同样，出现了许多不同的符号和约定——例如，`MOVE A,B`是将`A`移动到`B`，还是将`B`移动到`A`？不同的计算机同时使用了这两种约定。以下是一些有助于解决混淆的提示。

在这本书中，我们将主要采用从右到左的约定进行数据移动。例如，`add r1,r2,r2`表示将`r2`和`r3`相加，并将和放入`r1`。为了突出这一点，我经常将操作的源操作数用粗体字表示。

符号经常被赋予不同的含义。这一点在`#`、`@`和`%`上尤为明显。

+   `if x > y: z = 2`。如果`x`超过`y`，则使用`#`重置`z`。在 ARM 汇编语言中，`#`用于表示文本值——例如，`add r0,r1,#5`。`@`将整数`5`加到`r1`的内容中，并放入`r2`。

+   **@**：在 ARM 汇编语言中，`at`符号用于表示注释。

+   `add r1,r2,#%1010`表示以二进制形式表示的文本值。Python 使用前缀`0b`来表示二进制值（例如，0b1010）。

+   使用`0x`来表示十六进制值（例如，0xA10C）。

+   **寄存器间接寻址**：在汇编语言级别的编程中，一个关键概念是指针——即一个变量，它是内存中某个元素的地址。这种寻址模式被称为寄存器间接、基于指针或甚至索引寻址。

+   **上下**：在正常日常使用中，上下表示向天空（上）或向地面（下）的方向。在算术中，它们表示增加一个数字（上）或减少它（下）。在计算机中，当数据项被添加到栈中时，栈向上增长。然而，按照惯例，地址在添加项时向下增长。因此，当向栈中添加项时，栈指针会递减，当从栈中移除项时，栈指针会递增。

+   `ldr r0,=0x12345678`, 大端序的计算机会在字节内存中以递增的地址顺序存储字节 12,34,56,78，而小端序的计算机则会以 78,56,34,12 的顺序存储字节。ARM 是一个小端序机器，尽管它可以编程为在大端序模式下运行。在实践中，这意味着在调试程序和查看内存转储时必须小心。同样，在执行字值上的字节操作时，也必须小心，以确保选择正确的字节。

# 词汇

所有专业领域都有自己的词汇，编程也不例外。以下是一些有助于理解文本及其上下文的有用词汇。

+   `1`s 和 `0`s。人类用类似于英语的高级语言（如 Python）编写程序。在高级语言程序可以执行之前，一个叫做 *编译器* 的软件将其翻译成二进制代码。当你在电脑上运行 Python 程序时，你的源代码会由操作系统与编译器一起自动翻译成机器代码。幸运的是，你不必担心编译过程中在后台发生的所有无形操作。

+   `y = "4" + 1`。这是一个语法错误，因为我正在添加两个不能相加的不同实体。`"4"` 是一个字符（你可以打印它），而 `1` 是一个整数。你可以写 `y = "4" + "1"` 或 `z = 4 + 1`。这两个都是语法正确的，`y` 的值是 `"41"`，而 `z` 的值是 `5`。

+   **语义错误**：语义关注于意义。语法错误意味着句子在语法上是正确的，即使它在语法上是正确的。一个带有语义错误的英语句子示例是，“Twas brillig, and the slithy toves did gyre and gimble in the wabe。”这是语法正确的，但没有意义——也就是说，它是语义错误的。在计算机中，语义错误意味着你的程序没有按照你的意图执行。编译器可以检测语法错误，但通常不能检测语义错误。

+   `age = 25`，你创建了一个名为 `age` 的新变量，其值为 `25`。如果你引用 `age`，实际的值将被替换。表达式 `y = age + 10` 将会给 `y` 赋值为 `35`。一个变量有四个属性——它的名称（你如何称呼它）、它的地址（它在计算机中的存储位置）、它的值（它实际上是什么）和它的类型（例如，整数、列表、字符或字符串）。

+   `c = 2πr`，其中 `2` 和 `π` 是常数。`c` 和 `r` 都是变量。

+   `c` 及其半径 `r`。我们给无理数 `3.1415926` 赋予符号名 `π`。当程序编译成机器代码时，符号名会被实际值替换。

+   `1234`。程序员通常不必担心数据在内存中的实际位置。将程序中使用的逻辑地址转换为内存设备的物理地址是操作系统的领域。

+   `c = 2πr`，那么 `r` 是什么？我们（人类）将 `r` 视为半径值的符号名——比如说，5。但计算机将 `r` 视为内存地址 1234，必须读取它以提供 `r` 的实际值。如果我们写 `r = r + 1`，我们是想表示 `r = 5 + 1 = 6` 还是表示 `r = 1234 + 1 = 1235`？区分*地址*和其*内容*非常重要。当我们引入指针时，这个因素变得很重要。

+   `i` 实际上是一个指针；我们只是称它为*索引*。如果我们改变指针（索引），我们就可以遍历表、数组或矩阵的元素，并遍历元素 x1、x2、x3 和 x4。
