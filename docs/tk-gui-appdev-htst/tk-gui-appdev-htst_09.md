# 附录 B. 快速参考表

附录的大部分内容是从内置 Tkinter 文档生成的，因此相应地归 Python 软件基金会版权所有（版权所有 © 2001-2013 Python 软件基金会；版权所有）。

# 小部件共有选项

以下表格包含大多数小部件共有的选项、它们的职能以及不适用于这些选项的小部件列表：

| 小部件选项 | 功能 | 不适用于小部件 |
| --- | --- | --- |
| `background` (`bg`) | 选择背景颜色。 | （无内容） |
| `borderwidth` (`bd`) | 定义边框的像素宽度。 | （无内容） |
| `cursor` | 指定小部件使用的鼠标光标。 | （无内容） |
| `relief` | 指定小部件的边框样式。 | （无内容） |
| `takefocus` | 如果窗口在键盘遍历期间接受焦点。 | （无内容） |
| `width` | 一个整数，指定小部件的相对宽度。 | 菜单 |
| `font` | 指定字体家族和字体大小。 | 最顶层、画布、框架和滚动条 |
| `foreground` (`fg`) | 指定前景颜色。 | 最顶层、画布、框架和滚动条 |
| `highlightbackground` | 颜色 | 菜单 |
| `highlightcolor` | 颜色 | 菜单 |
| `highlightthickness` | 以像素为单位进行测量。 | 菜单 |
| `relief` | 指定应用于给定小部件的 3D 效果。有效值有 `RAISED`、`SUNKEN`、`FLAT`、`RIDGE`、`SOLID` 和 `GROOVE`。 | （无内容） |
| `takefocus` | 指定小部件在键盘标签遍历期间是否接受焦点。 | （无内容） |
| `width` | 指定小部件宽度的整数。 | 菜单 |

以下表格包含大多数小部件共有的选项、它们的职能以及适用于这些选项的小部件列表：

| 小部件选项 | 功能 | 适用于 |
| --- | --- | --- |
| `activebackground` | 当小部件处于活动状态时的背景颜色。 | 菜单、菜单按钮、按钮、复选框、单选按钮、刻度和滚动条 |
| `activeforeground` | 当小部件处于活动状态时的前景颜色。 | 菜单、菜单按钮、按钮、复选框和单选按钮 |
| `anchor` | 指示文本或位图在窗口组件上显示的位置。有效值有 `n`、`ne`、`e`、`se`、`s`、`sw`、`w`、`nw` 或 `center`。 | 菜单按钮、按钮、复选框、单选按钮、标签和信息框 |
| `bitmap` | 指示在窗口组件中显示的位图。 | 菜单按钮、按钮、复选框、单选按钮和标签 |
| `command` | 指示与窗口关联的命令回调，通常在窗口上鼠标按钮 1 释放时调用。 | 按钮、复选框、单选按钮、刻度和滚动条 |
| `disabledforeground` | 当小部件处于禁用状态时显示的前景色。 | 菜单、菜单按钮、按钮、复选框和单选按钮 |
| `height` | 表示小部件的高度，以给定小部件指定的字体单位表示。 | 最顶层、菜单按钮、按钮、复选框、单选按钮、标签、框架、列表框和画布 |
| `image` | 表示在部件中显示的图像。 | Menubutton、Button、Checkbutton、Radiobutton 和 Label |
| `justify` | 当在部件中显示多行文本时适用。这决定了文本行之间的对齐方式。必须是 LEFT、CENTER 或 RIGHT 之一。 | Menubutton、Button、Checkbutton、Radiobutton、Label、Entry 和 Message |
| `selectbackground` | 表示显示选中项时要显示的背景颜色。 | Text、Listbox、Entry 和 Canvas |
| `selectborderwidth` | 表示显示选中项时要显示的边框宽度。 | Text、Listbox、Entry 和 Canvas |
| `selectforeground` | 表示显示选中项时要显示的前景色。 | Text、Listbox、Entry 和 Canvas |
| `state` | 表示部件可能处于的两个或三个状态之一。有效值 `normal`、`active` 或 `disabled`。 | Menubutton、Button、Checkbutton、Radiobutton、Text、Entry 和 Scale |
| `text` | 表示要在部件内显示的字符串。 | Menubutton、Button、Checkbutton、Radiobutton、Label 和 Message |
| `textvariable` | 表示变量的名称。变量的值被更改为字符串以便在部件中显示。当变量值改变时，部件会自动更新。 | Menubutton、Button、Checkbutton、Radiobutton、Label、Entry 和 Message |
| `underline` | 表示在部件中下划线的字符的整数索引。 | Menubutton、Button、Checkbutton、Radiobutton 和 Label |
| `wraplength` | 表示具有单词换行的部件的最大行长度。 | Menubutton、Button、Checkbutton、Radiobutton 和 Label |

# 部件特定选项

我们并没有列出所有部件特定选项。您可以在 Python 交互式外壳中使用 help 命令获取给定部件的所有可用选项。

要获取任何 `Tkinter` 类的帮助，首先将 Tkinter 导入命名空间，如下所示：

```py
>>>import Tkinter

```

然后，可以使用以下命令获取特定部件的信息：

| Widget Name | 获取帮助 |
| --- | --- |
| Label | `help(Tkinter.Label)` |
| Button | `help(Tkinter.Button)` |
| Canvas | `help(Tkinter.Canvas)` |
| CheckButton | `help(Tkinter.Checkbutton)` |
| Entry | `help(Tkinter.Entry)` |
| Frame | `help(Tkinter.Frame)` |
| LabelFrame | `help(Tkinter.LabelFrame)` |
| Listbox | `help(Tkinter.Listbox)` |
| Menu | `help(Tkinter.Menu)` |
| Menubutton | `help(Tkinter.Menubutton)` |
| Message | `help(Tkinter.Message)` |
| OptionMenu | `help(Tkinter.OptionMenu)` |
| PanedWindow | `help(Tkinter.PanedWindow)` |
| RadioButton | `help(Tkinter.Radiobutton)` |
| Scale | `help(Tkinter.Scale)` |
| Scrollbar | `help(Tkinter.Scrollbar)` |
| Spinbox | `help(Tkinter.Spinbox)` |
| Text | `help(Tkinter.Text)` |
| Bitmap Class | `help(Tkinter.BitmapImage)` |
| Image Class | `help(Tkinter.Image)` |

# 包管理器

Pack 几何管理器是 Tk 和 Tkinter 中可用的最古老的几何管理器。Pack 几何管理器将子小部件放置在主小部件中，按照子小部件被引入的顺序逐个添加。下表显示了可用的`pack()`方法和选项：

| 方法 | 描述 |
| --- | --- |

| `config = configure = pack_configure(self, cnf={}, **kw)` | 在父小部件中打包小部件。使用以下选项：

+   `after=widget`: 在打包小部件后打包它

+   `anchor=NSEW` (或子集): 根据给定方向定位小部件

+   `before=widget`: 在打包小部件之前打包它

+   `expand=bool`: 如果父小部件大小增加，则扩展小部件

+   `fill=NONE`（或`X`、`Y`或`BOTH`）：如果小部件增长，则填充小部件

+   `in=master`: 使用主小部件包含此小部件

+   `in_=master`: 查看`in`选项描述

+   `ipadx=amount`: 在 x 方向添加内部填充

+   `ipady=amount`: 在 y 方向添加内部填充

+   `padx=amount`: 在 x 方向添加填充

+   `pady=amount`: 在 y 方向添加填充

+   `side=TOP`（或`BOTTOM`、`LEFT`或`RIGHT`）：添加此小部件的位置

|

| `forget = pack_forget(self)` | 解除映射此小部件，并不要用于打包顺序。 |
| --- | --- |
| `info = pack_info(self)` | 返回此小部件的打包选项信息。 |
| `propagate =pack_propagate(self, flag=['_noarg_']) from Tkinter.Misc` | 设置或获取几何信息传播的状态。布尔参数指定子小部件的几何信息是否将决定此小部件的大小。如果没有提供参数，则返回当前设置。 |
| `slaves = pack_slaves(self) from Tkinter.Misc` | 返回此小部件所有子小部件的打包顺序列表。 |

# 网格管理器

网格易于实现且易于修改，使其成为大多数用例中最受欢迎的选择。以下是使用`grid()`几何管理器进行布局管理可用的方法和选项列表：

| 在此处定义的方法 | 描述 |
| --- | --- |
| `bbox = grid_bbox(self, column=None, row=None, col2=None, row2=None) from Tkinter.Misc` | 返回由几何管理器网格控制的小部件边界框的整数坐标元组。如果提供了`column`和`row`，则边界框适用于从行和列 0 的单元格到指定的单元格。如果提供了`col2`和`row2`，则边界框从该单元格开始。返回的整数指定了主小部件中上左角的偏移量以及宽度和高度。 |
| `columnconfigure = grid_columnconfigure(self, index, cnf={}, **kw) from Tkinter.Misc` | 配置网格的`index`列。有效资源包括 minsize（列的最小大小）、weight（额外空间传播到该列的程度）和 pad（额外空间量）。 |

| `grid = config = configure = grid_configure(self, cnf={}, **kw)` | 在父小部件中按网格定位小部件。使用以下选项：

+   `column=number`: 使用给定列标识的单元格（从 0 开始）

+   `columnspan=number`: 此部件将跨越多个列

+   `in=master`: 使用主部件包含此部件

+   `in_=master`: 查看'in'选项描述

+   `ipadx=amount`: 在 x 方向添加内部填充

+   `ipady=amount`: 在 y 方向添加内部填充

+   `padx=amount`: 在 x 方向添加填充

+   `pady=amount`: 在 y 方向添加填充

+   `row=number`: 使用给定行标识的单元格（从 0 开始）

+   `rowspan=number`: 此部件将跨越多个行

+   `sticky=NSEW`: 如果单元格在哪个方向上更大，则此部件将粘附到单元格边界

|

| `forget = grid_forget(self)` | 取消映射此部件。 |
| --- | --- |
| `info = grid_info(self)` | 返回有关在此网格中定位此部件的选项的信息。 |
| `grid_location(self, x, y) from Tkinter.Misc` | 返回一个元组，表示列和行，这些列和行标识了主部件内 X 和 Y 位置像素所在的单元格。 |
| `grid_propagate(self, flag=['_noarg_']) from Tkinter.Misc` | 设置或获取几何信息传播的状态。一个布尔参数指定是否由从属部件的几何信息确定此部件的大小。如果没有给出参数，将返回当前设置。 |
| `grid_remove(self)` | 取消映射此部件，但记住网格选项。 |
| `grid_rowconfigure(self, index, cnf={}, **kw) from Tkinter.Misc` | 配置网格的`index`行。有效的资源有 minsize（行的最小大小）、weight（额外空间传播到本行的程度）和 pad（额外允许的空间）。 |
| `size = grid_size(self) from Tkinter.Misc` | 返回网格中列和行的数量 |
| `slaves = grid_slaves(self, row=None, column=None) from Tkinter.Misc` | 返回此部件在其打包顺序中的所有从属部件的列表。 |
| `location = grid_location(self, x, y) from Tkinter.Misc` | 返回一个元组，表示列和行，这些列和行标识了主部件内 X 和 Y 位置像素所在的单元格。 |
| `propagate = grid_propagate(self, flag=['_noarg_']) from Tkinter.Misc` | 设置或获取几何信息传播的状态。一个布尔参数指定是否由从属部件的几何信息确定此部件的大小。如果没有给出参数，将返回当前设置。 |
| `rowconfigure = grid_rowconfigure(self, index, cnf={}, **kw) from Tkinter.Misc` | 配置网格的`INDEX`行。有效的资源有 minsize（行的最小大小）、weight（额外空间传播到本行的程度）和 pad（额外允许的空间）。 |

# 位置管理器

`place()`几何管理器允许基于给定窗口的绝对或相对坐标精确定位部件。以下表格列出了在位置几何管理器下可用的方法和选项：

| 在此处定义的方法 | 描述 |
| --- | --- |

| `config = configure = place_configure(self, cnf={}, **kw)` | 在父小部件中放置一个小部件。使用以下选项：

+   `in=master`: 小部件放置的主控件相对位置

+   `in_=master`: 参见 'in' 选项描述

+   `x=amount`: 在主控件的 x 位置定位此小部件的锚点

+   `y=amount`: 在主控件的 y 位置定位此小部件的锚点

+   `relx=amount`: 在主控件的宽度范围内（0.0 到 1.0）定位此小部件的锚点（1.0 是右边缘）

+   rely=amount: 在主控件的 0.0 到 1.0 之间定位此小部件的锚点（1.0 是底部边缘）

+   `anchor=NSEW`（或子集）：根据给定方向定位锚点

+   `width=amount`: 此小部件的宽度（以像素为单位）

+   `height=amount`: 此小部件的高度（以像素为单位）

+   `relwidth=amount`: 此小部件的宽度，相对于主控件的 0.0 到 1.0（1.0 与主控件相同宽度）

+   `relheight=amount`: 此小部件的高度，相对于主控件的 0.0 到 1.0（1.0 与主控件相同高度）

+   `bordermode="inside"`（或`"outside"`）：是否考虑主小部件的边框宽度

|

| `forget = place_forget(self)` | 取消映射此小部件。 |
| --- | --- |
| `info = place_info(self)` | 返回此小部件放置选项的信息。 |
| `slaves = place_slaves(self) from Tkinter.Misc` | 返回此小部件所有子部件的列表，按其打包顺序排列。 |

# 事件类型

表示事件的通用格式如下：

```py
<[event modifier-]...event type [-event detail]>

```

对于任何事件绑定，必须指定事件类型。此外，请注意，事件类型、事件修饰符和事件细节在不同平台上可能有所不同。以下表格表示事件类型及其描述：

| 事件类型 | 描述 |
| --- | --- |
| Activate | 小部件的状态选项从非活动状态（灰色）变为活动状态。 |
| Button | 鼠标按钮按下。事件详细部分指定了哪个按钮。 |
| ButtonRelease | 按下的鼠标按钮释放。 |
| Configure | 小部件大小的更改。 |
| Deactivate | 小部件的状态选项从活动状态变为非活动状态（灰色）。 |
| Destroy | 使用 `widget.destroy` 方法销毁小部件。 |
| Enter | 鼠标指针进入小部件的可见部分。 |
| Expose | 小部件至少有一部分在另一个窗口覆盖后变得可见。 |
| FocusIn | 小部件由于用户事件（如使用*Tab*键或鼠标点击）或在小部件上调用`.focus_set()`而获得输入焦点 |
| FocusOut | 焦点从小部件中移出。 |
| KeyPress/Key | 按键盘中按键。事件详细部分指定了哪个键。 |
| KeyRelease | 按下的键释放。 |
| Leave | 鼠标指针从小部件中移出。 |
| 地图 | 小部件被映射（显示）。例如，当你对一个小部件调用几何管理器时发生。 |
| Motion | 鼠标指针完全在小部件内移动。 |
| Un-map | 小部件取消映射（变为不可见）。例如，当你使用`remove()`方法时。 |
| 可见性 | 窗口的一部分变为可见。 |

# 事件修饰符

事件修饰符是创建事件绑定时的一个可选组件。以下列出了事件修饰符列表。然而，请注意，大多数事件修饰符是平台特定的，可能不会在所有平台上工作。

| 修饰符 | 描述 |
| --- | --- |
| Alt | 当按下 *Alt* 键时为真。 |
| 任何 | 通用事件类型。例如 `<Any-KeyPress>` 在任何键被按下时为真。 |
| 控制 | 当按下 *Ctrl* (控制) 键时为真。 |
| 双击 | 指定两个事件快速连续发生。例如，`<Double-Button-1>` 是鼠标按钮 1 的双击。 |
| 锁定 | 如果按下 *Caps Lock*/*Shift* 锁则为真 |
| Shift | 如果按下 *Shift* 键则为真 |
| 三击 | 与双击类似（三个事件快速连续发生） |

# 事件详情

事件详情是创建事件绑定时的一个可选组件。它们通常使用缩写为 **keysym** 的键符号表示鼠标按钮或键盘按键的详细信息。

| 所有可用事件详情列表如下:.keysym | .keycode | .keysym_num | 键 |
| --- | --- | --- | --- |
| `Alt_L` | 64 | 65513 | 左 *Alt* 键 |
| `Alt_R` | 113 | 65514 | 右 *Alt* 键 |
| `BackSpace` | 22 | 65288 | *Backspace* |
| `Cancel` | 110 | 65387 | 断开 |
| `Caps_Lock` | 66 | 65549 | *CapsLock* |
| `Control_L` | 37 | 65507 | 左 *Ctrl* 键 |
| `Control_R` | 109 | 65508 | 右 *Ctrl* 键 |
| `Delete` | 107 | 65535 | *Delete* |
| `Down` | 104 | 65364 | 向下箭头键 |
| `End` | 103 | 65367 | *End* |
| `Escape` | 9 | 65307 | *Esc* |
| `Execute` | 111 | 65378 | *SysRq* |
| `F1 – F11` | 67 to 95 | 65470 to 65480 | 功能键 *F1* 到 *F11* |
| `F12` | 96 | 65481 | 功能键 *F12* |
| `Home` | 97 | 65360 | *Home* |
| `Insert` | 106 | 65379 | *Insert* |
| `Left` | 100 | 65361 | 左侧箭头键 |
| `Linefeed` | 54 | 106 | 换行符/*Ctrl* + *J* |
| `KP_0` | 90 | 65438 | 键盘上的 *0* |
| `KP_1` | 87 | 65436 | 键盘上的 *1* |
| `KP_2` | 88 | 65433 | 键盘上的 *2* |
| `KP_3` | 89 | 65435 | 键盘上的 *3* |
| `KP_4` | 83 | 65430 | 键盘上的 *4* |
| `KP_5` | 84 | 65437 | 键盘上的 *5* |
| `KP_6` | 85 | 65432 | 键盘上的 *6* |
| `KP_7` | 79 | 65429 | 键盘上的 *7* |
| `KP_8` | 80 | 65431 | 键盘上的 *8* |
| `KP_9` | 81 | 65434 | 键盘上的 *9* |
| `KP_Add` | 86 | 65451 | 键盘上的 *+* |
| `KP_Begin` | 84 | 65437 | 键盘上的中心键（与键 *5* 相同） |
| `KP_Decimal` | 91 | 65439 | 键盘上的小数 (*.*) 键 |
| `KP_Delete` | 91 | 65439 | 键盘上的 Delete (*Del*) 键 |
| `KP_Divide` | 112 | 65455 | 键盘上的 */* |
| `KP_Down` | 88 | 65433 | 键盘上的向下箭头键 |
| `KP_End` | 87 | 65436 | 键盘上的 *End* |
| `KP_Enter` | 108 | 65421 | 键盘上的 *Enter* |
| `KP_Home` | 79 | 65429 | 键盘上的 *Home* |
| `KP_Insert` | 90 | 65438 | 键盘上的 *Insert* |
| `KP_Left` | 83 | 65430 | 键盘上的左箭头键 |
| `KP_Multiply` | 63 | 65450 | 键盘上的 *** |
| `KP_Next` | 89 | 65435 | 键盘上的 *Page Down* |
| `KP_Prior` | 81 | 65434 | 键盘上的 *Page Up* |
| `KP_Right` | 85 | 65432 | 键盘上的右箭头键 |
| `KP_Subtract` | 82 | 65453 | 键盘上的 *-* |
| `KP_Up` | 80 | 65431 | 键盘上的向上箭头键 |
| `Next` | 105 | 65366 | *Page Down* |
| `Num_Lock` | 77 | 65407 | *Num Lock* |
| `Pause` | 110 | 65299 | *Pause* |
| `Print` | 111 | 65377 | *Prt Scr* |
| `Prior` | 99 | 65365 | *Page Up* |
| `Return` | 36 | 65293 | *Enter* 键 / *Ctrl* + *M* |
| `Right` | 102 | 65363 | 右箭头键 |
| `Scroll_Lock` | 78 | 65300 | *Scroll Lock* |
| `Shift_L` | 50 | 65505 | 左 *Shift* 键 |
| `Shift_R` | 62 | 65506 | 右 *Shift* 键 |
| `Tab` | 23 | 65289 | *Tab* 键 |
| `Up` | 98 | 65362 | 向上箭头键 |

# 其他与事件相关的方法

使用 `bind`、`bind_all`、`bind_class` 和 `tag_bind` 可以在各个级别将处理程序绑定到事件。

如果将事件绑定注册到回调函数，则回调函数将使用事件作为其第一个参数被调用。事件参数具有以下属性：

| 属性 | 描述 | 适用于事件类型 |
| --- | --- | --- |
| `event.serial` | 事件的序列号。 | 所有 |
| `event.num` | 按下的鼠标按钮。 | `ButtonPress` 和 `ButtonRelease` |
| `event.focus` | 窗口是否有焦点。 | `Enter` 和 `Leave` |
| `event.height` | 暴露窗口的高度。 | `Configure` 和 `Expose` |
| `event.width` | 暴露窗口的宽度。 | `Configure` 和 `Expose` |
| `event.keycode` | 按下的键的键码。 | `KeyPress` 和 `KeyRelease` |
| `event.state` | 事件的状态作为数字。 | `ButtonPress`、`ButtonRelease`、`Enter`、`KeyPress`、`KeyRelease`、`Leave` 和 `Motion` |
| `event.state` | 状态作为字符串。 | `Visibility` |
| `event.time` | 事件发生的时间。 | 所有 |
| `event.x` | 给出鼠标的 x 位置。 | 所有 |
| `event.y` | 给出鼠标的 y 位置。 | 所有 |
| `event.x_root` | 给出鼠标在屏幕上的 x 位置。 | `ButtonPress`、`ButtonRelease`、`KeyPress`、`KeyRelease` 和 `Motion` |
| `event.y_root` | 给出鼠标在屏幕上的 y 位置。 | `ButtonPress`、`ButtonRelease`、`KeyPress`、`KeyRelease` 和 `Motion` |
| `event.char` | 给出按下的字符。 | `KeyPress` 和 `KeyRelease` |
| `event.keysym` | 以字符串形式给出事件的 `keysym`。 | `KeyPress` 和 `KeyRelease` |
| `event.keysym_num` | 给出事件的 `keysym` 作为数字。 | `KeyPress` 和 `KeyRelease` |
| `event.type` | 事件类型作为数字。 | 所有 |
| `event.widget` | 发生事件的窗口小部件。 | 所有 |
| `event.delta` | 滚轮移动的增量。 | `MouseWheel` |

# 可用光标列表

光标小部件选项允许 Tk 程序员更改特定小部件的鼠标光标。Tk 在所有平台上识别的光标名称是：

| `X_cursor` | `arrow` | `based_arrow_down` | `based_arrow_up` | `boat` | `bogosity` |
| --- | --- | --- | --- | --- | --- |
| `bottom_left_corner` | `bottom_right_corner` | `bottom_side` | `box_spiral` | `center_ptr` | `circle` |
| `clock` | `coffee_mug` | `cross` | `cross_reverse` | `crosshair` | `diamond_cross` |
| `dot` | `dotbox` | `double_arrow` | `draft_large` | `draft_small` | `draped_box` |
| `exchange` | `fleur` | `gobbler` | `gumby` | `hand1` | `hand2` |
| `heart` | `icon` | `iron_cross` | `left_ptr` | `left_side` | `left_tee` |
| `leftbutton` | `ll_angle` | `lr_angle` | `man` | `bottom_tee` | `middlebutton` |
| `mouse` | `pencil` | `pirate` | `plus` | `question_arrow` | `right_ptr` |
| `right_side` | `right_tee` | `rightbutton` | `rtl_logo` | `sailboat` | `sb_down_arrow` |
| `sb_h_double_arrow` | `sb_left_arrow` | `sb_right_arrow` | `sb_up_arrow` | `sb_v_double_arrow` | `shuttle` |
| `sizing` | `spider` | `spraycan` | `star` | `target` | `tcross` |
| `top_left_arrow` | `top_left_corner` | `top_right_corner` | `top_side` | `top_tee` | `trek` |
| `ul_angle` | `umbrella` | `ur_angle` | `watch` | `xterm` |   |

*查看 9.01 所有光标演示.py* 以演示所有跨平台光标。

## 可携带性问题

+   Windows：在 Windows 上有原生映射的光标有，`arrow`，`center_ptr`，`crosshair`，`fleur`，`ibeam`，`icon`，`sb_h_double_arrow`，`sb_v_double_arrow`，`watch`，和`xterm`。

    可用的以下附加光标有，`no`，`starting`，`size`，`size_ne_sw`，`size_ns`，`size_nw_se`，`size_we`，`uparrow`，`wait`。

    可以指定`no`光标来消除光标。

+   Mac OS X：在 Mac OS X 系统上有原生映射的光标有，`arrow`，`cross`，`crosshair`，`ibeam`，`plus`，`watch`，`xterm`。

    可用的以下附加原生光标有，`copyarrow`，`aliasarrow`，`contextualmenuarrow`，`text`，`cross-hair`，`closedhand`，`openhand`，`pointinghand`，`resizeleft`，`resizeright`，`resizeleftright`，`resizeup`，`resizedown`，`resizeupdown`，`none`，`notallowed`，`poof`，`countinguphand`，`countingdownhand`，`countingupanddownhand`，`spinning`。

# 基本小部件方法

这些方法在 Tkinter 模块的 Widget 类下提供。您可以使用以下命令在交互式 shell 中查看这些方法的文档：

```py
>>> import Tkinter
>>>help(Tkinter.Widget)
```

Widgets 类下的可用方法如下：

| 方法 | 描述 |
| --- | --- |
| `after(self, ms, func=None, *args)` | 在给定时间后调用函数一次。MS 指定时间为毫秒。`FUNC`给出要调用的函数。其他参数作为函数调用的参数给出。返回：使用`after_cancel`取消调度的标识符。 |
| `after_cancel(self, id)` | 取消与 ID 标识的函数的调度。由 after 或`after_idle`返回的标识符必须作为第一个参数给出。 |
| `after_idle(self, func, *args)` | 如果 Tcl 主循环没有事件要处理，则调用 FUNC 一次。返回一个标识符，用于使用`after_cancel`取消调度。 |
| `bbox = grid_bbox(self, column=None, row=None, col2=None, row2=None)` | 返回由几何管理器 grid 控制的此小部件的边界框的整数坐标元组。如果提供了 `COLUMN` 和 `ROW`，则边界框适用于从行和列 0 的单元格到指定单元格的单元格。如果提供了 `COL2` 和 `ROW2`，则边界框从该单元格开始。返回的整数指定了主小部件中右上角的位置以及宽度和高度。 |
| `bind(self, sequence=None, func=None, add=None)` | 在事件 `SEQUENCE` 上为此小部件绑定一个调用函数 `FUNC`。`SEQUENCE` 是连接事件模式的字符串。事件模式的形式为 `<MODIFIER-MODIFIER-TYPE-DETAIL>`。事件模式也可以是形式为 `<<AString>>` 的虚拟事件，其中 `AString` 可以是任意的。此事件可以通过 `event_generate` 生成。如果事件连接，它们必须彼此紧挨着。如果事件序列发生，并且以事件实例作为参数，则调用 `FUNC`。如果 `FUNC` 的返回值为 "break"，则不会调用其他已绑定的函数。一个额外的布尔参数 `ADD` 指定 `FUNC` 是否将作为其他已绑定的函数调用，或者是否将替换先前的函数。bind 将返回一个标识符，允许使用 unbind 删除已绑定的函数，而不会发生内存泄漏。如果省略 `FUNC` 或 `SEQUENCE`，则返回已绑定的函数或已绑定的事件列表。 |
| `bind_all(self, sequence=None, func=None, add=None)` | 在所有小部件上绑定事件 `SEQUENCE`，调用函数 `FUNC`。一个额外的布尔参数 `ADD` 指定 `FUNC` 是否将作为其他已绑定的函数调用，或者是否将替换先前的函数。参见 bind 了解返回值。 |
| `bind_class(self, className, sequence=None, func=None, add=None)` | 在具有 bind 标签 `CLASSNAME` 的小部件上，在事件 `SEQUENCE` 上绑定一个调用函数 `FUNC`。一个额外的布尔参数 `ADD` 指定 `FUNC` 是否将作为其他已绑定的函数调用，或者是否将替换先前的函数。参见 bind 了解返回值。 |
| `bindtags(self, tagList=None)` | 设置或获取此小部件的 bindtags 列表。如果没有参数，则返回与此小部件关联的所有 bindtags 列表。如果有字符串列表作为参数，则将 bindtags 设置为此列表。bindtags 决定了事件处理的顺序（参见 bind）。 |
| `cget(self, key)` | 返回给定字符串形式的 `Key` 的资源值。 |
| `clipboard_append(self, string, **kw)` | 将 `String` 添加到 `Tk` 剪贴板。关键字参数中指定的可选显示小部件指定了目标显示。可以通过 `selection_get` 获取剪贴板。 |
| `clipboard_clear(self, **kw)` | 清除 Tk 剪贴板中的数据。关键字参数中指定的可选显示小部件指定了目标显示。 |
| `clipboard_get(self, **kw)` | 从窗口的显示中检索剪贴板数据。窗口关键字默认为 Tkinter 应用程序的根窗口。类型关键字指定返回数据的形式，应为一个原子名称，例如 STRING 或 FILE_NAME。类型默认为 `String`。此命令等同于：`selection_get(CLIPBOARD)`。 |
| `columnconfigure = grid_columnconfigure(self, index, cnf={}, **kw)` | 配置网格的 `Index` 列。有效的资源有 minsize（列的最小大小）、weight（额外空间传播到该列的程度）和 pad（额外空间）。 |
| `config = configure(self, cnf=None, **kw)` | 配置窗口的资源。资源值作为关键字参数指定。要获取允许的关键字参数的概述，请调用方法 keys。 |
| `event_add(self, virtual, *sequences)` | 将虚拟事件 `virtual`（形式为 `<<Name>>`）绑定到事件 `sequence`，使得虚拟事件在 SEQUENCE 发生时被触发。 |
| `event_delete(self, virtual, *sequences)` | 从 `sequence` 解绑虚拟事件 `virtual`。 |
| `event_generate(self, sequence, **kw)` | 生成事件 `sequence`。额外的关键字参数指定事件的参数（例如，x、y、rootx 和 rooty）。 |
| `event_info(self, virtual=None)` | 返回所有虚拟事件列表或绑定到虚拟事件 `virtual` 的 `sequence` 的信息。 |
| `focus = focus_set(self)` | 将输入焦点直接设置到这个小部件。如果应用程序当前没有焦点，并且通过窗口管理器获得焦点，则这个小部件将获得焦点。 |
| `focus_displayof(self)` | 返回在当前小部件所在显示上具有焦点的窗口小部件。如果没有应用程序具有焦点，则返回 None。 |
| `focus_force(self)` | 即使应用程序没有焦点，也将输入焦点直接设置到这个小部件。请谨慎使用！ |
| `focus_get(self)` | 返回当前在应用程序中具有焦点的窗口小部件。使用 `focus_displayof` 允许与多个显示一起工作。如果没有应用程序具有焦点，则返回 None。 |
| `focus_lastfor(self)` | 返回如果此小部件的顶层获得窗口管理器的焦点，则将具有焦点的窗口小部件。 |
| `focus_set(self)` | 将输入焦点直接设置到这个小部件。如果应用程序当前没有焦点，并且通过窗口管理器获得焦点，则这个小部件将获得焦点。 |
| `getboolean(self, s)` | 对于作为参数给出的 Tclboolean 值 true 和 false，返回一个布尔值。 |
| `getvar(self, name='PY_VAR')` | 返回 Tcl 变量 `name` 的值。 |
| `grab_current(self)` | 返回在此应用程序中当前具有抓取的窗口小部件或 None。 |
| `grab_release(self)` `)` | 如果当前设置了抓取，则释放此小部件的抓取。 |
| `grab_set(self)` | 为此小部件设置抓取。抓取将所有事件指向此小部件及其应用程序中的子小部件。 |
| `grab_set_global(self)` | 为此小部件设置全局抓取。全局抓取将所有事件指向显示上的此小部件及其子小部件。请谨慎使用 - 其他应用程序不再接收事件。 |
| `grab_status(self)` | 如果此小部件没有、局部或全局抓取，则返回 None、"local" 或 "global"。 |
| `grid_bbox(self, column=None, row=None, col2=None, row2=None)` | 返回一个整数坐标元组，表示由网格管理器控制的此小部件的边界框。如果提供了 `column` 和 `row`，则边界框适用于从行和列 0 开始的指定单元格。如果提供了 `col2` 和 `row2`，则边界框从该单元格开始。返回的整数指定了主小部件中左上角的偏移量以及宽度和高度。 |
| `grid_columnconfigure(self, index, cnf={}, **kw)` | 配置网格的列 `index`。有效的资源有 minsize（列的最小大小）、weight（额外空间传播到本列的程度）和 pad（额外留出的空间）。 |
| `grid_location(self, x, y)` | 返回一个元组，表示列和行，用于标识主小部件中位置在 `x` 和 `y` 的像素所在的单元格。 |
| `grid_propagate(self, flag=['_noarg_'])` | 设置或获取几何信息传播的状态。布尔参数指定是否由子小部件的几何信息确定此小部件的大小。如果没有给出参数，则返回当前设置。 |
| `grid_rowconfigure(self, index, cnf={}, **kw)` | 配置网格的行 `index`。有效的资源有 minsize（行的最小大小）、weight（额外空间传播到本行的程度）和 pad（额外留出的空间）。 |
| `grid_size(self)` | 返回网格中列和行的数量元组。 |
| `grid_slaves(self, row=None, column=None)` | 返回一个列表，包含此小部件在其包装顺序中的所有子小部件。 |
| `image_names(self)` | 返回所有现有图像名称列表。 |
| `image_types(self)` | 返回所有可用图像类型列表（例如 photo bitmap）。 |
| `keys(self)` | 返回此小部件的所有资源名称列表。 |
| `lift = tkraise(self, aboveThis=None)` | 在堆叠顺序中提升此小部件。 |
| `lower(self, belowThis=None)` | 在堆叠顺序中降低此小部件。 |
| `mainloop(self, n=0)` | 调用 Tk 的 `mainloop`。 |
| `nametowidget(self, name)` | 返回由其 Tcl 名称 NAME 标识的 Tkinter 实例的小部件。 |
| `option_add(self, pattern, value, priority=None)` | 为选项模式 PATTERN（第一个参数）设置一个 `value`（第二个参数）。可选的第三个参数给出数字优先级（默认为 80）。 |
| `option_clear(self)` | 清除选项数据库。如果调用 option_add，则将其重新加载。 |
| `option_get(self, name, className)` | 返回此小部件具有 `classname` 的选项 NAME 的值。优先级较高的值覆盖较低优先级的值。 |
| `option_readfile(self, fileName, priority=None)` | 将文件 `filename` 读取到选项数据库中。可选的第二个参数给出数字优先级。 |
| `propagate =pack_propagate(self, flag=['_noarg_'])` | 设置或获取几何信息传播的状态。布尔参数指定奴隶的几何信息是否将决定此小部件的大小。如果没有给出参数，则返回当前设置。 |
| `pack_slaves(self)` | 返回此小部件在其布局顺序中的所有奴隶列表。 |
| `quit(self)` | 退出 Tcl 解释器。所有小部件将被销毁。 |
| `register = _register(self, func, subst=None, needcleanup=1)` | 返回一个新创建的 Tcl 函数。如果调用此函数，Python 函数 `func` 将被执行。可以提供一个可选的函数 `subst`，它将在 `func` 执行之前执行。 |
| `rowconfigure = grid_rowconfigure(self, index, cnf={}, **kw)` | 配置网格的 `index` 行。有效的资源有 minsize（行的最小大小）、weight（额外空间传播到本行的程度）和 pad（额外空间）。 |
| `selection_clear(self, **kw)` | 清除当前的 X 选择。 |
| `selection_get(self, **kw)` | 返回当前 X 选择的内容。关键字参数选择指定选择的名称，默认为 PRIMARY。关键字参数 display 指定要使用的显示上的小部件。 |
| `selection_handle(self, command, **kw)` | 指定一个函数 `command`，如果此小部件拥有的 X 选择被另一个应用程序查询，则调用该函数。此函数必须返回选择的内 容。该函数将使用 OFFSET 和 LENGTH 参数调用，允许对非常长的选择进行分块。以下关键字参数可以提供：选择 - 选择的名称（默认 PRIMARY），类型 - 选择的类型（例如，`string`，FILE_NAME）。 |
| `selection_own(self, **kw)` | 成为 X 选择的所有者。关键字参数选择指定选择的名称（默认 PRIMARY）。 |
| `selection_own_get(self, **kw)` | 返回 X 选择的所有者。以下关键字参数可以提供：选择 - 选择的名称（默认 PRIMARY），类型 - 选择的类型（例如，STRING，FILE_NAME）。 |
| `send(self, interp, cmd, *args)` | 将 Tcl 命令 CMD 发送到不同的解释器 INTERP 执行。 |
| `setvar(self, name='PY_VAR', value='1')` | 将 Tcl 变量 NAME 设置为 VALUE。 |
| `size = grid_size(self)` | 返回网格中列和行的元组。 |
| `slaves = pack_slaves(self)` | 返回此小部件在其布局顺序中的所有奴隶列表。 |
| `tk_focusFollowsMouse(self)` | 鼠标下的小部件将自动获得焦点。无法轻易禁用。 |
| `tk_focusNext(self)` | 返回当前具有焦点的控件之后的下一个控件。焦点顺序首先指向下一个子控件，然后递归地指向子控件的子控件，最后指向堆叠顺序中较高的下一个兄弟控件。如果控件具有设置为 0 的 takefocus 资源，则该控件将被忽略。 |
| `tk_focusPrev(self)` | 返回焦点顺序中的上一个控件。有关详细信息，请参阅`tk_focusNext`。 |
| `tk_setPalette(self, *args, **kw)` | 为所有控件元素设置新的颜色方案。作为参数的单个颜色将导致 Tk 控件元素的所有颜色都由此颜色派生。或者，可以给出多个关键字参数及其关联的颜色。以下关键字是有效的：`activeBackground`、`foreground`、`selectColor`、`activeForeground`、`highlightBackground`、`selectBackground`、`background`、`highlightColor`、`selectForeground`、`disabledForeground`、`insertBackground`和`troughColor`。 |
| `tkraise(self, aboveThis=None)` | 在堆叠顺序中提升此控件。 |
| `unbind(self, sequence, funcid=None)` | 为此控件解绑事件 SEQUENCE 的由 FUNCID 标识的函数。 |
| `unbind_all(self, sequence)'` | 为所有控件解绑事件 SEQUENCE 的所有函数。 |
| `unbind_class(self, className, sequence)` | 解绑所有具有 bindtag `classname`的控件的事件 SEQUENCE 的所有函数。 |
| `update(self)` | 进入事件循环，直到所有挂起的事件都被 Tcl 处理。 |
| `update_idletasks(self)` | 进入事件循环，直到所有空闲回调都被调用。这将更新窗口的显示，但不会处理由用户引起的事件。 |
| `wait_variable(self, name='PY_VAR')` | 等待直到变量被修改。必须给出类型为`IntVar`、`StringVar`、`DoubleVar`或`BooleanVar`的参数。 |
| `wait_visibility(self, window=None)` | 等待直到一个 Widget 的可见性发生变化（例如，它出现）。如果没有给出参数，则使用 self。 |
| `wait_window(self, window=None)` | 等待直到一个 Widget 被销毁。如果没有给出参数，则使用 self。 |
| `waitvar = wait_variable(self, name='PY_VAR')` | 等待直到变量被修改。必须给出类型为`IntVar`、`StringVar`、`DoubleVar`或`BooleanVar`的参数。 |
| `winfo_atom(self, name, displayof=0)` | 返回表示原子名称的整数。 |
| `winfo_atomname(self, id, displayof=0)` | 返回具有标识符 ID 的原子名称。 |
| `winfo_cells(self)` | 返回此控件颜色映射中的单元格数。 |
| `winfo_children(self)` | 返回此控件的子控件列表。 |
| `winfo_class(self)` | 返回此控件的窗口类名。 |
| `winfo_colormapfull(self)` | 如果在最后的颜色请求中`colormap`已满，则返回 true。 |
| `winfo_containing(self, rootX, rootY, displayof=0)` | 返回在根坐标 root`X`、`rootY`处的控件。 |
| `winfo_depth(self)` | 返回每像素的位数。 |
| `winfo_exists(self)` | 如果此小部件存在，则返回 true。 |
| `winfo_fpixels(self, number)` | 返回给定距离 NUMBER（例如："3c"）的像素数，以浮点数形式返回。 |
| `winfo_geometry(self)` | 返回此小部件的几何字符串，格式为"widthxheight+X+Y"。 |
| `winfo_height(self)` | 返回此小部件的高度。 |
| `winfo_id(self)` | 返回此小部件的标识符 ID。 |
| `winfo_interps(self, displayof=0)` | 返回此显示的所有 Tcl 解释器的名称。 |
| `winfo_ismapped(self)` | 如果此小部件已映射，则返回 true。 |
| `winfo_manager(self)` | 返回此小部件的窗口管理器名称。 |
| `winfo_name(self)` | 返回此小部件的名称。 |
| `winfo_parent(self)` | 返回此小部件的父级名称。 |
| `winfo_pathname(self, id, displayof=0)` | 返回由 ID 指定的窗口的路径名。 |
| `winfo_pixels(self, num)` | winfo_fpixels 的舍入整数值。 |
| `winfo_pointerx(self)` | 返回根窗口上指针的 x 坐标。 |
| `winfo_pointerxy(self)` | 返回根窗口上指针的 x 和 y 坐标的元组。 |
| `winfo_pointery(self)` | 返回根窗口上指针的 y 坐标。 |
| `winfo_reqheight(self)` | 返回此小部件请求的高度。 |
| `winfo_reqwidth(self)` | 返回此小部件请求的宽度。 |
| `winfo_rgb(self, color)` | 返回此小部件中颜色`color`的红色、绿色、蓝色十进制值的元组。 |
| `winfo_rootx(self)` `/ winfo_rooty(self)` | 返回此小部件在根窗口上的左上角 x/y 坐标。 |
| `winfo_screen(self)` | 返回此小部件的屏幕名称。 |
| `winfo_screencells(self)` | 返回此小部件屏幕调色板中的单元格数。 |
| `winfo_screendepth(self)` | 返回此小部件屏幕根窗口的每像素位数。 |
| `winfo_screenheight(self)` | 返回此小部件屏幕高度（以像素为单位）的像素数。 |
| `winfo_screenmmheight(self)` | 返回此小部件屏幕高度（以毫米为单位）的像素数。 |
| `winfo_screenmmwidth(self)` | 返回此小部件屏幕宽度（以毫米为单位）的像素数。 |
| `winfo_screenwidth(self)` | 返回此小部件屏幕宽度（以像素为单位）的像素数。 |
| `winfo_toplevel(self)` | 返回此小部件的 Toplevel 小部件。 |
| `winfo_viewable(self)` | 如果小部件及其所有更高祖先都已映射，则返回 true。 |
| `winfo_visual(self) = winfo_screenvisual(self)` | 返回字符串之一`directcolor`、`grayscale`、`pseudocolor`、`staticcolor`、`staticgray`或`truecolor`，表示此小部件的`colormodel`。 |
| `winfo_visualid(self)` | 返回此小部件视觉效果的 X 标识符。 |
| `winfo_visualsavailable(self, includeids=0)` | 返回此小部件屏幕上所有可用的视觉效果的列表。 |
| `winfo_vrootheight(self)` | 返回与此小部件关联的虚拟根窗口的高度（以像素为单位）。如果没有虚拟根窗口，则返回屏幕的高度。 |
| `winfo_vrootwidth(self)` | 返回与此小部件关联的虚拟根窗口的宽度（以像素为单位）。如果没有虚拟根窗口，则返回屏幕的宽度。 |
| `winfo_vrootx(self)` | 返回此小部件相对于屏幕根窗口的虚拟根的 x 偏移量。 |
| `winfo_vrooty(self)` | 返回此小部件相对于屏幕根窗口的虚拟根的 y 偏移量。 |
| `winfo_width(self)` | 返回此小部件的宽度。 |
| `winfo_x(self)` | 返回此小部件在父窗口中的左上角 x 坐标。 |
| `winfo_y(self)` | 返回此小部件在父窗口中的左上角 y 坐标。 |

# ttk 小部件

ttk 小部件基于 TIP #48 ([`tip.tcl.tk/48`](http://tip.tcl.tk/48)) 指定的改进和增强版本的风格引擎。

文件：`path\to\python27\\lib\lib-tk\ttk.py`

基本思想是在尽可能的程度上将实现小部件行为的代码与实现其外观的代码分开。小部件类绑定主要负责维护小部件状态和调用回调，而小部件外观的所有方面都位于主题之下。

您可以将一些 Tkinter 小部件替换为其相应的 ttk 小部件（按钮、复选框、输入框、框架、标签、标签框架、菜单按钮、分割窗口、单选按钮、滑块和滚动条）。

然而，Tkinter 和 ttk 小部件并不完全兼容。主要区别是 Tkinter 小部件的样式选项（如 `fg`、`bg`、`relief` 等）不是 ttk 小部件的支持选项。这些样式选项被移动到 `ttk.Style()`。

这里是一个小的 Tkinter 代码示例：

```py
Label(text="Who", fg="white", bg="black")
Label(text="Are You ?", fg="white", bg="black")

```

以下是它的等价 ttk 代码：

```py
style = ttk.Style()
style.configure("BW.TLabel", foreground="white", background="black")
ttk.Label(text="Who", style="BW.TLabel")
ttk.Label(text="Are You ?", style="BW.TLabel")

```

ttk 还提供了六个新的小部件类，这些类在 Tkinter 中不可用。这些是 `Combobox`、`Notebook`、`Progressbar`、`Separator`、`Sizegrip` 和 `Treeview`。

ttk 风格名称如下：

| 小部件类 | 样式名称 |
| --- | --- |
| `Button` | `TButton` |
| `Checkbutton` | `TCheckbutton` |
| `Combobox` | `TCombobox` |
| `Entry` | `TEntry` |
| `Frame` | `TFrame` |
| `Label` | `TLabel` |
| `LabelFrame` | `TLabelFrame` |
| `Menubutton` | `TMenubutton` |
| `Notebook` | `TNotebook` |
| `PanedWindow` | `TPanedwindow`（注意窗口名称不区分大小写！） |
| `Progressbar` | `Horizontal.TProgressbar` 或 `Vertical.TProgressbar`，根据 orient 选项。 |
| `Radiobutton` | `TRadiobutton` |
| `Scale` | `Horizontal.TScale` 或 `Vertical.TScale`，根据 orient 选项。 |
| `Scrollbar` | `Horizontal.TScrollbar` 或 `Vertical.TScrollbar`，根据 orient 选项。 |
| `Separator` | `TSeparator` |
| `Sizegrip` | `TSizegrip` |
| `Treeview` | `Treeview` (注意只有一个 'T'，表示不是 TTreview!) |

所有 ttk 小部件都有的选项如下：

| 选项 | 描述 |
| --- | --- |
| `class` | 指定窗口类。当查询选项数据库以获取窗口的其他选项、确定窗口的默认 bindtags 以及选择小部件的默认布局和样式时使用该类。这是一个只读选项，只能在创建窗口时指定。 |
| `cursor` | 指定小部件显示的鼠标光标 |
| `takefocus` | 确定窗口在键盘遍历期间是否接受焦点。返回 0、1 或空字符串。如果为 0，则在键盘遍历期间应完全跳过窗口。如果为 1，则只要窗口是可见的，它就应该接收输入焦点。空字符串表示遍历脚本将决定是否将焦点放在窗口上。 |
| `style` | 可用于指定自定义小部件样式。 |

所有可滚动 ttk 小部件接受的选项如下：

| 选项 | 描述 |
| --- | --- |
| `xscrollcommand` | 用于与水平滚动条通信。当小部件窗口中的视图发生变化时，小部件将基于 scrollcommand 生成一个 Tcl 命令。通常，此选项由某个滚动条的 Scrollbar.set() 方法组成。这将导致滚动条在窗口中的视图发生变化时更新。 |
| `yscrollcommand` | 垂直滚动条的命令。 |

ttk.Widget 类的方法及其描述如下：

| 方法 | 描述 |
| --- | --- |
| `identify(self, x, y)` | 返回 x, y 位置处的元素名称，如果点不在任何元素内，则返回空字符串。x 和 y 是相对于小部件的像素坐标。 |
| `instate(self, statespec, callback=None, *args, **kw)` | 测试小部件的状态。如果没有指定回调函数，如果小部件状态与 statespec 匹配则返回 True，否则返回 False。如果指定了回调函数，则当小部件状态与 `statespec` 匹配时，将使用 `*args` 和 `**kw` 调用它。`statespec` 预期是一个序列。 |
| `state(self, statespec=None)` | 修改或查询小部件状态。如果 statespec 为 None，则返回小部件状态，否则根据 `statespec` 标志设置状态，然后返回一个新的状态规范，指示哪些标志已更改。`statespec` 预期是一个序列。 |

我们在此处不会显示所有 ttk 小部件特定选项。要获取 ttk 小部件可用选项的列表，请使用帮助命令。

要获取任何 ttk 小部件/类的帮助，请使用以下命令将 ttk 导入命名空间：

```py
>>>import ttk
```

然后，可以使用以下命令获取特定小部件的信息：

| 小部件名称 | 获取帮助 |
| --- | --- |
| Label | `help(ttk.Label)` |
| Button | `help(ttk.Button)` |
| CheckButton | `help(ttk.Checkbutton)` |
| Entry | `help(ttk.Entry)` |
| Frame | `help(ttk.Frame)` |
| LabelFrame | `help(ttk.LabelFrame)` |
| Menubutton | `help(ttk.Menubutton)` |
| OptionMenu | `help(ttk.OptionMenu)` |
| PanedWindow | `help(ttk.PanedWindow)` |
| 单选按钮 | `帮助(ttk.Radiobutton)` |
| 滚动条 | `帮助(ttk.Scale)` |
| 滚动条 | `帮助(ttk.Scrollbar)` |
| 组合框 | `帮助(ttk.Combobox)` |
| 笔记本 | `帮助(ttk.Notebook)` |
| 进度条 | `帮助(ttk.Progressbar)` |
| 分隔符 | `帮助(ttk.Separator)` |
| 大小调整手柄 | `帮助(ttk.Sizegrip)` |
| 树形视图 | `帮助(ttk.Treeview)` |

以下是一些 ttkVirtual 事件及其触发情况：

| 虚拟事件 | 触发时 |
| --- | --- |
| `<<ComboboxSelected>>` | 用户从 Combobox 小部件的值列表中选择了一个元素 |
| `<<NotebookTabChanged>>` | 在 Notebook 小部件中选择了新标签页 |
| `<<TreeviewSelect>>` | Treeview 小部件中的选择发生变化。 |
| `<<TreeviewOpen>>` | 在将焦点项设置为打开 = True 前立即。 |
| `<<TreeviewClose>>` | 在将焦点项设置为打开 = False 后立即。 |

ttk 中的每个小部件都分配了一个样式，该样式指定了组成小部件的元素集合以及它们的排列方式，以及元素选项的动态和默认设置。

默认情况下，样式名称与小部件的类名相同，但可能被小部件的样式选项覆盖。如果小部件的类名未知，请使用方法 `Misc.winfo_class()` (`somewidget.winfo_class()`)。以下是一些 ttk 样式的方法及其描述：

| 方法 | 描述 |
| --- | --- |
| `configure(self, style, query_opt=None, **kw)` | 查询或设置样式中指定选项的默认值。`kw` 中的每个键都是一个选项，每个值是标识该选项值的字符串或序列。 |
| `element_create(self, elementname, etype, *args, **kw)` | 在给定的 `etype` 当前主题中创建一个新元素。 |
| `element_names(self)` | 返回当前主题中定义的元素列表。 |
| `element_options(self, elementname)` | 返回 `elementname` 选项的列表。 |
| `layout(self, style, layoutspec=None)` | 定义给定样式的小部件布局。如果省略 `layoutspec`，则返回给定样式的布局规范。`layoutspec` 预期是一个列表或一个不同于 None 的对象，如果想要“关闭”该样式，则该对象应评估为 False。如果它是一个列表（或元组，或其他），则每个项应是一个元组，其中第一个项是布局名称，第二个项应具有以下描述的格式 |

布局可以是 `None`，如果它不包含选项，或者是一个选项字典，指定如何排列元素。布局机制使用简化版的 pack 几何管理器：给定一个初始内腔，每个元素都被分配一个包裹。

| 有效选项：值 | 描述 |
| --- | --- |
| `side: whichside` | 指定放置元素的内腔的哪一侧；top、right、bottom 或 left 之一。如果省略，则元素占据整个内腔。 |
| `sticky: nswe` | 指定元素在其分配的包裹内的放置位置。 |
| `children: [sublayout... ]` | 指定要放置在元素内部的元素列表。每个元素是一个元组（或其他序列），其中第一个项目是布局名称，其余的是布局。 |
| `lookup(self, style, option, state=None, default=None)` | 返回在样式中对选项指定的值。如果指定了状态，则它应是一个包含一个或多个状态的序列。如果设置了默认参数，则在找不到选项的指定时，它用作回退值。 |
| `map(self, style, query_opt=None, **kw)` | 查询或设置指定选项（s）在样式中的动态值。`kw` 中的每个键是一个选项，每个值应是一个列表或元组（通常是），其中包含按元组、列表或其他您偏好的方式分组的 `statespecs`。`statespec` 是由一个或多个状态和值组成的复合体。 |
| `theme_create(self, themename, parent=None, settings=None)` | 创建一个新的主题。如果 `themename` 已经存在，则是一个错误。如果指定了 `parent`，则新主题将从指定的父主题继承样式、元素和布局。如果存在设置，则它们应使用与 `theme_settings` 相同的语法。 |
| `theme_names(self)` | 返回所有已知主题的列表。 |
| `theme_settings(self, themename, settings)` | 临时将当前主题设置为 `themename`，应用指定的设置，然后恢复先前的主题。`settings` 中的每个键是一个样式，每个值可能包含 `configure`、`map`、`layout` 和 `element create` 的键，并且它们应具有与 `configure`、`map`、`layout` 和 `element_create` 方法指定的相同格式。 |
| `theme_use(self, themename=None)` | 如果 `themename` 为 None，则返回正在使用的主题；否则，将当前主题设置为 `themename`，刷新所有小部件并发出 `<<ThemeChanged>>` 事件。 |

# Toplevel 窗口方法

这些方法使与窗口管理器通信成为可能。它们在根窗口（Tk）和 Toplevel 实例上都是可用的。

注意，不同的窗口管理器表现不同。例如，一些窗口管理器不支持图标窗口；一些不支持窗口组，等等。

| `aspect = wm_aspect(self, minNumer=None, minDenom=None, maxNumer=None, maxDenom=None)` | 指示窗口管理器将此小部件的纵横比（宽度/高度）设置为介于 `minNumer`/`minDenom` 和 `maxNumer`/`maxDenom` 之间。如果没有提供参数，则返回实际值的元组。 |
| --- | --- |
| `attributes = wm_attributes(self, *args)` | 此子命令返回或设置平台特定的属性。第一种形式返回平台特定标志及其值的列表。第二种形式返回特定选项的值。第三种形式设置一个或多个值。值如下：在 Windows 上，-disabled 获取或设置窗口是否处于禁用状态。-toolwindow 获取或设置窗口转换为工具窗口的样式（如 MSDN 中定义）。-topmost 获取或设置此窗口是否为顶层窗口（显示在其他所有窗口之上）。在 Macintosh 上，`XXXXX`在 Unix 上，目前没有特殊的属性值。 |
| `client = wm_client(self, name=None)` | 将名称存储在此小部件的 `WM_CLIENT_MACHINE` 属性中。返回当前值。 |
| `colormapwindows = wm_colormapwindows(self, *wlist)` | 将窗口名称列表（`wlist`）存储在此小部件的 `WM_COLORMAPWINDOWS` 属性中。此列表包含与父窗口 `colormaps` 不同的窗口。如果 `wlist` 为空，则返回当前小部件列表。 |
| `command = wm_command(self, value=None)` | 在 `WM_COMMAND` 属性中存储 `value`。这是用于调用应用程序的命令。如果 `value` 为 `None`，则返回当前命令。 |
| `deiconify = wm_deiconify(self)` | `deiconify` 此小部件。如果它从未映射，则不会映射。在 Windows 上，它将提升此小部件并使其获得焦点。 |
| `focusmodel = wm_focusmodel(self, model=None)` | 将焦点模型设置为 `model`，"active" 表示此小部件将自行请求焦点，"passive" 表示窗口管理器应提供焦点。如果 `model` 为 `None`，则返回当前焦点模型。 |
| `frame = wm_frame(self)` | 如果存在，则返回此小部件装饰框架的标识符。 |
| `geometry = wm_geometry(self, newGeometry=None)` | 将 `geometry` 设置为 `newgeometry`，其形式为 `=widthxheight+x+y`。如果未提供，则返回当前值。 |
| `grid = wm_grid(self, baseWidth=None, baseHeight=None, widthInc=None, heightInc=None)` | 指示窗口管理器，此小部件只能在网格边界上调整大小。`widthInc` 和 `heightInc` 是网格单元的宽度和高度（以像素为单位）。`baseWidth` 和 `baseHeight` 是在 `Tk_GeometryRequest` 中请求的网格单元数。 |
| `group = wm_group(self, pathName=None)` | 将相关小部件的组领导者小部件设置为 `pathName`。如果未提供，则返回此小部件的组领导者。 |
| `iconbitmap = wm_iconbitmap(self, bitmap=None, default=None)` | 将图标化小部件的位图设置为 BITMAP。如果未提供，则返回位图。在 Windows 上，可以使用 DEFAULT 参数设置小部件及其未显式设置图标的任何后代的图标。DEFAULT 可以是到 `.ico` 文件的相对路径（例如：`root.iconbitmap(default='myicon.ico')`）。有关更多信息，请参阅 Tk 文档。 |
| `iconify = wm_iconify(self)` | 将小部件显示为图标。 |
| `iconmask = wm_iconmask(self, bitmap=None)` | 设置此小部件图标位图的掩码。如果未提供，则返回掩码。 |
| `iconname = wm_iconname(self, newName=None)` | 设置此小部件图标的名称。如果未提供，则返回名称。 |
| `iconposition = wm_iconposition(self, x=None, y=None)` | 将此小部件图标的 X 和 Y 位置设置为 X 和 Y。如果未提供，则返回 X 和 Y 的当前值。 |
| `iconwindow = wm_iconwindow(self, pathName=None)` | 将小部件`pathName`设置为显示图标。如果未提供，则返回当前值。 |
| `maxsize = wm_maxsize(self, width=None, height=None)` | 设置此小部件的最大`width`和`height`。如果窗口是网格化的，则值以网格单位给出。如果未提供，则返回当前值。 |
| `minsize = wm_minsize(self, width=None, height=None)` | 设置此小部件的最小`width`和`height`。如果窗口是网格化的，则值以网格单位给出。如果未提供，则返回当前值。 |
| `overrideredirect = wm_overrideredirect(self, boolean=None)` | 指示窗口管理器在布尔值为 1 时忽略此小部件。如果未提供，则返回当前值。 |
| `positionfrom = wm_positionfrom(self, who=None)` | 指示窗口管理器，如果`who`为"user"，则由用户定义此小部件的位置，如果`who`为"program"，则由其自身策略定义。 |
| `protocol = wm_protocol(self, name=None, func=None)` | 将函数`func`绑定到此小部件的命令`name`。如果未提供，则返回绑定到`name`的函数。`name`可以是例如`WM_SAVE_YOURSELF`或`WM_DELETE_WINDOW`。 |
| `resizable = wm_resizable(self, width=None, height=None)` | 指示窗口管理器是否可以在`width`或`height`中调整此宽度的大小。两个值都是布尔值。 |
| `sizefrom = wm_sizefrom(self, who=None)` | 指示窗口管理器，如果`who`为"user"，则由用户定义此小部件的大小，如果`who`为"program"，则由其自身策略定义。 |
| `state = wm_state(self, newstate=None)` | 查询或设置此小部件的状态，可以是正常、图标、图标化（参见`wm_iconwindow`）、撤回或缩放（仅限 Windows）。 |
| `title = wm_title(self, string=None)` | 设置此小部件的标题。 |
| `transient = wm_transient(self, master=None)` | 指示窗口管理器，此小部件相对于小部件`master`是瞬时的。 |
| `withdraw = wm_withdraw(self)` | 将此小部件从屏幕上撤回，使其未映射并被窗口管理器遗忘。使用`wm_deiconify`重新绘制它。 |
| `wm_aspect(self, minNumer=None, minDenom=None, maxNumer=None, maxDenom=None)` | 指示窗口管理器将此小部件的宽高比（宽度/高度）设置为介于`minNumer`/`minDenom`和`maxNumer`/`maxDenom`之间。如果没有提供参数，则返回实际值。 |
| `wm_attributes(self, *args)` | 此子命令返回或设置平台特定的属性。第一种形式返回平台特定标志及其值的列表。第二种形式返回特定选项的值。第三种形式设置一个或多个值。值如下：在 Windows 上，`-disabled` 获取或设置窗口是否处于禁用状态。`-toolwindow` 获取或设置窗口到工具窗口的样式（如 MSDN 中定义）。`-topmost` 获取或设置此是否为顶层窗口（显示在其他所有窗口之上）。在 Macintosh 上，`XXXXX`。在 Unix 上，目前没有特殊的属性值。 |
| `wm_client(self, name=None)` | 将 `name` 存储在此小部件的 `WM_CLIENT_MACHINE` 属性中。返回当前值。 |
| `wm_colormapwindows(self, *wlist)` | 将窗口名称列表（wlist）存储在此小部件的 `WM_COLORMAPWINDOWS` 属性中。此列表包含与父窗口 `colormaps` 不同的窗口。如果 wlist 为空，则返回当前小部件的列表。 |
| `wm_command(self, value=None)` | 将 `value` 存储在 `WM_COMMAND` 属性中。这是用于调用应用程序的命令。如果 `value` 为 `None`，则返回当前命令。 |
| `wm_deiconify(self)` | 取消图标化此小部件。如果它从未被映射，则不会进行映射。在 Windows 上，它将提升此小部件并使其获得焦点。 |
| `wm_focusmodel(self, model=None)` | 将焦点模型设置为 `model`。"active" 表示此小部件将自行请求焦点，"passive" 表示窗口管理器应提供焦点。如果 `model` 为 `None`，则返回当前焦点模型。 |
| `wm_frame(self)` | 如果存在，则返回此小部件装饰框架的标识符。 |
| `wm_geometry(self, newGeometry=None)` | 将 `geometry` 设置为 `newgeometry` 的形式 `=widthxheight+x+y`。如果未提供 `None`，则返回当前值。 |
| `wm_grid(self, baseWidth=None, baseHeight=None, widthInc=None, heightInc=None)` | 指示窗口管理器，此小部件只能在网格边界上调整大小。`widthInc` 和 `heightInc` 是像素中网格单元的宽度和高度。`baseWidth` 和 `baseHeight` 是在 `Tk_GeometryRequest` 中请求的网格单元数。 |
| `wm_group(self, pathName=None)` | 将相关小部件的组领导者小部件设置为 `pathname`。如果未提供 `None`，则返回此小部件的组领导者。 |
| `wm_iconbitmap(self, bitmap=None, default=None)` | 将图标化小部件的位图设置为 `bitmap`。如果未提供 `None`，则返回位图。在 Windows 上，可以使用 `default` 参数设置小部件及其未显式设置图标的任何后代的图标。DEFAULT 可以是 `.ico` 文件的相对路径（例如：`root.iconbitmap(default='myicon.ico')`）。有关更多信息，请参阅 Tk 文档。 |
| `wm_iconify(self)` | 将小部件显示为图标。 |
| `wm_iconmask(self, bitmap=None)` | 设置此小部件图标位图的掩码。如果未提供 `None`，则返回掩码。 |
| `wm_iconname(self, newName=None)` | 设置这个小部件图标的名称。如果未提供`None`，则返回名称。 |
| `wm_iconposition(self, x=None, y=None)` | 将这个小部件图标的 X 和 Y 位置设置为 X 和 Y。如果未提供`None`，则返回 X 和 Y 的当前值。 |
| `wm_iconwindow(self, pathName=None)` | 将小部件`pathname`设置为显示图标，而不是图标本身。如果未提供`None`，则返回当前值。 |
| `wm_maxsize(self, width=None, height=None)` | 设置这个小部件的最大`width`和`height`。如果窗口是网格化的，则这些值以网格单位给出。如果未提供`None`，则返回当前值。 |
| `wm_minsize(self, width=None, height=None)` | 设置这个小部件的最小`width`和`height`。如果窗口是网格化的，则这些值以网格单位给出。如果未提供`None`，则返回当前值。 |
| `wm_overrideredirect(self, boolean=None)` | 指示窗口管理器，如果提供了布尔值 1，则忽略这个小部件。如果未提供`None`，则返回当前值。 |
| `wm_positionfrom(self, who=None)` | 指示窗口管理器，如果`who`是"用户"，则由用户定义这个小部件的位置，如果`who`是"程序"，则由其自己的策略定义。 |
| `wm_protocol(self, name=None, func=None)` | 将函数`func`绑定到这个小部件的命令`name`。如果未提供`None`，则返回绑定到`name`的函数。名称可以是例如`WM_SAVE_YOURSELF`或`WM_DELETE_WINDOW`。 |
| `wm_resizable(self, width=None, height=None)` | 指示窗口管理器是否允许在`width`或`height`中调整这个小部件的大小。两个值都是布尔值。 |
| `wm_sizefrom(self, who=None)` | 指示窗口管理器，如果`who`是"用户"，则由用户定义这个小部件的大小，如果`who`是"程序"，则由其自己的策略定义。 |
| `wm_state(self, newstate=None)` | 查询或设置这个小部件的状态为`normal`、`icon`、`iconic`（见`wm_iconwindow`）、`withdrawn`或`zoomed`（仅限 Windows）。 |
| `wm_title(self, string=None)` | 设置这个小部件的标题。 |
| `wm_transient(self, master=None)` | 指示窗口管理器，这个小部件相对于小部件`master`是临时的。 |
| `wm_withdraw(self)` | 将这个小部件从屏幕上移除，使其未被映射且被窗口管理器遗忘。使用`wm_deiconify`重新绘制它。 |
