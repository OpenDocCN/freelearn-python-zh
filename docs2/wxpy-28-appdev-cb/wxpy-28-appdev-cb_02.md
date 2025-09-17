# 第二章 响应事件

在本章中，我们将涵盖：

+   处理事件

+   理解事件传播

+   处理键事件

+   使用 UpdateUI 事件

+   操纵鼠标

+   创建自定义事件类

+   使用 EventStack 管理事件处理器

+   使用验证器验证输入

+   处理 Apple 事件

# 简介

在事件驱动系统中，事件被用来将框架内的动作与那些事件相关联的回调函数连接起来。基于事件驱动框架构建的应用程序利用这些事件来确定何时响应由用户或系统发起的动作。在用户界面中，事件是了解何时点击了按钮、选择了菜单或用户在与应用程序界面交互时可能采取的广泛多样的其他动作的方式。

正如你所见，了解如何应对应用生命周期中发生的事件是创建一个功能应用的关键部分。因此，让我们深入 wxPython 的事件驱动世界吧。

# 处理事件

wxPython 是一个事件驱动系统。该系统的使用方法在框架中非常直接且规范。无论你的应用程序将交互的控制或事件类型如何，处理事件的基模式都是相同的。本食谱将介绍在 wxPython 事件系统中的基本操作方法。

## 如何做到这一点...

让我们创建一个简单的`Frame`，其中包含两个按钮，以展示如何处理事件：

```py
class MyFrame(wx.Frame):
    def __init__(self, parent, id=wx.ID_ANY, title="", 
                 pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.DEFAULT_FRAME_STYLE,
                 name="MyFrame"):
        super(MyFrame, self).__init__(parent, id, title,
                                      pos, size, style, name)

        # Attributes
        self.panel = wx.Panel(self)

        self.btn1 = wx.Button(self.panel, label="Push Me")
        self.btn2 = wx.Button(self.panel, label="push me too")

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.btn1, 0, wx.ALL, 10)
        sizer.Add(self.btn2, 0, wx.ALL, 10)
        self.panel.SetSizer(sizer)

        self.Bind(wx.EVT_BUTTON, self.OnButton, self.btn1)
        self.Bind(wx.EVT_BUTTON,
                  lambda event:
                  self.btn1.Enable(not self.btn1.Enabled),
                  self.btn2)

    def OnButton(self, event):
        """Called when self.btn1 is clicked"""
        event_id = event.GetId()
        event_obj = event.GetEventObject()
        print "Button 1 Clicked:"
        print "ID=%d" % event_id
        print "object=%s" % event_obj.GetLabel()

```

## 它是如何工作的...

在这个菜谱中需要注意的代码行是两个 `Bind` 调用。`Bind` 方法用于将事件处理函数与可能发送到控件的事件关联起来。事件总是沿着窗口层次结构向上传播，而不会向下传播。在这个例子中，我们将按钮事件绑定到了 `Frame`，但事件将起源于 `Panel` 的子 `Button` 对象。`Frame` 对象位于包含 `Panel` 的层次结构的顶部，而 `Panel` 又包含两个 `Buttons`。正因为如此，由于事件回调既没有被 `Button` 也没有被 `Panel` 处理，它将传播到 `Frame`，在那里我们的 `OnButton` 处理程序将被调用。

`Bind` 方法需要两个必填参数：

+   事件绑定对象 (`EVT_FOO`)

+   一个接受事件对象作为其第一个参数的可调用对象。这是当事件发生时将被调用的事件处理函数。

可选参数用于指定绑定事件处理器的源控件。在这个示例中，我们通过将`Button`对象指定为`Bind`函数的第三个参数，为每个按钮绑定了一个处理器。

`EVT_BUTTON` 是当应用程序用户点击 `Button` 时的事件绑定器。当第一个按钮被点击时，事件处理器 `OnButton` 将被调用以通知我们的程序这一动作已发生。事件对象将作为其第一个参数传递给处理器函数。事件对象有多个方法可以用来获取有关事件及其来源控件的信息。每个事件可能都有不同的数据可用，这取决于与事件来源的控件类型相关的事件类型。

对于我们的第二个`按钮`，我们使用了`lambda`函数作为创建事件处理函数的简写方式，无需定义新的函数。这是一种处理只需执行简单操作的事件的便捷方法。

## 参见

+   第一章中的 *应用程序对象* 菜单，*使用 wxPython 入门* 讲述了主循环，这是事件系统的核心。

+   第一章中的*理解窗口层次结构*配方描述了窗口包含层次结构。

+   第三章中的 *创建股票按钮* 菜单，*用户界面基本构建块* 详细解释了按钮。

+   第七章中的*使用 BoxSizer 布局*配方，*窗口布局与设计*解释了如何使用`BoxSizer`类来布局控件。

# 理解事件传播

在 wxPython 中，主要有两种事件对象，每种都有其独特的行为：

+   事件

+   命令事件

基本事件（`Events`）是指不会在窗口层次结构中向上传播的事件。相反，它们保持在它们被发送到的或起源的窗口的本地。第二种类型，`CommandEvents`，是更常见的事件类型，它们与常规事件的不同之处在于，它们会沿着窗口父级层次结构向上传播，直到被处理或到达应用程序对象的末尾。本食谱将探讨如何处理、理解和控制事件的传播。

## 如何做到这一点...

为了探索事件如何传播，让我们创建另一个简单的应用程序：

```py
import wx

ID_BUTTON1 = wx.NewId()
ID_BUTTON2 = wx.NewId()

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="Event Propagation")
        self.SetTopWindow(self.frame)
        self.frame.Show()

        self.Bind(wx.EVT_BUTTON, self.OnButtonApp)

        return True

    def OnButtonApp(self, event):
        event_id = event.GetId()
        if event_id == ID_BUTTON1:
            print "BUTTON ONE Event reached the App Object"

class MyFrame(wx.Frame):
    def __init__(self, parent, id=wx.ID_ANY, title="", 
                 pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.DEFAULT_FRAME_STYLE,
                 name="MyFrame"):
        super(MyFrame, self).__init__(parent, id, title,
                                      pos, size, style, name)

        # Attributes
        self.panel = MyPanel(self)

        self.btn1 = wx.Button(self.panel, ID_BUTTON1,
                              "Propagates")
        self.btn2 = wx.Button(self.panel, ID_BUTTON2,
                              "Doesn't Propagate")

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.btn1, 0, wx.ALL, 10)
        sizer.Add(self.btn2, 0, wx.ALL, 10)
        self.panel.SetSizer(sizer)

        self.Bind(wx.EVT_BUTTON, self.OnButtonFrame)

    def OnButtonFrame(self, event):
        event_id = event.GetId()
        if event_id == ID_BUTTON1:
            print "BUTTON ONE event reached the Frame"
            event.Skip()
        elif event_id == ID_BUTTON2:
            print "BUTTON TWO event reached the Frame"

class MyPanel(wx.Panel):
    def __init__(self, parent):
        super(MyPanel, self).__init__(parent)

        self.Bind(wx.EVT_BUTTON, self.OnPanelButton)

    def OnPanelButton(self, event):
        event_id = event.GetId()
        if event_id == ID_BUTTON1:
            print "BUTTON ONE event reached the Panel"
            event.Skip()
        elif event_id == ID_BUTTON2:
            print "BUTTON TWO event reached the Panel"
            # Not skipping the event will cause its 
            # propagation to end here
if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()

```

运行此程序将创建一个带有两个按钮的应用程序。点击每个按钮以查看事件如何不同地传播。

## 它是如何工作的...

事件处理程序链的调用将从事件起源的对象开始。在这种情况下，它将是我们的两个按钮之一。此应用程序窗口层次结构的每个级别都绑定了一个通用事件处理程序，它将接收任何按钮事件。

点击第一个按钮将显示所有的事件处理程序都被调用。这是因为对于第一个按钮，我们调用了事件的`Skip`方法。在事件上调用`Skip`将告诉它继续传播到事件处理程序层次结构中的下一级。这将是显而易见的，因为控制台将打印出三条语句。另一方面，点击第二个按钮将只调用一个事件处理程序，因为未调用`Skip`。

## 参见

+   本章中的*处理事件*菜谱解释了事件处理器是如何工作的。

+   第一章中的*理解窗口层次结构*配方，*使用 wxPython 入门*描述了事件传播通过的窗口层次结构。

# 处理关键事件

`KeyEvents`是与键盘操作相关的事件。许多控件可以接受键盘事件。每次在键盘上按下键时，都会向具有键盘焦点的控件发送两个或三个事件，具体取决于按下了哪个键。本食谱将创建一个简单的文本编辑窗口，以演示如何使用`KeyEvents`来过滤添加到`TextCtrl`中的文本。

## 如何做到这一点...

要查看一些`KeyEvents`的实际应用，让我们创建一个简单的窗口，该窗口上有一个`TextCtrl`：

```py
class MyFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        super(MyFrame, self).__init__(parent, *args, **kwargs)

        # Attributes
        self.panel = wx.Panel(self)
        self.txtctrl = wx.TextCtrl(self.panel, 
                                   style=wx.TE_MULTILINE)

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.txtctrl, 1, wx.EXPAND)
        self.panel.SetSizer(sizer)
        self.CreateStatusBar() # For output display

        # Event Handlers
        self.txtctrl.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.txtctrl.Bind(wx.EVT_CHAR, self.OnChar)
        self.txtctrl.Bind(wx.EVT_KEY_UP, self.OnKeyUp)

    def OnKeyDown(self, event):
        """KeyDown event is sent first"""
        print "OnKeyDown Called"
        # Get information about the event and log it to
        # the StatusBar for display.
        key_code = event.GetKeyCode()
        raw_code = event.GetRawKeyCode()
        modifiers = event.GetModifiers()
        msg = "key:%d,raw:%d,modifers:%d" % \
              (key_code, raw_code, modifiers)
        self.PushStatusText("KeyDown: " + msg)

        # Must Skip the event to allow OnChar to be called
        event.Skip()

    def OnChar(self, event):
        """The Char event comes second and is
        where the character associated with the
        key is put into the control.
        """
        print "OnChar Called"
        modifiers = event.GetModifiers()
        key_code = event.GetKeyCode()
        # Beep at the user if the Shift key is down
        # and disallow input.
        if modifiers & wx.MOD_SHIFT:
            wx.Bell()
        elif chr(key_code) in "aeiou":elif unichr(key_code) in   "aeiou":
            # When a vowel is pressed append a
            # question mark to the end.
            self.txtctrl.AppendText("?")
        else:
            # Let the text go in to the buffer
            event.Skip()

    def OnKeyUp(self, event):
        """KeyUp comes last"""
        print "OnKeyUp Called"
        event.Skip()

```

当在这个窗口中输入时，按下*Shift*键将不允许输入文本，并且会将所有元音字母转换成问号。

## 它是如何工作的...

`KeyEvents` 是按照以下顺序由系统发送的：

+   `EVT_KEY_DOWN`

+   `EVT_CHAR`（仅适用于与字符相关联的键）

+   `EVT_KEY_UP`

重要的是要注意，我们在 `TextCtrl` 上调用了 `Bind` 而不是 `Frame`。这是必要的，因为 `KeyEvents` 只会发送到具有键盘焦点的控件，在这个窗口中将是 `TextCtrl`。

每个 `KeyEvent` 都附带有一些属性，用于指定在事件中按下了哪个键以及在此事件期间按下了哪些其他修饰键，例如 *Shift, Alt* 和 *Ctrl* 键。

在事件上调用 `Skip` 允许控制处理它，并调用链中的下一个处理器。例如，不在 `EVT_KEY_DOWN` 处理器中跳过事件将阻止 `EVT_CHAR` 和 `EVT_KEY_UP` 处理器被调用。

在这个示例中，当键盘上的一个键被按下时，我们的`OnKeyDown`处理程序首先被调用。我们在那里的所有操作只是将一条消息`print`到`stdout`，并在`StatusBar`中显示一些关于事件的详细信息，然后调用`Skip`。然后，在我们的`OnChar`处理程序中，我们通过检查事件修改器掩码中是否包含*Shift 键*来对大写字母进行一些简单的过滤。如果是，我们向用户发出蜂鸣声，并且不调用事件上的`Skip`，以防止字符出现在`TextCtrl`中。此外，作为一个修改事件行为的示例，我们将原始键码转换为字符字符串，并检查该键是否为元音。如果是元音键，我们只需在`TextCtrl`中插入一个问号。最后，如果事件在`OnChar`处理程序中被跳过，我们的`OnKeyUp`处理程序将被调用，在那里我们简单地打印一条消息到`stdout`以显示它已被调用。

## 还有更多...

一些控件需要在它们的构造函数中指定`wx.WANTS_CHARS`样式标志，以便接收字符事件。`Panel`类是要求使用这种特殊样式标志以接收`EVT_CHAR`事件的常见示例。通常，这用于在创建一个从`Panel`派生的自定义控件类型时执行特殊处理。

## 参见

+   本章中关于*使用验证器验证输入*的配方使用`KeyEvents`来执行输入验证。

# 使用 UpdateUI 事件

`UpdateUIEvents` 是框架定期发送的事件，以便允许应用程序更新其控件的状态。这些事件对于执行诸如根据应用程序的业务逻辑更改控件启用或禁用时间等任务非常有用。本食谱将展示如何使用 `UpdateUIEvents` 根据 UI 的当前上下文更新菜单项的状态。

## 如何做到这一点...

在本例中，我们创建了一个简单的窗口，其中包含一个`编辑菜单`和一个`文本控件`。`编辑菜单`中有三个项目，这些项目将根据`文本控件`当前的选中状态通过使用`UpdateUIEvents`来启用或禁用。

```py
class TextFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        super(TextFrame, self).__init__(parent,
                                        *args,
                                        **kwargs)

        # Attributes
        self.panel = wx.Panel(self)
        self.txtctrl = wx.TextCtrl(self.panel,
                                   value="Hello World",
                                   style=wx.TE_MULTILINE)

        # Layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.txtctrl, 1, wx.EXPAND)
        self.panel.SetSizer(sizer)
        self.CreateStatusBar() # For output display

        # Menu
        menub = wx.MenuBar()
        editm = wx.Menu()
        editm.Append(wx.ID_COPY, "Copy\tCtrl+C")
        editm.Append(wx.ID_CUT, "Cut\tCtrl+X")
        editm.Append(ID_CHECK_ITEM, "Selection Made?",
                     kind=wx.ITEM_CHECK)
        menub.Append(editm, "Edit")
        self.SetMenuBar(menub)

        # Event Handlers
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateEditMenu)

    def OnUpdateEditMenu(self, event):
        event_id = event.GetId()
        sel = self.txtctrl.GetSelection()
        has_sel = sel[0] != sel[1]
        if event_id in (wx.ID_COPY, wx.ID_CUT):
            event.Enable(has_sel)
        elif event_id == ID_CHECK_ITEM:
            event.Check(has_sel)
        else:
            event.Skip()

```

## 它是如何工作的...

`UpdateUIEvents` 是在空闲时间由框架定期发送的，以便应用程序检查控件的状态是否需要更新。我们的 `TextFrame` 类在其编辑菜单中有三个菜单项，这些菜单项将由我们的 `OnUpdateUI` 事件处理器管理。在 `OnUpdateUI` 中，我们检查事件的 ID 以确定事件是为哪个对象发送的，然后调用事件上的适当 `UpdateUIEvent` 方法来更改控件的状态。我们每个菜单项的状态取决于 `TextCtrl` 中是否有选择。调用 `TextCtrl` 的 `GetSelection` 方法将返回一个包含选择开始和结束位置的元组。当两个位置不同时，控件中有选择，我们将 `Enable` `Copy` 和 `Cut` 项目，或者在我们的 `Selection Made` 项目中设置勾选标记。如果没有选择，则项目将变为禁用或未勾选。

在事件对象上调用该方法以更新控件，而不是在控件本身上调用该方法，这是非常重要的，因为它将允许更高效地更新。请参阅 wxPython API 文档中的`UpdateUIEvent`，以查看可用的所有方法列表。

## 还有更多...

`UpdateUIEvent` 类中提供了一些静态方法，这些方法允许应用程序更改事件传递的行为。其中最显著的两个方法如下：

1.  `wx.UpdateUIEvent.SetUpdateInterval`

1.  `wx.UpdateUIEvent.SetMode`

`SetUpdateInterval` 可以用来配置 `UpdateUIEvents` 发送频率。它接受一个表示毫秒数的参数。如果你发现在你的应用程序中处理 `UpdateUIEvents` 存在明显的开销，这将非常有用。你可以使用这个方法来降低这些事件发送的速率。

`SetMode` 可以用来配置哪些窗口将接收事件的行为，通过设置以下模式之一：

| 模式 | 描述 |
| --- | --- |
| `wx.UPDATE_UI_PROCESS_ALL` | 处理所有窗口的 `UpdateUI` 事件 |
| `wx.UPDATE_UI_PROCESS_SPECIFIED` | 仅处理设置了 `WS_EX_PROCESS_UI_UPDATES` 额外样式标志的窗口的 `UpdateUI` 事件。 |

## 参见

+   本章中的 *使用 EventStack 管理事件处理器* 菜谱展示了如何集中管理 `UpdateUI` 事件。

# 操纵鼠标

`MouseEvents` 可以用来与用户在窗口内进行的鼠标位置变化和鼠标按钮点击进行交互。本教程将快速介绍一些在程序中可用的常见鼠标事件。

## 如何做到这一点...

在这里，我们以一个示例来创建一个简单的`Frame`类，其中包含一个`Panel`和一个按钮，以了解如何与`MouseEvents`进行交互。

```py
class MouseFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        super(MouseFrame, self).__init__(parent,
                                         *args,
                                         **kwargs)

        # Attributes
        self.panel = wx.Panel(self)
        self.btn = wx.Button(self.panel)

        # Event Handlers
        self.panel.Bind(wx.EVT_ENTER_WINDOW, self.OnEnter)
        self.panel.Bind(wx.EVT_LEAVE_WINDOW, self.OnLeave)
        self.panel.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)

    def OnEnter(self, event):
        """Called when the mouse enters the panel"""
        self.btn.SetForegroundColour(wx.BLACK)
        self.btn.SetLabel("EVT_ENTER_WINDOW")
        self.btn.SetInitialSize()

    def OnLeave(self, event):
        """Called when the mouse leaves the panel"""
        self.btn.SetLabel("EVT_LEAVE_WINDOW")
        self.btn.SetForegroundColour(wx.RED)

    def OnLeftDown(self, event):
        """Called for left down clicks on the Panel"""
        self.btn.SetLabel("EVT_LEFT_DOWN")

    def OnLeftUp(self, event):
        """Called for left clicks on the Panel"""
        position = event.GetPosition()
        self.btn.SetLabel("EVT_LEFT_UP")
        # Move the button
        self.btn.SetPosition(position - (25, 25))

```

## 它是如何工作的...

在这个菜谱中，我们利用了鼠标光标进入`Panel`和左键点击`Panel`的事件来修改我们的`Button`。当鼠标光标进入窗口区域时，会向其发送一个`EVT_ENTER_WINDOW`事件；相反，当光标离开窗口时，它会收到一个`EVT_LEAVE_WINDOW`事件。当鼠标进入或离开`Panel`的区域时，我们更新按钮的标签以显示发生了什么。当我们的`Panel`接收到左键点击事件时，我们将`Button`移动到点击发生的位置。

重要的是要注意，我们直接在 `Panel` 上调用了 `Bind` 而不是在 `Frame` 上。这很重要，因为 `MouseEvents` 不是 `CommandEvents`，所以它们只会被发送到它们起源的窗口，而不会在包含层次结构中传播。

## 还有更多...

有大量可以用来与其他鼠标操作交互的 `MouseEvents`。以下表格包含了对每个事件的快速参考：

| 鼠标事件 | 描述 |
| --- | --- |
| `wx.EVT_MOUSEWHEEL` | 发送鼠标滚轮滚动事件。请参阅属于 `MouseEvent` 类的 `GetWheelRotation` 和 `GetWheelDelta` 方法，了解如何处理此事件。 |
| `wx.EVT_LEFT_DCLICK` | 发送用于左鼠标按钮双击的事件。 |
| `wx.EVT_RIGHT_DOWN` | 当鼠标右键被按下时发送。 |
| `wx.EVT_RIGHT_UP` | 当鼠标右键被释放时发送。 |
| `wx.EVT_RIGHT_DCLICK` | 发送用于右键鼠标双击的事件。 |
| `wx.EVT_MIDDLE_DOWN` | 当鼠标中键被按下时发送。 |
| `wx.EVT_MIDDLE_UP` | 当鼠标中键被释放时发送。 |
| `wx.EVT_MIDDLE_DCLICK` | 发送于鼠标中键双击事件。 |
| `wx.EVT_MOTION` | 每当鼠标光标在窗口内移动时发送。 |
| `wx.EVT_MOUSE_EVENTS` | 此事件绑定器可用于获取所有鼠标相关事件的通告。 |

## 参见

+   本章中关于*理解事件传播*的配方讨论了不同类型的事件是如何传播的。

# 创建自定义事件类

有时有必要定义自己的事件类型来表示自定义操作，以及从应用的一个地方传输数据到另一个地方。本食谱将展示创建自定义事件类的两种方法。

## 如何做到这一点...

在这个简短片段中，我们使用两种不同的方法定义了两种新的事件类型：

```py
import wx
import wx.lib.newevent

# Our first custom event
MyEvent, EVT_MY_EVENT = wx.lib.newevent.NewCommandEvent()

# Our second custom event
myEVT_TIME_EVENT = wx.NewEventType()
EVT_MY_TIME_EVENT = wx.PyEventBinder(myEVT_TIME_EVENT, 1)
class MyTimeEvent(wx.PyCommandEvent):
    def __init__(self, id=0, time="12:00:00"):
         evttype = myEVT_TIME_EVENT
        super(MyTimeEvent, self).__init__(evttype, id)wx.PyCommandEvent.__init__(self, myEVT_TIME_EVENT, id)

        # Attributes
        self.time = time

    def GetTime(self):
        return self.time

```

## 它是如何工作的...

第一个示例展示了创建自定义事件类的最简单方法。来自 wx.lib.newevent 模块的 `NewCommandEvent` 函数将返回一个包含新事件类及其事件绑定器的元组。返回的类定义可以用来构建事件对象。当你只需要一个新的事件类型而不需要随事件发送任何自定义数据时，这种方法创建新事件类型最为有用。

为了使用事件对象，该对象需要通过事件循环进行处理。这里有两种方法可以实现，其中之一是`PostEvent`函数。`PostEvent`函数接受两个参数：第一个是需要接收事件的窗口，第二个是事件本身。例如，以下两行代码可以用来创建并发送我们的自定义`MyEvent`实例到`Frame:`：

```py
event = MyEvent(eventID)
wx.PostEvent(myFrame, event)

```

发送事件进行处理的第二种方式是使用窗口的`ProcessEvent`方法：

```py
event = MyEvent(eventID)
myFrame.GetEventHandler().ProcessEvent(event)

```

这两者的区别在于，`PostEvent`会将事件放入应用程序的事件队列中，以便在`MainLoop`的下一个迭代中处理，而`ProcessEvent`则会导致事件立即被处理。

第二种方法展示了如何从`PyCommandEvent`基类派生出一个新的事件类型。为了以这种方式创建一个事件，需要完成以下三个步骤。

1.  使用 `NewEventType` 函数定义一个新的事件类型。

1.  使用`PyEventBinder`类创建事件绑定对象，该对象将事件类型作为其第一个参数。

1.  定义用于创建事件对象的的事件类。

这个`MyTimeEvent`类可以保存一个自定义值，我们使用它来发送格式化的时间字符串。必须从`PyCommandEvent`派生这个类，这样我们附加到这个对象上的自定义 Python 数据和方法才能通过事件系统传递。

这些事件现在可以通过使用`PostEvent`函数或 Windows 的`ProcessEvent`方法发送到任何事件处理对象。这两种方法中的任何一种都会导致事件被分发到通过调用`Bind`与之关联的事件处理程序。

## 参见

+   第一章中的*理解继承限制*配方，*wxPython 入门*解释了为什么需要某些类的 Py 版本。

+   本章的*处理事件*配方讨论了事件处理器的使用。

# 使用 EventStack 管理事件处理器

`EventStack` 是 `wx.lib` 模块中的一个模块，它提供了一个用于 wx 应用对象混合类，可用于帮助管理 `Menu` 和 `UpdateUI` 事件的处理器。在具有多个顶级窗口或需要根据具有焦点的控件切换调用处理器的上下文的应用程序中，这可能很有用。本食谱将展示一个用于管理基于 `Frame` 的应用程序中事件的简单框架，这些应用程序使用了 `AppEventHandlerMixin` 类。本食谱附带的一个完整的工作示例展示了如何使用本食谱中的类。

## 如何做到这一点...

使用此代码，我们定义了两个相互协作的类。首先，我们定义了一个使用`AppEventHandlerMixin`的`App`基类。

```py
import wx
import wx.lib.eventStack as eventStack 

class EventMgrApp(wx.App, eventStack.AppEventHandlerMixin):
    """Application object base class that
    event handler managment.
    """
    def __init__(self, *args, **kwargs):
        eventStack.AppEventHandlerMixin.__init__(self)
        wx.App.__init__(self, *args, **kwargs)

class EventMgrFrame(wx.Frame):
    """Frame base class that provides event
    handler managment.
    """
    def __init__(self, parent, *args, **kwargs):
        super(EventMgrFrame, self).__init__(parent,
                                            *args,
                                            **kwargs)

        # Attributes
        self._menu_handlers = []
        self._ui_handlers = []

        # Event Handlers
        self.Bind(wx.EVT_ACTIVATE, self._OnActivate)

    def _OnActivate(self, event):
        """Pushes/Pops event handlers"""
        app = wx.GetApp()
        active = event.GetActive()
        if active:
            mode = wx.UPDATE_UI_PROCESS_SPECIFIED
            wx.UpdateUIEvent.SetMode(mode)
            self.SetExtraStyle(wx.WS_EX_PROCESS_UI_UPDATES)

            # Push this instances handlers
            for handler in self._menu_handlers:
                app.AddHandlerForID(*handler)

            for handler in self._ui_handlers:
                app.AddUIHandlerForID(*handler)
        else:
            self.SetExtraStyle(0)
            wx.UpdateUIEvent.SetMode(wx.UPDATE_UI_PROCESS_ALL)
            # Pop this instances handlers
            for handler in self._menu_handlers:
                app.RemoveHandlerForID(handler[0])

            for handler in self._ui_handlers:
                app.RemoveUIHandlerForID(handler[0])

    def RegisterMenuHandler(self, event_id, handler):
        """Register a MenuEventHandler
        @param event_id: MenuItem ID
        @param handler: Event handler function
        """
        self._menu_handlers.append((event_id, handler))

    def RegisterUpdateUIHandler(self, event_id, handler):
        """Register a controls UpdateUI handler
        @param event_id: Control ID
        @param handler: Event handler function
        """
        self._ui_handlers.append((event_id, handler))

```

## 它是如何工作的...

`EventMgrApp` 类只是一个用于创建使用 `AppEventHandlerMixin` 的应用程序对象的基类。这个 `mixin` 提供了添加和删除 `MenuEvent` 和 `UpdateUIEvent` 处理程序的事件处理程序的方法。

`EventMgrFrame` 类是用于派生的框架的基类。此类将处理使用其 `RegisterMenuHandler` 或 `RegisterUpdateUIHandler` 方法注册的事件处理器的添加、移除和绑定。这些方法负责将事件处理器添加到堆栈中，当 `Frame` 被激活或停用时，这些处理器将被推入或弹出。`AppEventHandlerMixin` 将内部管理这些处理器的绑定和解绑。

## 参见

+   本章中关于*使用 UpdateUI 事件*的配方详细讨论了`UpdateUI`事件。

# 使用验证器验证输入

`验证器` 是一种用于验证数据和过滤输入到控制器的事件的通用辅助类。大多数接受用户输入的控制都可以动态地与一个 `Validator` 关联。本食谱将展示如何创建一个 `Validator`，该 `Validator` 检查输入到窗口中的数据是否是一个在给定值范围内的整数。

## 如何做到这一点...

在这里，我们将定义一个用于`TextCtrl`的验证器，它可以用来验证输入的值是否为整数，并且是否在给定的范围内。

```py
import wx
import sys

class IntRangeValidator(wx.PyValidator):
    """An integer range validator for a TextCtrl"""
    def __init__(self, min_=0, max_=sys.maxint):
        """Initialize the validator
        @keyword min: min value to accept
        @keyword max: max value to accept

        """
        super(IntRangeValidator, self).__init__()
        assert min_ >= 0, "Minimum Value must be >= 0"
        self._min = min_
        self._max = max_

        # Event managment
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def Clone(self):
        """Required override"""
        return IntRangeValidator(self._min, self._max)

    def Validate(self, win):
        """Override called to validate the window's value.
        @return: bool
        """
        txtCtrl = self.GetWindow()
        val = txtCtrl.GetValue()
        isValid = False
        if val.isdigit():
            digit = int(val)
            if digit >= self._min and digit <= self._max:
                isValid = True

        if not isValid:
            # Notify the user of the invalid value
            msg = "Value must be between %d and %d" % \
                  (self._min, self._max)
            wx.MessageBox(msg,
                          "Invalid Value",
                          style=wx.OK|wx.ICON_ERROR)

        return isValid

    def OnChar(self, event):
        txtCtrl = self.GetWindow()
        key = event.GetKeyCode()
        isDigit = False
        if key < 256:
            isDigit = chr(key).isdigit()

        if key in (wx.WXK_RETURN,
                   wx.WXK_DELETE,
                   wx.WXK_BACK) or \
           key > 255 or isDigit:
            if isDigit:
                # Check if in range
                val = txtCtrl.GetValue()
                digit = chr(key)
                pos = txtCtrl.GetInsertionPoint()
                if pos == len(val):
                    val += digit
                else:
                    val = val[:pos] + digit + val[pos:]

                val = int(val)
                if val < self._min or val > self._max:
                    if not wx.Validator_IsSilent():
                        wx.Bell()
                    return

            event.Skip()
            return

        if not wx.Validator_IsSilent():
            # Beep to warn about invalid input
            wx.Bell()

        return

    def TransferToWindow(self):
         """Overridden to skip data transfer"""
         return True

    def TransferFromWindow(self):
         """Overridden to skip data transfer"""
         return True

```

## 它是如何工作的...

`Validator` 类包含多个需要重写以使其正常工作的虚拟方法。因此，为了访问类的虚拟方法感知版本，重要的是从 `PyValidator` 类而不是从 `Validator` 类派生一个子类。

所有 `Validator` 子类都必须重写 `Clone` 方法。这个方法只需简单地返回 `Validator` 的一个副本。

调用 `Validate` 方法来检查值是否有效。如果控件是模态对话框的子控件，在调用 `EndModal` 方法为“确定”按钮之前，将会调用此方法。这是通知用户任何输入问题的好时机。

`校验器`也可以绑定到它们窗口可能绑定的任何事件，并且可以用来过滤事件。在事件被发送到窗口之前，这些事件将被发送到`校验器`的`OnChar`方法，允许`校验器`过滤哪些事件被允许到达控件。

如果您希望在`Dialog`显示或关闭时仅进行验证，则可以重写`TransferToWindow`和`TransferFromWindow`方法。当`Dialog`显示时，将调用`TransferToWindow`，而当`Dialog`关闭时，将调用`TransferFromWIndow`。从任一方法返回`True`表示数据有效，而返回`False`则表示存在无效数据。

## 参见

+   第一章中的*理解继承限制*配方讨论了类和重写虚拟方法使用 Python 版本的情况。

+   本章中关于*处理关键事件*的配方详细讨论了`KeyEvents`。

# 处理苹果事件

AppleEvents 是 Macintosh 操作系统使用的高级系统事件，用于在进程之间传递信息。为了处理诸如打开拖放到应用程序图标上的文件等操作，应用程序必须处理这些事件。wxPython 应用程序对象通过在应用程序对象中实现虚拟覆盖，提供了一些对最常见事件的内置支持。本食谱将展示如何创建一个可以利用内置的并且相对隐藏的事件回调函数的应用程序对象。

### 注意事项

这是一个针对 OS X 的特定配方，对其他平台将没有任何影响。

## 如何做到这一点...

这个小示例应用程序展示了在`App`中可用的所有内置回调方法，用于处理一些常见的`AppleEvents`。

```py
import wx

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="AppleEvents")
        self.SetTopWindow(self.frame)
        self.frame.Show()

        return True

    def MacNewFile(self):
        """Called for an open-application event"""
        self.frame.PushStatusText("MacNewFile Called")

    def MacOpenFile(self, filename):
        """Called for an open-document event"""
        self.frame.PushStatusText("MacOpenFile: %s" % \
                                  filename)

    def MacOpenURL(self, url):
        """Called for a get-url event"""
        self.frame.PushStatusText("MacOpenURL: %s" % url)

    def MacPrintFile(self, filename):
        """Called for a print-document event"""
        self.frame.PushStatusText("MacPrintFile: %s" % \
                                   filename)

    def MacReopenApp(self):
        """Called for a reopen-application event"""
        self.frame.PushStatusText("MacReopenApp")
        # Raise the application from the Dock
        if self.frame.IsIconized():
            self.frame.Iconize(False)
        self.frame.Raise()

```

## 它是如何工作的...

对于一些常见的 `AppleEvents`，有五种内置的处理方法。要在您的应用程序中使用它们，只需在应用程序对象中覆盖它们，如前所述。由于应用程序对这些事件的响应非常特定于应用程序本身，这个方法除了在调用方法时向框架的状态栏报告之外，并没有做太多的事情。

应该实现的最常见的两个事件是 `MacOpenFile` 和 `MacReopenApp` 方法，因为这些是实现在 OS X 上应用程序中标准预期行为所必需的。当用户将文件拖放到应用程序的 Dock 图标上时，会调用 `MacOpenFile`。在这种情况下，它将作为参数传递文件的路径。当用户左键单击正在运行的应用程序的 Dock 图标时，会调用 `MacReopenApp`。如配方中所示，这用于将应用程序带到前台，或者从 Dock 中的最小化状态中恢复出来。

## 还有更多...

在 wxPython 应用程序中添加对更多 `AppleEvents` 的支持是可能的，尽管这不是一项特别容易的任务，因为它需要编写一个本地扩展模块来捕获事件，阻塞 wx 的 `EventLoop`，然后在处理事件后将 Python 解释器的状态恢复到 wx。wxPython Wiki 中有一个相当不错的示例可以作为起点（见 [`wiki.wxpython.org/Catching%20AppleEvents%20in%20wxMAC`](http://wiki.wxpython.org/Catching%20AppleEvents%20in%20wxMAC)），如果你发现自己需要走这条路的话。

## 参见

+   第一章中的*理解继承限制*配方，*使用 wxPython 入门*包含了更多关于重写虚方法的信息。

+   第十二章中的*针对 OS X 优化*配方，在*应用程序基础设施*部分包含了更多关于使 wxPython 应用程序在 OS X 上运行良好的信息。
