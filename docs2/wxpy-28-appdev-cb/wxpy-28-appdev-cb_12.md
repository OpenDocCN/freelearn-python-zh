# 第十二章 建立和管理分发应用

在本章中，我们将涵盖：

+   使用 `StandardPaths`

+   保持用户界面状态

+   使用 `SingleInstanceChecker`

+   异常处理

+   优化针对 OS X

+   支持国际化

+   分发应用程序

# 简介

应用程序的基础设施为应用程序的内部运作提供了骨架，这些内部运作通常是用户无法直接看到的，但对于应用程序的功能至关重要。这包括诸如存储配置和外部数据文件、错误处理和安装等方面。这些领域的每一个都提供了重要的功能，并有助于提高应用程序的可用性和最终用户对应用程序的整体印象。在本章中，我们将深入探讨这些主题以及更多内容，以便为您提供适当的工具来帮助构建和分发您的应用程序。

# 与 StandardPaths 一起工作

几乎每个非平凡的应用都需要在程序使用之间存储数据以及加载资源，如图片。问题是把这些东西放在哪里？操作系统和用户期望在这些平台上找到这些文件的位置可能会有所不同。这个菜谱展示了如何使用 `wx.StandardPaths` 来管理应用程序的配置和资源文件。

## 如何做到这一点...

在这里，我们将创建一个薄封装实用类来帮助管理应用程序的配置文件和数据。构造函数将确保任何预定义的目录已经在系统配置存储位置设置好。

```py
class ConfigHelper(object):
    def __init__(self, userdirs=None):
        """@keyword userdirs: list of user config
                              subdirectories names
        """
        super(ConfigHelper, self).__init__()

        # Attributes
        self.userdirs = userdirs

        # Setup
        self.InitializeConfig()

    def InitializeConfig(self):
        """Setup config directories"""
        # Create main user config directory if it does
        # not exist.
        datap = wx.StandardPaths_Get().GetUserDataDir()
        if not os.path.exists(datap):
            os.mkdir(datap)
        # Make sure that any other application specific
        # config subdirectories have been created.
        if self.userdirs:
            for dname in userdirs:
                self.CreateUserCfgDir(dname)

```

在这里，我们添加一个辅助函数来在当前用户的资料目录中创建一个目录：

```py
    def CreateUserCfgDir(self, dirname):
        """Create a user config subdirectory"""
        path = wx.StandardPaths_Get().GetUserDataDir()
        path = os.path.join(path, dirname)
        if not os.path.exists(path):
            os.mkdir(path)

```

下一个函数可以用来获取用户数据目录中文件或目录的绝对路径：

```py
    def GetUserConfigPath(self, relpath):
        """Get the path to a resource file
        in the users configuration directory.
        @param relpath: relative path (i.e config.cfg)
        @return: string
        """
        path = wx.StandardPaths_Get().GetUserDataDir()
        path = os.path.join(path, relpath)
        return path

```

最后，本类中的最后一个方法可以用来检查指定的配置文件是否已经创建：

```py
    def HasConfigFile(self, relpath):
        """Does a given config file exist"""
        path = self.GetUserConfigPath(relpath)
        return os.path.exists(path)

```

## 它是如何工作的...

`ConfigHelper` 类只是对一些 `StandardPaths` 方法进行了一个简单的封装，以便使其使用起来更加方便。当对象被创建时，它会确保用户数据目录及其任何应用程序特定的子目录已经被创建。`StandardPaths` 单例使用应用程序的名称来确定用户数据目录的名称。因此，在创建 `App` 对象并使用 `SetAppName` 设置其名称之前，等待是很重要的。

```py
class SuperFoo(wx.App):
    def OnInit(self):
        self.SetAppName("SuperFoo")
        self.config = ConfigHelper()
        self.frame = SuperFooFrame(None, title="SuperFoo")
        self.frame.Show()
        return True

    def GetConfig(self):
        return self.config

```

`CreateUserCfgDir` 提供了一种方便的方法在用户的主要配置目录内创建一个新的目录。`GetUserConfigPath` 可以通过使用相对于主目录的路径来获取配置目录或子目录中文件或目录的完整路径。最后，`HasConfigFile` 是一种简单的方法来检查用户配置文件中是否存在文件。

## 还有更多...

`StandardPaths` 单例提供了许多其他方法来获取其他系统和特定安装的安装路径。以下表格描述了这些附加方法：

| 方法 | 描述 |
| --- | --- |
| `GetConfigDir()` | 返回系统级配置目录 |
| `GetDataDir()` | 返回应用程序的全局（非用户特定）数据目录 |
| `GetDocumentsDir()` | 返回当前用户的文档目录 |
| `GetExecutablePath()` | 返回当前运行的可执行文件的路径 |
| `GetPluginsDir()` | 返回应用程序插件应驻留的路径 |
| `GetTempDir()` | 返回系统 `TEMP` 目录的路径 |
| `GetUserConigDir()` | 返回当前用户的配置目录路径 |

## 参见

+   请参阅第九章中的*创建单例*配方，在*设计方法和技巧*中讨论了单例，例如`StandardPaths`对象，是什么。

+   有关存储确认信息的更多信息，请参阅本章中的*持久化 UI 状态*配方。

# 保持用户界面状态

许多应用程序的共同特点是在程序启动之间能够记住并恢复它们的窗口大小和位置。这不是工具包提供的内置功能，因此这个配方将创建一个简单的`Frame`基类，该类将在应用程序使用之间自动保存和恢复其在桌面上的大小和位置。

## 如何做到这一点...

这个例子展示了创建一个`Frame`类的一种方法，该类将在程序运行之间自动恢复其位置和大小：

```py
class PersistentFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(PersistentFrame, self).__init__(*args, **kwargs)

        # Setup
        wx.CallAfter(self.RestoreState)

        # Event Handlers
        self.Bind(wx.EVT_CLOSE, self._OnClose)

```

在这里，我们处理 `EVT_CLOSE` 事件，用于在 `Frame` 关闭时，将其位置和大小保存到 `Config` 对象中，在 Windows 上是注册表，在其他平台上是 `.ini` 文件：

```py
    def _OnClose(self, event):
        position = self.GetPosition()
        size = self.GetSize()
        cfg = wx.Config()
        cfg.Write('pos', repr(position.Get()))
        cfg.Write('size', repr(size.Get()))
        event.Skip()

```

`RestoreState` 方法恢复当前存储的窗口状态，或者如果没有存储任何内容，则恢复默认状态：

```py
    def RestoreState(self):
        """Restore the saved position and size"""
        cfg = wx.Config()
        name = self.GetName()
        position = cfg.Read(name + '.pos',
                            repr(wx.DefaultPosition))
        size = cfg.Read(name + '.size',
                        repr(wx.DefaultSize))
        # Turn strings back into tuples
        position = eval(position)
        size = eval(size)
        # Restore settings to Frame
        self.SetPosition(position)
        self.SetSize(size)

```

## 它是如何工作的...

应将`PersistentFrame`用作任何需要在退出时持久化其大小和位置的`Frame`的应用程序的基础类。这个类的工作方式相当简单，所以让我们快速了解一下它是如何工作的。

首先，为了节省其大小和位置，`PersisistentFrame` 将事件处理器绑定到 `EVT_CLOSE`。当用户关闭 `Frame` 时，将调用其 `_OnClose` 方法。在这个事件处理器中，我们简单地获取 `Frame` 的当前大小和位置，并将其保存到一个 `wx.Config` 对象中，在 Windows 上这将作为注册表，在其他平台上则是一个 `.ini` 文件。

相反，当创建`PersistentFrame`时，它会尝试从配置中读取之前保存的大小和位置。这发生在`RestoreState`方法中，该方法通过`CallAfter`来启动。这样做是为了确保我们不会在`Frame`创建之后恢复设置，这样如果子类设置了一些默认大小，它们就不会覆盖用户最后留下的最后状态。在`RestoreState`中，如果为`Frame`存储了信息，它将使用`eval`函数加载字符串并将它们转换回元组，然后简单地应用这些设置。

## 还有更多...

为了简化，我们只是使用了`wx.Config`来存储应用程序运行之间的设置。我们也可以使用`StandardPaths`并编写我们自己的配置文件到用户的配置目录，就像我们在之前的菜谱中所做的那样，以确保这些信息被保存在用户期望的位置。

## 参见

+   请参阅本章中的 *使用 StandardPaths* 菜单以获取有关另一个可帮助存储和定位配置信息的类的信息。

# 使用 SingleInstanceChecker

有时可能希望在任何给定时间只允许一个应用程序实例存在。`SingleInstanceChecker` 类提供了一种检测应用程序是否已有实例正在运行的方法。这个配方创建了一个 `App` 类，它使用 `SingleInstanceChecker` 来确保计算机上一次只运行一个应用程序实例，并且还使用了一个简单的 IPC 机制，允许任何后续的应用程序实例向原始实例发送消息，告知其打开一个新窗口。

## 如何做到这一点...

在这里，我们将创建一个`App`基类，确保同一时间只运行一个进程实例，并支持一个简单的基于套接字的进程间通信机制，以通知已运行的实例有一个新的实例尝试启动：

```py
import wx
import threading
import socket
import select

class SingleInstApp(wx.App):
    """App baseclass that only allows a single instance to
    exist at a time.
    """
    def __init__(self, *args, **kwargs):
        super(SingleInstApp, self).__init__(*args, **kwargs)

        # Setup (note this will happen after subclass OnInit)
        instid = "%s-%s" % (self.GetAppName(), wx.GetUserId())
        self._checker = wx.SingleInstanceChecker(instid)
        if self.IsOnlyInstance():
           # First instance so start IPC server
           self._ipc = IpcServer(self, instid, 27115)
           self._ipc.start()
           # Open a window
           self.DoOpenNewWindow()
        else:
            # Another instance so just send a message to
            # the instance that is already running.
            cmd = "OpenWindow.%s" % instid
            if not SendMessage(cmd, port=27115):
                print "Failed to send message!"

    def __del__(self):
        self.Cleanup()

```

当应用程序退出时，需要显式删除`SingleInstanceChecker`以确保它创建的文件锁被释放：

```py
    def Cleanup(self):
        # Need to cleanup instance checker on exit
        if hasattr(self, '_checker'):
            del self._checker
        if hasattr(self, '_ipc'):
            self._ipc.Exit()

    def Destroy(self):
        self.Cleanup()
        super(SingleInstApp, self).Destroy()

    def IsOnlyInstance(self):
        return not self._checker.IsAnotherRunning()

    def DoOpenNewWindow(self):
        """Interface for subclass to open new window
        on ipc notification.
        """
        pass

```

`IpcServer` 类通过在机器的本地回环上打开一个套接字连接来实现进程间通信。这已被实现为一个循环等待消息的后台线程，直到收到退出指令：

```py
class IpcServer(threading.Thread):
    """Simple IPC Server"""
    def __init__(self, app, session, port):
        super(IpcServer, self).__init__()

        # Attributes
        self.keeprunning = True
        self.app = app
        self.session = session
        self.socket = socket.socket(socket.AF_INET,
                                    socket.SOCK_STREAM)

        # Setup TCP socket
        self.socket.bind(('127.0.0.1', port))
        self.socket.listen(5)
        self.setDaemon(True)

```

`run` 方法运行服务器线程的主循环，检查套接字是否有消息，并使用 `CallAfter` 通知 `App` 调用其 `DoOpenNewWindow` 方法，当服务器接收到 `'OpenWindow'` 命令时：

```py
    def run(self):
        """Run the server loop"""
        while self.keeprunning:
            try:
                client, addr = self.socket.accept()

                # Read from the socket
                # blocking up to 2 seconds at a time
                ready = select.select([client,],[], [],2)
                if ready[0]:
                    recieved = client.recv(4096)

                if not self.keeprunning:
                    break

                # If message ends with correct session
                # ID then process it.
                if recieved.endswith(self.session):
                    if recieved.startswith('OpenWindow'):
                        wx.CallAfter(self.app.DoOpenNewWindow)
                    else:
                        # unknown command message
                        pass
                recieved = ''
            except socket.error, msg:
                print "TCP error! %s" % msg
                break

        # Shutdown the socket
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            pass

        self.socket.close()

    def Exit(self):
        self.keeprunning = False

```

`SendMessage` 函数用于打开到 `IpcServer` 的套接字的客户端连接并发送指定的消息：

```py
def SendMessage(message, port):
    """Send a message to another instance of the app"""
    try:
        # Setup the client socket
        client = socket.socket(socket.AF_INET,
                               socket.SOCK_STREAM)
        client.connect(('127.0.0.1', port))
        client.send(message)
        client.shutdown(socket.SHUT_RDWR)
        client.close()
    except Exception, msg:
        return False
    else:
        return True

```

与本章配套的代码中包含了一个完整的运行应用程序，展示了如何使用上述框架。为了测试它，尝试在同一台计算机上启动多个应用程序实例，并观察只有原始进程正在运行，并且每次后续启动都会在原始进程中打开一个新窗口。

## 它是如何工作的...

在这个菜谱中，我们在这段代码里塞进了许多内容，所以让我们来了解一下每个类是如何工作的。

`SingleInstApp` 类创建一个 `SingleInstanceChecker` 对象，以便能够检测是否有另一个应用程序实例正在运行。作为 `SingleInstanceChecker` 的 ID 的一部分，我们使用了用户的登录 ID，以确保实例检查器只检查同一用户启动的其他实例。

在我们的 `SingleInstanceApp` 对象的 `__init__` 方法中，重要的是要意识到当派生类被初始化时将要发生的操作顺序。调用基类 `wx.App` 的 `__init__` 方法将会导致派生类的虚拟 `OnInit` 被调用，然后之后 `SingleInstApp` 的 `__init__` 方法中的其余代码将执行。如果它检测到这是应用程序的第一个运行实例，它将创建并启动我们的 `IpcServer`。如果不是，它将简单地创建并发送一个简单的字符串命令到另一个已经运行的 `IpcServer` 对象，告诉它通知其他应用程序实例创建一个新窗口。

在继续查看 `IpcServer` 类之前，使用 `SingleInstanceChecker` 时需要牢记的一个重要事项是，当你完成使用后，需要显式地删除它。如果不删除，它用于确定另一个实例是否活跃的文件锁可能永远不会释放，这可能会在程序未来的启动中引起问题。

`IpcServer` 类是一个简单的从 `Thread` 派生的类，它使用 TCP 套接字进行进程间通信。正如所述，第一个启动的 `SingleInstanceApp` 将创建此服务器的实例。服务器将在自己的线程中运行，检查套接字上的消息。`IpcServer` 线程的 `run` 方法只是运行一个循环，检查套接字上的新数据。如果它能够读取一条消息，它将检查消息的最后部分是否与创建 `App's SingleInstanceChecker` 时使用的密钥匹配，以确保命令来自应用程序的另一个实例。我们目前只为我们的简单 IPC 协议设计了支持单个 `'OpenWindow'` 命令，但它可以很容易地扩展以支持更多。在接收到 `OpenWindow` 消息后，`IpcServer` 将使用 `CallAfter` 调用 `SingleInstanceApp` 的接口方法 `DoOpenNewWindow`，通知应用程序打开其主窗口的新实例。

这个小框架的最后一部分是`SendMessage`函数，它被用作客户端方法来连接并向`IpcServer`发送消息。

## 参见

+   请参阅第一章中的*理解继承限制*配方，在*wxPython 入门*中解释了 wxPython 类中重写虚拟方法的内容。

+   查阅第十一章中的*理解线程安全性*配方，以获取更多关于在 wxPython GUI 中处理线程的信息，详见*响应式界面*。

# 异常处理

即使在看似简单的应用中，也可能难以考虑到应用中可能发生的所有可能的错误条件。本食谱展示了如何处理未处理的异常，以及如何在应用退出之前向用户显示通知，让他们知道发生了意外的错误。

## 如何做到这一点...

对于这个菜谱，我们将展示如何创建一个简单的异常钩子来处理和通知用户在程序运行过程中发生的任何意外错误：

```py
import wx
import sys
import traceback

def ExceptionHook(exctype, value, trace):
    """Handler for all unhandled exceptions
    @param exctype: Exception Type
    @param value: Error Value
    @param trace: Trace back info
    """
    # Format the traceback
    exc = traceback.format_exception(exctype, value, trace)
    ftrace = "".join(exc)
    app = wx.GetApp()
    if app:
        msg = "An unexpected error has occurred: %s" % ftrace
        wx.MessageBox(msg, app.GetAppName(),
                      style=wx.ICON_ERROR|wx.OK)
        app.Exit()
    else:
        sys.stderr.write(ftrace)

class ExceptionHandlerApp(wx.App):
    def OnInit(self):
        sys.excepthook = ExceptionHook
        return True

```

## 它是如何工作的...

这个菜谱展示了一种非常简单的方法来创建一个异常钩子，以捕获应用程序中的未处理异常。在应用程序启动期间，我们所需做的所有事情就是用我们自己的`ExceptionHook`函数替换默认的`excepthook`函数。然后，每当应用程序中抛出未处理的异常时，`ExceptionHook`函数就会被调用。在这个函数中，我们只是弹出一个`MessageBox`来显示发生了意外错误，然后告诉`MainLoop`退出。

## 还有更多...

本例的目的是展示如何优雅地处理这些错误的处理过程。因此，我们通过仅使用一个`MessageBox`使其保持相当简单。很容易扩展和定制这个例子，以便记录错误，或者允许用户向应用程序的开发者发送通知，以便调试错误。

# 优化针对 OS X

在 wxPython 应用程序中，有许多事情可以做到，以帮助它在 Macintosh OS X 系统上运行时更好地适应。用户对 OS X 上的应用程序有一些期望，这个菜谱展示了确保您的应用程序在 OS X 以及其他平台上运行良好和外观美观的一些操作。这包括标准菜单和菜单项的正确定位、主窗口的行为，以及如何启用一些 Macintosh 特定的功能。

## 如何做到这一点...

作为考虑一些事项的例子，我们将创建一个简单的应用程序，展示如何使应用程序符合 Macintosh UI 标准：

```py
import wx
import sys

class OSXApp(wx.App):
    def OnInit(self):
        # Enable native spell checking and right
        # click menu for Mac TextCtrl's
        if wx.Platform == '__WXMAC__':
            spellcheck = "mac.textcontrol-use-spell-checker"
            wx.SystemOptions.SetOptionInt(spellcheck, 1)
        self.frame = OSXFrame(None,
                              title="Optimize for OSX")
        self.frame.Show()
        return True

    def MacReopenApp(self):
        self.GetTopWindow().Raise()

class OSXFrame(wx.Frame):
    """Main application window"""
    def __init__(self, *args, **kwargs):
        super(OSXFrame, self).__init__(*args, **kwargs)

        # Attributes
        self.textctrl = wx.TextCtrl(self,
                                    style=wx.TE_MULTILINE)

        # Setup Menus
        mb = wx.MenuBar()
        fmenu = wx.Menu()
        fmenu.Append(wx.ID_OPEN)
        fmenu.Append(wx.ID_EXIT)
        mb.Append(fmenu, "&File")
        emenu = wx.Menu()
        emenu.Append(wx.ID_COPY)
        emenu.Append(wx.ID_PREFERENCES)
        mb.Append(emenu, "&Edit")
        hmenu = wx.Menu()
        hmenu.Append(wx.NewId(), "&Online Help...")
        hmenu.Append(wx.ID_ABOUT, "&About...")
        mb.Append(hmenu, "&Help")

        if wx.Platform == '__WXMAC__':
            # Make sure we don't get duplicate
            # Help menu since we used non standard name
            app = wx.GetApp()
            app.SetMacHelpMenuTitleName("&Help")

        self.SetMenuBar(mb)
        self.SetInitialSize()

if __name__ == '__main__':
    app = OSXApp(False)
    app.MainLoop()

```

## 它是如何工作的...

这个简单的应用程序创建了一个包含`MenuBar`和`TextCtrl`的`Frame`，并演示了在准备部署到 Macintosh 系统上的应用程序时需要注意的一些事项。

从我们的 `OSXApp` 对象的 `OnInit` 方法开始，我们使用了 `SystemOptions` 单例来启用 OS X 上 `TextCtrl` 对象的本地上下文菜单和拼写检查功能。此选项默认是禁用的；将其设置为 `1` 可以启用它。同样，在我们的 `OSXApp` 类中，我们重写了 `MacReopenApp` 方法，这是一个当应用程序的 dock 图标被点击时发生的 `AppleEvent` 的回调。我们重写它以确保这个点击将使我们的应用程序主窗口被带到前台，正如预期的那样。

接下来，在我们的`OSXFrame`类中，可以看到对于`菜单`部分有一些特殊处理是必要的。所有原生 OS X 应用程序在其菜单中都有一些共同元素。所有应用程序都有一个帮助菜单、一个窗口菜单和一个应用程序菜单。如果你的应用程序需要创建自定义的帮助或窗口菜单，那么需要采取一些额外的步骤来确保它们在 OS X 上能够按预期工作。在我们之前的示例中，我们创建了一个包含标题中助记符加速器的自定义帮助菜单，以便 Windows/GTK 在键盘导航中使用。由于菜单标题与默认标题不同，我们需要在`App`对象上调用`SetMacHelpMenuTitleName`，以便它知道我们的帮助菜单应该被使用。如果我们省略这一步，我们的应用程序最终会在 OS X 的`MenuBar`中显示两个帮助菜单。另一个需要注意的重要事项是尽可能使用库存 ID 来为菜单项设置。特别是关于“关于”、“退出”和“首选项”的条目，在 OS X 的应用程序菜单下总是会被显示。通过使用这些项目的库存 ID，wxPython 将确保它们在每个平台上都出现在正确的位置。

## 还有更多...

以下包含一些额外的针对 Macintosh 系统的方法和注意事项，供快速查阅。

### wx.App Macintosh 特定方法

有一些其他额外的针对 Macintosh 系统的辅助方法属于`App`对象，可以用来自定义处理三个特殊菜单项。当应用程序在其他平台上运行时，这些方法将不会执行任何操作。

| 方法 | 描述 |
| --- | --- |
| `SetMacAboutMenuItemId` | 将用于识别“关于”菜单项的 ID 从`ID_ABOUT`更改为自定义值 |
| `SetMacExitMenuItemId` | 将用于识别 `Exit` 菜单项的 ID 从 `ID_EXIT` 更改为自定义值 |
| `SetMacPreferencesMenuItemId` | 将用于识别 `Preferences` 菜单项的 ID 从 `ID_PREFERENCES` 更改为自定义值 |
| `SetMacSupportPCMenuShortcuts` | 启用在 OS X 上使用菜单快捷键的功能 |

### wx.MenuBar

可以通过使用 `wx.MenuBar` 的静态 `SetAutoWindowMenu` 方法在 OS X 上禁用 Windows 菜单的自动创建。在创建 `MenuBar` 之前调用 `SetAutoWindowMenu` 并传入 `False` 值将阻止 Windows 菜单的创建。

## 参见

+   请参阅第一章中的*使用股票 ID*配方，以了解有关使用内置股票 ID 的详细讨论。

+   请参阅第二章中的*处理 Apple 事件*配方，了解在 wxPython 应用程序中如何处理 AppleEvents 的示例。

+   请参阅本章中的 *分发应用程序* 菜单，以了解如何在 OS X 上分发应用程序。

# 支持国际化

在我们今天所生活的这个互联互通的世界中，在开发应用程序界面时考虑国际化非常重要。在设计一个从一开始就完全支持国际化的应用程序时，损失非常小，但如果您不这样做，损失将会很大。这个指南将展示如何设置一个应用程序以使用 wxPython 内置的界面翻译支持。

## 如何做到这一点...

下面，我们将创建一个完整的示例应用程序，展示如何在 wxPython 应用程序的用户界面中支持本地化。首先要注意的是，我们下面使用的 `wx.GetTranslation` 的别名，用于将应用程序中所有的界面字符串包装起来：

```py
import wx
import os

# Make a shorter alias
_ = wx.GetTranslation

```

接下来，在创建我们的 `App` 对象期间，我们创建并保存对一个 `Locale` 对象的引用。然后我们告诉 `Locale` 对象我们存放翻译文件的位置，这样它就知道在调用 `GetTranslation` 函数时去哪里查找翻译。

```py
class I18NApp(wx.App):
    def OnInit(self):
        self.SetAppName("I18NTestApp")
        # Get Language from last run if set
        config = wx.Config()
        language = config.Read('lang', 'LANGUAGE_DEFAULT')

        # Setup the Locale
        self.locale = wx.Locale(getattr(wx, language))
        path = os.path.abspath("./locale") + os.path.sep
        self.locale.AddCatalogLookupPathPrefix(path)
        self.locale.AddCatalog(self.GetAppName())

        # Local is not setup so we can create things that
        # may need it to retrieve translations.
        self.frame = TestFrame(None,
                               title=_("Sample App"))
        self.frame.Show()
        return True

```

然后，在剩余部分，我们创建一个简单的用户界面，这将允许应用程序在英语和日语之间切换语言：

```py
class TestFrame(wx.Frame):
    """Main application window"""
    def __init__(self, *args, **kwargs):
        super(TestFrame, self).__init__(*args, **kwargs)

        # Attributes
        self.panel = TestPanel(self)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.SetInitialSize((300, 300))

class TestPanel(wx.Panel):
    def __init__(self, parent):
        super(TestPanel, self).__init__(parent)

        # Attributes
        self.closebtn = wx.Button(self, wx.ID_CLOSE)
        self.langch = wx.Choice(self,
                                choices=[_("English"),
                                         _("Japanese")])

        # Layout
        self.__DoLayout()

        # Event Handler
        self.Bind(wx.EVT_CHOICE, self.OnChoice)
        self.Bind(wx.EVT_BUTTON,
                  lambda event: self.GetParent().Close())

    def __DoLayout(self):
        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(self, label=_("Hello"))
        hsizer.AddStretchSpacer()
        hsizer.Add(label, 0, wx.ALIGN_CENTER)
        hsizer.AddStretchSpacer()

        langsz = wx.BoxSizer(wx.HORIZONTAL)
        langlbl = wx.StaticText(self, label=_("Language"))
        langsz.AddStretchSpacer()
        langsz.Add(langlbl, 0, wx.ALIGN_CENTER_VERTICAL)
        langsz.Add(self.langch, 0, wx.ALL, 5)
        langsz.AddStretchSpacer()

        vsizer.AddStretchSpacer()
        vsizer.Add(hsizer, 0, wx.EXPAND)
        vsizer.Add(langsz, 0, wx.EXPAND|wx.ALL, 5)
        vsizer.Add(self.closebtn, 0, wx.ALIGN_CENTER)
        vsizer.AddStretchSpacer()

        self.SetSizer(vsizer)

    def OnChoice(self, event):
        sel = self.langch.GetSelection()
        config = wx.Config()
        if sel == 0:
            val = 'LANGUAGE_ENGLISH'
        else:
            val = 'LANGUAGE_JAPANESE'
        config.Write('lang', val)

if __name__ == '__main__':
    app = I18NApp(False)
    app.MainLoop()

```

## 它是如何工作的...

上面的简单示例展示了如何在 wxPython 应用程序中利用翻译支持。在 `Choice` 控件中更改选定的语言并重新启动应用程序，将改变界面字符串在英语和日语之间的转换。利用翻译相当简单，所以让我们看看使其工作的重要部分。

首先，我们为函数 `wx.GetTranslation` 创建了一个别名 `_`，这样在编写时更短，也更易于阅读。这个函数应该被包裹在应用中任何将要显示给用户的界面字符串周围。

接下来，在我们的应用程序的 `OnInit` 方法中，我们做了一些事情来设置适当的区域信息，以便加载配置的翻译。首先，我们创建了一个 `Locale` 对象。保留对这个对象的引用是必要的，以确保它不会被垃圾回收。因此，我们将它保存到 `self.locale`。接下来，我们设置了 `Locale` 对象，让它知道我们的翻译资源文件所在的位置，首先通过调用 `AddCatalogLookupPathPrefix` 并传入我们保存翻译文件的目录。然后，我们通过调用 `AddCatalog` 并传入我们的应用程序对象名称来告诉它应用程序的资源文件名称。为了加载翻译，需要在目录查找路径前缀目录下的每个语言都需要以下目录结构：

```py
Lang_Canonical_Name/LC_MESSAGES/CatalogName.mo

```

因此，例如，对于我们的应用程序的日语翻译，我们在 locale 目录下有以下目录结构。

```py
ja_JP/LC_MESSAGES/I18NTestApp.mo

```

在创建`Locale`对象之后，任何对`GetTranslation`的调用都将使用该区域设置从`gettext`目录文件中加载适当的字符串。

## 还有更多...

wxPython 使用 `gettext 格式化` 文件来加载字符串资源。对于每种翻译，都有两个文件。`.po` 文件（可移植对象）是用于创建默认字符串到翻译版本映射的文件，需要编辑此文件。另一个文件是 `.mo` 文件（机器对象），它是 `.po` 文件的编译版本。要将 `.po` 文件编译成 `.mo` 文件，您需要使用 `msgfmt` 工具。这是任何 Linux 平台上 `gettext` 的一部分。它也可以通过 `fink` 在 OS X 上安装，通过 `Cygwin` 在 Windows 上安装。以下命令行语句将从给定的输入 `.po` 文件生成 `.mo` 文件。

```py
msgfmt ja_JP.po

```

# 分发应用程序

一旦你正在开发的应用程序完成，就需要准备一种方法将应用程序分发给用户。wxPython 应用程序可以像其他 Python 应用程序或脚本一样进行分发，通过创建一个`setup.py`脚本并使用`distutils`模块的`setup`函数。然而，这个菜谱将专注于如何通过创建一个使用`py2exe`和`py2app`分别针对两个目标平台构建的构建脚本，来创建 Windows 和 OS X 的独立可执行文件。创建一个独立的应用程序使得用户在自己的系统上安装应用程序变得更加容易，这意味着更多的人可能会使用它。

## 准备就绪

要构建独立的可执行文件，除了 wxPython 之外，还需要一些扩展模块。因此，如果您还没有这样做，您将需要安装`py2exe`（Windows）或`py2app`（OS X）。

## 如何做到这一点...

在这里，我们将创建一个简单的 `setup.py` 模板，通过一些简单的自定义设置，可以用来构建适用于大多数 wxPython 应用程序的 Windows 和 OS X 二进制文件。顶部这里的 **应用程序信息** 部分可以被修改，以指定应用程序的名称和其他特定信息。

```py
import wx
import sys

#---- Application Information ----#
APP = "FileEditor.py"
NAME = "File Editor"
VERSION = "1.0"
AUTHOR = "Author Name"
AUTHOR_EMAIL = "authorname@someplace.com"
URL = "http://fileeditor_webpage.foo"
LICENSE = "wxWidgets"
YEAR = "2010"

#---- End Application Information ----#

```

在这里，我们将定义一种方法，该方法使用 `py2exe` 从应用程序信息部分中指定的 `APP` 变量的 Python 脚本构建 Windows 可执行文件：

```py
RT_MANIFEST = 24

def BuildPy2Exe():
    """Generate the Py2exe files"""
    from distutils.core import setup
    try:
        import py2exe
    except ImportError:
        print "\n!! You dont have py2exe installed. !!\n"
        exit()

```

Windows 的二进制文件中嵌入了一个清单，该清单指定了依赖项和其他设置。本章附带的示例代码包括以下两个 XML 文件，这些文件将确保在 Windows XP 及更高版本上运行时 GUI 具有适当的主题控件：

```py
    pyver = sys.version_info[:2]
    if pyver == (2, 6):
        fname = "py26manifest.xml"
    elif pyver == (2, 5):
        fname = "py25manifest.xml"
    else:
        vstr = ".".join(pyver)
        assert False, "Unsupported Python Version %s" % vstr
    with open(fname, 'rb') as handle:
        manifest = handle.read()
        manifest = manifest % dict(prog=NAME)

```

`OPTS` 字典指定了 `py2exe` 选项。这些是一些适用于大多数应用的常规设置，但如果需要针对特定用例进行进一步调整，它们可以进行微调：

```py
    OPTS = {"py2exe" : {"compressed" : 1,
                        "optimize" : 1,
                        "bundle_files" : 2,
                        "excludes" : ["Tkinter",],
                        "dll_excludes": ["MSVCP90.dll"]}}

```

`setup` 函数中的 `windows` 关键字用于指定我们正在创建一个图形用户界面应用程序，并用于指定要嵌入二进制的应用程序图标和清单：

```py
    setup(
        name = NAME,
        version = VERSION,
        options = OPTS,
        windows = [{"script": APP,
                    "icon_resources": [(1, "Icon.ico")],
                    "other_resources" : [(RT_MANIFEST, 1,
                                          manifest)],
                  }],
        description = NAME,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        url = URL,
        )

```

接下来是我们的 OS X 构建方法，它使用 py2app 来构建二进制小程序包：

```py
def BuildOSXApp():
    """Build the OSX Applet"""
    from setuptools import setup

```

在这里，我们定义了一个`PLIST`，其用途与 Windows 二进制文件使用的清单非常相似。它用于定义一些关于应用程序的信息，操作系统使用这些信息来了解应用程序扮演的角色。

```py
    # py2app uses this to generate the plist xml for
    # the applet.
    copyright = "Copyright %s %s" % (AUTHOR, YEAR)
    appid = "com.%s.%s" % (NAME, NAME)
    PLIST = dict(CFBundleName = NAME,
             CFBundleIconFile = 'Icon.icns',
             CFBundleShortVersionString = VERSION,
             CFBundleGetInfoString = NAME + " " + VERSION,
             CFBundleExecutable = NAME,
             CFBundleIdentifier = appid,
             CFBundleTypeMIMETypes = ['text/plain',],
             CFBundleDevelopmentRegion = 'English',
             NSHumanReadableCopyright = copyright
             )

```

以下字典指定了`setup()`在构建应用程序时将使用的`py2app`选项：

```py
    PY2APP_OPTS = dict(iconfile = "Icon.icns",
                       argv_emulation = True,
                       optimize = True,
                       plist = PLIST)

    setup(
        app = [APP,],
        version = VERSION,
        options = dict( py2app = PY2APP_OPTS),
        description = NAME,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        url = URL,
        setup_requires = ['py2app'],
        )

if __name__ == '__main__':
    if wx.Platform == '__WXMSW__':
        # Windows
        BuildPy2Exe()
    elif wx.Platform == '__WXMAC__':
        # OSX
        BuildOSXApp()
    else:
        print "Unsupported platform: %s" % wx.Platform

```

## 它是如何工作的...

使用之前的设置脚本，我们可以在 Windows 和 OS X 上为我们的 `FileEditor` 脚本构建独立的可执行文件。因此，让我们分别查看这两个函数，`BuildPy2exe` 和 `BuildOSXApp`，看看它们是如何工作的。

`BuildPy2exe` 执行必要的准备工作，以便在 Windows 机器上使用 `py2exe` 运行 `setup` 来构建独立的二进制文件。在这个函数中，有三个重要的部分需要注意。首先是创建清单的部分。在 2.5 和 2.6 版本之间，用于构建 Python 解释器二进制的 Windows 运行时库发生了变化。因此，我们需要在我们的二进制清单中指定不同的依赖项，以便它能够加载正确的运行时，并给我们的 GUI 应用程序提供正确的主题外观。本主题的示例源代码中包含了 Python 2.5 或 2.6 的两种可能的清单。

第二个是`py2exe`选项字典。这个字典包含了在捆绑脚本时使用的`py2exe`特定选项。我们使用了五个选项：`compressed`、`optimize`、`bundle_files`、`excludes`和`dll_excludes`。`compressed`选项表示我们希望压缩生成的`.exe`文件。`optimize`选项表示要优化 Python 字节码。在这里我们可以指定`0`、`1`或`2`，以实现不同级别的优化。`bundle_files`选项指定将依赖项捆绑到`library.zip`文件的级别。数字越低（`1-3`），捆绑到 ZIP 文件中的文件数量就越多，从而减少了需要分发的单个文件的总数。使用`1`可能会经常导致 wxPython 应用程序出现问题，因此建议使用`2`或`3`。接下来，`excludes`选项是一个要排除在结果捆绑中的模块列表。在这里我们指定了`Tkinter`，只是为了确保其依赖项不会意外地被包含在我们的二进制文件中，从而使文件变大。最后，`dll_excludes`选项用于解决在使用`py2exe`与 Python 2.6 时遇到的问题。

第三点也是最后一点是`setup`命令中的`windows`参数。这个参数用于指定我们正在构建一个 GUI 应用程序，并且在这里我们指定要嵌入到`.exe`文件中的应用程序图标以及之前提到的清单文件。

使用 `py2exe` 运行 `setup` 与以下命令行语句一样简单：

```py
python setup.py py2exe

```

现在我们来看看 `py2app` 的工作原理。它与 `py2exe` 非常相似，实际上甚至更易于使用，因为无需担心像在 Windows 上那样的运行时依赖问题。主要区别在于 `PLIST`，它在某种程度上类似于 Windows 上的清单文件，但用于定义一些应用程序行为并存储操作系统使用应用程序信息。`Py2app` 将使用指定的字典在生成的应用程序中生成 `Plist` XML 文件。有关可用的 `Plist` 选项，请参阅在 [`developer.apple.com`](http://developer.apple.com) 提供的适当列出的文档。`PLIST` 字典通过设置函数的 `options` 参数传递给 `py2app`，以及其他我们指定的 `py2app` 选项，例如应用程序的图标。此外，与 `py2exe` 非常相似，运行 `py2app` 只需要以下命令行语句：

```py
python setup.py py2app

```

## 还有更多...

以下包含有关 Windows 应用程序的一些特定分布依赖性问题的附加信息，以及为 Windows 和 OS X 上的应用程序创建安装程序的参考资料。

### Py2Exe 依赖项

运行 `py2exe` 设置命令后，请确保您审查了末尾列出的未包含的依赖项列表。可能需要手动将一些额外的文件包含到您的应用程序的 `dist` 文件夹中，以便在不同计算机上部署时能够正常运行。对于 Python 2.5，通常需要 `msvcr71.dll` 和 `gdiplus.dll` 文件。对于 Python 2.6，则需要 `msvcr90.dll` 和 `gdiplus.dll` 文件。`msvcr .dll` 文件由微软版权所有，因此您应该审查许可条款，以确保您有权重新分发它们。如果没有，用户可能需要单独使用可以从微软网站下载的免费可重新分发的运行时包来安装它们。

### 安装程序

在使用 `py2exe` 或 `py2app` 构建你的应用程序后，你需要一种方法来帮助应用程序的用户将文件正确安装在他们的系统上。对于 Windows，有多个选项可用于构建安装程序：NSIS ([`nsis.sourceforge.net`](http://nsis.sourceforge.net)) 和 Inno Setup ([`www.jrsoftware.org/isinfo.php`](http://www.jrsoftware.org/isinfo.php)) 是两种流行的免费选项。在 OS X 上，必要的工具已经安装好了。只需使用磁盘工具应用程序创建一个磁盘镜像（`.dmg`）文件，然后将构建的应用程序复制到其中。
