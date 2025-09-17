# 第十章. 高级主题

在本章中，我们将探讨以下几个可以进一步扩展你的脚本的先进主题：

+   将 Python 功能封装在 MEL 中

+   使用上下文创建自定义工具

+   使用脚本作业触发自定义功能

+   使用脚本节点在场景中嵌入代码

+   结合脚本作业和脚本节点

# 简介

在本章中，我们将探讨一些可以用来给你的脚本增加额外光泽并使它们更容易为你的队友使用的先进主题。我们将了解如何使用上下文使你的脚本像 Maya 的内置工具一样工作，如何使用脚本作业在响应事件时触发自定义功能，以及如何使用脚本节点将代码嵌入场景中。

最后，我们将探讨一个可以用来在场景中嵌入自定义功能并在选择特定对象时触发它的工具（例如，用于调用角色绑定的复杂 UI 非常有用）。

# 将 Python 功能封装在 MEL 中

虽然 Python 无疑是 Maya 脚本编写的首选方式，但仍然有一些功能需要你使用 MEL。在本章中，我们将探讨这些功能中的几个，但首先我们需要看看如何从 MEL 中调用 Python 代码。

## 准备工作

首先，我们需要一个 Python 脚本进行调用。你可以使用你已编写的内容或者创建新的内容。为了这个示例，我将使用一个新脚本，该脚本简单地创建一个位于原点的 NURBS 球体，如下所示：

```py
# listing of pythonFromMel.py
import maya.cmds as cmds

def makeSphere():
    cmds.sphere()
```

## 如何做到这一点...

在这个示例中，我们将创建一个 MEL 脚本，该脚本将调用我们的 Python 脚本。创建一个新文件并添加以下代码，确保将其保存为`.mel`扩展名。在这种情况下，我们将创建一个名为`melToPython.mel`的文件：

```py
global proc melToPython()
{
    python "import pythonFromMel";
    python "pythonFromMel.makeSphere()";
}
```

注意，文件中定义的函数与文件本身的名称相同；这是创建 MEL 脚本时的标准做法，并且用于指示脚本的入口点。你当然可以在脚本中拥有多个函数，但通常应该始终有一个与文件名称相同的函数，并且该函数应该是脚本的开始点。

确保将脚本保存到 Maya 的默认脚本位置之一。在 Mac 系统上，这意味着：

```py
/Users/Shared/Autodesk/Maya/(Version)/scripts
```

在 PC 上，这意味着：

```py
\Documents and Settings\<username>\My Documents\maya
```

一旦你做了这件事，你需要确保 Maya 知道新的脚本，这意味着在 Maya 内部调用 rehash MEL 命令。通过点击文本字段左侧，显示为**Python**的位置切换你的命令行到 MEL。或者，切换到脚本编辑器的**MEL**选项卡并输入你的代码。

rehash 命令强制 Maya 重新检查其已知的脚本位置列表，并注意任何新添加的 MEL 脚本。这会在 Maya 每次启动时自动发生，但如果你在 Maya 打开时创建了一个新脚本并尝试在调用 rehash 之前运行它，Maya 会给你一个错误。

一旦运行了 rehash，您可以通过在命令行或脚本编辑器中输入脚本名称来运行我们的新 MEL 脚本。这样做应该会在原点处出现一个新的 NURBS 球体。

## 它是如何工作的...

MEL 脚本相当直观。请注意，函数的定义方式略有不同，只有一些细微的差异。`proc` 关键字（代表 *procedure*）与 Python 中的 `def` 具有相同的作用，表示一个命名的代码块。此外，与括号后跟冒号不同，实际代码用花括号括起来。

`global` 关键字表示这个特定的函数旨在从脚本外部调用。在编写 MEL 时，将具有与文件相同名称的全局过程作为脚本的入口点是常见的做法。

我们主要感兴趣的是让这个脚本调用一些 Python 功能。为此，我们依赖于 `python` MEL 命令。`python` 命令接受一个字符串作为参数，并尝试将那个字符串作为 Python 语句运行。

例如，如果我们想从 MEL 中调用 Python 的 `print` 命令，我们可以这样做：

```py
python "print('hello from Python')"
```

注意，MEL 与 Python 不同，内置函数的参数 *不* 用括号括起来。所以，在上一个例子中，`python` 命令接收一个字符串作为其单个参数。这个字符串被传递给 Python 解释器，在这种情况下，结果是一些文本被打印出来。

要从 MEL 实际运行 Python 脚本，我们需要做两件事：

+   使用 `import` 语句加载脚本

+   在脚本内部调用一个函数

这意味着我们需要调用 MEL 的 `python` 命令两次。导入相当简单：

```py
python "import pythonFromMel";
```

第二行需要一些解释。当我们使用 `import` 命令加载脚本时，脚本被加载为一个模块。脚本中定义的每个函数都是模块的属性。因此，为了调用脚本中定义的函数，我们将想要使用以下语法：

```py
moduleName.functionName()
```

将其包裹在字符串中并传递给 MEL，我们得到 `pythonFromMel` 脚本中定义的 `makeSphere()` 函数如下：

```py
python "pythonFromMel.makeSphere()";
```

我们可以选择将 `import` 语句和 `makeSphere` 调用合并到一行。为此，我们需要用分号分隔这两个语句。虽然 Python 不 *要求* 在语句末尾使用分号，但它允许这样做。在大多数情况下，这并不是必需的，但如果需要在单行上放置多个语句，这可能会很有用。

如果我们这样做，我们最终会得到以下结果：

```py
python "import pythonFromMel; pythonFromMel.makeSphere()";
```

这将在我们需要将 MEL 命令作为单行传递以调用 Python 功能时很有用。

## 还有更多...

应该提到的是，Maya 提供了一个内置实用程序，可以从给定的 Python 脚本创建 MEL 脚本，该实用程序是 `createMelWrapper` 命令，它是 `maya.mel` 库的一部分。

如果我们想在示例中使用的`makeSphere`函数上调用它，我们可以在脚本编辑器的**Python**标签中运行以下代码：

```py
import maya.mel as mel

maya.mel.createMelWrapper(pythonFromMel.makeSphere)
```

这将提示你保存创建的 MEL 脚本的位置。如果你打开创建的脚本，你会看到如下内容：

```py
global proc makeSphere () {
    python("from pythonFromMel import makeSphere");

    python("makeSphere()"); }
```

除了格式上的差异外，生成的脚本几乎与我们写的完全相同。唯一的真正区别是它明确导入了`makeSphere`命令，而不是整个`pythonFromMel`模块。

# 使用上下文创建自定义工具

许多 Maya 的工具都是交互式使用的，用户根据需要指定输入，动作在提供必要数量的输入或用户按下*Enter*键时发生。

到目前为止，我们的脚本还没有这样工作过——需要用户明确运行脚本或按按钮。这对许多事情来说都很好，但提供交互式输入可以为脚本增添很多润色。在这个例子中，我们将做的是 exactly that。

我们将创建一个脚本，一旦调用，就会提示用户选择两个或多个对象。当他们按下*Enter*键时，我们将在所有对象的平均位置创建一个定位器。为此，我们需要创建一个自定义上下文来实现我们自己的工具。

![使用上下文创建自定义工具](img/4657_10_01.jpg)

我们的自定义工具正在使用中。左图是使用工具时的样子（注意左侧的自定义"AVG"图标），右图显示了结果——在所选对象平均位置的新定位器。

## 准备工作

如此呈现的脚本使用了自定义图标。虽然这不是必需的，但这是一项很好的润色。如果你想这么做，创建一个 32x32 像素的透明 PNG 文件，并将其保存到图标文件夹中。在 mac 上，应该是这样的：

```py
/Users/Shared/Autodesk/Maya/icons/
```

...在 PC 上，这意味着：

```py
\Documents and Settings\<username>\My Documents\maya\icons\
```

## 如何操作...

创建一个新文件，并添加以下代码。确保将其命名为`customCtx.py`。

```py
import maya.cmds as cmds

def startCtx():
    print("starting context")

def finalizeCtx():
    objs = cmds.ls(selection=True)

    numObjs = len(objs)
    xpos = 0
    ypos = 0
    zpos = 0

    for o in objs:
        # print(o)
        pos = cmds.xform(o, query=True, worldSpace=True, translation=True)
        # print(pos)
        xpos += pos[0]
        ypos += pos[1]
        zpos += pos[2]

    xpos /= numObjs
    ypos /= numObjs
    zpos /= numObjs

    newLoc = cmds.spaceLocator()
    cmds.move(xpos, ypos, zpos, newLoc)

def createContext():
    toolStartStr = 'python("customCtx .startCtx()");'
    toolFinishStr = 'python("customCtx .finalizeCtx()");'

    newCtx = cmds.scriptCtx(i1='myTool.png', title='MyTool', setNoSelectionPrompt='Select at least two objects',toolStart=toolStartStr, finalCommandScript=toolFinishStr, totalSelectionSets=1, setSelectionCount=2, setAllowExcessCount=True, setAutoComplete=False, toolCursorType="create")

    cmds.setToolTo(newCtx)

createContext()
```

如果你运行脚本，你会看到 Maya 在你的左侧 UI 中激活了你的新图标，就像其他任何工具一样。*Shift*选择至少两个对象，然后按*Enter*键。你会看到一个新的定位器出现在所选对象平均位置。

作为附加功能，你会发现*Y*快捷键，它可以用来重新调用最近使用的工具，也会再次启动你的脚本。

## 它是如何工作的...

首先，我们创建几个函数，这些函数将被新上下文使用，一个在它启动时被调用，另一个在它结束时被调用。`start`脚本非常简单（只是打印一些文本），只是为了演示目的而包含在内。

```py
def startCtx():
    print("starting context")
```

在命令结束时调用的函数稍微复杂一些，但仍然不复杂。我们首先获取当前选定的对象，并设置一些变量——一个用于存储对象数量，一个用于我们将创建定位器的 x、y 和 z 位置。

```py
def finalizeCtx():
    objs = cmds.ls(selection=True)

    numObjs = len(objs)
    xpos = 0
    ypos = 0
    zpos = 0
```

接下来，我们遍历所有对象，并使用查询模式下的`xform`命令获取它们的位置。我们将每个 x、y 和 z 位置添加到我们的变量中，以创建一个位置的总计。

```py
    for o in objs:
        # print(o)
        pos = cmds.xform(o, query=True, worldSpace=True, translation=True)
        xpos += pos[0]
        ypos += pos[1]
        zpos += pos[2]
```

然后，我们将每个位置变量除以对象数量以平均位置，创建一个新的定位器，并将其移动到平均位置。

```py
    xpos /= numObjs
    ypos /= numObjs
    zpos /= numObjs

    newLoc = cmds.spaceLocator()
    cmds.move(xpos, ypos, zpos, newLoc)
```

现在是时候进行有趣的部分——实际上设置一个自定义上下文。我们首先创建可以用来调用我们的两个函数的 MEL 字符串。在两种情况下，它们只是调用我们脚本中定义的一个函数。

```py
def createContext():
    toolStartStr = 'python("customCtx .startCtx()");'
    toolFinishStr = 'python("customCtx .finalizeCtx()");'
```

注意，我们在调用函数之前没有明确导入`customCtx`（如前一个示例中所示）。这是因为我们正在使用同一脚本中定义的功能，所以如果此代码正在执行，则`customCtx`脚本必须已经导入。

现在我们已经准备好进行主要事件——使用`scriptCtx`命令创建一个新的上下文。

```py
newCtx = cmds.scriptCtx(i1='myTool.png', title='MyTool', setNoSelectionPrompt='Select at least two objects',toolStart=toolStartStr, finalCommandScript=toolFinishStr, totalSelectionSets=1, setSelectionCount=2, setAllowExcessCount=True, setAutoComplete=False, toolCursorType="create")
```

如您所见，这是一个相当大的命令，所以让我们来看看它的参数。首先，我们使用`i1`标志来指定工具要使用的图标。您可以省略这个标志，但如果这样做，Maya 将在您的工具激活时在 UI 中突出显示一个空白区域。务必制作 32x32 像素的图标，并将其放入图标文件夹中（见上面的*准备就绪*）。

接下来，我们设置标题。这也是可选的，但会使显示的文本对用户更有用。同样，我们可以省略`setNoSelectionPrompt`标志，但最好保留它。设置标题和`setNoSelectionPrompt`标志将在 Maya 界面的底部显示有用的文本。

现在我们来到了命令的核心部分，即`toolStart`和`finalCommandScript`标志。两者都必须传递一个字符串，该字符串对应于应在脚本开始时或按下*Enter*键时运行的 MEL 命令。我们传递为每个创建的 MEL 字符串，这将反过来调用 Python 功能。

接下来的一组标志都与选择的特定细节有关。首先，我们将选择集的数量设置为`1`，这意味着我们想要一个单独的项目集合。之后，我们使用`setSelectionCount`标志来指定至少需要选择两个项目以便工具能够运行。在这种情况下，我们还想允许用户选择超过两个对象，因此我们将`setAllowExcessCount`标志设置为`true`。由于我们希望允许用户指定可变数量的对象，并且不完成命令直到他们按下*Enter*，我们需要将`setAutoComplete`设置为`false`。将其设置为`true`会导致最终命令脚本在用户选择了等于`setSelectionCount`数量的对象时立即运行。这在某些情况下当然很有用，但不是我们想要的。

最后，我们将`toolCursorType`标志设置为`create`。这将设置在工具期间使用的光标。Maya 提供了一系列不同的选项，为您的目的选择最佳选项可以是一种很好的方式来为您的工具增添专业感（同时为用户提供一些质量反馈）。有关选项列表，请务必查看`scriptCtx`命令的文档。

呼——这有很多标志，但我们已经完成了，准备收尾。在这个脚本的这个点上，我们已经创建了新的上下文，但它尚未激活。要实际调用工具，我们需要使用`setToolTo`命令，并传入对`scriptCtx`的调用输出。

```py
cmds.setToolTo(newCtx)
```

有了这个，我们已经为 Maya 添加了一个全新的工具。

## 还有更多...

在这个示例中，我们创建了自己的自定义工具。您也可以通过使用适当的命令来创建该类型的上下文，然后使用`setToolTo`切换到它，来调用 Maya 的内置工具。

例如，您可能正在创建一个脚本，允许用户以半自动化的方式创建角色绑定。作为那部分的一部分，您可能希望用户创建一些骨骼，然后由您的系统进一步操作。您可以通过用户使用关节工具创建一些骨骼来开始这个过程。在调用您的脚本后直接将它们放入骨骼创建中，您可以使用以下方法：

```py
makeBoneCtx = cmds.jointCtx()
cmds.setToolTo(makeBoneCtx)
```

您可以创建大量上下文——请参阅 Maya 文档以获取完整列表。

另一件您可能会发现有用的功能是能够重置当前上下文，这将丢弃迄今为止的所有输入并重置当前工具。您可以使用自己的自定义工具或内置在 Maya 中的工具来完成此操作。无论如何，以下是如何重置当前工具：

```py
cmds.ctxAbort()
```

上下文是给脚本增添润色的一种好方法，但只有在用户以交互方式添加输入有意义，或者你预计用户会多次快速连续使用你的工具时才真正应该使用。如果你有一个只期望用户使用一次的脚本，并且输入数量有限（且固定），那么直接提供一个按钮可能更容易。然而，如果你的脚本需要处理可变数量的输入，或者需要在新的集合上再次调用而不重新调用脚本，你可能想要考虑创建一个上下文。另一种看待它的方式是，你应该只在上下文能够为用户提供净*减少*工作量（以点击次数衡量）时使用。

# 使用脚本作业触发自定义功能

脚本作业提供了另一种替代方法，可以明确调用脚本或按按钮来调用你的功能。通过使用脚本作业，可以根据特定的条件或特定的事件触发自定义功能。

在这个例子中，我们将创建一个脚本作业，该作业将响应选择更改事件，将所选对象的名称和类型打印到控制台。

## 准备工作

脚本作业之所以非常有用，其中一个原因是它们会持续存在（而不是只运行一次）。然而，这可能会使得开发使用它们的脚本变得有些困难，因为如果你更改了代码并重新运行脚本，你会在场景中结束多个脚本作业。因此，给自己一个轻松清除所有现有脚本作业的方法是很好的。以下脚本将做到这一点：

```py
import maya.cmds as cmds

def killAll():
    cmds.scriptJob(killAll=True, force=True)
    print('KILLED ALL JOBS')

killAll()
```

使用带有`killAll`标志的`scriptJob`命令将清除场景中所有正常的脚本作业。然而，脚本作业也可以创建为`受保护的`或`永久的`。添加强制标志也会清除受保护的脚本作业，但请小心，因为 Maya 使用`scriptJobs`来实现其一些 UI 功能。为了完全安全，请省略`force=True`标志，并确保你创建的`scriptJobs`不是受保护的。

永久性脚本作业将一直持续到您创建一个新的场景，但在开发过程中不应该出现这种情况。即使你真的想要一个永久性脚本作业，最好是以默认优先级进行开发，一旦你确定你得到了想要的功能，再将其升级为永久性。

在你开始使用脚本作业之前，确保有上述脚本（或类似）可用，因为它肯定会让你的人生变得更加轻松。

## 如何操作...

创建一个新的脚本并添加以下代码。请确保将文件命名为`selectionOutput.py`：

```py
import maya.cmds as cmds
import sys

def selectionChanged():
    objs = cmds.ls(selection=True)

    if len(objs) < 1:
        sys.stdout.write('NOTHING SELECTED')
    else:
        shapeNodes = cmds.listRelatives(objs[0], shapes=True)
        msg = objs[0]
        if (len(shapeNodes) > 0):
            msg += ": " + cmds.nodeType(shapeNodes[0])

        sys.stdout.write(msg)

def makeEventScriptJob():
    cmds.scriptJob(event=["SelectionChanged", selectionChanged], killWithScene=True)

makeEventScriptJob()
```

运行上面的脚本，每次你选择（或取消选择）一个对象时，你应该在 Maya 的 UI 底部看到文本出现。

## 它是如何工作的...

首先，请注意，我们除了导入标准的 `maya.cmds` 之外，还导入了 sys（或系统）库。这是为了允许我们将文本打印到命令行，以便即使在用户没有打开脚本编辑器的情况下，用户也能看到。关于这一点，我们稍后再详细说明。

在创建 `scriptJob` 之前，我们想要创建它需要调用的代码。在这种情况下，我们将触发代码，每当选择发生变化时，我们希望该代码检查当前选定的对象。我们像在其他示例中做的那样开始，使用 ls 来获取选择：

```py
def selectionChanged():
    objs = cmds.ls(selection=True)
```

然后，如果我们发现没有选择任何内容，我们将输出一些文本到命令行。

```py
    if len(objs) < 1:
        sys.stdout.write('NOTHING SELECTED')
```

正是这里，`sys` 库发挥了作用——通过使用 `sys.stdout.write`，我们能够直接将文本输出到命令行。这为向脚本的用户提供反馈提供了一种好方法，因为你不应该期望他们打开脚本编辑器。请注意，我们**也可以**使用错误或警告命令，但由于这段文本仅仅是输出，既不是错误也不是警告，因此使用 `stdout.write` 更为合适。

`selectionChanged` 函数的其余部分相当直接。唯一稍微棘手的是，如果我们查看选定节点的节点类型，我们保证只会得到变换。为了避免这种情况，我们首先检查是否有任何形状节点连接到相关的节点。如果有，我们将形状的节点类型追加到对象的名称中，并将其输出到命令行。

```py
    else:
        shapeNodes = cmds.listRelatives(objs[0], shapes=True)
        msg = objs[0]
        if (len(shapeNodes) > 0):
            msg += ": " + cmds.nodeType(shapeNodes[0])

        sys.stdout.write(msg)
```

现在我们已经准备好进行有趣的部分——实际上创建 `scriptJob`。所有 `scriptJobs` 都需要我们指定一个事件或条件，以及当事件被触发或条件假设给定值（true、false 或当它改变）时执行的代码。

重要的是要注意，事件和条件必须与 Maya 内置的事件和条件相对应。在这种情况下，我们将使用 `SelectionChanged` 事件作为触发器。每当选择因任何原因发生变化时，它都会触发，无论选择了多少对象（包括零个）。

要实际创建 `scriptJob`，我们使用 `scriptJob` 命令。

```py
cmds.scriptJob(event=["SelectionChanged", selectionChanged], killWithScene=True)
```

在这种情况下，我们使用事件标志来告诉 Maya，这个 `scriptJob` 应该基于事件（而不是基于条件）。我们传递给标志的值需要是一个数组，其中第一个元素是我们想要监视的事件对应的字符串，第二个是响应时需要调用的函数。

在这种情况下，我们希望在 `SelectionChanged` 事件发生时调用我们的 `selectionChanged` 函数。我们还包含了 `killWithScene` 标志，当离开当前场景时，这将导致 `scriptJob` 被销毁，这通常是一个好主意。当然，有合理的理由让 `scriptJob` 在场景之间持续存在，但除非你确定这是你想要的，否则通常最好防止这种情况发生。

就这样！现在每次选择改变时，我们都会调用自定义函数。

## 还有更多...

在*准备工作*部分，我们介绍了一个简单的脚本，用于删除*所有*`scriptJobs`。这在测试期间是可以的，但有时可能会有些过于强硬。有许多情况下你可能只想删除特定的`scriptJob`——可能是因为它所实现的功能不再必要。这很容易做到，但需要指定你想要删除的`scriptJob`。

创建新的脚本作业时，`scriptJob`命令将返回一个整数，可以用作创建的脚本作业的 ID。稍后，你可以使用这个数字来删除特定的脚本作业，同时保持场景中的其他脚本作业完好无损。如果你想稍后删除脚本作业，请确保将输出保存到变量中，如下所示：

```py
jobID = cmds.scriptJob(event=["SelectionChanged", selectionChanged], killWithScene=True)
```

然后，要删除脚本作业，再次调用`scriptJob`命令，但这次带有终止标志，并传入 ID，如下所示：

```py
cmds.scriptJob(kill=jobID)
```

如果你试图删除的脚本作业受保护，你需要也将`force`标志设置为`true`，如下所示：

```py
cmds.scriptJob(kill=jobID, force=True)
```

你还可以使用`scriptJob`命令来获取当前所有活动的脚本作业列表。为此，运行它时将`listJobs`标志设置为`True`。例如：

```py
jobs = cmds.scriptJob(listJobs=True)

for j in jobs:
        print(j)
```

...这会导致以下结果：

```py
0:  "-permanent" "-event" "PostSceneRead" "generateUvTilePreviewsPostSceneReadCB"
1:  "-permanent" "-parent" "MayaWindow" "-event" "ToolChanged" "changeToolIcon"
```

...以及 Maya 使用的其他长列表脚本作业，以及你添加的任何脚本作业。左侧的数字是作业的 ID，可以用来删除它（只要它不是*永久的*）。

作为删除所有作业或通过 ID 删除单个作业的替代方案，你还可以让 Maya 在删除给定的 UI 时删除脚本作业。例如，如果我们想要一个只在给定窗口打开时存在的脚本作业，我们可以做如下操作：

```py
def scriptJobUI():
    win = cmds.window(title="SJ", widthHeight=(300, 200))

    cmds.scriptJob(parent=win, event=["SelectionChanged", respondToSelection])

    cmds.showWindow(win)
```

注意在调用`cmds.scriptJob`时添加了`parent`标志。你可以包含该标志将脚本作业与特定的 UI 绑定。在这种情况下，我们将脚本作业绑定到窗口上。

# 使用脚本节点在场景中嵌入代码

我们迄今为止看到的所有示例都作为脚本存在，与它们运行的场景是分开的。这对于工具来说是可以的，但如果你创建了一个与特定场景紧密相关的脚本（例如，用于角色绑定的自定义控制 UI），你必须小心确保脚本文件始终与 Maya 文件一起分发。

对于这种情况，Maya 提供了一种更好的方法。可以使用脚本节点将脚本直接烘焙到场景中，这样它们就可以在没有任何外部依赖的情况下运行。此外，可以使用代码创建脚本节点。

在这个例子中，我们将创建一个脚本，它会提示用户输入 Python 文件，然后创建一个包含文件内容的脚本节点，并设置脚本，以便每次文件打开时执行。

## 准备工作

要使用我们创建的脚本，我们需要有一个准备嵌入的脚本。为了举例，我将使用一个简单的脚本，该脚本显示一个包含创建 NURBS 球体按钮的窗口。

完整的脚本如下：

```py
import maya.cmds as cmds

def testUI():
    win = cmds.window(title="Script Node", widthHeight=(300,200))
    cmds.columnLayout()
    cmds.button(label="Make Sphere", command="cmds.sphere()")
    cmds.showWindow(win)

testUI()
```

## 如何做到这一点...

创建一个新的脚本并添加以下代码：

```py
import maya.cmds as cmds

def createScriptNode():
    filePath = cmds.fileDialog2(fileMode=1, fileFilter="Python files (*.py)")

    if (filePath == None):
        return

    f = open(filePath[0], "r")

    scriptStr = ""

    line = f.readline()
    while (line):
        scriptStr += line
        line = f.readline()

    f.close()

    cmds.scriptNode(sourceType="python", scriptType=2, beforeScript=scriptStr)

createScriptNode()
```

运行脚本，并将生成的文件浏览器指向您想要嵌入的脚本。保存您的文件，然后重新打开它。您应该看到您的嵌入脚本自动运行。

## 它是如何工作的...

我们首先调用 `fileDialog2` 命令来提示用户提供一个 Python 文件。

```py
def createScriptNode():
    filePath = cmds.fileDialog2(fileMode=1, fileFilter="Python files (*.py)")
```

如果用户在未指定文件的情况下取消对话框，`filePath` 将为空。我们检查这一点，并在必要时提前结束脚本。

```py
    if (filePath == None):
        return
```

如果我们确实有一个文件，我们将以文本模式打开它进行读取。

```py
f = open(filePath[0], "r")
```

到目前为止，我们已经准备好为嵌入脚本做准备。`scriptNode` 命令将期望一个字符串，该字符串由构成脚本节点的代码组成，因此我们需要创建这样的字符串。为此，我们将从一个空字符串开始，并添加用户指定的 python 文件的每一行。

```py
    scriptStr = ""

    line = f.readline()
    while (line):
        scriptStr += line
        line = f.readline()
```

到目前为止，`scriptStr` 变量包含指定脚本的全部内容。由于我们已经完成了文件，我们将关闭它。

```py
f.close()
```

现在，我们实际上可以创建脚本节点了。创建脚本节点需要我们指定一些不同的事情。首先，我们需要指定脚本是否为 MEL 或 Python，我们使用 `sourceType` 标志来完成。

我们还需要指定脚本节点中的代码将运行的条件，这需要我们指定一个条件和代码是否应该在它之前或之后执行。在这种情况下，我们将使用可能是最标准的选项，即脚本将在场景首次加载时运行一次。

要做到这一点，我们想要使用 **文件加载时执行** 选项，并使用 `beforeScript` 标志设置我们的代码。将所有这些放在一起，我们得到以下内容：

```py
cmds.scriptNode(sourceType="python", scriptType=2, beforeScript=scriptStr)
```

`scriptType` 标志指定条件，需要是一个介于 `0` 和 `7` 之间的整数。使用 `2` 的值将节点绑定到非批处理模式下的场景打开时。如果您想在批处理模式下打开时也运行脚本，请使用 `1`。使用 `0` 的值将仅在代码被明确调用时运行代码——稍后我会详细介绍。其他选项使用较少——请参阅文档以获取详细信息。

注意，还有一个 `afterScript` 标志，可以用来将代码执行绑定到给定事件之后。如果您与文件加载选项（1 或 2）一起使用它，它将在文件关闭时执行代码。如果您想为 `beforeScript` 和 `afterScript` 标志指定脚本，您可以这样做。

## 还有更多...

你还可以使用`scriptNodes`嵌入不自行执行但直接触发的功能。为此，将`scriptType`的值指定为 0（对应于**按需执行**选项）。然后，当你想要调用代码时，可以按以下方式调用：

```py
cmds.scriptNode("scriptNodeName", executeBefore=True)
```

…运行“之前”脚本，或者..

```py
cmds.scriptNode("scriptNodeName", executeAfter=True)
```

...运行“之后”脚本。

当你与脚本节点一起工作时，验证它们是否已创建而不直接触发它们可能会有所帮助。为此，请转到**窗口** | **动画编辑器** | **表达式编辑器**。从表达式编辑器中，转到**选择过滤器** | **按脚本节点名称**。你会看到界面发生变化，并出现场景中脚本节点的列表。点击任何一个都会允许你更改其属性并查看或编辑相应的代码。

![还有更多...](img/4657_10_02.jpg)

如果需要，你也可以从这个窗口中删除脚本节点。

# 合并脚本作业和脚本节点

脚本作业和脚本节点中的一个优点是，你可以使用脚本节点确保给定的脚本作业随着场景移动。例如，你可能想使用脚本作业在用户选择场景中的某个特定对象时触发自定义角色绑定 UI。

在这个例子中，我们将创建一个脚本，这将使设置此类事情变得非常容易。我们的脚本将执行以下操作：

+   它将要求用户将其指向一个包含一个或多个函数以创建 UI 的 Python 文件

+   它将以滚动列表的形式向用户展示文件中定义的所有函数

+   它将允许用户从文件中选择场景中的对象和命名函数

+   它将函数的内容嵌入场景作为脚本节点，并附带一个脚本作业，每次选择指定的对象时都会运行该函数

## 准备工作

要使用我们将要编写的脚本，你需要有一个至少包含一个顶级函数定义的脚本。请注意，当前脚本的格式无法解析类中的功能，并且一次只能处理一个函数，所以请确保所有功能都包含在单个函数中。为了获得最佳效果，请确保你的输入文件看起来像这样：

```py
import maya.cmds as cmds

def testUI():
    win = cmds.window(title="Script Node", widthHeight=(300,200))
    # add some features here
    cmds.showWindow(win)

def otherUI():
    win = cmds.window(title="Other UI", widthHeight=(300,200))
    # add some features here
    cmds.showWindow(win)
```

## 如何做到这一点...

创建一个新的脚本并添加以下内容：

```py
import maya.cmds as cmds

class EmbedUI():

    def __init__(self):
        self.win = cmds.window(title="Embed UI", widthHeight=(300,400))
        self.commandList = {}

        cmds.columnLayout()

        self.loadButton = cmds.button(label="Load Script", width=300, command=self.loadScript)
        self.makeNodeBtn = cmds.button(label="Tie Script to Current Object", width=300, command=self.makeNode)

        self.functionList = cmds.textScrollList(width=300, numberOfRows=10, selectCommand=self.showCommand)

        cmds.showWindow(self.win)

    def loadScript(self, args):

        self.commandList = {}

        filePath = cmds.fileDialog2(fileMode=1, fileFilter="Python files (*.py)")

        if (filePath == None):
            return

        f = open(filePath[0], "r")

        functionName = ""
        functionStr = ""

        line = f.readline()

        while (line):
            parts = line.split()

            if (line.startswith("import")):
                pass

            elif (line.startswith("def")):
                if (functionName != "" and functionStr != ""):
                    self.commandList[functionName] = functionStr

                functionName = parts[1].replace("():", "")
                functionStr += line

            elif (line.strip() == ""):
                # possibly blank line, check for tab
                if (line.startswith("\t") == False):
                    # blank line, see if we have a function
                    if (functionName != "" and functionStr != ""):
                        self.commandList[functionName] = functionStr
                        functionName = ""
                        functionStr = ""
            else:
                functionStr += line

            line = f.readline()

        f.close()
        self.updateList()

    def updateList(self):
        cmds.textScrollList(self.functionList, edit=True, removeAll=True)

        for function in self.commandList:
            cmds.textScrollList(self.functionList, edit=True, append=function)

    def showCommand(self):
        command = cmds.textScrollList(self.functionList, query=True, selectItem=True)[0]

    def makeNode(self, args):
        command = cmds.textScrollList(self.functionList, query=True, selectItem=True)[0]

        objectName = ""
        objs = cmds.ls(selection=True)

        if (len(objs) > 0):
            objectName = objs[0]

        if (command != "" and objectName != ""):
            print("Tying " + command + " to " + objectName)

            nodeStr = "import maya.cmds as cmds\n\n"

            nodeStr += self.commandList[command] + "\n\n"

            nodeStr += 'def testSelection():\n'
            nodeStr += '\tobjs = cmds.ls(selection=True)\n'
            nodeStr += '\tif (len(objs) > 0):\n'
            nodeStr += '\t\tif (objs[0] == "' + objectName + '"):\n'
            nodeStr += '\t\t\t' + command + '()\n\n'

            nodeStr += 'cmds.scriptJob(killWithScene=True, event=["SelectionChanged", testSelection])'

            cmds.scriptNode(sourceType="python", scriptType=2, beforeScript=nodeStr)

        else:
            cmds.error("Please select a script and an object")

EmbedUI()
```

## 它是如何工作的...

首先，我们为我们的 UI 创建一个类，以便更容易地传递数据。

在`__init__`函数中，我们添加了三项：

+   一个按钮用于加载和解析源文件

+   一个按钮将特定函数与特定对象的选取绑定

+   一个`textScrollList`命令来保存函数名并允许用户选择它们

我们还为自己提供了一个`commandList`变量，这是一个我们将用它来保存文件中找到的命令的字典。每个元素的索引将是函数的名称，值将是该函数的整个源代码。

### 注意

字典是 Python 内置的数据结构之一，在其他语言中有时被称为**关联数组**。字典和列表之间的大不同在于，在列表中，你通过数字索引指定条目，而在字典中，你通过名称指定条目。

例如，你可以使用`myDict = {'foo':1, 'bar':2}`创建一个简单的字典。

…这将创建一个包含两个条目的字典——一个用于`foo`，另一个用于`bar`。访问这些值的方式与列表索引非常相似，只是用名称代替了数字，例如`print(myDict['foo'])`会打印出 1。

将所有这些放在一起，我们得到以下内容：

```py
class EmbedUI():

    def __init__(self):
        self.win = cmds.window(title="Embed UI", widthHeight=(300,400))
        self.commandList = {}

        cmds.columnLayout()

        self.loadButton = cmds.button(label="Load Script", width=300, command=self.loadScript)
        self.makeNodeBtn = cmds.button(label="Tie Script to Current Object", width=300, command=self.makeNode)

        self.functionList = cmds.textScrollList(width=300, numberOfRows=10, selectCommand=self.showCommand)

        cmds.showWindow(self.win)
```

接下来，我们实现`loadScript`函数。我们首先清除`commandList`变量，以防用户指定了一个新文件，然后要求他们指向一个要加载的 Python 源文件。

```py
    def loadScript(self, args):
        self.commandList = {}
        filePath = cmds.fileDialog2(fileMode=1, fileFilter="Python files (*.py)")
```

如果我们找到一个文件，我们以读取模式打开它。

```py
    if (filePath == None):
        return

    f = open(filePath[0], "r")
```

现在我们已经准备好实际读取文件了。我们首先创建两个变量——一个用来存储人类友好的函数名，我们将它在`textScrollList`命令中显示，另一个用来存储实际的源代码。

```py
functionName = ""
functionStr = ""
```

一旦我们做了这些，我们就开始解析文件。我们以与之前示例相同的方式遍历文件，逐行读取——唯一的区别是我们如何解析内容。暂时不考虑文件内容的处理，我们解析的外部部分应该看起来很熟悉：

```py
    line = f.readline()
    while (line):
        # code to handle contents
        line = f.readline()
```

接下来是解析——我们想要捕获每个函数的所有文本。这意味着我们想要从定义函数的行到函数结束的所有内容。然而，找到函数的结束需要一些思考。我们寻找的不仅是一个空白行，而是一个**不包含制表符**的空白行。

我们首先忽略导入语句。我们检查当前行是否以`import`开头，如果是，我们使用`pass`语句来跳过执行任何操作。

```py
        while (line):
            if (line.startswith("import")):
                pass
```

注意，我们可以使用`continue`语句跳过循环的其余部分，但这也会跳过负责读取文件下一行的行，导致无限循环。

接下来，我们检查该行是否以`def`开头，这表示它代表一个新的函数定义。

```py
elif (line.startswith("def")):
```

如果是，我们想要开始收集新函数的代码，但首先我们想要保存我们之前正在逐步执行的函数，如果有的话。为了做到这一点，我们检查`functionName`和`functionStr`变量是否为空。如果它们都有内容，这意味着我们之前保存了另一个函数，我们将它按以下方式插入到我们的函数列表中：

```py
if (functionName != "" and functionStr != ""):
    self.commandList[functionName] = functionStr
```

如果我们正在解析的文件在上一行函数的下一行有一个新的函数定义，且中间没有空白行，那么这种情况就会发生。

现在我们已经处理了之前的函数（如果有的话），我们准备开始存储我们的新函数。我们将从通过丢弃`def`关键字、括号和冒号来获取函数的更人性化的形式开始。

为了做到这一点，我们首先使用 split 函数通过空格将行拆分成一个数组，第一个索引是`def`，第二个索引是类似`myFunction():`的内容。然后我们使用 replace 来移除`():`。这给了我们：

```py
parts = line.split()
functionName = parts[1].replace("():", "")
```

最后，我们将`functionStr`变量设置为整行。在我们继续解析文件的过程中，我们将向这个变量添加额外的行。当我们遇到新的`def`语句或真正空白的（没有制表符）行时，我们将整个`functionStr`存储到我们的命令列表中。

```py
functionStr = line
```

说到空白行，这是我们接下来要检查的。如果该行只包含空白字符，通过运行`strip()`函数将得到一个空字符串。如果我们确实找到一个空字符串，我们可能处于当前函数的末尾，但我们需要通过测试当前行是否以制表符开头来确保这一点。

```py
elif (line.strip() == ""):
    # possibly blank line, check for tab
    if (line.startswith("\t") == False):
```

如果我们确实有一个真正的空白行（没有制表符），并且我们一直在构建一个函数，现在就是时候将它存储到我们的列表中。再次检查，确保我们的`functionName`和`functionStr`变量都有内容，如果有，我们将函数代码存储到我们的`commandList`中。

```py
    if (functionName != "" and functionStr != ""):
        self.commandList[functionName] = functionStr
        functionName = ""
        functionStr = ""
```

为了防止脚本存储同一个函数多次（在出现多个空白行的情况下），我们还将`functionName`和`functionStr`变量重置为空白。

如果上述代码没有任何一个被触发，我们知道我们有一个非空白行，它既不以`import`开头也不以`def`开头。我们将假设任何这样的行都是一条有效的代码行，并且是当前函数的一部分。因此，我们只需将其添加到我们的`functionStr`变量中。

```py
else:
    functionStr += line
```

有了这个，我们就完成了文件的解析，并关闭了它。在这个时候，我们的`commandList`字典将为文件中的每个函数都有一个条目。我们想要通过将它们添加到我们的滚动列表中向用户展示这些函数，我们在`updateList`函数中这样做。

```py
        f.close()
        self.updateList()
```

在`updateList`函数中，我们首先想要清空`scrollList`的内容，然后为找到的每个函数添加一个条目。这两个操作都可以通过在编辑模式下调用`textScrollList`命令轻松完成。首先，我们将其清空：

```py
    def updateList(self):
        cmds.textScrollList(self.functionList, edit=True, removeAll=True)
```

然后我们遍历我们的命令列表，并将每个命令的名称添加到带有`append`标志的列表中：

```py
        for function in self.commandList:
            cmds.textScrollList(self.functionList, edit=True, append=function)
```

现在剩下的就是实现创建脚本节点的函数。首先，我们想要确保用户已经从滚动列表中选择了一个命令，并且在场景中选择了一个对象。为了获取滚动列表中当前选中的项，我们再次使用`textScrollList`命令，但这次是在查询模式下。

```py
command = cmds.textScrollList(self.functionList, query=True, selectItem=True)[0]
```

注意，我们在`textScrollList`命令的末尾有一个`[0]`。这是必要的，因为`textScrollList`小部件可以允许多项目选择。因此，查询`selectItem`的输出可能有多个值，并以数组形式返回。添加`[0]`给我们第一个元素（如果有的话）。

我们获取所选对象的代码很简单，看起来确实很熟悉：

```py
        objectName = ""
        objs = cmds.ls(selection=True)

        if (len(objs) > 0):
            objectName = objs[0]
```

如果我们既有对象又有命令，我们就准备好深入脚本节点创建。如果没有，我们向用户显示错误消息。

对于我们的脚本节点，我们想要的代码将执行以下操作：

+   在场景开始时运行。

+   包含所选函数的定义。

+   包含一个可以在每次选择更改时运行的函数的定义。该函数需要将当前选定的对象与目标对象进行比较，如果匹配，则调用触发函数。

+   创建一个与`SelectionChanged`事件绑定的脚本作业。

![如何工作...](img/4657_10_03.jpg)

左侧：显示输入文件中函数列表的 UI。右侧：选择指定的球体将触发一个自定义 UI。

这是一系列步骤，但最终都归结为构建一个包含上述所有功能的字符串。我们首先将字符串设置为我们在所有脚本中使用的`import maya.cmds as cmds`行。

```py
    if (command != "" and objectName != ""):
        print("Tying " + command + " to " + objectName)
        nodeStr = "import maya.cmds as cmds\n\n"
```

注意，行尾有两个换行符。这会使内容更易于阅读，并在表达式编辑器中检查结果时，如果出现问题，会更容易。

接下来，我们添加我们想要触发的命令的代码。这非常简单，因为我们已经将所有代码存储在我们的`commandList`字典中。我们只需要使用用户选择的命令名称来索引它。

```py
        nodeStr += self.commandList[command] + "\n\n"
```

现在我们需要创建一个函数的代码，该函数负责检查当前选择与目标对象是否匹配，并运行目标脚本。为此，我们需要将一些模板代码和特定的名称（对象和函数的名称）组合在一起。

在这种情况下，首先写出给定特定输入的结果通常是有帮助的。假设我们想要在选择了名为`triggerObject`的对象时触发一个名为`myFunction`的函数。为此，我们可以使用以下函数：

```py
def testSelection():
    objs = cmds.ls(selection=True)
    if (len(objs) > 0):
        if (objs[0] == "triggerObject"):
            myFunction()
```

很简单，对吧？我们只需要将上述文本添加到我们的`nodeStr`变量中，确保替换对象和函数名称，并添加适当的制表符（`\t`）和换行符（`\n`），以便遵循正确的 Python 空白规则。

这最终给出了以下：

```py
    nodeStr += 'def testSelection():\n'
    nodeStr += '\tobjs = cmds.ls(selection=True)\n'
    nodeStr += '\tif (len(objs) > 0):\n'
    nodeStr += '\t\tif (objs[0] == "' + objectName + '"):\n'
    nodeStr += '\t\t\t' + command + '()\n\n'
```

剩下的只是添加将创建脚本作业以正确地将我们的`testSelection`方法绑定到`SelectionChanged`事件的代码。我们只需在`nodeStr`变量中添加一行来完成此操作，如下所示：

```py
nodeStr += 'cmds.scriptJob(killWithScene=True, event=["SelectionChanged", testSelection])'
```

我们已经非常接近完成，但我们目前拥有的仍然只是一大块文本。为了将其真正变成一个脚本节点，我们需要将其作为`beforeScript`值传递给`scriptNode`命令，并设置`scriptType=2`，以便在场景启动时运行。

```py
cmds.scriptNode(sourceType="python", scriptType=2, beforeScript=nodeStr)
```

就这样！我们现在有了一种将任意 UI 代码嵌入场景并在给定对象被选中时触发它的方法。

## 还有更多...

目前，这个例子更多的是一个概念验证，而不是一个合适的工具。出于简洁的考虑，我被迫省略了一些人们可能希望包含的内容，但脚本可以很容易地扩展以包括所有这些内容。

首先，这个脚本只处理单个函数。对于一个合适的角色绑定 UI，我们可能希望包括一个函数集合，可能捆绑在一个或多个类中。为了支持这一点，脚本需要被修改为要么复制源文件的全部内容到脚本节点，要么对文件内容进行更复杂的解析以包括多个函数。

此外，如果脚本在同一场景中被多次使用，它将不会按预期工作，因为每个函数和对象的配对都使用相同的名称（`testSelection`）来关联脚本作业。

为了解决这个问题，我们希望确保每个脚本作业都有一个唯一命名的函数来测试选择。实现这一目标的一种方法是将我们最终想要触发的函数名称附加到`testSelection`函数名称上，如下所示：

```py
selectionFunctionName = "testFor" + command

nodeStr += 'def ' + selectionFunctionName + '():\n'
nodeStr += '\tobjs = cmds.ls(selection=True)\n'
nodeStr += '\tif (len(objs) > 0):\n'
nodeStr += '\t\tif (objs[0] == "' + objectName + '"):\n'
nodeStr += '\t\t\t' + command + '()\n\n'

nodeStr += 'cmds.scriptJob(killWithScene=True, event=["SelectionChanged", ' + selectionFunctionName + '])'
```
