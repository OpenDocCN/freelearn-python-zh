# 准备软件进行分发

到目前为止，在这本书中，我们主要关注的是编写一个可工作的代码。我们的项目都是单个脚本，最多有几个支持数据文件。然而，完成一个项目并不仅仅是编写代码；我们还需要我们的项目能够轻松分发，这样我们就可以与其他人分享（或出售）它们。

在本章中，我们将探讨为分享和分发准备我们的代码的方法。

我们将涵盖以下主题：

+   项目结构

+   使用`setuptools`进行分发

+   使用 PyInstaller 编译

# 技术要求

在本章中，您将需要我们在整本书中使用的基本 Python 和 PyQt 设置。您还需要使用以下命令从 PyPI 获取`setuptools`、`wheel`和`pyinstaller`库：

```py
$ pip install --user setuptools wheel pyinstaller
```

Windows 用户将需要从[`www.7-zip.org/`](https://www.7-zip.org/)安装 7-Zip 程序，以便他们可以使用`tar.gz`文件，所有平台的用户都应该从[`upx.github.io/`](https://upx.github.io/)安装 UPX 实用程序。

最后，您将希望从存储库中获取示例代码[`github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter17`](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter17)。

查看以下视频，看看代码是如何运行的：[`bit.ly/2M5xH4J`](http://bit.ly/2M5xH4J)

# 项目结构

到目前为止，在这本书中，我们一直将每个示例项目中的所有 Python 代码放入单个文件中。然而，现实世界的 Python 项目受益于更好的组织。虽然没有关于如何构建 Python 项目的官方标准，但我们可以应用一些约定和一般概念来构建我们的项目结构，这不仅可以保持组织，还可以鼓励其他人贡献我们的代码。

为了看到这是如何工作的，我们将在 PyQt 中创建一个简单的井字棋游戏，然后花费本章的其余部分来准备分发。

# 井字棋

我们的井字棋游戏由三个类组成：

+   管理游戏逻辑的引擎类

+   提供游戏状态视图和进行游戏的方法的棋盘类

+   将其他两个类合并到 GUI 中的主窗口类

打开第四章中的应用程序模板的新副本，*使用 QMainWindow 构建应用程序*，并将其命名为`ttt-qt.py`。现在让我们创建这些类。

# 引擎类

我们的游戏引擎对象的主要责任是跟踪游戏并检查是否有赢家或游戏是否为平局。玩家将简单地由`'X'`和`'O'`字符串表示，棋盘将被建模为九个项目的列表，这些项目将是玩家或`None`。

它开始如下：

```py
class TicTacToeEngine(qtc.QObject):

    winning_sets = [
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    ]
    players = ('X', 'O')

    game_won = qtc.pyqtSignal(str)
    game_draw = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.board = [None] * 9
        self.current_player = self.players[0]
```

`winning_sets`列表包含`set`对象，其中包含构成胜利的每个棋盘索引的组合。我们将使用该列表来检查玩家是否获胜。我们还定义了信号，当游戏获胜或平局时发出（即，所有方块都填满了，没有人获胜）。构造函数填充了棋盘列表，并将当前玩家设置为`X`。

我们将需要一个方法来在每轮之后更新当前玩家，看起来是这样的：

```py
    def next_player(self):
        self.current_player = self.players[
            not self.players.index(self.current_player)]
```

接下来，我们将添加一个标记方块的方法：

```py
    def mark_square(self, square):
        if any([
                not isinstance(square, int),
                not (0 <= square < len(self.board)),
                self.board[square] is not None
        ]):
            return False
        self.board[square] = self.current_player
        self.next_player()
        return True
```

此方法首先检查给定方块是否应该被标记的任何原因，如果有原因则返回`False`；否则，我们标记方块，切换到下一个玩家，并返回`True`。

这个类中的最后一个方法将检查棋盘的状态，看看是否有赢家或平局：

```py
    def check_board(self):
        for player in self.players:
            plays = {
                index for index, value in enumerate(self.board)
                if value == player
            }
            for win in self.winning_sets:
                if not win - plays:  # player has a winning combo
                    self.game_won.emit(player)
                    return
        if None not in self.board:
            self.game_draw.emit()
```

该方法使用一些集合操作来检查每个玩家当前标记的方块是否与获胜组合列表匹配。如果找到任何匹配项，将发出`game_won`信号并返回。如果还没有人赢，我们还要检查是否有任何未标记的方块；如果没有，游戏就是平局。如果这两种情况都不成立，我们什么也不做。

# 棋盘类

对于棋盘 GUI，我们将使用一个`QGraphicsScene`对象，就像我们在第十二章中为坦克游戏所做的那样，*使用 QPainter 创建 2D 图形*。

我们将从一些类变量开始：

```py
class TTTBoard(qtw.QGraphicsScene):

    square_rects = (
        qtc.QRectF(5, 5, 190, 190),
        qtc.QRectF(205, 5, 190, 190),
        qtc.QRectF(405, 5, 190, 190),
        qtc.QRectF(5, 205, 190, 190),
        qtc.QRectF(205, 205, 190, 190),
        qtc.QRectF(405, 205, 190, 190),
        qtc.QRectF(5, 405, 190, 190),
        qtc.QRectF(205, 405, 190, 190),
        qtc.QRectF(405, 405, 190, 190)
    )

    square_clicked = qtc.pyqtSignal(int)
```

`square_rects`元组为棋盘上的九个方块定义了一个`QRectF`对象，并且每当点击一个方块时会发出一个`square_clicked`信号；随附的整数将指示点击了哪个方块（0-8）。

以下是`=__init__()`方法：

```py
    def __init__(self):
        super().__init__()
        self.setSceneRect(0, 0, 600, 600)
        self.setBackgroundBrush(qtg.QBrush(qtc.Qt.cyan))
        for square in self.square_rects:
            self.addRect(square, brush=qtg.QBrush(qtc.Qt.white))
        self.mark_pngs = {
            'X': qtg.QPixmap('X.png'),
            'O': qtg.QPixmap('O.png')
        }
        self.marks = []
```

该方法设置了场景大小并绘制了青色背景，然后在`square_rects`中绘制了每个方块。然后，我们加载了用于标记方块的`'X'`和`'O'`图像的`QPixmap`对象，并创建了一个空列表来跟踪我们标记的`QGraphicsSceneItem`对象。

接下来，我们将添加一个方法来绘制棋盘的当前状态：

```py
    def set_board(self, marks):
        for i, square in enumerate(marks):
            if square in self.mark_pngs:
                mark = self.addPixmap(self.mark_pngs[square])
                mark.setPos(self.square_rects[i].topLeft())
                self.marks.append(mark)
```

该方法将接受我们棋盘上的标记列表，并在每个方块中绘制适当的像素项，跟踪创建的`QGraphicsSceneItems`对象。

现在我们需要一个方法来清空棋盘：

```py
    def clear_board(self):
        for mark in self.marks:
            self.removeItem(mark)
```

该方法只是遍历保存的像素项并将它们全部删除。

我们需要做的最后一件事是处理鼠标点击：

```py
    def mousePressEvent(self, mouse_event):
        position = mouse_event.buttonDownScenePos(qtc.Qt.LeftButton)
        for square, qrect in enumerate(self.square_rects):
            if qrect.contains(position):
                self.square_clicked.emit(square)
                break
```

`mousePressEvent()`方法由`QGraphicsScene`在用户进行鼠标点击时调用。它包括一个`QMouseEvent`对象，其中包含有关事件的详细信息，包括鼠标点击的位置。我们可以检查此点击是否在我们的`square_rects`对象中的任何一个内部，如果是，我们将发出`square_clicked`信号并退出该方法。

# 主窗口类

在`MainWindow.__init__()`中，我们将首先创建一个棋盘和一个`QGraphicsView`对象来显示它：

```py
        self.board = TTTBoard()
        self.board_view = qtw.QGraphicsView()
        self.board_view.setScene(self.board)
        self.setCentralWidget(self.board_view)
```

现在我们需要创建一个游戏引擎的实例并连接它的信号。为了让我们能够一遍又一遍地开始游戏，我们将为此创建一个单独的方法：

```py
    def start_game(self):
        self.board.clear_board()
        self.game = TicTacToeEngine()
        self.game.game_won.connect(self.game_won)
        self.game.game_draw.connect(self.game_draw)
```

该方法清空了棋盘，然后创建了游戏引擎对象的一个实例，将引擎的信号连接到`MainWindow`方法以处理两种游戏结束的情况。

回到`__init__()`，我们将调用这个方法来自动设置第一局游戏：

```py
        self.start_game()
```

接下来，我们需要启用玩家输入。我们需要一个方法，该方法将尝试在引擎中标记方块，然后在标记成功时检查棋盘是否获胜或平局：

```py
    def try_mark(self, square):
        if self.game.mark_square(square):
            self.board.set_board(self.game.board)
            self.game.check_board()
```

该方法可以连接到棋盘的`square_clicked`信号；在`__init__()`中，添加以下代码：

```py
        self.board.square_clicked.connect(self.try_mark)
```

最后，我们需要处理两种游戏结束的情况：

```py
    def game_won(self, player):
        """Display the winner and start a new game"""
        qtw.QMessageBox.information(
            None, 'Game Won', f'Player {player} Won!')
        self.start_game()

    def game_draw(self):
        """Display the lack of a winner and start a new game"""
        qtw.QMessageBox.information(
            None, 'Game Over', 'Game Over.  Nobody Won...')
        self.start_game()
```

在这两种情况下，我们只会在`QMessageBox`中显示适当的消息，然后重新开始游戏。

这完成了我们的游戏。花点时间运行游戏，并确保您了解它在正常工作时的响应（也许找个朋友和您一起玩几局；如果您的朋友很年轻或者不太聪明，这会有所帮助）。

现在我们有了一个可用的游戏，是时候准备将其分发了。我们首先要做的是以一种使我们更容易维护和扩展的方式构建我们的项目，以及让其他 Python 程序员合作。

# 模块式结构

作为程序员，我们倾向于将应用程序和库视为两个非常不同的东西，但实际上，结构良好的应用程序与库并没有太大的不同。库只是一组现成的类和函数。我们的应用程序主要也只是类定义；它只是碰巧在最后有几行代码，使其能够作为应用程序运行。当我们以这种方式看待事物时，将我们的应用程序结构化为 Python 库模块是很有道理的。为了做到这一点，我们将把我们的单个 Python 文件转换为一个包含多个文件的目录，每个文件包含一个单独的代码单元。

第一步是考虑我们项目的名称；现在，那个名称是`ttt-qt.py`。当你开始着手一个项目时，想出一个快速简短的名称是很常见的，但这不一定是你要坚持的名称。在这种情况下，我们的名称相当神秘，由于连字符而不能作为 Python 模块名称。相反，让我们称之为`qtictactoe`，这是一个更明确的名称，避免了连字符。

首先，创建一个名为`QTicTacToe`的新目录；这将是我们的**项目根目录**。项目根目录是所有项目文件都将放置在其中的目录。

在该目录下，我们将创建一个名为`qtictactoe`的第二个目录；这将是我们的**模块目录**，其中将包含大部分我们的源代码。

# 模块的结构

为了开始我们的模块，我们将首先添加我们三个类的代码。我们将把每个类放在一个单独的文件中；这并不是严格必要的，但这将帮助我们保持代码解耦，并使得更容易找到我们想要编辑的类。

因此，在`qtictactoe`下，创建三个文件：

+   `engine.py`将保存我们的游戏引擎类。复制`TicTacToeEngine`的定义以及它所使用的必要的`PyQt5`导入语句。在这种情况下，你只需要`QtCore`。

+   `board.py`将保存`TTTBoard`类。也复制那段代码以及完整的`PyQt5`导入语句。

+   最后，`mainwindow.py`将保存`MainWindow`类。复制该类的代码以及`PyQt5`导入。

`mainwindow.py`还需要从其他文件中获取`TicTacToeEngine`和`TTTBoard`类的访问权限。为了提供这种访问权限，我们需要使用**相对导入**。相对导入是一种从同一模块中导入子模块的方法。

在`mainwindow.py`的顶部添加这行：

```py
from .engine import TicTacToeEngine
from .board import TTTBoard
```

在导入中的点表示这是一个相对导入，并且特指当前容器模块（在本例中是`qtictactoe`）。通过使用这样的相对导入，我们可以确保我们从自己的项目中导入这些模块，而不是从用户系统上的其他 Python 库中导入。

我们需要添加到我们模块的下一个代码是使其实际运行的代码。这通常是我们放在`if __name__ == '__main__'`块下的代码。

在模块中，我们将把它放在一个名为`__main__.py`的文件中：

```py
import sys
from PyQt5.QtWidgets import QApplication
from .mainwindow import MainWindow

def main():
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
```

`__main__.py`文件在 Python 模块中有着特殊的用途。每当我们使用`-m`开关运行我们的模块时，它就会被执行，就像这样：

```py
$ python3 -m qtictactoe
```

实质上，`__main__.py`是 Python 脚本中`if __name__ == '__main__':`块的模块等价物。

请注意，我们已经将我们的三行主要代码放在一个名为`main()`的函数中。当我们讨论`setuptools`的使用时，这样做的原因将变得明显。

我们需要在模块内创建的最后一个文件是一个名为`__init__.py`的空文件。Python 模块的`__init__.py`文件类似于 Python 类的`__init__()`方法。每当导入模块时，它都会被执行，并且其命名空间中的任何内容都被视为模块的根命名空间。但在这种情况下，我们将它留空。这可能看起来毫无意义，但如果没有这个文件，我们将要使用的许多工具将不会将这个 Python 文件夹识别为一个实际的模块。

此时，您的目录结构应该是这样的：

```py
QTicTacToe/
├── qtictactoe
    ├── board.py
    ├── engine.py
    ├── __init__.py
    ├── __main__.py
    └── mainwindow.py
```

现在，我们可以使用`python3 -m qtictactoe`来执行我们的程序，但对大多数用户来说，这并不是非常直观。让我们通过创建一个明显的文件来帮助一下执行应用程序。

在项目根目录下（模块外部），创建一个名为`run.py`的文件：

```py
from qtictactoe.__main__ import main
main()
```

这个文件的唯一目的是从我们的模块中加载`main()`函数并执行它。现在，您可以执行`python run.py`，您会发现它可以正常启动。但是，有一个问题——当您点击一个方块时，什么也不会发生。那是因为我们的图像文件丢失了。我们需要处理这些问题。

# 非 Python 文件

在 PyQt 程序中，处理诸如我们的`X`和`O`图像之类的文件的最佳方法是使用`pyrcc5`工具生成一个资源文件，然后像任何其他 Python 文件一样将其添加到您的模块中（我们在第六章中学习了这个）。然而，在这种情况下，我们将保留我们的图像作为 PNG 文件，以便我们可以探索处理非 Python 文件的选项。

关于这些文件应该放在项目目录的何处，目前还没有达成一致的意见，但是由于这些图像是`TTTBoard`类的一个必需组件，将它们放在我们的模块内是有意义的。为了组织起见，将它们放在一个名为`images`的目录中。

现在，您的目录结构应该是这样的：

```py
QTicTacToe/
├── qtictactoe
│   ├── board.py
│   ├── engine.py
│   ├── images
│   │   ├── O.png
│   │   └── X.png
│   ├── __init__.py
│   ├── __main__.py
│   └── mainwindow.py
└── run.py
```

我们编写`TTTBoard`的方式是，您可以看到每个图像都是使用相对文件路径加载的。在 Python 中，相对路径始终相对于当前工作目录，也就是用户启动脚本的目录。不幸的是，这是一个相当脆弱的设计，因为我们无法控制这个目录。我们也不能硬编码绝对文件路径，因为我们不知道我们的应用程序可能存储在用户系统的何处（请参阅我们在第六章中对这个问题的讨论，*Styling Qt Applications*，*Using Qt Resource files*部分）。

在 PyQt 应用程序中解决这个问题的理想方式是使用 Qt 资源文件；然而，我们将尝试一种不同的方法，只是为了说明在这种情况下如何解决这个问题。

为了解决这个问题，我们需要修改`TTTBoard`加载图像的方式，使其相对于我们模块的位置，而不是用户的当前工作目录。这将需要我们使用 Python 标准库中的`os.path`模块，因此在`board.py`的顶部添加这个：

```py
from os import path
```

现在，在`__init__()`中，我们将修改加载图像的行：

```py
        directory = path.dirname(__file__)
        self.mark_pngs = {
            'X': qtg.QPixmap(path.join(directory, 'images', 'X.png')),
            'O': qtg.QPixmap(path.join(directory, 'images', 'O.png'))
        }
```

`__file__`变量是一个内置变量，它始终包含当前文件（在本例中是`board.py`）的绝对路径。使用`path.dirname`，我们可以找到包含此文件的目录。然后，我们可以使用`path.join`来组装一个路径，以便在同一目录下的名为`images`的文件夹中查找文件。

如果您现在运行程序，您应该会发现它完美地运行，就像以前一样。不过，我们还没有完成。

# 文档和元数据

工作和组织良好的代码是我们项目的一个很好的开始；但是，如果您希望其他人使用或贡献到您的项目，您需要解决一些他们可能会遇到的问题。例如，他们需要知道如何安装程序，它的先决条件是什么，或者使用或分发的法律条款是什么。

为了回答这些问题，我们将包括一系列标准文件和目录：`LICENSE`文件，`README`文件，`docs`目录和`requirements.txt`文件。

# 许可文件

当您分享代码时，非常重要的是明确说明其他人可以或不可以对该代码做什么。在大多数国家，创建作品的人自动成为该作品的版权持有人；这意味着您对您的作品的复制行为行使控制。如果您希望其他人为您创建的作品做出贡献或使用它们，您需要授予他们一个**许可证**。

管理您项目的许可证通常以项目根目录中的一个名为`LICENSE`的纯文本文件提供。在我们的示例代码中，我们已经包含了这样一个文件，其中包含了**MIT 许可证**的副本。MIT 许可证是一种宽松的开源许可证，基本上允许任何人对代码做任何事情，只要他们保留我们的版权声明。它还声明我们对因某人使用我们的代码而发生的任何可怕事件不负责。

这个文件有时被称为`COPYING`，也可能有一个名为`txt`的文件扩展名。

您当然可以在许可证中加入任何条件；但是，对于 PyQt 应用程序，您需要确保您的许可证与 PyQt 的**通用公共许可证**（**GPL**）GNU 和 Qt 的**较宽松的通用公共许可证**（**LGPL**）GNU 的条款兼容。如果您打算发布商业或限制性许可的 PyQt 软件，请记住来自第一章，*PyQt 入门*，您需要从 Qt 公司和 Riverbank Computing 购买商业许可证。

对于开源项目，Python 社区强烈建议您坚持使用 MIT、BSD、GPL 或 LGPL 等知名许可证。可以在开放源代码倡议组织的网站[`opensource.org/licenses`](https://opensource.org/licenses)上找到已知的开源许可证列表。您还可以参考[`choosealicense.com`](https://choosealicense.com)，这是一个提供有关选择最符合您意图的许可证的指导的网站。

# README 文件

`README`文件是软件分发中最古老的传统之一。追溯到 20 世纪 70 年代中期，这个纯文本文件通常旨在在用户安装或运行软件之前向程序的用户传达最基本的一组指令和信息。

虽然没有关于`README`文件应包含什么的标准，但用户希望找到某些内容；其中一些包括以下内容：

+   软件的名称和主页

+   软件的作者（带有联系信息）

+   软件的简短描述

+   基本使用说明，包括任何命令行开关或参数

+   报告错误或为项目做出贡献的说明

+   已知错误的列表

+   诸如特定平台问题或说明之类的注释

无论您在文件中包含什么，您都应该力求简洁和有组织。为了方便一些组织，许多现代软件项目在编写`README`文件时使用标记语言；这使我们可以使用诸如标题、项目列表甚至表格等元素。

在 Python 项目中，首选的标记语言是**重新结构化文本**（**RST**）。这种语言是`docutils`项目的一部分，为 Python 提供文档实用程序。

当我们创建`qtictactoe`的`README.rst`文件时，我们将简要介绍 RST。从一个标题开始：

```py
============
 QTicTacToe
============
```

顶部行周围的等号表示它是一个标题；在这种情况下，我们只是使用了我们项目的名称。

接下来，我们将为项目的基本信息创建几个部分；我们通过简单地用符号划线下一行文本来指示部分标题，就像这样：

```py
Authors
=======
By Alan D Moore -  https://www.alandmoore.com

About
=====

This is the classic game of **tic-tac-toe**, also known as noughts and crosses.  Battle your opponent in a desperate race to get three in a line.
```

用于下划线部分标题的符号必须是以下之一：

```py
= - ` : ' " ~ ^ _ * + # < >
```

我们使用它们的顺序并不重要，因为 RST 解释器会假定第一个使用的符号作为表示顶级标题的下划线，下一个类型的符号是第二级标题，依此类推。在这种情况下，我们首先使用等号，所以无论我们在整个文档中使用它，它都会指示一个一级标题。

注意单词`tac-tac-toe`周围的双星号，这表示粗体文本。RST 还可以表示下划线、斜体和类似的排版样式。

例如，我们可以使用反引号来指示等宽代码文本：

```py
Usage
=====

Simply run `python qtictactoe.py` from within the project folder.

- Players take turns clicking the mouse on the playing field to mark squares.
- When one player gets 3 in a row, they win.
- If the board is filled with nobody getting in a row, the game is a draw.
```

这个例子还展示了一个项目列表：每行前面都加了一个破折号和空格。我们也可以使用`+`或`*`符号，并通过缩进创建子项目。

让我们用一些关于贡献的信息和一些注释来完成我们的`README.rst`文件：

```py
Contributing
============

Submit bugs and patches to the
`public git repository <http://git.example.com/qtictactoe>`_.

Notes
=====

    A strange game.  The only winning move is not to play.

    *—Joshua the AI, WarGames*
```

`Contributing`部分显示如何创建超链接：将超链接文本放在反引号内，URL 放在尖括号内，并在关闭反引号后添加下划线。`Notes`部分演示了块引用，只需将该行缩进四个空格即可。

虽然我们的文件作为文本是完全可读的，但是许多流行的代码共享网站会将 RST 和其他标记语言转换为 HTML。例如，在 GitHub 上，这个文件将在浏览器中显示如下：

![](img/74f798bd-47d8-4941-8a14-63b614ce31d7.png)

这个简单的`README.rst`文件对于我们的小应用已经足够了；随着应用的增长，它将需要进一步扩展以记录添加的功能、贡献者、社区政策等。这就是为什么我们更喜欢使用 RST 这样的纯文本格式，也是为什么我们将其作为项目仓库的一部分；它应该随着代码一起更新。

RST 语法的快速参考可以在[docutils.sourceforge.net/docs/user/rst/quickref.html](http://docutils.sourceforge.net/docs/user/rst/quickref.html)找到。

# 文档目录

虽然这个`README`文件对于`QTicTacToe`已经足够了，但是一个更复杂的程序或库可能需要更健壮的文档。放置这样的文档的标准位置是在`docs`目录中。这个目录应该直接位于我们的项目根目录下，并且可以包含任何类型的额外文档，包括以下内容：

+   示例配置文件

+   用户手册

+   API 文档

+   数据库图表

由于我们的程序不需要这些东西，所以我们不需要在这个项目中添加`docs`目录。

# `requirements.txt`文件

Python 程序通常需要标准库之外的包才能运行，用户需要知道安装什么才能让你的项目运行。你可以（而且可能应该）将这些信息放在`README`文件中，但你也应该将它放在`requirements.txt`中。

`requirements.txt`的格式是每行一个库，如下所示：

```py
PyQt5
PyQt5-sip
```

这个文件中的库名称应该与 PyPI 中使用的名称相匹配，因为这个文件可以被`pip`用来安装项目所需的所有库，如下所示：

```py
$ pip  install --user -r requirements.txt
```

我们实际上不需要指定`PyQt5-sip`，因为它是`PyQt5`的依赖项，会自动安装。我们在这里添加它是为了展示如何指定多个库。

如果需要特定版本的库，也可以使用版本说明符进行说明：

```py
PyQt5 >= 5.12
PyQt5-sip == 4.19.4
```

在这种情况下，我们指定了`PyQt5`版本`5.12`或更高，并且只有`PyQt5-sip`的`4.19.4`版本。

关于`requirements.txt`文件的更多信息可以在[`pip.readthedocs.io/en/1.1/requirements.html`](https://pip.readthedocs.io/en/1.1/requirements.html)找到。

# 其他文件

这些是项目文档和元数据的基本要素，但在某些情况下，你可能会发现一些额外的文件有用：

+   `TODO.txt`：需要处理的错误或缺失功能的简要列表

+   `CHANGELOG.txt`：主要项目变更和发布历史的日志

+   `tests`：包含模块单元测试的目录

+   `scripts`：包含对你的模块有用但不是其一部分的 Python 或 shell 脚本的目录

+   `Makefile`：一些项目受益于脚本化的构建过程，对此，像`make`这样的实用工具可能会有所帮助；其他选择包括 CMake、SCons 或 Waf

不过，此时你的项目已经准备好上传到你喜欢的源代码共享站点。在下一节中，我们将看看如何为 PyPI 做好准备。

# 使用 setuptools 进行分发

在本书的许多部分，你已经使用`pip`安装了 Python 包。你可能知道`pip`会从 PyPI 下载这些包，并将它们安装到你的系统、Python 虚拟环境或用户环境中。你可能不知道的是，用于创建和安装这些包的工具称为`setuptools`，如果我们想要为 PyPI 或个人使用制作自己的包，它就可以随时为我们提供。

尽管`setuptools`是官方推荐的用于创建 Python 包的工具，但它并不是标准库的一部分。但是，如果你在安装过程中选择包括`pip`，它通常会包含在大多数操作系统的默认发行版中。如果由于某种原因你没有安装`setuptools`，请参阅[`setuptools.readthedocs.io/en/latest/`](https://setuptools.readthedocs.io/en/latest/)上的文档，了解如何在你的平台上安装它。

使用`setuptools`的主要任务是编写一个`setup.py`脚本。在本节中，我们将学习如何编写和使用我们的`setup.py`脚本来生成可分发的包。

# 编写 setuptools 配置

`setup.py`的主要目的是使用关键字参数调用`setuptools.setup()`函数，这将定义我们项目的元数据以及我们的项目应该如何打包和安装。

因此，我们将首先导入该函数：

```py
from setuptools import setup

setup(
    # Arguments here
)
```

`setup.py`中的剩余代码将作为`setup()`的关键字参数。让我们来看看这些参数的不同类别。

# 基本元数据参数

最简单的参数涉及项目的基本元数据：

```py
    name='QTicTacToe',
    version='1.0',
    author='Alan D Moore',
    author_email='alandmoore@example.com',
    description='The classic game of noughts and crosses',
    url="http://qtictactoe.example.com",
    license='MIT',
```

在这里，我们已经描述了包名称、版本、简短描述、项目 URL 和许可证，以及作者的姓名和电子邮件。这些信息将被写入包元数据，并被 PyPI 等网站使用，以构建项目的个人资料页面。

例如，看一下 PyQt5 的 PyPI 页面：

![](img/816fd19d-7d1a-4e06-88c6-53ff5541c532.png)

在页面的左侧，你会看到一个指向项目主页的链接，作者（带有超链接的电子邮件地址）和许可证。在顶部，你会看到项目名称和版本，以及项目的简短描述。所有这些数据都可以从项目的`setup.py`脚本中提取出来。

如果你计划向 PyPI 提交一个包，请参阅[`www.python.org/dev/peps/pep-0440/`](https://www.python.org/dev/peps/pep-0440/)上的 PEP 440，了解你的版本号应该如何指定。

你在这个页面的主体中看到的长文本来自`long_description`参数。我们可以直接将一个长字符串放入这个参数，但既然我们已经有了一个很好的`README.rst`文件，为什么不在这里使用呢？由于`setup.py`是一个 Python 脚本，我们可以直接读取文件的内容，就像这样：

```py
    long_description=open('README.rst', 'r').read(),
```

在这里使用 RST 的一个优点是，PyPI（以及许多其他代码共享站点）将自动将你的标记渲染成格式良好的 HTML。

如果我们希望使我们的项目更容易搜索，我们可以包含一串空格分隔的关键字：

```py
    keywords='game multiplayer example pyqt5',
```

在这种情况下，搜索 PyPI 中的“multiplayer pyqt5”的人应该能找到我们的项目。

最后，你可以包含一个与项目相关的 URL 字典：

```py
    project_urls={
        'Author Website': 'https://www.alandmoore.com',
        'Publisher Website': 'https://packtpub.com',
        'Source Code': 'https://git.example.com/qtictactoe'
    },
```

格式为`{'label': 'URL'}`；你可能会在这里包括项目的 bug 跟踪器、文档站点、Wiki 页面或源代码库，特别是如果其中任何一个与主页 URL 不同的话。

# 包和依赖关系

除了建立基本元数据外，`setup()`还需要有关需要包含的实际代码或需要在系统上存在的环境的信息，以便执行此包。

这里我们需要处理的第一个关键字是`packages`，它定义了我们项目中需要包含的模块：

```py
    packages=['qtictactoe', 'qtictactoe.images'],
```

请注意，我们需要明确包括`qtictactoe`模块和`qtictactoe.images`模块；即使`images`目录位于`qtictactoe`下，也不会自动包含它。

如果我们有很多子模块，并且不想明确列出它们，`setuptools`也提供了自动解决方案：

```py
from setuptools import setup, find_package

setup(
    #...
    packages=find_packages(),
)
```

如果要使用`find_packages`，请确保每个子模块都有一个`__init__.py`文件，以便`setuputils`可以将其识别为模块。在这种情况下，您需要在`images`文件夹中添加一个`__init__.py`文件，否则它将被忽略。

这两种方法都有优点和缺点；手动方法更费力，但`find_packages`有时可能在某些情况下无法识别库。

我们还需要指定此项目运行所需的外部库，例如`PyQt5`。可以使用`install_requires`关键字来完成：

```py
    install_requires=['PyQt5'],
```

这个关键字接受一个包名列表，这些包必须被安装才能安装程序。当使用`pip`安装程序时，它将使用此列表自动安装所有依赖包。您应该在此列表中包括任何不属于标准库的内容。

就像`requirements.txt`文件一样，我们甚至可以明确指定每个依赖项所需的版本号：

```py
    install_requires=['PyQt5 >= 5.12'],
```

在这种情况下，`pip`将确保安装大于或等于 5.12 的 PyQt5 版本。如果未指定版本，`pip`将安装 PyPI 提供的最新版本。

在某些情况下，我们可能还需要指定特定版本的 Python；例如，我们的项目使用 f-strings，这是 Python 3.6 或更高版本才有的功能。我们可以使用`python_requires`关键字来指定：

```py
    python_requires='>=3.6',
```

我们还可以为可选功能指定依赖项；例如，如果我们为`qtictactoe`添加了一个可选的网络游戏功能，需要`requests`库，我们可以这样指定：

```py
    extras_require={
        "NetworkPlay": ["requests"]
    }
```

`extras_require`关键字接受一个特性名称（可以是任何您想要的内容）到包名称列表的映射。这些模块在安装您的包时不会自动安装，但其他模块可以依赖于这些子特性。例如，另一个模块可以指定对我们项目的`NetworkPlay`额外关键字的依赖，如下所示：

```py
    install_requires=['QTicTacToe[NetworkPlay]'],
```

这将触发一系列依赖关系，导致安装`requests`库。

# 非 Python 文件

默认情况下，`setuptools`将打包在我们项目中找到的 Python 文件，其他文件类型将被忽略。然而，在几乎任何项目中，都会有一些非 Python 文件需要包含在我们的分发包中。这些文件通常分为两类：一类是 Python 模块的一部分，比如我们的 PNG 文件，另一类是不是，比如`README`文件。

要包含*不*是 Python 包的文件，我们需要创建一个名为`MANIFEST.in`的文件。此文件包含项目根目录下文件路径的`include`指令。例如，如果我们想要包含我们的文档文件，我们的文件应该如下所示：

```py
include README.rst
include LICENSE
include requirements.txt
include docs/*
```

格式很简单：单词`include`后跟文件名、路径或匹配一组文件的模式。所有路径都是相对于项目根目录的。

要包含 Python 包的文件，我们有两种选择。

一种方法是将它们包含在`MANIFEST.in`文件中，然后在`setup.py`中将`include_package_data`设置为`True`：

```py
    include_package_data=True,
```

包含非 Python 文件的另一种方法是在`setup.py`中使用`package_data`关键字参数：

```py
    package_data={
        'qtictactoe.images': ['*.png'],
        '': ['*.txt', '*.rst']
    },
```

这个参数接受一个`dict`对象，其中每个条目都是一个模块路径和一个匹配包含的文件的模式列表。在这种情况下，我们希望包括在`qtictactoe.images`模块中找到的所有 PNG 文件，以及包中任何位置的 TXT 或 RST 文件。请记住，这个参数只适用于*模块目录中*的文件（即`qtictactoe`下的文件）。如果我们想要包括诸如`README.rst`或`run.py`之类的文件，那些应该放在`MANIFEST.in`文件中。

您可以使用任一方法来包含文件，但您不能在同一个项目中同时使用*两种*方法；如果启用了`include_package_data`，则将忽略`package_data`指令。

# 可执行文件

我们倾向于将 PyPI 视为安装 Python 库的工具；事实上，它也很适合安装应用程序，并且许多 Python 应用程序都可以从中获取。即使你正在创建一个库，你的库很可能会随附可执行的实用程序，比如 PyQt5 附带的`pyrcc5`和`pyuic5`实用程序。

为了满足这些需求，`setuputils` 为我们提供了一种指定特定函数或方法作为控制台脚本的方法；当安装包时，它将创建一个简单的可执行文件，在从命令行执行时将调用该函数或方法。

这是使用`entry_points`关键字指定的：

```py
    entry_points={
        'console_scripts': [
            'qtictactoe = qtictactoe.__main__:main'
        ]
    }
```

`entry_points`字典还有其他用途，但我们最关心的是`'console_scripts'`键。这个键指向一个字符串列表，指定我们想要设置为命令行脚本的函数。这些字符串的格式如下：

```py
'command_name = module.submodule:function'
```

您可以添加尽可能多的控制台脚本；它们只需要指向包中可以直接运行的函数或方法。请注意，您*必须*在这里指定一个实际的可调用对象；您不能只是指向一个要运行的 Python 文件。这就是为什么我们将所有执行代码放在`__main__.py`中的`main()`函数下的原因。

`setuptools`包含许多其他指令，用于处理不太常见的情况；有关完整列表，请参阅[`setuptools.readthedocs.io/en/latest/setuptools.html`](https://setuptools.readthedocs.io/en/latest/setuptools.html)。

# 源码分发

现在`setup.py`已经准备就绪，我们可以使用它来实际创建我们的软件包分发。软件包分发有两种基本类型：`源码`和`构建`。在本节中，我们将讨论如何使用**源码分发**。

源码分发是我们构建项目所需的所有源代码和额外文件的捆绑包。它包括`setup.py`文件，并且对于以跨平台方式分发您的项目非常有用。

# 创建源码分发

要构建源码分发，打开项目根目录中的命令提示符，并输入以下命令：

```py
$ python3 setup.py sdist
```

这将创建一些目录和许多文件：

+   `ProjectName.egg-info`目录（在我们的情况下是`QTicTacToe.egg-info`目录）将包含从我们的`setup.py`参数生成的几个元数据文件。

+   `dist`目录将包含包含我们分发的`tar.gz`存档文件。我们的文件名为`QTicTacToe-1.0.tar.gz`。

花几分钟时间来探索`QTicTacToe.egg-info`的内容；您会看到我们在`setup()`中指定的所有信息以某种形式存在。这个目录也包含在源码分发中。

此外，花点时间打开`tar.gz`文件，看看它包含了什么；你会看到我们在`MANIFEST.in`中指定的所有文件，以及`qtictactoe`模块和来自`QTicTacToe.egg-info`的所有文件。基本上，这是我们项目目录的完整副本。

Linux 和 macOS 原生支持`tar.gz`存档；在 Windows 上，您可以使用免费的 7-Zip 实用程序。有关 7-Zip 的信息，请参阅*技术要求*部分。

# 安装源码分发

源分发可以使用`pip`进行安装；为了在一个干净的环境中看到这是如何工作的，我们将在 Python 的**虚拟环境**中安装我们的库。虚拟环境是创建一个隔离的 Python 堆栈的一种方式，您可以在其中独立于系统 Python 安装添加或删除库。

在控制台窗口中，创建一个新目录，然后将其设置为虚拟环境：

```py
$ mkdir test_env
$ virtualenv -p python3 test_env
```

`virtualenv`命令将必要的文件复制到给定目录，以便可以运行 Python，以及一些激活和停用环境的脚本。

要开始使用您的新环境，请运行此命令：

```py
# On Linux and Mac
$ source test_env/bin/activate
# On Windows
$ test_env\Scripts\activate
```

根据您的平台，您的命令行提示可能会更改以指示您处于虚拟环境中。现在当您运行`python`或 Python 相关工具，如`pip`时，它们将在虚拟环境中执行所有操作，而不是在您的系统 Python 中执行。

让我们安装我们的源分发包：

```py
$ pip install QTicTacToe/dist/QTicTacToe-1.0.tar.gz
```

此命令将导致`pip`提取我们的源分发并在项目根目录内执行`python setup.py install`。`install`指令将下载任何依赖项，构建一个入口点可执行文件，并将代码复制到存储 Python 库的目录中（在我们的虚拟环境的情况下，那将是`test_env/lib/python3.7/site-packages/`）。请注意，`PyQt5`的一个新副本被下载；您的虚拟环境中除了 Python 和标准库之外没有安装任何依赖项，因此我们在`install_requires`中列出的任何依赖项都必须重新安装。

在`pip`完成后，您应该能够运行`qtictactoe`命令并成功启动应用程序。该命令存储在`test_env/bin`中，以防您的操作系统不会自动将虚拟环境目录附加到您的`PATH`。

要从虚拟环境中删除包，可以运行以下命令：

```py
$ pip uninstall QTicTacToe
```

这应该清理源代码和所有生成的文件。

# 构建分发

源分发对开发人员至关重要，但它们通常包含许多对最终用户不必要的元素，例如单元测试或示例代码。除此之外，如果项目包含编译代码（例如用 C 编写的 Python 扩展），那么该代码在目标上使用之前将需要编译。为了解决这个问题，`setuptools`提供了各种**构建分发**类型。构建分发提供了一组准备好的文件，只需要将其复制到适当的目录中即可使用。

在本节中，我们将讨论如何使用构建分发。

# 构建分发的类型

创建构建分发的第一步是确定我们想要的构建分发类型。`setuptools`库提供了一些不同的构建分发类型，我们可以安装其他库以添加更多选项。

内置类型如下：

+   **二进制分发**：这是一个`tar.gz`文件，就像源分发一样，但与源分发不同，它包含预编译的代码（例如`qtictactoe`可执行文件），并省略了某些类型的文件（例如测试）。构建分发的内容需要被提取和复制到适当的位置才能运行。

+   **Windows 安装程序**：这与二进制分发类似，只是它是一个在 Windows 上启动安装向导的可执行文件。向导仅用于将文件复制到适当的位置以供执行或库使用。

+   **RPM 软件包管理器**（**RPM**）**安装程序**：再次，这与二进制分发类似，只是它将代码打包在一个 RPM 文件中。RPM 文件被用于几个 Linux 发行版的软件包管理工具（如 Red Hat、CentOS、Suse、Fedora 等）。

虽然您可能会发现这些分发类型在某些情况下很有用，但它们在 2019 年都有点过时；今天分发 Python 的标准方式是使用**wheel 分发**。这些是您在 PyPI 上找到的二进制分发包。

让我们来看看如何创建和安装 wheel 包。

# 创建 wheel 分发

要创建一个 wheel 分发，您首先需要确保从 PyPI 安装了`wheel`库（请参阅*技术要求*部分）。之后，`setuptools`将有一个额外的`bdist_wheel`选项。

您可以使用以下方法创建您的`wheel`文件：

```py
$ python3 setup.py bdist_wheel
```

就像以前一样，这个命令将创建`QTicTacToe.egg-info`目录，并用包含您项目元数据的文件填充它。它还创建一个`build`目录，在那里编译文件被分阶段地压缩成`wheel`文件。

在`dist`下，我们会找到我们完成的`wheel`文件。在我们的情况下，它被称为`QTicTacToe-1.0-py3-none-any.whl`。文件名的格式如下：

+   项目名称（`QTicTacToe`）。

+   版本（1.0）。

+   支持的 Python 版本，无论是 2、3 还是`universal`（`py3`）。

+   `ABI`标签，它表示我们的项目依赖的 Python 二进制接口的特定版本（`none`）。如果我们已经编译了代码，这将被使用。

+   平台（操作系统和 CPU 架构）。我们的是`any`，因为我们没有包含任何特定平台的二进制文件。

二进制分发有三种类型：

+   **通用**类型只有 Python，并且与 Python 2 或 3 兼容

+   **纯 Python**类型只有 Python，但与 Python 2 或 Python 3 兼容

+   **平台**类型包括只在特定平台上运行的已编译代码

正如分发名称所反映的那样，我们的包是纯 Python 类型，因为它不包含已编译的代码，只支持 Python 3。PyQt5 是一个平台包类型的例子，因为它包含为特定平台编译的 Qt 库。

回想一下第十五章，*树莓派上的 PyQt*，我们无法在树莓派上从 PyPI 安装 PyQt，因为 Linux ARM 平台上没有`wheel`文件。由于 PyQt5 是一个平台包类型，它只能安装在已生成此`wheel`文件的平台上。

# 安装构建的分发

与源分发一样，我们可以使用`pip`安装我们的 wheel 文件：

```py
$ pip install qtictactoe/dist/QTicTacToe-1.0-py3-none-any.whl
```

如果您在一个新的虚拟环境中尝试这个，您应该会发现，PyQt5 再次从 PyPI 下载并安装，并且您之后可以使用`qtictactoe`命令。对于像`QTicTacToe`这样的程序，对最终用户来说并没有太大的区别，但对于一个包含需要编译的二进制文件的库（如 PyQt5）来说，这使得设置变得相当不那么麻烦。

当然，即使`wheel`文件也需要目标系统安装了 Python 和`pip`，并且可以访问互联网和 PyPI。这对许多用户或计算环境来说仍然是一个很大的要求。在下一节中，我们将探讨一个工具，它将允许我们从我们的 Python 项目创建一个独立的可执行文件，而无需任何先决条件。

# 使用 PyInstaller 编译

成功编写他们的第一个应用程序后，许多 Python 程序员最常见的问题是*如何将这段代码制作成可执行文件？*不幸的是，对于这个问题并没有一个单一的官方答案。多年来，许多项目已经启动来解决这个任务（例如 Py2Exe、cx_Freeze、Nuitka 和 PyInstaller 等），它们在支持程度、使用简单性和结果一致性方面各有不同。在这些特性方面，目前最好的选择是**PyInstaller**。

# PyInstaller 概述

Python 是一种解释语言；与 C 或 C++编译成机器代码不同，您的 Python 代码（或称为**字节码**的优化版本）每次运行时都会被 Python 解释器读取和执行。这使得 Python 具有一些使其非常易于使用的特性，但也使得它难以编译成机器代码以提供传统的独立可执行文件。

PyInstaller 通过将您的脚本与 Python 解释器以及运行所需的任何库或二进制文件打包在一起来解决这个问题。这些东西被捆绑在一起，形成一个目录或一个单一文件，以提供一个可分发的应用程序，可以复制到任何系统并执行，即使该系统没有 Python。

要查看这是如何工作的，请确保您已经从 PyPI 安装了 PyInstaller（请参阅*技术要求*部分），然后让我们为`QTicTacToe`创建一个可执行文件。

请注意，PyInstaller 创建的应用程序包是特定于平台的，只能在与编译平台兼容的操作系统和 CPU 架构上运行。例如，如果您在 64 位 Linux 上构建 PyInstaller 可执行文件，则它将无法在 32 位 Linux 或 64 位 Windows 上运行。

# 基本的命令行用法

理论上，使用 PyInstaller 就像打开命令提示符并输入这个命令一样简单：

```py
$ pyinstaller my_python_script.py
```

实际上，让我们尝试一下，使用第四章中的`qt_template.py`文件，*使用 QMainWindow 构建应用程序*；将其复制到一个空目录，并在该目录中运行`pyinstaller qt_template.py`。

您将在控制台上获得大量输出，并发现生成了几个目录和文件：

+   `build`和`__pycache__`目录主要包含在构建过程中生成的中间文件。这些文件在调试过程中可能有所帮助，但它们不是最终产品的一部分。

+   `dist`目录包含我们的可分发输出。

+   `qt_template.spec`文件保存了 PyInstaller 生成的配置数据。

默认情况下，PyInstaller 会生成一个包含可执行文件以及运行所需的所有库和数据文件的目录。如果要运行可执行文件，整个目录必须复制到另一台计算机上。

进入这个目录，寻找一个名为`qt_template`的可执行文件。如果运行它，您应该会看到一个空白的`QMainWindow`对象弹出。

如果您更喜欢只有一个文件，PyInstaller 可以将这个目录压缩成一个单独的可执行文件，当运行时，它会将自身提取到临时位置并运行主可执行文件。

这可以通过`--onefile`参数来实现；删除`dist`和`build`的内容，然后运行这个命令：

```py
$ pyinstaller --onefile qt_template.py
```

现在，在`dist`下，您只会找到一个单一的`qt_template`可执行文件。再次运行它，您将看到我们的空白`QMainWindow`。请记住，虽然这种方法更整洁，但它会增加启动时间（因为应用程序需要被提取），并且如果您的应用程序打开本地文件，可能会产生一些复杂性，我们将在下面看到。

如果对代码、环境或构建规范进行了重大更改，最好删除`build`和`dist`目录，可能还有`.spec`文件。

在我们尝试打包`QTicTacToe`之前，让我们深入了解一下`.spec`文件。

# .spec 文件

`.spec`文件是一个 Python 语法的`config`文件，包含了关于我们构建的所有元数据。您可以将其视为 PyInstaller 对`setup.py`文件的回答。然而，与`setup.py`不同，`.spec`文件是自动生成的。这是在我们运行`pyinstaller`时发生的，使用了从我们的脚本和通过命令行开关传递的数据的组合。我们也可以只生成`.spec`文件（而不开始构建）使用`pyi-makespec`命令。

生成后，可以编辑`.spec`文件，然后将其传递回`pyinstaller`，以重新构建分发，而无需每次都指定命令行开关：

```py
$ pyinstaller qt_template.spec
```

要查看我们可能在这个文件中编辑的内容，再次运行`pyi-makespec qt_template.py`，然后在编辑器中打开`qt_template.spec`。在文件内部，您将发现正在创建四种对象：`Analysis`、`PYZ`、`EXE`和`COLLECT`。

`Analysis`构造函数接收有关我们的脚本、数据文件和库的信息。它使用这些信息来分析项目的依赖关系，并生成五个指向应包含在分发中的文件的路径表。这五个表是：

+   `scripts`：作为入口点的 Python 文件，将被转换为可执行文件

+   `pure`：脚本所需的纯 Python 模块

+   `binaries`：脚本所需的二进制库

+   `datas`：非 Python 数据文件，如文本文件或图像

+   `zipfiles`：任何压缩的 Python`.egg`文件

在我们的文件中，`Analysis`部分看起来像这样：

```py
a = Analysis(['qt_template.py'],
             pathex=['/home/alanm/temp/qt_template'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
```

您会看到 Python 脚本的名称、路径和许多空关键字参数。这些参数大多对应于输出表，并用于手动补充分析结果，以弥补 PyInstaller 未能检测到的内容，包括以下内容：

+   `binaries` 对应于`binaries`表。

+   `datas` 对应于`datas`表。

+   `hiddenimports` 对应于`pure`表。

+   `excludes` 允许我们排除可能已自动包含但实际上并不需要的模块。

+   `hookspath` 和 `runtime_hooks` 允许您手动指定 PyInstaller **hooks**；hooks 允许您覆盖分析的某些方面。它们通常用于处理棘手的依赖关系。

接下来创建的对象是`PYZ`对象：

```py
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
```

`PYZ` 对象表示在分析阶段检测到的所有纯 Python 脚本的压缩存档。我们项目中的所有纯 Python 脚本将被编译为字节码（.pyc）文件并打包到这个存档中。

注意`Analysis`和`PYZ`中都有`cipher`参数；这个参数可以使用 AES256 加密进一步混淆我们的 Python 字节码。虽然它不能完全阻止代码的解密和反编译，但如果您计划商业分发，它可以成为好奇心的有用威慑。要使用此选项，请在创建文件时使用`--key`参数指定一个加密字符串，如下所示：

```py
$ pyi-makespec --key=n0H4CK1ngPLZ qt_template.py
```

在`PYZ`部分之后，生成了一个`EXE()`对象：

```py
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='qt_template',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
```

`EXE` 对象表示可执行文件。这里的位置参数表示我们要捆绑到可执行文件中的所有文件表。目前，这只是压缩的 Python 库和主要脚本；如果我们指定了`--onefile`选项，其他表（`binaries`、`zipfiles`和`datas`）也会包含在这里。

`EXE`的关键字参数允许我们控制可执行文件的各个方面：

+   `name` 是可执行文件的文件名

+   `debug` 切换可执行文件的调试输出

+   `upx` 切换是否使用**UPX**压缩可执行文件

+   `console` 切换在 Windows 和 macOS 中以控制台或 GUI 模式运行程序；在 Linux 中，它没有效果

UPX 是一个可用于多个平台的免费可执行文件打包工具，网址为[`upx.github.io/`](https://upx.github.io/)。如果您已安装它，启用此参数可以使您的可执行文件更小。

该过程的最后阶段是生成一个`COLLECT`对象：

```py
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='qt_template')
```

这个对象将所有必要的文件收集到最终的分发目录中。它只在单目录模式下运行，其位置参数包括要包含在目录中的组件。我们还可以覆盖文件夹的其他一些方面，比如是否在二进制文件上使用 UPX 以及输出目录的名称。

现在我们对 PyInstaller 的工作原理有了更多的了解，让我们来打包 QTicTacToe。

# 为 PyInstaller 准备 QTicTacToe

PyInstaller 在处理单个脚本时非常简单，但是在处理我们的模块式项目安排时该如何工作呢？我们不能将 PyInstaller 指向我们的模块，因为它会返回一个错误；它需要指向一个作为入口点的 Python 脚本，比如我们的`run.py`文件。

这似乎有效：

```py
$ pyinstaller run.py
```

然而，生成的分发和可执行文件现在被称为`run`，这并不太好。您可能会想要将`run.py`更改为`qtictactoe.py`；事实上，一些关于 Python 打包的教程建议这种安排（即，将`run`脚本与主模块具有相同的名称）。

然而，如果您尝试这样做，您可能会发现出现以下错误：

```py
Traceback (most recent call last):
  File "qtictactoe/__init__.py", line 3, in <module>
    from .mainwindow import MainWindow
ModuleNotFoundError: No module named '__main__.mainwindow'; '__main__' is not a package
[3516] Failed to execute script qtictactoe
```

因为 Python 模块可以是`.py`文件或目录，PyInstaller 无法确定哪一个构成了`qtictactoe`模块，因此两者具有相同的名称将失败。

正确的方法是在创建我们的`.spec`文件或运行`pyinstaller`时使用`--name`开关：

```py
$ pyinstaller --name qtictactoe run.py
# or, to just create the spec file:
# pyi-makespec --name qtictactoe run.py
```

这将创建`qtictactoe.spec`并将`EXE`和`COLLECT`的`name`参数设置为`qtictactoe`，如下所示：

```py
exe = EXE(pyz,
          #...
          name='qtictactoe',
          #...
coll = COLLECT(exe,
               #...
               name='qtictactoe')
```

当然，这也可以通过手动编辑`.spec`文件来完成。

# 处理非 Python 文件

我们的程序运行了，但我们又回到了`'X'`和`'O'`图像不显示的旧问题。这里有两个问题：首先，我们的 PNG 文件没有包含在分发中，其次，即使它们包含在分发中，程序也无法找到它们。

要解决第一个问题，我们必须告诉 PyInstaller 在构建的`Analysis`阶段将我们的文件包含在`datas`表中。我们可以在命令行中这样做：

```py
# On Linux and macOS:
$ pyinstaller --name qtictactoe --add-data qtictactoe/images:images run.py
# On Windows:
$ pyinstaller --name qtictactoe --add-data qtictactoe\images;images run.py
```

`--add-data`参数接受一个源路径和一个目标路径，两者之间用冒号（在 macOS 和 Linux 上）或分号（在 Windows 上）分隔。源路径是相对于我们正在运行`pyinstaller`的项目根目录（在本例中为`QTicTacToe`）的，目标路径是相对于分发根文件夹的。

如果我们不想使用长而复杂的命令行，我们还可以更新`qtictactoe.spec`文件的`Analysis`部分：

```py
a = Analysis(['run.py'],
             #...
             datas=[('qtictactoe/images', 'images')],
```

在这里，源路径和目标路径只是`datas`列表中的一个元组。源值也可以是一个模式，例如`qtictactoe/images/*.png`。如果您使用这些更改运行`pyinstaller qtictactoe.spec`，您应该会在`dist/qtictactoe`中找到一个`images`目录，其中包含我们的 PNG 文件。

这解决了图像的第一个问题，但我们仍然需要解决第二个问题。在*使用 setuptools 进行分发*部分，我们通过使用`__file__`内置变量解决了定位 PNG 文件的问题。但是，当您从 PyInstaller 可执行文件运行时，`__file__`的值*不是*可执行文件的路径；它实际上是一个临时目录的路径，可执行文件在其中解压缩字节码。此目录的位置也会根据我们是处于单文件模式还是单目录模式而改变。为了解决这个问题，我们需要更新我们的代码以检测程序是否已制作成可执行文件，并且如果是，则使用不同的方法来定位文件。

当我们运行 PyInstaller 可执行文件时，PyInstaller 会向`sys`模块添加两个属性来帮助我们：

+   `sys.frozen`属性，其值为`True`

+   `sys._MEIPASS`属性，存储可执行目录的路径

因此，我们可以将我们的代码在`board.py`中更新为以下内容：

```py
        if getattr(sys, 'frozen', False):
            directory = sys._MEIPASS
        else:  # Not frozen
            directory = path.dirname(__file__)
        self.mark_pngs = {
            'X': qtg.QPixmap(path.join(directory, 'images', 'X.png')),
            'O': qtg.QPixmap(path.join(directory, 'images', 'O.png'))
        }
```

现在，在从冻结的 PyInstaller 环境中执行时，我们的代码将能够正确地定位文件。重新运行`pyinstaller qtictactoe.spec`，您应该会发现`X`和`O`图形正确显示。万岁！

如前所述，在 PyQt5 应用程序中更好的解决方案是使用第六章中讨论的 Qt 资源文件，*Styling Qt Applications*。对于非 PyQt 程序，`setuptools`库有一个名为`pkg_resources`的工具可能会有所帮助。

# 进一步调试

如果您的构建继续出现问题，有几种方法可以获取更多关于正在进行的情况的信息。

首先，确保您的代码作为 Python 脚本正确运行。如果在任何模块文件中存在语法错误或其他代码问题，分发将在没有它们的情况下构建。这些遗漏既不会中止构建，也不会在命令行输出中提到。

确认后，检查构建目录以获取 PyInstaller 正在执行的详细信息。在`build/projectname/`下，您应该看到一些文件，可以帮助您进行调试，包括这些：

+   `warn-projectname.txt`：这个文件包含`Analysis`过程输出的警告。其中一些是无意义的（通常只是无法在您的平台上找到特定于平台的库），但如果库有错误或无法找到，这些问题将在这里记录。

+   `.toc`文件：这些文件包含构建过程各阶段创建的目录表；例如，`Analysis-00.toc`显示了`Analysis()`中找到的目录。您可以检查这些文件，看看项目的依赖项是否被错误地识别或从错误的位置提取。

+   `base_library.zip`：此存档应包含您的应用程序使用的所有纯 Python 模块的 Python 字节码文件。您可以检查这个文件，看看是否有任何遗漏。

如果您需要更详细的输出，可以使用`--log-level`开关来增加输出的详细程度到`warn-projectname.txt`。设置为`DEBUG`将提供更多细节：

```py
$ pyinstaller --log-level DEBUG my_project.py
```

更多调试提示可以在[`pyinstaller.readthedocs.io/en/latest/when-things-go-wrong.html`](https://pyinstaller.readthedocs.io/en/latest/when-things-go-wrong.html)找到。

# 总结

在本章中，您学会了如何与他人分享您的项目。您学会了使您的项目目录具有最佳布局，以便您可以与其他 Python 编码人员和 Python 工具进行协作。您学会了如何使用`setuptools`为诸如 PyPI 之类的站点制作可分发的 Python 软件包。最后，您学会了如何使用 PyInstaller 将您的代码转换为可执行文件。

恭喜！您已经完成了这本书。到目前为止，您应该对使用 Python 和 PyQt5 从头开始开发引人入胜的 GUI 应用程序的能力感到自信。从基本的输入表单到高级的网络、数据库和多媒体应用程序，您现在有了创建和分发惊人程序的工具。即使我们涵盖了所有的主题，PyQt 中仍有更多的发现。继续学习，创造伟大的事物！

# 问题

尝试回答这些问题，以测试您从本章中学到的知识：

1.  您已经在一个名为`Scan & Print Tool-box.py`的文件中编写了一个 PyQt 应用程序。您想将其转换为模块化组织形式；您应该做出什么改变？

1.  您的 PyQt5 数据库应用程序有一组包含应用程序使用的查询的`.sql`文件。当您的应用程序是与`.sql`文件在同一目录中的单个脚本时，它可以正常工作，但是现在您已将其转换为模块化组织形式后，无法找到查询。您应该怎么做？

1.  在将新应用程序上传到代码共享站点之前，您正在编写一个详细的`README.rst`文件来记录您的新应用程序。分别应使用哪些字符来下划线标记您的一级、二级和三级标题？

1.  您正在为您的项目创建一个`setup.py`脚本，以便您可以将其上传到 PyPI。您想要包括项目的常见问题解答页面的 URL。您该如何实现这一点？

1.  您在`setup.py`文件中指定了`include_package_data=True`，但由于某种原因，`docs`文件夹没有包含在您的分发包中。出了什么问题？

1.  您运行了`pyinstaller fight_fighter3.py`来将您的新游戏打包为可执行文件。然而出了些问题；您在哪里可以找到构建过程的日志？

1.  尽管名称如此，PyInstaller 实际上不能生成安装程序或包来安装您的应用程序。请为您选择的平台研究一些选项。

# 进一步阅读

有关更多信息，请参阅以下内容：

+   有关`ReStructuredText`标记的教程可以在[`docutils.sourceforge.net/docs/user/rst/quickstart.html`](http://docutils.sourceforge.net/docs/user/rst/quickstart.html)找到。

+   关于设计、构建、文档化和打包 Python GUI 应用程序的更多信息可以在作者的第一本书《Python GUI 编程与 Tkinter》中找到，该书可在 Packt Publications 上获得。

+   如果您有兴趣将软件包发布到 PyPI，请参阅[`blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/`](https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/)了解发布过程的教程。

+   解决在非 PyQt 代码中包含图像的问题的更好方法是`setuptools`提供的`pkg_resources`工具。您可以在[`setuptools.readthedocs.io/en/latest/pkg_resources.html`](https://setuptools.readthedocs.io/en/latest/pkg_resources.html)上了解更多信息。

+   PyInstaller 的高级用法在 PyInstaller 手册中有详细说明，可在[`pyinstaller.readthedocs.io/en/stable/`](https://pyinstaller.readthedocs.io/en/stable/)找到。
