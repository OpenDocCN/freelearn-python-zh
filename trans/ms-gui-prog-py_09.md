# 使用QtMultimedia处理音频-视频

无论是在游戏、通信还是媒体制作应用中，音频和视频内容通常是现代应用的重要组成部分。当使用本机API时，即使是最简单的音频-视频（AV）应用程序在支持多个平台时也可能非常复杂。然而，幸运的是，Qt为我们提供了一个简单的跨平台多媒体API，即`QtMultimedia`。使用`QtMultimedia`，我们可以轻松地处理音频内容、视频内容或摄像头和收音机等设备。

在这一章中，我们将使用`QtMultimedia`来探讨以下主题：

+   简单的音频播放

+   录制和播放音频

+   录制和播放视频

# 技术要求

除了[第1章](bce5f3b1-2979-4f78-817b-3986e7974725.xhtml)中描述的基本PyQt设置外，您还需要确保已安装`QtMultimedia`和`PyQt.QtMultimedia`库。如果您使用`pip`安装了PyQt5，则应该已经安装了。使用发行版软件包管理器的Linux用户应检查这些软件包是否已安装。

您可能还想从我们的GitHub存储库[https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter07](https://github.com/PacktPublishing/Mastering-GUI-Programming-with-Python/tree/master/Chapter07)下载代码，其中包含示例代码和用于这些示例的音频数据。

如果您想创建自己的音频文件进行处理，您可能需要安装免费的Audacity音频编辑器，网址为[https://www.audacityteam.org/](https://www.audacityteam.org/)。

最后，如果您的计算机没有工作的音频系统、麦克风和网络摄像头，您将无法充分利用本章。如果没有，那么其中一些示例将无法为您工作。

查看以下视频以查看代码的实际操作：[http://bit.ly/2Mjr8vx](http://bit.ly/2Mjr8vx)

# 简单的音频播放

很多时候，应用程序需要对GUI事件做出声音回应，就像在游戏中一样，或者只是为用户操作提供音频反馈。对于这种应用程序，`QtMultimedia`提供了`QSoundEffect`类。`QSoundEffect`仅限于播放未压缩音频，因此它可以使用**脉冲编码调制**（**PCM**）、**波形数据**（**WAV**）文件，但不能使用MP3或OGG文件。这样做的好处是它的延迟低，资源利用率非常高，因此虽然它不适用于通用音频播放器，但非常适合快速播放音效。

为了演示`QSoundEffect`，让我们构建一个电话拨号器。将[第4章](9281bd2a-64a1-4128-92b0-e4871b79c040.xhtml)中的应用程序模板*使用QMainWindow构建应用程序*复制到一个名为`phone_dialer.py`的新文件中，并在编辑器中打开它。

让我们首先导入`QtMultimedia`库，如下所示：

```py
from PyQt5 import QtMultimedia as qtmm
```

导入`QtMultimedia`将是本章所有示例的必要第一步，我们将一贯使用`qtmm`作为其别名。

我们还将导入一个包含必要的WAV数据的`resources`库：

```py
import resources
```

这个`resources`文件包含一系列**双音多频**（**DTMF**）音调。这些是电话拨号时电话生成的音调，我们包括了`0`到`9`、`*`和`#`。我们已经在示例代码中包含了这个文件；或者，您可以从自己的音频样本创建自己的`resources`文件（您可以参考[第6章](c3eb2567-0e73-4c37-9a9e-a0e2311e106c.xhtml)中关于如何做到这一点的信息）。

您可以使用免费的Audacity音频编辑器生成DTMF音调。要这样做，请从Audacity的主菜单中选择生成|DTMF。

一旦完成这些，我们将创建一个`QPushButton`子类，当单击时会播放声音效果，如下所示：

```py
class SoundButton(qtw.QPushButton):

    def __init__(self, wav_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wav_file = wav_file
        self.player = qtmm.QSoundEffect()
        self.player.setSource(qtc.QUrl.fromLocalFile(wav_file))
        self.clicked.connect(self.player.play)
```

如您所见，我们修改了构造函数以接受声音文件路径作为参数。这个值被转换为`QUrl`并通过`setSource()`方法传递到我们的`QSoundEffect`对象中。最后，`QSoundEffect.play()`方法触发声音的播放，因此我们将其连接到按钮的`clicked`信号。这就是创建我们的`SoundButton`对象所需的全部内容。

回到`MainWindow.__init__()`方法，让我们创建一些`SoundButton`对象并将它们排列在GUI中：

```py
        dialpad = qtw.QWidget()
        self.setCentralWidget(dialpad)
        dialpad.setLayout(qtw.QGridLayout())

        for i, symbol in enumerate('123456789*0#'):
            button = SoundButton(f':/dtmf/{symbol}.wav', symbol)
            row = i // 3
            column = i % 3
            dialpad.layout().addWidget(button, row, column)
```

我们已经设置了资源文件，以便可以通过`dtmf`前缀下的符号访问每个DTMF音调；例如，`':/dtmf/1.wav'`指的是1的DTMF音调。通过这种方式，我们可以遍历一串符号并为每个创建一个`SoundButton`对象，然后将其添加到三列网格中。

就是这样；运行这个程序并按下按钮。它应该听起来就像拨打电话！

# 录制和播放音频

`QSoundEffect`足以处理简单的事件声音，但对于更高级的音频项目，我们需要具备更多功能的东西。理想情况下，我们希望能够加载更多格式，控制播放的各个方面，并录制新的声音。

在这一部分，我们将专注于提供这些功能的两个类：

+   `QMediaPlayer`类，它类似于一个虚拟媒体播放器设备，可以加载音频或视频内容

+   `QAudioRecorder`类，用于管理将音频数据录制到磁盘

为了看到这些类的实际效果，我们将构建一个采样音效板。

# 初始设置

首先，制作一个新的应用程序模板副本，并将其命名为`soundboard.py`。然后，像上一个项目一样导入`QtMultimedia`，并布局主界面。

在`MainWindow`构造函数中，添加以下代码：

```py
        rows = 3
        columns = 3
        soundboard = qtw.QWidget()
        soundboard.setLayout(qtw.QGridLayout())
        self.setCentralWidget(soundboard)
        for c in range(columns):
            for r in range(rows):
                sw = SoundWidget()
                soundboard.layout().addWidget(sw, c, r)
```

我们在这里所做的只是创建一个空的中央小部件，添加一个网格布局，然后用`3`行`3`列的`SoundWidget`对象填充它。

# 实现声音播放

我们的`SoundWidget`类将是一个管理单个声音样本的`QWidget`对象。完成后，它将允许我们加载或录制音频样本，循环播放或单次播放，并控制其音量和播放位置。

在`MainWindow`构造函数之前，让我们创建这个类并给它一个布局：

```py
class SoundWidget(qtw.QWidget):

    def __init__(self):
        super().__init__()
        self.setLayout(qtw.QGridLayout())
        self.label = qtw.QLabel("No file loaded")
        self.layout().addWidget(self.label, 0, 0, 1, 2)
```

我们添加的第一件事是一个标签，它将显示小部件加载的样本文件的名称。我们需要的下一件事是一个控制播放的按钮。我们不只是一个普通的按钮，让我们运用一些我们的样式技巧来创建一个可以在播放按钮和停止按钮之间切换的自定义按钮。 

在`SoundWidget`类的上方开始一个`PlayButton`类，如下所示：

```py
class PlayButton(qtw.QPushButton):
    play_stylesheet = 'background-color: lightgreen; color: black;'
    stop_stylesheet = 'background-color: darkred; color: white;'

    def __init__(self):
        super().__init__('Play')
        self.setFont(qtg.QFont('Sans', 32, qtg.QFont.Bold))
        self.setSizePolicy(
            qtw.QSizePolicy.Expanding,
            qtw.QSizePolicy.Expanding
        )
        self.setStyleSheet(self.play_stylesheet)
```

回到`SoundWidget`类，我们将添加一个`PlayButton`对象，如下所示：

```py
        self.play_button = PlayButton()
        self.layout().addWidget(self.play_button, 3, 0, 1, 2)
```

现在我们有了一个控制按钮，我们需要创建将播放采样的`QMediaPlayer`对象，如下所示：

```py
        self.player = qtmm.QMediaPlayer()
```

您可以将`QMediaPlayer`视为硬件媒体播放器（如CD或蓝光播放器）的软件等效物。就像硬件媒体播放器有播放、暂停和停止按钮一样，`QMediaPlayer`对象有`play()`、`stop()`和`pause()`槽来控制媒体的播放。

让我们将我们的双功能`PlayButton`对象连接到播放器。我们将通过一个名为`on_playbutton()`的实例方法来实现这一点：

```py
        self.play_button.clicked.connect(self.on_playbutton)
```

`SoundWidget.on_playbutton()`将如何看起来：

```py
    def on_playbutton(self):
        if self.player.state() == qtmm.QMediaPlayer.PlayingState:
            self.player.stop()
        else:
            self.player.play()
```

这种方法检查了播放器对象的`state`属性，该属性返回一个常量，指示播放器当前是正在播放、已暂停还是已停止。如果播放器当前正在播放，我们就停止它；如果没有，我们就要求它播放。

由于我们的按钮在播放和停止按钮之间切换，让我们更新它的标签和外观。`QMediaPlayer`在其状态改变时发出`stateChanged`信号，我们可以将其发送到我们的`PlayButton`对象，如下所示：

```py
        self.player.stateChanged.connect(self.play_button.on_state_changed)
```

回到`PlayButton`类，让我们处理该信号，如下所示：

```py
    def on_state_changed(self, state):
        if state == qtmm.QMediaPlayer.PlayingState:
            self.setStyleSheet(self.stop_stylesheet)
            self.setText('Stop')
        else:
            self.setStyleSheet(self.play_stylesheet)
            self.setText('Play')
```

在这里，`stateChanged`传递了媒体播放器的新状态，我们用它来设置按钮的播放或停止外观。

# 加载媒体

就像硬件媒体播放器需要加载CD、DVD或蓝光光盘才能实际播放任何内容一样，我们的`QMediaPlayer`在播放任何音频之前也需要加载某种内容。让我们探讨如何从文件中加载声音。

首先在`SoundWidget`布局中添加一个按钮，如下所示：

```py
        self.file_button = qtw.QPushButton(
            'Load File', clicked=self.get_file)
        self.layout().addWidget(self.file_button, 4, 0)
```

这个按钮调用`get_file()`方法，看起来是这样的：

```py
    def get_file(self):
        fn, _ = qtw.QFileDialog.getOpenFileUrl(
            self,
            "Select File",
            qtc.QDir.homePath(),
            "Audio files (*.wav *.flac *.mp3 *.ogg *.aiff);; All files (*)"
        )
        if fn:
            self.set_file(fn)
```

这个方法简单地调用`QFileDialog`来检索文件URL，然后将其传递给另一个方法`set_file()`，我们将在下面编写。我们已经设置了过滤器来查找五种常见的音频文件类型，但如果你有不同格式的音频，可以随意添加更多——`QMediaPlayer`在加载方面非常灵活。

请注意，我们正在调用`getOpenFileUrl()`，它返回一个`QUrl`对象，而不是文件路径字符串。`QMediaPlayer`更喜欢使用`QUrl`对象，因此这将节省我们一个转换步骤。

`set_file()`方法是我们最终将媒体加载到播放器中的地方：

```py
    def set_file(self, url):
        content = qtmm.QMediaContent(url)
        self.player.setMedia(content)
        self.label.setText(url.fileName())
```

在我们可以将URL传递给媒体播放器之前，我们必须将其包装在`QMediaContent`类中。这为播放器提供了播放内容所需的API。一旦包装好，我们就可以使用`QMediaPlayer.setMedia()`来加载它，然后它就准备好播放了。你可以将这个过程想象成将音频数据放入CD（`QMediaContent`对象），然后将CD加载到CD播放器中（使用`setMedia()`）。

作为最后的修饰，我们已经检索了加载文件的文件名，并将其放在标签中。

# 跟踪播放位置

此时，我们的声音板可以加载和播放样本，但是看到并控制播放位置会很好，特别是对于长样本。`QMediaPlayer`允许我们通过信号和槽来检索和控制播放位置，所以让我们从我们的GUI中来看一下。

首先创建一个`QSlider`小部件，如下所示：

```py
        self.position = qtw.QSlider(
            minimum=0, orientation=qtc.Qt.Horizontal)
        self.layout().addWidget(self.position, 1, 0, 1, 2)
```

`QSlider`是一个我们还没有看过的小部件；它只是一个滑块控件，可以用来输入最小值和最大值之间的整数。

现在连接滑块和播放器，如下所示：

```py
        self.player.positionChanged.connect(self.position.setSliderPosition)
        self.player.durationChanged.connect(self.position.setMaximum)
        self.position.sliderMoved.connect(self.player.setPosition)
```

`QMediaPlayer`类以表示从文件开始的毫秒数的整数报告其位置，因此我们可以将`positionChanged`信号连接到滑块的`setSliderPosition()`槽。

然而，我们还需要调整滑块的最大位置，使其与样本的持续时间相匹配，否则滑块将不知道值代表的百分比。因此，我们已经将播放器的`durationChanged`信号（每当新内容加载到播放器时发出）连接到滑块的`setMaximum()`槽。

最后，我们希望能够使用滑块来控制播放位置，因此我们将`sliderMoved`信号设置为播放器的`setPosition()`槽。请注意，我们绝对要使用`sliderMoved`而不是`valueChanged`（当用户*或*事件更改值时，`QSlider`发出的信号），因为后者会在媒体播放器更改位置时创建一个反馈循环。

这些连接是我们的滑块工作所需的全部。现在你可以运行程序并加载一个长声音；你会看到滑块跟踪播放位置，并且可以在播放之前或期间移动以改变位置。

# 循环音频

在一次性播放我们的样本很好，但我们也想循环播放它们。在`QMediaPlayer`对象中循环音频需要稍微不同的方法。我们需要先将`QMediaContent`对象添加到`QMediaPlayList`对象中，然后告诉播放列表循环播放。

回到我们的`set_file()`方法，我们需要对我们的代码进行以下更改：

```py
    def set_file(self, url):
        self.label.setText(url.fileName())
        content = qtmm.QMediaContent(url)
        #self.player.setMedia(content)
        self.playlist = qtmm.QMediaPlaylist()
        self.playlist.addMedia(content)
        self.playlist.setCurrentIndex(1)
        self.player.setPlaylist(self.playlist)
```

当然，一个播放列表可以加载多个文件，但在这种情况下，我们只想要一个。我们使用`addMedia（）`方法将`QMediaContent`对象加载到播放列表中，然后使用`setCurrentIndex（）`方法将播放列表指向该文件。请注意，播放列表不会自动指向任何项目。这意味着如果您跳过最后一步，当您尝试播放播放列表时将不会发生任何事情。

最后，我们使用媒体播放器的`setPlaylist（）`方法添加播放列表。

现在我们的内容在播放列表中，我们将创建一个复选框来切换循环播放的开关：

```py
        self.loop_cb = qtw.QCheckBox(
            'Loop', stateChanged=self.on_loop_cb)
        self.layout().addWidget(self.loop_cb, 2, 0)
```

正如您所看到的，我们正在将复选框的`stateChanged`信号连接到一个回调方法；该方法将如下所示：

```py
    def on_loop_cb(self, state):
        if state == qtc.Qt.Checked:
            self.playlist.setPlaybackMode(
                qtmm.QMediaPlaylist.CurrentItemInLoop)
        else:
            self.playlist.setPlaybackMode(
                qtmm.QMediaPlaylist.CurrentItemOnce)
```

`QMediaPlaylist`类的`playbackMode`属性与CD播放器上的曲目模式按钮非常相似，可以用于在重复、随机或顺序播放之间切换。如下表所示，有五种播放模式：

| 模式 | 描述 |
| --- | --- |
| `CurrentItemOnce` | 播放当前曲目一次，然后停止。 |
| `CurrentItemInLoop` | 重复播放当前项目。 |
| `顺序` | 播放所有项目，然后停止。 |
| `循环` | 播放所有项目，然后重复。 |
| `随机` | 以随机顺序播放所有项目。 |

在这种方法中，我们根据复选框是否被选中来在`CurrentItemOnce`和`CurrentItemInLoop`之间切换。由于我们的播放列表只有一个项目，剩下的模式是没有意义的。

最后，当加载新文件时，我们将清除复选框。因此，请将以下内容添加到`set_file（）`的末尾：

```py
        self.loop_cb.setChecked(False)
```

在这一点上，您应该能够运行程序并循环播放示例。请注意，使用此方法循环音频可能无法保证无缝循环；取决于您的平台和系统功能，循环的迭代之间可能会有一个小间隙。

# 设置音量

我们的最终播放功能将是音量控制。为了让我们能够控制播放级别，`QMediaPlayer`有一个接受值从`0`（静音）到`100`（最大音量）的`volume`参数。

我们将简单地添加另一个滑块小部件来控制音量，如下所示：

```py
        self.volume = qtw.QSlider(
            minimum=0,
            maximum=100,
            sliderPosition=75,
            orientation=qtc.Qt.Horizontal,
            sliderMoved=self.player.setVolume
        )
        self.layout().addWidget(self.volume, 2, 1)
```

在设置最小和最大值后，我们只需要将`sliderMoved`连接到媒体播放器的`setVolume（）`槽。就是这样！

为了更平滑地控制音量，Qt文档建议将滑块的线性刻度转换为对数刻度。我们建议您阅读[https://doc.qt.io/qt-5/qaudio.html#convertVolume](https://doc.qt.io/qt-5/qaudio.html#convertVolume)，看看您是否可以自己做到这一点。

# 实现录音

Qt中的音频录制是通过`QAudioRecorder`类实现的。就像`QMediaPlayer`类类似于媒体播放设备一样，`QAudioRecorder`类类似于媒体录制设备，例如数字音频录音机（或者如果您是作者的一代人，磁带录音机）。录音机使用`record（）`、`stop（）`和`pause（）`方法进行控制，就像媒体播放器对象一样。

让我们向我们的`SoundWidget`添加一个录音机对象，如下所示：

```py
        self.recorder = qtmm.QAudioRecorder()
```

为了控制录音机，我们将创建另一个双功能按钮类，类似于我们之前创建的播放按钮：

```py
class RecordButton(qtw.QPushButton):

    record_stylesheet = 'background-color: black; color: white;'
    stop_stylesheet = 'background-color: darkred; color: white;'

    def __init__(self):
        super().__init__('Record')

    def on_state_changed(self, state):
        if state == qtmm.QAudioRecorder.RecordingState:
            self.setStyleSheet(self.stop_stylesheet)
            self.setText('Stop')
        else:
            self.setStyleSheet(self.record_stylesheet)
            self.setText('Record')
```

就像`PlayButton`类一样，每当从录音机的`stateChanged`信号接收到新的`state`值时，我们就会切换按钮的外观。在这种情况下，我们正在寻找录音机的`RecordingState`状态。

让我们向我们的小部件添加一个`RecordButtoon（）`方法，如下所示：

```py
        self.record_button = RecordButton()
        self.recorder.stateChanged.connect(
            self.record_button.on_state_changed)
        self.layout().addWidget(self.record_button, 4, 1)
        self.record_button.clicked.connect(self.on_recordbutton)
```

我们已经将`clicked`信号连接到`on_recordbutton（）`方法，该方法将处理音频录制的开始和停止。

这个方法如下：

```py
    def on_recordbutton(self):
        if self.recorder.state() == qtmm.QMediaRecorder.RecordingState:
            self.recorder.stop()
            url = self.recorder.actualLocation()
            self.set_file(url)
```

我们将首先检查录音机的状态。如果它当前正在录制，那么我们将通过调用`recorder.stop()`来停止它，这不仅会停止录制，还会将录制的数据写入磁盘上的音频文件。然后，我们可以通过调用录音机的`actualLocation()`方法来获取该文件的位置。此方法返回一个`QUrl`对象，我们可以直接将其传递给`self.set_file()`以将我们的播放设置为新录制的文件。

确保使用`actualLocation()`获取文件的位置。可以使用`setLocation()`配置录制位置，并且此值可以从`location()`访问器中获取。但是，如果配置的位置无效或不可写，Qt可能会回退到默认设置。`actualLocation()`返回文件实际保存的URL。

如果我们当前没有录制，我们将通过调用`recorder.record()`来告诉录音机开始录制：

```py
        else:
            self.recorder.record()
```

当调用`record()`时，音频录制器将在后台开始录制音频，并将一直保持录制，直到调用`stop()`。

在我们可以播放录制的文件之前，我们需要对`set_file()`进行一次修复。在撰写本文时，`QAudioRecorder.actualLocation()`方法忽略了向URL添加方案值，因此我们需要手动指定这个值：

```py
    def set_file(self, url):
        if url.scheme() == '':
            url.setScheme('file')
        content = qtmm.QMediaContent(url)
        #...
```

在`QUrl`术语中，`scheme`对象指示URL的协议，例如HTTP、HTTPS或FTP。由于我们正在访问本地文件，因此方案应为`'file'`。

如果`QAudioRecorder`的默认设置在您的系统上正常工作，则应该能够录制和播放音频。但是，这是一个很大的*如果*；很可能您需要对音频录制器对象进行一些配置才能使其正常工作。让我们看看如何做到这一点。

# 检查和配置录音机

即使`QAudioRecorder`类对您来说运行良好，您可能会想知道是否有一种方法可以控制它记录的音频类型和质量，它从哪里记录音频，以及它将音频文件写入的位置。

为了配置这些内容，我们首先必须知道您的系统支持什么，因为对不同音频录制功能的支持可能取决于硬件、驱动程序或操作系统的能力。`QAudioRecorder`有一些方法可以提供有关可用功能的信息。

以下脚本将显示有关系统支持的音频功能的信息：

```py
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

app = QCoreApplication([])
r = QAudioRecorder()
print('Inputs: ', r.audioInputs())
print('Codecs: ', r.supportedAudioCodecs())
print('Sample Rates: ', r.supportedAudioSampleRates())
print('Containers: ', r.supportedContainers())
```

您可以在您的系统上运行此脚本并获取受支持的`Inputs`、`Codecs`、`Sample Rates`和`container`格式的列表。例如，在典型的Microsoft Windows系统上，您的结果可能如下所示：

```py
Inputs:  ['Microhpone (High Defnition Aud']
Codecs:  ['audio/pcm']
Sample Rates:  ([8000, 11025, 16000, 22050, 32000,
                 44100, 48000, 88200, 96000, 192000], False)
Containers:  ['audio/x-wav', 'audio/x-raw']
```

要为`QAudioRecorder`对象配置输入源，您需要将音频输入的名称传递给`setAudioInput()`方法，如下所示：

```py
        self.recorder.setAudioInput('default:')
```

输入的实际名称可能在您的系统上有所不同。不幸的是，当您设置无效的音频输入时，`QAudioRecorder`不会抛出异常或注册错误，它只是简单地无法录制任何音频。因此，如果决定自定义此属性，请务必确保该值首先是有效的。

要更改记录的输出文件，我们需要调用`setOutputLocation()`，如下所示：

```py
        sample_path = qtc.QDir.home().filePath('sample1')
        self.recorder.setOutputLocation(
            qtc.QUrl.fromLocalFile(sample_path))
```

请注意，`setOutputLocation()`需要一个`QUrl`对象，而不是文件路径。一旦设置，Qt将尝试使用此位置来录制音频。但是，如前所述，如果此位置不可用，它将恢复到特定于平台的默认值。

容器格式是保存音频数据的文件类型。例如，`audio/x-wav`是用于WAV文件的容器。我们可以使用`setContainerFormat()`方法在记录对象中设置此值，如下所示：

```py
        self.recorder.setContainerFormat('audio/x-wav')
```

此属性的值应为`QAudioRecorder.supportedContainers()`返回的字符串。使用无效值将在您尝试录制时导致错误。

设置编解码器、采样率和质量需要一个称为`QAudioEncoderSettings`对象的新对象。以下示例演示了如何创建和配置`settings`对象：

```py
        settings = qtmm.QAudioEncoderSettings()
        settings.setCodec('audio/pcm')
        settings.setSampleRate(44100)
        settings.setQuality(qtmm.QMultimedia.HighQuality)
        self.recorder.setEncodingSettings(settings)
```

在这种情况下，我们已经将我们的音频配置为使用PCM编解码器以`44100` Hz进行高质量编码。

请注意，并非所有编解码器都与所有容器类型兼容。如果选择了两种不兼容的类型，Qt将在控制台上打印错误并且录制将失败，但不会崩溃或抛出异常。您需要进行适当的研究和测试，以确保您选择了兼容的设置。

根据所选择的编解码器，您可以在`QAudioEncoderSettings`对象上设置其他设置。您可以在[https://doc.qt.io/qt-5/qaudioencodersettings.html](https://doc.qt.io/qt-5/qaudioencodersettings.html)的Qt文档中查阅更多信息。

配置音频设置可能非常棘手，特别是因为支持在各个系统之间差异很大。最好在可以的时候让Qt使用其默认设置，或者让用户使用从`QAudioRecorder`的支持检测方法获得的值来配置这些设置。无论您做什么，如果您不能保证运行您的软件的系统将支持它们，请不要硬编码设置或选项。

# 录制和播放视频

一旦您了解了如何在Qt中处理音频，处理视频只是在复杂性方面迈出了一小步。就像处理音频一样，我们将使用一个播放器对象来加载和播放内容，以及一个记录器对象来记录它。但是，对于视频，我们需要添加一些额外的组件来处理内容的可视化并初始化源设备。

为了理解它是如何工作的，我们将构建一个视频日志应用程序。将应用程序模板从[第4章](9281bd2a-64a1-4128-92b0-e4871b79c040.xhtml) *使用QMainWindow构建应用程序*复制到一个名为`captains_log.py`的新文件中，然后我们将开始编码。

# 构建基本GUI

**船长的日志**应用程序将允许我们从网络摄像头录制视频到一个预设目录中的时间戳文件，并进行回放。我们的界面将在右侧显示过去日志的列表，在左侧显示预览/回放区域。我们将有一个分页式界面，以便用户可以在回放和录制模式之间切换。

在`MainWindow.__init__()`中，按照以下方式开始布局基本GUI：

```py
        base_widget = qtw.QWidget()
        base_widget.setLayout(qtw.QHBoxLayout())
        notebook = qtw.QTabWidget()
        base_widget.layout().addWidget(notebook)
        self.file_list = qtw.QListWidget()
        base_widget.layout().addWidget(self.file_list)
        self.setCentralWidget(base_widget)
```

接下来，我们将添加一个工具栏来容纳传输控件：

```py
        toolbar = self.addToolBar("Transport")
        record_act = toolbar.addAction('Rec')
        stop_act = toolbar.addAction('Stop')
        play_act = toolbar.addAction('Play')
        pause_act = toolbar.addAction('Pause')
```

我们希望我们的应用程序只显示日志视频，因此我们需要将我们的记录隔离到一个独特的目录，而不是使用记录的默认位置。使用`QtCore.QDir`，我们将以跨平台的方式创建和存储一个自定义位置，如下所示：

```py
        self.video_dir = qtc.QDir.home()
        if not self.video_dir.cd('captains_log'):
            qtc.QDir.home().mkdir('captains_log')
            self.video_dir.cd('captains_log')
```

这将在您的主目录下创建`captains_log`目录（如果不存在），并将`self.video_dir`对象设置为指向该目录。

我们现在需要一种方法来扫描这个目录以查找视频并填充列表小部件：

```py
    def refresh_video_list(self):
        self.file_list.clear()
        video_files = self.video_dir.entryList(
            ["*.ogg", "*.avi", "*.mov", "*.mp4", "*.mkv"],
            qtc.QDir.Files | qtc.QDir.Readable
        )
        for fn in sorted(video_files):
            self.file_list.addItem(fn)
```

`QDir.entryList()`返回我们的`video_dir`内容的列表。第一个参数是常见视频文件类型的过滤器列表，以便非视频文件不会在我们的日志列表中列出（可以随意添加您的操作系统喜欢的任何格式），第二个是一组标志，将限制返回的条目为可读文件。检索到这些文件后，它们将被排序并添加到列表小部件中。

回到`__init__()`，让我们调用这个函数来刷新列表：

```py
        self.refresh_video_list()
```

您可能希望在该目录中放入一个或两个视频文件，以确保它们被读取并添加到列表小部件中。

# 视频播放

我们的老朋友`QMediaPlayer`可以处理视频播放以及音频。但是，就像蓝光播放器需要连接到电视或监视器来显示它正在播放的内容一样，`QMediaPlayer`需要连接到一个实际显示视频的小部件。我们需要的小部件是`QVideoWidget`类，它位于`QtMultimediaWidgets`模块中。

要使用它，我们需要导入`QMultimediaWidgets`，如下所示：

```py
from PyQt5 import QtMultimediaWidgets as qtmmw
```

要将我们的`QMediaPlayer()`方法连接到`QVideoWidget()`方法，我们设置播放器的`videoOutput`属性，如下所示：

```py
        self.player = qtmm.QMediaPlayer()
        self.video_widget = qtmmw.QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
```

这比连接蓝光播放器要容易，对吧？

现在我们可以将视频小部件添加到我们的GUI，并将传输连接到我们的播放器：

```py
        notebook.addTab(self.video_widget, "Play")
        play_act.triggered.connect(self.player.play)
        pause_act.triggered.connect(self.player.pause)
        stop_act.triggered.connect(self.player.stop)
        play_act.triggered.connect(
            lambda: notebook.setCurrentWidget(self.video_widget))
```

最后，我们添加了一个连接，以便在单击播放按钮时切换回播放选项卡。

启用播放的最后一件事是将文件列表中的文件选择连接到加载和播放媒体播放器中的视频。

我们将在一个名为`on_file_selected()`的回调中执行此操作，如下所示：

```py
    def on_file_selected(self, item):
        fn = item.text()
        url = qtc.QUrl.fromLocalFile(self.video_dir.filePath(fn))
        content = qtmm.QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()
```

回调函数从`file_list`接收`QListWidgetItem`并提取`text`参数，这应该是文件的名称。我们将其传递给我们的`QDir`对象的`filePath()`方法，以获得文件的完整路径，并从中构建一个`QUrl`对象（请记住，`QMediaPlayer`使用URL而不是文件路径）。最后，我们将内容包装在`QMediaContent`对象中，将其加载到播放器中，并点击`play()`。

回到`__init__()`，让我们将此回调连接到我们的列表小部件：

```py
        self.file_list.itemDoubleClicked.connect(
            self.on_file_selected)
        self.file_list.itemDoubleClicked.connect(
            lambda: notebook.setCurrentWidget(self.video_widget))
```

在这里，我们连接了`itemDoubleClicked`，它将被点击的项目传递给槽，就像我们的回调所期望的那样。请注意，我们还将该操作连接到一个`lambda`函数，以切换到视频小部件。这样，如果用户在录制选项卡上双击文件，他们将能够在不手动切换回播放选项卡的情况下观看它。

此时，您的播放器已经可以播放视频。如果您还没有在`captains_log`目录中放入一些视频文件，请放入一些并查看它们是否可以播放。

# 视频录制

要录制视频，我们首先需要一个来源。在Qt中，此来源必须是`QMediaObject`的子类，其中可以包括音频来源、媒体播放器、收音机，或者在本程序中将使用的相机。

Qt 5.12目前不支持Windows上的视频录制，只支持macOS和Linux。有关Windows上多媒体支持当前状态的更多信息，请参阅[https://doc.qt.io/qt-5/qtmultimedia-windows.html](https://doc.qt.io/qt-5/qtmultimedia-windows.html)。

在Qt中，相机本身表示为`QCamera`对象。要创建一个可工作的`QCamera`对象，我们首先需要获取一个`QCameraInfo`对象。`QCameraInfo`对象包含有关连接到计算机的物理相机的信息。可以从`QtMultimedia.QCameraInfo.availableCameras()`方法获取这些对象的列表。

让我们将这些放在一起，形成一个方法，该方法将在您的系统上查找相机并返回一个`QCamera`对象：

```py
    def camera_check(self):
        cameras = qtmm.QCameraInfo.availableCameras()
        if not cameras:
            qtw.QMessageBox.critical(
                self,
                'No cameras',
                'No cameras were found, recording disabled.'
            )
        else:
            return qtmm.QCamera(cameras[0])
```

如果您的系统连接了一个或多个相机，`availableCameras()`应该返回一个`QCameraInfo`对象的列表。如果没有，那么我们将显示一个错误并返回空；如果有，那么我们将信息对象传递给`QCamera`构造函数，并返回表示相机的对象。

回到`__init__()`，我们将使用以下函数来获取相机对象：

```py
        self.camera = self.camera_check()
        if not self.camera:
            self.show()
            return
```

如果没有相机，那么此方法中剩余的代码将无法工作，因此我们将只显示窗口并返回。

在使用相机之前，我们需要告诉它我们希望它捕捉什么。相机可以捕捉静态图像或视频内容，这由相机的`captureMode`属性配置。

在这里，我们将其设置为视频，使用`QCamera.CaptureVideo`常量：

```py
        self.camera.setCaptureMode(qtmm.QCamera.CaptureVideo)
```

在我们开始录制之前，我们希望能够预览相机捕捉的内容（毕竟，船长需要确保他们的头发看起来很好以供后人纪念）。`QtMultimediaWidgets`有一个专门用于此目的的特殊小部件，称为`QCameraViewfinder`。

我们将添加一个并将我们的相机连接到它，如下所示：

```py
        self.cvf = qtmmw.QCameraViewfinder()
        self.camera.setViewfinder(self.cvf)
        notebook.addTab(self.cvf, 'Record')
```

相机现在已经创建并配置好了，所以我们需要通过调用`start()`方法来激活它：

```py
        self.camera.start()
```

如果您此时运行程序，您应该在录制选项卡上看到相机捕捉的实时显示。

这个谜题的最后一块是录制器对象。在视频的情况下，我们使用`QMediaRecorder`类来创建一个视频录制对象。这个类实际上是我们在声音板中使用的`QAudioRecorder`类的父类，并且工作方式基本相同。

让我们创建我们的录制器对象，如下所示：

```py
        self.recorder = qtmm.QMediaRecorder(self.camera)
```

请注意，我们将摄像头对象传递给构造函数。每当创建`QMediaRecorder`属性时，必须传递`QMediaObject`（其中`QCamera`是子类）。此属性不能以后设置，也不能在没有它的情况下调用构造函数。

就像我们的音频录制器一样，我们可以配置有关我们捕获的视频的各种设置。这是通过创建一个`QVideoEncoderSettings`类并将其传递给录制器的`videoSettings`属性来完成的：

```py
        settings = self.recorder.videoSettings()
        settings.setResolution(640, 480)
        settings.setFrameRate(24.0)
        settings.setQuality(qtmm.QMultimedia.VeryHighQuality)
        self.recorder.setVideoSettings(settings)
```

重要的是要理解，如果你设置了你的摄像头不支持的配置，那么录制很可能会失败，你可能会在控制台看到错误：

```py
CameraBin warning: "not negotiated"
CameraBin error: "Internal data stream error."
```

为了确保这不会发生，我们可以查询我们的录制对象，看看支持哪些设置，就像我们对音频设置所做的那样。以下脚本将打印每个检测到的摄像头在您的系统上支持的编解码器、帧速率、分辨率和容器到控制台：

```py
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

app = QCoreApplication([])

for camera_info in QCameraInfo.availableCameras():
    print('Camera: ', camera_info.deviceName())
    camera = QCamera(camera_info)
    r = QMediaRecorder(camera)
    print('\tAudio Codecs: ', r.supportedAudioCodecs())
    print('\tVideo Codecs: ', r.supportedVideoCodecs())
    print('\tAudio Sample Rates: ', r.supportedAudioSampleRates())
    print('\tFrame Rates: ', r.supportedFrameRates())
    print('\tResolutions: ', r.supportedResolutions())
    print('\tContainers: ', r.supportedContainers())
    print('\n\n')
```

请记住，在某些系统上，返回的结果可能为空。如果有疑问，最好要么进行实验，要么接受默认设置提供的任何内容。

现在我们的录制器已经准备好了，我们需要连接传输并启用它进行录制。让我们首先编写一个用于录制的回调方法：

```py
    def record(self):
        # create a filename
        datestamp = qtc.QDateTime.currentDateTime().toString()
        self.mediafile = qtc.QUrl.fromLocalFile(
            self.video_dir.filePath('log - ' + datestamp)
        )
        self.recorder.setOutputLocation(self.mediafile)
        # start recording
        self.recorder.record()
```

这个回调有两个作用——创建并设置要记录的文件名，并开始录制。我们再次使用我们的`QDir`对象，结合`QDateTime`类来生成包含按下记录时的日期和时间的文件名。请注意，我们不向文件名添加文件扩展名。这是因为`QMediaRecorder`将根据其配置为创建的文件类型自动执行此操作。

通过简单调用`QMediaRecorder`对象上的`record()`来启动录制。它将在后台记录视频，直到调用`stop()`插槽。

回到`__init__()`，让我们通过以下方式完成连接传输控件：

```py
        record_act.triggered.connect(self.record)
        record_act.triggered.connect(
            lambda: notebook.setCurrentWidget(self.cvf)
        )
        pause_act.triggered.connect(self.recorder.pause)
        stop_act.triggered.connect(self.recorder.stop)
        stop_act.triggered.connect(self.refresh_video_list)
```

我们将记录操作连接到我们的回调和一个lambda函数，该函数切换到录制选项卡。然后，我们直接将暂停和停止操作连接到录制器的`pause()`和`stop()`插槽。最后，当视频停止录制时，我们将希望刷新文件列表以显示新文件，因此我们将`stop_act`连接到`refresh_video_list()`回调。

这就是我们需要的一切；擦拭一下你的网络摄像头镜头，启动这个脚本，开始跟踪你的星际日期！

# 总结

在本章中，我们探索了`QtMultimedia`和`QMultimediaWidgets`模块的功能。您学会了如何使用`QSoundEffect`播放低延迟音效，以及如何使用`QMediaPlayer`和`QAudioRecorder`播放和记录各种媒体格式。最后，我们使用`QCamera`、`QMediaPlayer`和`QMediaRecorder`创建了一个视频录制和播放应用程序。

在下一章中，我们将通过探索Qt的网络功能来连接到更广泛的世界。我们将使用套接字进行低级网络和使用`QNetworkAccessManager`进行高级网络。

# 问题

尝试这些问题来测试你从本章学到的知识：

1.  使用`QSoundEffect`，你为呼叫中心编写了一个实用程序，允许他们回顾录制的电话呼叫。他们正在转移到一个将音频呼叫存储为MP3文件的新电话系统。你需要对你的实用程序进行任何更改吗？

1.  `cool_songs`是一个包含你最喜欢的歌曲路径字符串的Python列表。要以随机顺序播放这些歌曲，你需要做什么？

1.  你已经在你的系统上安装了`audio/mpeg`编解码器，但以下代码不起作用。找出问题所在：

```py
   recorder = qtmm.QAudioRecorder()
   recorder.setCodec('audio/mpeg')
   recorder.record()
```

1.  在几个不同的Windows、macOS和Linux系统上运行`audio_test.py`和`video_test.py`。输出有什么不同？有哪些项目在所有系统上都受支持？

1.  `QCamera`类的属性包括几个控制对象，允许您管理相机的不同方面。其中之一是`QCameraFocus`。在Qt文档中调查`QCameraFocus`，网址为[https://doc.qt.io/qt-5/qcamerafocus.html](https://doc.qt.io/qt-5/qcamerafocus.html)，并编写一个简单的脚本，显示取景器并让您调整数字变焦。

1.  您注意到录制到您的**船长日志**视频日志中的音频相当响亮。您想添加一个控件来调整它；您会如何做？

1.  在`captains_log.py`中实现一个停靠窗口小部件，允许您控制尽可能多的音频和视频录制方面。您可以包括焦点、变焦、曝光、白平衡、帧速率、分辨率、音频音量、音频质量等内容。

# 进一步阅读

您可以查阅以下参考资料以获取更多信息：

+   您可以在[https://doc.qt.io/qt-5/multimediaoverview.html](https://doc.qt.io/qt-5/multimediaoverview.html)上了解Qt多媒体系统及其功能。

+   PyQt的官方`QtMultimedia`和`QtMultimediaWidgets`示例可以在[https://github.com/pyqt/examples/tree/master/multimedia](https://github.com/pyqt/examples/tree/master/multimedia)和[https://github.com/pyqt/examples/tree/master/multimediawidgets](https://github.com/pyqt/examples/tree/master/multimediawidgets)找到。它们提供了更多使用PyQt进行媒体捕获和播放的示例代码。
