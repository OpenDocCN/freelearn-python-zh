# 将Raspbian 9升级到Raspbian 10

在[第15章](77583d1b-8a70-4118-8210-b0a5f09c9603.xhtml)中，*PyQt Raspberry Pi*，需要Raspbian 10，这样您就可以拥有足够新的Python和PyQt5版本。在出版时，Raspbian的当前版本是9版，预计2019年中至晚期将推出10版。但是，您可以升级到Raspbian 10的测试版本，这将正确地满足本书的目的。

要做到这一点，请按照以下步骤进行：

1.  首先，通过检查`/etc/issue`的内容来验证您是否正在使用Raspbian 9。它应该如下所示：

```py
 $ Rasbpian GNU/Linux 9 \n \l
```

1.  打开命令提示符，并使用`sudo`编辑`/etc/apt/sources.list`：

```py
 $ sudo -e /etc/apt/sources.list
```

1.  将每个`stretch`实例更改为`buster`。例如，第一行应该如下所示：

```py
deb http://raspbian.raspbrrypi.org/raspbian/
 buster main contrib non-free rpi
```

1.  运行`sudo apt update`命令，确保没有任何错误。

1.  现在运行`sudo apt upgrade`命令。此命令可能需要很长时间才能完成，因为它需要下载系统上每个软件包的更新副本并安装它。下载阶段结束后，还会有一些问题需要回答。一般来说，对这些问题采用默认答案。

1.  最后，重新启动您的Raspberry Pi。要清理旧的软件包，请运行以下命令：

```py
 $ sudo apt autoremove
```

就是这样；现在您应该正在运行Raspbian 10。如果遇到困难，请咨询[Raspbian社区](https://www.raspberrypi.org/forums/)。
