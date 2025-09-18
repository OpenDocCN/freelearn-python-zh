# 附录 C.未来开发

# Blender

Blender 是一个稳定且具有生产质量的软件包，但它自从开源以来一直在进行大量开发。几乎每个版本的 Blender 都带来了新功能——有些很小，有些确实非常复杂。Blender 的开发变化在 Blender 网站上得到了很好的记录：[`www.blender.org/development/release-logs/`](http://www.blender.org/development/release-logs/)。

Blender 当前的稳定版本是 2.49，这个版本可能会持续一段时间，因为 Blender 不再仅仅是 3D 爱好者使用的开源软件包，而是一个在专业工作室的生产流程中使用的可行生产工具。

然而，截至撰写本书时，Blender 新版本的开发正在全面进行。版本号 2.50 可能会让你认为这只是一个小改动，但实际上，它几乎是一个完全的重写。最明显的变化是完全不同的图形用户界面。这个界面几乎完全是用 Python 编写的，这为编写复杂的用户界面提供了无限的机会，以取代 2.49 中的有限可能性。

不仅用户界面发生了变化，内部结构也进行了彻底的改造，尽管在 Python API 中暴露的功能在本质上保持相似，但 Blender 的大多数模块都发生了许多变化。

一个主要的缺点是，新版本是在 Durian 项目（该项目将制作开源电影"Sintel"，见[durian.blender.org](http://durian.blender.org)）的同时开发的，因此 2.50 版本的主要开发目标是提供该项目所需的所有功能。这确实涵盖了大多数问题，但一些部分，特别是 Pynodes 和屏幕处理器，在首次生产版本中不会提供。

从积极的一面来看，将不再需要在 Blender 旁边安装完整的 Python 发行版，因为新版本将捆绑完整的 Python 发行版。

Blender 2.50 版本的开发路线图可以在[`www.blender.org/development/release-logs/blender-250/`](http://www.blender.org/development/release-logs/blender-250/)找到，但当然，那里提到的日程安排是非常初步的。完整的生产版本预计将在 2010 年底推出，并将具有 2.6 的版本号。

# Python

新版本的 Blender 将捆绑完整的 Python 3.x 发行版，从而消除了单独安装 Python 的需求。这个版本的 Python 已经非常稳定，未来不太可能实施重大更改。

3.x 版本与 2.6 版本不同，但最显著的变化都在表面之下，大多数不会影响到 Blender 脚本编写者。

尽管新版本的 Python 有一个显著的副作用：许多第三方包（即，未与发行版捆绑的 Python 包）尚未移植到 3.x 版本。在某些情况下，这可能会造成相当大的不便。从本书使用的包中，尚未移植到 3.x 版本的最显著的包是 PIL（Python 图像库）。这个包确实很受欢迎，因为它提供了 Blender 中不存在的复杂 2D 功能。

另一个尚未移植到 3.x 的包是 Psyco——即时编译器，但 Python 3 在许多情况下已经相当快了，因此像 Psyco 这样的包所能达到的速度提升可能不值得麻烦。

为了加快 Python 3.x 的接受速度，Python 的开发者宣布了对新特性的添加实行禁令，这样包的开发者就不必针对一个移动的目标进行瞄准。关于这个主题的更多信息可以在[`www.python.org/dev/peps/pep-3003/`](http://www.python.org/dev/peps/pep-3003/)找到。
