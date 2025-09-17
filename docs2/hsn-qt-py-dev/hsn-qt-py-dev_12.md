# 图形表示

在本章中，我们将探讨创建各种图形组件的下一部分，并在 GUI 应用程序中实现它们。这些类主要被用户用于创建图形元素，并在 Qt 库的标准图形类帮助下可视化它们。Qt 框架提供了表示这些图形的基本类。这些类是`QGraphicsScene`、`QGraphicsView`和`QGraphicsItem`。还有超过 50 个类实现了特殊功能，例如动画和图形元素。Qt 库实现了应用程序中创建的图形的模型-视图范式。模型-视图架构描述了如何将视图从模型中分离出来，并使用一个视图与不同的模型一起使用。具有图形元素的场景可以用不同的视图来表示。基于此，场景为项目提供了一个位置，这些项目可以是各种几何图形，而视图则可视化这个场景。虽然 Qt 是一个图形库，但我们将不会涵盖所有类，而只描述主要的一些。

在本章中，我们将涵盖以下主题：

+   `QObject`类

+   场景

+   视图

+   项目

# QObject

此类是所有与 Qt 一起使用的对象的基类。正如我们在引言中提到的，此类由`QWidget`类（与`QPaintDevice`相同）继承。我们将从这个类开始，因为`QGraphicsScene`继承了这个类。此类的声明语法如下：

```py
object = QtCore.QObject()
```

# QObject 函数

`QObject`类通过以下函数增强了功能。

# 设置

这些函数设置对象的参数/属性：

`object.setObjectName(str)`: 为此对象设置参数中指定的名称。

`object.setParent(QtCore.QObject)`: 为此对象设置参数中指定的父对象。

`object.setProperty(str, object)`: 将对象的名称（第一个参数）属性设置为值（第二个参数）。

# 是

这些函数返回与该对象状态相关的布尔值（`bool`）：

`object.isSignalConnected(QtCore.QMetaMethod)`: 如果在参数中指定的信号至少连接到一个接收器，则返回`True`。

`object.isWidgetType()`: 如果此对象具有小部件类型（或是一个小部件），则返回`True`。

`object.isWindowType()`: 如果此对象具有窗口类型（或是一个窗口），则返回`True`。

# 功能性

这些函数与返回此对象的当前值、功能变化等相关：

`object.blockSignals(bool)`: 如果参数为`True`，则此对象将阻止发出信号。

`object.children()`: 返回此对象的子对象列表。

`object.connect(QtCore.QObject, str, QtCore.QObject, str, QtCore.Qt.ConnectionType)`: 这从发送者（第一个参数）的信号（第二个参数）创建到接收者（第三个参数）的方法（第四个参数）的连接（第五个参数）。

`object.connectNotify(QtCore.QMetaMethod)`: 当对象与参数中指定的信号连接时，此函数将被调用。

`object.deleteLater()`: 这安排删除此对象。

`object.disconnect(QtCore.QObject, str)`: 这将断开对象的所有信号与接收者（第一个参数）的方法（第二个参数）的连接。

`object.disconnect(str, QtCore.QObject, str)`: 这将断开信号（第一个参数）与接收者（第二个参数）的方法（第三个参数）的连接。

`object.disconnect(QtCore.QObject, str, QtCore.QObject, str)`: 这将断开对象发送者（第一个参数）中的信号（第二个参数）与接收者（第三个参数）的方法（第四个参数）的连接。

`object.disconnect(QtCore.QObject, QtCore.QMetaMethod, QtCore.QObject, QtCore.QMetaMethod)`: 这将断开对象发送者（第一个参数）中的信号（第二个参数）与接收者（第三个参数）的方法（第四个参数）的连接。

`object.disconnectNotify(QtCore.QMetaMethod)`: 当对象从参数中指定的信号断开连接时，此函数将被调用。

`object.dumpObjectInfo()`: 这将输出该对象的信号连接。

`object.dumpObjectTree()`: 这将输出该对象的子对象树。

`object.dynamicPropertyNames()`: 这返回所有动态添加到该对象的`setProperty()`函数的所有属性的名称。

`object.findChild(type, str)`: 这将找到具有指定类型（第一个参数）和名称（第二个参数）的子对象。

`object.findChildren(type, str)`: 这将找到具有指定类型（第一个参数）和名称（第二个参数）的子对象。

`object.inherits(str)`: 如果该对象是参数中指定的类或其子类的实例，则返回`True`。

`object.killTimer(int)`: 这将杀死具有指定 ID（参数）的计时器。

`object.metaObject()`: 这返回该对象的元对象。

`object.moveToThread(QtCore.QThread)`: 这将此对象及其子对象的线程亲和力更改为参数中指定的线程。

`object.objectName()`: 这返回该对象的名字。

`object.parent()`: 这返回该对象的`QtCore.QObject`类型的父对象。

`object.property(str)`: 这返回对象名称属性的价值。

`object.receivers(SIGNAL)`: 这返回连接到参数中指定的信号的接收者数量。

`object.sender()`: 这返回发送信号的 `QtCore.QObject` 类型的发送者。

`object.senderSignalIndex()`: 这返回调用槽的信号的元方法索引，该槽是 `sender()` 函数返回的类的成员。

`object.signalsBlocked()`: 如果此对象的信号被阻止，则返回 `True`。

`object.startTimer(int, QtCore.Qt.TimerType)`: 这以间隔（第一个参数）和类型（第二个参数）启动计时器。

`object.thread()`: 这返回此对象正在运行的线程。

# events

这些函数与事件相关，例如事件处理器：

`object.childEvent(QtCore.QChildEvent)`: 此事件处理器接收此对象的子事件，事件通过参数传入。

`object.customEvent(QtCore.QEvent)`: 此事件处理器接收此对象的自定义事件，事件通过参数传入。

`object.event(QtCore.QEvent)`: 此函数接收发送给对象的信号，如果事件被识别并处理，则应返回 `True`。

`object.eventFilter(QtCore.QObject, QtCore.QEvent)`: 如果对象作为事件过滤器安装在此对象上，则过滤事件。

`object.installEventFilter(QtCore.QObject)`: 这在此对象上安装参数中指定的事件过滤器。

`object.removeEventFilter(QtCore.QObject)`: 这将从对象中移除参数中指定的事件过滤器。

`object.timerEvent(QtCore.QTimerEvent)`: 此事件处理器接收具有传入参数的事件的组件的计时器事件。

# signals

以下是在 `QObject` 类中可用的信号：

`object.destroyed(QtCore.QObject)`: 在对象被销毁之前发出此信号。

`object.objectNameChanged(str)`: 当对象名称已更改时发出此信号。新名称通过参数传入。

# QGraphicsScene

此类表示各种图形项的场景。这是图形视图架构的一部分，并提供图形视图场景。场景在应用程序中的作用如下：

+   管理项目的快速接口

+   未变换渲染

+   场中每个项目的信号

+   管理项目状态

场景的声明语法如下：

```py
scene = QtWidgets.QGraphicsScene()
```

# QGraphicsScene 函数

`QGraphicsScene` 类通过以下函数提高功能。

# add

这些函数添加场景元素：

`scene.addEllipse(QtCore.QRectF, QtGui.QPen, QtGui.QBrush)`: 使用矩形（第一个参数）的几何形状添加椭圆，指定笔（第二个参数）和刷（第三个参数）。

`scene.addEllipse(x, y, w, h, QtGui.QPen, QtGui.QBrush)`: 这在 *x* 轴上的 `x` 和 *y* 轴上的 `y` 处添加一个椭圆，宽度为 `w`，高度为 `h`。笔（第五个参数）和刷（第六个参数）被指定。

`scene.addItem(QtWidgets.QGraphicsItem)`: 这将在场景中添加一个由参数指定的图形项目。

`scene.addLine(QtCore.QLineF, QtGui.QPen)`: 这将在场景中添加一条线，其几何形状由第一个参数指定，而笔由第二个参数指定。

`scene.addLine(x1, y1, x2, y2, QtGui.QPen)`: 这将在点`(x1, y1)`开始并结束于点`(x2, y2)`的位置添加一条线。笔由第五个参数指定。

`scene.addPath(QtGui.QPainterPath, QtGui.QPen, QtGui.QBrush)`: 这将在场景中添加一个由第一个参数指定的路径，并带有笔（第二个参数）和画刷（第三个参数）。

`scene.addPixmap(QtGui.QPixmap)`: 这将在场景中添加一个指定的参数图样。

`scene.addPolygon(QtGui.QPolygonF, QtGui.QPen, QtGui.QBrush)`: 这将在场景中添加一个由第一个参数指定的多边形，并带有笔（第二个参数）和画刷（第三个参数）。

`scene.addRect(QtCore.QRectF, QtGui.QPen, QtGui.QBrush)`: 这将在场景中添加一个矩形，其几何形状由第一个参数指定，同时指定了笔（第二个参数）和画刷（第三个参数）。

`scene.addRect(x, y, w, h, QtGui.QPen, QtGui.QBrush)`: 这将在`(x, y)`处开始，宽度为`w`，高度为`h`的位置添加一个矩形。笔（第五个参数）和画刷（第六个参数）也进行了指定。

`scene.addSimpleText(str, QtGui.QFont)`: 这将在场景中添加一些简单文本（第一个参数），为`QtWidgets.QGraphicsSimpleTextItem`类型，并带有字体（第二个参数）。

`scene.addText(str, QtGui.QFont)`: 这将在场景中添加一些格式化文本（第一个参数），为`QtWidgets.QGraphicsTextItem`类型，并带有字体（第二个参数）。

`scene.addWidget(QtWidgets.QWidget, QtCore.Qt.WindowFlags)`: 这将在场景中添加一个`QtWidgets.QGraphicsProxyWidget`类型的新小部件。

# set

这些函数将参数/属性设置到场景中：

`scene.setActivePanel(QtWidgets.QGraphicsItem)`: 这将参数指定的项目设置为活动项目。

`scene.setActiveWindow(QtWidgets.QGraphicsWidget)`: 这将为参数指定的窗口设置活动状态。

`scene.setBackgroundBrush(QtGui.QBrush)`: 这将为场景设置参数指定的背景画刷。

`scene.setBspTreeDepth(int)`: 这将为场景设置参数指定的二叉空间划分（BSP）索引树的深度。

`scene.setFocus(QtCore.Qt.FocusReason)`: 这将设置场景的焦点，并通过参数传递焦点原因。

`scene.setFocusItem(QtWidgets.QGraphicsItem, QtCore.Qt.FocusReason)`: 这将为场景设置焦点项目（第一个参数），并带有焦点原因（第二个参数）。

`scene.setFocusOnTouch(bool)`: 如果参数为`True`，则在接收到触摸开始事件时，项目将获得焦点。

`scene.setFont(QtGui.QFont)`: 这将为场景设置指定的参数，作为场景的默认字体。

`scene.setForegroundBrush(QtGui.QBrush)`: 这将为场景设置指定的前景画笔。

`scene.setItemIndexMethod(QtWidgets.QGraphicsScene.ItemIndexMethod)`: 这将设置项目索引方法。可用方法如下：

+   `QtWidgets.QGraphicsScene.BspTreeIndex`—`0`: 应用了 BSP（静态场景）。

+   `QtWidgets.QGraphicsScene.NoIndex`—`1`: 未应用索引（动态场景）。

`scene.setMinimumRenderSize(float)`: 这将为要绘制的项目的最小视图变换大小设置。这将加快场景的渲染速度，在缩放视图下渲染许多对象。

`scene.setPalette(QtGui.QPalette)`: 这将为场景设置指定的调色板。

`scene.setSceneRect(QtCore.QRectF)`: 这将为场景设置指定的参数作为边界矩形。

`scene.setSceneRect(x, y, w, h)`: 这将为场景设置一个以`x`/`y`为起点，宽度为`w`，高度为`h`的边界矩形。

`scene.setSelectionArea(QtGui.QPainterPath, QtGui.QTransform)`: 这将设置选择区域为一个路径（第一个参数）以及应用了变换（第二个参数）。

`scene.setSelectionArea(QtGui.QPainterPath, QtCore.Qt.ItemSelectionMode, QtGui.QTransform)`: 这将设置选择区域为一个路径（第一个参数），带有模式（第二个参数）和应用了变换（第三个参数）。

`scene.setSelectionArea(QtGui.QPainterPath, QtCore.Qt.ItemSelectionOperation, QtCore.Qt.ItemSelectionMode, QtGui.QTransform)`: 这将设置选择区域为一个路径（第一个参数），带有模式（第三个参数）、应用了变换（第四个参数）以及当前选中项的选择操作（第二个参数）。

`scene.setStickyFocus(bool)`: 如果参数为`True`，则焦点将保持不变，点击场景的背景或不接受焦点的项。否则，焦点将被清除。

`scene.setStyle(QtWidgets.QStyle)`: 这将为场景设置指定的参数作为样式。

# has/is

这些函数返回与场景状态相关的布尔值（`bool`）：

`scene.hasFocus()`: 如果此场景有焦点，则返回`True`。

`scene.isActive()`: 如果此场景是活动的，则返回`True`。

# functional

这些函数与场景当前值的返回、功能变化等相关：

`scene.activePanel()`: 这将返回此场景的活动面板。

`scene.activeWindow()`: 这将返回此场景的活动窗口。

`scene.advance()`: 这将场景向前推进一步（适用于场景上的所有项目）。

`scene.backgroundBrush()`: 这将返回此场景背景的`QtGui.QBrush`类型的画笔。

`scene.bspTreeDepth()`: 这将返回此场景中 BSP 索引树的深度。

`scene.clear()`: 这清除此场景中的所有项目。

`scene.clearFocus()`: 这清除此场景的焦点。

`scene.clearSelection()`: 这清除此场景的当前选择。

`scene.collidingItems(QtWidgets.QGraphicsItem, QtCore.Qt.ItemSelectionMode)`: 这返回与项目（第一个参数）碰撞的项目列表，碰撞检测由模式（第二个参数）指定。

`scene.createItemGroup([QtWidgets.QGraphicsItem])`: 这将参数中指定的所有项目作为一个项目列表分组到新的项目组中。

`scene.destroyItemGroup(QtWidgets.QGraphicsItemGroup)`: 这从场景中删除参数中指定的项目组。

`scene.drawBackground(QtGui.QPainter, QtCore.QRectF)`: 这使用画家（第一个参数）和矩形（第二个参数）为此场景绘制背景。

`scene.drawForeground(QtGui.QPainter, QtCore.QRectF)`: 这使用画家（第一个参数）和矩形（第二个参数）为此场景绘制前景。

`scene.focusItem()`: 这返回此场景的当前焦点项目。

`scene.focusOnTouch()`: 如果项目在接收到触摸开始事件时获得焦点，则返回 `True`。

`scene.font()`: 这返回此场景的当前字体。

`scene.foregroundBrush()`: 这返回此场景前景的 `QtGui.QBrush` 类型的画笔。

`scene.invalidate(QtCore.QRectF, QtWidgets.QGraphicsScene.SceneLayers)`: 这安排在场景中矩形（第一个参数）的层（第二个参数）的重绘。

`scene.invalidate(x, y, w, h, QtWidgets.QGraphicsScene.SceneLayers)`: 这安排在以 `x`/`y` 开始的矩形中重绘层（第五个参数），宽度为 `w`，高度为 `h`。

`scene.itemAt(QtCore.QPointF, QtGui.QTransform)`: 这返回在特定点（第一个参数）的最顶层项目，带有应用变换（第二个参数）。

`scene.itemAt(x, y, QtGui.QTransform)`: 这返回由 `x` 和 `y` 指定的位置（带有应用变换的第三个参数）的最顶层项目。

`scene.itemIndexMethod()`: 这返回项目的索引方法。

`scene.items(QtCore.Qt.SortOrder)`: 这返回在参数中指定的堆叠顺序中所有场景项目的有序列表。

`scene.items(QtCore.QRectF, QtCore.Qt.ItemSelectionMode, QtCore.Qt.SortOrder, QtGui.QTransform)`: 这返回根据模式（第二个参数）在矩形（第一个参数）内或与之相交的所有可见项目，排序（第三个参数），并应用变换（第四个参数）。

`scene.items(QtCore.QPointF, QtCore.Qt.ItemSelectionMode, QtCore.Qt.SortOrder, QtGui.QTransform)`: 此函数根据模式（第二个参数）返回所有可见项目，该模式位于或与点（第一个参数）内部或相交，排序（第三个参数），并应用了变换（第四个参数）。

`scene.items(QtGui.QPainterPath, QtCore.Qt.ItemSelectionMode, QtCore.Qt.SortOrder, QtGui.QTransform)`: 此函数根据模式（第二个参数）返回所有可见项目，该模式位于或与指定的路径（第一个参数）内部或相交，排序（第三个参数），并应用了变换（第四个参数）。

`scene.items(QtGui.QPolygonF, QtCore.Qt.ItemSelectionMode, QtCore.Qt.SortOrder, QtGui.QTransform)`: 此函数根据模式（第二个参数）返回所有可见项目，该模式位于或与多边形（第一个参数）内部或相交，排序（第三个参数），并应用了变换（第四个参数）。

`scene.items(x, y, w, h, QtCore.Qt.ItemSelectionMode, QtCore.Qt.SortOrder, QtGui.QTransform)`: 此函数根据模式（第五个参数）返回所有可见项目，该模式位于或与以`x`/`y`为起始点、宽度为`w`、高度为`h`的矩形内部或相交，排序（第六个参数），并应用了变换（第七个参数）。

`scene.itemsBoundingRect()`: 此函数返回场景中所有项目的边界矩形。

`scene.minimumRenderSize()`: 此函数返回要绘制的项目的最小视图变换大小。

`scene.mouseGrabberItem()`: 此函数返回当前鼠标抓取项，该项接收发送到场景的所有鼠标事件。

`scene.palette()`: 此函数返回与场景一起使用的`QtGui.QPalette`类型的默认调色板。

`scene.removeItem(QtWidgets.QGraphicsItem)`: 此函数移除由参数指定的项目及其所有子项。

`scene.render(QtGui.QPainter, QtCore.QRectF, QtCore.QRectF, QtCore.Qt.AspectRatioMode)`: 此函数将场景中的源矩形（第三个参数）渲染到由画家（第一个参数）和模式（第四个参数）指定的矩形中。

`scene.sceneRect()`: 此函数返回场景的边界矩形。

`scene.selectedItems()`: 此函数返回当前选定的项目列表。

`scene.selectionArea()`: 此函数返回此场景的选择区域。

`scene.stickyFocus()`: 当用户点击场景的背景或项目时，此函数返回`True`，表示焦点将保持不变。

`scene.style()`: 此函数返回用于此场景的`QtWidgets.QStyle`类型的样式。

`scene.update(QtCore.QRectF)`: 此函数安排在场景中重绘由参数指定的矩形。

`scene.update(x, y, w, h)`: 此函数安排在场景上重绘以`x`/`y`为起始点、宽度为`w`、高度为`h`的区域。

`scene.views()`: 这返回显示在此场景上的所有视图，作为一个视图列表。

# 事件

这些函数与事件相关，例如事件处理程序：

`scene.contextMenuEvent(QtWidgets.QGraphicsSceneContextMenuEvent)`: 此事件处理程序接收上下文菜单事件。

`scene.dragEnterEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理程序接收场景参数中指定的拖动进入事件。

`scene.dragLeaveEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理程序接收场景参数中指定的拖动离开事件。

`scene.dragMoveEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理程序接收场景参数中指定的拖动移动事件。

`scene.dropEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理程序接收场景参数中指定的放下事件。

`scene.event(QtCore.QEvent)`: 这接收场景的事件，并且如果事件被识别和处理，应返回 `True`。

`scene.focusOutEvent(QtGui.QFocusEvent)`: 此事件处理程序接收场景的键盘焦点事件，当失去焦点时，这些事件通过事件参数传递。

`scene.focusInEvent(QtGui.QFocusEvent)`: 此事件处理程序接收场景的键盘焦点事件，当获得焦点时，这些事件通过事件参数传递。

`scene.helpEvent(QtWidgets.QGraphicsSceneHelpEvent)`: 此事件处理程序接收场景参数中指定的帮助事件。

`scene.inputMethodEvent(QtGui.QInputMethodEvent)`: 此事件处理程序接收输入法事件。

`scene.keyPressEvent(QtGui.QKeyEvent)`: 此事件处理程序接收场景的按键按下事件，事件通过参数传递。

`scene.keyReleaseEvent(QtGui.QKeyEvent)`: 此事件处理程序接收场景的按键释放事件，事件通过参数传递。

`scene.mouseDoubleClickEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理程序接收场景的鼠标双击事件，事件通过参数传递。

`scene.mouseMoveEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理程序接收场景的鼠标移动事件，事件通过参数传递。

`scene.mousePressEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理程序接收场景的鼠标按下事件，事件通过参数传递。

`scene.mouseReleaseEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理程序接收场景的鼠标释放事件，事件通过参数传递。

`scene.sendEvent(QtWidgets.QGraphicsItem, QtCore.QEvent)`: 这通过事件过滤器将事件（第二个参数）发送到项目（第一个参数）。

`scene.wheelEvent(QtWidgets.QGraphicsSceneWheelEvent)`: 此事件处理程序接收场景的鼠标滚轮事件，事件通过参数传递。

# 信号

`QGraphicsScene` 类的可用的信号如下：

`scene.changed([QtCore.QRectF])`: 如果场景内容发生变化，会发出此信号，参数包含一个矩形列表。

`scene.focusItemChanged(QtWidgets.QGraphicsItem, QtWidgets.QGraphicsItem, QtCore.Qt.FocusReason)`: 当场景中的焦点发生变化时，会发出此信号。参数包括一个之前有焦点的项目（第二个参数）、获得输入焦点的项目（第一个参数）和焦点原因（第三个参数）。

`scene.sceneRectChanged(QtCore.QRectF)`: 当场景的矩形发生变化时，会发出此信号。新矩形通过参数传递。

`scene.selectionChanged()`: 当场景的选择发生变化时，会发出此信号。

# QGraphicsView

此类表示一个具有显示场景小部件的视图。这是图形视图架构的一部分，并为应用程序中的场景提供图形表示。此类的声明语法如下：

```py
graphics_view = QtWidgets.QGraphicsView()
```

# QGraphicsView 函数

`QGraphicsView` 通过以下函数增强功能。

# set

这些函数设置图形视图的参数/属性：

`graphics_view.setAlignment(QtCore.Qt.Alignment)`: 这将设置图形视图中的场景对齐方式。

`graphics_view.setBackgroundBrush(QtGui.QBrush)`: 这将为图形视图中的场景设置指定的背景画刷。

`graphics_view.setCacheMode(QtWidgets.QGraphicsView.CacheMode)`: 这将设置描述视图哪些部分被缓存的缓存模式。可用的模式如下：

+   `QtWidgets.QGraphicsView.CacheNone`: 所有绘图都直接在视口中完成。

+   `QtWidgets.QGraphicsView.CacheBackground`: 背景被缓存。

`graphics_view.setDragMode(QtWidgets.QGraphicsView.DragMode)`: 这将设置拖动模式。可用的模式如下：

+   `QtWidgets.QGraphicsView.NoDrag`—`0`: 鼠标事件将被忽略。

+   `QtWidgets.QGraphicsView.ScrollHandDrag`—`1`: 光标将变为指向手，拖动鼠标将滚动滚动条。

+   `QtWidgets.QGraphicsView.RubberBandDrag`—`2`: 将使用橡皮筋。

`graphics_view.setForegroundBrush(QtGui.QBrush)`: 这将为图形视图中的场景设置指定的前景画刷。

`graphics_view.setInteractive(bool)`: 这将设置在视图中允许的场景交互。

`graphics_view.setOptimizationFlag(QtWidgets.QGraphicsView.OptimizationFlag, bool)`: 如果第二个参数是 `True`，则启用第一个参数指定的标志。

`graphics_view.setOptimizationFlags(QtWidgets.QGraphicsView.OptimizationFlag | QtWidgets.QGraphicsView.OptimizationFlag)`: 这将设置用于图形视图性能的优化标志，标志由参数指定。

`graphics_view.setRenderHint(QtGui.QPainter.RenderHint, bool)`: 如果第二个参数是 `True`，则启用第一个参数指定的渲染提示。

`graphics_view.setRenderHints(QtGui.QPainter.RenderHint | QtGui.QPainter.RenderHint)`: 这将设置参数中指定的渲染提示，这些提示将用作此视图的默认渲染提示。

`graphics_view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor)`: 这将设置参数中指定的锚点，该锚点将描述视图在调整大小时如何定位场景。可用的锚点如下：

+   `QtWidgets.QGraphicsView.NoAnchor`—`0`: 无锚点，位置不变。

+   `QtWidgets.QGraphicsView.AnchorViewCenter`—`1`: 将锚点设置为视图的中心。

+   `QtWidgets.QGraphicsView.AnchorUnderMouse`—`2`: 将锚点设置为鼠标下方的点。

`graphics_view.setRubberBandSelectionMode(QtCore.Qt.ItemSelectionMode)`: 这将设置描述如何使用橡皮筋拖动选择项的模式。

`graphics_view.setScene(QtWidgets.QGraphicsScene)`: 这将参数中指定的当前图形场景设置为图形视图。

`graphics_view.setSceneRect(QtCore.QRectF)`: 这将设置参数中指定的场景区域，该区域将使用此图形视图进行可视化。

`graphics_view.setSceneRect(x, y, w, h)`: 这将设置从`x`/`y`开始的场景区域，宽度为`w`，高度为`h`，该区域将使用此图形视图进行可视化。

`graphics_view.setTransform(QtGui.QTransform, bool)`: 这将设置第一个参数中指定的变换矩阵。如果第二个参数为`True`，则矩阵将与当前矩阵合并。

`graphics_view.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor)`: 这将设置参数中指定的变换锚点。这将描述图形视图在变换期间如何定位场景。

`graphics_view.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode)`: 这将设置用于更新视口内容的模式。可用的参数如下：

+   `QtWidgets.QGraphicsView.FullViewportUpdate`—`0`: 将更新整个视口，以反映对场景可见部分所做的更改。

+   `QtWidgets.QGraphicsView.MinimalViewportUpdate`—`1`: 将更新最少的视口区域。

+   `QtWidgets.QGraphicsView.SmartViewportUpdate`—`2`: 将更新最优的视口区域。

+   `QtWidgets.QGraphicsView.NoViewportUpdate`—`3`: 视口不会随着场景的变化而更新。

+   `QtWidgets.QGraphicsView.BoundingRectViewportUpdate`—`4`: 将更新视口中所有更改的边界矩形。

# 是

这些函数返回与图形视图状态相关的布尔值（`bool`）：

`graphics_view.isInteractive()`: 如果此视图允许在场景中进行交互，则返回`True`。

`graphics_view.isTransformed()`: 如果此视图已变换，则返回`True`。

# 功能性

这些函数返回图形视图的当前值、功能更改等：

`graphics_view.alignment()`: 这返回了图形视图小部件中场景的对齐方式。

`graphics_view.backgroundBrush()`: 这返回了在图形视图小部件中用于此场景的`QtGui.QBrush`类型的背景画刷。

`graphics_view.cacheMode()`: 这返回了`QtWidgets.QGraphicsView.CacheMode`类型的缓存模式。此视图的部分被缓存。

`graphics_view.centerOn(QtCore.QPointF)`: 这将视口的内容滚动到场景坐标点，由参数指定，这将位于视图中心。

`graphics_view.centerOn(QtWidgets.QGraphicsItem)`: 这将视口的内容滚动到场景项，由参数指定，这将位于视图中心。

`graphics_view.centerOn(x, y)`: 这将视口的内容滚动到场景坐标`x`和`y`，这将位于视图中心。

`graphics_view.dragMode()`: 这返回了在鼠标拖动此场景时的`QtWidgets.QGraphicsView.DragMode`类型的模式。

`graphics_view.drawBackground(QtGui.QPainter, QtCore.QRectF)`: 这使用画家（第一个参数）在坐标（第二个参数）中绘制此场景的背景。

`graphics_view.drawForeground(QtGui.QPainter, QtCore.QRectF)`: 这使用画家（第一个参数）在坐标（第二个参数）中绘制此场景的前景。

`graphics_view.ensureVisible(QtCore.QRectF, int, int)`: 这将视口的内容滚动到将可见的场景矩形（第一个参数），带有*x*边距（第二个参数）和*y*边距（第三个参数）。

`graphics_view.ensureVisible(QtWidgets.QGraphicsItem, int, int)`: 这将视口的内容滚动到将可见的场景项（第一个参数），带有*x*边距（第二个参数）和*y*边距（第三个参数）。

`graphics_view.ensureVisible(x, y, w, h, int, int)`: 这将视口的内容滚动到场景的起始位置`x`/`y`，宽度为`w`，高度为`h`，将在*x*边距（第五个参数）和*y*边距（第六个参数）内可见。

`graphics_view.fitInView(QtCore.QRectF, QtCore.Qt.AspectRatioMode)`: 这缩放视图矩阵并滚动滚动条，使场景矩形（第一个参数）根据纵横比（第二个参数）适合视口内。

`graphics_view.fitInView(QtWidgets.QGraphicsItem, QtCore.Qt.AspectRatioMode)`: 这缩放视图矩阵并滚动滚动条，使场景的项根据纵横比（第二个参数）紧密适合视图内。

`graphics_view.fitInView(x, y, w, h, QtCore.Qt.AspectRatioMode)`: 这缩放视图矩阵并滚动滚动条到场景的起始位置`x`/`y`，宽度为`w`，高度为`h`，使项目根据纵横比（第五个参数）适合视口内。

`graphics_view.foregroundBrush()`: 这返回了用于图形视图小部件中此场景的 `QtGui.QBrush` 类型的前景画笔。

`graphics_view.invalidateScene(QtCore.QRectF, QtWidgets.QGraphicsScene.SceneLayers)`: 这使矩形（第一个参数）内的层（第二个参数）无效并安排重绘。

`graphics_view.itemAt(QtCore.QPointF)`: 这返回了参数中指定的点的项目。

`graphics_view.itemAt(x, y)`: 这返回了由 `x` 和 `y` 坐标指定的位置的项。

`graphics_view.items()`: 这返回了场景中所有项目的列表，按降序堆叠顺序排列。

`graphics_view.items(QtCore.QRect, QtCore.Qt.ItemSelectionMode)`: 这返回了所有可见项目，取决于模式（第二个参数），这些项目位于或与矩形（第一个参数）内部或相交。

`graphics_view.items(QtCore.QPoint)`: 这返回了参数中指定的点的所有项。

`graphics_view.items(QtGui.QPainterPath, QtCore.Qt.ItemSelectionMode)`: 这返回了所有项目，取决于模式（第二个参数），这些项目位于或与指定的路径（第一个参数）内部或相交。

`graphics_view.items(QtGui.QPolygon, QtCore.Qt.ItemSelectionMode)`: 这返回了所有项目，取决于模式（第二个参数），这些项目位于或与多边形（第一个参数）内部或相交。

`graphics_view.items(x, y)`: 这返回了坐标 *x* 和 *y* 处的所有项。

`graphics_view.items(x, y, w, h, QtCore.Qt.ItemSelectionMode)`: 这返回了所有项目，取决于模式（第五个参数），这些项目位于或与以 `x`/`y` 为起点、宽度为 `w` 和高度为 `h` 的区域内部或相交。

`graphics_view.optimizationFlags()`: 这返回了用于调整视图性能的标志。

`graphics_view.render(QtGui.QPainter, QtCore.QRectF, QtCore.QRect, QtCore.Qt.AspectRatioMode)`: 这使用画家（第一个参数）和模式（第四个参数）将场景中的源矩形（第三个参数）渲染到矩形（第二个参数）中。

`graphics_view.renderHints()`: 这返回了此视图的默认渲染提示。

`graphics_view.resetCachedContent()`: 这重置了缓存内容并清除了视图缓存。

`graphics_view.resetTransform()`: 这将视图变换重置为恒等矩阵。

`graphics_view.resizeAnchor()`: 这返回了在视图大小调整时将与场景位置一起使用的锚点。

`graphics_view.rotate(float)`: 这按顺时针方向将当前视图变换旋转到参数中指定的角度度数。

`graphics_view.rubberBandRect()`: 这返回了使用项目选择时 `QtCore.QRect` 类型（在视口坐标中）的橡皮筋区域。

`graphics_view.rubberBandSelectionMode()`: 这返回了用于使用橡皮筋选择矩形选择项目的模式。

`graphics_view.scale(float, float)`: 此函数通过 `x`（第一个参数）和 `y`（第二个参数）缩放视图转换。

`graphics_view.scene()`: 这将返回由此图形视图可视化的 `QtWidgets.QGraphicsScene` 类型的场景。

`graphics_view.sceneRect()`: 这将返回由此图形视图可视化的 `QtCore.QRectF` 类型的场景区域。

`graphics_view.setupViewport(QtWidgets.QWidget)`: 在使用之前，此函数初始化一个新的视口小部件。

`graphics_view.shear(float, float)`: 此函数将当前视图转换水平（第一个参数）和垂直（第二个参数）剪切。

`graphics_view.transform()`: 这将返回此图形视图的当前转换矩阵，其类型为 `QtGui.QTransform`。

`graphics_view.transformationAnchor()`: 这将返回与该图形视图转换一起使用的 `QtWidgets.QGraphicsView.ViewportAnchor` 类型的锚点。

`graphics_view.translate(float, float)`: 此函数通过 `x`（第一个参数）和 `y`（第二个参数）平移视图转换。

`graphics_view.updateScene([QtCore.QRectF])`: 这将安排更新场景矩形，参数为一个矩形列表。

`graphics_view.updateSceneRect(QtCore.QRectF)`: 此函数通知图形视图场景的矩形已更改，参数为新的场景矩形。

`graphics_view.viewportTransform()`: 这将返回一个 `QtGui.QTransform` 类型的矩阵，该矩阵将场景坐标映射到视口坐标。

`graphics_view.viewportUpdateMode()`: 这将返回视口的更新模式。

# 映射

这些函数与映射相关：

`graphics_view.mapFromScene(QtCore.QRectF)`: 这将返回一个指定参数的矩形，并将其转换为视口坐标多边形。

`graphics_view.mapFromScene(QtCore.QPointF)`: 这将返回一个指定参数的点，并将其转换为视口坐标点。

`graphics_view.mapFromScene(QtGui.QPainterPath)`: 这将返回一个指定参数的场景坐标画家路径，并将其转换为视口坐标画家路径。

`graphics_view.mapFromScene(QtGui.QPolygonF)`: 这将返回一个指定参数的场景坐标多边形，并将其转换为视口坐标多边形。

`graphics_view.mapToScene(QtCore.QRectF)`: 这将返回一个指定参数的视口坐标多边形，并将其转换为场景坐标多边形。

`graphics_view.mapToScene(QtCore.QPointF)`: 这将返回一个指定参数的视口坐标点，并将其映射到场景坐标。

`graphics_view.mapToScene(QtGui.QPainterPath)`: 这将返回一个指定参数的视口画家路径，并将其转换为场景坐标画家路径。

`graphics_view.mapToScene(QtGui.QPolygonF)`: 这将返回一个指定参数的视口坐标多边形，并将其转换为场景坐标多边形。

# 事件

这些函数与事件相关，例如事件处理器：

`graphics_view.contextMenuEvent(QtGui.QContextMenuEvent)`: 此事件处理程序接收上下文菜单事件。

`graphics_view.dragEnterEvent(QtGui.QDragEnterEvent)`: 当鼠标进入此场景且正在拖动时，此事件处理程序会使用事件参数被调用。

`graphics_view.dragLeaveEvent(QtGui.QDragLeaveEvent)`: 当鼠标离开此场景且正在拖动时，此事件处理程序会使用事件参数被调用。

`graphics_view.dragMoveEvent(QtGui.QDragMoveEvent)`: 当发生某些条件时，例如光标进入或移动到该区域内，键盘上的修饰键在场景具有焦点时被按下，或者正在拖动时，此事件处理程序会使用事件参数被调用。

`graphics_view.dropEvent(QtGui.QDropEvent)`: 当拖动被放置到场景上时，此事件处理程序会使用事件参数被调用。

`graphics_view.event(QtCore.QEvent)`: 此接收场景的事件，如果事件被识别并处理，则应返回`True`。

`graphics_view.focusOutEvent(QtGui.QFocusEvent)`: 此事件处理程序接收场景的键盘焦点事件，当失去焦点时，这些事件会通过事件参数传递。

`graphics_view.focusInEvent(QtGui.QFocusEvent)`: 此事件处理程序接收场景的键盘焦点事件，当获得焦点时，这些事件会通过事件参数传递。

`graphics_view.inputMethodEvent(QtGui.QInputMethodEvent)`: 此事件处理程序接收场景的输入法事件。

`graphics_view.keyPressEvent(QtGui.QKeyEvent)`: 此事件处理程序接收传入参数的事件的按键事件。

`graphics_view.keyReleaseEvent(QtGui.QKeyEvent)`: 此事件处理程序接收传入参数的事件的按键释放事件。

`graphics_view.mouseDoubleClickEvent(QtGui.QMouseEvent)`: 此事件处理程序接收传入参数的事件的鼠标双击事件。

`graphics_view.mouseMoveEvent(QtGui.QMouseEvent)`: 此事件处理程序接收传入参数的事件的鼠标移动事件。

`graphics_view.mousePressEvent(QtGui.QMouseEvent)`: 此事件处理程序接收传入参数的事件的鼠标按下事件。

`graphics_view.mouseReleaseEvent(QtGui.QMouseEvent)`: 此事件处理程序接收传入参数的事件的鼠标释放事件。

`graphics_view.paintEvent(QtGui.QPaintEvent)`: 此事件处理程序接收传入参数的事件的绘制事件。

`graphics_view.resizeEvent(QtGui.QResizeEvent)`: 此事件处理程序接收传入参数的事件的尺寸调整事件。

`graphics_view.showEvent(QtGui.QShowEvent)`: 此事件处理程序接收传入参数的事件的显示事件。

`graphics_view.viewportEvent(QtCore.QEvent)`: 这是带有传入参数的事件的滚动区域的主要事件处理程序。

`graphics_view.wheelEvent(QtGui.QWheelEvent)`: 此事件处理器接收场景的鼠标滚轮事件，事件通过参数传入。

# 信号

`QGraphicsView` 类的可用的信号如下：

`graphics_view.rubberBandChanged(QtCore.QRect, QtCore.QPointF, QtCore.QPointF)`: 当橡皮筋矩形改变时，发出此信号。视口矩形由第一个参数指定，其中包含拖动开始位置（第二个参数）和拖动结束位置（第三个参数）。

# QGraphicsItem

这是所有可以通过 `QGraphicsScene` 类在场景中实现的图形项的基类。使用此基类，Qt 框架提供了一套标准图形项，如 `QGraphicsEllipseItem`、`QGraphicsLineItem`、`QGraphicsPathItem`、`QGraphicsPixmapItem`、`QGraphicsPolygonItem`、`QGraphicsRectItem`、`QGraphicsSimpleTextItem` 和 `QGraphicsTextItem`。这些通常用于在应用程序中创建内部图形组件。

# QGraphicsItem 函数

`QGraphicsItem` 类通过以下函数增强功能。

# set

这些函数将参数/属性设置到图形项中：

`setAcceptDrops(bool)`: 如果参数为 `True`，则为此项接受拖放事件。

`setAcceptedMouseButtons(QtCore.Qt.MouseButtons)`: 这为此项的鼠标事件设置参数中指定的鼠标按钮。

`setAcceptHoverEvents(bool)`: 如果参数为 `True`，则为此项接受悬停事件。

`setAcceptTouchEvents(bool)`: 如果参数为 `True`，则为此项接受触摸事件。

`setActive(bool)`: 如果参数为 `True`，则激活此项的面板。

`setBoundingRegionGranularity(float)`: 这为此项的边界区域粒度（`0.0` - `1.0`）设置参数，即设备分辨率与边界区域的比率。

`setCacheMode(QtWidgets.QGraphicsItem.CacheMode, QtCore.QSize)`: 这为此项设置缓存模式（第一个参数），并可选地设置大小（第二个参数）。可用的缓存模式如下：

+   `QtWidgets.QGraphicsItem.NoCache`—`0`: 禁用缓存。

+   `QtWidgets.QGraphicsItem.ItemCoordinateCache`—`1`: 为该项的局部坐标系启用缓存。

+   `QtWidgets.QGraphicsItem.DeviceCoordinateCache`—`2`: 为该项的设备坐标系启用缓存。

`setCursor(QtGui.QCursor)`: 这为此项设置参数中指定的光标形状。

`setData(int, object)`: 这将为此项的键（第一个参数）设置自定义数据为值（第二个参数）。

`setEnabled(bool)`: 如果参数为 `True`，则启用此项。

`setFiltersChildEvents(bool)`: 如果参数为 `True`，则设置此项以过滤所有子项的事件。

`setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag, bool)`: 如果第二个参数为 `True`，则将启用此项的标志（第一个参数）。

`setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag | QtWidgets.QGraphicsItem.GraphicsItemFlag)`: 这将为此项设置在参数中指定的标志。

`setFocus(QtCore.Qt.FocusReason)`: 这将为此项提供键盘输入焦点。指定的参数将被传递到任何生成的焦点事件中。

`setFocusProxy(QtWidgets.QGraphicsItem)`: 这将为此项设置焦点代理为参数中指定的项。

`setGraphicsEffect(QtWidgets.QGraphicsEffect)`: 这将为此项设置在参数中指定的图形效果。

`setGroup(QtWidgets.QGraphicsItemGroup)`: 这将为此项设置在参数中指定的组。

`setHandlesChildEvents(bool)`: 如果参数为 `True`，此项将处理所有子事件。

`setInputMethodHints(QtCore.Qt.InputMethodHints)`: 这将为此项设置在参数中指定的输入法提示。

`setOpacity(float)`: 这将为此项设置局部透明度，介于 `0.0`（完全透明）和 `1.0`（完全不透明）之间。

`setPanelModality(QtWidgets.QGraphicsItem.PanelModality)`: 这将为此项设置在参数中指定的模态。可用的模态如下：

+   `QtWidgets.QGraphicsItem.NonModal`—`0`: 面板不是模态的。

+   `QtWidgets.QGraphicsItem.PanelModal`—`1`: 面板是模态的（项）。

+   `QtWidgets.QGraphicsItem.SceneModal`—`2`: 窗口是模态的（场景）。

`setParentItem(QtWidgets.QGraphicsItem)`: 这将为此项设置由参数指定的父项。

`setPos(QtCore.QPointF)`: 这将为此项设置由参数指定的位置。

`setPos(x, y)`: 这将为此项设置由 `x` 和 `y` 坐标指定的位置。

`setRotation(float)`: 这将为此项设置围绕 *z* 轴的旋转，以度为单位。如果值为正，则项将顺时针旋转；如果值为负，则逆时针旋转。

`setScale(float)`: 这将为此项设置缩放因子。

`setSelected(bool)`: 如果参数为 `True`，则此项将被选中。

`setToolTip("Tool tip")`: 这将为此项设置在参数中指定的工具提示。

`setTransform(QtGui.QTransform, bool)`: 这将为此项设置变换矩阵（第一个参数）。如果第二个参数为 `True`，则矩阵将与当前矩阵合并。

`setTransformations([QGraphicsTransform])`: 这将为此项设置要应用到此项的图形变换列表。

`setTransformOriginPoint(QtCore.QPointF)`: 这将为此项设置在参数中指定的变换点。

`setTransformOriginPoint(x, y)`: 这将为此项设置由 `x` 和 `y` 坐标指定的变换点。

`setVisible(bool)`: 如果参数为 `True`，则将此项设置为可见。

`setX(float)`: 设置此项目位置的 *x* 坐标，该坐标由参数指定。

`setY(float)`: 设置此项目位置的 *y* 坐标，该坐标由参数指定。

`setZValue(float)`: 设置此项目的 `Z` 值，该值由参数指定。`Z` 值是兄弟项目的堆叠顺序，其中 `Z` 值最高的项目位于顶部。

# has/is

这些函数返回与图形项目状态相关的布尔值 (`bool`)：

`hasCursor()`: 如果为此项目设置了光标，则返回 `True`。

`hasFocus()`: 如果此项目具有键盘输入焦点，则返回 `True`。

`isActive()`: 如果此项目处于活动状态，则返回 `True`。

`isAncestorOf(QtWidgets.QGraphicsItem)`: 如果此项目是参数中指定的子项目的祖先，则返回 `True`。

`isBlockedByModalPanel()`: 如果此项目被模态面板阻塞，则返回 `True`。

`isClipped()`: 如果此项目被裁剪，则返回 `True`。

`isEnabled()`: 如果此项目处于启用状态，则返回 `True`。

`isObscured(QtCore.QRectF)`: 如果参数中指定的矩形被此项目上方的任何碰撞项目的非透明形状遮挡，则返回 `True`。

`isObscured(x, y, w, h)`: 如果从 `x`/`y` 开始，宽度为 `w`，高度为 `h` 的区域被此项目上方的任何碰撞项目的非透明形状遮挡，则返回 `True`。

`isObscuredBy(QtWidgets.QGraphicsItem)`: 如果此项目的边界矩形被参数中指定的项目的非透明形状遮挡，则返回 `True`。

`isPanel():` 如果此项目是一个面板，则返回 `True`。

`isSelected()`: 如果此项目被选中，则返回 `True`。

`isUnderMouse()`: 如果此项目位于鼠标指针下方，则返回 `True`。

`isVisible()`: 如果此项目可见，则返回 `True`。

`isVisibleTo(QtWidgets.QGraphicsItem)`: 如果此项目对参数中指定的父项目可见，则返回 `True`。

`isWidget()`: 如果此项目是 `QGraphicsWidget` 小部件，则返回 `True`。

`isWindow()`: 如果此项目是 `QGraphicsWidget` 窗口，则返回 `True`。

# functional

这些函数返回图形项目的当前值，功能更改等：

`acceptDrops()`: 如果项目接受拖放事件，则返回 `True`。

`acceptedMouseButtons()`: 返回此项目接受鼠标事件时 `QtCore.Qt.MouseButtons` 类型的鼠标按钮。

`acceptHoverEvents()`: 如果项目接受悬停事件，则返回 `True`。

`acceptTouchEvents()`: 如果项目接受触摸事件，则返回 `True`。

`advance(int)`: 返回相位。在第一个阶段，所有项目都使用相等于 `0` 的相位被调用。这意味着场景中的项目即将前进一步，然后所有项目都使用相等于 `1` 的相位被调用。

`boundingRect()`: 返回描述此项目外边界的 `QtCore.QRectF` 类型的矩形。

`boundingRegion(QtGui.QTransform)`: 这返回使用指定参数的项目边界区域。

`boundingRegionGranularity()`: 这返回项目边界区域的粒度（应该是一个介于 `0` 和 `1` 之间的数字）。

`cacheMode()`: 这返回 `QtWidgets.QGraphicsItem.CacheMode` 类型的缓存模式。

`childItems()`: 这返回一个包含此项目子项的列表。

`childrenBoundingRect()`: 这返回此项目的后代项的边界矩形。

`clearFocus()`: 这将从该项目中获取键盘输入焦点，并在该项目有焦点的情况下发送焦点外事件。

`clipPath()`: 这返回此项目的 `QtGui.QPainterPath` 类型的裁剪路径。

`collidesWithItem(QtWidgets.QGraphicsItem, QtCore.Qt.ItemSelectionMode)`: 如果此项目与第一个参数中的项目碰撞，并且模式（第二个参数）与指定项目相关，则返回 `True`。

`collidesWithPath(QtGui.QPainterPath, QtCore.Qt.ItemSelectionMode)`: 如果此项目与路径（第一个参数）碰撞，并且模式（第二个参数）与指定路径相关，则返回 `True`。

`collidingItems(QtCore.Qt.ItemSelectionMode)`: 这返回与该项目碰撞的所有项目的列表。碰撞检测由参数中指定的模式确定。

`commonAncestorItem(QtWidgets.QGraphicsItem)`: 这返回此项目和指定参数中的项目的最近祖先项。

`contains(QtCore.QPointF)`: 如果此项目包含指定参数中的点，则返回 `True`。

`cursor()`: 这返回此项目的光标形状。

`data(int)`: 这返回指定参数中指定的键的此项目的自定义数据。

`deviceTransform(QtGui.QTransform)`: 这返回此项目的设备变换矩阵。

`effectiveOpacity()`: 这返回此项目的有效不透明度，可以是 `0.0`（完全透明）或 `1.0`（完全不透明）。

`filtersChildEvents()`: 如果此项目过滤子事件，则返回 `True`。

`flags()`: 这返回用于此项目的标志。

`focusItem()`: 如果此项目的后代或子项具有输入焦点，则返回 `QtWidgets.QGraphicsItem` 类型的项目。

`focusProxy()`: 这返回该项目的 `QtWidgets.QGraphicsItem` 类型的焦点代理。

`grabKeyboard()`: 这获取键盘输入。

`grabMouse()`: 这获取鼠标输入。

`graphicsEffect()`: 如果此项目存在 `QtWidgets.QGraphicsEffect` 类型的效果，则返回该效果。

`group()`: 如果该项目是组的一部分，则返回 `QtWidgets.QGraphicsItemGroup` 类型的该项目组。

`handlesChildEvents()`: 如果此项目处理子事件，则返回 `True`。

`hide()`: 此函数隐藏项目。

`itemChange(QtWidgets.QGraphicsItem.GraphicsItemChange, value)`: 这会通知自定义项目状态的一部分已更改。更改（第一个参数）是正在更改的项目，新值（第二个参数）都传递过去。

`itemTransform(QtWidgets.QGraphicsItem)`: 这返回一个映射从该项目到参数中指定的项目的坐标的 `QtGui.QTransform` 类型的转换。

`moveBy(float, float)`: 这将项目水平（第一个参数）和垂直（第二个参数）移动。

`opacity()`: 这返回项目的局部不透明度，介于 `0.0`（完全透明）和 `1.0`（完全不透明）之间。

`opaqueArea()`: 这返回此项目不透明的 `QtGui.QPainterPath` 类型的区域。

`paint(QtGui.QPainter, QtWidgets.QStyleOptionGraphicsItem, QtWidgets.QWidget)`: 这使用绘图器（第一个参数）、样式选项（第二个参数）和绘画发生的窗口（可选第三个参数）来绘制项目的内容。

`panel()`: 这返回此项目的 `QtWidgets.QGraphicsItem` 类型的面板。

`panelModality()`: 这返回此项目的 `QtWidgets.QGraphicsItem.PanelModality` 类型的模式。

`parentItem()`: 这返回此项目的 `QtWidgets.QGraphicsItem` 类型的父级项目。

`parentObject()`: 这返回此项目的 `QtWidgets.QGraphicsObject` 类型的父级对象。

`parentWidget()`: 这返回此项目的 `QtWidgets.QGraphicsWidget` 类型的父级窗口。

`pos()`: 这返回此项目在父级（或场景）坐标中的 `QtCore.QPointF` 类型的位置。

`prepareGeometryChange()`: 这为此项目准备几何更改。

`resetTransform()`: 这重置此项目的转换矩阵。

`rotation()`: 这返回此项目的旋转，以度为单位，沿 *z* 轴顺时针旋转。

`scale()`: 这返回此项目的缩放因子。

`scene()`: 这返回对此项目当前有效的 `QtWidgets.QGraphicsScene` 类型的场景。

`sceneBoundingRect()`: 这返回此项目在场景坐标中的 `QtCore.QRectF` 类型的边界矩形。

`scenePos()`: 这返回此项目在场景坐标中的位置。

`sceneTransform()`: 这返回此项目场景的 `QtGui.QTransform` 类型的转换矩阵。

`scroll(float, float, QtCore.QRectF)`: 这通过 *x*（第一个参数）和 *y* 值（第二个参数）滚动矩形（第三个参数）的内容。

`shape()`: 这返回项目的 `QtGui.QPainterPath` 类型的形状。

`show()`: 这将显示此项目。

`stackBefore(QtWidgets.QGraphicsItem)`: 这将此项目堆叠在参数中指定的兄弟项目之前。

`toGraphicsObject()`: 这返回将图形项目转换为 `QtWidgets.QGraphicsObject` 类型的图形对象。

`toolTip()`: 这返回此项目的工具提示。

`topLevelItem()`: 这返回此项目的 `QtWidgets.QGraphicsItem` 类型的顶级项目。

`topLevelWidget()`: 这返回此项目的 `QtWidgets.QGraphicsWidget` 类型的顶级小部件。

`transform()`: 这返回此项目的 `QtGui.QTransform` 类型的变换矩阵。

`transformations()`: 这返回此项目的图形变换列表。

`transformOriginPoint()`: 这返回此项目 `QtCore.QPointF` 类型的变换原点。

`type()`: 这以整数值返回项目的类型。

`ungrabKeyboard()`: 这释放键盘捕获。

`ungrabMouse()`: 这释放鼠标捕获。

`unsetCursor()`: 这取消此项目的光标。

`update(QtCore.QRectF)`: 这安排在此项目中重绘参数指定的区域。

`update(x, y, w, h)`: 这安排从 `x`/`y` 开始的区域的重绘，宽度为 `w` 和高度为 `h`。

`updateMicroFocus()`: 这将更新此项目的微焦点。

`window()`: 这返回此项目的 `QtWidgets.QGraphicsWidget` 类型的窗口。

`zValue()`: 这返回此项目的 z 值。

# map

这些函数与图形项的映射相关：

`mapFromItem(QtWidgets.QGraphicsItem, QtCore.QPointF)`: 这将把点（第二个参数），映射到项目（第一个参数）的坐标系中，并返回映射后的坐标。

`mapFromItem(QtWidgets.QGraphicsItem, QtGui.QPolygonF)`: 这将把多边形（第二个参数），映射到项目（第一个参数）的坐标系中，并返回映射后的坐标。

`mapFromItem(QtWidgets.QGraphicsItem, QtCore.QRectF)`: 这将把矩形（第二个参数），映射到项目（第一个参数）的坐标系中，并返回映射后的坐标。

`mapFromItem(QtWidgets.QGraphicsItem, QtGui.QPainterPath)`: 这将把路径（第二个参数），映射到项目（第一个参数）的坐标系中，并返回映射后的坐标。

`mapFromItem(QtWidgets.QGraphicsItem, x, y)`: 这将把由 `x` 和 `y` 设置的位置的点映射到项目（第一个参数）的坐标系中，并返回映射后的坐标。

`mapFromItem(QtWidgets.QGraphicsItem, x, y, w, h)`: 这将把从 `x`/`y` 开始的区域，宽度为 `w` 和高度为 `h`，映射到项目（第一个参数）的坐标系中，并将其映射到本项目的坐标系中，并返回映射后的坐标。

`mapFromParent(QtCore.QPointF)`: 这将把参数指定的点映射到项目的父坐标系中，并返回映射后的点。

`mapFromParent(QtGui.QPolygonF)`: 这将参数中指定的多边形（位于项的父坐标系中）映射到该项的坐标系，并返回映射后的多边形。

`mapFromParent(QtCore.QRectF)`: 这将参数中指定的矩形（位于项的父坐标系中）映射到该项的坐标系，并返回映射后的多边形。

`mapFromParent(QtGui.QPainterPath)`: 这将参数中指定的路径（位于项的父坐标系中）映射到该项的坐标系，并返回映射后的路径。

`mapFromParent(x, y)`: 这将位于 `x` 和 `y` 位置的点（位于项的父坐标系中）映射到该项的坐标系，并返回映射后的点。

`mapFromParent(x, y, w, h)`: 这将位于 `x`/`y` 位置，宽度为 `w` 和高度为 `h` 的区域（位于项的父坐标系中）映射到该项的坐标系，并返回映射后的多边形。

`mapFromScene(QtCore.QPointF)`: 这将参数中指定的点（位于项的场景坐标系中）映射到该项的坐标系，并返回映射后的坐标。

`mapFromScene(QtGui.QPolygonF)`: 这将参数中指定的多边形（位于项的场景坐标系中）映射到该项的坐标系，并返回映射后的多边形。

`mapFromScene(QtCore.QRectF)`: 这将参数中指定的矩形（位于项的场景坐标系中）映射到该项的坐标系，并返回映射后的多边形。

`mapFromScene(QtGui.QPainterPath)`: 这将参数中指定的路径（位于项的场景坐标系中）映射到该项的坐标系，并返回映射后的路径。

`mapFromScene(x, y)`: 这将位于 `x` 和 `y` 位置的点（位于项的场景坐标系中）映射到该项的坐标系，并返回映射后的点。

`mapFromScene(x, y, w, h)`: 这将位于 `x`/`y` 位置，宽度为 `w` 和高度为 `h` 的区域（位于项的场景坐标系中）映射到该项的坐标系，并返回映射后的多边形。

`mapRectFromItem(QtWidgets.QGraphicsItem, QtCore.QRectF)`: 这将位于项（第一个参数）坐标系中的矩形（第二个参数）映射到该项的坐标系，并返回新的映射矩形。

`mapRectFromItem(QtWidgets.QGraphicsItem, x, y, w, h)`: 这将位于 `x`/`y` 位置，宽度为 `w` 和高度为 `h` 的区域（位于项的坐标系中，第一个参数）映射到该项的坐标系，并返回新的映射矩形。

`mapRectFromParent(QtCore.QRectF)`: 这将参数中指定的矩形（在该项目的父坐标系中）映射到该项目的坐标系，并返回新的映射矩形。

`mapRectFromParent(x, y, w, h)`: 这将起始于 `x`/`y`，宽度为 `w` 和高度为 `h` 的区域（在该项目的父坐标系中）映射到该项目的坐标系，并返回新的映射矩形。

`mapRectFromScene(QtCore.QRectF)`: 这将参数中指定的矩形（在场景坐标系中）映射到该项目的坐标系，并返回新的映射矩形。

`mapRectFromScene(x, y, w, h)`: 这将起始于 `x`/`y`，宽度为 `w` 和高度为 `h` 的区域（在场景坐标系中）映射到该项目的坐标系，并返回新的映射矩形。

`mapRectToItem(QtWidgets.QGraphicsItem, QtCore.QRectF)`: 这将矩形（第二个参数，即第二个参数），映射到项目（第一个参数，即第一个参数）的坐标系，并返回新的映射矩形。

`mapRectToItem(QtWidgets.QGraphicsItem, x, y, w, h)`: 这将起始于 `x`/`y`，宽度为 `w` 和高度为 `h` 的区域（在该项目的坐标系中）映射到项目（第一个参数）的坐标系，并返回新的映射矩形。

`mapRectToParent(QtCore.QRectF)`: 这将参数中指定的矩形（在该项目的坐标系中）映射到该项目的父坐标系，并返回新的映射矩形。

`mapRectToParent(x, y, w, h)`: 这将起始于 `x`/`y`，宽度为 `w` 和高度为 `h` 的区域（在该项目的坐标系中）映射到该项目的父坐标系，并返回新的映射矩形。

`mapRectToScene(QtCore.QRectF)`: 这将参数中指定的矩形（在该项目的坐标系中）映射到场景坐标系，并返回新的映射矩形。

`mapRectToScene(x, y, w, h)`: 这将起始于 `x`/`y`，宽度为 `w` 和高度为 `h` 的区域（在该项目的坐标系中）映射到场景坐标系，并返回新的映射矩形。

`mapToItem(QtWidgets.QGraphicsItem, QtCore.QPointF)`: 这将点（第二个参数），映射到项目（第一个参数）的坐标系，并返回映射的点。

`mapToItem(QtWidgets.QGraphicsItem, QtGui.QPolygonF)`: 这将多边形（第二个参数），映射到项目（第一个参数）的坐标系，并返回映射的多边形。

`mapToItem(QtWidgets.QGraphicsItem, QtCore.QRectF)`: 将矩形（第二个参数），位于此项目的坐标系中，映射到项目（第一个参数）的坐标系中，并返回映射后的矩形。

`mapToItem(QtWidgets.QGraphicsItem, QtGui.QPainterPath)`: 将路径（第二个参数），位于此项目的坐标系中，映射到项目（第一个参数）的坐标系中，并返回映射后的路径。

`mapToItem(QtWidgets.QGraphicsItem, x, y)`: 将由 `x` 和 `y` 设置的位置的点（位于此项目的坐标系中）映射到项目（第一个参数）的坐标系中，并返回映射后的点。

`mapToItem(QtWidgets.QGraphicsItem, x, y, w, h)`: 将从 `x`/`y` 开始，宽度为 `w` 和高度为 `h` 的区域（位于此项目的坐标系中）映射到项目（第一个参数）的坐标系中，并返回映射后的多边形。

`mapToParent(QtCore.QPointF)`: 将参数中指定的点（位于此项目的坐标系中）映射到此项目父项目的坐标系中，并返回映射后的点。

`mapToParent(QtGui.QPolygonF)`: 将参数中指定的多边形（位于此项目的坐标系中）映射到此项目父项目的坐标系中，并返回映射后的多边形。

`mapToParent(QtCore.QRectF)`: 将参数中指定的矩形（位于此项目的坐标系中）映射到此项目父项目的坐标系中，并返回映射后的矩形

`mapToParent(QtGui.QPainterPath)`: 将参数中指定的路径（位于此项目的坐标系中）映射到此项目父项目的坐标系中，并返回映射后的路径。

`mapToParent(x, y)`: 将由 `x` 和 `y` 定位的点（位于此项目的坐标系中）映射到此项目父项目的坐标系中，并返回映射后的点。

`mapToParent(QtWidgets.QGraphicsItem, x, y, w, h)`: 将从 `x`/`y` 开始，宽度为 `w` 和高度为 `h` 的区域（位于此项目的坐标系中）映射到此项目父项目的坐标系中，并返回映射后的多边形。

`mapToScene(QtCore.QPointF)`: 将参数中指定的点（位于此项目的坐标系中）映射到场景的坐标系中，并返回映射后的点。

`mapToScene(QtGui.QPolygonF)`: 将参数中指定的多边形（位于此项目的坐标系中）映射到场景的坐标系中，并返回映射后的多边形。

`mapToScene(QtCore.QRectF)`: 将参数中指定的矩形（位于此项目的坐标系中）映射到场景的坐标系中，并返回映射后的矩形。

`mapToScene(QtGui.QPainterPath)`: 此函数将位于此项目坐标系中的参数中指定的路径映射到场景坐标系，并返回映射的路径。

`mapToScene(x, y)`: 此函数将位于此项目坐标系中的由 `x` 和 `y` 定位的点映射到场景坐标系，并返回映射的点。

`mapToScene(x, y, w, h)`: 此函数将位于此项目坐标系中的以 `x`/`y` 为起点，宽度为 `w`，高度为 `h` 的区域映射到场景坐标系，并返回映射的多边形。

# 事件

这些函数与事件相关，例如事件处理器：

`contextMenuEvent(QtWidgets.QGraphicsSceneContextMenuEvent)`: 此事件处理器接收此项目的上下文菜单事件。

`dragEnterEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理器接收此项目针对参数中指定事件的拖动进入事件。

`dragLeaveEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理器接收此项目针对参数中指定事件的拖动离开事件。

`dragMoveEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理器接收此项目针对参数中指定事件的拖动移动事件。

`dropEvent(QtWidgets.QGraphicsSceneDragDropEvent)`: 此事件处理器接收此项目针对参数中指定事件的放下事件。

`focusOutEvent(QtGui.QFocusEvent)`: 此事件处理器接收在失去焦点时带有事件参数传递的键盘焦点事件。

`focusInEvent(QtGui.QFocusEvent)`: 此事件处理器接收在获得焦点时带有事件参数传递的键盘焦点事件。

`hoverEnterEvent(QtWidgets.QGraphicsSceneHoverEvent)`: 此事件处理器接收此项目针对参数中指定事件的悬停进入事件。

`hoverLeaveEvent(QtWidgets.QGraphicsSceneHoverEvent)`: 此事件处理器接收此项目针对参数中指定事件的悬停离开事件。

`hoverMoveEvent(QtWidgets.QGraphicsSceneHoverEvent)`: 此事件处理器接收此项目针对参数中指定的事件的悬停移动事件

`inputMethodEvent(QtGui.QInputMethodEvent)`: 此事件处理器接收此项目的输入法事件。

`installSceneEventFilter(QtWidgets.QGraphicsItem)`: 此函数为此项目安装参数中指定的事件过滤器。

`keyPressEvent(QtGui.QKeyEvent)`: 此事件处理器接收通过参数传递的事件的按键事件为此项目。

`keyReleaseEvent(QtGui.QKeyEvent)`: 此事件处理器接收通过参数传递的事件的按键释放事件为此项目。

`mouseDoubleClickEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理器接收通过参数传递的事件的鼠标双击事件为此项目。

`mouseMoveEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理器接收此项目的鼠标移动事件，事件通过参数传入。

`mousePressEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理器接收此项目的鼠标按下事件，事件通过参数传入。

`mouseReleaseEvent(QtWidgets.QGraphicsSceneMouseEvent)`: 此事件处理器接收此项目的鼠标释放事件，事件通过参数传入。

`removeSceneEventFilter(QtWidgets.QGraphicsItem)`: 这将移除为此项目指定的参数中的事件过滤器。

`sceneEvent(QtCore.QEvent)`: 这接收此项目的事件。

`sceneEventFilter(QtWidgets.QGraphicsItem, QtCore.QEvent)`: 这个过滤器为项目（第一个参数）和过滤事件（第二个参数）过滤事件。

`wheelEvent(QtWidgets.QGraphicsSceneWheelEvent)`: 此事件处理器接收此项目的鼠标滚轮事件，事件通过参数传入。

# 摘要

在本章中，我们提供了可用于 GUI 应用程序的图形元素最常用类的描述。

在下一章中，我们将描述这些元素的各种图形效果。我们还将介绍在应用程序开发期间实现附加技术的特殊模块。
