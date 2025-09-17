# 图形基础

Qt 框架为您提供了在应用程序中使用和/或表示各种图形的机会。在这里，我们将介绍我们可以用来绘制、绘制和使用图像的基本类。本章主要涉及使用 Qt 和 Python 绑定管理创建图形元素的基础知识。由于 Qt 是一个图形库，这个主题非常广泛，因此本章基本上是图形基础介绍，将在未来的章节中进一步展开。

在本章中，我们将涵盖以下主题：

+   基础

+   绘制和渐变

+   图片

# 基础

Qt 框架和 Python 绑定，如 PySide 和 PyQt，实际上绘制了在上一章中描述的图形组件。这种绘制与按钮、字段、标签和其他小部件相关。正如我们之前所描述的，`QWidget` 类继承自 `QObject` 类，它为图形元素提供了一些功能，以及 `QPaintDevice` 类，它提供了在 GUI 应用程序中绘制元素的功能。这种结构是 Qt 图形库的主要范式。此外，还有一些支持类，将在本节中首先介绍。然而，本章无法容纳所有需要学习的类，以便完全理解在开发过程中管理图形的机制。为了实现这一点，请查阅相关文档（Qt—[`doc.qt.io/`](https://doc.qt.io/)，PySide2—[`doc.qt.io/qtforpython/index.html`](https://doc.qt.io/qtforpython/index.html)，PyQt5—[`www.riverbankcomputing.com/static/Docs/PyQt5/`](https://www.riverbankcomputing.com/static/Docs/PyQt5/)）。

# QFont

此类提供了一个字体，该字体将用于绘制应用程序中组件的文本。还有其他用于处理字体的类，例如 `QFontInfo`、`QFontMetrics`、`QFontMetricsF` 和 `QFontDatabase`。在代码中使用 `QFont` 类可能的形式如下：

```py
font = QtGui.QFont()
```

`QFont` 通过以下功能提高功能。

# 设置

这些是与设置字体相关参数和属性相关的函数，包括设置字体家族和间距：

`font.setBold(bool)`: 如果参数为 `True`，则将当前字体设置为粗体。

`font.setCapitalization(QtGui.QFont.Capitalization)`: 这将设置文本的字母大小写；所有单词都将大写。可用的参数如下：

+   `QtGui.QFont.MixedCase`—`0`: 不应用大小写。

+   `QtGui.QFont.AllUppercase`—`1`: 全大写。

+   `QtGui.QFont.AllLowercase`—`2`: 全小写。

+   `QtGui.QFont.SmallCaps`—`3`: 小写字母。

+   `QtGui.QFont.Capitalize`—`4`: 每个单词的首字母大写。

`font.setFamily("Font family")`: 这将为此字体设置字体家族名称（不区分大小写）。

`font.setFixedPitch(bool)`: 如果参数为`True`，则为此字体设置固定间距。

`font.setHintingPreference(QtGui.QFont.HintingPreference)`: 这将为参数中指定的此字体的符号设置提示级别。这取决于操作系统。可用的参数如下：

+   `QtGui.QFont.PreferDefaultHinting`—`0`: 平台的默认提示级别。

+   `QtGui.QFont.PreferNoHinting`—`1`: 在可能的情况下，不提示的文本。

+   `QtGui.QFont.PreferVerticalHinting`—`2`: 在可能的情况下，垂直对齐符号，而不进行文本的水平提示。

+   `QtGui.QFont.PreferFullHinting`—`3`: 在可能的情况下，提供水平和垂直提示。

`font.setItalic(bool)`: 如果参数为`True`，则将字体设置为斜体（草书）。

`font.setKerning(bool)`: 如果参数为`True`，则为此字体设置启用间距（默认：`True`）；符号度量不累加。

`font.setLetterSpacing(QtGui.QFont.SpacingType, float)`: 这将为具有在第一个参数中指定的类型和第二个参数中指定的`float`值的间距的文本字体设置每个字母之间的间距。可用的间距类型如下：

+   `QtGui.QFont.PercentageSpacing`—`0`: 间距作为百分比；值为`200.0`时，间距将根据其字符宽度放大。

+   `QtGui.QFont.AbsoluteSpacing`—`0`: 像素间距。

`font.setOverline(bool)`: 如果参数为`True`，则将此字体的文本设置为上划线。

`font.setPixelSize(int)`: 这将设置参数中指定的字体文本的大小，以像素值表示。

`font.setPointSize(int)`: 这将设置参数中指定的字体文本的大小，以点值表示。

`font.setPointSizeF(float)`: 这将设置参数中指定的字体文本的大小，以点值表示，具有浮点精度。

`font.setRawName(str)`: 这将设置由其系统名称使用的字体。

`font.setStretch(int)`: 这将设置此字体的拉伸因子，从`1`到`4000`（因子为`250`时，所有字符都将宽 2.5 倍）。

`font.setStrikeOut(bool)`: 如果参数为`True`，则将此字体的文本设置为删除线。

`font.setStyle(QtGui.QFont.Style)`: 这将设置参数中指定的字体样式。可用的样式如下：

+   `QtGui.QFont.StyleNormal`—`0`: 正常符号。

+   `QtGui.QFont.StyleItalic`—`1`: 斜体符号。

+   `QtGui.QFont.StyleOblique`—`2`: 基于无样式符号的斜体外观符号。

`font.setStyleHint(QtGui.QFont.StyleHint, QtGui.QFont.StyleStrategy)`: 这将为字体匹配器首选的样式提示（第一个参数）和提示策略（第二个参数）设置样式提示。可用的样式提示如下：

+   `QtGui.QFont.SansSerif`—`Helvetica`: 优先使用无衬线字体。

+   `QtGui.QFont.Helvetica`—`0`: 优先使用无衬线字体。

+   `QtGui.QFont.Serif`—`Times`: 优先选择衬线字体。

+   `QtGui.QFont.Times`—`1`: 优先选择衬线字体。

+   `QtGui.QFont.TypeWriter`—`Courier`: 优先选择固定间距字体。

+   `QtGui.QFont.Courier`—`2`: 优先选择固定间距字体。

+   `QtGui.QFont.OldEnglish`—`3`: 优先选择装饰字体。

+   `QtGui.QFont.Decorative`—`OldEnglish`: 优先选择装饰字体。

+   `QtGui.QFont.System`—`4`: 优先选择系统字体。

+   `QtGui.QFont.AnyStyle`—`5`: 优先选择选择。

+   `QtGui.QFont.Cursive`—`6`: 优先选择草书字体家族。

+   `QtGui.QFont.Monospace`—`7`: 优先选择等宽字体家族。

+   `QtGui.QFont.Fantasy`—`8`: 优先选择幻想字体家族。

`font.setStyleName(str)`: 这将设置样式名称，名称由参数指定。

`font.setStyleStrategy(QtGui.QFont.StyleStrategy)`: 这将设置字体的样式策略。以下样式策略可用于指定字体匹配器应使用哪种类型来查找默认家族：

+   `QtGui.QFont.PreferDefault`: 不优先选择任何字体。

+   `QtGui.QFont.PreferBitmap`: 优先选择位图字体。

+   `QtGui.QFont.PreferDevice`: 优先选择设备字体。

+   `QtGui.QFont.ForceOutline`: 使用轮廓字体。

+   `QtGui.QFont.NoAntialias`: 不对字体进行抗锯齿。

+   `QtGui.QFont.NoSubpixelAntialias`: 如果可能，将不会对字体进行子像素抗锯齿。

+   `QtGui.QFont.PreferAntialias`: 优先选择抗锯齿。

+   `QtGui.QFont.OpenGLCompatible`: 使用与 OpenGL 兼容的字体。

+   `QtGui.QFont.NoFontMerging`: 如果书写系统不包含此字符，则禁用自动查找类似字体的功能。

+   `QtGui.QFont.PreferNoShaping`: 禁用不需要时应用复杂规则等特性。

可用于与 OR (`|`) 运算符一起使用的可用标志如下：

+   `QtGui.QFont.PreferMatch`: 优先选择精确匹配。

+   `QtGui.QFont.PreferQuality`: 优先选择最佳质量的字体。

+   `QtGui.QFont.ForceIntegerMetrics`: 在字体引擎中使用整数值。

`font.setUnderline(bool)`: 如果参数为 `True`，则将此字体的文本设置为下划线。

`font.setWeight(int)`: 这将设置字体的重量，重量由参数指定。字体重量的可用值如下：

+   `QtGui.QFont.Thin`: `0`

+   `QtGui.QFont.ExtraLight`: `12`

+   `QtGui.QFont.Light`: `25`

+   `QtGui.QFont.Normal`: `50`

+   `QtGui.QFont.Medium`: `57`

+   `QtGui.QFont.DemiBold`: `63`

+   `QtGui.QFont.Bold`: `75`

+   `QtGui.QFont.ExtraBold`: `81`

+   `QtGui.QFont.Black`: `87`

`font.setWordSpacing(float)`: 这将设置此字体中每个单词之间的间距，间距由参数指定。

# is

这些是返回与字体相关的布尔值（`bool`）的函数：

`font.isCopyOf(QtGui.QFont)`: 如果在参数中指定的字体是当前字体的副本，则返回 `True`。

# functional

这些是与当前字体值的返回相关的函数：

`font.bold()`: 如果 `font.weight()` 的值大于 `QFont.Medium`，则返回 `True`；否则返回 `False`。

`font.cacheStatistics()`: 这输出字体的缓存统计信息。

`font.capitalization()`: 这返回此字体`QtGui.QFont.Capitalization`类型的当前大写化。

`font.cleanup()`: 这清理字体系统。

`font.defaultFamily()`: 这返回用于当前样式提示的字体家族。

`font.exactMatch()`: 如果窗口系统中有与该字体设置匹配的字体，则返回`True`。

`font.family()`: 这返回与该字体一起使用的当前字体家族名称。

`font.fixedPitch()`: 如果为此字体设置了固定间距，则返回`True`。

`font.fromString(str)`: 这设置此字体，它将匹配参数中指定的描述，作为带有字体属性的逗号分隔列表。

`font.hintingPreference()`: 这返回当前首选的符号的提示级别，并将使用此字体进行渲染。

`font.initialize()`: 这初始化字体系统。

`font.insertSubstitution(str, str)`: 这将第二个参数中指定的替代名称插入到字体家族（第一个参数）的替换表中。

`font.insertSubstitution(str, [str])`: 这将第二个参数中指定的替代名称列表插入到字体家族（第一个参数）的替换表中。

`font.italic()`: 如果当前字体是斜体（草书），则返回`True`。

`font.kerning()`: 如果与此字体一起使用字距调整，则返回`True`。

`font.key()`: 这返回当前字体的键，作为此字体的文本表示。

`font.lastResortFamily()`: 这返回字体家族名称，作为最后的手段。

`font.lastResortFont()`: 这返回字体：最后的手段。

`font.letterSpacing()`: 这以浮点值返回每个字母之间的间距，该值与此字体一起使用。

`font.letterSpacingType()`: 这返回作为`QtGui.QFont.SpacingType`的间距类型，并用于字母间距。

`font.overline()`: 如果字体有上划线，则返回`True`。

`font.pixelSize()`: 这返回像素表示中的字体大小。

`font.pointSize()`: 这以点表示返回字体的大小。

`font.pointSizeF()`: 这以浮点值返回点表示中的字体大小。

`font.rawName()`: 这返回底层窗口系统中使用的字体名称。

`font.removeSubstitutions(str)`: 这删除参数中指定的字体家族名称的替换。

`font.resolve(QtGui.QFont)`: 这返回具有参数中指定的字体属性的新字体，这些属性尚未设置。

`font.stretch()`: 这返回此字体的拉伸因子。

`font.strikeOut()`: 如果此字体的文本有删除线，则返回`True`。

`font.style()`: 这返回此字体`QtGui.QFont.Style`类型的样式。

`font.styleHint()`: 这返回此字体`QtGui.QFont.StyleHint`类型的样式提示。

`font.styleName()`: 这返回用于此字体样式的样式名称。

`font.styleStrategy()`: 这返回`QtGui.QFont.StyleStrategy`类型的策略，并用于此字体的字体匹配器。

`font.substitute(str)`: 这返回与参数中指定的家族名称一起使用的第一个替换字体家族名称。

`font.substitutes(str)`: 这返回使用指定家族名称的替换字体家族名称列表。

`font.substitutions()`: 这返回一个排序后的替换字体家族名称列表。

`font.swap(QtGui.QFont)`: 这将此字体与参数中指定的字体进行交换。

`font.toString()`: 这返回以逗号分隔的描述此字体的列表。

`font.underline()`: 如果此字体的文本被下划线，则返回`True`。

`font.weight()`: 这返回此字体的权重。

`font.wordSpacing()`: 这返回文本中每个单词之间的间距，以浮点值表示。

# QColor

此类提供了用于应用程序组件的颜色操作。此类表示基于 **RGB** （**红色、绿色和蓝色**）、**HSV** （**色调、饱和度和亮度**）和 **CMYK** （**青色、品红色、黄色和黑色**）值颜色的模型。它们可以按以下方式使用：

+   **命名颜色**："white"。

+   **字符串/值**："#FFFFFF"，`(255, 255, 255)`，`(1, 1, 1)`，换句话说，*#redgreenblue* 或 (red, green, blue)。

+   **字符串/值**："#FFFFFFFF"，`(255, 255, 255, 255)`，`(1, 1, 1, 1)`，换句话说，*#redgreenbluealpha* 或 (red, green, blue, alpha)。

颜色的 alpha 通道（RGBA 中的字母*A*）是透明度。HSV 和 CMYK 可以以类似的方式用于其颜色模型的颜色。我们可以像这样使用`QColor`类：

```py
color = QtGui.QColor()
```

`QColor`通过以下函数提高功能。

# set

这些是与设置颜色相关参数和属性相关的函数，包括设置颜色或色调：

`color.setAlpha(a)`: 这将 alpha，`a`，设置为该颜色的整数值（`0`-`255`）。

`color.setAlphaF(a)`: 这将 alpha，`a`，设置为该颜色的浮点值（`0.0`-`1.0`）。

`color.setBlue(b)`: 这将蓝色，`b`，设置为该颜色的整数值（`0`-`255`）。

`color.setBlueF(b)`: 这将蓝色，`b`，设置为该颜色的浮点值（`0.0`-`1.0`）。

`color.setCmyk(c, m, y, k, a)`: 这将参数中指定的 CMYK 颜色设置为：

+   `c`: 青色 (`0`-`255`).

+   `m`: 品红色 (`0`-`255`).

+   `y`: 黄色 (`0`-`255`).

+   `k`: 黑色 (`0`-`255`).

+   `a`: Alpha (`0`-`255`).

`color.setCmykF(c, m, y, k, a)`: 这将参数中指定的 CMYK 颜色设置为：

+   `c`: 青色 (`0.0`-`1.0`).

+   `m`: 品红色 (`0.0`-`1.0`).

+   `y`: 黄色 (`0.0`-`1.0`).

+   `k`: 黑色 (`0.0`-`1.0`).

+   `a`: Alpha (`0.0`-`1.0`).

`color.setGreen(g)`: 这将绿色，`g`，设置为该颜色的整数值（`0`-`255`）。

`color.setGreenF(g)`: 这将绿色，`g`，设置为该颜色的浮点值（`0.0`-`1.0`）。

`color.setHsl(h, s, l, a)`: 这设置由参数指定的 HSL 颜色：

+   `h`: 色调 (`0`-`255`)。

+   `s`: 饱和度 (`0`-`255`)。

+   `l`: 亮度 (`0`-`255`)。

+   `a`: 透明度 (`0.0`-`255`)。

`color.setHslF(h, s, l, a)`: 这设置由参数指定的 HSL 颜色：

+   `h`: 色调 (`0.0`-`1.0`)。

+   `s`: 饱和度 (`0.0`-`1.0`)。

+   `l`: 亮度 (`0.0`-`1.0`)。

+   `a`: 透明度 (`0.0`-`1.0`)。

`color.setHsv(h, s, v, a)`: 这设置由参数指定的 HSV 颜色：

+   `h`: 色调 (`0`-`255`)。

+   `s`: 饱和度 (`0`-`255`)。

+   `v`: 亮度 (`0`-`255`)。

+   `a`: 透明度 (`0`-`255`)。

`color.setHsvF(h, s, v, a)`: 这设置由参数指定的 HSV 颜色：

+   `h`: 色调 (`0.0`-`1.0`)。

+   `s`: 饱和度 (`0.0`-`1.0`)。

+   `v`: 亮度 (`0.0`-`1.0`)。

+   `a`: 透明度 (`0.0`-`1.0`)。

`color.setNamedColor(str)`: 这将此颜色的 RGB 值设置为参数中指定的名称。将要命名的颜色格式如下：`"#RGB"`，`"#RRGGBB"`，`"#AARRGGBB"`，`"#RRRGGGBBB"`，和 `"#RRRRGGGGBBBB"`。

`color.setRed(r)`: 这将红色 `r` 设置为该颜色的整数值 (`0`-`255`)。

`color.setRedF(r)`: 这将红色 `r` 设置为该颜色的浮点值 (`0.0`-`1.0`)。

`color.setRgb(r, g, b, a)`: 这设置由参数指定的 RGB 颜色：

+   `r`: 红色 (`0`-`255`)。

+   `g`: 绿色 (`0`-`255`)。

+   `b`: 蓝色 (`0`-`255`)。

+   `a`: 透明度 (`0`-`255`)。

`color.setRgbF(r, g, b, a)`: 这设置由参数指定的 RGB 颜色：

+   `r`: 红色 (`0.0`-`1.0`)。

+   `g`: 绿色 (`0.0`-`1.0`)。

+   `b`: 蓝色 (`0.0`-`1.0`)。

+   `a`: 透明度 (`0.0`-`1.0`)。

# is

这些是返回与颜色相关的布尔值 (`bool`) 的函数：

`color.isValid()`: 如果此颜色有效，则返回 `True`。

`color.isValidColor(str)`: 如果参数中指定的颜色是有效颜色，则返回 `True`。

# functional

这些是与当前颜色值返回相关的函数：

`color.alpha()`: 这返回颜色的透明度粒子，以该颜色的整数值表示。

`color.alphaF()`: 这返回颜色的透明度粒子，以该颜色的浮点值表示。

`color.black()`: 这返回颜色的黑色粒子，以该颜色的整数值表示。

`color.blackF()`: 这返回颜色的黑色粒子，以该颜色的浮点值表示。

`color.blue()`: 这返回颜色的蓝色粒子，以该颜色的整数值表示。

`color.blueF()`: 这返回颜色的蓝色粒子，以该颜色的浮点值表示。

`color.colorNames()`: 这返回包含 Qt 颜色列表中可用颜色名称的字符串列表。

`color.convertTo(QtGui.QColor.Spec)`: 这将参数中的指定符的颜色复制并返回 `QtGui.QColor` 类型的颜色。可用的指定符如下：

+   `QtGui.QColor.Rgb`—`1`: 红色、绿色、蓝色。

+   `QtGui.QColor.Hsv`—`2`: 色调、饱和度、亮度。

+   `QtGui.QColor.Cmyk`—`3`: 青色、品红色、黄色、黑色。

+   `QtGui.QColor.Hsl`—`4`: 色调、饱和度、亮度。

`color.cyan()`: 此函数返回颜色的青色部分，以该颜色的整数值表示。

`color.cyanF()`: 此函数返回颜色的青色部分，以该颜色的 `float` 值表示。

`color.darker(int)`: 此函数返回 `QtGui.QColor` 类型的较暗或较亮颜色，参数中指定了一个整数值作为因子。如果因子大于 `100`，则颜色较暗；如果因子小于 `100`，则颜色较亮。

`color.fromCmyk(c, m, y, k, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 CMYK 颜色：

+   `c`: 青色 (`0`-`255`).

+   `m`: 品红色 (`0`-`255`).

+   `y`: 黄色 (`0`-`255`).

+   `k`: 黑色 (`0`-`255`).

+   `a`: 透明度 (`0`-`255`).

`color.fromCmykF(c, m, y, k, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 CMYK 颜色：

+   `c`: 青色 (`0.0`-`1.0`).

+   `m`: 品红色 (`0.0`-`1.0`).

+   `y`: 黄色 (`0.0`-`1.0`).

+   `k`: 黑色 (`0`-`1.0`).

+   `a`: 透明度 (`0.0`-`1.0`).

`color.fromHsl(h, s, l, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 HSL 颜色：

+   `h`: 色调 (`0`-`255`).

+   `s`: 饱和度 (`0`-`255`).

+   `l`: 亮度 (`0`-`255`).

+   `a`: 透明度 (`0`-`255`).

`color.fromHslF(h, s, l, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 HSL 颜色：

+   `h`: 色调 (`0.0`-`1.0`).

+   `s`: 饱和度 (`0.0`-`1.0`).

+   `l`: 亮度 (`0.0`-`1.0`).

+   `a`: 透明度 (`0.0`-`1.0`).

`color.fromHsv(h, s, v, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 HSV 颜色：

+   `h`: 色调 (`0`-`255`).

+   `s`: 饱和度 (`0`-`255`).

+   `v`: 价值 (`0`-`255`).

+   `a`: 透明度 (`0`-`255`).

`color.fromHsvF(h, s, v, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 HSV 颜色：

+   `h`: 色调 (`0.0`-`1.0`).

+   `s`: 饱和度 (`0.0`-`1.0`).

+   `v`: 价值 (`0.0`-`1.0`).

+   `a`: 透明度 (`0.0`-`1.0`).

`color.fromRgb(r, g, b, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 RGB 颜色：

+   `r`: 红色 (`0`-`255`).

+   `g`: 绿色 (`0`-`255`).

+   `b`: 蓝色 (`0`-`255`).

+   `a`: 透明度 (`0`-`255`).

`color.fromRgbF(r, g, b, a)`: 此函数返回具有指定参数的 `QtGui.QColor` 类型的 RGB 颜色：

+   `r`: 红色 (`0.0`-`1.0`) .

+   `g`: 绿色 (`0.0`-`1.0`).

+   `b`: 蓝色 (`0.0`-`1.0`).

+   `a`: 透明度 (`0.0`-`1.0`).

`color.getCmyk()`: 此函数检索 *c*、*m*、*y*、*k* 和 *a* 的内容，并以整数值设置组件为 CMYK 颜色的青色、品红色、黄色、黑色和透明度值。

`color.getCmykF()`: 此函数检索 *c*、*m*、*y*、*k* 和 *a* 的内容，并以 `float` 值设置组件为 CMYK 颜色的青色、品红色、黄色、黑色和透明度值。

`color.getHsl()`: 此函数检索 *h*、*s*、*l* 和 *a* 的内容，并以整数值设置组件为 HSL 颜色的色调、饱和度、亮度和透明度值。

`color.getHslF()`: 此函数检索 *h*、*s*、*l* 和 *a* 的内容，并以 `float` 值设置组件为 HSL 颜色的色调、饱和度、亮度和透明度值。

`color.getHsv()`: 这检索 *h*、*s*、*v* 和 *a* 的内容，并以整数值设置这些组件为 HSV 颜色的色调、饱和度、值和透明度。

`color.getHsvF()`: 这检索 *h*、*s*、*v* 和 *a* 的内容，并以浮点值设置这些组件为 HSV 颜色的色调、饱和度、值和透明度。

`color.getRgb()`: 这检索 *r*、*g*、*b* 和 *a* 的内容，并以整数值设置这些组件为 RGB 颜色的红色、绿色、蓝色和透明度值。

`color.getRgbF()`: 这检索 *r*、*g*、*b* 和 *a* 的内容，并以浮点值设置这些组件为 RGB 颜色的红色、绿色、蓝色和透明度。

`color.green()`: 这返回颜色的绿色粒子，以该颜色的整数值表示。

`color.greenF()`: 这返回颜色的绿色粒子，以该颜色的浮点值表示。

`color.hslHue()`: 这返回颜色的色调粒子，以该颜色的整数值表示。

`color.hslHueF()`: 这返回颜色的色调粒子，以该颜色的浮点值表示。

`color.hslSaturation()`: 这返回颜色的饱和度粒子，以该颜色的整数值表示。

`color.hslSaturationF()`: 这返回颜色的饱和度粒子，以该颜色的浮点值表示。

`color.hsvHue()`: 这返回颜色的色调粒子，以该颜色的整数值表示。

`color.hsvHueF()`: 这返回颜色的色调粒子，以该颜色的浮点值表示。

`color.hsvSaturation()`: 这返回颜色的饱和度粒子，以该颜色的整数值表示。

`color.hsvSaturationF()`: 这返回颜色的饱和度粒子，以该颜色的浮点值表示。

`color.hue()`: 这返回颜色的色调粒子，以该颜色的整数值表示。

`color.hueF()`: 这返回颜色的色调粒子，以该颜色的浮点值表示。

`color.lighter(int)`: 这返回一个 `QtGui.QColor` 类型的更亮或更暗的颜色，参数中指定的因子以整数值表示。如果因子大于 `100`，颜色更亮；如果因子小于 `100`，颜色更暗。

`color.lightness()`: 这返回颜色的亮度粒子，以该颜色的整数值表示。

`color.lightnessF()`: 这返回颜色的亮度粒子，以该颜色的浮点值表示。

`color.magenta()`: 这返回颜色的洋红色粒子，以该颜色的整数值表示。

`color.magentaF()`: 这返回颜色的洋红色粒子，以该颜色的浮点值表示。

`color.name()`: 这以 `"#RRGGBB"` 格式返回颜色名称。

`color.name(QtGui.QColor.NameFormat)`: 这返回参数中指定的格式的颜色名称。

`color.red()`: 这返回颜色的红色粒子，以该颜色的整数值表示。

`color.redF()`: 这返回颜色的红色粒子，以该颜色的浮点值表示。

`color.saturation()`: 这返回颜色的饱和度粒子，以该颜色的整数值形式。

`color.saturationF()`: 这返回颜色的饱和度粒子，以该颜色的浮点值形式。

`color.spec()`: 这返回该颜色的 `QtGui.QColor.Spec` 类型的指定符。

`color.toCmyk()`: 这创建并返回该颜色的 `QtGui.QColor` 类型的 CMYK 颜色。

`color.toHsl()`: 这创建并返回该颜色的 `QtGui.QColor` 类型的 HSL 颜色。

`color.toHsv()`: 这创建并返回该颜色的 `QtGui.QColor` 类型的 HSV 颜色。

`color.toRgb()`: 这创建并返回该颜色的 `QtGui.QColor` 类型的 RGB 颜色。

`color.value()`: 这返回颜色的值粒子，以该颜色的整数值形式。

`color.valueF()`: 这返回颜色的值粒子，以该颜色的浮点值形式。

`color.yellow()`: 这返回颜色的黄色粒子，以该颜色的整数值形式。

`color.yellowF()`: 这返回颜色的黄色粒子，以该颜色的浮点值形式。

# 绘制和渐变

Qt 框架提供了您可以在应用程序中用于处理绘图和绘画的类。几乎每个小部件都是为了可视化目的而构建的，并且使用 Qt 库类图形表示的那些小部件将根据这些类进行绘制或绘制。渐变使各种元素的色彩更加现代，并赋予它们更佳的外观。

# QPainter

`QPainter` 类是 Qt 图形系统的主要组件之一。此类提供以各种形式对图形元素进行低级绘制的功能。从 `QWidget` 继承的 `QPaintDevice` 类与该类一起绘制元素。`QPainter` 类用于执行与绘制和绘画相关的操作，以及 `QPaintDevice` 和 `QPaintEngine` 等类。此类的声明语法如下：

```py
painter = QtGui.QPainter()
```

此外，如果绘图设备是小部件，则此类只能在 `paintEvent()` 函数或调用 `paintEvent()` 函数的函数内部实现，如下所示：

```py
def paintEvent(self, event):
   painter = QtGui.QPainter(self)
```

`QPainter` 类通过以下函数提高功能。

# 设置

这些是与设置画家相关参数和属性相关的函数。

`painter.setBackground(QtGui.QBrush)`: 这将此画家的背景画笔设置为参数中指定的画笔。

`painter.setBackgroundMode(QtCore.Qt.BGMode)`: 这将此画家的背景模式设置为参数中指定的模式。

`painter.setBrush(QtGui.QBrush)`: 这将此画家的画笔设置为参数中指定的画笔。

`painter.setBrush(QtCore.Qt.BrushStyle)`: 这将此画家的画笔设置为黑色和指定的样式。

`painter.setBrushOrigin(QtCore.QPoint)`: 这将画笔原点设置为参数中指定的点，使用整数值。

`painter.setBrushOrigin(QtCore.QPointF)`: 这将设置画笔原点为参数中指定的具有浮点值的点。

`painter.setBrushOrigin(x, y)`: 这将设置画笔原点在 `x`（*x* 轴）和 `y`（*y* 轴）位置。

`painter.setClipping(bool)`: 如果参数是 `True`，则启用剪辑。

`painter.setClipPath(QtGui.QPainterPath, QtCore.Qt.ClipOperation)`: 这将为这个画家设置剪辑路径为路径（第一个参数）和剪辑操作（第二个参数）。

`painter.setClipRect(QtCore.QRect, QtCore.Qt.ClipOperation)`: 这将通过使用剪辑操作（第二个参数）将剪辑区域设置为具有整数值的矩形（第一个参数）。

`painter.setClipRect(QtCore.QRectF, QtCore.Qt.ClipOperation)`: 这将通过使用剪辑操作（第二个参数）将剪辑区域设置为具有浮点值的矩形（第一个参数）。

`painter.setClipRect(QtGui.QRegion, QtCore.Qt.ClipOperation)`: 这将通过使用剪辑操作（第二个参数）将剪辑区域设置为第一个参数中指定的区域。

`painter.setClipRect(x, y, w, h, QtCore.Qt.ClipOperation)`: 这将通过使用剪辑操作（第五个参数）将剪辑区域设置为从 `x`（*x* 轴）和 `y`（*y* 轴）开始的矩形，以及 `w`（宽度）和 `h`（高度）。

`painter.setCompositionMode(QtGui.QPainter.CompositionMode)`: 这将设置参数中指定的合成模式。

`painter.setFont(QtGui.QFont)`: 这将设置画家的字体为参数中指定的字体。

`painter.setLayoutDirection(QtCore.Qt.LayoutDirection)`: 这将设置绘制文本的布局方向。

`painter.setOpacity(float)`: 这将设置透明度，指定在参数中（`0.0`：完全透明，`1.0`：完全不透明），将与这个画家一起使用。

`painter.setPen(QtGui.QPen)`: 这将设置参数中指定的笔到这个画家。

`painter.setPen(QtGui.QColor)`: 这将设置具有参数中指定的颜色的笔到这个画家。

`painter.setPen(QtCore.Qt.PenStyle)`: 这将设置具有参数中指定的样式和黑色颜色的笔到这个画家。

`painter.setRenderHint(QtGui.QPainter.RenderHint, bool)`: 如果第二个参数是 `True`，则将渲染提示（第一个参数）设置为画家。可用的提示如下：

+   `QtGui.QPainter.Antialiasing`: 如果可能，将抗锯齿原始图形的边缘。

+   `QtGui.QPainter.TextAntialiasing`: 如果可能，将抗锯齿文本。

+   `QtGui.QPainter.SmoothPixmapTransform`: 引擎将使用平滑的位图变换。

+   `QtGui.QPainter.Qt4CompatiblePainting`: 引擎将使用与 Qt4 相同的填充规则。

`painter.setRenderHints(QtGui.QPainter.RenderHint | QtGui.QPainter.RenderHint, bool)`: 如果第二个参数是 `True`，则将渲染提示（第一个参数）设置为画家。

`painter.setTransform(QtGui.QTransform, bool)`: 这设置由第一个参数指定的变换矩阵。如果第二个参数为 `True`，则变换将与当前矩阵组合；否则，它将替换当前矩阵。

`painter.setViewTransformEnabled(bool)`: 如果参数为 `True`，则设置视图变换为启用。

`painter.setViewport(QtCore.QRect)`: 这将画笔的视口矩形设置为参数中指定的矩形。视口表示设备的坐标系。

`painter.setViewport(x, y, w, h)`: 这将画笔的视口矩形设置为指定的矩形，从 `x`（*x* 轴）和 `y`（*y* 轴）开始，以及 `w`（宽度）和 `h`（高度）。

`painter.setWindow(QtCore.QRect)`: 这将画笔的窗口设置为参数中指定的矩形。

`painter.setWindow(x, y, w, h)`: 这将画笔的窗口设置为指定的矩形，从 `x`（*x* 轴）和 `y`（*y* 轴）开始，以及 `w`（宽度）和 `h`（高度）。

`painter.setWorldMatrixEnabled(bool)`: 如果参数为 `True`，则设置世界变换为启用。

`painter.setWorldTransform(QtGui.QTransform, bool)`: 这设置由第一个参数指定的世界变换矩阵。如果第二个参数为 `True`，则变换将与当前矩阵组合；否则，它将替换当前矩阵。

# has 和 is

这些是返回与画笔相关的布尔值 (`bool`) 的函数：

`painter.hasClipping()`: 如果此画笔设置了裁剪，则返回 `True`。

`painter.isActive()`: 如果画笔是活动的并且已调用 `begin()`，则返回 `True`，而 `end()` 尚未调用。

# functional

这些是与当前画笔值的返回相关的函数：

`painter.background()`: 这返回用于背景的当前 `QtGui.QBrush` 类型的画笔。

`painter.backgroundMode()`: 这返回用于背景的当前 `QtCore.Qt.BGMode` 类型的模式。

`painter.begin(QtGui.QPaintDevice)`: 这使用参数中指定的绘图设备开始绘制，如果绘制成功则返回 `True`。

`painter.beginNativePainting()`: 这刷新绘图管道并准备对底层图形上下文进行绘制。

`painter.boundingRect(QtCore.QRectF, QtCore.Qt.Alignment, str)`: 这返回 `QtCore.QRectF` 类型的边界矩形，其中包含文本（第三个参数）、矩形（第一个参数）和标志（第二个参数），如它们在绘图中所出现。

`painter.brush()`: 这返回与该画笔一起使用的当前 `QtGui.QBrush` 类型的画笔。

`painter.brushOrigin()`: 这返回 `QtCore.QPoint` 类型的画笔原点。

`painter.clipBoundingRect()`: 如果存在裁剪，则返回当前裁剪的 `QtCore.QRectF` 类型的边界矩形。

`painter.combinedTransform()`: 这返回组合当前世界变换和窗口/视口的变换矩阵。

`painter.compositionMode()`: 这返回当前使用的 `QtGui.QPainter.CompositionMode` 类型的合成模式。

`painter.device()`: 这返回用于绘图的当前 `QtGui.QPaintDevice` 类型的绘图设备。

`painter.deviceTransform()`: 这返回将逻辑坐标转换为平台相关绘图设备的设备坐标的矩阵。

`painter.end()`: 这将结束绘图操作，如果绘图器结束操作并且不再活跃，则返回 `True`。

`painter.endNativePainting()`: 这在发出原生绘图命令后恢复绘图器。

`painter.eraseRect(QtCore.QRect)`: 这擦除参数中指定的整数值指定的矩形内的区域。

`painter.eraseRect(QtCore.QRectF)`: 这使用参数中指定的浮点值擦除矩形内的区域。

`painter.eraseRect(x, y, w, h)`: 这擦除从 `x`（*x* 轴）和 `y`（*y* 轴）开始的矩形内的区域，以及 `w`（宽度）和 `h`（高度）。

`painter.font()`: 这返回使用此绘图器绘制文本的当前字体。

`painter.fontInfo()`: 这返回与该绘图器一起使用的字体的字体信息。

`painter.fontMetrics()`: 这返回与该绘图器一起使用的字体的字体度量。

`painter.layoutDirection()`: 这返回与该绘图器一起用于绘制文本的 `QtCore.Qt.LayoutDirection` 类型的布局方向。

`painter.opacity()`: 这返回与该绘图器一起使用的透明度。

`painter.paintEngine()`: 这返回此绘图器正在操作的当前 `QtGui.QPaintEngine` 类型的绘图引擎。

`painter.pen()`: 这返回与该绘图器一起使用的当前 `QtGui.QPen` 类型的笔。

`painter.renderHints()`: 这返回用于此绘图器绘图的 `QtGui.QPainter.RenderHints` 类型的渲染提示。

`painter.resetTransform()`: 这重置使用此绘图器的 `translate()`、`scale()`、`shear()`、`rotate()`、`setWorldTransform()`、`setViewport()` 和 `setWindow()` 函数所做的变换。

`painter.restore()`: 这恢复此绘图器的当前状态。

`painter.rotate(float)`: 这以角度参数按顺时针方向旋转此绘图器的坐标系。

`painter.save()`: 这保存此绘图器的当前状态。

`painter.scale(x, y)`: 这通过 `x`（*x* 轴）和 `y`（*y* 轴）值缩放此绘图器的坐标系，这些值是浮点数。

`painter.shear(float, float)`: 这根据指定的参数剪切坐标系。

`painter.strokePath(QtGui.QPainterPath, QtGui.QPen)`: 这使用第二个参数指定的笔来绘制第一个参数指定的路径的轮廓。

`painter.testRenderHint(QtGui.QPainter.RenderHint)`: 如果参数中指定的提示设置为该画家，则返回 `True`。

`painter.transform()`: 这返回 `QtGui.QTransform` 类型的变换矩阵。

`painter.translate(QtCore.QPoint)`: 这将在参数中指定的点（以整数值指定）处平移坐标系。

`painter.translate(QtCore.QPointF)`: 这将在参数中指定的点（以浮点值指定）处平移坐标系。

`painter.translate(float, float)`: 这将根据指定的向量平移坐标系。

`painter.viewTransformEnabled()`: 如果此画家的视图变换被启用，则返回 `True`。

`painter.viewport()`: 这返回 `QtCore.QRect` 类型的视口矩形。

`painter.window()`: 这返回 `QtCore.QRect` 类型的窗口矩形。

`painter.worldMatrixEnabled()`: 如果此画家的世界变换被启用，则返回 `True`。

`painter.worldTransform()`: 这返回 `QtGui.QTransform` 类型的世界变换矩阵。

# draw

这些是与该画家相关的绘图操作函数：

`painter.drawArc(QtCore.QRect, int, int)`: 这将使用参数中指定的整数值（第一个参数）和起始角度（第二个参数）以及跨度角度（第三个参数）绘制圆弧。

`painter.drawArc(QtCore.QRectF, int, int)`: 这将使用参数中指定的浮点值（第一个参数）和起始角度（第二个参数）以及跨度角度（第三个参数）绘制圆弧。

`painter.drawArc(x, y, w, h, int, int)`: 这将使用指定的参数绘制圆弧——一个以 `x`（*x* 轴）和 `y`（*y* 轴）开始的矩形；具有 `w`（宽度）和 `h`（高度）；起始角度（第五个参数）和跨度角度（第六个参数）。

`painter.drawChord(QtCore.QRect, int, int)`: 这将使用参数中指定的整数值（第一个参数），起始角度（第二个参数）和跨度角度（第三个参数）绘制圆弧。

`painter.drawChord(QtCore.QRectF, int, int)`: 这将使用参数中指定的浮点值（第一个参数），起始角度（第二个参数）和跨度角度（第三个参数）绘制圆弧。

`painter.drawChord(x, y, w, h, int, int)`: 这将使用指定的参数绘制圆弧——一个以 `x`（*x* 轴）和 `y`（*y* 轴）开始的矩形；具有 `w`（宽度）和 `h`（高度）；起始角度（第五个参数）和跨度角度（第六个参数）。

`painter.drawConvexPolygon(QtGui.QPolygon)`: 这将使用参数中指定的多边形绘制凸多边形。

`painter.drawEllipse(QtCore.QRect)`: 这将使用参数中指定的整数值绘制定义的椭圆。

`painter.drawEllipse(QtCore.QRectF)`: 这将使用参数中指定的浮点值绘制定义的椭圆。

`painter.drawEllipse(QtCore.QPoint, int, int)`: 这将在中心位置以整数值（第一个参数）绘制椭圆，半径 *x*（第二个参数）和半径 *y*（第三个参数）。

`painter.drawEllipse(QtCore.QPointF, int, int)`: 这将在中心位置以浮点值（第一个参数）绘制椭圆，半径 *x*（第二个参数）和半径 *y*（第三个参数）。

`painter.drawEllipse(x, y, w, h)`: 这将在指定参数的椭圆上绘制——从 `x`（*x* 轴）和 `y`（*y* 轴）开始的矩形，宽度为 `w`，高度为 `h`。

`painter.drawImage(QtCore.QRect, QtGui.QImage)`: 这将在具有整数值（第一个参数）的矩形中绘制第二个参数指定的图像。

`painter.drawImage(QtCore.QRectF, QtGui.QImage)`: 这将在具有浮点值（第一个参数）的矩形中绘制第二个参数指定的图像。

`painter.drawImage(QtCore.QPoint, QtGui.QImage)`: 这将在具有整数值（第一个参数）的点处绘制第二个参数指定的图像。

`painter.drawImage(QtCore.QPointF, QtGui.QImage)`: 这将在具有浮点值（第一个参数）的点处绘制第二个参数指定的图像。

`painter.drawImage(QtCore.QRect, QtGui.QImage, QtCore.QRect, QtCore.Qt.ImageConversionFlags)`: 这将在具有整数值（第三个参数）的图像（第二个参数）的矩形源上绘制矩形，该矩形位于具有整数值（第一个参数）的画布设备中的矩形内，并带有标志（第四个参数）。

`painter.drawImage(QtCore.QPoint, QtGui.QImage, QtCore.QRect, QtCore.Qt.ImageConversionFlags)`: 这将在具有整数值（第三个参数）的图像（第二个参数）的矩形源上绘制浮点值（第一个参数）的点，并带有标志（第四个参数）。

`painter.drawImage(QtCore.QPointF, QtGui.QImage, QtCore.QRectF, QtCore.Qt.ImageConversionFlags)`: 这将在具有浮点值（第三个参数）的图像（第二个参数）的矩形源上绘制浮点值（第一个参数）的点，并带有标志（第四个参数）。

`painter.drawImage(QtCore.QRectF, QtGui.QImage, QtCore.QRectF, QtCore.Qt.ImageConversionFlags)`: 这将在具有浮点值（第三个参数）的图像（第二个参数）的矩形源上绘制矩形，该矩形位于具有浮点值（第一个参数）的画布设备中的矩形内，并带有标志（第四个参数）。

`painter.drawImage(x, y, QtGui.QImage, tx, ly, w, h, QtCore.Qt.ImageConversionFlags)`: 这将在位置绘制图像（第二个参数），从 `x`（*x* 轴）和 `y`（*y* 轴）开始；在图像中具有 `tx`（顶部）和 `ly`（左侧）点，宽度为 `w`，高度为 `h`，并带有标志（第八个参数）的画布设备中。

`painter.drawLine(QtCore.QLine)`: 这将绘制由参数中指定的具有整数值的线。

`painter.drawLine(QtCore.QLineF)`: 这条语句根据参数中指定的具有浮点值的线绘制一条线。

`painter.drawLine(QtCore.QPoint, QtCore.QPoint)`: 这条语句从具有整数值的点（第一个参数）绘制一条线到具有整数值的点（第二个参数）。

`painter.drawLine(QtCore.QPointF, QtCore.QPointF)`: 这条语句从具有浮点值的点（第一个参数）绘制一条线到具有浮点值的点（第二个参数）。

`painter.drawLine(x1, y1, x2, y2)`: 这条语句从 `x1`（x 轴）和 `y1`（y 轴）到 `x2`（x 轴）和 `y2`（y 轴）绘制一条线。

`painter.drawLines([QtCore.QLine])`: 这条语句绘制列表中指定的具有整数值的线。

`painter.drawLines([QtCore.QLineF])`: 这条语句绘制列表中指定的具有浮点值的线。

`painter.drawPath(QtGui.QPainterPath)`: 这条语句绘制参数中指定的路径。

`painter.drawPicture(QtCore.QPoint, QtGui.QPicture)`: 这条语句在具有整数值的点上绘制一个图片（第二个参数）。

`painter.drawPicture(QtCore.QPointF, QtGui.QPicture)`: 这条语句在具有浮点值的点上绘制一个图片（第二个参数）。

`painter.drawPicture(x, y, QPicture)`: 这条语句在由 `x`（x 轴）和 `y`（y 轴）指定的点上绘制一个图片（第三个参数）。

`painter.drawPie(QtCore.QRect, int, int)`: 这条语句通过矩形（第一个参数）和整数值（起始角度的第二个参数和跨度角度的第三个参数）绘制一个饼图。

`painter.drawPie(QtCore.QRectF, int, int)`: 这条语句通过具有浮点值的矩形（第一个参数）和起始角度（第二个参数）以及跨度角度（第三个参数）绘制一个饼图。

`painter.drawPie(x, y, w, h, int, int)`: 这条语句使用指定的参数绘制一个饼图——一个从 `x`（x 轴）和 `y`（y 轴）开始的矩形；具有 `w`（宽度）和 `h`（高度）；以及起始角度（第五个参数）和跨度角度（第六个参数）。

`painter.drawPixmap(QtCore.QRect, QtGui.QPixmap)`: 这条语句在具有整数值的矩形内绘制一个位图（第二个参数）。

`painter.drawPixmap(QtCore.QPoint, QtGui.QPixmap)`: 这条语句在具有整数值的点上绘制一个位图（第二个参数）。

`painter.drawPixmap(QtCore.QPointF, QtGui.QPixmap)`: 这条语句在具有浮点值的点上绘制一个位图（第二个参数）。

`painter.drawPixmap(QtCore.QRect, QtGui.QPixmap, QtCore.QRect)`: 这条语句在画布中绘制位图的矩形部分（第三个参数），该部分位于具有整数值的矩形（第一个参数）内。

`painter.drawPixmap(QtCore.QRectF, QtGui.QPixmap, QtCore.QRectF)`: 这将在画布设备中，以浮点值（第三个参数）的矩形部分（第二个参数）在具有浮点值（第一个参数）的矩形中绘制位图。

`painter.drawPixmap(QtCore.QPoint, QtGui.QPixmap, QtCore.QRect)`: 这将在画布设备中，以整数值（第三个参数）的矩形部分（第二个参数）在具有整数值（第一个参数）的点处绘制位图。

`painter.drawPixmap(QtCore.QPointF, QtGui.QPixmap, QtCore.QRectF)`: 这将在画布设备中，以浮点值（第三个参数）的矩形部分（第二个参数）在具有浮点值（第一个参数）的点处绘制位图。

`painter.drawPixmap(x, y, QtGui.QPixmap)`: 这将在根据 `x`（x 轴）和 `y`（y 轴）的位置开始绘制指定在第三个参数中的位图。

`painter.drawPixmap(x, y, QtGui.QPixmap, tx, ly, w, h)`: 这将在指定参数的情况下绘制位图（第三个参数），从 `x`（x 轴）和 `y`（y 轴）开始；使用 `tx`（x-top）和 `ly`（y-left）点；大小为 `w`（宽度）和 `h`（高度）。

`painter.drawPixmap(x, y, w, h, QtGui.QPixmap, tx, ly, pw, ph)`: 这将在指定参数的情况下绘制位图（第五个参数），从 `x`（x 轴）和 `y`（y 轴）开始；大小为 `w`（宽度）和 `h`（高度）；使用 `tx`（x-top）；`ly`（y-left）；`pw`（宽度）和 `ph`（高度）。

`painter.drawPixmap(x, y, w, h, QtGui.QPixmap)`: 这将在根据 `x`（x 轴）和 `y`（y 轴）的位置，大小为 `w`（宽度）和 `h`（高度）的矩形中绘制位图（第五个参数）。

`painter.drawPixmapFragments([QtGui.QPainter.PixmapFragment], int, QtGui.QPixmap, QtGui.QPainter.PixmapFragmentHints)`: 这将在多个位置（片段）绘制位图（第三个参数），具有不同的缩放、旋转和透明度。片段（第一个参数）是用于绘制每个位图片段的片段计数（第二个参数）的元素数组，以及构成绘图提示的提示（第四个参数）。

`painter.drawPoint(QtCore.QPoint)`: 这将在指定在点参数中的整数值位置绘制一个点。

`painter.drawPointF(QtCore.QPoint)`: 这将在指定在点参数中的浮点值位置绘制一个点。

`painter.drawPoint(x, y)`: 这将在指定 `x`（x 轴）和 `y`（y 轴）的位置绘制一个点。

`painter.drawPoints(QtGui.QPolygon)`: 这将根据向量中的点以整数值绘制多边形。

`painter.drawPoints(QtGui.QPolygonF)`: 这将根据向量中的点以浮点值绘制多边形。

`painter.drawPolygon(QtGui.QPolygon, QtCore.Qt.FillRule)`: 这将根据多边形中指定的整数值点绘制多边形，并使用规则（第二个参数）。

`painter.drawPolygon(QtGui.QPolygonF, QtCore.Qt.FillRule)`: 这将根据多边形中指定的浮点值点绘制多边形（第一个参数），并且有规则（第二个参数）。

`painter.drawPolyline(QtGui.QPolygon, QtCore.Qt.FillRule)`: 这将根据多边形中指定的整数值点绘制折线。

`painter.drawPolyline(QtGui.QPolygonF, QtCore.Qt.FillRule)`: 这将根据多边形中指定的浮点值点绘制折线。

`painter.drawRect(QtCore.QRect)`: 这将绘制由整数值指定的参数指定的矩形。

`painter.drawRect(QtCore.QRectF)`: 这将绘制由浮点值指定的参数指定的矩形。

`painter.drawRect(x, y, w, h)`: 这将绘制从左上角开始，由 `x`（*x* 轴）和 `y`（*y* 轴）指定的矩形，宽度为 `w`，高度为 `h`。

`painter.drawRects(QtCore.QRect, QtCore.QRect...)`: 这将绘制由整数值指定的参数指定的矩形。

`painter.drawRects([QtCore.QRectF])`: 这将绘制由浮点值指定的参数指定的矩形。

`painter.drawRoundRect(QtCore.QRect, rx, ry)`: 这将绘制具有圆角的矩形，圆角的大小由 `rx` 和 `ry`（矩形的圆角）指定。

`painter.drawRoundRect(QtCore.QRectF, rx, ry)`: 这将绘制具有圆角的矩形，圆角的大小由 `rx` 和 `ry`（矩形的圆角）指定，并且矩形的大小由浮点值指定（第一个参数）。

`painter.drawRoundRect(x, y, w, h, rx, ry)`: 这将绘制具有圆角的矩形，圆角的大小由 `rx` 和 `ry`（矩形的圆角）指定，并且从 `x`（*x* 轴）和 `y`（*y* 轴）的起点开始；宽度为 `w`，高度为 `h`。

`painter.drawRoundedRect(QtCore.QRect, rx, ry, QtCore.Qt.SizeMode)`: 这将绘制具有圆角的矩形，圆角的大小由 `rx` 和 `ry`（矩形的圆角）指定，并且有一个模式（第四个参数），矩形的大小由整数值指定（第一个参数）。

`painter.drawRoundedRect(QtCore.QRectF, rx, ry, QtCore.Qt.SizeMode)`: 这将绘制具有圆角的矩形，圆角的大小由 `rx` 和 `ry`（矩形的圆角）指定，并且有一个模式（第四个参数），矩形的大小由浮点值指定（第一个参数）。

`painter.drawRoundedRect(x, y, w, h, rx, ry, QtCore.Qt.SizeMode)`: 这将绘制具有圆角的矩形，圆角的大小由 `rx` 和 `ry`（矩形的圆角）指定，并且有一个模式（第七个参数），矩形的大小由参数指定，从点 `x`（*X* 轴）和 `y`（*Y* 轴）开始；宽度为 `w`，高度为 `h`。

`painter.drawStaticText(QtCore.QPoint, QtGui.QStaticText)`: 这将在具有整数值（第一个参数）的点处绘制静态文本（第二个参数）。

`painter.drawStaticText(QtCore.QPointF, QtGui.QStaticText)`: 这将在具有浮点值（第一个参数）的点绘制静态文本（第二个参数）。

`painter.drawStaticText(int, int, QStaticText)`: 这将在左（第一个参数）和顶（第二个参数）坐标绘制静态文本（第三个参数）。

`painter.drawText(QtCore.QRect, flags, str)`: 这将在提供的矩形内绘制文本（第三个参数），矩形由整数值（第一个参数）和标志（第二个参数）定义。可用的标志可以用 OR（`|`）组合，如下所示：

+   `QtCore.Qt.AlignLeft`: 左对齐。

+   `QtCore.Qt.AlignRight`: 右对齐。

+   `QtCore.Qt.AlignHCenter`: 水平居中对齐。

+   `QtCore.Qt.AlignJustify`: 段落对齐。

+   `QtCore.Qt.AlignTop`: 顶部对齐。

+   `QtCore.Qt.AlignBottom`: 底部对齐。

+   `QtCore.Qt.AlignVCenter`: 垂直居中对齐。

+   `QtCore.Qt.AlignCenter`: 居中对齐。

+   `QtCore.Qt.TextDontClip`: 文本不会裁剪。

+   `QtCore.Qt.TextSingleLine`: 文本单行。

+   `QtCore.Qt.TextExpandTabs`: 这使得 ASCII 制表符移动到下一个停止位置。

+   `QtCore.Qt.TextShowMnemonic`: 显示如`"&P"`之类的字符串为`P`。

+   `QtCore.Qt.TextWordWrap`: 按单词换行。

+   `QtCore.Qt.TextIncludeTrailingSpaces`: 如果使用此选项，`naturalTextWidth()` 和 `naturalTextRect()` 将返回一个包含文本末尾空格宽度的值；否则，它将被排除。

`painter.drawText(QtCore.QRectF, flags, str)`: 这将在提供的矩形内绘制文本（第三个参数），矩形由浮点值（第一个参数）和标志（第二个参数）定义。

`painter.drawText(x, y, str)`: 这将在根据`x`（*x*轴）和`y`（*y*轴）的位置绘制文本（第三个参数）。

`painter.drawText(x, y, w, h, flags, str)`: 这将在根据`x`（*x*轴）和`y`（*y*轴）的位置绘制文本（第六个参数），以及`w`（宽度）和`h`（高度）和标志（第五个参数）。

`painter.drawText(QtCore.QRectF, str, QtGui.QTextOption)`: 这将在具有浮点值（第一个参数）的矩形内绘制文本（第二个参数），并带有文本选项（第三个参数）。

`painter.drawText(QtCore.QPoint, str)`: 这将在具有整数值（第一个参数）的点绘制文本（第二个参数）。

`painter.drawText(QtCore.QPointF, str)`: 这将在具有浮点值（第一个参数）的点绘制文本（第二个参数）。

`painter.drawTextItem(QtCore.QPoint, QtGui.QTextItem)`: 这将在具有整数值（第一个参数）的点绘制文本项（第二个参数）。

`painter.drawTextItem(QtCore.QPointF, QtGui.QTextItem)`: 这将在具有浮点值（第一个参数）的点绘制文本项（第二个参数）。

`painter.drawTextItem(x, y, QtGui.QTextItem)`: 这将在根据`x`（*x*轴）和`y`（*y*轴）的位置绘制文本项（第三个参数）。

`painter.drawTiledPixmap(QtCore.QRect, QtGui.QPixmap, QtCore.QPoint)`: 这将在具有整数值的矩形（第一个参数）中绘制平铺的位图（第二个参数），从具有整数值的点（第三个参数）开始。

`painter.drawTiledPixmap(QtCore.QRectF, QtGui.QPixmap, QtCore.QPointF)`: 这将在具有浮点值的矩形（第一个参数）中绘制平铺的位图（第二个参数），从具有浮点值的点（第三个参数）开始。

`painter.drawTiledPixmap(x, y, w, h, QtGui.QPixmap, tx, ly)`: 这将在指定的矩形中绘制平铺的位图（第五个参数），从 `x`（*x* 轴）和 `y`（*y* 轴）开始；宽度为 `w`，高度为 `h`；并且以位图中的 `tx`（顶部）和 `ly`（左侧）点为起点。

# fill

这些是与使用此画家填充相关的函数：

`painter.fillPath(QtGui.QPainterPath, QtGui.QBrush)`: 这将使用画笔（第二个参数）填充第一个参数指定的路径。

`painter.fillRect(QtCore.QRect, QtCore.Qt.GlobalColor)`: 这将使用第二个参数指定的颜色填充由整数值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRectF, QtCore.Qt.GlobalColor)`: 这将使用第二个参数指定的颜色填充由浮点值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRect, QtGui.QColor)`: 这将使用第二个参数指定的颜色填充由整数值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRectF, QtGui.QColor)`: 这将使用第二个参数指定的颜色填充由浮点值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRect, QtGui.QBrush)`: 这将使用第二个参数指定的画笔填充由整数值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRectF, QtGui.QBrush)`: 这将使用第二个参数指定的画笔填充由浮点值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRect, QtGui.QGradient.Preset)`: 这将使用第二个参数指定的预设渐变填充由整数值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRectF, QtGui.QGradient.Preset)`: 这将使用第二个参数指定的预设渐变填充由浮点值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRect, QtCore.Qt.BrushStyle)`: 这将使用第二个参数指定的画笔样式填充由整数值表示的矩形（第一个参数）。

`painter.fillRect(QtCore.QRectF, QtCore.Qt.BrushStyle)`: 这将使用第二个参数指定的画笔样式填充由浮点值表示的矩形（第一个参数）。

`painter.fillRect(x, y, w, h, QtGui.QGradient.Preset)`：这用指定的预设渐变填充矩形，从`x`（*x*轴）和`y`（*y*轴）开始；使用`w`（宽度）和`h`（高度），在第五个参数指定的预设渐变中。

`painter.fillRect(x, y, w, h, QtCore.Qt.BrushStyle)`：这用指定的画刷样式填充矩形，从`x`（*x*轴）和`y`（*y*轴）开始；使用`w`（宽度）和`h`（高度），在第五个参数指定的画刷样式中。

`painter.fillRect(x, y, w, h, QtGui.QColor)`：这用指定的颜色填充矩形，从`x`（*x*轴）和`y`（*y*轴）开始；使用`w`（宽度）和`h`（高度），在第五个参数指定的颜色中。

`painter.fillRect(x, y, w, h, QtGui.QBrush)`：这用指定的画刷填充矩形，从`x`（*x*轴）和`y`（*y*轴）开始；使用`w`（宽度）和`h`（高度），在第五个参数指定的画刷中。

`painter.fillRect(x, y, w, h, QtCore.Qt.GlobalColor)`：这用指定的颜色填充矩形，从`x`（*x*轴）和`y`（*y*轴）开始；使用`w`（宽度）和`h`（高度），在第五个参数指定的颜色中。

# QPen

此类通过`QPainter`类提供了绘制形状和线条轮廓的方法。笔的声明如下：

```py
pen = QtGui.QPen()
```

`QPen`类通过以下函数提高了功能。

# set

这些是设置笔的参数和属性的函数：

`pen.setBrush(QtGui.QBrush)`：这设置了参数中指定的画刷，该画刷将用于填充线条。

`pen.setCapStyle(QtCore.Qt.PenCapStyle)`：这设置了笔的帽样式为参数中指定的样式。可用的样式有`QtCore.Qt.SquareCap`、`QtCore.Qt.FlatCap`和`QtCore.Qt.RoundCap`。

`pen.setColor(QtGui.QColor)`：这设置了此笔的画刷颜色为参数中指定的颜色。

`pen.setCosmetic(bool)`：如果参数为`True`，则将此笔设置为装饰性。装饰性笔以恒定宽度绘制线条，不受任何变换的影响。用装饰性笔绘制的形状将确保轮廓具有相同的厚度。

`pen.setDashOffset(float)`：这设置了参数中指定的虚线偏移。

`pen.setDashPattern([float])`：这设置了参数中指定的虚线模式，作为浮点值的可迭代对象，必须是`[1.0, 2.0, 3.0, 4.0]`，其中`1.0`和`3.0`是虚线，`2.0`和`4.0`是空格。

`pen.setJoinStyle(QtCore.Qt.PenJoinStyle)`：这设置了笔的连接样式为参数中指定的样式。可用的样式有`QtCore.Qt.BevelJoin`、`QtCore.Qt.MiterJoin`和`QtCore.Qt.RoundJoin`。

`pen.setMiterLimit(float)`：这设置了笔的斜接限制。

`pen.setStyle(QtCore.Qt.PenStyle)`：这设置了此笔的样式。可用的样式如下：

+   `QtCore.Qt.SolidLine`：实线。

+   `QtCore.Qt.DashLine`：分离的虚线。

+   `QtCore.Qt.DotLine`：分离的点。

+   `QtCore.Qt.DashDotLine`：点划线。

+   `QtCore.Qt.DashDotDotLine`: 一个破折号，两个点，一个破折号，两个点。

+   `QtCore.Qt.CustomDashLine`: 将是使用 `setDashPattern()` 定义的定制模式。

+   `QtCore.Qt.NoPen`: 无线条。

`pen.setWidth(int)`: 这将参数中指定的宽度作为整数值设置，并用作画笔的宽度。

`pen.setWidth(float)`: 这将参数中指定的宽度作为浮点值设置，并用作画笔的宽度。

# 是

这些是与画笔相关的函数，返回一个布尔值（`bool`）：

`pen.isCosmetic()`: 如果此画笔是装饰性的，则返回 `True`。

`pen.isSolid()`: 如果这是一个实心填充画笔，则返回 `True`。

# 功能性

这些是与当前画笔值的返回值相关的函数：

`pen.brush()`: 这返回 `QtGui.QBrush` 类型的画笔，并用于填充线条。

`pen.capStyle()`: 这返回与画笔一起使用的 `QtCore.Qt.PenCapStyle` 类型的端点样式。

`pen.color()`: 这返回与画笔的刷子一起使用的 `QtGui.QColor` 类型的颜色。

`pen.dashOffset()`: 这返回画笔的虚线偏移量。

`pen.dashPattern()`: 这返回画笔的虚线模式。

`pen.joinStyle()`: 这返回 `QtCore.Qt.PenJoinStyle` 类型的连接样式，如与该画笔一起使用。

`pen.miterLimit()`: 这返回画笔的斜接限制。

`pen.style()`: 这返回此画笔的样式。

`pen.swap(QtGui.QPen)`: 这将参数中指定的画笔与此画笔交换。

`pen.width()`: 这返回此画笔的宽度作为整数值。

`pen.widthF()`: 这返回此画笔的宽度作为浮点值。

# QBrush

画笔描述了使用 `QPainter` 类绘制的形状的填充模式。画笔的声明如下：

```py
brush = QtGui.QBrush()
```

`QBrush` 类通过以下函数提高了功能性。

# 设置

这些是与设置画笔相关参数和属性的函数：

`brush.setColor(QtGui.QColor)`: 这将设置参数中指定的此画笔的颜色。

`brush.setColor(QtCore.Qt.GlobalColor)`: 这将设置此画笔参数中指定的全局颜色。

`brush.setStyle(QtCore.Qt.BrushStyle)`: 这将设置参数中指定的此画笔的样式。可用的画笔样式如下：

+   `QtCore.Qt.NoBrush`—`0`: 无画笔。

+   `QtCore.Qt.SolidPattern`—`1`: 均匀画笔。

+   `QtCore.Qt.Dense1Pattern`—`2`: 非常密集的画笔。

+   `QtCore.Qt.Dense2Pattern`—`3`: 非常密集的画笔。

+   `QtCore.Qt.Dense3Pattern`—`4`: 有点密集的画笔。

+   `QtCore.Qt.Dense4Pattern`—`5`: 半密集的画笔。

+   `QtCore.Qt.Dense5Pattern`—`6`: 有点稀疏的画笔。

+   `QtCore.Qt.Dense6Pattern`—`7`: 非常稀疏的画笔。

+   `QtCore.Qt.Dense7Pattern`—`8`: 非常稀疏的画笔。

+   `QtCore.Qt.HorPattern`—`9`: 水平。

+   `QtCore.Qt.VerPattern`—`10`: 垂直。

+   `QtCore.Qt.CrossPattern`—`11`: 横向/纵向交叉。

+   `QtCore.Qt.BDiagPattern`—`12`: 反向对角。

+   `QtCore.Qt.FDiagPattern`—`13`: 正向对角。

+   `QtCore.Qt.DiagCrossPattern`—`14`: 对角交叉。

+   `QtCore.Qt.LinearGradientPattern`—`15`: 线性渐变画笔。

+   `QtCore.Qt.RadialGradientPattern`—`16`: 径向渐变画笔。

+   `QtCore.Qt.ConicalGradientPattern`—`17`: 锥形渐变画笔。

+   `QtCore.Qt.TexturePattern`—`24`: 自定义。

`brush.setTexture(QtGui.QPixmap)`: 这将参数中指定的位图设置为画笔位图。

`brush.setTextureImage(QtGui.QImage)`: 这将参数中指定的图像设置为画笔图像。

`brush.setTransform(QtGui.QTransform)`: 这将参数中指定的矩阵作为显式变换矩阵设置在画笔上。

# 是

此函数返回与画笔相关的布尔值 (`bool`)：

`brush.isOpaque()`: 如果画笔完全不透明，则返回 `True`。

# 功能性

这些是与画笔当前值返回相关的函数：

`brush.color()`: 这返回与该画笔一起使用的 `QtGui.QColor` 类型的颜色。

`brush.gradient()`: 这返回与该画笔一起使用的 `QtGui.QGradient` 类型的渐变。

`brush.style()`: 这返回与该画笔一起使用的 `QtCore.Qt.BrushStyle` 类型的样式。

`brush.swap(QtGui.QBrush)`: 这将此画笔与参数中指定的画笔交换。

`brush.texture()`: 这返回 `QtGui.QPixmap` 类型的自定义画笔图案。

`brush.textureImage()`: 这返回 `QtGui.QImage` 类型的自定义画笔图案。

`brush.transform()`: 这返回与该画笔一起使用的 `QtGui.QTransform` 类型的当前变换矩阵。

# QGradient

此类与 `QBrush` 风格一起使用，用于在图形创建中实现简单的渐变。声明语法如下：

```py
gradient = QtGui.QGradient()
```

`QGradient` 类通过以下函数提高功能。

# 设置

这些是与设置与渐变相关的参数和属性相关的函数：

`gradient.setColorAt(float, QtGui.QColor)`: 这在第一个参数指定的位置创建一个停止点（`0.0`–`1.0`），第二个参数指定颜色。

`gradient.setCoordinateMode(QtGui.QGradient.CoordinateMode)`: 这设置此渐变的参数中指定的坐标模式。可用的参数如下：

+   `QtGui.QGradient.LogicalMode`—`0`: 坐标在逻辑空间中指定。

+   `QtGui.QGradient.StretchToDeviceMode`—`1`: 坐标相对于绘图设备的矩形。

+   `QtGui.QGradient.ObjectMode`—`3`: 坐标相对于对象的矩形。

`gradient.setSpread(QtGui.QGradient.Spread)`: 这设置此渐变的参数中指定的扩散方法。可用的扩散方法如下：

+   `QtGui.QGradient.PadSpread`—`0`: 用最接近的停止颜色填充。

+   `QtGui.QGradient.ReflectSpread`—`1`: 在渐变区域外反射。

+   `QtGui.QGradient.RepeatSpread`—`2`: 在渐变区域外重复。

`gradient.setStops([float])`: 这用参数中指定的停止点替换当前设置的停止点。点必须按从最低点开始的顺序排序，并且在 `0.0` 到 `1.0` 的范围内。

# functional

这些是与梯度当前值返回相关的函数：

`gradient.coordinateMode()`: 这返回梯度坐标模式，类型为 `QtGui.QGradient.CoordinateMode`。

`gradient.spread()`: 这返回与该梯度一起使用的 `QtGui.QGradient.Spread` 类型的扩散方法。

`gradient.stops()`: 这返回与该梯度一起使用的停止点。

`gradient.type()`: 这返回梯度类型。类型如下：

+   `QtGui.QGradient.LinearGradient`—`0`: 起点和终点之间的色彩。

+   `QtGui.QGradient.RadialGradient`—`1`: 点与其终点之间的色彩。

+   `QtGui.QGradient.ConicalGradient`—`2`: 中心周围的色彩。

+   `QtGui.QGradient.NoGradient`—`3`: 无渐变。

# QLinearGradient

这是 `QGradient` 类的子类，表示在起点和终点之间填充颜色的线性梯度。声明如下：

```py
linear_gradient = QtGui.QLinearGradient()
```

`QLinearGradient` 从 `QGradient` 类继承，并通过以下函数改进了功能。

# set

这些是与设置与线性梯度相关的参数和属性相关的函数：

`linear_gradient.setFinalStop(QtCore.QPointF)`: 这使用参数中指定的浮点值在逻辑坐标中设置线性梯度的最终停止点。

`linear_gradient.setFinalStop(x, y)`: 这根据线性梯度在逻辑坐标中的`x` (*x* 轴) 和 `y` (*y* 轴) 位置，使用浮点值设置最终停止点。

`linear_gradient.setStart(QtCore.QPointF)`: 这使用参数中指定的浮点值在逻辑坐标中设置线性梯度的起点。

`linear_gradient.setStart(x, y)`: 这根据线性梯度在逻辑坐标中的`x` (*x* 轴) 和 `y` (*y* 轴) 位置，使用浮点值设置起点。

# functional

这些是与线性梯度当前值返回相关的函数：

`linear_gradient.finalStop()`: 这返回线性梯度的最终停止点，类型为 `QtCore.QPointF`，在逻辑坐标中。

`linear_gradient.start()`: 这返回线性梯度的起点，类型为 `QtCore.QPointF`，在逻辑坐标中。

# QRadialGradient

这是 `QGradient` 类的子类，表示在焦点和其端点之间填充颜色的径向梯度。声明如下：

```py
radial_gradient = QtGui.QRadialGradient()
```

`QRadialGradient` 从 `QGradient` 类继承，并通过以下函数改进了功能。

# set

这些是与设置与径向梯度相关的参数和属性相关的函数：

`radial_gradient.setCenter(QtCore.QPointF)`: 这将使用参数中指定的浮点值设置中心点，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.setCenter(x, y)`: 这将根据逻辑坐标中的`x`（*x*轴）和`y`（*y*轴）位置，使用浮点值设置中心点。

`radial_gradient.setCenterRadius(float)`: 这将设置中心半径，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.setFocalPoint(QtCore.QPointF)`: 这将使用参数中指定的浮点值设置焦点，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.setFocalPoint(x, y)`: 这将使用浮点值在`x`（*x*轴）和`y`（*y*轴）位置设置焦点，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.setFocalRadius(float)`: 这将设置参数中指定的焦点半径，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.setRadius(float)`: 这将设置参数中指定的半径，在逻辑坐标中，对于这个径向渐变。

# functional

这些是与径向渐变当前值返回相关的函数：

`radial_gradient.center()`: 这返回`QtCore.QPointF`类型的中心，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.centerRadius()`: 这返回中心半径作为一个浮点值，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.focalPoint()`: 这返回`QtCore.QPointF`类型的焦点，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.focalRadius()`: 这返回焦点半径作为一个浮点值，在逻辑坐标中，对于这个径向渐变。

`radial_gradient.radius()`: 这返回半径作为一个浮点值，在逻辑坐标中，对于这个径向渐变。

# QConicalGradient

这是一个`QGradient`类的子类，表示围绕中心点填充颜色的锥形渐变。声明如下：

```py
conical_gradient = QtGui.QConicalGradient()
```

`QConicalGradient`从`QGradient`类继承，并通过以下函数增强了功能。

# set

这些是与设置锥形渐变参数和属性相关的函数：

`conical_gradient.setAngle(float)`: 这将设置参数中指定的起始角度，在逻辑坐标中，对于这个锥形渐变。

`conical_gradient.setCenter(QtCore.QPointF)`: 这将使用参数中指定的浮点值设置中心点，在逻辑坐标中，对于这个锥形渐变。

`conical_gradient.setCenter(x, y)`: 这将根据逻辑坐标中的`x`（*x*轴）和`y`（y 轴）位置，使用浮点值设置中心点，对于这个锥形渐变。

# functional

这些是与锥形渐变当前值返回相关的函数：

`conical_gradient.angle()`: 这返回此圆锥渐变的起始角度，作为逻辑坐标中的浮点值。

`conical_gradient.center()`: 这返回此圆锥渐变的 `QtCore.QPointF` 类型的中心，在逻辑坐标中。

# 图片

Qt 库提供了一套完整的类，用于处理不同类型的图像。所有最流行的图像处理格式都得到了支持。主要格式在此描述。

# QPicture

此类表示重放和记录 `QPainter` 类命令的绘图设备。使用此类，可以将 `QPainter` 的绘图命令序列化为平台无关的格式，并作为图片表示。这用于保存绘图图片和加载它。图片将以类似于 `filename.pic` 的名称保存到文件中，以供以后使用。此类的声明如下：

```py
picture = QtGui.QPicture()
```

`QPicture` 类继承自 `QPaintDevice` 类，并通过以下函数增强了功能。

# 设置

这些是与图片相关的设置参数和属性的函数：

`picture.setBoundingRect(QtCore.QRect)`: 这为此图片设置参数中指定的边界矩形。

`picture.setData(bytes)`: 这将从参数中指定的数据设置此图片的数据。它还会复制输入数据。

# 是

此函数返回与图片相关的布尔值 (`bool`)：

`picture.isNull()`: 如果此图片不包含任何数据，则返回 `True`。

# 功能

这些是与当前图片值的返回或功能仪器相关的函数：

`picture.boundingRect()`: 这返回此图片的 `QtCore.QRect` 类型的边界矩形。

`picture.data()`: 这返回指向此图片数据的指针。

`picture.inputFormats()`: 这返回支持图片输入的格式列表。

`picture.load(r"/Path/To/filename.pic")`: 这从参数中指定的文件加载图片，如果成功则返回 `True`。

`picture.outputFormats()`: 这返回支持图片输出的格式列表。

`picture.pictureFormat(r"/Path/To/filename.pic")`: 这返回参数中指定的图片的格式。

`picture.play(QtGui.QPainter)`: 这使用参数中指定的画家重放图片，如果成功则返回 `True`。

`picture.save(r"/Path/To/filename.pic")`: 这将保存创建的图片到参数中指定的文件，如果成功则返回 `True`。

`picture.size()`: 这返回图片的数据大小。

`picture.swap(QtGui.QPicture)`: 这与此参数指定的图片交换。

# QPixmap

`QPixmap` 类提供了一种处理像素映射数据的方法。位图允许像素为任何颜色。它是图像的离屏表示，可用于直接访问和操作像素。此类的声明如下：

```py
pixmap = QtGui.QPixmap()
```

`QPixmap` 从 `QPaintDevice` 类继承，并通过以下函数改进了功能。

# set

这些是与设置与位图相关的参数和属性相关的函数：

+   `pixmap.setDevicePixelRatio(float)`: 这将指定参数中指定的设备像素比率应用于此位图。它表示位图像素与设备无关像素之间的比率。

+   `pixmap.setMask(QtGui.QBitmap)`: 这设置参数中指定的掩码位图。

# is/has

这些是与位图状态相关的返回布尔值（`bool`）的函数：

`pixmap.hasAlpha()`: 如果它为此位图具有 alpha 通道或掩码，则返回 `True`。

`pixmap.hasAlphaChannel()`: 如果它为此位图具有尊重 alpha 通道的格式，则返回 `True`。

`pixmap.isNull()`: 如果此位图是 `null`（无数据），则返回 `True`。

`pixmap.isQBitmap()`: 如果这是一个 `QtGui.QBitmap` 类型，则返回 `True`。

# functional

这些是与位图当前值或功能仪器相关的函数：

`pixmap.cacheKey()`: 这返回作为此位图标识符的键。

`pixmap.convertFromImage(QtGui.QImage, QtCore.Qt.ImageConversionFlags)`: 这用标志（第二个参数）中指定的转换替换图像（第一个参数）的位图数据。可用的转换标志有 `QtCore.Qt.AutoColor`、`QtCore.Qt.ColorOnly` 和 `QtCore.Qt.MonoOnly`（单色）。

`pixmap.copy(QtCore.QRect)`: 这返回指定参数中矩形的 `QtGui.QPixmap` 类型位图子集的深度副本。

`pixmap.copy(x, y, w, h)`: 这返回指定矩形的 `QtGui.QPixmap` 类型位图子集的深度副本，从 `x`（x 轴）和 `y`（y 轴）开始；具有 `w`（宽度）和 `h`（高度）。

`pixmap.createHeuristicMask(bool)`: 如果此参数为 `True`，则为此位图创建 `QtGui.QBitmap` 类型的启发式掩码。

`pixmap.createMaskFromColor(QtGui.QColor, QtCore.Qt.MaskMode)`: 这为此位图创建并返回 `QtGui.QBitmap` 类型的掩码。掩码将基于颜色（第一个参数）并相对于掩码模式（第二个参数）。

`pixmap.defaultDepth()`: 这返回此应用程序默认使用的位图深度。

`pixmap.depth()`: 这返回位图的深度（每像素位数（bpp）或位平面）。

`pixmap.detach()`: 这将此位图从此位图的共享数据中分离出来。

`pixmap.devicePixelRatio()`: 这返回设备像素与设备无关像素之间的比率。

`pixmap.fill(QtGui.QColor)`: 这用参数中指定的颜色填充此位图。

`pixmap.fromImage(QtGui.QImage, QtCore.Qt.ImageConversionFlags)`: 这使用标志（第二个参数）将图像（第一个参数）转换为位图。

`pixmap.fromImageReader(QtGui.QImageReader, QtCore.Qt.ImageConversionFlags)`: 这直接使用标志（第二个参数）从图像读取器（第一个参数）创建位图。

`pixmap.load(r"/Path/To/filename.png", str, QtCore.Qt.ImageConversionFlags)`: 这从文件（第一个参数）加载位图，使用格式（第二个参数）和标志（第三个参数）。

`pixmap.loadFromData(QtCore.QByteArray, str, QtCore.Qt.ImageConversionFlags)`: 这从二进制数据（第一个参数）加载位图，使用格式（第二个参数）和标志（第三个参数）。

`pixmap.mask()`: 这从位图的 alpha 通道中提取位图遮罩。

`pixmap.rect()`: 这返回此位图的包围矩形。

`pixmap.save(r"/Path/To/filename.png", str, int)`: 这将位图保存到文件（第一个参数），使用格式（第二个参数）和品质因子（第三个参数）。品质因子必须在`0`到`100`之间，或者为`-1`（默认设置）。

`pixmap.scaled(QtCore.QSize, QtCore.Qt.AspectRatioMode, QtCore.Qt.TransformationMode)`: 这将此位图缩放到指定的大小（第一个参数），保持宽高比（第二个参数）和变换模式（第三个参数）。

`pixmap.scaled(w, h, QtCore.Qt.AspectRatioMode, QtCore.Qt.TransformationMode)`: 这将此位图缩放到具有`w`（宽度）和`h`（高度）的矩形，并带有宽高比（第二个参数）和变换模式（第三个参数）。

`pixmap.scaledToHeight(h, QtCore.Qt.TransformationMode)`: 这根据`h`（高度）缩放此位图，并带有变换模式（第二个参数）。

`pixmap.scaledToWidth(w, QtCore.Qt.TransformationMode)`: 这根据`w`（宽度）缩放此位图，并带有变换模式（第二个参数）。

`pixmap.scroll(dx, dy, QtCore.QRect, QtGui.QRegion)`: 这根据`dx`和`dy`整数值滚动此位图的矩形区域（第三个参数）。暴露的区域（第四个参数）保持不变。

`pixmap.scroll(dx, dy, x, y, w, h, QtGui.QRegion)`: 这根据`dx`和`dy`整数值滚动指定矩形的区域，从`x`（*x*轴）和`y`（*y*轴）开始；`w`（宽度）和`h`（高度）为此位图的大小。暴露的区域（第四个参数）保持不变。

`pixmap.size()`: 这返回位图的大小。

`pixmap.swap(QtGui.QPixmap)`: 这与此参数指定的位图交换此位图。

`pixmap.toImage()`: 这将位图转换为`QtGui.QImage`类型的图像。

`pixmap.transformed(QtGui.QTransform, QtCore.Qt.TransformationMode)`: 这返回使用变换（第一个参数）和变换模式（第二个参数）转换后的位图副本。原始位图将不会改变。

`pixmap.trueMatrix(QtGui.QTransform, w, h)`: 这返回用于转换此位图的实际矩阵，以及矩阵（第一个参数）和 `w`（宽度）和 `h`（高度）。

# QBitmap

`QBitmap` 类提供了一种处理单色或 1 位深度像素映射数据的方法。它是一个离屏绘图设备，用于创建自定义光标和画笔，以及构建诸如 `QRegion`、位图和小部件的掩码等对象。此类的声明如下：

```py
bitmap = QtGui.QBitmap()
```

`QBitmap` 从 `QPixmap` 类继承，并通过以下函数改进了功能。

# functional

这些是与当前位图值的返回或与功能仪器相关的函数。

`bitmap.clear()`: 这将清除此位图并将所有位设置为 `Qt.color0`，或零像素值。

`bitmap.fromData(QtCore.QSize, bytes, QtGui.QImage.Format)`: 这使用大小（第一个参数）构建此位图，并将内容设置为位（第二个参数），字节对齐，按位顺序（第三个参数）。

`bitmap.swap(QtGui.QBitmap)`: 这将此位图与参数中指定的位图交换。

`bitmap.transformed(QtGui.QTransform)`: 这返回使用参数中指定的变换转换后的位图副本。

# QImage

此类提供硬件无关的图像表示，用于处理图像。它还允许直接访问像素数据。`QImage` 类的声明如下：

```py
image = QtGui.QImage()
```

`QImage` 从 `QPaintDevice` 类继承，并通过以下函数改进了功能。

# set

这些是与设置与图像相关的参数和属性相关的函数。

`image.setAlphaChannel(QtGui.QImage)`: 这将此图像的 alpha 通道设置为参数中指定的一个。

`image.setColor(int, int)`: 这会将颜色表中的索引（第一个参数）设置为颜色值（第二个参数）。

`image.setColorCount(int)`: 这会将颜色表计数调整为参数中指定的值。

`image.setColorTable([int])`: 这会将颜色表设置为参数中指定的颜色。

`image.setDevicePixelRatio(float)`: 这将为该图像设置参数中指定的设备像素比。这是图像像素与设备无关像素之间的比率。

`image.setDotsPerMeterX(int)`: 这将设置物理米中由 *x* 轴定位的像素数量。这将描述此图像的缩放和宽高比。

`image.setDotsPerMeterY(int):` 这设置由 *y* 轴在物理米中定位的像素数。这将描述此图像的缩放和宽高比。

`image.setOffset(QtCore.QPoint)`: 这设置相对于其他图像，图像偏移的像素数，到参数指定的点。

`image.setPixel(QtCore.QPoint, int)`: 这在点（第一个参数）处设置第二个参数指定的像素索引或颜色。

`image.setPixel(x, y, int)`: 这将在坐标 `x` 和 `y` 处设置第三参数指定的像素索引或颜色。

`image.setPixelColor(QtCore.QPoint, QtGui.QColor)`: 这将点（第一个参数）处的像素设置为颜色（第二个参数）。

`image.setPixel(x, y, QtGui.QColor)`: 这将坐标 `x` 和 `y` 处的像素设置为颜色（第三个参数）。

`image.setText(str, str)`: 这设置图像（第二个参数）的文本，并将其与键（第一个参数）关联。

# has and is

这些是返回与图像状态相关的布尔值 (`bool`) 的函数：

`image.hasAlphaChannel()`: 如果它具有尊重此图像的 alpha 通道的格式，则返回 `True`。

`image.isGrayscale()`: 如果此图像中的所有颜色都是灰色阴影，则返回 `True`。

`image.isNull()`: 如果此图像为 `null`（无数据），则返回 `True`。

# functional

这些是与图像当前值或功能仪器相关的函数：

`image.allGray()`: 如果此图像中的所有颜色都是灰色阴影，则返回 `True`。

`image.bitPlaneCount()`: 这返回图像位平面的数量——每个像素的颜色和透明度的位数。

`image.byteCount()`: 这返回此图像数据占用的字节数。

`image.bytesPerLine()`: 这返回每图像扫描行的字节数。

`image.cacheKey()`: 这返回作为此图像内容标识符的键。

`image.color(int)`: 这返回颜色表在参数指定的索引处的颜色。索引从 `0` 开始。

`image.colorTable()`: 这返回此图像颜色表中的颜色列表。

`image.constBits()`: 这返回指向第一个像素数据的指针。

`image.constScanLine(int)`: 这返回指向参数指定的索引处的扫描行像素数据的指针。索引从 `0` 开始。

`image.convertToFormat(QtGui.QImage.Format, [int], QtCore.Qt.ImageConversionFlags)`: 这返回一个图像副本，该图像已转换为格式（第一个参数），使用颜色表（第二个参数）和标志（第三个参数）。

`image.copy(QtCore.QRect)`: 这返回一个由参数中指定的矩形大小的图像子区域。

`image.copy(x, y, w, h)`: 这返回从指定矩形开始复制（以 `x`（x 轴）和 `y`（y 轴）为起点）的图像，宽度为 `w`，高度为 `h`。

`image.createAlphaMask(QtCore.Qt.ImageConversionFlags)`: 这从图像中的 alpha 缓冲区创建并返回一个每像素 1 位的掩码。

`image.createHeuristicMask(bool)`: 这为图像创建并返回一个每像素 1 位的掩码。如果参数为 `True`，则掩码足以覆盖像素；否则，掩码大于数据像素。

`image.createMaskFromColor(int, QtCore.Qt.MaskMode)`: 这根据颜色（第一个值）和模式（第二个参数）创建并返回掩码。

`image.depth()`: 这返回图像的深度（`bpp`）。

`image.devicePixelRatio()`: 这返回设备像素和设备无关像素之间的比率。

`image.dotsPerMeterX()`: 这返回在物理米中由 *X* 轴定位的像素数量。这描述了此图像的缩放和宽高比。

`image.dotsPerMeterY()`: 这返回在物理米中由 *Y* 轴定位的像素数量。这描述了此图像的缩放和宽高比。

`image.fill(QtCore.Qt.GlobalColor)`: 这用参数中指定的颜色填充此图像。

`image.fill(QtGui.QColor)`: 这用参数中指定的颜色填充此图像。

`image.fill(int)`: 这用参数中指定的像素值填充此图像。

`image.format()`: 这返回此图像的 `QtGui.QImage.Format` 类型的格式。

`image.fromData(bytes, str)`: 这从 `QtCore.QByteArray` 数据中加载此图像。

`image.invertPixels(QtGui.QImage.InvertMode)`: 这通过使用参数中指定的模式反转此图像的所有像素值。

`image.load(r"/Path/To/filename.png", str)`: 这以格式（第二个参数）从文件（第一个参数）中加载图像。

`image.load(QtCore.QIODevice, str)`: 这以格式（第二个参数）从设备（第一个参数）中读取图像。

`image.loadFromData(bytes, str)`: 这从字节（第一个参数）中加载图像，格式（第二个参数）。

`image.mirrored(bool, bool)`: 这返回图像的镜像；如果第一个参数为 `True`，则镜像方向为水平，如果第二个参数为 `True`，则镜像方向为垂直。

`image.offset()`: 这返回相对于其他图像要偏移的像素数。

`image.pixel(QtCore.QPoint)`: 这返回位于参数中指定点处的像素。

`image.pixel(x, y)`: 这返回了位于 `x` 和 `y` 坐标处的像素。

`image.pixelColor(QtCore.QPoint)`: 这返回位于参数中指定点处的像素颜色。

`image.pixelColor(x, y)`: 这返回了位于 `x` 和 `y` 坐标处的像素颜色。

`image.pixelFormat()`: 这返回了作为此图像的 `QtGui.QPixelFormat` 格式。

`image.pixelIndex(QtCore.QPoint)`: 这返回了位于参数中指定点位置的像素索引。

`image.pixelIndex(x, y)`: 这将返回`x`和`y`坐标处的像素索引。

`image.rect()`: 这将返回包围此图像的`QtCore.QRect`类型的矩形。

`image.reinterpretAsFormat(QtGui.QImage.Format)`: 这将图像的格式更改为参数中指定的格式；数据将不会改变。

`image.rgbSwapped()`: 这将返回所有像素的红色和蓝色分量交换后的图像。

`image.save(r"/Path/To/filename.png", str, int)`: 这将图像保存到文件（第一个参数），格式（第二个参数）和品质因子（第三个参数）。品质因子必须在`0`（小，压缩）到`100`（大，未压缩）或`-1`（默认设置）的范围内。

`image.scaled(QtCore.QSize, QtCore.Qt.AspectRatioMode, QtCore.Qt.TransformationMode)`: 这将返回一个副本，该副本按大小（第一个参数）缩放，并带有纵横比（第二个参数）和变换模式（第三个参数）。

`image.scaled(w, h, QtCore.Qt.AspectRatioMode, QtCore.Qt.TransformationMode)`: 这将返回一个副本，该副本按`w`（宽度）和`h`（高度）缩放，并带有纵横比（第二个参数）和变换模式（第三个参数）。

`image.scaledToHeight(h, QtCore.Qt.TransformationMode)`: 这将返回一个副本，该副本相对于`h`（高度）缩放，并带有变换模式（第二个参数）。

`image.scaledToWidth(w, QtCore.Qt.TransformationMode)`: 这将返回一个副本，该副本相对于`w`（宽度）缩放，并带有变换模式（第二个参数）。

`image.size()`: 这将返回此图像的大小。

`image.sizeInBytes()`: 这将返回图像数据的大小（以字节为单位）。

`image.smoothScaled(w, h)`: 这将返回一个图像副本，该图像经过平滑缩放，其大小为`w`（宽度）和`h`（高度）。

`image.swap(QtGui.QImage)`: 这将与此参数指定的图像交换此图像。

`image.text(str)`: 这将返回与参数中指定的键关联的图像文本。

`image.textKeys()`: 这将返回此图像文本的键列表。

`image.toImageFormat(QtGui.QPixelFormat)`: 这将参数中指定的格式转换为`QtGui.QImage.Format`类型。

`image.toPixelFormat(QtGui.QImage.Format)`: 这将参数中指定的格式转换为`QtGui.QPixelFormat`类型。

`image.transformed(QtGui.QTransform, QtCore.Qt.TransformationMode)`: 这将返回一个经过变换的图像副本，使用变换（第一个参数）和变换模式（第二个参数）。

`image.trueMatrix(QtGui.QTransform, w, h)`: 这将返回用于使用矩阵（第一个参数）和`w`（宽度）以及`h`（高度）变换此图像的实际矩阵。

`image.valid(QPoint)`: 如果参数中指定的点在此图像中是有效的坐标，则返回`True`。

`image.valid(x, y)`: 如果指定为坐标`x`和`y`的点在此图像中是有效坐标，则返回`True`。

# QIcon

此类实现了 GUI 应用程序中的可缩放图标。这些可以与小部件、按钮或作为窗口图标一起使用。图标的声明语法如下：

```py
icon = QtGui.QIcon()
```

`QIcon`通过以下函数提高功能。

# add

这些是与图标添加相关的函数：

`icon.addFile(r"/Path/To/filename.png", QtCore.QSize, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 将文件（第一个参数）中的图像添加到图标中。它通过大小（第二个参数）、模式（第三个参数）和状态（第四个参数）进行指定。

`icon.addPixmap(QtGui.QPixmap, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 将位图（第一个参数）添加到图标中。它通过模式（第二个参数）和状态（第三个参数）进行指定。

# set

这些是与图标相关的设置参数和属性的函数：

`icon.setFallbackSearchPaths([str])`: 将回退搜索路径设置为图标回退的路径列表。

`icon.setIsMask(bool)`: 如果参数为`True`，则指定此图标为遮罩图像。

`icon.setThemeName(str)`: 将图标主题名称设置为参数中指定的名称。

`icon.setThemeSearchPaths([str])`: 将主题搜索路径设置为图标主题的路径列表。

# has/is

这些是与图像状态相关的布尔值（`bool`）返回值的函数：

`icon.hasThemeIcon(str)`: 如果图标对于参数中指定的名称可用，则返回`True`。

`icon.isMask()`: 如果图标被标记为遮罩图像，则返回`True`。

`icon.isNull()`: 如果图标为空，则返回`True`。

# functional

这些是与当前图标值的返回或功能仪器相关的函数：

`icon.actualSize(QtCore.QSize, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 返回图标在大小（第一个参数）、模式（第二个参数）和状态（第三个参数）下的实际大小。可用的图标模式如下：

+   `QtGui.QIcon.Normal`—`0`: 与图标无交互；功能可用。

+   `QtGui.QIcon.Disabled`—`1`: 功能不可用。

+   `QtGui.QIcon.Active`—`2`: 与图标的交互；功能可用。

+   `QtGui.QIcon.Selected`—`3`: 图标被选中。

图标状态如下：

+   `QtGui.QIcon.On`—`0`: 小部件处于*开启*状态。

+   `QtGui.QIcon.Off`—`1`: 小部件处于*关闭*状态。

`icon.actualSize(QtGui.QWindow, QtCore.QSize, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 返回图标相对于窗口（第一个参数）的大小（第二个参数），模式（第三个参数）和状态（第四个参数）的实际大小。

`icon.availableSizes(QtGui.QIcon.Mode, QtGui.QIcon.State)`: 这返回与模式（第一个参数）和状态（第二个参数）相关的图标可用大小的列表。

`icon.cacheKey()`: 这返回图标内容的标识符。

`icon.fallbackSearchPaths()`: 这返回图标回退搜索路径的列表。

`icon.fromTheme(str)`: 这返回与参数中指定的名称对应的图标，在当前图标主题中。

`icon.fromTheme(str, QtGui.QIcon)`: 这返回与名称（第一个参数）对应的图标；如果找不到图标，则返回回退（第二个参数）。

`icon.name()`: 这返回图标的名称。

`icon.paint(QtGui.QPainter, QtCore.QRect, QtCore.Qt.Alignment, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 这使用画家（第一个参数）在指定的矩形（第二个参数）、对齐（第三个参数）、模式（第四个参数）和状态（第五个参数）中绘制图标。

`icon.paint(QtGui.QPainter, x, y, w, h, QtCore.Qt.Alignment, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 这使用画家（第一个参数）在以 `x`（*x* 轴）和 `y`（*y* 轴）开始的矩形内，使用 `w`（宽度）和 `h`（高度）大小，以及对齐（第三个参数），模式（第四个参数）和状态（第五个参数）来绘制图标。

`icon.pixmap(QtCore.QSize, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 这返回具有指定大小（第一个参数）、模式（第二个参数）和状态（第三个参数）的位图。

`icon.pixmap(w, h, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 这返回具有指定 `w`（宽度）和 `h`（高度）、模式（第三个参数）和状态（第四个参数）的位图。

`icon.pixmap(QtGui.QWindow, QtCore.QSize, QtGui.QIcon.Mode, QtGui.QIcon.State)`: 这返回具有指定窗口（第一个参数）、大小（第二个参数）、模式（第三个参数）和状态（第四个参数）的位图。

`icon.swap(QtGui.QIcon)`: 这将此图标与参数中指定的图标交换。

`icon.themeName()`: 这返回当前图标主题的名称。

`icon.themeSearchPaths()`: 这返回搜索图标主题的路径。

# 摘要

本章从管理图形元素和使用 Qt 库进行绘制的基本类的角度进行了描述。Qt 框架的目标是开发图形应用程序、图形工具以及与创建图形相关的类，并确保它们得到广泛应用。在可能的情况下，本书将讨论主要的内容。

虽然本章关注的是基本类，但在下一章中，我们将讨论在应用程序中创建图形表示的模型/视图范式以及实现此范式的类。
