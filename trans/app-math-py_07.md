# 第八章

回归和预测

统计学家或数据科学家最重要的任务之一是生成对两组数据之间关系的系统性理解。这可能意味着两组数据之间的“连续”关系，其中一个值直接取决于另一个变量的值。或者，它可能意味着分类关系，其中一个值根据另一个值进行分类。处理这些问题的工具是*回归*。在其最基本的形式中，回归涉及将一条直线拟合到两组数据的散点图中，并进行一些分析，以查看这条直线如何“拟合”数据。当然，我们通常需要更复杂的东西来模拟现实世界中存在的更复杂的关系。

时间序列代表这些回归类型问题的一种专门类别，其中一个值在一段时间内发展。与更简单的问题不同，时间序列数据通常在连续值之间有复杂的依赖关系；例如，一个值可能依赖于前两个值，甚至可能依赖于前一个“噪音”。时间序列建模在科学和经济学中非常重要，有各种工具可用于建模时间序列数据。处理时间序列数据的基本技术称为**ARIMA**，它代表**自回归综合移动平均**。该模型包括两个基本组件，一个**自回归**（**AR**）**组件和一个**移动平均**（**MA**）组件，用于构建观察数据的模型。

在本章中，我们将学习如何建立两组数据之间的关系模型，量化这种关系的强度，并对其他值（未来）生成预测。然后，我们将学习如何使用逻辑回归，在分类问题中，这是简单线性模型的一种变体。最后，我们将使用 ARIMA 为时间序列数据构建模型，并基于这些模型构建不同类型的数据。我们将通过使用一个名为 Prophet 的库来自动生成时间序列数据模型来结束本章。

在本章中，我们将涵盖以下内容：

+   使用基本线性回归

+   使用多元线性回归

+   使用对数回归进行分类

+   使用 ARMA 对时间序列数据进行建模

+   使用 ARIMA 从时间序列数据进行预测

+   使用 ARIMA 对季节性数据进行预测

+   使用 Prophet 对时间序列进行建模

让我们开始吧！

# 技术要求

在本章中，像往常一样，我们需要导入 NumPy 包并使用别名`np`，导入 Matplotlib `pyplot`模块并使用`plt`别名，以及导入 Pandas 包并使用`pd`别名。我们可以使用以下命令来实现：

```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

在本章中，我们还需要一些新的包。statsmodels 包用于回归和时间序列分析，`scikit-learn`包（`sklearn`）提供通用数据科学和机器学习工具，Prophet 包（`fbprophet`）用于自动生成时间序列数据模型。这些包可以使用您喜欢的包管理器（如`pip`）进行安装：

```py
          python3.8 -m pip install statsmodels sklearn fbprophet

```

Prophet 包可能在某些操作系统上安装起来比较困难，因为它的依赖关系。如果安装`fbprophet`出现问题，您可能希望尝试使用 Python 的 Anaconda 发行版及其包管理器`conda`，它可以更严格地处理依赖关系：

```py
          conda install fbprophet

```

最后，我们还需要一个名为`tsdata`的小模块，该模块包含在本章的存储库中。该模块包含一系列用于生成样本时间序列数据的实用程序。

本章的代码可以在 GitHub 存储库的`Chapter 07`文件夹中找到：[`github.com/PacktPublishing/Applying-Math-with-Python/tree/master/Chapter%2007`](https://github.com/PacktPublishing/Applying-Math-with-Python/tree/master/Chapter%2007)。

查看以下视频以查看代码实际操作：[`bit.ly/2Ct8m0B`](https://bit.ly/2Ct8m0B)。

# 使用基本线性回归

线性回归是一种建模两组数据之间依赖关系的工具，这样我们最终可以使用这个模型进行预测。名称来源于我们基于第二组数据形成一个线性模型（直线）。在文献中，我们希望建模的变量通常被称为*响应*变量，而我们在这个模型中使用的变量是*预测*变量。

在这个步骤中，我们将学习如何使用 statsmodels 包执行简单的线性回归，以建模两组数据之间的关系。

## 准备工作

对于这个步骤，我们需要导入 statsmodels 的`api`模块并使用别名`sm`，导入 NumPy 包并使用别名`np`，导入 Matplotlib 的`pyplot`模块并使用别名`plt`，以及一个 NumPy 默认随机数生成器的实例。所有这些都可以通过以下命令实现：

```py
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng(12345)
```

## 如何做到这一点...

以下步骤概述了如何使用 statsmodels 包对两组数据执行简单线性回归：

1.  首先，我们生成一些示例数据进行分析。我们将生成两组数据，这将说明一个良好的拟合和一个不太好的拟合：

```py
x = np.linspace(0, 5, 25)
rng.shuffle(x)
trend = 2.0
shift = 5.0
y1 = trend*x + shift + rng.normal(0, 0.5, size=25)
y2 = trend*x + shift + rng.normal(0, 5, size=25)
```

1.  执行回归分析的一个很好的第一步是创建数据集的散点图。我们将在同一组坐标轴上完成这一步：

```py
fig, ax = plt.subplots()
ax.scatter(x, y1, c="b", label="Good correlation")
ax.scatter(x, y2, c="r", label="Bad correlation")
ax.legend()
ax.set_xlabel("X"),
ax.set_ylabel("Y")
ax.set_title("Scatter plot of data with best fit lines")
```

1.  我们需要使用`sm.add_constant`实用程序例程，以便建模步骤将包括一个常数值：

```py
pred_x = sm.add_constant(x)
```

1.  现在，我们可以为我们的第一组数据创建一个`OLS`模型，并使用`fit`方法来拟合模型。然后，我们使用`summary`方法打印数据的摘要：

```py
model1 = sm.OLS(y1, pred_x).fit()
print(model1.summary())
```

1.  我们重复第二组数据的模型拟合并打印摘要：

```py
model2 = sm.OLS(y2, pred_x).fit()
print(model2.summary())
```

1.  现在，我们使用`linspace`创建一个新的*x*值范围，我们可以用它来在散点图上绘制趋势线。我们需要添加`constant`列以与我们创建的模型进行交互：

```py
model_x = sm.add_constant(np.linspace(0, 5))
```

1.  接下来，我们在模型对象上使用`predict`方法，这样我们就可以使用模型在前一步生成的每个*x*值上预测响应值：

```py
model_y1 = model1.predict(model_x)
model_y2 = model2.predict(model_x)
```

1.  最后，我们将在散点图上绘制在前两个步骤中计算的模型数据：

```py
ax.plot(model_x[:, 1], model_y1, 'b')
ax.plot(model_x[:, 1], model_y2, 'r')
```

散点图以及我们添加的最佳拟合线（模型）可以在下图中看到：

![](img/8bf4175e-e73a-4f3b-b5a1-860f8d0bc8f1.png)图 7.1：使用最小二乘法回归计算的数据散点图和最佳拟合线。

## 工作原理...

基本数学告诉我们，一条直线的方程如下所示：

![](img/def061a4-d8f7-482e-a3a3-78d4587c495f.png)

在这里，*c*是直线与*y*轴相交的值，通常称为*y*截距，*m*是直线的斜率。在线性回归的背景下，我们试图找到响应变量*Y*和预测变量*X*之间的关系，使其形式为一条直线，使以下情况发生：

![](img/2777a0a2-21f8-41b8-97d0-0969989f1e8f.png)

在这里，*c*和*m*现在是要找到的参数。我们可以用另一种方式来表示这一点，如下所示：

![](img/f85da231-6294-45c3-9e5a-b08dd6a6ef79.png)

这里，*E*是一个误差项，一般来说，它取决于*X*。为了找到“最佳”模型，我们需要找到*E*误差被最小化的*c*和*m*参数值（在适当的意义上）。找到使这个误差最小化的参数值的基本方法是最小二乘法，这给了这里使用的回归类型以它的名字：*普通最小二乘法*。一旦我们使用这种方法建立了响应变量和预测变量之间的某种关系，我们的下一个任务就是评估这个模型实际上如何代表这种关系。为此，我们形成了以下方程给出的*残差*：

![](img/c3ade12f-e352-49c2-a1f1-b07373253b4d.png)

我们对每个数据点*X[i]*和*Y[i]*进行这样的操作。为了对我们对数据之间关系建模的准确性进行严格的统计分析，我们需要残差满足某些假设。首先，我们需要它们在概率意义上是独立的。其次，我们需要它们围绕 0 呈正态分布，并且具有相同的方差。（在实践中，我们可以稍微放松这些条件，仍然可以对模型的准确性做出合理的评论。）

在这个配方中，我们使用线性关系从预测数据中生成响应数据。我们创建的两个响应数据集之间的差异在于每个值的误差的“大小”。对于第一个数据集`y1`，残差呈正态分布，标准差为 0.5，而对于第二个数据集`y2`，残差的标准差为 5.0。我们可以在*图 7.1*中的散点图中看到这种变异性，其中`y1`的数据通常非常接近最佳拟合线 - 这与用于生成数据的实际关系非常接近 - 而`y2`数据则远离最佳拟合线。

来自 statsmodels 包的`OLS`对象是普通最小二乘回归的主要接口。我们将响应数据和预测数据作为数组提供。为了在模型中有一个常数项，我们需要在预测数据中添加一列 1。`sm.add_constant`例程是一个简单的实用程序，用于添加这个常数列。`OLS`类的`fit`方法计算模型的参数，并返回一个包含最佳拟合模型参数的结果对象（`model1`和`model2`）。`summary`方法创建一个包含有关模型和拟合优度的各种统计信息的字符串。`predict`方法将模型应用于新数据。顾名思义，它可以用于使用模型进行预测。

除了参数值本身之外，摘要中报告了另外两个统计量。第一个是*R²*值，或者调整后的版本，它衡量了模型解释的变异性与总变异性之间的关系。这个值将在 0 和 1 之间。较高的值表示拟合效果更好。第二个是 F 统计量的 p 值，它表示模型的整体显著性。与 ANOVA 测试一样，较小的 F 统计量表明模型是显著的，这意味着模型更有可能准确地对数据进行建模。

在这个配方中，第一个模型`model1`的调整后的*R²*值为 0.986，表明该模型非常紧密地拟合了数据，p 值为 6.43e-19，表明具有很高的显著性。第二个模型的调整后的*R²*值为 0.361，表明该模型与数据的拟合程度较低，p 值为 0.000893，也表明具有很高的显著性。尽管第二个模型与数据的拟合程度较低，但从统计学的角度来看，并不意味着它没有用处。该模型仍然具有显著性，尽管不如第一个模型显著，但它并没有解释所有的变异性（或者至少是数据中的一个显著部分）。这可能表明数据中存在额外的（非线性）结构，或者数据之间的相关性较低，这意味着响应和预测数据之间的关系较弱（由于我们构造数据的方式，我们知道后者是真实的）。

## 还有更多...

简单线性回归是统计学家工具包中一个很好的通用工具。它非常适合找到两组已知（或被怀疑）以某种方式相互关联的数据之间关系的性质。衡量一个数据集依赖于另一个数据集的程度的统计测量称为*相关性*。我们可以使用相关系数来衡量相关性，例如*Spearman 秩相关系数*。高正相关系数表示数据之间存在强烈的正相关关系，就像在这个示例中看到的那样，而高负相关系数表示强烈的负相关关系，其中通过数据的最佳拟合线的斜率为负。相关系数为 0 意味着数据没有相关性：数据之间没有关系。

如果数据集之间明显相关，但不是线性（直线）关系，那么它可能遵循一个多项式关系，例如，一个值与另一个值的平方有关。有时，您可以对一个数据集应用转换，例如对数，然后使用线性回归来拟合转换后的数据。当两组数据之间存在幂律关系时，对数特别有用。

# 使用多元线性回归

简单线性回归，如前面的示例所示，非常适合产生一个响应变量和一个预测变量之间关系的简单模型。不幸的是，有一个单一的响应变量依赖于许多预测变量更为常见。此外，我们可能不知道从一个集合中选择哪些变量作为良好的预测变量。对于这个任务，我们需要多元线性回归。

在这个示例中，我们将学习如何使用多元线性回归来探索响应变量和几个预测变量之间的关系。

## 准备就绪

对于这个示例，我们将需要导入 NumPy 包作为`np`，导入 Matplotlib 的`pyplot`模块作为`plt`，导入 Pandas 包作为`pd`，并创建 NumPy 默认随机数生成器的实例，使用以下命令：

```py
from numpy.random import default_rng
rng = default_rng(12345)
```

我们还需要导入 statsmodels 的`api`模块作为`sm`，可以使用以下命令导入：

```py
import statsmodels.api as sm
```

## 如何做...

以下步骤向您展示了如何使用多元线性回归来探索几个预测变量和一个响应变量之间的关系：

1.  首先，我们需要创建要分析的预测数据。这将采用 Pandas 的`DataFrame`形式，有四个项。我们将通过添加一个包含 1 的列来添加常数项：

```py
p_vars = pd.DataFrame({
"const": np.ones((100,)),
"X1": rng.uniform(0, 15, size=100),
"X2": rng.uniform(0, 25, size=100),
"X3": rng.uniform(5, 25, size=100)
})

```

1.  接下来，我们将仅使用前两个变量生成响应数据：

```py
residuals = rng.normal(0.0, 12.0, size=100)
Y = -10.0 + 5.0*p_vars["X1"] - 2.0*p_vars["X2"] + residuals
```

1.  现在，我们将生成响应数据与每个预测变量的散点图：

```py
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,
   tight_layout=True)
ax1.scatter(p_vars["X1"], Y)
ax2.scatter(p_vars["X2"], Y)
ax3.scatter(p_vars["X3"], Y)

```

1.  然后，我们将为每个散点图添加轴标签和标题，因为这是一个很好的做法：

```py
ax1.set_title("Y against X1")
ax1.set_xlabel("X1")
ax1.set_ylabel("Y")
ax2.set_title("Y against X2")
ax2.set_xlabel("X2")
ax3.set_title("Y against X3")
ax3.set_xlabel("X3")
```

```py
        The resulting plots can be seen in the following figure:

          ![](img/151a3a34-831e-40fc-9d29-ec1db98ab665.png)

        Figure 7.2: Scatter plots of the response data against each of the predictor variables
        As we can see, there appears to be some correlation between the response data and the first two predictor columns, `X1` and `X2`. This is what we expect, given how we generated the data.

          5.  We use the same `OLS` class to perform multilinear regression; that is, providing the response array and the predictor `DataFrame`:

```

model = sm.OLS(Y, p_vars).fit()

print(model.summary())

```py

        The output of the `print` statement is as follows:

```

OLS 回归结果

==================================================================

因变量：y                  R 平方：0.770

模型：OLS 调整                   R 平方：0.762

方法：最小二乘法             F 统计量：106.8

日期：2020 年 4 月 23 日星期四            概率（F 统计量）：1.77e-30

时间：12:47:30                    对数似然：-389.38

观测数量：100             AIC：786.8

残差自由度：96                  BIC：797.2

模型自由度：3

协方差类型：非鲁棒

===================================================================

coef    std err     t      P>|t|      [0.025     0.975]

-------------------------------------------------------------------

常数   -9.8676    4.028   -2.450    0.016     -17.863    -1.872

X1       4.7234    0.303    15.602   0.000       4.122     5.324

X2      -1.8945    0.166   -11.413   0.000     -2.224     -1.565

X3      -0.0910    0.206   -0.441    0.660     -0.500      0.318

===================================================================

Omnibus: 0.296                Durbin-Watson: 1.881

Prob(Omnibus): 0.862          Jarque-Bera (JB): 0.292

偏度：0.123                   Prob(JB): 0.864

峰度：2.904               条件数 72.9

===================================================================

```py

        In the summary data, we can see that the `X3` variable is not significant since it has a p value of 0.66.

          6.  Since the third predictor variable is not significant, we eliminate this column and perform the regression again:

```

second_model = sm.OLS(Y, p_vars.loc[:, "const":"X2"]).fit()

print(second_model.summary())

```py

        This results in a small increase in the goodness of fit statistics.
        How it works...
        Multilinear regression works in much the same way as simple linear regression. We follow the same procedure here as in the previous recipe, where we use the statsmodels package to fit a multilinear model to our data. Of course, there are some differences behind the scenes. The model we produce using multilinear regression is very similar in form to the simple linear model from the previous recipe. It has the following form:

          ![](img/0839eae4-836a-497f-9f90-8dd8f2cd4a17.png)

        Here, *Y* is the response variable, *X[i]* represents the predictor variables, *E* is the error term, and *β*[*i*] is the parameters to be computed. The same requirements are also necessary for this context: residuals must be independent and normally distributed with a mean of 0 and a common standard deviation.
        In this recipe, we provided our predictor data as a Pandas `DataFrame` rather than a plain NumPy array. Notice that the names of the columns have been adopted in the summary data that we printed. Unlike the first recipe, *Using basic linear regression*, we included the constant column in this `DataFrame`, rather than using the `add_constant` utility from statsmodels.
        In the output of the first regression, we can see that the model is a reasonably good fit with an adjusted *R²* value of 0.762, and is highly significant (we can see this by looking at the regression F statistic p value). However, looking closer at the individual parameters, we can see that both of the first two predictor values are significant, but the constant and the third predictor are less so. In particular, the third predictor parameter, `X3`, is not significantly different from 0 and has a p value of 0.66\. Given that our response data was constructed without using this variable, this shouldn't come as a surprise. In the final step of the analysis, we repeat the regression without the predictor variable, `X3`, which is a mild improvement to the fit.
        Classifying using logarithmic regression
        Logarithmic regression solves a different problem to ordinary linear regression. It is commonly used for classification problems where, typically, we wish to classify data into two distinct groups, according to a number of predictor variables. Underlying this technique is a transformation that's performed using logarithms. The original classification problem is transformed into a problem of constructing a model for the **log-odds***.* This model can be completed with simple linear regression. We apply the inverse transformation to the linear model, which leaves us with a model of the probability that the desired outcome will occur, given the predictor data. The transform we apply here is called the **logistic function**, which gives its name to the method. The probability we obtain can then be used in the classification problem we originally aimed to solve.
        In this recipe, we will learn how to perform logistic regression and use this technique in classification problems.
        Getting ready
        For this recipe, we will need the NumPy package imported as `np`, the Matplotlib `pyplot`module imported as `plt`, the Pandas package imported as `pd`, and an instance of the NumPy default random number generator to be created using the following commands:

```

从 numpy.random 导入 default_rng

rng = default_rng(12345)

```py

        We also need several components from the `scikit-learn` package to perform logistic regression. These can be imported as follows:

```

从 sklearn.linear_model 导入 LogisticRegression

从 sklearn.metrics 导入 classification_report

```py

        How to do it...
        Follow these steps to use logistic regression to solve a simple classification problem:

          1.  First, we need to create some sample data that we can use to demonstrate how to use logistic regression. We start by creating the predictor variables:

```

df = pd.DataFrame({

"var1": np.concatenate([rng.normal(3.0, 1.5, size=50),

rng.normal(-4.0, 2.0, size=50)]),

"var2": rng.uniform(size=100),

"var3": np.concatenate([rng.normal(-2.0, 2.0, size=50),

rng.normal(1.5, 0.8, size=50)])

})

```py

          2.  Now, we use two of our three predictor variables to create our response variable as a series of Boolean values:

```

分数 = 4.0 + df["var1"] - df["var3"]

Y = score >= 0

```py

          3.  Next, we scatter plot the points, styled according to the response variable, of the `var3` data against the `var1` data, which are the variables used to construct the response variable:

```

fig1, ax1 = plt.subplots()

ax1.plot(df.loc[Y, "var1"], df.loc[Y, "var3"], "bo", label="True

数据")

ax1.plot(df.loc[~Y, "var1"], df.loc[~Y, "var3"], "rx", label="False

数据")

ax1.legend()

ax1.set_xlabel("var1")

ax1.set_ylabel("var3")

ax1.set_title("Scatter plot of var3 against var1")

```py

        The resulting plot can be seen in the following figure:

          ![](img/e972cdff-197a-4940-8f33-dbe849ddb774.png)

        Figure 7.3: Scatter plot of the var3 data against var1, with classification marked

          4.  Next, we create a `LogisticRegression` object from the `scikit-learn` package and fit the model to our data:

```

model = LogisticRegression()

model.fit(df, Y)

```py

          5.  Next, we prepare some extra data, different from what we used to fit the model, to test the accuracy of our model:

```

test_df = pd.DataFrame({

"var1": np.concatenate([rng.normal(3.0, 1.5, size=50),

rng.normal(-4.0, 2.0, size=50)]),

"var2": rng.uniform(size=100),

"var3": np.concatenate([rng.normal(-2.0, 2.0, size=50),

rng.normal(1.5, 0.8, size=50)])

})

test_scores = 4.0 + test_df["var1"] - test_df["var3"]

test_Y = test_scores >= 0

```py

          6.  Then, we generate predicted results based on our logistic regression model:

```

test_predicts = model.predict(test_df)

```py

          7.  Finally, we use the `classification_report` utility from `scikit-learn` to print a summary of predicted classification against known response values to test the accuracy of the model. We print this summary to the Terminal:

```

print(classification_report(test_Y, test_predicts))

```py

        The report that's generated by this routine looks as follows:

```

precision     recall      f1-score      support

False       1.00        1.00         1.00          18

True       1.00        1.00         1.00          32

accuracy                                1.00          50

macro avg       1.00        1.00         1.00          50

weighted avg       1.00        1.00         1.00          50

```py

        How it works...
        Logistic regression works by forming a linear model of the *log odds* ratio *(*or *logit*), which, for a single predictor variable, *x*, has the following form:

          ![](img/ef09118a-1d49-472f-9b70-c9925e410036.png)

        Here, *p*(*x*) represents the probability of a true outcome in response to the given the predictor, *x*. Rearranging this gives a variation of the logistic function for the probability:

          ![](img/4edf13c3-10f1-42a0-a475-91264d1ea732.png)

        The parameters for the log odds are estimated using a maximum likelihood method. 
        The `LogisticRegression` class from the `linear_model` module in `scikit-learn` is an implementation of logistic regression that is very easy to use. First, we create a new model instance of this class, with any custom parameters that we need, and then use the `fit` method on this object to fit (or train) the model to the sample data. Once this fitting is done, we can access the parameters that have been estimated using the `get_params` method. 
        The `predict` method on the fitted model allows us to pass in new (unseen) data and make predictions about the classification of each sample. We could also get the probability estimates that are actually given by the logistic function using the `predict_proba` method.
        Once we have built a model for predicting the classification of data, we need to validate the model. This means we have to test the model with some previously unseen data and check whether it correctly classifies the new data. For this, we can use `classification_report`, which takes a new set of data and the predictions generated by the model and computes the proportion of the data that was correctly predicted by the model. This is the *precision* of the model.
        The classification report we generated using the `scikit-learn` utility performs a comparison between the predicted results and the known response values. This is a common method for validating a model before using it to make actual predictions. In this recipe, we saw that the reported precision for each of the categories (`True` and `False`) was 1.00, indicating that the model performed perfectly in predicting the classification with this data. In practice, it is unlikely that the precision of a model will be 100%. 
        There's more...
        There are lots of packages that offer tools for using logistic regression for classification problems. The statsmodels package has the `Logit` class for creating logistic regression models. We used the `scikit-learn` package in this recipe, which has a similar interface. `Scikit-learn` is a general-purpose machine learning library and has a variety of other tools for classification problems.
        Modeling time series data with ARMA
        Time series, as the name suggests, tracks a value over a sequence of distinct time intervals. They are particularly important in the finance industry, where stock values are tracked over time and used to make predictions – known as forecasting – of the value at some future time. Good predictions coming from such data can be used to make better investments. Time series also appear in many other common situations, such as weather monitoring, medicine, and any places where data is derived from sensors over time.
        Time series, unlike other types of data, do not usually have independent data points. This means that the methods that we use for modeling independent data will not be particularly effective. Thus, we need to use alternative techniques to model data with this property. There are two ways in which a value in a time series can depend on previous values. The first is where there is a direct relationship between the value and one or more previous values. This is the *autocorrelation* property and is modeled by an *autoregressive* model. The second is where the noise that's added to the value depends on one or more previous noise terms. This is modeled by a *moving average* model. The number of terms involved in either of these models is called the *order* of the model.
        In this recipe, we will learn how to create a model for stationary time series data with ARMA terms.
        Getting ready
        For this recipe, we need the Matplotlib `pyplot` module imported as `plt` and the statsmodels package `api` module imported as `sm`. We also need to import the `generate_sample_data` routine from the `tsdata` package from this book's repository, which uses NumPy and Pandas to generate sample data for analysis:

```

from tsdata import generate_sample_data

```py

        How to do it...
        Follow these steps to create an autoregressive moving average model for stationary time series data:

          1.  First, we need to generate the sample data that we will analyze:

```

sample_ts, _ = generate_sample_data()

```py

          2.  As always, the first step in the analysis is to produce a plot of the data so that we can visually identify any structure:

```

ts_fig, ts_ax = plt.subplots()

sample_ts.plot(ax=ts_ax, label="Observed")

ts_ax.set_title("Time series data")

ts_ax.set_xlabel("Date")

ts_ax.set_ylabel("Value")

```py

        The resulting plot can be seen in the following figure. Here, we can see that there doesn't appear to be an underlying trend, which means that the data is likely to be stationary:

          ![](img/518372fa-7c61-409c-90aa-f0a13459785f.png)

        Figure 7.4: Plot of the time series data that we will analyze. There doesn't appear to be a trend in this data

          3.  Next, we compute the augmented Dickey-Fuller test. The null hypothesis is that the time series is not stationary:

```

adf_results = sm.tsa.adfuller(sample_ts)

adf_pvalue = adf_results[1]

print("Augmented Dickey-Fuller test:\nP-value:", adf_pvalue)

```py

        The reported p value is 0.000376 in this case, so we reject the null hypothesis and conclude that the series is stationary.

          4.  Next, we need to determine the order of the model that we should fit. For this, we'll plot the **autocorrelation function** (**ACF**) and the **partial autocorrelation function**(**PACF**) for the time series:

```

ap_fig, (acf_ax, pacf_ax) = plt.subplots(2, 1, sharex=True,

tight_layout=True)

sm.graphics.tsa.plot_acf(sample_ts, ax=acf_ax,

title="Observed autocorrelation")

sm.graphics.tsa.plot_pacf(sample_ts, ax=pacf_ax,

title="Observed partial autocorrelation")

pacf_ax.set_xlabel("Lags")

pacf_ax.set_ylabel("Value")

acf_ax.set_ylabel("Value")

```py

        The plots of the ACF and PACF for our time series can be seen in the following figure. These plots suggest the existence of both autoregressive and moving average processes:

          ![](img/093fb316-c237-48f3-b47b-a62afcde7afa.png)

        Figure 7.5: ACF and PACF for the sample time series data

          5.  Next, we create an ARMA model for the data, using the `ARMA` class from statsmodels, `tsa` module. This model will have an order 1 AR and an order 1 MA:

```

arma_model = sm.tsa.ARMA(sample_ts, order=(1, 1))

```py

          6.  Now, we fit the model to the data and get the resulting model. We print a summary of these results to the Terminal:

```

arma_results = arma_model.fit()

print(arma_results.summary())

```py

        The summary data given for the fitted model is as follows:

```

ARMA Model Results

===================================================================

Dep. Variable: y           No. Observations: 366

Model: ARMA(1, 1)          Log Likelihood -513.038

方法：css-mle            创新的标准偏差 0.982

日期：2020 年 5 月 1 日     AIC 1034.077

时间：12:40:00             BIC 1049.687

Sample: 01-01-2020         HQIC 1040.280

- 12-31-2020

===================================================================

coef   std err     z     P>|z|    [0.025    0.975]

-------------------------------------------------------------------

const   -0.0242   0.143   -0.169    0.866   -0.305    0.256

ar.L1.y  0.8292   0.057    14.562   0.000    0.718    0.941

ma.L1.y -0.5189   0.090    -5.792   0.000   -0.695    -0.343

Roots

===================================================================

实部       虚部       模数      频率

-------------------------------------------------------------------

AR.1    1.2059     +0.0000j        1.2059        0.0000

MA.1    1.9271     +0.0000j        1.9271        0.0000

-------------------------------------------------------------------

```py

        Here, we can see that both of the estimated parameters for the AR and MA components are significantly different from 0\. This is because the value in the `P >|z|` column is 0 to 3 decimal places.

          7.  Next, we need to verify that there is no additional structure remaining in the residuals (error) of the predictions from our model. For this, we plot the ACF and PACF of the residuals:

```

residuals = arma_results.resid

rap_fig, (racf_ax, rpacf_ax) = plt.subplots(2, 1,

sharex=True, tight_layout=True)

sm.graphics.tsa.plot_acf(residuals, ax=racf_ax,

title="Residual autocorrelation")

sm.graphics.tsa.plot_pacf(residuals, ax=rpacf_ax,

标题="残差部分自相关")

rpacf_ax.set_xlabel("滞后")

rpacf_ax.set_ylabel("值")

racf_ax.set_ylabel("值")

```py

        The ACF and PACF of the residuals can be seen in the following figure. Here, we can see that there are no significant spikes at lags other than 0, so we conclude that there is no structure remaining in the residuals:

          ![](img/6dc54aaa-0b81-403f-b4d8-2a67cb360f1e.png)

        Figure 7.6: ACF and PACF for the residuals from our model

          8.  Now that we have verified that our model is not missing any structure, we plot the values that are fitted to each data point on top of the actual time series data to see whether the model is a good fit for the data. We plot this model in the plot we created in *step 2*:

```

fitted = arma_results.fittedvalues

fitted.plot(c="r", ax=ts_ax, 标签="拟合")

ts_ax.legend()

```py

        The updated plot can be seen in the following figure:

          ![](img/67b312d7-3986-4bf8-aba6-a534a0cf9b89.png)

        Figure 7.7: Plot of the fitted time series data over the observed time series data
        The fitted values give a reasonable approximation of the behavior of the time series, but reduce the noise from the underlying structure.
        How it works...
        A time series is stationary if it does not have a trend. They usually have a tendency to move in one direction rather than another. Stationary processes are important because we can usually remove the trend from an arbitrary time series and model the underlying stationary series. The ARMA model that we used in this recipe is a basic means of modeling the behavior of stationary time series. The two parts of an ARMA model are the autoregressive and moving average parts, which model the dependence of the terms and noise, respectively, on previous terms and noise.
        An order 1 autoregressive model has the following form:

          ![](img/7a4a2e01-301a-4448-9ad2-32ab71c37a97.png)

        Here, *φ[i]* represents the parameters and *ε[t]* is the noise at a given step. The noise is usually assumed to be normally distributed with a mean of 0 and a standard deviation that is constant across all the time steps. The *Y[t]* value represents the value of the time series at the time step, *t*. In this model, each value depends on the previous value, though it can also depend on some constants and some noise. The model will give rise to a stationary time series precisely when the *φ[1]* parameter lies strictly between -1 and 1.
        An order 1 moving average model is very similar to an autoregressive model and is given by the following equation:

          ![](img/c71efac3-1651-4fc9-932c-e3930e7d80e7.png)

        Here, the variants of *θ[i]* are parameters. Putting these two models together gives us an ARMA(1, 1) model, which has the following form:

          ![](img/fcb7eb3b-9d02-4a2e-b954-b260ba87f8e0.png)

        In general, we can have an ARMA(p, q) model that has an order *p* AR component and an order q MA component. We usually refer to the quantities, *p* and *q*, as the orders of the model.
        Determining the orders of the AR and MA components is the most tricky aspect of constructing an ARMA model. The ACF and PACF give some information toward this, but even then, it can be quite difficult. For example, an autoregressive process will show some kind of decay or oscillating pattern on the ACF as lag increases, and a small number of peaks on the PACF and values that are not significantly different from 0 beyond that. The number of peaks that appear on the PAF plot can be taken as the order of the process. For a moving average process, the reverse is true. There are usually a small number of significant peaks on the ACF plot, and a decay or oscillating pattern on the PACF plot. Of course, sometimes, this isn't obvious.
        In this recipe, we plotted the ACF and PACF for our sample time series data. In the autocorrelation plot in *Figure 7.5* (top), we can see that the peaks decay rapidly until they lie within the confidence interval of zero (meaning they are not significant). This suggests the presence of an autoregressive component. On the partial autocorrelation plot in *Figure 7.5* (bottom), we can see that there are only two peaks that can be considered not zero, which suggests an autoregressive process of order 1 or 2\. You should try to keep the order of the model as small as possible. Due to this, we chose an order 1 autoregressive component. With this assumption, the second peak on the partial autocorrelation plot is indicative of decay (rather than an isolated peak), which suggests the presence of a moving average process. To keep the model simple, we try an order 1 moving average process. This is how the model that we used in this recipe was decided on. Notice that this is not an exact process, and you might have decided differently. 
        We use the augmented Dickey-Fuller test to test the likelihood that the time series that we have observed is stationary. This is a statistical test, such as those seen in Chapter 6, *Working with Data and Statistics*, that generates a test statistic from the data. This test statistic, in turn, determines a p-value that is used to determine whether to accept or reject the null hypothesis. For this test, the null hypothesis is that a unit root is present in the time series that's been sampled. The alternative hypothesis – the one we are really interested in – is that the observed time series is (trend) stationary. If the p-value is sufficiently small, then we can conclude with the specified confidence that the observed time series is stationary. In this recipe, the p-value was 0.000 to 3 decimal places, which indicates a strong likelihood that the series is stationary. Stationarity is an essential assumption for using the ARMA model for the data.
        Once we have determined that the series is stationary, and also decided on the orders of the model, we have to fit the model to the sample data that we have. The parameters of the model are estimated using a maximum likelihood estimator. In this recipe, the learning of the parameters is done using the `fit` method, in *step 6*.
        The statsmodels package provides various tools for working with time series, including utilities for calculating – and plotting –ACF and PACF of time series data, various test statistics, and creating ARMA models for time series. There are also some tools for automatically estimating the order of the model.
        We can use the **Akaike information criterion** (**AIC**), **Bayesian information criterion** (**BIC**), and **Hannan-Quinn Information Criterion** (**HQIC**) quantities to compare this model to other models to see which model best describes the data. A smaller value is better in each case.
        When using ARMA to model time series data, as in all kinds of mathematical modeling tasks, it is best to pick the simplest model that describes the data to the extent that is needed. For ARMA models, this usually means picking the smallest order model that describes the structure of the observed data.
        There's more...
        Finding the best combination of orders for an ARMA model can be quite difficult. Often, the best way to fit a model is to test multiple different configurations and pick the order that produces the best fit. For example, we could have tried ARMA(0, 1) or ARMA(1, 0) in this recipe, and compared it to the ARMA(1, 1) model we used to see which produced the best fit by considering the **Akaike Information Criteria** (**AIC**) statistic reported in the summary. In fact, if we build these models, we will see that the AIC value for ARMA(1, 1) – the model we used in this recipe – is the "best" of these three models.
        Forecasting from time series data using ARIMA
        In the previous recipe, we generated a model for a stationary time series using an ARMA model, which consists of an **autoregressive** (**AR**) component and an **m****oving average** (**MA**) component. Unfortunately, this model cannot accommodate time series that have some underlying trend; that is, they are not stationary time series. We can often get around this by *differencing* the observed time series one or more times until we obtain a stationary time series that can be modeled using ARMA. The incorporation of differencing into an ARMA model is called an ARIMA model, which stands for **Autoregressive** (**AR**) **Integrated** (**I**) **Moving Average** (**MA**).
        Differencing is the process of computing the difference of consecutive terms in a sequence of data. So, applying first-order differencing amounts to subtracting the value at the current step from the value at the next step (*t[i+1] - t[i]*). This has the effect of removing the underlying upward or downward linear trend from the data. This helps to reduce an arbitrary time series to a stationary time series that can be modeled using ARMA. Higher-order differencing can remove higher-order trends to achieve similar effects.
        An ARIMA model has three parameters, usually labeled *p*, *d*, and *q*. The *p* and *q* order parameters are the order of the autoregressive component and the moving average component, respectively, just as they are for the ARMA model. The third order parameter, *d*, is the order of differencing to be applied. An ARIMA model with these orders is usually written as ARIMA (*p*, *d*, *q*). Of course, we will need to determine what order differencing should be included before we start fitting the model.
        In this recipe, we will learn how to fit an ARIMA model to a non-stationary time series and use this model to make forecasts about future values.
        Getting ready
        For this recipe, we will need the NumPy package imported as `np`, the Pandas package imported as `pd`, the Matplotlib `pyplot` module as `plt`, and the statsmodels `api` module imported as `sm`. We will also need the utility for creating sample time series data from the `tsdata` module, which is included in this book's repository:

```

从 tsdata 导入生成样本数据

```py

        How to do it...
        The following steps show you how to construct an ARIMA model for time series data and use this model to make forecasts:

          1.  First, we load the sample data using the `generate_sample_data` routine:

```

sample_ts，test_ts = generate_sample_data(trend=0.2, undiff=True)

```py

          2.  As usual, the next step is to plot the time series so that we can visually identify the trend of the data:

```

ts_fig, ts_ax = plt.subplots(tight_layout=True)

sample_ts.plot(ax=ts_ax, c="b", 标签="观察到的")

ts_ax.set_title("训练时间序列数据")

ts_ax.set_xlabel("日期")

ts_ax.set_ylabel("值")

```py

        The resulting plot can be seen in the following figure. As we can see, there is a clear upward trend in the data, so the time series is certainly not stationary:

          ![](img/a5081c24-e36c-48ac-86e5-a5ec2cfccee0.png)

        Figure 7.8: Plot of the sample time series. There is an obvious positive trend in the data.

          3.  Next, we difference the series to see if one level of differencing is sufficient to remove the trend:

```

差分=sample_ts.diff().dropna()

```py

          4.  Now, we plot the ACF and PACF for the differenced time series: 

```

ap_fig, (acf_ax, pacf_ax) = plt.subplots(1, 2,

tight_layout=True, sharex=True)

sm.graphics.tsa.plot_acf(diffs, ax=acf_ax)

sm.graphics.tsa.plot_pacf(diffs, ax=pacf_ax)

acf_ax.set_ylabel("值")

pacf_ax.set_xlabel("滞后")

pacf_ax.set_ylabel("值")

```py

        The ACF and PACF can be seen in the following figure. We can see that there does not appear to be any trends left in the data and that there appears to be both an autoregressive component and a moving average component:

          ![](img/dc3a6b97-bdd4-4c6c-822a-d40b00a0a2e8.png)

        Figure 7.9: ACF and PACF for the differenced time series

          5.  Now, we construct the ARIMA model with order 1 differencing, an autoregressive component, and a moving average component. We fit this to the observed time series and print a summary of the model:

```

model = sm.tsa.ARIMA(sample_ts, order=(1,1,1))

fitted = model.fit(trend="c")

print(fitted.summary())

```py

        The summary information that's printed looks as follows:

```

ARIMA 模型结果

==================================================================

因变量：D.y             观察次数：365

模型：ARIMA(1, 1, 1)          对数似然-512.905

方法：css-mle                创新的标准差 0.986

日期：星期六，2020 年 5 月 2 日         AIC 1033.810

时间：14:47:25                 BIC 1049.409

样本：2020 年 01 月 02 日             HQIC 1040.009

- 12-31-2020

==================================================================

coef     std err     z      P>|z|     [0.025    0.975]

------------------------------------------------------------------

const     0.9548    0.148     6.464    0.000     0.665    1.244

ar.L1.D.y 0.8342    0.056     14.992   0.000     0.725    0.943

ma.L1.D.y -0.5204   0.088    -5.903    0.000    -0.693   -0.348

根

==================================================================

实际      虚数       模数        频率

------------------------------------------------------------------

AR.1      1.1987      +0.0000j       1.1987          0.0000

MA.1      1.9216      +0.0000j       1.9216          0.0000

------------------------------------------------------------------

```py

        Here, we can see that all three of our estimated coefficients are significantly different from 0 due to the fact that all three have 0 to 3 decimal places in the `P>|z|` column.

          6.  Now, we can use the `forecast` method to generate predictions of future values. This also returns the standard error and confidence intervals for predictions:

```

forecast, std_err, fc_ci = fitted.forecast(steps=50)

forecast_dates = pd.date_range("2021-01-01", periods=50)

forecast = pd.Series(forecast, index=forecast_dates)

```py

          7.  Next, we plot the forecast values and their confidence intervals on the figure containing the time series data:

```

forecast.plot(ax=ts_ax, c="g", 标签="预测")

ts_ax.fill_between(forecast_dates, fc_ci[:, 0], fc_ci[:, 1],

颜色="r", alpha=0.4)

```py

          8.  Finally, we add the actual future values to generate, along with the sample in *step 1*, to the plot (it might be easier if you repeat the plot commands from *step 1* to regenerate the whole plot here):

```

test_ts.plot(ax=ts_ax, c="k", 标签="实际")

ts_ax.legend()

```py

        The final plot containing the time series with the forecast and the actual future values can be seen in the following figure:

          ![](img/76b414a5-1847-4944-8811-28677d9e3853.png)

        Figure 7.10: Plot of the sample time series with forecast values and actual future values for comparison
        Here, we can see that the actual future values are within the confidence interval for the forecast values.
        How it works...
        The ARIMA model – with orders *p*, *d*, and *q –* is simply an ARMA (*p*, *q*) model that's applied to a time series. This is obtained by applying differencing of order *d* to the original time series data. It is a fairly simple way to generate a model for time series data. The statsmodels `ARIMA` class handles the creation of a model, while the `fit` method fits this model to the data. We passed the `trend="c"` keyword argument because we know, from *Figure 7.9*, that the time series has a constant trend.
        The model is fit to the data using a maximum likelihood method and the final estimates for the parameters – in this case, one parameter for the autoregressive component, one for the moving average component, the constant trend parameter, and the variance of the noise. These parameters are reported in the summary. From this output, we can see that the estimates for the AR coefficient (0.8342) and the MA constant (-0.5204) are very good approximations of the true estimates that were used to generate the data, which were 0.8 for the AR coefficient and -0.5  for the MA coefficient. These parameters are set in the `generate_sample_data` routine from the `tsdata.py` file in the code repository for this chapter. This generates the sample data in *step 1*. You might have noticed that the constant parameter (0.9548) is not 0.2, as specified in the `generate_sample_data` call in *step 1*. In fact, it is not so far from the actual drift of the time series.
        The `forecast` method on the fitted model (the output of the `fit` method) uses the model to make predictions about the value after a given number of steps. In this recipe, we forecast for up to 50 time steps beyond the range of the sample time series. The output of the `forecast` method is a tuple containing the forecast values, the standard error for the forecasts, and the confidence interval (by default, 95% confidence) for the forecasts. Since we provided the time series as a Pandas series, these are returned as `Series` objects (the confidence interval is a `DataFrame`).
        When you construct an ARIMA model for time series data, you need to make sure you use the smallest order differencing that removes the underlying trend. Applying more differencing than is necessary is called *overdifferencing* and can lead to problems with the model.
        Forecasting seasonal data using ARIMA
        Time series often display periodic behavior so that peaks or dips in the value appear at regular intervals. This behavior is called *seasonality* in the analysis of time series. The methods we have used to far in this chapter to model time series data obviously do not account for seasonality. Fortunately, it is relatively easy to adapt the standard ARIMA model to incorporate seasonality, resulting in what is sometimes called a SARIMA model.
        In this recipe, we will learn how to model time series data that includes seasonal behavior and use this model to produce forecasts.
        Getting ready
        For this recipe, we will need the NumPy package imported as `np`, the Pandas package imported as `pd`, the Matplotlib `pyplot`module as `plt`, and the statsmodels `api`module imported as `sm`. We will also need the utility for creating sample time series data from the `tsdata`module, which is included in this book's repository:

```

从 tsdata 导入生成样本数据

```py

        How to do it...
        Follow these steps to produce a seasonal ARIMA model for sample time series data and use this model to produce forecasts:

          1.  First, we use the `generate_sample_data` routine to generate a sample time series to analyze:

```

sample_ts，test_ts = generate_sample_data(undiff=True，

季节性=True)

```py

          2.  As usual, our first step is to visually inspect the data by producing a plot of the sample time series:

```

ts_fig, ts_ax = plt.subplots(tight_layout=True)

sample_ts.plot(ax=ts_ax, 标题="时间序列", 标签="观察到的")

ts_ax.set_xlabel("日期")

ts_ax.set_ylabel("值")

```py

        The plot of the sample time series data can be seen in the following figure. Here, we can see that there seem to be periodic peaks in the data:

          ![](img/f4ae55a8-7e3a-489b-b5bc-987923b794ab.png)

        Figure 7.11: Plot of the sample time series data

          3.  Next, we plot the ACF and PACF for the sample time series:

```

ap_fig，(acf_ax，pacf_ax) = plt.subplots(2, 1，

sharex=True, tight_layout=True)

sm.graphics.tsa.plot_acf(sample_ts, ax=acf_ax)

sm.graphics.tsa.plot_pacf(sample_ts, ax=pacf_ax)

pacf_ax.set_xlabel("滞后")

acf_ax.set_ylabel("值")

pacf_ax.set_ylabel("值")

```py

        The ACF and PACF for the sample time series can be seen in the following figure:

          ![](img/0d6132a1-0b39-4f7b-bf1a-0da288ad8c3c.png)

        Figure 7.12: ACF and PACF for the sample time series
        These plots possibly indicate the existence of autoregressive components, but also a significant spike on the PACF with lag 7.

          4.  Next, we difference the time series and produce plots of the ACF and PACF for the differenced series. This should make the order of the model clearer:

```

差分=sample_ts.diff().dropna()

dap_fig, (dacf_ax, dpacf_ax) = plt.subplots(2, 1, sharex=True,

tight_layout=True)

sm.graphics.tsa.plot_acf(diffs, ax=dacf_ax,

标题="差分 ACF")

sm.graphics.tsa.plot_pacf(diffs, ax=dpacf_ax,

标题="差分 PACF")

dpacf_ax.set_xlabel("滞后")

dacf_ax.set_ylabel("值")

dpacf_ax.set_ylabel("值")

```py

        The ACF and PACF for the differenced time series can be seen in the following figure. We can see that there is definitely a seasonal component with lag 7:

          ![](img/7e2f3382-2b4e-4423-9d38-0e0d4332f9b8.png)

        Figure 7.13: Plot of the ACF and PACF for the differenced time series

          5.  Now, we need to create a `SARIMAX` object that holds the model, with ARIMA order `(1, 1, 1)` and seasonal ARIMA order `(1, 0, 0, 7)`. We fit this model to the sample time series and print summary statistics. We plot the predicted values on top of the time series data:

```

model = sm.tsa.SARIMAX(sample_ts, order=(1, 1, 1),

季节性顺序=(1, 0, 0, 7))

fitted_seasonal = model.fit()

print(fitted_seasonal.summary())

fitted_seasonal.fittedvalues.plot(ax=ts_ax, c="r",

标签="预测")

```py

        The summary statistics that are printed to the Terminal look as follows:

```

SARIMAX 结果

===================================================================

因变量：y                      观察次数：366

模型：SARIMAX(1, 1, 1)x(1, 0, [], 7) 对数似然-509.941

日期：星期一，2020 年 5 月 4 日                AIC 1027.881

时间：18:03:27                        BIC 1043.481

样本：2020 年 01 月 01 日                    HQIC 1034.081

- 12-31-2020

协方差类型：                      opg

===================================================================

coef     std err     z       P>|z|      [0.025     0.975]

-------------------------------------------------------------------

ar.L1   0.7939    0.065     12.136    0.000      0.666     0.922

ma.L1   -0.4544   0.095    -4.793     0.000     -0.640    -0.269

ar.S.L7  0.7764   0.034     22.951    0.000      0.710     0.843

sigma2   0.9388   0.073     12.783    0.000      0.795     1.083

===================================================================

Ljung-Box (Q): 31.89                Jarque-Bera (JB): 0.47

Prob(Q): 0.82                       Prob(JB): 0.79

异方差性（H）: 1.15        偏度: -0.03

Prob(H) (双侧): 0.43           峰度: 2.84

===================================================================

警告:

[1] 使用外积计算的协方差矩阵

梯度的数量（复杂步骤）。

```py

          6.  This model appears to be a reasonable fit, so we move ahead and forecast `50` time steps into the future:

```

forecast_result = fitted_seasonal.get_forecast(steps=50)

forecast_index = pd.date_range("2021-01-01", periods=50)

预测 = 预测结果.预测均值

```py

          7.  Finally, we add the forecast values to the plot of the sample time series, along with the confidence interval for these forecasts:

```

forecast.plot(ax=ts_ax, c="g", label="预测")

conf = forecast_result.conf_int()

ts_ax.fill_between(forecast_index, conf["lower y"],

conf["upper y"], color="r", alpha=0.4)

test_ts.plot(ax=ts_ax, color="k", label="实际未来")

ts_ax.legend()

```py

        The final plot of the time series, along with the predictions and the confidence interval for the forecasts, can be seen in the following figure:

          ![](img/54ae8641-4c49-464e-9183-9c56e95c1145.png)

        Figure 7.14: Plot of the sample time series, along with the forecasts and confidence interval
        How it works...
        Adjusting an ARIMA model to incorporate seasonality is a relatively simple task. A seasonal component is similar to an autoregressive component, where the lag starts at some number larger than 1\. In this recipe, the time series exhibits seasonality with period 7 (weekly), which means that the model is approximately given by the following equation:

          ![](img/0a72f02a-4314-4245-b1cf-92c84d07882e.png)

        Here *φ[1]* and *Φ**[1]**are the parameters and *ε[t]* is the noise at time step *t*. The standard ARIMA model is easily adapted to include this additional lag term.* *The SARIMA model incorporates this additional seasonality into the ARIMA model. It has four additional order terms on top of the three for the underlying ARIMA model. These four additional parameters are the seasonal AR, differencing, and MA components, along with the period of the seasonality. In this recipe, we took the seasonal AR to be order 1, with no seasonal differencing or MA components (order 0), and a seasonal period of 7\. This gives us the additional parameters (1, 0, 0, 7) that we used in *step 5* of this recipe.

Seasonality is clearly important in modeling time series data that is measured over a period of time covering days, months, or years. It usually incorporates some kind of seasonal component based on the time frame that they occupy. For example, a time series of national power consumption measured hourly over several days would probably have a 24-hour seasonal component since power consumption will likely fall during the night hours.

Long-term seasonal patterns might be hidden if the time series data that you are analyzing does not cover a sufficiently large time period for the pattern to emerge. The same is true for trends in the data. This can lead to some interesting problems when trying to produce long-term forecasts from a relatively short period represented by observed data.

The `SARIMAX` class from the statsmodels package provides the means of modeling time series data using a seasonal ARIMA model. In fact, it can also model external factors that have an additional effect on the model, sometimes called *exogenous regressors*. (We will not cover these here.) This class works much like the `ARMA` and `ARIMA` classes that we used in the previous recipes. First, we create the model object by providing the data and orders for both the ARIMA process and the seasonal process, and then use the `fit` method on this object to create a fitted model object. We use the `get_forecasts` method to generate an object holding the forecasts and confidence interval data that we can then plot, thus producing the *Figure 7.14*.

## There's more...

There is a small difference in the interface between the `SARIMAX` class used in this recipe and the `ARIMA` class used in the previous recipe. At the time of writing, the statsmodels package (v0.11) includes a second `ARIMA` class that builds on top of the `SARIMAX` class, thus providing the same interface. However, at the time of writing, this new `ARIMA` class does not offer the same functionality as that used in this recipe.

# Using Prophet to model time series data 

The tools we have seen so far for modeling time series data are very general and flexible methods, but they require some knowledge of time series analysis in order to be set up. The analysis needed to construct a good model that can be used to make reasonable predictions into the future can be intensive and time-consuming, and may not be viable for your application. The Prophet library is designed to automatically model time series data quickly, without the need for input from the user, and make predictions into the future.

In this recipe, we will learn how to use Prophet to produce forecasts from a sample time series.

## Getting ready

For this recipe, we will need the Pandas package imported as `pd`, the Matplotlib `pyplot` package imported as `plt`, and the `Prophet` object from the Prophet library, which can be imported using the following command:

```

from fbprophet import Prophet

```py

We also need to import the `generate_sample_data` routine from the `tsdata` module, which is included in the code repository for this book:

```

from tsdata import generate_sample_data

```py

## How to do it...

The following steps show you how to use the Prophet package to generate forecasts for a sample time series:

1.  First, we use `generate_sample_data` to generate the sample time series data:

```

sample_ts, test_ts = generate_sample_data(undiff=True, trend=0.2)

```py

2.  We need to convert the sample data into a `DataFrame` that Prophet expects:

```

df_for_prophet = pd.DataFrame({

"ds": sample_ts.index,   # dates

"y": sample_ts.values    # values

})

```py

3.  Next, we make a model using the `Prophet` class and fit it to the sample time series:

```

model = Prophet()

model.fit(df_for_prophet)

```py

4.  Now, we create a new `DataFrame` that contains the time intervals for the original time series, plus the additional periods for the forecasts:

```

forecast_df = model.make_future_dataframe(periods=50)

```py

5.  Then, we use the `predict` method to produce the forecasts along the time periods we just created:

```

forecast = model.predict(forecast_df)

```py

6.  Finally, we plot the predictions on top of the sample time series data, along with the confidence interval and the true future values:

```

fig, ax = plt.subplots(tight_layout=True)

sample_ts.plot(ax=ax, label="观察到的", title="预测")

forecast.plot(x="ds", y="yhat", ax=ax, c="r",

label="预测")

ax.fill_between(forecast["ds"].values, forecast["yhat_lower"].values,

forecast["yhat_upper"].values, color="r", alpha=0.4)

test_ts.plot(ax=ax, c="k", label="未来")

ax.legend()

ax.set_xlabel("日期")

ax.set_ylabel("值")

```py

The plot of the time series, along with forecasts, can be seen in the following figure:

          ![](img/9a9b3e93-8bbb-491f-8f99-2357a722754a.png)

        Figure 7.15: Plot of sample time series data, along with forecasts and a confidence interval

## How it works...

Prophet is a package that's used to automatically produce models for time series data based on sample data, with little extra input needed from the user. In practice, it is very easy to use; we just need to create an instance of the `Prophet` class, call the `fit` method, and then we are ready to produce forecasts and understand our data using the model.

The `Prophet` class expects the data in a specific format: a `DataFrame` with columns named `ds` for the date/time index, and `y` for the response data (the time series values). This `DataFrame` should have integer indices. Once the model has been fit, we use `make_future_dataframe` to create a `DataFrame` in the correct format, with appropriate date intervals, and with additional rows for future time intervals. The `predict` method then takes this `DataFrame` and produces values using the model to populate these time intervals with predicted values. We also get other information, such as the confidence intervals, in this forecast's `DataFrame`.

## There's more...

Prophet does a fairly good job of modeling time series data without any input from the user. However, the model can be customized using various methods from the `Prophet` class. For example, we could provide information about the seasonality of the data using the `add_seasonality` method of the `Prophet` class, prior to fitting the model.

There are alternative packages for automatically generating models for time series data. For example, popular machine learning libraries such as TensorFlow can be used to model time series data.

# Further reading

A good textbook on regression in statistics is the book *Probability and Statistics* by Mendenhall, Beaver, and Beaver, as mentioned in Chapter 6, *Working with Data and Statistics*. The following books provide a good introduction to classification and regression in modern data science:

*   *James, G. and Witten, D., 2013\. An Introduction To Statistical Learning: With Applications In R. New York: Springer.*

*   *Müller, A. and Guido, S., 2016\. Introduction To Machine Learning With Python. Sebastopol: O'Reilly Media.*

A good introduction to time series analysis can be found in the following book:

*   *Cryer, J. and Chan, K., 2008\. Time Series Analysis. New York: Springer.** 
```**
