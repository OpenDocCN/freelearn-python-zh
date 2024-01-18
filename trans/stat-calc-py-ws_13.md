# 附录

# 1. Python 基础

## 活动 1.01：构建数独求解器

**解决方案**

1.  首先，我们定义`Solver`类，将其输入谜题存储在其`cells`属性中，如下所示：

```py
    from copy import deepcopy
    class Solver:
        def __init__(self, input_path):
            # Read in the input file and initialize the puzzle
            with open(input_path, 'r') as f:
                lines = f.readlines()
            self.cells = [list(map(int, line.split(','))) \
                          for line in lines]
    ```

1.  以漂亮的格式打印出谜题的辅助方法可以循环遍历谜题中的单元格，同时在适当的位置插入分隔字符`'-'`和`'|'`：

```py
        # Print out the initial puzzle or solution in a nice format.
        def display_cell(self):
            print('-' * 23)
            for i in range(9):
                for j in range(9):
                    print(self.cells[i][j], end=' ')
                    if j % 3 == 2:
                        print('|', end=' ')
                print()
                if i % 3 == 2:
                    print('-' * 23)
            print()
    ```

1.  `get_presence（）`方法可以维护三个单独的布尔变量列表，用于表示各行、列和象限中 1 到 9 之间数字的存在。这些布尔变量在开始时都应初始化为`False`，但我们可以循环遍历输入中的所有单元格，并根据需要将它们的值更改为`True`：

```py
            """ 
            True/False for whether a number is present in a row, 
            column, or quadrant.
            """
            def get_presence(cells):
                present_in_row = [{num: False for num in range(1, 10)}
                                  for _ in range(9)]
                present_in_col = [{num: False for num in range(1, 10)}
                                  for _ in range(9)]
                present_in_quad = [{num: False for num in range(1, 10)}
                                   for _ in range(9)]
                for row_id in range(9):
                    for col_id in range(9):
                        temp_val = cells[row_id][col_id]
                        """
                        If a cell is not empty, update the corresponding 
                        row, column, and quadrant.
                        """
                        if temp_val > 0:
                            present_in_row[row_id][temp_val] = True
                            present_in_col[col_id][temp_val] = True
                            present_in_quad[row_id // 3 * 3 \
                                            + col_id // 3]\
                                            [temp_val] = True
                return present_in_row, present_in_col, present_in_quad
    ```

对象象限进行索引可能有些棘手。上述代码使用公式`row_id // 3 * 3 + col_id // 3`，这实际上导致了从左上象限索引为`0`，上中`1`，上右`2`，中左`3`，...，底部中`7`，底部右`8`的计数。

1.  `get_possible_values（）`方法可以调用`get_presence（）`并生成剩余空单元格的可能值列表：

```py
            # A dictionary for empty locations and their possible values.
            def get_possible_values(cells):
                present_in_row, present_in_col, \
                present_in_quad = get_presence(cells)
                possible_values = {}
                for row_id in range(9):
                    for col_id in range(9):
                        temp_val = cells[row_id][col_id]
                        if temp_val == 0:
                            possible_values[(row_id, col_id)] = []
                            """ 
                            If a number is not present in the same row, 
                            column, or quadrant as an empty cell, add it 
                            to the list of possible values of that cell.
                            """
                            for num in range(1, 10):
                                if (not present_in_row[row_id][num]) and\
                                   (not present_in_col[col_id][num]) and\
                                   (not present_in_quad[row_id // 3 * 3 \
                                   + col_id // 3][num]):
                                    possible_values[(row_id, col_id)]\
                                    .append(num)
                return possible_values
    ```

1.  `simple_update（）`方法可以以相当直接的方式实现，其中我们可以使用一个标志变量（这里称为`update_again`）来指示我们在返回之前是否需要再次调用该方法：

```py
            # Fill in empty cells that have only one possible value.
            def simple_update(cells):
                update_again = False
                possible_values = get_possible_values(cells)
                for row_id, col_id in possible_values:
                    if len(possible_values[(row_id, col_id)]) == 1:
                        update_again = True
                        cells[row_id][col_id] = possible_values[\
                                                (row_id, col_id)][0]
                """
                Recursively update with potentially new possible values.
                """
                if update_again:
                    cells = simple_update(cells)
                return cells
    ```

1.  `recur_solve（）`方法包含多个教学组件，但逻辑流程简单易实现：

```py
            # Recursively solve the puzzle
            def recur_solve(cells):
                cells = simple_update(cells)
                possible_values = get_possible_values(cells)
                if len(possible_values) == 0:
                    return cells  # return when all cells are filled
                # Find the empty cell with fewest possible values.
                fewest_num_values = 10
                for row_id, col_id in possible_values:
                    if len(possible_values[(row_id, col_id)]) == 0:
                        return False  # return if an empty is invalid
                    if len(possible_values[(row_id, col_id)]) \
                       < fewest_num_values:
                        fewest_num_values = len(possible_values[\
                                                (row_id, col_id)])
                        target_location = (row_id, col_id)
                for value in possible_values[target_location]:
                    dup_cells = deepcopy(cells)
                    dup_cells[target_location[0]]\
                             [target_location[1]] = value
                    potential_sol = recur_solve(dup_cells)
                    # Return immediately when a valid solution is found.
                    if potential_sol:
                        return potential_sol
                return False  # return if no valid solution is found
    ```

1.  最后，我们将所有这些方法放在`solve（）`方法中，该方法在`self.cells`上调用`recur_solve（）`：

```py
        # Functions to find a solution.
        def solve(self):
            def get_presence(cells):
                ...
            def get_possible_values(cells):
                ...
            def simple_update(cells):
                ...
            def recur_solve(cells):
                ...
            print('Initial puzzle:')
            self.display_cell()
            final_solution = recur_solve(self.cells)
            if final_solution is False:
                print('A solution cannot be found.')
            else:
                self.cells = final_solution
                print('Final solution:')
                self.display_cell()
    ```

1.  按如下方式打印返回的解决方案：

```py
    solver = Solver('sudoku_input/sudoku_input_2.txt')
    solver.solve()
    ```

输出的一部分如下：

```py
    Initial puzzle:
    -----------------------
    0 0 3 | 0 2 0 | 6 0 0 | 
    9 0 0 | 3 0 5 | 0 0 1 | 
    0 0 1 | 8 0 6 | 4 0 0 | 
    -----------------------
    0 0 8 | 1 0 2 | 9 0 0 | 
    7 0 0 | 0 0 0 | 0 0 8 | 
    0 0 6 | 7 0 8 | 2 0 0 | 
    -----------------------
    0 0 2 | 6 0 9 | 5 0 0 | 
    8 0 0 | 2 0 3 | 0 0 9 | 
    0 0 5 | 0 1 0 | 3 0 0 | 
    -----------------------
    ```

注意

要访问此特定部分的源代码和最终输出，请参阅[`packt.live/3dWRsnE.`](https://packt.live/3dWRsnE )

您还可以在[`packt.live/2BBKreC.`](https://packt.live/2BBKreC )上在线运行此示例

# 2. Python 的统计主要工具

## 活动 2.01：分析社区和犯罪数据集

**解决方案**：

1.  数据集下载后，可以导入库，并使用 pandas 在新的 Jupyter 笔记本中读取数据集，如下所示：

```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df = pd.read_csv('CommViolPredUnnormalizedData.txt')
    df.head()
    ```

我们还打印出数据集的前五行，应该如下所示：

![图 2.21：数据集的前五行](img/B15968_02_211.jpg)

图 2.21：数据集的前五行

1.  要打印列名，我们可以简单地在`for`循环中迭代`df.columns`，如下所示：

```py
    for column in df.columns:
        print(column)
    ```

1.  可以使用 Python 中的`len（）`函数计算数据集中的列总数：

```py
    print(len(df.columns))
    ```

1.  要用`np.nan`对象替换特殊字符`'?'`，可以使用`replace（）`方法：

```py
    df = df.replace('?', np.nan)
    ```

1.  要打印出数据集中列的列表及其各自的缺失值数量，我们使用`isnull().sum()`方法的组合：

```py
    df.isnull().sum()
    ```

上述代码应产生以下输出：

```py
    communityname             0
    state                     0
    countyCode             1221
    communityCode          1224
    fold                      0
                           ... 
    autoTheftPerPop           3
    arsons                   91
    arsonsPerPop             91
    ViolentCrimesPerPop     221
    nonViolPerPop            97
    Length: 147, dtype: int64
    ```

1.  可以按如下方式访问并显示两个指定列的缺失值数量：

```py
    print(df.isnull().sum()['NumStreet'])
    print(df.isnull().sum()['PolicPerPop'])
    ```

您应该获得`0`和`1872`作为输出。

1.  使用条形图计算和可视化`'state'`中唯一值的计数（以及调整图的大小），可以使用以下代码：

```py
    state_count = df['state'].value_counts()
    f, ax = plt.subplots(figsize=(15, 10))
    state_count.plot.bar()
    plt.show()
    ```

这应该产生以下图表：

![图 2.22：州计数的条形图](img/B15968_02_221.jpg)

图 2.22：州计数的条形图

1.  使用以下代码可以计算和可视化相同信息的饼图：

```py
    f, ax = plt.subplots(figsize=(15, 10))
    state_count.plot.pie()
    plt.show()
    ```

将生成以下可视化：

![图 2.23：州计数的饼图](img/B15968_02_23.jpg)

图 2.23：州计数的饼图

1.  使用直方图计算和可视化人口分布，可以使用以下代码：

```py
    f, ax = plt.subplots(figsize=(15, 10))
    df['population'].hist(bins=200)
    plt.show()
    ```

这应该产生以下图表：

![图 2.24：人口分布的直方图](img/B15968_02_24.jpg)

图 2.24：人口分布的直方图

1.  要计算和可视化家庭规模分布，可以使用以下代码：

```py
    f, ax = plt.subplots(figsize=(15, 10))
    df['householdsize'].hist(bins=200)
    plt.show()
    ```

这应该产生以下图表：

![图 2.25：家庭规模分布的直方图](img/B15968_02_25.jpg)

图 2.25：家庭规模分布的直方图

注意

要访问此特定部分的源代码，请参阅[`packt.live/2BB5BJT`](https://packt.live/2BB5BJT)

你也可以在[`packt.live/38nbma9`](https://packt.live/38nbma9)上线运行此示例。

# 3. Python 的统计工具箱

## 活动 3.01：重新审视社区和犯罪数据集

**解决方案**

1.  可以导入库，并使用 pandas 读取数据集，如下所示：

```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv('CommViolPredUnnormalizedData.txt')
    df.head()
    ```

您的输出应该如下所示：

![图 3.29：数据集的前五行](img/B15968_03_29.jpg)

图 3.29：数据集的前五行

1.  要用`np.nan`对象替换特殊字符，我们可以使用以下代码：

```py
    df = df.replace('?', np.nan)
    ```

1.  要计算不同年龄组的实际计数，我们可以简单地使用表达式`df['population'] * df['agePct...']`，以向量化方式计算计数：

```py
    age_groups = ['12t21', '12t29', '16t24', '65up']

    for group in age_groups:
        df['ageCnt' + group] = (df['population'] * \
                                df['agePct' + group]).astype(int)
    df[['population'] \
      + ['agePct' + group for group in age_groups] \
      + ['ageCnt' + group for group in age_groups]].head()
    ```

请注意，我们正在使用`astype(int)`将最终答案四舍五入为整数。这些新创建的列的前五行应该如下所示：

![图 3.30：不同年龄组的实际计数](img/B15968_03_30.jpg)

图 3.30：不同年龄组的实际计数

1.  表达式`df.groupby('state')`给我们一个`GroupBy`对象，将我们的数据集聚合成不同的组，每个组对应`'state'`列中的唯一值。然后我们可以在该对象上调用`sum()`并检查相关的列：

```py
    group_state_df = df.groupby('state')
    group_state_df.sum()[['ageCnt' + group for group in age_groups]]
    ```

这应该打印出每个州不同年龄组的计数。输出的前五列应该如下所示：

![图 3.31：每个州不同年龄组的计数](img/B15968_03_31.jpg)

图 3.31：每个州不同年龄组的计数

1.  使用`df.describe()`方法，您可以获得以下输出：![图 3.32：数据集的描述](img/B15968_03_32.jpg)

图 3.32：数据集的描述

1.  可生成可视化各种犯罪数量的箱线图，如下所示：

```py
    crime_df = df[['burglPerPop','larcPerPop',\
                   'autoTheftPerPop', 'arsonsPerPop',\
                   'nonViolPerPop']]
    f, ax = plt.subplots(figsize=(13, 10))
    sns.boxplot(data=crime_df)
    plt.show()
    ```

这应该产生以下图表：

![图 3.33：各种犯罪数量的箱线图](img/B15968_03_33.jpg)

图 3.33：各种犯罪数量的箱线图

1.  从图表中，我们可以看到五种犯罪中非暴力犯罪是最常见的，而纵火犯罪是最不常见的。

1.  可以使用与给定列对应的相关矩阵的热图来可视化所需的信息：

```py
    feature_columns = ['PctPopUnderPov', 'PctLess9thGrade', \
                       'PctUnemployed', 'ViolentCrimesPerPop', \
                       'nonViolPerPop']
    filtered_df = df[feature_columns]
    f, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(filtered_df.dropna().astype(float).corr(), \
                                     center=0, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
    ```

这应该产生以下热图：

![图 3.34：各种人口特征的热图](img/B15968_03_34.jpg)

图 3.34：各种人口特征的热图

从图表中，我们可以看到贫困线以下人口的百分比和失业率之间高度相关（相关系数为`0.77`）。这是一个可以理解但富有启发性的洞察，揭示了各种与犯罪相关的因素是如何相互联系的。

注意

要访问此特定部分的源代码，请参阅[`packt.live/3f8taZn`](https://packt.live/3f8taZn)

你也可以在[`packt.live/3ikxjeF`](https://packt.live/3ikxjeF)上线运行此示例。

# 4. 使用 Python 进行函数和代数

## 活动 4.01：多变量收支平衡分析

**解决方案**：

1.  让*x*表示餐厅每月生产的汉堡数量，*y*表示每个汉堡的价格。然后，月收入将是*xy*，成本将是*6.56x + 1312.13*，最后，总利润将是两者之间的差异：*xy - 6.56x - 1312.13*。

1.  要达到收支平衡，生产的汉堡数量*x*必须等于需求，这给我们带来了方程：*x = 4000/y*。此外，总利润应该为零，这导致*xy - 6.56x = 1312.13*。

总的来说，我们有以下方程组：

![图 4.48：方程组](img/B15968_04_48.jpg)

图 4.48：方程组

1.  从第一个方程中，我们可以解出*x = 409.73628*。将这个值代入第二个方程，我们可以解出*y = 9.76237691*。

要在 Python 中解决这个系统，我们首先声明我们的变量和常数：

```py
    COST_PER_BURGER = 6.56
    FIXED_COST = 1312.13
    AVG_TOWN_BUDGET = 4000
    x = Symbol('x')  # number of burgers to be sold
    y = Symbol('y')  # price of a burger
    ```

然后我们可以在对应的函数列表上调用 SymPy 中的`solve()`函数：

```py
    solve([x * (y - COST_PER_BURGER) - FIXED_COST,\
           x * y - AVG_TOWN_BUDGET])
    ```

这段代码应该产生以下输出，对应于系统的实际解决方案：

```py
    [{x: 409.736280487805, y: 9.76237690066856}]
    ```

1.  这个函数最具挑战性的一点是，如果餐厅生产的汉堡数量*x*超过需求*4000/y*，他们的收入仍然是*4000*。然而，如果汉堡的数量较少，那么收入就是*xy*。因此，我们的函数需要有一个条件来检查这个逻辑：

```py
    def get_profit(x, y):
        demand = AVG_TOWN_BUDGET / y
        if x > demand:
            return AVG_TOWN_BUDGET - x * COST_PER_BURGER \
                                       - FIXED_COST

        return x * (y - COST_PER_BURGER) - FIXED_COST
    ```

1.  以下代码生成了指定的列表和相应的图表，当每个汉堡的价格为$9.76 时：

```py
    xs = [i for i in range(300, 501)]
    profits_976 = [get_profit(x, 9.76) for x in xs]
    plt.plot(xs, profits_976)
    plt.axhline(0, c='k')
    plt.xlabel('Number of burgers produced')
    plt.ylabel('Profit')
    plt.show()
    ```

输出应该如下所示：

![图 4.49：售价为$9.76 的盈亏平衡图](img/B15968_04_49.jpg)

图 4.49：售价为$9.76 的盈亏平衡图

利润曲线的倒置 V 形与水平线在`0`处的交点表示分析中每个汉堡价格固定为$9.76 的盈亏平衡点。这个交点的*x*坐标略高于`400`，大致对应于*步骤 3*中的盈亏平衡解，当*x*大约为`410`，*y*大约为 9.76 时。

1.  以下代码生成了指定的列表和相应的图表，当每个汉堡的价格为$9.99 时：

```py
    xs = [i for i in range(300, 501)]
    profits_999 = [get_profit(x, 9.99) for x in xs]
    plt.plot(xs, profits_999)
    plt.axhline(0, c='k')
    plt.xlabel('Number of burgers produced')
    plt.ylabel('Profit')
    plt.show()
    ```

输出应该如下所示：

![图 4.50：售价为$9.99 的盈亏平衡图](img/B15968_04_50.jpg)

图 4.50：售价为$9.99 的盈亏平衡图

类似地，利润曲线与水平线交点处的两个交点`0`表示分析中每个汉堡价格固定为$9.99 的盈亏平衡点。

我们看到，随着生产的汉堡数量增加，餐厅的利润呈线性增长。然而，当这个数量满足需求并且利润曲线达到峰值后，曲线开始线性下降。这是当餐厅过度生产并且增加产品数量不再有利时。

1.  以下代码生成了指定的列表：

```py
    xs = [i for i in range(300, 501, 2)]
    ys = np.linspace(5, 10, 100)
    profits = [[get_profit(x, y) for y in ys] for x in xs]
    ```

`profits`是一个相当大的二维列表，但该列表中的前几个元素应如下所示：

![图 4.51：利润的二维列表](img/B15968_04_51.jpg)

图 4.51：利润的二维列表

1.  可以使用以下代码生成指定的热图：

```py
    plt.imshow(profits)
    plt.colorbar()
    plt.xticks([0, 20, 40, 60, 80],\
               [5, 6, 7, 8, 9, 10])
    plt.xlabel('Price for each burger')
    plt.yticks([0, 20, 40, 60, 80],\
               [300, 350, 400, 450, 500])
    plt.ylabel('Number of burgers produced')
    plt.show()
    ```

输出应该如下所示：

![图 4.52：利润的热图作为生产和价格的函数](img/B15968_04_52.jpg)

图 4.52：利润的热图作为生产和价格的函数

从图中我们可以看出，有特定的*x*和*y*的组合来控制餐厅的利润行为。

例如，当每个汉堡的价格较低时（地图的左侧区域），总利润明显低于 0。当我们移动到图的右侧时，最亮的区域代表了两个变量的组合将产生最高利润。

注意

要访问此特定部分的源代码，请参阅[`packt.live/2C6dKWz`](https://packt.live/2C6dKWz)。

您也可以在[`packt.live/2NTfEwG`](https://packt.live/2NTfEwG)上在线运行此示例。

# 5\. 使用 Python 进行更多数学运算

## 活动 5.01：使用级数计算您的退休计划

**解决方案**：

执行以下步骤完成此活动：

1.  首先，我们需要确定输入变量，并注意问题归结为计算具有公比（1 + 利息）和年薪比例的等比数列的*n*项。

`annual_salary`和百分比*contrib*是我们为计划做出的贡献。`current_balance`是我们在第 0 年拥有的钱，应该加到总金额中。`annual_cap`是我们可以贡献的最大百分比；任何超出该值的输入值应该等于`contrib_cap`。`annual_salary_increase`告诉我们每年我们的工资预计增加多少。`employer_match`给我们雇主为计划贡献的百分比（通常在 0.5 和 1 之间）。最后，当前年龄，计划的年限，预期寿命以及计划可能产生的任何其他费用都是输入变量。`per_month`布尔变量确定输出是作为每年还是每月的回报金额打印。

1.  定义第一个函数`retirement_n`，来计算我们序列的第 n 项，它返回贡献和雇主匹配作为逗号分隔的元组：

```py
    def retirement_n(current_balance, annual_salary, \
                     annual_cap, n, contrib, \
                     annual_salary_increase, employer_match, \
                     match_cap, rate):
        '''
        return :: retirement amount at year n
        '''

        annual_salary_n = annual_salary*\
                          (1+annual_salary_increase)**n

        your_contrib = contrib*annual_salary_n
        your_contrib = min(your_contrib, annual_cap)
        employer_contrib = contrib*annual_salary_n*employer_match
        employer_contrib = min(employer_contrib,match_cap\
                               *annual_salary_n*employer_match)

        contrib_total = your_contrib + employer_contrib

        return your_contrib, employer_contrib,         current_balance + contrib_total*(1+rate)**n
    ```

如此所示的输入是当前余额和绝对值的年薪。我们还定义了贡献，贡献上限（即允许的最大值），年薪的增加，雇主匹配以及回报率作为相对值（0 到 1 之间的浮点数）。年度上限也应该被视为绝对值。

1.  定义一个函数，将每年的个人金额相加，并计算我们计划的总价值。这将把这个数字除以计划要使用的年数（偿还期限），以便函数返回计划的每年回报。作为输入，它应该读取当前年龄，计划的持续时间和预期寿命（偿还期限是通过从`预期寿命`中减去`当前年龄+计划年限`来找到的）：

```py
    def retirement_total(current_balance, annual_salary, \
        annual_cap=18000, contrib=0.05, \
        annual_salary_increase=0.02, employer_match=0.5, \
        match_cap=0.06, rate=0.03, current_age=35, \
        plan_years=35, life_expectancy=80, fees=0, \
        per_month=False):

        i = 0
        result = 0
        contrib_list = []; ematch_list = []; total_list = []

        while i <= plan_years:
            cn = retirement_n(current_balance=current_balance, \
                 annual_salary=annual_salary, \
                 annual_cap=annual_cap, n=i, \
                 contrib=contrib, match_cap=match_cap, \
                 annual_salary_increase=annual_salary_increase,\
                 employer_match=employer_match, rate=rate)

            contrib_list.append(cn[0])
            ematch_list.append(cn[1]) 
            total_list.append(cn[2])

            result = result + cn[2]
            i+=1
    ```

前一个函数的主要操作是设置一个循环（`while`迭代），在其中调用前一个函数，并找到每年的计划价值*n*（我们在这里称它为*cn*以简洁起见）。结果是所有年份的价值之和，并存储在`result`变量中。我们切片*cn（cn[0]，cn[1]，cn[2])*，因为`retirement_n`函数返回三个数量的元组。我们还将贡献（员工），匹配（员工）和总额的值存储在三个单独的列表中。这些将从此函数返回。

1.  最后，减去可能需要包括的任何费用并返回结果：

```py
        result = result - fees

        years_payback = life_expectancy - (current_age + plan_years)

        if per_month:
            months = 12
        else:
            months = 1
        result = result / (years_payback*months)
        print('You get back:',result)

        return result, contrib_list, ematch_list, total_list
    ```

1.  检查我们的函数和输出：

```py
    result, contrib, ematch, total = retirement_total(current_balance=1000, plan_years=35,\
                     current_age=36, annual_salary=40000, \
                     per_month=True)
    ```

输出如下：

```py
    You get back: 3029.952393422356
    ```

1.  绘制您的发现。绘制已计算的内容总是一个很好的做法，因为它有助于您理解主要信息。此外，可以检查函数是否存在潜在错误：

```py
    from matplotlib import pyplot as plt
    years = [i for i in range(len(total))]
    plt.plot(years, total,'-o',color='b')
    width=0.85
    p1 = plt.bar(years, total, width=width)
    p2 = plt.bar(years, contrib, width=width)
    p3 = plt.bar(years, ematch, width=width)
    plt.xlabel('Years')
    plt.ylabel('Return')
    plt.title('Retirement plan evolution')
    plt.legend((p1[0], p2[0], p3[0]), ('Investment returns','Contributions','Employer match'))
    plt.show()
    ```

将以以下方式显示情节：

![图 5.26：退休计划演变情节](img/B15968_05_26.jpg)

图 5.26：退休计划演变情节

有了这个，我们创建了一个 Python 程序，根据当前的贡献和一组其他参数来计算退休计划的每月或每年回报。我们已经看到了我们对序列和级数的知识如何应用到现实生活场景中，以产生有关金融和社会利益的结果。

注意

要访问此特定部分的源代码，请参阅[`packt.live/2YVgQWE`](https://packt.live/2YVgQWE)

您还可以在[`packt.live/38rOHts`](https://packt.live/38rOHts)上在线运行此示例。

# 6. 使用 Python 进行矩阵和马尔可夫链

## 活动 6.01：使用马尔可夫链构建文本预测器

**解决方案**：

有几种方法可以解决这个问题，值得一提的是，我们将采取的方法可能是使用文本预测的最简单方式。在实际实践中，文本预测要复杂得多，并且有许多其他因素会影响它们，我们将在活动结束时简要介绍。

1.  我们将使用温斯顿·丘吉尔在第二次世界大战期间从敦刻尔克被解救的盟军士兵后在英国下议院发表的演讲的文本。这篇演讲本身值得一读，如果您感兴趣，可以在网上轻松找到。

注意

您可以从[`packt.live/38rZy6v`](https://packt.live/38rZy6v)下载演讲稿。

1.  这个列表存储在名为`churchill.txt`的文本文件中。阅读该文本文件：

```py
    # Churchill's speech
    churchill = open('churchill.txt').read()
    keywords = churchill.split()
    print(keywords)
    ```

我们将其保存在名为`churchill`的字符串对象中，然后使用字符串中的`split()`函数对我们拥有的文本进行标记化，并将其存储在名为`keywords`的列表中。这将产生以下输出：

```py
    ['The', 'position', 'of', 'the', 'B.', 'E.F', 'had',  'now', 'become', 'critical', 'As', 'a', 'result', 'of',  'a', 'most', 'skillfully', 'conducted', 'retreat',….]
    ```

1.  接下来，我们遍历列表并将元素附加到一个新列表中，该列表将存储关键字和其后的单词：

```py
    keylist = []
    for i in range(len(keywords)-1):
        keylist.append( (keywords[i], keywords[i+1]))
    print(keylist)
    ```

这将产生以下输出：

```py
    [('The', 'position'), ('position', 'of'), ('of', 'the'),  ('the', 'B.'), ('B.', 'E.F'), ('E.F', 'had'), ('had',  'now'), ('now', 'become'), ('become', 'critical'),  ('critical', 'As'),….]
    ```

注意

这里的列表已经初始化，并且是一个元组列表，如果您愿意，可以将其转换为列表，但这并非必要。

1.  然后，初始化一个名为`word_dict`的字典。一旦我们有了字典，我们就会遍历先前的`keylist`数组，并将左侧的单词添加到字典中的键的前一个元组中，并将右侧的单词添加为该字典中的值。如果左侧的单词已添加到字典中，我们只需将右侧的单词附加到字典中的相应值中：

```py
    # Create key-value pairs based on follow-up words
    word_dict = {}
    for beginning, following in keylist:
        if beginning in word_dict.keys():
            word_dict[beginning].append(following)
        else:
            word_dict[beginning] = [following]
    print(word_dict)
    ```

这将产生以下输出：

```py
    {'magnetic': ['mines'], 'comparatively': ['slowly'],  'four': ['hundred', 'thousand', 'days', 'or', 'to'],  'saved': ['the', 'not'], 'forget': ['the'],….}
    ```

1.  做到这一点后，我们现在准备构建我们的预测器。首先，我们定义一个 NumPy 字符串，它从先前的关键字集合中选择一个随机单词，这将是我们的第一个单词：

```py
    first_word = np.random.choice(keywords)
    while first_word.islower():
        first_word = np.random.choice(keywords)
    ```

前面代码的第二部分旨在确保我们的句子以大写字母开头。如果不深入了解自然语言处理的工作原理，只要我们了解在原始文本中使用的大写字母单词将为构建更全面的陈述铺平道路，就足够简单了。只要它存在于我们使用的关键字语料库中，我们也可以在这里指定一个特定的单词，而不是随机选择它。

1.  将这个单词添加到一个新列表中：

```py
    word_chain = [first_word]
    ```

这里的第一个单词是从我们使用的文本文件中单词语料库中随机生成的，使用`random`函数。

然后，我们将根据先前建立的字典附加其他单词。

1.  通常，我们将查看我们刚刚附加到`word_chain`的单词，从列表中的第一个单词开始。将其用作我们创建的字典中的键，并随机跟随先前创建的字典中该特定键的值列表：

```py
    WORDCOUNT = 40
    for i in range(WORDCOUNT):
        word_chain.append(np.random.choice(word_dict[\
                                           word_chain[-1]]))
    ```

注意使用我们初始化的静态变量`WORDCOUNT`，它指定我们希望句子有多长。如果您不习惯广泛使用嵌套的 Python 函数，只需从最内部的函数开始解决，并使用`outer`函数的值。

1.  最后，我们将定义一个名为`sentence`的字符串变量，这将是我们的输出：

```py
    sentence = ' '.join(word_chain)
    print(sentence)
    ```

注意

由于这里选择的第一个单词和字典中的值都是随机选择的，因此我们每次都会得到不同的输出。

让我们看一些我们将生成的输出：

```py
    Output 1: 
    British tanks and all the New World, with little or fail. We have been reposed is so plainly marked the fighters which we should the hard and fierce. Suddenly the sharpest form. But this Island home, some articles of all fall
    Output 2
    That expansion had been effectively stamped out. Turning once again there may be very convenient, if necessary to guard their knowledge of the question of His son has given to surrender. He spurned the coast to be held by the right
    Output 3:
    Air Force. Many are a great strength and four days of the British and serious raids, could approach or at least two armored vehicles of the government would observe that has cleared, the fine Belgian Army compelled the retreating British Expeditionary
    Output 4
    30,000 men we can be defended Calais were to cast aside their native land and torpedoes. It was a statement, I feared it was in adverse weather, under its main French Army away; and thus kept open our discussions free, without
    Output 5
    German bombers and to give had the House by views freely expressed in their native land. I thought-and some articles of British and in the rescue and more numerous Air Force, and brain of it be that Herr Hitler has often
    ```

注意

要访问此特定部分的源代码，请参阅[`packt.live/3gr5uQ5`](https://packt.live/3gr5uQ5)。

您还可以在[`packt.live/31JeD2b`](https://packt.live/31JeD2b)上在线运行此示例。

# 7. 使用 Python 进行基本统计

## 活动 7.01：查找评分很高的策略游戏

**解决方案**：

1.  加载`numpy`和`pandas`库如下：

```py
    import pandas as pd
    import numpy as np
    ```

1.  加载策略游戏数据集（在本章的`dataset`文件夹中）：

```py
    games = pd.read_csv('../data/appstore_games.csv')
    ```

注意

您可以从 GitHub 存储库下载数据集[`packt.live/2O1hv2B`](https://packt.live/2O1hv2B)。

1.  执行我们在本章第一部分所做的所有转换。更改变量的名称：

```py
    original_colums_dict = {x: x.lower().replace(' ','_') \
                            for x in games.columns}
    games.rename(columns = original_colums_dict,\
                 inplace = True)
    ```

1.  将`'id'`列设置为`index`：

```py
    games.set_index(keys = 'id', inplace = True)
    ```

1.  删除`'url'`和`'icon_url'`列：

```py
    games.drop(columns = ['url', 'icon_url'], \
               inplace = True)
    ```

1.  将`'original_release_date'`和`'current_version_release_date'`更改为`datetime`：

```py
    games['original_release_date'] = pd.to_datetime\
                                     (games['original_release_date'])
    games['current_version_release_date'] = \
    pd.to_datetime(games['current_version_release_date'])
    ```

1.  从 DataFrame 中删除`'average_user_rating'`为空的行：

```py
    games = games.loc[games['average_user_rating'].notnull()]
    ```

1.  在 DataFrame 中仅保留`'user_rating_count'`等于或大于`30`的行：

```py
    games = games.loc[games['user_rating_count'] >= 30]
    ```

1.  打印数据集的维度。您必须有一个包含`4311`行和`15`列的 DataFrame。您应该得到以下输出：

```py
    (4311, 15)
    games.shape
    ```

1.  用字符串`EN`填充`languages`列中的缺失值，以指示这些游戏仅以英语提供：

```py
    games['languages'] = games['languages'].fillna('EN')
    ```

1.  创建一个名为`free_game`的变量，如果游戏的价格为零，则具有`free`的值，如果价格高于零，则具有`paid`的值：

```py
    games['free_game'] = (games['price'] == 0).astype(int)
                          .map({0:'paid', 1:'free'})
    ```

1.  创建一个名为`multilingual`的变量，如果`language`列只有一个语言字符串，则具有`monolingual`的值，如果`language`列至少有两个语言字符串，则具有`multilingual`的值：

```py
    number_of_languages = games['languages'].str.split(',') \
                                            .apply(lambdax: len(x))
    games['multilingual'] = number_of_languages == 1
    games['multilingual'] = games['multilingual'].astype(int)
                            .map({0:'multilingual', 1:'monolingual'})
    ```

1.  创建一个变量，其中包含上一步中创建的两个变量的四种组合（`free-monolingual`，`free-multilingual`，`paid-monolingual`和`paid-multilingual`）：

```py
    games['price_language'] = games['free_game'] + '-' \
                            + games['multilingual']
    ```

1.  计算`price_language`变量中每种类型的观察次数。您应该得到以下输出：

```py
    games['price_language'].value_counts()
    ```

输出将如下所示：

```py
    free-monolingual     2105
    free-multilingual    1439
    paid-monolingual     467
    paid-multilingual    300
    Name: price_language, dtype: int64
    ```

1.  在`games` DataFrame 上使用`groupby`方法，按新创建的变量进行分组，然后选择`average_user_rating`变量并计算描述性统计信息：

```py
    games.groupby('price_language')['average_user_rating']\
                                   .describe()
    ```

输出将如下所示：

![图 7.35：按 price_language 类别分组的摘要统计信息](img/B15968_07_35.jpg)

图 7.35：按 price_language 类别分组的摘要统计信息

注意

要访问此特定部分的源代码，请参阅[`packt.live/2VBGtJZ`](https://packt.live/2VBGtJZ)。

您也可以在[`packt.live/2BwtJNK`](https://packt.live/2BwtJNK)上在线运行此示例。

# 8.基础概率概念及其应用

## 活动 8.01：在金融中使用正态分布

**解决方案**：

执行以下步骤完成此活动：

1.  使用 pandas，从`data`文件夹中读取名为`MSFT.csv`的 CSV 文件：

```py
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    %matplotlib inline
    msft = pd.read_csv('../data/MSFT.csv')
    ```

1.  可选地，重命名列，使其易于使用：

```py
    msft.rename(columns=lambda x: x.lower().replace(' ', '_'),\
                inplace=True)
    ```

1.  将`date`列转换为适当的`datetime`列：

```py
    msft['date'] = pd.to_datetime(msft['date'])
    ```

1.  将`date`列设置为 DataFrame 的索引：

```py
    msft.set_index('date', inplace = True)
    ```

1.  在金融中，股票的日收益被定义为每日收盘价的百分比变化。通过计算`adj close`列的百分比变化，在 MSFT DataFrame 中创建`returns`列。使用`pct_change`系列 pandas 方法来实现：

```py
    msft['returns'] = msft['adj_close'].pct_change()
    ```

1.  将分析期限限制在`2014-01-01`和`2018-12-31`之间的日期（包括在内）：

```py
    start_date = '2014-01-01'
    end_date = '2018-12-31'
    msft = msft.loc[start_date: end_date]
    ```

1.  使用直方图来可视化收益列的分布。使用 40 个箱子来做到这一点。看起来像正态分布吗？

```py
    msft['returns'].hist(ec='k', bins=40);
    ```

输出应如下所示：

![图 8.24：MSFT 股票收益直方图](img/B15968_08_24.jpg)

图 8.24：MSFT 股票收益直方图

1.  计算`returns`列的描述性统计信息：

```py
    msft['returns'].describe()
    ```

输出如下：

```py
    count    1258.000000
    mean        0.000996
    std         0.014591
    min        -0.092534
    25%        -0.005956
    50%         0.000651
    75%         0.007830
    max         0.104522
    Name: returns, dtype: float64
    ```

1.  创建一个名为`R_rv`的随机变量，表示*MSFT 股票的日收益*。使用返回列的均值和标准差作为此分布的参数：

```py
    R_mean = msft['returns'].mean()
    R_std = msft['returns'].std()
    R_rv = stats.norm(loc = R_mean, scale = R_std)
    ```

1.  绘制`R_rv`的分布和实际数据的直方图。使用`plt.hist()`函数和`density=True`参数，使真实数据和理论分布以相同的比例显示：

```py
    fig, ax = plt.subplots()
    ax.hist(x = msft['returns'], ec = 'k', \
            bins = 40, density = True,);
    x_values = np.linspace(msft['returns'].min(), \
                           msft['returns'].max(), num=100)
    densities = R_rv.pdf(x_values)
    ax.plot(x_values, densities, color='r')
    ax.grid();
    ```

输出如下：

![图 8.25：MSFT 股票收益直方图](img/B15968_08_25.jpg)

图 8.25：MSFT 股票收益直方图

注意

要访问此特定部分的源代码，请参阅[`packt.live/2Zw18Ah`](https://packt.live/2Zw18Ah)。

您也可以在[`packt.live/31EmOg9`](https://packt.live/31EmOg9)上在线运行此示例。

在查看前面的图后，你会说正态分布是否为微软股票的日收益提供了准确的模型？

*不，正态分布并不能提供关于股票分布的非常准确的近似，因为理论分布并不完全遵循直方图的一般形状。尽管直方图关于中心对称且“钟形”，我们可以清楚地观察到零附近的值的频率比我们在正态分布中期望的要高得多，这就是为什么我们可以观察到柱子在图的中心处的红色曲线上方。此外，我们可以观察到许多极端值（左右两侧的小柱子），这些值在正态分布中不太可能出现。*

# 9. Python 中级统计

## 活动 9.01：标准化测试表现

**解决方案**：

1.  我们将使用之前创建的 t-置信区间函数来计算 95%的置信区间。我已经在这里重新创建了它以确保完整性：

```py
    # We will use the T-Confidence Interval Function 
    # we wrote earlier in the Chapter
    print("For Math:")
    t_confidence_interval(list(data['Math']),0.95)
    print("For Reading:")
    t_confidence_interval(list(data['Reading']),0.95)
    print("For Science:")
    t_confidence_interval(list(data['Science']),0.95)
    ```

这段代码的输出应该是以下内容：

```py
    For Math:
    Your 0.95 t confidence interval is (448.2561338314995,473.6869804542148)
    For Reading:
    Your 0.95 t confidence interval is (449.1937943789569,472.80078847818595)
    For Science:
    Your 0.95 t confidence interval is (453.8991748650865,476.9790108491992)
    ```

看起来我们可以以 95%的置信度说，一个国家的数学平均分在`448.3`和`473.7`之间，在阅读方面在`449.2`和`472.8`之间，在科学方面在`453.9`和`477.0`之间。

1.  我们将数据集分成两个不同的数据集；一个是每 100 人中有超过`50`个互联网用户的数据集，另一个是每 100 人中有`50`个或更少互联网用户的数据集：

```py
    # Using A Hypothesis Test, evaluate whether having 
    # widespread internet infrastructure could have an 
    # impact on scores
    # We need to divide the data set into majority 
    # internet (more than 50 users out of 100) and 
    # minority internet(50 users or less) 
    data1 = data[data['internet_users_per_100'] > 50]
    data0 = data[data['internet_users_per_100'] <= 50]
    print(data1)
    print(data0)
    ```

这里有两个数据集，`data1`和`data0`。请注意`data1`包含所有每 100 人中有超过 50 个互联网用户的国家，而`data0`包含每 100 人中有 50 个或更少互联网用户的国家：

```py
                  internet_users   Math      Reading   Science
                  _per_100
    Country Code                                                    
    ALB           63.252933        413.1570  405.2588  427.2250
    ARE           90.500000        427.4827  433.5423  436.7311
    ARG           68.043064        409.0333  425.3031  432.2262
    AUS           84.560519        493.8962  502.9006  509.9939
    AUT           83.940142        496.7423  484.8656  495.0375
    ...           ...              ...       ...       ...
    SWE           90.610200        493.9181  500.1556  493.4224
    TTO           69.198471        417.2434  427.2733  424.5905
    TUR           53.744979        420.4540  428.3351  425.4895
    URY           64.600000        417.9919  436.5721  435.3630
    USA           74.554202        469.6285  496.9351  496.2424
    [63 rows x 4 columns]
                  internet_users   Math      Reading   Science
                  _per_100
    Country Code                                                      
    DZA           38.200000        359.6062  349.8593  375.7451
    GEO           47.569760        403.8332  401.2881  411.1315
    IDN           21.976068        386.1096  397.2595  403.0997
    PER           40.900000        386.5606  397.5414  396.6836
    THA           39.316127        415.4638  409.1301  421.3373
    TUN           48.519836        366.8180  361.0555  386.4034
    VNM           43.500000        494.5183  486.7738  524.6445
    ```

1.  由于我们要比较两个可能具有不同方差的样本，我们将使用`scipy.stats`包中的 2 样本 t 检验函数。我们的显著性水平将是 5%。由于我们想要测试互联网用户多数的均值是否更高，这将是一个上尾检验。这意味着我们将把 p 值除以 2，并且只有在检验统计量为正时才接受结果为显著。以下代码将运行我们的测试（注意——这是代码的截断版本；完整的代码可以在 GitHub 存储库中找到）：

```py
    import scipy.stats as sp
    math_test_results = sp.ttest_ind(data1['Math'],\
                        data0['Math'],equal_var=False)
    print(math_test_results.statistic)
    print(math_test_results.pvalue / 2)
    reading_test_results = sp.ttest_ind(data1['Reading'],\
                           data0['Reading'],equal_var=False)
    print(reading_test_results.statistic)
    print(reading_test_results.pvalue / 2)
    science_test_results = sp.ttest_ind(data1['Science'],\
                           data0['Science'],equal_var=False)
    print(science_test_results.statistic)
    print(science_test_results.pvalue / 2)
    ```

结果如下：

```py
    For Math: (note - statistic must be positive in     order for there to be significance.)
    3.6040958108257897
    0.0036618262642996438
    For Reading: (note - statistic must be positive     in order for there to be significance.)
    3.8196670837378237
    0.0028727977455195778
    For Science: (note - statistic must be positive     in order for there to be significance.)
    2.734488895919944
    0.01425936325938158
    ```

对于数学、阅读和科学，p 值（第二个数字）小于 0.05，检验统计量（第一个数字）为正。这意味着在所有三个测试中，多数互联网用户组的测试成绩显著提高，而少数互联网用户组的测试成绩。

注意

这样的结果总是会引起统计学中一个著名的说法——相关并不意味着因果。这意味着仅仅因为我们发现了互联网多数群体平均分的显著增加，并不意味着互联网导致了分数的增加。可能存在一些第三个未知变量，称为**潜在变量**，可能导致差异。例如，财富可能是增加分数和互联网使用背后的原因。

1.  对于我们的最后任务，我们将建立一个线性回归模型，描述数学成绩与阅读和科学成绩的关系。首先，让我们从我们的 DataFrame 中提取分数，并将数学分数放在一个独立的 DataFrame 中，与阅读和科学分数分开。我们将使用`sklearn.linear_model`中的`LinearRegression`函数，并将其分配给它自己的变量。然后，我们将使用较小的 DataFrame 拟合模型。最后，我们将打印回归方程的截距和系数：

```py
    #import sklearn linear model package
    import sklearn.linear_model as lm
    # Construct a Linear Model that can predict math 
    #    scores from reading and science scores
    y = data['Math']
    x = data[['Science','Reading']]
    model = lm.LinearRegression()
    model.fit(x,y)
    print(model.coef_)
    print(model.intercept_)
    ```

结果如下：

```py
    [1.02301989 0.0516567 ]
    -38.99549267679242
    ```

系数按顺序列出，所以科学是第一个，然后是阅读。这将使你的方程为：

![图 9.23：数学成绩与阅读和科学成绩的公式](img/B15968_09_23.jpg)

图 9.23：以阅读和科学成绩为基础的数学成绩的公式

1.  最后，我们将绘制点和回归，并注意线性模型很好地拟合了数据：

```py
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(data['Science'], data['Reading'],\
                     data['Math'])
    threedee.set_xlabel('Science Score')
    threedee.set_ylabel('Reading Score')
    threedee.set_zlabel('Math Score')
    xline = np.linspace(0, 600, 600)
    yline = np.linspace(0, 600, 600)
    zline = xline*1.02301989 + \
            yline*0.0516567-38.99549267679242
    threedee.plot3D(xline, yline, zline, 'red')
    plt.show()
    ```

结果如下：

![图 9.24：线性方程似乎很好地适合我们的数据](img/B15968_09_24.jpg)

图 9.24：线性方程似乎很好地适合我们的数据

注意

要访问此特定部分的源代码，请参阅[`packt.live/3is2GE8。`](https://packt.live/3is2GE8 )

您还可以在[`packt.live/3dWmz2o`](https://packt.live/3dWmz2o )上在线运行此示例

# 10\. 使用 Python 进行基础微积分

## 活动 10.01：最大圆锥体积

解决方案：

1.  要找到所得圆锥的体积，您需要圆锥的高度和底部的半径，就像*图 10.33*右侧的图中所示。首先，我们找到底部的周长，它等于左侧切割圆上的弧长 AB。您可以将*R*设置为`1`，因为我们感兴趣的只是角度。

弧度测量使得找到弧长变得容易。它只是从切割中剩下的角度，即*2π - θ*乘以半径*R*，我们将其设置为`1`。因此*θ*也是圆锥底部的周长。我们可以建立一个方程并解决*r*：

![图 10.34：计算半径的公式](img/B15968_10_34.jpg)

图 10.34：计算半径的公式

1.  我们将把它编码到我们的程序中。我们需要从 Python 的`math`模块导入一些东西并定义`r`变量：

```py
    from math import pi,sqrt,degrees
    def v(theta):
        r = (2*pi - theta)/(2*pi) 
    ```

1.  圆锥的高度可以使用毕达哥拉斯定理找到，因为圆锥的斜高，即原始圆的半径，我们设置为`1`：![图 10.35：计算斜边的公式](img/B15968_10_35.jpg)

图 10.35：计算斜边的公式

圆锥的体积是：

![图 10.36：计算圆锥体积的公式](img/B15968_10_36.jpg)

图 10.36：计算圆锥体积的公式

1.  所以，我们将把它添加到我们的函数中：

```py
        h = sqrt(1-r**2)
        return (1/3)*pi*r**2*h
    ```

不难，是吗？这就是我们在使用 Python 时所要做的一切。如果我们按照传统方式进行微积分，我们需要一个仅用一个变量*θ*表示体积*V*的表达式。但是我们有一个关于*θ*的*r*表达式，一个关于*r*的*h*表达式，以及一个关于*h*和*r*的体积表达式。我们的程序将几乎瞬间计算出体积。

1.  现在我们可以通过我们的`find_max_mins`函数运行它。角度以弧度测量，所以我们将从`0`到`6.28`进行检查，并打印出度的版本：

```py
    find_max_mins(v,0,6.28)
    ```

输出将如下所示：

```py
    Max/Min at x= 1.1529999999999838 y= 0.40306652536733706
    ```

因此，从原始圆中切出的最佳角度是 1.15 弧度，约为 66 度。

注意

要访问此特定部分的源代码，请参阅[`packt.live/3iqx6Xj。`](https://packt.live/3iqx6Xj )

您还可以在[`packt.live/2VJHIqB`](https://packt.live/2VJHIqB )上在线运行此示例

# 11\. 使用 Python 进行更多微积分

## 活动 11.01：寻找曲面的最小值

解决方案：

1.  我们需要导入`random`模块以使用其`uniform`函数，该函数在给定范围内选择一个随机小数值：

```py
    import random
    from math import sin, cos,sqrt,pi
    ```

1.  创建一个函数，它将为我们提供`f`相对于`u`在（`v，w`）处的偏导数：

```py
    def partial_d(f,u,v,w,num=10000):
        """returns the partial derivative of f
        with respect to u at (v,w)"""
        delta_u = 1/num
        try:
            if u == 'x':
                return (f(v+delta_u,w) - f(v,w))/delta_u
            else:
                return (f(v,w+delta_u) - f(v,w))/delta_u
        except ValueError:
             pass
    ```

1.  接下来，我们将需要一个曲面函数，*x*的范围，*y*的范围和一个步长：

```py
    def min_of_surface(f,a,b,c,d,step = 0.01):
    ```

1.  我们将调用`random`模块的`uniform`函数来生成起始点的`x`和`y`值：

```py
        x,y = random.uniform(a,b),random.uniform(c,d)
    ```

1.  我们可能也会打印出测试目的的起始点。如果我们只是说`print(x，y，f(x，y))`，我们会得到不必要的长小数，所以我们在打印时会将所有内容四舍五入到*两*个小数位：

```py
        print(round(x,2),round(y,2),round(f(x,y),2))
    ```

1.  1 万步可能足够了。我们也可以将其设置为无限循环，使用`while True`：

```py
        for i in range(100000):
    ```

1.  在（x，y）处计算偏导数：

```py
            dz_dx = partial_d(f,'x',x,y, 10000)
            dz_dy = partial_d(f,'y',x,y, 10000)
    ```

1.  如果偏导数都非常接近 0，那意味着我们已经下降到了*z*的最小值。这可能是局部最小值，但对于这个随机起点，再走更多步也不会有任何进展：

```py
            if abs(dz_dx) < 0.001 and abs(dz_dy) < 0.001:
                print("Minimum:", round(x,2),round(y,2),round(f(x,y),2))
                break
    ```

1.  向*x*方向迈出一个微小步骤，与偏导数的值相反。这样，我们总是在*z*值下降。对*y*也是一样：

```py
            x -= dz_dx*step
            y -= dz_dy*step
    ```

1.  如果*x*或*y*超出了我们给定的值范围，打印`Out of Bounds`并跳出循环：

```py
            if x < a or x > b or y < c or y > d:
                print("Out of Bounds")
                break
    ```

1.  最后，打印出我们最终到达的位置的值，以及它的*z*值：

```py
        print(round(x,2),round(y,2),round(f(x,y),2))
    ```

1.  让我们在一个我们知道最小值的表面上进行测试：一个抛物面（3D 抛物线），其最小值为 0，在点(0,0)。我们将在-5 到 5 之间的值上进行测试。以下是表面的方程：![图 11.48：3D 抛物面的方程](img/B15968_11_48.jpg)

图 11.48：3D 抛物面的方程

1.  在 Python 中，它看起来像这样：

```py
    def surface(x,y):
        return x**2 + y**2
    ```

表面的样子如下：

![图 11.49：抛物面的图形](img/B15968_11_49.jpg)

图 11.49：抛物面的图形

1.  我们选择这个，因为与其二维等价物类似，最小点在(0,0)，最小的*z*值是 0。让我们在抛物面上运行`min_of_surface`函数：

```py
    min_of_surface(surface,-5,5,-5,5)
    ```

输出如下：

```py
    -1.55 2.63 9.29
    Minimum: -0.0 0.0 0.0
    ```

选择的随机点是(-1.55, 2.63)，产生了一个 z 值为 9.29。在它的行走后，它找到了*在(0,0)处的最小点，z 值为 0*。如果重新运行代码，它将从不同的随机点开始，但最终会到达(0,0)。

1.  现在我们对`min_of_surface`函数的工作很有信心，让我们尝试另一个表面：![图 11.50：另一个表面的方程](img/B15968_11_50.jpg)

图 11.50：另一个表面的方程

我们将使用*-1 < x < 5*和*-1 < y < 5*。

1.  首先，重新定义表面函数，然后为指定的范围运行`min_of_surface`函数：

```py
    def surface(x,y):
        return 3*cos(x)+5*x*cos(x)*cos(y)
    min_of_surface(surface,-1,5,-1,5)
    ```

输出将如下所示：

```py
    -0.05 4.07 3.14
    Minimum: 1.1 3.14 -1.13
    ```

看起来从这个随机点找到的最小点是(1.1,3.14)，最小的*z*值是`-1.13`。

1.  当我们重新运行代码以确保一切正确时，有时会收到`Out of Bounds`消息，有时会得到相同的结果，但很多时候，我们最终会到达这一点：

```py
    3.24 0.92 -12.8
    Minimum: 3.39 0.0 -19.34
    ```

1.  让我们将`min_of_surface`放入循环中，这样我们就可以运行多次试验：

```py
    for i in range(10):
        min_of_surface(surface,-1,5,-1,5)
    ```

以下是输出：

```py
    1.62 4.65 -0.12
    Out of Bounds
    2.87 0.47 -15.24
    Minimum: 3.39 0.0 -19.34
    2.22 0.92 -5.91
    Minimum: 3.39 0.0 -19.34
    -0.78 -0.85 0.32
    Out of Bounds
    1.23 3.81 -0.61
    Minimum: 1.1 3.14 -1.13
    1.96 -0.21 -4.82
    Minimum: 3.39 -0.0 -19.34
    -0.72 3.0 4.93
    Out of Bounds
    2.9 -0.51 -15.23
    Minimum: 3.39 -0.0 -19.34
    1.73 -0.63 -1.58
    Minimum: 3.39 -0.0 -19.34
    2.02 2.7 2.63
    Minimum: 1.1 3.14 -1.13
    ```

每次程序产生`Minimum`时，它都是我们已经看到的两个点中的一个。发生了什么？让我们看一下函数的图形：

![图 11.51：f(x,y) = 3cos(x) + 5x cos(x) * cos(y)的图形](img/B15968_11_51.jpg)

图 11.51：![1](img/B15968_11_InlineEquation81.png)的图形

图表显示的是存在多个最小值。有一个全局最小值，在这个函数深入负数，还有一个局部最小值，在该*valley*中的任何点都会简单地下降到点(1.1, 3.14)，无法离开。

注意

要访问此特定部分的源代码，请参考[`packt.live/2ApkzCc。`](https://packt.live/2ApkzCc )

您还可以在[`packt.live/2Avxt1K`](https://packt.live/2Avxt1K)上在线运行此示例。

# 12. 使用 Python 进行中级微积分

## 活动 12.01：找到粒子的速度和位置

**解决方案**：

1.  对于第一部分，我们只需要找到![2](img/B15968_12_InlineEquation3.png)的位置。让我们为*dx/dt*和*dy/dt*编写函数：

```py
    from math import sqrt,sin,cos,e
    def dx(t):
        return 1 + 3*sin(t**2)
    def dy(t):
        return 15*cos(t**2)*sin(e**t)
    ```

1.  现在，我们可以从 0 到 1.5 循环，并查看*dy/dt*从正变为负或反之的位置：

```py
    t = 0.0
    while t<=1.5:
        print(t,dy(t))
        t += 0.05
    ```

以下是输出的重要部分：

```py
    1.0000000000000002 3.3291911769931715
    1.0500000000000003 1.8966982923409172
    1.1000000000000003 0.7254255490661741
    1.1500000000000004 -0.06119060343046955
    1.2000000000000004 -0.3474047235245454
    1.2500000000000004 -0.04252527324380706
    1.3000000000000005 0.8982461584089145
    1.3500000000000005 2.4516137491656442
    1.4000000000000006 4.5062509856573225
    1.4500000000000006 6.850332845507693
    ```

我们可以看到*dy/dt*在 1.1 和 1.15 之间的某处为零，并且在 1.25 和 3 之间再次为零，因为输出改变了符号。

1.  让我们使用二分搜索来缩小这些范围。这与之前的`bin_search`函数相同，只是`guess =`行不同。我们只是将平均值插入`f`函数以获得我们的猜测：

```py
    def bin_search(f,lower,upper,target):
        def average(a,b):
            return (a+b)/2
        for i in range(40):
            avg = average(lower,upper)
            guess = f(avg)
            if guess == target:
                return guess
            if guess < target:
                upper = avg
            else:
                lower = avg
        return avg
    print(bin_search(dy,1.1,1.15,0))
    ```

答案是`t = 1.145`。

1.  对于其他范围，您必须将`if guess < target`更改为`if guess > target`，并以这种方式调用函数：

```py
    print(bin_search(dy,1.25,1.3,0))
    ```

答案是`t = 1.253`。但那太容易了。挑战在于找到这些时间点粒子的确切*x-y*位置。

1.  我们需要一个`position`函数，它将采取微小的步骤，就像我们的球问题一样：

```py
    def position(x0,y0,t):
        """Calculates the height a projectile given the
        initial height and velocity and the elapsed time."""
    ```

1.  首先，我们设置我们的增量变量，并将名为`elapsed`的变量设置为`0`：

```py
        inc = 0.001
        elapsed = 0
    ```

1.  我们的`vx`和`vy`的初始值将是 0 时的导数，`x`和`y`也将从 0 开始：

```py
        vx,vy = dx(0),dy(0)
        x,y = x0,y0
    ```

1.  现在，我们开始循环并运行，直到经过的时间达到所需的`t`：

```py
        while elapsed <= t:
    ```

1.  我们计算水平和垂直速度，然后增加`x`和`y`以及循环计数器：

```py
            vx,vy = dx(elapsed),dy(elapsed)
            x += vx*inc
            y += vy*inc
            elapsed += inc
        return x,y
    ```

1.  现在，我们将找到的时间放入`position`函数中，以获取我们知道导数为 0 的时间点粒子的位置：

```py
    times = [1.145,1.253]
    for t in times:
        print(t,position(-2,3,t))
    ```

输出结果如下：

```py
    1.145 (0.4740617265786189, 15.338128944560578)
    1.253 (0.9023867438757808, 15.313033269941062)
    ```

这些是垂直速度为 0 的位置。

1.  对于第二部分，在那里我们需要找到*t = 1*时粒子的速度，速度将是由垂直速度和水平速度形成的直角三角形的斜边：

```py
    def speed(t):
        return sqrt(dx(t)**2+dy(t)**2)
    speed(1.0)
    ```

输出如下：

```py
    4.848195599011939
    ```

粒子的速度是每秒 4.85 个单位。

注意

要访问此特定部分的源代码，请参阅[`packt.live/3dQjSzy.`](https://packt.live/3dQjSzy )

您也可以在[`packt.live/3f0IBCE`](https://packt.live/3f0IBCE)上在线运行此示例。
