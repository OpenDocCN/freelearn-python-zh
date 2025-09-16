# 8

# 性能模式

在上一章中，我们介绍了并发和异步模式，这些模式对于编写能够同时处理多个任务的效率软件非常有用。接下来，我们将讨论一些特定的性能模式，这些模式有助于提高应用程序的速度和资源利用率。

性能模式解决常见的瓶颈和优化挑战，为开发者提供经过验证的方法来提高执行时间、减少内存使用并有效扩展。

在本章中，我们将涵盖以下主要主题：

+   缓存旁路模式

+   缓存模式

+   懒加载模式

# 技术要求

请参阅第一章中提出的要求。本章讨论的代码的附加技术要求如下：

+   使用以下命令将`Faker`模块添加到您的 Python 环境中：`python -m pip install faker`

+   使用以下命令将`Redis`模块添加到您的 Python 环境中：`python -m pip install redis`

+   使用 Docker 安装 Redis 服务器并运行它：`docker run --name myredis -p 6379:6379 redis`

    如果需要，请遵循[`redis.io/docs/latest/`](https://redis.io/docs/latest/)上的文档。

# 缓存旁路模式

在数据读取频率高于更新的情况下，应用程序使用缓存来优化对存储在数据库或数据存储中的信息的重复访问。在某些系统中，这种类型的缓存机制是内置的，并且可以自动工作。当这种情况不成立时，我们必须在应用程序中自行实现它，使用适合特定用例的缓存策略。

其中一种策略被称为**缓存旁路**，在这种策略中，为了提高性能，我们将频繁访问的数据存储在缓存中，从而减少从数据存储中重复获取数据的需求。

## 现实世界案例

我们可以在软件领域引用以下示例：

+   Memcached 通常用作缓存服务器。它是一个流行的内存键值存储，用于存储来自数据库调用、API 调用或 HTML 页面内容的小块数据。

+   Redis 是另一种用于缓存的服务器解决方案。如今，它是我用于缓存或应用内存存储用例的首选服务器，在这些用例中，它表现出色。

+   根据文档网站([`docs.aws.amazon.com/elasticache/`](https://docs.aws.amazon.com/elasticache/))的说明，亚马逊的 ElastiCache 是一种云服务，它使得在云中设置、管理和扩展分布式内存数据存储或缓存环境变得容易。

## 缓存旁路模式的使用案例

当我们需要在我们的应用程序中减少数据库负载时，缓存旁路模式非常有用。通过缓存频繁访问的数据，可以减少发送到数据库的查询次数。它还有助于提高应用程序的响应速度，因为缓存数据可以更快地检索。

注意，这种模式适用于不经常变化的数据，以及不依赖于存储中一组条目一致性的数据存储（多个键）。例如，它可能适用于某些类型的文档存储或数据库，其中键永远不会更新，偶尔会删除数据条目，但没有强烈的要求在一段时间内继续提供服务（直到缓存刷新）。

## 实现缓存旁路模式。

我们可以总结实现 Cache-Aside 模式所需的步骤，涉及数据库和缓存，如下所示：

+   **案例 1 – 当我们想要获取数据项时**：如果缓存中找到该项，则从缓存中返回该项。如果没有在缓存中找到，则从数据库中读取数据。将我们得到的项目放入缓存并返回。

+   **案例 2 – 当我们想要更新数据项时**：在数据库中写入该项，并从缓存中删除相应的条目。

让我们尝试一个简单的实现，使用一个数据库，用户可以通过应用程序请求检索一些引语。我们在这里的重点是实现**案例 1**部分。

这里是我们为这个实现需要在机器上安装的额外软件依赖项的选择：

+   SQLite 数据库，因为我们可以使用 Python 的标准模块 `sqlite3` 来查询 SQLite 数据库。

+   Redis 服务器和 `redis-py` Python 模块。

我们将使用一个脚本（在 `ch08/cache_aside/populate_db.py` 文件中）来处理创建数据库和 `quotes` 表，并将示例数据添加到其中。出于实际考虑，我们也在那里使用 `Faker` 模块生成假引语，这些引语用于填充数据库。

我们的代码从所需的导入开始，然后创建我们将用于生成假引语的 Faker 实例，以及一些常量或模块级变量：

```py
import sqlite3
from pathlib import Path
from random import randint
import redis
from faker import Faker
fake = Faker()
DB_PATH = Path(__file__).parent / Path("quotes.sqlite3")
cache = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
```

然后，我们编写一个函数来处理数据库设置部分，如下所示：

```py
def setup_db():
    try:
        with sqlite3.connect(DB_PATH) as db:
            cursor = db.cursor()
            cursor.execute(
                """
                CREATE TABLE quotes(id INTEGER PRIMARY KEY, text TEXT)
            """
            )
            db.commit()
            print("Table 'quotes' created")
    except Exception as e:
        print(e)
```

然后，我们定义一个中心函数，该函数负责根据句子列表或文本片段添加一组新的引语。在众多事情中，我们将引语标识符与引语关联，用于数据库表中的 `id` 列。为了简化问题，我们只是随机选择一个数字，使用 `quote_id = randint(1, 100)`。`add_quotes()` 函数定义如下：

```py
def add_quotes(quotes_list):
    added = []
    try:
        with sqlite3.connect(DB_PATH) as db:
            cursor = db.cursor()
            for quote_text in quotes_list:
                quote_id = randint(1, 100) # nosec
                quote = (quote_id, quote_text)
                cursor.execute(
                    """INSERT OR IGNORE INTO quotes(id, text) VALUES(?, ?)""", quote
                )
                added.append(quote)
            db.commit()
    except Exception as e:
        print(e)
    return added
```

接下来，我们添加一个 `main()` 函数，实际上它将包含几个部分；我们想要使用命令行参数解析。请注意以下内容：

+   如果我们传递 `init` 参数，我们调用 `setup_db()` 函数。

+   如果我们传递 `update_all` 参数，我们将引语注入数据库并添加到缓存中。

+   如果我们传递 `update_db_only` 参数，我们只将引语注入数据库。

当运行 Python 脚本时调用的 `main()` 函数的代码如下：

```py
def main():
    msg = "Choose your mode! Enter 'init' or 'update_db_only' or 'update_all': "
    mode = input(msg)
    if mode.lower() == "init":
        setup_db()
    elif mode.lower() == "update_all":
        quotes_list = [fake.sentence() for _ in range(1, 11)]
        added = add_quotes(quotes_list)
        if added:
            print("New (fake) quotes added to the database:")
            for q in added:
                print(f"Added to DB: {q}")
                print("  - Also adding to the cache")
                cache.set(str(q[0]), q[1], ex=60)
    elif mode.lower() == "update_db_only":
        quotes_list = [fake.sentence() for _ in range(1, 11)]
        added = add_quotes(quotes_list)
        if added:
            print("New (fake) quotes added to the database ONLY:")
            for q in added:
                print(f"Added to DB: {q}")
```

那部分已经完成。现在，我们将创建另一个模块和脚本，用于缓存旁路相关的操作本身（在 `ch08/cache_aside/cache_aside.py` 文件中）。

我们这里也需要一些导入，然后是常量：

```py
import sqlite3
from pathlib import Path
import redis
CACHE_KEY_PREFIX = "quote"
DB_PATH = Path(__file__).parent / Path("quotes.sqlite3")
cache = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
```

接下来，我们定义一个 `get_quote()` 函数，通过标识符获取引语。如果我们不在缓存中找到引语，我们将查询数据库以获取它，并在返回之前将其放入缓存。函数定义如下：

```py
def get_quote(quote_id: str) -> str:
    out = []
    quote = cache.get(f"{CACHE_KEY_PREFIX}.{quote_id}")
    if quote is None:
        # Get from the database
        query_fmt = "SELECT text FROM quotes WHERE id = {}"
        try:
            with sqlite3.connect(DB_PATH) as db:
                cursor = db.cursor()
                res = cursor.execute(query_fmt.format(quote_id)).fetchone()
                if not res:
                    return "There was no quote stored matching that id!"
                quote = res[0]
                out.append(f"Got '{quote}' FROM DB")
        except Exception as e:
            print(e)
            quote = ""
        # Add to the cache
        if quote:
            key = f"{CACHE_KEY_PREFIX}.{quote_id}"
            cache.set(key, quote, ex=60)
            out.append(f"Added TO CACHE, with key '{key}'")
    else:
        out.append(f"Got '{quote}' FROM CACHE")
    if out:
        return " - ".join(out)
    else:
        return ""
```

最后，在脚本的主体部分，我们要求用户输入一个引语标识符，并调用 `get_quote()` 来获取引语。代码如下：

```py
def main():
    while True:
        quote_id = input("Enter the ID of the quote: ")
        if quote_id.isdigit():
            out = get_quote(quote_id)
            print(out)
        else:
            print("You must enter a number. Please retry.")
```

现在是测试我们脚本的时机，请按照以下步骤进行。

首先，通过调用 `python ch08/cache_aside/populate_db.py` 并选择 `"init"` 作为模式选项，我们可以看到在 `ch08/cache_aside/` 文件夹中创建了一个 `quotes.sqlite3` 文件，因此我们可以得出结论，数据库已经创建，并在其中创建了一个 `quotes` 表。

然后，我们调用 `python ch08/cache_aside/populate_db.py` 并传递 `update_all` 模式；我们得到以下输出：

```py
Choose your mode! Enter 'init' or 'update_db_only' or 'update_all': update_all
New (fake) quotes added to the database:
Added to DB: (62, 'Instead not here public.')
- Also adding to the cache
Added to DB: (26, 'Training degree crime serious beyond management and.')
- Also adding to the cache
Added to DB: (25, 'Agree hour example cover game bed.')
- Also adding to the cache
Added to DB: (23, 'Dark team exactly really wind.')
- Also adding to the cache
Added to DB: (46, 'Only loss simple born remain.')
- Also adding to the cache
Added to DB: (13, 'Clearly statement mean growth executive mean.')
- Also adding to the cache
Added to DB: (88, 'West policy a human job structure bed.')
- Also adding to the cache
Added to DB: (25, 'Work maybe back play.')
- Also adding to the cache
Added to DB: (18, 'Here certain require consumer strategy.')
- Also adding to the cache
Added to DB: (48, 'Discover method many by hotel.')
python ch08/cache_aside/populate_db.py and choose the update_db_only mode. In that case, we get the following output:

```

选择你的模式！输入 'init' 或 'update_db_only' 或 'update_all'：update_db_only

仅向数据库中添加了新的（虚假的）引语：

添加到数据库中：（73，'Whose determine group what site.'）

添加到数据库中：（77，'Standard much career either will when chance.'）

添加到数据库中：（5，'Nature when event appear yeah.'）

添加到数据库中：（81，'By himself in treat.'）

添加到数据库中：（88，'Establish deal sometimes stage college everybody close thank.'）

添加到数据库中：（99，'Room recently authority station relationship our knowledge occur.'）

添加到数据库中：（63，'Price who a crime garden doctor eat.'）

添加到数据库中：（43，'Significant hot those think heart shake ago.'）

添加到数据库中：（80，'Understand and view happy.'）

`python ch08/cache_aside/cache_aside.py` 命令，然后我们被要求输入一个尝试获取匹配引语的输入。以下是我根据提供的值得到的不同输出：

```py
Enter the ID of the quote: 23
Got 'Dark team exactly really wind.' FROM DB - Added TO CACHE, with key 'quote.23'
Enter the ID of the quote: 12
There was no quote stored matching that id!
Enter the ID of the quote: 43
Got 'Significant hot those think heart shake ago.' FROM DB - Added TO CACHE, with key 'quote.43'
Enter the ID of the quote: 45
There was no quote stored matching that id!
Enter the ID of the quote: 77
Got 'Standard much career either will when chance.' FROM DB - Added TO CACHE, with key 'quote.77'
```

            因此，每次我输入一个与仅存储在数据库中的引语匹配的标识符（如前一个输出所示），具体的输出都显示数据首先从数据库中获取，然后从缓存（它立即被添加到其中）返回。

            我们可以看到一切按预期工作。缓存 aside 实现的更新部分（在数据库中写入条目并从缓存中删除相应的条目）留给你去尝试。你可以添加一个 `update_quote()` 函数，用于在传递 `quote_id` 给它时更新一个引语，并使用正确的命令行（例如 `python` `cache_aside.py update`）来调用它。

            记忆化模式

            **记忆化** 模式是软件开发中一个关键的优化技术，通过缓存昂贵函数调用的结果来提高程序的效率。这种方法确保了如果函数多次使用相同的输入被调用，则返回缓存的值，从而消除了重复和昂贵的计算需求。

            真实世界的例子

            我们可以将计算斐波那契数列视为记忆化模式的经典示例。通过存储序列之前计算过的值，算法避免了重新计算，这极大地加快了序列中更高数值的计算速度。

            另一个例子是文本搜索算法。在处理大量文本的应用中，如搜索引擎或文档分析工具，缓存先前搜索的结果意味着相同的查询可以立即返回结果，这显著提高了用户体验。

            记忆化模式的用例

            记忆化模式可以用于以下用例：

                1.  **加速递归算法**：记忆化将递归算法从具有高时间复杂度转变为低时间复杂度。这对于计算斐波那契数等算法特别有益。

                1.  **减少计算开销**：记忆化通过避免不必要的重新计算来节省 CPU 资源。这在资源受限的环境或处理大量数据处理时至关重要。

                1.  **提高应用性能**：记忆化的直接结果是应用性能的显著提升，使用户感觉应用响应更快、更高效。

            实现记忆化模式

            让我们讨论使用 Python 的 `functools.lru_cache` 装饰器实现记忆化模式的示例。这个工具对于具有昂贵计算且重复使用相同参数调用的函数特别有效。通过缓存结果，具有相同参数的后续调用将直接从缓存中检索结果，显著减少执行时间。

            对于我们的示例，我们将记忆化应用于一个经典问题，其中使用了递归算法：计算斐波那契数。

            我们首先需要以下 `import` 语句：

```py
from datetime import timedelta
from functools import lru_cache
```

            第二，我们创建了一个名为 `fibonacci_func1` 的函数，该函数使用递归（不涉及任何缓存）来计算斐波那契数。我们将用它来进行比较：

```py
def fibonacci_func1(n):
    if n < 2:
        return n
    return fibonacci_func1(n - 1) + fibonacci_func1(n - 2)
```

            第三，我们定义了一个名为 `fibonacci_func2` 的函数，代码与之前相同，但这次我们使用了 `lru_cache` 装饰器来启用记忆化。这里发生的情况是，函数调用的结果被存储在内存中的缓存中，具有相同参数的重复调用将直接从缓存中获取结果，而不是执行函数的代码。代码如下：

```py
@lru_cache(maxsize=None)
def fibonacci_func2(n):
    if n < 2:
        return n
    return fibonacci_func2(n - 1) + fibonacci_func2(n - 2)
```

            最后，我们创建了一个 `main()` 函数来测试使用 `n=30` 作为输入调用两个函数，并测量每个执行的耗时。测试代码如下：

```py
def main():
    import time
    n = 30
    start_time = time.time()
    result = fibonacci_func1(n)
    duration = timedelta(time.time() - start_time)
    print(f"Fibonacci_func1({n}) = {result}, calculated in {duration}")
    start_time = time.time()
    result = fibonacci_func2(n)
    duration = timedelta(time.time() - start_time)
    print(f"Fibonacci_func2({n}) = {result}, calculated in {duration}")
```

            要测试实现，请运行以下命令：`python ch08/memoization.py`。你应该得到以下输出：

```py
Fibonacci_func1(30) = 832040, calculated in 7:38:53.090973
Fibonacci_func2(30) = 832040, calculated in 0:00:02.760315
```

            当然，你得到的时间可能与我不同，但使用缓存功能的第二个函数的时间应该短于没有缓存的函数的时间。而且，两者之间的时间差应该是重要的。

            这是一个演示，说明记忆化减少了计算斐波那契数所需的递归调用次数，尤其是对于大的`n`值。通过减少计算开销，记忆化不仅加快了计算速度，还节省了系统资源，从而使得应用程序更加高效和响应。

            懒加载模式

            **懒加载**模式是软件工程中的一个关键设计方法，尤其在优化性能和资源管理方面特别有用。懒加载的理念是在资源真正需要时才延迟初始化或加载资源。这样，应用程序可以实现更有效的资源利用，减少初始加载时间，并提升整体用户体验。

            真实世界的例子

            浏览在线艺术画廊提供了一个例子。网站不会一开始就加载数百张高分辨率图片，而是只加载当前视图中的图片。当你滚动时，额外的图片会无缝加载，从而提升你的浏览体验，而不会耗尽设备的内存或网络带宽。

            另一个例子是按需视频流媒体服务，如 Netflix 或 YouTube。这样的平台通过分块加载视频提供不间断的观看体验。这种方法不仅最小化了开始时的缓冲时间，还能适应不断变化的网络条件，确保视频质量一致，中断最少。

            在像 Microsoft Excel 或 Google Sheets 这样的应用程序中，处理大量数据集可能非常耗费资源。懒加载允许这些应用程序仅加载与当前视图或操作相关的数据，例如特定的工作表或单元格范围。这显著加快了操作速度并减少了内存使用。

            懒加载模式的用例

            我们可以将以下与性能相关的用例视为懒加载模式：

                1.  **减少初始加载时间**：这在网页开发中尤其有益，较短的加载时间可以转化为更高的用户参与度和留存率。

                1.  **保护系统资源**：在多样化的设备时代，从高端台式机到入门级智能手机，优化资源使用对于在所有平台上提供一致的用户体验至关重要。

                1.  **提升用户体验**：用户期望与软件进行快速、响应式的交互。懒加载通过最小化等待时间并使应用程序感觉更加响应来对此做出贡献。

            实现懒加载模式 – 懒属性加载

            考虑一个执行复杂数据分析或基于用户输入生成复杂可视化的应用程序。背后的计算是资源密集型和耗时的。在这种情况下实现懒加载可以显著提高性能。但为了演示目的，我们将不会像复杂的数据分析应用场景那样雄心勃勃。我们将使用一个模拟昂贵计算并返回用于类属性值的函数。

            对于这个懒加载示例，我们的想法是只有当属性第一次被访问时才初始化属性。这种方法在初始化属性是资源密集型，并且你希望推迟这个过程直到必要时常用的场景中。 

            我们从`LazyLoadedData`类的初始化部分开始，将`_data`属性设置为`None`。在这里，昂贵的资源尚未被加载：

```py
class LazyLoadedData:
    def __init__(self):
        self._data = None
```

            我们添加了一个`data()`方法，使用`@property`装饰器，使其像属性（一个属性）一样工作，并添加了懒加载的逻辑。在这里，我们检查`_data`是否为`None`。如果是，我们调用`load_data()`方法：

```py
    @property
    def data(self):
        if self._data is None:
            self._data = self.load_data()
        return self._data
```

            我们添加了一个`load_data()`方法，模拟一个昂贵的操作，使用`sum(i * i for i in range(100000))`。在现实世界的场景中，这可能涉及从远程数据库获取数据，执行复杂的计算或其他资源密集型任务：

```py
    def load_data(self):
        print("Loading expensive data...")
        return sum(i * i for i in range(100000))
```

            然后我们添加一个`main()`函数来测试实现。我们创建`LazyLoadedData`类的一个实例，并两次访问`_data`属性：

```py
def main():
    obj = LazyLoadedData()
    print("Object created, expensive attribute not loaded yet.")
    print("Accessing expensive attribute:")
    print(obj.data)
    print("Accessing expensive attribute again, no reloading occurs:")
    print(obj.data)
```

            要测试实现，运行`python ch08/lazy_loading/lazy_attribute_loading.py`命令。你应该得到以下输出：

```py
Object created, expensive attribute not loaded yet.
Accessing expensive attribute:
Loading expensive data...
333328333350000
Accessing expensive attribute again, no reloading occurs:
_data. On subsequent accesses, the data stored is retrieved (from the attribute) without re-performing the expensive operation.
			The lazy loading pattern, applied this way, is very useful for improving performance in applications where certain data or computations are needed from time to time but are expensive to produce.
			Implementing the lazy loading pattern – using caching
			In this second example, we consider a function that calculates the factorial of a number using recursion, which can become quite expensive computationally as the input number grows. While Python’s `math` module provides a built-in function for calculating factorials efficiently, implementing it recursively serves as a good example of an expensive computation that could benefit from caching. We will use caching with `lru_cache`, as in the previous section, but this time for the purpose of lazy loading.
			We start with importing the modules and functions we need:

```

import time

from datetime import timedelta

from functools import lru_cache

```py

			Then, we create a `recursive_factorial()` function that calculates the factorial of a number `n` recursively:

```

def recursive_factorial(n):

"""计算阶乘（对于大的 n 来说很昂贵）"""

if n == 1:

return 1

else:

return n * recursive_factorial(n - 1)

```py

			Third, we create a `cached_factorial()` function that returns the result of calling `recursive_factorial()` and is decorated with `@lru_cache`. This way, if the function is called again with the same arguments, the result is retrieved from the cache instead of being recalculated, significantly reducing computation time:

```

@lru_cache(maxsize=128)

def cached_factorial(n):

return recursive_factorial(n)

```py

			We create a `main()` function as usual for testing the functions. We call the non-cached function, and then we call the `cached_factorial` function twice, showing the computation time for each case. The code is as follows:

```

def main():

# 测试性能

n = 20

# Without caching

start_time = time.time()

print(f"{ n }的递归阶乘：{ recursive_factorial(n) }")

duration = timedelta(time.time() - start_time)

print(f"无缓存的计算时间：{ duration }。")

# With caching

start_time = time.time()

print(f"缓存的{ n }的阶乘：{ cached_factorial(n) }")

duration = timedelta(time.time() - start_time)

print(f"带缓存的计算时间：{ duration }。")

start_time = time.time()

print(f"缓存的{ n }的阶乘，重复：{ cached_factorial(n) }")

duration = timedelta(time.time() - start_time)

print(f"带缓存的第二次计算时间：{ duration }。")

```py

			To test the implementation, run the `python ch08/lazy_loading/lazy_loading_with_caching.py` command. You should get the following output:

```

递归阶乘的 20：2432902008176640000

无缓存的计算时间：0:00:04.840851

缓存的 20 的阶乘：2432902008176640000

带缓存的计算时间：0:00:00.865173

缓存的 20 的阶乘，重复：2432902008176640000

带缓存的第二次计算时间：0:00:00.350189

```py

			You will notice the time taken for the initial calculation of the factorial without caching, then the time with caching, and finally, the time for a repeated calculation with caching.
			Also, `lru_cache` is inherently a memoization tool, but it can be adapted and used in cases where, for example, there are expensive initialization processes that need to be executed only when required and not make the application slow. In our example, we used factorial computation to simulate such expensive processes.
			If you are asking yourself what is the difference from memoization, the answer is that the context in which caching is used here is for managing resource initialization.
			Summary
			Throughout this chapter, we have explored patterns that developers can use to enhance the efficiency and scalability of applications.
			The cache-aside pattern teaches us how to manage cache effectively, ensuring data is fetched and stored in a manner that optimizes performance and consistency, particularly in environments with dynamic data sources.
			The memoization pattern demonstrates the power of caching function results to speed up applications by avoiding redundant computations. This pattern is beneficial for expensive, repeatable operations and can dramatically improve the performance of recursive algorithms and complex calculations.
			Finally, the lazy loading pattern emphasizes delaying the initialization of resources until they are needed. This approach not only improves the startup time of applications but also reduces memory overhead, making it ideal for resource-intensive operations that may not always be necessary for the user’s interactions.
			In the next chapter, we are going to discuss patterns that govern distributed systems.

```

```py

```
