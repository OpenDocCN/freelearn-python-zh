# 8

# 创建协程、事件和消息驱动的交易

FastAPI 框架是一个基于 asyncio 平台的异步框架，它利用了 ASGI 协议。它因其对异步端点和非阻塞任务 100%的支持而闻名。本章将重点介绍我们如何通过异步任务和事件驱动以及消息驱动的交易来创建高度可扩展的应用程序。

我们在 *第二章* *探索核心功能* 中了解到，*Async/Await* 或异步编程是一种设计模式，它允许其他服务或交易在主线程之外运行。该框架使用 `async` 关键字创建将在其他线程池之上运行的异步进程，并将被 *await*，而不是直接调用它们。外部线程的数量在 Uvicorn 服务器启动期间通过 `--worker` 选项定义。

在本章中，我们将深入研究框架并仔细审查 FastAPI 框架的各个组件，这些组件可以使用多个线程异步运行。以下要点将帮助我们理解异步 FastAPI：

+   实现协程

+   创建异步后台任务

+   理解 Celery 任务

+   使用 RabbitMQ 构建消息驱动的交易

+   使用 Kafka 构建发布/订阅消息

+   在任务中应用响应式编程

+   自定义事件

+   实现异步 **服务器端事件**（**SSE**）

+   构建异步 WebSocket

# 技术要求

本章将涵盖异步功能、软件规范以及 *新闻亭管理系统* 原型的组件。讨论将使用这个在线报纸管理系统原型作为样本来理解、探索和实现异步交易，这些交易将管理 *报纸内容*、*订阅*、*计费*、*用户资料*、*客户* 以及其他与业务相关的交易。所有代码都已上传到 [`github.com/PacktPublishing/Building-Python-Microservices-with-FastAPI`](https://github.com/PacktPublishing/Building-Python-Microservices-with-FastAPI) 下的 `ch08` 项目。

# 实现协程

在 FastAPI 框架中，始终存在一个 *线程池* 来执行每个请求的同步 API 和非 API 交易。对于理想情况，即这两种交易都具有最小的性能开销，无论是 *CPU 密集型* 还是 *I/O 密集型* 交易，使用 FastAPI 框架的整体性能仍然优于那些使用非 ASGI 基础平台的框架。然而，当由于高 CPU 密集型流量或重 CPU 工作负载导致竞争时，由于 *线程切换*，FastAPI 的性能开始下降。

线程切换是在同一进程内从一个线程切换到另一个线程的上下文切换。因此，如果我们有多个具有不同工作负载的事务在后台和浏览器上运行，FastAPI 将在线程池中运行这些事务，并执行多个上下文切换。这种情况将导致对较轻工作负载的竞争和性能下降。为了避免性能问题，我们采用 *协程切换* 而不是线程。

## 应用协程切换

FastAPI 框架通过称为 *协程切换* 的机制以最佳速度运行。这种方法允许事务调优的任务通过允许其他运行进程暂停，以便线程可以执行并完成更紧急的任务，并在不抢占线程的情况下恢复 "awaited" 事务。这些协程切换是程序员定义的组件，与内核或内存相关的功能无关。在 FastAPI 中，有两种实现协程的方法：(a) 应用 `@asyncio.coroutine` 装饰器，和 (b) 使用 `async`/`await` 构造。

### 应用 @asyncio.coroutine

`asyncio` 是一个 Python 扩展，它使用单线程和单进程模型实现 Python 并发范式，并提供用于运行和管理协程的 API 类和方法。此扩展提供了一个 `@asyncio.coroutine` 装饰器，将 API 和原生服务转换为基于生成器的协程。然而，这是一个旧的方法，只能在使用 Python 3.9 及以下版本的 FastAPI 中使用。以下是我们 *新闻亭管理系统* 原型中实现为协程的登录服务事务：

```py
@asyncio.coroutine
def build_user_list(query_list):
    user_list = []
    for record in query_list:
        yield from asyncio.sleep(2)
        user_list.append(" ".join([str(record.id), 
            record.username, record.password]))
    return user_list
```

`build_user_list()` 是一个原生服务，它将所有登录记录转换为 `str` 格式。它使用 `@asyncio.coroutine` 装饰器将事务转换为异步任务或协程。协程可以使用 `yield from` 子句调用另一个协程函数或方法。顺便说一句，`asyncio.sleep()` 方法是 `asyncio` 模块中最广泛使用的异步实用工具之一，它可以使进程暂停几秒钟，但并不是理想的。另一方面，以下代码是一个作为协程实现的 API 服务，它可以最小化客户端执行中的竞争和性能下降：

```py
@router.get("/login/list/all")
@asyncio.coroutine
def list_login():
    repo = LoginRepository()
    result = yield from repo.get_all_login()
    data = jsonable_encoder(result)
    return data
```

`list_login()` API 服务通过在 *GINO ORM* 中实现的协程 CRUD 事务检索应用程序用户的全部登录详情。API 服务再次使用 `yield from` 子句来运行和执行 `get_all_login()` 协程函数。

协程函数可以使用 `asyncio.gather()` 工具并发调用和等待多个协程。这个 `asyncio` 方法管理一个协程列表，并等待直到所有协程完成其任务。然后，它将返回对应协程的结果列表。以下是一个通过异步 CRUD 事务检索登录记录的 API，然后并发调用 `count_login()` 和 `build_user_list()` 来处理这些记录：

```py
@router.get("/login/list/records")
@asyncio.coroutine
def list_login_records():
    repo = LoginRepository()
    login_data = yield from repo.get_all_login()
    result = yield from 
       asyncio.gather(count_login(login_data), 
            build_user_list(login_data))
    data = jsonable_encoder(result[1])
    return {'num_rec': result[0], 'user_list': data}
```

`list_login_records()` 使用 `asyncio.gather()` 来运行 `count_login()` 和 `build_user_list()` 任务，并在之后提取它们对应的返回值进行处理。

### 使用 async/await 构造

实现协程的另一种方法是使用 `async`/`await` 构造。与之前的方法一样，这种语法创建了一个任务，在执行过程中可以随时暂停，直到到达末尾。但这种方法产生的协程被称为 *原生协程*，它不能像生成器类型那样迭代。`async`/`await` 语法还允许创建其他异步组件，例如 `async with` 上下文管理器和 `async for` 迭代器。以下代码是之前在基于生成器的协程服务 `list_login_records()` 中调用的 `count_login()` 任务：

```py
async def count_login(query_list):
    await asyncio.sleep(2)
    return len(query_list)
```

`count_login()` 原生服务是一个原生协程，因为它在方法定义之前放置了 `async` 关键字。它只使用 `await` 来调用其他协程。`await` 关键字暂停当前协程的执行，并将线程控制权传递给被调用的协程函数。在被调用的协程完成其处理后，线程控制权将返回给调用协程。使用 `yield from` 构造而不是 `await` 会引发错误，因为我们的协程不是基于生成器的。以下是一个作为原生协程实现的 API 服务，用于管理新管理员资料的录入：

```py
@router.post("/admin/add")
async def add_admin(req: AdminReq):
    admin_dict = req.dict(exclude_unset=True)
    repo = AdminRepository()
    result = await repo.insert_admin(admin_dict)
    if result == True: 
        return req 
    else: 
        return JSONResponse(content={'message':'update 
            trainer profile problem encountered'}, 
              status_code=500)
```

基于生成器和原生的协程都由一个 *事件循环* 监控和管理，它代表线程内部的一个无限循环。技术上，它是在 *线程* 中找到的一个对象，线程池中的每个 *线程* 只能有一个事件循环，其中包含一个称为 *任务* 的辅助对象列表。每个任务，无论是预先生成的还是手动创建的，都会执行一个协程。例如，当先前的 `add_admin()` API 服务调用 `insert_admin()` 协程事务时，事件循环将挂起 `add_admin()` 并将其任务标记为 *等待* 任务。之后，事件循环将分配一个任务来运行 `insert_admin()` 事务。一旦任务完成执行，它将控制权交还给 `add_admin()`。在执行转换期间，管理 FastAPI 应用程序的线程不会被中断，因为事件循环及其任务参与了 *协程切换* 机制。现在让我们使用这些协程来构建我们的应用程序

## 设计异步事务

在为我们的应用程序创建协程时，我们可以遵循几种编程范式。在过程中利用更多的协程切换可以帮助提高软件性能。在我们的 *newsstand* 应用程序中，`admin.py` 路由器中有一个端点 `/admin/login/list/enc`，它返回一个加密的用户详情列表。在其 API 服务中，如下所示代码所示，每条记录都由一个 `extract_enc_admin_profile()` 事务调用管理，而不是将整个数据记录传递给单个调用，从而允许任务的并发执行。这种策略比在没有 *上下文切换* 的情况下运行大量事务在线程中更好：

```py
@router.get("/admin/login/list/enc")
async def generate_encypted_profile():
    repo = AdminLoginRepository()
    result = await repo.join_login_admin()
    encoded_data = await asyncio.gather(
       *(extract_enc_admin_profile(rec) for rec in result))
    return encoded_data
```

现在，`extract_enc_admin_profile()` 协程，如下所示代码所示，实现了一个链式设计模式，其中它通过链调用其他较小的协程。将单体和复杂的过程简化并分解成更小但更健壮的协程，通过利用更多的上下文切换来提高应用程序的性能。在这个 API 中，`extract_enc_admin_profile()` 在链中创建了三个上下文切换，比线程切换更优：

```py
async def extract_enc_admin_profile(admin_rec):
    p = await extract_profile(admin_rec)
    pinfo = await extract_condensed(p)
    encp = await decrypt_profile(pinfo)
    return encp
```

另一方面，以下实现是 `extract_enc_admin_profile()` 等待和执行的较小子程序：

```py
async def extract_profile(admin_details):
    profile = {}
    login = admin_details.parent
    profile['firstname'] = admin_details.firstname
    … … … … … …
    profile['password'] = login.password 
    await asyncio.sleep(1)
    return profile
async def extract_condensed(profiles):
    profile_info = " ".join([profiles['firstname'], 
       profiles['lastname'], profiles['username'], 
       profiles['password']])
    await asyncio.sleep(1)
    return profile_info 
async def decrypt_profile(profile_info):
    key = Fernet.generate_key()
    fernet = Fernet(key)
    encoded_profile = fernet.encrypt(profile_info.encode())
    return encoded_profile
```

这三个子程序将给主协程提供包含管理员配置文件详情的加密 `str`。所有这些加密字符串都将由 API 服务使用 `asyncio.gather()` 工具汇总。

利用协程切换的另一种编程方法是使用 `asyncio.Queue` 创建的管道。在这种编程设计中，队列结构是两个任务之间的共同点：（a）将值放入队列的任务称为 *生产者*，（b）从队列中获取项的任务称为 *消费者*。我们可以使用这种方法实现 *单生产者/单消费者* 交互或 *多生产者/多消费者* 设置。

以下代码突出了构建 *生产者/消费者* 交易流程的本地服务 `process_billing()`。`extract_billing()` 协程是生产者，它从数据库中检索账单记录，并将每个记录逐个传递到队列中。另一方面，`build_billing_sheet()` 是消费者，它从队列结构中获取记录并生成账单表：

```py
async def process_billing(query_list):
    billing_list = []

    async def extract_billing(qlist, q: Queue):
        assigned_billing = {}
        for record in qlist:
            await asyncio.sleep(2)
            assigned_billing['admin_name'] = "{} {}"
              .format(record.firstname, record.lastname)
            if not len(record.children) == 0:
                assigned_billing['billing_items'] = 
                      record.children
            else:
                assigned_billing['billing_items'] = None

            await q.put(assigned_billing)
    async def build_billing_sheet(q: Queue):
        while True: 
            await asyncio.sleep(2)
            assigned_billing = await q.get()
            name = assigned_billing['admin_name']
            billing_items = 
                assigned_billing['billing_items']
            if not billing_items == None:
                for item in billing_items:
                    billing_list.append(
                    {'admin_name': name, 'billing': item})
            else: 
                billing_list.append(
                    {'admin_name': name, 'billing': None})
            q.task_done()
```

在这种编程设计中，`build_billing()` 协程将明确等待由 `extract_billing()` 队列的记录。这种设置是由于 `asyncio.create_task()` 工具，它直接分配和调度任务给每个协程。

队列是协程的唯一公共方法参数，因为它它们的共同点。`asyncio.Queue` 的 `join()` 确保所有通过 `extract_billing()` 传递到管道的项都被 `build_billing_sheet()` 获取和处理。它还阻止影响协程交互的外部控制。以下代码展示了如何创建 `asyncio.Queue` 并调度任务执行：

```py
    q = asyncio.Queue()
    build_sheet = asyncio.create_task(
               build_billing_sheet(q))
    await asyncio.gather(asyncio.create_task(
             extract_billing(query_list, q)))

    await q.join()
    build_sheet.cancel()
    return billing_list
```

顺便说一句，总是在协程完成处理后立即传递 `cancel()` 给任务。另一方面，我们也可以采用其他方法，以便提高我们协程的性能。

## 使用 HTTP/2 协议

在运行在 *HTTP/2* 协议上的应用程序中，协程执行可以更快。我们可以用 *Hypercorn* 替换 *Uvicorn* 服务器，现在它支持基于 ASGI 的框架，如 FastAPI。但首先，我们需要使用 `pip` 安装 `hypercorn`：

```py
pip install hypercorn
```

要使 *HTTP/2* 工作，我们需要创建一个 SSL 证书。使用 OpenSSL，我们的应用程序有两个 *PEM* 文件用于我们的 *newsstand* 原型：（a）私有加密密钥（`key.pem`）和（b）证书信息（`cert.pem`）。我们在执行以下 `hypercorn` 命令以运行我们的 FastAPI 应用程序之前，将这些文件放置在主项目文件夹中：

```py
hypercorn --keyfile key.pem --certfile cert.pem main:app       --bind 'localhost:8000' --reload
```

现在，让我们探索其他可以使用协程的 FastAPI 任务。

# 创建异步后台任务

在*第二章**，探索核心功能*中，我们首先展示了 `BackgroundTasks` 注入式 API 类，但并未提及创建异步后台任务。在本讨论中，我们将专注于使用 `asyncio` 模块和协程创建异步后台任务。

## 使用协程

框架支持使用 `async`/`await` 结构创建和执行异步后台进程。以下原生服务是一个异步事务，在后台以 CSV 格式生成账单表：

```py
async def generate_billing_sheet(billing_date, query_list):
    filepath = os.getcwd() + '/data/billing-' + 
                  str(billing_date) +'.csv'
    with open(filepath, mode="a") as sheet:
        for vendor in query_list:
            billing = vendor.children
            for record in billing:
                if billing_date == record.date_billed:
                    entry = ";".join(
             [str(record.date_billed), vendor.account_name, 
              vendor.account_number, str(record.payable),
              str(record.total_issues) ])
                    sheet.write(entry)
                await asyncio.sleep(1) 
```

这个 `generate_billing_sheet()` 协程服务将在以下 API 服务 `save_vendor_billing()` 中作为后台任务执行：

```py
@router.post("/billing/save/csv")
async def save_vendor_billing(billing_date:date, 
              tasks: BackgroundTasks):
    repo = BillingVendorRepository()
    result = await repo.join_vendor_billing()
    tasks.add_task(generate_billing_sheet, 
            billing_date, result)
    tasks.add_task(create_total_payables_year, 
            billing_date, result)
    return {"message" : "done"}
```

现在，在定义后台进程方面，并没有什么变化。我们通常将 `BackgroundTasks` 注入 API 服务方法，并使用 `add_task()` 提供任务调度、分配和执行。但由于现在的方法是利用协程，后台任务将使用事件循环而不是等待当前线程完成其工作。

如果后台进程需要参数，我们将在 `add_task()` 的 *第一个参数* 之后传递这些参数。例如，`generate_billing_sheet()` 的 `billing_date` 和 `query_list` 参数的参数应该在将 `generate_billing_sheet` 注入 `add_task()` 之后放置。此外，`billing_date` 的值应该在 `result` 参数之前传递，因为 `add_task()` 仍然遵循 `generate_billing_sheet()` 中参数声明的顺序，以避免类型不匹配。

所有异步后台任务将持续执行，即使它们的协程 API 服务已经向客户端返回了响应，也不会被 *await*。

## 创建多个任务

`BackgroundTasks` 允许创建多个异步事务，这些事务将在后台并发执行。在 `save_vendor_billing()` 服务中，为一个新的交易创建了一个名为 `create_total_payables_year()` 的事务，它需要与 `generate_billing_sheet()` 相同的参数。再次强调，这个新创建的任务将使用事件循环而不是线程。

当后台进程具有高 CPU 工作负载时，应用程序总是遇到性能问题。此外，由 `BackgroundTasks` 生成的任务无法从事务中返回值。让我们寻找另一种解决方案，其中任务可以管理高工作量并执行带有返回值的进程。

# 理解 Celery 任务

*Celery* 是一个在分布式系统上运行的非阻塞任务队列。它可以管理具有大量和重 CPU 工作负载的异步后台进程。它是一个第三方工具，因此我们需要首先通过 `pip` 安装它：

```py
pip install celery
```

它在单个服务器或分布式环境中并发地安排和运行任务。但它需要一个消息传输来发送和接收消息，例如*Redis*，一个可以用于字符串、字典、列表、集合、位图和流类型的消息代理的内存数据库。我们还可以在 Linux、macOS 和 Windows 上安装 Redis。现在，安装后，运行其`redis-server.exe`命令以启动服务器。在 Windows 中，Redis 服务默认设置为安装后运行，这会导致*TCP 绑定监听器*错误。因此，在运行启动命令之前，我们需要停止它。*图 8.1*显示了 Redis 服务在 Windows**任务管理器**中处于**停止**状态：

![图 8.1 – 停止 Redis 服务

![img/Figure_8.01_B17975.jpg]

图 8.1 – 停止 Redis 服务

停止服务后，我们现在应该看到 Redis 正在运行，如图*图 8.2*所示：

![图 8.2 – 运行中的 Redis 服务器

![img/Figure_8.02_B17975.jpg]

图 8.2 – 运行中的 Redis 服务器

## 创建和配置 Celery 实例

在创建 Celery 任务之前，我们需要将 Celery 实例放置在我们应用程序的专用模块中。*newsstand*原型将 Celery 实例放在`/services/billing.py`模块中，以下代码展示了 Celery 实例化的过程：

```py
from celery import Celery
from celery.utils.log import get_task_logger 
celery = Celery("services.billing",   
   broker='redis://localhost:6379/0', 
   backend='redis://localhost', 
   include=["services.billing", "models", "config"])
class CeleryConfig:
    task_create_missing_queues = True
    celery_store_errors_even_if_ignored = True
    task_store_errors_even_if_ignored = True 
    task_ignore_result = False
    task_serializer = "pickle"
    result_serializer = "pickle"
    event_serializer = "json"
    accept_content = ["pickle", "application/json", 
          "application/x-python-serialize"]
    result_accept_content = ["pickle", "application/json",
          "application/x-python-serialize"]
celery.config_from_object(CeleryConfig)
celery_log = get_task_logger(__name__)
```

要创建 Celery 实例，我们需要以下详细信息：

+   包含 Celery 实例的当前模块的名称（第一个参数）

+   作为我们的消息代理的 Redis 的 URL（`broker`）

+   存储和监控任务结果的后端（`backend`）

+   在消息体或 Celery 任务中使用的其他模块的列表（`include`）

实例化后，如果有的话，我们需要设置适当的序列化和内容类型来处理涉及的任务的传入和传出消息体。为了允许传递具有非 JSON 可序列化值的完整 Python 对象，我们需要将`pickle`作为支持的内容类型包括在内，然后向对象流声明默认的任务和结果序列化器。然而，使用`pickle`序列化器会带来一些安全问题，因为它倾向于暴露一些事务数据。为了避免损害应用程序，在执行消息操作之前，对消息对象进行清理，例如删除敏感值或凭据。

除了序列化选项之外，其他重要的属性，如`task_create_missing_queues`、`task_ignore_result`和与错误相关的配置，也应该成为`CeleryConfig`类的一部分。现在，我们在一个自定义类中声明所有这些细节，然后将其注入到 Celery 实例的`config_from_object()`方法中。

此外，我们可以通过其`get_task_logger()`方法创建一个 Celery 日志记录器，其名称为当前任务。

## 创建任务

*Celery 实例*的主要目标是注释 Python 方法以成为任务。Celery 实例有一个`task()`装饰器，我们可以将其应用于我们想要定义为异步任务的所有可调用过程。`task()`装饰器的一部分是任务的`name`，这是一个可选的唯一名称，由*package*、*module name(s)*和*transaction*的*method name*组成。它还有其他属性，可以添加更多细化到任务定义中，例如`auto_retry`列表，它注册了可能引起执行重试的`Exception`类，以及`max_tries`，它限制了任务的重试执行次数。顺便说一下，Celery 5.2.3 及以下版本只能从*non-coroutine methods*定义任务。

这里显示的`services.billing.tasks.create_total_payables_year_celery`任务将每天的应付金额相加，并返回总额：

```py
@celery.task(
    name="services.billing.tasks
            .create_total_payables_year_celery", 
                auto_retry=[ValueError, TypeError], 
                  max_tries=5)
def create_total_payables_year_celery(billing_date,
              query_list):
        total = 0.0
        for vendor in query_list:
            billing = vendor.children
            for record in billing:
                if billing_date == record.date_billed:
                    total += record.payable      
        celery_log.info('computed result: ' + str(total))
        return total   
```

给定的任务在运行时遇到`ValueError`或`TypeError`时只有五次(`5`)重试来恢复。此外，它是一个返回计算金额的函数，当使用`BackgroundTasks`时，这是不可能创建的。所有功能任务都使用*Redis*数据库作为它们返回值的临时存储，这也是为什么在 Celery 构造函数中有后端参数的原因。

## 调用任务

FastAPI 服务可以使用`apply_async()`或`delay()`函数调用这些任务。后者是一个更简单的选项，因为它预先配置好，只需要提供事务的参数即可获取结果。`apply_async()`函数是一个更好的选项，因为它接受更多细节，可以优化任务执行。这些细节包括`queue`、`time_limit`、`retry`、`ignore_result`、`expires`以及一些`kwargs`参数。但这两个函数都返回一个`AsyncResult`对象，该对象返回资源，如任务的`state`、帮助任务完成操作的`wait()`函数以及返回其计算值或异常的`get()`函数。以下是一个协程 API 服务，它使用`apply_async`方法调用`services.billing.tasks.create_total_payables_year_celery`任务：

```py
@router.post("/billing/total/payable")
async def compute_payables_yearly(billing_date:date):
    repo = BillingVendorRepository()
    result = await repo.join_vendor_billing()
    total_result = create_total_payables_year_celery
       .apply_async(queue='default', 
            args=(billing_date, result))
    total_payable = total_result.get(timeout=1)
    return {"total_payable": total_payable }
```

在`CeleryConfig`设置中将`task_create_missing_queues`设置为`True`总是推荐的做法，因为它会在工作服务器启动时自动创建任务`queue`，无论是默认的还是有其他名称。工作服务器将所有加载的任务放置在任务`queue`中以便执行、监控和检索结果。因此，在提取`AsyncResult`之前，我们应该始终在`apply_async()`函数的参数中定义一个任务`queue`。

`AsyncResult`对象有一个`get()`方法，它从`AsyncResult`实例中释放任务返回的值，无论是否有超时。在`compute_payables_yearly()`服务中，使用 5 秒的超时通过`get()`函数检索`AsyncResult`中的应付金额。现在让我们使用 Celery 服务器部署和运行我们的任务。

## 启动工作服务器

运行 Celery 工作进程创建一个处理和管理所有排队任务的单个进程。工作进程需要知道 Celery 实例是在哪个模块中创建的，以及任务以建立服务器进程。在我们的原型中，`services.billing` 模块是我们放置 Celery 应用程序的地方。因此，启动工作进程的完整命令如下：

```py
celery  -A services.billing worker -Q default -P solo -c 2 -l info
```

在这里，`-A` 指定了我们的 Celery 对象和任务的模块。`-Q` 选项表示工作进程将使用 *低*、*正常* 或 *高* 优先级队列。但首先，我们需要在 Celery 设置中将 `task_create_missing_queues` 设置为 `True`。我们还需要通过添加 `-c` 选项来指定工作进程执行任务所需的线程数。`-P` 选项指定工作进程将利用的 *线程池* 类型。默认情况下，Celery 工作进程使用适用于大多数 CPU 密集型事务的 `prefork pool`。其他选项有 *solo*、*eventlet* 和 *gevent*，但我们的设置将使用 *solo*，这是在微服务环境中运行 CPU 密集型任务的最佳选择。另一方面，`-l` 选项启用了我们在设置期间使用 `get_task_logger()` 设置的记录器。现在，也有方法来监控我们的运行任务，其中之一就是使用 Flower 工具。

## 监控任务

*花* 是芹菜的监控工具，它通过在基于网页的平台生成实时审计来观察和监控所有任务执行。但首先，我们需要使用 `pip` 来安装它：

```py
pip install flower
```

然后，我们使用以下带有 `flower` 选项的 `celery` 命令：

```py
celery -A services.billing flower
```

要查看审计，我们在浏览器中运行 `http://localhost:5555/tasks`。*图 8.3* 显示了由 `services.billing.tasks.create_total_payables_year_celery` 任务引起的执行日志的 *Flower* 快照：

![图 8.3 – Flower 监控工具](img/Figure_8.03_B17975.jpg)

图 8.3 – Flower 监控工具

到目前为止，我们已经使用 Redis 作为我们的内存后端数据库来存储任务结果和消息代理。现在让我们使用另一个可以替代 Redis 的异步消息代理，即 *RabbitMQ*。

# 使用 RabbitMQ 构建消息驱动的交易

RabbitMQ 是一个轻量级的异步消息代理，支持多种消息协议，如 *AMQP*、*STOM*、*WebSocket* 和 *MQTT*。在 Windows、Linux 或 macOS 上正常工作之前，它需要 *erlang*。其安装程序可以从 [`www.rabbitmq.com/download.html`](https://www.rabbitmq.com/download.html) 下载。

## 创建 Celery 实例

与使用 Redis 作为代理相比，RabbitMQ 作为消息代理是一个更好的替代品，它将在客户端和 Celery 工作线程之间中继消息。对于多个任务，RabbitMQ 可以命令 Celery 工作进程一次处理这些任务中的一个。RabbitMQ 代理适用于大量消息，并将这些消息保存到磁盘内存中。

首先，我们需要设置一个新的 Celery 实例，该实例将使用其 *guest* 账户利用 RabbitMQ 消息代理。我们将使用 AMQP 协议作为生产者/消费者类型消息设置的机制。以下是替换先前 Celery 配置的代码片段：

```py
celery = Celery("services.billing",   
    broker='amqp://guest:guest@127.0.0.1:5672', 
    result_backend='redis://localhost:6379/0', 
    include=["services.billing", "models", "config"])
```

如 Celery 的 `backend_result` 所示，Redis 仍然是后端资源，因为它在消息流量增加时仍然简单且易于控制和管理工作。现在让我们使用 RabbitMQ 来创建和管理消息驱动的交易。

## 监控 AMQP 消息

我们可以配置 RabbitMQ 管理仪表板以监控 RabbitMQ 处理的消息。设置完成后，我们可以使用账户详情登录仪表板来设置代理。*图 8.4* 展示了 RabbitMQ 对 API 服务多次调用 `services.billing.tasks.create_total_payables_year_celery` 任务的截图：

![图 8.4 – RabbitMQ 管理工具](img/Figure_8.04_B17975.jpg)

图 8.4 – RabbitMQ 管理工具

如果 RabbitMQ 仪表板无法捕获任务的执行行为，*Flower* 工具将始终是收集关于任务参数、`kwargs`、UUID、状态和处理日期详情的选项。如果 RabbitMQ 不是合适的消息工具，我们总是可以求助于 *Apache Kafka*。

# 使用 Kafka 构建发布/订阅消息

与 RabbitMQ 类似，*Apache Kafka* 是一种异步消息工具，应用程序使用它来在生产者和消费者之间发送和存储消息。然而，它比 RabbitMQ 更快，因为它使用带有分区的 *topics*，生产者可以在这些类似文件夹的结构中追加各种类型的消息。在这种架构中，消费者可以并行消费所有这些消息，这与基于队列的消息传递不同，后者允许生产者向只能按顺序消费消息的队列发送多个消息。在这个发布/订阅架构中，Kafka 可以以连续和实时模式每秒处理大量数据交换。

我们可以使用三个 Python 扩展将 FastAPI 服务与 Kafka 集成，即 `kafka-python`、`confluent-kafka` 和 `pykafka` 扩展。我们的在线 *newsstand* 原型将使用 `kafka-python`，因此我们需要使用 `pip` 命令安装它：

```py
pip install kafka-python
```

在这三个扩展中，只有 `kafka-python` 可以将 Java API 库通道并应用于 Python，以实现客户端的实现。我们可以从 [`kafka.apache.org/downloads`](https://kafka.apache.org/downloads) 下载 Kafka。

## 运行 Kafka 代理和服务器

Kafka 有一个 *ZooKeeper* 服务器，它管理和同步 Kafka 分布式系统内消息的交换。ZooKeeper 服务器作为代理运行，监控和维护 Kafka 节点和主题。以下命令启动服务器：

```py
C:\..\kafka\bin\windows\zookeeper-server-start.bat            C:\..\kafka\config\zookeeper.properties
```

现在，我们可以通过运行以下控制台命令来启动 Kafka 服务器：

```py
C:\..\kafka\bin\windows\kafka-server-start.bat                C:\..\kafka\config\server.properties
```

默认情况下，服务器将在本地的 `9092` 端口上运行。

## 创建主题

当两个服务器都已启动时，我们现在可以通过以下命令创建一个名为 `newstopic` 的主题：

```py
C:\..\kafka-topics.bat --create --bootstrap-server             localhost:9092 --replication-factor 1 --partitions 3         --topic newstopic
```

`newstopic` 主题有三个（`3`）分区，将保存我们 FastAPI 服务的所有附加消息。这些也是消费者将同时访问所有已发布消息的点。

## 实施发布者

在创建主题之后，我们现在可以实施一个生产者，该生产者将消息发布到 Kafka 集群。`kafka-python` 扩展有一个 `KafkaProducer` 类，它为所有运行的 FastAPI 线程实例化一个线程安全的生产者。以下是一个 API 服务，它将报纸信使记录发送到 Kafka 的 `newstopic` 主题，以便消费者访问和处理：

```py
from kafka import KafkaProducer
producer = KafkaProducer(
     bootstrap_servers='localhost:9092')
def json_date_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Data %s not serializable" % 
             type(obj))
@router.post("/messenger/kafka/send")
async def send_messnger_details(req: MessengerReq): 
    messenger_dict = req.dict(exclude_unset=True)
    producer.send("newstopic", 
       bytes(str(json.dumps(messenger_dict, 
          default=json_date_serializer)), 'utf-8')) 
    return {"content": "messenger details sent"}
```

协程 API 服务 `send_messenger_details()` 请求有关报纸信使的详细信息，并将它们存储在 `BaseModel` 对象中。然后，它以字节格式将配置文件详细信息的字典发送到集群。现在，消费 Kafka 任务的选项之一是运行其内置的 `kafka-console-consumer.bat` 命令。

## 在控制台上运行消费者

从控制台运行以下命令是消费 `newstopic` 主题当前消息的一种方法：

```py
kafka-console-consumer.bat --bootstrap-server                                                                                                                                                                                                                 127.0.0.1:9092 --topic newstopic
```

此命令创建一个消费者，该消费者将连接到 Kafka 集群，实时读取生产者发送的 `newtopic` 中的当前消息。*图 8.5* 展示了消费者在控制台上运行时的捕获情况：

![Figure 8.5 – The Kafka consumer]

![img/Figure_8.05_B17975.jpg]

图 8.5 – Kafka 消费者

如果我们希望消费者从 Kafka 服务器和代理开始运行的位置读取所有由生产者发送的消息，我们需要在命令中添加 `--from-beginning` 选项。以下命令将读取 `newstopic` 中的所有消息，并持续实时捕获传入的消息：

```py
kafka-console-consumer.bat --bootstrap-server 127.0.0.1:9092 --topic newstopic --from-beginning
```

使用 FastAPI 框架实现消费者的一种方法是使用 SSE（服务器发送事件）。典型的 API 服务实现无法满足 Kafka 消费者的需求，因为我们需要一个持续运行的服务，该服务订阅 `newstopic` 以获取实时数据。因此，现在让我们探讨如何在 FastAPI 框架中创建 SSE 以及它将如何消费 Kafka 消息。

# 实施异步服务器发送事件（SSE）

SSE 是一种服务器推送机制，它在不重新加载页面的情况下将数据发送到浏览器。一旦订阅，它就会为各种目的实时生成事件驱动的流。

在 FastAPI 框架中创建 SSE 只需要以下步骤：

+   来自 `sse_starlette.see` 模块的 `EventSourceResponse` 类

+   事件生成器

总而言之，该框架还允许使用协程实现整个服务器推送机制的非阻塞实现，这些协程甚至可以在 *HTTP/2* 上运行。以下是一个使用 SSE 的开放和轻量级协议实现 Kafka 消费者的协程 API 服务：

```py
from sse_starlette.sse import EventSourceResponse
@router.get('/messenger/sse/add')
async def send_message_stream(request: Request):

    async def event_provider():
        while True:
            if await request.is_disconnected():
                break
            message = consumer.poll()
            if not len(message.items()) == 0:
                for tp, records in message.items():
                   for rec in records:
                     messenger_dict = 
                      json.loads(rec.value.decode('utf-8'),
                       object_hook=date_hook_deserializer )

                     repo = MessengerRepository()
                     result = await 
                      repo.insert_messenger(messenger_dict)
                     id = uuid4()
                     yield {
                       "event": "Added … status: {},  
                           Received: {}". format(result, 
                            datetime.utcfromtimestamp(
                               rec.timestamp // 1000)
                               .strftime("%B %d, %Y 
                                      [%I:%M:%S %p]")),
                       "id": str(id),
                       "retry": SSE_RETRY_TIMEOUT,
                       "data": rec.value.decode('utf-8')
                      }

            await asyncio.sleep(SSE_STREAM_DELAY)
    return EventSourceResponse(event_provider())
```

`send_message_stream()` 是一个协程 API 服务，它实现了整个 SSE。它返回由 `EventSourceResponse` 函数生成的特殊响应。当 HTTP 流打开时，它持续从其源检索数据，并将任何内部事件转换为 SSE 信号，直到连接关闭。

另一方面，事件生成函数创建内部事件，这些事件也可以是异步的。例如，`send_message_stream()` 有一个嵌套的生成函数 `event_provider()`，它使用 `consumer.poll()` 方法消费生产者服务发送的最后一条消息。如果消息有效，生成器将检索到的消息转换为 `dict` 对象，并通过 `MessengerRepository` 将所有详细信息插入到数据库中。然后，它为 `EventSourceResponse` 函数产生所有内部详细信息，以便转换为 SSE 信号。*图 8.6* 展示了由 `send_message_stream()` 渲染的浏览器生成数据流：

![图 8.6 – SSE 数据流![图片](img/Figure_8.06_B17975.jpg)

图 8.6 – SSE 数据流

另一种实现 Kafka 消费者是通过 *WebSocket*。但这次，我们将关注如何使用 FastAPI 框架创建异步 WebSocket 应用程序的一般步骤。

# 构建异步 WebSocket

与 SSE 不同，WebSocket 的连接始终是**双向的**，这意味着服务器和客户端通过一个长 TCP 套接字连接相互通信。通信总是实时的，并且不需要客户端或服务器对发送的每个事件进行回复。

## 实现异步 WebSocket 端点

FastAPI 框架允许实现一个可以在 *HTTP/2* 协议上运行的异步 WebSocket。以下是一个使用协程块创建的异步 WebSocket 的示例：

```py
import asyncio
from fastapi import WebSocket
@router.websocket("/customer/list/ws")
async def customer_list_ws(websocket: WebSocket):
    await websocket.accept()
    repo = CustomerRepository()
    result = await repo.get_all_customer()

    for rec in result:
        data = rec.to_dict()
        await websocket.send_json(json.dumps(data, 
           default=json_date_serializer))
        await asyncio.sleep(0.01)
        client_resp = await websocket.receive_json()
        print("Acknowledging receipt of record id 
           {}.".format(client_resp['rec_id']))
    await websocket.close()    
```

首先，当使用 APIRouter 时，我们用 `@router.websocket()` 装饰协程函数，或者当使用 FastAPI 装饰器时，用 `@api.websocket()` 来声明 WebSocket 组件。装饰器还必须为 WebSocket 定义一个唯一的端点 URL。然后，WebSocket 函数必须将 `WebSocket` 注入为其第一个方法参数。它还可以包括其他参数，如查询和头参数。

`WebSocket` 注入器有四种发送消息的方式，即 `send()`、`send_text()`、`send_json()` 和 `send_bytes()`。应用 `send()` 将默认将每个消息管理为纯文本。之前的 `customer_list_ws()` 协程是一个以 JSON 格式发送每个客户记录的 WebSocket。

另一方面，WebSocket 注入器还可以提供四种方法，这些是 `receive()`、`receive_text()`、`receive_json()` 和 `receive_bytes()` 方法。`receive()` 方法默认期望消息是纯文本格式。现在，我们的 `customer_list_ws()` 端点期望从客户端接收 JSON 响应，因为它在其发送消息操作之后调用了 `receive_json()` 方法。

WebSocket 端点必须在交易完成后立即关闭连接。

## 实现 WebSocket 客户端

创建 WebSocket 客户端有许多方法，但这一章将专注于利用一个协程 API 服务，该服务在浏览器或 `curl` 命令被调用时将与异步的 `customer_list_ws()` 端点进行握手。以下是使用 `websockets` 库实现的 WebSocket 客户端代码，该库运行在 `asyncio` 框架之上：

```py
import websockets
@router.get("/customer/wsclient/list/")  
async def customer_list_ws_client():
    uri = "ws://localhost:8000/ch08/customer/list/ws"
    async with websockets.connect(uri) as websocket:
        while True:
           try:
             res = await websocket.recv()
             data_json = json.loads(res, 
                object_hook=date_hook_deserializer)

             print("Received record: 
                       {}.".format(data_json))

             data_dict = json.loads(data_json)
             client_resp = {"rec_id": data_dict['id'] }
             await websocket.send(json.dumps(client_resp))

           except websockets.ConnectionClosed:
                 break
        return {"message": "done"}
```

在使用 `websockets.connect()` 方法成功建立握手之后，`customer_list_ws_client()` 将会持续运行一个循环，从 WebSocket 端点获取所有传入的消费者详情。接收到的消息将被转换成其他进程所需的字典格式。现在，我们的客户端也会向 WebSocket 协程发送一个包含配置文件 *客户 ID* 的 JSON 数据的确认通知消息。一旦 WebSocket 端点关闭其连接，循环将停止。

让我们现在探索其他可以与 FastAPI 框架一起工作的异步编程特性。

# 在任务中应用反应式编程

反应式编程是一种涉及生成一系列操作以在过程中传播某些变化的流生成范式。Python 有一个 *RxPY* 库，它提供了我们可以应用于这些流以异步提取订阅者所需终端结果的方法。

在反应式编程范式中，所有在流中工作的中间操作符都会执行，以传播一些变化，如果在此之前有一个 `Observable` 实例和一个订阅此实例的 `Observer`。这个范式的主要目标是使用函数式编程在传播过程的最后达到预期的结果。

## 使用协程创建 Observable 数据

所有这一切都始于一个协程函数的实现，该函数将根据业务流程生成这些数据流。以下是一个 `Observable` 函数，它以 `str` 格式发布那些销售表现良好的出版物的详细信息：

```py
import asyncio
from rx.disposable import Disposable
async def process_list(observer):
      repo = SalesRepository()
      result = await repo.get_all_sales()

      for item in result:
        record = " ".join([str(item.publication_id),  
          str(item.copies_issued), str(item.date_issued), 
          str(item.revenue), str(item.profit), 
          str(item.copies_sold)])
        cost = item.copies_issued * 5.0
        projected_profit = cost - item.revenue
        diff_err = projected_profit - item.profit
        if (diff_err <= 0):
            observer.on_next(record)
        else:
            observer.on_error(record)
      observer.on_completed()
```

一个 `Observable` 函数可以是同步的或异步的。我们的目标是创建一个异步的，例如 `process_list()`。协程函数应该有以下回调方法，以符合 `Observable` 函数的资格：

+   一个 `on_next()` 方法，在给定一定条件下发出项目

+   一个 `on_completed()` 方法，在函数完成操作时执行一次

+   当在 `Observable` 上发生错误时被调用的 `on_error()` 方法

我们的 `process_list()` 发射获得一些利润的出版物的详细信息。然后，我们为 `process_list()` 协程的调用创建一个 `asyncio` 任务。我们创建了一个嵌套函数 `evaluate_profit()`，它返回 RxPY 的 `create()` 方法所需的 `Disposable` 任务，用于生成 Observable 流。当 Observable 流全部被消费时，此任务会被取消。以下是对执行异步 Observable 函数和使用 `create()` 方法从该 Observable 函数生成数据流的完整实现：

```py
def create_observable(loop):
    def evaluate_profit(observer, scheduler):
        task = asyncio.ensure_future(
            process_list(observer), loop=loop)
        return Disposable(lambda: task.cancel())
    return rx.create(evaluate_profit)
```

由 `create_observable()` 创建的订阅者是我们应用程序的 `list_sales_by_quota()` API 服务。它需要获取当前运行的事件循环，以便方法可以生成 Observable。之后，它调用 `subscribe()` 方法向流发送订阅并提取所需的结果。Observable 的 `subscribe()` 方法被调用以使客户端订阅流并观察发生的传播：

```py
@router.get("/sales/list/quota")
async def list_sales_by_quota():
    loop = asyncio.get_event_loop()
    observer = create_observable(loop)

    observer.subscribe(
        on_next=lambda value: print("Received Instruction 
              to buy {0}".format(value)),
        on_completed=lambda: print("Completed trades"),
        on_error=lambda e: print(e),
        scheduler = AsyncIOScheduler(loop) 
    )
    return {"message": "Notification 
           sent to the background"}
```

`list_sales_by_quote()` 协程服务展示了如何订阅一个 Observable。订阅者应利用以下回调方法：

+   一个 `on_next()` 方法来消费流中的所有项目

+   一个 `on_completed()` 方法来指示订阅的结束

+   一个在订阅过程中标记错误的 `on_error()` 方法

由于 Observable 的处理是异步的，因此调度器是一个可选参数，它提供了正确的管理器来调度和运行这些进程。API 服务使用 `AsyncIOScheduler` 作为订阅的适当调度。但还有其他生成 Observables 的快捷方式，这些快捷方式不使用自定义函数。

## 创建后台进程

当我们创建持续运行的 Observables 时，我们使用 `interval()` 函数而不是使用自定义的 `Observable` 函数。一些 Observables 被设计为成功结束，但一些则是为了在后台持续运行。以下 Observable 会定期在后台运行，以提供关于从报纸订阅中收到的总金额的一些更新：

```py
import asyncio
import rx
import rx.operators as ops
async def compute_subscriptions():
    total = 0.0
    repo = SubscriptionCustomerRepository()
    result = await repo.join_customer_subscription_total()

    for customer in result:
        subscription = customer.children
        for item in subscription:
            total = total + (item.price * item.qty)
    await asyncio.sleep(1)
    return total
def fetch_records(rate, loop) -> rx.Observable:
    return rx.interval(rate).pipe(
        ops.map(lambda i: rx.from_future(
          loop.create_task(compute_subscriptions()))),
        ops.merge_all()
    )
```

`interval()`方法以秒为单位定期创建数据流。但由于`pipe()`方法的执行，这个 Observable 对其流施加了一些传播。Observable 的`pipe()`方法创建了一个称为中间操作符的反应式操作符的管道。这个管道可以由一系列一次运行一个操作符的链式操作符组成，以改变流中的项目。似乎这一系列操作在订阅者上创建了多个订阅。因此，`fetch_records()`在其管道中有一个`map()`操作符，用于从`compute_subcription()`方法中提取结果。它在管道的末尾使用`merge_all()`来合并和展平创建的所有子流，形成一个最终的流，这是订阅者期望的流。现在，我们也可以从文件或 API 响应生成 Observable 数据。

## 访问 API 资源

创建 Observable 的另一种方法是使用`from_()`方法，它从文件、数据库或 API 端点提取资源。Observable 函数从我们的应用程序的 API 端点生成的 JSON 文档中检索其数据。假设我们正在使用`hypercorn`运行应用程序，它使用*HTTP/2*，因此我们需要通过将`httpx.AsyncClient()`的`verify`参数设置为`False`来绕过 TLS 证书。

以下代码突出了`fetch_subscription()`操作中的`from_()`，它创建了一个 Observable，从`https://localhost:8000/ch08/subscription/list/all`端点发出`str`数据流。Observable 的这些反应式操作符，即`filter()`、`map()`和`merge_all()`，用于在流中传播所需上下文：

```py
async def fetch_subscription(min_date:date, 
         max_date:date, loop) -> rx.Observable:
    headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    async with httpx.AsyncClient(http2=True, 
             verify=False) as client:
        content = await 
          client.get('https://localhost:8000/ch08/
            subscription/list/all', headers=headers)
    y = json.loads(content.text)
    source = rx.from_(y)
    observable = source.pipe(
      ops.filter(lambda c: filter_within_dates(
               c, min_date, max_date)),
      ops.map(lambda a: rx.from_future(loop.create_task(
            convert_str(a)))),
      ops.merge_all(),
    )
    return observable
```

`filter()`方法是另一个管道操作符，它从验证规则返回布尔值。它执行以下`filter_within_dates()`以验证从 JSON 文档检索的记录是否在订阅者指定的日期范围内：

```py
def filter_within_dates(rec, min_date:date, max_date:date):
    date_pur = datetime.strptime(
             rec['date_purchased'], '%Y-%m-%d')
    if date_pur.date() >= min_date and 
             date_pur.date() <= max_date:
        return True
    else:
        return False
```

另一方面，以下`convert_str()`是一个由`map()`操作符执行的协程函数，用于生成从 JSON 数据派生的报纸订阅者的简洁配置文件详情：

```py
async def convert_str(rec):
    if not rec == None:
        total = rec['qty'] * rec['price']
        record = " ".join([rec['branch'], 
            str(total), rec['date_purchased']])
        await asyncio.sleep(1)
        return record
```

运行这两个函数将原始发出的数据流从 JSON 修改为日期过滤的`str`数据流。另一方面，`list_dated_subscription()`协程 API 服务订阅`fetch_subscription()`以提取`min_date`和`max_date`范围内的报纸订阅：

```py
@router.post("/subscription/dated")
async def list_dated_subscription(min_date:date, 
            max_date:date):

    loop = asyncio.get_event_loop()
    observable = await fetch_subscription(min_date, 
             max_date, loop)

    observable.subscribe(
       on_next=lambda item: 
         print("Subscription details: {}.".format(item)),
       scheduler=AsyncIOScheduler(loop)
    )
```

尽管 FastAPI 框架尚未完全支持反应式编程，我们仍然可以创建可以与各种 RxPY 实用程序一起工作的协程。现在，我们将探讨协程不仅用于后台进程，也用于 FastAPI 事件处理器。

# 定制事件

FastAPI 框架具有一些称为 *事件处理器* 的特殊功能，这些功能在应用程序启动之前和关闭期间执行。每当 `uvicorn` 或 `hypercorn` 服务器重新加载时，都会激活这些事件。事件处理器也可以是协程。

## 定义启动事件

*启动事件* 是服务器在启动时执行的事件处理器。我们使用 `@app.on_event("startup")` 装饰器来创建启动事件。应用程序可能需要一个启动事件来集中处理某些事务，例如某些组件的初始配置或与数据相关的资源的设置。以下是一个应用程序启动事件示例，该事件为 GINO 存储库事务打开数据库连接：

```py
app = FastAPI()
@app.on_event("startup")
async def initialize():
    engine = await db.set_bind("postgresql+asyncpg://
          postgres:admin2255@localhost:5433/nsms")
```

这个 `initialize()` 事件定义在我们的应用程序的 `main.py` 文件中，以便 GINO 可以在每次服务器重新加载或重启时仅创建一次连接。

## 定义关闭事件

同时，*关闭事件*清理不需要的内存，销毁不需要的连接，并记录关闭应用程序的原因。以下是我们应用程序的关闭事件，该事件关闭了 GINO 数据库连接：

```py
@app.on_event("shutdown")
async def destroy():
    engine, db.bind = db.bind, None
    await engine.close()
```

我们可以在 APIRouter 中定义启动和关闭事件，但请确保这不会导致事务重叠或与其他路由器冲突。此外，事件处理器在挂载的子应用程序中不起作用。

# 摘要

除了使用基于 ASGI 的服务器之外，协程的使用是使 FastAPI 微服务应用程序快速的因素之一。本章已经证明，使用协程实现 API 服务比在线程池中利用更多线程能更好地提高性能。由于框架运行在 asyncio 平台上，我们可以利用 asyncio 工具设计各种设计模式来管理 CPU 密集型和 I/O 密集型服务。

本章使用了 Celery 和 Redis 来创建和管理后台异步任务，例如日志记录、系统监控、时间分片计算和批量作业。我们了解到 RabbitMQ 和 Apache Kafka 为构建 FastAPI 组件之间的异步和松散耦合通信提供了一个集成解决方案，特别是对于这些交互的消息传递部分。最重要的是，协程被应用于创建这些异步和非阻塞的后台进程以及消息传递解决方案，以提升性能。本章还介绍了通过 RxPy 扩展模块引入的反应式编程。

总体而言，本章得出结论，FastAPI 框架已准备好构建一个具有 *可靠*、*异步*、*消息驱动*、*实时消息传递* 和 *分布式核心系统* 的微服务应用程序。下一章将突出介绍其他 FastAPI 功能，这些功能提供了与 UI 相关的工具和框架的集成、使用 OpenAPI 规范的 API 文档、会话处理以及绕过 CORS。

# 第三部分：基础设施相关问题、数值和符号计算以及微服务测试

在本书的最后一部分，我们将讨论其他重要的微服务功能，例如分布式跟踪和日志记录、服务注册、虚拟环境和 API 度量。还将介绍使用 Docker 和 Docker Compose 进行无服务器部署，其中 NGINX 作为反向代理。此外，我们还将探讨 FastAPI 作为构建科学应用的框架，使用来自`numpy`、`scipy`、`sympy`和`pandas`模块的数值算法来模拟、分析和可视化其 API 服务的数学和统计解决方案。

本部分包括以下章节：

+   *第九章**，利用其他高级功能*

+   *第十章**，解决数值、符号和图形问题*

+   *第十一章**，添加其他微服务功能*
