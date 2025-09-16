

# 第三章：使用 FastAPI 构建 RESTful API

在本章中，我们将深入探讨构建**RESTful API**的基本要素。RESTful API 是网络服务的骨架，它使得应用程序能够高效地进行通信和数据交换。

您将构建一个用于任务管理应用程序的 RESTful API。该应用程序将与 CSV 文件交互，尽管对于此类应用程序，典型的做法是使用数据库，如 SQL 或 NoSQL。这种方法是非传统的，并且由于可扩展性和性能限制，不建议在大多数场景中使用。然而，在某些情况下，特别是在遗留系统或处理大量结构化数据文件时，通过 CSV 管理数据可能是一个实用的解决方案。

我们的任务管理 API 将允许用户**创建、读取、更新和删除**（**CRUD**）任务，每个任务都表示为 CSV 文件中的一个记录。本例将提供在 FastAPI 中处理非标准格式数据的见解。

我们将了解如何测试 API 的端点。随着 API 的增长，管理复杂查询和过滤变得至关重要。我们将探讨实现高级查询功能的技术，增强 API 的可用性和灵活性。

此外，我们将解决 API 版本化的重要问题。版本化是随着时间的推移演进 API 而不破坏现有客户端的关键。您将学习管理 API 版本的战略，确保向后兼容性和用户平滑过渡。

最后，我们将介绍使用 OAuth2 保护 API，这是一种行业标准的授权协议。安全性在 API 开发中至关重要，您将获得实施身份验证和保护端点的实践经验。

在本章中，我们将涵盖以下食谱：

+   创建 CRUD 操作

+   创建 RESTful 端点

+   测试您的 RESTful API

+   处理复杂查询和过滤

+   API 版本化

+   使用 OAuth2 保护您的 API

+   使用 Swagger 和 Redoc 记录您的 API

# 技术要求

为了在*FastAPI 食谱集*中充分参与本章的学习，并有效地构建 RESTful API，您需要安装和配置以下技术和工具：

+   **Python**：请确保您的环境中安装了高于 3.9 版本的 Python。

+   **FastAPI**：应安装所有必需的依赖项。如果您尚未从前面的章节中安装，您可以从终端简单地使用以下命令进行安装：

    ```py
    $ pip install fastapi[all]
    ```

+   **Pytest**：您可以通过运行以下命令来安装此框架：

    ```py
    $ pip install pytest
    ```

注意，已经对 Pytest 框架有所了解可能会非常有用，以便更好地遵循*测试您的 RESTful API*食谱。

本章中使用的代码可在 GitHub 上找到，地址为：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter03`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter03)。

随时可以跟随或查阅，以防遇到困难。

# 创建 CRUD 操作

这个配方将向您展示如何使用作为数据库的 CSV 文件来实现基本的 CRUD 操作。

我们将开始为简单的任务列表草拟一个 CSV 格式的草案，并将操作放在一个单独的 Python 模块中。到配方结束时，您将拥有所有准备通过 API 端点使用的操作。

## 如何做到这一点…

让我们先创建一个名为 `task_manager_app` 的项目根目录，用于存放我们的应用程序代码库：

1.  进入根项目文件夹，创建一个 `tasks.csv` 文件，我们将将其用作数据库，并在其中放入一些任务：

    ```py
    id,title,description,status
    1,Task One,Description One,Incomplete
    2,Task Two,Description Two,Ongoing
    ```

1.  然后，创建一个名为 `models.py` 的文件，其中包含我们将用于内部代码的 Pydantic 模型。它看起来如下所示：

    ```py
    from pydantic import BaseModel
    class Task(BaseModel):
        title: str
        description: str
        status: str
    class TaskWithID(Task):
        id: int
    ```

    我们创建了两个独立的任务对象类，因为 `id` 在整个代码中都不会使用。

1.  在一个名为 `operations.py` 的新文件中，我们将定义与我们的数据库交互的函数。

    我们可以开始创建 CRUD 操作

    创建一个从 `.csv` 文件中检索所有任务的函数：

    ```py
    import csv
    from typing import Optional
    from models import Task, TaskWithID
    DATABASE_FILENAME = "tasks.csv"
    column_fields = [
        "id", "title", "description", "status"
    ]
    def read_all_tasks() -> list[TaskWithID]:
        with open(DATABASE_FILENAME) as csvfile:
            reader = csv.DictReader(
                csvfile,
            )
            return [TaskWithID(**row) for row in reader]
    ```

1.  现在，我们需要创建一个基于 `id` 读取特定任务的函数：

    ```py
    def read_task(task_id) -> Optional[TaskWithID]:
        with open(DATABASE_FILENAME) as csvfile:
            reader = csv.DictReader(
                csvfile,
            )
            for row in reader:
                if int(row["id"]) == task_id:
                    return TaskWithID(**row)
    ```

1.  要编写一个任务，我们需要一个策略来为新写入数据库的任务分配一个新的 `id`：

    一个好的策略是实施一个基于数据库中已存在的 ID 的逻辑，然后将任务写入我们的 CSV 文件，并将这两个操作组合到一个新的函数中。我们可以将创建任务操作拆分为三个函数。

    首先，让我们创建一个基于数据库中现有 ID 的函数来检索新 ID：

    ```py
    def get_next_id():
        try:
            with open(DATABASE_FILENAME, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                max_id = max(
                    int(row["id"]) for row in reader
                )
                return max_id + 1
        except (FileNotFoundError, ValueError):
            return 1
    ```

    然后，我们定义一个将任务写入 CSV 文件中具有 ID 的函数：

    ```py
    def write_task_into_csv(
        task: TaskWithID
    ):
        with open(
            DATABASE_FILENAME, mode="a", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=column_fields,
            )
            writer.writerow(task.model_dump())
    ```

    之后，我们可以利用这两个最后的功能来定义创建任务的函数：

    ```py
    def create_task(
        task: Task
    ) -> TaskWithID:
        id = get_next_id()
        task_with_id = TaskWithID(
            id=id, **task.model_dump()
        )
        write_task_into_csv(task_with_id)
        return task_with_id
    ```

1.  然后，让我们创建一个修改任务的函数：

    ```py
    def modify_task(
        id: int, task: dict
    ) -> Optional[TaskWithID]:
        updated_task: Optional[TaskWithID] = None
        tasks = read_all_tasks()
        for number, task_ in enumerate(tasks):
            if task_.id == id:
                tasks[number] = (
                    updated_task
                ) = task_.model_copy(update=task)
        with open(
            DATABASE_FILENAME, mode="w", newline=""
        ) as csvfile:  # rewrite the file
            writer = csv.DictWriter(
                csvfile,
                fieldnames=column_fields,
            )
            writer.writeheader()
            for task in tasks:
                writer.writerow(task.model_dump())
        if updated_task:
            return updated_task
    ```

1.  最后，让我们创建一个删除具有特定 `id` 的任务的函数：

    ```py
    def remove_task(id: int) -> bool:
        deleted_task: Optional[Task] = None
        tasks = read_all_tasks()
        with open(
            DATABASE_FILENAME, mode="w", newline=""
        ) as csvfile:  # rewrite the file
            writer = csv.DictWriter(
                csvfile,
                fieldnames=column_fields,
            )
            writer.writeheader()
            for task in tasks:
                if task.id == id:
                    deleted_task = task
                    continue
                writer.writerow(task.model_dump())
        if deleted_task:
            dict_task_without_id = (
                deleted_task.model_dump()
            )
            del dict_task_without_id["id"]
            return Task(**dict_task_wihtout_id)
    ```

您刚刚创建了基本的 CRUD 操作。我们现在准备通过 API 端点公开这些操作。

## 它是如何工作的…

您的 API 结构在 RESTful 设计中至关重要。它涉及定义端点（URI）并将它们与 HTTP 方法关联以执行所需的操作。

在我们的任务管理系统（Task Management system）中，我们将创建处理任务的端点，以反映常见的 CRUD 操作。以下是概述：

+   `列出任务` (`GET /tasks`) 获取所有任务的列表

+   `检索任务` (`GET /tasks/{task_id}`) 获取特定任务的详细信息

+   `创建任务` (`POST /task`) 添加一个新任务

+   `更新任务` (`PUT /tasks/{task_id}`) 修改现有任务

+   `删除任务` (`DELETE /tasks/{task_id}`) 删除一个任务

每个端点代表 API 中的一个特定函数，定义明确且目的明确。FastAPI 的路由系统允许我们轻松地将这些操作映射到 Python 函数。

练习

尝试为每个 CRUD 操作编写单元测试。如果您跟随 GitHub 仓库，您可以在 `Chapter03/task_manager_rest_api/test_operations.py` 文件中找到测试。

# 创建 RESTful 端点

现在，我们将创建路由来通过特定的端点公开每个 CRUD 操作。在这个菜谱中，我们将看到 FastAPI 如何利用 Python 类型注解来定义预期的请求和响应数据类型，从而简化验证和序列化数据的过程。

## 准备工作…

在开始菜谱之前，请确保您知道如何设置本地环境并创建一个基本的 FastAPI 服务器。您可以在 *第一章* 的 *创建新的 FastAPI 项目* 和 *理解 FastAPI 基础* 菜谱中查看它。

此外，我们还将使用之前菜谱中创建的 CRUD 操作。

## 如何做到这一点…

让我们在项目根目录中创建一个 `main.py` 文件来编写带有端点的服务器。FastAPI 简化了不同 HTTP 方法的实现，使它们与相应的 CRUD 操作相匹配。

现在，让我们为每个操作编写端点：

1.  使用 `read_all_tasks` 操作创建一个端点来列出所有任务的服务器：

    ```py
    from fastapi import FastAPI, HTTPException
    from models import (
        Task,
        TaskWithID,
    )
    from operations import read_all_tasks
    app = FastAPI()
    @app.get("/tasks", response_model=list[TaskWithID])
    def get_tasks():
        tasks = read_all_tasks()
        return tasks
    ```

1.  现在，让我们编写一个端点来根据 `id` 读取特定的任务：

    ```py
    @app.get("/task/{task_id}")
    def get_task(task_id: int):
        task = read_task(task_id)
        if not task:
            raise HTTPException(
                status_code=404, detail="task not found"
            )
        return task
    ```

1.  添加任务的端点如下：

    ```py
    from operations import create_task
    @app.post("/task", response_model=TaskWithID)
    def add_task(task: Task):
        return create_task(task)
    ```

1.  要更新任务，我们可以修改每个字段（`description`、`status` 或 `title`）。为此，我们创建一个用于正文的特定模型，称为 `UpdateTask`。端点将如下所示：

    ```py
    from operations import modify_task
    class UpdateTask(BaseModel):
        title: str | None = None
        description: str | None = None
        status: str | None = None
    @app.put("/task/{task_id}", response_model=TaskWithID)
    def update_task(
        task_id: int, task_update: UpdateTask
    ):
        modified = modify_task(
            task_id,
            task_update.model_dump(exclude_unset=True),
        )
        if not modified:
            raise HTTPException(
                status_code=404, detail="task not found"
            )
        return modified
    ```

1.  最后，这是删除任务的端点：

    ```py
    from operations import remove_task
    @app.delete("/task/{task_id}", response_model=Task)
    def delete_task(task_id: int):
        removed_task = remove_task(task_id)
        if not removed_task:
            raise HTTPException(
                status_code=404, detail="task not found"
            )
        return removed_task
    ```

您刚刚实现了与用作数据库的 CSV 文件交互的操作。

在项目根目录级别的命令行中，使用 `uvicorn` 命令启动服务器：

```py
$ uvicorn main:app
```

在浏览器中，访问 `http://localhost:8000/docs`，您将看到您刚刚创建的 RESTful API 的端点。

您可以通过创建一些任务，然后列出它们，更新它们，并直接通过交互式文档删除一些任务来实验。

# 测试您的 RESTful API

测试是 API 开发的一个关键部分。在 FastAPI 中，您可以使用各种测试框架，如 `pytest`，来编写 API 端点的测试。

在这个菜谱中，我们将为之前创建的每个端点编写单元测试。

## 准备工作…

如果尚未完成，请确保您已经通过运行以下命令在您的环境中安装了 `pytest`：

```py
$ pip install pytest
```

在测试中，使用一个专门的数据库来避免与生产数据库交互是一个好的实践。为了实现这一点，我们将创建一个测试固定装置，在每次测试之前生成数据库。

我们将在 `conftest.py` 模块中定义它，以便固定装置应用于项目根目录下的所有测试。让我们在项目根目录中创建该模块，并首先定义一个测试任务列表和用于测试的 CSV 文件名称：

```py
TEST_DATABASE_FILE = "test_tasks.csv"
TEST_TASKS_CSV = [
    {
        "id": "1",
        "title": "Test Task One",
        "description": "Test Description One",
        "status": "Incomplete",
    },
    {
        "id": "2",
        "title": "Test Task Two",
        "description": "Test Description Two",
        "status": "Ongoing",
    },
]
TEST_TASKS = [
    {**task_json, "id": int(task_json["id"])}
    for task_json in TEST_TASKS_CSV
]
```

我们现在可以创建一个将用于所有测试的固定装置。这个固定装置将在每个测试函数执行之前设置测试数据库。

我们可以通过将`autouse=True`参数传递给`pytest.fixture`装饰器来实现这一点，这表示该功能将在每个测试之前运行：

```py
import csv
import os
from pathlib import Path
from unittest.mock import patch
import pytest
@pytest.fixture(autouse=True)
def create_test_database():
    database_file_location = str(
        Path(__file__).parent / TEST_DATABASE_FILE
    )
    with patch(
        "operations.DATABASE_FILENAME",
        database_file_location,
    ) as csv_test:
        with open(
            database_file_location, mode="w", newline=""
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "id",
                    "title",
                    "description",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(TEST_TASKS_CSV)
            print("")
        yield csv_test
        os.remove(database_file_location)
```

由于固定装置定义在`conftest.py`模块中，每个测试模块将自动导入它。

现在，我们可以继续创建之前配方中创建的端点的实际单元测试函数：

## 如何操作…

为了测试端点，FastAPI 提供了一个特定的`TestClient`类，允许在不运行服务器的情况下测试端点。

在一个名为`test_main.py`的新模块中，让我们定义我们的测试客户端：

```py
from main import app
from fastapi.testclient import TestClient
client = TestClient(app)
```

我们可以像以下这样为每个端点创建测试。

1.  让我们从`GET /tasks`端点开始，该端点列出数据库中的所有任务：

    ```py
    from conftest import TEST_TASKS
    def test_endpoint_read_all_tasks():
        response = client.get("/tasks")
        assert response.status_code == 200
        assert response.json() == TEST_TASKS
    ```

    我们正在断言响应的状态码和`json`正文。

1.  就这么简单，我们可以通过创建`GET /tasks/{task_id}`的测试来继续，以读取具有特定`id`的任务：

    ```py
    def test_endpoint_get_task():
        response = client.get("/task/1")
        assert response.status_code == 200
        assert response.json() == TEST_TASKS[0]
        response = client.get("/task/5")
        assert response.status_code == 404
    ```

    除了现有任务的`200`状态码外，我们还断言当任务不存在于数据库中时，状态码等于`404`。

1.  以类似的方式，我们可以通过断言任务的新分配`id`来测试`POST /task`端点，以便将新任务添加到数据库中：

    ```py
    from operations import read_all_tasks
    def test_endpoint_create_task():
        task = {
            "title": "To Define",
            "description": "will be done",
            "status": "Ready",
        }
        response = client.post("/task", json=task)
        assert response.status_code == 200
        assert response.json() == {**task, "id": 3}
        assert len(read_all_tasks()) == 3
    ```

1.  修改任务的`PUT /tasks/{task_id}`端点的测试将是以下内容：

    ```py
    from operations import read_task
    def test_endpoint_modify_task():
        updated_fields = {"status": "Finished"}
        response = client.put(
            "/task/2", json=updated_fields
        )
        assert response.status_code == 200
        assert response.json() == {
             *TEST_TASKS[1],
             *updated_fields,
        }
        response = client.put(
            "/task/3", json=updated_fields
        )
        assert response.status_code == 404
    ```

1.  最后，我们测试`DELETE /tasks/{task_id}`端点以删除任务：

    ```py
    def test_endpoint_delete_task():
        response = client.delete("/task/2")
        assert response.status_code == 200
        expected_response = TEST_TASKS[1]
        del expected_response["id"]
        assert response.json() == expected_response
        assert read_task(2) is None
    ```

你已经为每个 API 端点编写了所有单元测试。

你现在可以从项目根目录运行测试，在终端中运行或在您最喜欢的编辑器的 GUI 支持下运行：

```py
$ pytest .
```

Pytest 将收集所有测试并运行它们。如果你正确编写了测试，你将在控制台输出中看到一条消息，表明你获得了 100%的分数。

## 参见

你可以在 Pytest 文档中检查测试固定装置：

+   *Pytest Fixtures* *参考*: [`docs.pytest.org/en/7.1.x/reference/fixtures.xhtml`](https://docs.pytest.org/en/7.1.x/reference/fixtures.xhtml)

你可以在官方文档中深入了解 FastAPI 测试工具和`TestClient` API：

+   *FastAPI* *Testing*: [`fastapi.tiangolo.com/tutorial/testing/`](https://fastapi.tiangolo.com/tutorial/testing/)

+   *FastAPI* *TestClient*: [`fastapi.tiangolo.com/reference/testclient/`](https://fastapi.tiangolo.com/reference/testclient/)

# 处理复杂查询和过滤

在任何 RESTful API 中，提供基于某些标准过滤数据的功能是至关重要的。在这个配方中，我们将增强我们的任务管理 API，允许用户根据不同的参数过滤任务并创建一个搜索端点。

## 准备中…

过滤功能将在现有的`GET /tasks`端点中实现，以展示如何超载端点，而搜索功能将在全新的端点中展示。在继续之前，请确保您已经实现了至少 CRUD 操作。

## 如何操作…

我们将首先通过过滤器对`GET /tasks`端点进行过度充电。我们修改端点以接受两个查询参数：`status`和`title`。

该端点将看起来如下所示：

```py
@app.get("/tasks", response_model=list[TaskWithID])
def get_tasks(
    status: Optional[str] = None,
    title: Optional[str] = None,
):
    tasks = read_all_tasks()
    if status:
        tasks = [
            task
            for task in tasks
            if task.status == status
        ]
    if title:
        tasks = [
            task for task in tasks if task.title == title
        ]
    return tasks
```

这两个参数可以可选地指定以过滤匹配其值的任务。

接下来，我们实现搜索功能。除了基本的过滤外，实现搜索功能可以显著提高 API 的可用性。我们将在新的端点中添加一个搜索功能，允许用户根据标题或描述中的关键词查找任务：

```py
@app.get("/tasks/search", response_model=list[TaskWithID])
def search_tasks(keyword: str):
    tasks = read_all_tasks()
    filtered_tasks = [
        task
        for task in tasks
        if keyword.lower()
        in (task.title + task.description).lower()
    ]
    return filtered_tasks
```

在`search_tasks`端点中，该函数会过滤任务，只包括标题或描述中包含关键词的任务。

要像往常一样启动服务器，请在命令行中运行此命令：

```py
$ uvicorn main:app
```

然后，转到交互式文档地址`http://localhost:8000/docs`，您将看到我们刚刚创建的新端点。

通过指定可能出现在您任务标题或描述中的某些关键词来尝试一下。

# API 版本控制

**API 版本控制**对于维护和演进网络服务而不中断现有用户至关重要。它允许开发者在提供向后兼容性的同时引入更改、改进或甚至破坏性更改。在这个食谱中，我们将实现我们的任务管理 API 的版本控制。

## 准备中…

要遵循食谱，您需要已经定义了端点。如果您还没有，可以先查看*创建 RESTful* *端点*的食谱。

## 如何做到这一点...

对于 API 版本控制，有几种策略。我们将使用最常见的方法，即 URL 路径版本控制，来为我们的 API 使用。

让我们考虑我们想要通过添加一个名为`priority`的新`str`字段来改进任务信息，该字段默认设置为`"lower"`。让我们通过以下步骤来完成它。

1.  让我们在`models.py`模块中创建一个名为`TaskV2`的对象类：

    ```py
    from typing import Optional
    class TaskV2(BaseModel):
        title: str
        description: str
        status: str
        priority: str | None = "lower"
    class TaskV2WithID(TaskV2):
        id: int
    ```

1.  在`operations.py`模块中，让我们创建一个名为`read_all_tasks_v2`的新函数，该函数读取所有任务，并添加`priority`字段：

    ```py
    from models import TaskV2WIthID
    def read_all_tasks_v2() -> list[TaskV2WIthID]:
        with open(DATABASE_FILENAME) as csvfile:
            reader = csv.DictReader(
                csvfile,
            )
            return [TaskV2WIthID(**row) for row in reader]
    ```

1.  我们现在已经拥有了创建`read_all_tasks`函数第二个版本所需的一切。我们将在`main.py`模块中完成这项工作：

    ```py
    from models import TaskV2WithID
    @app.get(
        "/v2/tasks",
        response_model=list[TaskV2WithID]
    )
    def get_tasks_v2():
        tasks = read_all_tasks_v2()
        return tasks
    ```

您刚刚创建了端点的第二个版本。这样，您就可以通过端点的几个版本来开发和改进您的 API。

为了测试它，让我们通过手动将新字段添加到`tasks.csv`文件中来修改它，以测试新端点：

```py
id,title,description,status,priority
1,Task One,Description One,Incomplete
2,Task Two,Description Two,Ongoing,higher
```

再次从命令行启动服务器：

```py
$ uvicorn main:app
```

现在，交互式文档`http://localhost:8000/docs`将显示新的`GET /v2/tasks`端点，以列出所有以版本 2 模式运行的任务。

检查端点是否列出了带有新`priority`字段的任务，并且旧的`GET /tasks`是否仍然按预期工作。

练习

你可能已经注意到，使用 CSV 文件作为数据库可能不是最可靠的解决方案。如果在更新或删除过程中进程崩溃，你可能会丢失所有数据。因此，通过使用与 SQLite 数据库交互的操作函数的新版本端点来改进 API。

## 更多内容…

当你对 API 进行版本控制时，你实际上是在提供一个区分不同 API 发布或版本的方法，允许客户端选择他们想要交互的版本。

除了我们在配方中使用的基于 URL 的方法之外，还有其他常见的 API 版本控制方法，例如以下内容：

+   **查询参数版本控制**：版本信息作为 API 请求中的查询参数传递。例如，参见以下内容：

    ```py
    https://api.example.com/resource?version=1
    ```

    这种方法保持了不同版本之间的基础 URL 统一。

+   **头部版本控制**：版本信息在 HTTP 请求的自定义头部中指定：

    ```py
    GET /resource HTTP/1.1
    Host: api.example.com
    X-API-Version: 1
    ```

    这种方法保持了 URL 的简洁性，但要求客户端在他们的请求中显式设置版本。

+   **基于消费者的版本控制**：这种策略允许客户选择他们需要的版本。他们在第一次交互时保存的版本将与他们的详细信息一起使用，并在所有未来的交互中使用，除非他们进行更改。

此外，可以使用`MAJOR.MINOR.PATCH`格式。`MAJOR`版本的更改表示不兼容的 API 更改，而`MINOR`和`PATCH`版本的更改表示向后兼容的更改。

版本控制允许 API 提供商在不破坏现有客户端集成的情况下引入更改（如添加新功能、修改现有行为或弃用端点和日落策略）。

它还让消费者控制何时以及如何采用新版本，最小化中断并保持 API 生态系统的稳定性。

## 参见

你可以查看 Postman 博客上关于 API 版本控制策略的一篇有趣的文章：

+   *Postman 博客 API* *版本控制*：[`www.postman.com/api-platform/api-versioning/`](https://www.postman.com/api-platform/api-versioning/)

# 使用 OAuth2 保护您的 API

在 Web 应用程序中，保护端点免受未经授权的用户访问至关重要。**OAuth2**是一个常见的授权框架，它允许应用程序通过具有受限权限的用户账户访问。它是通过发行令牌而不是凭据来工作的。本配方将展示如何在我们的任务管理器 API 中使用 OAuth2 来保护端点。

## 准备中…

FastAPI 支持使用密码的 OAuth2，包括使用外部令牌。数据合规性法规要求密码不以明文形式存储。相反，通常的方法是存储散列操作的输出，这会将明文转换为人类无法读取的字符串，并且无法逆转。

重要提示

仅为了展示功能，我们将使用简单的机制来模拟散列机制以及令牌创建。出于明显的安全原因，请不要在生产环境中使用。

## 如何实现…

让我们在项目根目录中创建一个 `security.py` 模块，我们将在这里实现所有用于保护我们服务的工具。然后，让我们创建一个如下所示的安全端点。

1.  首先，让我们创建一个包含用户名和密码的用户列表的字典：

    ```py
    fake_users_db = {
        "johndoe": {
            "username": "johndoe",
            "hashed_password": "hashedsecret",
        },
        "janedoe": {
            "username": "janedoe",
            "hashed_password": "hashedsecret2",
        },
    }
    ```

1.  密码不应该以纯文本形式存储，而应该加密或散列。为了演示这个特性，我们通过在密码字符串前插入 `"hashed"` 来模拟散列机制：

    ```py
    def fakely_hash_password(password: str):
        return f"hashed{password}"
    ```

1.  让我们创建处理用户和从我们创建的 `dict` 数据库中检索用户的函数的类：

    ```py
    class User(BaseModel):
        username: str
    class UserInDB(User):
        hashed_password: str
    def get_user(db, username: str):
        if username in db:
            user_dict = db[username]
            return UserInDB(**user_dict)
    ```

1.  使用与我们刚才用于散列的类似逻辑，让我们创建一个模拟令牌生成器和模拟令牌解析器：

    ```py
    def fake_token_generator(user: UserInDB) -> str:
        # This doesn't provide any security at all
        return f"tokenized{user.username}"
    def fake_token_resolver(
        token: str
    ) -> UserInDB | None:
        if token.startswith("tokenized"):
            user_id = token.removeprefix("tokenized")
            user = get_user(fake_users_db, user_id)
            return user
    ```

1.  现在，让我们创建一个函数来从令牌中检索用户。为此，我们将使用 `Depends` 类来利用 FastAPI 提供的依赖注入（见 [`fastapi.tiangolo.com/tutorial/dependencies/`](https://fastapi.tiangolo.com/tutorial/dependencies/)），使用 `OAuthPasswordBearer` 类来处理令牌：

    ```py
    from fastapi import Depends, HTTPException, status
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    def get_user_from_token(
        token: str = Depends(oauth2_scheme),
    ) -> UserInDB:
        user = fake_token_resolver(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Invalid authentication credentials"
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    ```

    `oauth2scheme` 包含了将被交互式文档用于认证浏览器的 `/token` URL 端点。

重要提示

我们已经使用依赖注入从 `get_user_token` 函数中检索令牌，使用了 `fastapi.Depends` 对象。依赖注入模式不是 Python 语言的原生特性，它与 FastAPI 框架紧密相关。在*第八章* *高级特性和最佳实践*中，你可以找到一个专门的配方，称为 *实现* *依赖注入*。

1.  让我们在 `main.py` 模块中创建端点：

    ```py
    from fastapi import Depends, HTTPException
    from fastapi.security import OAuth2PasswordRequestForm
    from security import (
        UserInDB,
        fake_token_generator,
        fakely_hash_password,
        fake_users_db
    )
    @app.post("/token")
    async def login(
        form_data: OAuth2PasswordRequestForm = Depends(),
    ):
        user_dict = fake_users_db.get(form_data.username)
        if not user_dict:
            raise HTTPException(
                status_code=400,
                detail="Incorrect username or password",
            )
        user = UserInDB(**user_dict)
        hashed_password = fakely_hash_password(
            form_data.password
        )
        if not hashed_password == user.hashed_password:
            raise HTTPException(
                status_code=400,
                detail="Incorrect username or password",
            )
        token = fake_token_generator(user)
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    ```

    现在我们已经拥有了创建带有 OAuth2 认证的安全端点所需的一切。

1.  我们将要创建的端点将根据提供的令牌返回有关当前用户的信息。如果令牌没有授权，它将返回一个 `400` 异常：

    ```py
    from security import get_user_from_token
    @app.get("/users/me", response_model=User)
    def read_users_me(
        current_user: User = Depends(get_user_from_token),
    ):
        return current_user
    ```

    我们刚刚创建的端点将只能被允许的用户访问。

现在我们来测试我们的安全端点。在项目根目录的命令行终端中，通过运行以下命令启动服务器：

```py
$ uvicorn main:app
```

然后，打开浏览器，访问 `http://localhost:8000/docs`，你将注意到交互式文档中的新 `token` 和 `users/me` 端点。

你可能会在 `users/me` 端点注意到一个小锁形图标。如果你点击它，你会看到一个表单窗口，允许你获取令牌并将其直接存储在你的浏览器中，这样你就不必每次调用安全端点时都提供它。

练习

你刚刚学习了如何为你的 RESTful API 创建一个安全端点。现在，尝试在之前配方中创建的一些端点上实现安全性。

## 还有更多…

使用 OAuth2，我们可以定义一个**作用域**参数，该参数用于指定访问令牌在用于访问受保护资源时授予客户端应用的访问级别。作用域可以用来定义客户端应用代表用户可以执行或访问哪些操作或资源。

当客户端从资源所有者（用户）请求授权时，它会在授权请求中包含一个或多个作用域。在 FastAPI 中，这些作用域以`dict`的形式表示，其中键代表作用域的名称，值是描述。

授权服务器随后使用这些作用域来确定在颁发访问令牌时授予客户端应用的适当访问控制和权限。

本食谱的目的不是深入探讨在 FastAPI 中实现 OAuth2 作用域的细节。然而，您可以在官方文档页面找到实用示例，链接为：[`fastapi.tiangolo.com/advanced/security/oauth2-scopes/`](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/)。

## 参见

您可以在以下链接中查看 FastAPI 如何集成 OAuth2：

+   *简单的基于密码和 Bearer 的 OAuth2*：https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/

此外，您还可以在官方文档页面找到更多关于 FastAPI 中依赖注入的信息：

+   *依赖项*：[`fastapi.tiangolo.com/tutorial/dependencies/`](https://fastapi.tiangolo.com/tutorial/dependencies/)

# 使用 Swagger 和 Redoc 记录 API

当启动服务器时，FastAPI 会自动使用**Swagger UI**和**Redoc**为您的 API 生成文档。

此文档是从您的路由函数和 Pydantic 模型中派生出来的，对开发团队或 API 消费者来说非常有用。

在本食谱中，我们将看到如何根据特定需求自定义文档。

## 准备中…

默认情况下，FastAPI 提供了两个文档接口：

+   `/docs` 端点（例如，`http://127.0.0.1:8000/docs`）

+   `/redoc` 端点（例如，`http://127.0.0.1:8000/redoc`）

这些界面提供动态文档，用户可以查看和测试 API 端点和其详细信息。然而，这两份文档都可以进行修改。

## 如何实现...

FastAPI 允许自定义 Swagger UI。您可以通过`FastAPI`类的参数添加元数据、自定义外观和添加额外的文档。

您可以通过在`main.py`模块中的`app`对象提供额外的元数据，如`title`、`description`和`version`来增强您的 API 文档。

```py
app = FastAPI(
    title="Task Manager API",
    description="This is a task management API",
    version="0.1.0",
)
```

这些元数据将出现在 Swagger UI 和 Redoc 文档中。

如果您需要在某些条件下将 Swagger UI 暴露给第三方用户，您可以进一步自定义它。

让我们尝试隐藏文档中的`/token`端点。

在这种情况下，您可以使用 FastAPI 提供的`utils`模块，以以下方式在`dict`对象中检索 Swagger UI 的 OpenAPI 模式：

```py
from fastapi.openapi.utils import get_openapi
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Customized Title",
        version="2.0.0",
        description="This is a custom OpenAPI schema",
        routes=app.routes,
    )
    del openapi_schema["paths"]["/token"]
    app.openapi_schema = openapi_schema
    return app.openapi_schema
app = FastAPI(
    title="Task Manager API",
    description="This is a task management API",
    version="0.1.0",
)
app.openapi = custom_openapi
```

这就是您需要自定义 API 文档的所有内容。

如果您使用`uvicorn main:app`命令启动服务器并访问两个文档页面之一，`/token`端点将不再出现。

您现在可以自定义 API 文档，以提升您向客户展示的方式。

## 参见

您可以在官方文档页面上了解更多关于 FastAPI 生成元数据、特性和 OpenAPI 集成的信息：

+   *元数据和文档* *URLs*: [`fastapi.tiangolo.com/tutorial/metadata/`](https://fastapi.tiangolo.com/tutorial/metadata/)

+   *FastAPI* *特性*: [`fastapi.tiangolo.com/features/`](https://fastapi.tiangolo.com/features/)

+   *扩展* *OpenAPI*: [`fastapi.tiangolo.com/how-to/extending-openapi/`](https://fastapi.tiangolo.com/how-to/extending-openapi/)
