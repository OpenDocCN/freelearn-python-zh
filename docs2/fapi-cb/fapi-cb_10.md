

# 第十章：将 FastAPI 与其他 Python 库集成

在本章中，我们将深入探讨通过将 FastAPI 与其他 **Python** 库集成来扩展 **FastAPI** 功能的过程。通过利用外部工具和库的力量，您可以增强 FastAPI 应用程序的功能，并为创建动态和功能丰富的 Web 服务解锁新的可能性。

在本章的整个过程中，您将学习如何将 FastAPI 与各种 Python 库集成，每个库都服务于不同的目的，并提供了独特的功能。从利用 **Cohere** 和 **LangChain** 的高级自然语言处理能力，到通过 **gRPC** 和 **GraphQL** 集成实时通信功能，您将发现如何结合其他流行的 Python 工具充分利用 FastAPI 的全部潜力。

通过将 FastAPI 与其他 Python 库集成，您将能够构建超越简单 **REST API** 的复杂 Web 应用程序。无论是开发由自然语言处理驱动的聊天机器人，还是集成 **机器学习**（**ML**）模型进行智能决策，可能性是无限的。

到本章结束时，您将具备利用外部工具和资源的能力，使您能够构建满足用户需求的复杂和功能丰富的 API。

本章包括以下配方：

+   将 FastAPI 与 gRPC 集成

+   将 FastAPI 与 GraphQL 连接

+   使用 Joblib 与 ML 模型

+   将 FastAPI 与 Cohere 集成

+   将 FastAPI 与 LangChain 集成

# 技术要求

要遵循本章中的配方，对 FastAPI 有良好的理解至关重要。此外，由于本章演示了如何将 FastAPI 与外部库集成，对每个库的基本了解可能有益。

然而，我们将提供外部链接供您回顾配方中使用的任何概念。您也可以在需要将技术与 FastAPI 集成时参考此章节。

本章使用的代码托管在 GitHub 上，网址为 [`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter10`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter10)。

建议在项目根目录中为项目设置一个虚拟环境，以有效地管理依赖项并保持项目隔离。

对于每个配方，您可以通过使用 GitHub 仓库中提供的 `requirements.txt` 文件，在您的虚拟环境中一次性安装所有依赖项：

```py
$ pip install –r requirements.txt
```

让我们开始深入探讨这个配方，并发现将 FastAPI 与外部库结合使用的潜力。

# 将 FastAPI 与 gRPC 集成

gRPC 是一个高性能、开源的通用**远程过程调用**（RPC）框架，最初由 Google 开发。它被设计为高效、轻量级，并且可以在不同的编程语言和平台之间互操作。将 FastAPI 与 gRPC 集成允许您利用 RPC 的力量来构建高效、可扩展和可维护的 API。

这个食谱将展示如何使用 FastAPI 在 REST 客户端和 gRPC 服务器之间构建网关。

## 准备工作

为了遵循这个食谱，有一些关于协议缓冲区的先验知识可能会有所帮助。您可以在[`protobuf.dev/overview/`](https://protobuf.dev/overview/)查看官方文档。

此外，我们将使用 proto3 版本来定义`.proto`文件。您可以在[`protobuf.dev/programming-guides/proto3/`](https://protobuf.dev/programming-guides/proto3/)查看语言指南。

我们将为这个食谱创建一个专门的根项目文件夹，命名为`grpc_gateway`。

除了`fastapi`和`uvicorn`之外，您还需要安装`grpcio`和`grpcio-tools`包。您可以通过使用 GitHub 仓库中提供的`requirements.txt`文件在您的环境中执行此操作，或者通过在您的环境中使用`pip`命令显式指定包，如下所示：

```py
$ pip install fastapi uvicorn grpcio grpcio-tools
```

在开始食谱之前，让我们按照以下步骤构建一个基本的 gRPC 服务器，该服务器有一个方法，该方法从客户端接收消息并发送回消息。

1.  在根项目下，让我们创建一个`grpcserver.proto`文件，其中包含我们服务器的定义，如下所示：

    ```py
    syntax = "proto3";
    service GrpcServer{
        rpc GetServerResponse(Message)
        returns (MessageResponse) {}
    }
    ```

1.  在同一文件中，我们将定义`Message`和`MessageResponse`消息，如下所示：

    ```py
    message Message{
    string message = 1;
    }
    message MessageResponse{
    string message = 1;
    bool received = 2;
    }
    ```

    从我们刚刚创建的`.proto`文件中，我们可以自动生成必要的 Python 代码，以集成服务和 gRPC 客户端，以及使用 proto 编译器。

1.  然后，在命令行终端中，运行以下命令：

    ```py
    $ python -m grpc_tools.protoc \
    --proto_path=. ./grpcserver.proto \
    --python_out=. \
    grpcserver_pb2_grpc.py and grpcserver_pb2.py. The grpcserver_pb2_grpc.py file contains the class to build the server a support function and a stub class that will be used by the client, while the grpcserver_pb2.py module contains the classes that define the messages. In our case, these are Message and MessageResponse.
    ```

1.  现在，让我们编写一个运行 gRPC 服务器的脚本。让我们创建一个名为`grpc_server.py`的文件，并定义服务器类，如下所示：

    ```py
    from grpcserver_pb2 import MessageResponse
    from grpcserver_pb2_grpc import GrpcServerServicer
    class Service(GrpcServerServicer):
        async def GetServerResponse(
            self, request, context
        ):
            message = request.message
            logging.info(f"Received message: {message}")
            result = (
                "Hello I am up and running, received: "
                f"{message}"
            )
            result = {
                "message": result,
                "received": True,
            }
            return MessageResponse(**result)
    ```

1.  然后，我们将定义一个函数，在本地主机的`50015`端口上运行服务器，如下所示：

    ```py
    import grpc
    from grpcserver_pb2_grpc import (
        add_GrpcServerServicer_to_server
    )
    async def serve():
        server = grpc.aio.server()
        add_GrpcServerServicer_to_server(
            Service(), server
        )
        server.add_insecure_port("[::]:50051")
        logging.info("Starting server on port 50051")
        await server.start()
        await server.wait_for_termination()
    ```

1.  我们通过将`serve`函数运行到事件循环中来关闭脚本：

    ```py
    import asyncio
    import logging
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        asyncio.run(serve())
    ```

这就是我们构建 gRPC 服务器所需的所有内容。现在我们可以从命令行运行脚本：

```py
$ python ./grpc_server.py
```

如果一切设置正确，您将在终端看到以下日志消息：

```py
INFO:root:Starting server on port 50051
```

当 gRPC 服务器运行时，我们现在可以利用 FastAPI 创建我们的网关。

## 如何做到这一点...

我们将创建一个 FastAPI 应用程序，其中有一个`GET /grpc`端点，该端点将接收一个消息作为参数，将请求转发到 gRPC 服务器，并将从 gRPC 服务器返回的消息发送给客户端。让我们按照以下步骤构建一个基本的网关应用程序。

1.  在项目根目录下，让我们创建一个名为`app`的文件夹，其中包含一个`main.py`模块，该模块包含服务器，如下所示：

    ```py
    from fastapi import FastAPI
    app = FastAPI()
    ```

1.  现在，让我们使用 Pydantic 创建响应类模式，该模式将反映 `MessageResponse` proto 类，如下所示：

    ```py
    from pydantic import BaseModel
    class GRPCResponse(BaseModel):
        message: str
        received: bool
    ```

1.  然后，我们将初始化 `grpc_channel` 对象，这是一个包含 gRPC 服务 URL 的抽象层，如下所示：

    ```py
    grpc_channel = grpc.aio.insecure_channel(
        "localhost:50051"
    )
    ```

1.  最后，我们可以按照以下方式创建我们的端点：

    ```py
    @app.get("/grpc")
    async def call_grpc(message: str) -> GRPCResponse:
        async with grpc_channel as channel:
            grpc_stub = GrpcServerStub(channel)
            response = await grpc_stub.GetServerResponse(
                Message(message=message)
            )
            return response
    ```

一旦我们创建了我们的 FastAPI 应用程序，让我们从命令行启动服务器：

```py
$ uvicorn app.main:app
```

打开交互式文档 `http://localhost:8000/docs`，您将看到新的端点，该端点将接受一个消息参数并从 gRPC 服务器返回响应。如果您尝试调用它，您还将看到 gRPC 服务器终端上的调用日志消息。

您已成功使用 FastAPI 设置了 REST-gRPC 网关！

## 还有更多...

我们创建了一个支持单一 RPC 的网关，单一 RPC 是类似于正常函数调用的简单 RPC。它涉及向服务器发送一个定义在 `.proto` 文件中的单个请求，并从服务器接收单个响应。然而，有各种类型的 RPC 实现可用，允许从客户端到服务器或从服务器到客户端的消息流，以及允许双向通信的实现。

使用 FastAPI 创建 REST 网关是一个相对简单的工作。有关如何在 Python 中实现不同类型的 gRPC 的更多信息，您可以参考以下文章：[`www.velotio.com/engineering-blog/grpc-implementation-using-python`](https://www.velotio.com/engineering-blog/grpc-implementation-using-python)。一旦您掌握了这些概念，您就可以轻松地将它们集成到 FastAPI 中，并为 gRPC 服务构建一个完整的网关。

## 参见

您可以在官方文档中深入了解协议缓冲区以及如何在 Python 代码中使用它：

+   *Protocol Buffer Python Generated* *代码*: [`protobuf.dev/reference/python/python-generated/`](https://protobuf.dev/reference/python/python-generated/)

您可以在 gRPC 官方文档页面上了解更多关于如何在 Python 代码中实现 gRPC 的信息：

+   *gRPC Python* *教程*: [`grpc.io/docs/languages/python/basics/`](https://grpc.io/docs/languages/python/basics/)

此外，还可以查看 GitHub 上的示例：

+   *gRPC Python GitHub* *示例*: [`github.com/grpc/grpc/tree/master/examples/python`](https://github.com/grpc/grpc/tree/master/examples/python)

# 将 FastAPI 与 GraphQL 连接

GraphQL 是一种用于 API 的查询语言和执行查询的运行时。它通过允许客户端指定他们确切需要的数据，提供了一个高效、强大且灵活的替代传统 REST API 的方案。将 FastAPI 与 GraphQL 集成使您能够构建高度可定制的 API，能够处理复杂的数据需求。在本食谱中，我们将了解如何将 FastAPI 与 GraphQL 连接起来以查询用户数据库，允许您在 FastAPI 应用程序中创建 GraphQL 模式、定义解析器并公开 GraphQL 端点。

## 准备工作

为了遵循这个食谱，确保您已经对 GraphQL 有一些基本了解是有益的。您可以在[`graphql.org/learn/`](https://graphql.org/learn/)查看官方文档。

在本章的 GitHub 仓库文件夹中，有一个名为`graphql`的文件夹，我们将将其视为根项目文件夹。为了实现 GraphQL，我们将利用 Strawberry 库。请确保您已经将其安装到您的环境中，并包含 FastAPI。您可以通过在项目根目录中找到的`requirements.txt`文件来安装它，或者通过运行以下`pip`命令来安装：

```py
$ pip install fastapi uvicorn strawberry-graphql[fastapi]
```

安装完成后，我们可以开始制作。

## 如何做到这一点...

让我们创建一个基本的 GraphQL 端点，从数据库中检索特定国家的用户。让我们通过以下步骤来完成：

1.  让我们创建一个包含我们将用作数据库源的用户的`database.py`模块。如下定义`User`类：

    ```py
    from pydantic import BaseModel
    class User(BaseModel):
        id: int
        username: str
        phone_number: str
        country: str
    ```

1.  然后，您可以编写一个`users_db`对象，它将是一个`User`类对象的列表，或者从 GitHub 仓库中相应的`database.py`文件复制一个，仓库地址为[`raw.githubusercontent.com/PacktPublishing/FastAPI-Cookbook/main/Chapter10/graphql/database.py`](https://raw.githubusercontent.com/PacktPublishing/FastAPI-Cookbook/main/Chapter10/graphql/database.py)。

    它将看起来像这样：

    ```py
    users_db: list[User] = [
        User(
            id=1,
            username="user1",
            phone_number="1234567890",
            country="USA",
        ),
    # other users
    ]
    ```

    我们将使用这个列表作为我们简单查询的数据库。

1.  在一个名为`graphql_utils.py`的单独模块中，我们将定义查询。但首先，让我们如下定义查询返回的模型：

    ```py
    import strawberry
    @strawberry.type
    class User:
        username: str
        phone_number: str
        country: str
    ```

1.  然后我们将如下定义查询：

    ```py
    @strawberry.type
    class Query:
        @strawberry.field
        def users(
            self, country: str | None
        ) -> list[User]:
            return [
                User(
                    username=user.username,
                    phone_number=user.phone_number,
                    country=user.country,
                )
                for user in users_db
                if user.country == country
            ]
    ```

    查询接受一个国家作为参数，并返回该国家的所有用户。

1.  现在，在同一个文件中，让我们使用 FastAPI 路由器创建 GraphQL 模式：

    ```py
    from strawberry.fastapi import GraphQLRouter
    schema = strawberry.Schema(Query)
    graphql_app = GraphQLRouter(schema)
    ```

    最后一行将创建一个`fastapi.Router`实例，该实例将处理端点。

1.  让我们通过将路由器添加到主 FastAPI 实例的单独`main.py`模块中来最终确定端点，如下所示：

    ```py
    from fastapi import FastAPI
    from graphql_utils import graphql_app
    app = FastAPI()
    app.include_router(graphql_app, prefix="/graphql")
    ```

    我们将端点添加到 FastAPI 实例中，并定义了`/``graphql`路径。

这就是您在 FastAPI 中设置 GraphQL 端点所需的所有内容。

为了探索端点的潜力，让我们从命令行运行服务器：

```py
http://localhost:8000/graphql. You will see an interactive page for the endpoint. The page is divided into two panels. The left contains the query editor and the right will show you the response.
Try to make the following GraphQL query:

```

{

users(country: "USA") {

username

country

phoneNumber

}

}

```py

 You will see the result on the right panel, which will look something like this:

```

{

"data": {

"users": [

{

"username": "user1",

"country": "USA",

"phoneNumber": "1234567890"

}

]

}

}

```py

 You have learned how to create an interactive GraphQL endpoint. By combining RESTful endpoints with GraphQL, the potential for data querying and modification can be greatly expanded. Real-world scenarios may involve using REST endpoints to modify the database by adding, modifying, or removing records. GraphQL can then be used to query the database and extract valuable insights.
See also
You can consult the FastAPI official documentation on how to integrate GraphQL:

*   *FastAPI GraphQL* *Documentation*: [`fastapi.tiangolo.com/how-to/graphql/`](https://fastapi.tiangolo.com/how-to/graphql/)

Also, in the Strawberry documentation, you can find a dedicated page on FastAPI integration:

*   *Integrate FastAPI with* *Strawberry*: [`strawberry.rocks/docs/integrations/fastapi`](https://strawberry.rocks/docs/integrations/fastapi)

Using ML models with Joblib
ML models are powerful tools for data analysis, prediction, and decision-making in various applications. FastAPI provides a robust framework for building web services, making it an ideal choice for deploying ML models in production environments. In this recipe, we will see how to integrate an ML model with FastAPI using **Joblib**, a popular library for model serialization and deserialization in Python.
We will develop an AI-powered doctor application that can diagnose diseases by analyzing the symptoms provided.
Warning
Note that the diagnoses provided by the AI doctor should not be trusted in real-life situations, as it is not reliable.
Getting ready
Prior knowledge of ML is not mandatory but having some can be useful to help you follow the recipe.
We will apply the recipe to a new project, so create a folder named `ai_doctor` that we will use as the project root folder.
To ensure that you have all the necessary packages in your environment, you can install them using the `requirements.txt` file provided in the GitHub repository or from the command line:

```

$ pip install fastapi[all] joblib scikit-learn

```py

 We will download the model from the Hugging Face Hub, a centralized hub hosting pre-trained ML models that are ready to be used.
We will use the `human-disease-prediction` model, which is a relatively lightweight linear logistic regression model developed with the `scikit-learn` package. You can check it out at the following link: [`huggingface.co/AWeirdDev/human-disease-prediction`](https://huggingface.co/AWeirdDev/human-disease-prediction).
To download it, we will leverage the provided `huggingface_hub` Python package, so make sure you have it in your environment by running the following:

```

$ pip install huggingface_hub

```py

 Once the installation is complete, we can proceed with building our AI doctor.
How to do it…
Let’s follow these steps to create our AI doctor:

1.  Let’s start by writing the code to accommodate the ML model. In the project root folder, let's create the `app` folder containing a module called `utils.py`. In the module, we will declare a `symptoms_list` list containing all the symptoms accepted by the model. You can download the file directly from the GitHub repository at [`raw.githubusercontent.com/PacktPublishing/FastAPI-Cookbook/main/Chapter10/ai_doctor/app/utils.py`](https://raw.githubusercontent.com/PacktPublishing/FastAPI-Cookbook/main/Chapter10/ai_doctor/app/utils.py).

    You can find the complete list on the model’s documentation page at [`huggingface.co/AWeirdDev/human-disease-prediction`](https://huggingface.co/AWeirdDev/human-disease-prediction).

2.  Still in the `app` folder, let’s create the `main.py` module that will contain the `FastAPI` server class object and the endpoint. To incorporate the model into our application, we will utilize the FastAPI lifespan feature.

    We can define the lifespan context manager as follows:

    ```

    from fastapi import FastAPI

    from contextlib import asynccontextmanager

    ml_model = {}

    REPO_ID = "AWeirdDev/human-disease-prediction"

    FILENAME = "sklearn_model.joblib"

    @asynccontextmanager

    async def lifespan(app: FastAPI):

    ml_model["doctor"] = joblib.load(

    hf_hub_download(

    repo_id=REPO_ID, filename=FILENAME

    )

    )

    yield

    ml_model.clear()

    ```py

    The `lifespan` context manager serves as middleware and carries out operations before and after server start and shutdown. It retrieves the model from the Hugging Face Hub and stores it in the `ml_model` dictionary, so it to be used across the endpoints without the need to reload it every time it is called.

     3.  Once it has been defined, we need to pass it to the `FastAPI` object class as follows:

    ```

    app = FastAPI(

    title="AI Doctor",

    lifespan=lifespan

    )

    ```py

     4.  Now we need to create the endpoint that will take the symptoms as parameters and return the diagnosis.

    The idea is to return each symptom as a path parameter. Since we have `132` possible symptoms, we will create the parameters object dynamically with Pydantic and restrict our model to the first ten symptoms. In the `main.py` file, let’s create the `Symptoms` class used to accept the parameters with the `pydantic.create_model` function as follows:

    ```

    from pydantic import create_model

    from app.utils import symptoms_list

    query_parameters = {

    symp: (bool, False)

    for symp in symptoms_list[:10]

    }

    Symptoms = create_model(

    "Symptoms", **query_params

    )

    ```py

    We now have all that we need to create our `GET /``diagnosis` endpoint.

     5.  Let’s create our endpoint as follows:

    ```

    @app.get("/diagnosis")

    async def get_diagnosis(

    symptoms: Annotated[Symptoms, Depends()],

    ):

    array = [

    int(value)

    for _, value in symptoms.model_dump().items()

    ]

    array.extend(

    # adapt array to the model's input shape

    [0] * (len(symptoms_list) - len(array))

    )

    len(symptoms_list)

    diseases = ml_model["doctor"].predict([array])

    return {

    "diseases": [disease for disease in diseases]

    }

    ```py

To test it, as usual, spin up the server with `uvicorn` from the command line by running the following:

```

$ uvicorn app.main:app

```py

 Open the interactive documentation from the browser at `http://localhost:8000/docs`. You will see the only `GET /diagnosis` endpoint and you will be able to select the symptoms. Try to select some of them and get your diagnosis from the AI doctor you have just created.
You have just created a FastAPI application that integrates an ML model. You can use the same model for different endpoints, but you can also integrate multiple models within the same application with the same strategy.
See also
You can check the guidelines on how to integrate an ML model into FastAPI on the official documentation page:

*   *Lifespan* *Events*: [`fastapi.tiangolo.com/advanced/events/?h=machine+learning#use-case`](https://fastapi.tiangolo.com/advanced/events/?h=machine+learning#use-case)

You can have a look at the Hugging Face Hub platform documentation at the link:

*   *Hugging Face Hub* *Documentation*: [`huggingface.co/docs/hub/index`](https://huggingface.co/docs/hub/index)

Take a moment to explore the capabilities of the `scikit-learn` package by referring to the official documentation:

*   *Scikit-learn* *Documentation*: [`scikit-learn.org/stable/`](https://scikit-learn.org/stable/)

Integrating FastAPI with Cohere
Cohere offers powerful language models and APIs that enable developers to build sophisticated AI-powered applications capable of understanding and generating human-like text.
State-of-the-art language models, such as the **Generative Pre-trained Transformer** (**GPT**) series, have revolutionized how machines comprehend and generate natural language. These models, which are trained on vast amounts of text data, deeply understand human language patterns, semantics, and context.
By leveraging Cohere AI’s models, developers can empower their applications to engage in natural language conversations, answer queries, generate creative content, and perform a wide range of language-related tasks.
In this recipe, we will create an AI-powered chatbot using FastAPI and Cohere that suggests Italian cuisine recipes based on user queries.
Getting ready
Before starting the recipe, you will need a Cohere account and an API key.
You can create your account at the page [`dashboard.cohere.com/welcome/login`](https://dashboard.cohere.com/welcome/login) by clicking the **Sign up** button at the top. At the time of writing, you can create an account by using your existing GitHub or Google account.
Once logged in, you will see a welcome page and a platform menu on the left with some options. Click on **API keys** to access the API menu.
By default, you will have a trial key that is free of charge, but it is rate limited and it cannot be used for commercial purposes. For the recipe, it will be largely sufficient.
Now create the project root folder called `chef_ai` and store your API key in a file called `.env` under the project root folder as follows:

```

COHERE_API_KEY="your-cohere-api-key"

```py

 Warning
If you develop your project with a versioning system control such as Git, for example, make sure to not track any API keys. If you have done this already, even unintentionally, revoke the key from the Cohere API keys page and generate a new one.
Aside from the API key, make sure that you also have all the required packages in your environment. We will need `fastapi`, `uvicorn`, `cohere`, and `python-dotenv`. This last package will enable importing environment variables from the `.``env` file.
You can install all the packages with the `requirements.txt` file provided in the GitHub repository in the `chef_ai` project folder by running the following:

```

$ pip install -r requirements.txt

```py

 Alternatively you can install them one by one:

```

$ pip install fastapi uvicorn cohere python-dotenv

```py

 Once the installation is complete, we can dive into the recipe and create our “chef de cuisine” assistant.
How to do it…
We will create our chef cuisine assistant by using a message completion chat. Chat completion models take a list of messages as input and return a model-generated message as output. The first message to provide is the **system message**.
A system message defines how a chatbot behaves in a conversation, such as adopting a specific tone or acting as a specialist such as a senior UX designer or software engineer. In our case, the system message will tell the chatbot to behave like a chef de cuisine.
Let’s create an endpoint to call our chat through the following steps:

1.  Let’s create a `handlers.py` module under the project root and import the Cohere API key from the `.``env` file:

    ```

    from dotenv import load_dotenv

    load_dotenv()

    ```py

     2.  Let’s write the system message as follows:

    ```

    SYSTEM_MESSAGE = (

    "You are a skilled Italian top chef "

    "expert in Italian cuisine tradition "

    "that suggest the best recipes unveiling "

    "tricks and tips from Grandma's Kitchen"

    "shortly and concisely."

    )

    ```py

     3.  Define the Cohere asynchronous client as follows:

    ```

    from cohere import AsyncClient

    client = AsyncClient()

    ```py

     4.  Before creating the function the generate the message, let’s import the required modules as:

    ```

    from cohere import ChatMessage

    from cohere.core.api_error import ApiError

    from fastapi import HTTPException

    ```py

     5.  Then, we can define the function to generate our message:

    ```

    async def generate_chat_completion(

    user_query=" ", messages=[]

    ) -> str:

    try:

    response = await client.chat(

    message=user_query,

    model="command-r-plus",

    preamble=SYSTEM_MESSAGE,

    chat_history=messages,

    )

    messages.extend(

    [

    ChatMessage(

    role="USER", message=user_query

    ),

    ChatMessage(

    role="CHATBOT",

    message=response.text,

    ),

    ]

    )

    return response.text

    except ApiError as e:

    raise HTTPException(

    status_code=e.status_code, detail=e.body

    )

    ```py

    The function will take in input the user query and the messages previously exchanged during the conversation. If the response is returned with no errors, the messages list is updated with the new interaction, otherwise an HTTPException error is raised.

    We utilized `main.py` module, located under the project root folder, we can start defining the `messages` list in the application state at the startup with the lifespan context manager:

    ```

    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    @asynccontextmanager

    async def lifespan(app: FastAPI):

    yield {"messages": []}

    ```py

     6.  We then pass the `lifespan` context manager to the app object as:

    ```

    app = FastAPI(

    title="Chef Cuisine Chatbot App",

    lifespan=lifespan,

    )

    ```py

     7.  Finally, we can create our endpoint as follows:

    ```

    from typing import Annotated

    from fastapi import Body, Request

    from handlers import generate_chat_completion

    @app.post("/query")

    async def query_chat_bot(

    request: Request,

    query: Annotated[str, Body(min_length=1)],

    ) -> str:

    answer = await generate_chat_completion(

    query, request.state.messages

    )

    return answer

    ```py

    We enforce a minimum length for the query message `(Body(min_length=1))` to prevent the model from returning an error response.

You have just created an endpoint that interacts with our chef de cuisine chatbot.
To test it, spin up the server with `uvicorn`:

```

$ uvicorn main:app

```py

 Open the interactive documentation and start testing the endpoint. For example, you can prompt the model with a message such as the following:

```

"Hello, could you suggest a quick recipe for lunch to be prepared in less than one hour?"

```py

 Read the answer, then try asking the bot to replace some ingredients and continue the chat. Once you have completed your recipe, enjoy your meal!
Exercise
We have created a chatbot endpoint to interact with our assistant. However, for real-life applications, it can be useful to have an endpoint that returns all the messages exchanged.
Create a `GET /messages` endpoint that returns all the messages in a formatted way.
Also create an endpoint `POST /restart-conversation` that will flush all the messages and restart the conversation without any previous messages.
See also
You can have a look at the Cohere quickstart on building a chatbot on the official documentation page:

*   *Building a* *Chatbot*: [`docs.cohere.com/docs/building-a-chatbot`](https://docs.cohere.com/docs/building-a-chatbot)

In production environment, depending on the project’s needs and budget, you might want to choose from the several models available. You can see an overview of the models provided by Cohere here:

*   *Models* *Overview*: [`docs.cohere.com/docs/models`](https://docs.cohere.com/docs/models)

Integrating FastAPI with LangChain
LangChain is a versatile interface for nearly any **Large Language Model** (**LLM**) that allows developers to create LLM applications and integrate them with external data sources and software workflows. It was launched in October 2022 and quickly became a top open source project on GitHub.
We will use LangChain and FastAPI to create an AI-powered assistant for an electronic goods store that provides recommendations and helps users.
We will set up a **Retrieval-Augmented Generation** (**RAG**) application, which involves empowering the model with personalized data to be trained. In this particular case, that would be a document of frequently asked questions.
This recipe will guide you through the process of integrating FastAPI with LangChain to create dynamic and interactive AI assistants that enhance the customer shopping experience.
Getting ready
Before starting the recipe, you will need a Cohere API key. If you don’t have it, you can check the *Getting ready* section of the *Integrating FastAPI with* *Cohere* recipe.
Create a project directory called `ecotech_RAG` and place the API key within a `.env` file, labeled as `COHERE_API_KEY`.
Previous knowledge of LLM and RAG is not required but having it would help.
Aside from the `fastapi` and `uvicorn` packages, you will need to install `python-dotenv` and the packages related to LangChain. You can do this by using `requirements.txt` or by installing them with `pip` as follows:

```

$ pip install fastapi uvicorn python-dotenv

$ pip install langchain

$ pip install langchain-community langchain-cohere

$ pip install chromadb unstructured

```py

 Once the installation is complete, we can start building our AI shop assistant.
How to do it…
We are going to create an application with a single endpoint that interacts with an LLM from Cohere.
The idea behind LangChain is to provide a series of interconnected modules, forming a chain to establish a workflow linking the user query with the model output.
We will split the process of creating the endpoint to interact with the RAG AI assistant into the following steps:

1.  Defining the prompts
2.  Ingesting and vectorizing the documents
3.  Building the model chain
4.  Creating the endpoint

Let’s start building our AI-powered assistant.
Defining the prompts
Like for the previous recipe, we will utilize a chat model that takes a list message as input. For this specific use case, however, we will supply the model with two messages: the system message and the user message. LangChain includes template objects for specific messages. Here are the steps to set up our prompts:

1.  Under the root project, create a module called `prompting.py`. Let’s start the module by defining a template message that will be used as the system message:

    ```

    template: str = """

    You are a customer support Chatbot.

    You assist users with general inquiries

    and technical issues.

    You will answer to the question:

    {question}

    你的答案将仅基于以下知识

    以下是你所训练的上下文。

    -----------

    {context}

    -----------

    如果你不知道答案，

    你将询问用户

    重新措辞问题或

    将用户重定向到 support@ecotech.com

    总是友好并乐于助人

    在对话结束时，

    询问用户是否满意

    如果是，提供答案，

    说再见并结束对话

    """

    ```py

    This is a common prompt for customer assistants that contains two variables: `question` and `context`. Those variables will be required to query the model.

     2.  With that template, we can define the system message as follows:

    ```

    从 langchain.prompts 导入(

    SystemMessagePromptTemplate,

    )

    system_message_prompt = (

    SystemMessagePromptTemplate.from_template(

    模板

    )

    )

    ```py

     3.  The user message does not require specific context and can be defined as follows:

    ```

    从 langchain.prompts 导入(

    HumanMessagePromptTemplate,

    )

    human_message_prompt = (

    HumanMessagePromptTemplate.from_template(

    template="{question}",

    )

    )

    ```py

     4.  Then we can group both messages under the dedicated chat message `template` object as follows:

    ```

    从 langchain.prompts 导入 ChatPromptTemplate

    chat_prompt_template = (

    ChatPromptTemplate.from_messages(

    [system_message_prompt, human_message_prompt]

    )

    )

    ```py

This is all we need to set up the prompt object to query our model.
Ingesting and vectorizing the documents
Our assistant will answer user questions by analyzing the documents we will provide to the model. Let’s create a `docs` folder under the project root that will contain the documents. First, download the `faq_ecotech.txt` file from the GitHub repository in the `ecotech_RAG/docs` project folder and save it in the local `docs` folder.
You can download it directly at [`raw.githubusercontent.com/PacktPublishing/FastAPI-Cookbook/main/Chapter10/ecotech_RAG/docs/faq_ecotech.txt`](https://raw.githubusercontent.com/PacktPublishing/FastAPI-Cookbook/main/Chapter10/ecotech_RAG/docs/faq_ecotech.txt).
Alternatively, you can create your own FAQ file. Just ensure that each question and answer is separated by one empty line.
The information contained in the file will be used by our assistant to help the customers. However, to retrieve the information, we will need to split our documents into chunks and store them as vectors to optimize searching based on similarity.
To split the documents, we will use a character-based text splitter. To store chunks, we will use Chroma DB, an in-memory vector database.
Then, let’s create a `documents.py` module that will contain the `load_documents` helper function that will upload the files into a variable as follows:

```

从 langchain.text_splitter 导入(

CharacterTextSplitter,

)

从 langchain_core.documents.base 导入 Document

从 langchain_community.document_loaders 导入(

DirectoryLoader,

)

从 langchain_community.vectorstores 导入 Chroma

async def load_documents(

db: Chroma,

):

text_splitter = CharacterTextSplitter(

chunk_size=100, chunk_overlap=0

)

raw_documents = DirectoryLoader(

"docs", "*.txt"

).load()

chunks = text_splitter.split_documents(

raw_documents

)

等待 db.aadd_documents(chunks)

```py

 The `DirectoryLoader` class uploads the content of all the `.txt` files from the `docs` folder, then the `text_splitter` object reorganizes the documents into document chunks of `100` characters that will be then added to the `Chroma` database.
By utilizing the vectorized database alongside the user query, we can retrieve the relevant context to feed into our model, which will analyze the most significant portion.
We can write a function for this called `get_context` as follows:

```

def get_context(

user_query: str, db: Chroma

) -> str:

docs = db.similarity_search(user_query)

return "\n\n".join(

doc.page_content for doc in docs

)

```py

 The documents have to be stored and vectorized in numerical representations called embedding. This can be done with Chroma, an AI-native vector database.
Then, through a similarity search operation (`db.similaratiry_search`) between the user query and the document chunks, we can retrieve the relevant content to pass as context to the model.
We have now retrieved the context to provide in the chat model system message template.
Building the model chain
Once we have defined the mechanism to retrieve the context, we can build the chain model. Let’s build it through the following steps:

1.  Let’s create a new module called `model.py`. Since we will use Cohere, we will upload the environment variables from the `.env` file with the `dotenv` package as follows:

    ```

    从 dotenv 导入 load_dotenv

    load_dotenv()

    ```py

     2.  Then we will define the model we are going to use:

    ```

    从 langchain_cohere 导入 ChatCohere

    model = ChatCohere(model="command-r-plus")

    ```py

    We will use the same module we used in the previous recipe, Command R+.

     3.  Now we can gather the pieces we have created to leverage the power of LangChain by creating the chain pipeline to query the model as follows:

    ```

    从 langchain.schema 导入 StrOutputParser

    从 prompting 导入 chat_prompt_template

    chain = (

    chat_prompt_template | model | StrOutputParser()

    )

    ```py

We will use the chain object to create our endpoint to expose through the API.
Creating the endpoint
We will make the `app` object instance with the endpoint in the `main.py` module under the project root folder. As always, let’s follow these steps to create it:

1.  The operation of loading the documents can be quite CPU-intensive, especially in real-life applications. Therefore, we will define a lifespan context manager to execute this process only at server startup. The `lifespan` function will be structured as follows:

    ```

    从 contextlib 导入 asynccontextmanager

    从 fastapi 导入 FastAPI

    从 langchain_cohere 导入 CohereEmbeddings

    从 langchain_community.vectorstores 导入 Chroma

    从 documents 导入 load_documents

    @asynccontextmanager

    async def lifespan(app: FastAPI):

    db = Chroma(

    )

    )

    等待 load_documents(db)

    yield {"db": db}

    ```py

     2.  We can then pass it to the FastAPI object as follows:

    ```

    app = FastAPI(

    title="Ecotech AI Assistant",

    lifespan=lifespan

    )

    ```py

     3.  Now, we can define a `POST /message` endpoint as follows:

    ```

    从 typing 导入 Annotated

    从 fastapi 导入 Body, Request

    从文档中导入 get_context

    从 model 导入 chain

    @app.post("/message")

    async def query_assistant(

    request: Request,

    question: Annotated[str, Body()],

    ) -> str:

    context = get_context(question, request.state.db)

    response = 等待 chain.ainvoke(

    {

    "question": question,

    "context": context,

    }

    )

    return response

    ```py

     4.  The endpoint will accept a body string text as input and will return the response from the model as a string based on the documents provided in the `docs` folder at startup.

To test it, you can spin up the server from the following command:

```

$ uvicorn main:app

```py

 Once the server has started, open the interactive documentation at `http://localhost:8000/docs` and you will see the `POST /message` endpoint we just created.
Try first to send a message that is not related to the assistance, something like the following:

```

"什么是比利时的首都？"

```py

 You will receive an answer like this:

```

"很抱歉，我无法回答这个问题，因为它超出了我的知识库。我是一个专门训练来回答与 EcoTech Electronics 相关的特定问题的 FAQ 聊天机器人，包括我们的产品与智能家居系统的兼容性、国际运费以及针对新客户的促销活动。如果您对这些话题有任何问题，我会很乐意帮助您！否则，对于一般性咨询，您可以联系我们的支持团队 support@ecotech.com。今天关于 EcoTech Electronics 还有其他我可以帮助您的事情吗？"

```py

 Then try to ask, for example, the following:

```

"我们接受哪些支付方式？"

```py

 You will get your assistance answer, which should be something like this:

```

"我们希望确保您在我们这里的购物体验尽可能顺畅和安全。对于在线购买，我们目前接受主要信用卡：Visa、Mastercard 和 American Express。您还可以选择通过 PayPal 支付，它提供额外的安全性和便利性。 \n\n 这些支付方式已集成到我们简单的在线结账流程中，确保交易快速高效。 \n\n 您是否对使用特定的支付方式感兴趣，或者您对我们接受的支付方式有任何进一步的问题？我们希望确保您的安心和购物体验的整体满意度。 \n\n 您对答案满意吗？"

```py

 You can double check that the answer is in line with what is written in the FAQ document in the `docs` folder.
You have just implemented a RAG AI-powered assistant with LangChain and FastAPI. You will now be able to implement your own AI assistant for your application.
Exercise
We have implemented the endpoint to interact with the chat model that will answer based on the document provided. However, real-life API applications will allow the addition of new documents interactively.
Create a new `POST /document` endpoint that will add a file in the `docs` folder and reload the documents in the code.
Have a look at the *Working with file uploads and downloads* recipe in *Chapter 2*, *Working with Data*, to see how to upload files in FastAPI.
See also
You can have a look at the quickstart in the LangChain documentation:

*   *LangChain* *Quickstart*: [`python.langchain.com/v0.1/docs/get_started/quickstart/`](https://python.langchain.com/v0.1/docs/get_started/quickstart/)

We have used Chroma, a vector database largely used for ML applications. Feel free to have a look at the documentation:

*   *Chroma*: [`docs.trychroma.com/`](https://docs.trychroma.com/)

```
