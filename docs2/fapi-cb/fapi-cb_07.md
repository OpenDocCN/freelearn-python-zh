# 7

# 将 FastAPI 与 NoSQL 数据库集成

在本章中，我们将探讨 FastAPI 与 **NoSQL** 数据库的集成。通过构建音乐流媒体平台应用程序的后端，您将学习如何使用 FastAPI 设置和使用流行的 NoSQL 数据库 **MongoDB**。

您还将学习如何执行 **创建、读取、更新和删除** （**CRUD**） 操作，使用索引进行性能优化，以及处理 NoSQL 数据库中的关系。此外，您还将学习如何将 FastAPI 与 **Elasticsearch** 集成以实现强大的搜索功能，保护敏感数据，并使用 **Redis** 实现缓存。

到本章结束时，您将深入理解如何有效地使用 FastAPI 与 NoSQL 数据库结合，以提升应用程序的性能和功能。

在本章中，我们将介绍以下食谱：

+   使用 FastAPI 设置 MongoDB

+   MongoDB 中的 CRUD 操作

+   处理 NoSQL 数据库中的关系

+   在 MongoDB 中使用索引

+   从 NoSQL 数据库中公开敏感数据

+   将 FastAPI 与 Elasticsearch 集成

+   在 FastAPI 中使用 Redis 进行缓存

# 技术要求

要跟随本章的食谱，请确保您的设置包括以下基本要素：

+   **Python**：应在您的计算机上安装版本 3.7 或更高版本

+   您的工作环境中的 `fastapi` 包

+   `asyncio`：熟悉 `asyncio` 框架和 `async`/`await` 语法，因为我们将贯穿整个食谱使用它们

本章中使用的代码托管在 GitHub 上，地址为：[`github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter07`](https://github.com/PacktPublishing/FastAPI-Cookbook/tree/main/Chapter07)。

您可以在项目根目录内为项目创建一个虚拟环境，以高效管理依赖项并保持项目隔离。

在您的虚拟环境中，您可以使用 `requirements.txt` 一次性安装所有依赖项，该文件位于 GitHub 仓库的项目文件夹中：

```py
$ pip install –r requirements.txt
```

对每个食谱中将要使用的工具的一般了解可能有益，尽管不是强制性的。每个食谱都将为您提供对所使用工具的最小解释。

# 使用 FastAPI 设置 MongoDB

在这个食谱中，您将学习如何使用 FastAPI 设置流行的文档型 NoSQL 数据库 **MongoDB**。您将学习如何管理 Python 包以与 MongoDB 交互，创建数据库，并将其连接到 FastAPI 应用程序。到食谱结束时，您将深入理解如何将 MongoDB 与 FastAPI 集成以存储和检索应用程序数据。

## 准备工作

要跟随这个食谱，您需要在您的环境中安装 Python 和 `fastapi package`。

此外，对于这个配方，确保你有一个正在运行且可访问的 MongoDB 实例，如果没有，请设置一个本地的。根据你的操作系统和你的个人偏好，你可以通过以下几种方式设置本地的 MongoDB 实例。请自由查阅以下链接上的官方文档，了解如何在你的本地机器上安装 MongoDB 社区版：[`www.mongodb.com/try/download/community`](https://www.mongodb.com/try/download/community)。

在整个章节中，我们将考虑运行在 http://localhost:27017 的 MongoDB 的本地实例。如果你在远程机器上运行 MongoDB 实例，或者使用不同的端口，请相应地调整 URL 引用。

你还需要在你的环境中安装 `motor` 包。如果你还没有使用 `requirements.txt` 安装包，你可以从命令行在你的环境中安装 `motor`：

```py
$ pip install motor
```

`asyncio` 库。

一旦我们有了正在运行且可访问的 MongoDB 实例，并且你的环境中已安装了 `motor` 包，我们就可以继续进行配方。

## 如何做到这一点...

让我们先创建一个名为 `streaming_platform` 的项目根文件夹，其中包含一个 `app` 子文件夹。在 `app` 中，我们创建一个名为 `db_connection.py` 的模块，其中将包含与 MongoDB 的连接信息。

现在，我们将通过以下步骤设置连接：

1.  在 `db_connecion.py` 模块中，让我们定义 MongoDB 客户端：

    ```py
    from motor.motor_asyncio import AsyncIOMotorClient
    mongo_client = AsyncIOMotorClient(
        "mongodb://localhost:27017"
    )
    ```

    我们将每次需要与运行在 http://localhost:27017 的 MongoDB 实例交互时使用 `mongo_client` 对象。

1.  在 `db_connection.py` 模块中，我们将创建一个函数来 ping MongoDB 实例以确保它正在运行。但首先，我们检索 FastAPI 服务器使用的 `uvicorn` 日志记录器，以便将消息打印到终端：

    ```py
    import logging
    logger = logging.getLogger("uvicorn.error")
    ```

1.  然后，让我们创建一个函数来 ping MongoDB，如下所示：

    ```py
    async def ping_mongo_db_server():
        try:
            await mongo_client.admin.command("ping")
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(
                f"Error connecting to MongoDB: {e}"
            )
            raise e
    ```

    该函数将 ping 服务器，如果它没有收到任何响应，它将传播一个错误，这将停止代码的运行。

1.  最后，我们需要在启动 FastAPI 服务器时运行 `ping_mongo_db_server` 函数。在 `app` 文件夹中，让我们创建一个 `main.py` 模块，其中包含用于启动和关闭我们的 FastAPI 服务器的上下文管理器：

    ```py
    from contextlib import asynccontextmanager
    from app.db_connection import (
        ping_mongo_db_server,
    )
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await ping_mongo_db_server(),
        yield
    ```

    `lifespan` 上下文管理器必须作为参数传递给 `FastAPI` 对象：

    ```py
    from fastapi import FastAPI
    app = FastAPI(lifespan=lifespan)
    ```

    服务器被包装在 `lifespan` 上下文管理器中，以在启动时执行数据库检查。

为了测试它，确保你的 MongoDB 实例已经运行，并且像往常一样，让我们从命令行启动服务器：

```py
$ uvicorn app.main:app
```

你将在输出中看到以下日志消息：

```py
INFO:    Started server process [1364]
INFO:    Waiting for application startup.
INFO:    Connected to MongoDB
INFO:    Application startup complete.
```

此消息确认我们的应用程序正确地与 MongoDB 实例进行了通信。

你刚刚设置了 FastAPI 应用程序和 MongoDB 实例之间的连接。

## 参见

你可以在 MongoDB 官方文档页面上了解更多关于 Motor 异步驱动程序的信息：

+   *Motor Async Driver* *设置*: https://www.mongodb.com/docs/drivers/motor/

对于 FastAPI 服务器的启动和关闭事件，您可以在本页面上找到更多信息：

+   *FastAPI 生命周期* *事件*: [`fastapi.tiangolo.com/advanced/events/`](https://fastapi.tiangolo.com/advanced/events/)

# MongoDB 中的 CRUD 操作

CRUD 操作是数据库数据操作的基础，使用户能够以高效、灵活和可扩展的方式创建、读取、更新和删除数据实体。

这个食谱将演示如何在 FastAPI 中创建端点，用于从 MongoDB 数据库创建、读取、更新和删除文档，这是我们流平台的核心。

## 准备工作

要跟随这个食谱，您需要一个数据库连接，MongoDB 已经与您的应用程序一起设置好了，否则，请回到之前的食谱，*使用 FastAPI 设置 MongoDB*，它将详细展示如何进行设置。

## 如何做到这一点…

在创建 CRUD 操作的端点之前，我们必须在 MongoDB 实例上初始化一个数据库，用于我们的流应用程序。

让我们在`app`目录下的一个名为`database.py`的专用模块中这样做，如下所示：

```py
from app.db_connection import mongo_client
database = mongo_client.beat_streaming
```

我们已经定义了一个名为`beat_streaming`的数据库，它将包含我们应用程序的所有集合。

在 MongoDB 服务器端，我们不需要采取任何行动，因为`motor`库将自动检查名为`beat_streaming`的数据库以及最终集合的存在性，如果它们不存在，它将创建它们。

在同一个模块中，我们可以创建一个函数来返回将作为端点依赖项使用的数据库，以提高代码的可维护性：

```py
def mongo_database():
    return database
```

现在，我们可以在`main.py`中定义我们的端点，用于每个 CRUD 操作，步骤如下。

1.  让我们从创建添加歌曲到`songs`集合的端点开始：

    ```py
    from bson import ObjectId
    from fastapi import Body, Depends
    from app.database import mongo_database
    from fastapi.encoders import ENCODERS_BY_TYPE
    ENCODERS_BY_TYPE[ObjectId] = str
    @app.post("/song")
    async def add_song(
        song: dict = Body(
            example={
                "title": "My Song",
                "artist": "My Artist",
                "genre": "My Genre",
            },
        ),
        mongo_db=Depends(mongo_database),
    ):
        await mongo_db.songs.insert_one(song)
        return {
            "message": "Song added successfully",
            "id": song["_id"],
        }
    ```

    该端点在体中接受一个通用的 JSON，并从数据库返回受影响的 ID。`ENCONDERS_BY_TYPE[ObjectID] = str`这一行指定 FastAPI 服务器，`song["_id"]`文档 ID 必须解码为`string`。

    选择 NoSQL 数据库的一个原因是不受 SQL 模式限制，这允许在管理数据时具有更大的灵活性。然而，在文档中提供一个示例可能会有所帮助。这是通过使用带有示例参数的`Body`对象类来实现的。

1.  获取歌曲的端点将非常直接：

    ```py
    @app.get("/song/{song_id}")
    async def get_song(
        song_id: str,
        db=Depends(mongo_database),
    ):
        song = await db.songs.find_one(
            {
                "_id": ObjectId(song_id)
                if ObjectId.is_valid(song_id)
                else None
            }
        )
        if not song:
            raise HTTPException(
                status_code=404,
                detail="Song not found"
            )
        return song
    ```

    应用程序将搜索具有指定 ID 的歌曲，如果找不到，则返回`404`错误。

1.  要更新歌曲，端点将看起来像这样：

    ```py
    @app.put("/song/{song_id}")
    async def update_song(
        song_id: str,
        updated_song: dict,
        db=Depends(mongo_database),
    ):
        result = await db.songs.update_one(
            {
                "_id": ObjectId(song_id)
                if ObjectId.is_valid(song_id)
                else None
            },
            {"$set": updated_song},
        )
        if result.modified_count == 1:
          return {
              "message": "Song updated successfully"
          }
        raise HTTPException(
            status_code=404, detail="Song not found"
        )
    ```

    如果歌曲 ID 不存在，端点将返回`404`错误，否则它将只更新请求体中指定的字段。

1.  最后，`delete`操作端点可以如下完成：

    ```py
    @app.delete("/song/{song_id}")
    async def delete_song(
        song_id: str,
        db=Depends(mongo_database),
    ):
        result = await db.songs.delete_one(
            {
                "_id": ObjectId(song_id)
                if ObjectId.is_valid(song_id)
                else None
            }
        )
        if result.deleted_count == 1:
            return {
                "message": "Song deleted successfully"
            }
        raise HTTPException(
            status_code=404, detail="Song not found"
        )
    ```

    您刚刚创建了与 MongoDB 数据库交互的端点。

现在，从命令行启动服务器并测试您刚刚在 http://localhost:8000/docs 的交互式文档中创建的端点。

如果您跟随 GitHub 存储库，您还可以使用链接中的脚本`fill_mongo_db_database.py`预先填充数据库：[`github.com/PacktPublishing/FastAPI-Cookbook/blob/main/Chapter07/streaming_platform/fill_mongo_db_database.py`](https://github.com/PacktPublishing/FastAPI-Cookbook/blob/main/Chapter07/streaming_platform/fill_mongo_db_database.py)

确保您还下载了同一文件夹中的`songs_info.py`。

您可以从终端按照以下方式运行脚本：

```py
$ python fill_mongo_db_database.py
```

如果您调用端点`GET /songs`，您将有一个预先填充的长列表歌曲以测试您的 API。

## 参考以下内容

您可以在官方文档链接中进一步调查`motor`提供的操作：

+   *Motor MongoDB Aynscio* *教程*: [`motor.readthedocs.io/en/stable/tutorial-asyncio.xhtml`](https://motor.readthedocs.io/en/stable/tutorial-asyncio.xhtml)

# 在 NoSQL 数据库中处理关系

与关系型数据库不同，NoSQL 数据库不支持连接或外键来定义集合之间的关系。

无模式数据库，如 MongoDB，不强制执行像传统关系型数据库那样的关系。相反，可以使用两种主要方法来处理关系：**嵌入**和**引用**。

嵌入涉及在单个文档中存储相关数据。这种方法适用于所有类型的关联，前提是嵌入的数据与父文档紧密相关。这种技术对于频繁访问的数据和单个文档的原子更新来说，对读取性能很有好处。然而，如果嵌入的数据频繁更改，它很容易导致数据重复和潜在的不一致性，从而引发大小限制问题。

**引用**涉及使用它们的对象 ID 或其他唯一标识符存储相关文档的引用。这种方法适用于多对一和多对多关系，其中相关数据很大，并且跨多个文档共享。

这种技术减少了数据重复，提高了独立更新相关数据的灵活性，但另一方面，由于多个查询导致读取操作的复杂性增加，从而在检索相关数据时性能变慢。

在这个食谱中，我们将通过向我们的流平台添加新的集合并使它们交互，来探索在 MongoDB 中处理数据实体之间关系的技术。

## 准备工作

我们将继续构建我们的流平台。请确保您已经遵循了本章中所有之前的食谱，或者您可以将这些步骤应用于与 NoSQL 数据库交互的现有应用程序。

## 如何实现它...

让我们看看如何实现嵌入和引用技术的关系。

### 嵌入

展示歌曲嵌入关系的合适候选者是专辑集合。一旦发布，专辑信息很少改变，甚至从不改变。

`album`文档将嵌套字段嵌入到`song`文档中：

```py
{
    "title": "Title of the Song",
    "artist": "Singer Name",
    "genre": "Music genre",
    "album": {
        "title": "Album Title",
        "release_year": 2017,
    },
}
```

当使用 MongoDB 时，我们可以使用相同的端点检索有关专辑和歌曲的信息。这意味着当我们创建一首新歌时，我们可以直接添加它所属专辑的信息。我们指定文档 song 的存储方式，MongoDB 负责其余部分。

启动服务器并测试`POST /song`端点。在 JSON 体中包含有关专辑的信息。注意检索到的 ID，并使用它来调用`GET /song`端点。由于我们尚未在响应模型中定义任何响应模式限制，端点将返回从数据库检索到的所有文档信息，包括专辑。

对于这个用例示例，没有必要担心，但对于某些应用程序，你可能不希望向最终用户披露一个字段。你可以定义一个响应模型（参见*第一章*，*使用 FastAPI 的初步步骤*，在*定义和使用请求和响应模型*食谱中）或者在该字段从`dict`对象返回之前将其删除。

你刚刚定义了一个使用嵌入策略的多对一关系，将歌曲与专辑相关联。

### 引用

引用关系的典型用例可以是创建播放列表。播放列表包含多首歌曲，每首歌曲可以出现在不同的播放列表中。此外，播放列表通常会被更改或更新，因此需要一个引用策略来管理这些关系。

在数据库方面，我们不需要采取任何行动，因此我们将直接创建创建播放列表和检索包含所有歌曲信息的播放列表的端点。

1.  你可以在`main.py`模块中定义创建播放列表的端点：

    ```py
    class Playlist(BaseModel):
        name: str
        songs: list[str] = []
    @app.post("/playlist")
    async def create_playlist(
        playlist: Playlist = Body(
            example={
                "name": "My Playlist",
                "songs": ["song_id"],
            }
        ),
        db=Depends(mongo_database),
    ):
        result = await db.playlists.insert_one(
            playlist.model_dump()
        )
        return {
            "message": "Playlist created successfully",
            "id": str(result.inserted_id),
        }
    ```

    该端点需要一个 JSON 体，指定播放列表名称和要包含的歌曲 ID 列表，并返回播放列表 ID。

1.  获取播放列表的端点将接受播放列表 ID 作为参数。你可以这样编写代码：

    ```py
    @app.get("/playlist/{playlist_id}")
    async def get_playlist(
        playlist_id: str,
        db=Depends(mongo_database),
    ):
        playlist = await db.playlists.find_one(
            {
                "_id": ObjectId(playlist_id)
                if ObjectId.is_valid(playlist_id)
                else None
            }
        )
        if not playlist:
            raise HTTPException(
                status_code=404,
                detail="Playlist not found"
            )
        songs = await db.songs.find(
            {
                "_id": {
                    "$in": [
                        ObjectId(song_id)
                        for song_id in playlist["songs"]
                    ]
                }
            }
        ).to_list(None)
        return {
            "name": playlist["name"],
            "songs": songs
        }
    ```

    注意，播放列表集合中的歌曲 ID 存储为字符串，而不是`ObjectId`，这意味着在查询时必须进行转换。

    此外，为了接收播放列表的歌曲列表，我们不得不进行两次查询：一次用于播放列表，一次用于根据 ID 检索歌曲。

现在你已经构建了创建和检索播放列表的端点，启动服务器：

```py
http://localhost:8000/docs and you will see the new endpoints: POST /playlist and GET /playlist.
To test the endpoints, create some songs and note their IDs. Then, create a playlist and retrieve the playlist with the `GET /playlist` endpoint. You will see that the response will contain the songs with all the information including the album.
At this point, you have all the tools to manage relationships between collections in MongoDB.
See also
We just saw how to manage relationships with MongoDB and create relative endpoints. Feel free to check the official MongoDB guidelines at this link:

*   *MongoDB Model* *Relationships*: [`www.mongodb.com/docs/manual/applications/data-models-relationships/`](https://www.mongodb.com/docs/manual/applications/data-models-relationships/)

Working with indexes in MongoDB
An **index** is a data structure that provides a quick lookup mechanism for locating specific pieces of data within a vast dataset. Indexes are crucial for enhancing query performance by enabling the database to quickly locate documents based on specific fields.
By creating appropriate indexes, you can significantly reduce the time taken to execute queries, especially for large collections. Indexes also facilitate the enforcement of uniqueness constraints and support the execution of sorted queries and text search queries.
In this recipe, we’ll explore the concept of indexes in MongoDB and we will create indexes to improve search performances for songs in our streaming platform.
Getting ready
To follow along with the recipe, you need to have a MongoDB instance already set up with at least a collection to apply indexes. If you are following along with the cookbook, make sure you went through the *Setting up MongoDB with FastAPI* and *CRUD operations in* *MongoDB* recipes.
How to do it…
Let’s imagine we need to search for songs released in a certain year. We can create a dedicated endpoint directly in the `main.py` module as follows:

```

@app.get("/songs/year")

async def get_songs_by_released_year(

year: int,

db=Depends(mongo_database),

):

query = db.songs.find({"album.release_year": year})

songs = await query.to_list(None)

返回 songs

```py

 The query will fetch all documents and filter the one with a certain `release_year`. To speed up the query, we can create a dedicated index on the release year. We can do it at the server startup in the `lifespan` context manager in `main.py`. A text search in MongoDB won’t be possible without a text index.
First, at the startup server, let’s create a text index based on the `artist` field of the collection document. To do this, let’s modify the `lifespan` context manager in the `main.py` module:

```

@asynccontextmanager

async def lifespan(app: FastAPI):

等待 ping_mongo_db_server()，

db = mongo_database()

await db.songs.create_index({"album.release_year": -1})

yield

```py

 The `create_index` method will create an index based on the `release_year` field sorted in descending mode because of the `-``1` value.
You’ve just created an index based on the `release_year` field.
How it works…
The index just created is automatically used by MongoDB when running the query.
Let’s check it by leveraging the explain query method. Let’s add the following log message to the endpoint to retrieve songs released in a certain year:

```

@app.get("/songs/year")

async def get_songs_by_released_year(

year: int,

db=Depends(mongo_database),

):

query = db.songs.find({"album.release_year": year})

explained_query = await query.explain()

logger.info(

"Index used: %s",

explained_query.get("queryPlanner", {})

.get("winningPlan", {})

.get("inputStage", {})

.get("indexName", "No index used"),

)

songs = await query.to_list(None)

return songs

```py

 The `explained_query` variable holds information about the query such as the query execution or index used for the search.
If you run the server and call the `GET /songs/year` endpoint, you will see the following message log on the terminal output:

```

INFO:    Index used: album.release_year_-1

```py

 This confirms that the query has correctly used the index we created to run.
There’s more…
Database indexes become necessary to run text search queries. Imagine we need to retrieve the songs of a certain artist.
To query and create the endpoint, we need to make a text index on the `artist` field. We can do it at the server startup like the previous index on `album.release_year`.
In the `lifespan` context manager, you can add the index creation:

```

@asynccontextmanager

async def lifespan(app: FastAPI):

await ping_mongodb_server(),

db = mongo_database()

await db.songs.drop_indexes()

await db.songs.create_index({"release_year": -1})

await db.songs.create_index({"artist": "text"})

yield

```py

 Once we have created the index, we can proceed to create the endpoint to retrieve the song based on the artist’s name.
In the same `main.py` module, create the endpoint as follows:

```

@app.get("/songs/artist")

async def get_songs_by_artist(

artist: str,

db=Depends(mongo_database),

):

query = db.songs.find(

{"$text": {"$search": artist}}

)

explained_query = await query.explain()

logger.info(

"Index used: %s",

explained_query.get("queryPlanner", {})

.get("winningPlan", {})

.get("indexName", "No index used"),

)

songs = await query.to_list(None)

return songs

```py

 Spin up the server from the command line with the following:

```

$ uvicorn app.main:app

```py

 Go to the interactive documentation at `http:/localhost:8000/docs` and try to run the new `GET /``songs/artist` endpoint.
Text searching allow you to fetch records based on text matching. If you have filled the database with the `fill_mongo_db_database.py` script you can try searching for Bruno Mars’s songs by specifying the family name `"mars"`. The query will be:

```

http://localhost:8000/songs/artist?artist=mars

```py

 This will return at the least the song:

```

[

{

"_id": "667038acde3a00e55e764cf7",

"title": "Uptown Funk",

"artist": "Mark Ronson ft. Bruno Mars",

"genre": "Funk/pop",

"album": {

"title": "Uptown Special",

"release_year": 2014

}

}

]

```py

 Also, you will see a message on the terminal output like:

```

INFO:    Index used: artist_text

```py

 That means that the database has used the correct index to fetch the data.
Important note
By using the `explanation_query` variable, you can also check the difference in the execution time. However, you need a huge number of documents in your collection to appreciate the improvement.
See also
We saw how to build a text index for the search over the artist and a numbered index for the year of release. MongoDB allows you to do more, such as defining 2D sphere index types or compound indexes. Have a look at the documentation to discover the potential of indexing your MongoDB database:

*   *Mongo* *Indexes*: https://www.mongodb.com/docs/v5.3/indexes/
*   *MongoDB Text* *Search*: [`www.mongodb.com/docs/manual/core/link-text-indexes/`](https://www.mongodb.com/docs/manual/core/link-text-indexes/)

Exposing sensitive data from NoSQL databases
The way to expose sensitive data in NoSQL databases is pivotal to protecting sensitive information and maintaining the integrity of your application.
In this recipe, we will demonstrate how to securely view our data through database aggregations with the intent to expose it to a third-party consumer of our API. This technique is known as **data masking**. Then, we will explore some strategies and best practices for securing sensitive data in MongoDB and NoSQL databases in general.
By following best practices and staying informed about the latest security updates, you can effectively safeguard your MongoDB databases against potential security threats.
Getting ready
To follow the recipe, you need to have a running FastAPI application with a MongoDB connection already set up. If don’t have it yet, have a look at the *Setting up MongoDB with FastAPI* recipe. In addition, you need a collection of sensitive data such as **Personal Identifiable Information** (**PII**) or other restricted information.
Alternatively, we can build a collection of users into our MongoDB database, `beat_streaming`. The document contains PIIs such as names and emails, as well as users actions on the platform. The document will look like this:

```

{

"name": "John Doe",

"email": "johndoe@email.com",

"year_of_birth": 1990,

"country": "USA",

"consent_to_share_data": True,

"actions": [

{

"action": "basic subscription",

"date": "2021-01-01",

"amount": 10,

},

{

"action": "unscription",

"date": "2021-05-01",

},

],

}

```py

 The `consent_to_share_data` field stores the consent of the user to share behavioral data with third-party partners.
Let’s first fill the collection users in our database. You can do this with a user’s sample by running the script provided in the GitHub repository:

```

$ python fill_users_in_mongo.py

```py

 If everything runs smoothly, you should have the collection users in your MongoDB instance.
How to do it…
Imagine we need to expose users data for marketing research to a third-party API consumer for commercial purposes. The third-party consumer does not need PII information such as names or emails, and they are also not allowed to have data from users who didn’t give their consent. This is a perfect use case to apply data masking.
In MongoDB, you can build aggregation pipelines in stages. We will do it step by step.

1.  Since the database scaffolding is an infrastructure operation rather than an application, let’s create the pipeline with the view in a separate script that we will run separately from the server.

    In a new file called `create_aggregation_and_user_data_view.py`, let’s start by defining the client:

    ```

    from pymongo import MongoClient

    client = MongoClient("mongodb://localhost:27017/")

    ```py

    Since we don’t have any need to manage high traffic, we will use the simple `pymongo` client instead of the asynchronous one. We will reserve the asynchronous to the sole use of the application interactions.

     2.  The pipeline stage follows a specific aggregations framework. The first step of the pipeline will be to filter out the users who didn’t approve the consent. This can be done with a `$``redact` stage:

    ```

    pipeline_redact = {

    "$redact": {

    "$cond": {

    "if": {

    "$eq": [

    "$consent_to_share_data", True

    ]

    },

    "then": "$$KEEP",

    "else": "$$PRUNE",

    }

    }

    }

    ```py

     3.  Then, we filter out the emails that shouldn’t be shared with a `$``unset` stage:

    ```

    pipeline_remove_email_and_name = {

    "$unset": ["email", "name"]

    }

    ```py

     4.  This part of the pipeline will prevent emails and names from appearing in the pipeline’s output. We will split stage definition into three dictionaries for a better understanding.

    First, we define the action to obfuscate the day for each date:

    ```

    obfuscate_day_of_date = {

    "$concat": [

    {

    "$substrCP": [

    "$$action.date",

    0,

    7,

    ]

    },

    "-XX",

    ]

    }

    ```py

     5.  Then, we map the new `date` field for each element of the actions list:

    ```

    rebuild_actions_elements = {

    "input": "$actions",

    "as": "action",

    "in": {

    "$mergeObjects": [

    "$$action",

    {"date": obfuscate_day_of_date},

    ]

    },

    }

    ```py

     6.  Then, we use a `$set` operation to apply the `rebuild_actions_element` operation to every record like that:

    ```

    pipeline_set_actions = {

    "$set": {

    "actions": {"$map": rebuild_actions_elements},

    }

    }

    ```py

     7.  Then, we gather the pipelines just created to define the entire pipeline stage:

    ```

    pipeline = [

    pipeline_redact,

    pipeline_remove_email_and_name,

    pipeline_set_actions,

    ]

    ```py

     8.  We can use the list of aggregation stages to retrieve results and create the view in the `__main__` section of the script:

    ```

    if __name__ == "__main__":

    client["beat_streaming"].drop_collection(

    "users_data_view"

    )

    client["beat_streaming"].create_collection(

    "users_data_view",

    viewOn="users",

    pipeline=pipeline,

    )

    users_data_view view will be created in our beat_streaming database.

    ```py

     9.  Once we have the view, we can create a dedicated endpoint to expose this view to a third-party customer without exposing any sensible data. We can create our endpoint in a separate module for clarity. In the `app` folder, let’s create the `third_party_endpoint.py` module. In the module, let’s create the module router as follows:

    ```

    from fastapi import APIRouter, Depends

    from app.database import mongo_database

    router = APIRouter(

    prefix="/thirdparty",

    tags=["third party"],

    )

    ```py

     10.  Then, we can define the endpoint:

    ```

    @router.get("/users/actions")

    async def get_users_with_actions(

    db=Depends(mongo_database),

    ):

    users = [

    user

    async for user in db.users_data_view.find(

    {}, {"_id": 0}

    )

    ]

    return users

    ```py

     11.  Once the endpoint function has been created, let’s include the new router in the `FastAPI` object in the `main.py` module:

    ```

    from app import third_party_endpoint

    ## rest of the main.py code

    app = FastAPI(lifespan=lifespan)

    app.include_router(third_party_endpoint.router)

    ## rest of the main.py code

    ```py

The endpoint is now implemented in our API. Let’s start the server by running the following command:

```

通过访问 http://localhost:8000/docs，您可以检查新创建的端点是否存在，并调用它以检索创建的视图中的所有用户，而无需任何敏感信息。

您刚刚创建了一个安全地公开用户数据的端点。可以通过在端点上实现**基于角色的访问控制**（RBAC），例如在配方*设置* *RBAC*中的第四章 *身份验证和授权*中，添加一个额外的安全层。

更多内容...

除了数据脱敏之外，通常还会添加额外的层来保护您的数据应用。其中最重要的如下：

+   **静态加密**

+   **传输加密**

+   **基于角色的访问控制**（RBAC）

MongoDB 的企业版本提供了三个现成的服务解决方案。是否使用它们由软件架构师自行决定。

**静态加密**涉及加密存储在 MongoDB 数据库中的数据，以防止未经授权访问敏感信息。MongoDB 的企业版本通过使用专用存储引擎提供内置的加密功能。通过启用静态加密，您可以确保数据在磁盘上加密，使得没有适当加密密钥的人无法读取。

**传输加密**确保在您的应用程序和 MongoDB 服务器之间传输的数据被加密，以防止窃听和篡改。MongoDB 支持使用**传输层安全性**（**TLS**）进行传输加密，它加密在您的应用程序和 MongoDB 服务器之间通过网络发送的数据。

**基于角色的访问控制**（RBAC）对于限制 MongoDB 数据库中敏感数据的访问至关重要。MongoDB 提供了强大的身份验证和授权机制来控制对数据库、集合和文档的访问。您可以根据不同的角色和权限创建用户账户，以确保只有授权用户可以访问和操作敏感数据。

MongoDB 支持 RBAC，允许您根据用户的责任分配特定的角色，并相应地限制对敏感数据的访问。

参见

在配方中，我们简要地了解了如何在 MongoDB 中创建聚合和视图。您可以自由地查看官方文档页面上的更多内容：

+   *MongoDB 聚合* *快速入门*: [`www.mongodb.com/developer/languages/python/python-quickstart-aggregation/`](https://www.mongodb.com/developer/languages/python/python-quickstart-aggregation/)

+   *MongoDB 视图* *文档*: [`www.mongodb.com/docs/manual/core/views/`](https://www.mongodb.com/docs/manual/core/views/)

在此链接中可以找到一个很好的例子，展示了如何在 MongoDB 中通过数据库聚合推进数据脱敏：

+   *MongoDB 数据脱敏* *示例*: [`github.com/pkdone/mongo-data-masking?tab=readme-ov-file`](https://github.com/pkdone/mongo-data-masking?tab=readme-ov-file)

您可以在官方文档页面上了解更多关于聚合框架命令的信息：

+   *聚合* 阶段：[`www.mongodb.com/docs/manual/reference/operator/aggregation-pipeline/`](https://www.mongodb.com/docs/manual/reference/operator/aggregation-pipeline/)

此外，一本关于 MongoDB 聚合的全面书籍，免费查阅，可在以下链接找到：

+   *《实用 MongoDB 聚合》* 书籍：[`www.practical-mongodb-aggregations.com`](https://www.practical-mongodb-aggregations.com)

将 FastAPI 与 Elasticsearch 集成

**Elasticsearch** 是一个功能强大的搜索引擎，提供快速高效的全文本搜索、实时分析和更多功能。通过将 Elasticsearch 与 FastAPI 集成，您可以启用高级搜索功能，包括关键字搜索、过滤和聚合。我们将逐步介绍在 FastAPI 应用程序中集成 Elasticsearch、索引数据、执行搜索查询和处理搜索结果的过程。

在本菜谱中，我们将为我们的流媒体平台创建一个特定的端点，以启用分析和增强您的 Web 应用程序的搜索功能。具体来说，我们将根据指定国家的观看次数检索前十个艺术家。

到本菜谱结束时，您将具备利用 Elasticsearch 在 FastAPI 项目中实现强大搜索功能的知识和工具。

准备工作

要跟随本菜谱，您需要一个正在运行的应用程序或继续在我们的流媒体平台上工作。

此外，您需要一个运行中的 Elasticsearch 实例，并且可以通过此地址访问：`http://localhost:9200`。

您也可以通过遵循官方指南在您的机器上安装 Elasticsearch：[`www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.xhtml`](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.xhtml)。

然后，如果您还没有使用 `requirements.txt` 安装包，您需要使用 `pip` 从命令行在您的环境中安装 Elasticsearch Python 客户端和 `aiohttp` 包。您可以使用以下命令完成此操作：

```py
$ pip install "elasticsearch>=8,<9" aiohttp
```

对 Elasticsearch 的**领域特定语言**（**DSL**）有基本了解将有助于更深入地理解我们将要实施的查询。

查看此链接的官方文档：[`www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.xhtml`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.xhtml)。

一旦您安装并运行了 Elasticsearch，我们就可以将其集成到我们的应用程序中。

如何操作…

我们将整个过程分解为以下步骤：

1.  在我们的 FastAPI 应用程序中设置 Elasticsearch，以便我们的 API 可以与 Elasticsearch 实例通信。

1.  创建一个 Elasticsearch 索引，以便我们的歌曲可以被索引并由 Elasticsearch 查询。

1.  构建查询以查询我们的歌曲索引。

1.  创建 FastAPI 端点以向 API 用户公开我们的分析端点。

让我们详细查看这些步骤。

在我们的 FastAPI 应用程序中设置 Elasticsearch

要与 Elasticsearch 服务器交互，我们需要在我们的 Python 代码中定义客户端。在已经定义了 MongoDB 参数的`db_connection.py`模块中，让我们定义 Elasticsearch 异步客户端：

```py
from elasticsearch import AsyncElasticsearch,
es_client = AsyncElasticsearch(
    "localhost:27017"
)
```

我们可以在同一模块中创建一个函数来检查与 Elasticsearch 的连接：

```py
from elasticsearch import (
    TransportError,
)
async def ping_elasticsearch_server():
    try:
        await es_client.info()
        logger.info(
            "Elasticsearch connection successful"
        )
    except TransportError as e:
        logger.error(
            f"Elasticsearch connection failed: {e}"
        )
        raise e
```

如果 ping 失败，函数将 ping Elasticsearch 服务器并传播错误。

然后，我们可以在`main.py`模块的`lifetime`上下文管理器中调用该函数：

```py
@asynccontextmanager
async def lifespan(app: FastAPI):
    await ping_mongo_db_server(),
    await ping_elasticsearch_server()
# rest of the code
```

这将确保应用程序在启动时检查与 Elasticsearch 服务器的连接，如果 Elasticsearch 服务器没有响应，它将传播错误。

创建 Elasticsearch 索引

首先，我们应该开始用歌曲文档集合填充我们的 Elasticsearch 实例。在 Elasticsearch 中，集合被称为*索引*。

歌曲文档应包含一个额外的字段，用于跟踪每个国家的观看信息。例如，新的文档歌曲将如下所示：

```py
{
    "title": "Song Title",
    "artist": "Singer Name",
    "album": {
    "title": "Album Title",
    "release_year": 2012,
    },
    "genre": "rock pop",
    "views_per_country": {
    "India": 50_000_000,
    "UK": 35_000_150_000,
    "Mexico": 60_000_000,
    "Spain": 40_000_000,
    },
}
```

你可以在项目 GitHub 仓库中的`songs_info.py`文件中找到一个采样歌曲列表。如果你使用该文件，你还可以定义一个函数来填充索引，如下所示：

```py
from app.db_connection import es_client
async def fill_elastichsearch():
    for song in songs_info:
        await es_client.index(
            index="songs_index", body=song
        )
    await es_client.close()
```

要根据国家的观看次数分组我们的歌曲，我们需要根据`views_per_country`字段获取数据，而对于前十位艺术家，我们将根据`artist`字段进行分组。

应将这些信息提供给索引过程，以便 Elasticsearch 了解如何索引索引内的文档以运行查询。

在一个名为`fill_elasticsearch_index.py`的新模块中，我们可以将此信息存储在一个`python`字典中：

```py
mapping = {
    "mappings": {
        "properties": {
            "artist": {"type": "keyword"},
            "views_per_country": {
                "type": "object",
                "dynamic": True,
            },
        }
    }
}
```

当创建索引时，`mapping`对象将作为参数传递给 Elasticsearch 客户端。我们可以定义一个函数来创建我们的`songs_index`：

```py
from app.db_connection import es_client
async def create_index():
    await es_client.options(
        ignore_status=[400, 404]
    ).indices.create(
        index="songs_index",
        body=mapping,
    )
    await es_client.close()
```

你可以将该函数运行在`main()`分组中，并使用模块的`__main__`部分如下运行：

```py
async def main():
    await create_index()
    await fill_elastichsearch() # only if you use it
if __name__ == "__main__":
    import asyncio
    asyncio.run(create_index())
```

然后，你可以从终端运行脚本：

```py
$ python fill_elasticsearch_index.py
```

现在索引已创建，我们只需将歌曲添加到索引中。你可以通过创建一个单独的脚本或运行 GitHub 仓库中提供的`fill_elasticsearch_index.py`来实现这一点。

我们刚刚在我们的 Elasticsearch 索引中设置了一个索引，填充了文档。让我们看看如何构建查询。

构建查询

我们将构建一个函数，根据指定的国家返回查询。

我们可以在`app`文件夹中的单独模块`es_queries.py`中这样做。查询应获取包含特定国家`views_per_country`映射索引的所有文档，并按降序排序：

```py
def top_ten_songs_query(country) -> dict:
    views_field = f"views_per_country.{country}"
    query = {
        "bool": {
            "must": {"match_all": {}},
            "filter": [
                {"exists": {"field": views_field}}
            ],
        }
    }
    sort = {views_field: {"order": "desc"}}
```

然后，我们过滤出我们希望在响应中包含的字段，如下所示：

```py
    source = [
        "title",
        views_field,
        "album.title",
        "artist",
    ]
```

最后，我们通过指定我们期望的列表大小来以字典的形式返回查询：

```py
      return {
        "index": "songs_index",
        "query": query,
        "size": 10,
        "sort": sort,
        "source": source,
    }
```

现在我们已经有了构建查询以检索指定国家前十个艺术家的函数，我们将在我们的端点中使用它。

创建 FastAPI 端点

一旦我们设置了 Elasticsearch 连接并制定了查询，创建端点就是一个简单的过程。让我们在`app`文件夹下的一个新模块`main_search.py`中定义它。让我们首先定义路由器：

```py
from fastapi import APIRouter
router = APIRouter(prefix="/search", tags=["search"])
```

然后，端点将是：

```py
from fastapi import Depends, HTTPException
from app.db_connection import es_client
def get_elasticsearch_client():
    return es_client
@router.get("/top/ten/artists/{country}")
async def top_ten_artist_by_country(
    country: str,
    es_client=Depends(get_elasticsearch_client),
):
    try:
        response = await es_client.search(
         *top_ten_artists_query(country)
    )
    except BadRequestError as e:
        logger.error(e)
        raise HTTPException(
            status_code=400,
            detail="Invalid country",
        )
    return [
        {
            "artist": record.get("key"),
            "views": record.get("views", {}).get(
                "value"
            ),
        }
        for record in response["aggregations"][
            "top_ten_artists"
        ]["buckets"]
    ]
```

在返回之前，查询结果将进一步调整，以提取我们感兴趣的唯一值，即艺术家和观看次数。

最后一步是将路由器包含到我们的`FastAPI`对象中，以包含端点。

在`main.py`模块中，我们可以添加路由器如下：

```py
import main_search
## existing code in main.py
app = FastAPI(lifespan=lifespan)
app.include_router(third_party_endpoint.router)
app.include_router(main_search.router)
## rest of the code
```

现在，如果您使用`uvicorn app.main:app`命令启动服务器并转到`http://localhost:8000/docs`的交互式文档，您将看到根据歌曲观看次数检索一个国家前十个艺术家的新创建的端点。

您刚刚创建了一个与 Elasticsearch 实例交互的 FastAPI 端点。您可以自由地创建自己的新端点。例如，您可以创建一个返回某个国家前十个歌曲的端点。

参见

由于我们使用了 Elasticsearch Python 客户端，您可以自由地深入了解官方文档页面：

+   *Elasticsearch Python* *客户端*：[`www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.xhtml`](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.xhtml)

+   *使用 Asyncio 与* *Elasticsearch*：[`elasticsearch-py.readthedocs.io/en/7.x/async.xhtml`](https://elasticsearch-py.readthedocs.io/en/7.x/async.xhtml)

要了解更多关于 Elasticsearch 索引的信息，请查看 Elasticsearch 文档：

+   *索引* *API*：[`www.elastic.co/guide/en/elasticsearch/reference/current/docs-index_.xhtml`](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-index_.xhtml)

您可以在此链接中找到映射指南：

+   *映射*：[`www.elastic.co/guide/en/elasticsearch/reference/current/mapping.xhtml`](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.xhtml)

最后，您可以在以下链接中深入了解搜索查询语言：

+   *查询* *DSL*：[`www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.xhtml`](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.xhtml)

在 FastAPI 中使用 Redis 进行缓存

Redis 是一个内存数据存储，可以用作缓存来提高 FastAPI 应用程序的性能和可伸缩性。通过在 Redis 中缓存频繁访问的数据，您可以减少对数据库的负载并加快 API 端点的响应时间。

在这个菜谱中，我们将探讨如何将 Redis 缓存集成到我们的流平台应用程序中，并将缓存一个端点作为示例。

准备工作

要跟随这个菜谱，您需要一个运行中的 Redis 实例，可通过 http://localhost:6379 地址访问。

根据您的机器和偏好，您有多种安装和运行它的方法。查看 Redis 文档以了解如何在您的操作系统上执行此操作：[`redis.io/docs/install/install-redis/`](https://redis.io/docs/install/install-redis/)。

此外，您还需要一个具有耗时端点的 FastAPI 应用程序。

或者，如果您遵循流平台，请确保您已经从之前的菜谱中创建了前十个艺术家的端点，*将 FastAPI* *与 Elasticsearch*集成。

您的环境还需要 Python 的 Redis 客户端。如果您还没有使用`requirements.txt`安装包，可以通过运行以下命令来完成：

```py
$ pip install redis
```

安装完成后，我们可以继续进行菜谱。

如何做到这一点…

一旦 Redis 运行并可通过`localhost:6379`访问，我们就可以将 Redis 客户端集成到我们的代码中：

1.  在`db_connection.py`模块中，我们已经为 Mongo 和 Elasticsearch 定义了客户端，现在让我们添加 Redis 的客户端：

    ```py
    from redis import asyncio as aioredis
    redis_client = aioredis.from_url("redis://localhost")
    ```

    2.  类似于其他数据库，我们可以在应用程序启动时创建一个 ping Redis 服务器的函数。该函数可以定义为以下内容：

    ```py
    async def ping_redis_server():
        try:
            await redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(
                f"Error connecting to Redis: {e}"
            )
            raise e
    ```

    3.  然后，将其包含在`main.py`中的`lifespan`上下文管理器中：

    ```py
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await ping_mongo_db_server(),
        await ping_elasticsearch_server(),
        await ping_redis_server(),
        yield
    ```

    现在，我们可以使用`redis_client`对象来缓存我们的端点。我们将缓存用于查询 Elasticsearch 的`GET /search/top/ten/artists`端点。

    4.  在`main_search.py`中，我们可以定义一个函数来检索 Redis 客户端作为依赖项：

    ```py
    def get_redis_client():
        return redis_client
    ```

    5.  然后，您可以按如下方式修改端点：

    ```py
    @router.get("/top/ten/artists/{country}")
    async def top_ten_artist_by_country(
        country: str,
        es_client=Depends(get_elasticsearch_client),
        redis_client=Depends(get_redis_client),
    ):
    ```

    6.  在函数的开始处，我们检索存储值的键，并检查该值是否已经存储在 Redis 中：

    ```py
        cache_key = f"top_ten_artists_{country}"
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            logger.info(
                f"Returning cached data for {country}"
            )
            return json.loads(cached_data)
    ```

    7.  然后，当我们看到数据不存在时，我们继续从 Elasticsearch 获取数据：

    ```py
        try:
            response = await es_client.search(
                 *top_ten_artists_query(country)
            )
        except BadRequestError as e:
            logger.error(e)
            raise HTTPException(
                status_code=400,
                detail="Invalid country",
            )
        artists = [
            {
                "artist": record.get("key"),
                "views": record.get("views", {}).get(
                    "value"
                ),
            }
            for record in response["aggregations"][
                "top_ten_artists"
            ]["buckets"]
        ]
    ```

    8.  一旦我们检索到列表，我们将其存储在 Redis 中，以便在后续调用中检索：

    ```py
        await redis_client.set(
            cache_key, json.dumps(artists), ex=3600
        )
        return artists
    ```

    9.  我们指定了一个过期时间，即记录将在 Redis 中停留的秒数。在此时间之后，记录将不再可用，艺术家列表将从 Elasticsearch 中重新调用。

现在，如果您使用`uvicorn app.main:app`命令运行服务器并尝试调用意大利的端点，您将注意到第二次调用的响应时间将大大减少。

您已经使用 Redis 为我们的应用程序的一个端点实现了缓存。使用相同的策略，您可以自由地缓存所有其他端点。

还有更多…

在撰写本文时，有一个有希望的库，`fastapi-cache`，它使 FastAPI 中的缓存变得非常简单。请查看 GitHub 仓库：[`github.com/long2ice/fastapi-cache`](https://github.com/long2ice/fastapi-cache)。

该库支持多个缓存数据库，包括 Redis 和内存缓存。通过简单的端点装饰器，您可以指定缓存参数，如存活时间、编码器和缓存响应头。

参见

Redis 的 Python 客户端支持更多高级功能。您可以在官方文档中自由探索其潜力：

+   *Redis Python* *客户端*: [`redis.io/docs/connect/clients/python/`](https://redis.io/docs/connect/clients/python/)

+   *Redis Python 异步* *客户端*: [`redis-py.readthedocs.io/en/stable/examples/asyncio_examples.xhtml`](https://redis-py.readthedocs.io/en/stable/examples/asyncio_examples.xhtml)

```py

```
