# 第三章：3. 使用 SQLAlchemy 操作数据库

## 学习目标

在本章结束时，你将能够：

+   使用 pgAdmin 工具管理数据库

+   使用 SQLAlchemy 操作数据库

+   使用 Flask-Migrate 创建数据库表

+   将数据持久化到数据库中

+   对机密密码数据进行哈希处理

本章涵盖了使用 SQLAlchemy 访问数据库，包括构建模型、加密密码、确保每个电子邮件地址唯一，然后将食谱数据保存到数据库中。

## 简介

在上一章中，我们只是在应用程序内存中存储我们的数据。虽然这样做编码起来很容易，但一旦服务器重启，数据就会消失。这显然不是理想的，因为我们期望数据在服务器重启或应用程序迁移等情况下仍然持久化。因此，在本章中，我们将讨论在数据库中持久化数据。我们首先将在本地机器上安装 Postgres 数据库。然后，我们将使用 pgAdmin 创建数据库，并使用 SQLAlchemy 的 **ORM**（**对象关系映射**）包与之交互。ORM 允许我们通过对象而不是 SQL 查询与数据库交互。之后，我们将定义用户和食谱模型，将它们链接起来，并使用 Flask-Migrate 在数据库中创建相应的表。一旦这部分完成，我们将通过练习来理解 SQLAlchemy 在 Python 控制台中的使用。最后，我们将添加用户资源，以便可以通过 API 创建新用户。

## 数据库

你可能之前听说过数据库这个词。它基本上是一个数据存储系统。但为什么我们需要一个系统来存储数据？为什么我们不能只是将所有内容存储在文本文件中，然后保存在文件夹系统中？显然，数据库的功能不仅仅是存储数据。它对数据进行分类和组织，并帮助以更少的冗余存储数据。它还使数据更容易维护，使其更安全、更一致。数据库通常由一个**数据库管理系统**（**DBMS**）管理

### 数据库管理系统

DBMS 是一个操作和管理数据库的应用程序。它促进了用户与数据库之间的通信。用户可以使用此应用程序创建、使用和维护数据库。

DBMS 对于数据安全和完整性至关重要。流行的数据库软件和 DBMS 包括 PostgreSQL、MySQL、Microsoft SQL Server、MariaDB 和 Oracle 数据库。大多数 DBMS 使用**结构化查询语言**（**SQL**）来插入和提取数据。

在这本书中，我们将使用 PostgreSQL 作为我们的后端数据库系统。我们还将使用 pgAdmin，这是一个用于管理 PostgreSQL 的工具。PostgreSQL 是一个拥有 15 年历史的强大、开源的对象关系型数据库管理系统。由于其稳定性和数据完整性，它得到了广泛的认可。

## SQL

**SQL** 是一种专门为管理和操作数据而发明的语言。它可以进一步分为以下类型：

+   `SELECT column1, column2 FROM table WHERE conditions`，它可以查询表并提取满足一定条件的数据（`column1, column2`）。

+   `INSERT`、`UPDATE` 和 `DELETE`。

+   **数据控制语言**（**DCL**）用于控制数据访问。

虽然我们在这里介绍了许多不同的语言，但好事是我们不需要学习所有这些。事实上，我们不会使用 SQL 查询我们的数据库。我们只需要用 Python 编码，ORM 包将在幕后将我们的 Python 代码转换为 SQL。现在与数据库一起工作要容易得多。

## ORM

**对象关系映射**（**ORM**）是一种编程技术，允许开发者在编程语言中将对象映射到数据库中的数据模型。不再需要使用 SQL 与数据库交互。这种技术的优点是，开发者可以用自己的编程语言进行编码，并且它可以在不同类型的数据库上工作。

映射工作如下：

+   Python 中的类 = 数据库中的表架构

+   类中的属性 = 表架构中的字段

+   对象 = 表中的数据行

SQLAlchemy 是 Python 社区中最受欢迎的 ORM。接下来，让我们进一步深入，尝试创建一个数据库。

### 练习 18：设置 Smilecook 数据库

应用程序现在大多需要数据库来存储和管理数据。我们的应用程序 Smilecook 也不例外。它是一个食谱分享平台，对公众开放。显然，它将需要存储用户数据和食谱数据。在这个练习中，我们将创建数据库管理员并为我们的 Smilecook 应用程序设置数据库：

1.  首先，我们将创建一个角色。角色是 PostgreSQL 用于管理访问的一个简单概念。在这里，我们可以将其视为用户。在 **Servers** 下的 **PostgreSQL 11** 上**右键单击**，选择 **Create**，然后选择 **Login/Group Role…**：![图 3.1：选择登录/组角色…    ![图 3.4：使用创建的账户登录数据库](img/C15309_03_01.jpg)

    ###### 图 3.1：选择登录/组角色…

1.  填写登录名，稍后将用于连接到数据库：![图 3.2：填写登录名    ![图 3.4：使用创建的账户登录数据库](img/C15309_03_02.jpg)

    ###### 图 3.2：填写登录名

1.  然后，选择 **Definition** 并设置密码。点击 **Save**：![图 3.3：设置密码    ![图 3.4：使用创建的账户登录数据库](img/C15309_03_03.jpg)

    ###### 图 3.3：设置密码

1.  现在，转到 **Privileges**，并选择 **Yes** 以允许 **Can login**？这将允许我们使用此账户登录到数据库：![图 3.4：使用创建的账户登录数据库](img/C15309_03_04.jpg)

    ![图 3.4：使用创建的账户登录数据库](img/C15309_03_04.jpg)

    ###### 图 3.4：使用创建的账户登录数据库

1.  **右键单击** **Databases**，并从那里创建数据库：![图 3.5：创建数据库    ![图 3.4：使用创建的账户登录数据库](img/C15309_03_05.jpg)

    ###### 图 3.5：创建数据库

1.  将数据库命名为`smilecook`，并将我们刚刚创建的角色设置为**所有者**。点击**保存**：![图 3.6：命名数据库和设置角色    ](img/C15309_03_06.jpg)

###### 图 3.6：命名数据库和设置角色

现在我们已经创建了 Smilecook 数据库，但目前它是空的。在下一个练习中，我们将使用 Flask-SQLAlchemy 和 Flask-Migrate 来创建我们的数据库表。你会注意到其中没有涉及 SQL 查询。

## 定义我们的模型

在我们进入实现之前，我们需要首先定义和了解我们将要工作的字段。我们将涵盖两个基本模型：**User**和**Recipe**。模型在数据库中类似于模式。一个模型是一个类，可以被实例化。它包含与数据库模式中的字段相对应的属性。

**用户模型**

用户模型将被映射到数据库中的用户表。我们为我们的用户模型定义的字段和方法如下：

+   `id`：用户的标识。

+   `username`：用户的用户名。允许的最大长度是 80 个字符。它不能为空，并且是一个唯一字段。

+   `email`：用户的电子邮件。允许的最大长度是 200。它不能为空，并且是一个唯一字段。

+   `password`：用户的密码。允许的最大长度是 200。

+   `is_active`：这是用来表示账户是否通过电子邮件激活。它是一个布尔字段，默认值为`False`。

+   `recipes`：这不会在数据库表中创建一个字段。这只是为了定义与菜谱模型的关系。因此，我们可以使用`user.recipes`获取所有菜谱。

+   `created_at`：用户的创建时间。

+   `updated_at`：用户的最后更新时间。

我们还将在用户模型中定义三个方法：

+   `get_by_username`：这个方法用于通过用户名搜索用户。

+   `get_by_email`：这个方法用于通过电子邮件搜索用户。

+   `save`：这是为了将数据持久化到数据库中。

**菜谱模型**

菜谱模型将被映射到数据库中的用户表。我们为我们的菜谱模型定义的字段如下：

+   `id`：菜谱的标识。

+   `name`：菜谱的名称。允许的最大长度是 100 个字符。它不能为空。

+   `description`：菜谱的描述。允许的最大长度是 200。

+   `num_of_servings`：份量数量。这需要是一个整数。

+   `cook_time`：烹饪时间（分钟）。此字段只接受整数。

+   `directions`：菜谱的说明。这个字段的最大长度为 1,000。

+   `is_publish`：这是用来表示菜谱是否已发布。默认设置为`False`。

+   `created_at`：菜谱的创建时间。

+   `updated_at`：菜谱的最后更新时间。

在我们心中有了模型设计之后，我们现在准备在下一个练习中使用这些模型。在此之前，让我们也简要了解一下我们将要使用的一些关键包。这些包如下：

+   Flask-SQLAlchemy：这是一个非常流行的 ORM 软件包，它允许我们访问对象而不是数据库表来处理数据。使用 ORM，我们不再需要依赖 SQL。

+   Flask-Migrate：这是一个数据库迁移软件包；它基于 Alembic。

+   Psycopg2-binary：这是 Postgres 数据库的适配器。

+   Passlib：这是一个 Python 密码散列库。

### 练习 19：安装软件包和定义模型

本练习旨在安装必要的软件包并定义用户和食谱模型。用户和食谱模型将是 Python 类；本练习中不会涉及任何 SQL 编码。我们想通过简单的 Python 编码向您展示如何与数据库交互：

1.  我们将在`requirements.txt`文件中添加所需的软件包。如果您还记得，通过在`requirements.txt`中放置软件包名称和版本，我们可以通过单个`pip`命令在 Python 虚拟环境中安装它们：

    ```py
    Flask-SQLAlchemy==2.4.0
    Flask-Migrate==2.5.2
    psycopg2-binary==2.8.3
    passlib==1.7.1
    ```

1.  我们可以通过以下`pip install`命令安装必要的软件包：

    ```py
    pip install -r requirements.txt
    ```

    安装结果将在屏幕上显示：

    ```py
    Installing collected packages: SQLAlchemy, Flask-SQLAlchemy, alembic, Flask-Migrate, psycopg2-binary, passlib
      Running setup.py install for SQLAlchemy ... done
      Running setup.py install for alembic ... done
    Successfully installed Flask-Migrate-2.5.2 Flask-SQLAlchemy-2.4.0 SQLAlchemy-1.3.6 alembic-1.0.11 passlib-1.7.1 psycopg2-binary-2.8.3
    ```

1.  创建一个`Config.py`文件并输入以下代码：

    ```py
    class Config:
        DEBUG = True
        SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{your_name}:{your_password}@localhost/{db_name}
        SQLALCHEMY_TRACK_MODIFICATIONS = False
    ```

    我们可以在这里设置`DEBUG = True`以进行调试。至于`SQLALCHEMY_DATABASE_URI`，这是数据库的路径。请将用户名和密码替换为我们为`pgAdmin`创建的。同时，也请替换数据库名称。

1.  现在，在 Smilecook 项目下创建`extensions.py`并输入以下代码：

    ```py
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()
    ```

1.  在`models`文件夹下创建`user.py`并输入以下代码：

    ```py
    from extensions import db
    class User(db.Model):
        __tablename__ = 'user'
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), nullable=False, unique=True)
        email = db.Column(db.String(200), nullable=False, unique=True)
        password = db.Column(db.String(200))
        is_active = db.Column(db.Boolean(), default=False)
        created_at = db.Column(db.DateTime(), nullable=False, server_default=db.func.now())
        updated_at = db.Column(db.DateTime(), nullable=False, server_default=db.func.now(), onupdate=db.func.now())
        recipes = db.relationship('Recipe', backref='user')
        @classmethod
        def get_by_username(cls, username):
            return cls.query.filter_by(username=username).first()
        @classmethod
        def get_by_email(cls, email):
            return cls.query.filter_by(email=email).first()

        def save(self):
            db.session.add(self)
            db.session.commit()
    ```

1.  将`recipe.py`替换为以下代码。我们在这里添加了`import db`语句，并且也修改了`Recipe`类。与`recipe_list`相关的代码仍然有效，因此我们保留了这部分代码：

    ```py
    from extensions import db
    recipe_list = []
    def get_last_id():
        if recipe_list:
            last_recipe = recipe_list[-1]
        else:
            return 1
        return last_recipe.id + 1
    class Recipe(db.Model):
        __tablename__ = 'recipe'
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False)
        description = db.Column(db.String(200))
        num_of_servings = db.Column(db.Integer)
        cook_time = db.Column(db.Integer)
        directions = db.Column(db.String(1000))
        is_publish = db.Column(db.Boolean(), default=False)
        created_at = db.Column(db.DateTime(), nullable=False, server_default=db.func.now())
        updated_at = db.Column(db.DateTime(), nullable=False, server_default=db.func.now(), onupdate=db.func.now())
        user_id = db.Column(db.Integer(), db.ForeignKey("user.id"))
    ```

1.  现在，用以下代码重写`app.py`。我们以更合适的方式组织代码，使其更易于阅读和维护。首先，在代码文件的开头导入所需的软件包。

    #### 注意

    ```py
    from flask import Flask
    from flask_migrate import Migrate
    from flask_restful import Api
    from config import Config
    from extensions import db
    from models.user import User
    from resources.recipe import RecipeListResource, RecipeResource, RecipePublishResource
    ```

1.  使用`create_app()`函数创建 Flask 应用程序。这将调用`register_extensions(app)`以初始化 SQLAlchemy 并设置 Flask-Migrate。然后它将调用`register_resources(app)`来设置资源路由：

    ```py
    def create_app():
        app = Flask(__name__)
        app.config.from_object(Config)
        register_extensions(app)
        register_resources(app)
        return app
    def register_extensions(app):
        db.init_app(app)
        migrate = Migrate(app, db)
    def register_resources(app):
        api = Api(app)
        api.add_resource(RecipeListResource, '/recipes')
        api.add_resource(RecipeResource, '/recipes/<int:recipe_id>')
        api.add_resource(RecipePublishResource, '/recipes/<int:recipe_id>/publish')
    ```

1.  最后，使用`app = create_app()`创建 Flask 应用程序，并使用`app.run()`启动应用程序：

    ```py
    if __name__ == '__main__':
        app = create_app()
        app.run()
    ```

1.  保存`app.py`并在其上右键单击以运行应用程序。然后 Flask 将在本地主机（**127.0.0.1**）的 5000 端口启动并运行：

![图 3.7：在本地主机上启动 Flask]

![图片 C15309_03_07.jpg](img/C15309_03_07.jpg)

###### 图 3.7：在本地主机上启动 Flask

我们已成功安装必要的 ORM 相关软件包，并定义了用户和食谱模型。首先安装软件包后，我们在虚拟环境中运行了安装。我们创建了`config.py`、`extensions.py`和`user.py`文件，并替换了`app.py`。最后，我们对 Flask 应用程序进行了重构，并看到了它运行得有多好。

### 练习 20：使用 Flask-Migrate 构建数据库升级脚本

成功理解如何使用我们的两个主要模型，用户和食谱后，我们现在已经建立了完美的基础。下一步是执行。我们将使用 Flask-Migrate 来构建一个脚本来创建用户和食谱表：

1.  在终端中使用以下命令初始化我们的数据库。这将创建一个迁移存储库：

    ```py
    flask db init
    ```

    您应该在屏幕上看到以下内容：

    ```py
    Creating directory /Python-API-Development-Fundamentals/smilecook/migrations ... done
    Creating directory /Python-API-Development-Fundamentals/smilecook/migrations/versions ... done
    Generating /Python-API-Development-Fundamentals/smilecook/migrations/script.py.mako ... done
    Generating /Python-API-Development-Fundamentals/smilecook/migrations/env.py ... done
    Generating /Python-API-Development-Fundamentals/smilecook/migrations/README ... done
    Generating /Python-API-Development-Fundamentals/smilecook/migrations/alembic.ini ... done
    Please edit configuration/connection/logging settings in '/Python-API-Development-
    Fundamentals/smilecook/migrations/alembic.ini' before proceeding.
    ```

    您现在应该在 PyCharm 中看到以下新文件：

    ![图 3.8：PyCharm 中的新文件夹    ](img/C15309_03_08.jpg)

    ###### 图 3.8：PyCharm 中的新文件夹

1.  现在，运行 `flask db migrate` 命令来创建数据库和表。我们在这里不需要使用 SQL：

    ```py
    flask db migrate
    ```

    Flask-Migrate 检测到两个对象（`user` 和 `recipe`）并为它们创建了相应的两个表：

    ```py
    INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
    INFO  [alembic.runtime.migration] Will assume transactional DDL.
    INFO  [alembic.autogenerate.compare] Detected added table 'user'
    INFO  [alembic.autogenerate.compare] Detected added table 'recipe'
      Generating /Python-API-Development-Fundamentals/smilecook/migrations/versions/a6d248ab7b23_.py ... done
    ```

1.  现在，请检查 `versions` 文件夹下的 `/migrations/versions/a6d248ab7b23_.py`。此文件由 Flask-Migrate 创建。请注意，您在这里可能得到不同的修订 ID。请在运行 flask `db` upgrade 命令之前查看该文件。因为有时它可能无法检测到您对模型所做的每个更改：

    ```py
    """empty message
    Revision ID: a6d248ab7b23
    Revises: 
    Create Date: 2019-07-22 16:10:41.644737
    """
    from alembic import op
    import sqlalchemy as sa
    # revision identifiers, used by Alembic.
    revision = 'a6d248ab7b23'
    down_revision = None
    branch_labels = None
    depends_on = None
    def upgrade():
        # ### commands auto generated by Alembic - please adjust! ###
        op.create_table('user',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=80), nullable=False),
        sa.Column('email', sa.String(length=200), nullable=False),
        sa.Column('password', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
        )
        op.create_table('recipe',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('num_of_servings', sa.Integer(), nullable=True),
        sa.Column('cook_time', sa.Integer(), nullable=True),
        sa.Column('directions', sa.String(), nullable=True),
        sa.Column('is_publish', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        # ### end Alembic commands ###
    def downgrade():
        # ### commands auto generated by Alembic - please adjust! ###
        op.drop_table('recipe')
        op.drop_table('user')
        # ### end Alembic commands ###
    ```

    在此自动生成的文件中有两个函数；一个是升级的，用于将新的食谱和用户添加到表中，另一个是降级的，用于回到上一个版本。

1.  然后，我们将执行 `flask db upgrade` 命令，这将使我们的数据库符合模型中最新规范：

    ```py
    flask db upgrade
    ```

    此命令将调用 `upgrade()` 来升级数据库：

    ```py
    INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
    INFO  [alembic.runtime.migration] Will assume transactional DDL.
    INFO  [alembic.runtime.migration] Running upgrade  -> a6d248ab7b23, empty message
    ```

    #### 注意

    在未来，无论何时我们需要升级数据库，我们只需调用 `flask db migrate` 和 `flask db upgrade`。

1.  在 `pgAdmin` 中检查数据库表。现在，我们可以查看表是否已在数据库中创建。转到 **smilecook** >> **Schemas** >> **Tables to verify**：![图 3.9：检查数据库表    ](img/C15309_03_09.jpg)

###### 图 3.9：检查数据库表

如果您在我们的 **Smilecook** 数据库中看到食谱和用户表，这意味着您已成功在 Python 中创建它们，而没有使用任何 SQL。这不是很酷吗？!

接下来，我们将尝试数据库插入。让我们看看以下练习。

### 练习 21：应用数据库插入

这个练习是为了让我们测试数据库插入。我们首先创建一个用户，然后在该用户下创建两个食谱：

1.  在 Python 控制台中导入模块。在 PyCharm 底部打开 Python 控制台，并输入以下代码以导入必要的类：

    ```py
    from app import *
    from models.user import User
    from models.recipe import Recipe
    app = create_app()
    ```

1.  创建我们的第一个 `user` 对象，并将其保存到数据库中，请在 Python 控制台中输入以下代码：

    ```py
    user = User(username='jack', email='jack@gmail.com', password='WkQa')
    db.session.add(user)
    db.session.commit()
    ```

1.  现在，检查 `user` 的详细信息。请注意，用户的 ID 已经被分配为 `1`：

    ```py
    >>>user.username
    'jack'
    >>>user.id
    1
    >>>user.email
    'jack@gmail.com'
    ```

1.  由于用户已保存在数据库中，我们将验证那里：![图 3.10：在数据库中验证用户    ](img/C15309_03_10.jpg)

    ###### 图 3.10：在数据库中验证用户

1.  我们可以在 `user` 表中看到一条记录：

1.  ![图 3.11：用户表中的记录    ](img/C15309_03_11.jpg)

    ###### 图 3.11：用户表中的记录

1.  接下来，我们将使用以下代码创建两个菜谱。需要注意的是，菜谱的 `user_id` 属性被设置为 `user.id`。这是为了表明菜谱是由用户 `Jack` 创建的：

    ```py
    pizza = Recipe(name='Cheese Pizza', description='This is a lovely cheese pizza recipe', num_of_servings=2, cook_time=30, directions='This is how you make it', user_id=user.id)
    db.session.add(pizza)
    db.session.commit()
    pasta = Recipe(name='Tomato Pasta', description='This is a lovely tomato pasta recipe', num_of_servings=3, cook_time=20, directions='This is how you make it', user_id=user.id)
    db.session.add(pasta)
    db.session.commit()
    ```

1.  我们将检查是否在数据库中创建了两个菜谱：![图 3.12：检查两个菜谱是否已创建    ![img/C15309_03_12.jpg](img/C15309_03_12.jpg)

    ###### 图 3.12：检查两个菜谱是否已创建

1.  我们将在数据库中搜索用户名为 `jack` 的用户，并获取该用户创建的所有菜谱，这些菜谱存储在他们的对象属性 `recipes` 中：

    ```py
    >>> user = User.query.filter_by(username='jack').first()
    >>> user.recipes
    ```

    我们将获取两个菜谱的列表：

    ```py
    [<Recipe 1>, <Recipe 2>]
    ```

1.  我们可以使用 `for` 循环来显示菜谱的详细信息。我们通过 `recipe.name` 获取菜谱名称，而通过 `recipe.user.username` 获取用户的名称：

    ```py
    >>> for recipe in user.recipes:
        print('{} recipe made by {} can serve {} people.'.format(recipe.name, recipe.user.username, recipe.num_of_servings))
    ```

    你应该在屏幕上看到以下结果：

    ```py
    Cheese Pizza recipe made by jack can serve 2 people.
    Tomato Pasta recipe made by jack can serve 3 people.
    ```

你刚刚学会了如何使用 Python 控制台来命令你的应用程序。你已经创建了用户和菜谱模型，并将它们保存在数据库中。整个过程是无 SQL 的，正如你所看到的。让我们通过一个活动来巩固你的知识。

### 活动 5：创建用户和菜谱

在这个活动中，我们将通过运行一些额外的测试用例来测试我们的 API。我们想要创建一个新用户 `Peter`，并在数据库中为他创建两个菜谱。让我们看看你是否知道如何在 Python 交互控制台中编写这段代码：

1.  导入 `User` 和 `Recipe` 类，并使用 Python 控制台创建 Flask 应用程序。

1.  创建一个新用户，`Peter`。

1.  创建两个菜谱并将 `Peter` 设为作者。

    #### 注意

    本活动的解决方案可以在第 302 页找到。

如果你看到数据已成功创建在数据库中，恭喜你——你已经知道如何使用 Python 控制台与数据库进行交互！接下来，我们将实现用户注册功能。

## 密码哈希

哈希是一种单向数学函数。将明文字符串转换为哈希值（哈希）需要很少的计算能力。然而，要从哈希值中检索原始字符串则需要巨大的计算能力（几乎是不可能的）。因此，我们称它为单向函数：

![图 3.13：哈希函数的工作原理![img/C15309_03_13.jpg](img/C15309_03_13.jpg)

###### 图 3.13：哈希函数的工作原理

利用这一特性，哈希函数非常适合用于密码哈希。在我们将其保存到数据库之前，我们将用户的密码哈希化，使其不可识别且不可逆。下次用户登录时，平台所做的是将输入的密码转换为它的哈希值，然后将其与数据库中存储的哈希值进行比较。这样，我们就可以在不向他人泄露敏感密码信息的情况下执行密码比较。

### 练习 22：实现用户注册功能并哈希用户密码

在这个练习中，我们将处理用户注册功能。我们还将实现两个用于哈希用户密码的函数：

1.  在应用程序项目文件夹下创建`utils.py`，并输入以下代码。这段代码用于哈希密码。由于安全考虑，我们不希望在数据库中存储明文密码。因此，我们将使用`passlib`模块进行哈希。我们在这里定义了两种方法：

    ```py
    from passlib.hash import pbkdf2_sha256
    def hash_password(password):
        return pbkdf2_sha256.hash(password)
    def check_password(password, hashed):
        return pbkdf2_sha256.verify(password, hashed)
    ```

    `hash_password(password)`函数用于密码哈希，`check_password(password, hashed)`用于用户认证。它将用户输入的密码进行哈希处理，并将其与我们保存在数据库中的密码进行比较。

1.  在`resources`文件夹中创建`user.py`，然后输入以下代码。我们首先导入必要的模块并在`UserListResource`中实现`Post`方法：

    ```py
    from flask import request
    from flask_restful import Resource
    from http import HTTPStatus
    from utils import hash_password
    from models.user import User
    class UserListResource(Resource):
        def post(self):
            json_data = request.get_json()
            username = json_data.get('username')
            email = json_data.get('email')
            non_hash_password = json_data.get('password')
    ```

    当客户端请求以 HTTP `POST`方法击中`http://localhost/users`时，应用程序将获取请求中的 JSON 格式数据。应该有一个用户名、电子邮件和密码。

1.  通过`User.get_by_user(username)`检查用户是否已存在于数据库中。如果找到这样的条目，这意味着用户已经注册，我们将简单地返回一个错误消息。我们也会对`email`进行相同的检查：

    ```py
            if User.get_by_username(username):
                return {'message': 'username already used'}, HTTPStatus.BAD_REQUEST
            if User.get_by_email(email):
                return {'message': 'email already used'}, HTTPStatus.BAD_REQUEST
    ```

1.  一旦所有验证都通过，就可以在数据库中创建用户。密码将被哈希，然后创建用户对象。然后使用`user.save()`将用户对象保存到数据库中。最后，以 JSON 格式返回用户详细信息，并带有`HTTP`状态码`201`：

    ```py
        password = hash_password(non_hash_password)
            user = User(
                username=username,
                email=email,
                password=password
            )
            user.save()
            data = {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
            return data, HTTPStatus.CREATED
    ```

1.  将用户资源路由添加到`app.py`：

    ```py
    from extensions import db
    from resources.user import UserListResource
    from resources.recipe import RecipeListResource, RecipeResource, RecipePublishResource
    def register_resources(app):
        api = Api(app)
        api.add_resource(UserListResource, '/users')
        api.add_resource(RecipeListResource, '/recipes')
    ```

    将`app.py`中的`from models.user import User`替换为`from resources.user import UserListResource`。用户模型已经在`resources.user`中导入，因此不需要再次导入。请将`api.add_resource(UserListResource, '/users')`也添加到代码中。

    运行应用程序。Flask 将在本地主机（`127.0.0.1`）的`5000`端口上启动并运行：

    ![图 3.14：Flask 在本地主机上启动    ](img/C15309_03_07.jpg)

###### 图 3.14：Flask 在本地主机上启动

因此，我们刚刚完成了密码哈希练习。从现在起，每当有新用户在我们的 Smilecook 应用程序中注册时，他们的密码将被哈希并安全地存储在数据库中。让我们在下一个练习中测试一下这是否如此。

#### 注意

我们在这里不讨论食谱资源的原因是，食谱中会有一个作者 ID。作者 ID 将是一个外键，它将链接到用户模型。我们将在下一章中讨论用户登录功能。只有在那之后，我们才能获取用户 ID 并完成食谱资源。

### 练习 23：在 Postman 中测试应用程序

在这个练习中，我们将测试 Postman 中的应用程序。我们首先注册一个用户账户，并确保用户数据已存储在数据库中。我们还需要验证密码是否已哈希。创建用户后，现在让我们测试我们的 API 端点：

1.  在 Postman 中点击**集合**选项卡。

1.  创建一个新的集合，并将其命名为`User`。

1.  在该集合下创建一个新的请求，`UserList`。你可以通过点击`User`集合来完成此操作。

1.  编辑`POST`。

1.  在 URL 字段中输入`http://localhost:5000/users`。

1.  转到**主体**选项卡，将数据类型选择为**raw**，然后选择数据格式为**JSON (application/json)**。

1.  插入以下用户详细信息并保存：

    ```py
    {
        "username": "jack",
        "email": "jack@gmail.com",
        "password": "WkQa"
    }
    ```

1.  点击**发送**。结果如下所示截图：![图 3.15：使用现有用户名创建用户    ![图片](img/C15309_03_15.jpg)

    ###### 图 3.15：使用现有用户名创建用户

    你将看到以下返回的数据；HTTP 状态为**400 BAD REQUEST**。我们还可以在**主体**字段中看到错误消息，显示用户名已被注册。

1.  使用以下代码创建另一个具有以下详细信息的用户：

    ```py
    {
        "username": "ray",
        "email": "ray@gmail.com",
        "password": "WkQa"
    }
    ```

    结果如下所示截图：

    ![图 3.16：创建另一个用户    ![图片](img/C15309_03_16.jpg)

    ###### 图 3.16：创建另一个用户

    现在，第二个账户已成功创建。

1.  按如下方式检查数据库中的数据：![图 3.17：检查数据库中的数据    ![图片](img/C15309_03_17.jpg)

###### 图 3.17：检查数据库中的数据

现在，我们可以看到数据库表中创建了一个新的用户记录。并且你可以看到密码已被哈希。

通过进行此测试练习，我们可以确保我们的用户注册工作流程运行良好。最重要的是，用户密码以哈希值的形式保存在数据库中。这是一种更安全的存储密码的方式，即使数据库管理员也无法看到它。

### 活动 6：升级和降级数据库

1.  在此活动中，我们将升级和降级数据库以模拟我们需要在`user`类下添加属性的场景，但后来我们改变了主意，需要删除它。以下是我们完成此活动所需执行的高级步骤：

1.  向`user`类添加一个新属性。此属性应命名为`bio`，它将是一个表示用户信息的字符串。

1.  运行`flask db migrate`命令以创建数据库和表。

1.  现在，检查`versions`文件夹下的`/migrations/versions/6971bd62ec60_.py`。此文件由 Flask-Migrate 创建。

1.  执行`flask db upgrade`命令以将我们的数据库升级到符合我们模型中最新的规范。

1.  检查新字段是否已创建在数据库中。

1.  运行`downgrade`命令以删除新字段。

1.  检查字段是否已被删除。

    #### 注意

    此活动的解决方案可在第 303 页找到。

如果你看到新字段已被删除，这意味着你已经在 Python 中成功降级了数据库，而没有编写任何 SQL。别忘了删除 models/user.py 中用户模型的 bio 属性，也删除我们在 migrations/versions 文件夹中创建的脚本`a6d248ab7b23.py`。你刚刚学会了一个非常有用的技能，你将来可能会经常用到。给你一个提示，你应该在数据库模式更新之前备份你的数据库。这是为了确保数据不会丢失。

## 摘要

在本章中，我们在本地构建了 Postgres 数据库，并学习了如何使用 pgAdmin 工具来管理它。然后，通过 SQLAlchemy 模块，我们开发了一个对象库来操作数据库。这比直接使用 SQL 语法要容易得多。而且，只要我们定义了模型之间的关系，我们就可以轻松地获取我们想要的信息。这导致代码可读性更高，代码行数更少，并消除了重复的 SQL。然后，我们使用 Flask-Migrate 构建所有数据表。然后，当我们将来迁移数据库时，我们只需要两个命令——`flask db migrate`和`flask db upgrade`；这很简单，也很容易。尽管 Flask-Migrate 可以帮助我们更轻松地设置和迁移数据库，但在生产环境中，执行此类迁移仍然需要额外的谨慎。我们应该始终备份数据库以保护我们宝贵的数据。

在开发过程中，我们应该经常测试我们的代码以确保其按预期行为。我们不应该等到最后才进行大范围测试。一旦完成，我们可以对函数和 API 端点进行单元测试。使用 Python 控制台进行此类简单测试是推荐的。迭代测试我们的应用程序也可以培养最佳编程实践。这迫使我们思考如何以优雅的方式组织代码并避免技术债务的积累。

最后，我们为用户注册创建了一个 API。在下一章中，我们将致力于为认证用户开发用户登录和食谱创建功能。
