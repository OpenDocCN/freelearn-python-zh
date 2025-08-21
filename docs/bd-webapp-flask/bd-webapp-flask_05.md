# 第五章：你把东西放在哪里？

我就像一只松鼠。我偶尔会在家里的秘密藏匿处留下一些钱，以防我被抢劫，或者在一个月里花费太多。我真的忘记了我所有的藏匿处在哪里，这有点有趣也有点悲哀（对我来说）。

现在，想象一下，你正在存储一些同样重要甚至更重要的东西，比如客户数据或者甚至你公司的数据。你能允许自己将它存储在以后可能会丢失或者可以被某人干扰的地方吗？我们正处于信息时代；信息就是力量！

在网络应用程序世界中，我们有两个大的数据存储玩家：**关系数据库**和**NoSQL 数据库**。第一种是传统的方式，其中您的数据存储在表和列中，事务很重要，期望有 ACID，规范化是关键（双关语）！它使用**SQL**来存储和检索数据。在第二种方式中，情况变得有点疯狂。您的数据可能存储在不同的结构中，如文档、图形、键值映射等。写入和查询语言是特定于供应商的，您可能不得不放弃 ACID 以换取速度，大量的速度！

你可能已经猜到了！这一章是关于**MVC**中的**M**层，也就是如何以透明的方式存储和访问数据的章节！我们将看一下如何使用查询和写入两种数据库类型的示例，以及何时选择使用哪种。

### 提示

ACID 是原子性、一致性、隔离性和持久性的缩写。请参考[`en.wikipedia.org/wiki/ACID`](http://en.wikipedia.org/wiki/ACID)了解一个舒适的定义和概述。

# SQLAlchemy

SQLAlchemy 是一个与关系数据库一起工作的惊人库。它是由 Pocoo 团队制作的，他们也是 Flask 的创始人，被认为是“事实上”的 Python SQL 库。它可以与 SQLite、Postgres、MySQL、Oracle 和所有 SQL 数据库一起使用，这些数据库都有兼容的驱动程序。

SQLite 自称为一个自包含、无服务器、零配置和事务性 SQL 数据库引擎（[`sqlite.org/about.html`](https://sqlite.org/about.html)）。其主要目标之一是成为应用程序和小型设备的嵌入式数据库解决方案，它已经做到了！它也非常容易使用，这使得它非常适合我们的学习目的。

尽管所有的例子都将以 SQLite 为主要考虑对象进行给出和测试，但它们应该在其他数据库中也能够以很少或没有改动的方式工作。在适当的时候，将会不时地给出特定于数据库的提示。

### 注意

请参考[`www.w3schools.com/sql/default.asp`](http://www.w3schools.com/sql/default.asp)了解广泛的 SQL 参考。

在我们的第一个例子之前，我们是否应该复习一下几个关系数据库的概念？

## 概念

**表**是低级抽象结构，用于存储数据。它由**列**和**行**组成，其中每一列代表数据的一部分，每一行代表一个完整的记录。通常，每个表代表一个类模型的低级抽象。

**行**是给定类模型的单个记录。您可能需要将多个行记录分散到不同的表中，以记录完整的信息。一个很好的例子是**MxN 关系**。

**列**代表存储的数据本身。每一列都有一个特定的类型，并且只接受该类型的输入数据。您可以将其视为类模型属性的抽象。

**事务**是用来将要执行的操作分组的方式。它主要用于实现原子性。这样，没有操作是半途而废的。

**主键**是一个数据库概念，记录的一部分数据用于标识数据库表中的给定记录。通常由数据库通过约束来实现。

**外键**是一个数据库概念，用于在不同表之间标识给定记录的一组数据。它的主要用途是在不同表的行之间构建关系。通常由数据库通过约束来实现。

在使用关系数据库时的一个主要关注点是数据规范化。在关系数据库中，相关数据存储在不同的表中。您可能有一个表来保存一个人的数据，一个表来保存这个人的地址，另一个表来保存他/她的汽车，等等。

每个表都与其他表隔离，通过外键建立的关系可以检索相关数据！数据规范化技术是一组规则，用于允许数据在表之间适当分散，以便轻松获取相关表，并将冗余保持最小。

### 提示

请参考[`en.wikipedia.org/wiki/Database_normalization`](http://en.wikipedia.org/wiki/Database_normalization)了解数据库规范化的概述。

有关规范形式的概述，请参阅以下链接：

[`en.wikipedia.org/wiki/First_normal_form`](http://en.wikipedia.org/wiki/First_normal_form)

[`en.wikipedia.org/wiki/Second_normal_form`](http://en.wikipedia.org/wiki/Second_normal_form)

[`en.wikipedia.org/wiki/Third_normal_form`](http://en.wikipedia.org/wiki/Third_normal_form)

我们现在可以继续了！

## 实际操作

让我们开始将库安装到我们的环境中，并尝试一些示例：

```py
pip install sqlalchemy

```

我们的第一个示例！让我们为一家公司（也许是你的公司？）创建一个简单的员工数据库：

```py
from sqlalchemy import create_engine
db = create_engine('sqlite:///employees.sqlite')
# echo output to console
db.echo = True

conn = db.connect()

conn.execute("""
CREATE TABLE employee (
  id          INTEGER PRIMARY KEY,
  name        STRING(100) NOT NULL,
  birthday    DATE NOT NULL
)""")

conn.execute("INSERT INTO employee VALUES (NULL, 'marcos mango', date('1990-09-06') );")
conn.execute("INSERT INTO employee VALUES (NULL, 'rosie rinn', date('1980-09-06') );")
conn.execute("INSERT INTO employee VALUES (NULL, 'mannie moon', date('1970-07-06') );")
for row in conn.execute("SELECT * FROM employee"):
    print row
# give connection back to the connection pool
conn.close()
```

前面的例子非常简单。我们创建了一个 SQLAlchemy 引擎，从**连接池**中获取连接（引擎会为您处理），然后执行 SQL 命令来创建表，插入几行数据并查询是否一切都如预期发生。

### 提示

访问[`en.wikipedia.org/wiki/Connection_pool`](http://en.wikipedia.org/wiki/Connection_pool)了解连接池模式概述。（这很重要！）

在我们的插入中，我们为主键`id`提供了值`NULL`。请注意，SQLite 不会使用`NULL`填充主键；相反，它会忽略`NULL`值，并将列设置为新的、唯一的整数。这是 SQLite 特有的行为。例如，**Oracle**将要求您显式插入序列的下一个值，以便为主键设置一个新的唯一列值。

我们之前的示例使用了一个名为**autocommit**的功能。这意味着每次执行方法调用都会立即提交到数据库。这样，您无法一次执行多个语句，这在现实世界的应用程序中是常见的情况。

要一次执行多个语句，我们应该使用**事务**。我们可以通过事务重写我们之前的示例，以确保所有三个插入要么一起提交，要么根本不提交（严肃的表情...）。

```py
# we start our transaction here
# all actions now are executed within the transaction context
trans = conn.begin()

try:
    # we are using a slightly different insertion syntax for convenience, here; 
    # id value is not explicitly provided
    conn.execute("INSERT INTO employee (name, birthday) VALUES ('marcos mango', date('1990-09-06') );")
    conn.execute("INSERT INTO employee (name, birthday) VALUES ('rosie rinn', date('1980-09-06') );")
    conn.execute("INSERT INTO employee (name, birthday) VALUES ('mannie moon', date('1970-07-06') );")
    # commit all
    trans.commit()
except:
    # all or nothing. Undo what was executed within the transaction
    trans.rollback()
    raise
```

到目前为止还没有什么花哨的。在我们的例子中，我们从连接创建了一个事务，执行了一些语句，然后提交以完成事务。如果在事务开始和结束之间发生错误，`except`块将被执行，并且在事务中执行的所有语句将被回滚或“撤消”。

我们可以通过在表之间创建关系来完善我们的示例。想象一下，我们的员工在公司档案中注册了一个或多个地址。我们将创建一个 1xN 关系，其中一个员工可以拥有一个或多个地址。

```py
# coding:utf-8
from sqlalchemy import create_engine

engine = create_engine('sqlite:///employees.sqlite')
engine.echo = True

conn = engine.connect()

conn.execute("""
CREATE TABLE employee (
  id          INTEGER PRIMARY KEY,
  name        STRING(100) NOT NULL,
  birthday    DATE NOT NULL
)""")

conn.execute("""
CREATE TABLE address(
  id      INTEGER PRIMARY KEY,
  street  STRING(100) NOT NULL,
  number  INTEGER,
  google_maps STRING(255),
  id_employee INTEGER NOT NULL,
  FOREIGN KEY(id_employee) REFERENCES employee(id)
)""")

trans = conn.begin()
try:
    conn.execute("INSERT INTO employee (name, birthday) VALUES ('marcos mango', date('1990-09-06') );")
    conn.execute("INSERT INTO employee (name, birthday) VALUES ('rosie rinn', date('1980-09-06') );")
    conn.execute("INSERT INTO employee (name, birthday) VALUES ('mannie moon', date('1970-07-06') );")
    # insert addresses for each employee
    conn.execute(
        "INSERT INTO address (street, number, google_maps, id_employee) "
        "VALUES ('Oak', 399, '', 1)")
    conn.execute(
        "INSERT INTO address (street, number, google_maps, id_employee) "
        "VALUES ('First Boulevard', 1070, '', 1)")
    conn.execute(
        "INSERT INTO address (street, number, google_maps, id_employee) "
        "VALUES ('Cleveland, OH', 10, 'Cleveland,+OH,+USA/@41.4949426,-81.70586,11z', 2)")
    trans.commit()
except:
    trans.rollback()
    raise

# get marcos mango addresses
for row in conn.execute("""
  SELECT a.street, a.number FROM employee e
  LEFT OUTER JOIN address a
  ON e.id = a.id_employee
  WHERE e.name like '%marcos%';
  """):
    print "address:", row
conn.close()
```

在我们新的和更新的示例中，我们记录了一些员工的地址，确保使用正确的外键值（`id_employee`），然后我们使用`LEFT JOIN`查找名为`'marcos mango'`的员工的地址。

我们已经看到了如何创建表和关系，运行语句来查询和插入数据，并使用 SQLAlchemy 进行事务处理；我们还没有完全探索 SQLAlchemy 库的强大功能。

SQLAlchemy 具有内置的 ORM，允许您像使用本机对象实例一样使用数据库表。想象一下，读取列值就像读取实例属性一样，或者通过方法查询复杂的表关系，这就是 SQLAlchemy 的 ORM。

让我们看看使用内置 ORM 的示例会是什么样子：

```py
# coding:utf-8

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.ext.declarative import declarative_base

from datetime import datetime

engine = create_engine('sqlite:///employees.sqlite')
engine.echo = True

# base class for our models
Base = declarative_base()

# we create a session binded to our engine
Session = sessionmaker(bind=engine)

# and then the session itself
session = Session()

# our first model
class Address(Base):
    # the table name we want in the database
    __tablename__ = 'address'

    # our primary key
    id = Column(Integer, primary_key=True)
    street = Column(String(100))
    number = Column(Integer)
    google_maps = Column(String(255))
    # our foreign key to employee
    id_employee = Column(Integer, ForeignKey('employee.id'))

    def __repr__(self):
         return u"%s, %d" % (self.street, self.number)

class Employee(Base):
    __tablename__ = 'employee'

    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    birthday = Column(Date)
    # we map 
    addresses = relationship("Address", backref="employee")

    def __repr__(self):
         return self.name

# create our database from our classes
Base.metadata.create_all(engine)

# execute everything inside a transaction
session.add_all([
        Employee(name='marcos mango', birthday=datetime.strptime('1990-09-06', '%Y-%m-%d')), 
        Employee(name='rosie rinn', birthday=datetime.strptime('1980-09-06', '%Y-%m-%d')),
        Employee(name='mannie moon', birthday=datetime.strptime('1970-07-06', '%Y-%m-%d'))
    ])
session.commit()

session.add_all([
    Address(street='Oak', number=399, google_maps='', id_employee=1),
    Address(street='First Boulevard', number=1070, google_maps='', id_employee=1),
    Address(street='Cleveland, OH', number=10, 
             google_maps='Cleveland,+OH,+USA/@41.4949426,-81.70586,11z', id_employee=2)
])
session.commit()

# get marcos, then his addresses
marcos = session.query(Employee).filter(Employee.name.like(r"%marcos%")).first()
for address in marcos.addresses:
    print 'Address:', address
```

前面的示例介绍了相当多的概念。首先，我们创建了我们的引擎，即第一个示例中使用的 SQLAlchemy 引擎，然后我们创建了我们的基本模型类。虽然`Employee`将被`create_all`映射到一个名为`employee`的表中，但每个定义的`Column`属性都将被映射到数据库中给定表的列中，并具有适当的约束。例如，对于`id`字段，它被定义为主键，因此将为其创建主键约束。`id_employee`是一个外键，它是对另一个表的主键的引用，因此它将具有外键约束，依此类推。

我们所有的类模型都应该从中继承。然后我们创建一个`session`。会话是您使用 SQLAlchemy ORM 模型的方式。

会话具有内部正在进行的事务，因此它非常容易具有类似事务的行为。它还将您的模型映射到正确的引擎，以防您使用多个引擎；但等等，还有更多！它还跟踪从中加载的所有模型实例。例如，如果您将模型实例添加到其中，然后修改该实例，会话足够聪明，能够意识到其对象的更改。因此，它会将自身标记为脏（内容已更改），直到调用提交或回滚。

在示例中，在找到 marcos 之后，我们可以将"Marcos Mango's"的名字更改为其他内容，比如`"marcos tangerine"`，就像这样：

```py
marcos.name = "marcos tangerine"
session.commit()
```

现在，在`Base.metadata`之后注释掉整个代码，并添加以下内容：

```py
marcos = session.query(Employee).filter(Employee.name.like(r"%marcos%")).first()
marcos_last_name = marcos.name.split(' ')[-1]
print marcos_last_name
```

现在，重新执行示例。Marcos 的新姓氏现在是"tangerine"。神奇！

### 提示

有关使用 SQLAlchemy ORM 进行查询的惊人、超级、强大的参考，请访问[`docs.sqlalchemy.org/en/rel_0_9/orm/tutorial.html#querying`](http://docs.sqlalchemy.org/en/rel_0_9/orm/tutorial.html#querying)。

在谈论了这么多关于 SQLAlchemy 之后，您能否请醒来，因为我们将谈论 Flask-SQLAlchemy，这个扩展将库与 Flask 集成在一起。

## Flask-SQLAlchemy

Flask-SQLAlchemy 是一个轻量级的扩展，它将 SQLAlchemy 封装在 Flask 周围。它允许您通过配置文件配置 SQLAlchemy 引擎，并为每个请求绑定一个会话，为您提供了一种透明的处理事务的方式。让我们看看如何做到这一点。首先，确保我们已经安装了所有必要的软件包。加载虚拟环境后，运行：

```py
pip install flask-wtf flask-sqlalchemy

```

我们的代码应该是这样的：

```py
# coding:utf-8
from flask import Flask, render_template, redirect, flash
from flask_wtf import Form
from flask.ext.sqlalchemy import SQLAlchemy

from wtforms.ext.sqlalchemy.orm import model_form

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/employees.sqlite'
app.config['SQLALCHEMY_ECHO'] = True

# initiate the extension
db = SQLAlchemy(app)

# define our model
class Employee(db.Model):
    __tablename__ = 'employee'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    birthday = db.Column(db.Date, nullable=False)

    def __repr__(self):
        return 'employee %s' % self.name

# create the database
db.create_all()

# auto-generate form for our model
EmployeeForm = model_form(Employee, base_class=Form, field_args={
    'name': {
    'class': 'employee'
  }
})

@app.route("/", methods=['GET', 'POST'])
def index():
    # as you remember, request.POST is implicitly provided as argument
    form = EmployeeForm()

    try:
        if form.validate_on_submit():
            employee = Employee()
            form.populate_obj(employee)
            db.session.add(employee)
            db.session.commit()
            flash('New employee add to database')
            return redirect('/')
    except Exception, e:
        # log e
        db.session.rollback()
        flash('An error occurred accessing the database. Please, contact administration.')

    employee_list=Employee.query.all()
    return render_template('index.html', form=form, employee_list=employee_list)

if __name__ == '__main__':
    app.debug = True
    app.run()
```

前面的示例非常完整。它具有表单验证、CSRF 保护、从模型自动生成的表单以及数据库集成。让我们只关注到目前为止我们还没有提到的内容。

自动生成表单非常方便。使用`model_form`，您可以自省定义的模型类并生成适合该模型的表单类。您还可以通过`model_form`参数`field_args`为字段提供参数，这对于添加元素类或额外验证器非常有用。

您可能还注意到`Employee`扩展了`db.Model`，这是您的 ORM 模型基类。所有您的模型都应该扩展它，以便被`db`所知，它封装了我们的引擎并保存我们的请求感知会话。

在 index 函数内部，我们实例化表单，然后检查它是否通过 POST 提交并且有效。在`if`块内部，我们实例化我们的员工模型，并使用`populate_obj`将表单的值放入模型实例中。我们也可以逐个字段地进行操作，就像这样：

```py
employee.name = form.name.data
employee. birthday = form.birthday.data
```

`populate_obj`只是更方便。在填充模型后，我们将其添加到会话中以跟踪它，并提交会话。在此块中发生任何异常时，我们将其放在一个带有准备回滚的 try/except 块中。

请注意，我们使用`Employee.query`来查询存储在我们数据库中的员工。每个模型类都带有一个`query`属性，允许您从数据库中获取和过滤结果。对`query`的每个过滤调用将返回一个`BaseQuery`实例，允许您堆叠过滤器，就像这样：

```py
queryset = Employee.query.filter_by(name='marcos mango')
queryset = queryset.filter_by(birthday=datetime.strptime('1990-09-06', '%Y-%m-%d'))
queryset.all()  # <= returns the result of both filters applied together
```

这里有很多可能性。为什么不现在就尝试一些例子呢？

### 注意

与 Web 应用程序和数据库相关的最常见的安全问题是**SQL 注入攻击**，攻击者将 SQL 指令注入到您的数据库查询中，获取他/她不应该拥有的权限。SQLAlchemy 的引擎对象“自动”转义您的查询中的特殊字符；因此，除非您明确绕过其引用机制，否则您应该是安全的。

# MongoDB

MongoDB 是一个广泛使用的强大的 NoSQL 数据库。它允许您将数据存储在文档中；一个可变的、类似字典的、类似对象的结构，您可以在其中存储数据，而无需担心诸如“我的数据是否规范化到第三范式？”或“我是否必须创建另一个表来存储我的关系？”等问题。

MongoDB 文档实际上是 BSON 文档，是 JSON 的超集，支持扩展的数据类型。如果您知道如何处理 JSON 文档，您应该不会有问题。

### 提示

如果 JSON 对您毫无意义，只需查看[`www.w3schools.com/json/`](http://www.w3schools.com/json/)。

让我们在本地安装 MongoDB，以便尝试一些例子：

```py
sudo apt-get install mongodb

```

现在，从控制台输入：

```py
mongo

```

您将进入 MongoDB 交互式控制台。从中，您可以执行命令，向数据库添加文档，查询、更新或删除。您可以通过控制台实现的任何语法，也可以通过控制台实现。现在，让我们了解两个重要的 MongoDB 概念：数据库和集合。

在 MongoDB 中，您的文档被分组在集合内，而集合被分组在数据库内。因此，在连接到 MongoDB 后，您应该做的第一件事是选择要使用的数据库。您不需要创建数据库，连接到它就足以创建数据库。对于集合也是一样。您也不需要在使用文档之前定义其结构，也不需要实现复杂的更改命令，如果您决定文档结构应该更改。这里有一个例子：

```py
> use example
switched to db example
> db.employees.insert({name: 'marcos mango', birthday: new Date('Sep 06, 1990')})
WriteResult({ "nInserted" : 1 })
> db.employees.find({'name': {$regex: /marcos/}})
```

在上述代码中，我们切换到示例数据库，然后将一个新文档插入到员工集合中（我们不需要在使用之前创建它），最后，我们使用正则表达式搜索它。MongoDB 控制台实际上是一个 JavaScript 控制台，因此新的`Date`实际上是 JavaScript 类`Date`的实例化。非常简单。

### 提示

如果您不熟悉 JavaScript，请访问[`www.w3schools.com/js/default.asp`](http://www.w3schools.com/js/default.asp)了解一个很好的概述。

我们可以存储任何 JSON 类型的文档，还有其他一些类型。访问[`docs.mongodb.org/manual/reference/bson-types/`](http://docs.mongodb.org/manual/reference/bson-types/)获取完整列表。

关于正确使用 MongoDB，只需记住几个黄金规则：

+   避免将数据从一个集合保留到另一个集合，因为 MongoDB 不喜欢*连接*

+   在 MongoDB 中，将文档值作为列表是可以的，甚至是预期的

+   在 MongoDB 中，适当的文档索引（本书未涉及）对性能至关重要

+   写入比读取慢得多，可能会影响整体性能

## MongoEngine

MongoEngine 是一个非常棒的 Python 库，用于访问和操作 MongoDB 文档，并使用**PyMongo**，MongoDB 推荐的 Python 库。

### 提示

由于 PyMongo 没有**文档对象映射器**（**DOM**），我们不直接使用它。尽管如此，有些情况下 MongoEngine API 将不够用，您需要使用 PyMongo 来实现您的目标。

它有自己的咨询 API 和文档到类映射器，允许您以与使用 SQLAlchemy ORM 类似的方式处理文档。这是一个好事，因为 MongoDB 是无模式的。它不像关系数据库那样强制执行模式。这样，您在使用之前不必声明文档应该是什么样子。MongoDB 根本不在乎！

在实际的日常开发中，确切地知道您应该在文档中存储什么样的信息是一个很好的反疯狂功能，MongoEngine 可以直接为您提供。

由于您的机器上已经安装了 MongoDB，只需安装 MongoEngine 库即可开始使用它编码：

```py
pip install mongoengine pymongo==2.8

```

让我们使用我们的新库将“Rosie Rinn”添加到数据库中：

```py
# coding:utf-8

from mongoengine import *
from datetime import datetime

# as the mongo daemon, mongod, is running locally, we just need the database name to connect
connect('example')

class Employee(Document):
    name = StringField()
    birthday = DateTimeField()

    def __unicode__(self):
        return u'employee %s' % self.name

employee = Employee()
employee.name = 'rosie rinn'
employee.birthday = datetime.strptime('1980-09-06', '%Y-%m-%d')
employee.save()

for e in Employee.objects(name__contains='rosie'):
    print e
```

理解我们的示例：首先，我们使用`example`数据库创建了一个 MongoDB 连接，然后像使用 SQLAlchemy 一样定义了我们的员工文档，最后，我们插入了我们的员工“Rosie”并查询是否一切正常。

在声明我们的`Employee`类时，您可能已经注意到我们必须使用适当的字段类型定义每个字段。如果 MongoDB 是无模式的，为什么会这样？MongoEngine 强制执行每个模型字段的类型。如果您为模型定义了`IntField`并为其提供了字符串值，MongoEngine 将引发验证错误，因为那不是适当的字段值。此外，我们为`Employee`定义了一个`__unicode__`方法，以便在循环中打印员工的姓名。`__repr__`在这里不起作用。

由于 MongoDB 不支持事务（MongoDB 不是 ACID，记住？），MongoEngine 也不支持，我们进行的每个操作都是原子的。当我们创建我们的“Rosie”并调用`save`方法时，“Rosie”立即插入数据库；不需要提交更改或其他任何操作。

最后，我们有数据库查询，我们搜索“Rosie”。要查询所选集合，应使用每个 MongoEngine 文档中可用的`objects`处理程序。它提供了类似 Django 的界面，支持操作，如`contains`，`icontains`，`ne`，`lte`等。有关查询运算符的完整列表，请访问[`mongoengine-odm.readthedocs.org/guide/querying.html#query-operators`](https://mongoengine-odm.readthedocs.org/guide/querying.html#query-operators)。

## Flask-MongoEngine

MongoEngine 本身非常容易，但有人认为事情可以变得更好，于是我们有了 Flask-MongoEngine。它为您提供了三个主要功能：

+   Flask-DebugToolbar 集成（嘿嘿！）

+   类似 Django 的查询集（`get_or_404`，`first_or_404`，`paginate`，`paginate_field`）

+   连接管理

Flask-DebugToolbar 是一个漂亮的 Flask 扩展，受到 Django-DebugToolbar 扩展的启发，它跟踪应用程序在幕后发生的事情，例如请求中使用的 HTTP 标头，CPU 时间，活动 MongoDB 连接的数量等。

类似 Django 的查询是一个很有用的功能，因为它们可以帮助你避免一些无聊的编码。`get_or_404(*args, **kwargs)`查询方法会在未找到要查找的文档时引发 404 HTTP 页面（它在内部使用`get`）。如果你正在构建一个博客，你可能会喜欢在加载特定的文章条目时使用这个小家伙。`first_or_404()`查询方法类似，但适用于集合。如果集合为空，它会引发 404 HTTP 页面。`paginate(page, per_page)`查询实际上是一个非常有用的查询方法。它为你提供了一个开箱即用的分页界面。它在处理大型集合时效果不佳，因为在这些情况下 MongoDB 需要不同的策略，但大多数情况下，它就是你所需要的。`paginate_field(field_name, doc_id, page, per_page)`是 paginate 的更具体版本，因为你将对单个文档字段进行分页，而不是对集合进行分页。当你有一个文档，其中一个字段是一个巨大的列表时，它非常有用。

现在，让我们看一个完整的`flask-mongoengine`示例。首先，在我们的虚拟环境中安装这个库：

```py
pip install flask-mongoengine

```

现在开始编码：

```py
# coding:utf-8

from flask import Flask, flash, redirect, render_template
from flask.ext.mongoengine import MongoEngine
from flask.ext.mongoengine.wtf import model_form
from flask_wtf import Form

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['MONGODB_SETTINGS'] = {
    # 'replicaset': '',
    'db': 'example',
    # 'host': '',
    # 'username': '',
    # 'password': ''
}
db = MongoEngine(app)

class Employee(db.Document):
    name = db.StringField()
    # mongoengine does not support datefield
    birthday = db.DateTimeField()

    def __unicode__(self):
        return u'employee %s' % self.name

# auto-generate form for our model
EmployeeForm = model_form(Employee, base_class=Form, field_args={
    'birthday': {
        # we want to use date format, not datetime
        'format': '%Y-%m-%d'
    }
})

@app.route("/", methods=['GET', 'POST'])
def index():
    # as you remember, request.POST is implicitly provided as argument
    form = EmployeeForm()

    try:
        if form.validate_on_submit():
            employee = Employee()
            form.populate_obj(employee)
            employee.save()
            flash('New employee add to database')
            return redirect('/')
    except:
        # log e
        flash('An error occurred accessing the database. Please, contact administration.')

    employee_list=Employee.objects()
    return render_template('index.html', form=form, employee_list=employee_list)

if __name__ == '__main__':
    app.debug = True
    app.run()
```

我们的 Flask-MongoEngine 示例与 Flask-SQLAlchemy 示例非常相似。除了导入的差异之外，还有 MongoDB 的配置，因为 MongoDB 需要不同的参数；我们有`birthday`字段类型，因为 MongoEngine 不支持`DateField`；有生日格式的覆盖，因为`datetimefield`的默认字符串格式与我们想要的不同；还有`index`方法的更改。

由于我们不需要使用 Flask-MongoEngine 处理会话，我们只需删除所有与它相关的引用。我们还改变了`employee_list`的构建方式。

### 提示

由于 MongoDB 不会解析你发送给它的数据以尝试弄清楚查询的内容，所以你不会遇到 SQL 注入的问题。

# 关系型与 NoSQL

你可能会想知道何时使用关系型数据库，何时使用 NoSQL。嗯，鉴于今天存在的技术和技术，我建议你选择你感觉更适合的类型来工作。NoSQL 吹嘘自己是无模式、可扩展、快速等，但关系型数据库对于大多数需求也是相当快速的。一些关系型数据库，比如 Postgres，甚至支持文档。那么扩展呢？嗯，大多数项目不需要扩展，因为它们永远不会变得足够大。其他一些项目，只需与它们的关系型数据库一起扩展。

如果没有*重要*的原因来选择原生无模式支持或完整的 ACID 支持，它们中的任何一个都足够好。甚至在安全方面，也没有值得一提的大差异。MongoDB 有自己的授权方案，就像大多数关系型数据库一样，如果配置正确，它们都是一样安全的。通常，应用层在这方面更加麻烦。

# 摘要

这一章非常紧凑！我们对关系型和 NoSQL 数据库进行了概述，学习了 MongoDB 和 MongoEngine，SQLite 和 SQLAlchemy，以及如何使用扩展来将 Flask 与它们集成。知识积累得很快！现在你能够创建更复杂的带有数据库支持、自定义验证、CSRF 保护和用户通信的网络应用程序了。

在下一章中，我们将学习关于 REST 的知识，它的优势，以及如何创建服务供应用程序消费。
