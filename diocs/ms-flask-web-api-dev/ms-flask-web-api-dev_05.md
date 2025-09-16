# 第六章：<st c="0">5</st>

# <st c="2">构建异步事务</st>

<st c="36">在详细讨论了 Flask 3.0 框架的核心组件和高级特性之后，本章将探讨 Flask 管理请求和响应的异步能力以及其执行异步服务和</st> <st c="282">存储库事务的能力。</st>

<st c="306">Flask 最初是一个在</st> `<st c="589">asyncio</st>` <st c="596">工具上运行的</st> `<st c="659">SQLAlchemy 2.x</st>` <st c="596">构建异步存储库事务的标准 Python 框架。</st>

<st c="674">本章还将探讨其他途径，帮助 Flask 应用程序通过异步机制获得最快的性能，例如 Celery 任务、任务队列、WebSocket 和服务器推送。</st> <st c="880">本章还将介绍</st> **<st c="912">Quart</st>**<st c="917">，这是一个基于异步的 Flask 平台，可以与原始的</st> `<st c="1035">Flask 框架</st>` <st c="917">相比，构建和运行所有组件都是异步的。</st>

<st c="1051">本章将重点介绍以下主题：</st> <st c="1090">

+   <st c="1105">创建异步</st> <st c="1128">Flask 组件</st>

+   <st c="1144">构建异步 SQLAlchemy</st> <st c="1181">存储层</st>

+   <st c="1197">使用</st> <st c="1230">`asyncio`</st> <st c="1235">实现异步事务</st>

+   <st c="1242">利用异步</st> <st c="1266">信号通知</st>

+   <st c="1286">使用 Celery 和 Redis 构建后台任务</st> <st c="1329">

+   <st c="1338">使用异步事务构建 WebSocket</st> <st c="1364">

+   <st c="1389">实现异步</st> **<st c="1416">服务器端发送</st>** **<st c="1428">事件</st>** <st c="1434">(**<st c="1436">SSE</st>**<st c="1439">)</st>

+   <st c="1441">使用 RxPy 应用响应式编程</st> <st c="1471">

+   <st c="1480">选择 Quart 而非</st> <st c="1501">Flask 2.x</st>

# <st c="1510">技术要求</st>

本章将突出介绍一个具有一些异步任务和后台进程的**《在线投票系统》**原型，这些进程用于管理来自不同地区的候选人申请和与选举相关的提交的高带宽，以及满足从各个政党同时检索投票计数的需要。该系统由三个独立的项目组成，分别是 `<st c="1900">ch05-api</st>`<st c="1908">，它包含 API 端点，</st> `<st c="1939">ch05-web</st>`<st c="1947">，它实现了 SSE、WebSocket 和基于模板的结果，以及</st> `<st c="2018">ch05-quart</st>`<st c="2028">，它为应用程序提供了使用 Quart 框架的另一个平台。</st> `<st c="2101">所有这些项目都使用了</st>` **《应用工厂设计模式》** `<st c="2128">，并且可在</st> [<st c="2184">https://github.com/PacktPublishing/Mastering-Flask-Web-Development/tree/main/ch05</st>](https://github.com/PacktPublishing/Mastering-Flask-Web-Development/tree/main/ch05)<st c="2265">找到。</st>`

# 创建异步 Flask 组件

`<st c="2305">Flask 2.3 及以上版本</st>` `<st c="2322">到当前版本支持在其基于 WSGI 的平台上运行异步 API 端点和基于 Web 的视图函数。</st>` `<st c="2447">然而，要完全使用此功能，请使用以下</st>` `<st c="2535">pip</st>` `<st c="2538">命令安装</st>` `<st c="2495">flask[async]</st>` `<st c="2507">模块：</st>`

```py
 pip install flask[async]
```

在安装了 `<st c="2572">flask[async]</st>` `<st c="2606">模块后，现在可以使用</st>` `<st c="2656">async</st>`<st c="2661">/</st>`<st c="2663">await</st>` `<st c="2668">设计模式来实现同步视图。</st> `<st c="2692">现在这变得可行了。</st>`

## 实现异步视图和端点

与 `<st c="2750">Django 或 FastAPI</st>` 类似，在 Flask 框架中创建异步视图和端点涉及应用 `<st c="2862">async</st>`<st c="2867">/</st>`<st c="2869">await</st>` `<st c="2874">关键字。</st> `<st c="2885">以下来自</st>` `<st c="2913">ch05-web</st>` `<st c="2921">的网页视图向用户显示欢迎问候信息，并描述了我们的</st>` **《在线》** **《投票》** `<st c="3011">应用程序：</st>`

```py
 @current_app.route('/ch05/web/index') <st c="3063">async</st> def welcome():
    return render_template('index.html'), 200
```

`<st c="3125">来自另一个应用程序的另一个异步视图函数</st>` `<st c="3189">ch05-api</st>`<st c="3197">，在以下</st>` `<st c="3229">API 端点中展示，该端点向</st> **《数据库》** `<st c="3281">（</st>**《DB》**<st c="3291">）**<st c="3293">）**中添加新的登录凭证：</st>`

```py
 @current_app.post('/ch05/login/add') <st c="3334">async</st> def add_login():
   async with db_session() as sess:
            repo = LoginRepository(sess)
            login_json = request.get_json()
            login = Login(**login_json)
            result = await repo.insert(login)
            if result:
                content = jsonify(login_json)
                return make_response(content, 201)
            else:
                raise DuplicateRecordException("add login credential has failed")
```

给定的视图和 API 函数都使用 `<st c="3703">async</st>` `<st c="3708">路由</st>` 来管理它们各自的请求和响应对象。<st c="3773">使用</st>` `<st c="3800">async</st>` `<st c="3805">定义这些路由</st>` 创建协程，Flask 2.x 可以使用 `<st c="3871">run()</st>` `<st c="3876">工具</st>` 的 `<st c="3892">asyncio</st>` `<st c="3899">模块</st>` 意外地运行。<st c="3908">但是，Flask 框架如何在 WSGI 平台的缺陷中管理协程执行呢？</st>

Flask 创建一个 `<st c="4084">async</st>`，框架从工作线程创建一个 *<st c="4115">子线程</st>* 以创建 `<st c="4215">asyncio</st>` `<st c="4222">工具。</st>` 尽管存在异步过程，但由于环境仍然在 WSGI，一个同步平台中，因此 `<st c="4309">async</st>` `<st c="4314">可以推进的极限仍然有限。</st>` 然而，对于不太复杂的非阻塞事务，`<st c="4463">flask[async]</st>` `<st c="4475">框架足以提高软件质量和性能。</st>`

`<st c="4540">异步组件</st>` `<st c="4566">不仅限于视图和 API 函数，还有 Flask `<st c="4629">事件处理器</st>`。

## 实现异步的 `before_request` 和 `after_request` 处理器

除了视图和端点之外，Flask 3.0 框架允许实现异步的 `<st c="4808">before_request</st>` 和 `<st c="4827">after_request</st>` 处理器，就像以下 `<st c="4875">ch05-api</st>` `<st c="4883">处理器</st>` 那样记录每个 API `<st c="4912">请求事务</st>`：

```py
<st c="4932">@app.before_request</st>
<st c="4952">async</st> def init_request():
    app.logger.info('executing ' + request.endpoint + ' starts') <st c="5040">@app.after_request</st>
<st c="5058">async</st> def return_response(response):
    app.logger.info('executing ' + request.endpoint + ' stops')
    return response
```

这些事件处理器仍然使用在 `<st c="5231">main.py</st>` `<st c="5238">中创建的 `<st c="5207">app</st>` `<st c="5210">实例</st>`，使用内置的 `<st c="5282">logging</st>` `<st c="5289">模块</st>` 创建日志文件。

另一方面，`<st c="5321">flask[async]</st>` `<st c="5333">模块可以允许创建异步的错误处理器。</st>`

## 创建异步错误处理器

Flask 2.x 可以使用 `<st c="5472">@errorhandler</st>` `<st c="5485">装饰器</st>` 装饰协程，以管理引发的异常和 HTTP 状态码。<st c="5594">ch05-api</st>` `<st c="5602">项目的以下异步错误处理器放置在</st>` `<st c="5621">main.py</st>` `<st c="5628">:</st>`

```py
<st c="5630">@app.errorhandler(404)</st>
<st c="5652">async</st> def not_found(e):
    return jsonify(error=str(e)), 404 <st c="5711">@app.errorhandler(400)</st>
<st c="5733">async</st> def bad_request(e):
    return jsonify(error=str(e)), 400 <st c="5794">@app.errorhandler(DuplicateRecordException)</st>
<st c="5837">async</st> def insert_record_exception(e):
    return jsonify(e.to_dict()), e.code <st c="5912">async</st> def server_error(e):
    return jsonify(error=str(e)), 500
app.register_error_handler(500, server_error)
```

<st c="6018">现在，所有这些</st> <st c="6034">异步 Flask 组件也可以等待其他异步操作，例如应用程序的存储层。</st> <st c="6158">实际上，异步 Flask 环境对集成异步第三方扩展模块，如 async</st> <st c="6290">SQLAlchemy 2.x</st>，持开放态度。

# <st c="6305">构建异步 SQLAlchemy 存储层</st>

<st c="6358">更新的</st> <st c="6362">flask-sqlalchemy</st> <st c="6387">扩展模块支持 SQLAlchemy 2.x，它提供了 API 工具，这些工具使用</st> `<st c="6472">asyncio</st>` <st c="6479">环境，以</st> `<st c="6497">greenlet</st>` <st c="6505">作为主要库，允许在 API 的内部过程中传播</st> `<st c="6555">await</st>` <st c="6560">关键字。</st> <st c="6602">我们的</st> `<st c="6606">ch05-web</st>` <st c="6614">和</st> `<st c="6619">ch05-api</st>` <st c="6627">项目具有异步事务，这些事务调用这些等待的</st> <st c="6688">SQLAlchemy</st> `<st c="6791">/models/config.py</st>` <st c="6808">文件，该文件利用一个</st> `<st c="6831">asyncpg</st>` <st c="6838">驱动程序来构建一个用于非阻塞</st> <st c="6882">存储层事务的会话。</st>

## <st c="6906">设置数据库连接</st>

<st c="6937">要开始</st> <st c="6946">配置，请安装</st> `<st c="6983">asyncpg</st>` <st c="6990">数据库驱动程序或方言，该驱动程序或方言是</st> `<st c="7021">asyncio</st>`<st c="7028">-驱动的 SQLAlchemy 模块所必需的，使用</st> `<st c="7074">pip</st>` <st c="7077">命令：</st>

```py
 pip install asyncpg
```

<st c="7106">此外，如果绿色库尚未成为虚拟环境的一部分，请将其包含在安装中：</st>

```py
 pip install greenlet
```

<st c="7232">设置还需要</st> <st c="7257">一个</st> `<st c="7339">asyncpg</st>` <st c="7346">协议</st>，用户凭据，数据库服务器的主机地址，端口号以及模式名称。</st> <st c="7430">我们的项目使用连接字符串</st> `<st c="7470">postgresql+asyncpg://postgres:admin2255@localhost:5433/ovs</st>`<st c="7528">，来生成</st> `<st c="7546">AsyncEngine</st>` <st c="7557">实例，该实例使用 SQLAlchemy 框架的</st> `<st c="7591">create_async_engine()</st>` <st c="7612">函数，这是其</st> `<st c="7674">create_engine()</st>` <st c="7689">实用工具的异步版本。</st> <st c="7699">除了数据库 URL 之外，该方法还需要将它的</st> `<st c="7746">future</st>` <st c="7752">参数设置为</st> `<st c="7776">True</st>`<st c="7780">，将</st> `<st c="7782">pool_pre_pring</st>` <st c="7796">设置为</st> `<st c="7810">True</st>`<st c="7814">，并将</st> `<st c="7820">poolclass</st>` <st c="7829">设置为连接池策略，例如</st> `<st c="7881">NullPool</st>`<st c="7889">。</st> `<st c="7891">poolclass</st>` <st c="7900">管理 SQLAlchemy 在 CRUD 操作期间使用的线程，将其设置为</st> `<st c="7995">NullPool</st>` <st c="8003">将限制一个 Python 线程仅运行一个事件循环以执行一个 CRUD 操作。</st> `<st c="8087">pool_pre_ping</st>`<st c="8100">，另一方面，有助于处理断开连接的悲观方法的连接池。</st> <st c="8202">如果它确定数据库连接不可用或无效，那么在执行新操作之前，该连接及其之前的连接将立即回收。</st> <st c="8364">最重要的是，必须将</st> `<st c="8386">future</st>` <st c="8392">参数设置为</st> `<st c="8418">True</st>` <st c="8422">以启用 SQLAlchemy 2.x 的异步功能，否则异步 SQLAlchemy 设置将</st> <st c="8521">无法工作。</st>

<st c="8530">在成功创建之后，</st> `<st c="8566">sessionmaker</st>` <st c="8578">可调用对象将需要</st> `<st c="8602">AsyncEngine</st>` <st c="8613">实例来实例化每个 CRUD 操作所需的会话。</st> <st c="8686">然而，这次，会话将是</st> `<st c="8733">AsyncSession</st>` <st c="8745">类型，并且</st> `<st c="8760">async_scoped_session</st>` <st c="8780">可调用对象将帮助通过其提供的</st> `<st c="8836">scopefunc</st>` <st c="8845">参数来管理仓库层中的轻量级线程局部会话操作。</st> <st c="8941">每个仓库类都需要这个</st> `<st c="8982">AsyncSession</st>` <st c="8994">实例来实现异步</st> <st c="9068">Flask 平台</st>的每个必要的数据库事务。</st>

<st c="9083">Now, there is nothing new with the</st> `<st c="9119">declarative_base()</st>` <st c="9137">method, for it will still provide the needed helper classes to generate the model classes for the repository layer, like in the standard SQLAlchemy setup.</st> <st c="9293">The following is the complete module script of the</st> <st c="9344">specified</st> <st c="9354">SQLAlchemy setup:</st>

```py
<st c="9371">from sqlalchemy.ext.asyncio</st> import <st c="9407">create_async_engine</st>, <st c="9428">AsyncSession, async_scoped_session</st>
<st c="9462">from sqlalchemy.orm</st> import declarative_base, <st c="9508">sessionmaker</st>
<st c="9520">from sqlalchemy.pool import NullPool</st>
<st c="9557">from asyncio import current_task</st> DB_URL = "postgresql+<st c="9612">asyncpg</st>:// postgres:admin2255@localhost:5433/ovs"
engine = <st c="9673">create_async_engine</st>(DB_URL, <st c="9702">future=True</st>, echo=True, <st c="9726">pool_pre_ping=True</st>, <st c="9746">poolclass=NullPool</st>)
db_session = <st c="9780">async_scoped_session</st>(<st c="9802">sessionmaker</st>(engine, expire_on_commit=False, class_=AsyncSession), <st c="9870">scopefunc=current_task</st>)
Base = declarative_base() <st c="9921">def init_db():</st> import app.model.db
```

<st c="9955">The</st> `<st c="9960">echo</st>` <st c="9964">parameter of the given</st> `<st c="9988">create_async_engine</st>` <st c="10007">enables logging for the</st> `<st c="10032">AsyncEngine</st>`<st c="10043">-related transactions.</st> <st c="10067">Now, the</st> `<st c="10076">init_db()</st>` <st c="10085">method from the preceding configuration exposes the model classes to the different areas of the application.</st> <st c="10195">These model classes, built using the</st> `<st c="10232">Base</st>` <st c="10236">instance, help auto-generate the table schemas of our DB through the</st> *<st c="10306">Flask-Migrate</st>* <st c="10319">extension module, which still works with</st> `<st c="10361">flask[async]</st>` <st c="10373">and</st> `<st c="10378">flask-sqlalchemy</st>` <st c="10394">integration.</st>

<st c="10407">Let us now use the derived</st> `<st c="10435">async_scoped_session()</st>` <st c="10457">to build</st> <st c="10467">repository classes.</st>

## <st c="10486">Building the asynchronous repository layer</st>

<st c="10529">The</st> <st c="10533">asynchronous repository layer of the application requires the</st> `<st c="10596">AsyncSession</st>` <st c="10608">and the model classes to be created in the setup.</st> <st c="10659">The following is a</st> `<st c="10678">VoterRepository</st>` <st c="10693">class implementation that provides CRUD transactions for managing the</st> `<st c="10764">Voter</st>` <st c="10769">records:</st>

```py
<st c="10778">from sqlalchemy import update, delete, insert</st>
<st c="10824">from sqlalchemy.future import select</st>
<st c="10861">from sqlalchemy.orm import Session</st>
<st c="10896">from app.model.db import Voter</st> from datetime import datetime
class <st c="10964">VoterRepository</st>: <st c="11128">Session</st> object is always part of the constructor parameters of the repository class, like in the preceding <st c="11235">VoterRepository</st>.
			<st c="11251">Every operation under the</st> `<st c="11278">AsyncSession</st>` <st c="11290">scope requires an</st> `<st c="11309">await</st>` <st c="11314">process to finish its execution, which means every repository transaction must be</st> *<st c="11397">coroutines</st>*<st c="11407">. Every repository transaction requires an event loop to pursue its execution because of the</st> `<st c="11500">async</st>`<st c="11505">/</st>`<st c="11507">await</st>` <st c="11512">design pattern delegated</st> <st c="11538">by</st> `<st c="11541">AsyncSession</st>`<st c="11553">.</st>
			<st c="11554">The best-fit approach to applying the asynchronous</st> *<st c="11606">INSERT</st>* <st c="11612">operation is to utilize the</st> `<st c="11641">insert()</st>` <st c="11649">method from SQLAlchemy utilities.</st> <st c="11684">The</st> `<st c="11688">insert()</st>` <st c="11696">method will establish the</st> *<st c="11723">INSERT</st>* <st c="11729">command, which</st> `<st c="11745">AsyncSession</st>` <st c="11757">will</st> *<st c="11763">execute</st>*<st c="11770">,</st> *<st c="11772">commit</st>*<st c="11778">, or</st> *<st c="11783">roll back</st>* <st c="11792">asynchronously.</st> <st c="11809">The following is</st> `<st c="11826">VoterRepository</st>`<st c="11841">’s</st> <st c="11845">INSERT transaction:</st>

```

<st c="11864">async</st> def insert_voter(self, voter: Voter) -> bool:

        try: <st c="11922">sql = insert(Voter).values(mid=voter.mid,</st> <st c="11963">precinct=voter.precinct,</st> <st c="11988">voter_id=voter.voter_id,</st> <st c="12013">last_vote_date=datetime.strptime(</st><st c="12047">voter.last_vote_date, '%Y-%m-%d').date())</st><st c="12089">await</st> self.sess.execute(sql) <st c="12119">await</st> self.sess.commit() <st c="12144">await</st> self.sess.close()

            return True

        except Exception as e:

            print(e)

        return False

```py

			<st c="12224">As depicted in the preceding snippet, the transaction awaits the</st> `<st c="12290">execute()</st>`<st c="12299">,</st> `<st c="12301">commit()</st>`<st c="12309">, and</st> `<st c="12315">close()</st>` <st c="12322">methods to finish their respective tasks, which is a clear indicator that a repository operation needs to be a coroutine before executing these</st> `<st c="12467">AsyncSession</st>` <st c="12479">member methods.</st> <st c="12496">The same applies to the following UPDATE transaction of</st> <st c="12552">the</st> <st c="12555">repository:</st>

```

<st c="12567">async</st> def update_voter(self, id:int, details:Dict[str, Any]) -> bool:

    try: <st c="12643">sql = update(Voter).where(Voter.id ==</st> <st c="12680">id).values(**details)</st><st c="12702">await</st> self.sess.execute(sql) <st c="12732">await</st> self.sess.commit() <st c="12757">await</st> self.sess.close()

        return True

    except Exception as e:

        print(e)

    return False

```py

			<st c="12837">The preceding</st> `<st c="12852">update_voter()</st>` <st c="12866">also uses the same asynchronous approach as</st> `<st c="12911">insert_voter()</st>` <st c="12925">using the</st> `<st c="12936">AsyncSession</st>` <st c="12948">methods.</st> `<st c="12958">update_voter()</st>` <st c="12972">also needs an event loop from</st> <st c="13003">Flask to run successfully as an</st> <st c="13035">asynchronous task:</st>

```

<st c="13053">async</st> def delete_voter(self, id:int) -> bool:

        try: <st c="13105">sql = delete(Voter).where(Voter.id == id)</st><st c="13146">等待</st> self.sess.execute(sql) <st c="13176">等待</st> self.sess.commit() <st c="13201">等待</st> self.sess.close()

        return True

        except Exception as e:

            print(e)

        return False

```py

			<st c="13281">For the query transactions, the following are the repository’s coroutines that implement its SELECT</st> <st c="13382">operations:</st>

```

<st c="13393">异步</st> def select_all_voter(self): <st c="13428">sql = select(Voter)</st> q = <st c="13452">等待</st> self.sess.execute(sql)

        records = q.scalars().all() <st c="13509">等待</st> self.sess.close()

        return records

```py

			<st c="13547">Both</st> `<st c="13553">select_all_voter()</st>` <st c="13571">and</st> `<st c="13576">select_voter()</st>` <st c="13590">use the</st> `<st c="13599">select()</st>` <st c="13607">method from the</st> `<st c="13624">sqlalchemy</st>` <st c="13634">or</st> `<st c="13638">sqlalchemy.future</st>` <st c="13655">module.</st> <st c="13664">With the same objective as the</st> `<st c="13695">insert()</st>`<st c="13703">,</st> `<st c="13705">update()</st>`<st c="13713">, and</st> `<st c="13719">delete()</st>` <st c="13727">utilities, the</st> `<st c="13743">select()</st>` <st c="13751">method establishes a</st> *<st c="13773">SELECT</st>* <st c="13779">command object, which requires the asynchronous</st> `<st c="13828">execute()</st>` <st c="13837">utility for its execution.</st> <st c="13865">Thus, both query implementations are</st> <st c="13902">also coroutines:</st>

```

<st c="13918">异步</st> def select_voter(self, id:int): <st c="13957">sql = select(Voter).where(Voter.id == id)</st> q = <st c="14003">等待</st> self.sess.execute(sql)

        record = q.scalars().all() <st c="14059">等待</st> self.sess.close()

        return record

```py

			<st c="14096">In SQLAlchemy, the</st> *<st c="14116">INSERT</st>*<st c="14122">,</st> *<st c="14124">UPDATE</st>*<st c="14130">, and</st> *<st c="14136">DELETE</st>* <st c="14142">transactions technically utilize the model attributes that refer to the primary keys of the models’ corresponding DB tables, such as</st> `<st c="14276">id</st>`<st c="14278">. Conventionally, SQLAlchemy recommends updating and removing retrieved records based on</st> <st c="14366">their</st> `<st c="14546">update_precinct()</st>` <st c="14563">and</st> `<st c="14568">delete_voter_by_precinct()</st>` <st c="14594">of</st> <st c="14598">the repository:</st>

```

<st c="14613">异步</st> def update_precinct(self, old_prec:str,   new_prec:str) -> bool:

    try:

        sql = update(Voter).<st c="14708">where(Voter.precinct == old_prec).values(precint=new_prec)</st><st c="14767">sql.execution_options(synchronize_session=</st> <st c="14810">"fetch")</st> await self.sess.execute(sql)

        await self.sess.commit()

        await self.sess.close()

        return True

    except Exception as e:

        print(e)

    return False

```py

			`<st c="14954">update_precinct()</st>` <st c="14972">searches a</st> `<st c="14984">Voter</st>` <st c="14989">record with an existing</st> `<st c="15014">old_prec</st>` <st c="15022">(old precinct) and replaces it with</st> `<st c="15059">new_prec</st>` <st c="15067">(new precinct).</st> <st c="15084">There is no</st> `<st c="15096">id</st>` <st c="15098">primary key used to search the records for updating.</st> <st c="15152">The same scenario is also depicted in</st> `<st c="15190">delete_voter_by_precinct()</st>`<st c="15216">, which uses the</st> `<st c="15233">precinct</st>` <st c="15241">non-primary key value for record removal.</st> <st c="15284">Both</st> <st c="15288">transactions do not conform with the</st> <st c="15326">ideal</st> **<st c="15332">object-relational</st>** **<st c="15350">mapper</st>** <st c="15356">persistence:</st>

```

<st c="15369">异步</st> def delete_voter_by_precinct(self, precinct:str) -> bool:

        try:

        sql = delete(Voter).<st c="15458">where(Voter.precinct == precinct)</st><st c="15491">sql.execution_options(synchronize_session=</st> <st c="15534">"fetch")</st> await self.sess.execute(sql)

        await self.sess.commit()

        await self.sess.close()

        return True

        except Exception as e:

            print(e)

        return False

```py

			<st c="15678">In this regard, it is mandatory to perform</st> `<st c="15722">execution_options()</st>` <st c="15741">to apply the necessary synchronization strategy, preferably the</st> `<st c="15806">fetch</st>` <st c="15811">strategy, before executing the</st> *<st c="15843">UPDATE</st>* <st c="15849">and</st> *<st c="15854">DELETE</st>* <st c="15861">operations that do not conform with the ORM persistence.</st> <st c="15918">This mechanism provides the session with the resolution to manage the changes reflected by these two operations.</st> <st c="16031">For instance, the</st> `<st c="16049">fetch</st>` <st c="16054">strategy will let the session retrieve the primary keys of those records retrieved through the arbitrary values and will eventually update the in-memory objects or records affected by the operations and merge them into the actual table records.</st> <st c="16300">This setup is essential for the asynchronous</st> <st c="16345">SQLAlchemy operations.</st>
			<st c="16367">After</st> <st c="16374">building the repository layer, let us call these CRUD transactions in our view or</st> <st c="16456">API functions.</st>
			<st c="16470">Utilizing the asynchronous DB transactions</st>
			<st c="16513">To call the</st> <st c="16526">repository transactions, the asynchronous view and endpoint functions require an asynchronous context manager to create and manage</st> `<st c="16657">AsyncSession</st>` <st c="16669">for the repository class.</st> <st c="16696">The following is an</st> `<st c="16716">add_login()</st>` <st c="16727">API function that adds a new</st> `<st c="16757">Login</st>` <st c="16762">credential to</st> <st c="16777">the DB:</st>

```

from app.model.db import Login

from app.repository.login import LoginRepository <st c="16865">from app.model.config import db_session</st> @current_app.post('/ch05/login/add') <st c="16942">异步</st> def add_login(): <st c="16965">异步 with db_session() as sess:</st><st c="16997">异步 with sess.begin():</st><st c="17022">repo = LoginRepository(sess)</st> login_json = request.get_json()

            login = Login(**login_json) <st c="17112">结果 = 等待(repo.insert_login(login))</st> if 结果:

                content = jsonify(login_json)

                return make_response(content, 201)

            else:

                abort(500)

```py

			<st c="17244">The view function uses the</st> `<st c="17272">async with</st>` <st c="17282">context manager to localize the session for the coroutine or task execution.</st> <st c="17360">It opens the session for that specific task that will run the</st> `<st c="17422">insert_login()</st>` <st c="17436">transaction of</st> `<st c="17452">LoginRepository</st>`<st c="17467">. Then, eventually, the session will be closed by</st> <st c="17516">the repository or the context</st> <st c="17547">manager itself.</st>
			<st c="17562">Now, let us focus on another way of running asynchronous transactions using the</st> `<st c="17643">asyncio</st>` <st c="17650">library.</st>
			<st c="17659">Implementing async transactions with asyncio</st>
			<st c="17704">The</st> `<st c="17709">asyncio</st>` <st c="17716">module</st> <st c="17723">is an easy-to-use library for implementing asynchronous tasks.</st> <st c="17787">Compared to the</st> `<st c="17803">threading</st>` <st c="17812">module, the</st> `<st c="17825">asyncio</st>` <st c="17832">utilities use an event loop to execute each task, which is lightweight and easier to control.</st> <st c="17927">Threading uses one whole thread to run one specific operation, while</st> `<st c="17996">asyncio</st>` <st c="18003">utilizes only a single event loop to run all registered tasks concurrently.</st> <st c="18080">Thus, constructing an event loop is more resource friendly than running multiple threads to build</st> <st c="18178">concurrent transactions.</st>
			`<st c="18202">asyncio</st>` <st c="18210">is seamlessly compatible with</st> `<st c="18241">flask[async]</st>`<st c="18253">, and the clear proof is the following API function that adds a new voter to the DB using the task created by the</st> `<st c="18367">create_task()</st>` <st c="18380">method:</st>

```

from app.model.db import Member

from app.repository.member import MemberRepository

from app.model.config import db_session <st c="18512">from asyncio import create_task, ensure_future,</st> <st c="18559">InvalidStateError</st> from app.exceptions.db import DuplicateRecordException <st c="18633">@current_app.post("/ch05/member/add")</st> async def add_member():

    async with db_session() as sess:

        async with sess.begin():

            repo = MemberRepository(sess)

            member_json = request.get_json()

            member = Member(**member_json)

            try: <st c="18852">插入任务 =</st> <st c="18865">创建任务(repo.insert(member))</st><st c="18898">等待插入任务</st><st c="18916">结果 = 插入任务的结果</st> if 结果:

                    content = jsonify(member_json)

                    return make_response(content, 201)

                else:

                    raise DuplicateRecordException("插入成员记录失败")

            except InvalidStateError:

                abort(500)

```py

			<st c="19128">The</st> `<st c="19133">create_task()</st>` <st c="19146">method</st> <st c="19153">requires a coroutine to create a task and schedule its execution in an event loop.</st> <st c="19237">So, coroutines are not tasks at all, but they are the core inputs for generating these tasks.</st> <st c="19331">Running the scheduled task requires the</st> `<st c="19371">await</st>` <st c="19376">keyword.</st> <st c="19386">After its execution, the task returns a</st> `<st c="19426">Future</st>` <st c="19432">object that requires the task’s</st> `<st c="19465">result()</st>` <st c="19473">built-in method to retrieve its actual returned value.</st> <st c="19529">The given API transaction creates an</st> *<st c="19566">INSERT</st>* <st c="19572">task from the</st> `<st c="19587">insert_login()</st>` <st c="19601">coroutine and retrieves a</st> `<st c="19628">bool</st>` <st c="19632">result</st> <st c="19640">after execution.</st>
			<st c="19656">Now,</st> `<st c="19662">create_task()</st>` <st c="19675">automatically utilizes Flask’s internal event loop in running its tasks.</st> <st c="19749">However, for complex cases such as executing scheduled tasks,</st> `<st c="19811">get_event_loop()</st>` <st c="19827">or</st> `<st c="19831">get_running_loop()</st>` <st c="19849">are more applicable to utilize than</st> `<st c="19886">create_task()</st>` <st c="19899">due to their flexible settings.</st> `<st c="19932">get_event_loop()</st>` <st c="19948">gets the current running event loop, while</st> `<st c="19992">get_running_loop()</st>` <st c="20010">uses the running event in the current</st> <st c="20049">system’s thread.</st>
			<st c="20065">Another way of creating tasks from the coroutine is through</st> `<st c="20126">asyncio</st>`<st c="20133">’s</st> `<st c="20137">ensure_future()</st>`<st c="20152">. The following API uses this utility to spawn a task that lists all</st> <st c="20221">user accounts:</st>

```

<st c="20235">@current_app.get("/ch05/member/list/all")</st>

<st c="20277">async</st> def list_all_member():

    async with db_session() as sess:

        async with sess.begin():

            repo = MemberRepository(sess) <st c="20395">list_member_task =</st> <st c="20413">ensure_future(repo.select_all_member())</st><st c="20453">await list_member_task</st><st c="20476">records = list_member_task.result()</st> member_rec = [rec.to_json() for rec in records]

            return make_response(member_rec, 201)

```py

			<st c="20598">The only difference</st> <st c="20618">between</st> `<st c="20627">create_task()</st>` <st c="20640">and</st> `<st c="20645">ensure_future()</st>` <st c="20660">is that the former strictly requires coroutines, while the latter can accept coroutines,</st> `<st c="20750">Future</st>`<st c="20756">, or any awaitable objects.</st> `<st c="20784">ensure_future()</st>` <st c="20799">also invokes</st> `<st c="20813">create_task()</st>` <st c="20826">to wrap a</st> `<st c="20837">coroutine()</st>` <st c="20848">argument or directly return a</st> `<st c="20879">Future</st>` <st c="20885">result from a</st> `<st c="20900">Future</st>` <st c="20906">parameter object.</st>
			<st c="20924">On the other hand,</st> `<st c="20944">flask[async]</st>` <st c="20956">supports creating and running multiple tasks concurrently using</st> `<st c="21021">asyncio</st>`<st c="21028">. Its</st> `<st c="21034">gather()</st>` <st c="21042">method has</st> <st c="21054">two parameters:</st>

				*   <st c="21069">The first parameter is the sequence of coroutines,</st> `<st c="21121">Future</st>`<st c="21127">, or any</st> <st c="21136">awaitable objects.</st>
				*   <st c="21154">The second parameter is</st> `<st c="21179">return_exceptions</st>`<st c="21196">, which is set to</st> `<st c="21214">False</st>` <st c="21219">by default.</st>

			<st c="21231">The following is an endpoint function that inserts multiple profiles of candidates using</st> <st c="21321">concurrent tasks:</st>

```

<st c="21338">@current_app.post('/ch05/candidates/party')</st>

<st c="21382">async</st> def add_list_candidates():

    candidates = request.get_json()

    count_rec_added = 0 <st c="21468">results = await gather( *[insert_candidate_task(data)</st> <st c="21521">for data in candidates])</st> for success in results:

        如果成功：

            count_rec_added = count_rec_added + 1

    return jsonify(message=f'there are {count_rec_added} newly added candidates'), 201

```py

			<st c="21703">The given API</st> <st c="21717">expects a list of candidate profile details from</st> `<st c="21767">request</st>`<st c="21774">. A service named</st> `<st c="21792">insert_candidate_task()</st>` <st c="21815">will create a task that will convert the dictionary of objects to a</st> `<st c="21884">Candidate</st>` <st c="21893">instance and add the model instance to the DB through the</st> `<st c="21952">insert_candidate()</st>` <st c="21970">transaction of</st> `<st c="21986">CandidateRepository</st>`<st c="22005">. The following code showcases the complete implementation of this</st> <st c="22072">service task:</st>

```

<st c="22085">from asyncio import create_task</st> … … … … … … <st c="22129">async</st> def insert_candidate_task(data): <st c="22168">async with db_session() as sess:</st><st c="22200">async with sess.begin():</st> repo = CandidateRepository(sess) <st c="22259">insert_task =</st> <st c="22272">create_task(repo.insert_candidate(</st><st c="22307">Candidate(**data)))</st><st c="22327">await insert_task</st><st c="22345">result = insert_task.result()</st> return result

```py

			<st c="22389">Since our SQLAlchemy connection pooling is</st> `<st c="22433">NullPool</st>`<st c="22441">, which means connection pooling is disabled, we cannot utilize the same</st> `<st c="22514">AsyncSession</st>` <st c="22526">for all the</st> `<st c="22539">insert_candidate()</st>` <st c="22557">transactions.</st> <st c="22572">Otherwise,</st> `<st c="22583">gather()</st>` <st c="22591">will throw</st> `<st c="22603">RuntimeError</st>` <st c="22615">object.</st> <st c="22624">Thus, each</st> `<st c="22635">insert_candidate_task()</st>` <st c="22658">will open a new localized session for every</st> `<st c="22703">insert_candidate()</st>` <st c="22721">task execution.</st> <st c="22738">To add connection pooling, replace</st> `<st c="22773">NullPool</st>` <st c="22781">with</st> `<st c="22787">QueuePool</st>`<st c="22796">,</st> `<st c="22798">AsyncAdaptedQueuePool</st>`<st c="22819">,</st> <st c="22821">or</st> `<st c="22824">SingletonThreadPool</st>`<st c="22843">.</st>
			<st c="22844">Now, the</st> `<st c="22854">await</st>` <st c="22859">keyword will concurrently run the sequence of tasks registered in</st> `<st c="22926">gather()</st>` <st c="22934">and propagate all results in the resulting</st> `<st c="22978">tuple</st>` <st c="22983">of</st> `<st c="22987">Future</st>` <st c="22993">once these tasks have finished their execution successfully.</st> <st c="23055">The order of these</st> `<st c="23074">Future</st>` <st c="23080">objects is the same as the sequence of the awaitable objects provided in</st> `<st c="23154">gather()</st>`<st c="23162">. If a task has encountered failure or exception, it will not throw any exception and pre-empt the other task execution because</st> `<st c="23290">return_exceptions</st>` <st c="23307">of</st> `<st c="23311">gather()</st>` <st c="23319">is</st> `<st c="23323">False</st>`<st c="23328">. Instead, the failed task will join as a typical awaitable object in the</st> <st c="23402">resulting</st> `<st c="23412">tuple</st>`<st c="23417">.</st>
			<st c="23418">By the way, the</st> <st c="23434">given</st> `<st c="23441">add_list_candidates()</st>` <st c="23462">API function will return the number of successful INSERT tasks that persisted in the</st> <st c="23548">candidate profiles.</st>
			<st c="23567">The next section will discuss how to de-couple Flask components using the event-driven behavior of</st> <st c="23667">Flask</st> **<st c="23673">signals</st>**<st c="23680">.</st>
			<st c="23681">Utilizing asynchronous signal notifications</st>
			<st c="23725">Flask has a</st> <st c="23737">built-in lightweight event-driven mechanism called signals that can establish a loosely coupled software architecture using subscription-based event handling.</st> <st c="23897">It can trigger single or multiple transactions depending on the purpose.</st> <st c="23970">The</st> `<st c="23974">blinker</st>` <st c="23981">module provides the building blocks for Flask signal utilities, so install</st> `<st c="24057">blinker</st>` <st c="24064">using the</st> `<st c="24075">pip</st>` <st c="24078">command if it is not yet in the</st> <st c="24111">virtual environment.</st>
			<st c="24131">Flask has built-in signals and listens to many Flask events and callbacks such as</st> `<st c="24214">render_template()</st>`<st c="24231">,</st> `<st c="24233">before_request()</st>`<st c="24249">, and</st> `<st c="24255">after_request()</st>`<st c="24270">. These signals, such as</st> `<st c="24295">request_started</st>`<st c="24310">,</st> `<st c="24312">request_finished</st>`<st c="24328">,</st> `<st c="24330">message_flashed</st>`<st c="24345">, and</st> `<st c="24351">template_rendered</st>`<st c="24368">, are found in the</st> `<st c="24387">flask</st>` <st c="24392">module.</st> <st c="24401">For instance, once a component connects to</st> `<st c="24444">template_rendered</st>`<st c="24461">, it will run its callback method after</st> `<st c="24501">render_template()</st>` <st c="24518">finishes posting a Jinja template.</st> <st c="24554">However, our target is to create custom</st> *<st c="24594">asynchronous signals</st>*<st c="24614">.</st>
			<st c="24615">To create custom signals, import the</st> `<st c="24653">Namespace</st>` <st c="24662">class from the</st> `<st c="24678">flask.signals</st>` <st c="24691">module and instantiate it.</st> <st c="24719">Use its instance to define and instantiate specific custom signals, each having a unique name.</st> <st c="24814">The following is a snippet from our applications that creates an event signal for election date verification and another for retrieving all the</st> <st c="24958">election details:</st>

```

<st c="24975">from flask.signals import Namespace</st>

<st c="25011">election_ns = Namespace()</st> check_election = election_ns.<st c="25067">signal('check_election')</st> list_elections = election_ns.<st c="25281">check_election_event</st>，例如，它具有以下使用<st c="25365">ElectionRepository</st>验证选举日期的异步方法：

```py
<st c="25411">@check_election.connect</st>
<st c="25435">async</st> def check_election_event(<st c="25467">app</st>, election_date):
    async with db_session() as sess:
        async with sess.begin():
            repo = ElectionRepository(sess)
            records = await repo.select_all_election()
            election_rec = [rec.to_json() for rec in records if rec.election_date == datetime.strptime(election_date, '%Y-%m-%d').date()]
            if len(election_rec) > 0:
                return True
            return False
```

            同时，我们的<st c="25798">`list_all_election()`</st> <st c="25814">API 端点具有以下</st> <st c="25833">`list_elections_event()`</st> <st c="25865">，它返回 JSON 格式的记录列表：</st> <st c="25922">：

```py
<st c="25934">@list_elections.connect</st>
<st c="25958">async</st> def list_elections_event(app):
    async with db_session() as sess:
        async with sess.begin():
            repo = ElectionRepository(sess)
            records = await repo.select_all_election()
            election_rec = [rec.to_json() for rec in records]
            return election_rec
```

            <st c="26198">事件或</st> <st c="26207">信号函数必须接受一个</st> *<st c="26239">发送者</st> <st c="26245">或</st> *<st c="26249">监听器</st> <st c="26257">作为第一个局部参数，后面跟着其他对事件事务至关重要的自定义</st> `<st c="26326">args</st>` <st c="26330">对象。</st> <st c="26375">如果事件机制是类作用域的一部分，函数的值必须是</st> `<st c="26460">self</st>` <st c="26464">或类实例本身。</st> <st c="26495">否则，如果信号用于全局事件处理，其第一个参数必须是 Flask</st> `<st c="26589">app</st>` <st c="26592">实例。</st>

            信号有一个 `<st c="26618">connect()</st>` 函数或装饰器，用于注册事件或函数作为其实现。这些事件将在调用者发出信号时执行一次。Flask 组件可以通过调用信号的 `<st c="26827">send()</st>` 或 `<st c="26833">send_async()</st>` 工具函数，并传递事件函数参数来发出信号。以下 `<st c="26907">verify_election()</st>` 端点通过 `<st c="26965">check_election</st>` 信号从数据库检查特定日期是否发生了选举：

```py
<st c="27031">@current_app.post('/ch05/election/verify')</st>
<st c="27074">async</st> def verify_election():
    election_json = request.get_json()
    election_date = election_json['election_date'] <st c="27186">result_tuple = await</st> <st c="27206">check_election.send_async(current_app,</st> <st c="27245">election_date=election_date)</st> isApproved = result_tuple[0][1]
    if isApproved:
        return jsonify(message=f'election for {election_date} is approved'), 201
    else:
        return jsonify(message=f'election for {election_date} is disabled'), 201
```

            如果事件函数是一个标准的 Python 函数，则通过信号 `<st c="27588">send()</st>` 方法发送其执行的通知。然而，如果它是一个异步方法，就像我们的情况一样，使用 `<st c="27667">send_async()</st>` 创建并运行协程的任务，并使用 `<st c="27730">await</st>` 提取其 `<st c="27751">Future</st>` 值。

            通常，信号可以采用可扩展应用程序中组件的解耦，以减少依赖关系并提高模块化和可维护性。这也有助于构建具有分布式架构设计的应用程序。然而，随着需求的复杂化和信号订阅者的数量增加，通知可能会降低整个应用程序的性能。因此，如果调用者和事件函数可以减少对彼此参数、返回值和条件的依赖，那么这是一个良好的设计。订阅者必须对事件函数有独立的范围。此外，创建一个灵活且目标不太狭窄的事件函数是一种良好的编程方法，这样许多 `<st c="28532">组件</st>` 可以订阅它。

            在探索了 Flask 如何使用其信号支持事件处理之后，现在让我们学习如何使用其平台创建后台进程。

            使用 Celery 和 Redis 构建后台任务

            <st c="28756">在 Flask 中使用其</st> <st c="28763">flask[async]</st> <st c="28804">平台</st> <st c="28807">创建后台进程</st> <st c="28823">或</st> <st c="28840">事务</st> <st c="28852">是不可能的。</st> <st c="28863">运行异步视图或端点的任务的事件循环不允许启动另一个事件循环来处理后台任务，因为它不能等待视图或端点完成其处理后再结束后台进程。</st> <st c="29131">然而，通过一些第三方组件，如任务队列，对于 Flask 平台来说，后台处理是可行的。</st>

            <st c="29252">其中一个解决方案是使用 Celery，它是一个异步任务队列，可以在应用程序上下文之外运行进程。</st> <st c="29391">因此，当事件循环正在运行视图或端点的协程时，它们可以将后台事务的管理委托给 Celery。</st>

            <st c="29533">设置 Celery 任务队列</st>

            <st c="29566">在用 Celery 编写后台进程时，有一些</st> <st c="29583">考虑因素，首先是使用</st> <st c="29681">celery</st> <st c="29687">扩展模块通过</st> <st c="29715">pip</st> <st c="29718">命令进行安装：</st>

```py
 pip install celery
```

            <st c="29746">然后，我们在 WSGI 服务器中指定一些本地工作者来运行 Celery 队列中的后台任务，但在我们的应用程序中，我们的 Flask 服务器将只使用一个工作者来运行所有</st> <st c="29945">进程。</st>

            <st c="29959">现在让我们安装 Redis 服务器，它将作为消息代理为 Celery 服务。</st>

            <st c="30050">安装 Redis 数据库</st>

            <st c="30074">指定工作者后，Celery 需要一个消息代理来让工作者与客户端应用程序通信，以便运行后台任务。</st> <st c="30232">我们的应用程序使用 Redis 数据库作为代理。</st> <st c="30281">因此，在 Windows 上使用</st> <st c="30315">**<st c="30320">Windows Subsystem for Linux</st>** <st c="30347">(</st>**<st c="30349">WSL2</st>**<st c="30353">) shell</st> <st c="30347">或通过在</st> [<st c="30405">https://github.com/microsoftarchive/redis/releases</st>](https://github.com/microsoftarchive/redis/releases)<st c="30455">下载 Windows 安装程序</st> <st c="30402">来安装 Redis。</st>

            <st c="30456">下一步是向</st> <st c="30563">app</st> <st c="30566">实例</st> <st c="30563">添加必要的 Celery 配置变量，包括</st> `<st c="30537">CELERY_BROKER_URL</st>`<st c="30554">。</st>

            <st c="30576">设置 Celery 客户端配置</st>

            `<st c="30619">由于我们的项目使用</st>` `<st c="30642">TOML</st>` `<st c="30645">文件来设置配置环境变量，Celery 将从这些文件中作为 TOML 变量获取所有配置详细信息。</st> `<st c="30791">以下是对</st>` `<st c="30826">config_dev.toml</st>` `<st c="30841">文件的快照，该文件包含 Celery</st>` `<st c="30868">设置变量：</st>`

```py
 CELERY_BROKER_URL = "redis://127.0.0.1:6379/0"
CELERY_RESULT_BACKEND = "redis://127.0.0.1:6379/0 <st c="30982">[CELERY]</st> celery_store_errors_even_if_ignored = true
task_create_missing_queues = true
task_store_errors_even_if_ignored = true
task_ignore_result = false
broker_connection_retry_on_startup = true
celery_task_serializer = "pickle"
celery_result_serializer = "pickle"
celery_event_serializer = "json"
celery_accept_content = ["pickle", "application/json", "application/x-python-serialize"]
celery_result_accept_content = ["pickle", "application/json", "application/x-python-serialize"]
```

            Celery 客户端模块需要的两个最重要的变量是 `<st c="31465">CELERY_BROKER_URL</st>` `<st c="31478">和</st> `<st c="31483">CELERY_RESULT_BACKEND</st>` `<st c="31505">，它们分别提供 Redis 代理和后端服务器的地址、端口和 DB 名称。</st> `<st c="31682">Redis 有 DB</st>` `<st c="31696">0</st>` `<st c="31697">到</st>` `<st c="31701">15</st>` `<st c="31703">，但我们的应用程序仅使用 DB</st>` `<st c="31742">0</st>` `<st c="31743">作为默认用途。</st> `<st c="31766">由于</st>` `<st c="31776">CELERY_RESULT_BACKEND</st>` `<st c="31797">在此配置中不是那么重要，因此将</st>` `<st c="31843">CELERY_RESULT_BACKEND</st>` `<st c="31864">设置为定义的代理 URL 或从配置中删除它是可接受的。</st>`

            首先，创建包含 Celery 实例在管理后台任务执行所需详细信息的 `<st c="31943">CELERY</st>` `<st c="31961">TOML</st>` 字典。<st c="32081">首先，</st> `<st c="32088">celery_store_errors_even_if_ignored</st>` `<st c="32123">和</st> `<st c="32128">task_store_errors_even_if_ignored</st>` `<st c="32161">必须</st> `<st c="32170">设置为</st>` `<st c="32174">True</st>` `<st c="32178">以启用 Celery 执行期间的错误审计跟踪功能。</st> `<st c="32250">broker_connection_retry_on_startup</st>` `<st c="32284">应该</st> `<st c="32295">设置为</st>` `<st c="32299">True</st>` `<st c="32303">以防 Redis 仍在关闭模式。</st> `<st c="32341">另一方面，</st>` `<st c="32360">task_ignore_result</st>` `<st c="32378">必须</st> `<st c="32387">设置为</st>` `<st c="32392">False</st>` `<st c="32396">因为我们的一些协程作业将返回一些值给调用者。</st> `<st c="32471">此外，</st>` `<st c="32481">task_create_missing_queues</st>` `<st c="32507">设置为</st>` `<st c="32518">True</st>` `<st c="32522">以防在流量期间有未定义的任务队列供应用程序使用。</st> `<st c="32612">顺便说一下，默认任务队列的名称</st>` `<st c="32654">是</st>` `<st c="32657">celery</st>` `<st c="32663">。</st>`

            <st c="32664">其他细节包括任务可以接受用于其协程的资源 mime-type（</st>`<st c="32760">celery_accept_content</st>`<st c="32782">）以及这些后台进程可以向调用者返回的返回值（</st>`<st c="32868">celery_result_accept_content</st>`<st c="32897">）。</st> <st c="32901">任务序列化器也是细节的一部分，因为它们是将任务的传入参数和返回</st> <st c="33039">值转换为可接受状态和有效</st> <st c="33089">mime-type 类型</st>的机制。

            <st c="33105">现在，让我们专注于构建我们项目的 Celery 客户端模块，从创建</st> <st c="33218">Celery 实例</st>开始。

            <st c="33234">创建客户端实例</st>

            <st c="33263">由于本章中的所有</st> <st c="33274">项目都使用应用程序工厂方法，识别应用程序作为 Celery 客户端的设置发生在</st> `<st c="33405">app/__init__.py</st>`<st c="33420">中。</st> 然而，确切的</st> `<st c="33441">Celery</st>` <st c="33447">类实例化发生在另一个模块中，</st> `<st c="33494">celery_config.py</st>`<st c="33510">，以避免循环导入错误。</st> <st c="33545">以下代码片段显示了在</st> `<st c="33614">celery_config.py</st>`<st c="33630">中创建</st> `<st c="33598">Celery</st>` <st c="33604">类的实例：</st>

```py
<st c="33632">from celery import Celery, Task</st> from flask import Flask
def <st c="33692">celery_init_app</st>(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object: <st c="33818">with app.app_context():</st> return self.run(*args, **kwargs)
    celery_app = <st c="33888">Celery(app.name, task_cls=FlaskTask,</st> <st c="33924">broker=app.config["CELERY_BROKER_URL"],</st> <st c="33964">backend=app.config["CELERY_RESULT_BACKEND"])</st><st c="34009">celery_app.config_from_object(app.config["CELERY"])</st><st c="34061">celery_app.set_default()</st> return celery_app
```

            <st c="34104">从前面的代码片段中，创建</st> `<st c="34158">Celery</st>` <st c="34164">类的实例严格需要 Celery 应用程序名称，</st> `<st c="34218">CELERY_BROKER_URL</st>`<st c="34235">，以及工作任务。</st> <st c="34258">第一个参数，Celery 应用程序名称，可以有任何指定的名称，或者直接使用 Flask 应用程序的名称，因为 Celery 客户端模块将在应用程序的线程中运行后台作业（</st>`<st c="34423">FlaskTask</st>`<st c="34433">）。</st>

            <st c="34456">在</st> <st c="34462">实例化 Celery 之后，Celery 实例，</st> `<st c="34510">celery_app</st>`<st c="34520">，需要从 Flask</st> `<st c="34578">app</st>` <st c="34581">加载</st> `<st c="34540">CELERY</st>` <st c="34546">TOML 字典来配置任务队列及其消息代理。</st> <st c="34634">最后，</st> `<st c="34642">celery_app</st>` <st c="34652">必须调用</st> `<st c="34665">set_default()</st>` <st c="34678">来封闭配置。</st> <st c="34706">现在，</st> `<st c="34711">app/__init__.py</st>` <st c="34726">将导入</st> `<st c="34743">celery_init_app()</st>` <st c="34760">工厂，以最终从 Flask 应用程序中创建 Celery 客户端。</st>

            <st c="34853">现在，让我们使用自定义任务来构建 Celery 客户端模块。</st>

            <st c="34914">实现 Celery 任务</st>

            <st c="34944">To avoid circular</st> <st c="34963">import problems, it is not advisable to import</st> `<st c="35010">celery_app</st>` <st c="35020">and use it to decorate functions with the</st> `<st c="35063">task()</st>` <st c="35069">decorator.</st> <st c="35081">The</st> `<st c="35085">shared_task()</st>` <st c="35098">decorator from the</st> `<st c="35118">celery</st>` <st c="35124">module is enough proxy to define functions as Celery tasks.</st> <st c="35185">Here is a Celery task that adds a new vote to</st> <st c="35231">a candidate:</st>

```py
<st c="35243">from celery import shared_task</st>
<st c="35274">from asyncio import run</st>
<st c="35298">@shared_task</st> def add_vote_task_wrapper(details): <st c="35348">async</st> def add_vote_task(details):
        try: <st c="35387">async</st> with db_session() as sess: <st c="35420">async</st> with sess.begin():
                repo = VoteRepository(sess)
                details_dict = loads(details)
                print(details_dict)
                election = Vote(**details_dict)
                result = await repo.insert(election)
                if result: <st c="35603">return str(True)</st> else: <st c="35626">return str(False)</st> except Exception as e:
            print(e) <st c="35676">return str(False)</st> return <st c="35776">add_vote_task_wrapper()</st>, must not be a coroutine. A Celery task is a class generated by any callable decorated by <st c="35890">@shared_task</st>, which means it cannot propagate the <st c="35940">await</st> keyword outwards with the <st c="35972">async</st> function call. However, it can enclose an asynchronous local method to handle all the operations asynchronously, such as <st c="36099">add_vote_task()</st>, which wraps and executes the INSERT transactions for new vote details. The Celery task can apply the <st c="36217">asyncio</st>’s <st c="36228">run()</st> utility method to run its async local function.
			<st c="36281">Since our Celery app does not ignore the result, our task returns a Boolean value converted into a string, a safe object type that a task can return to the caller.</st> <st c="36446">Although it is feasible to use pickling, through the</st> `<st c="36499">pickle</st>` <st c="36505">module, to pass an argument to or transport return values from Celery tasks to the callers, it might open vulnerabilities that can pose security risks to the application, such as accidentally exposing confidential information stored in the pickled object or unpickling/de-serializing</st> <st c="36790">malicious objects.</st>
			<st c="36808">Another approach to manage the Celery task’s input arguments and returned values, especially if they are collection types, is through the</st> `<st c="36947">loads()</st>` <st c="36954">and</st> `<st c="36959">dumps()</st>` <st c="36966">utilities of the</st> `<st c="36984">json</st>` <st c="36988">module.</st> <st c="36997">This</st> `<st c="37002">loads()</st>` <st c="37009">function deserializes a JSON string into a Python object while</st> `<st c="37073">dumps()</st>` <st c="37080">serializes Python objects (e.g., dictionaries, lists, etc.) into a JSON formatted string.</st> <st c="37171">However, sometimes, using</st> `<st c="37197">dumps()</st>` <st c="37204">to convert these objects to strings is not certain.</st> <st c="37257">There are data in the string payload that can cause serialization error, because Celery does not support their default format, such as</st> `<st c="37392">time</st>`<st c="37396">,</st> `<st c="37398">date</st>`<st c="37402">, and</st> `<st c="37408">datetime</st>`<st c="37416">. In this</st> <st c="37425">scenario, the</st> `<st c="37440">dumps()</st>` <st c="37447">method needs a custom serializer to convert these temporal data types to their equivalent</st> *<st c="37538">ISO 8601</st>* <st c="37546">formats.</st> <st c="37556">The following Celery task has the same problem, thus the presence of a</st> <st c="37627">custom</st> `<st c="37634">json_date_serializer()</st>`<st c="37656">:</st>

```

<st c="37658">@shared_task</st> def list_all_votes_task_wrapper():

    async def list_all_votes_task():

    async with db_session() as sess:

        async with sess.begin():

            repo = VoteRepository(sess)

            records = await repo.select_all_vote()

            vote_rec = [rec.to_json() for rec in records]

            return <st c="37917">dumps(vote_rec,</st> <st c="37932">default=json_date_serializer)</st> return <st c="37970">run(list_all_votes_task())</st>

<st c="37996">def json_date_serializer(obj):</st> if isinstance(obj, time):

        return obj.isoformat()

    raise TypeError ("Type %s not …" % type(obj))

```py

			<st c="38122">Among the many</st> <st c="38137">ways to implement a date serializer,</st> `<st c="38175">json_date_serializer()</st>` <st c="38197">uses the</st> `<st c="38207">time</st>`<st c="38211">’s</st> `<st c="38215">isoformat()</st>` <st c="38226">method to convert the time object to an</st> *<st c="38267">ISO 8601</st>* <st c="38275">or</st> *<st c="38279">HH:MM:SS:ssssss</st>* <st c="38294">formatted string value so that the task can return the list of vote records without conflicts on the</st> `<st c="38396">date</st>` <st c="38400">types.</st>
			<st c="38407">Running the Celery worker server</st>
			<st c="38440">After creating the</st> <st c="38460">Celery tasks, the next step is to run the built-in Celery server through the following command to check whether the server can</st> <st c="38587">recognize them:</st>

```

celery -A main.celery_app worker --loglevel=info -P solo

```py

			`<st c="38659">main</st>` <st c="38664">in the command is the</st> `<st c="38687">main.py</st>` <st c="38694">module, and</st> `<st c="38707">celery_app</st>` <st c="38717">is the Celery instance found in the</st> `<st c="38754">main.py</st>` <st c="38761">module.</st> <st c="38770">The</st> `<st c="38774">loglevel</st>` <st c="38782">option creates a console logger for the server, and the</st> `<st c="38839">P</st>` <st c="38840">option indicates the</st> *<st c="38862">concurrency pool</st>*<st c="38878">, which is</st> `<st c="38889">solo</st>` <st c="38893">in the given command.</st> *<st c="38916">Figure 5</st>**<st c="38924">.1</st>* <st c="38926">shows the screen details after the</st> <st c="38962">server started.</st>
			![Figure 5.1 – Server details after Celery server startup](img/B19383_05_001.jpg)

			<st c="39751">Figure 5.1 – Server details after Celery server startup</st>
			<st c="39806">Celery server fetched the</st> `<st c="39833">add_vote_task_wrapper()</st>` <st c="39856">and</st> `<st c="39861">list_all_votes_task_wrapper()</st>` <st c="39890">tasks, as indicated in</st> *<st c="39914">Figure 5</st>**<st c="39922">.1</st>*<st c="39924">. Thus, Flask views and endpoints can now use these tasks to cast and view the votes from users.</st> <st c="40021">Aside from the list of ready-to-use tasks, the server logs also show details of the default task queue,</st> `<st c="40125">celery</st>`<st c="40131">. Also, it indicates the concurrency pool type, which is</st> `<st c="40188">solo</st>`<st c="40192">, and has a concurrency worker limit of</st> `<st c="40232">8</st>`<st c="40233">. Among the</st> `<st c="40245">prefork</st>`<st c="40252">,</st> `<st c="40254">eventlet</st>`<st c="40262">,</st> `<st c="40264">gevent</st>`<st c="40270">, and</st> `<st c="40276">solo</st>` <st c="40280">concurrency options, our applications use</st> `<st c="40323">solo</st>` <st c="40327">and</st> `<st c="40332">eventlet</st>`<st c="40340">. However, to use</st> `<st c="40358">eventlet</st>`<st c="40366">, install the</st> `<st c="40380">eventlet</st>` <st c="40388">module using the</st> `<st c="40406">pip</st>` <st c="40409">command:</st>

```

pip install eventlet

```py

			<st c="40439">Our application uses the solo Celery execution pool because it runs within the worker process, which makes a task’s performance fast.</st> <st c="40574">This pool is fit for running resource-intensive tasks.</st> <st c="40629">Other better options are</st> `<st c="40654">eventlet</st>` <st c="40662">and</st> `<st c="40667">gevent</st>`<st c="40673">, which spawn greenlets, sometimes called green threads, cooperative threads, or coroutines.</st> <st c="40766">Most Input/Output-bound tasks run better with</st> `<st c="40812">eventlet</st>` <st c="40820">or</st> `<st c="40824">gevent</st>` <st c="40830">because they generate more threads and emulate a multi-threading environment</st> <st c="40908">for efficiency.</st>
			<st c="40923">Once the Celery</st> <st c="40940">server loads and recognizes the tasks with a worker managing the message queues, Flask view and endpoint functions can invoke the tasks now using Celery</st> <st c="41093">utility methods.</st>
			<st c="41109">Utilizing the Celery tasks</st>
			<st c="41136">Once the</st> <st c="41146">Celery worker server runs with the list of tasks, Flask’s</st> `<st c="41204">async</st>` <st c="41209">views and endpoints can now access and run these tasks like signals.</st> <st c="41279">These tasks will execute only when the caller invokes their built-in</st> `<st c="41348">delay()</st>` <st c="41355">or</st> `<st c="41359">apply_async()</st>` <st c="41372">methods.</st> <st c="41382">The following endpoint function runs</st> `<st c="41419">add_vote_task_wrapper()</st>` <st c="41442">to cast a vote for</st> <st c="41462">a user:</st>

```

<st c="41469">@current_app.post('/ch05/vote/add')</st>

<st c="41505">async</st> def add_vote():

    vote_json = request.get_json() <st c="41559">vote_str = dumps(vote_json)</st><st c="41586">task =</st> <st c="41593">add_vote_task_wrapper.apply_async(args=[vote_str])</st><st c="41644">result = task.get()</st> return jsonify(message=result), 201

```py

			<st c="41700">The given</st> `<st c="41711">add_vote()</st>` <st c="41721">endpoint retrieves the request JSON data and converts it to a string before passing it as an argument to</st> `<st c="41827">add_vote_task_wrapper()</st>`<st c="41850">. Without using the</st> `<st c="41870">await</st>` <st c="41875">keyword, the Celery task has</st> `<st c="41905">apply_async()</st>`<st c="41918">, which the invoker can use to trigger its execution with the argument.</st> `<st c="41990">apply_async()</st>` <st c="42003">returns an</st> `<st c="42015">AsyncResult</st>` <st c="42026">object with a</st> `<st c="42041">get()</st>` <st c="42046">method that returns the returned value, if any.</st> <st c="42095">It also has a</st> `<st c="42109">traceback</st>` <st c="42118">variable</st> <st c="42127">that retrieves an exception stack trace when the execution raises</st> <st c="42194">an exception.</st>
			<st c="42207">From creating asynchronous background tasks, let us move on to WebSocket implementation with</st> <st c="42301">asynchronous transactions.</st>
			<st c="42327">Building WebSockets with asynchronous transactions</st>
			<st c="42378">WebSocket is</st> <st c="42391">a well-known bi-directional communication between a server and browser-based clients.</st> <st c="42478">Many popular frameworks such as Spring, JSF, Jakarta EE, Django, FastAPI, Angular, and React support this technology, and Flask is one of them.</st> <st c="42622">However, this chapter will focus on implementing WebSocket and its client applications using the</st> <st c="42719">asynchronous paradigm.</st>
			<st c="42741">Creating the client-side application</st>
			<st c="42778">Our WebSocket implementation with the</st> <st c="42817">client-side application is in the</st> `<st c="42851">ch05-web</st>` <st c="42859">project.</st> <st c="42869">Calling</st> `<st c="42877">/ch05/votecount/add</st>` <st c="42896">from the</st> `<st c="42906">vote_count.py</st>` <st c="42919">view module will give us the following HTML form in</st> *<st c="42972">Figure 5</st>**<st c="42980">.2</st>*<st c="42982">, which handles the data entry for the final vote tally per precinct or</st> <st c="43054">election district:</st>
			![Figure 5.2 – Client-side application for adding final vote counts](img/B19383_05_002.jpg)

			<st c="43230">Figure 5.2 – Client-side application for adding final vote counts</st>
			<st c="43295">Our</st> <st c="43299">WebSocket captures election data from officers and then updates DB records in real time.</st> <st c="43389">It retrieves a string message from the server as a response.</st> <st c="43450">The HTML form and the</st> `<st c="43507">WebSocket</st>` <st c="43516">are in</st> `<st c="43524">pages/vote_count_add.html</st>` <st c="43549">of</st> `<st c="43553">ch05-web</st>`<st c="43561">. The following snippet is the JS code that communicates with our</st> <st c="43627">server-side</st> `<st c="43639">WebSocket</st>`<st c="43648">:</st>

```

<script> <st c="43660">const add_log = (message) => {</st> document.getElementById('add_log').innerHTML += `<span>${message}</span><br>`;

    }; <st c="43773">const socket = new WebSocket('ws://' + location.host</st> <st c="43825">+ '/ch05/vote/save/ws');</st><st c="43850">socket.addEventListener('message', msg => {</st><st c="43894">add_log('server: ' + msg.data);</st><st c="43926">});</st> document.getElementById('vote_form').onsubmit = data => {

            data.preventDefault();

            const election_id = document.getElementById('election_id');

            const precinct = document.getElementById('precinct');

            const final_tally = document.getElementById('final_tally');

            const approved_date = document.getElementById('approved_date');

            var vote_count = new Object();

            vote_count.election_id = election_id.value;

            vote_count.precinct  = precinct.value;

            vote_count.final_tally = final_tally.value;

            vote_count.approved_date = approved_date.value;

            var vote_count_json = JSON.stringify(vote_count);

            add_log('client: ' + vote_count_json); <st c="44544">socket.send(vote_count_json);</st> election_id.value = '';

            precinct.value = '';

            final_tally.value = '';

            approved_date.value = '';

    };

    </script>

```py

			<st c="44681">The preceding JS script will connect to the Flask server through</st> `<st c="44747">ws://localhost:5001/ch05/vote/save/ws</st>` <st c="44784">by instantiating the</st> `<st c="44806">WebSocket</st>` <st c="44815">API.</st> <st c="44821">When the connection is ready, the client can ask for vote details from the client through the form components.</st> <st c="44932">Submitting the data will create a JSON object out of the form data before sending the JSON formatted details to the server through the</st> `<st c="45067">WebSocket</st>` <st c="45076">connection.</st>
			<st c="45088">On the other</st> <st c="45102">hand, to capture the message from the server, the client must create a listener to the message emitter by calling the WebSocket’s</st> `<st c="45232">addEventListener()</st>`<st c="45250">, which will watch and retrieve any JSON message from the Flask server.</st> <st c="45322">The custom</st> `<st c="45333">add_log()</st>` <st c="45342">function will render the message to the front end using the</st> `<st c="45403"><</st>``<st c="45404">span></st>` <st c="45409">tag.</st>
			<st c="45414">Next, let us focus on the WebSocket implementation per se using the</st> `<st c="45483">flask-sock</st>` <st c="45493">module.</st>
			<st c="45501">Creating server-side transactions</st>
			<st c="45535">There are many ways to implement a</st> <st c="45571">server-side message emitter, such as</st> `<st c="45608">WebSocket</st>`<st c="45617">, in Flask, and many Flask extensions can provide support for it, such as</st> `<st c="45691">flask-socketio</st>`<st c="45705">,</st> `<st c="45707">flask-sockets</st>`<st c="45720">, and</st> `<st c="45726">flask-sock</st>`<st c="45736">. This chapter will use the</st> `<st c="45764">flask-sock</st>` <st c="45774">module to create WebSocket routes because it can implement WebSocket communication with minimal configuration and setup.</st> <st c="45896">So, to start, install the</st> `<st c="45922">flask-sock</st>` <st c="45932">extension using the</st> `<st c="45953">pip</st>` <st c="45956">command:</st>

```

pip install flask-sock

```py

			<st c="45988">Then, integrate the extension to Flask by instantiating the</st> `<st c="46049">Sock</st>` <st c="46053">class with the</st> `<st c="46069">app</st>` <st c="46072">instance as its required argument.</st> <st c="46108">The following</st> `<st c="46122">app/__init__.py</st>` <st c="46137">snippet shows the</st> `<st c="46156">flask-sock</st>` <st c="46166">setup:</st>

```

<st c="46173">from flask_sock import Sock</st>

<st c="46201">sock = Sock()</st> def create_app(config_file):

    app = Flask(__name__, template_folder='../app/pages', static_folder="../app/resources")

    app.config.from_file(config_file, toml.load)

    在<st c="46468">/api/ votecount_websocket.py 模块</st>中初始化<st c="46451">sock</st>实例以定义 WebSocket 路由。 <st c="46536">ws://localhost:5001/ch05/vote/save/ws</st>，这是由前面的 JS 代码调用的，具有以下路由实现：

```py
 from app import sock <st c="46680">@sock.route('/ch05/vote/save/ws')</st> def add_vote_count_server(<st c="46740">ws</st>):
    async def add_vote_count():
        while True:
            vote_count_json = ws.receive()
            vote_count_dict = loads(vote_count_json)
            async with db_session() as sess:
                repo = VoteCountRepository(sess)
                vote_count = VoteCount(**vote_count_dict)
                result = await repo.insert(vote_count)
                if result:
                    ws.send("data added")
                else:
                    ws.send("data not added")
    run(add_vote_count())
```

            <st c="47092">Sock</st> <st c="47101">实例有一个</st> `<st c="47117">route()</st>` <st c="47124">装饰器，用于定义 WebSocket 实现。</st> <st c="47174">WebSocket 路由函数或处理程序始终是非异步的，并需要一个接受从</st> `<st c="47310">Sock</st>`<st c="47314">注入的 WebSocket 对象的必需参数。</st> 这个<st c="47321">ws</st> <st c="47323">对象有一个<st c="47337">send()</st> <st c="47343">方法，用于向客户端应用程序发送数据，一个<st c="47396">receive()</st> <st c="47405">实用工具，用于接受来自客户端的消息，以及</st> `<st c="47457">close()</st>` <st c="47464">用于在运行时异常或与服务器相关的问题发生时强制断开双向通信。</st>

            <st c="47582">WebSocket 处理程序通常保持一个</st> *<st c="47622">开环过程</st> <st c="47639">，其中它可以首先通过</st> `<st c="47685">receive()</st>` <st c="47694">接收消息，然后使用</st> `<st c="47727">send()</st>` <st c="47733">连续地发出其消息，具体取决于消息的目的。</st>

            <st c="47790">在<st c="47806">add_vote_count_server()</st> <st c="47829">的情况下，它需要等待异步的</st> `<st c="47865">VoteCountRepository</st>` <st c="47884">的 INSERT 事务，WebSocket 路由函数内部必须存在一个类似于 Celery 任务的</st> `<st c="47911">async</st>` <st c="47916">本地方法。</st> <st c="48010">这个本地方法将封装异步操作，并且</st> `<st c="48077">asyncio</st>` <st c="48084">的</st> `<st c="48088">run()</st>` <st c="48093">将在路由函数内部执行它。</st>

            <st c="48136">现在，为了见证消息交换，</st> *<st c="48179">图 5</st>**<st c="48187">.3</st>* <st c="48189">显示了我们的 JS 客户端与运行时的</st> `<st c="48258">add_vote_count_server()</st>` <st c="48281">处理程序</st> `<st c="48290">之间的通信快照：</st>

            ![图 5.3 – 一个 JS 客户端与 flask-sock WebSocket 之间的消息交换](img/B19383_05_003.jpg)

            <st c="48668">图 5.3 – 一个 JS 客户端与 flask-sock WebSocket 之间的消息交换</st>

            <st c="48744">除了基于 Web 的客户端之外，WebSocket 还可以将数据传播或发送到</st> <st c="48820">API 客户端。</st>

            <st c="48832">创建 Flask API 客户端应用程序</st>

            <st c="48872">另一种通过 WebSocket 发射器连接的方式是通过 Flask 组件，而不是 JS 代码。</st> <st c="48888">有时，客户端应用程序不是由 HTML、CSS 和前端 JS 框架组成的支持 WebSocket 通信的 Web 组件。</st> <st c="49112">例如，在我们的</st> `<st c="49133">ch05-api</st>` <st c="49141">项目中，一个 POST API 函数</st> `<st c="49172">bulk_check_vote_count()</st>`<st c="49195">，要求列出候选人以计算他们在选举期间获得的选票。</st> <st c="49277">API 的输入是一个 JSON 字符串，如下所示</st> <st c="49338">样本数据：</st>

```py
 [
    {
        "election_id": 1,
        "cand_id": "PHL-101"
    },
    {
        "election_id": 1,
        "cand_id": "PHL-111"
    },
    {
        "election_id": 1,
        "cand_id": "PHL-005"
    }
]
```

            <st c="49485">然后，API 函数将此 JSON 输入转换为包含候选人和选举 ID 的字典列表。</st> <st c="49603">以下是实现此 API 函数的代码，该函数作为 WebSocket 的客户端：</st> <st c="49678">客户端：</st>

```py
<st c="49690">from simple_websocket import Client</st>
<st c="49726">from json import dumps</st>
<st c="49749">@current_app.post("/ch05/check/vote/counts/client")</st> def bulk_check_vote_count(): <st c="49831">ws =</st> <st c="49835">Client('ws://127.0.0.1:5000/ch05/check/vote/counts/ws',</st><st c="49891">headers={"Access-Control-Allow-Origin": "*"})</st><st c="49937">candidates = request.get_json()</st> for candidate in candidates:
            try:
                print(f'client sent: {candidate}') <st c="50039">ws.send(dumps(candidate))</st><st c="50064">vote_count = ws.receive()</st> print(f'client recieved: {vote_count}')
            except Exception as e:
                print(e)
    return jsonify(message="done client transaction"), 201
```

            <st c="50217">对于与</st> `<st c="50275">flask-sock</st>` <st c="50285">最兼容的 WebSocket 客户端扩展是</st> `<st c="50289">simple-websocket</st>`<st c="50305">，请使用以下</st> `<st c="50337">pip</st>` <st c="50340">命令安装此模块：</st>

```py
 pip install simple-websocket
```

            <st c="50378">从</st> `<st c="50417">simple-websocket</st>` <st c="50433">模块实例化</st> `<st c="50495">Client</st>` <st c="50401">类</st> <st c="50407">以连接到具有</st> `<st c="50459">flask-sock</st>` <st c="50469">WebSocket 发射器，并使用</st> `<st c="50493">Access-Control-Allow-Origin</st>` <st c="50520">允许跨域访问。</st> <st c="50551">然后，API 将通过</st> `<st c="50643">Client</st>`<st c="50649">的</st> `<st c="50653">send()</st>` <st c="50659">方法将转换为字符串的字典详情发送到发射器。</st>

            <st c="50667">另一方面，将接收来自</st> `<st c="50755">bulk_check_vote_count()</st>` <st c="50778">客户端 API 的选举详情的 WebSocket 路由具有以下实现：</st>

```py
<st c="50823">@sock.route("/ch05/check/vote/counts/ws")</st> def bulk_check_vote_count_ws(<st c="50895">websocket</st>): <st c="50909">async</st> def vote_count():
      While True:
        try: <st c="50950">candidate = websocket.receive()</st> candidate_map = loads(candidate)
          print(f'server received: {candidate_map}')
          async with db_session() as sess:
            async with sess.begin():
               repo = VoteRepository(sess) <st c="51144">count = await repo.count_votes_by_candidate(</st> <st c="51188">candidate_map["cand_id"],</st> <st c="51214">int(candidate_map["election_id"]))</st> vote_count_data = {"cand_id": candidate_map["cand_id"], "vote_count": count} <st c="51327">websocket.send(dumps(vote_count_data))</st> print(f'server sent: {candidate_map}')
        except  Exception as e:
          print(e)
          break <st c="51527">run()</st> from <st c="51538">asyncio</st> to execute asynchronous query transactions from <st c="51594">VoteRepository</st> and extract the total number of votes for each candidate sent by the API client. The emitter will send a newly formed dictionary containing the candidate’s ID and counted votes back to the client API in string format. So, the handshake in this setup is between two Flask components, the WebSocket route and an async Flask API.
			<st c="51935">There are other client-server interactions that</st> `<st c="51984">flask[async]</st>` <st c="51996">can build, and one of these is</st> <st c="52028">the SSE.</st>
			<st c="52036">Implementing asynchronous SSE</st>
			<st c="52066">Like the</st> <st c="52075">WebSocket, the SSE is a real-time mechanism for sending messages from the server to client applications.</st> <st c="52181">However, unlike the WebSocket, it establishes unidirectional communication between the server and</st> <st c="52279">client applications.</st>
			<st c="52299">There are many ways to build server push solutions in Flask, but our applications prefer using the built-in</st> <st c="52408">response’s</st> `<st c="52419">text/event-stream</st>`<st c="52436">.</st>
			<st c="52437">Implementing the message publisher</st>
			<st c="52472">SSE is a</st> *<st c="52482">server push</st>* <st c="52493">solution</st> <st c="52503">that requires an input source where it can listen for incoming data or messages in real time and push that data to its client applications.</st> <st c="52643">One of the reliable sources that will work with SSE is</st> <st c="52698">a</st> **<st c="52700">message broker</st>**<st c="52714">, which can store messages from various resources.</st> <st c="52765">It can also help the SSE generator function to listen for incoming messages before yielding them to</st> <st c="52865">the clients.</st>
			<st c="52877">In this chapter, our</st> `<st c="52899">ch05-web</st>` <st c="52907">application utilizes Redis as the broker, which our</st> `<st c="52960">ch05-api</st>` <st c="52968">project used for invoking the Celery background tasks.</st> <st c="53024">However, in this scenario, there is a need to create a Redis client application that will implement its publisher-subscribe pattern.</st> <st c="53157">So, install the</st> *<st c="53173">redis-py</st>* <st c="53181">extension by using the</st> `<st c="53205">pip</st>` <st c="53208">command:</st>

```

pip install redis

```py

			<st c="53235">This extension will provide us with the</st> `<st c="53276">Redis</st>` <st c="53281">client that will connect to the Redis server once instantiated in the</st> `<st c="53352">main</st>` <st c="53356">module.</st> <st c="53365">The following</st> `<st c="53379">main.py</st>` <st c="53386">snippet shows the setup of the Redis</st> <st c="53424">client application:</st>

```

from app import create_app <st c="53471">from redis import Redis</st> app = create_app('../config_dev.toml') <st c="53534">redis_conn = Redis(</st><st c="53553">db = 0,</st><st c="53561">host='127.0.0.1',</st><st c="53579">port=6379,</st><st c="53590">decode_responses=True</st> )

```py

			<st c="53614">The Redis callable requires details about the DB (</st>`<st c="53664">db</st>`<st c="53667">), port, and host address of the installed Redis server as its parameters for setup.</st> <st c="53753">Since Celery tasks can return bytes, the</st> `<st c="53794">Redis</st>` <st c="53799">constructor should set its</st> `<st c="53827">decode_response</st>` <st c="53842">parameter to</st> `<st c="53856">True</st>` <st c="53860">to enable binary message data decoding mechanism and receive decoded strings.</st> <st c="53939">The instance,</st> `<st c="53953">redis_conn</st>`<st c="53963">, will be the key to the message publisher implementation needed by the SSE.</st> <st c="54040">In the complaint module of the application, our input source is a form view function that requests the user its statement and voter’s ID before pushing these details to the Redis</st> <st c="54219">broker.</st> <st c="54227">The following is the view that publishes data to the</st> <st c="54280">Redis server:</st>

```

<st c="54293">from main import redis_conn</st> from  json import dumps <st c="54345">@current_app.route('/ch05/election/complaint/form', methods = ['GET','POST'])</st>

<st c="54422">async</st> def create_complaint():

    if request.method == "GET":

        return render_template('complaint_form.html')

    else:

        voter_id = request.form['voter_id']

        complaint = request.form['complaint']

        record = {'voter_id': voter_id, 'complaint': complaint} <st c="54663">redis_conn.publish("complaint_channel",</st> <st c="54702">dumps(record))</st> return render_template('complaint_form.html')

```py

			<st c="54763">The</st> `<st c="54768">Redis</st>` <st c="54773">client</st> <st c="54780">instance,</st> `<st c="54791">redis_conn</st>`<st c="54801">, has a</st> `<st c="54809">publish()</st>` <st c="54818">method that stores a message to Redis under a specific topic or channel, a point where a subscriber will fetch the message from the broker.</st> <st c="54959">The name of our Redis channel</st> <st c="54989">is</st> `<st c="54992">complaint_channel</st>`<st c="55009">.</st>
			<st c="55010">Building the server push</st>
			<st c="55035">Our SSE will be</st> <st c="55051">the subscriber to</st> `<st c="55070">complaint_channel</st>`<st c="55087">. It will create a subscriber object first, through</st> `<st c="55139">redis_conn</st>`<st c="55149">’s</st> `<st c="55153">pubsub()</st>` <st c="55161">method, to connect to Redis and eventually use the broker to listen for any published message from the form view.</st> <st c="55276">The following is our SSE implementation using the</st> `<st c="55326">async</st>` <st c="55331">Flask route:</st>

```

<st c="55344">@current_app.route('/ch05/elec/comaplaint/stream')</st>

<st c="55395">async</st> def elec_complaint_sse():

    def process_complaint_event(): <st c="55459">connection = redis_conn.pubsub()</st><st c="55491">connection.subscribe('complaint_channel')</st> for message in <st c="55549">connection.listen()</st>:

            time.sleep(1)

            if message is not None and message['type'] == 'message': <st c="55642">data = message['data']</st><st c="55664">yield 'data: %s\n\n' % data</st> return <st c="55765">process_complaint_event()</st> 在给定的 SSE 路由中是 *<st c="55822">生成器函数</st>*，它创建订阅者对象 (<st c="55877">connection</st>)，通过调用 <st c="55926">subscribe()</st> 方法连接到 Redis，并构建一个开放循环事务，该事务将连续从代理监听当前发布的消息。它从订阅者对象的 <st c="56096">listen()</st> 实用程序检索到的消息是一个包含有关消息类型、通道和表单视图发布者发布的 <st c="56215">数据</st> 的 JSON 实体。<st c="56258">elec_complaint_sse()</st> 只需要产生消息的 <st c="56303">数据</st> 部分。现在，运行 <st c="56349">process_complaint_event()</st> 生成器需要 SSE 路由返回 Flask 的 <st c="56426">Response</st>，这将执行并渲染它为一个 <st c="56474">text/event-stream</st> 类型的对象。*<st c="56505">图 5</st>**<st c="56513">.4</st>* 展示了为投票者提供投诉的表单视图：

            ![图 5.4 – 从 Redis 发布的数据的投诉表单视图](img/B19383_05_004.jpg)

            <st c="56748">图 5.4 – 发布到 Redis 的数据投诉表单视图</st>

            *<st c="56813">图 5</st>**<st c="56822">.5</st>* 提供了包含从 Redis 代理推送的消息的 SSE 客户端页面的快照。

            ![图 5.5 – 从 Redis 推送数据渲染的 SSE 客户端页面](img/B19383_05_005.jpg)

            <st c="57234">图 5.5 – 从 Redis 推送数据渲染的 SSE 客户端页面</st>

            <st c="57298">除了代理消息之外，Flask 还支持其他库，这些库在创建其组件时使用发布者-订阅者设计模式。<st c="57433">下一个主题将展示其中之一，即</st> `<st c="57481">reactivex</st> <st c="57490">模块。</st>`

            <st c="57498">使用 RxPy 进行响应式编程</st>

            **<st c="57538">响应式编程</st>** <st c="57559">是当今兴起的一种流行编程范式之一，它侧重于异步数据流和可以管理执行、事件、存储库和异常传播的操作。<st c="57757">它利用发布者-订阅者编程方法，该方法在软件组件和事务之间建立异步交互。</st> <st c="57883">它支持 Flask 的其他库，这些库在创建其组件时使用发布者-订阅者设计模式。</st>

            <st c="57900">The</st> <st c="57904">library used to apply reactive streams to build services transactions and API functions in this chapter is</st> `<st c="58012">reactivex</st>`<st c="58021">, so install the module using the</st> `<st c="58055">pip</st>` <st c="58058">command:</st>

```py
 pip install reactivex
```

            <st c="58089">The</st> `<st c="58094">reactivex</st>` <st c="58103">module has an</st> `<st c="58118">Observable</st>` <st c="58128">class that generates data sources for the subscribers to consume.</st> `<st c="58195">Observer</st>` <st c="58203">is another API class that pertains to the subscriber entities.</st> `<st c="58267">reactivex</st>` <st c="58276">will not be a complete reactive programming library without its</st> *<st c="58341">operators</st>*<st c="58350">. The following is a vote-counting service implementation that uses the</st> `<st c="58422">reactivex</st>` <st c="58431">utilities:</st>

```py
<st c="58442">from reactivex import Observable, Observer, create</st>
<st c="58493">from reactivex.disposable import Disposable</st>
<st c="58537">from asyncio import ensure_future</st>
<st c="58571">async</st> def extract_precinct_tally(rec_dict):
    del rec_dict['id']
    del rec_dict['election_id']
    del rec_dict['approved_date']
    return str(rec_dict) <st c="58714">async</st> def create_tally_data(<st c="58742">observer</st>):
    async with db_session() as sess:
          async with sess.begin():
            repo = VoteCountRepository(sess)
            records = await repo.select_all_votecount()
            votecount_rec = [rec.to_json() for rec in records]
            print(votecount_rec)
            for vc in votecount_rec:
                rec_str = await extract_precinct_tally(vc) <st c="59030">observer.on_next(rec_str)</st><st c="59055">observer.on_completed()</st>
<st c="59388">create_tally_data()</st> and <st c="59412">extract_precinct_tally()</st> service operations that utilize these <st c="59475">async</st> queries also asynchronous. The objective is not to call these <st c="59543">async</st> services directly from the API layer but to wrap these service transactions in one <st c="59632">Observable</st> object through <st c="59658">create_observable()</st> and let the API functions subscribe to it. However, the problem is that <st c="59750">create_observable()</st> can’t be <st c="59779">async</st> because <st c="59793">reactivex</st> does not allow <st c="59818">async</st> to deal with its operators such as <st c="59859">create()</st>, <st c="59869">from_iterable()</st>, and <st c="59890">from_list()</st>.
			<st c="59902">With that, the</st> `<st c="59918">create_observable()</st>` <st c="59937">custom function needs a local-scoped subscriber function,</st> `<st c="59996">on_subscribe()</st>`<st c="60010">, that will invoke</st> `<st c="60029">create_task()</st>` <st c="60042">or</st> `<st c="60046">ensure_future()</st>` <st c="60061">with an event loop to create a task for the</st> `<st c="60106">create_tally_data()</st>` <st c="60125">coroutine and return it as a</st> `<st c="60155">Disposable</st>` <st c="60165">resource object.</st> <st c="60183">A disposable resource link allows for the cleaning up of the resources used by the observable operators during the subscription.</st> <st c="60312">Creating the</st> `<st c="60325">async</st>` <st c="60330">subscriber disposable will help manage the</st> <st c="60374">Flask resources.</st>
			<st c="60390">In connection with this setup,</st> `<st c="60422">create_tally_data()</st>` <st c="60441">will now emit the vote counts from the repository to the observer or subscriber.</st> <st c="60523">The only goal now of</st> `<st c="60544">create_observable()</st>` <st c="60563">is to return its created</st> `<st c="60589">Observable</st>` <st c="60599">based on the</st> `<st c="60613">on_subscribe()</st>` <st c="60627">emissions.</st>
			<st c="60638">The API transaction needs</st> <st c="60665">to run the</st> `<st c="60676">create_tally_date()</st>` <st c="60695">service and extract all the emitted vote counts by invoking</st> `<st c="60756">create_observable()</st>` <st c="60775">and subscribing to its returned</st> `<st c="60808">Observable</st>` <st c="60818">through the</st> `<st c="60831">subscribe()</st>` <st c="60842">method.</st> <st c="60851">The following is the</st> `<st c="60872">list_votecount_tally()</st>` <st c="60894">endpoint function that creates a subscription to the</st> <st c="60948">returned</st> `<st c="60957">Observable</st>`<st c="60967">:</st>

```

from app.services.vote_count import create_observable

from asyncio import get_event_loop, Future <st c="61067">@current_app.get("/ch05/votecount/tally")</st> async def list_votecount_tally():

    finished = Future() <st c="61163">loop = get_event_loop()</st> def on_completed():

        finished.set_result(0)

    tally = [] <st c="61241">disposable</st> = <st c="61254">create_observable(loop).subscribe(</st><st c="61288">on_next = lambda i: tally.append(i),</st><st c="61325">on_error = lambda e: print("Error</st> <st c="61359">Occurred: {0}".format(e)),</st><st c="61386">on_completed = on_completed)</st><st c="61415">await finished</st><st c="61430">disposable.dispose()</st> return jsonify(tally=tally), 201

```py

			`<st c="61484">subscribe()</st>` <st c="61496">has three</st> <st c="61506">callback methods that are all active and ready to run anytime</st> <st c="61569">when triggered:</st>

				*   `<st c="61584">on_next()</st>`<st c="61594">: This executes when</st> `<st c="61616">Observer</st>` <st c="61624">receives</st> <st c="61634">emitted data.</st>
				*   `<st c="61647">on_error()</st>`<st c="61658">: This executes when</st> `<st c="61680">Observable</st>` <st c="61690">encounters an exception along</st> <st c="61721">its operators.</st>
				*   `<st c="61735">on_completed()</st>`<st c="61750">: This runs when</st> `<st c="61768">Observable</st>` <st c="61778">completes</st> <st c="61789">its task.</st>

			<st c="61798">Our</st> `<st c="61803">on_next()</st>` <st c="61812">callback adds all the emitted data to the</st> <st c="61855">tally list.</st>
			<st c="61866">Now, the execution of the</st> `<st c="61893">Observable</st>` <st c="61903">operations will not be possible without the event loop.</st> <st c="61960">The API function needs the currently running event loop for the</st> `<st c="62024">create_tally_data()</st>` <st c="62043">coroutine execution, and thus its</st> `<st c="62078">get_event_loop()</st>` <st c="62094">invocation.</st> <st c="62107">The API will return the tally list once it disposes of the task running</st> <st c="62179">in</st> `<st c="62182">Observable</st>`<st c="62192">.</st>
			<st c="62193">Even though our framework is asynchronous Flask or the solutions applied to our applications are reactive and asynchronous, Flask will remain a WSGI-based framework, unlike FastAPI.</st> <st c="62376">The platform is still not 100% asynchronous friendly.</st> <st c="62430">However, if the application requires a 100% Flask environment, replace Flask with one of its variations called the</st> *<st c="62545">Quart</st>* <st c="62550">framework.</st>
			<st c="62561">Choosing Quart over Flask 2.x</st>
			<st c="62591">Quart</st> <st c="62597">is a Flask framework in and out but with a platform that runs entirely on</st> `<st c="62672">asyncio</st>`<st c="62679">. Many of the core features from Flask are part of the Quart framework, except for the main application class.</st> <st c="62790">The framework has its</st> `<st c="62812">Quart</st>` <st c="62817">class to set up</st> <st c="62834">an application.</st>
			<st c="62849">Moreover, Quart supports the</st> `<st c="63016">hypercorn</st>` <st c="63025">server, which supports HTTP/2</st> <st c="63056">request-response transactions.</st>
			<st c="63086">Since Quart and</st> <st c="63102">Flask are almost the same, migration of Flask applications to Quart is seamless and straightforward.</st> `<st c="63204">ch05-quart</st>` <st c="63214">is a product of migrating our</st> `<st c="63245">ch05-web</st>` <st c="63253">and</st> `<st c="63258">ch05-api</st>` <st c="63266">projects into using the Quart platform.</st> <st c="63307">The following is the</st> `<st c="63328">app/__init__.py</st>` <st c="63343">configuration of</st> <st c="63361">that project:</st>

```

<st c="63374">from quart import Quart</st> import toml

from app.model.config import init_db

from app.api.home import home, welcome

from app.api.login import add_login, list_all_login

def create_app(config_file): <st c="63568">app = Quart(__name__, template_folder='../app/pages',</st> <st c="63621">static_folder="../app/resources")</st><st c="63655">app.config.from_file(config_file, toml.load)</st> init_db()

    app.<st c="63715">add_url_rule</st>('/ch05/home', view_func=home, endpoint='home')

    app.<st c="63781">add_url_rule</st>('/ch05/welcome', view_func=welcome, endpoint='welcome')

    app.<st c="63856">add_url_rule</st>('/ch05/login/add', view_func=add_login, endpoint='add_login')

    app.<st c="63937">add_url_rule</st>('/ch05/login/list/all', view_func=list_all_login, endpoint='list_all_login')

    return app

```py

			<st c="64039">The</st> <st c="64044">Quart framework has a</st> `<st c="64066">Quart</st>` <st c="64071">class to build the application.</st> <st c="64104">Its constructor parameters, such as</st> `<st c="64140">template_folder</st>` <st c="64155">and</st> `<st c="64160">static_folder</st>`<st c="64173">, are the same as those of Flask.</st> <st c="64207">The framework can also recognize TOML</st> <st c="64245">configuration files.</st>
			<st c="64265">On the repository layer, the framework has a</st> `<st c="64311">quart-sqlalchemy</st>` <st c="64327">extension module that supports asynchronous ORM operations for Quart applications.</st> <st c="64411">There is no need to rewrite the model and repository classes during the migration because all the helper classes and utilities are the same as the</st> `<st c="64558">flask-sqlalchemy</st>` <st c="64574">extension.</st> <st c="64586">The same</st> `<st c="64595">init_db()</st>` <st c="64604">from the project’s application factory will set up and load the helper functions, methods, and model classes of the</st> `<st c="64721">quart-sqlalchemy</st>` <st c="64737">ORM.</st>
			<st c="64742">Quart also supports blueprint, application factory design, or even the hybrid approach in building the application.</st> <st c="64859">However, the current version,</st> *<st c="64889">Quart 0.18.4</st>*<st c="64901">, does not have an easy way to manage the asynchronous request context so that modules inside the application can access the</st> `<st c="65026">current_app</st>` <st c="65037">proxy for view or API implementation.</st> <st c="65076">That’s why, from the given configuration, the views and endpoints can be defined inside</st> `<st c="65164">create_app()</st>` <st c="65176">using</st> `<st c="65183">add_url_rule()</st>`<st c="65197">. Decorating them with</st> `<st c="65220">route()</st>` <st c="65227">in their respective module script using the</st> `<st c="65272">app</st>` <st c="65275">object or</st> `<st c="65286">current_app</st>` <st c="65297">raises an exception.</st> <st c="65319">Now, the following are the view and endpoint implementations in the</st> <st c="65387">Quart platform:</st>

```

<st c="65402">from quart import jsonify, render_template, request,</st> <st c="65455">make_response</st>

<st c="65469">async def add_login():</st> async with db_session() as sess:

            repo = LoginRepository(sess)

            login_json = request.get_json()

            login = Login(**login_json)

            result = await repo.insert(login)

            if result: <st c="65660">content = jsonify(login_json)</st><st c="65689">return await make_response(content, 201)</st> else:

                content = jsonify(message="insert complaint details record encountered a problem") <st c="65820">return await make_response(content, 500)</st>

<st c="65860">async</st> def welcome(): <st c="65949">render_template()</st> 和 <st c="65971">make_response()</st> 在 Quart 中需要使用 <st c="66010">await</st> 关键字。另一个区别是 Quart 使用 <st c="66062">hypercorn</st> 来运行其应用程序而不是 Werkzeug 服务器。因此，使用 <st c="66140">hypercorn</st> 命令安装 <st c="66160">pip</st>：

```py
 pip install hypercorn
```

            <st c="66194">然后，使用</st> `<st c="66230">hypercorn</st>` `<st c="66240">main:app</st>` <st c="66248">命令运行应用程序。</st>

            <st c="66257">到目前为止，在</st> <st c="66269">一般情况下，Quart 已经是一个有前途的异步框架。</st> <st c="66329">让我们希望与创建者、支持组和爱好者的合作能够帮助在不久的将来升级和扩展这个框架。</st>

            <st c="66472">摘要</st>

            <st c="66480">Flask 2.2 现在与其他支持并利用异步解决方案来提高应用程序运行性能的框架相当。</st> <st c="66629">它的视图和 API 函数现在可以是</st> `<st c="66667">async</st>` <st c="66672">并且可以在 Flask 创建的事件循环上运行。</st> <st c="66721">异步服务和事务现在可以在 Flask 平台上作为由</st> `<st c="66834">create_task()</st>` <st c="66847">和</st> `<st c="66852">ensure_future()</st>`<st c="66867">创建的任务执行和等待。</st>

            <st c="66868">最新的</st> *<st c="66880">SQLAlchemy[async]</st>* <st c="66897">可以轻松集成到 Flask 应用程序中，以提供异步 CRUD 事务。</st> <st c="66989">此外，使用 Flask 2.2 创建异步任务以分解 Celery 后台进程中的阻塞事务序列、WebSocket 消息传递和可观察操作现在成为可能。</st>

            <st c="67186">此外，通过内置的</st> <st c="67363">异步信号</st>，现在可以使用 Flask 2.2 设计松散耦合的组件、应用程序范围的跨切关注点解决方案和一些分布式设置。</st>

            <st c="67384">甚至有一个 100%异步的 Flask 框架叫做 Quart，可以构建快速响应的</st> <st c="67479">请求-响应事务。</st>

            <st c="67509">尽管 Flask 中异步支持的目的在于性能，但它在我们的应用程序中可以成为一部分的边界仍然存在。</st> <st c="67654">当与 <st c="67733">asyncio</st> <st c="67740">一起使用时，某些组件或实用程序将降低其运行时间。</st> <st c="67873">其他，如 CRUD 操作，由于数据库规范不符合异步设置，将减慢数据库访问速度。</st> <st c="67988">因此，异步编程的效果仍然取决于项目的需求和应用程序使用的资源。</st>

            <st c="68005">下一章将带我们进入 Flask 的计算世界，它涉及</st> `<st c="68091">numpy</st>`<st c="68096">,</st> `<st c="68098">pandas</st>`<st c="68104">, 图表，统计，文件序列化以及其他 Flask</st> <st c="68197">可以提供的科学解决方案。</st>

```py

```

```py

```

```py

```

```py

```
