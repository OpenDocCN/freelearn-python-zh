# 第八章. 与 Web 服务协作 – XML-RPC、SOAP 和 REST

在本章中，我们将涵盖以下食谱：

+   查询本地 XML-RPC 服务器

+   编写一个多线程、多调用 XML-RPC 服务器

+   使用基本 HTTP 身份验证运行 XML-RPC 服务器

+   使用 REST 从 Flickr 收集一些照片信息

+   从 Amazon S3 Web 服务中搜索 SOAP 方法

+   在 Google 上搜索自定义信息

+   通过产品搜索 API 在 Amazon 上搜索书籍

# 简介

本章介绍了使用三种不同方法（即**XML 远程过程调用**（**XML-RPC**）、**简单对象访问协议**（**SOAP**）和**表征状态转移**（**REST**））在 Web 服务中的一些有趣的 Python 食谱。Web 服务的理念是通过精心设计的协议在 Web 上使两个软件组件之间进行交互。接口是机器可读的。使用各种协议来促进 Web 服务。

在这里，我们提供了三个常用协议的示例。XML-RPC 使用 HTTP 作为传输媒介，通信使用 XML 内容进行。实现 XML-RPC 的服务器等待来自合适客户端的调用。客户端调用该服务器以执行具有不同参数的远程过程。XML-RPC 更简单，并考虑了最小安全性。另一方面，SOAP 有一套丰富的协议，用于增强远程过程调用。REST 是一种促进 Web 服务的架构风格。它使用 HTTP 请求方法操作，即`GET`、`POST`、`PUT`和`DELETE`。本章介绍了这些 Web 服务协议和风格的实际应用，以实现一些常见任务。

# 查询本地 XML-RPC 服务器

如果你做很多 Web 编程，你很可能会遇到这个任务：从一个运行 XML-RPC 服务的网站上获取一些信息。在我们深入研究 XML-RPC 服务之前，让我们先启动一个 XML-RPC 服务器并与它进行通信。

## 准备工作

在这个食谱中，我们将使用 Python Supervisor 程序，这是一个广泛用于启动和管理多个可执行程序的程序。Supervisor 可以作为后台守护进程运行，可以监控子进程，并在它们意外死亡时重新启动。我们可以通过简单地运行以下命令来安装 Supervisor：

```py
$pip install supervisor

```

## 如何操作...

我们需要为 Supervisor 创建一个配置文件。本食谱提供了一个示例配置。在这个例子中，我们定义了 Unix HTTP 服务器套接字和一些其他参数。注意`rpcinterface:supervisor`部分，其中`rpcinterface_factory`被定义为与客户端通信。

在`program:8_2_multithreaded_multicall_xmlrpc_server.py`部分，我们使用 Supervisor 配置一个简单的服务器程序，通过指定命令和一些其他参数。

列表 8.1a 给出了最小化 Supervisor 配置的代码，如下所示：

```py
[unix_http_server]
file=/tmp/supervisor.sock   ; (the path to the socket file)
chmod=0700                 ; socket file mode (default 0700)

[supervisord]
logfile=/tmp/supervisord.log 
loglevel=info                
pidfile=/tmp/supervisord.pid 
nodaemon=true               

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[program:8_2_multithreaded_multicall_xmlrpc_server.py]
command=python 8_2_multithreaded_multicall_xmlrpc_server.py ; the 
program (relative uses PATH, can take args)
process_name=%(program_name)s ; process_name expr (default 
%(program_name)s)
```

如果你使用你喜欢的编辑器创建前面的管理员配置文件，你可以通过简单地调用它来运行管理员。

现在，我们可以编写一个 XML-RPC 客户端，它可以充当管理员代理，并给我们提供有关正在运行进程的信息。

列表 8.1b 给出了查询本地 XML-RPC 服务器的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.
import supervisor.xmlrpc
import xmlrpclib

def query_supervisr(sock):
    transport = supervisor.xmlrpc.SupervisorTransport(None, None,
                'unix://%s' %sock)
    proxy = xmlrpclib.ServerProxy('http://127.0.0.1',
            transport=transport)
    print "Getting info about all running processes via Supervisord..."
    print proxy.supervisor.getAllProcessInfo()

if __name__ == '__main__':
    query_supervisr(sock='/tmp/supervisor.sock')
```

如果你运行管理员守护进程，它将显示类似于以下内容的输出：

```py
chapter8$ supervisord
2013-09-27 16:40:56,861 INFO RPC interface 'supervisor' initialized
2013-09-27 16:40:56,861 CRIT Server 'unix_http_server' running 
without any HTTP authentication checking
2013-09-27 16:40:56,861 INFO supervisord started with pid 27436
2013-09-27 16:40:57,864 INFO spawned: 
'8_2_multithreaded_multicall_xmlrpc_server.py' with pid 27439
2013-09-27 16:40:58,940 INFO success: 
8_2_multithreaded_multicall_xmlrpc_server.py entered RUNNING state, 
process has stayed up for > than 1 seconds (startsecs)

```

注意，我们的子进程 `8_2_multithreaded_multicall_xmlrpc_server.py` 已经启动。

现在，如果你运行客户端代码，它将查询管理员服务器的 XML-RPC 服务器接口并列出正在运行的进程，如下所示：

```py
$ python 8_1_query_xmlrpc_server.py 
Getting info about all running processes via Supervisord...
[{'now': 1380296807, 'group': 
'8_2_multithreaded_multicall_xmlrpc_server.py', 'description': 'pid 
27439, uptime 0:05:50', 'pid': 27439, 'stderr_logfile': 
'/tmp/8_2_multithreaded_multicall_xmlrpc_server.py-stderr---
supervisor-i_VmKz.log', 'stop': 0, 'statename': 'RUNNING', 'start': 
1380296457, 'state': 20, 'stdout_logfile': 
'/tmp/8_2_multithreaded_multicall_xmlrpc_server.py-stdout---
supervisor-eMuJqk.log', 'logfile': 
'/tmp/8_2_multithreaded_multicall_xmlrpc_server.py-stdout---
supervisor-eMuJqk.log', 'exitstatus': 0, 'spawnerr': '', 'name': 
'8_2_multithreaded_multicall_xmlrpc_server.py'}]

```

## 它是如何工作的...

这个菜谱依赖于在后台运行配置了 `rpcinterface` 的管理员守护进程。管理员启动了另一个 XML-RPC 服务器，如下所示：`8_2_multithreaded_multicall_xmlrpc_server.py`。

客户端代码有一个 `query_supervisr()` 方法，该方法接受一个管理员套接字参数。在这个方法中，使用 Unix 套接字路径创建了一个 `SupervisorTransport` 实例。然后，通过传递服务器地址和先前创建的 `transport`，通过实例化 `xmlrpclib` 的 `ServerProxy()` 类创建了一个 XML-RPC 服务器代理。

XML-RPC 服务器代理随后调用管理员的 `getAllProcessInfo()` 方法，该方法打印子进程的进程信息。这个过程包括 `pid`、`statename`、`description` 等等。

# 编写多线程多调用 XML-RPC 服务器

你可以让你的 XML-RPC 服务器同时接受多个调用。这意味着多个函数调用可以返回单个结果。除此之外，如果你的服务器是多线程的，那么你可以在单个线程中启动服务器后执行更多代码。程序的主线程将以这种方式不会被阻塞。

## 如何做...

我们可以创建一个继承自 `threading.Thread` 类的 `ServerThread` 类，并将一个 `SimpleXMLRPCServer` 实例封装为该类的属性。这可以设置为接受多个调用。

然后，我们可以创建两个函数：一个启动多线程、多调用的 XML-RPC 服务器，另一个创建对该服务器的客户端。

列表 8.2 给出了编写多线程、多调用 XML-RPC 服务器的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import xmlrpclib
import threading

from SimpleXMLRPCServer import SimpleXMLRPCServer

# some trivial functions
def add(x,y):
  return x+y

def subtract(x, y):
  return x-y

def multiply(x, y):
  return x*y

def divide(x, y):
  return x/y

class ServerThread(threading.Thread):
  def __init__(self, server_addr):
    threading.Thread.__init__(self)
    self.server = SimpleXMLRPCServer(server_addr)
    self.server.register_multicall_functions()
    self.server.register_function(add, 'add')
    self.server.register_function(subtract, 'subtract')
    self.server.register_function(multiply, 'multiply')
    self.server.register_function(divide, 'divide')

  def run(self):
    self.server.serve_forever()

def run_server(host, port):
  # server code
  server_addr = (host, port)
  server = ServerThread(server_addr)
  server.start() # The server is now running
  print "Server thread started. Testing the server..."

def run_client(host, port):
  # client code
  proxy = xmlrpclib.ServerProxy("http://%s:%s/" %(host, port))
  multicall = xmlrpclib.MultiCall(proxy)
  multicall.add(7,3)
  multicall.subtract(7,3)
  multicall.multiply(7,3)
  multicall.divide(7,3)
  result = multicall()
  print "7+3=%d, 7-3=%d, 7*3=%d, 7/3=%d" % tuple(result)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Multithreaded 
multicall XMLRPC Server/Proxy')
  parser.add_argument('--host', action="store", dest="host", 
default='localhost')
  parser.add_argument('--port', action="store", dest="port", 
default=8000, type=int)
  # parse arguments
  given_args = parser.parse_args()
  host, port =  given_args.host, given_args.port
  run_server(host, port)
  run_client(host, port)
```

如果你运行此脚本，你将看到类似于以下内容的输出：

```py
$ python 8_2_multithreaded_multicall_xmlrpc_server.py --port=8000
Server thread started. Testing the server...
localhost - - [25/Sep/2013 17:38:32] "POST / HTTP/1.1" 200 -
7+3=10, 7-3=4, 7*3=21, 7/3=2 

```

## 它是如何工作的...

在这个菜谱中，我们创建了一个继承自 Python 线程库的 `Thread` 类的 `ServerThread` 子类。这个子类初始化一个服务器属性，该属性创建一个 `SimpleXMLRPC` 服务器实例。XML-RPC 服务器地址可以通过命令行输入提供。为了启用多调用功能，我们在服务器实例上调用 `register_multicall_functions()` 方法。

然后，使用此 XML-RPC 服务器注册了四个简单的函数：`add()`、`subtract()`、`multiply()` 和 `divide()`。这些函数的操作正好与它们的名称所暗示的相同。

为了启动服务器，我们将主机和端口传递给`run_server()`函数。使用之前讨论过的`ServerThread`类创建服务器实例。这个服务器实例的`start()`方法启动 XML-RPC 服务器。

在客户端，`run_client()`函数从命令行接受相同的 host 和 port 参数。然后通过调用`xmlrpclib`中的`ServerProxy()`类创建之前讨论过的 XML-RPC 服务器的代理实例。这个代理实例随后被传递给`MultiCall`类实例`multicall`。现在，前面提到的四个简单的 RPC 方法可以运行，例如`add`、`subtract`、`multiply`和`divide`。最后，我们可以通过单个调用获取结果，例如`multicall()`。结果元组随后在一行中打印出来。

# 使用基本 HTTP 身份验证运行 XML-RPC 服务器

有时，你可能需要实现 XML-RPC 服务器的身份验证。这个配方提供了一个基本 HTTP 身份验证的 XML-RPC 服务器的示例。

## 如何做到这一点...

我们可以创建`SimpleXMLRPCServer`的子类，并覆盖其请求处理器，以便当请求到来时，它将与给定的登录凭证进行验证。

列表 8.3a 给出了运行具有基本 HTTP 身份验证的 XML-RPC 服务器的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import xmlrpclib
from base64 import b64decode
from SimpleXMLRPCServer  import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

class SecureXMLRPCServer(SimpleXMLRPCServer):

  def __init__(self, host, port, username, password, *args, 
**kargs):
    self.username = username
    self.password = password
    # authenticate method is called from inner class
    class VerifyingRequestHandler(SimpleXMLRPCRequestHandler):
      # method to override
      def parse_request(request):
        if\ SimpleXMLRPCRequestHandler.parse_request(request):
        # authenticate
          if self.authenticate(request.headers):
        return True
          else:
            # if authentication fails return 401
              request.send_error(401, 'Authentication\ failed 
ZZZ')
            return False
          # initialize
         SimpleXMLRPCServer.__init__(self, (host, port), 
requestHandler=VerifyingRequestHandler, *args, **kargs)

  def authenticate(self, headers):
    headers = headers.get('Authorization').split()
    basic, encoded = headers[0], headers[1]
    if basic != 'Basic':
      print 'Only basic authentication supported'
    return False
    secret = b64decode(encoded).split(':')
    username, password = secret[0], secret[1]
  return True if (username == self.username and password == 
self.password) else False

def run_server(host, port, username, password):
  server = SecureXMLRPCServer(host, port, username, password)
  # simple test function
  def echo(msg):
    """Reply client in  upper case """
    reply = msg.upper()
    print "Client said: %s. So we echo that in uppercase: %s" 
%(msg, reply)
  return reply
  server.register_function(echo, 'echo')
  print "Running a HTTP auth enabled XMLRPC server on %s:%s..." 
%(host, port)
  server.serve_forever()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Multithreaded 
multicall XMLRPC Server/Proxy')
  parser.add_argument('--host', action="store", dest="host", 
default='localhost')
  parser.add_argument('--port', action="store", dest="port", default=8000, type=int)
  parser.add_argument('--username', action="store", 
dest="username", default='user')
  parser.add_argument('--password', action="store", 
dest="password", default='pass')
  # parse arguments
  given_args = parser.parse_args()
  host, port =  given_args.host, given_args.port
  username, password = given_args.username, given_args.password
  run_server(host, port, username, password)
```

如果运行此服务器，则默认可以看到以下输出：

```py
$ python 8_3a_xmlrpc_server_with_http_auth.py 
Running a HTTP auth enabled XMLRPC server on localhost:8000...
Client said: hello server.... So we echo that in uppercase: HELLO 
SERVER...
localhost - - [27/Sep/2013 12:08:57] "POST /RPC2 HTTP/1.1" 200 -

```

现在，让我们创建一个简单的客户端代理，并使用与服务器相同的登录凭证。

列表 8.3b 给出了 XML-RPC 客户端的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import xmlrpclib

def run_client(host, port, username, password):
  server = xmlrpclib.ServerProxy('http://%s:%s@%s:%s' %(username, 
password, host, port, ))
  msg = "hello server..."
  print "Sending message to server: %s  " %msg
  print "Got reply: %s" %server.echo(msg)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Multithreaded 
multicall XMLRPC Server/Proxy')
  parser.add_argument('--host', action="store", dest="host", 
default='localhost')
  parser.add_argument('--port', action="store", dest="port", 
default=8000, type=int)
  parser.add_argument('--username', action="store", 
dest="username", default='user')
  parser.add_argument('--password', action="store", 
dest="password", default='pass')
  # parse arguments
  given_args = parser.parse_args()
  host, port =  given_args.host, given_args.port
  username, password = given_args.username, given_args.password
  run_client(host, port, username, password)
```

如果你运行客户端，那么它将显示以下输出：

```py
$ python 8_3b_xmprpc_client.py 
Sending message to server: hello server... 
Got reply: HELLO SERVER...

```

## 它是如何工作的...

在服务器脚本中，通过从`SimpleXMLRPCServer`继承创建`SecureXMLRPCServer`子类。在这个子类的初始化代码中，我们创建了`VerifyingRequestHandler`类，该类实际上拦截请求并使用`authenticate()`方法进行基本身份验证。

在`authenticate()`方法中，HTTP 请求作为参数传递。该方法检查`Authorization`值的是否存在。如果其值设置为`Basic`，则使用`base64`标准模块中的`b64decode()`函数解码编码后的密码。在提取用户名和密码后，它随后与服务器最初设置的凭证进行验证。

在`run_server()`函数中，定义了一个简单的`echo()`子函数，并将其注册到`SecureXMLRPCServer`实例中。

在客户端脚本中，`run_client()`简单地获取服务器地址和登录凭证，并将它们传递给`ServerProxy()`实例。然后通过`echo()`方法发送单行消息。

# 使用 REST 从 Flickr 收集一些照片信息

许多互联网网站通过它们的 REST API 提供 Web 服务接口。**Flickr**，一个著名的照片分享网站，有一个 REST 接口。让我们尝试收集一些照片信息来构建一个专门的数据库或其他与照片相关的应用程序。

## 如何做到这一点...

我们需要 REST URL 来执行 HTTP 请求。为了简化，这个菜谱中将 URL 硬编码。我们可以使用第三方`requests`模块来执行 REST 请求。它有方便的`get()`、`post()`、`put()`和`delete()`方法。

为了与 Flickr Web 服务通信，您需要注册并获取一个秘密 API 密钥。这个 API 密钥可以放在`local_settings.py`文件中，或者通过命令行提供。

列表 8.4 展示了使用 REST 从 Flickr 收集一些照片信息的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import json
import requests

try:
    from local_settings import flickr_apikey
except ImportError:
    pass

def collect_photo_info(api_key, tag, max_count):
    """Collects some interesting info about some photos from Flickr.com for a given tag """
    photo_collection = []
    url =  "http://api.flickr.com/services/rest/?method=flickr.photos.search&tags=%s&format=json&nojsoncallback=1&api_key=%s" %(tag, api_key)
    resp = requests.get(url)
    results = resp.json()
    count  = 0
    for p in results['photos']['photo']:
        if count >= max_count:
            return photo_collection
        print 'Processing photo: "%s"' % p['title']
        photo = {}
        url = "http://api.flickr.com/services/rest/?method=flickr.photos.getInfo&photo_id=" + p['id'] + "&format=json&nojsoncallback=1&api_key=" + api_key
        info = requests.get(url).json()
        photo["flickrid"] = p['id']
        photo["title"] = info['photo']['title']['_content']
        photo["description"] = info['photo']['description']['_content']
        photo["page_url"] = info['photo']['urls']['url'][0]['_content']

        photo["farm"] = info['photo']['farm']
        photo["server"] = info['photo']['server']
        photo["secret"] = info['photo']['secret']

        # comments
        numcomments = int(info['photo']['comments']['_content'])
        if numcomments:
            #print "   Now reading comments (%d)..." % numcomments
            url = "http://api.flickr.com/services/rest/?method=flickr.photos.comments.getList&photo_id=" + p['id'] + "&format=json&nojsoncallback=1&api_key=" + api_key
            comments = requests.get(url).json()
            photo["comment"] = []
            for c in comments['comments']['comment']:
                comment = {}
                comment["body"] = c['_content']
                comment["authorid"] = c['author']
                comment["authorname"] = c['authorname']
                photo["comment"].append(comment)
        photo_collection.append(photo)
        count = count + 1
    return photo_collection     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get photo info from Flickr')
    parser.add_argument('--api-key', action="store", dest="api_key", default=flickr_apikey)
    parser.add_argument('--tag', action="store", dest="tag", default='Python')
    parser.add_argument('--max-count', action="store", dest="max_count", default=3, type=int)
    # parse arguments
    given_args = parser.parse_args()
    api_key, tag, max_count =  given_args.api_key, given_args.tag, given_args.max_count
    photo_info = collect_photo_info(api_key, tag, max_count)
    for photo in photo_info:
        for k,v in photo.iteritems():
            if k == "title":
                print "Showing photo info...."  
            elif k == "comment":
                "\tPhoto got %s comments." %len(v)
            else:
                print "\t%s => %s" %(k,v) 
```

您可以通过将 API 密钥放入`local_settings.py`文件或从命令行（通过`--api-key`参数）提供它来运行这个菜谱。除了 API 密钥外，还可以提供搜索标签和结果的最大计数参数。默认情况下，这个菜谱将搜索`Python`标签，并将结果限制为三个条目，如下面的输出所示：

```py
$ python 8_4_get_flickr_photo_info.py 
Processing photo: "legolas"
Processing photo: ""The Dance of the Hunger of Kaa""
Processing photo: "Rocky"
 description => Stimson Python
Showiing photo info....
 farm => 8
 server => 7402
 secret => 6cbae671b5
 flickrid => 10054626824
 page_url => http://www.flickr.com/photos/102763809@N03/10054626824/
 description => &quot; 'Good. Begins now the dance--the Dance of the Hunger of Kaa. Sit still and watch.'

He turned twice or thrice in a big circle, weaving his head from right to left. 
Then he began making loops and figures of eight with his body, and soft, oozy triangles that melted into squares and five-sided figures, and coiled mounds, never resting, never hurrying, and never stopping his low humming song. It grew darker and darker, till at last the dragging, shifting coils disappeared, but they could hear the rustle of the scales.&quot;
(From &quot;Kaa's Hunting&quot; in &quot;The Jungle Book&quot; (1893) by Rudyard Kipling)

These old abandoned temples built around the 12th century belong to the abandoned city which inspired Kipling's Jungle Book.
They are rising at the top of a mountain which dominates the jungle at 811 meters above sea level in the centre of the jungle of Bandhavgarh located in the Indian state Madhya Pradesh.
Baghel King Vikramaditya Singh abandoned Bandhavgarh fort in 1617 when Rewa, at a distance of 130 km was established as a capital. 
Abandonment allowed wildlife development in this region.
When Baghel Kings became aware of it, he declared Bandhavgarh as their hunting preserve and strictly prohibited tree cutting and wildlife hunting...

Join the photographer at <a href="http://www.facebook.com/laurent.goldstein.photography" rel="nofollow">www.facebook.com/laurent.goldstein.photography</a>

© All photographs are copyrighted and all rights reserved.
Please do not use any photographs without permission (even for private use).
The use of any work without consent of the artist is PROHIBITED and will lead automatically to consequences.
Showiing photo info....
 farm => 6
 server => 5462
 secret => 6f9c0e7f83
 flickrid => 10051136944
 page_url => http://www.flickr.com/photos/designldg/10051136944/
 description => Ball Python
Showiing photo info....
 farm => 4
 server => 3744
 secret => 529840767f
 flickrid => 10046353675
 page_url => 
http://www.flickr.com/photos/megzzdollphotos/10046353675/

```

## 它是如何工作的...

这个菜谱展示了如何使用其 REST API 与 Flickr 进行交互。在这个例子中，`collect_photo_info()`标签接受三个参数：Flickr API 密钥、搜索标签和期望的搜索结果数量。

我们构建第一个 URL 来搜索照片。请注意，在这个 URL 中，方法参数的值是`flickr.photos.search`，期望的结果格式是 JSON。

第一次`get()`调用的结果存储在`resp`变量中，然后通过在`resp`上调用`json()`方法将其转换为 JSON 格式。现在，JSON 数据通过循环读取`['photos']['photo']`迭代器。创建一个`photo_collection`列表来返回经过信息整理后的结果。在这个列表中，每张照片信息由一个字典表示。这个字典的键是通过从早期的 JSON 响应和另一个`GET`请求中提取信息来填充的。

注意，为了获取关于照片的评论，我们需要进行另一个`get()`请求，并从返回的 JSON 的`['comments']['comment']`元素中收集评论信息。最后，这些评论被附加到一个列表中，并附加到照片字典条目中。

在主函数中，我们提取`photo_collection`字典并打印有关每张照片的一些有用信息。

# 从 Amazon S3 Web 服务搜索 SOAP 方法

如果您需要与实现简单对象访问过程（SOAP）的 Web 服务交互，那么这个菜谱可以帮助您找到一个起点。

## 准备工作

我们可以使用第三方`SOAPpy`库来完成这个任务。可以通过运行以下命令来安装它：

```py
$pip install SOAPpy

```

## 如何操作...

在我们可以调用它们之前，我们创建一个代理实例并检查服务器方法。

在这个菜谱中，我们将与 Amazon S3 存储服务进行交互。我们已获取了 Web 服务 API 的测试 URL。执行这个简单任务需要一个 API 密钥。

列表 8.5 给出了从 Amazon S3 网络服务中搜索 SOAP 方法的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter – 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import SOAPpy

TEST_URL = 'http://s3.amazonaws.com/ec2-downloads/2009-04-04.ec2.wsdl'

def list_soap_methods(url):
    proxy = SOAPpy.WSDL.Proxy(url)
    print '%d methods in WSDL:' % len(proxy.methods) + '\n'
    for key in proxy.methods.keys():
 "Key Details:"
        for k,v in proxy.methods[key].__dict__.iteritems():
            print "%s ==> %s" %(k,v)

if __name__ == '__main__':
    list_soap_methods(TEST_URL)
```

如果你运行此脚本，它将打印出支持 Web 服务定义语言（WSDL）的可用方法的总数以及一个任意方法的详细信息，如下所示：

```py
$ python 8_5_search_amazonaws_with_SOAP.py 
/home/faruq/env/lib/python2.7/site-packages/wstools/XMLSchema.py:1280: UserWarning: annotation is 
ignored
 warnings.warn('annotation is ignored')
43 methods in WSDL:

Key Name: ReleaseAddress
Key Details:
 encodingStyle ==> None
 style ==> document
 methodName ==> ReleaseAddress
 retval ==> None
 soapAction ==> ReleaseAddress
 namespace ==> None
 use ==> literal
 location ==> https://ec2.amazonaws.com/
 inparams ==> [<wstools.WSDLTools.ParameterInfo instance at 
0x8fb9d0c>]
 outheaders ==> []
 inheaders ==> []
 transport ==> http://schemas.xmlsoap.org/soap/http
 outparams ==> [<wstools.WSDLTools.ParameterInfo instance at 
0x8fb9d2c>]

```

## 它是如何工作的...

此脚本定义了一个名为`list_soap_methods()`的方法，它接受一个 URL 并通过调用`SOAPpy`的`WSDL.Proxy()`方法来构建 SOAP 代理对象。可用的 SOAP 方法都位于此代理的方法属性下。

通过遍历代理的方法键来迭代，`for`循环仅打印单个 SOAP 方法的详细信息，即键的名称及其详细信息。

# 搜索谷歌以获取自定义信息

在谷歌上搜索以获取关于某物的信息似乎对许多人来说是日常活动。让我们尝试使用谷歌搜索一些信息。

## 准备工作

此食谱使用第三方 Python 库`requests`，可以通过以下命令安装：

```py
$ pip install SOAPpy

```

## 如何操作...

谷歌有复杂的 API 来进行搜索。然而，它们要求你注册并按照特定方式获取 API 密钥。为了简单起见，让我们使用谷歌旧的普通**异步 JavaScript**（**AJAX**）API 来搜索有关 Python 书籍的一些信息。

列表 8.6 给出了搜索谷歌自定义信息的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 8
# This program is optimized for Python 2.7.# It may run on any other version with/without modifications.
import argparse
import json
import urllib
import requests

BASE_URL = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0' 

def get_search_url(query):
  return "%s&%s" %(BASE_URL, query)

def search_info(tag):
  query = urllib.urlencode({'q': tag})
  url = get_search_url(query)
  response = requests.get(url)
  results = response.json()

  data = results['responseData']
  print 'Found total results: %s' % 
data['cursor']['estimatedResultCount']
  hits = data['results']
  print 'Found top %d hits:' % len(hits)
  for h in hits: 
    print ' ', h['url']
    print 'More results available from %s' % 
data['cursor']['moreResultsUrl']

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Search info from 
Google')
  parser.add_argument('--tag', action="store", dest="tag", 
default='Python books')
  # parse arguments
  given_args = parser.parse_args()
  search_info(given_args.tag)
```

如果你通过指定`--tag`参数中的搜索查询来运行此脚本，那么它将搜索谷歌并打印出总结果数和前四个命中页面，如下所示：

```py
$ python 8_6_search_products_from_Google.py 
Found total results: 12300000
Found top 4 hits:
 https://wiki.python.org/moin/PythonBooks
 http://www.amazon.com/Python-Languages-Tools-Programming-
Books/b%3Fie%3DUTF8%26node%3D285856
 http://pythonbooks.revolunet.com/
 http://readwrite.com/2011/03/25/python-is-an-increasingly-popu
More results available from 
http://www.google.com/search?oe=utf8&ie=utf8&source=uds&start=0&hl=en
&q=Python+books

```

## 它是如何工作的...

在此食谱中，我们定义了一个简短的功能`get_search_url()`，它从`BASE_URL`常量和目标查询中构建搜索 URL。

主要搜索函数`search_info()`接受搜索标签并构建查询。使用`requests`库进行`get()`调用。然后，将返回的响应转换为 JSON 数据。

通过访问`'responseData'`键的值从 JSON 数据中提取搜索结果。然后通过访问结果数据的相关键提取估计结果和命中数。然后将前四个命中 URL 打印到屏幕上。

# 通过产品搜索 API 搜索亚马逊上的书籍

如果你喜欢在亚马逊上搜索产品并将其中一些包含在你的网站或应用程序中，这个食谱可以帮助你做到这一点。我们可以看看如何搜索亚马逊上的书籍。

## 准备工作

此食谱依赖于第三方 Python 库`bottlenose`。你可以使用以下命令安装此库：

```py
$ pip install  bottlenose

```

首先，你需要将你的亚马逊账户的访问密钥、秘密密钥和联盟 ID 放入`local_settings.py`。提供了一个带有书籍代码的示例设置文件。你也可以编辑此脚本并将其放置在此处。

## 如何操作...

我们可以使用实现了亚马逊产品搜索 API 的`bottlenose`库。

列表 8.7 给出了通过产品搜索 API 搜索亚马逊书籍的代码，如下所示：

```py
#!/usr/bin/env python
# Python Network Programming Cookbook -- Chapter - 8
# This program is optimized for Python 2.7.
# It may run on any other version with/without modifications.

import argparse
import bottlenose
from xml.dom import minidom as xml

try:
  from local_settings import amazon_account
except ImportError:
  pass 

ACCESS_KEY = amazon_account['access_key'] 
SECRET_KEY = amazon_account['secret_key'] 
AFFILIATE_ID = amazon_account['affiliate_id'] 

def search_for_books(tag, index):
  """Search Amazon for Books """
  amazon = bottlenose.Amazon(ACCESS_KEY, SECRET_KEY, AFFILIATE_ID)
  results = amazon.ItemSearch(
    SearchIndex = index,
    Sort = "relevancerank",
    Keywords = tag
  )
  parsed_result = xml.parseString(results)

  all_items = []
  attrs = ['Title','Author', 'URL']

  for item in parsed_result.getElementsByTagName('Item'):
    parse_item = {}

  for attr in attrs:
    parse_item[attr] = ""
    try:
      parse_item[attr] = 
item.getElementsByTagName(attr)[0].childNodes[0].data
    except:
      pass
    all_items.append(parse_item)
  return all_items

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Search info from 
Amazon')
  parser.add_argument('--tag', action="store", dest="tag", 
default='Python')
  parser.add_argument('--index', action="store", dest="index", 
default='Books')
  # parse arguments
  given_args = parser.parse_args()
  books = search_for_books(given_args.tag, given_args.index)    

  for book in books:
    for k,v in book.iteritems():
      print "%s: %s" %(k,v)
      print "-" * 80
```

如果你使用搜索标签和索引运行这个食谱，你可以看到一些类似以下输出的结果：

```py
$ python 8_7_search_amazon_for_books.py --tag=Python --index=Books
URL: http://www.amazon.com/Python-In-Day-Basics-Coding/dp/tech-data/1490475575%3FSubscriptionId%3DAKIAIPPW3IK76PBRLWBA%26tag%3D7052-6929-7878%26linkCode%3Dxm2%26camp%3D2025%26creative%3D386001%26creativeASIN%3D1490475575
Author: Richard Wagstaff
Title: Python In A Day: Learn The Basics, Learn It Quick, Start Coding Fast (In A Day Books) (Volume 1)
--------------------------------------------------------------------------------
URL: http://www.amazon.com/Learning-Python-Mark-Lutz/dp/tech-data/1449355730%3FSubscriptionId%3DAKIAIPPW3IK76PBRLWBA%26tag%3D7052-6929-7878%26linkCode%3Dxm2%26camp%3D2025%26creative%3D386001%26creativeASIN%3D1449355730
Author: Mark Lutz
Title: Learning Python
--------------------------------------------------------------------------------
URL: http://www.amazon.com/Python-Programming-Introduction-Computer-Science/dp/tech-data/1590282418%3FSubscriptionId%3DAKIAIPPW3IK76PBRLWBA%26tag%3D7052-6929-7878%26linkCode%3Dxm2%26camp%3D2025%26creative%3D386001%26creativeASIN%3D1590282418
Author: John Zelle
Title: Python Programming: An Introduction to Computer Science 2nd Edition
---------------------------------------------------------------------
-----------

```

## 它是如何工作的...

这个食谱使用第三方`bottlenose`库的`Amazon()`类来创建一个对象，通过产品搜索 API 搜索亚马逊。这是通过顶级`search_for_books()`函数完成的。这个对象的`ItemSearch()`方法通过传递`SearchIndex`和`Keywords`键的值来调用。它使用`relevancerank`方法对搜索结果进行排序。

搜索结果使用`xml`模块的`minidom`接口进行处理，该接口有一个有用的`parseString()`方法。它返回解析后的 XML 树形数据结构。这个数据结构上的`getElementsByTagName()`方法有助于找到物品的信息。然后查找物品属性，并将它们放置在解析物品的字典中。最后，所有解析的物品都附加到`all_items()`列表中，并返回给用户。
