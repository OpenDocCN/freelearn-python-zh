# 网络的测试驱动开发

**测试驱动开发**（TDD）的想法已经存在一段时间了。美国软件工程师肯特·贝克等人通常被认为是带领 TDD 运动的人，同时也是敏捷软件开发的领导者。敏捷软件开发需要非常短的构建-测试-部署开发周期；所有的软件需求都被转化为测试用例。这些测试用例通常是在编写代码之前编写的，只有当测试通过时，软件代码才会被接受。

相同的想法也可以与网络工程并行。当我们面临设计现代网络的挑战时，我们可以将这个过程分解为以下步骤：

+   我们从新网络的整体需求开始。为什么我们需要设计一个新的网络或部分新的网络？也许是为了新的服务器硬件，新的存储网络，或者新的微服务软件架构。

+   新的需求被分解为更小、更具体的需求。这可以是查看新的交换机平台，更高效的路由协议，或者新的网络拓扑（例如，fat-tree）。每个更小的需求都可以分解为必须和可选的类别。

+   我们制定测试计划，并根据潜在的解决方案进行评估。

+   测试计划将按相反的顺序进行；我们将从测试功能开始，然后将新功能集成到更大的拓扑中。最后，我们将尽可能接近生产环境来运行我们的测试。

关键是，即使我们没有意识到，我们可能已经在网络工程中采用了测试驱动的开发方法。这是我在学习 TDD 思维方式时的一部分启示。我们已经在不正式规范方法的情况下隐式地遵循了这一最佳实践。

通过逐渐将网络的部分作为代码移动，我们可以更多地使用 TDD 来进行网络。如果我们的网络拓扑以 XML 或 JSON 的分层格式描述，每个组件都可以正确映射并以所需的状态表达。这是我们可以编写测试用例的期望状态。例如，如果我们的期望状态要求交换机的全网状，我们可以始终编写一个测试用例来检查我们的生产设备的 BGP 邻居数量。

# 测试驱动开发概述

TDD 的顺序大致基于以下六个步骤：

1.  以结果为目标编写测试

1.  运行所有测试，看新测试是否失败

1.  编写代码

1.  再次运行测试

1.  如果测试失败，则进行必要的更改

1.  重复

我只是松散地遵循指南。TDD 过程要求在编写任何代码之前编写测试用例，或者在我们的情况下，在构建网络的任何组件之前。出于个人偏好的原因，我总是喜欢在编写测试用例之前看到一个工作的网络或代码版本。这给了我更高的信心水平。我也在测试的级别之间跳来跳去；有时我测试网络的一小部分；其他时候我进行系统级的端到端测试，比如 ping 或 traceroute 测试。

关键是，我不认为在测试方面有一种适合所有情况的方法。这取决于个人偏好和项目的范围。这对我合作过的大多数工程师来说都是真的。牢记框架是个好主意，这样我们就有了一个可行的蓝图，但你是解决问题风格的最佳评判者。

# 测试定义

让我们来看看 TDD 中常用的一些术语：

+   **单元测试**：检查小段代码。这是针对单个函数或类运行的测试。

+   **集成测试**：检查代码库的多个组件；多个单元组合在一起并作为一个组进行测试。这可以是针对 Python 模块或多个模块的测试

+   **系统测试**：端到端检查。这是一个尽可能接近最终用户所看到的测试

+   **功能测试**：针对单个功能的检查

+   **测试覆盖**：一个术语，用于确定我们的测试用例是否覆盖了应用程序代码。通常通过检查运行测试用例时有多少代码被执行来完成这一点。

+   **测试装置**：形成运行测试的基线的固定状态。测试装置的目的是确保测试运行在一个众所周知和固定的环境中，以便它们是可重复的

+   设置和拆卸：所有先决步骤都添加在设置中，并在拆卸中清理

这些术语可能看起来非常侧重于软件开发，并且有些可能与网络工程无关。请记住，这些术语是我们用来传达概念或步骤的一种方式，我们将在本章的其余部分中使用这些术语。随着我们在网络工程上下文中更多地使用这些术语，它们可能会变得更清晰。让我们深入探讨将网络拓扑视为代码。

# 拓扑作为代码

在我们宣称网络太复杂，无法总结成代码之前！让我们保持开放的心态。如果我告诉您，我们已经在本书中使用代码来描述我们的拓扑，这会有所帮助吗？

如果您查看本书中使用的任何 VIRL 拓扑图，它们只是包含节点之间关系描述的 XML 文件。

在本章中，我们将使用以下拓扑进行实验：

![](img/7a7b33f1-6e47-4a88-9d3e-11e5497c9237.png)

如果我们用文本编辑器打开拓扑文件`chapter13_topology.virl`，我们会看到该文件是一个描述节点和节点之间关系的 XML 文件。顶级根节点是带有`<node>`子节点的`<topology>`节点。每个子节点都包含各种扩展和条目。设备配置也嵌入在文件中：

```py
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<topology   schemaVersion="0.95" xsi:schemaLocation="http://www.cisco.com/VIRL https://raw.github.com/CiscoVIRL/schema/v0.95/virl.xsd">
    <extensions>
        <entry key="management_network" type="String">flat</entry>
    </extensions>
    <node name="iosv-1" type="SIMPLE" subtype="IOSv" location="182,162" ipv4="192.168.0.3">
        <extensions>
            <entry key="static_ip" type="String">172.16.1.20</entry>
            <entry key="config" type="string">! IOS Config generated on 2018-07-24 00:23
! by autonetkit_0.24.0
!
hostname iosv-1
boot-start-marker
boot-end-marker
!
...
    </node>
    <node name="nx-osv-1" type="SIMPLE" subtype="NX-OSv" location="281,161" ipv4="192.168.0.1">
        <extensions>
            <entry key="static_ip" type="String">172.16.1.21</entry>
            <entry key="config" type="string">! NX-OSv Config generated on 2018-07-24 00:23
! by autonetkit_0.24.0
!
version 6.2(1)
license grace-period
!
hostname nx-osv-1

...
<node name="host2" type="SIMPLE" subtype="server" location="347,66">
        <extensions>
            <entry key="static_ip" type="String">172.16.1.23</entry>
            <entry key="config" type="string">#cloud-config
bootcmd:
- ln -s -t /etc/rc.d /etc/rc.local
hostname: host2
manage_etc_hosts: true
runcmd:
- start ttyS0
- systemctl start getty@ttyS0.service
- systemctl start rc-local
    <annotations/>
    <connection dst="/virl:topology/virl:node[1]/virl:interface[1]" src="/virl:topology/virl:node[3]/virl:interface[1]"/>
    <connection dst="/virl:topology/virl:node[2]/virl:interface[1]" src="/virl:topology/virl:node[1]/virl:interface[2]"/>
    <connection dst="/virl:topology/virl:node[4]/virl:interface[1]" src="/virl:topology/virl:node[2]/virl:interface[2]"/>
</topology>
```

通过将网络表示为代码，我们可以为我们的网络声明一个真理源。我们可以编写测试代码来比较实际生产值与此蓝图。我们将使用此拓扑文件作为基础，并将生产网络值与其进行比较。但首先，我们需要从 XML 文件中获取我们想要的值。在`chapter13_1_xml.py`中，我们将使用`ElementTree`来解析`virl`拓扑文件，并构建一个包含我们设备信息的字典：

```py
#!/usr/env/bin python3

import xml.etree.ElementTree as ET
import pprint

with open('chapter13_topology.virl', 'rt') as f:
    tree = ET.parse(f)

devices = {}

for node in tree.findall('./{http://www.cisco.com/VIRL}node'):
    name = node.attrib.get('name')
    devices[name] = {}
    for attr_name, attr_value in sorted(node.attrib.items()):
        devices[name][attr_name] = attr_value

# Custom attributes
devices['iosv-1']['os'] = '15.6(3)M2'
devices['nx-osv-1']['os'] = '7.3(0)D1(1)'
devices['host1']['os'] = '16.04'
devices['host2']['os'] = '16.04'

pprint.pprint(devices)
```

结果是一个 Python 字典，其中包含根据我们的拓扑文件的设备。我们还可以向字典中添加习惯项目：

```py
$ python3 chapter13_1_xml.py
{'host1': {'location': '117,58',
           'name': 'host1',
           'os': '16.04',
           'subtype': 'server',
           'type': 'SIMPLE'},
 'host2': {'location': '347,66',
           'name': 'host2',
           'os': '16.04',
           'subtype': 'server',
           'type': 'SIMPLE'},
 'iosv-1': {'ipv4': '192.168.0.3',
            'location': '182,162',
            'name': 'iosv-1',
            'os': '15.6(3)M2',
            'subtype': 'IOSv',
            'type': 'SIMPLE'},
 'nx-osv-1': {'ipv4': '192.168.0.1',
              'location': '281,161',
              'name': 'nx-osv-1',
              'os': '7.3(0)D1(1)',
              'subtype': 'NX-OSv',
              'type': 'SIMPLE'}}
```

我们可以使用我们在第三章中的示例，*API 和意图驱动网络*，`cisco_nxapi_2.py`，来检索 NX-OSv 版本。当我们结合这两个文件时，我们可以比较我们从拓扑文件中收到的值以及生产设备信息。我们可以使用 Python 内置的`unittest`模块编写测试用例。

我们稍后将讨论`unittest`模块。如果您愿意，可以跳过并回到这个例子。

以下是`chapter13_2_validation.py`中相关的`unittest`部分：

```py
import unittest

# Unittest Test case
class TestNXOSVersion(unittest.TestCase):
    def test_version(self):
        self.assertEqual(nxos_version, devices['nx-osv-1']['os'])

if __name__ == '__main__':
    unittest.main()
```

当我们运行验证测试时，我们可以看到测试通过了，因为生产中的软件版本与我们预期的相匹配：

```py
$ python3 chapter13_2_validation.py
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

如果我们手动更改预期的 NX-OSv 版本值以引入失败案例，我们将看到以下失败的输出：

```py
$ python3 chapter13_3_test_fail.py
F
======================================================================
FAIL: test_version (__main__.TestNXOSVersion)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "chapter13_3_test_fail.py", line 50, in test_version
 self.assertEqual(nxos_version, devices['nx-osv-1']['os'])
AssertionError: '7.3(0)D1(1)' != '7.4(0)D1(1)'
- 7.3(0)D1(1)
? ^
+ 7.4(0)D1(1)
? ^

----------------------------------------------------------------------
Ran 1 test in 0.004s

FAILED (failures=1)
```

我们可以看到测试用例的结果返回为失败；失败的原因是两个值之间的版本不匹配。

# Python 的 unittest 模块

在前面的例子中，我们看到了如何使用`assertEqual()`方法来比较两个值，以返回`True`或`False`。以下是内置的`unittest`模块比较两个值的示例：

```py
$ cat chapter13_4_unittest.py
#!/usr/bin/env python3

import unittest

class SimpleTest(unittest.TestCase):
    def test(self):
        one = 'a'
        two = 'a'
        self.assertEqual(one, two)
```

使用`python3`命令行界面，`unittest`模块可以自动发现脚本中的测试用例：

```py
$ python3 -m unittest chapter13_4_unittest.py
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

除了比较两个值之外，这里还有更多的例子，测试预期值是否为`True`或`False`。当发生失败时，我们还可以生成自定义的失败消息：

```py
$ cat chapter13_5_more_unittest.py
#!/usr/bin/env python3
# Examples from https://pymotw.com/3/unittest/index.html#module-unittest

import unittest

class Output(unittest.TestCase):
    def testPass(self):
        return

    def testFail(self):
        self.assertFalse(True, 'this is a failed message')

    def testError(self):
        raise RuntimeError('Test error!')

    def testAssesrtTrue(self):
        self.assertTrue(True)

    def testAssertFalse(self):
        self.assertFalse(False)
```

我们可以使用`-v`选项来显示更详细的输出：

```py
$ python3 -m unittest -v chapter13_5_more_unittest.py
testAssertFalse (chapter13_5_more_unittest.Output) ... ok
testAssesrtTrue (chapter13_5_more_unittest.Output) ... ok
testError (chapter13_5_more_unittest.Output) ... ERROR
testFail (chapter13_5_more_unittest.Output) ... FAIL
testPass (chapter13_5_more_unittest.Output) ... ok

======================================================================
ERROR: testError (chapter13_5_more_unittest.Output)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "/home/echou/Master_Python_Networking_second_edition/Chapter13/chapter13_5_more_unittest.py", line 14, in testError
 raise RuntimeError('Test error!')
RuntimeError: Test error!

======================================================================
FAIL: testFail (chapter13_5_more_unittest.Output)
----------------------------------------------------------------------
Traceback (most recent call last):
 File "/home/echou/Master_Python_Networking_second_edition/Chapter13/chapter13_5_more_unittest.py", line 11, in testFail
 self.assertFalse(True, 'this is a failed message')
AssertionError: True is not false : this is a failed message

----------------------------------------------------------------------
Ran 5 tests in 0.001s

FAILED (failures=1, errors=1)
```

从 Python 3.3 开始，`unittest`模块默认包含`module`对象库（[`docs.python.org/3/library/unittest.mock.html`](https://docs.python.org/3/library/unittest.mock.html)）。这是一个非常有用的模块，可以对远程资源进行假的 HTTP API 调用，而无需实际进行调用。例如，我们已经看到了使用 NX-API 来检索 NX-OS 版本号的示例。如果我们想运行我们的测试，但没有 NX-OS 设备可用怎么办？我们可以使用`unittest`模拟对象。

在`chapter13_5_more_unittest_mocks.py`中，我们创建了一个简单的类，其中包含一个用于进行 HTTP API 调用并期望 JSON 响应的方法：

```py
# Our class making API Call using requests
class MyClass:
    def fetch_json(self, url):
        response = requests.get(url)
        return response.json()
```

我们还创建了一个模拟两个 URL 调用的函数：

```py
# This method will be used by the mock to replace requests.get
def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

    if args[0] == 'http://url-1.com/test.json':
        return MockResponse({"key1": "value1"}, 200)
    elif args[0] == 'http://url-2.com/test.json':
        return MockResponse({"key2": "value2"}, 200)

    return MockResponse(None, 404)
```

最后，我们在我们的测试用例中对两个 URL 进行 API 调用。然而，我们使用`mock.patch`装饰器来拦截 API 调用：

```py
# Our test case class
class MyClassTestCase(unittest.TestCase):
    # We patch 'requests.get' with our own method. The mock object is passed in to our test case method.
    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_fetch(self, mock_get):
        # Assert requests.get calls
        my_class = MyClass()
        # call to url-1
        json_data = my_class.fetch_json('http://url-1.com/test.json')
        self.assertEqual(json_data, {"key1": "value1"})
        # call to url-2
        json_data = my_class.fetch_json('http://url-2.com/test.json')
        self.assertEqual(json_data, {"key2": "value2"})
        # call to url-3 that we did not mock
        json_data = my_class.fetch_json('http://url-3.com/test.json')
        self.assertIsNone(json_data)

if __name__ == '__main__':
    unittest.main()
```

当我们运行测试时，我们会看到测试通过，而无需实际调用远程端点的 API：

```py
$ python3 -m unittest -v chapter13_5_more_unittest_mocks.py
test_fetch (chapter13_5_more_unittest_mocks.MyClassTestCase) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

有关`unittest`模块的更多信息，Doug Hellmann 的 Python 模块一周（[`pymotw.com/3/unittest/index.html#module-unittest`](https://pymotw.com/3/unittest/index.html#module-unittest)）是一个关于`unittest`模块的简短而精确的示例的绝佳来源。一如既往，Python 文档也是一个很好的信息来源：[`docs.python.org/3/library/unittest.html`](https://docs.python.org/3/library/unittest.html)。

# 更多关于 Python 测试的内容

除了内置的`unittest`库之外，社区中还有许多其他 Python 测试框架。Pytest 是另一个强大的 Python 测试框架，值得一看。`pytest`可以用于各种类型和级别的软件测试。它可以被开发人员、QA 工程师、练习测试驱动开发的个人和开源项目使用。许多大型开源项目已经从`unittest`或`nose`转换到`pytest`，包括 Mozilla 和 Dropbox。`pytest`的主要吸引力在于第三方插件模型、简单的装置模型和断言重写。

如果您想了解更多关于`pytest`框架的信息，我强烈推荐 Brian Okken 的*Python Testing with PyTest*（ISBN 978-1-68050-240-4）。另一个很好的来源是`pytest`文档：[`docs.pytest.org/en/latest/`](https://docs.pytest.org/en/latest/)。

`pytest`是命令行驱动的；它可以自动找到我们编写的测试并运行它们：

```py
$ sudo pip install pytest
$ sudo pip3 install pytest
$ python3
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pytest
>>> pytest.__version__
'3.6.3'
```

让我们看一些使用`pytest`的例子。

# pytest 示例

第一个`pytest`示例将是对两个值的简单断言：

```py
$ cat chapter13_6_pytest_1.py
#!/usr/bin/env python3

def test_passing():
    assert(1, 2, 3) == (1, 2, 3)

def test_failing():
    assert(1, 2, 3) == (3, 2, 1)
```

当您使用`-v`选项运行时，`pytest`将为我们提供一个相当强大的失败原因的答案：

```py
$ pytest -v chapter13_6_pytest_1.py
============================== test session starts ===============================
platform linux -- Python 3.5.2, pytest-3.6.3, py-1.5.4, pluggy-0.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/echou/Master_Python_Networking_second_edition/Chapter13, inifile:
collected 2 items

chapter13_6_pytest_1.py::test_passing PASSED [ 50%]
chapter13_6_pytest_1.py::test_failing FAILED [100%]

==================================== FAILURES ====================================
__________________________________ test_failing __________________________________

 def test_failing():
> assert(1, 2, 3) == (3, 2, 1)
E assert (1, 2, 3) == (3, 2, 1)
E At index 0 diff: 1 != 3
E Full diff:
E - (1, 2, 3)
E ? ^ ^
E + (3, 2, 1)
E ? ^ ^

chapter13_6_pytest_1.py:7: AssertionError
======================= 1 failed, 1 passed in 0.03 seconds =======================
```

在第二个示例中，我们将创建一个`router`对象。`router`对象将使用一些值进行初始化，其中一些值为`None`，另一些值为默认值。我们将使用`pytest`来测试一个具有默认值的实例和一个没有默认值的实例：

```py
$ cat chapter13_7_pytest_2.py
#!/usr/bin/env python3

class router(object):
    def __init__(self, hostname=None, os=None, device_type='cisco_ios'):
        self.hostname = hostname
        self.os = os
        self.device_type = device_type
        self.interfaces = 24

def test_defaults():
    r1 = router()
    assert r1.hostname == None
    assert r1.os == None
    assert r1.device_type == 'cisco_ios'
    assert r1.interfaces == 24

def test_non_defaults():
    r2 = router(hostname='lax-r2', os='nxos', device_type='cisco_nxos')
    assert r2.hostname == 'lax-r2'
    assert r2.os == 'nxos'
    assert r2.device_type == 'cisco_nxos'
    assert r2.interfaces == 24
```

当我们运行测试时，我们将看到实例是否准确地应用了默认值：

```py
$ pytest chapter13_7_pytest_2.py
============================== test session starts ===============================
platform linux -- Python 3.5.2, pytest-3.6.3, py-1.5.4, pluggy-0.6.0
rootdir: /home/echou/Master_Python_Networking_second_edition/Chapter13, inifile:
collected 2 items

chapter13_7_pytest_2.py .. [100%]

============================ 2 passed in 0.04 seconds ============================
```

如果我们要用`pytest`替换之前的`unittest`示例，在`chapter13_8_pytest_3.py`中，我们将有一个简单的测试用例：

```py
# pytest test case
def test_version():
    assert devices['nx-osv-1']['os'] == nxos_version
```

然后我们使用`pytest`命令行运行测试：

```py
$ pytest chapter13_8_pytest_3.py
============================== test session starts ===============================
platform linux -- Python 3.5.2, pytest-3.6.3, py-1.5.4, pluggy-0.6.0
rootdir: /home/echou/Master_Python_Networking_second_edition/Chapter13, inifile:
collected 1 item

chapter13_8_pytest_3.py . [100%]

============================ 1 passed in 0.19 seconds ============================
```

如果我们为自己编写测试，我们可以自由选择任何模块。在`unittest`和`pytest`之间，我发现`pytest`是一个更直观的工具。然而，由于`unittest`包含在标准库中，许多团队可能更喜欢使用`unittest`模块进行测试。

# 编写网络测试

到目前为止，我们大多数时间都在为我们的 Python 代码编写测试。我们使用了`unittest`和`pytest`库来断言`True/False`和`equal/Non-equal`值。我们还能够编写模拟来拦截我们的 API 调用，当我们没有实际的 API 可用设备但仍想运行我们的测试时。

几年前，Matt Oswalt 宣布了**Testing On Demand: Distributed**（**ToDD**）验证工具，用于测试网络连接和分布式容量。这是一个旨在测试网络连通性和分布式容量的开源框架。您可以在其 GitHub 页面上找到有关该项目的更多信息：[`github.com/toddproject/todd`](https://github.com/toddproject/todd)。Oswalt 还在 Packet Pushers Priority Queue 81 上谈到了该项目，标题是 Network Testing with ToDD：[`packetpushers.net/podcast/podcasts/pq-show-81-network-testing-todd/`](https://packetpushers.net/podcast/podcasts/pq-show-81-network-testing-todd/)。

在这一部分，让我们看看如何编写与网络世界相关的测试。在网络监控和测试方面，商业产品并不少见。多年来，我接触过许多这样的产品。然而，在这一部分，我更喜欢使用简单的开源工具进行测试。

# 可达性测试

通常，故障排除的第一步是进行小范围的可达性测试。对于网络工程师来说，`ping`是我们在进行网络可达性测试时的好朋友。这是一种通过向目的地发送一个小数据包来测试 IP 网络上主机可达性的方法。

我们可以通过`OS`模块或`subprocess`模块自动化`ping`测试：

```py
>>> import os
>>> host_list = ['www.cisco.com', 'www.google.com']
>>> for host in host_list:
...     os.system('ping -c 1 ' + host)
...
PING e2867.dsca.akamaiedge.net (69.192.206.157) 56(84) bytes of data.
64 bytes from a69-192-206-157.deploy.static.akamaitechnologies.com (69.192.206.157): icmp_seq=1 ttl=54 time=14.7 ms

--- e2867.dsca.akamaiedge.net ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 14.781/14.781/14.781/0.000 ms
0
PING www.google.com (172.217.3.196) 56(84) bytes of data.
64 bytes from sea15s12-in-f196.1e100.net (172.217.3.196): icmp_seq=1 ttl=54 time=12.8 ms

--- www.google.com ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 12.809/12.809/12.809/0.000 ms
0
>>>
```

`subprocess`模块提供了捕获输出的额外好处：

```py
>>> import subprocess
>>> for host in host_list:
...     print('host: ' + host)
...     p = subprocess.Popen(['ping', '-c', '1', host], stdout=subprocess.PIPE)
...     print(p.communicate())
...
host: www.cisco.com
(b'PING e2867.dsca.akamaiedge.net (69.192.206.157) 56(84) bytes of data.\n64 bytes from a69-192-206-157.deploy.static.akamaitechnologies.com (69.192.206.157): icmp_seq=1 ttl=54 time=14.3 ms\n\n--- e2867.dsca.akamaiedge.net ping statistics ---\n1 packets transmitted, 1 received, 0% packet loss, time 0ms\nrtt min/avg/max/mdev = 14.317/14.317/14.317/0.000 ms\n', None)
host: www.google.com
(b'PING www.google.com (216.58.193.68) 56(84) bytes of data.\n64 bytes from sea15s07-in-f68.1e100.net (216.58.193.68): icmp_seq=1 ttl=54 time=15.6 ms\n\n--- www.google.com ping statistics ---\n1 packets transmitted, 1 received, 0% packet loss, time 0ms\nrtt min/avg/max/mdev = 15.695/15.695/15.695/0.000 ms\n', None)
>>>
```

这两个模块在许多情况下都非常有用。我们可以通过`OS`或`subprocess`模块执行在 Linux 和 Unix 环境中可以执行的任何命令。

# 网络延迟测试

网络延迟的话题有时可能是主观的。作为网络工程师，我们经常面对用户说网络很慢的情况。然而，慢是一个非常主观的词。如果我们能构建测试，将主观的词转化为客观的值，那将非常有帮助。我们应该始终如一地这样做，这样我们就可以比较一系列数据的值。

这有时可能很难做到，因为网络是无状态的设计。成功发送一个数据包并不保证下一个数据包也会成功。多年来我见过的最好的方法就是经常使用 ping 跨多个主机，并记录数据，进行 ping-mesh 图。我们可以利用前面示例中使用的相同工具，捕获返回结果的时间，并保留记录：

```py
$ cat chapter13_10_ping.py
#!/usr/bin/env python3

import subprocess

host_list = ['www.cisco.com', 'www.google.com']

ping_time = []

for host in host_list:
    p = subprocess.Popen(['ping', '-c', '1', host], stdout=subprocess.PIPE)
    result = p.communicate()[0]
    host = result.split()[1]
    time = result.split()[14]
    ping_time.append((host, time))

print(ping_time)
```

在这种情况下，结果被保存在一个元组中，并放入一个列表中：

```py
$ python3 chapter13_10_ping.py
[(b'e2867.dsca.akamaiedge.net', b'time=13.8'), (b'www.google.com', b'time=14.8')]
```

这绝不是完美的，只是监控和故障排除的起点。然而，在没有其他工具的情况下，这提供了一些客观值的基线。

# 安全测试

我们已经在第六章中看到了我认为是用 Python 进行网络安全测试的最佳工具，即 Scapy。有很多开源安全工具，但没有一个能提供构建数据包的灵活性。

网络安全测试的另一个很好的工具是`hping3`（[`www.hping.org/`](http://www.hping.org/)）。它提供了一种简单的方法来一次生成大量的数据包。例如，您可以使用以下一行命令生成 TCP Syn flood：

```py
# DON'T DO THIS IN PRODUCTION #
echou@ubuntu:/var/log$ sudo hping3 -S -p 80 --flood 192.168.1.202
HPING 192.168.1.202 (eth0 192.168.1.202): S set, 40 headers + 0 data bytes
hping in flood mode, no replies will be shown
^C
--- 192.168.1.202 hping statistic ---
2281304 packets transmitted, 0 packets received, 100% packet loss
round-trip min/avg/max = 0.0/0.0/0.0 ms
echou@ubuntu:/var/log$
```

同样，由于这是一个命令行工具，我们可以使用`subprocess`模块来自动化任何我们想要的`hping3`测试。

# 交易测试

网络是基础设施的重要组成部分，但它只是其中的一部分。用户关心的通常是运行在网络之上的服务。如果用户试图观看 YouTube 视频或收听播客却无法做到，他们会认为服务出现了故障。我们可能知道这不是网络传输的问题，但这并不能让用户感到安慰。

因此，我们应该实施尽可能接近用户体验的测试。在 YouTube 视频的例子中，我们可能无法 100%复制 YouTube 体验（除非你是 Google 的一部分），但我们可以尽可能接近网络边缘实现第七层服务。然后，我们可以模拟来自客户端的交易作为事务性测试。

Python 的`HTTP`标准库模块是我在需要快速测试第七层可达性的 Web 服务时经常使用的模块：

```py
# Python 2
$ python -m SimpleHTTPServer 8080
Serving HTTP on 0.0.0.0 port 8080 ...
127.0.0.1 - - [25/Jul/2018 10:14:39] "GET / HTTP/1.1" 200 -

# Python 3 
$ python3 -m http.server 8080
Serving HTTP on 0.0.0.0 port 8080 ...
127.0.0.1 - - [25/Jul/2018 10:15:23] "GET / HTTP/1.1" 200 -
```

如果我们可以模拟预期服务的完整交易，那就更好了。但是标准库中的 Python 简单`HTTP`服务器模块始终是运行一些临时 Web 服务测试的好选择。

# 网络配置测试

在我看来，对网络配置的最佳测试是使用标准化模板生成配置并经常备份生产配置。我们已经看到了如何使用 Jinja2 模板来根据设备类型或角色标准化我们的配置。这将消除许多由人为错误引起的错误，比如复制和粘贴。

一旦配置生成，我们可以针对我们在将配置推送到生产设备之前期望的已知特征编写测试。例如，在网络的所有部分中，当涉及到环回 IP 时，IP 地址不应重叠，因此我们可以编写一个测试来查看新配置是否包含在我们的设备中唯一的环回 IP。

# 测试 Ansible

在我使用 Ansible 的时间里，我记不起来使用类似`unittest`的工具来测试 Playbook。在大多数情况下，Playbooks 使用了模块，这些模块是由模块开发人员测试过的。

Ansible 为他们的模块库提供单元测试。目前，Ansible 中的单元测试是从 Python 驱动测试的唯一方式。今天运行的单元测试可以在`/test/units` ([`github.com/ansible/ansible/tree/devel/test/units`](https://github.com/ansible/ansible/tree/devel/test/units))下找到。

可以在以下文档中找到 Ansible 测试策略：

+   **测试 Ansible**: [`docs.ansible.com/ansible/2.5/dev_guide/testing.html`](https://docs.ansible.com/ansible/2.5/dev_guide/testing.html)

+   **单元测试**: [`docs.ansible.com/ansible/2.5/dev_guide/testing_units.html`](https://docs.ansible.com/ansible/2.5/dev_guide/testing_units.html)

+   **单元测试 Ansible 模块**: [`docs.ansible.com/ansible/2.5/dev_guide/testing_units_modules.html`](https://docs.ansible.com/ansible/2.5/dev_guide/testing_units_modules.html)

Ansible 测试框架中有一个有趣的工具是**molecule** ([`pypi.org/project/molecule/2.16.0/`](https://pypi.org/project/molecule/2.16.0/))。它旨在帮助开发和测试 Ansible 角色。Molecule 支持使用多个实例、操作系统和发行版进行测试。我没有使用过这个工具，但如果我想对我的 Ansible 角色进行更多测试，这是我会开始的地方。

# Jenkins 中的 Pytest

**持续集成**（**CI**）系统，如 Jenkins，经常用于在每次代码提交后启动测试。这是使用 CI 系统的主要好处之一。想象一下，有一个隐形的工程师一直在观察网络中的任何变化；在检测到变化后，工程师将忠实地测试一堆功能，以确保没有任何故障。谁不想要这样的工程师呢？

让我们看一个将`pytest`集成到 Jenkins 任务中的例子。

# Jenkins 集成

在我们将测试用例插入到我们的持续集成之前，让我们安装一些可以帮助我们可视化操作的插件。我们将安装的两个插件是 build-name-setter 和 Test Result Analyzer：

![](img/cd23c244-3916-4292-bcf2-4a708cd95195.png)

Jenkins 插件安装

我们将运行的测试将会连接到 NXOS 设备并检索操作系统版本号。这将确保我们可以通过 API 访问 Nexus 设备。完整的脚本内容可以在`chapter13_9_pytest_4.py`中阅读，相关的`pytest`部分和结果如下：

```py
def test_transaction():
     assert nxos_version != False

## Test Output
$ pytest chapter13_9_pytest_4.py
============================== test session starts ===============================
platform linux -- Python 3.5.2, pytest-3.6.3, py-1.5.4, pluggy-0.6.0
rootdir: /home/echou/Chapter13, inifile:
collected 1 item

chapter13_9_pytest_4.py . [100%]

============================ 1 passed in 0.13 seconds ============================
```

我们将使用`--junit-xml=results.xml`选项来生成 Jenkins 需要的文件：

```py
$ pytest --junit-xml=results.xml chapter13_9_pytest_4.py
$ cat results.xml
<?xml version="1.0" encoding="utf-8"?><testsuite errors="0" failures="0" name="pytest" skips="0" tests="1" time="0.134"><testcase classname="chapter13_9_pytest_4" file="chapter13_9_pytest_4.py" line="25" name="test_transaction" time="0.0009090900421142578"></testcase></testsuite>
```

下一步将是将此脚本检入 GitHub 存储库。我倾向于将测试放在其目录下。因此，我创建了一个`/test`目录，并将测试文件放在那里：

![](img/974185c8-eb51-4ed4-86d5-5470266069bd.png)

项目存储库

我们将创建一个名为`chapter13_example1`的新项目：

![](img/5456017e-169a-4ecd-af1c-350097468fc4.png)

第十三章示例 1

我们可以复制上一个任务，这样我们就不需要重复所有步骤：

![](img/1705abee-1ac9-4b3c-b55b-4cffa1080c59.png)

从第十二章示例 2 复制任务

在执行 shell 部分，我们将添加`pytest`步骤：

![](img/10883f4c-c1a0-4230-8588-97ed8a5f2b02.png)

项目执行 shell

我们将添加一个发布 JUnit 测试结果报告的构建后步骤：

![](img/628c4028-6460-47d6-9802-16b8133f00bb.png)

构建后步骤

我们将指定`results.xml`文件作为 JUnit 结果文件：

![](img/5fc9e8fd-82ee-4259-9692-6b27b45c5cd8.png)

测试报告 XML 位置

运行构建几次后，我们将能够看到测试结果分析器图表：

![](img/91ab28c8-3cf3-475c-a267-5ebf0147fef7.png)

测试结果分析器

测试结果也可以在项目主页上看到。让我们通过关闭 Nexus 设备的管理接口来引入一个测试失败。如果有测试失败，我们将能够立即在项目仪表板上的测试结果趋势图上看到它：

![](img/b17f8844-1578-47c8-b6ee-32dce3b3d765.png)

测试结果趋势

这是一个简单但完整的例子。我们可以将测试集成到 Jenkins 中的许多方式。

# 总结

在本章中，我们看了测试驱动开发以及如何将其应用于网络工程。我们从 TDD 的概述开始；然后我们看了使用`unittest`和`pytest` Python 模块的示例。Python 和简单的 Linux 命令行工具可以用来构建各种测试，包括网络可达性、配置和安全性。

我们还看了如何在 Jenkins 中利用测试，这是一个持续集成工具。通过将测试集成到我们的 CI 工具中，我们可以更加确信我们的更改是合理的。至少，我们希望在用户之前捕捉到任何错误。

简而言之，如果没有经过测试，就不能信任。我们网络中的一切都应尽可能地进行程序化测试。与许多软件概念一样，测试驱动开发是一个永无止境的服务轮。我们努力实现尽可能多的测试覆盖率，但即使在 100%的测试覆盖率下，我们总是可以找到新的方法和测试用例来实现。这在网络中尤其如此，网络通常是互联网，而互联网的 100%测试覆盖是不可能的。
