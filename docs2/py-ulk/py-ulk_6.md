# 第六章。测试驱动开发

在本章中，我们将讨论一些在测试期间要应用的良好概念。首先，我们将看看我们如何可以轻松地创建模拟或存根来测试系统中不存在的功能。然后，我们将介绍如何编写参数化的测试用例。自定义测试运行器对于为特定项目编写测试实用程序非常有帮助。然后，我们将介绍如何测试线程化应用程序，并利用并发执行来减少测试套件运行的总时间。我们将涵盖以下主题：

+   测试用的 Mock

+   参数化

+   创建自定义测试运行器

+   测试线程化应用程序

+   并行运行测试用例

# 测试用的 Mock

**关键 1：模拟你所没有的。**

当我们使用测试驱动开发时，我们必须为依赖于尚未编写或执行时间很长的其他组件的组件编写测试用例。在我们创建模拟和存根之前，这几乎是不可行的。在这种情况下，存根或模拟非常有用。我们使用一个假对象而不是真实对象来编写测试用例。如果我们使用语言提供的工具，这可以变得非常简单。例如，在以下代码中，我们只有工作类接口，没有其实施。我们想测试`assign_if_free`函数。

我们不是自己编写任何存根，而是使用`create_autospec`函数从 Worker 抽象类的定义中创建一个模拟对象。我们还为检查工作是否忙碌的函数调用设置了返回值：

```py
import six
import unittest
import sys
import abc
if sys.version_info[0:2] >= (3, 3):
    from unittest.mock import Mock, create_autospec
else:
    from mock import Mock, create_autospec
if six.PY2:
    import thread
else:
    import _thread as thread

class IWorker(six.with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def execute(self, *args):
        """ execute an api task """
        pass

    @abc.abstractmethod
    def is_busy(self):
        pass

    @abc.abstractmethod
    def serve_api(self,):
        """register for api hit"""
        pass

class Worker(IWorker):
    def __init__(self,):
        self.__running = False

    def execute(self,*args):
        self.__running = True
        th = thread.start_new_thread(lambda x:time.sleep(5))
        th.join()
        self.__running = False

    def is_busy(self):
        return self.__running == True

def assign_if_free(worker, task):
    if not worker.is_busy():
        worker.execute(task)
        return True
    else:
        return False

class TestWorkerReporting(unittest.TestCase):

    def test_worker_busy(self,):
        mworker = create_autospec(IWorker)
        mworker.configure_mock(**{'is_busy.return_value':True})
        self.assertFalse(assign_if_free(mworker, {}))

    def test_worker_free(self,):
        mworker = create_autospec(IWorker)
        mworker.configure_mock(**{'is_busy.return_value':False})
        self.assertTrue(assign_if_free(mworker, {}))

if __name__ == '__main__':
    unittest.main()
```

要设置返回值，我们也可以使用函数来返回条件响应，如下所示：

```py
>>> STATE = False
>>> worker = create_autospec(Worker,)
>>> worker.configure_mock(**{'is_busy.side_effect':lambda : True if not STATE else False})
>>> worker.is_busy()
True
>>> STATE=True
>>> worker.is_busy()
False
```

我们还可以使用 mock 的`side_effect`属性来设置方法抛出异常，如下所示：

```py
>>> worker.configure_mock(**{'execute.side_effect':Exception('timeout for execution')})
>>> 
>>> worker.execute()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.4/unittest/mock.py", line 896, in __call__
    return _mock_self._mock_call(*args, **kwargs)
  File "/usr/lib/python3.4/unittest/mock.py", line 952, in _mock_call
    raise effect
Exception: timeout for execution
```

另一个用途是检查方法是否被调用以及调用时使用的参数，如下所示：

```py
>>> worker = create_autospec(IWorker,)
>>> worker.configure_mock(**{'is_busy.return_value':True})
>>> assign_if_free(worker,{})
False
>>> worker.execute.called
False
>>> worker.configure_mock(**{'is_busy.return_value':False})
>>> assign_if_free(worker,{})
True
>>> worker.execute.called
True
```

# 参数化

**关键 2：可管理的测试输入。**

对于我们必须测试同一功能或转换的多种输入的测试，我们必须编写测试用例来覆盖不同的输入。在这里，我们可以使用参数化。这样，我们可以用不同的输入调用相同的测试用例，从而减少与它相关的时间和错误。较新的 Python 版本 3.4 或更高版本包括一个非常有用的方法，`unittest.TestCase`中的`subTest`，这使得添加参数化测试变得非常容易。在测试输出中，请注意参数化值也是可用的：

```py
import unittest
from itertools import combinations
from functools import wraps

def convert(alpha):
    return ','.join([str(ord(i)-96) for i in alpha])

class TestOne(unittest.TestCase):

    def test_system(self,):
        cases = [("aa","1,1"),("bc","2,3"),("jk","4,5"),("xy","24,26")]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(convert(case[0]),case[1])

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

这将给出以下输出：

```py
(py3)arun@olappy:~/codes/projects/pybook/book/ch6$ python parametrized.py
test_system (__main__.TestOne) ... 
======================================================================
FAIL: test_system (__main__.TestOne) (case=('jk', '4,5'))
----------------------------------------------------------------------
Traceback (most recent call last):
  File "parametrized.py", line 14, in test_system
    self.assertEqual(convert(case[0]),case[1])
AssertionError: '10,11' != '4,5'
- 10,11
+ 4,5

======================================================================
FAIL: test_system (__main__.TestOne) (case=('xy', '24,26'))
----------------------------------------------------------------------
Traceback (most recent call last):
  File "parametrized.py", line 14, in test_system
    self.assertEqual(convert(case[0]),case[1])
AssertionError: '24,25' != '24,26'
- 24,25
?     ^
+ 24,26
?     ^

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (failures=2)
```

这也意味着，如果我们需要运行所有输入组合的*柯里化*测试，那么这可以非常容易地完成。我们必须编写一个返回柯里化参数的函数，然后我们可以使用`subTest`来运行带有柯里化参数的迷你测试。这样，向团队中的新成员解释如何用最少的语言术语编写测试用例就变得非常容易，如下所示：

```py
import unittest
from itertools import combinations
from functools import wraps

def entry(number,alpha):
    if 0 < number < 4 and 'a' <= alpha <= 'c':
        return True
    else:
        return False

def curry(*args):
    if not args:
        return []
    else:
        cases = [ [i,] for i in args[0]]
        if len(args)>1:
            for i in range(1,len(args)):
                ncases = []
                for j in args[i]:
                    for case in cases:
                        ncases.append(case+[j,])
                cases = ncases
        return cases

class TestOne(unittest.TestCase):

    def test_sample2(self,):
         case1 = [1,2]
         case2 = ['a','b','d']
         for case in curry(case1,case2):
             with self.subTest(case=case):
                 self.assertTrue(entry(*case), "not equal")

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

这将给出以下输出：

```py
(py3)arun@olappy:~/codes/projects/pybook/book/ch6$ python parametrized_curry.py 
test_sample2 (__main__.TestOne) ... 
======================================================================
FAIL: test_sample2 (__main__.TestOne) (case=[1, 'd'])
----------------------------------------------------------------------
Traceback (most recent call last):
  File "parametrized_curry.py", line 33, in test_sample2
    self.assertTrue(entry(*case), "not equal")
AssertionError: False is not true : not equal

======================================================================
FAIL: test_sample2 (__main__.TestOne) (case=[2, 'd'])
----------------------------------------------------------------------
Traceback (most recent call last):
  File "parametrized_curry.py", line 33, in test_sample2
    self.assertTrue(entry(*case), "not equal")
AssertionError: False is not true : not equal

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (failures=2)
```

但是，这仅适用于 Python 的新版本。对于旧版本，我们可以利用语言的动态性执行类似的工作。我们可以自己实现这个功能，如下面的代码片段所示。我们使用装饰器将参数化值粘接到测试用例上，然后在`metaclass`中创建一个新的包装函数，该函数使用所需的参数调用原始函数：

```py
from functools import wraps
import six
import unittest
from datetime import datetime, timedelta

class parameterize(object):
    """decorator to pass parameters to function 
    we need this to attach parameterize 
    arguments on to the function, and it attaches
    __parameterize_this__ attribute which tells 
    metaclass that we have to work on this attribute
    """
    def __init__(self,names,cases):
        """ save parameters """
        self.names = names
        self.cases = cases

    def __call__(self,func):
        """ attach parameters to same func """
        func.__parameterize_this__ = (self.names, self.cases)
        return func

class ParameterizeMeta(type):

    def __new__(metaname, classname, baseclasses, attrs):
        # iterate over attribute and find out which one have __parameterize_this__ set
        for attrname, attrobject in six.iteritems(attrs.copy()):
            if attrname.startswith('test_'):
                pmo = getattr(attrobject,'__parameterize_this__',None)
                if pmo:
                    params,values = pmo
                    for case in values:
                        name = attrname + '_'+'_'.join([str(item) for item in case])
                        def func(selfobj, testcase=attrobject,casepass=dict(zip(params,case))):
                            return testcase(selfobj, **casepass)
                        attrs[name] = func
                        func.__name__ = name
                    del attrs[attrname]
        return type.__new__(metaname, classname, baseclasses, attrs)

class MyProjectTestCase(six.with_metaclass(ParameterizeMeta,unittest.TestCase)):
    pass

class TestCase(MyProjectTestCase):

    @parameterize(names=("input","output"),
                 cases=[(1,2),(2,4),(3,6)])
    def test_sample(self,input,output):
        self.assertEqual(input*2,output)

    @parameterize(names=("in1","in2","output","shouldpass"),
                  cases=[(1,2,3,True),
                         (2,3,6,False)]
                 )
    def test_sample2(self,in1,in2,output,shouldpass):
        res = in1 + in2 == output
        self.assertEqual(res,shouldpass)

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

上述代码的输出如下：

```py
test_sample2_1_2_3_True (__main__.TestCase) ... ok
test_sample2_2_3_6_False (__main__.TestCase) ... ok
test_sample_1_2 (__main__.TestCase) ... ok
test_sample_2_4 (__main__.TestCase) ... ok
test_sample_3_6 (__main__.TestCase) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK
```

# 创建自定义测试运行器

**关键 3：从测试系统中获取信息。**

单元测试的流程如下：`unittest.TestProgram`在`unittest.main`中是运行一切的主要对象。测试用例通过测试发现或通过命令行传递的模块加载来收集。如果没有指定给主函数的测试运行器，则默认使用`TextTestRunner`。测试套件传递给运行器的`run`函数，以返回一个`TestResult`对象。

自定义测试运行器是获取特定输出格式信息、管理运行顺序、将结果存储在数据库中或为项目需求创建新功能的好方法。

现在我们来看一个例子，创建测试用例的 XML 输出，你可能需要这样的东西来与只能处理某些 XML 格式的持续集成系统集成。如下面的代码片段所示，`XMLTestResult`是提供 XML 格式测试结果的类。`TsRunner`类测试运行器然后将相同的信息放在`stdout`流上。我们还添加了测试用例所需的时间。`XMLify`类以 XML 格式向测试`TsRunner`运行器类发送信息。`XMLRunner`类将此信息以 XML 格式放在`stdout`上，如下所示：

```py
""" custom test system classes """

import unittest
import sys
import time
from xml.etree import ElementTree as ET
from unittest import TextTestRunner

class XMLTestResult(unittest.TestResult):
    """converts test results to xml format"""

    def __init__(self, *args,**kwargs):#runner):
        unittest.TestResult.__init__(self,*args,**kwargs )
        self.xmldoc = ET.fromstring('<testsuite />')

    def startTest(self, test):
        """called before each test case run"""
        test.starttime = time.time()
        test.testxml = ET.SubElement(self.xmldoc,
                                     'testcase',
                                     attrib={'name': test._testMethodName,
                                             'classname': test.__class__.__name__,
                                             'module': test.__module__})

    def stopTest(self, test):
        """called after each test case"""
        et = time.time()
        time_elapsed = et - test.starttime
        test.testxml.attrib['time'] = str(time_elapsed)

    def addSuccess(self, test):
        """
        called on successful test case run
        """
        test.testxml.attrib['result'] = 'ok'

    def addError(self, test, err):
        """
        called on errors in test case
        :param test: test case
        :param err: error info
        """
        unittest.TestResult.addError(self, test, err)
        test.testxml.attrib['result'] = 'error'
        el = ET.SubElement(test.testxml, 'error', )
        el.text = self._exc_info_to_string(err, test)

    def addFailure(self, test, err):
        """
        called on failures in test cases.
        :param test: test case
        :param err: error info
        """
        unittest.TestResult.addFailure(self, test, err)
        test.testxml.attrib['result'] = 'failure'
        el = ET.SubElement(test.testxml, 'failure', )
        el.text = self._exc_info_to_string(err, test)

    def addSkip(self, test, reason):
        # self.skipped.append(test)
        test.testxml.attrib['result'] = 'skipped'
        el = ET.SubElement(test.testxml, 'skipped', )
        el.attrib['message'] = reason

class XMLRunner(object):
    """ custom runner class"""

    def __init__(self, *args,**kwargs):
        self.resultclass = XMLTestResult

    def run(self, test):
        """ run given test case or suite"""
        result = self.resultclass()
        st = time.time()
        test(result)
        time_taken = float(time.time() - st)
        result.xmldoc.attrib['time'] = str(time_taken)

        ET.dump(result.xmldoc)
        #tree = ET.ElementTree(result.xmldoc)
        #tree.write("testm.xml", encoding='utf-8')
        return result
```

假设我们在测试用例上使用此`XMLRunner`，如下面的代码所示：

```py
import unittest

class TestAll(unittest.TestCase):
    def test_ok(self):
        assert 1 == 1

    def test_notok(self):
        assert 1 >= 3

    @unittest.skip("not needed")
    def test_skipped(self):
        assert 2 == 4

class TestAll2(unittest.TestCase):
    def test_ok2(self):
        raise IndexError
        assert 1 == 1

    def test_notok2(self):
        assert 1 == 3

    @unittest.skip("not needed")
    def test_skipped2(self):
        assert 2 == 4

if __name__ == '__main__':
    from ts2 import XMLRunner
unittest.main(verbosity=2, testRunner=XMLRunner)
```

我们将得到以下输出：

```py
<testsuite time="0.0005891323089599609"><testcase classname="TestAll" module="__main__" name="test_notok" result="failure" time="0.0002377033233642578"><failure>Traceback (most recent call last):
  File "test_cases.py", line 8, in test_notok
    assert 1 &gt;= 3
AssertionError
</failure></testcase><testcase classname="TestAll" module="__main__" name="test_ok" result="ok" time="2.6464462280273438e-05" /><testcase classname="TestAll" module="__main__" name="test_skipped" result="skipped" time="9.059906005859375e-06"><skipped message="not needed" /></testcase><testcase classname="TestAll2" module="__main__" name="test_notok2" result="failure" time="9.34600830078125e-05"><failure>Traceback (most recent call last):
  File "test_cases.py", line 20, in test_notok2
    assert 1 == 3
AssertionError
</failure></testcase><testcase classname="TestAll2" module="__main__" name="test_ok2" result="error" time="8.440017700195312e-05"><error>Traceback (most recent call last):
  File "test_cases.py", line 16, in test_ok2
    raise IndexError
IndexError
</error></testcase><testcase classname="TestAll2" module="__main__" name="test_skipped2" result="skipped" time="7.867813110351562e-06"><skipped message="not needed" /></testcase></testsuite>
```

# 测试线程应用程序

**关键 4：使线程应用程序测试类似于非线程化测试。**

我在测试线程应用程序方面的经验是执行以下操作：

+   尽可能使线程应用程序在测试中尽可能非线程化。我的意思是，在一个代码段中，将非线程化的逻辑分组。不要尝试用线程逻辑测试业务逻辑。尽量将它们分开。

+   尽可能地使用最少的全局状态。函数应该传递所需工作的对象。

+   尝试创建任务队列以同步它们。而不是自己创建生产者消费者链，首先尝试使用队列。

+   还要注意，sleep 语句会使测试用例运行得更慢。如果你在代码中添加了超过 20 个 sleep，整个测试套件开始变慢。线程代码应该通过事件和通知传递信息，而不是通过 while 循环检查某些条件。

Python 2 中的`_thread`模块和 Python 3 中的`_thread`模块非常有用，因为你可以以线程的形式启动函数，如下所示：

```py
>>> def foo(waittime):
...     time.sleep(waittime)
...     print("done")
>>> thread.start_new_thread(foo,(3,))
140360468600576
>> done
```

# 并行运行测试用例

**关键 5：加快测试套件执行速度**

当我们在项目中积累了大量测试用例时，执行所有测试用例需要花费很多时间。我们必须使测试并行运行以减少整体所需的时间。在这种情况下，`py.test`测试框架在简化并行运行测试的能力方面做得非常出色。为了使这成为可能，我们首先需要安装`py.test`库，然后使用其运行器来运行测试用例。`py.test`库有一个`xdist`插件，它增加了并行运行测试的能力，如下所示：

```py
(py35) [ ch6 ] $ py.test -n 3 test_system.py
========================================== test session starts ===========================================
platform linux -- Python 3.5.0, pytest-2.8.2, py-1.4.30, pluggy-0.3.1
rootdir: /home/arun/codes/workspace/pybook/ch6, inifile: 
plugins: xdist-1.13.1
gw0 [5] / gw1 [5] / gw2 [5]
scheduling tests via LoadScheduling
s...F
================================================ FAILURES ================================================
___________________________________________ TestApi.test_api2 ____________________________________________
[gw0] linux -- Python 3.5.0 /home/arun/.pyenv/versions/py35/bin/python3.5
self = <test_system.TestApi testMethod=test_api2>

    def test_api2(self,):
        """api2
            simple test1"""
        for i in range(7):
            with self.subTest(i=i):
>               self.assertLess(i, 4, "not less")
E               AssertionError: 4 not less than 4 : not less

test_system.py:40: AssertionError
============================= 1 failed, 3 passed, 1 skipped in 0.42 seconds ==============================
```

如果你想深入了解这个主题，可以参考[`pypi.python.org/pypi/pytest-xdist`](https://pypi.python.org/pypi/pytest-xdist)。

# 摘要

在创建稳定的应用程序中，测试非常重要。在本章中，我们讨论了如何模拟对象以创建易于分离关注点的环境来测试不同的组件。参数化对于测试各种转换逻辑非常有用。最重要的经验是尝试创建项目所需的测试实用程序功能。尽量坚持使用`unittest`模块。使用其他库进行并行执行，因为它们也支持`unittest`测试。

在下一章中，我们将介绍 Python 的优化技术。
