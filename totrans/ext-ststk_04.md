# 第四章：将状态包裹在执行模块周围

现在我们已经介绍了执行模块和配置模块，是时候讨论配置管理了。状态模块背后的想法是使用执行模块作为一个机制，将资源带到某种状态：一个软件包处于安装状态，一个服务处于运行状态，一个文件的内容与 Master 上定义的状态相匹配。在本章中，我们将讨论：

+   基本状态模块布局背后的概念

+   决定每个状态要推进多远

+   故障排除状态模块

# 构建状态模块

状态模块比大多数其他类型的模块更有结构，但正如你很快就会看到的，这实际上使它们更容易编写。

## 确定状态

状态模块必须执行一系列操作以完成其工作，并且在这些操作执行过程中，会存储某些数据。让我们从一个伪代码片段开始，并依次解释每个组件：

```py
def __virtual__():
    '''
    Only load if the necesaary modules available in __salt__
    '''
    if 'module.function' in __salt__:
        return True
    return False

def somestate(name):
    '''
    Achieve the desired state

    nane
        The name of the item to achieve statefulness
    '''
    ret = {'name': name,
           'changes': {},
           'result': None,
           'comment': ''}
    if <item is already in the desired state>:
        ret['result'] = True
        ret['comment'] = 'The item is already in the desired state'
        return ret
    if __opts__['test']:
        ret['comment'] = 'The item is not in the desired state'
        return ret
    <attempt to configure the item correctly>
    if <we are able to put the item in the correct state>:
        ret['changes'] = {'desired state': name}
        ret['result'] = True
        ret['comment'] = 'The desired state was successfully achieved'
        return ret
    else:
        ret['result'] = False
        ret['comment'] = 'The desired state failed to be achieved'
        return ret
```

### `__virtual__()` 函数

到现在为止，你已经熟悉这个函数了，但我想在这里再次提到它。因为执行模块旨在执行繁重的工作，所以在尝试使用它们之前确保它们可用是至关重要的。

很有可能你需要在你的状态模块内部跨调用多个函数。通常，你会调用至少一个函数来检查相关项的状态，至少再调用一个来将项带入所需的配置。但如果它们都在同一个执行模块中，你实际上只需要检查其中一个的存在。

假设你将要编写一个使用 `http.query` 执行模块进行查找和更改 Web 资源的状态的函数。这个函数应该始终可用，但为了演示的目的，我们将假设我们需要检查它。编写这个函数的一种方法可以是：

```py
def __virtual__():
    '''
    Check for http.query
    '''
    if 'http.query' in __salt__:
        return True
    return False
```

也有一种更简短的方式来做到这一点：

```py
def __virtual__():
    '''
    Check for http.query
    '''
    return 'http.query' in __salt__
```

### 设置默认值

在处理完 `__virtual__()` 函数之后，我们可以继续讨论状态函数本身。首先，我们在字典中设置一些默认变量。在我们的示例中，以及在大多数状态模块中，这个字典被称为 `ret`。这仅是一种惯例，并不是实际的要求。然而，字典内部的键及其数据类型是硬性要求。这些键包括：

+   `name`（字符串）- 这是传递到状态中的资源的名称。这也被称为状态中的 ID。例如，在以下状态中：

    ```py
    nginx:
      - pkg.installed
    ```

    +   传入的名称将是 `nginx`。

+   `changes`（字典）- 如果状态对 Minion 应用了任何更改，这个字典将包含对已应用的每个更改的条目。例如，如果使用了 `pkg.installed` 来安装 `nginx`，则 `changes` 字典将如下所示：

    ```py
    {'nginx': {'new': '1.8.0-2',  'old': ''}}
    ```

    +   对存储在`changes`中的数据类型没有限制，只要`changes`本身是一个字典。如果进行了更改，则此字典*必须*包含一些内容。

+   `result`（布尔值）- 此字段是三个值之一：`True`、`False`或`None`。如果指定的资源已经处于它应该处于的状态，或者它已经被成功配置到该状态，则此字段将为`True`。如果资源不在正确的状态，但`salt`以`test=True`运行，则此字段设置为`None`。如果资源不在正确的状态，并且 Salt 无法将其置于正确的状态，则此字段设置为`False`。

    +   在执行状态运行，如`state.highstate`时，结果值将影响输出的颜色。状态为`True`但没有`changes`的状态将是绿色。状态为`True`且有`changes`的状态将是蓝色。状态为`None`的状态将是黄色。状态为`False`的状态将是红色。

+   `comment`（字符串）- 此字段完全自由格式：它可以包含任何你想要的注释，或者没有注释。然而，最好有一些注释，即使像“请求的资源已经处于期望状态”这样简短也行。如果结果是`None`或`False`，则`comment`应包含尽可能有帮助的消息，说明为什么资源没有正确配置，以及如何纠正。

    +   我们在示例中使用的默认值几乎适用于任何状态：

        ```py
            ret = {'name': name,
                   'changes': {},
                   'result': None,
                   'comment': ''}
        ```

### 检查真实性

在设置默认值之后，接下来的任务是检查资源，看看它是否处于期望的状态：

```py
    if <item is already in the desired state>:
        ret['result'] = True
        ret['comment'] = 'The item is already in the desired state'
        return ret
```

这可能是一个使用执行模块中的单个函数进行的快速检查，或者可能包含需要跨调用几个函数的更多逻辑。不要在这里添加任何不必要的代码来检查资源的状态；记住，所有重负载都应该在执行模块中完成。

如果发现资源配置得当，则将`result`设置为`True`，添加一个有用的`comment`，然后函数`return`s。如果资源没有正确配置，则继续下一部分。

### 检查测试模式

如果代码通过了真实性检查，那么我们可以假设有问题。但在对系统进行任何更改之前，我们需要查看`salt`是否以`test=True`被调用。

```py
    if __opts__['test']:
        ret['comment'] = 'The item is not in the desired state'
        return ret
```

如果是这样，我们为用户设置一个有用的`comment`，然后`return``ret`字典。如果一旦确定`salt`正在`test`模式下运行，还有更多的逻辑发生，那么它应该只用于在注释中为用户提供更多信息。在`test`模式下永远不应该进行任何更改！

### 尝试配置资源

如果我们通过了`test`模式的检查，那么我们知道我们可以尝试更改以正确配置资源：

```py
    <attempt to configure the item correctly>
    if <we are able to put the item in the correct state>:
        ret['changes'] = {'desired state': name}
        ret['result'] = True
        ret['comment'] = 'The desired state was successfully achieved'
        return ret
```

同样，这段代码应该只包含足够的逻辑来正确配置相关资源，并在成功时通知用户。如果更改成功，那么我们更新`changes`字典，添加一个描述如何实现这些`changes`的`comment`，将`result`设置为`True`，然后`return`。

### 通知关于错误

如果我们通过了那段代码，我们现在可以确信出了问题，我们无法修复它：

```py
    else:
        ret['result'] = False
        ret['comment'] = 'The desired state failed to be achieved'
        return ret
```

这是代码中最重要的部分，因为用户交互很可能会被用来修复问题。

可能是 SLS 文件只是写得不好，下一次状态运行会修复它。也可能是状态模块存在需要修复的 bug。或者可能存在一些 Salt 无法控制的其他情况，例如一个暂时不可用的网络服务。注释应该包含尽可能多的信息，以便追踪和修复问题，而不要过多。这也是在`return`之前将结果设置为`False`的时候。

## 示例：检查 HTTP 服务

已经有一个用于联系网络服务的状态：`http.query`状态。然而，它非常通用，直接使用它的用途有限。实际上，它并没有执行更多逻辑的真正逻辑，而只是检查 URL 是否按预期响应。为了使其更智能，我们需要添加一些自己的逻辑。

### 检查凭证

让我们从设置我们的`docstring`、库导入以及一个带有理论网络服务凭证的`__virtual__()`函数开始：

```py
'''
This state connects to an imaginary web service.
The following credentials must be configured:

    webapi_username: <your username>
    webapi_password: <your password>

This module should be saved as salt/states/fake_webapi.py
'''
import salt.utils.http

def __virtual__():
    '''
    Make sure there are credentials
    '''
    username = __salt__'config.get'
    password = __salt__'config.get'
    if username and password:
        return True
    return False
```

在这个情况下，我们不是检查`http.query`函数的存在；正如我们之前所说的，它已经存在了。但是，如果没有能够连接到网络服务，这个模块将无法工作，所以我们快速检查以确保凭证已经就位。

我们没有检查服务本身是否响应，或者凭证是否正确。`__virtual__()`函数在 Minion 启动时进行检查，那时进行所有这些检查是不必要的，而且在停机事件中可能是不准确的。更好的做法是在我们实际调用服务时再进行检查。

### 第一个状态函数

接下来，我们需要设置一个状态函数。在我们的例子中，我们将允许用户确保该网络服务上的特定用户账户已被锁定。首先，我们设置默认值，然后检查该用户的账户是否已被锁定：

```py
def locked(name):
    '''
    Ensure that the user is locked out
    '''
    username = __salt__'config.get'
    password = __salt__'config.get'

    ret = {'name': name,
           'changes': {},
           'result': None,
           'comment': ''}

    result = salt.utils.http.query(
        'https://api.example.com/v1/users/{0}'.format(name),
        username=username,
        password=password,
        decode=True,
        decode_type='json',
    )

    if result('dict', {}).get('access', '') == 'locked':
        ret['result'] = True
        ret['comment'] = 'The account is already locked'
        return ret
```

你可能立刻就会发现问题。进行认证的网络调用有点重，尤其是当你必须解码返回数据时，无论你如何做。我们将在这个函数中再次进行网络调用，在其他函数中还会进行更多调用。让我们将我们可以的部分拆分到另一个函数中：

```py
def _query(action, resource='', data=None):
    '''
    Make a query against the API
    '''
    username = __salt__'config.get'
    password = __salt__'config.get'

    result = salt.utils.http.query(
        'https://api.example.com/v1/{0}/{1}'.format(action, resource),
        username=username,
        password=password,
        decode=True,
        decode_type='json',
        data=data,
    )

def locked(name):
    '''
    Ensure that the user is locked out
    '''
    ret = {'name': name,
           'changes': {},
           'result': None,
           'comment': ''}

    result = _query('users', name)
    if result('dict', {}).get('access', '') == 'locked':
        ret['result'] = True
        ret['comment'] = 'The account is already locked'
        return ret
```

新的`_query()`函数至少需要一个参数：将要执行的操作（`action`）的类型。这种类型的 API 通常期望在未指定特定资源的情况下列出该查询的所有项目，所以我们允许资源为空。我们还设置了一个名为`data`的可选参数，我们将在稍后使用它。

现在我们有一个检查账户是否被锁定，并且如果它是的话，我们可以返回`True`。如果我们通过了这一点，我们知道账户没有被锁定，所以让我们进行对`test`模式的检查：

```py
    if __opts__['test']:
        ret['comment'] = 'The {0} account is not locked'.format(name)
        return ret
```

这部分很容易；我们已经有了一切需要的`test`模式信息，我们不需要做任何事情，除了返回它。让我们尝试将正确的设置应用到账户上。

```py
    _query('users', name, {'access': 'locked'})
```

记住那个`data`选项吗？我们用它传递一个字典，将用户的访问值设置为`locked`。这也是使用 Web API 修改数据的一种非常常见的方式。

当然，我们不一定知道设置是否被正确应用，所以让我们再进行一次检查，以确保：

```py
    result = _query('users', name)
    if result('dict', {}).get('access', '') == 'locked':
        ret['changes'] = {'locked': name}
        ret['result'] = True
        ret['comment'] = 'The {0} user account is now locked'.format(name)
        return ret
    else:
        ret['result'] = False
        ret['comment'] = 'Failed to set the {0} user account to locked'.format(name)
        return ret
```

如果账户现在被锁定，那么我们可以返回我们已成功。如果账户仍然没有被锁定，那么我们可以返回一个失败信息。

### 另一个状态函数

让我们继续添加另一个函数，以便解锁用户账户。我们也将借此机会向您展示整个模块，包括所有公共和私有函数：

```py
'''
This state connects to an imaginary web service.
The following credentials must be configured:

    webapi_username: <your username>
    webapi_password: <your password>

This module should be saved as salt/states/fake_webapi.py
'''
import salt.utils.http

def __virtual__():
    '''
    Make sure there are credentials
    '''
    username = __salt__'config.get'
    password = __salt__'config.get'
    if username and password:
        return True
    return False

def _query(action, resource='', data=None):
    '''
    Make a query against the API
    '''
    username = __salt__'config.get'
    password = __salt__'config.get'

    result = salt.utils.http.query(
        'https://api.example.com/v1/{0}/{1}'.format(action, resource),
        username=username,
        password=password,
        decode=True,
        decode_type='json',
        data=data,
    )
return result

def locked(name):
    '''
    Ensure that the user is locked out
    '''
    ret = {'name': name,
           'changes': {},
           'result': None,
           'comment': ''}

    result = _query('users', name)
    if result('dict', {}).get('access', '') == 'locked':
        ret['result'] = True
        ret['comment'] = 'The account is already locked'
        return ret

    if __opts__['test']:
        ret['comment'] = 'The {0} account is not locked'.format(name)
        return ret

    _query('users', name, {'access': 'locked'})

    result = _query('users', name)
    if result('dict', {}).get('access', '') == 'locked':
        ret['changes'] = {'locked': name}
        ret['result'] = True
        ret['comment'] = 'The {0} user account is now locked'.format(name)
        return ret
    else:
        ret['result'] = False
        ret['comment'] = 'Failed to set the {0} user account to locked'.format(name)
        return ret

def unlocked(name):
    '''
    Ensure that the user is NOT locked out
    '''
    ret = {'name': name,
           'changes': {},
           'result': None,
           'comment': ''}

    result = _query('users', name)
    if result('dict', {}).get('access', '') == 'unlocked':
        ret['result'] = True
        ret['comment'] = 'The account is already unlocked'
        return ret

    if __opts__['test']:
        ret['comment'] = 'The {0} account is locked'.format(name)
        return ret

    _query('users', name, {'access': 'unlocked'})

    result = _query('users', name)
    if result('dict', {}).get('access', '') == 'unlocked':
        ret['changes'] = {'locked': name}
        ret['result'] = True
        ret['comment'] = 'The {0} user account is no longer locked'.format(name)
        return ret
    else:
        ret['result'] = False
        ret['comment'] = 'Failed to unlock the {0} user account'.format(name)
        return ret
```

你可以看到这两个函数之间没有太大的区别。实际上，它们确实做了完全相同的事情，但逻辑相反：一个锁定账户，另一个解锁账户。

状态模块通常包含同一配置的两个相反值。你经常会看到像`installed`和`removed`、`present`和`absent`、`running`和`dead`这样的函数名。

# 调试状态模块

尽管代码结构更清晰，但调试状态模块可能有点棘手。这是因为你需要测试所有四种类型的返回结果：

+   正确 – 资源已正确配置

+   无 – 资源配置不正确，且`test`模式为真

+   正确并更改 – 资源之前配置不正确，但现在已正确

+   错误 – 资源配置不正确

使这更加棘手的是，在调试过程中，你可能会多次更改配置，从正确到错误，然后再回到正确，直到代码正确为止。我建议将其拆分。

## 第一步：测试真值

在设置默认值之后，你的第一步是检查资源是否正确配置。这可能需要你手动切换设置以确保它正确地检查了所需和不需要的配置。添加两个返回值：一个用于`True`，一个用于`False`：

```py
    ret = {'name': name,
           'changes': {},
           'result': None,
           'comment': ''}
    if <item is already in the desired state>:
        ret['result'] = True
        ret['comment'] = 'The item is already in the desired state'
        return ret
    ret['result'] = False
    return ret
```

一旦你知道代码是正确的，你可以稍后删除最后两行。你不需要设置整个 SLS 文件来测试你的状态；你可以使用`state.single`来执行一次性的状态命令：

```py
# salt-run --local state.single fake_webapi.locked larry

```

## 第 2 步：测试模式

一旦你确信它能够正确地检测当前配置，手动将配置设置为不期望的值，并确保`test`模式工作正常：

## 第 3 步：应用更改

当你确信你的代码在尝试应用更改之前不会尝试检查测试模式，你可以继续应用更改。

这是最难的部分，有两个原因。首先，你将不得不频繁地设置和重置你的配置。这最多可能有些繁琐，但这是不可避免的。其次，你将同时设置正确的配置，然后测试以查看它是否被设置：

```py
    <attempt to configure the item correctly>
    if <we are able to put the item in the correct state>:
        ret['changes'] = {'desired state': name}
        ret['result'] = True
        ret['comment'] = 'The desired state was successfully achieved'
        return ret
    else:
        ret['result'] = False
        ret['comment'] = 'The desired state failed to be achieved'
        return ret
```

你可能认为你可以将这部分拆分，但很快你可能会意识到，为了确保配置被正确应用，你仍然需要执行与你在自己的测试中通常执行相同的检查，所以你不妨现在就把它解决掉。

## 测试相反的操作

幸运的是，如果你正在编写执行相反功能的函数，第二个通常要快得多。这是因为一旦你完成了第一个，你可以继续运行它来将配置重置为第二个不期望的值。在我们的例子中，一旦你能够锁定一个账户，你就可以在测试解锁功能时轻松地锁定它。

# 摘要

状态模块比执行模块更有结构，但这通常使它们更容易编写。状态返回的结果可以是 True（绿色），None（黄色），True with changes（蓝色），或 False（红色）。状态模块通常包含执行相反功能的函数对。

现在你已经知道如何编写状态模块了，是时候看看我们传递给它们的那些数据了。接下来是：渲染器！
