# 第十一章：处理系统中的变更、依赖和秘密

在本章中，我们将描述与多个微服务交互的不同元素。

我们将研究如何制定服务描述其版本的策略，以便依赖的微服务可以发现它们，并确保它们已经部署了正确的依赖关系。这将允许我们在依赖服务中定义部署顺序，并且如果不是所有依赖关系都准备好，将停止服务的部署。

本章描述了如何定义集群范围的配置参数，以便它们可以在多个微服务之间共享，并在单个位置进行管理，使用 Kubernetes ConfigMap。我们还将学习如何处理那些属于秘密的配置参数，比如加密密钥，这些密钥不应该对团队中的大多数人可见。

本章将涵盖以下主题：

+   理解微服务之间的共享配置

+   处理 Kubernetes 秘密

+   定义影响多个服务的新功能

+   处理服务依赖关系

在本章结束时，您将了解如何为安全部署准备依赖服务，以及如何在微服务中包含不会在其预期部署之外可访问的秘密。

# 技术要求

代码可在 GitHub 上的以下 URL 找到：[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11)。请注意，该代码是`Chapter10`代码的扩展，其中包含本章描述的额外元素。结构相同，有一个名为`microservices`的子目录，其中包含代码，另一个名为`kubernetes`的子目录，其中包含 Kubernetes 配置文件。

要安装集群，您需要使用以下命令构建每个单独的微服务：

```py
$ cd Chapter11/microservices/
$ cd rsyslog
$ docker-compose build
...
$ cd frontend
$ ./build-test.sh
...
$ cd thoughts_backend
$./build-test.sh
...
$ cd users_backend
$ ./build-test.sh
... 
```

这将构建所需的服务。

请注意，我们使用`build-test.sh`脚本。我们将在本章中解释它的工作原理。

然后，创建`namespace`示例，并使用`Chapter11/kubernetes`子目录中的配置启动 Kubernetes 集群：

```py
$ cd Chapter11/kubernetes
$ kubectl create namespace example
$ kubectl apply --recursive -f .
...
```

这将在集群中部署微服务。

`Chapter11`中包含的代码存在一些问题，**在修复之前**将无法正确部署。这是预期的行为。在本章中，我们将解释两个问题：无法配置秘密，以及无法满足前端的依赖关系，导致无法启动。

继续阅读本章以找到所描述的问题。解决方案将作为评估提出。

要能够访问不同的服务，您需要更新您的`/etc/hosts`文件，包括以下行：

```py
127.0.0.1 thoughts.example.local
127.0.0.1 users.example.local
127.0.0.1 frontend.example.local
```

有了这些，您就可以访问本章的服务了。

# 理解微服务之间的共享配置

某些配置可能适用于多个微服务。在我们的示例中，我们正在为数据库连接重复相同的值。我们可以使用 ConfigMap 并在不同的部署中共享它，而不是在每个部署文件中重复这些值。

我们已经看到如何在第十章 *监控日志和指标* 的*设置指标*部分中添加 ConfigMap 以包含文件。尽管它只用于单个服务。

ConfigMap 是一组键/值元素。它们可以作为环境变量或文件添加。在下一节中，我们将添加一个包含集群中所有共享变量的通用配置文件。

# 添加 ConfigMap 文件

`configuration.yaml`文件包含系统的公共配置。它位于`Chapter11/kubernetes`子目录中：

```py
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: shared-config
  namespace: example
data:
  DATABASE_ENGINE: POSTGRES
  POSTGRES_USER: postgres
  POSTGRES_HOST: "127.0.0.1"
  POSTGRES_PORT: "5432"
  THOUGHTS_BACKEND_URL: http://thoughts-service
  USER_BACKEND_URL: http://users-service
```

与数据库相关的变量，如`DATABASE_ENGINE`、`POSTGRES_USER`、`POSTGRES_HOST`和`POSTGRES_PORT`，在 Thoughts Backend 和 Users Backend 之间共享。

`POSTGRES_PASSWORD`变量是一个密钥。我们将在本章的*处理 Kubernetes 密钥*部分中描述这一点。

`THOUGHTS_BACKEND_URL`和`USER_BACKEND_URL`变量在前端服务中使用。尽管它们在集群中是通用的。任何想要连接到 Thoughts Backend 的服务都应该使用与`THOUGHTS_BACKEND_URL`中描述的相同 URL。

尽管它目前只在单个服务 Frontend 中使用，但它符合系统范围的变量的描述，并应包含在通用配置中。

拥有共享变量存储库的一个优点是将它们合并。

在创建多个服务并独立开发它们的同时，很常见的情况是最终以两种略有不同的方式使用相同的信息。独立开发的团队无法完美共享信息，这种不匹配会发生。

例如，一个服务可以将一个端点描述为`URL=http://service/api`，另一个使用相同端点的服务将其描述为`HOST=service PATH=/api`。每个服务的代码处理配置方式不同，尽管它们连接到相同的端点。这使得以统一方式更改端点更加困难，因为需要在两个或更多位置以两种方式进行更改。

共享位置是首先检测这些问题的好方法，因为如果每个服务保留自己独立的配置，这些问题通常会被忽略，然后调整服务以使用相同的变量，减少配置的复杂性。

在我们的示例中，ConfigMap 的名称是`shared-config`，如元数据中所定义的，像任何其他 Kubernetes 对象一样，可以通过`kubectl`命令进行管理。

# 使用 kubectl 命令

可以使用通常的一组`kubectl`命令来检查 ConfigMap 信息。这使我们能够发现集群中定义的 ConfigMap 实例：

```py
$ kubectl get configmap -n example shared-config
NAME               DATA AGE
shared-config      6    46m
```

请注意，ConfigMap 包含的键或变量的数量是显示的；在这里，它是`6`。要查看 ConfigMap 的内容，请使用`describe`：

```py
$ kubectl describe configmap -n example shared-config
Name: shared-config
Namespace: example
Labels: <none>
Annotations: kubectl.kubernetes.io/last-applied-configuration:
 {"apiVersion":"v1","data":{"DATABASE_ENGINE":"POSTGRES","POSTGRES_HOST":"127.0.0.1","POSTGRES_PORT":"5432","POSTGRES_USER":"postgres","THO...

Data
====
POSTGRES_HOST:
----
127.0.0.1
POSTGRES_PORT:
----
5432
POSTGRES_USER:
----
postgres
THOUGHTS_BACKEND_URL:
----
http://thoughts-service
USER_BACKEND_URL:
----
http://users-service
DATABASE_ENGINE:
----
POSTGRES
```

如果需要更改 ConfigMap，可以使用`kubectl edit`命令，或者更好的是更改`configuration.yaml`文件，并使用以下命令重新应用它：

```py
$ kubectl apply -f kubernetes/configuration.yaml
```

这将覆盖所有的值。

配置不会自动应用到 Kubernetes 集群。您需要重新部署受更改影响的 pod。最简单的方法是删除受影响的 pod，并允许部署重新创建它们。

另一方面，如果配置了 Flux，它将自动重新部署依赖的 pod。请记住，更改 ConfigMap（在所有 pod 中引用）将触发在该情况下所有 pod 的重新部署。

我们现在将看到如何将 ConfigMap 添加到部署中。

# 将 ConfigMap 添加到部署

一旦 ConfigMap 就位，它可以用于与不同部署共享其变量，保持一个中央位置来更改变量并避免重复。

让我们看看微服务（Thoughts Backend、Users Backend 和 Frontend）的每个部署如何使用`shared-config` ConfigMap。

# Thoughts Backend ConfigMap 配置

Thoughts Backend 部署定义如下：

```py
spec:
    containers:
        - name: thoughts-backend-service
          image: thoughts_server:v1.5
          imagePullPolicy: Never
          ports:
              - containerPort: 8000
          envFrom:
              - configMapRef:
                    name: shared-config
          env:
              - name: POSTGRES_DB
                value: thoughts
          ...
```

完整的`shared-config` ConfigMap 将被注入到 pod 中。请注意，这包括以前在 pod 中不可用的`THOUGHTS_BACKEND_URL`和`USER_BACKEND_URL`环境变量。可以添加更多环境变量。在这里，我们保留了`POSTGRES_DB`，而没有将其添加到 ConfigMap 中。

我们可以在 pod 中使用`exec`来确认它。

请注意，为了能够连接到密钥，它应该被正确配置。请参阅*处理 Kubernetes 密钥*部分。

要在容器内部检查，请检索 pod 名称并在其中使用`exec`，如下面的命令所示：

```py
$ kubectl get pods -n example
NAME                              READY STATUS  RESTARTS AGE
thoughts-backend-5c8484d74d-ql8hv 2/2   Running 0        17m
...
$ kubectl exec -it thoughts-backend-5c8484d74d-ql8hv -n example /bin/sh
Defaulting container name to thoughts-backend-service.
/opt/code $ env | grep POSTGRES
DATABASE_ENGINE=POSTGRESQL
POSTGRES_HOST=127.0.0.1
POSTGRES_USER=postgres
POSTGRES_PORT=5432
POSTGRES_DB=thoughts
/opt/code $ env | grep URL
THOUGHTS_BACKEND_URL=http://thoughts-service
USER_BACKEND_URL=http://users-service
```

`env`命令返回所有环境变量，但 Kubernetes 会自动添加很多环境变量。

# 用户后端 ConfigMap 配置

用户后端配置与我们刚刚看到的前一种类型的配置类似：

```py
spec:
    containers:
        - name: users-backend-service
          image: users_server:v2.3
          imagePullPolicy: Never
          ports:
              - containerPort: 8000
          envFrom:
              - configMapRef:
                    name: shared-config
          env:
              - name: POSTGRES_DB
                value: thoughts
          ...
```

`POSTGRES_DB`的值与 Thoughts 后端中的相同，但我们将其留在这里以展示如何添加更多环境变量。

# 前端 ConfigMap 配置

前端配置仅使用 ConfigMap，因为不需要额外的环境变量：

```py
spec:
    containers:
        - name: frontend-service
          image: thoughts_frontend:v3.7
          imagePullPolicy: Never
          ports:
              - containerPort: 8000
          envFrom:
              - configMapRef:
                    name: shared-config
```

前端 pod 现在还将包括连接到数据库的信息，尽管它不需要。对于大多数配置参数来说，这是可以的。

如果需要，您还可以使用多个 ConfigMaps 来描述不同的配置组。不过，将它们放在一个大桶中处理会更简单。这将有助于捕获重复的参数，并确保所有微服务中都有所需的参数。

然而，一些配置参数必须更加小心处理，因为它们将是敏感的。例如，我们从`shared-config` ConfigMap 中省略了`POSTGRES_PASSWORD`变量。这允许我们登录到数据库，并且不应该存储在任何带有其他参数的文件中，以避免意外暴露。

为了处理这种信息，我们可以使用 Kubernetes 秘密。

# 处理 Kubernetes 秘密

秘密是一种特殊的配置。它们需要受到保护，以免被其他使用它们的微服务读取。它们通常是敏感数据，如私钥、加密密钥和密码。

记住，读取秘密是有效的操作。毕竟，它们需要被使用。秘密与其他配置参数的区别在于它们需要受到保护，因此只有授权的来源才能读取它们。

秘密应该由环境注入。这要求代码能够检索配置秘密并在当前环境中使用适当的秘密。它还避免了在代码中存储秘密。

记住*永远*不要在 Git 存储库中提交生产秘密。即使删除了 Git 树，秘密也是可检索的。这包括 GitOps 环境。

还要为不同的环境使用不同的秘密。生产秘密需要比测试环境中的秘密更加小心。

在我们的 Kubernetes 配置中，授权的来源是使用它们的微服务以及通过`kubectl`访问的系统管理员。

让我们看看如何管理这些秘密。

# 在 Kubernetes 中存储秘密

Kubernetes 将秘密视为一种特殊类型的 ConfigMap 值。它们可以在系统中定义，然后以与 ConfigMap 相同的方式应用。与一般的 ConfigMap 的区别在于信息在内部受到保护。虽然它们可以通过`kubectl`访问，但它们受到意外暴露的保护。

可以通过`kubectl`命令在集群中创建秘密。它们不应该通过文件和 GitOps 或 Flux 创建，而应该手动创建。这样可以避免将秘密存储在 GitOps 存储库下。

需要秘密来操作的 pod 将在其部署文件中指示。这是安全的存储在 GitOps 源代码控制下，因为它不存储秘密，而只存储对秘密的引用。当 pod 被部署时，它将使用适当的引用并解码秘密。

登录到 pod 将授予您对秘密的访问权限。这是正常的，因为在 pod 内部，应用程序需要读取其值。授予对 pod 中执行命令的访问权限将授予他们对内部秘密的访问权限，因此请记住这一点。您可以阅读 Kubernetes 文档了解秘密的最佳实践，并根据您的要求进行调整（[`kubernetes.io/docs/concepts/configuration/secret/#best-practices`](https://kubernetes.io/docs/concepts/configuration/secret/#best-practices)）。

既然我们知道如何处理它们，让我们看看如何创建这些秘密。

# 创建秘密

让我们在 Kubernetes 中创建这些秘密。我们将存储以下秘密：

+   PostgreSQL 密码

+   用于签署和验证请求的公钥和私钥

我们将它们存储在同一个 Kubernetes 秘密中，该秘密可以有多个密钥。以下命令显示了如何生成一对密钥：

```py
$ openssl genrsa -out private_key.pem 2048
Generating RSA private key, 2048 bit long modulus
........+++
.................+++
e is 65537 (0x10001)
$ openssl rsa -in private_key.pem -outform PEM -pubout -out public_key.pub
writing RSA key
$ ls 
private_key.pem public_key.pub
```

这些密钥是唯一的。我们将使用它们来替换前几章中存储的示例密钥。

# 在集群中存储秘密

将秘密存储在集群中，在`thoughts-secrets`秘密下。请记住将其存储在`example`命名空间中：

```py
$ kubectl create secret generic thoughts-secrets --from-literal=postgres-password=somepassword --from-file=private_key.pem --from-file=public_key.pub -n example
```

您可以列出命名空间中的秘密：

```py
$ kubectl get secrets -n example
NAME             TYPE   DATA AGE
thoughts-secrets Opaque 3    41s
```

您还可以描述更多信息的秘密：

```py
$ kubectl describe secret thoughts-secrets -n example
Name: thoughts-secrets
Namespace: default
Labels: <none>
Annotations: <none>

Type: Opaque

Data
====
postgres-password: 12 bytes
private_key.pem: 1831 bytes
public_key.pub: 408 bytes
```

您可以获取秘密的内容，但数据以 Base64 编码检索。

Base64 是一种编码方案，允许您将二进制数据转换为文本，反之亦然。它被广泛使用。这使您可以存储任何二进制秘密，而不仅仅是文本。这也意味着在检索时秘密不会以明文显示，从而在意外显示在屏幕上等情况下增加了一层保护。

要获取秘密，请使用如下所示的常规`kubectl get`命令。我们使用`base64`命令对其进行解码：

```py
$ kubectl get secret thoughts-secrets -o yaml -n example
apiVersion: v1
data:
 postgres-password: c29tZXBhc3N3b3Jk
 private_key.pem: ...
 public_key.pub: ...
$ echo c29tZXBhc3N3b3Jk | base64 --decode
somepassword
```

同样，如果要编辑秘密以更新它，输入应该以 Base64 编码。

# 秘密部署配置

我们需要在部署配置中配置秘密的使用，以便在所需的 pod 中可用。例如，在用户后端的`deployment.yaml`配置文件中，我们有以下代码：

```py
spec:
    containers:
    - name: users-backend-service
      ...
      env:
      ...
      - name: POSTGRES_PASSWORD
        valueFrom:
          secretKeyRef:
            name: thoughts-secrets
            key: postgres-password
        volumeMounts:
        - name: sign-keys
          mountPath: "/opt/keys/"

    volumes:
    - name: sign-keys
      secret:
        secretName: thoughts-secrets
        items:
        - key: public_key.pub
          path: public_key.pub
        - key: private_key.pem
          path: private_key.pem
```

我们创建了来自秘密的`POSTGRES_PASSWORD`环境变量。我们还创建了一个名为`sign-keys`的卷，其中包含两个密钥文件，`public_key.pub`和`private_key.pem`。它挂载在`/opt/keys/`路径中。

类似地，Thoughts 后端的`deployment.yaml`文件包括秘密，但只包括 PostgreSQL 密码和`public_key.pub`。请注意，私钥没有添加，因为 Thoughts 后端不需要它，也不可用。

对于前端，只需要公钥。现在，让我们来建立如何检索这些秘密。

# 应用程序检索秘密

对于`POSTGRES_PASSWORD`环境变量，我们不需要更改任何内容。它已经是一个环境变量，代码已经从中提取它。

但是对于存储为文件的秘密，我们需要从适当的位置检索它们。存储为文件的秘密是签署身份验证标头的关键。公共文件在所有微服务中都是必需的，而私钥仅在用户后端中使用。

现在，让我们来看一下用户后端的`config.py`文件：

```py
import os
PRIVATE_KEY = ...
PUBLIC_KEY = ...

PUBLIC_KEY_PATH = '/opt/keys/public_key.pub'
PRIVATE_KEY_PATH = '/opt/keys/private_key.pem'

if os.path.isfile(PUBLIC_KEY_PATH):
    with open(PUBLIC_KEY_PATH) as fp:
        PUBLIC_KEY = fp.read()

if os.path.isfile(PRIVATE_KEY_PATH):
    with open(PRIVATE_KEY_PATH) as fp:
        PRIVATE_KEY = fp.read()
```

当前密钥仍然作为默认值存在。当秘密文件没有挂载时，它们将用于单元测试。

再次强调，请*不要*使用这些密钥。这些仅用于运行测试，并且对于任何可以访问本书的人都是可用的。

如果`/opt/keys/`路径中存在文件，它们将被读取，并且内容将被存储在适当的常量中。用户后端需要公钥和私钥。

在 Thoughts 后端的`config.py`文件中，我们只检索公钥，如下所示：

```py
import os
PUBLIC_KEY = ...

PUBLIC_KEY_PATH = '/opt/keys/public_key.pub'

if os.path.isfile(PUBLIC_KEY_PATH):
    with open(PUBLIC_KEY_PATH) as fp:
        PUBLIC_KEY = fp.read()
```

前端服务将公钥添加到`settings.py`文件中：

```py
TOKENS_PUBLIC_KEY = ...

PUBLIC_KEY_PATH = '/opt/keys/public_key.pub'

if os.path.isfile(PUBLIC_KEY_PATH):
    with open(PUBLIC_KEY_PATH) as fp:
        TOKENS_PUBLIC_KEY = fp.read()
```

此配置使秘密对应用程序可用，并为秘密值关闭了循环。现在，微服务集群使用来自秘密值的签名密钥，这是一种安全存储敏感数据的方式。

# 定义影响多个服务的新功能

我们谈到了单个微服务领域内的更改请求。但是，如果我们需要部署在两个或多个微服务中运行的功能，该怎么办呢？

这种类型的功能应该相对罕见，并且是与单体应用程序相比微服务中的开销的主要原因之一。在单体应用程序中，这种情况根本不可能发生，因为一切都包含在单体应用程序的墙内。

与此同时，在微服务架构中，这是一个复杂的更改。这至少涉及到每个相关微服务中的两个独立功能，这些功能位于两个不同的存储库中。很可能这些存储库将由两个不同的团队开发，或者至少负责每个功能的人将不同。

# 逐个更改

为了确保功能可以顺利部署，一次一个，它们需要保持向后兼容。这意味着您需要能够在服务 A 已部署但服务 B 尚未部署的中间阶段生存。微服务中的每个更改都需要尽可能小，以最小化风险，并且应逐个引入更改。

为什么不同时部署它们？因为同时发布两个微服务是危险的。首先，部署不是瞬时的，因此会有时刻，过时的服务将发送或接收系统尚未准备处理的调用。这将导致可能影响您的客户的错误。

但是存在一种情况，其中一个微服务不正确并且需要回滚。然后，系统将处于不一致状态。依赖的微服务也需要回滚。这本身就是有问题的，但是当在调试此问题期间，两个微服务都卡住并且在问题得到解决之前无法更新时，情况会变得更糟。

在健康的微服务环境中，部署会经常发生。因为另一个服务需要工作而不得不停止微服务的流水线是一个糟糕的处境，它只会增加压力和紧迫感。

记住我们谈到了部署和变更的速度。经常部署小的增量通常是确保每次部署都具有高质量的最佳方式。增量工作的持续流非常重要。

由于错误而中断此流程是不好的，但是如果无法部署影响了多个微服务的速度，影响会迅速扩大。

同时部署多个服务也可能导致死锁，其中两个服务都需要进行修复工作。这会使开发和解决问题的时间变得更加复杂。

需要进行分析以确定哪个微服务依赖于另一个而不是同时部署。大多数情况下，这是显而易见的。在我们的例子中，前端依赖于 Thoughts 后端，因此任何涉及它们两者的更改都需要从 Thoughts 后端开始，然后转移到前端。

实际上，用户后端是两者的依赖项，因此假设有一个影响它们三者的更改，您需要首先更改用户后端，然后是 Thoughts 后端，最后是前端。

请记住，有时部署可能需要跨多个服务进行多次移动。例如，让我们想象一下，我们对身份验证标头的签名机制进行了更改。然后，流程应该如下：

1.  在用户后端实施新的身份验证系统，但通过配置更改继续使用旧系统生成令牌。到目前为止，集群仍在使用旧的身份验证流程。

1.  更改 Thoughts 后端以允许与旧系统和新的身份验证系统一起工作。请注意，它尚未激活。

1.  更改前端以使其与两种身份验证系统一起工作。但是，此时新系统尚未被使用。

1.  在用户后端更改配置以生成新的身份验证令牌。现在是新系统开始使用的时候。在部署过程中，可能会生成一些旧系统令牌。

1.  用户后端和前端将使用系统中的任何令牌，无论是新的还是旧的。旧令牌将随着时间的推移而消失，因为它们会过期。只有新令牌才会被创建。

1.  作为可选阶段，可以从系统中删除旧的身份验证系统。三个系统可以在没有任何依赖关系的情况下删除它们，因为此时系统不再使用。

在整个过程的任何步骤中，服务都不会中断。每个单独的更改都是安全的。该过程正在慢慢使整个系统发展，但如果出现问题，每个单独的步骤都是可逆的，并且服务不会中断。

系统往往通过添加新功能来发展，清理阶段并不常见。通常，即使功能在任何地方都没有使用，系统也会长时间保留已弃用的功能。

我们将在《第十二章》*跨团队协作和沟通*中更详细地讨论清理工作。

此过程也可能需要进行配置更改。例如，更改用于签署身份验证标头的私钥将需要以下步骤：

1.  使 Thoughts 后端和前端能够处理多个公钥。这是一个先决条件和一个新功能。

1.  更改 Thoughts 后端中处理的密钥，使其同时具有旧公钥和新公钥。到目前为止，系统中没有使用新密钥签名的标头。

1.  更改前端中处理的密钥，使其同时具有旧密钥和新密钥。但是，系统中仍没有使用新密钥签名的标头。

1.  更改用户后端的配置以使用新的私钥。从现在开始，系统中有用新私钥签名的标头。其他微服务能够处理它们。

1.  系统仍然接受用旧密钥签名的标头。等待一个安全期以确保所有旧标头都已过期。

1.  删除用户后端的旧密钥配置。

步骤 2 至 6 可以每隔几个月重复使用新密钥。

这个过程被称为**密钥轮换**，被认为是一种良好的安全实践，因为它减少了密钥有效的时间，缩短了系统暴露于泄露密钥的时间窗口。为简单起见，我们没有在示例系统中实施它，但建议您这样做。尝试更改示例代码以实现此密钥轮换示例！

完整的系统功能可能涉及多个服务和团队。为了帮助协调系统的依赖关系，我们需要知道某个服务的特定依赖项何时部署并准备就绪。我们将在《第十二章》*跨团队协作和沟通*中讨论团队间的沟通，但我们可以通过使服务 API 明确描述已部署的服务版本来通过编程方式进行帮助，正如我们将在*处理服务依赖关系*部分中讨论的那样。

如果新版本出现问题，刚刚部署的版本可以通过回滚快速恢复。

# 回滚微服务

回滚是将微服务之一迅速退回到先前版本的过程。

当新版本出现灾难性错误时，可以触发此过程，以便快速解决问题。鉴于该版本已经兼容，可以在非常短的反应时间内放心地进行此操作。通过 GitOps 原则，可以执行`revert`提交以恢复旧版本。

`git revert`命令允许您创建一个撤消另一个提交的提交，以相反的方式应用相同的更改。

这是撤消特定更改的快速方法，并允许稍后*撤消撤消*并重新引入更改。您可以查看 Git 文档以获取更多详细信息（[`git-scm.com/docs/git-revert`](https://git-scm.com/docs/git-revert)）。

鉴于保持前进的战略性方法，回滚是一种临时措施，一旦实施，将停止微服务中的新部署。应尽快创建一个解决导致灾难性部署的错误的新版本，以保持正常的发布流程。

随着部署次数的增加，并且在适当的位置进行更好的检查，回滚将变得越来越少。

# 处理服务依赖关系

为了让服务检查它们的依赖项是否具有正确的版本，我们将使服务通过 RESTful 端点公开它们的版本。

我们将遵循 GitHub 上的 Thoughts Backend 示例，网址为：[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11/microservices/thoughts_backend`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11/microservices/thoughts_backend)。

在前端检查版本是否可用（[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11/microservices/frontend`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11/microservices/frontend)）。

该过程的第一步是为每个服务正确定义版本。

# 服务版本控制

为了清晰地了解软件的进展，我们需要命名要部署的不同版本。由于我们使用`git`来跟踪更改，系统中的每个提交都有一个独立的提交 ID，但它没有遵循任何特定的模式。

为了赋予其意义并对其进行排序，我们需要开发一个版本模式。有多种制定版本模式的方法，包括按发布日期（Ubuntu 使用此方法）或按`major.minor.patch`。

在所有地方使用相同的版本控制方案有助于在团队之间发展共同的语言和理解。它还有助于管理了解变化，无论是在发布时的变化还是变化的速度。与您的团队商定一个在您的组织中有意义的版本控制方案，并在所有服务中遵循它。

在此示例中，我们将使用`vMajor.Minor`模式，并将用户后端的版本设置为`v2.3`。

软件版本控制中最常见的模式是语义版本控制。这种版本控制模式对于软件包和面向客户的 API 非常有用，但对于内部微服务 API 则不太有用。让我们看看它的特点是什么。

# 语义版本控制

语义版本控制对不同版本号的每个更改赋予了含义。这使得很容易理解各个版本之间的变化范围，以及更新是否对依赖系统有风险。

语义版本控制使用三个数字定义每个版本：主要版本、次要版本和补丁版本，通常描述为`major.minor.patch`。

增加这些数字中的任何一个都具有特定的含义，如下所示：

+   增加主要版本号会产生不兼容的变化。

+   增加次要版本号会添加新功能，但保持向后兼容。

+   增加补丁号修复错误，但不添加任何新功能。

例如，Python 按照以下模式工作：

+   Python 3 与 Python 2 包含了兼容性变化。

+   Python 3.7 版本与 Python 3.6 相比引入了新功能。

+   Python 3.7.4 版本相对于 Python 3.7.3 增加了安全性和错误修复。

这种版本控制方案在与外部合作伙伴的沟通中非常有用，并且非常适用于大型发布和标准软件包。但对于微服务中的小型增量变化，它并不是非常有用。

正如我们在前面的章节中讨论的那样，持续集成的关键是进行非常小的更改。它们不应该破坏向后兼容性，但随着时间的推移，旧功能将被删除。每个微服务都以受控的方式与其他服务协同工作。与外部包相比，没有必要具有如此强烈的功能标签。服务的消费者是集群中受严格控制的其他微服务。

一些项目由于操作方式的改变而放弃了语义版本。例如，Linux 内核停止使用语义版本来生成没有特定含义的新版本（[`lkml.iu.edu/hypermail/linux/kernel/1804.1/06654.html`](http://lkml.iu.edu/hypermail/linux/kernel/1804.1/06654.html)），因为从一个版本到下一个版本的更改相对较小。

Python 也将版本 4.0 视为*在 3.9 之后的版本*，并且不像 Python 3 那样有重大变化（[`www.curiousefficiency.org/posts/2014/08/python-4000.html`](http://www.curiousefficiency.org/posts/2014/08/python-4000.html)）。

这就是为什么在内部*不*建议使用语义版本。保持类似的版本方案可能是有用的，但不要强制它进行兼容性更改，只需增加数字，而不对何时更改次要或主要版本做出具体要求。

然而，从外部来看，版本号可能仍然具有营销意义。对于外部可访问的端点，使用语义版本可能是有趣的。

一旦确定了服务的版本，我们就可以着手创建一个公开此信息的端点。

# 添加版本端点

要部署的版本可以从 Kubernetes 部署或 GitOps 配置中读取。但是存在一个问题。一些配置可能会误导或不唯一地指向单个镜像。例如，`latest`标签可能在不同时间代表不同的容器，因为它会被覆盖。

此外，还存在访问 Kubernetes 配置或 GitOps 存储库的问题。对于开发人员来说，也许这些配置是可用的，但对于微服务来说不会（也不应该）。

为了让集群中的其他微服务发现服务的版本，最好的方法是在 RESTful API 中明确创建一个版本端点。服务版本的发现是被授予的，因为它使用与任何其他请求中将使用的相同接口。让我们看看如何实现它。

# 获取版本

为了提供版本，我们首先需要将其记录到服务中。

正如我们之前讨论过的，版本是存储为 Git 标签的。这将是我们版本的标准。我们还将添加提交的 Git SHA-1，以避免任何差异。

SHA-1 是一个唯一的标识符，用于标识每个提交。它是通过对 Git 树进行哈希处理而生成的，因此能够捕获任何更改——无论是内容还是树历史。我们将使用 40 个字符的完整 SHA-1，尽管有时它会被缩写为八个或更少。

提交的 SHA-1 可以通过以下命令获得：

```py
$ git log --format=format:%H -n 1
```

这将打印出最后一次提交的信息，以及带有`%H`描述符的 SHA。

要获取此提交所指的标签，我们将使用`git-describe`命令：

```py
$ git describe --tags
```

基本上，`git-describe`会找到最接近当前提交的标签。如果此提交由标签标记，正如我们的部署应该做的那样，它将返回标签本身。如果没有，它将在标签后缀中添加有关提交的额外信息，直到达到当前提交。以下代码显示了如何使用`git describe`，具体取决于代码的提交版本。请注意，与标签不相关的代码将返回最接近的标签和额外的数字：

```py
$ # in master branch, 17 commits from the tag v2.3
$ git describe
v2.3-17-g2257f9c
$ # go to the tag
$ git checkout v2.3
$ git describe
v2.3
```

这将始终返回一个版本，并允许我们一目了然地检查当前提交的代码是否在`git`中标记。

将部署到环境中的任何内容都应该被标记。本地开发是另一回事，因为它包括尚未准备好的代码。

我们可以以编程方式存储这两个值，从而使我们能够自动地进行操作，并将它们包含在 Docker 镜像中。

# 将版本存储在镜像中

我们希望在镜像内部有版本可用。由于镜像是不可变的，所以在构建过程中实现这一目标是我们的目标。我们需要克服的限制是 Dockerfile 过程不允许我们在主机上执行命令，只能在容器内部执行。我们需要在构建时向 Docker 镜像中注入这些值。

一个可能的替代方案是在容器内安装 Git，复制整个 Git 树，并获取值。通常不鼓励这样做，因为安装 Git 和完整的源代码树会给容器增加很多空间，这是不好的。在构建过程中，我们已经有了 Git 可用，所以我们只需要确保外部注入它，这在构建脚本中很容易做到。

通过`ARG`参数传递值的最简单方法。作为构建过程的一部分，我们将把它们转换为环境变量，这样它们将像配置的任何其他部分一样容易获取。让我们来看看以下代码中的 Dockerfile：

```py
# Prepare the version
ARG VERSION_SHA="BAD VERSION"
ARG VERSION_NAME="BAD VERSION"
ENV VERSION_SHA $VERSION_SHA
ENV VERSION_NAME $VERSION_NAME
```

我们接受一个`ARG`参数，然后通过`ENV`参数将其转换为环境变量。为了简单起见，两者都具有相同的名称。`ARG`参数对于特殊情况有一个默认值。

使用`build.sh`脚本构建后，这使得版本在构建后（在容器内部）可用，该脚本获取值并调用`docker-compose`进行构建，使用版本作为参数，具体步骤如下：

```py
# Obtain the SHA and VERSION
VERSION_SHA=`git log --format=format:%H -n 1`
VERSION_NAME=`git describe --tags`
# Build using docker-compose with arguments
docker-compose build --build-arg VERSION_NAME=${VERSION_NAME} --build-arg VERSION_SHA=${VERSION_SHA}
# Tag the resulting image with the version
docker tag thoughts_server:latest throughs_server:${VERSION_NAME}
```

在构建过程之后，版本作为标准环境变量在容器内部可用。

在本章的每个微服务中都包含了一个脚本（例如，[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/blob/master/Chapter11/microservices/thoughts_backend/build-test.sh`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/blob/master/Chapter11/microservices/thoughts_backend/build-test.sh)）。这个脚本模拟 SHA-1 和版本名称，以创建一个用于测试的合成版本。它为用户后端设置了`v2.3`版本，为思想后端设置了`v1.5`版本。这些将被用作我们代码中的示例。

检查 Kubernetes 部署是否包含这些版本（例如，[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/blob/master/Chapter11/microservices/thoughts_backend/docker-compose.yaml#L21`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/blob/master/Chapter11/microservices/thoughts_backend/docker-compose.yaml#L21)镜像是`v1.5`版本）。

此外，`VERSION_NAME`也可以作为 CI 管道的参数传递给脚本。为此，您需要替换脚本以接受外部参数，就像在`build-ci.sh`脚本中看到的那样：

```py
#!/bin/bash
if [ -z "$1" ]
  then
    # Error, not version name
    echo "No VERSION_NAME supplied"
    exit -1
fi

VERSION_SHA=`git log --format=format:%H -n 1`
VERSION_NAME=$1

docker-compose build --build-arg VERSION_NAME=${VERSION_NAME} --build-arg VERSION_SHA=${VERSION_SHA}
docker tag thoughts_server:latest throughs_server:${VERSION_NAME}
```

所有这些脚本的版本都包括使用`VERSION_NAME`作为标签对镜像进行标记。

我们可以在 Python 代码中在容器内检索包含版本的环境变量，并在端点中返回它们，使版本通过外部 API 轻松访问。

# 实现版本端点

在`admin_namespace.py`文件中，我们将使用以下代码创建一个新的`Version`端点：

```py
import os

@admin_namespace.route('/version/')
class Version(Resource):

    @admin_namespace.doc('get_version')
    def get(self):
        '''
        Return the version of the application
        '''
        data = {
            'commit': os.environ['VERSION_SHA'],
            'version': os.environ['VERSION_NAME'],
        }

        return data
```

现在，这段代码非常简单。它使用`os.environ`来检索在构建过程中注入的环境变量作为配置参数，并返回一个包含提交 SHA-1 和标签（描述为版本）的字典。

可以使用`docker-compose`在本地构建和运行服务。要测试对`/admin/version`端点的访问并进行检查，请按照以下步骤进行：

```py
$ cd Chapter11/microservices/thoughts_backend
$ ./build.sh
...
Successfully tagged thoughts_server:latest
$ docker-compose up -d server
Creating network "thoughts_backend_default" with the default driver
Creating thoughts_backend_db_1 ... done
Creating thoughts_backend_server_1 ... done
$ curl http://localhost:8000/admin/version/
{"commit": "2257f9c5a5a3d877f5f22e5416c27e486f507946", "version": "tag-17-g2257f9c"}
```

由于版本可用，我们可以更新自动生成的文档以显示正确的值，如`app.py`中所示：

```py
import os
...
VERSION = os.environ['VERSION_NAME']
...

def create_app(script=False):
    ...
    api = Api(application, version=VERSION, 
              title='Thoughts Backend API',
              description='A Simple CRUD API')
```

因此，版本将在自动生成的 Swagger 文档中正确显示。一旦微服务的版本通过 API 中的端点可访问，其他外部服务就可以访问它以发现版本并加以利用。

# 检查版本

通过 API 能够检查版本使我们能够以编程方式轻松访问版本。这可以用于多种目的，比如生成一个仪表板，显示不同环境中部署的不同版本。但我们将探讨引入服务依赖的可能性。

当微服务启动时，可以检查其所依赖的服务，并检查它们是否高于预期版本。如果不是，它将不会启动。这可以避免在依赖服务更新之前部署依赖服务时出现配置问题。这可能发生在部署协调不佳的复杂系统中。

在`start_server.sh`中启动服务器时，要检查版本，我们将首先调用一个检查依赖项的小脚本。如果不可用，它将产生错误并停止。我们将检查前端是否具有 Thought 后端的可用版本，甚至更高版本。

我们将在我们的示例中调用的脚本称为`check_dependencies_services.py`，并且在前端的`start_server.sh`中调用它。

`check_dependencies_services`脚本可以分为三个部分：所需依赖项列表；一个依赖项的检查；以及一个主要部分，其中检查每个依赖项。让我们来看看这三个部分。

# 所需版本

第一部分描述了每个依赖项和所需的最低版本。在我们的示例中，我们规定`thoughts_backend`需要是版本`v1.6`或更高：

```py
import os

VERSIONS = {
    'thoughts_backend': 
        (f'{os.environ["THOUGHTS_BACKEND_URL"]}/admin/version',
         'v1.6'),
}
```

这里重用环境变量`THOUGHTS_BACKEND_URL`，并使用特定版本路径完成 URL。

主要部分遍历了所有描述的依赖项以进行检查。

# 主要函数

主要函数遍历`VERSIONS`字典，并对每个版本执行以下操作：

+   调用端点

+   解析结果并获取版本

+   调用`check_version`来查看是否正确

如果失败，它以`-1`状态结束，因此脚本报告为失败。这些步骤通过以下代码执行：

```py
import requests

def main():
    for service, (url, min_version) in VERSIONS.items():
        print(f'Checking minimum version for {service}')
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f'Error connecting to {url}: {resp}')
            exit(-1)

        result = resp.json()
        version = result['version']
        print(f'Minimum {min_version}, found {version}')
        if not check_version(min_version, version):
            msg = (f'Version {version} is '
                    'incorrect (min {min_version})')
            print(msg)
            exit(-1)

if __name__ == '__main__':
    main()
```

主要函数还打印一些消息，以帮助理解不同的阶段。为了调用版本端点，它使用`requests`包，并期望`200`状态代码和可解析的 JSON 结果。

请注意，此代码会遍历`VERSION`字典。到目前为止，我们只添加了一个依赖项，但用户后端是另一个依赖项，可以添加。这留作练习。

版本字段将在`check_version`函数中进行检查，我们将在下一节中看到。

# 检查版本

`check_version`函数检查当前返回的版本是否高于或等于最低版本。为了简化，我们将使用`natsort`包对版本进行排序，然后检查最低版本。

您可以查看`natsort`的完整文档（[`github.com/SethMMorton/natsort`](https://github.com/SethMMorton/natsort)）。它可以对许多自然字符串进行排序，并且可以在许多情况下使用。

基本上，`natsort`支持常见的版本排序模式，其中包括我们之前描述的标准版本模式（`v1.6`高于`v1.5`）。以下代码使用该库对版本进行排序，并验证最低版本是否为较低版本：

```py
from natsort import natsorted

def check_version(min_version, version):
    versions = natsorted([min_version, version])
    # Return the lower is the minimum version
    return versions[0] == min_version
```

有了这个脚本，我们现在可以启动服务，并检查 Thoughts 后端是否具有正确的版本。如果您按照*技术要求*部分中描述的方式启动了服务，您会发现前端无法正确启动，并产生`CrashLoopBackOff`状态，如下所示：

```py
$ kubectl get pods -n example
NAME READY STATUS RESTARTS AGE
frontend-54fdfd565b-gcgtt 0/1 CrashLoopBackOff 1 12s
frontend-7489cccfcc-v2cz7 0/1 CrashLoopBackOff 3 72s
grafana-546f55d48c-wgwt5 1/1 Running 2 80s
prometheus-6dd4d5c74f-g9d47 1/1 Running 2 81s
syslog-76fcd6bdcc-zrx65 2/2 Running 4 80s
thoughts-backend-6dc47f5cd8-2xxdp 2/2 Running 0 80s
users-backend-7c64564765-dkfww 2/2 Running 0 81s
```

检查一个前端 pod 的日志，以查看原因，使用`kubectl logs`命令，如下所示：

```py
$ kubectl logs frontend-54fdfd565b-kzn99 -n example
Checking minimum version for thoughts_backend
Minimum v1.6, found v1.5
Version v1.5 is incorrect (min v1.6)
```

要解决问题，您需要构建一个具有更高版本的 Thoughts 后端版本，或者减少依赖要求。这将作为本章结束时的评估留下。

# 总结

在本章中，我们学习了如何处理同时与多个微服务一起工作的元素。

首先，我们讨论了在新功能需要更改多个微服务时要遵循的策略，包括如何以有序的方式部署小的增量，并且能够在出现灾难性问题时回滚。

然后，我们讨论了定义清晰的版本模式，并向 RESTful 接口添加了一个版本端点，以允许微服务自我发现版本。这种自我发现可以用来确保依赖于另一个微服务的微服务在没有依赖项的情况下不会被部署，这有助于协调发布。

本章中的前端 GitHub 代码（[`github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11/microservices/frontend`](https://github.com/PacktPublishing/Hands-On-Docker-for-Microservices-with-Python/tree/master/Chapter11/microservices/frontend)）包含对 Thoughts 后端的依赖，这将阻止部署。请注意，原样的代码无法工作。修复留作练习。

我们还学习了如何使用 ConfigMap 来描述在 Kubernetes 集群中共享的配置信息。我们随后描述了如何使用 Kubernetes secrets 来处理敏感且需要额外注意的配置。

在下一章中，我们将看到协调不同团队与不同微服务高效工作的各种技术。

# 问题

1.  在微服务架构系统和单体架构中发布更改的区别是什么？

1.  在微服务架构中，为什么发布的更改应该很小？

1.  语义版本化是如何工作的？

1.  微服务架构系统中内部接口的语义版本化存在哪些问题？

1.  添加版本端点的优势是什么？

1.  我们如何修复本章代码中的依赖问题？

1.  我们应该在共享的 ConfigMap 中存储哪些配置变量？

1.  您能描述将所有配置变量放在单个共享的 ConfigMap 中的优缺点吗？

1.  Kubernetes ConfigMap 和 Kubernetes secret 之间有什么区别？

1.  我们如何更改 Kubernetes secret？

1.  假设根据配置，我们决定将`public_key.pub`文件从秘密更改为 ConfigMap。我们需要实施哪些更改？

# 进一步阅读

要处理 AWS 上的秘密，您可以与一个名为 CredStash 的工具交互（[`github.com/fugue/credstash`](https://github.com/fugue/credstash)）。您可以在书籍*AWS SysOps Cookbook – Second Edition* ([`www.packtpub.com/cloud-networking/aws-administration-cookbook-second-edition`](https://www.packtpub.com/cloud-networking/aws-administration-cookbook-second-edition))中了解更多信息。
