

# 第四章：使用 React API 获取数据

在过去的几年里，对数据库驱动 Web 应用程序的需求有所增加。这种增加是当前数据丰富性的结果。随着互联网的广泛采用，企业利用 Web 应用程序与客户、员工和其他利益相关者进行互动。

比以往任何时候，Web 开发者都面临着诸如数据组织和消费等任务。内部和外部数据都需要我们拥有智能且以业务为导向的数据库驱动 Web 应用程序。

作为全栈软件工程师，你的前端任务之一将是消费数据，无论是来自内部开发的 API 还是第三方 API。在我们深入探讨在 React 项目中获取数据的方法或工具之前，让我们简要地讨论一下 API 是什么以及为什么它们正在重新定义构建用户界面和 Web 应用程序的方式。

API 简单地说就是允许系统之间通过一组标准接受的格式中的规则进行通信。在 Web 开发中，HTTP 协议定义了基于 Web 的系统通信的规则集。HTTP 是一种用于在互联网上获取资源的数据交换协议。

数据交换有两种主要格式：**XML**和**JSON**。JSON 在这两种广泛使用的数据交换格式中赢得了人气竞赛。JSON 专门设计用于数据交换，无缝处理数组，并且在开发者中被广泛使用。

在 React 生态系统内，开发者可以访问一系列公开的接口，旨在简化从各种来源获取数据。这些 API 旨在赋予 React 开发者创建直观用户界面并提升与 Web 应用程序交互的整体用户体验。

在本章中，我们将学习一些在 React 前端开发中用于从不同来源获取数据的方法和技术。在本章中，我们将涵盖以下主题：

+   在 React 中使用 Fetch API 获取数据

+   使用`async/await`语法获取数据

+   使用 Axios 获取数据

+   使用 React Query 获取数据

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter04`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter04).

# 在 React 中使用 Fetch API 获取数据

**Fetch API**是 Web 浏览器内建的一个 API，它提供了使用 HTTP 进行互联网通信的 JavaScript 接口。每个 Web 浏览器都有一个 JavaScript 引擎作为运行时来编译和运行 JavaScript 代码。

React 生态系统无疑依赖于 JavaScript。这是一个事实，也是为什么在深入 React 应用程序开发之前，你被期望理解现代 JavaScript 的原因之一。

作为一名 React 开发者，你需要网络资源来构建网络应用。`fetch()`方法为你提供了访问和操作 HTTP 对象请求以及 HTTP 协议响应的手段。假设在我们的网络应用中，我们想要显示会议演讲者和他们相关的数据。这些信息存储在另一个资源数据库服务器上。

从第三方公开 API 中，我们将消费用户的资源以获取假设数据，用于我们的 React 应用，如下所示：

```py
import React, { useEffect, useState } from 'react';const App = () => {
  const [data, setData] = useState([]);
  const getSpeakers = ()=>{
    fetch("https://jsonplaceholder.typicode.com/users")
       .then((response) => response.json())
       .then((data) => {
         setData( data);
       })
    }
    useEffect(() => {
      getSpeakers()
    },[]);
    return (
      <>
        <h1>Displaying Speakers Information</h1>
        <ul>
          {data.map(speaker => (
            <li key={speaker.id}>
              {speaker.name},  <em> {speaker.email} </em>
            </li>
          ))}
        </ul>
      </>
    );
};
export default App;
```

让我们详细讨论一下前面的`fetch`数据片段：

+   `import React, { useEffect, useState } from 'react'`：这一行导入了 React 的核心函数和一些 Hooks，用于在我们的组件中使用。

+   初始化`useState`：我们在组件中通过调用`useState`来初始化我们的状态，如下所示：

    ```py
    const [data, setData] = useState([]);//using a destructuring array to write concise code.
    ```

    `useState`接受一个初始状态为空数组（`useState([])`），并返回两个值，`data`和`setData`：

    +   `data`：当前状态

    +   `setData`：状态更新函数（此函数负责初始状态的新状态）

+   `useState([])`是带有初始值空数组`[]`的`useState`：

+   以下代码片段包含一个全局的`fetch()`方法，它接受端点 URL，`https://jsonplaceholder.typicode.com/users`，其中包含我们假设的资源，用于演讲者：

    ```py
    const getSpeakers = ()=>{  fetch("https://jsonplaceholder.typicode.com/users")    .then((response) => response.json())    .then((data) => {      setData( data);    })
    ```

    上一段代码中的 URL 是我们的资源端点。它返回 JSON 格式的数据。`setData()`函数接受新的状态，即返回的 JSON 数据。

+   使用`useEffect`钩子来调用`getSpeaker`函数：

    ```py
    useEffect(() => {getSpeakers()    },[]);
    ```

+   在数据数组上调用`map()`函数，用于遍历演讲者的数据并在屏幕上显示详细信息：

    ```py
    {data.map(speaker => (        <li key={speaker.id}>          {speaker.name},  <em> {speaker.email} </em>        </li>      ))}
    ```

总结来说，`fetch()`函数接受资源 URL（[`jsonplaceholder.typicode.com/users`](https://jsonplaceholder.typicode.com/users)）作为参数，这是我们所感兴趣的网络资源路径，并且当请求的资源响应可用时，返回一个状态为已满足的 Promise。

注意

在现实世界的应用中，有效地管理网络错误至关重要，尤其是在数据检索遇到问题或数据缺失时。此外，实现加载状态可以显著提升整体用户体验。

接下来，我们将探讨在 React 项目中使用`async/await`和 ECMAScript 2017 特性来获取数据的另一种技术。

# 使用`async/await`语法获取数据

在纯 JavaScript 中编写异步代码有三种方式：回调函数、Promise 和`async/await`。在本节中，我们将重点关注`async` `/await`，并探讨它如何在 React 网络应用中使用。`async/await`是对 Promise 的改进。

以下代码片段解释了如何使用基于 Promise 的方法通过`async/await`从 API 获取数据：

```py
import React, { useEffect, useState } from 'react';const App = () => {
    const [data, setData] = useState([]);
    const API_URL = "https://dummyjson.com/users";
    const fetchSpeakers = async () => {
        try {
            const response = await fetch(API_URL);
            const data = await response.json();
            setData(data.users);
        } catch (error) {
            console.log("error", error);
        }
    };
    useEffect(() => {
        fetchSpeakers();
    },[]);
    return (
      <> [Text Wrapping Break]
           <h1>Displaying Speakers Information</h1>
[Text Wrapping Break]
           <ul>
               {data.map(item => (
                   <li key={item.id}>
                       {item.firstName} {item.lastName}
                   </li>
               ))}
           </ul>
      </>
    );
};
export default App;
```

让我们讨论一下前面的代码片段，它展示了如何使用`async/await`异步获取数据：

+   `import React, { useEffect, useState } from 'react'`: 这行代码导入 React 核心函数和一些 Hooks 以用于我们的组件。

+   初始化 `useState`: 我们通过在组件中调用 `useState` 来初始化我们的状态，如下所示：

    ```py
    const [data, setData] = useState([]);//using a destructuring array to write a concise code.
    ```

    `useState` 接受一个空数组的初始状态（`useState([])`）并返回两个值，`data` 和 `setData`：

    +   `data`: 当前状态

    +   `setData`: 状态更新函数（此函数负责初始状态的新状态）

+   `useState([])` 是带有空数组初始值 `[]` 的 `useState`。

    ```py
    const API_URL = "https://dummyjson.com/users";  const fetchSpeakers = async () => {      try {          const response = await fetch(API_URL);          const data = await response.json();          setData(data.users);      } catch (error) {          console.log("error", error);      }  };
    ```

    在前面的端点中，我们有假设的演讲者资源。这是我们的资源端点。它返回 JSON 格式的数据。`setData()` 接受新的状态，即返回的 JSON 数据。

+   `useEffect` 钩子用于调用 `fetchSpeakers` 函数，该函数从端点 `const API_URL = "`[`dummyjson.com/users`](https://dummyjson.com/users)`"` 异步获取数据：

    ```py
      useEffect(() => {        fetchSpeakers();    },[data]);
    ```

    数组依赖项提供数据状态。当数据状态发生变化时，可能是由于列表中添加或删除演讲者，组件会重新渲染并显示更新后的状态。

+   最后，`map()` 在数据上被调用，它用于遍历演讲者的数据以将详细信息渲染到屏幕上：

    ```py
    return (        <>        <ul>      {data.map(item => (        <li key={item.id}>          {item.firstName} {item.lastName}        </li>      ))}    </ul>
    ```

使用 `async/await` 方法获取数据可以为你的代码提供更好的组织结构，并提高你的 React 应用的响应性和性能。`async/await` 的非阻塞模式意味着你可以在等待大量数据运行任务响应的同时继续执行其他代码操作。

接下来，我们将探讨另一种从 API 获取数据的方法，使用名为 Axios 的第三方 npm 包。

# 使用 Axios 获取数据

**Axios** 是一个轻量级的基于 Promise 的 HTTP 客户端，用于消费 API 服务。它主要用于浏览器和 Node.js。要在我们的项目中使用 Axios，请打开项目终端并输入以下命令：

```py
npm install axios
```

现在，让我们看看如何在以下代码片段中使用 Axios：

```py
import React, { useEffect, useState } from 'react';import axios from 'axios';
const App = () => {
    const [data, setData] = useState([]);
    const getSpeakers = ()=>{
        axios.get(
            "https://jsonplaceholder.typicode.com/users")
            .then(response => {
                setData(response.data)
            })
    }
    useEffect(() => {
        getSpeakers()
    },[]);
    return (
        <>
           <h1>Displaying Speakers Information</h1>
           <ul>
               {data.map(speaker => (
                   <li key={speaker.id}>
                       {speaker.name},  <em>
                           {speaker.email} </em>
                   </li>
               ))}
           </ul>
        </>
    );
};
export default App;
```

让我们检查前面的代码片段，看看 Axios 如何在数据获取中使用：

+   `import React, { useEffect, useState } from 'react'`: 这行代码导入 React 核心函数和一些 Hooks 以用于我们的组件。

+   `import axios from "axios"`: 这行代码将已安装的 Axios 包引入到项目中以便使用。

+   初始化 `useState`: 我们通过在组件中调用 `useState` 来初始化我们的状态，如下所示：

    ```py
    const [data, setData] = useState([]);//using a destructuring array to write a concise code.
    ```

+   `useState` 接受一个空数组的初始状态（`useState([])`）并返回两个值，`data` 和 `setData`：

    +   `data`: 当前状态

    +   `setData`: 状态更新函数（此函数负责初始状态的新状态）

+   `useState([])` 是带有空数组初始值 `[]` 的 `useState`。

    ```py
    const getSpeakers = ()=>{    axios.get(        "https://jsonplaceholder.typicode.com/users")        .then(response => {            setData(response.data)        })}
    ```

    这是我们的资源端点。它返回 JSON 格式的数据。`setData()` 接受新的状态，即返回的 JSON 数据。

    `getSpeakers` 函数使用 `axios.get()` 从端点获取外部数据并返回一个承诺。状态值被更新，我们从响应对象中获取一个新的状态 `setData`：

    ```py
    useEffect(() => {getSpeakers()    },[]);
    ```

+   使用 `useEffect` 钩子调用 `getSpeaker()` 并渲染组件：

    ```py
    <ul>    {data.map(speaker => (        <li key={speaker.id}>            {speaker.name},  <em> {speaker.email}                </em>        </li>    ))}</ul>
    ```

    最后，使用 `map()` 函数遍历演讲者的数据，并在屏幕上显示姓名和电子邮件。

接下来，让我们看看 React 中的数据获取技术，我们将探讨使用 React Query 获取数据的一种新方法。

# 在 React 中使用 React Query 获取数据

**React Query** 是一个用于数据获取目的的 npm 包库，其中包含大量功能。在 React Query 中，状态管理、数据预取、请求重试和缓存都是开箱即用的。React Query 是 React 生态系统的一个关键组件，每周下载量超过一百万。

让我们重构我们在 *使用 Axios 获取数据* 部分使用的代码片段，并体验 React Query 的神奇之处：

1.  首先，安装 React Query。在项目的根目录中，执行以下操作：

    ```py
    npm install react-query
    ```

1.  在 `App.js` 中添加以下内容：

    ```py
    import {useQuery} from 'react-query'import axios from 'axios';function App() {  const{data, isLoading, error} = useQuery(    "speakers",    ()=>{ axios(      "https://jsonplaceholder.typicode.com/users")  );  if(error) return <h4>Error: {error.message},    retry again</h4>  if(isLoading) return <h4>...Loading data</h4>  console.log(data);  return (      <>         <h1>Displaying Speakers Information</h1>         <ul>             {data.data.map(speaker => (                 <li key={speaker.id}>                     {speaker.name},  <em>                         {speaker.email} </em>                 </li>             ))}         </ul>      </>  );}export default App;
    ```

检查前面的 *使用 Axios 获取数据* 部分，比较代码片段。React Query 的代码片段要短得多，更简洁。`useState` 和 `useEffect` 钩子的需求已经被 `useQuery()` 钩子开箱即用处理。

让我们分析前面的代码：

+   `useQuery` 接受两个参数：查询键（`speakers`）和一个使用 `axios()` 从资源端点获取假设演讲者的回调函数。

+   `useQuery` 使用变量进行解构 – `{data, isLoading, error}`。然后我们检查是否有来自错误对象的错误消息。

+   一旦我们有了数据，`return()` 函数就返回一个演讲者数据的数组。

在 `index.js` 中添加以下代码。假设现有的 `index.js` 代码已经存在：

```py
import { QueryClient, QueryClientProvider } from    react-query";
const queryClient = new QueryClient();
root.render(
    <QueryClientProvider client={queryClient}>
        <App /> </QueryClientProvider>
);
```

让我们对 `index.js` 中的代码片段进行一些解释：

+   从 React Query 导入 `{ QueryClient, QueryClientProvider }`：`QueryClient` 允许我们在 React Query 中利用所有查询和变更的全局默认值。`QueryClientProvider` 连接到应用程序并提供一个 `QueryClient`。

+   创建一个新的 `QueryClient` 实例 `queryClient`：将组件包裹在 `QueryClientProvider` 中——在这个例子中，`<App/>` 是组件，并将新实例作为属性值传递。

如果 `localhost:3000` 没有运行，现在运行 `npm start`。屏幕上应该显示以下内容：

![图 4.1 – 屏幕截图展示了 React Query 在获取数据中的应用](img/Figure_4.1_B18554.jpg)

图 4.1 – 屏幕截图展示了 React Query 在获取数据中的应用

React Query 在从 API 资源获取数据方面非常有效。它封装了可能由 `useState` 和 `useEffect` 需要的函数。通过引入带有 `queryKey` 的强大缓存机制，React Query 根本重新定义了我们在 React 应用程序中获取数据的方式。

与手动管理数据获取和缓存不同，React Query 以透明的方式处理这些操作。React Query 允许开发者仅用几行代码就轻松地获取和缓存数据，从而减少样板代码并提高性能。

图书馆提供了各种钩子和实用工具，这些工具简化了数据获取、错误处理以及与服务器之间的数据同步，从而带来了更高效、更流畅的用户体验。进一步探索 React Query 可以打开处理复杂数据获取场景和优化 React 应用程序数据管理的新世界。

# 摘要

处理数据是任何网络应用程序的关键组成部分。React 已经证明在处理大量数据时非常高效和可扩展。在本章中，我们讨论了您可以在项目中利用的各种处理数据获取的方法。我们讨论了使用 Fetch API、`async/await`、Axios 和 React Query 获取数据。

在下一章中，我们将讨论 JSX 以及如何在 React 中显示列表。
