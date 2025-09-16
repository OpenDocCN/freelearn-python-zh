# 8

# 构建应用程序的前端

在上一章中，您探讨了如何构建您的 FastAPI 后端并连接到 MongoDB。这将用于本章中您将构建的 React 前端。该应用程序将简单且功能丰富，最重要的是，它将允许您看到堆栈的各个部分协同工作。

在本章中，您将构建一个全栈 FARM 应用程序的前端。您将学习如何设置 React Vite 应用程序并安装和设置 React Router，以及加载内容的各种方法。该应用程序将允许认证用户插入新项目（汽车），同时将有多页用于显示汽车。

您将开发一个网站，该网站将列出待售的二手车，并且只允许登录用户发布新的汽车广告。您将首先使用 Vite 创建一个 React 应用程序，然后使用 React Router 布局页面结构，并逐步引入认证、受保护页面和数据加载等功能。在本章之后，您将能够轻松地利用 React Router 为您的**单页应用程序**（**SPAs**）提供支持，并使用强大的**React Hook Form**（**RHF**）进行细粒度表单控制。

本章将涵盖以下主题：

+   使用 Vite 创建新的 React 应用程序

+   设置 React Router 以进行 SPA 页面导航

+   使用数据加载器管理数据

+   RHF 和 Zod 的数据验证简介

+   使用 Context API 进行认证和授权

+   使用 React Router 页面保护路由和显示数据

# 技术要求

本章的技术要求与*第四章**，*使用 FastAPI 入门*中列出的类似。您需要以下内容：

+   Node 版本 18.14

+   一个好的代码编辑器，例如 Visual Studio Code

+   节点包管理器

# 创建 Vite React 应用程序

在本节中，您将构建 Vite React 应用程序并设置 Tailwind CSS 进行样式化。此过程已在*第五章*，*设置 React 工作流程*中介绍，您可以参考它。请确保完成*第五章*中的简要教程，因为以下指南在很大程度上基于其中介绍的概念。

您将使用`create vite`命令与 Node 包管理器通过以下步骤创建您的项目：

1.  在包含先前创建的后端文件夹的项目目录中打开您的终端客户端，并执行以下命令以创建 Vite React 项目：

    ```py
    npm create vite@latest frontend-app -- --template react
    ```

1.  现在，将目录更改为新创建的`frontend-app`文件夹，并安装依赖项和 Tailwind：

    ```py
    npm install -D tailwindcss postcss autoprefixer
    ```

1.  初始化 Tailwind 配置——以下命令创建一个空的 Tailwind 配置文件：

    ```py
    npx tailwindcss init -p
    ```

1.  最后，根据最新的文档配置生成的 `tailwind.config.js` 和 React 的 `index.css` 文件，文档地址为 [`tailwindcss.com/docs/guides/vite`](https://tailwindcss.com/docs/guides/vite)。

您的 `index.css` 应现在只包含 Tailwind 的导入：

```py
@tailwind base;
@tailwind components;
@tailwind utilities;
```

为了测试 Tailwind 是否已正确配置，修改 `App.jsx` 文件并启动开发服务器：

```py
export default function App() {
  return ( <
    h1 className = “text-3xl font-bold” >
    Cars FARM <
    /h1>
  )
}
```

当您刷新应用程序时，您应该看到一个带有文本 **Cars FARM** 的白色页面。

在设置好一个功能性的 React 应用程序和 Tailwind 之后，是时候介绍可能最重要的第三方 React 包——React Router。

## React Router

到目前为止，由于您正在构建单页应用（SPA），所有组件都适合在单个页面上。为了使您的应用程序能够根据提供的路由显示完全不同的页面，您将使用一个名为 React Router 的包——在 React 中进行页面路由的事实标准。

虽然有一些非常好且健壮的替代方案，例如 TanStack Router ([`tanstack.com/router/`](https://tanstack.com/router/))，但 React Router 被广泛采用，了解其基本机制将极大地帮助您，作为一名开发者，因为您很可能会遇到基于它的代码。

React Router 的第 6.4 版本有一些重大变化，同时保留了之前的基本原则，您将使用这些原则来构建您的前端。然而，截至 2024 年 5 月，还宣布了更多激进的变化——**React Remix**，这是一个完整的全栈框架（具有与 Next.js 相当的功能），它基于 React Router，而 React Router 本身应该合并到一个单一的项目中。在本节中，您将了解最重要的组件，这些组件将允许您创建单页体验，无需页面重新加载或了解 React Router 6.4，这在以后将非常有用，因为它是最广泛采用的 React 路由解决方案。

React Router 的基本底层原理是监听 URL 路径变化（如 `/about` 或 `/login`），并根据条件在布局中显示组件。显示的组件可以被视为“页面”，而布局则保留了一些始终应显示的页面部分——例如页脚和导航。

在查看 React Router 之前，请回顾一下您应用程序中的页面：

+   `/`) 路径

+   `/cars`)

+   `/cars/car_id`)

+   `/login`)

+   **“插入新车辆”页面**：这将只为认证用户提供表单

为了简化，您将不包括注册路由（因为只有几个认证员工），前端也不会有删除或更新功能。在下一节中，您将安装和配置 React Router，并将其作为您应用程序的基础。

## 安装和设置 React Router

React Router 只是一个 Node.js 包，因此安装过程很简单。然而，在应用程序内部设置路由器包括许多功能和不同选项。你将使用最强大且推荐的带数据路由器，它提供数据加载，并且是 React Router 团队建议的选项。

使用路由器通常涉及两个步骤：

1.  使用提供的生成所需路线的方法之一（[`reactrouter.com/en/main/routers/picking-a-router`](https://reactrouter.com/en/main/routers/picking-a-router)）。

1.  创建组件，通常被称为 `Login.jsx` 和 `Home.jsx`。此外，你几乎总是会创建一个或多个布局，这些布局将包含如导航或页脚等常见组件。

现在，你将执行安装 React Router 到你的应用程序中所需的步骤：

1.  第一步，与任何第三方包一样，是安装 `router` 包：

    ```py
    npm i react-router-dom@6.23.1
    ```

    版本号对应于写作时的最新版本，因此你可以重现确切的功能。

    在本章中，应用程序的 CSS 样式将被有意保持到最小——仅足以区分组件。

1.  首先，在 `/src` 文件夹内创建一个名为 `/pages` 的新目录，并搭建所有你的页面。页面名称将是 `Home`、`Cars`、`Login`、`NewCar`、`NotFound` 和 `SingleCar`，所有这些都有 `.jsx` 扩展名，你将以与 `Home.jsx` 页面相同的方式搭建这些其他页面。

    位于 `/src/pages/Home.jsx` 的第一个组件将看起来像这样：

    ```py
    const Home = () => {
        return (
            <div>Home</div>
        )
    }
    export default Home
    ```

    虽然在讨论 React Router 时，它们通常被称为页面，但这些页面实际上不过是普通的 React 组件。这种区别，以及它们通常被组织在名为 `pages` 的目录中，纯粹是基于这些组件对应于单页应用（SPA）的页面结构，并且通常不打算在其他地方重用。

1.  在搭建好所需的页面后，实现路由器。此过程包括创建路由器并将其插入到顶级 React 组件中。你将使用 `App.jsx` 组件，该组件加载并插入整个 React 应用程序到 DOM 中。

自从 6.4 版本以来，React Router 引入了在需要数据的路由（或页面）加载之前获取数据的功能，通过简单的函数 `createBrowserRouter` 实现（[`reactrouter.com/en/main/routers/create-browser-router`](https://reactrouter.com/en/main/routers/create-browser-router)），因为它如文档所述，是所有 React Router 网络项目的推荐路由器。

在选择 `createBrowserRouter` 作为创建路由器的所需方法后，是时候将其集成到你的应用程序中了。

### 将路由器与应用程序集成

在以下步骤中，你将集成路由器到你的应用程序中，创建一个 `Layout` 组件，并将组件（页面）连接到每个定义的 URI：

1.  为了正确配置路由器，你需要另一个组件——`Layout` 组件，在其中将渲染之前创建的页面。在 `/src` 文件夹内，创建一个 `/layouts` 文件夹，并在其中创建一个 `RootLayout.jsx` 文件：

    ```py
    const RootLayout = () => {
      return (
        <div>RootLayout</div>
      )
    }
    export default RootLayout
    ```

    你将要使用的 React 路由器以及支持数据加载的路由器基于 `react-router-dom` 包中的三个导入：`createBrowserRouter`、`createRoutesFromElements` 和 `Route`。

1.  打开 `App.jsx` 文件并导入包和之前创建的页面：

    ```py
    import {
      createBrowserRouter,
      Route,
      createRoutesFromElements,
      RouterProvider
    } from “react-router-dom”
    import RootLayout from “./layouts/RootLayout”
    import Cars from “./pages/Cars”
    import Home from “./pages/Home”
    import Login from “./pages/Login”
    import NewCar from “./pages/NewCar”
    import SingleCar from “./pages/SingleCar”
    ```

1.  现在，继续使用相同的 `App.jsx` 文件，将你刚刚导入并定义的元素创建的路由连接起来：

    ```py
    const router = createBrowserRouter(
      createRoutesFromElements(
        <Route path=”/” element={<RootLayout />}>
          <Route index element={<Home />} />
          <Route path=”cars” element={<Cars />} />
          <Route path=”login” element={<Login />} />
          <Route path=”new-car” element={<NewCar />} />
          <Route path=”cars/:id” element={<SingleCar />} />
        </Route>
      )
    )
    export default function App() {
      return (
        <RouterProvider router={router} />
      )
    }
    ```

在前面的代码中，有几个重要的事项需要注意。在创建路由器后，你调用了名为 `createRoutesFromElements` 的 React Router 函数，该函数创建了实际的路线。路由用于定义与组件对应和映射的单独路径；它可以是一个自闭合标签（如用于页面的那些），或者它可以包含其他路由——例如主页路径，它反过来对应于 `RootLayout`。

如果你再次启动 React 服务器并访问页面 `http://localhost:5173`，你将只会看到文本 `RootLayout`。尝试导航到路由器中定义的任何路由：`/cars`、`/cars/333` 或 `/login`。你将看到相同的 `RootLayout` 文本，但如果你输入一个未定义的路径，例如 `/about`，React 将会显示一个类似于以下的消息来告知页面不存在：`意外应用程序错误！404 找不到`。

这意味着路由器确实在运行；它没有设置为处理用户导航到未定义路由的情况，并且不会显示页面内容。现在你将修复这两个问题。

### 创建布局和未找到页面

为了正常工作，路由器需要一个地方来显示页面内容——记住，“页面”只是 React 组件。现在你将创建 `Layout.jsx` 并处理用户访问不存在的 URI 导致的 `页面未找到` 错误的情况：

1.  首先，在 `/src/pages` 目录下创建一个新页面，命名为 `NotFound.jsx`，内容如下：

    ```py
    const NotFound = () => {
      return (
        <div>This page does not exist yet!</div>
      )
    }
    export default NotFound
    ```

    现在，创建一个通配符路由，当路径不匹配任何定义的路由时，将显示 `Not Found` 页面。记住路由的顺序很重要——React Router 将按顺序尝试匹配路由，因此使用 `*` 符号来捕获所有之前未定义的路由并将它们与 `NotFound` 组件关联是有意义的。

1.  更新 `App.jsx` 文件，将 `NotFound` 路由作为 `RootLayout` 路由中的最后一个路由显示：

    ```py
      createRoutesFromElements(
        <Route path=”/” element={<RootLayout />}>
          <Route index element={<Home />} />
    	// more routes here…
          <Route path=”*” element={<NotFound />} />
        </Route>
      )
    <Route path=”/” element={<RootLayout />}>
    ```

    所有其他页面都是嵌套的。你需要修改 `RootLayout`（即使对于非现有路由也会始终加载！）并为渲染特定页面组件提供 `Outlet` 组件。

1.  打开 `RootLayout.jsx` 并进行修改：

    ```py
    import { Outlet } from “react-router-dom”
    const RootLayout = () => {
        return (
            <div className=” bg-blue-200 min-h-screen p-2”>
                <h2>RootLayout</h2>
                <main className=”p-8 flex flex-col flex-1 bg-white “>
     <Outlet />
                </main>
            </div>
        )
    }
    export default RootLayout
    ```

    现在已经有了 `Outlet` 组件，你已经实现了路由。如果你尝试导航到路由器中定义的页面，你应该会看到页面更新，其中布局如之前所示，但 `Outlet` 组件会改变并显示 URL 中选择的页面内容。

    使用路由器的整个目的是通过“页面”进行导航，而无需重新加载页面。

1.  现在，为了最终完成 `RootLayout` 组件，你将更新组件并添加一些链接，使用提供的 React Router 的 `NavLink` 组件：

    ```py
    import {
      Outlet,
      NavLink
    } from “react-router-dom”
    const RootLayout = () => {
      return (
        <div className=” bg-blue-200 min-h-screen p-2”>
          <h2>RootLayout</h2>
          <header className=”p-8 w-full”>
            <nav className=”flex flex-row 
              justify-between”>
              <div className=”flex flex-row space-x-3”>
                <NavLink to=”/”>Home</NavLink>
                <NavLink to=”/cars”>Cars</NavLink>
                <NavLink to=”/login”>Login</NavLink>
                <NavLink to=”/new-car”>New Car</NavLink>
              </div>
            </nav>
          </header>
          <main className=”p-8 flex flex-col flex-1
            bg-white “>
            <Outlet />
         </main>
       </div>
      )
    }
    export default RootLayout
    ```

现在你已经实现了简单的导航，并且当需要时 `NotFound` 页面会加载。路由器还提供了导航历史，因此浏览器的后退和前进按钮是可用的。应用样式故意简约，仅用于强调不同的组件。

到目前为止，你只有一个布局，但可能还有更多——一个是汽车列表页面和单个汽车页面——嵌入到主布局中。就像 FastAPI 中的 APIRouters 一样，React 路由和布局可以嵌套。React Router 的嵌套是一个强大的功能，它能够构建只加载或更新必要组件的分层网站。

在设置好 React Router 之后，让我们探索一个仅在使用数据路由时才可用的重要功能，例如你使用的——数据加载器——允许开发者以更有效的方式访问数据的特殊函数。

## React Router 加载器

加载器是简单的函数，可以在路由加载之前提供数据（[`reactrouter.com/en/main/route/loader`](https://reactrouter.com/en/main/route/loader)）通过一个简单的 React 钩子。

为了使用一些数据，首先创建一个新的 `.env` 文件，并添加你 Python 后端的地址：

```py
VITE_API_URL=http://127.0.0.1:8000
```

如果你现在重启服务器，Vite 将能够获取你代码中的地址，URI 将在 `import.meta.env.VITE_API_URL` 中可用。

注意

要了解更多关于 Vite 如何处理环境变量的信息，请查看他们的文档：[`vitejs.dev/guide/env-and-mode`](https://vitejs.dev/guide/env-and-mode)。

现在，你将学习 React Router 如何管理数据加载和预取。执行以下步骤，将后端数据加载到 React 应用程序中，并学习如何使用强大且简单的 `useLoader` 钩子。

首先，处理 `/src/pages/Cars.jsx` 组件，看看数据加载器如何帮助你管理组件数据：

1.  创建一个 `src/components` 文件夹，并在其中创建一个简单的静态 React 组件，名为 `CarCard.jsx`，用于显示单个汽车：

    ```py
    const CarCard = ({ car }) => {
      return (
        <div className=”flex flex-col p-3 text-black 
          bg-white rounded-xl overflow-hidden shadow-md
          hover:scale-105 transition-transform
          duration-200”>
          <div>{car.brand} {car.make} {car.year} {car.cm3}
            {car.price} {car.km}
          </div>
          <img src={car.picture_url} alt={car.make}
            className=”w-full h-64 object-cover
            object-center” />
        </div>
      )
    }
    export default CarCard
    ```

    在处理完 `Card` 组件后，你现在可以查看数据加载器是如何工作的。

    加载器是函数，在组件渲染之前向路由器中的组件提供数据。这些函数通常由同一组件定义和导出，尽管这不是强制性的。

1.  打开 `Cars.jsx` 并相应地更新它：

    ```py
    import { useLoaderData } from “react-router-dom”
    import CarCard from “../components/CarCard”
    const Cars = () => {
      const cars = useLoaderData()
      return (
        <div>
          <h1>Available cars</h1>
          <div className=”md:grid md:grid-cols-3 sm:grid
            sm:grid-cols-2 gap-5”>
            {cars.map(car => (
              <CarCard key={car.id} car={car} />
            ))}
          </div>
        </div>
      )
    }
    export default Cars
    ```

    组件导入 `useLoaderData`——这是 React Router 提供的一个自定义钩子，其唯一目的是将加载函数的数据提供给需要它的组件。这种范式是 React Remix 的核心，类似于一些 Next.js 功能，因此了解它是有用的。`useLoader` 函数将包含来自服务器的数据，通常以 JSON 格式。

1.  现在，在同一文件中也将 `carsLoader` 函数导出：

    ```py
    export const carsLoader = async () => {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL}/cars?limit=30`
        )
      const response = await res.json()
      if (!res.ok){
        throw new Error(response.message)
      }
      return response[‘cars’]
    }
    ```

注意

这两个部分——组件和函数——尚未连接。这种连接必须在路由器中发生，并允许在路由器级别预加载数据。

1.  现在，你将通过路由器将组件和加载器连接起来。打开 `App.jsx` 文件，通过向 `/cars` 路由提供加载器参数来修改代码：

    ```py
    import Cars, { carsLoader } from “./pages/Cars”
    // continues
      <Route path=”/” element={<RootLayout />}>
        <Route index element={<Home />} />
        <Route path=”cars” element={<Cars />}     
          loader={carsLoader} />
          <Route path=”login” element={<Login />} />
          <Route path=”new-car” element={<NewCar />} />
          <Route path=”cars/:id” 
            element={<SingleCar />} />
          <Route path=”*” element={<NotFound />} />
        </Route>
    ```

现在加载函数已经就位，你可以测试你的 `/cars` 页面了，它应该显示到目前为止保存的汽车集合。

接下来的几节将探讨实现你在每个 React（或 Next.js，或一般意义上的 web 开发）项目中都可能遇到的功能——使用 RHF 处理表单。你将借助处理 React 表单最流行的第三方包来实现登录功能，并使用 Zod 包进行数据验证。

# React Hook Form 和 Zod

处理 React 表单有许多方法，其中最常见的一种模式在第 *第五章* 中展示，即 *设置 React 工作流程*。状态变量使用 `useState` 钩子创建，表单提交被阻止并拦截，最后数据通过 JSON 或表单数据传递。当处理简单数据和少量字段时，这种工作流程是可以接受的，但在需要跟踪数十个字段、它们的约束和可能状态的情况下，它很快就会变得难以管理。

RHF 是一个成熟的项目，拥有繁荣的社区，它与其他类似库的区别在于其速度、渲染量最小以及与 TypeScript 和 JavaScript 中最受欢迎的数据验证库（如 Zod 和 Yup）的深度集成。在这种情况下，你将学习 Zod 的基础知识。

## 使用 Zod 进行数据验证

目前，JavaScript 和 TypeScript 生态系统中有几个验证库——Zod 和 Yup 可能是最受欢迎的。Zod 是一个以 TypeScript 为首的架构声明和验证库，它提供了数据结构的验证。Zod 为 JavaScript 应用程序中的对象和值提供了简单直观的对象语法，以创建复杂的验证规则，并极大地简化了确保应用程序数据完整性的过程。

这些包的基本思想是提供所需数据结构的原型，并对数据与定义的数据结构进行验证：

1.  首先，安装该包：

    ```py
    npm i react-hook-form@7.51.5
    ```

    由于撰写本文时和书中仓库中使用的版本号是 7.51.5，如果你想重现仓库中的确切代码，请使用前面的命令。

1.  更新 `Login.jsx` 组件，使其显示 `LoginForm`，你将在稍后创建它：

    ```py
    import LoginForm from “../components/LoginForm”
    const Login = () => {
      return (
      <div>
        <h1>Login</h1>
        <LoginForm />
      </div>
      )
    }
    export default Login
    ```

1.  现在，`/src/components/LoginForm.jsx` 文件将包含所有表单功能以及使用 Zod 的数据验证：

    ```py
    import { useForm } from “react-hook-form”
    import { z } from ‘zod’;
    import { zodResolver } from ‘@hookform/resolvers/zod’;
    const schema = z.object({
      username: z.string().min(4, ‘Username must be at least 4 characters long’).max(10, ‘Username cannot exceed 10 characters’),
      password: z.string().min(4, ‘Password must be at least 4 characters long’).max(10, ‘Password cannot exceed 10 characters’),
    });
    ```

    组件开始于导入——`useForm` 钩子、Zod 以及与表单钩子集成的 Zod 解析器。在 Zod 中的数据验证类似于 Pydantic 中的方式——你定义一个对象，并在各个字段上设置所需的属性。在这种情况下，我们设置用户名和密码长度在 4 到 10 个字符之间，但 Zod 允许进行一些非常复杂的验证，正如你可以在他们的网站上看到的那样（[`zod.dev/`](https://zod.dev/))。

    `useForm` 钩子提供了几个有用的函数：

    +   `register` 用于使用钩子注册单个表单字段

    +   `handleSubmit` 是提交时将被调用的函数

    +   `formState` 包含有关表单状态的不同信息（[`react-hook-form.com/docs/useform/formstate`](https://react-hook-form.com/docs/useform/formstate)）

1.  现在，设置 `form` 钩子：

    ```py
    const LoginForm = () => {
      const { register, handleSubmit, 
        formState: { errors } } = useForm({
          resolver: zodResolver(schema),
        });
      const onSubmitForm = (data) => {
          console.log(data)
        }
    ```

    在这种情况下，你将只跟踪错误（与之前用 Zod 定义的验证相关），但这个对象跟踪的内容要多得多。在你的代码中，一旦验证通过，你只需将数据输出到控制台即可。

1.  现在，构建表单的 JSX 并添加一些样式以查看发生了什么：

    ```py
    return (
      <div className=”flex items-center justify-center”>
        <div className=”w-full max-w-xs”>
        <form className=”bg-white shadow-md rounded 
          px-8 pt-6 pb-8 mb-4”                      
          onSubmit={handleSubmit(onSubmit event is bound to the handle. This process is quite simple: the form has an onSubmit method that you handed over to the handleSubmit method of RHF. This handleSubmit method is destructured from the hook itself, along with the register function (for mapping input fields) and the errors that reside in the form state. After establishing the connection, the handleSubmit method needs to know which function should process the form and its data. In this case, it should pass the handling to the onSubmitForm function.The two form fields, for the username and the password, are nearly identical:

    ```

    `<div className=”mb-4”>`

    `<label htmlFor=”username” className=”block`

        `text-gray-700 text-sm font-bold mb-2”>

        用户名

    `<label>`

    `<input id=”username” type=”text”`

        `placeholder=”Username” required`

    `{...register(‘username’)}`

        `className=”shadow appearance-none border`

        `rounded w-full py-2 px-3 text-gray-700`

        `leading-tight focus:outline-none`

        `focus:shadow-outline”/>

        `{errors.username && <p className=”text-red-500

    `text-xs italic”>{errors.username.message}</p>}

    </div>

    ```py

    ```

代码中突出显示的部分是使用 `useForm` 钩子注册字段——这是让表单知道预期哪些字段以及与各自字段相关的错误（如果有的话）的一种方式。

这样，字段通过这个扩展运算符语法注册到钩子表单中。由于表单提供的错误绑定到字段上，利用这个机会，将它们显示在报告错误的字段旁边，以提供更好的用户体验。

组件的其余部分直观易懂，涵盖了`密码`字段和`提交`按钮：

```py
<div className=”mb-6”>
  <label htmlFor=”password” className=”block text-gray-700   
    text-sm font-bold mb-2”>Password</label>
  <input id=”password” type=”password” placeholder=”****”
    required
    {...register(‘password’)}
    className=”shadow appearance-none border rounded w-full
    py-2 px-3 text-gray-700 mb-3 leading-tight
    focus:outline-none focus:shadow-outline” />
  {errors.password && <p className=”text-red-500 text-xs
 italic”>{errors.password.message}</p>}
</div>
<div className=”flex items-center justify-between”>
          <button type=”submit”>Sign In</button>
        </div>
      </form>
    </div>
  </div>
  )
}
export default LoginForm
```

书中的完整代码可在书库中找到。

表单现在已准备就绪，并由带有 Zod 验证的钩子表单完全处理。如果您尝试输入不符合验证标准的数据（例如用户名或密码少于四个字符），您将在字段旁边收到错误消息。在设置登录表单后，您将创建一个认证上下文，允许用户保持登录状态。认证过程——创建 React 上下文和存储 JWT——将与*第六章*“认证和授权”中介绍的过程非常相似，因此下一节仅涵盖并突出代码中的重要部分。

# 认证上下文和存储 JWT

在本节中，您将使用由 RHF 提供动力的全新表单，并将其连接到上下文 API。定义 React Context API 的流程在*第四章*“FastAPI 入门”中进行了详细说明，本章中，您将应用这些知识并创建一个类似上下文来跟踪应用程序的认证状态：

1.  在`/src`目录中创建一个新的文件夹，命名为`contexts`。在此文件夹内，创建一个名为`AuthContext.jsx`的新文件，并创建提供者：

    ```py
    import { createContext, useState, useEffect } from ‘react’;
    import { Navigate } from ‘react-router-dom’;
    export const AuthContext = createContext();
    export const AuthProvider = ({ children }) => {
      const [user, setUser] = useState(null);
      const [jwt, setJwt] =  useState(localStorage.getItem('jwt')||null);
      const [message, setMessage] = useState(
        “Please log in”
      );
    ```

    您正在创建的上下文相当简单，包含一些状态变量和设置器，这些变量和设置器将用于认证流程：用户名（其存在或不存在将指示用户是否已认证）、JWT 以及一个辅助消息，在这种情况下，它仅用于调试和说明。

    初始值通过`useState`钩子设置为`null`和通用消息——用户名设置为`null`，JWT 设置为空字符串，消息设置为“请登录”。

1.  接下来，添加一个`useEffect`钩子，它将在上下文加载或页面重新加载时触发：

    ```py
    useEffect(() => {
      const storedJwt = localStorage
        .getItem(‘jwt’);
      if(storedJwt) {
        setJwt(storedJwt);
        fetch(
          `${import.meta.env.VITE_API_URL}/users/me`, {
          headers: {
          Authorization: `Bearer ${storedJwt}`,
          },
            })
        .then(res => res.json())
    ```

    `useEffect`钩子的第一部分检查本地存储中是否存在 JWT。如果存在，`useEffect`钩子将对 FastAPI 服务器执行 API 调用，以确定 JWT 是否能够返回有效的用户：

    ```py
    .then(data => {
      if(data.username) {
        setUser({user: data.username});
        setMessage(`Welcome back, ${data.username}!`);
      } else {
    ```

    如果令牌无效或被篡改或已过期，`useEffect`钩子将其从本地存储中删除，将上下文状态变量设置为`null`，并设置适当的消息给用户：

    ```py
    localStorage.removeItem(
      ‘jwt’);
    setJwt(null);
    setUser(null);
    setMessage(data.message)
    }
    })
    .catch(() => {
      localStorage
        .removeItem(
          ‘jwt’);
      setJwt(null);
      setUser(null);
      setMessage(
        ‘Please log in or register’
      );
    });
    }
    else {
      setJwt(null);
      setUser(null);
      setMessage(
        ‘Please log in or register’
      );
    }
    }, []); };
    ```

总结一下，`useEffect`钩子执行了一个周期。首先，它检查本地存储，如果没有找到 JWT，它将从上下文中删除 JWT，将用户名设置为`null`，并提示用户登录。如果使用现有 JWT 对`/me`路由的 API 调用没有返回有效的用户名，也会得到相同的结果。这意味着令牌存在，但无效或已过期。如果 JWT 确实存在并且可以用来获取有效的用户名，那么将设置用户名并将 JWT 存储在*上下文*中。由于依赖项数组为空，此钩子将在第一次渲染时只运行一次。

## 实现登录功能

为了简单起见，登录函数将再次位于上下文中，尽管它也可以在单独的文件中。以下为登录流程：

1.  用户提供他们的用户名和密码。

1.  执行了对后端的 fetch 调用。

1.  如果响应具有 HTTP 状态`200`并且返回了 JWT，则`localStorage`和上下文都会被设置，用户被认证。

1.  如果响应没有返回 HTTP 状态`200`，这意味着登录信息未被接受，在这种情况下，上下文中的 JWT 和用户名值都被设置为`null`，从而有效失效。

要实现登录功能，执行以下步骤：

1.  首先，`login`函数需要调用带有提供的用户名和密码的登录 API 路由。将以下代码粘贴到`AuthContext.jsx`文件的末尾：

    ```py
    const login = async (username,
      password) => {  const response = await fetch(`${import.meta.env.VITE_API_URL}/users/login`, {
          method: ‘POST’,
          headers: {
            ‘Content-Type’: ‘application/json’,
          },
          body: JSON.stringify({
            username,
            password
          }),
        });
    ```

1.  接下来，根据响应，函数将相应地设置上下文中的状态变量：

    ```py
      const data = await response
        .json();
      if(response.ok) {
        setJwt(data.token);
        localStorage.setItem(‘jwt’, data
          .token);
        setUser(data.username);
        setMessage(
          `Login successful: welcome  ${data.username}`
        );
      } else {
        setMessage(‘Login failed: ‘ +
          data.detail);
        setUser(null)
        setJwt(null);
        localStorage.removeItem(‘jwt’);
      }
      return data
    };
    ```

    逻辑与`useEffect`钩子中应用的方法类似——如果找到有效的用户，上下文状态变量（用户名和 JWT）将被设置；否则，它们将被设置为`null`。

1.  最后的部分只是`logout`函数和上下文提供者的返回。下面的`logout`函数是在`AuthProvider`内部定义的：

    ```py
        const logout = () => {
          setUser(null);
          setJwt(null);
          localStorage.removeItem(‘jwt’);
          setMessage(‘Logout successful’);
        };
        return ( <
          AuthContext.Provider value = {
            {
              username,
              jwt,
              login,
              logout,
              message,
              setMessage
            }
          } > {
            children
          } <
          /AuthContext.Provider>
        );
    ```

到目前为止，你已经完成了相当多的事情：你设置了上下文，定义了登录和注销函数，并创建了上下文提供者。现在，为了方便使用上下文，你将创建一个简单的自定义 React 钩子，基于内置的`useContext`钩子。

### 创建用于访问上下文的自定义钩子

在设置好 Context API 之后，你现在可以继续创建一个位于新文件夹`/src/hooks`中的`useAuth.jsx`文件，这将允许从各个地方轻松访问上下文：

1.  在新文件夹中创建`useAuth.jsx`文件：

    ```py
    import {
      useContext
    } from “react”;
    import {
      AuthContext
    } from “../contexts/AuthContext”;
    export const useAuth = () => {
      const context = useContext(
        AuthContext)
      if (!context) {
        throw new Error(
          ‘Must be used within an AuthProvider’
          )
      }
      return context
    }
    ```

    如果钩子不在上下文中访问，`useAuth`钩子将包含一个错误消息——但你的上下文将包围整个应用程序。

    使用 React 上下文的最后一步是将需要访问它的组件包裹起来；在你的情况下，这将涉及`App.jsx`——根组件。

1.  打开`App.jsx`文件，并将当前返回的唯一组件`RouterProvider`包裹在`AuthProvider`内部：

    ```py
    import { AuthProvider } from “./contexts/AuthContext”
    // continues
    export default function App() {
      return (
        <AuthProvider>
          <RouterProvider router={router} />
        </AuthProvider>
      )
    }
    ```

    最后，在当前托管所有页面的`RootLayout`组件中显示上下文数据和状态变量。这是在使用 React Context API 时的一种有用的调试技术；你不需要频繁地在开发者工具之间切换。

1.  打开`RootLayout.jsx`并编辑文件：

    ```py
    import { Outlet, NavLink } from “react-router-dom”
    import { useAuth } from “../hooks/useAuth”
    const RootLayout = () => {
        logout function, you can now add a little bit of JSX conditional rendering and create a dynamic menu:

    ```

    const RootLayout = () => {

    const {

        user,

        消息，

        logout

    } = useAuth()

    return (

        <div className=” bg-blue-200 min-h-screen p-2”>

        <h2>根布局</h2>

        <p className=”text-red-500 p-2 border”>

            {message}

        </p>

        <p>用户名: {user}</p>

        <header className=”p-3 w-full”>

        <nav className=”flex flex-row justify-between

            mx-auto”>

        <div className=”flex flex-row space-x-3”>

            <NavLink to=”/”>主页</NavLink>

            <NavLink to=”/cars”>汽车</NavLink>

            {user ? <>

            <NavLink to=”/new-car”>新车</NavLink>

            <button onClick={logout}>登出</button>

            </> : <>

            <NavLink to=”/login”>登录</NavLink>

            </>}

        </div>

        </nav>

    </header>

    <main className=”p-8 flex flex-col flex-1

        bg-white “>

        <Outlet />

    </main>

        </div>

    )

    }

    export default RootLayout

    ```py

    ```

该应用程序相当简单，但很好地展示了登录/登出过程。作为一个练习，你可以轻松实现注册页面——API 端点已经存在，你应该创建处理注册表单的逻辑。

以下部分将专注于完成一些更多功能——插入新车的路由对于未登录的用户仍然可访问，而且表单还不存在。现在你将保护资源创建端点，并使用 React Router 创建受保护页面。

## 保护路由

受保护的路由是所有人无法访问的路由和页面——它们通常要求用户登录或拥有某些权限（管理员或创建者）。在 React Router 中有很多种保护路由的方法。一种流行的模式是通过高阶组件——它们是包裹需要登录用户的路由的包装组件。新的 React Router 及其`Outlet`组件允许你轻松实现门控逻辑，并在需要授权时重定向用户。

创建一个基本的组件，用于检查用户的存在（通过用户名）。如果用户存在，该组件将使用`Outlet`组件让被包裹的路由到达浏览器；否则，将重定向到登录页面：

1.  在`/src/components`文件夹中创建一个新的组件，命名为`AuthRequired.jsx`：

    ```py
    import {
      Outlet,
      Navigate
    } from “react-router-dom”
    import {
      useAuth
    } from “../hooks/useAuth”
    const AuthRequired = () => {
      const {
        jwt
      } = useAuth()
      return (
        <div>
                <h1>AuthRequired</h1>
                {jwt ? <Outlet /> : <Navigate to=”/login” />}
            </div>
      )
    }
    export default AuthRequired
    ```

    逻辑很简单；该组件确保你执行 JWT 存在性检查。然后它像一个信号量或简单的 IF 结构，检查条件——如果 JWT 存在，`Outlet`组件将显示封装的组件（在我们的例子中只有一个：`NewCar`页面），如果不存在，则使用 React Router 的`Navigate`组件进行程序性导航到主页。

    这个简单的解决方案不会强制认证用户在重新加载受保护的页面时被重定向到主页，因为`Layout.jsx`中的`useEffect`钩子只有在组件加载后才会检测 JWT 是否无效。如果 JWT 确实无效，`useEffect`钩子将使 JWT 无效，从而触发重定向。

1.  现在，更新`App.jsx`组件，导入`AuthRequired`组件，并将`NewCar`页面包围起来：

    ```py
    import AuthRequired from “./components/AuthRequired”
    import { AuthProvider } from “./contexts/AuthContext”
    // code continues
    const router = createBrowserRouter(
      createRoutesFromElements(
        <Route path=”/” element={<RootLayout />}>
          <Route index element={<Home />} />
          <Route path=”cars” element={<Cars />} loader={carsLoader} />
          <Route path=”login” element={<Login />} />
     <Route element={<AuthRequired />}>
     <Route path=”new-car” element={<NewCar />} />
     </Route>
          <Route path=”cars/:id” element={<SingleCar />} />
    ```

你已经学会了如何保护需要认证的路由。现在，你将构建另一个表单来插入关于新车的数据，并通过 FastAPI 将图像（每辆车一张图像）上传到 Cloudinary。

## 创建插入新车的页面

插入新车到集合的页面——`NewCar.jsx`组件——是受保护的，并且只能由认证用户访问。在本节中，你将构建一个更复杂的表单，并逐步模块化代码：

1.  首先，更新`NewCar.jsx`页面并添加一个`CarForm`组件，你很快就会构建它：

    ```py
    import CarForm from “../components/CarForm”
    const NewCar = () => {
        return (
            <div>
     <CarForm />
            </div>
        )
    }
    export default NewCar
    ```

1.  现在，在`/src/components`文件夹中创建这个组件。在这个文件夹中，创建一个新文件并命名为`CarForm.jsx`。在开始编写表单代码之前，快速回顾一下表单需要收集哪些类型的数据并将其发送到 API：

    +   **Brand**: 字符串

    +   **Make**: 字符串

    +   **Year**: 整数

    +   **Price**: 整数

    +   **Km**: 整数

    +   **Cm3**: 整数

    +   **Picture**: 文件对象

    如果将表单中的每个字段都作为单独的输入创建，并且只是将所有内容复制粘贴到文件中，将会非常繁琐且重复。相反，你可以抽象输入字段并使其成为一个可重用的组件。这个组件将需要接受一些属性，例如名称和类型（数字或字符串），并且 RHF 可以将其注册并关联到该字段上的任何错误。因此，在开始表单之前，创建另一个将被多次重用的组件，这将显著简化创建和更新表单的过程——在实际场景中，汽车可能有数百个字段。

1.  在`/src/components`文件夹中创建一个新文件并命名为`InputField.jsx`：

    ```py
    const InputField = ({ props }) => {
      const { name, type, error } = props;
      return (
        <div className=”mb-4”>
          <label
            className=”block text-gray-700 text-sm mb-2”
            htmlFor={name}
          >
            {name}
          </label>
          <input
            className=”shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline”
            id={name}
            name={name}
            type={type}
            placeholder={name}
            required
            autoComplete=”off”
            {...props}
          />
          {error && <p className=”text-red-500 text-xs italic”>{error.message}</p>}
        </div>
      );
    };
    export default InputField;
    ```

    字段组件简单而有用——它抽象了所有功能，甚至添加了一些样式。

1.  现在，回到`CarForm`文件并开始导入：

    ```py
    import { useForm } from “react-hook-form”
    import { z } from ‘zod’;
    import { zodResolver } from ‘@hookform/resolvers/zod’;
    import { useNavigate } from “react-router-dom”;
    import { useAuth } from “../hooks/useAuth”;
    import InputField from “./InputField”;
    ```

1.  你将再次使用 Zod 进行数据验证，因此添加一个模式——它应该理想地与后端的 Pydantic 验证规则相匹配以保持一致性：

    ```py
    const schema = z.object({
        brand: z.string().min(2, ‘Brand must contain at least two letters’).max(20, ‘Brand cannot exceed 20 characters’),
        make: z.string().min(1, ‘Car model must be at least 1 character long’).max(20, ‘Model cannot exceed 20 characters’),
        year: z.coerce.number().gte(1950).lte(2025),
        price: z.coerce.number().gte(100).lte(1000000),
        km: z.coerce.number().gte(0).lte(500000),
        cm3: z.coerce.number().gt(0).lte(5000),
        picture: z.any()
            .refine(file => file[0] && file[0].type.startsWith(‘image/’), { message: ‘File must be an image’ })
            .refine(file => file[0] && file[0].size <= 1024 * 1024, { message: ‘File size must be less than 1MB’ }),
    });
    ```

    Zod 模式语法相当直观，尽管可能有一些方面需要小心——数字需要被强制转换，因为 HTML 表单默认发送字符串，并且可以通过方便的函数验证文件。

1.  现在，开始实际的表单组件：

    ```py
    const CarForm = () => {
        const navigate = useNavigate();
        const { jwt } = useAuth();
        const { register, handleSubmit, 
        formState: { errors, isSubmitting } } = useForm({
            resolver: zodResolver(schema),
        });
    ```

    `useNavigate`钩子用于在提交完成后从页面导航离开，而`useForm`与用于登录用户的钩子类似。

1.  创建一个简单的 JavaScript 数组，包含表单所需的字段数据：

    ```py
        let formArray = [
            {
                name: “brand”,
                type: “text”,
                error: errors.brand
            },
            {
                name: “make”,
                type: “text”,
                error: errors.make
            },
            {
                name: “year”,
                type: “number”,
                error: errors.year
            },
            {
                name: “price”,
                type: “number”,
                error: errors.price
            },
            {
                name: “km”,
                type: “number”,
                error: errors.km
            },
            {
                name: “cm3”,
                type: “number”,
                error: errors.cm3
            },
            {
                name: “picture”,
                type: “file”,
                error: errors.picture
            }
        ]
    ```

1.  使用这个数组，表单代码变得更加易于管理。看看`onSubmit`函数：

    ```py
    const onSubmit = async (data) => {
      const formData = new FormData();
      formArray.forEach((field) => {
        if (field == “picture”) {
          formData.append(field, data[field][0]);
        } else {
          formData.append(field.name, data[field.name]);
        }
      });
    };
    ```

    突然，`onSubmit`函数变得更加简洁——它遍历数组并将字段添加到`formData`对象中。记住，`file`字段是特殊的——它是一个数组，你只想获取第一个元素，即图片。

1.  为了完成`onSubmit`函数，你需要向 API 发送`POST`请求：

    ```py
    const result = await fetch(`${import.meta.env.VITE_API_URL}/cars/`, {
      method: “POST”,
      body: formData,
      headers: {
        Authorization: `Bearer ${jwt}`,
      },
    });
    const json = await result.json();
    if (result.ok) {
      navigate(“/cars”);
    } else if (json.detail) {
      setMessage(JSON.stringify(json));
      navigate(“/”);
    }
    ```

    获取调用很简单。在你得到结果后，你可以应用自定义逻辑。在这种情况下，你将 JSON 化——将错误对象渲染为 JSON 字符串，并将消息设置为显示它，如果错误来自服务器。

1.  最后，由于你的`InputField`组件和`formArray`，JSX 变得非常简单，同时你也使用了`useForm`钩子中的提交值：

    ```py
    return (
      <div className=”flex items-center justify-center”>
        <div className=”w-full max-w-xs”>
          <form
            className=”bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4 “
            encType=”multipart/form-data”
            onSubmit={handleSubmit(onSubmit)}
          >
            <h2 className=”text-center text-2xl font-bold mb-6”>Insert new car</h2>
            {formArray.map((item, index) => (
              <InputField
                key={index}
                props={{
                  name: item.name,
                  type: item.type,
                  error: item.error,
                  ...register(item.name),
                }}
              />
            ))}
            <div className=”flex items-center justify-between”>
              <button
                className=”bg-gray-900 hover:bg-gray-700 text-white w-full font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline”
                type=”submit”
                disabled={isSubmitting}
              >
                {isSubmitting ? “Saving...” : “Save new car”}
              </button>
            </div>
          </form>
        </div>
      </div>
    );}
    export default CarForm
    ```

提交按钮现在被重用作为提交指示器——在提交时显示不同的消息，并且也被禁用以防止多次请求。

创建一个用于更新汽车的页面将与之前的端点非常相似——RHF 与可以从现有对象中填充的初始或默认数据配合得非常好，你还可以使用在线表单构建器：[`react-hook-form.com/form-builder`](https://react-hook-form.com/form-builder)。删除汽车也相对简单，因为请求只需要认证并包含汽车 ID。

你现在已经创建了一个汽车创作页面，它可以以多种方式扩展。你已经学会了如何模块化你的 React 代码，以及如何根据数据流向和从服务器提供有意义的信息和逻辑给你的应用程序。现在你将创建一个用于显示单个汽车的页面，并再次使用加载器。

## 显示单个汽车

现在你已经创建了用于显示多个项目（汽车）、认证和创建新项目的页面，创建一个单独的汽车页面，看看 React Router 是如何处理 URL 中的参数的：

1.  编辑`SingleCar.jsx`文件，并引入`useLoaderData`钩子，它已经在汽车页面中用于预加载数据：

    ```py
    import { useLoaderData } from “react-router-dom”;
    import CarCard from “../components/CarCards”;
    const SingleCar = () => {
        const car = useLoaderData()
        return (
            <CarCard car={car} />
        );
    };
    export default SingleCar
    ```

    为了节省空间，我们重用了`CarCard`函数来显示汽车的数据。然而，在现实场景中，这个页面可能包含一个图片库、更多的数据，也许还有一些评论或笔记等等。但在这里的目标只是展示创建加载器函数的另一种方式。

1.  打开当前托管路由器的`App.jsx`文件，并更新`cars/:id`路由，记住冒号表示一个参数，在这种情况下，是 MongoDB 集合中汽车`ObjectId`组件的字符串版本：

    ```py
    import fetchCarData from “./utils/fetchCarData”
    // continues
    const router = createBrowserRouter(
      createRoutesFromElements(
        <Route path=”/” element={<RootLayout />}>
          <Route index element={<Home />} />
          <Route path=”cars” element={<Cars />} loader={carsLoader} />
          <Route path=”login” element={<Login />} />
          <Route element={<AuthRequired />}>
            <Route path=”new-car” element={<NewCar />} />
          </Route>
          <Route
            path=”cars/:id”
            element={<SingleCar />}
     loader={async ({ params }) => {
     return fetchCarData(params.id);
     }}
     errorElement={<NotFound />} />
          <Route path=”*” element={<NotFound />} />
        </Route>
      )
    )
    ```

    路线上只有两个变化：一个是`loader`函数，它是作为异步函数的一部分提供的，该函数接收参数 ID，另一个是`errorElement`。如果`loader`函数在获取数据时遇到错误，将显示`NotFound`组件。在这里，你再次重用了一个现有元素，但它可以进行定制。

1.  最后一部分是位于`/src/utils`文件夹中的`fetchCarData.js`文件：

    ```py
    export default async function fetchCarData(id) {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/cars/${id}`)
        const response = await res.json()
        if (!res.ok) {
            throw new Error(response.message)
        }
        return response
    }
    ```

`async`函数仅执行单个 API 调用以检索与单个实体相关的数据，如果发生错误，将触发`errorElement`。

加载函数非常实用。通过预加载数据，它们使用户拥有更好的用户体验，应用程序感觉更快。

# 摘要

在本章中，你使用现代 Vite 设置创建了一个 React 应用程序，并实现了基本功能——创建新资源、列出和显示汽车。本章还对你关于基本 React 钩子，如`useState`和`useEffect`，以及 Context API 进行了复习。你还学习了 React Router 的基本知识，包括其强大的加载函数。在本章中，你创建了两个表单，使用 RHF 并学习了如何管理 API 使用过程中涉及的各种步骤和状态。

下一章将探讨 Next.js 14 版本——这是最强大且功能丰富的基于 React.js 的全栈框架。
