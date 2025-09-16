

# 第十章：使用 Next.js 14 进行 Web 开发

Next.js 是一个用于构建全栈 Web 应用的 React 框架。虽然 React 是一个用于构建用户界面（Web 或原生）的库，但 Next.js 是一个完整的框架，基于 React 构建，提供了数十个特性，最重要的是，为从简单网站（如本章中将要构建的网站）到极其复杂的应用程序的项目结构。

虽然 React.js 是一个用于构建 UI 的无意见声明性库，但作为一个框架，Next.js 提供了配置、工具、打包、编译等功能，使开发者能够专注于构建应用程序。

本章将涵盖以下主题：

+   如何创建 Next.js 项目并将其部署

+   最新的 Next.js App Router 及其特性

+   不同类型的页面渲染：动态、服务器端、静态

+   Next.js 实用工具：`Image`组件和`Head`组件

+   服务器操作以及基于 cookie 的认证

# 技术要求

要创建本章中的示例应用程序，您应该具备以下条件：

+   Node.js 版本 18.17 或更高

+   用于运行上一章后端的 Python 3.11.7（无论是本地还是从部署，如 Render）

要求与上一章相同，您将要安装的新包将在介绍时进行描述。

# Next.js 简介

Next.js 14 是流行的基于 React 的框架的最新版本，用于创建全栈和可生产就绪的 Web 应用程序。

Next.js 甚至提供了通过名为**Route Handlers**（[`nextjs.org/docs/app/building-your-application/routing/route-handlers`](https://nextjs.org/docs/app/building-your-application/routing/route-handlers)）的新特性来创建后端服务器的可能性。这个特性提供了允许你创建自定义 HTTP 请求处理器，并通过使用 Web 请求和响应 API 来创建完整 API 的函数。

这些路由处理器类似于 FastAPI（`GET`、`POST`等）公开 HTTP 方法，并允许构建支持中间件、缓存、动态函数、设置和获取 cookie 和头部的复杂 API 等。

在接下来的几节中，您将能够插入自己的基于 Python 的服务器，并让该服务器独立运行，可能同时服务于其他应用程序（例如移动应用程序）。您将能够释放 Python 生态系统在集成某些数据科学或 AI 库方面的力量，并快速拥有与 Python 的出色开发者体验。

注意

如需了解特定主题的更详细说明，您可以参考以下网站：[`nextjs.org/docs`](https://nextjs.org/docs)。

# 创建 Next.js 14 项目

在这个以项目为导向的部分，你将学习如何利用你的 React 知识创建和部署你的项目。你将通过执行一系列简单的步骤来创建一个全新的 Next.js 应用。该项目将使用 Tailwind CSS（集成到 Next.js 中）和 JavaScript 而不是 TypeScript。

本章中你将构建的前端需要运行后端——来自上一章。它可以在你的本地机器上运行，或者在执行部署的情况下，从**Render**.com 运行。在开发过程中，在单独的终端中本地运行上一章的背景，并激活虚拟环境，将会更容易和更快。

要创建一个全新的 Next.js 项目并按照我们指定的方式设置（使用 JavaScript 而不是 TypeScript，新的 App Router 等），请执行以下步骤：

1.  在你选择的文件夹中打开终端并输入以下命令：

    ```py
    npx create-next-app@latest
    ```

    提示将询问你是否希望安装最新的`create-next-app`包，在撰写本文时是*版本 14.2.4*。确认安装。

    在安装`create-next-app`包并使用之前的命令启动它之后，CLI 工具将提出一系列问题（[`nextjs.org/docs/getting-started/installation`](https://nextjs.org/docs/getting-started/installation)）。对于你的项目，你应该选择以下选项：

    +   你的项目叫什么名字？`src/`目录？`@/*`)? 使用`cd` `FARM`命令并运行开发服务器：

        ```py
        npm run dev
        ```

        CLI 将通知你服务器正在 URL `http://127.0.0.1:3000`上运行。如果你在浏览器中访问这个页面，页面的首次渲染可能会有些延迟，这是正常的，因为 Next.js 会编译第一个也是目前唯一的页面。

    +   当前页面显示了很多 Next.js 特定的样式，因此为了从零开始，打开`/src/app/page.js`中的唯一自动定义的页面，并将其变成一个空的 React 组件（你可以使用 React Snippets 扩展的`rafce`快捷键）：

        ```py
        const Home = () => {
          return (
            <div>Home</div>
          )
        }
        export default Home
        ```

    +   此外，从`/src/app/globals.css`文件中删除 Next.js 特定的样式，只留下顶部的三个 Tailwind 导入：

        ```py
        @tailwind base;
        @tailwind components;
        @tailwind utilities;
        ```

现在你已经运行了一个空的 Next.js 应用，并且你准备好定义应用程序的页面。Next.js 使用与 React Router 不同的路由系统。在下一节中，你将学习如何根据需要使用 Next.js 框架的最重要功能。在继续之前，你将简要观察 Next.js 项目结构，并在下一节中熟悉主要文件夹和文件。

## Next.js 项目结构

虽然文档详细解释了每个文件和文件夹的功能（[`nextjs.org/docs/getting-started/project-structure`](https://nextjs.org/docs/getting-started/project-structure)），但了解你从哪里开始是很重要的。`/app`文件夹是应用程序的中心。其结构将决定以下部分中将要介绍的应用程序路由。

定义 Next.js 项目结构的最重要文件和文件夹如下：

+   根项目目录中的`/public`文件夹可用于提供静态文件，并且它们通过基本 URL 进行引用。

+   `next.config.js`文件是一个 Node.js 模块，用于配置你的 Next.js 应用程序——从该文件中可以配置前缀资产、`gzip`压缩、管理自定义头、允许远程图像托管、日志记录等等（[`nextjs.org/docs/app/api-reference/next-config-js`](https://nextjs.org/docs/app/api-reference/next-config-js)）。

+   `globals.css`文件是导入到每个路由的全局 CSS 样式。在你的应用程序中，你保持它最小化，并仅导入 Tailwind 指令。

+   可选地，你可以创建一个`middleware.js`函数，该函数将包含将在每个或仅选定请求上应用的中件。查看中件文档以了解更多信息：[`nextjs.org/docs/app/building-your-application/routing/middleware`](https://nextjs.org/docs/app/building-your-application/routing/middleware)

+   可选地，你可以在`/app`文件夹（具有特殊路由角色）外部创建一个`/components`目录，并在其中创建你的 React 组件。

现在你已经了解了简要的项目结构，你将创建应用程序的页面，并在过程中学习 Next.js App Router 的基础知识。你将故意将样式保持到最小，以展示功能性和组件边界。

## 使用 Next.js 14 进行路由

Next.js 中最新且推荐的路由系统依赖于`src/App`文件夹——通常，每个 URL 都有一个对应名称的文件夹，其中包含一个`page.js`文件。这种结构允许你甚至用`route.js`文件替换`page.js`文件，然后将其视为 API 端点。你将创建一个简单的路由处理程序用于演示目的，但在项目中你不会使用路由处理程序。

注意

在 Next.js 文档网站上可以找到对 App Router 的详细介绍（[`nextjs.org/docs/pages/building-your-application/routing`](https://nextjs.org/docs/pages/building-your-application/routing)）。

现在，你将构建基本页面结构：一个主页、一个显示所有汽车以及单个汽车的页面、一个仅供授权用户插入新汽车的私有页面，以及一个登录页面。

### 使用 App Router 创建页面结构

你已经在`App`目录的根目录中有一个`page.js`文件；它映射到网站的`/root` URL。

现在，您将构建剩余页面的路由：

1.  要创建一个显示汽车的路由（在 URL 中的`/cars`），在`/app`目录中创建一个新的文件夹，命名为`cars`，并在其中创建一个简单的`page.js`文件（文件名`page.js`是强制性的）：

    ```py
    const Cars = () => {
        return (
            <div>Cars</div>
        )
    }
    export default Cars
    ```

1.  当在`/src/app/cars`目录内时，创建一个基于汽车 ID 显示单个汽车的嵌套文件夹。在`cars`目录内创建另一个文件夹，并命名为`[id]`。这将告诉路由器该路由应该映射到`/cars/someID`。`/cars/`部分是基于文件夹位于`/cars`目录内的事实，而括号语法通知 Next.js 存在一个动态参数（在这种情况下是`id`）。在`[id]`文件夹内创建一个`page.js`文件，并将组件命名为`CarDetails`。

1.  重复相同的步骤，创建一个`/app/login/page.js`文件和一个`/app/private/page.js`文件，并使用相应的文件结构。运行`rafce`命令，为每个页面创建一个简单的组件。

现在，您已经定义了页面，可以通过手动访问各种 URL 来测试它们的功能：`/`, `/cars`, `/private`, 和 `/login`。

这是个很好的时机来比较 App Router 与其他我们在前几章中使用过的解决方案——即 React Router。

### Next.js 中的布局

与 React Router 及其`Slot`组件类似，Next.js App Router 提供了一个强大的`Layout`组件，它融合到目录结构概念中。`Layout`是一个在路由间共享的用户界面；它保留状态，保持交互性，并且不会重新渲染。与 React Router 中使用的`Slot`组件不同，Next.js 布局接受一个`children`属性，它将在基本页面内部渲染——实际上，整个应用程序都将加载在这个布局组件内部。

您可以检查整个 Next.js 应用程序中使用的强制根布局，它位于`/app/layout.js`。尝试在 body 内部和`{{children}}`组件之前添加一个元素，并检查该元素在哪些页面上可见——它应该在每一页上都可见。根布局不是您能使用的唯一布局；实际上，您可以为相关路由创建布局，以封装共同的功能或用户界面元素。

要创建一个简单的布局，该布局将被用于汽车列表页面和单个汽车页面（因此它将位于`/app/cars`文件夹内），在`/app/cars`目录内创建一个名为`layout.js`的文件：

```py
const layout = ({ children }) => {
    return (
        <div className="p-4 bg-slate-300 border-2
            border-black">
            <h2>Cars Layout</h2>
            <p>More common cars functionality here.</p>
            {children}
        </div>
    )
}
export default layout
```

您会注意到布局影响了`/cars`和`/cars/id`路由，但不会影响其他路由；布局文件的位置定义了它何时会被加载。这个功能使您能够创建不同的嵌套路由，并基于您的应用程序逻辑保持可重用的 UI 功能。

在继续之前，需要提到 Next.js 路由器的几个特性：

+   `template.js` 包裹整个子布局或页面，但不会跨请求持久化。例如，它可以与 Framer Motion 一起使用，以添加不同页面之间的页面转换和动画。

+   `[… folderName]`。这些段将匹配更多的路径参数。关于 Next.js 路由段文档，请参阅 https://nextjs.org/docs/app/building-your-application/routing/dynamic-routes#catch-all-segments。

+   **路由组**在您想防止文件夹包含在路由的 URL 路径中，同时保留布局功能时非常有用。路由组文档请参阅 [`nextjs.org/docs/app/building-your-application/routing/route-groups`](https://nextjs.org/docs/app/building-your-application/routing/route-groups)。

在创建了必要的页面并了解了 App Router 的主要功能之后，在下一节中，您将学习 Next.js 组件以及如何在应用程序结构中利用布局。

### Next.js 组件

Next.js 的一个主要新概念是区分 `localstorage` 等。

注意

Next.js 文档解释了这里的主要差异以及更微妙的不同之处：[`nextjs.org/docs/app/building-your-application/rendering/composition-patterns`](https://nextjs.org/docs/app/building-your-application/rendering/composition-patterns)。

一般而言，由于服务器组件可以直接在服务器上访问数据，因此它们更适合数据获取和敏感信息（访问令牌、API 密钥等）处理等任务。客户端组件更适合经典的 React **单页应用**（SPA）任务：添加交互性、使用 React 钩子、依赖于状态的自定义钩子、与浏览器接口、地理位置等。

默认情况下，Next.js 组件是 **服务器** 组件。要将它们转换为客户端组件，您必须在第一行添加 `"use client"` 指令。此指令定义了服务器和客户端组件模块之间的边界。

### 创建导航组件

开始构建 Next.js 组件，现在您将创建一个简单的导航组件，并了解 Next.js 中的 `Link` 组件。

要创建导航组件，请执行以下步骤：

1.  在 `/app` 文件夹旁边创建一个名为 `/src/components/` 的文件夹（不要放在里面，因为这些页面不会被用户导航）并在其中创建 `NavBar.js` 文件：

    ```py
    import Link from "next/link"
    const Navbar = async () => {
        return (
            <nav className="flex justify-between
                items-center bg-gray-800 p-4">
                <h1 className="text-white">Farm Cars</h1>
                <div className="flex space-x-4 text-white
                    child-hover:text-yellow-400">
                    <Link href="/">Home</Link>
                    <Link href="/cars">Cars</Link>
                    <Link href="/private">Private</Link>
                    <Link href="/login">Login</Link>
                </div>
            </nav>
        )
    }
    export default Navbar
    ```

    `NavBar.js` 组件与前面章节中创建的组件非常相似。然而，在这里，您已经导入了 `Link` 组件——这是扩展 `<a>` 元素（原生的 HTML 链接组件）并提供数据预获取的 Next.js 组件。[`nextjs.org/docs/app/api-reference/components/link`](https://nextjs.org/docs/app/api-reference/components/link)

1.  之前的代码使用了一个 Tailwind 插件，它允许开发者直接定位后代选择器。要使用它，打开 `tailwind.config.js` 文件并编辑内容，通过更改 `plugins` 数组值：

    ```py
      plugins: [
        function ({ addVariant }) {
          addVariant('child', '& > *');
          addVariant('child-hover', '& > *:hover');
        }
      ],
    ```

1.  现在打开位于 `/src/app/layout.js` 的根布局，在 `children` 属性之前插入 `NavBar.js` 组件，用以下代码替换现有的 `RootLayout` 函数：

    ```py
    import Navbar from "@/components/NavBar";
    ...
    export default function RootLayout({ children }) {
      return (
        <html lang="en">
          <body>
            <Navbar />
            {children}
          </body>
        </html>
      );
    }
    ```

在这一步中，您将新创建的组件添加到根布局中，因为它将在每个页面上显示。

您现在已定义了路由，构建了应用程序的基本页面，并创建了一个简单的导航菜单。在下一节中，您将看到 Next.js 如何通过服务器组件简化数据加载。

## 使用服务器组件加载数据

以下过程将帮助您学习如何从您的 FastAPI 服务器加载数据到 `/cars` 页面，而不需要使用钩子和状态，并了解 Next.js 如何扩展原生的 fetch 功能。

要在不使用钩子的情况下从您的 FastAPI 服务器加载数据到 `/cars` 页面，请执行以下步骤：

1.  在创建应显示您当前汽车收藏中所有汽车信息的页面之前，在 Next.js 项目的根目录中（与 `/src` 文件夹平行）创建一个 `.env` 文件，并使用它来映射您的 API 地址：

    ```py
    API_URL=http://127.0.0.1:8000
    ```

    此值将在您部署并希望使用您的 Render.com API URL 或您可能选择的任何后端部署解决方案时需要更改。

1.  一旦在环境中设置，地址将在您的代码中可用：

    ```py
    process.env.API_URL
    ```

    重要的是要记住，为了在浏览器中可见，环境变量需要以 `NEXT_PUBLIC_` 字符串开头。然而，在这种情况下，您正在服务器组件中进行数据获取，所以隐藏 API 地址是完全可以接受的。

    现在您已经准备好执行第一次服务器端获取。请确保您的后端服务器正在指定的端口 `8000` 上运行。

1.  打开 `/app/cars/page.js` 文件并编辑它：

    ```py
    import Link from "next/link"
    const Cars = async () => {
        const data = await fetch(
            `${process.env.API_URL}/cars/`, {
            next: {
                revalidate: 10
            }
        }
        )
        const cars = await data.json()
        return (
            <>
                <h1>Cars</h1>
                <div>
                    {cars.map((car) => (
                        <div key={car._id} className="m-4 bg-white p-2">
                            <Link href={`/cars/${car._id}`}>
                                <p>{car.brand} {car.make} from {car.year}</p>
                            </Link>
                        </div>
                    ))}
                </div>
            </>
        )
    }
    export default Cars
    ```

之前的代码可能看起来很简单，但它代表了基于 React 的开发中的一种全新的范式。

您使用了 Next.js 的 `fetch` 函数，它扩展了原生的 Web API `fetch` 方法并提供了一些额外的功能。它是一个 `async` 函数，因此整个组件是异步的，调用被等待。

注意

此 fetch 功能在 Next.js 网站上有详细的解释：[`nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating`](https://nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating)。

在提供各种功能，如访问头和 cookie 的同时，`fetch` 函数允许对缓存和重新验证接收到的数据进行细粒度控制。在此上下文中，重新验证意味着缓存失效和重新获取最新数据。你的汽车页面可能非常频繁地更新，你可以设置内容的时间限制。在前面的代码中，内容每 10 秒重新验证一次。在某些情况下，在几小时或几天后重新验证数据可能是有意义的。

在学习框架提供的专用组件之前，你将了解用于在布局和路由组边界内捕获错误的 `error.js` 文件。

### Next.js 中的错误页面

为了捕获服务器组件和客户端组件中可能出现的意外错误，并显示备用用户界面，你可以在所需文件夹内创建一个名为 `error.js` 的文件（文件名是强制性的）：

1.  创建一个文件，`/src/app/cars/error.js`，包含以下简单内容：

    ```py
    "use client"
    const error = () => {
      return (
        <div className="bg-red-800 text-white p-3">
          There was an error while fetching car data!
        </div>
      )
    }
    export default error
    ```

    组件必须按照文档使用 `"use client"` 指令。

1.  你可以通过在 `[id]/page.js` 中抛出一个通用错误来测试错误处理页面：

    ```py
    const SingleCar = () => {
      throw new Error('Error')
    }
    export default SingleCar
    ```

如果你现在尝试导航到任何车辆详情页面，你会看到页面已加载——导航存在，主布局和车辆布局已渲染。只有包含 `error.js` 文件的内部最深层路由组显示错误信息。

在学习如何直接从服务器获取页面内部数据之后，在下一节中，你将创建一个静态生成的单车辆页面，并了解强大的 Next.js `Image` 组件。

### 静态页面生成和 Image 组件

Next.js 提供了另一种生成页面的方法——*静态渲染*。在这种情况下，页面是在构建时（而不是在请求时）渲染的，或者在数据重新验证的情况下，在后台进行。然后生成的页面被缓存并推送到内容分发网络，以实现高效和快速的服务。这使得 Next.js 有效地表现得像一个静态网站生成器，就像 Gatsby.js 或 Hugo 一样，并在网站速度方面实现最大性能。

然而，并非所有路由都适合静态渲染；个性化页面和包含特定用户数据的页面是不应进行静态生成的页面示例。然而，博客文章、文档页面，甚至是汽车广告，都不是应该向不同用户显示不同功能的页面。

在本节中，你将首先生成单个汽车页面作为服务器端渲染页面，就像之前的汽车页面一样，然后你将修改页面（们）以进行静态渲染。

在你开始使用 `Image` 组件之前，修改 `next.js.mjs` 文件——Next.js 配置文件——并让 Next.js 知道它应该允许来自外部域的图片——在你的情况下，Cloudinary——因为我们的汽车图片就是托管在那里。

执行以下步骤：

1.  打开 `next.config.mjs` 文件并编辑配置：

    ```py
    /** @type {import('next').NextConfig} */
    const nextConfig = {
      images: {
        remotePatterns: [
          {
            hostname: 'res.cloudinary.com',
          },
        ]
      }
    };
    export default nextConfig;
    ```

1.  在此修改之后，手动重新启动 Next.js 开发服务器：

    ```py
    npm run dev
    ```

    现在您将创建汽车页面的服务器端渲染版本。

1.  打开 `/app/cars/[id]/page.js` 并相应地修改它：

    ```py
    import {
      redirect
    } from "next/navigation"
    import Image from "next/image"
    const CarDetails = async ({
      params
    }) => {
      const carId = params.id
      const res = await fetch(
        `${process.env.API_URL}/cars/${carId}`, {
          next: {
            revalidate: 10
          }
        }
      )
      if(!res.ok) {
        redirect("/error")
      }
      const data = await res.json()
    ```

    在前面的代码中，您导入了 `next/image` 组件，并将 URL 的参数解构为 `params`。然后，您执行了一个类似的 `fetch` 请求并检查了结果状态。如果发生错误，您使用了 Next.js 的 `redirect` 函数将用户重定向到尚未创建的错误页面。

1.  现在，继续编辑组件并返回一些基本的 JSX：

    ```py
    return (
      <div className="p-4 flex flex-col justify-center
        items-center min-h-full bg-white">
        <h1>{data.brand} {data.make} ({data.year})</h1>
        <p>{data.description}</p>
        <div className="p-2 shadow-md bg-white">
          <Image src={data.picture_url}
            alt={`${data.brand} ${data.make}`}
            width={600} height={400}
            className="object-cover w-full" />
        </div>
        <div className="grid grid-cols-2 gap-3 my-3">
          {data.pros && <div className="bg-green-200
            p-5 flex flex-col justify-center
            items-center">
            <h2>Pros</h2>
            <ol className="list-decimal">
              {data.pros.map((pro, index) => (
                <li key={index}>{pro}</li>
              ))}
            </ol>
          </div>}
          {data.cons && <div className="bg-red-200 p-5
            flex flex-col justify-center items-center">
            <h2>Cons</h2>
            <ol className="list-decimal">
              {data.cons.map((con, index) => (
                <li key={index}>{con}</li>
              ))}
            </ol>
          </div>}
        </div>
      </div >
      )
    }
    export default CarDetails
    ```

    功能组件的其余部分相当简单。您使用了 `Image` 组件并提供了必要的数据，例如 `width`、`height` 和 `alt` 文本。Image 组件有一个丰富的 API，在 Next.js 网站上有文档（[`nextjs.org/docs/app/api-reference/components/image`](https://nextjs.org/docs/app/api-reference/components/image)），并且尽可能使用它，因为它大大提高了您网站的性能。

    `redirect` 函数是从 `next/navigation` 导入的（[`nextjs.org/docs/app/building-your-application/routing/redirecting`](https://nextjs.org/docs/app/building-your-application/routing/redirecting)）。

    页面的静态生成版本包括向页面提供一个 `generateStaticParams()` 函数并将其导出；Next.js 使用此函数在构建时知道要生成哪些页面。

1.  对于您的 `/app/cars/[id]/page.js` 文件，此函数需要遍历所有需要静态页面的汽车（在这种情况下是所有汽车）并提供一个包含 ID 的数组：

    ```py
    export async function generateStaticParams() {
      const cars = await fetch(
        `${process.env.API_URL}/cars/`).then((res) =>
        res.json())
      return cars.map((car) => ({id: car._id,}))
    }
    ```

如果您将前面的 `generateStaticParams()` 函数添加到组件中，请停止开发服务器并运行另一个 Next.js 命令：

```py
npm run build
```

Next.js 将生成整个站点的优化构建，在构建时将单个汽车页面渲染为静态 HTML 页面。如果您检查控制台，您将看到路由列表和一个图例，显示哪些页面是在构建时渲染的。

使用以下命令可以运行生产构建：

```py
npm run start
```

在关闭本节之前，让我们处理用户点击错误 URL，导致不存在汽车的情况。为了处理这些 `404 页面未找到` 错误，创建一个名为 `/src/app/not-found.js` 的新文件并填充它：

```py
import Link from "next/link"
const NotFoundPage = () => {
  return (
    <div className="min-h-screen flex flex-col
      justify-center items-center">
      <h1>Custom Not Found Page</h1>
      <p>take a look at <Link href="/cars"
        className="text-blue-500">our cars</Link>
      </p>
   </div>
  )
}
export default NotFoundPage
```

此路由将涵盖所有路由组，类似于 React Router 包中的 `*` 路由。

在创建了动态服务器端和静态生成的页面，并探索了 Next.js 的一些最重要的功能之后，您将在下一节中学习如何使用现有的 API 对用户进行身份验证。

# Next.js 中的身份验证和服务器操作

你已经了解了相当多的 Next.js 功能，这些功能使它成为首屈一指的 Web 框架，但如果没有对 **Server Actions** 的简要介绍，最重要的功能列表将不完整。

服务器操作是仅在实际服务器上执行的异步函数，旨在处理数据获取和变更（通过 `POST`、`PUT` 和 `DELETE` 方法），并且可以通过普通表单提交（默认浏览器表单处理方法）调用，也可以通过事件处理程序（React 风格的方法）或通过第三方库如 Axios 调用。

这种方法的好处有很多。性能得到提升，因为客户端 JavaScript 显著减少，并且由于操作仅在服务器上运行，应用程序的整体安全性得到增强，甚至可以在禁用 JavaScript 的情况下运行，就像几十年前的老式应用程序一样。

你现在将创建第一个用于登录用户的服务器操作，这需要使用一个名为 `localStorage` 的包：签名和加密 cookies。使用方法相当简单，这里有所记录：[`github.com/vvo/iron-session`](https://github.com/vvo/iron-session)。

1.  使用以下命令安装 Iron Session 包：

    ```py
    npm i iron-session
    ```

1.  要使用 `iron-session` 功能，在名为 `/src/lib.js` 的文件中创建一个 `sessionOptions` 对象：

    ```py
    export const sessionOptions = {
      password:
       "complex_password_at_least_32_characters_long",
      cookieName: "farmcars_session",
      cookieOptions: {
        httpOnly: true,
        secure: false,
        maxAge: 60 * 60,
      }
    };
    ```

配置对象定义了用于 cookie 加密和解密的选项，你应该使用一个强大、由计算机生成的随机密码。

Iron Session API 非常简单，因为会话对象允许设置和获取类似字典的值。你将用它来设置两个简单的值：当前登录的用户名以及 `jwt` 本身，这对于调用你的 FastAPI 端点至关重要。

现在你将开始创建应用程序所需的服务器操作，从用于验证用户的登录操作开始：

1.  创建一个 `/src/actions.js` 文件并导入必要的包：

    ```py
    "use server";
    import { cookies } from "next/headers"
    import { getIronSession } from "iron-session"
    import { sessionOptions } from "./lib"
    import { redirect } from "next/navigation"
    export const getSession = async () => {
      const session = await getIronSession(
        cookies(), sessionOptions)
        return session
    }
    ```

    之前的代码从 Next.js 导入了 cookies，以及来自 Iron Session 的 `getIronSession()` 函数，以及你之前定义的 `sessionOptions` 类。然后你创建了一个简单的函数来获取当前会话及其中的数据。

1.  现在，在同一个文件中处理登录功能：

    ```py
    export const login = async (status, formData) => {
      const username = formData.get("username")
      const password = formData.get("password")
      const result = await fetch(
        `${process.env.API_URL}/users/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ username, password })
        })
      const data = await result.json()
      const session = await getSession()
      if (result.ok) {
        session.username = data.username
        session.jwt = data.token
        await session.save()
        redirect("/private")
        } else {
          session.destroy()
          return { error: data.detail }
      }
    }
    ```

    代码结构简单，与你在 React Router 和 `localStorage` 解决方案中看到的代码相似。重要的是与会话对象相关的部分——如果 `fetch` 调用返回成功响应，这意味着找到了有效的用户，并且会话已通过用户名和相应的 `jwt` 设置。如果没有，会话将被销毁。

    只有当用户登录并且会话成功设置时，才会执行重定向到 `/private` 页面的操作。

    现在你已经创建了第一个 Server Action，你就可以创建一个 Next.js 客户端组件了——登录表单，它将在登录页面上使用。

1.  创建一个新的组件文件，`/src/app/components/LoginForm.js`：

    ```py
    "use client"
    import {login} from "@/actions"
    import { useFormState } from "react-dom";
    const LoginForm = () => {
      const [state, formAction] = useFormState(login, {})
    ```

    `LoginForm` 与之前创建的 `NavBar` 组件不同，是一个客户端组件，这意味着它将在客户端渲染，因此需要以 `"use` `client"` 指令开始。

    `useFormState` 钩子是 React 生态系统中最新的添加之一（实际上，它是从 React-Dom 包中导入的，而不是 Next.js），它允许你根据表单操作更新状态（[`pl.react.dev/reference/react-dom/hooks/useFormState`](https://pl.react.dev/reference/react-dom/hooks/useFormState)）。

1.  继续构建 `LoginForm` 组件：

    ```py
    return (
        <div className="flex flex-col items-center justify-center max-w-sm mx-auto mt-10">
            <form className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4" action={formAction}>
                <div className="mb-4">
                    <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="username">
                        Username
                    </label>
                    <input
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="username" name="username" type="text" placeholder="Username" required />
                </div>
                <div className="mb-6">
                    <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="password">
                        Password
                    </label>
                    <input className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline" id="password" name="password" type="password" placeholder="******************" required />
                </div>
                <div className="flex items-center justify-between">
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 w-full rounded focus:outline-none focus:shadow-outline" type="submit">
                        Sign In
                    </button>
                </div>
                <pre>{JSON.stringify(state, null, 2)}</pre>
            </form>
        </div >
    )
    }
    export default LoginForm
    ```

    此登录表单使用 `useFormState` 钩子，它提供了状态——本质上是一个错误对象和 `formAction`。在表单中，你将状态显示为字符串化的 JSON 对象，但在实际场景中，你可以访问服务器（在你的情况下是 FastAPI）提供的所有单个错误，并相应地显示它们。

1.  在更新 `/src/app/login/page.js` 页面并简单地添加 `LoginForm` 组件后，你将得到以下内容：

    ```py
    import LoginForm from "@/components/LoginForm"
    const page = () => {
      return (
        <div>
          <h2>Login Page</h2>
          <LoginForm />
        </div>
      )
    }
    export default page
    ```

现在，如果你尝试导航到 `/login` 路由并输入一些无效凭据，错误将以字符串化的 JSON 格式打印在表单下方。如果凭据有效，你应该被重定向到 `/private` 路由，并在整个应用程序中可用的 `jwt` 中。

你已经通过使用 `iron-session` 包和 Next.js 服务器操作添加了认证功能。

在下一节中，你将创建一个仅对认证用户可见的受保护页面。尽管在 Next.js 中有不同方式来保护页面，包括使用 Next.js 中间件，但你将使用简单的会话验证来保护一个页面。

## 创建受保护的页面

在本节中，你将创建一个受保护的页面——用于将新车插入 MongoDB 数据库集合的页面。使用 Iron Session 检查 cookie 的有效性，并将登录用户的用户名和 `jwt` 值跨页面传递。

你将通过验证会话中的数据来创建一个受保护的页面。如果会话存在（并且包含用户名和 `jwt`），用户将能够导航到它并通过表单和相关的服务器操作创建新车。如果没有，用户将被重定向到登录页面。

在这个应用程序中，你将需要的唯一认证页面是用于插入新车的页面，Iron Session 使这项工作变得非常简单：

1.  打开 `/src/app/private/page.js` 并编辑该文件：

    ```py
    import { getSession } from "@/actions"
    import { redirect } from "next/navigation"
    const page = async () => {
      const session = await getSession()
      if (!session?.jwt) {
        redirect("/login")
      }
      return (
        <div className="p-4">
          <h1>Private Page</h1>
          <pre>{JSON.stringify(session, null, 2)}</pre>
        </div>
      )
    }
    export default page
    ```

    之前的代码使用了 Iron Session 对象：如果会话中的 `jwt` 存在，用户能够看到当前包含会话数据的页面。如果会话无效，用户将被重定向到 `/login` 页面。

1.  要使用会话添加注销功能，请向 `/src/actions.js` 文件添加另一个操作：

    ```py
    export const logout = async () => {
      const session = await getSession()
      session.destroy()
      redirect("/")
    }
    ```

    现在，这个操作可以从 `NavBar` 组件中调用，并且可以使用会话对象相应地显示或隐藏登录和登出链接。

1.  要将登出功能集成到网站中，在新的 `LogoutForm.js` 文件中创建一个简单的单按钮表单用于登出：

    ```py
    import { logout } from "@/actions"
    const LogoutForm = () => {
      return (
        <form action={logout}>
          <button className="bg-blue-500
              hover:bg-blue-700" type="submit">
              Logout
          </button>
       </form>
      )
    }
    export default LogoutForm
    ```

    `LogoutForm` 只包含一个按钮，该按钮调用之前定义的登出操作。让我们使用一些条件逻辑将其添加到导航（`NavBar.js`）组件中。

1.  打开 `src/components/Navbar.js` 文件并编辑导航组件：

    ```py
    import Link from "next/link"
    import { getSession } from "@/actions";
    import LogoutForm from "./LogoutForm";
    ```

    在导入 `getSession` 函数（用于跟踪用户是否已登录）和 `LogoutForm` 按钮之后，您可以定义组件：

    ```py
    const Navbar = async () => {
      const session = await getSession()
      return (
        <nav className="flex justify-between items-center
          bg-gray-800 p-4">
          <h1 className="text-white">Farm Cars</h1>
          <div className="flex space-x-4 text-white
            child-hover:text-yellow-400">
            <Link href="/">Home</Link>
            <Link href="/cars">Cars</Link>
            <Link href="/private">Private</Link>
            {!session?.jwt && <Link
              href="/login">Login</Link>}
            {session?.jwt && <LogoutForm />}
          </div>
        </nav>
      )
    }
    export default Navbar
    ```

该组件现在跟踪已登录用户，并根据用户的登录状态条件性地显示登录或登出链接。私有链接故意总是可见的，但您可以测试一下；如果您未登录，您将无法访问该页面，并将被重定向到登录页面。

您现在已完全实现了登录功能。有几个因素需要考虑，首先是 cookie 的持续时间——通过文件 `/src/lib.js` 中的 `maxAge` 属性设置——它应该与 FastAPI 从后端提供的 `jwt` 持续时间相匹配。该应用程序故意缺少用户注册功能，因为想法是有几个员工——可以通过 API 直接创建的用户。作为一个练习，您可以编写用户注册页面并使用 FastAPI 的 `/users/register` 端点。

在下一节中，您将通过创建一个仅对认证用户可见且仅允许销售人员插入新汽车的私有页面来最终完成应用程序。

## 实现新汽车页面

在本节中，您将创建插入新汽车的表单。您将不会使用表单验证库，因为这在*第八章*中已经介绍过，即使用 Zod 库构建应用程序的前端。在实际应用中，表单肯定会有类似类型的验证。您将创建一个新的服务器操作来执行 POST API 调用，并再次使用 `useFormState`——这是您用于登录用户的相同模式。 

由于插入汽车的表单包含许多字段（并且可能会有更多），您将首先将表单字段抽象成一个单独的组件。新汽车广告创建的实现将分为以下步骤：

1.  在名为 `/src/components/InputField.js` 的文件中创建一个新的 `Field` 组件：

    ```py
    const InputField = ({ props }) => {
      // eslint-disable-next-line react/prop-types
      const { name, type } = props
      return (
        <div className="mb-4">
          <label className="block text-gray-700
            text-sm font-bold mb-2" htmlFor={name}>
              {name}
          </label>
          <input className="shadow appearance-none
            border rounded w-full py-2 px-3
            text-gray-700 leading-tight
            focus:outline-none focus:shadow-outline"
            id={name}
            name={name}
            type={type}
            placeholder={name}
            required
            autoComplete="off"
          />
        </div>
      )
    }
    export default InputField
    ```

    现在 `InputField` 已经处理完毕，创建 `CarForm`。

1.  在 `/src/components/CarForm.js` 文件中创建一个新的组件，并从导入和所需字段的数组开始：

    ```py
    "use client"
    import { createCar } from "@/actions"
    import { useFormState } from "react-dom"
    import InputField from "./InputField"
    const CarForm = () => {
      let formArray = [
        {
          name: "brand",
          type: "text"
        },
        {
          name: "make",
          type: "text"
        },
        {
          name: "year",
          type: "number"
        },
        {
          name: "price",
          type: "number"
        },
        {
          name: "km",
          type: "number"
        },
        {
          name: "cm3",
          type: "number"
        },
        {
          name: "picture",
          type: "file"
        }
      ]
    ```

    该组件使用 `useFormState` 钩子；您已经知道它需要一个客户端组件。

1.  组件的其余部分只是对 `fields` 数组的映射和钩子的实现：

    ```py
    const [state, formAction] = useFormState(
      createCar, {})
      return (
        <div className="flex items-center justify-center">
          <pre>{JSON.stringify(state, null, 2)}</pre>
            <div className="w-full max-w-xs">
              <form className="bg-white shadow-md rounded
                px-8 pt-6 pb-8 mb-4"
                action={formAction}>
                  <h2 className="text-center text-2xl
                    font-bold mb-6">Insert new car
                  </h2>
                  {formArray.map((item, index) => (
                  <InputField key={index}
                    props={{
                    name: item.name, type: item.type
                    }} />
                   ))}
                   <div className="flex items-center
                     justify-between">
                     <button className="bg-gray-900
                       hover:bg-gray-700 text-white w-full
                       font-bold py-2 px-4 rounded
                       focus:outline-none
                       focus:shadow-outline"
                       type="submit">Save new car
                     </button>
                   </div>
                 </form>
               </div>
             </div>
           )
      }
    export default CarForm
    ```

    表单使用 `createCar` 动作，您将在后续步骤中定义该动作的 `actions.js` 文件。

1.  表单需要在私有页面上显示，因此编辑 `/src/app/private/page.js` 文件：

    ```py
    import CarForm from "@/components/CarForm"
    import {getSession} from "@/actions"
    import { redirect } from "next/navigation"
    const page = async () => {
      const session = await getSession()
      if (!session?.jwt) {
        redirect("/login")
        }
      return (
        <div className="p-4">
          <h1>Private Page</h1>
          <CarForm />
        </div>
      )
    }
    export default page
    ```

    表单已创建，并在 `/private` 页面上显示。唯一缺少的是相应的动作，您将在下一步中创建。

1.  打开 `/src/actions.js` 文件，并在文件末尾添加以下动作以创建新汽车：

    ```py
    export const createCar = async (state, formData) => {
      const session = await getSession()
      const jwt = session.jwt
      const result = await fetch(`${
        process.env.API_URL}/cars/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${jwt}`,
            },
            body: formData
        })
        const data = await result.json()
        if (result.ok) {
          redirect("/")
        } else {
          return { error: data.detail }
        }
    }
    ```

动作很简单——这就是服务器动作的美丽之处。它只是一个检查会话和 `jwt` 并执行 `API POST` 请求的函数。该函数还应包括在找不到 `jwt` 的情况下将用户重定向到登录页面的早期重定向，但这样您可以让 `useFormState` 钩子显示来自后端的任何错误。

您已实现了网站规范——用户能够登录并插入新汽车，在重新验证期（15-20 秒）后，汽车将在 `/car` 页面以及新插入汽车的专用页面上显示。

在下一节中，您将部署您的应用程序到 Netlify，并学习如何简化流程，同时提供环境变量并配置部署设置。

## 提供元数据

Next.js 的一个主要特性是提供比 SPAs 更好的 **搜索引擎优化**（**SEO**）。虽然生成易于爬虫抓取的静态内容很重要，但提供有用的页面元数据是至关重要的。

元数据是每个网络应用程序或网站的重要特性，Next.js 通过 `Metadata` 组件以优雅的方式解决了这个问题。元数据使与搜索引擎（如 Google）的直接通信成为可能，提供有关网站内容、标题和描述的精确信息，以及页面特定的信息。

在本节中，您将学习如何设置页面的标题标签。Next.js 文档非常详细（[`nextjs.org/docs/app/building-your-application/optimizing/metadata`](https://nextjs.org/docs/app/building-your-application/optimizing/metadata)），并解释了可以设置的各种信息片段，但在此情况下，您只需设置页面标题：

1.  打开 `src/app/layout.js` 页面并编辑 `metadata` 部分：

    ```py
    export const metadata = {
      title: "Farm Cars App",
      description: "Next.js + FastAPI + MongoDB App",
    };
    ```

    这个简单的更改将导致布局内的所有页面都拥有新设置的标题和描述。由于您已编辑包含所有页面的 `Root` 布局，这意味着网站上的每个页面都将受到影响。这些可以在每个页面上进行覆盖。

1.  打开 `/src/app/cars/[id]/page.js` 以访问单个汽车页面，并添加以下导出：

    ```py
    export async function generateMetadata({ params }, parent) {
        const carId = params.id
        const car = await fetch(`${process.env.API_URL}/cars/${carId}`).then((res) => res.json())
        const title = `FARM Cars App - ${car.brand} ${car.make} (${car.year})`
        return { title }
    }
    ```

上述导出向 Next.js 传达了只有这些页面应该有从函数返回的标题，而其他页面将保留未更改的标题。

你已经成功编辑了页面的元数据，现在是时候将应用程序部署到互联网上了，下一节将详细介绍。

# Netlify 部署

Next.js 可以说是最受欢迎的全栈和前端框架，并且有大量的部署选项。

在本节中，你将学习如何在 Netlify 上部署你的 Next.js 应用程序——Netlify 是最受欢迎的用于部署、内容编排、持续集成等功能的 Web 平台之一。

为了在 Netlify 上部署你的网站，你需要部署 FastAPI 后端。如果你还没有这样做，请参考*第七章*，*使用 FastAPI 构建后端*，了解如何进行操作。一旦你有了后端地址（在你的例子中，部署的 FastAPI 应用程序的 URL 是[`chapter9backend2ed.onrender.com`](https://chapter9backend2ed.onrender.com)），它将被用作 Next.js 前端的 API URL。

为了执行 Netlify 的部署，请执行以下步骤：

+   **创建 Netlify 账户**：使用 GitHub 账户登录并创建一个免费的 Netlify 账户，因为 Netlify 将从你为 Next.js 应用程序创建的仓库中提取代码。

+   **创建 GitHub 仓库**：为了能够部署到 Netlify（或者 Vercel），你需要为你的 Next.js 项目创建一个 GitHub 仓库。

要创建 GitHub 仓库，请执行以下步骤：

1.  在你的终端中，进入项目文件夹并输入以下命令：

    ```py
    git add .
    ```

    此命令将修改后的和新创建的文件添加到仓库中。

1.  接下来，提交更改：

    ```py
    git commit -m "Next.js project"
    ```

1.  现在你的项目已置于版本控制之下，在你的 GitHub 账户中创建一个新的仓库并选择一个合适的名称。在你的情况下，仓库被命名为`chapter10frontend`。

## 推送更改到 GitHub

现在你可以将新的源添加到你的本地仓库。在项目中的同一终端内，输入以下命令：

1.  首先，将分支名称设置为`main`：

    ```py
    git branch -M main
    ```

1.  然后，将源设置为新建的仓库：

    ```py
    git remote add origin https://github.com/<your username>/<name_of_the_repo>.git
    ```

    在这里，你需要替换仓库名称和你的用户名：`(<username>`和`<name_of_the_repo>`)。

1.  最后，将项目推送到 GitHub：

    ```py
    git push -u origin main
    ```

现在，你可以以下述方式在 Netlify 上部署仓库：

1.  （在你的例子中是`chapter10frontend`）。

1.  `main`，因为这是你唯一的分支

1.  基础目录：保持为空

1.  构建命令：保持为`npm run build`

1.  发布目录：保持为`.next`

1.  设置唯一的环境变量：点击`API_URL`，其值将是 FastAPI 后端 URL。如果你遵循了上一章中关于在 Render 上托管后端的步骤，该值将是[`chapter9backend2ed.onrender.com`](https://chapter9backend2ed.onrender.com)。

+   点击**部署**（**你的** **repo**）按钮！

一段时间后，你应该能在页面上显示的地址看到你的网站已部署。然而，请注意，API 必须处于工作状态，例如，如果你使用 Render.com 的免费层作为后端部署选项（如果你使用了 Render 作为后端部署选项），在数据过时后可能需要一分钟才能唤醒，因此请准备好唤醒 API。建议等待后端响应——你可以通过简单地访问 API 地址来检查它——然后开始部署过程。这样，你将防止潜在的部署和页面生成错误。

这是个分析你提供给 Netlify 以构建网站的命令——`build`命令——的好时机。如果你在 Next.js 命令行中运行`npm run build`，Next.js 将执行一系列操作并生成一个优化的构建。

这些操作包括代码优化（如压缩和代码拆分）、创建包含优化、生产就绪代码的`.next`目录，以及实际上在互联网上提供服务的目录。

`build`命令还会生成静态页面和路由处理程序。在成功完成构建后，你可以使用以下命令测试构建：

```py
npm run start
```

你现在已经成功部署了一个优化的、由 FastAPI MongoDB 驱动的 Next.js 网站，你准备好使用这个强大且灵活的堆栈来处理大量的 Web 开发任务了。

# 摘要

在本章中，你学习了 Next.js 的基础知识，这是一个流行的基于 React 的全栈框架，结合 FastAPI 和 MongoDB，允许你构建几乎任何类型的 Web 应用程序。

你已经学会了如何创建新的 Next.js 项目，如何使用新的 App Router 实现路由，以及如何使用服务器组件获取数据。

还介绍了并实现了重要的 Next.js 概念，如服务器操作、表单处理和 cookies。除此之外，你还探索了一些 Next.js 优化，例如用于提供优化图像的`Image`组件、`Metadata`标签以及如何创建生产构建。

最后，你在 Netlify 上部署了你的 Next.js 应用程序，但其他提供者的部署基本原理仍然相同。

Next.js 本身就是一个丰富且复杂的生态系统，你应该将本章视为你下一个应用的起点，该应用融合了三个世界的最佳之处：FastAPI、MongoDB 和 React，并添加了应用程序可能需要的第三方外部服务。

下一章将分享一些在实际使用 FARM 堆栈时对你有帮助的实用建议，以及一些可以帮助你立即开始的项目想法。
