# 10

# 集成 React 前端与 Flask 后端

本章代表了我们构建全栈 Web 应用程序过程中的一个关键点。在本章中，你将了解到如何将 Flask Web 服务器连接到 React 前端的一系列指令。你将学习如何将 React 前端表单输入传递到 Flask 后端。在此集成之后，你就可以正式被称为**全栈** **Web 开发者**。

React Web 应用程序通常具有简洁的外观和感觉，被认为是现代前端 Web 应用程序的劳斯莱斯。React 拥有一个直观的用户界面库，能够轻松地驱动生产级的 Web 和移动应用程序。

强大的 React 生态系统与 React 的工具和库相结合，促进了端到端的 Web 开发。当你将 React 令人难以置信的基于组件的设计模式与一个简约轻量级的 Flask 框架相结合时，你将得到一个能够经受时间考验并大规模扩展的丰富 Web 应用程序。

本章将帮助你理解在开发有价值的软件产品时，将 React（一个前端库）和 Flask（一个后端框架）集成的动态。你还将学习在本章中 React 如何处理与 Flask 后端相关的表单。

自从 Web 的出现以来，在 Web 应用程序中就需要更多动态和响应式的表单形式。我们将探讨服务器端表单元素的处理、验证和安全问题。

在本章中，我们将涵盖以下主题：

+   *Bizza*应用程序结构

+   配置 React 前端

+   准备 Flask 后端

+   在 React 和 Flask 中处理表单

+   React 前端和 Flask 后端的故障排除技巧

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter10`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter10)。

# *Bizza*应用程序结构

在本节中，我们将深入探讨本书中将要构建的应用程序的结构。正如之前所述，我们将把这个虚构的 Web 应用程序命名为*Bizza*，一个会议活动 Web 应用程序。

这个*Bizza* Web 应用程序将作为信息技术行业演讲者的会议活动的数字中心，提供众多功能和特性，以增强演讲者和与会者的整体体验。让我们深入了解*Bizza*应用程序结构。

## 应用程序概述

*Bizza*是一个虚构的数据驱动事件应用程序，允许信息技术行业的主题专家分享他们的见解和经验，为活动参与者提供有价值的知识，以提升他们的技能。

*Bizza*让你可以看到演讲者和研讨会日程的列表，并查看详细信息。这个网站允许用户注册并浏览研讨会。本质上，该应用程序将具有以下功能：

+   展示活动演讲者和可用活动日程（包括地点和主题）的主页

+   事件参加者注册表单

+   具有感兴趣主题的演讲者注册表单

+   用户登录应用程序的页面

+   包含演讲者姓名和详细信息的页面

接下来，我们将深入探讨 *Bizza* 应用程序，并将其分解为其前端和后端组件。通过这样做，我们将全面了解每个组件在应用程序中扮演的独特角色和功能。

## 将代码结构分解为前端和后端

在软件开发的世界里，前端和后端就像阴阳一样——相反但相辅相成，共同提供和谐的数字体验。“阴阳”是中国哲学中的一个概念，描述了相反但相互关联的力量。

简而言之，将应用程序分解为其前端和后端组件提供了关注点的清晰分离，促进了代码的重用性和可移植性，实现了可扩展性和性能优化，并促进了协作和并行开发。这种方法最终有助于网络应用程序开发过程的总体成功。

在 20 世纪 90 年代末和 21 世纪初，随着基于 Web 的应用程序的兴起，软件开发中开始重视将前端和后端组件分离。在此期间，Web 技术瞬息万变，对可扩展和模块化应用程序的需求变得明显。

早在 2000 年代初期，JavaScript 框架如 jQuery 的引入使得前端用户界面更加动态和交互。这导致了网络应用程序的表现层（前端）和数据处理层（后端）之间更加清晰的区分。

随着 **单页应用程序**（**SPAs**）的出现以及 AngularJS、React 和 Vue.js 等 JavaScript 框架和库的普及，前端和后端之间的分离变得更加标准化和广泛采用。SPAs 将渲染和管理 UI 的责任转移到了客户端，而后端 API 处理数据检索和操作。

现在我们已经讨论了分解代码结构的关键原因，让我们来检查 *Bizza* 网络应用程序的前端和后端组件。

下面的代码结构代表了前端和后端之间的高端级别代码拆分。这使我们能够分离关注点并提高代码的可重用性：

```py
bizza/├── backend/
├── frontend/
```

### 前端结构

首先，让我们提供一个详细的 `frontend` 结构概述：

```py
frontend/├── node_modules
├── package.json
├── public
    ├──favicon.ico
├──index.html
├── src
    ├──components/
    ├──pages/
    ├──hooks/
    ├──assets/
    └──App.js
    └──App.css
    └──index.js
    └──index.css
    └──setupTests.js
├──.gitignore
├──.prettierrc
├──package-lock.json
├──package.json
├──README.md
```

前端代码结构主要包括 `node_modules`、`package.json`、`public`、`src`、`.gitignore`、`.prettierrc`、`package-lock.json` 和 `README.md`。

让我们快速分析主要的目录和文件：

+   `node_modules`: 此目录包含你的应用程序所依赖的所有包（库和框架）。这些包列在 `package.json` 文件的 `dependencies` 和 `devDependencies` 部分中。

+   `package.json`: 此文件包含有关你的应用程序的元数据，包括其名称、版本和依赖项。它还包括你可以用来构建、测试和运行应用程序的脚本。

+   `public`: 此目录包含你的应用程序将使用的静态资源，例如 favicon 和主 HTML 文件（`index.html`）。

+   `src`: 此目录包含你的应用程序的源代码。它组织成组件、页面、钩子和资产的子目录。`src` 目录对于采用的 React 前端设计模式至关重要。`components` 文件夹包含我们打算在 *Bizza* 应用程序中使用的所有组件，`pages` 包含应用程序的展示组件，`hooks` 包含自定义钩子，最后，`assets` 文件夹包含应用程序中使用的所有资产，例如 `images`、`logos` 和 `svg`。

+   `.gitignore`: 此文件告诉 Git 在你将代码提交到仓库时应忽略哪些文件和目录。

+   `.prettierrc`: 此文件指定了 Prettier 代码格式化工具的配置选项。Prettier 是一款流行的代码格式化工具，它确保你的代码库风格一致。它通常放置在 JavaScript 项目的 `root` 目录中，并包含用于定义格式化规则的 JSON 语法。

+   `package-lock.json`: 此文件记录了应用程序所依赖的所有包的确切版本，以及这些包所依赖的任何包。它确保每次安装应用程序时，它都使用其依赖项的相同版本。

+   `README.md`: 此文件包含你的应用程序的文档，例如安装和运行它的说明。

### 后端结构

接下来，我们将检查后端的结构：

```py
backend/├── app.py
├── models
├── config
│   ├── config.py
├── .flaskenv
├── requirements.txt
```

上述内容代表了 Flask 后端应用程序的文件和目录结构。

让我们分解目录和文件：

+   `app.py`: 此文件包含你的后端应用程序的主要代码，包括处理 HTTP 请求的路由和逻辑。

+   `models`: 此目录包含数据库模型定义的每个模型的模块。

+   `config`: 此目录包含应用程序的配置选项文件，例如数据库连接字符串或密钥。

+   `.flaskenv`: 此文件包含特定于 Flask 应用的环境变量。

+   `requirements.txt`: 此文件列出了应用程序所依赖的包，包括任何第三方库。你可以通过运行 `pip install -r requirements.txt` 来使用此文件安装必要的依赖项。

接下来，我们将了解如何配置 React 前端并准备它以消费后端 API 服务。

# 为 API 消费配置 React 前端

在本节中，您将配置前端 React 应用通过在 React 中设置代理与后端 Flask 服务器进行通信，以从 Flask 服务器消费 API。

为了配置 React 代理以用于 API 消费，您需要更新前端 React 应用的`package.json`文件中的`proxy`字段。`proxy`字段允许您指定一个 URL，该 URL 将用作从 React 应用发出的所有 API 请求的基础。

让我们更新`package.json`文件：

1.  使用文本编辑器在`project`目录中打开`package.json`文件，然后向`package.json`文件中添加一个`proxy`字段，并将其设置为 Flask 服务器的 URL：

    ```py
    {  "name": "bizza",  "version": "0.1.0",  "proxy": "http://localhost:5000"}
    ```

1.  接下来，您需要从 React 前端向 Flask 服务器发送 HTTP 请求。我们将使用`Fetch()`方法作为 Axios 的替代方案。

    Axios 是一个允许您从浏览器中发送 HTTP 请求的 JavaScript 库。它是一个基于 promise 的库，使用现代技术使异步请求变得容易处理。使用 Axios，您可以发送 HTTP 请求从服务器检索数据，提交表单数据，或将数据发送到服务器。

    Axios 支持多种不同的请求方法，如`GET`、`POST`、`PUT`、`DELETE`和`PATCH`，并且它可以处理 JSON 和 XML 数据格式。Axios 在开发者中很受欢迎，因为它有一个简单直接的 API，使得初学者和经验丰富的开发者都很容易使用。

    Axios 还具有许多使它灵活强大的功能，例如自动转换数据、支持拦截器（允许您在发送或接收之前修改请求或响应），以及取消请求的能力。

1.  您可以通过在终端中运行以下命令来安装 Axios：

    ```py
    npm install axios
    ```

    一旦安装了 Axios，您就可以使用它从 React 前端向 Flask 服务器发送 HTTP 请求。

1.  确保前端 React 应用和后端 Flask 服务器在不同的端口上运行。默认情况下，React 开发服务器在端口`3000`上运行，而 Flask 开发服务器在端口`5000`上运行。

接下来，您需要在 Flask 后端定义路由和函数来处理来自 React 前端发出的 HTTP 请求。

# 使 Flask 后端准备就绪

在*第一章*的*使用 React 和 Flask 准备全栈开发环境*部分，我们为 Flask 服务器设置了开发环境。请确保您的虚拟环境已激活。您可以通过运行以下命令来实现：

+   **对于 Mac/Linux**：

    ```py
    source venv/bin/activate
    ```

+   **对于 Windows**：

    ```py
    Venv/Scripts/activate
    ```

您的虚拟环境现在应该已激活，并且您的终端提示符应该以虚拟环境名称为前缀（例如，`(``venv) $`）。

接下来，让我们直接进入定义事件注册路由，该路由作为 Bizza 应用程序模型要求的一部分。

让我们添加一个模型来处理活动参加者的注册。你将在下一节中使用它来接受来自 React 前端的要求，在那里我们将处理 React 和 Flask 中的表单输入。

应用程序根目录中的`app.py`文件仍然是 Flask 应用程序的主要入口点。更新`app.py`以以下代码片段定义模型和端点以处理活动注册：

```py
class EventRegistration(db.Model):    __tablename__ = 'attendees'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), unique=True, nullable=False)
    last_name = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(100), unique=True, nullable=False)
    job_title = db.Column(db.String(100), unique=True, nullable=False)
    company_name = db.Column(db.String(100), unique=True,         nullable=False)
    company_size = db.Column(db.String(50), unique=True,         nullable=False)
    subject = db.Column(db.String(250), nullable=False)
def format(self):
    return {
        'id': self.id,
        'first_name': self.first_name,
        'last_name': self.last_name,
        'email': self.email,
        'phone': self.phone,
        'job_title': self.job_title,
        'company_name': self.job_title,
        'company_size': self.company_size,
        'subject': self.subject
    }
```

在前面的代码片段中，`EventRegistration`类代表数据库中活动注册的模型。

`__tablename__`属性指定了数据库中存储此模型的表的名称。`db.Model`类是`Flask-SQLAlchemy`中所有模型的基类，`db.Column`对象定义了模型字段，每个字段都有一个类型和一些附加选项。

`format`方法返回模型实例的字典表示形式，键对应字段名称，值对应字段值。

现在，让我们定义路由或端点，`/api/v1/events-registration`：

```py
@app.route("/api/v1/events-registration", methods=['POST'])def add_attendees():
    if request.method == 'POST':
        first_name = request.get_json().get('first_name')
        last_name = request.get_json().get('last_name')
        email = request.get_json().get('email')
        phone = request.get_json().get('phone')
        job_title = request.get_json().get('job_title')
        company_name = request.get_json().get('company_name')
        company_size = request.get_json().get('company_size')
        subject = request.get_json().get('subject')
        if first_name and last_name and email and phone and subject:
            all_attendees = EventRegistration.query.filter_by(
                email=email).first()
            if all_attendees:
                return jsonify(message="Email address already                     exists!"), 409
            else:
                new_attendee = EventRegistration(
                    first_name = first_name,
                    last_name = last_name,
                    email = email,
                    phone = phone,
                    job_title = job_title,
                    company_name = company_name,
                    company_size = company_size,
                    subject = subject
                )
                db.session.add(new_attendee)
                db.session.commit()
                return jsonify({
                    'success': True,
                    'new_attendee': new_attendee.format()
                }), 201
        else:
            return jsonify({'error': 'Invalid input'}), 400
```

`/api/v1/events-registration`端点函数处理对`/api/v1/events-registration`路由的 HTTP `POST`请求。此端点允许用户通过提供他们的姓名、电子邮件地址、电话号码和主题来注册活动。

端点函数首先检查请求方法是否确实是`POST`，然后从请求体中提取名称、电子邮件、电话和主题值，预期请求体为 JSON 格式。

接下来，该函数检查所有必需的输入值（`first_name`、`last_name`、`email`、`phone`和`subject`）是否都已存在。如果存在，它将检查数据库中是否已存在具有相同电子邮件地址的参与者。如果存在，它将返回一个 JSON 响应，其中包含一条消息指出电子邮件地址已被使用，以及 HTTP `409`状态码（冲突）。

如果电子邮件地址未被使用，该函数将使用输入值创建一个新的`EventRegistration`对象，将其添加到数据库会话中，并将更改提交到数据库。然后，它将返回一个包含成功消息和新的参与者详情的 JSON 响应，以及 HTTP `201`状态码（已创建）。

如果任何必需的输入值缺失，该函数将返回一个包含错误消息和 HTTP `400`状态码（错误请求）的 JSON 响应。现在，让我们更新数据库并添加一个`eventregistration`表格。`eventregistration`表格将接受所有活动注册的条目。

以下步骤在数据库中创建`eventregistration`表格。在`project`目录的终端中，输入以下命令：

```py
flask shellfrom app import db, EventRegistration
db.create_all()
```

或者，你可以继续使用迁移工具：

```py
flask db migrate –m "events attendee table added"flask db upgrade
```

使用这些选项中的任何一个，后端都将包含新的表格。

在终端中执行`flask run`以在`localhost`上使用默认端口（`5000`）启动 Flask 开发服务器。

就这样！后端现在已准备好接收来自 React 前端的表单条目。让我们在 React 中设计表单组件并将表单条目提交到 Flask 后端。

# 处理 React 和 Flask 中的表单

在 Web 开发中，处理 React 前端和 Flask 后端的表单是一种常见模式。在这个模式中，React 前端向 Flask 后端发送 HTTP 请求以提交或检索表单数据。

在 React 前端方面，你可以使用表单组件来渲染表单并处理表单提交。你可以使用受控组件，如 `input`、`textarea` 和 `select`，来控制表单值并在用户输入数据时更新组件状态。

当用户提交表单时，你可以使用事件处理器来阻止默认的表单提交行为，并使用类似 Axios 的库向 Flask 后端发送 HTTP 请求。在本节中，我们将使用 Axios 库。

在 Flask 后端方面，你可以定义一个路由来处理 HTTP 请求并从请求对象中检索表单数据。然后你可以处理表单数据并向前端返回响应。

`EventRegistration` 组件为未认证用户提供了一个简单的表单，用于在 *Bizza* 应用程序的前端注册活动。该表单包括用户姓名、电子邮件地址、电话号码和主题字段——即他们注册的活动主题或标题。

让我们深入了解与 Flask 后端协同工作的 React 表单实现：

1.  在项目目录中，在 `components` 文件夹内创建 `EventRegistration/EventRegistration.jsx`。

1.  将以下代码片段添加到 `EventRegistration.jsx` 文件中：

    ```py
    import React, { useState, useEffect } from 'react';import axios from 'axios';const EventRegistration = () => {  // Initial form values  const initialValues = {    firstname: '',    lastname: '',    email: '',    phone: '',    job_title: '',    company_name: '',    company_size: '',    subject: '' };  // State variables  const [formValues, setFormValues] =    useState(initialValues); // Stores the form field                                values  const [formErrors, setFormErrors] = useState({});// Stores the form field for the validation errors  const [isSubmitted, setIsSubmitted] =    useState(false); // Tracks whether the form has                        been submitted{/* Rest of the form can be found at the GitHub link - https://github.com/PacktPublishing/Full-Stack-Flask-Web-Development-with-React/tree/main/Chapter-10/ */}            <div id="btn-section">              <button>Join Now</button>            </div>        </form>      </div>    </div>  </div></>);};export default EventRegistration;
    POST request to the /api/v1/events-registration route with the form data. It then updates the component’s state with the response from the server and displays a success or error message to the user.
    ```

    `EventRegistration` 组件还包括一个 `validate` 函数，用于检查表单值中的错误，以及一个 `onChangeHandler` 函数，用于在用户输入表单字段时更新表单值。

让我们讨论前面代码中使用的组件状态变量：

+   `表单值`: 这是一个对象，用于存储表单字段的当前值（姓名、电子邮件、电话和主题）

+   `表单错误`: 这是一个对象，用于存储在表单值中发现的任何错误

+   `response`: 这是一个对象，用于存储表单提交后从服务器返回的响应

+   `反馈`: 这是一个字符串，用于存储要显示给用户的反馈消息（例如，**注册成功！**）

+   `状态`: 这是一个字符串，用于存储表单提交的状态（例如，**成功**或**错误**）

我们然后定义以下函数：

+   `validate`: 这是一个函数，接受表单值并返回一个包含在值中发现的任何错误的对象。

+   `onChangeHandler`: 这是一个函数，用于在用户在表单字段中输入时更新 `表单值` 状态变量。

+   `handleSubmit`：这是一个在表单提交时被调用的函数。它阻止默认的表单提交行为，调用 `validate` 函数来检查错误，然后使用 `sendEventData` 函数将表单数据发送到服务器。它还会根据服务器的响应更新反馈和状态状态变量。

+   `sendEventData`：这是一个 `async` 函数，它向 `/api/v1/events-registration` 路由发送带有表单数据的 HTTP `POST` 请求，并使用服务器的响应更新响应状态变量。

`EventRegistration` 组件同样包含一个 `useEffect` 钩子，当 `formValues` 状态变量发生变化时，会调用 `sendEventData` 函数。最后，`EventRegistration` 组件渲染一个包含表单字段的表单元素，并向用户显示反馈信息和状态。

现在，使用 `npm start` 启动 React 前端并提交您的表单条目。确保 Flask 服务器也在运行。在任何一个开发过程中，问题和错误都是不可避免的。我们将探讨一些有价值的故障排除技巧，帮助您在 React 前端和 Flask 后端集成过程中调试和修复问题。

# React 前端和 Flask 后端的故障排除技巧

将 React 前端与 Flask 后端集成可以是一个强大的组合，用于构建动态和可扩展的 Web 应用程序。然而，像任何集成一样，它可能带来自己的一套不可避免的问题。在 React-Flask 集成过程中出现问题时，需要系统的方法来识别和有效地解决问题。

本节将讨论您在将前端与后端集成时可能遇到的某些问题的解决方法。通过遵循这些技巧，您将能够诊断并解决在应用程序的开发和部署过程中可能出现的常见问题。

让我们深入了解 React 前端和 Flask 后端集成的故障排除技巧：

+   **验证** **Flask 设置**：

    +   确保 Flask 已正确配置并在服务器上运行。

    +   检查 Flask 服务器控制台中的任何错误消息或异常，这些可能表明配置错误。

    +   确认已安装必要的 Flask 包和依赖。

    +   通过测试基本端点来验证 Flask 服务器是否可访问并响应请求。

+   **检查** **React 配置**：

    +   确保 React 应用程序已正确配置并运行。

    +   确认 React 项目中已安装必要的依赖和包。

    +   在浏览器开发者工具的控制台中检查任何可能表明前端设置问题的 JavaScript 错误或警告。

    +   确保在 `package.json` 中添加了代理属性，并指向 Flask 服务器地址 - 例如，`http://127.0.0.1:5000`。

+   **调查** **网络请求**：

    +   使用浏览器的开发者工具来检查 React 应用程序发出的网络请求

    +   确认请求是否发送到正确的 Flask 端点

    +   检查网络响应状态码以识别任何服务器端错误

    +   检查响应负载以确保数据正确传输

    +   如果 React 前端和 Flask 后端托管在不同的域名或端口上，请注意**跨源资源共享**（**CORS**）问题

通过遵循这些故障排除技巧，您将具备诊断和解决 React-Flask 集成问题的必要知识。这将确保您的 Web 应用程序集成平稳且健壮。

# 摘要

在本章中，我们广泛讨论了应用程序代码结构以及集成 React 前端与 Flask 后端所需的一些关键步骤。首先，您需要设置前端以与后端通信，使用 HTTP 客户端库，并处理表单和用户输入。

然后，您需要设置 Flask 后端，包括必要的路由和函数来处理前端发出的请求并处理表单数据。最后，您需要测试整个应用程序以确保其正确且按预期工作。

通过这些步骤，您可以成功地将 React 前端与 Flask 后端集成到您的 Web 应用程序中。在下一章中，我们将通过创建更多表格来扩展 React-Flask 交互。这些表格将具有关系，我们将能够获取并显示数据。
