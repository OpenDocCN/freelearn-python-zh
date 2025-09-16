

# 第十一章：在 React-Flask 应用程序中获取和显示数据

在上一章中，您成功地将 React 前端集成到 Flask 后端。这是全栈 Web 开发者旅程中的一个重要里程碑。在本章中，您将在此基础上继续学习，并深入探讨全栈 Web 应用程序中的数据获取。

在 Web 应用程序中，数据获取非常重要，因为它允许应用程序从后端服务器、API 或数据库中检索数据并向用户显示这些数据。如果没有获取数据的能力，Web 应用程序将仅限于显示硬编码的数据，这不会很有用或动态。通过从后端服务器或 API 获取数据，应用程序可以向用户显示最新和动态的数据。

此外，数据获取通常与用户交互和数据更新结合使用，使应用程序能够执行诸如在数据库或 API 中插入、更新或删除数据等操作。这使得应用程序能够更加互动并对用户的操作做出响应。

在本章中，您将了解数据获取的复杂性及其在 Web 应用程序中的关键作用，更重要的是，它如何涉及将 React 前端与 Flask 后端集成。您将了解数据获取在使 Web 应用程序能够从后端服务器或 API 获取数据并确保显示当前和动态信息方面的作用。

我们将讨论结合用户交互使用数据获取来执行诸如检索、插入、更新或删除数据库或 API 中的数据等操作。最后，我们将讨论如何在 React-Flask 应用程序中管理分页。

到本章结束时，您将了解如何向数据库中添加数据、显示数据库数据以及如何在 React-Flask Web 应用程序中处理分页。

在本章中，我们将涵盖以下主题：

+   获取和显示数据 – React-Flask 方法

+   向数据库中添加数据 – React-Flask 方法

+   编辑数据 – React-Flask 方法

+   从数据库中删除数据 – React-Flask 方法

+   在 React-Flask 应用程序中管理分页

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter11`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter11)。

由于页面数量限制，一些代码块已被截断。请参阅 GitHub 以获取完整代码。

# 获取和显示数据 – React-Flask 方法

在本章中，首先，我们将检索演讲者的数据并将其显示给应用程序的用户。但在进入这一部分之前，让我们进行一些代码重构。你需要重构后端以适应项目目录中`app.py`文件内容的增长。将代码划分为不同的组件可以改善应用程序的整体结构和组织。

而不是将所有代码放在一个模块中，你可以将代码结构化以分离关注点。我们将在*第十四章*中讨论更多关于大型应用程序的代码结构化，即*模块化架构 – 蓝图的威力*。通过这种代码拆分，开发者可以轻松地定位和修改代码库的特定部分，而不会影响其他组件。这种模块化方法也促进了代码的可重用性。

现在，回到代码，你将在后端项目目录（`bizza/backend/models.py`）中添加`models.py`，以存放所有数据库交互的模型。这将帮助我们分离应用程序的关注点。`app.py`文件将用于处理端点和它们相关的逻辑，而`models.py`文件包含应用程序数据模型。

重新结构化的`app.py`和`models.py`文件可以在 GitHub 上找到，网址为[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter11`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter11)。

实质上，我们将为我们的*Bizza*应用程序模拟一个管理页面，以便我们可以创建、显示和编辑演讲者数据，并通过管理页面进行分页管理。目前，我们仅为了演示目的设置管理页面；我们不会去烦恼数据验证、身份验证和授权的实现。

在本节中，重点将是如何从后端检索数据并在 React 前端显示它。能够从数据库中显示数据非常重要，因为它允许你以视觉和交互式的方式向用户展示数据。通过在 Web 应用程序中显示数据，你可以创建一个用户友好的界面，使用户能够查看、搜索、过滤和按需操作数据。

为了创建一个功能强大且有用的 Web 应用程序，你需要从数据库中获取并显示数据。为了从后端检索数据，我们将使用 Axios 进行网络请求。你可以使用 Axios 向后端服务器发送`GET`请求并检索所需的数据。

让我们深入了解如何从后端检索演讲者列表及其详细信息，并在我们的*Bizza*应用程序的管理页面中显示它们。

## 从 Flask 中检索演讲者列表

Flask 后端将通过简单的 API 管理演讲者的列表及其详细信息。在`models.py`中，添加以下代码以创建`Speaker`模型类：

```py
from datetime import datetimeclass Speaker(db.Model):
    __tablename__ = 'speakers'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.String(200), nullable=False)
    speaker_avatar = db.Column(db.String(100),
        nullable=True)
    created_at = db.Column(db.DateTime,
        default=datetime.utcnow)
    updated_at = db.Column(db.DateTime,
        default=datetime.utcnow, onupdate=datetime.utcnow)
    def __repr__(self):
        return f'<Speaker {self.name}>'
    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'company': self.company,
            'position': self.position,
            'bio': self.bio,
            'speaker_avatar': self.speaker_avatar,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
```

上述代码定义了一个`Speaker`模型，并具有`__repr__()`和`serialize()`方法。`__repr__`方法是 Python 中的一个内置方法，用于创建对象的字符串表示。在这种情况下，它用于创建`Speaker`对象的字符串表示。

`serialize()`方法用于将`Speaker`对象转换为字典格式，可以轻松地转换为 JSON。这在您需要将`Speaker`对象作为 API 端点的响应返回时非常有用。

该方法返回一个包含`Speaker`对象所有属性的字典，如`id`、`name`、`email`、`company`、`position`、`bio`、`speaker_avatar`、`created_at`和`updated_at`。`created_at`和`updated_at`属性使用`isoformat()`方法转换为字符串格式。

现在，让我们创建一个端点来处理显示演讲者数据的逻辑：

```py
@app.route('/api/v1/speakers', methods=['GET'])def get_speakers():
    speakers = Speaker.query.all()
    if not speakers:
        return jsonify({"error": "No speakers found"}), 404
    return jsonify([speaker.serialize() for speaker in
        speakers]), 200
```

上述代码使用`get_speakers()`函数从数据库中检索演讲者列表。现在，您需要更新 React 前端目录以消费 API 演讲者列表端点。

## 在 React 中显示数据

在 React 前端，您需要创建一个路由，在`http://127.0.0.1:3000/admin`路径上渲染一个组件。

以下代码片段将创建用于管理员的路由系统：

```py
const router = createBrowserRouter([  {
    path: "/admin",
    element: <AdminPage/>,
    children: [
      {
        path: "/admin/dashboard",
        element: <Dashboard />,
      },
      {
        path: "/admin/speakers",
        element: <Speakers />,
      },
      {
        path: "/admin/venues",
        element: <Venues />,
      },
      {
        path: "/admin/events",
        element: <Events />,
      },
      {
        path: "/admin/schedules",
        element: <Schedules />,
      },
      {
        path: "/admin/sponsors",
        element: <Sponsors />,
      },
    ],
  },
]);
```

现在，让我们在`/src/pages/Admin/AdminPage/AdminPage.jsx`文件中创建`AdminPage`。`AdminPage`将作为管理员的索引组件页面，并渲染必要的组件，包括演讲者的 CRUD 操作。

将以下代码添加到`AdminPage.jsx`文件中：

```py
import React from "react";import { Outlet } from "react-router-dom";
import Sidebar from
    "../../../components/admin/Sidebar/Sidebar";
import './AdminPage.css'
const AdminPage = () => {
    return (
        <div className="container">
            <div><Navbar/></div>
            <div><Outlet /></div>
        </div>
    );
};
export default AdminPage;
```

上述代码显示了`AdminPage`组件，它代表了`admin`页面的结构和内容。`Sidebar`组件被导入并作为子组件渲染，以渲染管理员的侧边菜单列表。然后，我们有从`react-router-dom`包中导入的`Outlet`组件，它用于渲染当前路由的特定内容。

接下来，我们将创建一个用于查看数据库中演讲者列表的数据获取组件。

### 使用 ViewSpeakers 组件显示演讲者列表

我们将使用`ViewSpeakers`组件开始对演讲者的 CRUD 操作，该组件将处理从后端到管理员用户的演讲者数据的显示。

首先，我们将创建一个名为`SpeakersAPI.js`的模块来处理所有的 API 调用。`SpeakersAPI.js`模块封装了 API 调用，抽象出了制作 HTTP 请求的低级细节。这还将允许应用程序的其他部分以更直接的方式与 API 交互，而无需直接处理 Axios 库的复杂性。总的来说，您会从拥有这个独立的模块来处理 API 调用中受益，因为它促进了代码的组织、可重用性、错误处理、头部管理以及代码库的可扩展性和可维护性。

现在，让我们深入了解`SpeakersAPI`模块。

在 `bizza/frontend/src` 项目目录中，创建 `SpeakersAPI.js` 并添加以下代码片段：

```py
import axios from 'axios';const API_URL = 'http://localhost:5000/api/v1';
// Function to handle errors
const handleErrors = (error) => {
    if (error.response) {
    // The request was made and the server responded with a
       status code
    console.error('API Error:', error.response.status,
        error.response.data);
    } else if (error.request) {
    // The request was made but no response was received
    console.error('API Error: No response received',
        error.request);
    } else {
    // Something else happened while making the request
    console.error('API Error:', error.message);
    }
    throw error;
};
// Function to set headers with Content-Type:
   application/json
const setHeaders = () => {
    axios.defaults.headers.common['Content-Type'] =
        'application/json';
};
// Function to get speakers
export const getSpeakers = async () => {
    try {
        setHeaders();
        const response =
            await axios.get(`${API_URL}/speakers`);
        return response.data;
    } catch (error) {
        handleErrors(error);
    }
};
```

前面的代码为使用 Axios 向 API 发送 HTTP 请求设置了基本配置，并提供了一个从 API 获取演讲者的函数。它处理错误并设置请求所需的必要头信息。

接下来，我们将定义 `ViewSpeakers` 组件并使用前面的 `SpeakersAPI` 模块。

在 `src/pages/Admin/Speakers/` 目录中创建 `ViewSpeakers.js` 组件并添加以下代码：

```py
import React, { useEffect, useState } from 'react';import { getSpeakers } from
    '../../../services/SpeakersAPI';
const ViewSpeakers = () => {
    const [speakers, setSpeakers] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const fetchSpeakers = async () => {
        try {
            const speakerData = await getSpeakers();
            setSpeakers(speakerData);
            setIsLoading(false);
        } catch (error) {
            setError(error.message);
            setIsLoading(false);
        }
    };
    useEffect(() => {
        fetchSpeakers();
    }, []);
```

前面的代码设置了一个名为 `ViewSpeakers` 的 React 组件，该组件使用 `getSpeakers` 函数获取演讲者数据并相应地更新组件的状态。它处理加载和错误状态，并在组件挂载时触发数据获取过程。`ViewSpeakers.js` 的完整代码可以在 GitHub 仓库中找到。

接下来，我们将探讨如何使用 Flask-React 方法将数据添加到数据库中。

# 将数据添加到数据库 – React-Flask 方法

我们将数据添加到数据库以存储和组织易于访问、管理和更新的信息。这是持久化存储数据的一种方式，了解如何执行此操作是任何全栈开发者的关键要求。这种知识使您能够构建动态和交互式网络应用程序。然后您就有能力高效地检索和使用数据，用于各种目的，如报告、分析和决策制定。

## 将数据添加到 Flask

现在，让我们创建一个端点来处理将演讲者数据添加到数据库的逻辑：

```py
    @app.route('/api/v1/speakers', methods=['POST'])    def add_speaker():
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        company = data.get('company')
        position = data.get('position')
        bio = data.get('bio')
        avatar = request.files.get('speaker_avatar')
        # Save the uploaded avatar
        if avatar and allowed_file(avatar.filename):
            filename = secure_filename(avatar.filename)
            avatar.save(os.path.join(app.config[
                'UPLOAD_FOLDER'], filename))
        else:
            filename = 'default-avatar.jpg'
        if not name or not email or not company or not
            position or not bio:
            return jsonify({"error": "All fields are
                required"}), 400
        existing_speaker =
            Speaker.query.filter_by(email=email).first()
        if existing_speaker:
            return jsonify({"error": "Speaker with that
                email already exists"}), 409
        speaker = Speaker(name=name, email=email,
            company=company, position=position, bio=bio,
                speaker_avatar=avatar)
        db.session.add(speaker)
        db.session.commit()
        return jsonify(speaker.serialize()), 201
  # Function to check if the file extension is allowed
    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower(
            ) in app.config['ALLOWED_EXTENSIONS']
```

之前的代码定义了一个 `/api/v1/speakers` 路由，该路由处理添加新演讲者的 `POST` 请求。它从请求中提取所需的演讲者信息，验证数据，如果提供则保存头像文件，检查重复的电子邮件，创建新的演讲者对象，将其添加到数据库中，并返回包含创建的演讲者数据的响应。

前面的代码显示了在向指定路由发出 `POST` 请求时执行的 `add_speaker` 函数。

`add_speaker` 函数使用 `request.get_json()` 从请求中检索 JSON 数据，并从数据中提取演讲者的姓名、电子邮件、公司、职位、个人简介和 `speaker_avatar`（一个上传的文件）。

如果提供了 `speaker_avatar` 并且文件扩展名是允许的（在 `allowed_file` 函数检查后），则将头像文件以安全文件名保存到服务器的上传文件夹中。否则，将分配一个默认的头像文件名。

函数随后检查是否提供了所有必需的字段（`name`、`email`、`company`、`position` 和 `bio`）。如果任何字段缺失，它将返回一个包含错误消息和状态码 `400`（错误请求）的 JSON 响应。

接下来，`add_speaker()` 函数查询数据库以检查是否存在具有相同电子邮件的演讲者。如果找到具有相同电子邮件的演讲者，它将返回一个包含错误消息和状态码 `409`（冲突）的 JSON 响应。

如果演讲者是新的（没有具有相同电子邮件的现有演讲者），则使用提供的信息（包括头像文件）创建一个新的 `Speaker` 对象。然后，演讲者被添加到数据库会话并提交。

最后，`add_speaker()` 函数返回一个包含序列化演讲者数据和状态码 `201`（已创建）的 JSON 响应，以指示演讲者创建成功。该代码还包括一个辅助函数 `allowed_file`，该函数根据应用程序的配置检查给定的文件名是否具有允许的文件扩展名。

接下来，我们将设置 React 组件以将演讲者数据添加到后端。

## 使用 CreateSpeaker 组件将演讲者数据添加到后端

在本节中，我们将向后端添加演讲者数据。我们将创建一个名为 `CreateSpeaker` 的组件。此组件将处理添加新演讲者的表单输入并将数据发送到后端 API 以进行存储。

首先，我们将 `AddSpeaker` 函数添加到 API 调用服务模块 `SpeakersAPI.js`：

```py
// API function to add a speakerexport const addSpeaker = (speakerData) => {
    const url = `${API_URL}/speakers`;
    return axios
        .post(url, speakerData, { headers: addHeaders() })
        .then((response) => response.data)
        .catch(handleErrors);
};
```

上述代码提供了一个 `addSpeaker` 函数，该函数使用 Axios 向后端 API 发送 `POST` 请求以添加新的演讲者。它适当地处理请求、响应和错误情况。

现在，我们将在 `src/pages/Admin/Speakers` 内创建 `CreateSpeaker.js` 组件并添加以下代码：

```py
import React, { useState } from 'react';import { addSpeaker } from
    '../../../services/SpeakersAPI'LP;
import { useNavigate } from 'react-router-dom';
const CreateSpeaker = () => {
    const [name, setName] = useState('');
{/* Rest of inputs states */}
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [successMessage, setSuccessMessage] =
        useState('');
    const navigate = useNavigate();
    const handleSubmit = async (event) => {
        event.preventDefault();
        setIsLoading(true);
        setError(null);
        try {
            ...
            await addSpeaker(formData);
            setIsLoading(false);
            // Reset the form fields
            setName('');
            setEmail('');
            setCompany('');
            setPosition('');
            setBio('');
            setAvatar(null);
            // Display a success message
            ...
        )}
        </div>
    );
};
export default CreateSpeaker;
```

上述代码定义了一个 `CreateSpeaker` 组件，该组件处理新演讲者的创建。它管理表单输入值、头像文件选择、加载状态、错误消息和成功消息。当表单提交时，该组件将数据发送到后端 API 并相应地处理响应：

+   组件导入必要的依赖项，包括 `React`、`useState` 钩子、从 `SpeakersAPI` 导入的 `addSpeaker` 函数以及从 `react-router-dom` 导入的 `useNavigate` 钩子。

+   在 `CreateSpeaker` 组件内部，它使用 `useState` 钩子设置状态变量以存储表单输入值（`name`、`email`、`company`、`position` 和 `bio`）、头像文件、加载状态、错误消息和成功消息。`CreateSpeaker` 组件还使用 `useNavigate` 钩子来处理导航。

+   组件定义了一个 `handleSubmit` 函数，当表单提交时触发。它首先阻止默认的表单提交行为。然后，将加载状态设置为 true 并清除任何之前的错误消息。在 `handleSubmit` 函数内部，组件构造一个 `FormData` 对象并将表单输入值和头像文件附加到它。

+   从 `SpeakersAPI` 导入的 `addSpeaker` 函数与构造的 `FormData` 对象一起调用，该对象向后端 API 发送 `POST` 请求以创建新的演讲者。

+   如果请求成功，将加载状态设置为 false，并重置表单输入值。显示成功消息，并将用户导航到`/speakers`页面。如果在 API 请求过程中发生错误，将加载状态设置为 false，并将错误消息存储在状态中。

+   该组件还包括一个`handleAvatarChange`函数，用于在头像输入字段中选择文件时更新头像状态变量。

+   组件的渲染函数返回 JSX 元素，包括带有表单输入和提交按钮的表单。它还根据相应的状态变量显示错误和成功消息。

现在，让我们进入下一节，探索如何在 React-Flask 应用程序中编辑数据。

# 数据编辑 – React-Flask 方法

除了显示和添加数据外，对于 Web 应用程序来说，允许用户编辑数据也很重要。在本节中，您将学习如何在 React-Flask Web 应用程序中实现数据编辑。

## 在 Flask 中编辑数据

现在，让我们添加端点来处理在数据库中更新演讲者数据的逻辑。将以下代码添加到`app.py`中：

```py
from flask import jsonify, requestfrom werkzeug.utils import secure_filename
@app.route('/api/v1/speakers/<int:speaker_id>',
    methods=['PUT'])
def update_speaker(speaker_id):
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    company = data.get('company')
    position = data.get('position')
    bio = data.get('bio')
    avatar = request.files.get('speaker_avatar')
    speaker = Speaker.query.get(speaker_id)
    if not speaker:
        return jsonify({"error": "Speaker not found"}), 404
    if not all([name, email, company, position, bio]):
        return jsonify({"error": "All fields are
            required"}), 400
    if email != speaker.email:
        existing_speaker =
            Speaker.query.filter_by(email=email).first()
```

上述代码定义了一个新的路由，用于在`/api/v1/speakers/int:speaker_id`端点更新演讲者的信息，该端点接受`PUT`请求。使用`@app.route`装饰器定义端点，并将`methods`参数设置为`['PUT']`，以指定此路由只能接受`PUT`请求。《int:speaker_id》部分是路径参数，允许路由接受演讲者 ID 作为 URL 的一部分。

代码定义了`update_speaker`函数，它接受一个`speaker_id`参数，该参数对应于端点中的路径参数。

代码首先获取请求的 JSON 有效负载，并从中提取演讲者的信息。然后，使用`Speaker.query.get(speaker_id)`方法从数据库中检索演讲者的信息。该函数根据提供的`speaker_id`查询数据库以检索现有的演讲者对象。如果没有找到演讲者，它将返回一个包含错误消息和状态码`404`（未找到）的 JSON 响应。

`update_speaker()`检查是否提供了所有必需的字段（`name`、`email`、`company`、`position`和`bio`）。如果任何字段缺失，它将返回一个包含错误消息和状态码`400`（错误请求）的 JSON 响应。

如果保存图像时出现异常，它将删除之前的头像图像，并返回错误消息和状态码。然后，`update_speaker`函数更新数据库中的演讲者信息。`update_speaker`函数尝试将更改提交到数据库；如果失败，它将回滚事务并返回错误消息和状态码`500`。

最后，如果一切顺利，代码将返回更新后的演讲者信息作为 JSON 对象和状态码`200`。

接下来，我们将创建一个 React 组件来处理更新演讲者数据。

## 在 React 中显示编辑后的数据

在本节中，我们将提供编辑演讲者信息的功能。要在 React 中编辑数据，我们可以通过修改组件的状态来使用更新的值，并在用户界面中反映这些更改。我们将首先添加`UpdateSpeaker`组件。在`frontend/src/pages/Admin/Speakers/UpdateSpeaker.js`中，添加以下代码：

```py
import React, { useState, useEffect } from 'react';import { updateSpeaker } from
    '../../../services/SpeakersAPI';
import { useNavigate } from 'react-router-dom';
const UpdateSpeaker = ({ speakerId }) => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [company, setCompany] = useState('');
    const [position, setPosition] = useState('');
    const [bio, setBio] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [successMessage, setSuccessMessage] =
        useState('');
    const navigate=useNavigate();
    useEffect(() => {
        // Fetch the speaker data based on speakerId
        fetchSpeaker();
    }, [speakerId]);
    const fetchSpeaker = async () => {
        try {
            // Fetch the speaker data from the backend
                based on speakerId
            const speakerData =
                await getSpeaker(speakerId);
            setName(speakerData.name);
            setEmail(speakerData.email);
            setCompany(speakerData.company);
            setPosition(speakerData.position);
            setBio(speakerData.bio);
        } catch (error) {
            setError(error.message);
        }
    };
{/* The rest of the code snippet can be found on GitHub */}
```

上述代码定义了一个名为`UpdateSpeaker`的组件。该组件允许用户通过使用`SpeakersAPI.js`文件中的`updateSpeaker`函数向服务器发送`PUT`请求来更新演讲者的信息。

组件首先从 React 库中导入`React`、`useState`和`useEffect`，从`SpeakersAPI.js`模块中导入`updateSpeaker`。当表单提交时，将调用`handleSubmit`函数；它调用`SpeakersAPI.js`文件中的`updateSpeaker`函数，并传递`speakerId`和一个包含更新后的演讲者信息的对象。如果请求成功，它将成功状态设置为 true，如果有错误，它将错误状态设置为`error.message`。

现在，你需要更新`src/services/SpeakersAPI.js`中的`SpeakersAPI.js`文件，以添加`updateSpeaker` API 调用函数：

```py
// API function to update a speakerexport const updateSpeaker = (speakerId, speakerData) => {
    const url = `${API_URL}/speakers/${speakerId}`;
    return axios
        .put(url, speakerData, { headers: addHeaders() })
        .then((response) => response.data)
        .catch(handleErrors);
};
```

上述代码定义了一个用于在后端更新演讲者信息的`updateSpeaker` API 函数：

+   该函数接受两个参数：`speakerId`（表示要更新的演讲者的 ID）和`speakerData`（一个包含更新后的演讲者信息的对象）。

+   它通过将`speakerId`附加到基本 URL 来构造 API 端点的 URL。

+   该函数使用 Axios 库向构造的 URL 发送一个`PUT`请求，将`speakerData`作为请求负载传递，并使用`addHeaders`函数包含适当的头信息。

+   如果请求成功，它返回响应数据。如果在请求过程中发生错误，它捕获错误并调用`handleErrors`函数来处理和传播错误。

接下来，你将学习如何从数据库中删除演讲者数据。

# 从数据库中删除数据 – React-Flask 方法

从数据库中删除数据涉及从表中删除一个或多个记录或行。在本节中，你将学习如何在 React-Flask 网络应用程序中处理删除请求。

## 在 Flask 中处理删除请求

让我们创建一个端点来处理从数据库中删除演讲者数据的逻辑：

```py
@app.route('/api/v1/speakers/<int:speaker_id>',    methods=['DELETE'])
def delete_speaker(speaker_id):
    speaker = Speaker.query.get_or_404(speaker_id)
    if not current_user.has_permission("delete_speaker"):
        abort(http.Forbidden("You do not have permission to
            delete this speaker"))
    events =
        Event.query.filter_by(speaker_id=speaker_id).all()
    if events:
        abort(http.Conflict("This speaker has associated
            events, please delete them first"))
    try:
        if speaker.speaker_avatar:
            speaker_avatar.delete(speaker.speaker_avatar)
        with db.session.begin():
            db.session.delete(speaker)
    except Exception:
        abort(http.InternalServerError("Error while
            deleting speaker"))
    return jsonify({"message": "Speaker deleted
        successfully"}), http.OK
```

上述代码定义了一个用于删除演讲者的 API 路由，执行必要的检查，从数据库中删除演讲者，处理错误，并返回适当的响应。

接下来，我们将探讨用于处理前端删除请求的 React 组件。

## 在 React 中处理删除请求

当构建一个 React 应用程序时，你可以通过创建一个与后端 API 交互的组件来处理删除请求以删除演讲者资源。此组件将向适当的端点发送删除请求，处理任何潜在的错误，并根据删除演讲者资源更新组件的状态。

让我们从创建一个 `DeleteSpeaker` 组件开始。在 `frontend/src/pages/Admin/Speakers/DeleteSpeaker.js` 中，添加以下代码：

```py
import React, { useState, useEffect } from "react";import { useParams, useNavigate } from "react-router-dom";
import { deleteSpeaker } from "./api/SpeakersAPI";
const DeleteSpeaker = () => {
    const { speakerId } = useParams();
    const navigate = useNavigate();
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const handleDelete = async () => {
        try {
            setIsLoading(true);
            await deleteSpeaker(speakerId);
            setIsLoading(false);
            navigate("/speakers"); // Redirect to speakers
                                      list after successful
                                      deletion
        } catch (err) {
            setIsLoading(false);
            setError("Failed to delete speaker.");
        }
    };
    useEffect(() => {
        return () => {
            // Clear error message on component unmount
            setError("");
        };
    }, []);
    return (
        <div>
            {error && <p className="error">{error}</p>}
            <p>Are you sure you want to delete this
                speaker?</p>
            <button onClick={handleDelete}
                disabled={isLoading}>
                {isLoading ? "Deleting..." : "Delete
                Speaker"}
            </button>
        </div>
    );
};
export default DeleteSpeaker;
```

上述代码定义了一个组件，允许用户通过 `id` 删除演讲者。组件首先从 `react-router-dom` 中导入 `useParams` 和 `useNavigate` 钩子，以从 URL 中提取 `speakerId` 值。它还从 `src/services/SpeakersAPI.js` 中导入 `deleteSpeaker` 函数，以通过后端 API 调用处理演讲者的删除。然后，组件使用 `useState` 钩子初始化两个状态变量：`error` 和 `success`。

该组件有一个单按钮，点击时会触发 `handleDelete` 函数。此函数阻止默认的表单提交行为，然后调用 `deleteSpeaker` 函数，并将 `speakerId` 作为参数传递。如果删除成功，它将成功状态设置为 true；否则，它将错误状态设置为从 API 返回的错误信息。然后，该组件渲染一条消息，指示删除是否成功或存在错误。

现在，你需要更新 `src/api/SpeakersAPI.js` 中的 `SpeakersAPI.js` 文件，以添加 `deleteSpeaker` API 调用函数：

```py
// API function to delete a speakerexport const deleteSpeaker = async (speakerId) => {
    const url = `/api/v1/speakers/${speakerId}`;
    try {
        const speakerResponse = await axios.get(url);
        const speaker = speakerResponse.data;
        if (!speaker) {
            throw new Error("Speaker not found");
        }
      const eventsResponse = await
          axios.get(`/api/v1/events?speakerId=${speakerId}`
          );
      const events = eventsResponse.data;
      if (events.length > 0) {
        throw new Error("This speaker has associated
            events, please delete them first");
      }
      await axios.delete(url);
      return speaker;
    } catch (err) {
        if (err.response) {
            const { status, data } = err.response;
            throw new Error(`${status}: ${data.error}`);
        } else if (err.request) {
            throw new Error('Error: No response received
                from server');
        } else {
            throw new Error(err.message);
        }
    }
};
```

上述代码定义了一个 `deleteSpeaker` 函数，该函数接受一个 `speakerId` 作为其参数。该函数使用 Axios 库向服务器发送 HTTP 请求。该函数首先尝试通过向 `/api/v1/speakers/{speakerId}` 端点发送 `GET` 请求从服务器获取演讲者详细信息。

然后，它会检查演讲者是否存在。如果演讲者不存在，该函数会抛出一个错误，错误信息为向 `/api/v1/events?speakerId=${speakerId}` 端点发送 `GET` 请求以获取与演讲者相关的事件列表。然后，它会检查事件长度是否大于 `0`。如果是，它会抛出一个错误，错误信息为 **“此演讲者有关联事件，请先删除”** **它们**。

最后，该函数向 `/api/v1/speakers/{speakerId}` 端点发送一个 `DELETE` 请求以删除演讲者。如果在过程中出现错误，该函数会检查错误并抛出一个适当的错误信息。然后，该函数导出 `deleteSpeaker` 函数，以便在其他应用程序的部分中导入和使用。

接下来，我们将讨论如何在 React-Flask 应用程序中处理分页。

# 在 React-Flask 应用程序中管理分页

当处理大量数据集时，实现分页对于使大量数据集对用户更易于管理非常重要。分页是一种将大量数据集划分为更小、更易于管理的块（称为**页面**）的技术。每个页面包含总数据的一个子集，使用户能够以受控的方式浏览数据。

分页提供了一种高效展示大量数据集的方法，通过使数据更易于访问来提高性能并增强用户体验。在本节中，你将学习如何在 React-Flask 网络应用程序中实现分页。要实现分页，你需要对后端服务器进行一些修改以处理分页请求。

你可以使用 Flask-SQLAlchemy 库在后台处理分页。在 Flask 后端，你可以使用 Flask-SQLAlchemy 库的分页功能来实现 `speaker` 模型的分页。让我们深入了解如何为 `Speaker` 模型实现分页。

在 `app.py` 文件中更新 `get_speakers()` 函数如下所示：

```py
from flask_sqlalchemy import Pagination@app.route('/api/v1/speakers', methods=['GET'])
def get_speakers():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    speakers = Speaker.query.paginate(page, per_page,
        False)
    if not speakers.items:
        return jsonify({"error": "No speakers found"}), 404
    return jsonify({
        'speakers': [speaker.serialize() for speaker in
            speakers.items],
        'total_pages': speakers.pages,
        'total_items': speakers.total
    }), 200
```

在前面的代码中，我们使用 Flask-SQLAlchemy 的 `paginate()` 方法为演讲者集合添加分页功能。`page` 和 `per_page` 参数作为查询参数在 `GET` 请求中传递。`page` 的默认值为 `1`，`per_page` 的默认值为 `10`。

对于 React 前端，你可以在函数组件中使用 `useState` 和 `useEffect` 钩子来处理分页。

让我们修改 `ViewSpeakers` 组件并为其添加分页功能：

```py
import React, { useState, useEffect } from 'react';import { getSpeakers } from
    '../../../services/SpeakersAPI';
const ViewSpeakers = () => {
    const [speakers, setSpeakers] = useState([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [speakersPerPage] = useState(10);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    useEffect(() => {
        fetchSpeakers();
    }, []);
};
export default ViewSpeakers;
```

上述代码定义了一个组件，该组件使用分页显示演讲者列表。该组件利用 React 钩子来管理其状态。`speakers` 状态变量用于存储演讲者列表，而 `page` 和 `perPage` 状态变量分别用于存储当前页码和每页显示的项目数。

使用 `useEffect` 钩子在组件挂载时以及 `page` 或 `perPage` 状态变量更改时从服务器获取演讲者信息。`fetchSpeakers` 函数使用 Axios 库向 `'/api/v1/speakers?page=${page}&per_page=${perPage}'` 端点发起 `GET` 请求，并将当前页码和每页的项目数作为查询参数传递。

响应数据随后存储在 `speakers` 状态变量中。`ViewSpeakers` 组件随后遍历演讲者数组并显示每位演讲者的姓名和电子邮件。该组件还包括两个按钮，一个用于导航到上一页，另一个用于导航到下一页。

这些按钮的 `onClick` 处理程序相应地更新页面状态变量，并使用 `1` 防止用户导航到不存在的上一页。

# 摘要

在本章中，我们详细讨论了如何在 React-Flask 网络应用程序中获取和显示数据。我们考察了处理获取和显示数据的一种方法。您能够从后端开始，定义`Speaker`模型类，并实现各种端点来处理从数据库中获取数据，以及在该数据库上添加、更新和删除数据。

我们使用了 Axios 库向 Flask 后端发送请求，然后从数据库检索数据并将其以响应的形式返回到前端。React 前端随后处理响应并将数据显示给最终用户。最后，我们实现了分页作为一种高效展示大量数据集并提高 React-Flack 网络应用程序项目性能的方法。

接下来，我们将讨论在 React-Flask 应用程序中的身份验证和授权，并检查确保您的应用程序安全且准备就绪的最佳实践。
