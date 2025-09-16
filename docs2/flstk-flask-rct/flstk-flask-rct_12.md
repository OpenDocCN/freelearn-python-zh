

# 第十二章：身份验证和授权

在构建全栈 Web 应用程序时，你通常会希望实现一个系统，让用户信任你处理他们的敏感信息。作为一名全栈 Web 开发者，了解如何实现强大的身份验证和授权机制至关重要。你需要知道如何保护用户数据的安全和应用程序的完整性。想象一下，你正在构建一个允许用户在线购物的电子商务网站。

如果你未能正确地验证和授权用户，那么有人可能会未经授权访问网站并使用他人的个人信息下订单。这可能导致合法用户的财务损失，并损害在线业务或你的客户的声誉。

此外，如果你未能正确地验证和授权用户，这也可能使你的 Web 应用程序容易受到诸如 SQL 注入等攻击，攻击者可以访问存储在数据库中的敏感信息。这可能导致客户数据的丢失，并且可能面临法律后果。

在本章中，我们将深入 Web 安全的世界，探索保护 Flask Web 应用程序的最佳实践和技术。正如著名的计算机科学家布鲁斯·施奈尔（Bruce Schneier）曾经说过：“安全是一个过程，而不是一个产品”([`www.schneier.com/essays/archives/2000/04/the_process_of_secur.html`](https://www.schneier.com/essays/archives/2000/04/the_process_of_secur.html))。本章将为你提供了解信息安全的重要性以及如何在 Flask 应用程序中实施它的知识和技能。

从理解身份验证和授权的基本原理到管理用户会话和创建具有安全密码的账户，本章将涵盖 Web 应用程序安全的关键要素。我们将检查保护你的 Flask 应用程序的过程，并展示如何在实践中实现这些概念。

在本章中，你将学习以下主题：

+   理解信息安全的基本原理

+   定义 Web 应用程序中的身份验证和身份验证角色

+   实现密码安全和散列密码

+   理解 Web 应用程序开发中的访问和授权

+   将身份验证添加到你的 Flask 应用程序中

+   识别系统用户并管理他们的信息

+   会话管理

+   创建受密码保护的仪表板

+   在 Flask 中实现闪存消息

# 技术要求

本章的完整代码可在 GitHub 上找到：[`github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter12`](https://github.com/PacktPublishing/Full-Stack-Flask-and-React/tree/main/Chapter12)。

# 理解信息安全的基本原理

信息安全是 Web 应用程序开发的关键方面。在当今数字时代，个人和敏感信息通常通过 Web 应用程序存储和传输，这使得它们容易受到各种类型的网络安全威胁。这些威胁的范围从简单的攻击，如**SQL 注入**和**跨站脚本**（**XSS**），到更复杂的攻击，如**中间人攻击**（**MITM**）和**分布式拒绝服务**（**DDoS**）。

让我们深入了解一些可能危害您的 Web 应用程序安全性的各种威胁类型：

+   `用户名`和`密码`详细信息。如果应用程序容易受到 SQL 注入攻击，攻击者可以在密码字段中输入类似`' OR '1'='1`的内容。

    SQL 查询可能变为`SELECT * FROM users WHERE username = 'username' AND password = '' OR '1'='1';`，这可能会让攻击者无需有效密码即可登录。

+   `<script>malicious_scripts()</script>`，其他查看评论部分的用户可能会无意中执行该脚本。

+   **跨站请求伪造**（**CSRF**）：这是一种攻击，攻击者诱使用户不知情地向用户已认证的 Web 应用程序发出请求。这可能导致未经用户同意代表用户执行未经授权的操作。

    CSRF 攻击利用了网站对用户浏览器的信任。例如，一个毫无戒心的用户登录到在线银行网站并获取会话 cookie。攻击者创建了一个包含隐藏表单的恶意网页，该表单提交请求将用户的账户资金转移到攻击者的账户。

    用户访问攻击者的网页，并使用用户的会话 cookie 提交隐藏表单，导致未经授权的数据传输。这种攻击利用了网站对用户浏览器执行未经授权操作的信任。

+   **分布式拒绝服务**（**DDoS**）攻击：这种攻击涉及从多个来源向目标服务器、服务或网络发送大量流量，使其对合法用户不可访问。例如，攻击者可能使用僵尸网络（被破坏的计算机网络）向 Web 应用程序发送大量流量。这可能导致 Web 应用程序变慢或完全无法向用户提供服务。

然而，有一些方法可以减轻这些恶意威胁，它们能够破坏您的 Web 应用程序。现在，我们将强调一些保护 Web 应用程序的最佳实践。

+   **输入验证**：您需要确保所有输入数据都经过适当的清理和验证，以防止 SQL 注入和 XSS 攻击。

+   在 Flask 中使用`SQLAlchemy`，为您处理 SQL 查询的构建，并提供一种安全且高效的方式与数据库交互。

+   **密码存储**：使用强大的哈希算法和为每个用户生成唯一的盐值来安全地存储密码。

+   **使用 HTTPS**：使用 HTTPS 加密客户端和服务器之间的所有通信，以防止窃听和中间人攻击。

+   **会话管理**：正确管理会话以防止会话劫持并修复 Web 应用程序中的会话固定漏洞。

+   **访问控制**：使用基于角色的访问控制来限制对敏感资源和功能的访问。

+   **日志和监控**：您需要持续记录所有应用程序活动的详细日志，并监控可疑活动。

+   **使用最新软件**：您需要定期更新框架、库以及 Web 应用程序使用的所有依赖项，以确保已知漏洞得到修补。

+   使用`X-XSS-Protection`、`X-Frame-Options`和`Content-Security-Policy`来防止某些类型的攻击。

+   **定期测试漏洞**：定期进行渗透测试和漏洞扫描，以识别和修复任何安全漏洞。

在本章的剩余部分，我们将讨论和实现 Flask Web 应用程序中的身份验证和授权，以帮助您确保您的应用程序及其用户数据的安全。

接下来，我们将讨论 Web 应用程序中的身份验证和身份验证角色。这将提高您对如何验证用户身份以及各种身份验证类型的理解。

# 定义 Web 应用程序中的身份验证和身份验证角色

**身份验证**是验证用户身份并确保只有授权用户才能访问应用程序的资源和服务的过程。身份验证是任何 Web 应用程序的重要方面，包括使用 Flask 构建的应用程序。

这通常是通过提示用户提供一组凭证，例如用户名和密码，Web 应用程序可以使用这些凭证来确认用户身份来完成的。在 Web 应用程序开发中，身份验证的目的是确保只有授权用户可以访问敏感信息并在 Web 应用程序中执行某些操作。

在 Web 开发中，我们有几种可以在任何 Web 应用程序项目中使用的身份验证方法。以下是一些最常用的方法：

+   **基于密码的身份验证**：这是我们日常使用中最常见的身份验证形式，涉及用户输入用户名/电子邮件和密码以获取对 Web 应用程序的访问权限。这种方法简单易行，但存在其弱点。基于密码的身份验证容易受到暴力破解和字典攻击等攻击。

+   **多因素认证（MFA）**：这种方法通过要求用户提供多种身份验证形式来增加一个额外的安全层。例如，用户可能需要输入密码，并提供发送到他们的手机或电子邮件的一次性代码。MFA 比基于密码的认证更安全，但可能会对用户体验产生负面影响。

+   **基于令牌的认证**：这种方法涉及向用户发放一个令牌，他们必须向 Web 应用程序出示以获得访问权限。令牌可以是 JWT 或 OAuth 令牌的形式，通常存储在浏览器的 cookies 或本地存储中。令牌可以轻松撤销，这使得维护安全性变得更加容易。

+   **生物识别认证**：这种方法涉及使用生物特征，如指纹、面部识别或语音识别来验证用户的身份。生物识别认证被认为比其他方法更安全，但实施成本可能更高。

当你决定使用哪种认证方法时，考虑 Web 应用程序所需的安全级别和用户体验至关重要。这些认证方法各有优缺点。选择适合你应用程序的正确方法是至关重要的。

例如，如果你正在构建一个需要高度安全性的网络应用程序，你可能想要考虑使用多因素认证（MFA）或生物识别认证。当然，生物识别认证很少在公共或通用网络应用程序中使用。如果你正在构建一个不需要高度安全性的简单网络应用程序，基于密码的认证可能是安全且足够的。

接下来，我们将讨论在 Flask Web 应用程序中实现密码安全和哈希密码的概念。

# 实现密码安全和哈希密码

在任何需要访问的 Web 应用程序中，密码通常是防止未经授权访问的第一道防线。作为开发者，你将想要确保在构建 Flask 应用程序时，密码被安全地管理。Web 应用程序中密码管理的关键组成部分是永远不要以明文形式存储密码。

相反，密码应该被哈希处理，这是一个单向加密过程，它产生一个固定长度的输出，无法被逆转。当用户输入他们的密码时，它会被哈希处理并与存储的哈希值进行比较。如果两个哈希值匹配，则密码正确。哈希密码可以帮助保护免受暴力攻击和字典攻击等攻击。

暴力攻击涉及尝试所有可能的字符组合以找到匹配项，而字典攻击则涉及尝试预计算的单词列表。哈希密码使得攻击者无法在计算上逆转哈希并发现原始密码变得不可行。

在 Flask 中，你可以使用 `Flask-Bcrypt` 这样的库来处理密码哈希。`Flask-Bcrypt` 是一个 Flask 扩展，为 Flask 提供了 `bcrypt` 密码哈希功能。`Flask-Bcrypt` 提供了简单的接口用于哈希和检查密码。你还可以使用 `Flask-Bcrypt` 生成用于密码哈希的随机盐。

让我们快速通过一个使用 `Flask-Bcrypt` 进行密码哈希的例子：

```py
from flask import Flask, render_template, requestfrom flask_bcrypt import Bcrypt
app = Flask(__name__)
bcrypt = Bcrypt()
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        password = request.form.get("password")
        password_hash =
            bcrypt.generate_password_hash(password)
                .decode('utf-8')
        return render_template("index.html",
            password_hash=password_hash)
    else:
        return render_template("index.html")
@app.route("/login", methods=["POST"])
def login():
    password = request.form.get("password")
    password_hash = request.form.get("password_hash")
//Check GitHub for the complete code
if __name__ == "__main__":
    app.run(debug=True)
```

上一段代码使用了 `Flask Bcrypt` 库来哈希和检查密码。它导入了 `Bcrypt` 类和 `check_password_hash` 函数，使用 Flask 应用程序创建了一个 `Bcrypt` 实例。当表单提交时，使用 `flask_bcrypt` 扩展对密码进行哈希处理，并将哈希后的密码在同一页面上显示给用户。`render_template` 函数用于渲染 HTML 模板，而 `Bcrypt` 扩展用于安全的密码哈希。

接下来，我们将讨论网络应用程序开发中的访问和授权。

# 理解网络应用程序开发中的访问和授权

网络应用程序开发中的访问和授权是控制谁可以访问 Web 应用程序中特定资源和操作的过程。作为一个开发者，你将希望设计和确保用户只能执行他们被授权执行的操作，并访问他们被授权访问的资源。

如前所述，认证是验证用户身份的过程。授权是确定用户在 Web 应用程序中可以做什么的过程。当你结合这两个机制时，你就有了一个系统，确保只有授权用户才能访问敏感信息并在 Web 应用程序中执行某些操作。

在 Web 应用程序开发中可以使用多种不同的访问控制方法。我们将讨论其中一些，并具体说明 Flask 如何处理访问和授权：

+   `Flask-Login` 和 `Flask-Security`。

+   `Flask-OAuthlib`。此扩展提供了对 `OAuth 1.0a` 和 `OAuth 2.0` 的支持。`Flask-OAuthlib` 使得开发者在 Flask 应用程序中实现 OAuth 变得容易。

+   `Flask-JWT` 和 `Flask-JWT-Extended`。

    这些扩展提供了令牌生成、验证和过期等功能，以及根据 JWT 中包含的声明来限制对某些资源和操作的访问，以确保它是由可信源生成的且未被篡改。

+   `Flask-RBAC`。

    `Flask-RBAC` 扩展提供了角色管理、权限管理和基于用户角色的限制对某些资源和操作的访问的能力。

+   `Flask-Policies`。`Flask-Policies` 提供了策略管理、执行以及根据策略中指定的条件限制对某些资源和操作的访问的能力。

通过使用这些库，你可以轻松处理用户角色和权限，并根据用户的角色限制对某些视图和路由的访问。接下来，我们将探讨如何在 Flask Web 应用程序中实现身份验证。

# 为你的 Flask 应用程序添加身份验证

JWT 是现代 Web 应用程序中流行的身份验证方法。JWT 是一个经过数字签名的 JSON 对象，可以通过在各方之间传输声明（例如授权服务器和资源服务器）来用于身份验证用户。在 Flask Web 应用程序中，你可以使用`PyJWT`库来编码和解码 JWT 以进行身份验证。

当用户登录 Flask 应用程序时，后端验证用户的凭据，如他们的邮箱和密码，如果它们有效，则生成 JWT 并发送给客户端。客户端将 JWT 存储在浏览器的本地存储或作为 cookie。对于后续请求受保护的路线和资源，客户端在请求头中发送 JWT。

后端解码 JWT 以验证用户的身份，授予或拒绝对请求资源的访问，并为后续请求生成新的 JWT。JWT 身份验证允许无状态身份验证。这意味着身份验证信息存储在 JWT 中，可以在不同的服务器之间传递，而不是存储在服务器的内存中。这使得扩展应用程序更容易，并降低了数据丢失或损坏的风险。

JWT 身份验证还通过使用数字签名来防止数据篡改，从而增强了安全性。签名使用服务器和客户端之间共享的秘密密钥生成。签名确保 JWT 中的数据在传输过程中未被更改。JWT 身份验证是 Flask 应用程序中安全且高效的用户身份验证方法。

通过在 Flask 应用程序中实现 JWT 身份验证，开发者可以简化用户身份验证的过程，并降低安全漏洞的风险。让我们来检查 JWT 的后端和前端实现。

## Flask 后端

以下代码定义了两个 Flask 端点 – `/api/v1/login` 和 `/api/v1/dashboard`：

```py
@app.route('/api/v1/login', methods=['POST'])def login():
    email = request.json.get('email', None)
    password = request.json.get('password', None)
    if email is None or password is None:
        return jsonify({'message': 'Missing email or
            password'}), 400
    user = User.query.filter_by(email=email).first()
    if user is None or not bcrypt.check_password_hash
        (user.password, password):
        return jsonify({'message': 'Invalid email or
            password'}), 401
    access_token = create_access_token(identity=user.id)
    return jsonify({'access_token': access_token}), 200
@app.route('/api/v1/dashboard', methods=['GET'])
@jwt_required
def dashboard():
    current_user = get_jwt_identity()
    user = User.query.filter_by(id=current_user).first()
    return jsonify({'email': user.email}), 200
```

`/api/v1/login` 端点是用于处理用户登录请求的。它接收一个包含两个属性的 JSON 请求：`email` 和 `password`。如果这两个属性中的任何一个缺失，函数将返回一个包含消息“缺少邮箱或密码”和状态码`400`（错误请求）的 JSON 响应。

接下来，该函数查询数据库以查找具有给定邮箱的用户。如果不存在这样的用户，或者提供的密码与数据库中存储的散列密码不匹配，则函数返回一个包含消息“无效的邮箱或密码”和状态码`401`（未授权）的 JSON 响应。

否则，函数会使用 `create_access_token` 函数生成一个 JWT，并将其作为 JSON 响应返回，状态码为 `200`（OK）。JWT 可以用于在后续请求中对用户进行认证。`/api/v1/dashboard` 端点是一个受保护的端点，只有拥有有效 JWT 的用户才能访问。

使用 `jwt_required` 装饰器来强制执行此限制。当访问此端点时，JWT 用于提取用户的身份，然后从数据库中检索用户的 `email`。然后，该电子邮件作为 JSON 响应返回，状态码为 `200`（OK）。

## React 前端

以下代码展示了登录表单和仪表板。`LoginForm` 组件有三个状态 – `email`、`password` 和 `accessToken`。当表单提交时，它会对 `/api/v1/login` 端点发送一个带有电子邮件和密码数据的 `POST` 请求，并将请求的响应存储在 `accessToken` 状态中：

```py
import React, { useState } from 'react';import axios from 'axios';
const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const res = await axios.post('/api/v1/login', {
        email, password });
      setAccessToken(res.data.access_token);
    } catch (err) {
      console.error(err);
    }
  };
  return (
    <>
      {accessToken ? (
        <Dashboard accessToken={accessToken} />
      ) : (
        <form onSubmit={handleSubmit}>
          ....
          />
          <button type="submit">Login</button>
        </form>
      )}
    </>
  );
};
};
 export default LoginForm;
```

`Dashboard` 组件接受一个 `accessToken` 属性，并有一个状态，`email`。它会对 `/api/v1/dashboard` 端点发送一个带有设置为 `accessToken` 的授权头的 `GET` 请求，并将响应存储在 `email` 状态中。该组件显示一条消息，内容为 `"欢迎来到` `dashboard, [email]!"`。

`LoginForm` 组件根据 `accessToken` 是否为真返回 `Dashboard` 组件或登录表单。

接下来，我们将讨论如何识别网络应用程序用户并管理他们的信息。

# 识别系统用户和管理他们的信息

在大多数网络应用程序中，用户通过唯一的标识符（如用户名或电子邮件地址）进行识别。通常，在 Flask 应用程序中，你可以使用数据库来存储用户信息，例如用户名、电子邮件地址和散列密码。

当用户尝试登录时，输入的凭据（用户名和密码）将与数据库中存储的信息进行比较。如果输入的凭据匹配，则用户被认证，并为该用户创建一个会话。在 Flask 中，你可以使用内置的会话对象来存储和检索用户信息。

通过使用会话，你可以在 Flask 网络应用程序中轻松识别用户并检索他们的信息。然而，需要注意的是，会话容易受到会话劫持攻击。因此，使用诸如登录后重新生成会话 ID 和使用安全 cookie 等安全的会话管理技术是至关重要的。

让我们考察一个实现示例：

```py
from flask import Flask, request, redirect, session, jsonifyapp = Flask(__name__)
app.secret_key = 'secret_key'
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    session['email'] = email
    return jsonify({'message': 'Login successful'}), 201
@app.route('/dashboard', methods=['GET'])
def dashboard():
    email = session.get('email')
    user = User.query.filter_by(email=email).first()
    return jsonify({'email': email, 'user':
        user.to_dict()}), 200
```

在前面的代码中，第一行从 Flask 库中导入所需的模块。下一行创建了一个 `Flask` 类的实例，并将其分配给 `app` 变量。`app.secret_key` 属性被设置为 `'secret_key'`，它用于安全地签名会话 cookie。

登录功能被定义为 `api/v1/login` 路径上的 POST 端点。此端点使用 `request.get_json()` 方法从请求体中获取 JSON 数据并提取 `email` 和 `password` 的值。然后使用 `session['email'] = email` 将 `email` 存储在会话中。该函数返回一个包含消息 `"Login successful"` 和状态码 `201` 的 JSON 响应，表示成功创建资源。

然后，仪表板功能被定义为 `api/v1/dashboard` 路径上的 GET 端点。它使用 `session.get('email')` 从会话中检索 `email`。然后，该函数使用 `User.query.filter_by(email=email).first()` 查询数据库以获取具有指定电子邮件的用户。`email` 和用户数据（使用 `to_dict()` 转换为字典）以 JSON 响应的形式返回，状态码为 200，表示成功检索资源。

您还可以使用基于令牌的认证方法在 Flask 应用程序中识别用户。在此方法中，当用户登录时，会向用户颁发一个令牌，并将该令牌存储在用户的浏览器中作为 cookie 或放置在本地存储中。然后，此令牌随用户发出的每个后续请求一起发送，服务器使用此令牌来识别用户。JWT 是常用的令牌格式，`Flask-JWT` 和 `Flask-JWT-Extended` 等库使得在 Flask 中实现基于 JWT 的认证变得简单。

接下来，我们将深入了解在 Web 应用程序中跟踪用户会话。

# 会话管理

`Flask-Session`；在前端 React 端，你可以使用 React 的 `localStorage` 或 `sessionStorage`。

Flask 作为 Python 的首选框架，以其简洁性而闻名，使得构建从小型到企业级大小的 Web 应用程序变得容易。Flask 可以使用内置的会话对象和一些社区成员提供的 Flask 扩展来管理用户会话。

会话对象是一个类似字典的对象，存储在服务器上，可以通过安全的会话 cookie 由客户端访问。要使用会话对象，必须在 Flask 应用程序中设置一个 *密钥*。此密钥用于加密和签名会话数据，这些数据存储在客户端浏览器的安全 cookie 中。当用户访问受保护的资源时，服务器验证会话 cookie，如果 cookie 有效，则授予访问权限。

让我们在 Flask 后端和 React 前端中实现会话管理。我们将创建一个计数器端点，用于跟踪用户访问仪表板页面的次数。

## Flask 后端

我们将使用 `Flask-Session` 来存储会话数据并安全地管理会话。要使用 `Flask-Session`，您需要先安装它。您可以在终端中运行 `pip install flask-session` 命令来完成此操作。

安装 `Flask-Session` 后，您需要将以下代码添加到您的 Flask 应用程序中：

```py
from flask import Flask, sessionfrom flask_session import Session
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
@app.route("/api/v1/couters")
def visit_couter():
    session["counter"] = session.get("counter", 0) + 1
    return "Hey , you have visited this page:
        {}".format(session["counter"])
```

上述代码展示了在 Flask 后端中实现会话管理的简单示例：

1.  第一行导入 Flask 模块，而第二行导入 `Flask-Session` 扩展。

1.  接下来的几行创建了一个 Flask 应用程序对象，并配置会话类型存储在文件系统中。

1.  然后，使用 Flask 应用程序对象作为其参数初始化 `Session` 对象。

1.  `@app.route` 装饰器为 `visit_counter` 函数创建了一个路由 - 在这种情况下，是 `/api/v1/counters` 路由。

1.  `visit_counter` 函数检索会话中 `counter` 键的当前值，如果不存在则将其设置为 `0`，然后增加 `1`。然后，将更新后的值作为响应返回给用户。

让我们探索这个实现中的 React 前端部分。

## React 前端

你可以使用 Axios 库向 Flask 服务器发送 HTTP 请求。如果尚未安装，可以使用 `npm install axios` 命令安装 Axios。

一旦安装了 Axios，你就可以使用它向 Flask 服务器发送 HTTP 请求来设置或获取会话数据：

```py
import React, { useState } from "react";import axios from "axios";
function VisitCouter() {
    const [counter, setCounter] = useState(0);
    const getCounter = async () => {
        const response = await axios.get(
            "http://localhost:5000/api/v1/counters");
        setCounter(response.data.counter);
        };
        return (
          <div>
            <h1>You have visited this page: {counter}
              times!</h1>
            <button onClick={getCounter}>Get Counter
              </button>
          </div>
        );
}
export default VisitCounter;
```

上述代码演示了 React 前端的前端实现，它从 Flask 后端检索访问计数器。

1.  第一行导入所需的库 - 即 `React` 和 `axios`。

1.  下一个部分声明了 `VisitCounter` 函数组件，它返回一个用户视图。

1.  在组件内部，使用 `useState` 钩子初始化状态变量 `counter`。

1.  `getCounter` 函数使用 `axios` 库向 Flask 后端的 `/api/v1/counters` 端点发送 `GET` 请求。然后，使用来自后端的响应（其中包含更新的计数器值）来更新计数器状态变量。

1.  组件返回一个 div，显示计数器的值，以及一个按钮，当点击时，会触发 `getCounter` 函数从后端检索更新的计数器值。

接下来，我们将讨论如何在 Flask-React Web 应用程序中创建密码保护的仪表板。

# 创建密码保护的仪表板

在 Web 应用程序中保护页面对于维护安全和隐私至关重要。通过扩展，这有助于防止未经授权访问敏感信息。在本节中，你将在 Flask-React Web 应用程序中实现一个受保护的仪表板页面。

仪表板是一个用户友好的界面，提供了数据和信息的概览。仪表板上显示的数据可以来自各种来源，例如数据库、电子表格和 API。

## Flask 后端

以下代码演示了一个实现，允许管理员用户登录并查看受保护的仪表板页面。我们将实现最小化的登录和注销端点，定义登录和注销功能并保护 `dashboard` 端点。应用程序使用 `Flask-Session` 库在文件系统中存储会话数据：

```py
from flask import Flask, request, jsonify, sessionfrom flask_session import Session
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
@app.route("/api/v1/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    if username == "admin" and password == "secret":
        session["logged_in"] = True
        return jsonify({"message": "Login successful"})
    else:
        return jsonify({"message": "Login failed"}), 401
@app.route("/api/v1/logout")
def logout():
    session.pop("logged_in", None)
    return jsonify({"message": "Logout successful"})
@app.route("/api/v1/dashboard")
def dashboard():
    if "logged_in" not in session:
        return jsonify({"message": "Unauthorized access"}),
            401
    else:
        return jsonify({"message": "Welcome to the
            dashboard"})
```

在`login`端点，应用程序接收一个包含请求体中 JSON 格式的`username`和`password`参数的`POST`请求。代码检查`username`和`password`参数是否与预定义的值匹配——即`admin`和`secret`。如果值匹配，代码将会话数据中的`logged_in`键设置为`True`，表示用户已登录。

它返回一个包含声明`Login successful`的消息的 JSON 响应。如果值不匹配，代码返回一个包含声明`Login failed`和`401` HTTP 状态代码的 JSON 响应，表示未授权访问。

`logout`端点从会话数据中删除`logged_in`键，表示用户已注销。它返回一个包含声明`Logout successful`的消息的 JSON 响应。

仪表板端点检查会话数据中是否存在`logged_in`键。如果不存在，代码返回一个包含声明`Unauthorized access`和`401` HTTP 状态代码的 JSON 响应。如果`logged_in`键存在，代码返回一个包含声明`"Welcome to` `the dashboard"`的 JSON 响应。

## React 前端

以下代码片段是一个 React 组件，用于显示用户的仪表板。它使用 React 钩子，特别是`useState`和`useEffect`，来管理其状态和更新用户界面：

```py
import React, { useState, useEffect } from "react";import axios from "axios";
function Dashboard() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [message, setMessage] = useState("");
  const checkLogin = async () => {
    const response = await axios.get(
      "http://localhost:5000/api/v1/dashboard");
    if (response.status === 200) {
      setIsLoggedIn(true);
      setMessage(response.data.message);
    }
  };
  useEffect(() => {
  checkLogin();
  }, []);
  if (!isLoggedIn) {
    return <h1>Unauthorized access</h1>;
  }
  return <h1>{message}</h1>;
}
export default Dashboard;
```

当组件渲染时，它使用`axios`库向`http://localhost:5000/api/v1/dashboard`发出 HTTP `GET`请求。这是在`checkLogin`函数中完成的，该函数在组件挂载时由`useEffect`钩子调用。

如果服务器的响应是`200 OK`，这意味着用户有权访问仪表板。组件的状态通过将`isLoggedIn`设置为`true`和`message`设置为从服务器返回的消息来更新，以反映这一点。如果响应不是`200 OK`，这意味着用户未授权，`isLoggedIn`保持为`false`。

最后，组件返回一个消息，告诉用户他们是否有权访问仪表板。如果`isLoggedIn`为`false`，它返回`Unauthorized access`。如果`isLoggedIn`为`true`，它返回来自服务器的消息。

以这种方式，您可以使用 React 和 Flask 创建一个密码保护的仪表板，只有经过身份验证的用户才能访问，从而为您的应用程序增加安全性。

接下来，您将学习如何在 Flask 和 React Web 应用程序中实现 Flash 消息。

# 在 Flask 中实现 Flash 消息

Flash 消息增强了任何 Web 应用程序的用户体验，为用户提供及时和有用的反馈。Flash 用于在重定向后显示网页上的状态或错误消息。例如，在表单提交成功后，可以将消息存储在 Flash 中，以便在重定向页面上显示成功消息。

闪存信息存储在用户的会话中，这是一个类似于字典的对象，可以在请求之间存储信息。使用闪存信息，您可以在请求之间安全高效地传递信息。这对于显示不需要长时间持续或只需要显示一次的消息很有用，例如成功或错误消息。由于闪存信息存储在用户的会话中，它们只能由服务器访问，并且不会以纯文本形式发送到客户端，这使得它们更加安全。

让我们修改登录和注销端点以显示闪存信息。

## Flask 后端

以下代码演示了带有登录和注销端点的闪存信息系统的实现。代码首先导入必要的模块并创建一个 Flask 应用程序。`app.secret_key = "secret_key"`这一行设置了密钥，该密钥用于加密存储在会话中的闪存信息：

```py
from flask import Flask, request, jsonify, session, flashfrom flask_session import Session
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = "secret_key"
Session(app)
@app.route("/api/v1/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    if username == "admin" and password == "secret":
        session["logged_in"] = True
        flash("Login successful")
        return jsonify({"message": "Login successful"})
    else:
        flash("Login failed")
        return jsonify({"message": "Login failed"}), 401
@app.route("/api/v1/logout")
def logout():
    session.pop("logged_in", None)
    flash("Logout successful")
    return jsonify({"message": "Logout successful"})
```

登录端点由`login`函数定义，该函数绑定到`/api/v1/login` URL。该函数从请求中的 JSON 数据中检索`username`和`password`值，并检查它们是否与预定义的`"admin"`和`"secret"`值匹配。如果值匹配，用户的会话通过在会话中设置`logged_in`键标记为已登录，并设置一个表示登录成功的闪存信息。

函数随后返回一个 JSON 响应，指示登录成功。如果值不匹配，设置一个闪存信息，指示登录失败，并返回一个表示登录失败的 JSON 响应。注销端点由`logout`函数定义，该函数绑定到`/api/v1/logout` URL。

函数从会话中删除`logged_in`键，表示用户不再登录，并设置一个表示注销成功的闪存信息。随后返回一个表示注销成功的 JSON 响应。

## React 前端

以下代码片段演示了一个表示从后端处理闪存信息的 Web 应用程序仪表板的 React 函数组件。`Dashboard`组件使用了`useState`和`useEffect`钩子：

```py
import React, { useState, useEffect } from "react";import axios from "axios";
function Dashboard() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [message, setMessage] = useState("");
    const [flashMessage, setFlashMessage] = useState("");
    const checkLogin = async () => {
        const response = await axios.get(
            "http://localhost:5000/api/v1/dashboard");
        if (response.status === 200) {
            setIsLoggedIn(true);
            setMessage(response.data.message);
        }
    };
                    .....
        if (!isLoggedIn) {
            return (
                <div>
                    <h1>Unauthorized access</h1>
                    <h2>{flashMessage}</h2>
                    <button onClick={() =>
                        handleLogin("admin", "secret")}>
                        Login</button>
```

`Dashboard`组件跟踪以下状态变量：

+   `isLoggedIn`: 一个表示用户是否登录的布尔值。它最初设置为`false`。

+   `message`: 一个表示在仪表板上显示的消息的字符串值。

+   `flashMessage`: 一个表示在页面上显示的闪存信息的字符串值。

`Dashboard`组件有三个功能：

+   `checkLogin`: 一个异步函数，它向`/api/v1/dashboard`端点发送`GET`请求以检查用户是否已登录。如果响应状态是`200`，它将`isLoggedIn`状态变量更新为`true`，并显示`response.data.message`的值。

+   `handleLogin`: 一个异步函数，它使用提供的`username`和`password`值作为请求体，向`/api/v1/login`端点发送`POST`请求。如果响应状态是`200`，它将`isLoggedIn`状态变量更新为`true`并将`flashMessage`更新为`response.data.message`的值。如果响应状态不是`200`，它将`flashMessage`更新为`response.data.message`的值。

+   `handleLogout`: 一个异步函数，它向`/api/v1/logout`端点发送`GET`请求。如果响应状态是`200`，它将`isLoggedIn`状态变量更新为`false`并将`flashMessage`更新为`response.data.message`的值。

使用`useEffect`钩子在组件挂载时调用`checkLogin`函数。

最后，组件根据`isLoggedIn:`的值返回一个 UI。如果用户未登录，它将显示一条消息说“未经授权访问”和“``登录成功"”。

以这种方式，你可以在 React 应用程序的前端使用闪存消息向用户提供反馈，然后使用 Flask 后端来增强用户体验。总的来说，闪存消息使 Web 应用程序更加互动和用户友好。

# 摘要

本章提供了关于信息安全基础以及如何使用身份验证和授权来确保 Flask Web 应用程序的全面概述。你了解了最佳实践，并提供了在 Flask 应用程序中实现身份验证和授权的用例。我们还讨论了不同的身份验证方法和访问控制方法。

你探索了如何管理用户会话和实现密码保护的仪表板。此外，本章还展示了如何使用闪存消息向 Web 应用程序的用户提供反馈。你应已对如何确保 Flask 应用程序的安全以及如何在项目中实现身份验证和授权有了坚实的理解。

在下一章中，我们将讨论如何处理 Flask Web 应用程序中的错误，其中 React 处理前端部分。我们将深入研究内置的 Flask 调试功能，并学习如何在 React-Flask 应用程序中处理自定义错误消息。
