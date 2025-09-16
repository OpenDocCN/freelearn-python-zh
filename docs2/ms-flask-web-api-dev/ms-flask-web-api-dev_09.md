# <st c="0">9</st>

# <st c="2">保护 Flask 应用程序</st>

<st c="29">像任何 Web 应用程序一样，Flask 应用程序都有需要保护免受外部攻击的漏洞，这些攻击利用了这些软件缺陷。</st> <st c="181">这些网络攻击主要是由于访问控制问题不完善，</st> **<st c="251">跨站脚本</st>** <st c="271">(</st>**<st c="273">XSS</st>**<st c="276">)，</st> **<st c="280">跨站请求伪造</st>** <st c="306">(</st>**<st c="308">CSRF</st>**<st c="312">)，</st> **<st c="316">服务器端请求伪造</st>** <st c="344">(</st>**<st c="345">SSRF</st>**<st c="349">)，</st> **<st c="353">SQL 注入</st>**<st c="366">，以及</st> **<st c="372">拒绝服务</st>** <st c="389">(</st>**<st c="391">DoS</st>**<st c="394">)，以及过时的模块</st> <st c="426">和库。</st>

<st c="440">实施安全措施必须是任何 Flask 应用程序的最高优先级，尤其是当它在构建模型、仓库层和工作流相关事务时更加依赖外部模块时。</st> <st c="660">使用第三方库可能会给 Flask 应用程序带来风险，因为一些库代码可能包含编码错误或漏洞。</st> <st c="805">这对于来自过时第三方模块和库的代码来说尤其如此，这些模块和库的来源不可靠。</st>

<st c="919">使用外部模块，如使用</st> **<st c="1072">Authlib</st>** <st c="1079">模块而不是从头开始构建，来构建 Flask 组件和功能更容易。</st> <st c="1131">为了减少，如果不是避免，网络攻击的机会，应该制定一个仅使用可靠和更新模块的安全计划。</st> <st c="1271">这将保护应用程序免受</st> <st c="1310">外部攻击者。</st>

<st c="1328">本章的主要目标是提供可能的解决方案，以帮助 Flask 应用程序避免使用 Flask 内置组件进行的一些知名网络攻击，以及一些最新且可靠的</st> <st c="1543">第三方库。</st>

<st c="1565">以下是我们将在帮助保护我们的</st> <st c="1645">Flask 应用程序的上下文中涵盖的主题：</st>

+   <st c="1664">添加对</st> <st c="1688">网络漏洞</st>的保护

+   <st c="1707">保护</st> <st c="1717">响应数据</st>

+   <st c="1730">管理</st> <st c="1740">用户凭据</st>

+   <st c="1756">实现网络</st> <st c="1774">表单认证</st>

+   <st c="1793">防止</st> <st c="1805">CSRF 攻击</st>

+   <st c="1817">实现用户身份验证</st> <st c="1851">和授权</st>

+   <st c="1868">控制视图或</st> <st c="1893">API 访问</st>

# <st c="1903">技术要求</st>

<st c="1926">本章介绍的是</st> `<st c="2705">Flask[async]</st>` <st c="2717">功能，包括异步</st> `<st c="2751">Flask-SQLAlchemy</st>` <st c="2767">事务。</st> <st c="2782">它们可在</st> [<st c="2804">https://github.com/PacktPublishing/Mastering-Flask-Web-Development/tree/main/ch09</st>](https://github.com/PacktPublishing/Mastering-Flask-Web-Development/tree/main/ch09)<st c="2885">找到。</st>

# <st c="2886">添加对网络漏洞的保护</st>

<st c="2929">SQL 注入、SSRF 和 XSS 攻击是最常见的网络漏洞，它们破坏了许多网络应用程序。</st> <st c="3043">它们还影响任何使用基于 HTTP 的事务的应用程序，例如</st> `<st c="3119">POST</st>`<st c="3123">,</st> `<st c="3125">PUT</st>`<st c="3128">,</st> `<st c="3130">PATCH</st>`<st c="3135">, 和</st> `<st c="3141">DELETE</st>`<st c="3147">。</st> SQL 注入发生在攻击者渗透管理受信任应用程序内容的后端数据存储时。</st> <st c="3274">嵌入</st> <st c="3283">恶意 SQL 代码可以篡改数据，生成不想要的页面或破坏数据库。</st> <st c="3381">XSS 攻击通常会将恶意脚本插入到应用程序的页面中，以从系统中窃取 cookie、会话数据和敏感凭证。</st> <st c="3536">另一方面，CSRF 发生在认证环境中。</st> <st c="3604">它发生在有效用户执行 HTTP 事务时，浏览器中的恶意脚本篡改了有效的凭据，用虚假和无效的凭据来引导事务到</st> <st c="3785">不受信任的系统。</st>

## <st c="3803">将表单验证应用于请求数据</st>

<st c="3844">避免这些攻击的一个解决方案是设计一个表单验证，它不会占用几行代码或给视图或 API 函数增加更多性能开销</st> <st c="3976">开销。</st> <st c="4016">The</st> `<st c="4066">FlaskForm</st>` <st c="4075">类通过将属性映射到适当的字段类来子类化表单模型。</st> <st c="4168">每个字段类（例如，</st> `<st c="4192">StringField</st>`<st c="4203">,</st> `<st c="4205">BooleanField</st>`<st c="4217">,</st> `<st c="4219">DateField</st>`<st c="4228">, 或</st> `<st c="4233">TimeField</st>`<st c="4242">）都有属性和内置验证器（例如，</st> `<st c="4291">Length()</st>`<st c="4299">,</st> `<st c="4301">Email()</st>`<st c="4308">, 或</st> `<st c="4313">DataRequired()</st>`<st c="4327">），并支持自定义验证器</st> <st c="4364">当验证过程需要复杂条件时。</st> <st c="4425">有了这些验证器，它可以在一定程度上保护应用程序免受利用。</st> <st c="4514">关于使用 Flask-WTF 的进一步讨论包括在</st> *<st c="4567">第四章</st>*<st c="4576">。</st>

<st c="4577">如果应用程序不是基于 Web 或不需要表单模型类，</st> `<st c="5199">pip</st>` <st c="5202">命令：</st>

```py
 pip install flask-gladiator
```

<st c="5239">安装后，该模块无需进一步设置。</st> <st c="5304">在构建验证规则时可以立即使用，例如以下验证规则将仔细审查</st> <st c="5434">管理员资料：</st>

```py
<st c="5458">import gladiator as glv</st>
<st c="5482">from gladiator.core import ValidationResult</st> def validate_form(form_data):
    field_validations = (
        ('adminid', <st c="5591">glv.required</st>, <st c="5605">glv.length_max(12)</st>),
        ('username', glv.required, <st c="5654">glv.type_(str)</st>),
        ('firstname', glv.required, glv.length_max(50), <st c="5720">glv.regex_('[a-zA-Z][a-zA-Z ]+')</st>),
        ('midname', glv.required, glv.length_max(50), glv.regex_('[a-zA-Z][a-zA-Z ]+')),
        ('lastname', glv.required, glv.length_max(50), glv.regex_('[a-zA-Z][a-zA-Z ]+')),
        ('email', glv.required, glv.length_max(25), <st c="5963">glv.format_email</st>),
        ('mobile', glv.required, glv.length_max(15)),
        ('position', glv.required,  glv.length_max(100)),
        ('status', glv.required, <st c="6103">glv.in_(['true', 'false'])</st>),
        ('gender', glv.required, glv.in_(['male', 'female'])),
    ) <st c="6190">result:ValidationResult = glv.validate(field_validations, form_data)</st> return <st c="6317">gladiator</st> module is the <st c="6341">validate()</st> method, which has two required parameters: <st c="6395">form_data</st> and <st c="6409">validators</st>. The validators are placed in a tuple of tuples, as shown in the preceding code, wherein each tuple contains the request parameter name followed by all its validators. Our <st c="6592">ch09-web-passphrase</st> project uses the following validators:

				*   `<st c="6650">required()</st>`<st c="6661">: Requires the parameter to have</st> <st c="6695">a value.</st>
				*   `<st c="6703">length_max()</st>`<st c="6716">: Checks whether the given string length is lower than or equal to a</st> <st c="6786">maximum value.</st>
				*   `<st c="6800">type_()</st>`<st c="6808">: Checks the type of the request data (e.g., a form parameter is always</st> <st c="6881">a string).</st>
				*   `<st c="6891">regex_ ()</st>`<st c="6901">: Matches the string to a</st> <st c="6928">regular expression.</st>
				*   `<st c="6947">format_email()</st>`<st c="6962">: Checks whether the request data follows the</st> <st c="7009">email regex.</st>
				*   `<st c="7021">in_()</st>`<st c="7027">: Checks whether the value is within the list</st> <st c="7074">of options.</st>

			<st c="7085">The list shows only a few of the many validator functions that the</st> `<st c="7153">gladiator</st>` <st c="7162">module can provide to establish the validation</st> <st c="7209">rules.</st> <st c="7217">Now, the</st> `<st c="7226">validate()</st>` <st c="7236">method returns a</st> `<st c="7254">ValidationResult</st>` <st c="7270">object, which has a boolean</st> `<st c="7299">success</st>` <st c="7306">variable that yields</st> `<st c="7328">True</st>` <st c="7332">if all the validators have no</st> <st c="7362">hits.</st> <st c="7369">Otherwise, it yields</st> `<st c="7390">False</st>`<st c="7395">. The following code shows how the</st> `<st c="7430">ch09-web-passphrase</st>`<st c="7449">’s</st> `<st c="7453">add_admin_profile()</st>` <st c="7472">method utilizes the given</st> `<st c="7499">validate_form()</st>` <st c="7514">view function:</st>

```

@current_app.route('/admin/profile/add', methods=['GET', 'POST'])

async def add_admin_profile():

    if not session.get("user"):

        return redirect('/login/auth')

    … … … … … …

    if request.method == 'GET':

        return render_template('admin/add_admin_profile.html', admin=admin_rec), 200

    else:

        result = validate_form(request.form)

        if result == False:

            flash(f'验证问题。', 'error')

            return render_template('admin/add_admin_profile.html', admin=admin_rec), 200

        … … … … … …

        return render_template('admin/add_admin_profile.html', admin=admin_rec), 200

```py

			<st c="8070">Now, filtering malicious text can be effective if we combine the validation and sanitation of this form data.</st> <st c="8181">Sanitizing inputs</st> <st c="8199">means encoding special</st> <st c="8221">characters that might trigger the execution of malicious scripts from</st> <st c="8292">the browser.</st>
			<st c="8304">Sanitizing form inputs</st>
			<st c="8327">Aside from validation, view or API functions must also sanitize incoming request data by converting special characters and suspicious</st> <st c="8462">symbols to purely text so that XML- and HTML-based templates can render them without side effects.</st> <st c="8561">This process is</st> <st c="8576">known as</st> `<st c="8600">markupsafe</st>` <st c="8610">module has an</st> `<st c="8625">escape()</st>` <st c="8633">method that can normalize request data with query strings that intend to control the JavaScript codes, modify the UI experience, or tamper browser cookies when Jinja2 templates render them.</st> <st c="8824">The following snippet is a portion of the</st> `<st c="8866">add_admin_profile()</st>` <st c="8885">view function that sanitizes the form data after</st> `<st c="8935">gladiator</st>` <st c="8944">validation:</st>

```

@current_app.route('/admin/profile/add', methods=['GET', 'POST'])

async def add_admin_profile():

    … … … … … …

    result = validate_form(request.form)

    if result == False:

        flash(f'验证问题。', 'error')

        return render_template('admin/add_admin_profile.html', admin=admin_rec), 200

    username = request.form['username']

    … … … … … …

    admin_details = {

        "adminid": escape(request.form['adminid'].strip()),

        "username": escape(request.form['username'].strip()),

        "firstname": escape(request.form['firstname'].strip()),

        … … … … … …

        "gender": escape(request.form['gender'].strip())

    }

    admin = Administrator(**admin_details)

    result = await repo.insert_admin(admin)

    if result == False:

        flash(f'添加 … 资料时出错。', 'error')

    else:

        flash(f'成功添加用户 … )

    return render_template('admin/add_admin_profile.html', admin=admin_rec), 200

```py

			<st c="9792">Removing leading and trailing whitespaces or defined suspicious characters using Python’s</st> `<st c="9883">strip()</st>` <st c="9890">method with the escaping process may lower the risk of injection and XSS attacks.</st> <st c="9973">However, be sure that the validation rules and sanitation techniques combined will neither ruin the performance of the view or API function nor change the actual request data.</st> <st c="10149">Also, tight</st> <st c="10161">validation rules can affect the overall runtime performance, so choose the appropriate number and types of validators for</st> <st c="10283">every form.</st>
			<st c="10294">To avoid SQL injection, use</st> <st c="10323">an ORM such as</st> **<st c="10338">SQLAlchemy</st>**<st c="10348">,</st> **<st c="10350">Pony</st>**<st c="10354">, or</st> **<st c="10359">Peewee</st>** <st c="10365">that can provide a</st> <st c="10384">more abstract form of SQL transactions and even escape utilities to sanitize column</st> <st c="10469">values before persistence.</st> <st c="10496">Avoid using native and dynamic queries where the field values are concatenated to the query string because they are prone to manipulation</st> <st c="10634">and exploitation.</st>
			<st c="10651">Sanitation can also be applied to response data to</st> <st c="10703">avoid another type of attack called the</st> **<st c="10743">Server-Side Template Injection</st>** <st c="10773">(</st>**<st c="10775">SSTI</st>**<st c="10779">).</st> <st c="10783">Let us now discuss how to protect the application from SSTIs by managing the</st> <st c="10859">response data.</st>
			<st c="10874">Securing response data</st>
			<st c="10897">Jinja2 has a built-in escaping mechanism to avoid SSTIs.</st> <st c="10955">SSTIs allow attackers to inject malicious template scripts or fragments that can run in the background.</st> <st c="11059">These then ruin the response or perform unwanted</st> <st c="11107">executions that can ruin server-side operations.</st> <st c="11157">Thus, applying the</st> `<st c="11176">safe</st>` <st c="11180">filter in Jinja templates to perform dynamic content augmentation is not a good practice.</st> <st c="11271">The</st> `<st c="11275">safe</st>` <st c="11279">filter turns off the Jinja2’s escaping mechanism and allows for running these malicious attacks.</st> <st c="11377">In connection with this, avoid</st> <st c="11408">using</st> `<st c="11448"><a></st>` <st c="11451">tag in templates (e.g.,</st> `<st c="11476"><a href="{{ var_link }}">Click Me</a></st>`<st c="11513">).</st> <st c="11517">Instead, utilize the</st> `<st c="11538">url_for()</st>` <st c="11547">utility method to call dynamic view functions because it validates and checks whether the Jinja variable in the expression is a valid view name.</st> *<st c="11693">Chapter 1</st>* <st c="11702">discusses how to apply</st> `<st c="11726">url_for()</st>` <st c="11735">for hyperlinks.</st>
			<st c="11751">On the other hand, there are also issues in Flask that need handling to prevent injection attacks on the Jinja templates, such as managing how the view functions will render the context data and add security</st> <st c="11960">response headers.</st>
			<st c="11977">Rendering Jinja2 variables</st>
			<st c="12004">There is no ultimate solution to avoid injection but to</st> <st c="12060">apply escaping to context data before rendering them to Jinja2 templates.</st> <st c="12135">Moreover, avoid using</st> `<st c="12157">render_template_string()</st>` <st c="12181">even if this is part of the Flask framework.</st> <st c="12227">Rendering HTML page-generated content may accidentally run malicious data from inputs overlooked by</st> <st c="12326">filtering and escaping.</st> <st c="12351">It is always good practice to place all HTML content in a file with an</st>`<st c="12421">.html</st>` <st c="12426">extension, or XML content in a</st> `<st c="12458">.xml</st>` <st c="12462">file, to enable Jinja2’s default escaping feature.</st> <st c="12514">Then, render them using the</st> `<st c="12542">render_template()</st>` <st c="12559">method with or without the escaped and validated context data.</st> <st c="12623">All our projects use</st> `<st c="12644">render_template()</st>` <st c="12661">in rendering</st> <st c="12675">Jinja2 templates.</st>
			<st c="12692">Security response headers must also be part of the response object when rendering every view template.</st> <st c="12796">Let us explore these security response headers and learn where to</st> <st c="12862">build them.</st>
			<st c="12873">Adding security response headers</st>
			<st c="12906">HTTP security response headers are directives used by many web applications to mitigate vulnerability attacks, such as XXS and public exposure of user details.</st> <st c="13067">They are headers added in the response object</st> <st c="13113">during the rendition</st> <st c="13133">of the Jinja2 templates or JSON results.</st> <st c="13175">Some of these headers include</st> <st c="13205">the following:</st>

				*   `<st c="13371">UTF-8</st>` <st c="13376">charset to</st> <st c="13388">avoid XSS.</st>
				*   `<st c="13471">content-type</st>`<st c="13483">. It also blocks the browser’s</st> `<st c="13514">media-type</st>` <st c="13524">sniffing, so its value should</st> <st c="13555">be</st> `<st c="13558">nosniff</st>`<st c="13565">.</st>
				*   `<st c="13653"><frame></st>`<st c="13660">,</st> `<st c="13662"><iframe></st>`<st c="13670">,</st> `<st c="13672"><embed></st>`<st c="13679">, or</st> `<st c="13684"><objects></st>`<st c="13693">. Possible values include</st> `<st c="13719">DENY</st>` <st c="13723">and</st> `<st c="13728">SAMEORIGIN</st>`<st c="13738">. The</st> `<st c="13744">DENY</st>` <st c="13748">option disallows rending pages on a frame, while</st> `<st c="13798">SAMEORIGIN</st>` <st c="13808">allows rendering a page on a frame with the same URL site as</st> <st c="13870">the page.</st>
				*   **<st c="13879">Strict-Transport-Security</st>**<st c="13905">: This indicates that the browser can only access the page through the</st> <st c="13977">HTTPS protocol.</st>

			<st c="13992">In our</st> `<st c="14000">ch09-web-passphrase</st>` <st c="14019">project, the global</st> `<st c="14040">@after_request</st>` <st c="14054">function creates a list of security response headers for every view function call.</st> <st c="14138">The following code snippet in the</st> `<st c="14172">main.py</st>` <st c="14179">module shows this</st> <st c="14198">function implementation:</st>

```

<st c="14222">@app.after_request</st> def create_sec_resp_headers(response):

    response.headers['Content-Type'] = 'text/html; charset=UTF-8'

    response.headers['X-Content-Type-Options'] = 'nosniff'

    response.headers['X-Frame-Options'] = 'SAMEORIGIN'

    response.headers['Strict-Transport-Security'] = 'Strict-Transport-Security: max-age=63072000; includeSubDomains; preload'

    return response

```py

			<st c="14586">Here,</st> `<st c="14593">Content-Type</st>`<st c="14605">,</st> `<st c="14607">X-Content-Type-Options</st>`<st c="14629">,</st> `<st c="14631">X-Frame-Options</st>`<st c="14646">, and</st> `<st c="14652">Strict-Transport-Security</st>` <st c="14677">are the most essential response headers for web applications.</st> <st c="14740">By the way,</st> `<st c="14752">SAMEORIGIN</st>` <st c="14762">is the ideal value for</st> `<st c="14786">X-Frame-Options</st>` <st c="14801">because it prevents view pages from</st> <st c="14838">displaying outside the site domain of the</st> <st c="14880">project, mitigating</st> `<st c="15007">/</st>``<st c="15008">admin/profile/add</st>` <st c="15025">view.</st>
			![Figure 9.1 – The response headers when running the view function](img/B19383_09_001.jpg)

			<st c="15768">Figure 9.1 – The response headers when running the view function</st>
			<st c="15832">On the other hand, another</st> <st c="15859">way to manage security response headers is</st> <st c="15903">through the Flask module</st> `<st c="15990">flask-talisman</st>` <st c="16004">module using the following</st> `<st c="16032">pip</st>` <st c="16035">command:</st>

```

pip install flask-talisman

```py

			<st c="16071">Afterward, instantiate the</st> `<st c="16099">Talisman</st>` <st c="16107">class in the</st> `<st c="16121">create_app()</st>` <st c="16133">method and integrate the module into the Flask application by adding and configuring the web application’s security response headers using Talisman libraries, as shown in the</st> <st c="16309">following snippet:</st>

```

<st c="16327">from flask_talisman import Talisman</st> def create_app(config_file):

    app = Flask(__name__,template_folder= '../modules/pages', static_folder= '../modules/resources')

    app.config.from_file(config_file, toml.load)

    … … … … … … <st c="16547">talisman = Talisman(app)</st> csp = {

        'default-src': [

            '\'self\'',

            'https://code.jquery.com',

            'https://cdnjs.com',

            'https://cdn.jsdelivr.net',

        ]

    }

    hsts = {

        'max-age': 31536000,

        'includeSubDomains': True

    } <st c="16747">talisman.force_https = True</st> talisman.force_file_save = True <st c="16807">talisman.x_xss_protection = True</st><st c="16839">talisman.session_cookie_secure = True</st> talisman.frame_options_allow_from = 'https://www.google.com' <st c="17226">default-src</st>, <st c="17239">image-src</st>, <st c="17250">style-src</st>, <st c="17261">media-src</st>, <st c="17272">object-src</st> ). 在我们的配置中，JS 文件必须仅来自<st c="17337">https://code.jquery.com</st>、<st c="17362">https://cdnjs.com</st>、<st c="17381">https://cdn.jsdelivr.net</st>和本地主机，而 CSS 和图像必须如<st c="17503">default-src</st>中所示，从本地主机获取，作为每个视图页面的后备资源。通过指定具有特定 JS 资源的<st c="17570">script-src</st>、具有 CSS 资源的<st c="17607">style-src</st>和具有目标图像的<st c="17641">image-src</st>，将绕过<st c="17692">default-src</st>设置。

            除了 CSP 之外，Talisman 还可以添加<st c="17750">X-XSS-Protection</st>、<st c="17768">Referrer-Policy</st>、<st c="17783">和<st c="17789">Set-Cookie</st>，以及之前由<st c="17867">@after_request</st> <st c="17881">函数</st>包含在响应中的头部。在结合这两种方法时需要谨慎，因为头部设置的<st c="17892">重叠</st>可能会发生。

            <st c="17992">在响应中添加</st> `<st c="18004">Strict-Transport-Security</st>` <st c="18029">头部并设置 Talisman 属性的</st> `<st c="18069">force_https</st>` <st c="18080">为</st> `<st c="18107">True</st>` <st c="18111">需要以 HTTPS 模式运行应用程序。</st> <st c="18160">让我们探索为</st> <st c="18224">Flask 应用程序</st>启用 HTTPS 的最新和最简单的方法。

            <st c="18242">使用 HTTPS 运行请求/响应事务</st>

            <st c="18291">HTTPS 是一种 TLS 加密的 HTTP 协议。</st> <st c="18332">它建立了数据发送者和接收者之间的安全通信，保护了在交换过程中流动的 cookies、URLs 和敏感信息</st> <st c="18425">。它还保护了数据的完整性和用户的真实性，因为它需要用户的私钥来允许访问。</st> <st c="18503">因此，要启用 HTTPS 协议，WSGI 服务器必须使用由 SSL 密钥生成器生成的公钥和私钥证书运行。</st> <st c="18630">按照惯例，证书必须保存在项目目录内或主机服务器上的安全位置。</st> <st c="18773">本章</st> <st c="18897">使用</st> **<st c="18911">OpenSSL</st>** <st c="18918">工具生成</st> <st c="18935">证书。</st>

            <st c="18951">使用以下</st> `<st c="18971">pyopenssl</st>` <st c="18980">命令安装最新版本：</st> `<st c="19001">pip</st>` <st c="19004">命令：</st>

```py
 pip install pyopenssl
```

            <st c="19035">现在，要运行应用程序，请将</st> <st c="19076">私钥和公钥</st> <st c="19104">包含在</st> `<st c="19109">run()</st>` <st c="19109">中</st> 通过其 <st c="19122">ssl_context</st> <st c="19133">参数。</st> <st c="19145">以下</st> `<st c="19159">main.py</st>` <st c="19166">代码片段展示了如何使用 HTTPS</st> <st c="19220">在开发服务器上</st> <st c="19225">运行应用程序：</st>

```py
 app, celery_app, auth = create_app('../config_dev.toml')
… … … … … …
if __name__ == '__main__': <st c="19398">python main.py</st> command with the <st c="19430">ssl_context</st> parameter will show a log on the terminal console, as shown in *<st c="19505">Figure 9</st>**<st c="19513">.2</st>*:
			![Figure 9.2 – The server log when running on HTTPS](img/B19383_09_002.jpg)

			<st c="19742">Figure 9.2 – The server log when running on HTTPS</st>
			<st c="19791">When opening the</st> `<st c="19809">https://127.0.0.1:5000/</st>` <st c="19832">link on a browser, a warning page will pop up on the screen, such as the one depicted in</st> *<st c="19922">Figure 9</st>**<st c="19930">.3</st>*<st c="19932">, indicating that we are entering a secured page from a</st> <st c="19988">non-secured browser.</st>
			![Figure 9.3 – A warning page on opening secured links](img/B19383_09_003.jpg)

			<st c="20319">Figure 9.3 – A warning page on opening secured links</st>
			<st c="20371">Another way to run Flask applications</st> <st c="20409">on an HTTP protocol is to include the key files in the command line, such as</st> `<st c="20487">python main.py --cert=cert.pem --key=key.pem</st>`<st c="20531">. In the production environment, we run Flask applications according to the procedure followed by the</st> <st c="20633">secured</st> <st c="20641">production server.</st>
			<st c="20659">Encryption does not apply only when establishing an HTTP connection but also when securing sensitive user information such as usernames and passwords.</st> <st c="20811">In the next section, we will discuss the</st> <st c="20851">different ways of</st> **<st c="20870">hashing</st>** <st c="20877">and encrypting</st> <st c="20893">user credentials.</st>
			<st c="20910">Managing user credentials</st>
			<st c="20936">The most common procedure for protecting any application from attacks is to control access to the user’s sensitive details, such as their username and password.</st> <st c="21098">Direct use of saved raw user credentials for login</st> <st c="21149">validation will not protect the application from attacks unless the application derives passphrases from the passwords, saves them into the database, and applies them for user</st> <st c="21325">validation instead.</st>
			<st c="21344">This topic will</st> <st c="21360">cover password</st> <st c="21375">hashing using</st> `<st c="21478">sqlalchemy_utils</st>` <st c="21494">module for the seamless and automatic encryption of</st> <st c="21547">sensitive data.</st>
			<st c="21562">Encrypting user passwords</st>
			<st c="21588">Generating a passphrase from the username and password of the user is the typical and easiest way to protect the</st> <st c="21702">application from attackers who want to crack down or hack a user account.</st> <st c="21776">In Flask, there are two ways to generate a passphrase from</st> <st c="21835">user credentials:</st>

				*   **<st c="21852">The hashing process</st>**<st c="21872">: A one-way</st> <st c="21885">approach that involves generating a fixed-length passphrase of the</st> <st c="21952">original text.</st>
				*   **<st c="21966">The encryption process</st>**<st c="21989">: A two-way</st> <st c="22002">approach that involves generating a variable-length text using random symbols that can be traced back to its</st> <st c="22111">original text.</st>

			<st c="22125">The</st> `<st c="22130">ch09-api-bcrypt</st>` <st c="22145">and</st> `<st c="22150">ch09-auth-basic</st>` <st c="22165">projects use hashing to manage the passwords of a user.</st> <st c="22222">The</st> `<st c="22226">ch09-auth-basic</st>` <st c="22241">project utilizes Hashlib as its primary hashing library for passphrase generation.</st> <st c="22325">Flask has the</st> `<st c="22339">werkzeug.security</st>` <st c="22356">module that provides</st> `<st c="22378">generate_password_hash()</st>`<st c="22402">, a function that uses Hashlib’s</st> `<st c="22435">scrypt</st>` <st c="22441">algorithm to generate a passphrase from a text.</st> <st c="22490">The project’s</st> `<st c="22504">add_signup()</st>` <st c="22516">API endpoint function that utilizes the</st> `<st c="22557">werkzeug.security</st>` <st c="22574">module in generating the passphrase from the user’s password is</st> <st c="22639">as follows:</st>

```

<st c="22650">从 werkzeug.security 模块导入 generate_password_hash</st> @current_app.post('/login/signup')

async def add_signup():

    login_json = request.get_json() <st c="22795">密码 = login_json["password"]</st><st c="22828">passphrase = generate_password_hash(password)</st> async with db_session() as sess:

        async with sess.begin():

        repo = LoginRepository(sess)

        login = Login(username=login_json["username"], <st c="23009">密码=passphrase</st>, role=login_json["role"])

        result = await repo.insert_login(login)

        if result == False:

            return jsonify(message="插入错误"), 201

        return jsonify(record=login_json), 200

```py

			<st c="23200">The</st> `<st c="23205">generate_password_hash()</st>` <st c="23229">method has</st> <st c="23241">three parameters:</st>

				*   <st c="23258">The actual password is</st> <st c="23281">the</st> <st c="23286">first parameter.</st>
				*   <st c="23302">The hashing method is the second parameter with a default value</st> <st c="23367">of</st> `<st c="23370">scrypt</st>`<st c="23376">.</st>
				*   <st c="23377">The</st> **<st c="23382">salt</st>** <st c="23386">length is the</st> <st c="23401">third parameter.</st>

			<st c="23417">The salt length will determine the number of alphanumerics that the method will use to generate a salt.</st> <st c="23522">A salt is the additional alphanumerics with a fixed length that are added to the end of the password to make the passphrase more unbreachable or uncrackable.</st> <st c="23680">The process of adding salt to hashing</st> <st c="23717">is</st> <st c="23721">called</st> **<st c="23728">salting</st>**<st c="23735">.</st>
			<st c="23736">On the other hand, the</st> `<st c="23760">werkzeug.security</st>` <st c="23777">module also supports</st> `<st c="23799">pbkdf2</st>` <st c="23805">as an option for the hashing method parameter.</st> <st c="23853">However, it is less secure than the Scrypt algorithm.</st> <st c="23907">Scrypt is a simple and effective hashing algorithm that requires salt to hash a password.</st> <st c="23997">The</st> `<st c="24001">generate_password_hash()</st>` <st c="24025">method defaults the salt length to</st> `<st c="24061">16</st>`<st c="24063">, which can be replaced anytime by passing any preferred length.</st> <st c="24128">Moreover, Scrypt is memory intensive, since it needs storage to temporarily hold all the initial salted random alphanumerics</st> <st c="24252">until it returns the</st> <st c="24274">final passphrase.</st>
			<st c="24291">Since there is no way to re-assemble the passphrase to extract the original text, the</st> `<st c="24378">werkzeug.security</st>` <st c="24395">module has a</st> `<st c="24409">check_password_hash()</st>` <st c="24430">method that returns</st> `<st c="24451">True</st>` <st c="24455">if the given text value matches the hashed value.</st> <st c="24506">The following snippet validates the password of an authenticated user if it matches an account in the database with the same username but a</st> <st c="24646">hashed password:</st>

```

<st c="24662">从 werkzeug.security 模块导入 check_password_hash</st> @auth.verify_password

def verify_password(username, password):

    task = get_user_task_wrapper.apply_async( args=[username])

    login:Login = task.get()

    if login == None:

        abort(403) <st c="24889">if check_password_hash(login.password, password)</st>: <st c="24940">return login.username</st> else:

        abort(403)

```py

			<st c="24978">The</st> `<st c="24983">check_password_hash()</st>` <st c="25004">method requires two parameters, namely the passphrase as the first and the original password as the second.</st> <st c="25113">If the</st> `<st c="25120">werkzeug.security</st>` <st c="25137">module is not the option for your requirement due to its slowness, the</st> `<st c="25317">hashlib</st>` <st c="25324">module using the following</st> `<st c="25352">pip</st>` <st c="25355">command before</st> <st c="25371">applying it:</st>

```

pip install hashlib

```py

			<st c="25403">On the other hand,</st> `<st c="25423">ch09-api-bcrypt</st>` <st c="25438">uses the Bcrypt algorithm to generate a passphrase for a password.</st> <st c="25506">Since</st> `<st c="25597">pip</st>` <st c="25600">command:</st>

```

pip install bcrypt

```py

			<st c="25628">Afterward, instantiate the</st> `<st c="25656">Bcrypt</st>` <st c="25662">class container in the</st> `<st c="25686">create_app()</st>` <st c="25698">factory method and integrate the</st> <st c="25731">module into the Flask application through the</st> `<st c="25778">app</st>` <st c="25781">instance.</st> <st c="25792">The following snippet shows the setup of the Bcrypt module in the</st> <st c="25858">Flask application:</st>

```

<st c="25876">从 flask_bcrypt 模块导入 Bcrypt</st>

<st c="25908">bcrypt = Bcrypt()</st> def create_app(config_file):

    app = Flask(__name__, template_folder='../modules/pages', static_folder='../modules/resources')

    app.config.from_file(config_file, toml.load)

    app.config.from_prefixed_env()

    … … … … … … <st c="26140">bcrypt.init_app(app)</st> … … … … … …

```py

			<st c="26171">The</st> `<st c="26176">bcrypt</st>` <st c="26182">object will provide every module component with the utility methods to hash credential details such as passwords.</st> <st c="26297">The following</st> `<st c="26311">ch09-api-bcrypt</st>`<st c="26326">’s version of the</st> `<st c="26345">add_signup()</st>` <st c="26357">endpoint</st> <st c="26366">hashes the password of an account using the imported</st> `<st c="26420">bcrypt</st>` <st c="26426">object before saving the user’s credentials into</st> <st c="26476">the database:</st>

```

<st c="26489">从 modules 模块导入 bcrypt</st> @current_app.post('/login/signup')

async def add_signup():

    login_json = request.get_json()

    password = login_json["password"] <st c="26642">passphrase = bcrypt.generate_password_hash(password)</st><st c="26694">.decode('utf-8')</st> async with db_session() as sess:

        async with sess.begin():

        repo = LoginRepository(sess)

        … … … … … …

        result = await repo.insert_login(login)

        if result == False:

            return jsonify(message="插入错误"), 201

        return jsonify(record=login_json), 200

```py

			<st c="26955">Like Hashlib algorithms (e.g.,</st> `<st c="26987">scrypt</st>` <st c="26993">or</st> `<st c="26997">pbkdf2</st>`<st c="27003">), Bcrypt is not capable of extracting the original password from the passphrase.</st> <st c="27086">However, it also has a</st> `<st c="27109">check_password_hash()</st>` <st c="27130">method, which validates whether a password has the correct passphrase.</st> <st c="27202">However, compared to</st> <st c="27223">Hashlib, Bcrypt is more secure and modern because it uses the</st> **<st c="27285">Blowfish Cipher</st>** <st c="27300">algorithm.</st> <st c="27312">Its only drawback is its slow</st> <st c="27342">hashing process, which may affect the application’s</st> <st c="27394">overall performance.</st>
			<st c="27414">Aside from hashing, the encryption algorithms can also help secure the internal data of any Flask application, especially passwords.</st> <st c="27548">A well-known module that can provide reliable encryption methods is the</st> `<st c="27620">cryptography</st>` <st c="27632">module.</st> <st c="27641">So, let us first install the module using the following</st> `<st c="27697">pip</st>` <st c="27700">command before using its cryptographic recipes</st> <st c="27748">and utilities:</st>

```

pip install cryptography

```py

			<st c="27787">The</st> `<st c="27792">cryptography</st>` <st c="27804">module offers both symmetric and asymmetric cryptography.</st> <st c="27863">The former uses one key to initiate the encryption and decryption algorithms, while the latter uses two keys: the public and private keys.</st> <st c="28002">Since our application only needs one key to encrypt user credentials, it will use symmetric cryptography through the</st> `<st c="28119">Fernet</st>` <st c="28125">class, the utility class that implements symmetric cryptography for the module.</st> <st c="28206">Now, after the installation, call</st> `<st c="28240">Fernet</st>` <st c="28246">in the</st> `<st c="28254">create_app()</st>` <st c="28266">method to generate a key through its</st> `<st c="28304">generate_key()</st>` <st c="28318">class</st> <st c="28325">method.</st> <st c="28333">The following snippet in the factory method shows how the application created and kept the key for the</st> <st c="28436">entire runtime:</st>

```

<st c="28451">从 cryptography.fernet 模块导入 Fernet</st> def create_app(config_file):

    app = Flask(__name__, template_folder='../modules/pages', static_folder='../modules/resources')

    app.config.from_file(config_file, toml.load)

    … … … … … … <st c="28764">Fernet</st> 令牌或密钥是一个 URL 安全的 base64 编码的字母数字，它将启动加密和解密算法。应用程序应在启动时将密钥存储在安全且不可访问的位置，例如在安全目录内的文件中保存。缺少密钥将导致 <st c="29069">cryptography.fernet.InvalidToken</st> 和 <st c="29106">cryptography.exceptions.InvalidSignature</st> 错误。

            <st c="29154">生成密钥后，使用密钥作为构造函数参数实例化</st> `<st c="29204">Fernet</st>` <st c="29210">类以发出</st> `<st c="29270">encrypt()</st>` <st c="29279">方法。</st> <st c="29288">以下</st> `<st c="29302">ch09-auth-digest</st>`<st c="29318">版本的</st> `<st c="29333">add_signup()</st>` <st c="29345">使用</st> `<st c="29379">Fernet</st>`<st c="29385">加密用户密码</st>：<st c="29373">:</st>

```py
<st c="29387">from cryptography.fernet import Fernet</st> @current_app.post('/login/signup')
async def add_signup():
     … … … … … …
     password = login_json["password"] <st c="29531">with open("enc_key.txt", mode="r") as file:</st><st c="29574">enc_key = bytes(file.read(), "utf-8")</st><st c="29612">fernet = Fernet(enc_key)</st><st c="29637">passphrase = fernet.encrypt(bytes(password, 'utf-8'))</st> async with db_session() as sess:
         async with sess.begin():
           … … … … … …
           result = await repo.insert_login(login)
           … … … … …
           return jsonify(record=login_json), 200
```

            <st c="29850">为了实例化</st> `<st c="29866">Fernet</st>`<st c="29872">，</st> `<st c="29874">add_signup()</st>` <st c="29886">必须从文件中提取令牌，将其转换为字节，并将其作为构造函数参数传递给</st> `<st c="29993">Fernet</st>` <st c="29999">类。</st> <st c="30007">`<st c="30011">Fernet</st>` <st c="30017">实例提供了一个</st> `<st c="30039">encrypt()</st>` <st c="30048">方法，该方法使用</st> `<st c="30174">decrypt()</st>` <st c="30183">方法从加密消息中提取原始</st> `<st c="30194">密码。</st>` <st c="30252">以下是</st> `<st c="30269">ch09-auth-digest</st>`<st c="30285">的密码验证方案，该方案从数据库中检索带有编码密码的用户凭据，并解密编码消息以提取</st> `<st c="30444">实际密码：</st>`

```py
 @auth.get_password
def get_passwd(username):
    task = get_user_task_wrapper.apply_async( args=[username])
    login:Login = task.get() <st c="30590">with open("enc_key.txt", mode="r") as file:</st><st c="30633">enc_key = bytes(file.read(), "utf-8")</st><st c="30671">fernet = Fernet(enc_key)</st><st c="30696">password = fernet.decrypt(login.password)</st><st c="30738">.decode('utf-8')</st> if login == None:
        return None
    else:
        return password
```

            <st c="30806">再次强调，</st> `<st c="30814">get_passwd()</st>` <st c="30826">需要从文件中获取令牌以实例化</st> `<st c="30872">Fernet</st>`<st c="30878">。使用</st> `<st c="30890">Fernet</st>` <st c="30896">实例，</st> `<st c="30907">get_passwd()</st>` <st c="30919">可以发出</st> `<st c="30933">decrypt()</st>` <st c="30942">方法来重新组装加密消息并从</st> `<st c="31025">UTF-8</st>` <st c="31030">格式中提取实际密码。</st> <st c="31039">与哈希相比，加密涉及使用令牌将明文重新组装成不可读且无法破解的密文，并将该密文还原为其原始的可读形式。</st> <st c="31150">因此，它是一个双向过程，与</st> <st c="31262">哈希不同。</st>

            <st c="31273">如果目标是持久化编码数据到数据库中，而不添加可能减慢软件性能的不必要的加密错误，解决方案是</st> <st c="31441">使用</st> `<st c="31445">sqlalchemy_utils</st>`<st c="31461">。</st>

            <st c="31462">使用 sqlalchemy_utils 对加密列进行操作</st>

            <st c="31507">`sqlalchemy_utils`</st> <st c="31512">模块为 SQLAlchemy 模型类提供了额外的实用方法和列类型，其中包括</st> `<st c="31632">StringEncryptedType</st>`<st c="31651">。由于该模块使用了 cryptography 模块的加密方案，在使用</st> `<st c="31766">sqlalchemy_utils</st>` <st c="31782">之前，请务必使用以下</st> `<st c="31803">pip</st>` <st c="31806">命令安装该模块：</st>

```py
 pip install cryptography sqlalchemy_utils
```

            <st c="31857">之后，通过将</st> `<st c="31907">StringEncryptedType</st>` <st c="31926">应用于需要</st> `<st c="31954">Fernet</st>`<st c="31960">加密的表列，例如</st> `<st c="31988">用户名</st>` <st c="31996">和</st> `<st c="32001">密码</st>` <st c="32009">列。</st> <st c="32019">以下是在</st> `<st c="32040">Login</st>` <st c="32045">模型类中包含</st> `<st c="32065">ch09-web-passphrase</st>` <st c="32084">项目，并具有</st> `<st c="32098">用户名</st>` <st c="32106">和</st> `<st c="32111">密码</st>` <st c="32119">列</st> <st c="32128">的</st> `<st c="32131">StringEncryptedType</st>`<st c="32150">的</st>示例：

```py
<st c="32152">from sqlalchemy_utils import StringEncryptedType</st>
<st c="32200">enc_key = "packt_pazzword"</st> class Login(Base):
   __tablename__ = 'login'
   id = Column(Integer, Sequence('login_id_seq', increment=1), primary_key = True) <st c="32351">username = Column(StringEncryptedType(String(20), enc_key), nullable=False, unique=True)</st><st c="32439">password = Column(StringEncryptedType(String(50), enc_key), nullable=False)</st> role = Column(Integer, nullable=False)
   … … … … … …
```

            `<st c="32566">StringEncryptedType</st>` <st c="32586">会在</st> `<st c="32637">INSERT</st>` <st c="32643">事务期间自动加密列数据，并在</st> `<st c="32704">SELECT</st>` <st c="32710">语句中解密编码的字段值。</st> <st c="32723">要将实用类应用于列，请将其映射到包含实际 SQLAlchemy 列</st> `<st c="32830">类型</st>` <st c="32862">和自定义生成的</st> `<st c="32862">Fernet</st>` <st c="32868">令牌的列字段。</st> <st c="32876">它看起来像是一个列字段包装器，它将过滤和加密插入的字段值，并在检索时解密。</st> <st c="33003">对于这些字段值，不需要从</st> `<st c="33039">视图</st>` <st c="33043">函数或存储库层进行其他额外的编码来执行加密</st> `<st c="33109">和解密过程。</st>`

            <st c="33157">当使用</st> `<st c="33169">Flask-Migrate</st>`<st c="33182">时，在运行</st> `<st c="33243">db init</st>` <st c="33249">命令后，并在运行</st> `<st c="33254">script.py.mako</st>` <st c="33268">文件中的</st> `<st c="33286">migrations</st>` <st c="33296">文件夹内的</st> `<st c="33243">env.py</st>` <st c="33249">和</st> `<st c="33254">script.py.mako</st>` <st c="33268">文件之前，将</st> `<st c="33192">import sqlalchemy_utils</st>` <st c="33215">语句添加到生成的</st> `<st c="33243">env.py</st>` <st c="33249">和</st> `<st c="33254">script.py.mako</st>` <st c="33268">文件中。</st> <st c="33399">以下是在导入</st> `<st c="33484">sqlalchemy_utils</st>` <st c="33500">模块后修改的</st> `<st c="33430">ch09-web-passphrase</st>` <st c="33449">迁移文件：</st>

```py
 (env.py)
import logging
from logging.config import fileConfig
from flask import current_app
from alembic import context <st c="33629">import sqlalchemy_utils</st> … … … … … …
(script.py.mako)
"""${message}
Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa <st c="33838">import sqlalchemy_utils</st> ${imports if imports else ""}
… … … … … …
```

            <st c="33903">在提供的迁移文件中提供的突出显示的行是添加用于 SQLAlchemy 模型类的额外导入的正确位置。</st> <st c="34045">这包括不仅</st> `<st c="34072">sqlalchemy_util</st>` <st c="34087">类，还包括可能帮助建立所需</st> `<st c="34160">模型层</st>` 的其他库。</st>

            <st c="34172">在自定义用户认证时，应用程序利用默认的 Flask 会话来存储用户信息，例如用户名。</st> <st c="34307">此会话将信息保存到浏览器。</st> <st c="34354">为了保护应用程序免受破坏性访问控制攻击，您可以使用可靠的认证和授权机制，或者如果自定义基于会话的认证符合您的需求，可以通过</st> **<st c="34525">Flask-Session</st>** <st c="34538">模块应用服务器端会话处理。</st> <st c="34591">如果</st> <st c="34538">自定义会话基于认证</st> <st c="34591">满足您的需求。</st>

            <st c="34609">利用服务器端会话</st>

            <st c="34644">在</st> *<st c="34648">第四章</st>*<st c="34657">中，将</st> `<st c="35144">username</st>`<st c="35152">公开。</st>

            <st c="35168">在通过</st> `<st c="35190">Session</st>` <st c="35197">模块的</st> `<st c="35230">app</st>` <st c="35233">实例设置</st> `<st c="35261">session</st>` <st c="35268">字典对象后，Flask 的内置</st> `<st c="35261">session</st>` <st c="35268">会话对象可以轻松地在服务器端存储会话数据。</st> <st c="35338">以下</st> `<st c="35352">login_user()</st>` <st c="35364">视图函数在用户</st> <st c="35390">username</st> <st c="35398">凭证</st><st c="35415">确认后，将凭证</st> <st c="35456">credential</st> <st c="35467">存储到服务器端会话中：</st>

```py
 @current_app.route('/login/auth', methods=['GET', 'POST'])
async def login_user():
    if request.method == 'GET':
        return render_template('login/authenticate.html'), 200
    username = request.form['username'].strip()
    password = request.form['password'].strip()
    async with db_session() as sess:
        async with sess.begin():
            repo = LoginRepository(sess)
            records = await repo.select_login_username_passwd(username, password)
            login_rec = [rec.to_json() for rec in records]
            if len(login_rec) >= 1:
                session["user"] = username
                return redirect('/menu')
            else:
                … … … … … …
                return render_template('login/authenticate.html'), 200
```

            <st c="36087">在登出时需要清除所有会话数据。</st> <st c="36143">删除所有会话数据将会</st> `<st c="36294">logout()</st>` <st c="36302">视图函数的</st> `<st c="36324">ch09-web-paraphrase</st>` <st c="36343">项目：</st>

```py
 @current_app.route('/logout', methods=['GET'])
async def logout(): <st c="36420">session["user"] = None</st> return redirect('/login/auth')
```

            <st c="36473">除了将</st> <st c="36496">会话属性设置为</st> `<st c="36518">None</st>`<st c="36522">之外，</st> `<st c="36528">pop()</st>` <st c="36533">方法也可以帮助删除会话数据。</st> <st c="36602">删除所有会话数据等同于使当前会话失效。</st>

            <st c="36676">现在，如果自定义网页登录实现不符合您的需求，</st> **<st c="36757">Flask-Login</st>** <st c="36768">模块可以提供用户登录和登出的内置实用工具。</st> <st c="36832">现在让我们讨论如何使用 Flask-Login 进行</st> <st c="36880">Flask 应用程序。</st>

            <st c="36898">实现网页表单认证</st>

            `<st c="37184">flask-login</st>`<st c="37195">，首先使用以下</st> `<st c="37234">pip</st>` <st c="37237">命令安装它：</st>

```py
 pip install flask-login
```

            <st c="37270">同时，安装并设置 Flask-Session 模块，以便 Flask-Login 将其用户会话存储在</st> <st c="37366">文件系统中。</st>

            <st c="37381">然后，要将 Flask-login 集成到 Flask 应用中，需要在</st> `<st c="37457">LoginManager</st>` <st c="37469">类中</st> `<st c="37483">create_app()</st>` <st c="37495">方法中实例化它，并通过</st> `<st c="37529">app</st>` <st c="37532">实例</st> `<st c="37543">设置它。</st> <st c="37581">session_protection</st>`<st c="37599">属性需要安装 Flask-Bcrypt，而</st> `<st c="37654">login_view</st>`<st c="37664">属性则指定了</st> `<st c="37687">登录视图</st>` <st c="37697">函数。</st> <st c="37708">以下代码片段</st> <st c="37730">展示了为我们的</st> `<st c="37769">ch09-web-login</st>` <st c="37783">项目</st> `<st c="37730">设置 Flask-Login 的方法：</st>

```py
<st c="37792">from flask_login import LoginManager</st> def create_app(config_file):
    app = Flask(__name__,template_folder= '../modules/pages', static_folder=   '../modules/resources')
    app.config.from_file(config_file, toml.load)
    app.config.from_prefixed_env()
    … … … … … … <st c="38224">Login</st> model class to the model layer through your desired ORM and sub-class it with the <st c="38312">UserMixin</st> helper class of the <st c="38342">flask-login</st> module. The following is the <st c="38383">Login</st> model of our project that will persist the user’s <st c="38439">id</st>, <st c="38443">username</st>, <st c="38453">passphrase</st>, and <st c="38469">role</st>:

```

<st c="38475">from flask_login import UserMixin</st> from sqlalchemy_utils import StringEncryptedType

enc_key = "packt_pazzword" <st c="38585">class Login(UserMixin, Base):</st> __tablename__ = 'login'

id = Column(Integer, Sequence('login_id_seq', increment=1), primary_key = True)

username = Column(StringEncryptedType(String(20), enc_key), nullable=False, unique=True)

password = Column(StringEncryptedType(String(50), enc_key), nullable=False)

role = Column(Integer, nullable=False)

… … … … … …

```py

			<st c="38934">Instead of utilizing the Flask-Bcrypt module, our application uses the built-in</st> `<st c="39015">StringEncryptedType</st>` <st c="39034">hashing mechanism from the</st> `<st c="39062">sqlalchemy_utils</st>` <st c="39078">module.</st> <st c="39087">Now, the use of the</st> `<st c="39107">UserMixin</st>` <st c="39116">superclass allows the use of some properties, such as</st> `<st c="39171">is_authenticated</st>`<st c="39187">,</st> `<st c="39189">is_active</st>`<st c="39198">, and</st> `<st c="39204">is_anonymous</st>`<st c="39216">, as well as some utility methods, such</st> <st c="39255">as</st> `<st c="39259">get_id ()</st>`<st c="39268">, provided by the</st> `<st c="39286">current_user</st>` <st c="39298">object from the</st> `<st c="39315">flask_login</st>` <st c="39326">module.</st>
			<st c="39334">Flask-Login stores the</st> `<st c="39358">id</st>` <st c="39360">of a user in the session after a successful login authentication.</st> <st c="39427">With</st> `<st c="39432">Flask-Session</st>`<st c="39445">, it will store the</st> `<st c="39465">id</st>` <st c="39467">somewhere that has been secured.</st> <st c="39501">The</st> `<st c="39505">id</st>` <st c="39507">is vital to the</st> `<st c="39702">id</st>` <st c="39704">from the session, retrieves the object using its</st> `<st c="39754">id</st>` <st c="39756">parameter and a repository class, and returns the</st> `<st c="39807">Login</st>` <st c="39812">object to the application.</st> <st c="39840">Here is</st> `<st c="39848">ch09-web-login</st>`<st c="39862">’s implementation for the</st> <st c="39889">user loader:</st>

```

<st c="39901">@login_auth.user_loader</st> def load_user(<st c="39940">id</st>):

    task = get_user_task_wrapper.apply_async(args=[id])

    result = task.get() <st c="40077">main.py</st> 模块。现在，使用 <st c="40108">get_user_task_wrapper()</st> Celery 任务，<st c="40149">load_user()</st> 使用 <st c="40170">select_login()</st> 事务的 <st c="40204">LoginRepository</st> 来根据给定的 <st c="40234">id</st> 参数检索一个 <st c="40266">Login</st> 记录。应用程序会自动为每个请求访问视图而调用 <st c="40316">load_user()</st>。对回调函数的连续调用检查用户的合法性。返回的 <st c="40458">Login</st> 对象作为令牌，允许用户访问应用程序。要声明用户加载回调函数，创建一个带有本地 <st c="40615">id</st> 参数的函数，并用 <st c="40653">userloader()</st> 装饰器装饰 <st c="40683">LoginManager</st> 实例。

            <st c="40705">登录视图</st> `<st c="40710">功能</st>` <st c="40720">缓存</st> `<st c="40741">登录</st>` <st c="40746">对象，保存</st> `<st c="40765">登录</st>`<st c="40770">的</st> `<st c="40774">id</st>` <st c="40776">到会话中，并将其映射到</st> `<st c="40812">当前用户</st>` <st c="40824">的</st> `<st c="40845">flask_login</st>` <st c="40856">模块的代理对象。</st> <st c="40865">以下代码片段</st> <st c="40887">显示了</st> `<st c="40897">登录视图</st>` <st c="40907">功能，该功能在我们的</st> `<st c="40930">设置</st>` <st c="40930">中指定：</st>

```py
<st c="40940">from flask_login import login_user</st> @current_app.route('/login/auth', methods=['GET', 'POST']) <st c="41035">async def login_valid_user():</st> if request.method == 'GET':
     return render_template('login/authenticate.html'), 200
  username = request.form['username'].strip()
  password = request.form['password'].strip()
  async with db_session() as sess:
      async with sess.begin():
         repo = LoginRepository(sess)
         records = await repo.select_login_username_passwd( username, password)
         if(len(records) >= 1): <st c="41417">login_user(records[0])</st> return render_template('login/signup.html'), 200
         else:
            … … … … … …
           return render_template( 'login/authenticate.html'), 200
```

            <st c="41562">如果根据数据库验证，</st> `<st c="41570">POST</st>`<st c="41574">-提交的用户凭据是正确的，那么</st> `<st c="41653">登录视图</st>` <st c="41663">功能，即我们的</st> `<st c="41685">login_valid_user()</st>`<st c="41703">，应该调用</st> `<st c="41723">login_user()</st>` <st c="41735">模块的</st> `<st c="41750">flask_login</st>` <st c="41761">方法。</st> <st c="41770">视图必须将包含用户登录凭据的查询</st> `<st c="41801">登录</st>` <st c="41806">对象传递给</st> `<st c="41861">login_user()</st>` <st c="41873">函数。</st> <st c="41884">除了</st> `<st c="41899">登录</st>` <st c="41904">对象之外，该方法还有其他选项，例如</st> <st c="41951">以下内容：</st>

                +   `<st c="41965">记住</st>`<st c="41974">: 一个布尔</st> <st c="41987">参数，用于启用</st> `<st c="42014">记住我</st>` <st c="42025">功能，该功能允许用户会话在浏览器意外退出后仍然保持活跃。</st>

                +   `<st c="42116">fresh</st>`<st c="42122">: 一个布尔参数，用于将用户设置为</st> `<st c="42165">未新鲜</st>` <st c="42174">如果他们的会话在浏览器关闭后立即有效。</st>

                +   `<st c="42238">强制</st>`<st c="42244">: 一个布尔参数，用于强制用户登录。</st>

                +   `<st c="42302">duration</st>`<st c="42311">: 在</st> `<st c="42344">remember_me</st>` <st c="42356">cookie 过期之前的时间。</st>

            <st c="42371">在成功认证之后，用户现在可以访问受限视图或 API，这些视图或 API 对未认证用户不可用：那些带有</st> `<st c="42524">@login_required</st>` <st c="42539">装饰器的视图。</st> <st c="42551">以下是我们</st> `<st c="42599">ch09-web-login</st>` <st c="42613">的示例视图函数，它需要认证</st> <st c="42639">用户访问：</st>

```py
 from flask_login import login_required
@current_app.route('/doctor/profile/add', methods=['GET', 'POST'])
@login_required
async def add_doctor_profile():
    if request.method == 'GET':
        async with db_session() as sess:
            async with sess.begin():
                repo = LoginRepository(sess)
                records = await repo.select_all_doctor()
                doc_rec = [rec.to_json() for rec in records]
                return render_template('doctor/add_doctor_profile.html', docs=doc_rec), 200
    else:
        username = request.form['username']
        … … … … … …
        return render_template('doctor/add_doctor_profile.html', doctors=doc_rec), 200
```

            <st c="43215">除了装饰器之外，</st> `<st c="43246">当前用户</st>`<st c="43258">的</st> `<st c="43262">is_authenticated</st>` <st c="43278">属性还可以限制视图和</st> `<st c="43320">Jinja 模板</st>` <st c="43360">中某些代码片段的执行。</st>

            <st c="43376">最后，为了完成 Flask-Login 的集成，实现一个</st> <st c="43468">logout_user()</st>` <st c="43481">的 flask_login</st> <st c="43497">模块的实用工具。</st> <st c="43517">以下代码是我们项目的注销视图实现：</st>

```py
<st c="43586">from flask_login import logout_user</st> @current_app.route("/logout") <st c="43653">async def logout():</st><st c="43672">logout_user()</st> return <st c="43694">redirect(url_for('login_valid_user'))</st>
```

            <st c="43731">确保注销视图将用户重定向到登录视图页面，而不是仅仅渲染登录页面，以</st> <st c="43846">避免</st> **<st c="43853">HTTP 状态码 405</st>** <st c="43873">(</st>*<st c="43875">方法不允许</st>*<st c="43893">)</st> <st c="43896">在重新登录期间。</st>

            <st c="43912">拥有一个安全的</st> <st c="43934">Web 表单认证能否防止 CSRF 攻击的发生？</st> <st c="43996">让我们专注于保护我们的应用程序免受那些想要将事务转移到其他</st> <st c="44096">可疑网站上的攻击者。</st>

            <st c="44113">防止 CSRF 攻击</st>

            <st c="44137">CSRF 攻击是一种攻击方式，通过欺骗已认证用户将敏感数据转移到隐藏和恶意网站。</st> <st c="44255">这种攻击发生在</st> <st c="44274">用户执行</st> `<st c="44294">POST</st>`<st c="44298">,</st> `<st c="44300">DELETE</st>`<st c="44306">,</st> `<st c="44308">PUT</st>`<st c="44311">, 或</st> `<st c="44316">PATCH</st>` <st c="44321">事务时，此时表单数据被检索并提交到应用程序。</st> <st c="44402">在 Flask 中，最常用的解决方案是使用</st> `<st c="44447">Flask-WTF</st>` <st c="44456">，因为它有一个内置的</st> `<st c="44483">CSRFProtect</st>` <st c="44494">类，可以全局保护应用程序的每个表单事务。</st> <st c="44567">一旦启用，</st> `<st c="44581">CSRFProtect</st>` <st c="44592">允许为每个表单事务生成唯一的令牌。</st> <st c="44660">那些不会生成令牌的表单提交将导致</st> `<st c="44725">CSRFProtect</st>` <st c="44736">触发错误消息，检测到</st> <st c="44778">CSRF 攻击。</st>

            *<st c="44790">第四章</st>* <st c="44800">强调了在 Flask 应用程序中设置 Flask-</st>`<st c="44835">WTF</st>` <st c="44839">模块。</st> <st c="44871">安装后，导入</st> `<st c="44902">CSRFProtect</st>` <st c="44913">并在</st> `<st c="44936">create_app()</st>`<st c="44948">中实例化它，如下面的</st> <st c="44976">代码片段</st>所示：

```py
<st c="44989">from flask_wtf.csrf import CSRFProtect</st> def create_app(config_file):
    app = Flask(__name__,template_folder= '../modules/pages', static_folder= '../modules/resources')
    … … … … … … <st c="45223">SECRET_KEY</st> or <st c="45237">WTF_CSRF_SECRET_KEY</st> to be defined in the configuration file. Now, after integrating it into the application through the <st c="45357">app</st> instance, all <st c="45375"><form></st> in Jinja templates must have the <st c="45415">{{ csrf_token() }}</st> component or a <st c="45449"><input type="hidden" name="csrf_token" value = "{{ csrf_token() }}" /></st> component to impose CSRF protection.
			<st c="45556">If there is no plan to use the entire</st> `<st c="45595">Flask-WTF</st>`<st c="45604">, another</st> <st c="45614">option is to apply</st> **<st c="45633">Flask-Seasurf</st>** <st c="45646">instead.</st>
			<st c="45655">After showcasing the web-based authentication strategies, it is now time to discuss the different authentication types</st> <st c="45774">for</st> <st c="45779">API-based applications.</st>
			<st c="45802">Implementing user authentication and authorization</st>
			<st c="45853">There is a strong foundation of extension modules that can secure API services from unwanted access, such as the</st> `<st c="46426">Flask-HTTPAuth</st>` <st c="46440">module in</st> <st c="46451">our application.</st>
			<st c="46467">Utilizing the Flask-HTTPAuth module</st>
			<st c="46503">After you have installed the</st> `<st c="46533">Flask-HTTPAuth</st>` <st c="46547">module and its extensions, it can provide its</st> `<st c="46594">HTTPBasicAuth</st>` <st c="46607">class to</st> <st c="46617">build Basic authentication, the</st> `<st c="46649">HTTPDigestAuth</st>` <st c="46663">class to implement Digest authentication, and the</st> `<st c="46714">HTTPTokenAuth</st>` <st c="46727">class for the Bearer token</st> <st c="46755">authentication scheme.</st>
			<st c="46777">Basic authentication</st>
			<st c="46798">Basic authentication requires</st> <st c="46829">an unencrypted base64 format of the user’s</st> `<st c="46872">username</st>` <st c="46880">and</st> `<st c="46885">password</st>` <st c="46893">credentials through the</st> `<st c="46918">Authorization</st>` <st c="46931">request</st> <st c="46939">header.</st> <st c="46948">To implement this authentication type in Flask, instantiate the module’s</st> `<st c="47021">HTTPBasicAuth</st>` <st c="47034">in</st> `<st c="47038">create_app()</st>` <st c="47050">and register the instance to the Flask</st> `<st c="47090">app</st>` <st c="47093">instance, as shown in the</st> <st c="47120">following snippet:</st>

```

<st c="47138">from flask_httpauth import HTTPBasicAuth</st> def create_app(config_file):

    app = Flask(__name__,template_folder= '../modules/pages', static_folder= '../modules/resources')

    app.config.from_file(config_file, toml.load)

    … … … … … … <st c="47408">HTTPBasicAuth</st> 实现需要一个回调函数，该函数将从客户端检索用户名和密码，查询数据库以检查用户记录，并在存在有效用户的情况下将有效用户名返回给应用程序，如下面的代码所示：

```py
 app, celery_app, <st c="47681">auth</st> = <st c="47688">create_app('../config_dev.toml')</st> … … … … … … <st c="47732">@auth.verify_password</st> def verify_password(<st c="47774">username, password</st>): <st c="47797">task = get_user_task_wrapper.apply_async( args=[username])</st><st c="47855">login:Login = task.get()</st> if login == None: <st c="47899">abort(403)</st><st c="47909">if check_password_hash(login.password,password) == True:</st><st c="47966">return login.username</st> else: <st c="48012">HTTPBasicAuth</st>’s callback function, the given <st c="48058">check_password()</st> must have the <st c="48089">username</st> and <st c="48102">password</st> parameters and should be annotated with <st c="48151">HTTPBasicAuth</st>’s <st c="48168">verify_password()</st> decorator. Our callback uses the Celery task to search and retrieve the <st c="48258">Login</st> object containing the <st c="48286">username</st> and <st c="48299">password</st> details and raises <st c="48481">HTTPBasicAuth</st>’s <st c="48498">login_required()</st> decorator.
			<st c="48525">The</st> `<st c="48530">Flask-HTTPAuth</st>` <st c="48544">module has built-in authorization</st> <st c="48579">support.</st> <st c="48588">If the basic authentication needs a</st> <st c="48623">role-based authorization, the application only needs to have a separate callback function decorated by</st> `<st c="48727">get_user_roles()</st>` <st c="48743">from the</st> `<st c="48753">HTTPBasicAuth</st>` <st c="48766">class.</st> <st c="48774">The following is</st> `<st c="48791">ch09-auth-basic</st>`<st c="48806">’s callback function for retrieving user roles from</st> <st c="48859">its users:</st>

```

<st c="48869">from werkzeug.datastructures.auth import Authorization</st> app, celery_app, <st c="48942">auth</st> = <st c="48949">create_app('../config_dev.toml')</st> … … … … … … <st c="48993">@auth.get_user_roles</st> def get_scope(<st c="49028">user:Authorization</st>):

    task = get_user_task_wrapper.apply_async( args=[user.username])

    login:Login = task.get() <st c="49173">get_scope()</st> 自动从请求中检索 <st c="49213">werkzeug.datastructures.auth.Authorization</st> 对象。该 <st c="49285">Authorization</st> 对象包含 <st c="49319">username</st>，这是 <st c="49341">get_user_task_wrapper()</st> Celery 任务将基于其从数据库中搜索用户的 <st c="49406">Login</st> 记录对象的基础。回调函数的返回值可以是字符串格式的单个角色，也可以是分配给用户的角色列表。来自 <st c="49623">ch09-auth-digest</st> 项目的 <st c="49590">del_doctor_profile_id()</st> 不允许任何经过身份验证的用户，除了那些 <st c="49713">role</st> 等于 <st c="49739">1</st> 的医生：

```py
 @current_app.delete('/doctor/profile/delete/<int:id>') <st c="49802">@auth.login_required(role="1")</st> async def del_doctor_profile_id(id:int):
    async with db_session() as sess:
      async with sess.begin():
          repo = DoctorRepository(sess)
          … … … … … …
          return jsonify(record="deleted record"), 200
```

            <st c="50018">在这里，</st> `<st c="50025">del_doctor_profile_id()</st>` <st c="50048">是一个 API 函数，用于在数据库中删除医生的个人信息。</st> <st c="50129">只有医生本人（</st>`<st c="50181">role=1</st>`<st c="50188">）才能执行此交易。</st>

            <st c="50207">摘要认证</st>

            <st c="50229">另一方面，该模块的</st> `<st c="50262">HTTPDigestAuth</st>` <st c="50276">为基于 API 的应用程序构建摘要认证方案，该方案</st> <st c="50352">加密了凭证以及一些附加头信息，例如</st> `<st c="50441">realm</st>`<st c="50446">,</st> `<st c="50448">nonce</st>`<st c="50453">,</st> `<st c="50455">opaque</st>`<st c="50461">，以及</st> `<st c="50467">nonce count</st>`<st c="50478">。因此，它比</st> <st c="50509">基本认证方案</st> <st c="50542">更安全。</st> <st c="50542">以下代码片段展示了如何在</st> `<st c="50613">create_app()</st>` <st c="50625">工厂中设置摘要认证：</st>

```py
<st c="50634">from flask_httpauth import HTTPDigestAuth</st> def create_app(config_file):
    app = Flask(__name__,template_folder= '../modules/pages', static_folder= '../modules/resources')
    app.config.from_file(config_file, toml.load)
    … … … … … … <st c="50900">HTTPDigestAuth</st>’s constructors have five parameters, two of which have default values, namely <st c="50994">qop</st> and <st c="51002">algorithm</st>. The <st c="51061">auth</st> value, which means that the application is at the basic protection level of the digest scheme. So far, the highest protection level is <st c="51201">auth-int</st>, which is, at the time of writing this book, not yet functional in the <st c="51281">Flask-HTTPAuth</st> module. The other parameter, <st c="51325">algorithm</st>, has the <st c="51344">md5</st> default value for the encryption, but the requirement can change it to <st c="51419">md5-sess</st> or any supported encryption method. Now, the three other optional parameters are the following:

				*   `<st c="51523">realm</st>`<st c="51529">: This contains the</st> `<st c="51550">username</st>` <st c="51558">of the user and the name of the</st> <st c="51591">application’s host.</st>
				*   `<st c="51610">scheme</st>`<st c="51617">: This is a replacement to the default value of the</st> `<st c="51670">Digest scheme</st>` <st c="51683">header in the</st> `<st c="51698">WWW-Authenticate</st>` <st c="51714">response.</st>
				*   `<st c="51724">use_hw1_pw</st>`<st c="51735">: If this is set to</st> `<st c="51756">True</st>`<st c="51760">, the</st> `<st c="51766">get_password()</st>` <st c="51780">callback function must return a</st> <st c="51813">hashed password.</st>

			<st c="51829">In digest authentication, the user must submit their username, password, nonce, opaque, and nonce count to the application for verification.</st> *<st c="51971">Figure 9</st>**<st c="51979">.4</st>* <st c="51981">shows a postman client submitting the header information to the</st> `<st c="52046">ch09-auth-digest</st>` <st c="52062">app:</st>
			![Figure 9.4 – The Digest authentication scheme’s additional headers](img/B19383_09_004.jpg)

			<st c="52864">Figure 9.4 – The Digest authentication scheme’s additional headers</st>
			<st c="52930">A nonce is a unique base64 or hexadecimal string that the server generates for every</st> **<st c="53016">HTTP status code 401</st>** <st c="53036">response.</st> <st c="53047">The content of the compressed string, usually the estimated timestamp when the client</st> <st c="53133">received the response, must be unique to</st> <st c="53174">every access.</st>
			<st c="53187">Also specified by the server is</st> **<st c="53220">opaque</st>**<st c="53226">, a base64 or hexadecimal string value that the client needs to return to the</st> <st c="53304">server for validation if it is the same value as</st> <st c="53353">generated before.</st>
			<st c="53370">The nonce count value or</st> `<st c="53463">0000001</st>`<st c="53470">, that checks the integrity of the user credentials and protects data from playback attacks.</st> <st c="53563">The server increments its copy of the nc-value when it receives the same nonce value from a new request.</st> <st c="53668">Every authentication request must bear a new nonce value.</st> <st c="53726">Otherwise, it is</st> <st c="53743">a replay.</st>
			`<st c="53752">Flask-HTTPAuth</st>`<st c="53767">’s digest authentication scheme will only work if our API application provides the following</st> <st c="53861">callback implementations:</st>

```

<st c="53886">server_nonce = "H9OVSzjcB57StMQFPInmX22uZ0Kwu_4JptsWrj0oPpU"</st>

<st c="53947">server_opaque = "XJIXDX615CMGXXL0COHQQ0IJRG33OFTNGNFYT72VJ8XF5U3RYZ"</st>

<st c="54016">@auth.generate_nonce</st> def gen_nonce():

    return server_nonce <st c="54075">@auth.generate_opaque</st> def gen_opaque():

    return server_opaque

```py

			<st c="54135">The application must generate the nonce and opaque values using the likes of the</st> `<st c="54217">gen_nonce()</st>` <st c="54228">and</st> `<st c="54233">gen_opaque()</st>` <st c="54245">callbacks.</st> <st c="54257">These are just trivial implementations of the methods in our</st> `<st c="54318">ch09-auth-digest</st>` <st c="54334">application and need better solutions that use a UUID generator</st> <st c="54398">or a</st> `<st c="54404">secrets</st>` <st c="54411">module to generate the values.</st> <st c="54443">The nonce generator callback must have a</st> `<st c="54484">generate_nonce()</st>` <st c="54500">decorator, while the opaque generator must be decorated by the</st> `<st c="54564">generate_opaque()</st>` <st c="54581">annotation.</st>
			<st c="54593">Aside from these generators, the authentication scheme also needs the following validators of nonce and opaque values that the server needs from the</st> <st c="54743">client request:</st>

```

<st c="54758">@auth.verify_nonce</st> def verify_once(nonce):

    if nonce == server_nonce:

        return True

    else:

        return False <st c="54859">@auth.verify_opaque</st> def verify_opaque(opaque):

    if opaque == server_opaque:

        return True

    else:

        return False

```py

			<st c="54964">The validators check whether the values from the request are correct based on the server’s corresponding values.</st> <st c="55078">Now, the last</st> <st c="55092">requirement for the digest authentication to work is the</st> `<st c="55149">get_password()</st>` <st c="55163">callback that</st> <st c="55178">retrieves the password from the client for database validation of the user’s existence, as shown in the</st> <st c="55282">following snippet:</st>

```

<st c="55300">@auth.get_password</st> def get_passwd(username):

    print(username)

    task = get_user_task_wrapper.apply_async(args=[username])

    login:Login = task.get()

    … … … … … …

    if login == None: <st c="55475">return None</st> else: <st c="55548">get_password()</st> 方法每次访问 API 资源时都会调用，并提供有效用户的密码作为令牌。

            <st c="55652">Bearer 令牌认证</st>

            <st c="55680">除了基本和摘要之外，</st> `<st c="55714">Flask-HTTPAuth</st>` <st c="55728">模块还通过使用</st> `<st c="55806">HTTPTokenAuth</st>` <st c="55819">类</st> <st c="55827">支持</st> `<st c="55841">create_app()</st>` <st c="55853">片段</st> <st c="55861">的</st> `<st c="55869">ch09-auth-token</st>` <st c="55884">项目</st> <st c="55897">设置</st> `<st c="55912">Bearer</st>` <st c="55912">令牌认证：</st>

```py
<st c="55933">from flask_httpauth import HTTPTokenAuth</st> def create_app(config_file):
    app = Flask(__name__,template_folder= '../modules/pages', static_folder=   '../modules/resources')
    app.config.from_file(config_file, toml.load)
    … … … … … … <st c="56236">token</st> column field of the <st c="56262">string</st> type in the <st c="56281">Login</st> model to persist the token associated with the user. The application generates the token after the user signs up for login credentials. Our application uses the <st c="56448">PyJWT</st> module for token generation, as depicted in the following endpoint function:

```

<st c="56530">from jwt import encode</st>

<st c="56553">@current_app.post('/login/signup')</st> async def add_signup():

    login_json = request.get_json()

    password = login_json["password"]

    passphrase = generate_password_hash(password) <st c="56725">token = encode({'username': login_json["username"],</st> <st c="56776">'exp': int(time()) + 3600},</st> <st c="56804">current_app.config['SECRET_KEY'],</st> <st c="56838">algorithm='HS256')</st> async with db_session() as sess:

        async with sess.begin():

            repo = LoginRepository(sess)

            login = Login(username=login_json["username"], password=passphrase, <st c="57013">token=token</st>,   role=login_json["role"])

            result = await repo.insert_login(login)

            … … … … … …

            return jsonify(record=login_json), 200

```py

			<st c="57141">The token’s</st> `<st c="57175">username</st>` <st c="57183">and the token’s supposed expiration time in seconds.</st> <st c="57237">The encoding indicated in the given</st> `<st c="57273">add_signup()</st>` <st c="57285">API method is the</st> `<st c="57371">SECRET_KEY</st>`<st c="57381">. The token is always part of the</st> <st c="57414">request’s</st> `<st c="57425">Authorization</st>` <st c="57438">header with the</st> `<st c="57455">Bearer</st>` <st c="57461">value.</st> <st c="57469">Now, a callback function retrieves the bearer token from the request and checks whether it is the saved token of the user.</st> <st c="57592">The</st> <st c="57596">following is</st> `<st c="57609">ch09-auth-token</st>`<st c="57624">’s callback</st> <st c="57637">function implementation:</st>

```

<st c="57661">from jwt import decode</st>

<st c="57684">@auth.verify_token</st> def verify_token(token):

    try: <st c="57734">data = decode(token, app.config['SECRET_KEY'],</st><st c="57780">algorithms=['HS256'])</st> except:

        return False

    if 'username' in data:

        return data['username']

```py

			<st c="57870">The Bearer token’s callback function must have the</st> `<st c="57922">verify_token()</st>` <st c="57936">method decorator.</st> <st c="57955">It has the</st> `<st c="57966">token</st>` <st c="57971">parameter and it returns either a boolean value or the username.</st> <st c="58037">It must use the same</st> `<st c="58155">PyJWT</st>` <st c="58160">module encodes and decodes</st> <st c="58188">the token.</st>
			<st c="58198">Like basic, the digest and bearer token</st> <st c="58239">authentication schemes use the</st> `<st c="58270">login_required()</st>` <st c="58286">decorator to impose restrictions on API endpoints.</st> <st c="58338">Also, both can implement role-based authorization</st> <st c="58388">with the</st> `<st c="58397">get_user_roles()</st>` <st c="58413">callback.</st>
			<st c="58423">The next Flask module,</st> `<st c="58447">Authlib</st>`<st c="58454">, has core classes and methods for implementing OAuth2, OpenID Connect, and JWT Token-based authentication schemes.</st> <st c="58570">Let us now</st> <st c="58581">showcase it.</st>
			<st c="58593">Utilizing the Authlib module</st>
			`<st c="59046">authlib</st>` <st c="59053">module, install its module using the following</st> `<st c="59101">pip</st>` <st c="59104">command:</st>

```

pip install authlib

```py

			<st c="59133">If the application to secure is not running on an HTTPS protocol, set the</st> `<st c="59208">AUTHLIB_INSECURE_TRANSPORT</st>` <st c="59234">environment variable to</st> `<st c="59259">1</st>` <st c="59260">or</st> `<st c="59264">True</st>` <st c="59268">for Authlib to work because it is for a</st> <st c="59309">secured environment.</st>
			<st c="59329">Unlike the HTTP Basic, Digest, and Bearer Token authentication schemes, the OAuth2.0 scheme uses an authorization server that provides several endpoints for authorization procedures, as well as issuing tokens, refreshing tokens, and revoking tokens.</st> <st c="59580">The authorization server is always part of an application that protects its resources from malicious access and attacks.</st> <st c="59701">Our</st> `<st c="59705">ch09-oauth2-password</st>` <st c="59725">project implements the Vaccine and Reports applications with the OAuth2 Resource Owner Password authorization scheme using Authlib.</st> <st c="59858">The following</st> `<st c="59872">create_app()</st>` <st c="59884">factory method shows how to set up</st> <st c="59920">this scheme:</st>

```

<st c="59932">from authlib.integrations.flask_oauth2 import AuthorizationServer</st>

<st c="59998">from authlib.integrations.flask_oauth2 import ResourceProtector</st> from modules.security.oauth2_config import <st c="60106">PasswordGrant</st>, <st c="60121">query_client</st>, <st c="60135">save_token</st>

<st c="60145">require_oauth = ResourceProtector()</st>

<st c="60181">oauth_server = AuthorizationServer()</st> def create_app(config_file):

    app = Flask(__name__,template_folder= '../modules/pages', static_folder=   '../modules/resources')

    … … … … … …

    oauth_server.init_app(app, query_client=<st c="60397">query_client</st>, save_token=<st c="60423">save_token</st>)

    oauth_server.register_grant(<st c="60485">AuthorizationServer</st> 类管理应用程序的认证请求和响应。它提供了适合应用程序强制执行的认证授予的不同类型的端点。现在，实例化该类是构建客户端或其他应用程序的 OAuth2 授权服务器的第一步。它需要 <st c="60830">query_client()</st> 和 <st c="60849">save_token()</st> 来进行令牌生成和授权机制的授权类型。

            <st c="60937">Authlib</st> <st c="60946">提供</st> `<st c="60959">ResourceOwnerPasswordCredentialsGrant</st>` <st c="60996">类以实现</st> `<st c="61133">authenticate_user()</st>` <st c="61152">在执行 <st c="61197">query_client()</st> 和 <st c="61211">以及 <st c="61216">save_token()</st> <st c="61228">方法</st> 之前进行验证。</st> <st c="61238">以下代码片段显示了</st> `<st c="61270">ResourceOwnerPasswordCredentialsGrant</st>` <st c="61307">子类，它是我们</st> `<st c="61324">ch09-oauth2-password</st>` <st c="61344">项目的：</st>

```py
<st c="61353">from authlib.oauth2.rfc6749.grants import</st> <st c="61395">ResourceOwnerPasswordCredentialsGrant</st> class PasswordGrant(<st c="61454">ResourceOwnerPasswordCredentialsGrant</st>): <st c="61496">TOKEN_ENDPOINT_AUTH_METHODS</st> = [
      'client_secret_basic', 'client_secret_post' ]
    def authenticate_user(self, <st c="61602">username</st>, <st c="61612">password</st>):
         task = get_user_task_wrapper.apply_async(args=[username])
         login:Login = task.get()
         if login is not None and check_password_hash( login.password, password) == True: <st c="61805">PasswordGrant</st> custom class is in the <st c="61842">/modules/security/oauth2_config.py</st> module with the <st c="61893">query_client()</st> and <st c="61912">save_token()</st> authorization server methods. The first component of <st c="61978">PasswordGrant</st> to configure is its <st c="62012">TOKEN_ENDPOINT_AUTH_METHODS</st>, which, from its default <st c="62065">public</st> value, needs to be set to <st c="62098">client_secret_basic</st>, <st c="62119">client_secret_post</st>, or both. The <st c="62152">client_secret_basic</st> is a client authentication that passes client secrets through a basic authentication scheme, while <st c="62271">client_secret_post</st> utilizes form parameters to pass client secrets to the authorization server. On the other hand, the overridden <st c="62401">authenticate_user()</st> retrieves the <st c="62435">username</st> and <st c="62448">password</st> from the token generator endpoint through basic authentication or form submission. It also retrieves the <st c="62562">Login</st> record object from the database through a <st c="62610">get_user_task_wrapper()</st> Celery task and validates the <st c="62664">Login</st>’s hashed password with the retrieved password from the client. The method returns the <st c="62757">Login</st> object that will signal the execution of the <st c="62808">query_client()</st> method. The following snippet shows our <st c="62863">query_client()</st> implementation:

```

def query_client(<st c="62911">client_id</st>):

    task = get_client_task_wrapper.apply_async(args=[<st c="62975">client_id</st>])

    client:Client = task.get()

    return client

```py

			<st c="63029">The</st> `<st c="63034">query_client()</st>` <st c="63048">is a necessary method of the</st> `<st c="63078">AuthorizationServer</st>` <st c="63097">instance.</st> <st c="63108">Its goal is to find the client who requested the authentication and return the</st> `<st c="63187">Client</st>` <st c="63193">object.</st> <st c="63202">It retrieves</st> <st c="63215">the</st> `<st c="63219">client_id</st>` <st c="63228">from the</st> `<st c="63238">AuthorizationServer</st>` <st c="63258">endpoint and uses it to search for the</st> `<st c="63297">Client</st>` <st c="63303">object from the database.</st> <st c="63330">The following snippet shows how to build the</st> `<st c="63375">Client</st>` <st c="63381">blueprint with</st> `<st c="63397">Authlib</st>`<st c="63404">’s</st> `<st c="63408">OAuth2ClientMixin</st>`<st c="63425">:</st>

```

<st c="63427">from authlib.integrations.sqla_oauth2 import OAuth2ClientMixin</st> class Client(Base, <st c="63509">OAuth2ClientMixin</st>):

    __tablename__ = 'oauth2_client'

    id = Column(Integer, Sequence('oauth2_client_id_seq', increment=1), primary_key = True)

    user_id = Column(String(20), ForeignKey('login.username'), unique=True)

    login = relationship('Login', back_populates="client")

    … … … … … …

```py

			<st c="63788">Authlib’s</st> `<st c="63799">OAuth2ClientMixin</st>` <st c="63816">will pad all the necessary column fields to the model class, including those that are optional.</st> <st c="63913">The required pre-tokenization fields, such as</st> `<st c="63959">id</st>`<st c="63961">,</st> `<st c="63963">user_id</st>` <st c="63970">or</st> `<st c="63974">username</st>`<st c="63982">,</st> `<st c="63984">client_id</st>`<st c="63993">,</st> `<st c="63995">client_id_issued_at</st>`<st c="64014">, and</st> `<st c="64020">client_secret</st>`<st c="64033">, must be submitted to the database during client signup before the authentication starts.</st> <st c="64124">Now, if the client is valid, the</st> `<st c="64157">save_token()</st>` <st c="64169">will execute to retrieve the</st> `<st c="64199">access_token</st>` <st c="64211">from the authorization server and save it to the database.</st> <st c="64271">The following snippet is our</st> <st c="64299">implementation</st> <st c="64315">for</st> `<st c="64319">save_token()</st>`<st c="64331">:</st>

```

<st c="64333">from authlib.integrations.flask_oauth2.requests import FlaskOAuth2Request</st> def save_token(token_data, <st c="64434">request:FlaskOAuth2Request</st>):

    if request.user:

        user_id = request.user.user_id

    else:

        user_id = request.client.user_id

    token_dict = dict()

    token_dict['client_id'] = request.client.client_id

    token_dict['user_id'] = user_id

    token_dict['issued_at'] = request.client.client_id_issued_at

    token_dict['access_token_revoked_at'] = 0

    token_dict['refresh_token_revoked_at'] = 0

    token_dict['scope'] = request.client.client_metadata["scope"]

    token_dict.update(token_data)

    token_str = dumps(token_dict) <st c="64998">token_data</st> 包含了 <st c="65022">access_token</st>，并且请求包含了从 <st c="65091">query_client()</st> 获取的 <st c="65060">Client</st> 数据。该方法将这些详细信息合并到一个 <st c="65152">token_dict</st> 中，使用 <st c="65198">token_dict</st> 作为参数实例化 <st c="65181">Token</st> 类，并将对象记录存储在数据库中。以下是 <st c="65307">Token</st> 模型的蓝图：

```py
<st c="65319">from authlib.integrations.sqla_oauth2 import OAuth2TokenMixin</st> class Token(Base, <st c="65400">OAuth2TokenMixin</st>):
    __tablename__ = 'oauth2_token'
    id = Column(Integer, Sequence('oauth2_token_id_seq', increment=1), primary_key=True)
    user_id = Column(String(40), ForeignKey('login.username'), nullable=False)
    login = relationship('Login', back_populates="token")
    … … … … … …
```

            <st c="65676">The</st> `<st c="65681">OAuth2TokenMixin</st>` <st c="65697">pads the</st> `<st c="65707">Token</st>` <st c="65712">class with the attributes related to</st> `<st c="65750">access_token</st>`<st c="65762">, such as</st> `<st c="65772">id,</st>` `<st c="65775">user_id</st>`<st c="65783">,</st> `<st c="65785">client_id</st>`<st c="65794">,</st> `<st c="65796">token_type</st>`<st c="65806">,</st> `<st c="65808">refresh_token</st>`<st c="65821">, and</st> `<st c="65827">scope</st>`<st c="65832">. By the way,</st> `<st c="65846">scope</st>` <st c="65851">is a mandatory field in Authlib that restricts access to the API resources based on some access level</st> <st c="65954">or role.</st>

            <st c="65962">To trigger the authorization</st> <st c="65992">server, the client must access the</st> `<st c="66027">/oauth/token</st>` <st c="66039">endpoint through basic authentication or form-based transactions.</st> <st c="66106">The following code shows the endpoint implementation of</st> <st c="66162">our application:</st>

```py
 from flask import current_app, request <st c="66218">from modules import oauth_server</st> @current_app.route('/oauth/token', <st c="66286">methods=['POST']</st>)
async def issue_token(): <st c="66443">POST</st> transaction mode. The <st c="66470">Authorization</st> <st c="66483">Server</st> object from the <st c="66507">create_app()</st> provides the <st c="66533">create_token_response()</st> with details that the method needs to return for the user to capture the <st c="66630">access_token()</st>. Given the <st c="66656">client_id</st> of <st c="66669">Xd3LH9mveF524LOscPq4MzLY</st> and <st c="66698">client_secret</st> <st c="66711">t8w56Y9OBRsxdVV9vrNwdtMzQ8gY4hkKLKf4b6F6RQZlT2zI</st> with the <st c="66770">sjctrags</st> username, the following <st c="66803">curl</st> command shows how to run the <st c="66837">/</st><st c="66838">oauth/token</st> endpoint:

```

curl -u Xd3LH9mveF524LOscPq4MzLY:t8w56Y9OBRsxdVV9vrNwdtMzQ 8gY4hkKLKf4b6F6RQZlT2zI -XPOST http://localhost:5000/oauth/token -F grant_type=password -F username=sjctrags -F password=sjctrags -F scope=user_admin -F token_endpoint_auth_method=client_secret_basic

```py

			<st c="67118">A sample result of executing</st> <st c="67148">the preceding command will contain the following details aside from</st> <st c="67216">the</st> `<st c="67220">access_token</st>`<st c="67232">:</st>

```

{"access_token": "fVFyaS06ECKIKFVtIfVj3ykgjhQjtc80JwCKyTMlZ2", "expires_in": 864000, "scope": "user_admin", "token_type": "Bearer"}

```py

			<st c="67366">As indicated in the result, the</st> `<st c="67399">token_type</st>` <st c="67409">is</st> `<st c="67413">Bearer</st>`<st c="67419">, so we can use the</st> `<st c="67439">access_token</st>` <st c="67451">to access or run an API endpoint through a bearer Token authentication, like in the following</st> `<st c="67546">curl</st>` <st c="67550">command:</st>

```

curl -H "Authorization: Bearer fVFyaS06ECKIKFVtIfVj3y kgjhQjtc80JwCKyTMlZ2" http://localhost:5000/doctor/profile/add

```py

			<st c="67676">A secured API endpoint must have the</st> `<st c="67714">require_oauth("user_admin")</st>` <st c="67741">method decorator, wherein</st> `<st c="67768">require_oath</st>` <st c="67780">is the</st> `<st c="67788">ResourceProtector</st>` <st c="67805">instance from the</st> `<st c="67824">create_app()</st>`<st c="67836">. A sample secured endpoint is the following</st> `<st c="67881">add_doctor_profile()</st>` <st c="67901">API function:</st>

```

<st c="67915">from modules import require_oauth</st> @current_app.route('/doctor/profile/add', methods = ['GET', 'POST']) <st c="68019">@require_oauth("user_admin")</st> async def add_doctor_profile():

    … … … … … …

    async with db_session() as sess:

        async with sess.begin():

        repo = DoctorRepository(sess)

        doc = Doctor(**doctor_json)

        result = await repo.insert_doctor(doc)

        … … … … … …

        return jsonify(record=doctor_json), 200

```py

			<st c="68298">Aside from the Resource Owner Password grant, Authlib has an</st> `<st c="68360">AuthorizationCodeGrant</st>` <st c="68382">class to</st> <st c="68391">implement an</st> `<st c="68449">JWTBearerGrant</st>` <st c="68463">for implementing the</st> `<st c="68559">ch09-oauth-code</st>` <st c="68574">project will</st> <st c="68588">showcase the full implementation</st> <st c="68620">of the OAuth2 authorization code flow, while</st> `<st c="68666">ch09-oauth2-jwt</st>` <st c="68681">will implement the JWT authorization scheme (</st>`<st c="68749">pyjwt</st>` <st c="68754">module.</st>
			<st c="68762">If Flask supports popular and ultimate authentication and authorization modules, like Authlib, it also supports unpopular but reliable extension modules that can secure web-based and API-based Flask applications.</st> <st c="68976">One of</st> <st c="68983">these modules is</st> *<st c="69000">Flask-Limiter</st>*<st c="69013">, which can prevent</st> `<st c="69103">ch09-web-passphrase</st>` <st c="69122">project.</st>
			<st c="69131">Controlling the view or API access</st>
			<st c="69166">DoS attacks happen when a user maliciously accesses a web page or API multiple times to disrupt the traffic and make the</st> <st c="69287">resources inaccessible to others.</st> `<st c="69322">Flask-Limiter</st>` <st c="69335">can provide an immediate solution by managing the number of access of a user to an API endpoint.</st> <st c="69433">First, install the</st> `<st c="69452">Flask-Limiter</st>` <st c="69465">module using the following</st> `<st c="69493">pip</st>` <st c="69496">command:</st>

```

pip install flask-limiter

```py

			<st c="69531">Also, install the module dependency for caching its configuration details to the</st> <st c="69613">Redis server:</st>

```

pip install flask-limiter[redis]

```py

			<st c="69659">Now, we can set up the module’s</st> `<st c="69692">Limiter</st>` <st c="69699">class in the</st> `<st c="69713">create_app()</st>` <st c="69725">factory method, like in the</st> <st c="69754">following snippet:</st>

```

<st c="69772">from flask_limiter import Limiter</st> from flask_limiter.util import get_remote_address

def create_app(config_file):

    app = Flask(__name__,template_folder= '../modules/pages', static_folder= '../modules/resources')

    … … … … … …

    global limiter

    limiter = <st c="70020">Limiter</st>(

    app=app, key_func=get_remote_address,

    default_limits=["30 per day", "5 per hour"],

    storage_uri="memory://", )

```py

			<st c="70139">Instantiating the</st> `<st c="70158">Limiter</st>` <st c="70165">class requires at least the</st> `<st c="70194">app</st>` <st c="70197">instance, the host of the application through the</st> `<st c="70248">get_remote_address()</st>`<st c="70268">, the</st> `<st c="70274">default_limits</st>` <st c="70288">(e.g.,</st> `<st c="70296">10 per hour</st>`<st c="70307">,</st> `<st c="70309">10 per 2 hours</st>`<st c="70323">, or</st> `<st c="70328">10/hour</st>`<st c="70335">), and the storage URI for the Redis server.</st> <st c="70381">The</st> `<st c="70385">Limiter</st>` <st c="70392">instance will</st> <st c="70407">provide each protected API with the</st> `<st c="70443">limit()</st>` <st c="70450">decorator that specifies the number of accesses not lower than the set default limit.</st> <st c="70537">The following API is restricted not to be accessed by a user more than a</st> *<st c="70610">maximum count of 5 times</st>* *<st c="70635">per minute</st>*<st c="70645">:</st>

```

<st c="70647">from modules import limiter</st> @current_app.route('/login/auth', methods=['GET', 'POST']) <st c="70734">@limiter.limit("5 per minute")</st> async def login_user():

    if request.method == 'GET':

        return render_template('login/authenticate.html'), 200

    username = request.form['username'].strip()

    password = request.form['password'].strip()

    async with db_session() as sess:

        async with sess.begin():

            repo = LoginRepository(sess)

            … … … … … …

                return render_template('login/authenticate.html'), 200

```py

			<st c="71115">Running</st> `<st c="71124">login_user()</st>` <st c="71136">more than the</st> <st c="71151">limit will give us the message shown in</st> *<st c="71191">Figure 9</st>**<st c="71199">.5</st>*<st c="71201">.</st>
			![Figure 9.5 – Accessing /login/auth more than the limit](img/B19383_09_005.jpg)

			<st c="71287">Figure 9.5 – Accessing /login/auth more than the limit</st>
			<st c="71341">Violating the number of access rules set by Talisman will lead users to its built-in error handling mechanism: the application rendering an error page with its</st> <st c="71502">error message.</st>
			<st c="71516">Summary</st>
			<st c="71524">In this chapter, we learned that compared to FastAPI and Tornado, there is quite a list of extension modules that provide solutions to secure a Flask application against various attacks.</st> <st c="71712">For instance, Flask-Seasurf and Flask-WTF can help minimize CSRF attacks.</st> <st c="71786">When pursuing web authentication, Flask-Login can provide a reliable authentication mechanism with added password hashing and encryption mechanisms, as we learned in</st> <st c="71952">this chapter.</st>
			<st c="71965">On the other hand, Flask-HTTPAuth can provide API-based applications with HTTP basic, digest, and bearer token authentication schemes.</st> <st c="72101">We learned that OAuth2 Authorization server grants and OAuth2 JWT Token-based types can also protect Flask applications from other</st> <st c="72232">applications’ access.</st>
			<st c="72253">The Flask-Talisman ensures security rules on response headers to filter the outgoing response of every API endpoint.</st> <st c="72371">Meanwhile, the Flask-Session module saves Flask sessions in the filesystem to avoid browser-based attacks.</st> <st c="72478">Escaping, stripping of whitespaces, and form validation of incoming inputs using modules like Gladiator and Flask-WTF helps prevent injection attacks by eliminating suspicious text or alphanumerics in</st> <st c="72679">the inputs.</st>
			<st c="72690">This chapter proved that several updated and version-compatible modules can help protect our applications from malicious and unwanted attacks.</st> <st c="72834">These modules can save time and effort compared to ground-up solutions in securing</st> <st c="72917">our applications.</st>
			<st c="72934">The next chapter will be about testing Flask components before running and deploying them to</st> <st c="73028">production servers.</st>

```

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

```py

```

# <st c="0">第三部分：测试、部署和构建企业级应用程序</st>

<st c="70">在本节的最后部分，您将学习一些测试、部署和运行我们的 Flask 3 应用程序的选择和解决方案。</st> <st c="185">此外，您还将了解如何使用互操作性功能将 Flask 应用程序集成到 GraphQL、React 表单、Flutter 移动应用程序以及其他使用 FastAPI、Django、Tornado 和 Flask 构建的应用程序中。</st> <st c="402">。</st>

<st c="427">本部分包括以下章节：</st> <st c="451">：</st>

+   *<st c="470">第十章</st>*<st c="481">,</st> *<st c="483">为 Flask 创建测试用例</st>*

+   *<st c="512">第十一章</st>*<st c="523">,</st> *<st c="525">部署 Flask 应用程序</st>*

+   *<st c="553">第十二章</st>*<st c="564">,</st> *<st c="566">将 Flask 集成到其他工具和框架中</st>*
