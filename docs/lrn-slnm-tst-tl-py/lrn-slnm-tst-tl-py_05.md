# 第五章：同步测试

构建稳健和可靠的测试是自动化 UI 测试的关键成功因素之一。然而，你可能会遇到测试条件因测试而异的情况。当你的脚本搜索元素或应用程序的某种状态，并且由于突然的资源限制或网络延迟导致应用程序开始缓慢响应，无法再找到这些元素时，测试会报告假阴性结果。我们需要通过在测试脚本中引入延迟来匹配测试脚本的速率与应用程序的速率。换句话说，我们需要将脚本与应用程序的响应同步。WebDriver 提供了隐式和显式等待来同步测试。

在本章中，你将学习以下主题：

+   使用隐式和显式等待

+   何时使用隐式和显式等待

+   使用预期条件

+   创建自定义等待条件

# 使用隐式等待

隐式等待提供了一个通用的方法来同步 WebDriver 中的整个测试或一系列步骤。隐式等待在处理由于网络速度或使用 Ajax 调用动态渲染元素的应用程序响应时间不一致的情况时非常有用。

当我们在 WebDriver 上设置隐式等待时，它会搜索 DOM 一段时间以查找元素或元素，如果它们当时不可用。默认情况下，隐式等待超时设置为`0`。

一旦设置，隐式等待将应用于 WebDriver 实例的生命周期或整个测试期间，WebDriver 将为所有在页面上查找元素的步骤应用此隐式等待，除非我们将它重新设置为`0`。

`webdriver`类提供了`implicitly_wait()`方法来配置超时。我们在第二章中创建了一个`SearchProductTest`测试，*使用 unittest 编写测试*。我们将修改这个测试，并在`setUp()`方法中添加一个 10 秒的超时隐式等待，如下面的代码示例所示。当测试执行时，如果 WebDriver 找不到元素，它将等待最多 10 秒。当达到超时，即本例中的 10 秒时，它将抛出`NoSuchElementException`。

```py
import unittest
from selenium import webdriver

class SearchProductTest(unittest.TestCase):
    def setUp(self):
        # create a new Firefox session
        self.driver = webdriver.Firefox()
        self.driver.implicitly_wait(30)
        self.driver.maximize_window()

        # navigate to the application home page
        self.driver.get("http://demo.magentocommerce.com/")

    def test_search_by_category(self):

        # get the search textbox
        self.search_field = self.driver.find_element_by_name("q")
        self.search_field.clear()

        # enter search keyword and submit
        self.search_field.send_keys("phones")
        self.search_field.submit()

        # get all the anchor elements which have product names # displayed currently on result page using # find_elements_by_xpath method
        products = self.driver\
            .find_elements_by_xpath("//h2[@class='product-name']/a")

        # check count of products shown in results
        self.assertEqual(2, len(products))

    def tearDown(self):
        # close the browser window
        self.driver.quit()

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

### 小贴士

在测试中最好避免使用隐式等待，并尝试使用显式等待来处理同步问题，与隐式等待相比，显式等待提供了更多的控制。

# 使用显式等待

显式等待是 WebDriver 中用于同步测试的另一种等待机制。与隐式等待相比，显式等待提供了更好的控制。与隐式等待不同，我们可以在脚本继续下一步之前使用一组预定义或自定义条件来等待。

显式等待只能在需要脚本同步的特定情况下实现。WebDriver 提供了`WebDriverWait`和`expected_conditions`类来实现显式等待。

`expected_conditions`类提供了一组预定义的条件，用于在代码中进一步执行之前等待。

让我们创建一个简单的测试，使用预期条件显式等待一个元素的可见性，如下面的代码所示：

```py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
import unittest

class ExplicitWaitTests(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
        self.driver.get("http://demo.magentocommerce.com/")

    def test_account_link(self):
        WebDriverWait(self.driver, 10)\
            .until(lambda s: s.find_element_by_id("select-language").get_attribute("length") == "3")

        account = WebDriverWait(self.driver, 10)\
            .until(expected_conditions.visibility_of_element_located((By.LINK_TEXT, "ACCOUNT")))
        account.click()

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    unittest.main(verbosity=2)
```

在这个测试中，使用显式等待来等待直到**登录**链接在 DOM 中可见，使用预期的`visibility_of_element_located`条件。此条件需要我们想要等待的元素的定位策略和定位详情。脚本将等待最多 10 秒钟寻找可见的元素。一旦元素通过指定的定位器可见，预期的条件将返回定位到的元素回脚本。

如果在指定的超时时间内，元素通过指定的定位器不可见，将引发`TimeoutException`异常。

# 预期条件类

下表显示了我们在自动化由`expected_conditions`类支持的网页浏览器时经常遇到的常见条件及其示例：

| 预期条件 | 描述 | 参数 | 示例 |
| --- | --- | --- | --- |
| `element_to_be_clickable(locator)` | 这将等待一个元素被定位、可见并且可点击，以便可以点击。此方法将返回定位到的元素到测试中。 | `locator`: 这是一个`(by, locator)`的元组。 | `WebDriverWait(self.driver, 10).until(expected_conditions.element_to_be_clickable((By.NAME,"is_subscribed")))` |
| `element_to_be_selected(element)` | 这将等待直到指定的元素被选中。 | `element`: 这是一个 WebElement。 | `subscription = self.driver.find_element_by_name("is_subscribed")``WebDriverWait(self.driver, 10).until(expected_conditions.element_to_be_selected(subscription))` |
| `invisibility_of_element_located(locator)` | 这将等待一个元素要么不可见，要么不在 DOM 上。 | `locator`: 这是一个`(by, locator)`的元组。 | `WebDriverWait(self.driver, 10).until(expected_conditions.invisibility_of_element_located((By.ID,"loading_banner")))` |
| `presence_of_all_elements_located(locator)` | 这将等待直到至少一个匹配定位器的元素出现在网页上。此方法在元素被定位后返回 WebElements 列表。 | `locator`: 这是一个`(by, locator)`的元组。 | `WebDriverWait(self.driver, 10).until(expected_conditions.presence_of_all_elements_located((By.CLASS_NAME,"input-text")))` |
| `presence_of_element_located(locator)` | 这将等待直到匹配定位器的元素出现在网页上或在 DOM 上可用。此方法在元素被定位后返回一个元素。 | `locator`: 这是一个`(by, locator)`的元组。 | `WebDriverWait(self.driver, 10).until(expected_conditions.presence_of_element_located((By.ID,"search")))` |
| `text_to_be_present_in_element(locator, text_)` | 这将等待直到找到元素并且具有给定的文本。 | `locator`: 这是一个 `(by, locator)` 的元组。`text`: 这是需要检查的文本。 | `WebDriverWait(self.driver,10).until(expected_conditions.text_to_be_present_in_element((By.ID,"select-language"),"English"))` |
| `title_contains(title)` | 这将等待页面标题包含一个大小写敏感的子串。此方法如果标题匹配则返回 `true`，否则返回 `false`。 | `title`: 这是需要检查的标题子串。 | `WebDriverWait(self.driver, 10).until(expected_conditions.title_contains("Create New Customer Account"))` |
| `title_is(title)` | 这将等待页面标题等于预期的标题。此方法如果标题匹配则返回 `true`，否则返回 `false`。 | `title`: 这是页面的标题。 | `WebDriverWait(self.driver, 10).until(expected_conditions.title_is("Create New Customer Account - Magento Commerce Demo Store"))` |
| `visibility_of(element)` | 这将等待直到元素在 DOM 中存在，可见，并且其宽度和高度都大于零。此方法在元素变为可见时返回（相同的）WebElement。 | `element`: 这是 WebElement。 | `first_name = self.driver.find_element_by_id("firstname")` `WebDriverWait(self.driver, 10).until(expected_conditions.visibility_of(first_name))` |
| `visibility_of_element_located(locator)` | 这将等待直到要定位的元素在 DOM 中存在，可见，并且其宽度和高度都大于零。此方法在元素变为可见时返回 WebElement。 | `locator`: 这是一个 `(by, locator)` 的元组。 | `WebDriverWait(self.driver, 10).until(expected_conditions.visibility_of_element_located((By.ID,"firstname")))` |

你可以在[`selenium.googlecode.com/git/docs/api/py/webdriver_support/selenium.webdriver.support.expected_conditions.html#module-selenium.webdriver.support.expected_conditions`](http://selenium.googlecode.com/git/docs/api/py/webdriver_support/selenium.webdriver.support.expected_conditions.html#module-selenium.webdriver.support.expected_conditions)找到预期条件的完整列表。

让我们在接下来的部分中探索更多预期条件的示例。

## 等待元素变为可用

如我们之前所见，`expected_conditions` 类提供了各种等待条件，我们可以在脚本中实现。在下面的示例中，我们将等待一个元素变为可用或可点击。我们可以在基于其他表单字段值或筛选器的表单字段启用或禁用的 Ajax 重量级应用程序中使用此条件。在这个例子中，我们点击 **登录** 链接，然后等待登录页面上的 **创建账户** 按钮变为可点击，该按钮在登录页面上显示。然后我们将点击 **创建账户** 按钮，并等待显示下一页。

```py
def test_create_new_customer(self):
    # click on Log In link to open Login page
    self.driver.find_element_by_link_text("ACCOUNT").click()

    # wait for My Account link in Menu
    my_account = WebDriverWait(self.driver, 10)\
        .until(expected_conditions.visibility_of_element_located((By.LINK_TEXT, "My Account")))
    my_account.click()

    # get the Create Account button
    create_account_button = WebDriverWait(self.driver, 10)\
        .until(expected_conditions.element_to_be_clickable((By.LINK_TEXT, "CREATE AN ACCOUNT")))

    # click on Create Account button. This will displayed new account
    create_account_button.click()
    WebDriverWait(self.driver, 10)\
        .until(expected_conditions.title_contains("Create New Customer Account"))
```

我们可以使用`element_to_be_clickable`条件等待并检查元素是否可点击。这需要定位策略和定位值。当该元素变为可点击或换句话说变为启用时，它将返回定位的元素到脚本中。

前面的测试还通过检查指定的文本标题来等待创建新客户账户页面加载。我们使用了`title_contains`条件，该条件确保子字符串与页面标题匹配。

## 等待警报

我们还可以在警报和框架上使用显式等待。复杂的 JavaScript 处理或后端请求可能需要时间来向用户显示警报。这可以通过以下方式处理预期的`alert_is_present`条件：

```py
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
import unittest

class CompareProducts(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
        self.driver.get("http://demo.magentocommerce.com/")

    def test_compare_products_removal_alert(self):
        # get the search textbox
        search_field = self.driver.find_element_by_name("q")
        search_field.clear()

        # enter search keyword and submit
        search_field.send_keys("phones")
        search_field.submit()

        # click the Add to compare link
        self.driver.\
            find_element_by_link_text("Add to Compare").click()

        # wait for Clear All link to be visible
        clear_all_link = WebDriverWait(self.driver, 10)\
            .until(expected_conditions.visibility_of_element_located((By.LINK_TEXT, "Clear All")))

        # click on Clear All link,
        # this will display an alert to the user
        clear_all_link.click()

        # wait for the alert to present
        alert = WebDriverWait(self.driver, 10)\
            .until(expected_conditions.alert_is_present())

        # get the text from alert
        alert_text = alert.text

        # check alert text
        self.assertEqual("Are you sure you would like
  to remove all products from your comparison?", alert_text)
        # click on Ok button
        alert.accept()

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    unittest.main(verbosity=2)
```

前面的测试验证了从应用程序的产品比较功能中删除产品。当用户从比较中删除产品时，会向用户发送确认警报。使用`alert_is_present`条件来检查警报是否显示给用户，并将其返回到脚本以进行后续操作。脚本将等待 10 秒钟检查警报的存在，否则将引发异常。

# 实现自定义等待条件

如我们之前所见，`expected_conditions`类提供了各种预定义的条件以供等待。我们还可以使用`WebDriverWait`构建自定义条件。当没有合适的预期条件可用时，这非常有用。

让我们修改本章早期创建的一个测试，并实现一个自定义等待条件来检查下拉列表项的数量：

```py
def testLoginLink(self):
    WebDriverWait(self.driver, 10).until(lambda s: s.find_element_by_id "select-language").get_attribute("length") == "3")

    login_link = WebDriverWait(self.driver, 10).until(expected_conditions.visibility_of_element_located((By.LINK_TEXT,"Log In")))
       login_link.click();
```

我们可以使用 Python lambda 表达式通过`WebDriverWait`实现自定义等待条件。在这个例子中，脚本将等待 10 秒钟，直到**选择语言**下拉列表有八个选项可供选择。这个条件在下拉列表通过 Ajax 调用填充，并且脚本需要等待直到所有选项都可供用户选择时非常有用。

# 摘要

在本章中，我们认识到同步的需要及其在构建高度可靠的测试中的重要性。我们探讨了隐式等待以及如何使用隐式等待作为一个通用的等待机制。然后我们探讨了提供更灵活同步测试方式的显式等待。`expected_conditions`类提供了各种内置的等待条件。我们已经实现了一些这些条件。

`WebDriverWait`类还提供了一种非常强大的方式来在`expected_conditions`之上实现自定义等待条件。我们在下拉列表上实现了一个自定义等待条件。

在下一章中，你将学习如何使用`RemoteWebDriver`和 Selenium Server 实现跨浏览器测试，在远程机器上运行测试，并使用 Selenium Grid 进行并行执行。
