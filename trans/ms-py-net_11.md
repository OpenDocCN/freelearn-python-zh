# 使用Git

我们已经使用Python、Ansible和许多其他工具在网络自动化的各个方面进行了工作。如果您一直在阅读本书的前九章的示例，我们已经使用了超过150个文件，其中包含超过5300行代码。对于可能主要使用命令行界面的网络工程师来说，这是相当不错的！有了我们的新一套脚本和工具，我们现在准备好去征服我们的网络任务了，对吗？嗯，我的同行网络忍者们，不要那么快。

我们面对的第一个任务是如何将代码文件保存在一个位置，以便我们和其他人可以检索和使用。理想情况下，这个位置应该是保存文件的最新版本的唯一位置。在初始发布之后，我们可能会在未来添加功能和修复错误，因此我们希望有一种方式来跟踪这些更改并保持最新版本可供下载。如果新的更改不起作用，我们希望回滚更改并反映文件历史中的差异。这将给我们一个关于代码文件演变的良好概念。

第二个问题是我们团队成员之间的协作过程。如果我们与其他网络工程师合作，我们将需要共同在文件上工作。这些文件可以是Python脚本、Ansible Playbook、Jinja2模板、INI风格的配置文件等等。关键是任何一种基于文本的文件都应该被多方输入跟踪，以便团队中的每个人都能看到。

第三个问题是责任制。一旦我们有了一个允许多方输入和更改的系统，我们需要用适当的记录来标记这些更改，以反映更改的所有者。记录还应包括更改的简要原因，以便审查历史的人能够理解更改的原因。

这些是版本控制（或源代码控制）系统试图解决的一些主要挑战。公平地说，版本控制可以存在于专用系统以外的形式。例如，如果我打开我的Microsoft Word程序，文件会不断保存自身，并且我可以回到过去查看更改或回滚到以前的版本。我们在这里关注的版本控制系统是具有主要目的跟踪软件更改的独立软件工具。

在软件工程中，有各种不同的源代码控制工具，既有专有的也有开源的。一些更受欢迎的开源版本控制系统包括CVS、SVN、Mercurial和Git。在本章中，我们将专注于源代码控制系统**Git**，这是我们在本书中使用的许多`.software`软件包中下载的工具。我们将更深入地了解这个工具。Git是许多大型开源项目的事实上的版本控制系统，包括Python和Linux内核。

截至2017年2月，CPython开发过程已经转移到GitHub。自2015年1月以来一直在进行中。有关更多信息，请查看[https://www.python.org/dev/peps/pep-0512/](https://www.python.org/dev/peps/pep-0512/)上的PEP 512。

在我们深入了解Git的工作示例之前，让我们先来看看Git系统的历史和优势。

# Git简介

Git是由Linux内核的创造者Linus Torvalds于2005年4月创建的。他幽默地称这个工具为“来自地狱的信息管理者”。在Linux基金会的一次采访中，Linus提到他觉得源代码控制管理在计算世界中几乎是最不有趣的事情。然而，在Linux内核开发社区和当时他们使用的专有系统BitKeeper之间发生分歧后，他还是创建了这个工具。

Git这个名字代表什么？在英国俚语中，Git是一个侮辱性词语，表示一个令人不愉快、恼人、幼稚的人。Linus以他的幽默说他是一个自负的混蛋，所以他把所有的项目都以自己的名字命名。首先是Linux，现在是Git。然而，也有人建议这个名字是**全球信息跟踪器**（**GIT**）的缩写。你可以做出判断。

这个项目很快就成形了。在创建后大约十天（没错，你没看错），Linus觉得Git的基本理念是正确的，开始用Git提交第一个Linux内核代码。其余的，就像他们说的那样，就成了历史。在创建十多年后，它仍然满足Linux内核项目的所有期望。尽管切换源代码控制系统存在固有的惯性，它已经成为许多其他开源项目的版本控制系统。在多年托管Python代码后，该项目于2017年2月在GitHub上切换到Git。

# Git的好处

像Linux内核和Python这样的大型分布式开源项目的成功托管，证明了Git的优势。这尤其重要，因为Git是一个相对较新的源代码控制工具，人们不倾向于切换到新工具，除非它比旧工具有显著的优势。让我们看看Git的一些好处：

+   **分布式开发**：Git支持在私人仓库中进行并行、独立和同时的离线开发。与其他一些版本控制系统需要与中央仓库进行不断同步相比，这为开发人员提供了更大的灵活性。

+   **扩展以处理成千上万的开发人员**：许多开源项目的开发人员数量达到了成千上万。Git支持可靠地集成他们的工作。

+   **性能**：Linus决心确保Git快速高效。为了节省空间和传输时间，仅Linux内核代码的更新量就需要压缩和增量检查来使Git快速高效。

+   **责任和不可变性**：Git强制在每次更改文件的提交时记录更改日志，以便对所有更改和更改原因进行跟踪。Git中的数据对象在创建并放入数据库后无法修改，使它们不可变。这进一步强化了责任。

+   **原子事务**：确保仓库的完整性，不同但相关的更改要么一起执行，要么不执行。这将确保仓库不会处于部分更改或损坏的状态。

+   **完整的仓库**：每个仓库都有每个文件的所有历史修订版本的完整副本。

+   **自由，就像自由**：Git工具的起源源于Linux内核的免费版本与BitKeeper VCS之间的分歧，因此这个工具有一个非常自由的使用许可证。

让我们来看看Git中使用的一些术语。

# Git术语

以下是一些我们应该熟悉的Git术语：

+   **Ref**：以`refs`开头指向对象的名称。

+   **存储库**：包含项目所有信息、文件、元数据和历史记录的数据库。它包含了所有对象集合的`ref`。

+   **分支**：活跃的开发线。最近的提交是该分支的`tip`或`HEAD`。存储库可以有多个分支，但您的`工作树`或`工作目录`只能与一个分支关联。有时这被称为当前或`checked out`分支。

+   **检出**：将工作树的全部或部分更新到特定点的操作。

+   **提交**：Git历史中的一个时间点，或者可以表示将新的快照存储到存储库中。

+   **合并**：将另一个分支的内容合并到当前分支的操作。例如，我正在将`development`分支与`master`分支合并。

+   **获取**：从远程存储库获取内容的操作。

+   **拉取**：获取并合并存储库的内容。

+   **标签**：存储库中某个时间点的标记。在[第4章](2784e1ec-c5d2-4b04-9e57-7db3caf0e310.xhtml)中，*Python自动化框架- Ansible基础*，我们看到标签用于指定发布点，`v2.5.0a1`。

这不是一个完整的列表；请参考Git术语表，[https://git-scm.com/docs/gitglossary](https://git-scm.com/docs/gitglossary)，了解更多术语及其定义。

# Git和GitHub

Git和GitHub并不是同一回事。对于新手来说，这有时会让工程师感到困惑。Git是一个版本控制系统，而GitHub，[https://github.com/](https://github.com/)，是Git存储库的集中式托管服务。

因为Git是一个分散的系统，GitHub存储了我们项目的存储库的副本，就像其他任何开发人员一样。通常，我们将GitHub存储库指定为项目的中央存储库，所有其他开发人员将其更改推送到该存储库，并从该存储库拉取更改。

GitHub通过使用`fork`和`pull requests`机制，进一步将这个在分布式系统中的集中存储库的概念发扬光大。对于托管在GitHub上的项目，鼓励开发人员`fork`存储库，或者复制存储库，并在该复制品上工作作为他们的集中存储库。在做出更改后，他们可以向主项目发送`pull request`，项目维护人员可以审查更改，并在适当的情况下`commit`更改。GitHub还除了命令行之外，还为存储库添加了Web界面；这使得Git更加用户友好。

# 设置Git

到目前为止，我们只是使用Git从GitHub下载文件。在本节中，我们将进一步设置Git变量，以便开始提交我们的文件。我将在示例中使用相同的Ubuntu 16.04主机。安装过程有很好的文档记录；如果您使用的是不同版本的Linux或其他操作系统，快速搜索应该能找到正确的指令集。

如果您还没有这样做，请通过`apt`软件包管理工具安装Git：

```py
$ sudo apt-get update
$ sudo apt-get install -y git
$ git --version
git version 2.7.4
```

安装了`git`之后，我们需要配置一些东西，以便我们的提交消息可以包含正确的信息：

```py
$ git config --global user.name "Your Name"
$ git config --global user.email "email@domain.com"
$ git config --list
user.name=Your Name
user.email=email@domain.com
```

或者，您可以修改`~/.gitconfig`文件中的信息：

```py
$ cat ~/.gitconfig
[user]
 name = Your Name
 email = email@domain.com
```

Git中还有许多其他选项可以更改，但是名称和电子邮件是允许我们提交更改而不会收到警告的选项。个人而言，我喜欢使用VIM，而不是默认的Emac，作为我的文本编辑器来输入提交消息：

```py
(optional)
$ git config --global core.editor "vim"
$ git config --list
user.name=Your Name
user.email=email@domain.com
core.editor=vim
```

在我们继续使用Git之前，让我们先了解一下`gitignore`文件的概念。

# Gitignore

有时，有些文件您不希望Git检查到GitHub或其他存储库中。这样做的最简单方法是在`repository`文件夹中创建`.gitignore`；Git将使用它来确定在进行提交之前应该忽略哪些文件。这个文件应该提交到存储库中，以便与其他用户共享忽略规则。

这个文件可以包括特定于语言的文件，例如，让我们排除Python的`Byte-compiled`文件：

```py
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
```

我们还可以包括特定于您的操作系统的文件：

```py
# OSX
# =========================

.DS_Store
.AppleDouble
.LSOverride
```

您可以在GitHub的帮助页面上了解更多关于`.gitignore`的信息：[https://help.github.com/articles/ignoring-files/](https://help.github.com/articles/ignoring-files/)。以下是一些其他参考资料：

+   Gitignore手册：[https://git-scm.com/docs/gitignore](https://git-scm.com/docs/gitignore)

+   GitHub的`.gitignore`模板集合：[https://github.com/github/gitignore](https://github.com/github/gitignore)

+   Python语言`.gitignore`示例：[https://github.com/github/gitignore/blob/master/Python.gitignore](https://github.com/github/gitignore/blob/master/Python.gitignore)

+   本书存储库的`.gitignore`文件：[https://github.com/PacktPublishing/Mastering-Python-Networking-Second-Edition/blob/master/.gitignore](https://github.com/PacktPublishing/Mastering-Python-Networking-Second-Edition/blob/master/.gitignore)

我认为`.gitignore`文件应该与任何新存储库同时创建。这就是为什么这个概念尽早被引入的原因。我们将在下一节中看一些Git使用示例。

# Git使用示例

大多数时候，当我们使用Git时，我们会使用命令行：

```py
$ git --help
usage: git [--version] [--help] [-C <path>] [-c name=value]
 [--exec-path[=<path>]] [--html-path] [--man-path] [--info-path]
 [-p | --paginate | --no-pager] [--no-replace-objects] [--bare]
 [--git-dir=<path>] [--work-tree=<path>] [--namespace=<name>]
 <command> [<args>]
```

我们将创建一个`repository`并在其中创建一个文件：

```py
$ mkdir TestRepo
$ cd TestRepo/
$ git init
Initialized empty Git repository in /home/echou/Master_Python_Networking_second_edition/Chapter11/TestRepo/.git/
$ echo "this is my test file" > myFile.txt
```

当使用Git初始化存储库时，会在目录中添加一个新的隐藏文件夹`.git`。它包含所有与Git相关的文件：

```py
$ ls -a
. .. .git myFile.txt

$ ls .git/
branches config description HEAD hooks info objects refs
```

Git接收其配置的位置有几个层次结构。您可以使用`git config -l`命令来查看聚合配置：

```py
$ ls .git/config
.git/config

$ ls ~/.gitconfig
/home/echou/.gitconfig

$ git config -l
user.name=Eric Chou
user.email=<email>
core.editor=vim
core.repositoryformatversion=0
core.filemode=true
core.bare=false
core.logallrefupdates=true
```

当我们在存储库中创建一个文件时，它不会被跟踪。为了让`git`意识到这个文件，我们需要添加这个文件：

```py
$ git status
On branch master

Initial commit

Untracked files:
 (use "git add <file>..." to include in what will be committed)

 myFile.txt

nothing added to commit but untracked files present (use "git add" to track)

$ git add myFile.txt
$ git status
On branch master

Initial commit

Changes to be committed:
 (use "git rm --cached <file>..." to unstage)

 new file: myFile.txt
```

当您添加文件时，它处于暂存状态。为了使更改生效，我们需要提交更改：

```py
$ git commit -m "adding myFile.txt"
[master (root-commit) 5f579ab] adding myFile.txt
 1 file changed, 1 insertion(+)
 create mode 100644 myFile.txt

$ git status
On branch master
nothing to commit, working directory clean
```

在上一个示例中，我们在发出提交语句时使用了`-m`选项来提供提交消息。如果我们没有使用该选项，我们将被带到一个页面上来提供提交消息。在我们的情况下，我们配置了文本编辑器为vim，因此我们将能够使用vim来编辑消息。

让我们对文件进行一些更改并提交它：

```py
$ vim myFile.txt
$ cat myFile.txt
this is the second iteration of my test file
$ git status
On branch master
Changes not staged for commit:
 (use "git add <file>..." to update what will be committed)
 (use "git checkout -- <file>..." to discard changes in working directory)

 modified: myFile.txt
$ git add myFile.txt
$ git commit -m "made modificaitons to myFile.txt"
[master a3dd3ea] made modificaitons to myFile.txt
 1 file changed, 1 insertion(+), 1 deletion(-)
```

`git commit`号是一个`SHA1哈希`，这是一个重要的特性。如果我们在另一台计算机上按照相同的步骤操作，我们的`SHA1哈希`值将是相同的。这就是Git知道这两个存储库在并行工作时是相同的方式。

我们可以使用`git log`来显示提交的历史记录。条目以相反的时间顺序显示；每个提交显示作者的姓名和电子邮件地址，日期，日志消息，以及提交的内部标识号：

```py
$ git log
commit a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 09:58:24 2018 -0700

 made modificaitons to myFile.txt

commit 5f579ab1e9a3fae13aa7f1b8092055213157524d
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 08:05:09 2018 -0700

 adding myFile.txt
```

我们还可以使用提交ID来显示更改的更多细节：

```py
$ git show a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
commit a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 09:58:24 2018 -0700

 made modificaitons to myFile.txt

diff --git a/myFile.txt b/myFile.txt
index 6ccb42e..69e7d47 100644
--- a/myFile.txt
+++ b/myFile.txt
@@ -1 +1 @@
-this is my test file
+this is the second iteration of my test file
```

如果您需要撤消所做的更改，您可以选择`revert`和`reset`之间。`revert`将特定提交的所有文件更改回到它们在提交之前的状态：

```py
$ git revert a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
[master 9818f29] Revert "made modificaitons to myFile.txt"
 1 file changed, 1 insertion(+), 1 deletion(-)

# Check to verified the file content was before the second change. 
$ cat myFile.txt
this is my test file
```

`revert`命令将保留您撤消的提交并创建一个新的提交。您将能够看到到那一点的所有更改，包括撤消：

```py
$ git log
commit 9818f298f477fd880db6cb87112b50edc392f7fa
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 13:11:30 2018 -0700

 Revert "made modificaitons to myFile.txt"

 This reverts commit a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038.

 modified: reverted the change to myFile.txt

commit a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 09:58:24 2018 -0700

 made modificaitons to myFile.txt

commit 5f579ab1e9a3fae13aa7f1b8092055213157524d
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 08:05:09 2018 -0700

 adding myFile.txt
```

`reset`选项将将存储库的状态重置为旧版本，并丢弃其中的所有更改：

```py
$ git reset --hard a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
HEAD is now at a3dd3ea made modificaitons to myFile.txt

$ git log
commit a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 09:58:24 2018 -0700

 made modificaitons to myFile.txt

commit 5f579ab1e9a3fae13aa7f1b8092055213157524d
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 08:05:09 2018 -0700

 adding myFile.txt
```

就个人而言，我喜欢保留所有历史记录，包括我所做的任何回滚。因此，当我需要回滚更改时，我通常选择`revert`而不是`reset`。

`git`中的`分支`是存储库内的开发线。Git允许在存储库内有许多分支和不同的开发线。默认情况下，我们有主分支。分支的原因有很多，但大多数代表单个客户发布或开发阶段，即`dev`分支。让我们在我们的存储库中创建一个`dev`分支：

```py
$ git branch dev
$ git branch
 dev
* master
```

要开始在分支上工作，我们需要`检出`该分支：

```py
$ git checkout dev
Switched to branch 'dev'
$ git branch
* dev
 master
```

让我们在`dev`分支中添加第二个文件：

```py
$ echo "my second file" > mySecondFile.txt
$ git add mySecondFile.txt
$ git commit -m "added mySecondFile.txt to dev branch"
[dev c983730] added mySecondFile.txt to dev branch
 1 file changed, 1 insertion(+)
 create mode 100644 mySecondFile.txt
```

我们可以回到`master`分支并验证两行开发是分开的：

```py
$ git branch
* dev
 master
$ git checkout master
Switched to branch 'master'
$ ls
myFile.txt
$ git checkout dev
Switched to branch 'dev'
$ ls
myFile.txt mySecondFile.txt
```

将`dev`分支中的内容写入`master`分支，我们需要将它们`合并`：

```py
$ git branch
* dev
 master
$ git checkout master
$ git merge dev master
Updating a3dd3ea..c983730
Fast-forward
 mySecondFile.txt | 1 +
 1 file changed, 1 insertion(+)
 create mode 100644 mySecondFile.txt
$ git branch
 dev
* master
$ ls
myFile.txt mySecondFile.txt
```

我们可以使用`git rm`来删除文件。让我们创建第三个文件并将其删除：

```py
$ touch myThirdFile.txt
$ git add myThirdFile.txt
$ git commit -m "adding myThirdFile.txt"
[master 2ec5f7d] adding myThirdFile.txt
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 myThirdFile.txt
$ ls
myFile.txt mySecondFile.txt myThirdFile.txt
$ git rm myThirdFile.txt
rm 'myThirdFile.txt'
$ git status
On branch master
Changes to be committed:
 (use "git reset HEAD <file>..." to unstage)

 deleted: myThirdFile.txt
$ git commit -m "deleted myThirdFile.txt"
[master bc078a9] deleted myThirdFile.txt
 1 file changed, 0 insertions(+), 0 deletions(-)
 delete mode 100644 myThirdFile.txt
```

我们将能够在日志中看到最后两次更改：

```py
$ git log
commit bc078a97e41d1614c1ba1f81f72acbcd95c0728c
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 14:02:02 2018 -0700

 deleted myThirdFile.txt

commit 2ec5f7d1a734b2cc74343ce45075917b79cc7293
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 14:01:18 2018 -0700

 adding myThirdFile.txt
```

我们已经了解了Git的大部分基本操作。让我们看看如何使用GitHub共享我们的存储库。

# GitHub示例

在这个例子中，我们将使用GitHub作为同步我们的本地存储库并与其他用户共享的集中位置。

我们将在GitHub上创建一个存储库。默认情况下，GitHub有一个免费的公共存储库；在我的情况下，我支付一个小额的月费来托管私人存储库。在创建时，您可以选择创建许可证和`.gitignore`文件：

![](assets/62b8ab5d-063e-438a-9705-88d7472f9f8f.png)

GitHub私人存储库

存储库创建后，我们可以找到该存储库的URL：

![](assets/f0203a9a-1927-4afc-9b44-492d918e31d4.png)

GitHub存储库URL

我们将使用此URL创建一个`远程`目标；我们将其命名为`gitHubRepo`：

```py
$ git remote add gitHubRepo https://github.com/ericchou1/TestRepo.git
$ git remote -v
gitHubRepo https://github.com/ericchou1/TestRepo.git (fetch)
gitHubRepo https://github.com/ericchou1/TestRepo.git (push)
```

由于我们选择在创建时创建`README.md`和`LICENSE`文件，远程存储库和当前存储库不同。如果我们将本地更改推送到GitHub存储库，将收到以下错误：

```py
$ git push gitHubRepo master
Username for 'https://github.com': echou@yahoo.com
Password for 'https://echou@yahoo.com@github.com':
To https://github.com/ericchou1/TestRepo.git
 ! [rejected] master -> master (fetch first)
```

我们将继续使用`git pull`从GitHub获取新文件：

```py
$ git pull gitHubRepo master
Username for 'https://github.com': <username>
Password for 'https://<username>@github.com':
From https://github.com/ericchou1/TestRepo
 * branch master -> FETCH_HEAD
Merge made by the 'recursive' strategy.
 .gitignore | 104 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 LICENSE | 21 +++++++++++++
 README.md | 2 ++
 3 files changed, 127 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 LICENSE
 create mode 100644 README.md
```

现在我们将能够将内容`推送`到GitHub：

```py
$ git push gitHubRepo master
Username for 'https://github.com': <username>
Password for 'https://<username>@github.com':
Counting objects: 15, done.
Compressing objects: 100% (9/9), done.
Writing objects: 100% (15/15), 1.51 KiB | 0 bytes/s, done.
Total 15 (delta 1), reused 0 (delta 0)
remote: Resolving deltas: 100% (1/1), done.
To https://github.com/ericchou1/TestRepo.git
 a001b81..0aa362a master -> master
```

我们可以在网页上验证GitHub存储库的内容：

![](assets/d7d01d22-b4a7-45e3-9230-faa30494b5af.png)

GitHub存储库

现在另一个用户可以简单地制作存储库的副本，或`克隆`：

```py
[This is operated from another host]
$ cd /tmp
$ git clone https://github.com/ericchou1/TestRepo.git
Cloning into 'TestRepo'...
remote: Counting objects: 20, done.
remote: Compressing objects: 100% (13/13), done.
remote: Total 20 (delta 2), reused 15 (delta 1), pack-reused 0
Unpacking objects: 100% (20/20), done.
$ cd TestRepo/
$ ls
LICENSE myFile.txt
README.md mySecondFile.txt
```

这个复制的存储库将是我原始存储库的精确副本，包括所有提交历史：

```py
$ git log
commit 0aa362a47782e7714ca946ba852f395083116ce5 (HEAD -> master, origin/master, origin/HEAD)
Merge: bc078a9 a001b81
Author: Eric Chou <echou@yahoo.com>
Date: Fri Jul 20 14:18:58 2018 -0700

 Merge branch 'master' of https://github.com/ericchou1/TestRepo

commit a001b816bb75c63237cbc93067dffcc573c05aa2
Author: Eric Chou <ericchou1@users.noreply.github.com>
Date: Fri Jul 20 14:16:30 2018 -0700

 Initial commit
...
```

我还可以在存储库设置下邀请另一个人作为项目的合作者：

![](assets/49c734fa-c413-4bc2-98fe-b392520dec52.png)

存储库邀请

在下一个例子中，我们将看到如何分叉存储库并为我们不维护的存储库发起拉取请求。

# 通过拉取请求进行协作

如前所述，Git支持开发人员之间的合作，用于单个项目。我们将看看当代码托管在GitHub上时是如何完成的。

在这种情况下，我将查看这本书的GitHub存储库。我将使用不同的GitHub句柄，所以我会以不同的用户身份出现。我将点击分叉按钮，在我的个人帐户中制作存储库的副本：

![](assets/5e9563c6-d5df-41b6-a780-797aa8e88afb.png)

Git分叉底部

制作副本需要几秒钟：

![](assets/7c4ffe85-465d-4874-a9aa-12e02fa89634.png)

Git正在进行分叉

分叉后，我们将在我们的个人帐户中拥有存储库的副本：

![](assets/6ec1babb-d96b-4b24-9dc5-e160acb2440f.png)

Git分叉

我们可以按照之前使用过的相同步骤对文件进行一些修改。在这种情况下，我将对`README.md`文件进行一些更改。更改完成后，我可以点击“新拉取请求”按钮来创建一个拉取请求：

![](assets/080bb0d8-2ecc-4778-b9be-8753ed38db7b.png)

拉取请求

在发起拉取请求时，我们应尽可能填写尽可能多的信息，以提供更改的理由：

![](assets/87417bd2-789c-42f5-b5cc-d79e341451cd.png)

拉取请求详细信息

存储库维护者将收到拉取请求的通知；如果被接受，更改将传递到原始存储库：

![](assets/8cecc7d3-149e-4400-a506-7b4404e530bf.png)

拉取请求记录

GitHub为与其他开发人员合作提供了一个出色的平台；这很快成为了许多大型开源项目的事实开发选择。在接下来的部分，让我们看看如何使用Python与Git。

# 使用Python的Git

有一些Python包可以与Git和GitHub一起使用。在本节中，我们将看一下GitPython和PyGithub库。

# GitPython

我们可以使用GitPython包[https://gitpython.readthedocs.io/en/stable/index.html](https://gitpython.readthedocs.io/en/stable/index.html)来处理我们的Git存储库。我们将安装该包并使用Python shell来构建一个`Repo`对象。从那里，我们可以列出存储库中的所有提交：

```py
$ sudo pip3 install gitpython
$ python3
>>> from git import Repo
>>> repo = Repo('/home/echou/Master_Python_Networking_second_edition/Chapter11/TestRepo')
>>> for commits in list(repo.iter_commits('master')):
... print(commits)
...
0aa362a47782e7714ca946ba852f395083116ce5
a001b816bb75c63237cbc93067dffcc573c05aa2
bc078a97e41d1614c1ba1f81f72acbcd95c0728c
2ec5f7d1a734b2cc74343ce45075917b79cc7293
c98373069f27d8b98d1ddacffe51b8fa7a30cf28
a3dd3ea8e6eb15b57d1f390ce0d2c3a03f07a038
5f579ab1e9a3fae13aa7f1b8092055213157524d

```

我们还可以查看索引条目：

```py
>>> for (path, stage), entry in index.entries.items():
... print(path, stage, entry)
...
mySecondFile.txt 0 100644 75d6370ae31008f683cf18ed086098d05bf0e4dc 0 mySecondFile.txt
LICENSE 0 100644 52feb16b34de141a7567e4d18164fe2400e9229a 0 LICENSE
myFile.txt 0 100644 69e7d4728965c885180315c0d4c206637b3f6bad 0 myFile.txt
.gitignore 0 100644 894a44cc066a027465cd26d634948d56d13af9af 0 .gitignore
README.md 0 100644 a29fe688a14d119c20790195a815d078976c3bc6 0 README.md
>>>
```

GitPython与所有Git功能集成良好。但是它并不是最容易使用的。我们需要了解Git的术语和结构，以充分利用GitPython。但是要记住，以防我们需要它用于其他项目。

# PyGitHub

让我们看看如何使用PyGitHub包[http://pygithub.readthedocs.io/en/latest/](http://pygithub.readthedocs.io/en/latest/)与GitHub存储库进行交互。该包是围绕GitHub APIv3的包装器[https://developer.github.com/v3/](https://developer.github.com/v3/)：

```py
$ sudo pip install pygithub
$ sudo pip3 install pygithub
```

让我们使用Python shell来打印用户当前的存储库：

```py
$ python3
>>> from github import Github
>>> g = Github("ericchou1", "<password>")
>>> for repo in g.get_user().get_repos():
...     print(repo.name)
...
ansible
...
-Hands-on-Network-Programming-with-Python
Mastering-Python-Networking
Mastering-Python-Networking-Second-Edition
>>>
```

为了更多的编程访问，我们还可以使用访问令牌创建更细粒度的控制。Github允许令牌与所选权限关联：

![](assets/ee987cae-5c00-4fc8-a0d7-78299cbf0e9a.png)

GitHub令牌生成

如果使用访问令牌作为认证机制，输出会有些不同：

```py
>>> from github import Github
>>> g = Github("<token>")
>>> for repo in g.get_user().get_repos():
...     print(repo)
...
Repository(full_name="oreillymedia/distributed_denial_of_service_ddos")
Repository(full_name="PacktPublishing/-Hands-on-Network-Programming-with-Python")
Repository(full_name="PacktPublishing/Mastering-Python-Networking")
Repository(full_name="PacktPublishing/Mastering-Python-Networking-Second-Edition")
...
```

现在我们熟悉了Git、GitHub和一些Python包，我们可以使用它们来处理技术。在接下来的部分，我们将看一些实际的例子。

# 自动化配置备份

在这个例子中，我们将使用PyGithub来备份包含我们路由器配置的目录。我们已经看到了如何使用Python或Ansible从我们的设备中检索信息；现在我们可以将它们检入GitHub。

我们有一个子目录，名为`config`，其中包含我们的路由器配置的文本格式：

```py
$ ls configs/
iosv-1 iosv-2

$ cat configs/iosv-1
Building configuration...

Current configuration : 4573 bytes
!
! Last configuration change at 02:50:05 UTC Sat Jun 2 2018 by cisco
!
version 15.6
service timestamps debug datetime msec
...
```

我们可以使用以下脚本从我们的GitHub存储库中检索最新的索引，构建我们需要提交的内容，并自动提交配置：

```py
$ cat Chapter11_1.py
#!/usr/bin/env python3
# reference: https://stackoverflow.com/questions/38594717/how-do-i-push-new-files-to-github

from github import Github, InputGitTreeElement
import os

github_token = '<token>'
configs_dir = 'configs'
github_repo = 'TestRepo'

# Retrieve the list of files in configs directory
file_list = []
for dirpath, dirname, filenames in os.walk(configs_dir):
    for f in filenames:
        file_list.append(configs_dir + "/" + f)

g = Github(github_token)
repo = g.get_user().get_repo(github_repo)

commit_message = 'add configs'
master_ref = repo.get_git_ref('heads/master')
master_sha = master_ref.object.sha
base_tree = repo.get_git_tree(master_sha)

element_list = list()

for entry in file_list:
    with open(entry, 'r') as input_file:
        data = input_file.read()
    element = InputGitTreeElement(entry, '100644', 'blob', data)
    element_list.append(element)

# Create tree and commit
tree = repo.create_git_tree(element_list, base_tree)
parent = repo.get_git_commit(master_sha)
commit = repo.create_git_commit(commit_message, tree, [parent])
master_ref.edit(commit.sha)
```

我们可以在GitHub存储库中看到`configs`目录：

![](assets/bbc515e8-57e3-4942-87ce-cd5f36ba8662.png)

Configs目录

提交历史显示了我们脚本的提交：

![](assets/74cb1602-7bad-4888-b920-78a54d3c3051.png)

提交历史

在*GitHub示例*部分，我们看到了如何通过分叉存储库并发出拉取请求与其他开发人员合作。让我们看看如何进一步使用Git进行协作。

# 与Git协作

Git是一种很棒的协作技术，而GitHub是一种非常有效的共同开发项目的方式。GitHub为世界上任何有互联网访问权限的人提供了一个免费分享他们的想法和代码的地方。我们知道如何使用Git和一些基本的GitHub协作步骤，但是我们如何加入并为一个项目做出贡献呢？当然，我们想回馈给那些给予我们很多的开源项目，但是我们如何开始呢？

在本节中，我们将看一些关于使用Git和GitHub进行软件开发协作的要点：

+   **从小开始**：理解的最重要的事情之一是我们在团队中可以扮演的角色。我们可能擅长网络工程，但是Python开发水平一般。有很多事情我们可以做，不一定要成为高技能的开发者。不要害怕从小事做起，文档编写和测试是成为贡献者的好方法。

+   **学习生态系统**：对于任何项目，无论大小，都有一套已经建立的惯例和文化。我们都被Python的易于阅读的语法和初学者友好的文化所吸引；他们还有一个围绕这种意识形态的开发指南（[https://devguide.python.org/](https://devguide.python.org/)）。另一方面，Ansible项目还有一个广泛的社区指南（[https://docs.ansible.com/ansible/latest/community/index.html](https://docs.ansible.com/ansible/latest/community/index.html)）。它包括行为准则、拉取请求流程、如何报告错误以及发布流程。阅读这些指南，了解感兴趣项目的生态系统。

+   **创建分支**：我犯了一个错误，分叉了一个项目并为主分支提出了拉取请求。主分支应该留给核心贡献者进行更改。我们应该为我们的贡献创建一个单独的分支，并允许在以后的某个日期合并该分支。

+   **保持分叉存储库同步**：一旦您分叉了一个项目，就没有规则强制克隆存储库与主存储库同步。我们应该定期执行`git pull`（获取代码并在本地合并）或`git fetch`（获取本地任何更改的代码）以确保我们拥有主存储库的最新副本。

+   **友善相处**：就像现实世界一样，虚拟世界也不容忍敌意。讨论问题时，要文明友好，即使意见不一致也是如此。

Git和GitHub为任何有动力的个人提供了一种方式，使其易于在项目上进行协作，从而产生影响。我们都有能力为任何我们感兴趣的开源或私有项目做出贡献。

# 总结

在本章中，我们看了一下被称为Git的版本控制系统及其近亲GitHub。Git是由Linus Torvolds于2005年开发的，用于帮助开发Linux内核，后来被其他开源项目采用为源代码控制系统。Git是一个快速、分布式和可扩展的系统。GitHub提供了一个集中的位置在互联网上托管Git存储库，允许任何有互联网连接的人进行协作。

我们看了如何在命令行中使用Git，以及它的各种操作，以及它们在GitHub中的应用。我们还研究了两个用于处理Git的流行Python库：GitPython和PyGitHub。我们以一个配置备份示例和关于项目协作的注释结束了本章。

在[第12章](5a99fe1f-da17-491c-96a2-4511ff2f4803.xhtml)中，*使用Jenkins进行持续集成*，我们将看另一个流行的开源工具，用于持续集成和部署：Jenkins。
