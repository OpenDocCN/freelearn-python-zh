# 第十三章：维护

在本章中，我们将涵盖以下主题：

+   创建和恢复 MySQL 数据库备份

+   创建和恢复 PostgreSQL 数据库备份

+   为常规任务设置 cron 作业

+   记录事件以进行进一步审查

+   通过电子邮件获取详细的错误报告

# 介绍

此时，您应该已经开发和发布了一个或多个 Django 项目。在开发周期的最后阶段，我们将看看如何维护您的项目并监视它们以进行优化。敬请关注最后的细节和片段！

# 技术要求

要使用本章的代码，您需要最新稳定版本的 Python、MySQL 或 PostgreSQL 数据库以及一个带有虚拟环境的 Django 项目。

您可以在 GitHub 存储库的`ch13`目录中找到本章的所有代码：[`github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition`](https://github.com/PacktPublishing/Django-3-Web-Development-Cookbook-Fourth-Edition)。

# 创建和恢复 MySQL 数据库备份

为了网站的稳定性，能够从硬件故障和黑客攻击中恢复是非常重要的。因此，您应该始终进行备份并确保它们有效。您的代码和静态文件通常会驻留在版本控制中，可以从中恢复，但数据库和媒体文件应定期备份。

在这个配方中，我们将向您展示如何为 MySQL 数据库创建备份。

# 准备工作

确保您的 Django 项目正在运行一个 MySQL 数据库。将该项目部署到远程生产（或暂存）服务器。

# 如何做到...

要备份和恢复您的 MySQL 数据库，请执行以下步骤：

1.  在项目的主目录下的`commands`目录中，创建一个 bash 脚本：`backup_mysql_db.sh`。按照以下方式开始脚本，包括变量和函数定义：

```py
/home/myproject/commands/backup_mysql_db.sh
#!/usr/bin/env bash
SECONDS=0
export DJANGO_SETTINGS_MODULE=myproject.settings.production
PROJECT_PATH=/home/myproject
REPOSITORY_PATH=${PROJECT_PATH}/src/myproject
LOG_FILE=${PROJECT_PATH}/logs/backup_mysql_db.log
DAY_OF_THE_WEEK=$(LC_ALL=en_US.UTF-8 date +"%w-%A")
DAILY_BACKUP_PATH=${PROJECT_PATH}/db_backups/${DAY_OF_THE_WEEK}.sql
LATEST_BACKUP_PATH=${PROJECT_PATH}/db_backups/latest.sql
error_counter=0

echoerr() { echo "$@" 1>&2; }

cd ${PROJECT_PATH}
mkdir -p logs
mkdir -p db_backups

source env/bin/activate
cd ${REPOSITORY_PATH}

DATABASE=$(echo "from django.conf import settings; print(settings.DATABASES['default']['NAME'])" | python manage.py shell -i python)
USER=$(echo "from django.conf import settings; print(settings.DATABASES['default']['USER'])" | python manage.py shell -i python)
PASSWORD=$(echo "from django.conf import settings; print(settings.DATABASES['default']['PASSWORD'])" | python manage.py shell -i python)

EXCLUDED_TABLES=(
django_session
)

IGNORED_TABLES_STRING=''
for TABLE in "${EXCLUDED_TABLES[@]}"; do
    IGNORED_TABLES_STRING+=" --ignore-table=${DATABASE}.${TABLE}"
done
```

1.  然后，添加命令来创建数据库结构和数据的转储：

```py
echo "=== Creating DB Backup ===" > ${LOG_FILE}
date >> ${LOG_FILE}

echo "- Dump structure" >> ${LOG_FILE}
mysqldump -u "${USER}" -p"${PASSWORD}" --single-transaction --no-data "${DATABASE}" > "${DAILY_BACKUP_PATH}" 2>> ${LOG_FILE}
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command mysqldump for dumping database structure 
         failed with exit code ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

echo "- Dump content" >> ${LOG_FILE}
# shellcheck disable=SC2086
mysqldump -u "${USER}" -p"${PASSWORD}" "${DATABASE}" ${IGNORED_TABLES_STRING} >> "${DAILY_BACKUP_PATH}" 2>> ${LOG_FILE}
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command mysqldump for dumping database content 
         failed with exit code ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

```

1.  添加命令来压缩数据库转储并创建一个符号链接`latest.sql.gz`：

```py
echo "- Create a *.gz archive" >> ${LOG_FILE}
gzip --force "${DAILY_BACKUP_PATH}"
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command gzip failed with exit code 
         ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

echo "- Create a symlink latest.sql.gz" >> ${LOG_FILE}
if [ -e "${LATEST_BACKUP_PATH}.gz" ]; then
    rm "${LATEST_BACKUP_PATH}.gz"
fi
ln -s "${DAILY_BACKUP_PATH}.gz" "${LATEST_BACKUP_PATH}.gz"
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command ln failed with exit code 
         ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

```

1.  通过记录执行前面命令所花费的时间来完成脚本：

```py
duration=$SECONDS
echo "------------------------------------------" >> ${LOG_FILE}
echo "The operation took $((duration / 60)) minutes and $((duration % 60)) seconds." >> ${LOG_FILE}
exit $error_counter
```

1.  在同一目录中，创建一个名为`restore_mysql_db.sh`的 bash 脚本，内容如下：

```py
# home/myproject/commands/restore_mysql_db.sh
#!/usr/bin/env bash
SECONDS=0
PROJECT_PATH=/home/myproject
REPOSITORY_PATH=${PROJECT_PATH}/src/myproject
LATEST_BACKUP_PATH=${PROJECT_PATH}/db_backups/latest.sql
export DJANGO_SETTINGS_MODULE=myproject.settings.production

cd "${PROJECT_PATH}"
source env/bin/activate

echo "=== Restoring DB from a Backup ==="

echo "- Fill the database with schema and data"
cd "${REPOSITORY_PATH}"
zcat "${LATEST_BACKUP_PATH}.gz" | python manage.py dbshell

duration=$SECONDS
echo "------------------------------------------"
echo "The operation took $((duration / 60)) minutes and $((duration % 60)) seconds."
```

1.  使这两个脚本都可执行：

```py
$ chmod +x *.sh
```

1.  运行数据库备份脚本：

```py
$ ./backup_mysql_db.sh
```

1.  运行数据库恢复脚本（如果在生产中请谨慎）：

```py
$ ./restore_mysql_db.sh
```

# 它是如何工作的...

备份脚本将在`/home/myproject/db_backups/`目录下创建备份文件，并将日志保存在`/home/myproject/logs/backup_mysql_db.log`，类似于这样：

```py
=== Creating DB Backup ===
Fri Jan 17 02:12:14 CET 2020
- Dump structure
mysqldump: [Warning] Using a password on the command line interface can be insecure.
- Dump content
mysqldump: [Warning] Using a password on the command line interface can be insecure.
- Create a *.gz archive
- Create a symlink latest.sql.gz
------------------------------------------
The operation took 0 minutes and 2 seconds.
```

如果操作成功，脚本将返回退出代码`0`；否则，退出代码将是执行脚本时的错误数量。日志文件将显示错误消息。

在`db_backups`目录中，将有一个带有星期几的压缩 SQL 备份，例如`0-Sunday.sql.gz`，`1-Monday.sql.gz`等，以及另一个文件，实际上是一个符号链接，名为`latest.sql.gz`。基于工作日的备份允许您在正确设置 cron 作业时拥有最近 7 天的备份，并且符号链接允许您快速或自动将最新备份传输到另一台计算机上通过 SSH。

请注意，我们从 Django 设置中获取数据库凭据，然后在 bash 脚本中使用它们。

我们正在转储除了会话表之外的所有数据，因为会话本来就是临时的，而且占用内存很多。

当我们运行`restore_mysql_db.sh`脚本时，我们会得到如下输出：

```py
=== Restoring DB from a Backup ===
- Fill the database with schema and data
mysql: [Warning] Using a password on the command line interface can be insecure.
------------------------------------------
The operation took 0 minutes and 2 seconds.
```

# 另请参阅

+   第十二章*部署*中的*在 Apache 上使用 mod_wsgi 部署生产环境*配方

+   第十二章**部署**中的*在 Nginx 和 Gunicorn 上部署生产环境*配方

+   *创建和恢复 PostgreSQL 数据库备份*配方

+   *为常规任务设置 cron 作业*配方

# 创建和恢复 PostgreSQL 数据库备份

在本食谱中，您将学习如何备份 PostgreSQL 数据库，并在硬件故障或黑客攻击发生时恢复它们。

# 准备工作

确保已经运行了一个带有 PostgreSQL 数据库的 Django 项目。将该项目部署到远程暂存或生产服务器。

# 操作方法

要备份和恢复 MySQL 数据库，请执行以下步骤：

1.  在项目的主目录下的`commands`目录中，创建一个名为`backup_postgresql_db.sh`的 bash 脚本。开始脚本时，定义变量和函数，如下所示：

```py
/home/myproject/commands/backup_postgresql_db.sh
#!/usr/bin/env bash
SECONDS=0
PROJECT_PATH=/home/myproject
REPOSITORY_PATH=${PROJECT_PATH}/src/myproject
LOG_FILE=${PROJECT_PATH}/logs/backup_postgres_db.log
DAY_OF_THE_WEEK=$(LC_ALL=en_US.UTF-8 date +"%w-%A")
DAILY_BACKUP_PATH=${PROJECT_PATH}/db_backups/${DAY_OF_THE_WEEK}.backup
LATEST_BACKUP_PATH=${PROJECT_PATH}/db_backups/latest.backup
error_counter=0

echoerr() { echo "$@" 1>&2; }

cd ${PROJECT_PATH}
mkdir -p logs
mkdir -p db_backups

source env/bin/activate
cd ${REPOSITORY_PATH}

DATABASE=$(echo "from django.conf import settings; print(settings.DATABASES['default']['NAME'])" | python manage.py shell -i python)

```

1.  然后，添加一个命令以创建数据库转储：

```py
echo "=== Creating DB Backup ===" > ${LOG_FILE}
date >> ${LOG_FILE}

echo "- Dump database" >> ${LOG_FILE}
pg_dump --format=p --file="${DAILY_BACKUP_PATH}" ${DATABASE}
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command pg_dump failed with exit code 
         ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

```

1.  添加命令以压缩数据库转储并创建一个名为`latest.backup.gz`的符号链接：

```py
echo "- Create a *.gz archive" >> ${LOG_FILE}
gzip --force "${DAILY_BACKUP_PATH}"
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command gzip failed with exit code 
         ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

echo "- Create a symlink latest.backup.gz" >> ${LOG_FILE}
if [ -e "${LATEST_BACKUP_PATH}.gz" ]; then
    rm "${LATEST_BACKUP_PATH}.gz"
fi
ln -s "${DAILY_BACKUP_PATH}.gz" "${LATEST_BACKUP_PATH}.gz"
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Command ln failed with exit code 
         ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

```

1.  通过记录执行前一个命令所花费的时间来完成脚本：

```py
duration=$SECONDS
echo "------------------------------------------" >> ${LOG_FILE}
echo "The operation took $((duration / 60)) minutes and $((duration % 60)) seconds." >> ${LOG_FILE}
exit $error_counter

```

1.  在同一目录中，创建一个名为`restore_postgresql_db.sh`的 bash 脚本，内容如下：

```py
# /home/myproject/commands/restore_postgresql_db.sh
#!/usr/bin/env bash
SECONDS=0
PROJECT_PATH=/home/myproject
REPOSITORY_PATH=${PROJECT_PATH}/src/myproject
LATEST_BACKUP_PATH=${PROJECT_PATH}/db_backups/latest.backup
export DJANGO_SETTINGS_MODULE=myproject.settings.production

cd "${PROJECT_PATH}"
source env/bin/activate

cd "${REPOSITORY_PATH}"

DATABASE=$(echo "from django.conf import settings; print(settings.DATABASES['default']['NAME'])" | python manage.py shell -i python)
USER=$(echo "from django.conf import settings; print(settings.DATABASES['default']['USER'])" | python manage.py shell -i python)
PASSWORD=$(echo "from django.conf import settings; print(settings.DATABASES['default']['PASSWORD'])" | python manage.py shell -i python)

echo "=== Restoring DB from a Backup ==="

echo "- Recreate the database"
psql --dbname=$DATABASE --command='SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid();' 
dropdb $DATABASE
 createdb --username=$USER $DATABASE

echo "- Fill the database with schema and data"
zcat "${LATEST_BACKUP_PATH}.gz" | python manage.py dbshell

duration=$SECONDS
echo "------------------------------------------"
echo "The operation took $((duration / 60)) minutes and $((duration % 60)) seconds."

```

1.  使这两个脚本都可执行：

```py
$ chmod +x *.sh
```

1.  运行数据库备份脚本：

```py
$ ./backup_postgresql_db.sh
```

1.  运行数据库恢复脚本（如果在生产中，请谨慎）：

```py
$ ./restore_postgresql_db.sh
```

# 工作原理

备份脚本将在`/home/myproject/db_backups/`下创建备份文件，并将日志保存在`/home/myproject/logs/backup_postgresql_db.log`中，类似于这样：

```py
=== Creating DB Backup ===
Fri Jan 17 02:40:55 CET 2020
- Dump database
- Create a *.gz archive
- Create a symlink latest.backup.gz
------------------------------------------
The operation took 0 minutes and 1 seconds.
```

如果操作成功，脚本将返回退出代码`0`；否则，退出代码将是执行脚本时出现的错误数量。日志文件将显示错误消息。

在`db_backups`目录中，将有一个带有星期几的压缩 SQL 备份文件，例如`0-Sunday.backup.gz`，`1-Monday.backup.gz`等等，还有另一个文件，实际上是一个符号链接，名为`latest.backup.gz`。基于工作日的备份允许您在正确设置 cron 作业时拥有最近 7 天的备份，符号链接允许您通过 SSH 快速或自动将最新备份传输到另一台计算机。

请注意，我们从 Django 设置中获取数据库凭据，然后在 bash 脚本中使用它们。

当我们运行`restore_postgresql_db.sh`脚本时，我们会得到如下输出：

```py
=== Restoring DB from a Backup ===
- Recreate the database
 pg_terminate_backend
----------------------
(0 rows)

- Fill the database with schema and data
SET
SET
SET
SET
SET
 set_config
------------

(1 row)

SET

…

ALTER TABLE
ALTER TABLE
ALTER TABLE
------------------------------------------
The operation took 0 minutes and 2 seconds.
```

# 另请参阅

+   第十二章*部署*中的*在 Apache 上使用 mod_wsgi 部署生产环境*食谱

+   第十二章*部署*中的*在 Nginx 和 Gunicorn 上部署生产环境*食谱

+   *创建和恢复 PostgreSQL 数据库备份*食谱

+   *为常规任务设置 cron 作业*食谱

# 为常规任务设置 cron 作业

通常，网站有一些后台管理任务需要定期执行，例如每周一次、每天一次或每小时一次。这可以通过使用定时任务（通常称为 cron 作业）来实现。这些是在服务器上在指定时间段后运行的脚本。在本食谱中，我们将创建两个 cron 作业：一个用于从数据库中清除会话，另一个用于备份数据库数据。两者都将在每晚运行。

# 准备工作

首先，将 Django 项目部署到远程服务器。然后，通过 SSH 连接到服务器。这些步骤假定您正在使用虚拟环境，但是可以为 Docker 项目创建类似的 cron 作业，并且甚至可以直接在应用程序容器中运行。提供了备用语法的代码文件，步骤基本相同。

# 操作方法

让我们创建这两个脚本，并通过以下步骤定期运行它们：

1.  在生产或暂存服务器上，导航到项目用户的主目录，其中包含`env`和`src`目录。

1.  如果尚不存在，请按以下方式创建`commands`、`db_backups`和`logs`文件夹，如下所示：

```py
(env)$ mkdir commands db_backups logs
```

1.  在`commands`目录中，创建一个`clear_sessions.sh`文件。您可以使用终端编辑器（如 vim 或 nano）编辑它，添加以下内容：

```py
# /home/myproject/commands/clear_sessions.sh
#!/usr/bin/env bash
SECONDS=0
export DJANGO_SETTINGS_MODULE=myproject.settings.production
PROJECT_PATH=/home/myproject
REPOSITORY_PATH=${PROJECT_PATH}/src/myproject
LOG_FILE=${PROJECT_PATH}/logs/clear_sessions.log
error_counter=0

echoerr() { echo "$@" 1>&2; }

cd ${PROJECT_PATH}
mkdir -p logs

echo "=== Clearing up Outdated User Sessions ===" > ${LOG_FILE}
date >> ${LOG_FILE}

source env/bin/activate
cd ${REPOSITORY_PATH}
python manage.py clearsessions >> "${LOG_FILE}" 2>&1
function_exit_code=$?
if [[ $function_exit_code -ne 0 ]]; then
    {
        echoerr "Clearing sessions failed with exit code 
         ($function_exit_code)."
        error_counter=$((error_counter + 1))
    } >> "${LOG_FILE}" 2>&1
fi

duration=$SECONDS
echo "------------------------------------------" >> ${LOG_FILE}
echo "The operation took $((duration / 60)) minutes and $((duration % 60)) seconds." >> ${LOG_FILE}
exit $err
or_counter
```

1.  使`clear_sessions.sh`文件可执行，如下所示：

```py
$ chmod +x *.sh
```

1.  假设您正在使用 PostgreSQL 作为项目的数据库。然后，在相同的目录中，按照上一个配方*创建和恢复 PostgreSQL 数据库备份*的说明创建一个备份脚本。

1.  测试脚本以查看它们是否正确执行，方法是运行它们，然后检查日志目录中的`*.log`文件，如下所示：

```py
$ ./clear_sessions.sh
$ ./backup_postgresql_db.sh
```

1.  在远程服务器上的项目主目录中，创建一个`crontab.txt`文件，内容如下：

```py
# /home/myproject/crontab.txt
MAILTO=""
HOME=/home/myproject
PATH=/home/myproject/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
SHELL=/bin/bash
00 01 * * * /home/myproject/commands/clear_sessions.sh
00 02 * * * /home/myproject/commands/backup_postgresql_db.sh
```

1.  按照以下方式将`crontab`任务安装为`myproject`用户：

```py
(env)$ crontab crontab.txt
```

# 工作原理...

使用当前设置，每天晚上`clear_sessions.sh`将在凌晨 1:00 执行，`backup_postgresql_db.sh`将在凌晨 2:00 执行。执行日志将保存在`~/logs/clear_sessions.sh`和`~/logs/backup_postgresql_db.log`中。如果出现任何错误，您应该检查这些文件以获取回溯信息。

每天，`clear_sessions.sh`将执行`clearsessions`管理命令，正如其名称所暗示的那样，它将使用默认数据库设置从数据库中清除过期会话。

数据库备份脚本稍微复杂一些。每周的每一天，它都会创建一个备份文件，使用命名方案`0-Sunday.backup.gz`，`1-Monday.backup.gz`等等。因此，您将能够恢复 7 天前或更晚备份的数据。

crontab 文件遵循特定的语法。每行包含特定的一天时间，由一系列数字表示，然后是在给定时刻运行的任务。时间分为五部分，用空格分隔，如下列表所示：

+   分钟，从 0 到 59

+   小时，从 0 到 23

+   每月的日期，从 1 到 31

+   月份，从 1 到 12

+   每周的日期，从 0 到 7，其中 0 是星期日，1 是星期一，依此类推，7 又是星期日

星号（`*`）表示将使用每个时间段。因此，以下任务定义了`clear_sessions.sh`将在每个月的每一天，每个月和每周的每一天的 1:00 执行：

```py
00 01 * * * /home/myproject/commands/clear_sessions.sh
```

您可以在[`en.wikipedia.org/wiki/Cron`](https://en.wikipedia.org/wiki/Cron)了解有关 crontab 的具体信息。

# 还有更多...

我们定义了将定期执行的命令，并且还激活了结果的记录，但是我们还不能确定 cron 作业是否成功执行，除非我们每天手动登录服务器并检查日志。为了解决单调的手动劳动问题，您可以使用**Healthchecks**服务（[`healthchecks.io/`](https://healthchecks.io/)）自动监视 cron 作业。

使用 Healthchecks，您可以修改 crontab，以便在每次成功执行作业后 ping 特定 URL。如果脚本失败并以非零代码退出，Healthchecks 将知道它未成功执行。每天，您将通过电子邮件获取 cron 作业及其执行状态的概述。

# 另请参见

+   *在 Apache 上使用 mod_wsgi 部署生产环境*配方在第十二章*，部署*

+   *在 Nginx 和 Gunicorn 上部署生产环境*配方在第十二章*，部署*

+   *创建和恢复 MySQL 数据库备份*配方

+   *创建和恢复 PostgreSQL 数据库备份*配方

# 记录事件以进行进一步审查

在以前的配方中，您可以看到如何记录 bash 脚本的工作。但是您也可以记录发生在 Django 网站上的事件，例如用户注册、将产品添加到购物车、购买门票、银行交易、发送短信、服务器错误等。

您永远不应记录敏感信息，例如用户密码或信用卡详细信息。

此外，使用分析工具而不是 Python 记录来跟踪整体网站使用情况。

在本配方中，我们将指导您如何将有关您的网站的结构化信息记录到日志文件中。

# 准备工作

让我们从第四章*，模板和 JavaScript*中的*实现喜欢小部件*食谱开始。

在 Django 项目的虚拟环境中，安装`django-structlog`，如下所示：

```py
(env)$ pip install django-structlog==1.3.5

```

# 如何做...

要在 Django 网站中设置结构化日志记录，请按照以下步骤进行：

1.  在项目的设置中添加`RequestMiddleware`：

```py
# myproject/settings/_base.py MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django.middleware.locale.LocaleMiddleware",
 "django_structlog.middlewares.RequestMiddleware",
]
```

1.  同样在同一文件中，添加 Django 日志配置：

```py
# myproject/settings/_base.py
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json_formatter": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer(),
        },
        "plain_console": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(),
        },
        "key_value": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor":  
             structlog.processors.KeyValueRenderer(key_order=
             ['timestamp', 'level', 'event', 'logger']),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain_console",
        },
        "json_file": {
            "class": "logging.handlers.WatchedFileHandler",
            "filename": os.path.join(BASE_DIR, "tmp", "json.log"),
            "formatter": "json_formatter",
        },
        "flat_line_file": {
            "class": "logging.handlers.WatchedFileHandler",
            "filename": os.path.join(BASE_DIR, "tmp", 
           "flat_line.log"),
            "formatter": "key_value",
        },
    },
    "loggers": {
        "django_structlog": {
            "handlers": ["console", "flat_line_file", "json_file"],
            "level": "INFO",
        },
    }
}
```

1.  还要在那里设置`structlog`配置：

```py
# myproject/settings/_base.py
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.ExceptionPrettyPrinter(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=structlog.threadlocal.wrap_dict(dict),
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
) 
```

1.  在`likes`应用程序的`views.py`中，让我们记录将被喜欢或取消喜欢的对象：

```py
# myproject/apps/likes/views.py
import structlog

from django.contrib.contenttypes.models import ContentType
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt

from .models import Like
from .templatetags.likes_tags import liked_count

logger = structlog.get_logger("django_structlog")

@never_cache
@csrf_exempt
def json_set_like(request, content_type_id, object_id):
    """
    Sets the object as a favorite for the current user
    """
    result = {
        "success": False,
    }
    if request.user.is_authenticated and request.method == "POST":
        content_type = ContentType.objects.get(id=content_type_id)
        obj = content_type.get_object_for_this_type(pk=object_id)

        like, is_created = Like.objects.get_or_create(
            content_type=ContentType.objects.get_for_model(obj),
            object_id=obj.pk,
            user=request.user)
        if is_created:
            logger.info("like_created",  
            content_type_id=content_type.pk, 
            object_id=obj.pk)
        else:
            like.delete()
            logger.info("like_deleted",  
            content_type_id=content_type.pk, 
            object_id=obj.pk) 

        result = {
            "success": True,
            "action": "add" if is_created else "remove",
            "count": liked_count(obj),
        }

    return JsonResponse(result)
```

# 它是如何工作的...

当访问者浏览您的网站时，特定事件将记录在`tmp/json.log`和`tmp/flat_line.log`文件中。`django_structlog.middlewares.RequestMiddleware`记录 HTTP 请求处理的开始和结束。此外，我们还记录了在我们的 Django 项目中创建或删除`Like`实例时的情况。

`json.log`文件包含以 JSON 格式记录的日志。这意味着您可以以编程方式解析、检查和分析它们：

```py
{"request_id": "ad0ef355-77ef-4474-a91a-2d9549a0e15d", "user_id": 1, "ip": "127.0.0.1", "request": "<WSGIRequest: POST '/en/likes/7/1712dfe4-2e77-405c-aa9b-bfa64a1abe98/'>", "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36", "event": "request_started", "timestamp": "2020-01-18T04:27:00.556135Z", "logger": "django_structlog.middlewares.request", "level": "info"}
{"request_id": "ad0ef355-77ef-4474-a91a-2d9549a0e15d", "user_id": 1, "ip": "127.0.0.1", "content_type_id": 7, "object_id": "UUID('1712dfe4-2e77-405c-aa9b-bfa64a1abe98')", "event": "like_created", "timestamp": "2020-01-18T04:27:00.602640Z", "logger": "django_structlog", "level": "info"}
{"request_id": "ad0ef355-77ef-4474-a91a-2d9549a0e15d", "user_id": 1, "ip": "127.0.0.1", "code": 200, "request": "<WSGIRequest: POST '/en/likes/7/1712dfe4-2e77-405c-aa9b-bfa64a1abe98/'>", "event": "request_finished", "timestamp": "2020-01-18T04:27:00.604577Z", "logger": "django_structlog.middlewares.request", "level": "info"}
```

`flat_line.log`文件以更短的格式包含日志，这可能更容易手动阅读：

```py
(env)$ tail -3 tmp/flat_line.log
timestamp='2020-01-18T04:27:03.437759Z' level='info' event='request_started' logger='django_structlog.middlewares.request' request_id='a74808ff-c682-4336-aeb9-f043f11a7316' user_id=1 ip='127.0.0.1' request=<WSGIRequest: POST '/en/likes/7/1712dfe4-2e77-405c-aa9b-bfa64a1abe98/'> user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
timestamp='2020-01-18T04:27:03.489198Z' level='info' event='like_deleted' logger='django_structlog' request_id='a74808ff-c682-4336-aeb9-f043f11a7316' user_id=1 ip='127.0.0.1' content_type_id=7 object_id=UUID('1712dfe4-2e77-405c-aa9b-bfa64a1abe98')
timestamp='2020-01-18T04:27:03.491927Z' level='info' event='request_finished' logger='django_structlog.middlewares.request' request_id='a74808ff-c682-4336-aeb9-f043f11a7316' user_id=1 ip='127.0.0.1' code=200 request=<WSGIRequest: POST '/en/likes/7/1712dfe4-2e77-405c-aa9b-bfa64a1abe98/'>
```

# 另请参阅

+   *创建和恢复 MySQL 数据库备份*食谱

+   *创建和恢复 PostgreSQL 数据库备份*食谱

+   *为定期任务设置 cron 作业*食谱

# 通过电子邮件获取详细的错误报告

为执行系统日志记录，Django 使用 Python 的内置日志记录模块或前一食谱中提到的`structlog`模块。默认的 Django 配置似乎相当复杂。在本食谱中，您将学习如何对其进行微调，以便在发生错误时以与 Django 在 DEBUG 模式下提供的完整 HTML 类似的方式发送错误电子邮件。

# 准备工作

定位虚拟环境中的 Django 项目。

# 如何做...

以下过程将向您发送有关错误的详细电子邮件：

1.  如果您的项目尚未设置`LOGGING`设置，请先设置。找到 Django 日志实用程序文件，位于`env/lib/python3.7/site-packages/django/utils/log.py`。将`DEFAULT_LOGGING`字典复制到项目的设置中作为`LOGGING`字典。

1.  将`include_html`设置添加到`mail_admins`处理程序。前两个步骤的结果应该类似于以下内容：

```py
# myproject/settings/production.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'formatters': {
        'django.server': {
            '()': 'django.utils.log.ServerFormatter',
            'format': '[{server_time}] {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
        },
        'django.server': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'django.server',
        },
        'mail_admins': {
            'level': 'ERROR',
            'filters': ['require_debug_false'],
            'class': 'django.utils.log.AdminEmailHandler',
 'include_html': True,
        }
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'mail_admins'],
            'level': 'INFO',
        },
        'django.server': {
            'handlers': ['django.server'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}
```

# 它是如何工作的...

日志配置由四个部分组成：记录器、处理程序、过滤器和格式化程序。以下列表对它们进行了描述：

+   **记录器**是日志系统的入口点。每个记录器都可以有一个日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR`或`CRITICAL`。当消息被写入记录器时，消息的日志级别将与记录器的级别进行比较。如果满足或超过记录器的日志级别，则将由处理程序进一步处理。否则，消息将被忽略。

+   **处理程序**是定义记录器中每条消息发生的情况的引擎。它们可以被写入控制台，通过电子邮件发送给管理员，保存到日志文件，发送到 Sentry 错误记录服务等等。在我们的情况下，我们为`mail_admins`处理程序设置了`include_html`参数，因为我们希望在我们的 Django 项目中发生错误时获得包含回溯和本地变量的完整 HTML 的错误消息。

+   **过滤器**提供对从记录器传递到处理程序的消息的额外控制。例如，在我们的情况下，仅当 DEBUG 模式设置为 false 时才会发送电子邮件。

+   **格式化程序**用于定义如何将日志消息呈现为字符串。在本示例中未使用它们；但是，有关日志记录的更多信息，您可以参考官方文档[`docs.djangoproject.com/en/3.0/topics/logging/`](https://docs.djangoproject.com/en/3.0/topics/logging/)。

# 还有更多...

我们刚刚定义的配置将发送有关发生在您的网站上的每个服务器错误的电子邮件。如果您的网站流量很大，比如数据库崩溃，您将收到大量电子邮件，这些邮件将淹没您的收件箱，甚至可能挂起您的电子邮件服务器。

为了避免这样的问题，您可以使用 Sentry ([`sentry.io/for/python/`](https://sentry.io/for/python/))。它会在服务器上跟踪所有服务器错误，并针对每种错误类型仅发送一封通知电子邮件给您。

# 另请参阅

+   第十二章*部署*中的*在 Apache 上使用 mod_wsgi 进行生产环境部署*食谱

+   第十二章*部署*中的*在 Nginx 和 Gunicorn 上进行生产环境部署*食谱

+   *用于进一步审查的日志事件*食谱
