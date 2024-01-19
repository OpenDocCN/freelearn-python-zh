# 使用Nmap扫描器

本章涵盖了如何使用python-nmap进行网络扫描，以收集有关网络、主机和主机上运行的服务的信息。一些允许端口扫描和自动检测服务和开放端口的工具，我们可以在Python中找到，其中我们可以突出显示python-nmap。Nmap是一个强大的端口扫描器，可以帮助您识别打开、关闭或过滤的端口。它还允许编程例程和脚本来查找给定主机可能存在的漏洞。

本章将涵盖以下主题：

+   学习和理解Nmap协议作为端口扫描器，以识别主机上运行的服务

+   学习和理解使用Nmap的`python-nmap`模块，这是一个非常有用的工具，可以优化与端口扫描相关的任务

+   学习和理解使用`python-nmap模块`进行同步和异步扫描

+   学习和理解Nmap脚本，以便检测网络或特定主机中的漏洞

# 技术要求

本章的示例和源代码可在GitHub存储库的`chapter8`文件夹中找到：

[https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security](https://github.com/PacktPublishing/Mastering-Python-for-Networking-and-Security)。

您需要在本地机器上安装一个至少有4GB内存的Python发行版。在本章中，我们将使用一个**虚拟机**，用于进行与端口分析和漏洞检测相关的一些测试。它可以从`sourceforge`页面下载：

[https://sourceforge.net/projects/metasploitable/files/Metasploitable2](https://sourceforge.net/projects/metasploitable/files/Metasploitable2)

要登录，您必须使用用户名`msfadmin`和密码`msfadmin`：

![](assets/a762810d-c72e-4099-a79a-7b232836446a.png)

如果我们执行`ifconfig`命令，我们可以看到网络的配置和我们可以用来执行测试的IP地址。在这种情况下，我们本地网络的IP地址是**192.168.56.101**：

![](assets/ae16ef46-b2ea-4c49-b9ef-88ce95a6e9cc.png)

如果我们使用`nmap`命令进行端口扫描，我们可以看到虚拟机中打开的端口：

![](assets/c573f22c-7a09-47bf-b71f-5994d039b4db.png)

基本上，Metasploitable虚拟机（vm）是Ubuntu Linux的一个易受攻击的版本，旨在测试安全工具并演示常见的漏洞。

您可以在以下指南中找到有关此虚拟机的更多信息：[https://metasploit.help.rapid7.com/docs/metasploitable-2-exploitability-guide.](https://metasploit.help.rapid7.com/docs/metasploitable-2-exploitability-guide)

# 介绍使用Nmap进行端口扫描

在这一部分，我们将回顾Nmap工具用于端口扫描以及它支持的主要扫描类型。我们将了解Nmap作为一个端口扫描器，它允许我们分析机器上运行的端口和服务。

# 介绍端口扫描

一旦我们在我们的网络中确定了端点，下一步就是进行端口扫描。支持通信协议的计算机利用端口来建立连接。为了支持与多个应用程序的不同对话，端口用于区分同一台机器或服务器中的各种通信。例如，Web服务器可以使用**超文本传输协议**（**HTTP**）来提供对使用TCP端口号`80`的网页的访问。**简单邮件传输协议**或**SMTP**使用端口`25`来发送或传输邮件消息。对于每个唯一的IP地址，协议端口号由一个16位数字标识，通常称为端口号`0-65,535`。端口号和IP地址的组合提供了通信的完整地址。根据通信的方向，需要源地址和目标地址（IP地址和端口组合）。

# 使用Nmap进行扫描的类型

网络映射器（Nmap）是用于网络发现和安全审计的免费开源工具。它可以在所有主要计算机操作系统上运行，并且针对Linux、Windows和Mac OS X提供官方二进制包。python-nmap库有助于以编程方式操纵Nmap的扫描结果，以自动执行端口扫描任务。

Nmap工具主要用于识别和扫描特定网络段中的端口。从网站[https://nmap.org](https://nmap.org)，我们可以根据要安装的操作系统下载最新版本。

如果我们从控制台运行Nmap工具，我们会得到这个：

![](assets/c3307890-497e-4407-b904-5f7d0890ea6b.png)

我们可以看到我们有以下**扫描类型**：

sT（TCP Connect扫描）：这通常用于检测端口是否打开或关闭的选项，但通常是最受审计和最受入侵检测系统监视的机制。使用此选项，如果服务器在发送具有SYN标志的数据包时响应一个包含ACK标志的数据包，则端口是打开的。

sS（TCP Stealth扫描）：这是一种基于TCP Connect扫描的扫描类型，其不同之处在于不完全进行指定端口的连接。它包括在检查发送具有SYN标志的数据包之前检查目标的响应数据包。如果目标以激活了RST标志的数据包响应，则可以检查端口是打开还是关闭。

u（UDP扫描）：这是一种基于UDP协议的扫描类型，其中不进行连接过程，而只是发送一个UDP数据包来确定端口是否打开。如果答案是另一个UDP数据包，则意味着该端口是打开的。如果答案返回，端口是关闭的，并且将收到ICMP类型3（目的地不可达）的数据包。

sA（TCP ACK扫描）：这种扫描类型让我们知道我们的目标机器是否运行任何类型的防火墙。这种扫描发送一个激活了ACK标志的数据包到目标机器。如果远程机器以激活了RST标志的数据包响应，可以确定该端口没有被任何防火墙过滤。如果远程不响应，或者以ICMP类型的数据包响应，可以确定有防火墙过滤发送到指定端口的数据包。

sN（TCP空扫描）：这是一种扫描类型，它向目标机器发送一个没有任何标志的TCP数据包。如果远程机器没有发出响应，可以确定该端口是打开的。否则，如果远程机器返回一个RST标志，我们可以说该端口是关闭的。

sF（TCP FIN扫描）：这是一种向目标机器发送带有FIN标志的TCP数据包的扫描类型。如果远程机器没有发出响应，可以确定该端口是打开的。如果远程机器返回一个RST标志，我们可以说该端口是关闭的。

sX（TCP XMAS扫描）：这是一种向目标机器发送带有PSH、FIN或URG标志的TCP数据包的扫描类型。如果远程机器没有发出响应，可以确定该端口是打开的。如果远程机器返回一个RST标志，我们可以说该端口是关闭的。如果在响应数据包中获得ICMP类型3的响应，则端口被过滤。

默认扫描类型可能会因运行它的用户而异，因为在扫描期间允许发送数据包的权限不同。扫描类型之间的区别在于每种类型生成的“噪音”，以及它们避免被安全系统（如防火墙或入侵检测系统）检测到的能力。

如果我们想创建一个端口扫描程序，我们将不得不为每个打开端口的套接字创建一个线程，并通过交通灯管理屏幕的共享使用。通过这种方法，我们将有一个很长的代码，而且我们只会执行一个简单的TCP扫描，而不是Nmap工具包提供的ACK、SYN-ACK、RST或FIN。

由于Nmap响应格式是XML，因此很容易编写一个Python模块，允许解析此响应格式，提供与Nmap的完全集成，并能够运行更多类型的扫描。因此，`python-nmap`模块成为执行这些类型任务的主要模块。

# 使用python-nmap进行端口扫描

在本节中，我们将回顾Python中用于端口扫描的`python-nmap`模块。我们将学习`python-nmap`模块如何使用Nmap，以及它如何是一个非常有用的工具，用于优化有关在特定目标（域、网络或IP地址）上发现服务的任务。

# 介绍python-nmap

在Python中，我们可以通过python-nmap库使用Nmap，这使我们可以轻松地操作扫描结果。此外，对于系统管理员或计算机安全顾问来说，它可以是自动化渗透测试过程的完美工具。

python-nmap是在安全审计或入侵测试范围内使用的工具，其主要功能是发现特定主机开放的端口或服务。此外，它的优势在于与2.x和3.x版本兼容。

您可以从Bitbucket存储库获取python-nmap的源代码：

[https://bitbucket.org/xael/python-nmap](https://bitbucket.org/xael/python-nmap)

最新版本的python-nmap可以从以下网站下载：

[http://xael.org/pages/python-nmap-en.html](http://xael.org/pages/python-nmap-en.html)

[https://xael.org/norman/python/python-nmap](https://xael.org/norman/python/python-nmap/)

# 安装python-nmap

要进行安装，请解压下载的软件包，跳转到新目录，并执行安装命令。

在此示例中，我们正在安装源包的版本0.5：

![](assets/9bd17528-a2bf-482d-99f8-022e05defafe.png)

还可以使用`pip install`工具安装模块，因为它在官方存储库中。要安装模块，需要以管理员权限执行命令或使用系统超级用户（`sudo`）：

```py
sudo apt-get install python-pip nmap
sudo pip install python-nmap
```

# 使用python-nmap

现在，您可以导入python-nmap模块，我们可以从脚本或交互式终端中调用它，例如：

![](assets/a1a8c4c9-d51c-404b-9639-4d9f6096f9b0.png)

一旦我们验证了模块的安装，我们就可以开始对特定主机执行扫描。为此，我们必须对`PortScanner()`类进行实例化，以便访问最重要的方法：`scan()`。了解函数、方法或对象的工作原理的一个好方法是使用`**help()**`或`dir()`函数来查找模块中可用的方法：

![](assets/cf3f783d-4fc0-46e8-9f07-7d2461a46018.png)

如果我们执行`help (port_scan.scan)`命令，我们会看到`PortScanner`类的`scan`方法接收三个参数，主机、端口和参数，最后添加参数（所有参数都必须是字符串）。

使用`help`命令，我们可以看到信息：

![](assets/4862ae4b-1d55-43a2-bfd5-8221da7839e6.png)

我们首先要做的是导入Nmap库并创建我们的对象，以开始与`PortScanner()`进行交互。

我们使用`scan ('ip', 'ports')`方法启动我们的第一次扫描，其中第一个参数是IP地址，第二个是端口列表，第三个参数是可选的。如果我们不定义它，将执行标准的Nmap扫描：

```py
import nmap
nm = nmap.PortScanner()
results = nm.scan('192.168.56.101', '1-80','-sV')
```

在这个例子中，对具有IP地址`192.168.56.101`的虚拟机在`1-80`范围内的端口进行扫描。使用`**参数-sV**`，我们告诉你在调用扫描时检测版本。

扫描结果是一个包含与直接使用Nmap进行扫描返回的相同信息的字典。我们还可以返回到我们用`PortScanner()`类实例化的对象并测试其方法。我们可以在下一个截图中看到已执行的`nmap`命令，使用`command_line()`方法。

要获取运行在特定端口上的服务器的更多信息，我们可以使用`tcp()`方法来实现。

在这个例子中，我们可以看到如何使用`tcp`方法获取有关特定端口的信息：

![](assets/209c0f9e-bbaa-4c09-b95f-8dbe80527859.png)

我们还可以使用`state()`函数来查看主机是否处于启动状态，该函数返回我们在上一个截图中看到的状态属性：

```py
nmap['192.168.56.101'].state()
```

我们还有`all_hosts()`方法来扫描所有主机，通过它我们可以看到哪些主机是启动的，哪些是关闭的：

```py
for host in nmap.all_hosts():
    print('Host : %s (%s)' % (host, nmap[host].hostname()))
    print('State : %s' % nmap[host].state())
```

我们还可以看到在扫描过程中哪些服务给出了某种响应，以及使用的`scanning`方法：

```py
nm.scaninfo()
```

我们还扫描所有协议：

```py
for proto in nmap[host].all_protocols():
    print('Protocol : %s' % proto)
listport = nmap[host]['tcp'].keys()
listport.sort()
for port in listport:
    print('port : %s\tstate : %s' % (port,nmap[host][proto][port]['state']))
```

以下脚本尝试使用python-nmap在以下条件下进行扫描。

+   要扫描的端口：`21,22,23,80,8080`。

+   -n选项不执行DNS解析。

+   一旦获取了扫描数据，将其保存在`scan.txt`文件中。

您可以在文件名`Nmap_port_scanner.py`中找到以下代码：

```py
#!/usr/bin/python

#import nmap module
import nmap

#initialize portScanner                       
nm = nmap.PortScanner()

# we ask the user for the host that we are going to scan
host_scan = raw_input('Host scan: ')
while host_scan == "":
    host_scan = raw_input('Host scan: ')

#execute scan in portlist
portlist="21,22,23,25,80,8080"
nm.scan(hosts=host_scan, arguments='-n -p'+portlist)

#show nmap command
print nm.command_line()

hosts_list = [(x, nm[x]['status']['state']) for x in nm.all_hosts()]
#write in scan.txt file
file = open('scan.txt', 'w')
for host, status in hosts_list:
    print host, status
    file.write(host+'\n')

#show state for each port
array_portlist=portlist.split(',')
for port in array_portlist:
state= nm[host_scan]['tcp'][int(port)]['state']
    print "Port:"+str(port)+" "+"State:"+state
    file.write("Port:"+str(port)+" "+"State:"+state+'\n')

#close file
file.close()
```

`Nmap_port_scanner.py`执行：

在这个截图中，我们可以看到以指定IP地址的Metasploitable虚拟机作为参数传递的端口的状态：

![](assets/76bd821e-7d69-4d1c-a42b-7b16caa161b4.png)

# 使用python-nmap进行扫描模式

在这一部分中，我们回顾了`python-nmap`模块支持的扫描模式。`python-nmap`允许在两种模式下自动执行端口扫描任务和报告：同步和异步。使用异步模式，我们可以定义一个`callback`函数，当特定端口的扫描完成时将执行该函数，并且在此函数中，如果端口已打开，我们可以进行额外的处理，例如为特定服务（HTTP、FTP、MySQL）启动Nmap脚本。

# 同步扫描

在这个例子中，我们实现了一个允许我们扫描IP地址和作为参数传递给脚本的端口列表的类。

在主程序中，我们添加了处理输入参数所需的配置。我们执行一个循环，处理每个通过参数发送的端口，并调用`NmapScanner`类的`nmapScan(ip, port)`方法。

您可以在文件名`NmapScanner.py`中找到以下代码：

```py
import optparse, nmap

class NmapScanner:

    def __init__(self):
        self.nmsc = nmap.PortScanner()

    def nmapScan(self, host, port):
        self.nmsc.scan(host, port)
        self.state = self.nmsc[host]['tcp'][int(port)]['state']
        print " [+] "+ host + " tcp/" + port + " " + self.state

def main():
    parser = optparse.OptionParser("usage%prog " + "-H <target host> -p <target port>")
    parser.add_option('-H', dest = 'host', type = 'string', help = 'Please, specify the target host.')
    parser.add_option('-p', dest = 'ports', type = 'string', help = 'Please, specify the target port(s) separated by comma.')
    (options, args) = parser.parse_args()

    if (options.host == None) | (options.ports == None):
        print '[-] You must specify a target host and a target port(s).'
        exit(0)
    host = options.host
    ports = options.ports.split(',')

    for port in ports:
        NmapScanner().nmapScan(host, port)

if __name__ == "__main__":
    main()
```

我们可以在命令行中执行前面的脚本以显示选项：

```py
python NmapScanner.py -h
```

使用`-h`参数，我们可以查看脚本选项：

![](assets/3a18d316-58bb-4571-87f6-1c07f4a8e128.png)

这是在使用前面的参数执行脚本时的输出：

![](assets/1e37e36d-73fd-4d0d-9f45-1e948e29b36e.png)

除了执行端口扫描并通过控制台返回结果外，我们还可以生成一个JSON文档来存储给定主机的开放端口的结果。在这种情况下，我们使用`csv()`函数以便以易于收集所需信息的格式返回扫描结果。在脚本的末尾，我们看到如何调用定义的方法，通过参数传递IP和端口列表。

您可以在文件名`NmapScannerJSONGenerate.py`中找到以下代码：

```py
def nmapScanJSONGenerate(self, host, ports):
    try:
        print "Checking ports "+ str(ports) +" .........."
        self.nmsc.scan(host, ports)

        # Command info
        print "[*] Execuing command: %s" % self.nmsc.command_line()

        print self.nmsc.csv()
        results = {} 

        for x in self.nmsc.csv().split("\n")[1:-1]:
            splited_line = x.split(";")
            host = splited_line[0]
            proto = splited_line[1]
            port = splited_line[2]
            state = splited_line[4]

            try:
                if state == "open":
                    results[host].append({proto: port})
            except KeyError:
                results[host] = []
                results[host].append({proto: port})

        # Store info
        file_info = "scan_%s.json" % host
        with open(file_info, "w") as file_json:
            json.dump(results, file_json)

         print "[*] File '%s' was generated with scan results" % file_info 

 except Exception,e:
     print e
 print "Error to connect with " + host + " for port scanning" 
     pass
```

在这个截图中，我们可以看到`NmapScannerJSONGenerate`脚本的执行输出：

![](assets/151f4d1e-e673-4c60-a8c9-6a21495607f6.png)

# 异步扫描

我们可以使用`PortScannerAsync()`类执行异步扫描。在这种情况下，当执行扫描时，我们可以指定一个额外的回调参数，其中我们定义`return`函数，该函数将在扫描结束时执行：

```py
import nmap

nmasync = nmap.PortScannerAsync()

def callback_result(host, scan_result):
    print host, scan_result

nmasync.scan(hosts='127.0.0.1', arguments='-sP', callback=callback_result)
while nmasync.still_scanning():
    print("Waiting >>>")
    nmasync.wait(2)
```

通过这种方式，我们可以定义一个`回调`函数，每当Nmap对我们正在分析的机器有结果时就会执行。

以下脚本允许我们使用Nmap异步执行扫描，以便通过输入参数请求目标和端口。脚本需要做的是在`MySQL端口（3306）`上异步执行扫描，并执行MySQL服务可用的Nmap脚本。

为了测试它，我们可以在虚拟机**Metasploitable2**上运行它，该虚拟机的`3306`端口是开放的，除了能够执行Nmap脚本并获取有关正在运行的MySQL服务的附加信息。

你可以在文件名`NmapScannerAsync.py`中找到以下代码：

```py
import optparse, nmap
import json
import argparse

def callbackMySql(host, result):
    try:
        script = result['scan'][host]['tcp'][3306]['script']
        print "Command line"+ result['nmap']['command_line']
        for key, value in script.items():
            print 'Script {0} --> {1}'.format(key, value)
    except KeyError:
        # Key is not present
        pass

class NmapScannerAsync:

 def __init__(self):
        self.nmsync = nmap.PortScanner()
        self.nmasync = nmap.PortScannerAsync()

    def scanning(self):
        while self.nmasync.still_scanning():
            self.nmasync.wait(5)
```

这是检查作为参数传递的端口并以异步方式启动与MySQL相关的Nmap脚本的方法：

```py
def nmapScan(self, hostname, port):
        try:
            print "Checking port "+ port +" .........."
            self.nmsync.scan(hostname, port)
            self.state = self.nmsync[hostname]['tcp'][int(port)]['state']
            print " [+] "+ hostname + " tcp/" + port + " " + self.state 
            #mysql
            if (port=='3306') and self.nmsync[hostname]['tcp'][int(port)]['state']=='open':
                print 'Checking MYSQL port with nmap scripts......'
                #scripts for mysql:3306 open
                print 'Checking mysql-audit.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-audit.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-brute.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-brute.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-databases.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-databases.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-databases.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-dump-hashes.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-dump-hashes.nse.....'                                           self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-empty-password.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-enum.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-enum.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-info.nse".....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-info.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-query.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-query.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-users.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-users.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-variables.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-variables.nse",callback=callbackMySql)
                self.scanning()

                print 'Checking mysql-vuln-cve2012-2122.nse.....'
                self.nmasync.scan(hostname,arguments="-A -sV -p3306 --script mysql-vuln-cve2012-2122.nse",callback=callbackMySql)
                self.scanning()

    except Exception,e:
        print str(e)
        print "Error to connect with " + hostname + " for port scanning"
        pass

```

这是我们的主程序，用于请求目标和端口作为参数，并为每个端口调用`nmapScan(ip,port)`函数：

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nmap scanner async')
    # Main arguments
    parser.add_argument("-target", dest="target", help="target IP / domain", required=True)
    parser.add_argument("-ports", dest="ports", help="Please, specify the target port(s) separated by comma[80,8080 by default]", default="80,8080")
    parsed_args = parser.parse_args()   
    port_list = parsed_args.ports.split(',')
    ip = parsed_args.target
    for port in port_list:
        NmapScannerAsync().nmapScan(ip, port)
```

现在我们将使用目标和端口参数执行**NmapScannerAsync**：

![](assets/51b57cc9-9cee-40c0-be75-1d1b5ac14250.png)

# Nmap脚本中的漏洞

在本节中，我们将回顾`python-nmap`模块支持的扫描模式。我们将学习如何检测系统或网络段的开放端口，以及执行高级操作以收集有关其目标的信息，并检测FTP服务中的漏洞。

# 执行Nmap脚本以检测漏洞

Nmap最有趣的功能之一是执行符合**Nmap脚本引擎（NSE）**规范的脚本的能力。Nmap使您能够进行漏洞评估和利用，这要归功于其强大的Lua脚本引擎。通过这种方式，我们还可以执行更复杂的例程，允许我们过滤有关特定目标的信息。

目前，它包括使用脚本来检查一些最知名的漏洞：

+   **Auth：**执行所有可用的认证脚本

+   **默认：**默认情况下执行工具的基本脚本

+   **发现：**从目标或受害者中检索信息

+   **外部：**使用外部资源的脚本

+   **侵入式：**使用被认为对受害者或目标具有侵入性的脚本

+   **恶意软件：**检查是否有恶意代码或后门打开的连接

+   **安全：**执行不具侵入性的脚本

+   **Vuln：**发现最知名的漏洞

+   **全部：**执行所有可用的NSE扩展脚本

为了检测开放的端口服务可能存在的漏洞，我们可以利用模块安装时可用的Nmap脚本。在**UNIX**机器上，脚本位于路径：`/usr/share/nmap/scripts.`

在**Windows**机器上，脚本位于路径：**C:\Program Files (x86)\Nmap\scripts**.

脚本允许编程例程以查找给定主机可能存在的漏洞。脚本可以在以下URL中找到：

[https://nmap.org/nsedoc/scripts](https://nmap.org/nsedoc/scripts)

对于我们想要了解更多的每种类型的服务，都有很多脚本。甚至有一些允许使用字典或暴力攻击，并利用机器暴露的一些服务和端口中的某些漏洞。

要执行这些脚本，需要在`nmap`命令中传递**--script选项**。

在这个例子中，我们使用认证脚本（`auth`）执行Nmap，它将检查是否有空密码的用户或默认存在的用户和密码。

使用这个命令，它可以在MySQL和web服务器（tomcat）的服务中找到用户和密码：

```py
nmap -f -sS -sV --script auth 192.168.56.101
```

在这个例子中，显示了**mysql端口3306**允许使用空密码连接到root帐户。它还显示了从端口`80`收集的信息，例如计算机名称和操作系统版本（Metasploitable2 - Linux）：

![](assets/90b57a41-d3b0-4a84-8c3e-267f25a4e12d.png)

Nmap还包含的另一个有趣的脚本是**discovery**，它允许我们了解有关我们正在分析的虚拟机上运行的服务的更多信息。

通过`discovery`选项，我们可以获取有关在虚拟机上运行的应用程序相关的服务和路由的信息：

![](assets/d9b6cec9-5f46-46d0-8c3d-3efff85a14f7.png)

# 检测FTP服务中的漏洞

如果我们在端口`21`上在目标机器上运行**ftp-anon脚本**，我们可以知道FTP服务是否允许匿名身份验证而无需输入用户名和密码。在这种情况下，我们看到FTP服务器上确实存在这样的身份验证：

![](assets/38c765d5-5135-4478-b027-4ff95611be2d.png)

在下面的脚本中，我们以异步方式执行扫描，以便我们可以在特定端口上执行扫描并启动并行脚本，因此当一个脚本完成时，将执行`defined`函数。在这种情况下，我们执行为FTP服务定义的脚本，每次从脚本获得响应时，都会执行**`callbackFTP`**函数，这将为我们提供有关该服务的更多信息。

您可以在文件名`NmapScannerAsync_FTP.py`中找到以下代码：

```py
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import optparse, nmap
import json
import argparse

def callbackFTP(host, result):
    try:
        script = result['scan'][host]['tcp'][21]['script']
        print "Command line"+ result['nmap']['command_line']
        for key, value in script.items():
            print 'Script {0} --> {1}'.format(key, value)
    except KeyError:
        # Key is not present
        pass

class NmapScannerAsyncFTP:

    def __init__(self):
        self.nmsync = nmap.PortScanner()
        self.nmasync = nmap.PortScannerAsync()

    def scanning(self):
        while self.nmasync.still_scanning():
            self.nmasync.wait(5)
```

这是检查传递的端口并以异步方式启动与FTP相关的Nmap脚本的方法：

```py

    def nmapScanAsync(self, hostname, port):
        try:
            print "Checking port "+ port +" .........."
            self.nmsync.scan(hostname, port)
            self.state = self.nmsync[hostname]['tcp'][int(port)]['state']
            print " [+] "+ hostname + " tcp/" + port + " " + self.state 

             #FTP
             if (port=='21') and self.nmsync[hostname]['tcp'][int(port)]['state']=='open':
                print 'Checking ftp port with nmap scripts......'
                #scripts for ftp:21 open
                print 'Checking ftp-anon.nse .....'
                self.nmasync.scan(hostname,arguments="-A -sV -p21 --script ftp-anon.nse",callback=callbackFTP)
                self.scanning()
                print 'Checking ftp-bounce.nse .....'
                self.nmasync.scan(hostname,arguments="-A -sV -p21 --script ftp-bounce.nse",callback=callbackFTP)
                self.scanning()
                print 'Checking ftp-brute.nse .....'
                self.nmasync.scan(hostname,arguments="-A -sV -p21 --script ftp-brute.nse",callback=callbackFTP)
                self.scanning()
                print 'Checking ftp-libopie.nse .....'
                self.nmasync.scan(hostname,arguments="-A -sV -p21 --script ftp-libopie.nse",callback=callbackFTP)
                self.scanning()
                print 'Checking ftp-proftpd-backdoor.nse .....'
                self.nmasync.scan(hostname,arguments="-A -sV -p21 --script ftp-proftpd-backdoor.nse",callback=callbackFTP)
                self.scanning()
                print 'Checking ftp-vsftpd-backdoor.nse .....'
                self.nmasync.scan(hostname,arguments="-A -sV -p21 --script ftp-vsftpd-backdoor.nse",callback=callbackFTP)
                self.scanning()

    except Exception,e:
        print str(e)
        print "Error to connect with " + hostname + " for port scanning" 
        pass

```

这是我们的主程序，用于请求目标和端口作为参数，并调用`nmapScanAsync(ip,port)`函数来处理每个端口：

```py
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nmap scanner async')
    # Main arguments
    parser.add_argument("-target", dest="target", help="target IP / domain", required=True)
    parser.add_argument("-ports", dest="ports", help="Please, specify the target port(s) separated by comma[80,8080 by default]", default="80,8080")

    parsed_args = parser.parse_args()

    port_list = parsed_args.ports.split(',')

    ip = parsed_args.target

    for port in port_list:
        NmapScannerAsyncFTP().nmapScanAsync(ip, port)
```

现在，我们将使用目标和端口参数执行**NmapScannerAsync_fFTP**。

在这种情况下，我们对FTP端口（`21`）进行扫描，我们可以看到它执行了为该端口定义的每个脚本，并返回了更多信息，我们可以在以后的攻击或利用过程中使用。

我们可以通过执行上一个脚本来获取有关FTP易受攻击服务的信息：

```py
python NmapScannerAsync.py -target 192.168.56.101 -ports 21
```

![](assets/134612ea-bb70-4217-a4ef-9127fddd1f81.png)

# 总结

这个主题的一个目标是了解允许在特定域或服务器上执行端口扫描的模块。在Python中执行端口扫描的最佳工具之一是python-nmap，它是`nmap`命令的包装器模块。还有其他选择，比如Scrapy，也可以很好地完成这些任务，并且还允许我们更深入地了解这些工具的工作原理。

在下一章中，我们将更多地探讨与Metasploit框架交互的编程包和Python模块，以利用漏洞。

# 问题

1.  哪种方法允许我们查看已被扫描的机器？

1.  如果我们想要执行异步扫描并在扫描结束时执行脚本，调用`scan`函数的方法是什么？

1.  我们可以使用哪种方法以字典格式获取扫描结果？

1.  用于执行异步扫描的`Nmap`模块是什么类型？

1.  用于执行同步扫描的`Nmap`模块是什么类型？

1.  如果我们使用指令`self.nmsync = nmap.PortScanner()`初始化对象，我们如何在给定主机和给定端口上启动同步扫描？

1.  我们可以使用哪种方法来检查特定网络中的主机是否启动？

1.  使用`PortScannerAsync()`类进行异步扫描时，需要定义哪个函数？

1.  如果我们需要知道FTP服务是否允许匿名身份验证而无需输入用户名和密码，我们需要在端口`21`上运行哪个脚本？

1.  如果我们需要知道MySQL服务是否允许匿名身份验证而无需输入用户名和密码，我们需要在端口`3306`上运行哪个脚本？

# 进一步阅读

在这些链接中，您将找到有关先前提到的工具的更多信息，以及我们用于脚本执行的Metasploitable虚拟机的官方文档。

+   [http://xael.org/pages/python-nmap-en.html](http://xael.org/pages/python-nmap-en.html)

+   [https://nmap.org/nsedoc/scripts](https://nmap.org/nsedoc/scripts)

+   [https://metasploit.help.rapid7.com/docs/metasploitable-2-exploitability-guide](https://metasploit.help.rapid7.com/docs/metasploitable-2-exploitability-guide)

+   [https://information.rapid7.com/download-metasploitable-2017.html](https://information.rapid7.com/download-metasploitable-2017.html)

+   [https://media.blackhat.com/bh-us-10/whitepapers/Vaskovitch/BlackHat-USA-2010-Fyodor-Fifield-NMAP-Scripting-Engine-wp.pdf](https://media.blackhat.com/bh-us-10/whitepapers/Vaskovitch/BlackHat-USA-2010-Fyodor-Fifield-NMAP-Scripting-Engine-wp.pdf)

+   SPARTA端口扫描：[https://sparta.secforce.com](https://sparta.secforce.com)

SPARTA是一个用Python开发的工具，允许进行端口扫描、渗透测试和安全检测，用于检测已打开的服务，并与Nmap工具集成进行端口扫描。SPARTA将要求您指定要扫描的IP地址范围。扫描完成后，SPARTA将识别任何机器，以及任何打开的端口或正在运行的服务。
