要配置SSH仅允许通过密钥认证而禁止密码登录，可以按照以下步骤操作：

	1.	生成SSH密钥对（如果还没有密钥）：
在客户端机器上运行以下命令生成SSH密钥对：

ssh-keygen -t rsa -b 4096

按提示选择密钥存放路径（默认是 ~/.ssh/id_rsa）并设置密钥的密码（可选）。

	2.	将公钥上传到服务器：
使用以下命令将公钥上传到服务器：

ssh-copy-id username@server_ip

或者手动将~/.ssh/id_rsa.pub内容添加到服务器的~/.ssh/authorized_keys文件中。

	3.	修改SSH配置文件：
在服务器上编辑SSH配置文件 /etc/ssh/sshd_config：

sudo nano /etc/ssh/sshd_config

然后，确保以下选项的设置：

PasswordAuthentication no
PubkeyAuthentication yes

这些设置将禁用密码认证，仅允许公钥认证。

	4.	重启SSH服务：
修改配置文件后，重启SSH服务以使更改生效：
	•	在Debian/Ubuntu上：

sudo systemctl restart ssh


	•	在CentOS/Fedora上：

sudo systemctl restart sshd


	5.	测试连接：
在客户端测试通过密钥的SSH连接，以确保配置成功。