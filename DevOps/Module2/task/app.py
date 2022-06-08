import paramiko
import time

def solution_password():
    host = '127.0.0.1'
    port = '52022'
    user = 'root'
    password = 'test'

    session = paramiko.SSHClient()
    session.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    session.connect(
        hostname=host,
        port=port,
        username=user,
        password=password
    )

    stdin, stdout, stderr = session.exec_command('cat /root/pwd.txt')
    time.sleep(1)
    print(stdout.read().decode())
    session.close()

def solution():
    host = '127.0.0.1'
    port = '52022'
    user = 'root'

    session = paramiko.SSHClient()
    session.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_file = paramiko.RSAKey.from_private_key_file('/Users/molozey/.ssh/DevOpsTask')

    session.connect(
        hostname=host,
        port=port,
        username=user,
        pkey=key_file
    )

    stdin, stdout, stderr = session.exec_command('cat /root/pwd.txt')
    time.sleep(1)
    session.close()
    with open('result.txt', 'w') as result:
        result.write(str(stdout.read().decode()))
    return stdout.read().decode()

solution()