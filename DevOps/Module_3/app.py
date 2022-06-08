import paramiko
import re
import time
import os

def solution():
    HOST = "127.0.0.1"
    PORT = "52022"
    USER = "root"

    session = paramiko.SSHClient()
    session.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    session.connect(
        hostname=HOST,
        port=PORT,
        username=USER,
        pkey=paramiko.RSAKey.from_private_key_file('/Users/molozey/.ssh/module3')
    )
    stdin, stdout, stderr = session.exec_command('cat /var/log/bootstrap.log')
    time.sleep(.5)
    string = str(stdout.read().decode())


    matches = re.findall('2021-09-21 16:48:40.+', string)
    with open(f"result.txt", 'w') as f:
        for _ in matches[:-1]:
            f.write(_ + '\n')
        f.write(matches[-1])
    session.close()

solution()
