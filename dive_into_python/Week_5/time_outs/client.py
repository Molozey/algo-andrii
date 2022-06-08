import socket

with socket.create_connection(('127.0.0.1', 10001), 5) as sock:
    sock.settimeout(20)
    try:
        sock.sendall('bigtime'.encode('utf-8'))
    except socket.timeout:
        print('send data timeout')
    except socket.error as ex:
        print('sending error', ex)
