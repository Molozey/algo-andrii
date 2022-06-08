import socket

with socket.socket() as sock:
    sock.bind(('127.0.0.1', 10001))
    sock.listen()

    while True:
        conn, addr = sock.accept()
        with conn:
            while True:
                data = conn.recv(1024)
                if data.decode('utf-8') == 'BABABOY':
                    print('Партия гордится тобой')
                if not data:
                    break
                print(data.decode('utf-8'))
