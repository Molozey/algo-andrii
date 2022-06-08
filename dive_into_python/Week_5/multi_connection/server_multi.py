import socket
import threading


def process_request(conn, addr):
    print('connected user:', addr)
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(data.decode('utf-8'))


with socket.socket() as sock:
    sock.bind(("127.0.0.1", 10001))
    sock.listen()
    while True:
        conn, addr = sock.accept()
        th = threading.Thread(target=process_request, args=(conn, addr))
        th.start()
