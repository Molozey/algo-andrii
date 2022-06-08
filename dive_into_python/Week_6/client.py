import socket
import time

class ClientError(Exception):
    pass


class Client:
    def __init__(self, ip, port, timeout=None):
        self.ip = ip
        self.port = int(port)
        if timeout:
            self.timeout = int(timeout)

    def send(self, request):
        with socket.create_connection((self.ip, self.port), self.timeout) as sock:
            sock.sendall(request.encode('utf-8'))
            answer = sock.recv(1024)
            return answer.decode('utf-8')

    def put(self, metric_name, value, timestamp=None):
        resp = self.send('put ' + metric_name + ' ' + str(value) + ' ' + str(timestamp if timestamp else int(time.time())) + '\n')
        if resp[0:3] != 'ok\n':
            raise ClientError(resp)

    def get(self, metric_name):
        resp = self.send('get ' + metric_name + '\n')
        if resp[0:3] != 'ok\n':
            raise ClientError(resp)
        if resp != 'ok\n\n':
            _ = []
            buffer = resp.split('\n')
            _.append(buffer[0])
            try:
                _.extend(buffer[1].split(' '))
            except:
                raise ClientError(resp)
            if len(_) < 3:
                raise ClientError(resp)

        answer = dict()
        lines = resp.split('\n')
        for _ in lines[1:-2]:
            metric = _.split(' ')
            answer_key = metric[0]
            try:
                answer_value = float(metric[1])
                answer_time = int(metric[2])
            except:
                raise ClientError(resp)

            if not answer_key in answer:
                answer[answer_key] = list()
            answer[answer_key].append((answer_time, answer_value))
            answer[answer_key].sort(key=lambda cort: cort[0])

        return answer


    @classmethod
    def request_chosen(cls, input_string):
        req_str = input_string.split(' ')
        command = req_str[0]
        value = req_str[1][:-1]
        return command, value


client = Client(ip='127.0.0.1', port=8888, timeout=5)
client.put(metric_name='alm.cpu', value=100, timestamp=203040340)
print(client.get(metric_name='alm.cpu'))
