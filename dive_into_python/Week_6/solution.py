import asyncio

g_store = dict()

class ClientServerProtocol(asyncio.Protocol):

    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        self.transport.write(self._process(data.decode('utf-8').strip('\r\n')).encode('utf-8'))
        buffer = data.decode('utf-8').strip('\r\n').split(' ')
        cmd = buffer[0]



    def _process(self, data):
        cmd = data.split(' ')
        if cmd[0] == 'get':
            return self._process_get(cmd[1])
        if cmd[0] == 'put':
            return self._process_put(cmd[1], cmd[2], cmd[3])
        else:
            return 'error\nwrong command\n\n'

    def _process_get(self, obj):
        response = 'ok\n'
        if obj == '*':
            for key, values in g_store.items():
                for value in values:
                    response = response + key + ' ' + value[1] + ' ' + value[0] + '\n'
        else:
            if obj in g_store:
                for value in g_store[obj]:
                    response = response + obj + ' ' + value[1] + ' ' + value[0] + '\n'

        return response + '\n'

    def _process_put(self, obj, value, timestamp=None):
        if obj == '*':
            return 'error\nkey cannot contain *\n\n'
        if not obj in g_store:
            g_store[obj] = list()
        if not (timestamp, value) in g_store[obj]:
            g_store[obj].append((timestamp, value))
            g_store[obj].sort(key=lambda tup: tup[0])
        return 'ok\n\n'

def run_server(self, host, port):
    loop = asyncio.get_event_loop()
    coro = loop.create_server(ClientServerProtocol, host, port)
    server = loop.run_until_complete(coro)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()

#if __name__ == '__main__':
#    run_server('127.0.0.1', 8888)