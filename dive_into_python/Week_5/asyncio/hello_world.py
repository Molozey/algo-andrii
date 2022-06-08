import asyncio


@asyncio.coroutine
def hello_world_old():
    while True:
        print('Hello world')
        yield from asyncio.sleep(1)


async def hello_world_new():
    while True:
        print('Hello World!')
        await asyncio.sleep(1)

loop = asyncio.get_event_loop()
loop.run_until_complete(hello_world_new())
loop.close()
