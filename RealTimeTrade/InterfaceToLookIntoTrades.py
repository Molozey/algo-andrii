from connectorInterface import SaxoOrderInterface
import time
import pprint


saxo = SaxoOrderInterface()
while True:
    print(saxo.portfolio_open_positions())
    time.sleep(5)

