import argparse
from ib_insync import util, Forex, Future
from datetime import datetime, timedelta
from utils import loadData, connect




parser = argparse.ArgumentParser(description='Load data from IB.')

parser.add_argument('instrument', nargs=1, help='Name of instrument you want to download')
parser.add_argument('type', nargs=1, help='Type of instrument, one of: Forex, Future, Stock')
parser.add_argument('start', nargs=1, help='Start of period to be downloaded, format yyyymmdd-HH:MM:SS')
parser.add_argument('end', nargs=1, help='End of period to be downloaded, format yyyymmdd-HH:MM:SS')
parser.add_argument('-f', '--file', nargs='?', help='Filename of saved data(default is instrument name)')

args = vars(parser.parse_args())

instrument = args['instrument'][0]
type_ = args['type'][0]
start = datetime.strptime(args['start'][0], '%Y%m%d-%H:%M:%S')
end = datetime.strptime(args['end'][0], '%Y%m%d-%H:%M:%S')






ib = connect()

# 3.10 only
# match type_.lower():
#     case 'forex':
#         contract = Forex(instrument)
#     case 'stock':
#         contract = Stock(instrument)
#     case _:
#         print('For now only Forex and Stock are implemented')
#         exit()

if type_.lower() == 'forex':
    contract = Forex(instrument)
elif type_.lower() == 'stock':
    contract = Stock(instrument)
else:
    print('For now only Forex and Stock are implemented')
    exit()

data = loadData(ib, contract, start, end)

print('Data loaded. Saving')

if args['file'] is not None:
    data.to_csv(args['file'], index=False)
else:
    data.to_csv(f'{instrument}.csv', index=False)
