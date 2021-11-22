import argparse
from ib_insync import util, Forex, Future, Stock
from datetime import datetime, timedelta
from utils import SQLConnector, IBConnector, saveDF, timeSectorToDateSet


parser = argparse.ArgumentParser(description='Load data from IB.')

parser.add_argument('instrument', nargs=1, help='Name of instrument you want to download')
parser.add_argument('type', nargs=1, help='Type of instrument, one of: Forex, Future, Stock')
parser.add_argument('start', nargs=1, help='Start of period to be downloaded, format yyyymmdd')
parser.add_argument('end', nargs=1, help='End of period to be downloaded, format yyyymmdd')
parser.add_argument('-f', '--file', nargs='?', help='Filename of saved data(default is instrument name)')

args = vars(parser.parse_args())

instrument = args['instrument'][0]
type_ = args['type'][0]
start = datetime.strptime(args['start'][0]+'-00:00:00', '%Y%m%d-%H:%M:%S')
end = datetime.strptime(args['end'][0]+'-23:59:00', '%Y%m%d-%H:%M:%S')

requestedDateList = timeSectorToDateSet(start, end)




sqlConnection = SQLConnector()
sqlConnection.connect('23.23.90.225', 'research', '0xResearch', 'ib_data')

availableDateList = sqlConnection.loadDaysList(instrument)

if requestedDateList.issubset(availableDateList):
    data = sqlConnection.loadData(instrument, start, end)
    print('Data downloaded.')
    saveDF(data, instrument, args['file'])
    exit()
    
answer = input('Data not fully present on a server. Download from IB?[Y/n]')

if answer != '' and answer.lower() != 'y':
    print('Ok. Not downloading.')
    exit()








missingDateList = requestedDateList - availableDateList

try:
    ibConnection = IBConnector()
    ibConnection.connect()
except:
    exit()

if type_.lower() == 'forex':
    contract = Forex(instrument)
elif type_.lower() == 'stock':
    contract = Stock(instrument)
else:
    print('For now only Forex and Stock are implemented')
    exit()

print('Loading available data from database')
data = sqlConnection.loadData(instrument, start, end)
print('Done')
saveDF(data, instrument, args['file'])

print('Loading missing data from IB')
data_ib = ibConnection.loadDataDaysList(contract, missingDateList)
instrument_id = sqlConnection.createInstrument(instrument, type_)
data_ib['instrument'] = instrument_id
print('Data loaded.')

data_ib = data_ib[~data_ib['time'].isin(set(data['time']))]
print('Saving data to database')
sqlConnection.saveData(data_ib)
print('Data saved. Saving locally')

data = data.append(data_ib)
saveDF(data, instrument, args['file'])
print('All done')



