import argparse
from ib_insync import util, Forex, Future, Stock
from datetime import datetime, timedelta, date
from utils import SQLConnector, IBConnector, saveDF, timeSectorToDates


parser = argparse.ArgumentParser(description='Load data from IB.')

parser.add_argument('instrument', nargs='?', help='Name of instrument you want to download')
parser.add_argument('start', nargs='?', help='Start of period to be downloaded, format yyyymmdd. Default: -inf')
parser.add_argument('end', nargs='?', help='End of period to be downloaded, format yyyymmdd. Default: +inf')
parser.add_argument('-f', '--file', nargs='?', help='Filename of saved data. Default is instrument name')

args = vars(parser.parse_args())

sqlConnection = SQLConnector()
sqlConnection.connect('23.23.90.225', 'research', '0xResearch', 'ib_data')

if args['instrument'] == None:
    df = sqlConnection.loadIndex()
    print('Use -h to get info about a program.')
    print('Currently available instruments:')
    print(df)

    exit()

instrument = args['instrument']

if args['start'] == None:
    start = None
else:
    start = datetime.strptime(args['start'], '%Y%m%d').date()

if args['end'] == None:
    end = None
else:
    end = datetime.strptime(args['end'], '%Y%m%d').date()

fname = args['file']

df = sqlConnection.loadData(instrument, start, end)

saveDF(df, instrument, fname)

