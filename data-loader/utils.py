from datetime import datetime, timedelta
from ib_insync import util, IB

def connect():
    # util.startLoop()
    try:
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
    except:
        exit()
    return ib

def loadData(ib, contract, start, end, barSize='1 min'):
    bars = ib.reqHistoricalData(
        contract, endDateTime=end.strftime('%Y%m%d %H:%m:%S'), 
        durationStr='1 D', barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
    end -= timedelta(days=1)
    df = util.df(bars)
    k = 0
    while end > start:
        print('Got ' +end.strftime('%Y-%m-%d'))
        bars = ib.reqHistoricalData(
            contract, endDateTime=end.strftime('%Y%m%d %H:%m:%S'), 
            durationStr='1 D', barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
        end -= timedelta(days=1)
        df = df.append(util.df(bars))
        k += 1
        # checkpoints
#         if k % 100 == 0:
#             df.drop(['volume', 'average', 'barCount'], axis=1).sort_values('date', axis=0).to_csv(f'{name}_{k/100}.csv', index=False)
    df.rename(columns={'date' : 'time'}, inplace=True)
    return df.sort_values('time', axis=0).drop_duplicates(subset=['time'])