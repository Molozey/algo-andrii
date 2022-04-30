import pandas as pd


class Strategy:
    def __init__(self, data):
        self.data = data
        self.TimeBorder = pd.Timedelta('300T')
        self.intTimeBorder = self.TimeBorder // '1T'
        self.Y_threshold = 3    # In sigmas
        self.scanTime = '400T'  # For collecting half-time

    def calculate_half_times(self):
        pass

