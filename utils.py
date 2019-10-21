from datetime import datetime
import numpy as np
import os
import csv
import signal
import logging
import io

class Scaler(object):
    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Logger(object):
    def __init__(self, logname, date, time, loglevel):
        path = os.path.join('logs', date, logname)
        if not os.path.exists(path) :
            os.makedirs(path)
        path1 = os.path.join(path, time + '.csv')
        path2 = os.path.join(path, time + '.log')

        self.write_header = True
        self.log_entry = {}
        self.log_entry2 = {}
        self.f1 = open(path1, 'a')
        self.writer = None  # DictWriter created with first call to write() method

        self.loglevel = loglevel
        print(path2)
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path2)
        print(filename)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',level=self.GetLogLevel(loglevel))
        logging.debug("Test message")

    def GetLogLevel(self, loglevel):
        if loglevel == 1 :
            return logging.DEBUG
        elif  loglevel == 2 :
            return logging.INFO
        elif  loglevel == 3 :
            return logging.WARNING
        elif  loglevel == 4 :
            return logging.ERROR
        elif  loglevel == 5 :
            return logging.CRITICAL

    def write(self, display=True):
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f1, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        Time = datetime.now().strftime("%H-%M-%S")
        print("Time", Time, end = '')
        print(' Episode {}, Mean R = {:.1f} '.format(log['_Episode'],
                                                               log['_MeanReward']), end = '')
        logging.critical('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                                          log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print(' {:s}: {:.3g} '.format(key, log[key]), end = '')
        #print('\n')

    def logCSV(self, items):
        self.log_entry.update(items)
        self.log_entry2.update(items)
        self.logCSV2(self.log_entry2)
        self.log_entry2 = {}

    def logCSV2(self, log):
        log_keys = [k for k in log.keys()]
        log_keys.sort()

        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                logging.info('{:s}: {:.3g}'.format(key, log[key]))
        self.log_entry2 = {}

    def Debug(self, *args, end='', **kwargs):
        sio = io.StringIO()
        print(*args, **kwargs, end=end, file=sio)
        if self.loglevel <= 1 :
            print(sio.getvalue())
        logging.debug(sio.getvalue())

    def Info(self, *args, end='', **kwargs):
        sio = io.StringIO()
        print(*args, **kwargs, end=end, file=sio)
        if self.loglevel <= 2 :
            print(sio.getvalue())
        logging.info(sio.getvalue())

    def Warning(self, *args, end='', **kwargs):
        sio = io.StringIO()
        print(*args, **kwargs, end=end, file=sio)
        if self.loglevel <= 3 :
            print(sio.getvalue())
        logging.warning(sio.getvalue())

    def Error(self, *args, end='', **kwargs):
        sio = io.StringIO()
        print(*args, **kwargs, end=end, file=sio)
        if self.loglevel <= 4:
            print(sio.getvalue())
        logging.error(sio.getvalue())

    def Critical(self, *args, end='', **kwargs):
        sio = io.StringIO()
        print(*args, **kwargs, end=end, file=sio)
        if self.loglevel <= 5 :
            print(sio.getvalue())
        logging.critical(sio.getvalue())

    def close(self):
        self.f1.close()

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

