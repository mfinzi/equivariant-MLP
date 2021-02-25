import pickle, atexit
import logging

import time
import io

from tqdm.auto import tqdm
tqdm.get_lock().locks = []

prod = lambda c: reduce(lambda a,b:a*b,c)

# class TqdmToLogger(io.StringIO):
#     """
#         Output stream for TQDM which will output to logger module instead of
#         the StdOut.
#     """
#     logger = None
#     level = None
#     buf = ''
#     def __init__(self,logger,level=None):
#         super(TqdmToLogger, self).__init__()
#         self.logger = logger
#         self.level = level or logging.INFO
#     def write(self,buf):
#         self.buf = buf.strip('\r\n\t ')
#     def flush(self):
#         self.logger.log(self.level, self.buf)
#)

log_levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                        'warn': logging.WARNING,'warning': logging.WARNING,
                        'info': logging.INFO,'debug': logging.DEBUG}

def ltqdm(*args,level='info',**kwargs):
    if logging.root.level<=log_levels[level.lower()]:
        return tqdm(*args,**kwargs)
    else:
        return args[0]

# class NoCache(object):
#     def __enter__(self):
#         self.settings = CacheSettings.disk_caching
#         CacheSettings.disk_caching=False
#         return self
#     def __exit__(self, *exc):
#         CacheSettings.disk_caching=self.settings
#         return False

# class CacheSettings(object):
#     memory_caching=True
#     disk_caching=True

# def make_key(args, kwds, kwd_mark = (object(),)):
#     key = args
#     if kwds:
#         key += kwd_mark
#         for item in kwds.items():
#             key += item
#     return key



# def disk_cache(file_name):
#     try:
#         with open(file_name, 'rb') as f:
#             cache = pickle.load(f)
#     except (IOError, ValueError):
#         cache = {}

#     atexit.register(lambda: pickle.dump(cache, open(file_name, 'wb')))

#     def decorator(func):
#         def new_func(*args,**kwargs):
#             if not CacheSettings.disk_caching: return func(*args,**kwargs)
#             key = make_key(args,kwargs)
#             if key not in cache:
#                 logging.info(f"{key} cache miss")
#                 cache[key] = func(*args,**kwargs)
#             logging.debug(f"{key} cache hit")
#             return cache[key]
#         return new_func

#     return decorator