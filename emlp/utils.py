import pickle, atexit

class CacheSettings(object):
    memory_caching=True
    disk_caching=True

def make_key(args, kwds, kwd_mark = (object(),)):
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    return key

def disk_cache(file_name):
    try:
        with open(file_name, 'rb') as f:
            cache = pickle.load(f)
    except (IOError, ValueError):
        cache = {}

    atexit.register(lambda: pickle.dump(cache, open(file_name, 'wb')))

    def decorator(func):
        def new_func(*args,**kwargs):
            if not CacheSettings.disk_caching: return func(*args,**kwargs)
            key = make_key(args,kwargs)
            if key not in cache:
                cache[key] = func(*args,**kwargs)
            return cache[key]
        return new_func

    return decorator