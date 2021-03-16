def validate(*params_name):
    def check_accepts(f):
        def new_f(*args, **kwds):
            parameters = None
            for arg in args:
                if isinstance(arg, dict):
                    parameters = arg
            for param in params_name:
                if parameters is None:
                    raise Exception("Missing parameters in function {}".format(f.__name__))
                if parameters.get(param) is None:
                    raise Exception("Missing parameter {} in function {}".format(param, f.__name__))
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f
    return check_accepts
