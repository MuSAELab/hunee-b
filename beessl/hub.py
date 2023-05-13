from upstream.baseline.hubconf import *

def options():
    all_options = []
    for name, value in globals().items():
        if not name.startswith("_") and callable(value) and name != "options":
            all_options.append(name)

    return all_options
