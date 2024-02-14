from upstream.baseline.hubconf import *
from upstream.beeyol.hubconf import *
from upstream.byola.hubconf import *

def options():
    all_options = set()
    for name, value in globals().items():
        if not name.startswith("_") and callable(value) and name != "options":
            all_options.add(name)

    all_options.remove("UpstreamExpert")
    all_options.remove("load_hyperpyyaml")

    return all_options
