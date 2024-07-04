import cProfile
from functools import wraps


def profile(output_file="profile_results.prof"):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            result = prof.runcall(func, *args, **kwargs)
            prof.dump_stats(output_file)
            return result

        return wrapper

    return inner
