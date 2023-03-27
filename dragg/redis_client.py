import os

from redis import StrictRedis

_connection = None

def connection(url = "redis://localhost"):
    """Return the Redis connection to the URL given by the environment
    variable REDIS_URL, creating it if necessary.

    """
    global _connection
    if _connection is None:
        _connection = StrictRedis.from_url(url, decode_responses=True)
    return _connection