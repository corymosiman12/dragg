# import os
# import redis

# class Singleton(type):

#     _instances = {}

#     def __call__(cls, arg):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(arg)
#         return cls._instances[cls]

# class RedisClient(metaclass=Singleton):

#     def __init__(self, redis_url="redis://localhost"):
#         self.pool = redis.ConnectionPool(host = redis_url, decode_responses = True, db = 0)

#     @property
#     def conn(self):
#         if not hasattr(self, '_conn'):
#             self.getConnection()
#         return self._conn

#     def getConnection(self):
#         self._conn = redis.Redis(connection_pool = self.pool)


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