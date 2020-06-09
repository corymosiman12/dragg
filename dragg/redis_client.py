import os
import redis

class Singleton(type):

    _instances = {}

    def __call__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__()
        return cls._instances[cls]

class RedisClient(metaclass=Singleton):

    def __init__(self):
        self.pool = redis.ConnectionPool(host = os.environ.get('REDIS_HOST', 'localhost'), decode_responses = True, db = 0)

    @property
    def conn(self):
        if not hasattr(self, '_conn'):
            self.getConnection()
        return self._conn

    def getConnection(self):
        self._conn = redis.Redis(connection_pool = self.pool)
