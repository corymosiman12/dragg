from dragg.aggregator import Aggregator
import argparse

REDIS_URL = 'redis://localhost'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--redis", help="Redis host URL", default=REDIS_URL)

    args = parser.parse_args()

    a = Aggregator(redis_url=args.redis)
    a.run()