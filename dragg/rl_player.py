import os
import numpy as np
import cvxpy as cp
from redis import StrictRedis
import redis
import scipy.stats
import logging
import pathos
from collections import defaultdict
import json
from copy import deepcopy
import gym

import asyncio
import aioredis
import async_timeout

from dragg.redis_client import RedisClient
from dragg.logger import Logger
from dragg.mpc_calc import MPCCalc

def manage_home(home):
    """
    Calls class method as a top level function (picklizable by pathos)
    :return: None
    """
    home.step()#np.random.uniform(19,22))
    return

class RLPlayer(gym.Env):
    def __init__(self):#, home):
        # home = self.set_home()
        # self.home = MPCCalc(home)
        # self.name = self.home.name
        with open('data/rl_data/state_action.json','r') as file:
            states_actions = json.load(file)
        self.states = [k for k, v in states_actions['states'].items() if v]
        self.actions = [k for k, v in states_actions['actions'].items() if v]

    def set_home(self):
        self.redis_client = RedisClient()
        home = self.redis_client.conn.hgetall("home_values")
        home['hvac'] = self.redis_client.conn.hgetall("hvac")
        home['wh'] = self.redis_client.conn.hgetall("wh")
        home['hems'] = self.redis_client.conn.hgetall("hems")
        home['hems']["weekday_occ_schedule"] = [[19,8],[17,18]]
        if 'battery' in home['type']:
            home['battery'] = self.redis_client.conn.hgetall("battery")
        if 'pv' in home['type']:
            home['pv'] = self.redis_client.conn.hgetall("pv")
        return home

    def get_obs(self):
        """ gets the corresponding values for each of the desired state values, as set in state_action.json
        :return: list of float values"""
        obs = []
        for state in self.states:
            if state in self.home.optimal_vals.keys():
                obs += [self.home.optimal_vals[state]]
            else:
                obs += [0]

        return obs

    def get_reward(self):
        """ determines a reward, function can be redefined by user
        :return: float value normalized to [-1,1] """
        reward = 1#self.redis_client.conn.hget("current_demand")

        return reward

    # def step(self, action=None):
    #     """redefines the OpenAI Gym environment.
    #     :return: observation, reward, is_done, debug_info"""

    #     # do stuff
    #     fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
    #     fh.setLevel(logging.WARN)

    #     self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

    #     self.redis_client = RedisClient()
    #     self.home.redis_get_initial_values()
    #     self.home.cast_redis_timestep()

    #     if self.home.timestep > 0:
    #         self.home.redis_get_prev_optimal_vals()

    #     if action:
    #         temp = self.home.t_in_max 
    #         start = (self.home.timestep * self.home.sub_subhourly_steps * self.home.dt) % (24 * self.home.sub_subhourly_steps * self.home.dt)
    #         stop = start + (self.home.dt * self.home.sub_subhourly_steps)
    #         self.home.t_in_max[start:stop] = [action + 0.5 * self.home.t_deadband] * self.home.dt * self.home.sub_subhourly_steps
    #         self.home.t_in_min[start:stop] = [action - 0.5 * self.home.t_deadband] * self.home.dt * self.home.sub_subhourly_steps

    #     self.home.get_initial_conditions()
    #     self.home.solve_type_problem()
    #     self.home.cleanup_and_finish()
    #     self.home.redis_write_optimal_vals()

    #     self.home.log.removeHandler(fh)

    #     if action:
    #         print(self.home.timestep)
    #         self.home.t_in_max = temp # reset to default values (@akp, clean up)

    #     return self.get_obs(), self.get_reward(), False, {}

    def step(self, action=None):
        """redefines the OpenAI Gym environment.
        :return: observation, reward, is_done, debug_info"""

        # do stuff
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.name}.log"))
        fh.setLevel(logging.WARN)

        self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.name)

        self.redis_client = RedisClient()
        self.home.redis_get_initial_values()
        self.home.cast_redis_timestep()

        if self.home.timestep > 0:
            self.home.redis_get_prev_optimal_vals()

        if action:
            temp = self.home.t_in_max 
            start = (self.home.timestep * self.home.sub_subhourly_steps * self.home.dt) % (24 * self.home.sub_subhourly_steps * self.home.dt)
            stop = start + (self.home.dt * self.home.sub_subhourly_steps)
            self.home.t_in_max[start:stop] = [action + 0.5 * self.home.t_deadband] * self.home.dt * self.home.sub_subhourly_steps
            self.home.t_in_min[start:stop] = [action - 0.5 * self.home.t_deadband] * self.home.dt * self.home.sub_subhourly_steps

        self.home.get_initial_conditions()
        self.home.solve_type_problem()
        self.home.cleanup_and_finish()
        self.home.redis_write_optimal_vals()

        self.home.log.removeHandler(fh)

        if action:
            print(self.home.timestep)
            self.home.t_in_max = temp # reset to default values (@akp, clean up)

        return self.get_obs(), self.get_reward(), False, {}

    async def reader(self, channel: aioredis.client.PubSub, redis):
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await channel.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        print(f"(Reader) Message Received: {message}")
                        if message["data"].decode() == "ts_complete":
                            print(f"(Reader) doing step")
                            # await redis.publish("channel:1", "mpc_update")
                            redis.publish("channel:1", "mpc_update")
                            print("here's where we can implement the action")

                        elif message["data"].decode() == "get_rewards":
                            print("here's where we can get rewards")
                        # if message["data"].decode() == "time_update":
                        #     print("(Reader) timestep moved forward")
                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass


    async def main(self):
        # redis = aioredis.from_url("redis://localhost")
        # pubsub = redis.pubsub()
        # await pubsub.subscribe("channel:1", "channel:2")

        # future = asyncio.create_task(reader(pubsub))

        # await redis.publish("channel:1", "Hello")
        # await redis.publish("channel:2", "World")
        # await redis.publish("channel:1", "stop")

        # await future

        

        redis = aioredis.from_url("redis://localhost")
        pubsub = redis.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")

        # future = asyncio.create_task(self.reader(pubsub, redis))

        # await future

        
        await redis.publish("channel:1", "mpc_update")
        # await redis.publish("channel:2", "mpc_update")
        # await redis.publish("channel:1", "time_update")

if __name__=="__main__":
    rlp = RLPlayer()
    asyncio.run(rlp.main())
