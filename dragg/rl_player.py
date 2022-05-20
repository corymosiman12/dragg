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

REDIS_URL = "redis://localhost"

def manage_home(home):
    """
    Calls class method as a top level function (picklizable by pathos)
    :return: None
    """
    home.step()#np.random.uniform(19,22))
    return

class RLPlayer(gym.Env):
    def __init__(self):#, home):
        # redis = aioredis.from_url(REDIS_URL)
        # pubsub = redis.pubsub()
        # await pubsub.subscribe("channel:1", "channel:2")
        home = asyncio.run(self.set_home())
        # await redis.publish("channel:1", "mpc_started")

        
        self.home = MPCCalc(home)
        self.name = self.home.name
        with open('data/rl_data/state_action.json','r') as file:
            states_actions = json.load(file)
        self.states = [k for k, v in states_actions['states'].items() if v]
        self.actions = [k for k, v in states_actions['actions'].items() if v]

    async def set_home(self):
        async_redis = aioredis.from_url(REDIS_URL)
        pubsub = async_redis.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")

        redis_client = RedisClient()
        home = redis_client.conn.hgetall("home_values")
        home['hvac'] = redis_client.conn.hgetall("hvac_values")
        home['wh'] = redis_client.conn.hgetall("wh_values")
        home['hems'] = redis_client.conn.hgetall("hems_values")
        home['hems']["weekday_occ_schedule"] = [[19,8],[17,18]]
        if 'battery' in home['type']:
            home['battery'] = redis_client.conn.hgetall("battery_values")
        if 'pv' in home['type']:
            home['pv'] = redis_client.conn.hgetall("pv_values")
        home['wh']['draw_sizes'] = redis_client.conn.lrange('draw_sizes', 0, -1)
        home['hems']['weekday_occ_schedule'] = redis_client.conn.lrange('weekday_occ_schedule', 0, -1)
        print(home['name'])

        await async_redis.publish("channel:1", "mpc_started")
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

    # def get_reward(self):
    #     """ determines a reward, function can be redefined by user
    #     :return: float value normalized to [-1,1] """
    #     reward = 1#self.redis_client.conn.hget("current_demand")

    #     return reward

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

    async def step(self, action=None):
        """redefines the OpenAI Gym environment.
        :return: observation, reward, is_done, debug_info"""

        # connect to logger and redis (for parallelization this must be done within the method)
        fh = logging.FileHandler(os.path.join("home_logs", f"{self.home.name}.log"))
        fh.setLevel(logging.WARN)
        self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.home.name)
        # self.home.redis_client = RedisClient() # RedisClient is a singleton class

        # connect to the aioredis database (for asynchronous updates)
        # must cast to self.home.redis_client based on MPCCalc structure
        async_redis_client = aioredis.from_url(REDIS_URL)
        pubsub = async_redis_client.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")
        
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

        # # tell the aggregator that you are done
        # await self.async_redis_client.publish("channel:1", "mpc_updated")
        
        # pubsub = self.async_redis_client.pubsub()
        # await pubsub.subscribe("channel:1", "channel:2")

        # self.get_reward(pubsub, async_redis_client)
        reward = 1
        print(reward)

        self.home.log.removeHandler(fh)

        if action:
            print(self.home.timestep)
            self.home.t_in_max = temp # reset to default values (@akp, clean up)

        return self.get_obs(), reward, False, {}

    def get_reward(self, channel: aioredis.client.PubSub, redis_client):
        # while True:
        #     try:
        #         async with async_timeout.timeout(1):
        #             message = await channel.get_message(ignore_subscribe_messages=True)
        #             if message is not None:
        #                 print(f"(Reader) Message Recieved: {message}")
        #                 if message["data"].decode() == "ts_complete":
        #                     current_demand = redis_client.conn.hget("current_demand")
        #     except asyncio.TimeoutError:
        #         pass
        current_demand = 1
        return current_demand

    async def reader(self, channel: aioredis.client.PubSub, redis_client):
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await channel.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        print(f"(Reader) Message Received: {message}")
                        if message["data"].decode() == "ts_complete":
                            print(f"(Reader) doing step")
                            await redis_client.publish("channel:1", "mpc_update")

                        elif message["data"].decode() == "get_rewards":
                            print("here's where we can get rewards")
                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass
        return

    async def main(self, action=None):

        fh = logging.FileHandler(os.path.join("home_logs", f"{self.home.name}.log"))
        fh.setLevel(logging.WARN)
        self.home.log = pathos.logger(level=logging.INFO, handler=fh, name=self.home.name)
        # self.home.redis_client = RedisClient() # RedisClient is a singleton class
        
        # connect to the aioredis database (for asynchronous updates)
        # must cast to self.home.redis_client based on MPCCalc structure
        async_redis_client = aioredis.from_url(REDIS_URL)
        pubsub = async_redis_client.pubsub()
        await pubsub.subscribe("channel:1", "channel:2")

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
        # self.home.initialize_environmental_variables()
        self.home.solve_type_problem()
        # self.home.cleanup_and_finish()
        # self.home.redis_write_optimal_vals()
        # redis = aioredis.from_url(REDIS_URL)
        # pubsub = redis.pubsub()
        # await pubsub.subscribe("channel:1", "channel:2")
        # await redis.publish("channel:1", "mpc_update")
        self.home.redis_client.conn.close()
        self.home.log.removeHandler(fh)
        return

    def main_wrapper(self):
        # asyncio.run(self.step())
        self.step()
        return

    async def problem_child(self):
        self.home.get_initial_conditions()
        self.home.solve_type_problem()

if __name__=="__main__":
    # asyncio.set_event_loop(asyncio.ProactorEventLoop())

    rlp = RLPlayer()
    # asyncio.run(rlp.main())
    asyncio.run(rlp.problem_child())
    # rlp.main_wrapper()
