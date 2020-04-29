from policy import PolicyGradientModel, clone_policy
from sampler import BatchSampler
import multiprocessing as mp


sampler = BatchSampler('Maze-v0',
                           batch_size=20,
                           num_workers=mp.cpu_count() - 1)

print (sampler.envs.observation_space.shape)
print (sampler.envs.action_space.shape)


tasks = sampler.sample_tasks(num_tasks=40)