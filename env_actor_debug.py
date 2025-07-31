from verl.workers.rollout.env_workers.maniskill_env_worker import EnvActor

env = EnvActor()

init_data = env.init_venv(
    env_ids=['StackCube-v1'],
    env_unique_ids=[0],
    task_instructions='pick and place',
    is_valid=True,
    global_steps=0,
    max_steps=200
)