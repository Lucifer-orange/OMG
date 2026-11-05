

This code is based on [pymarl](https://github.com/oxwhirl/pymarl).

# Example
```
python main.py --config=4iql3qmix1iql_omg --multi_algs --env-config=sc2 with env_args.map_name=8m
```

# Other Parameter Configurations
- `--config` Algorithm Description, eg: 'iql' or '7qmix1iql_omg'
- `--multi_algs` mark for multi algorithm
- `--train_alg` training algorithm in a multi-algorithm setting
- `--mac` if multi algorithm, use "multi_module_mac" else "basic_mac"
- `--learner` choose learner for training
- `--am_model` for pre-train opponent model
- `--init_param_path` init param path
- `--is_agent_shift` in the multi-algorithm, agents switch positions
- `--obs_is_state` just for pretrain VAE