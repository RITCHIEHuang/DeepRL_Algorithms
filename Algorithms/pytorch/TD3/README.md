# TD3 算法

算法的更新核心代码在 [td3_step.py](td3_step.py) 中。

## 1.训练

训练 TD3 的主要代码在 [main.py](main.py), 使用 [click](https://click.palletsprojects.com/en/7.x/) 解析命令行参数, 因此也可以使用命令行配置参数。
执行 `python -m PolicyGradient.TD3.main --help` 可以查看所有参数:

``` text
Options:
  --env_id TEXT                   Environment Id
  --render BOOLEAN                Render environment or not
  --num_process INTEGER           Number of process to run environment
  --lr_p FLOAT                    Learning rate for Policy Net
  --lr_v FLOAT                    Learning rate for Value Net
  --gamma FLOAT                   Discount factor
  --polyak FLOAT                  Interpolation factor in polyak averaging for
                                  target networks
  --target_action_noise_std FLOAT
                                  Std for noise of target action
  --target_action_noise_clip FLOAT
                                  Clip ratio for target action noise
  --explore_size INTEGER          Explore steps before execute deterministic
                                  policy
  --memory_size INTEGER           Size of replay memory
  --step_per_iter INTEGER         Number of steps of interaction in each
                                  iteration
  --batch_size INTEGER            Batch size
  --min_update_step INTEGER       Minimum interacts for updating
  --update_step INTEGER           Steps between updating policy and critic
  --max_iter INTEGER              Maximum iterations to run
  --eval_iter INTEGER             Iterations to evaluate the model
  --save_iter INTEGER             Iterations to save the model
  --action_noise FLOAT            Noise for action
  --policy_update_delay INTEGER   Frequency for policy update
  --model_path TEXT               Directory to store model
  --log_path TEXT                 Directory to save logs
  --seed INTEGER                  Seed for reproducing
  --help                          Show this message and exit.

```

相较于 DDPG， 这里多了对 target action 的 noise (正态分布) 的standard deviation， 对 noise 的 clip ratio 以及 policy 更新频率。

## 2.测试

训练好的模型保存在[trained_models](trained_models)下, 执行 [test.py](test.py) 加载对应的模型以测试模型性能，
其命令行参数与 [main.py](main.py) 基本一致。


