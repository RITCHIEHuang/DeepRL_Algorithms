# PPO 算法

算法的更新核心代码在 [ppo_step.py](ppo_step.py) 中。

## 1.训练

训练PPO的主要代码在 [main.py](main.py), 使用 [click](https://click.palletsprojects.com/en/7.x/) 解析命令行参数, 因此也可以使用命令行配置参数。
执行 `python -m PolicyGradient.PPO.main --help` 可以查看所有参数:

``` text
Options:
  --env_id TEXT                  Environment Id
  --render BOOLEAN               Render environment or not
  --num_process INTEGER          Number of process to run environment
  --lr_p FLOAT                   Learning rate for Policy Net
  --lr_v FLOAT                   Learning rate for Value Net
  --gamma FLOAT                  Discount factor
  --tau FLOAT                    GAE factor
  --epsilon FLOAT                Clip rate for PPO
  --batch_size INTEGER           Batch size
  --ppo_mini_batch_size INTEGER  PPO mini-batch size (0 -> not use mini-batch update)
  --ppo_epochs INTEGER           PPO step
  --max_iter INTEGER             Maximum iterations to run
  --eval_iter INTEGER            Iterations to evaluate the model
  --save_iter INTEGER            Iterations to save the model
  --model_path TEXT              Directory to store model
  --seed INTEGER                 Seed for reproducing
```

## 2.测试

训练好的模型保存在[trained_models](trained_models)下, 执行 [test.py](test.py) 加载对应的模型以测试模型性能，
其命令行参数与 [main.py](main.py) 基本一致。

