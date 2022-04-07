- [Experiment records](#experiment-records)
  - [2022/3/19](#2022319)
    - [13_50_47](#13_50_47)
  - [2022/3/20](#2022320)
    - [09_19_37](#09_19_37)
    - [11_25_21](#11_25_21)
    - [13_57_25](#13_57_25)

# Experiment records
## 2022/3/19
### 13_50_47
- Hyperparameters
  ```
  The hyperparameters are as follows:
  The dataset is: texts/唐诗2.txt
  The learning rate is: 0.001
  The number of epochs is: 100
  The batch size is: 32
  The input word count is: 10
  The vocabulary length is: 2045
  The ratio of train/val is: 4 : 1
  The network is: LSTM_enhanced(
    (Embedding): Embedding(2045, 50)
    (LSTM): LSTM(50, 100, num_layers=2, batch_first=True, dropout=0.5)
    (Linear): Linear(in_features=100, out_features=2045, bias=True)
  )
  The device is: cpu
  The optimizer is: Adam (
  Parameter Group 0
      amsgrad: False
      betas: (0.9, 0.999)
      eps: 1e-08
      initial_lr: 0.001
      lr: 5.920529220333994e-06
      weight_decay: 0
  )
  The criterion is: CrossEntropyLoss()
  The lr decay schedule is: <torch.optim.lr_scheduler.ExponentialLR object at 0x7fe8a71d2340> , and the lr decay rate is: 0.95
  ```
- Results
  
  Final train loss: 2.642

  Final val loss: 2.601
- Tests

  输入：清华大学

  输出1：
  > 清迹静公在，华之实南零。<br>
  > 大玩归渚望，学斜引上起。

  输出2:
  > 清荣尚知邑，华女晓石求。<br>
  > 大让不讴狎，学夺朝孤来。

## 2022/3/20
### 09_19_37
- Hyperparameters
  ```
  The hyperparameters are as follows:
  The dataset is: texts/唐诗2.txt
  The learning rate is: 0.001
  The number of epochs is: 100
  The batch size is: 32
  The input word count is: 10
  The vocabulary length is: 2045
  The ratio of train/val is: 4 : 1
  The network is: LSTM_enhanced(
    (Embedding): Embedding(2045, 50)
    (LSTM): LSTM(50, 50, num_layers=4, batch_first=True, dropout=0.5)
    (Linear): Linear(in_features=50, out_features=2045, bias=True)
  )
  The device is: cpu
  The optimizer is: Adam (
  Parameter Group 0
      amsgrad: False
      betas: (0.9, 0.999)
      eps: 1e-08
      initial_lr: 0.001
      lr: 0.00036603234127322915
      weight_decay: 0
  )
  The criterion is: CrossEntropyLoss()
  The lr decay schedule is: <torch.optim.lr_scheduler.ExponentialLR object at 0x7f81123bcaf0> , and the lr decay rate is: 0.99
  ```
- Results

  Final train loss: 3.663

  Final val loss: 3.73

- Tests

  输入：清华大学

  输出1：
  > 清中霜有倾，华日奏应郊。<br>
  > 大看南虽园，学荣庶相周。

  输出2:
  > 清丧客意志，华结古吹砾。<br>
  > 大销空楚续，学谷挂馀空。

- Marker
  
  Too much layers may spoil the model!

### 11_25_21
- Hyperparameters
  ```
  The hyperparameters are as follows:
  The dataset is: texts/唐诗2.txt
  The learning rate is: 0.001
  The number of epochs is: 100
  The batch size is: 32
  The input word count is: 10
  The vocabulary length is: 2045
  The ratio of train/val is: 4 : 1
  The network is: vanilla_LSTM(
    (Embedding): Embedding(2045, 100)
    (LSTM): LSTM(100, 100, batch_first=True)
    (Linear): Linear(in_features=100, out_features=2045, bias=True)
  )
  The device is: cpu
  The optimizer is: Adam (
  Parameter Group 0
      amsgrad: False
      betas: (0.9, 0.999)
      eps: 1e-08
      initial_lr: 0.001
      lr: 0.00036603234127322915
      weight_decay: 0
  )
  The criterion is: CrossEntropyLoss()
  The lr decay schedule is: <torch.optim.lr_scheduler.ExponentialLR object at 0x7f92a3478ac0> , and the lr decay rate is: 0.99
  ```
- Results

  Final train loss: 0.5869

  Final val loss: 1.18

- Tests

  输入：清华大学

  输出1：
  > 清风摇玉树，华殊灼灼殊。<br>
  > 大吹胡碧砌，学步野楼烟。

  输出2:
  > 清汉臣久逢，华净柳空泫。<br>
  > 大人胡尘未，学月九月秋。

- Marker
  
  1. Less layers, much better!

  2. No dropout, overfitting!


### 13_57_25
- Hyperparameters
  ```
  The hyperparameters are as follows:
  The dataset is: texts/唐诗2.txt
  The learning rate is: 0.001
  The number of epochs is: 100
  The batch size is: 32
  The input word count is: 10
  The vocabulary length is: 2045
  The ratio of train/val is: 4 : 1
  The network is: vanilla_GRU(
    (Embedding): Embedding(2045, 50)
    (GRU): GRU(50, 100, num_layers=2, batch_first=True, dropout=0.5)
    (Linear): Linear(in_features=100, out_features=2045, bias=True)
  )
  The device is: cpu
  The optimizer is: Adam (
  Parameter Group 0
      amsgrad: False
      betas: (0.9, 0.999)
      eps: 1e-08
      initial_lr: 0.001
      lr: 0.00036603234127322915
      weight_decay: 0
  )
  The criterion is: CrossEntropyLoss()
  The lr decay schedule is: <torch.optim.lr_scheduler.ExponentialLR object at 0x7fccdc9d7760> , and the lr decay rate is: 0.99
  ```
- Results

  Final train loss: 1.891

  Final val loss: 1.702

- Tests

  输入：清华大学

  输出1：
  > 清尊客芳阙，华发青青光。<br>
  > 大郊自理满，学府九围空。

  输出2:
  > 清野曲西岸，华发旅风枝。<br>
  > 大运自盈背，学疑在长安。

- Marker
  
  Generally, GRU is slightly better than LSTM.