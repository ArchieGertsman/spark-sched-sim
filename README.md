# spark-sched-sim

An Apache Spark job scheduling simulator, implemented as a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment.

| ![](https://i.imgur.com/w5YqUf4.jpg)| 
|:--:| 
| *10 executors processing jobs in parallel across the vertical dimension, where each job is identified by a unique color. Decima achieves better resource packing and lower average job completion time than Spark's standard fair scheduler.* |

_What is job scheduling in Spark?_
- A Spark _application_ is a long-running program within the cluster that submits _jobs_ to be processed by its share of the cluster's resources. Each job encodes a directed acyclic graph (DAG) of _stages_ that depend on each other, where a dependency $A\to B$ means that stage $A$ must finish executing before stage $B$ can begin. Each stage consists of many identical _tasks_ which are units of work that operate over different shards of data. Tasks are processed by _executors_, which are JVM's running on the cluster's _worker_ nodes.
- Scheduling jobs means designating which tasks runs on which executors at each time.
- For more backround about Spark, see [this article](https://spark.apache.org/docs/latest/job-scheduling.html).

_Why this simulator?_
- Job scheduling is important, because a smarter scheduling algorithm can result in faster job turnaround time.
- This simulator allows researchers to test scheduling heuristics and train neural schedulers using reinforcement learning.

This repository is a PyTorch Geometric implementaion of the [Decima codebase](https://github.com/hongzimao/decima-sim), adhering to the Gymnasium interface. It also includes enhancements to the reinforcement learning algorithm and model design, which I developed as part of my Master's thesis, along with a basic PyGame renderer, which generates the above charts in real time.

To start out, you can try running examples via `examples.py --sched {fair,decima}`. To train Decima from scratch, you can modify the provided config file `config/decima_ppo.yaml`, then provide the config to `train.py -f FILE`.
