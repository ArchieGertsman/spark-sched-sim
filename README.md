# spark-sched-sim

An Apache Spark job scheduling simulator, implemented as a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment.

| ![](https://i.imgur.com/6BpPWxI.png)| 
|:--:| 
| *Two Gantt charts comparing the behavior of different job scheduling algorithms. In these experiments, 50 jobs are identified by unique colors and processed in parallel by 10 identical executors (stacked vertically). Decima achieves better resource packing and lower average job completion time than Spark's fair scheduler.* |

_What is job scheduling in Spark?_
- A Spark _application_ is a long-running program within the cluster that submits _jobs_ to be processed by its share of the cluster's resources. Each job encodes a directed acyclic graph (DAG) of _stages_ that depend on each other, where a dependency $A\to B$ means that stage $A$ must finish executing before stage $B$ can begin. Each stage consists of many identical _tasks_ which are units of work that operate over different shards of data. Tasks are processed by _executors_, which are JVM's running on the cluster's _worker_ nodes.
- Scheduling jobs means designating which tasks runs on which executors at each time.
- For more backround on Spark, see [this article](https://spark.apache.org/docs/latest/job-scheduling.html).

_Why this simulator?_
- Job scheduling is important, because a smarter scheduling algorithm can result in faster job turnaround time.
- This simulator allows researchers to test scheduling heuristics and train neural schedulers using reinforcement learning.

---

This repository is a PyTorch Geometric implementaion of the [Decima codebase](https://github.com/hongzimao/decima-sim), adhering to the Gymnasium interface. It also includes enhancements to the reinforcement learning algorithm and model design, along with a basic PyGame renderer that generates the above charts in real time.

Enhancements include:
- Continuously discounted returns, improving training speed
- Proximal Polixy Optimization (PPO), improving training speed and stability
- A restricted action space, encouraging a fairer policy to be learned
- Multiple different job sequences experienced per training iteration, reducing variance in the policy gradient (PG) estimate
- No learning curriculum, improving training speed

---

After cloning this repo, please run `pip install -r requirements.txt` to install the project's dependencies. Then, please manually install `torch_scatter` and `torch_sparse` by running e.g.
```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html
```
They are commented out in `requirements.txt` because `torch` needs to be installed first. See [here](https://github.com/pyg-team/pytorch_geometric/issues/861) for more.

To start out, try running examples via `examples.py --sched [fair|decima]`. To train Decima from scratch, modify the provided config file `config/decima_tpch.yaml` as needed, then provide the config to `train.py -f CFG_FILE`.