# spark-sched-sim

An Apache Spark job scheduling simulator, implemented as a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment.

| ![ezgif com-video-to-gif](https://user-images.githubusercontent.com/20342690/217462386-ceb6ea2e-f51f-4b78-8251-24672c382371.gif)| 
|:--:| 
| *10 workers processing jobs of different colors in parallel across the vertical axis* |

What is job scheduling in Spark?
- Terminology: a Spark _application_ is a long-running program within the cluster that submits _jobs_ to be processed by its share of the cluster's resources. Each job is a directed acyclic graph (dag) of _stages_ that depend on each other, where a dependency $A\to B$ means that stage $A$ must finish executing before stage $B$ can begin. Each stage consists of many identical _tasks_ which are units of work that operate over different shards of data. Tasks are processed by _executors_, which are JVM's running on the cluster's _worker_ nodes.
- Scheduling jobs means assigning executors to their tasks. If $A$ is a stage whose dependencies are all satisfied and $X$ is an idle executor, then $X$ can be assigned to one of $A$'s tasks.
- For more info, see [this article](https://spark.apache.org/docs/latest/job-scheduling.html).

Why the simulator?
- Job scheduling is important, because a smarter scheduling algorithm can result in faster job turnaround time.
- Job scheduling is generally hard.
- This simulator allows researchers to test scheduling heuristics and train neural schedulers using RL.

This repository is a re-implementaion of the [Decima codebase](https://github.com/hongzimao/decima-sim). This version:
- implements the Gymnasium interface,
- uses PyTorch instead of TensorFlow 1.x,
- uses [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) for the GNN components,
- includes different training algorithms in addition to the original VPG,
- offers rendering, and
- aims for good organization and documentation