# gym-dagsched

A dag scheduling environment adhering to the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) interface.

![ezgif com-optimize](https://user-images.githubusercontent.com/20342690/217447704-5f5a6ad2-4d16-4e2b-9515-99b2ee1e70b0.gif)


What is (job) dag scheduling?
- a job dag (directed acyclic graph) is used to represent jobs made of operations (nodes) that depend on each other (edges). A dependency A->B means that A must complete before B can begin.
- scheduling job dags means assigning resources to process them, according to some set of rules. If A is an operation whose dependencies are all satisfied, and X is an idle worker, then X can be assigned to A.
- dag scheduling is important. A smarter scheduling algorithm can result in faster job turnaround time.
- dag scheduling is hard. It's not obvious how to allocate resources when many jobs are competing for them, especially when more jobs randomly keep arriving.

Example: a cloud computing cluster is responsible for receiving jobs (aka workflows) and executing them using its resources. A machine learning workflow may consist of numerous operations, such as data preperation, training/validation, hyperparameter tuning, testing, etc. These operations depend on each other, e.g. data prep comes before training. Data prep can further be broken into identical tasks, where each task entails prepping a different shard of the data, and these tasks can easily be parallelized.

This repository is a re-implementaion of the [Decima codebase](https://github.com/hongzimao/decima-sim) [1]. Key features of this version:
- PyTorch is used in place of TensorFlow 1.x 
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) is used for the graph neural network part instead of coding it from scratch
- I plan to implement other policy-gradient methods such as PPO [3], in addition to the original REINFORCE
- I am actively attempting to improve the organization and documentation of the code

[[1]](https://dl.acm.org/doi/abs/10.1145/3341302.3342080) Mao, H., Schwarzkopf, M., Venkatakrishnan, S.B., Meng, Z. and Alizadeh, M., 2019. Learning scheduling algorithms for data processing clusters. In Proceedings of the ACM special interest group on data communication (pp. 270-288).

[[3]](https://arxiv.org/pdf/1707.06347.pdf) Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.





