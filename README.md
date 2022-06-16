# gym-dagsched

An OpenAI-Gym-style simulation environment for scheduling 
streaming jobs consisting of interdependent operations. 

What?
- "job": consists of "operations" that need to be completed by "workers"
- "streaming": arriving stochastically and continuously over time
- "scheduling": assigning workers to jobs
- "operation": can be futher split into "tasks" which are identical and can be worked on in parallel. The number of tasks in a operation is equal to the number of workers that can work on the operation in parallel.
- "interdependent": some operations may depend on the results of others, and therefore cannot begin execution until those dependencies are satisfied. These dependencies are encoded in directed acyclic graphs (dags) where an edge from operation (a) to operation (b) means that (a) must complete before (b) can begin.

Example: a cloud computing cluster is responsible for receiving workflows (i.e. jobs) and executing them using its resources. A  machine learning workflow may consist of numerous operations, such as data prep, training/validation, hyperparameter tuning, testing, etc. and these operations depend on each other, e.g. data prep comes before training. Data prep can further be broken into tasks, where each task entails prepping a subset of the data, and these tasks can easily be parallelized. 
    
The process of assigning workers to jobs is crutial, as sophisticated scheduling algorithms can significantly increase the system's efficiency. Yet, it turns out to be a very challenging problem.

