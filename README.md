# gym-dagsched

A simulation environment, adhering to the [OpenAI Gym](https://github.com/openai/gym) interface, for scheduling streaming jobs with interdependent stages.

What?
- "job": consists of "stages" that need to be completed by "workers"
- "streaming": arriving stochastically and continuously over time
- "scheduling": assigning a worker to a job
- "stage": can be futher split into "tasks" which are identical and 
    can be worked on in parallel. The number of tasks in a stage is
    effectively equal to the number of workers that can work on the 
    stage in parallel.
- "interdependent": some stages may depend on the results of other
    stages, and therefore cannot begin work until those dependencies 
    are completed. These dependencies are encoded in directed acyclic
    graphs (a.k.a dags) where an edge from stage (a) to stage (b) means
    that (a) must finish before (b) can begin.