# Software Artifacts for Grid-Interactive Data Center Experiments
This repository contains software artifacts for the paper titled "Turning AI
Data Centers into Grid-Interactive Assets: Results from a Field Demonstration
in Phoenix, Arizona" including pseudocode for the [Emerald
Conductor](#emerald-ai-conductor-pseudocode) and key implementation code snippets
from the [AI Orchestration](#ai-orchestration-usage) layer.

## Data
The following files contain data used in our work:

* [data/SRP\_0503.csv](data/SRP_0503.csv): Hourly power demand at SRP on May 3, 2025
* [data/SRP\_total\_power.csv](data/SRP_total_power.csv): Total power load from the SRP Emerald AI experiment
* [data/SRP\_exp\_performance.csv](data/SRP_exp_performance.csv): Scaled performance of each workload in the SRP experiment
* [data/CAISO-netdemand-cleaned.csv](data/CAISO-netdemand-cleaned.csv): CAISO demand and net demand on August 14, 2020
* [data/CAISOexp\_totalpower\_cleaned.csv](data/CAISOexp_totalpower_cleaned.csv): Measured power during the re-enacted two-peak power event on May 2, 2025.
* [data/dvfs\_sweep.csv](data/dvfs_sweep.csv): Throughput of power-capped AI workloads on the 256 GPU cluster.
* [data/Fig2\_data.xlsx](data/Fig2_data.xlsx): Data used to generate the SRP experiment summary for Figure 2 of our experimental evaluation paper.
* [data/Fig345\_data.xlsx](data/Fig345_data.xlsx): Data used to generate the APS experiment summary, re-enacted CAISO power event, and simulator power accuracy plots in Figures 3-5.

## AI Orchestration Usage

1. Launch `memcached` on a server reachable by both controller and compute hosts.

2. On a compute node with Docker and NVIDIA GPUs, run a training job:

```bash
export MEMCACHED_HOST=<IP address>[:<port>]
export EMERALD_CONTROL_NAME=job1
./train.sh
```

3. On any other (controller) node, run orchestration commands:

```bash
# Instal uv to manage python deps
curl -LsSf https://astral.sh/uv/install.sh | sh

# Must be same instance used for training job above
export MEMCACHED_HOST=<IP address>[:<port>]

./orchestrator_commands.py power_cap job1 200
./orchestrator_commands.py checkpoint --stop job1
```

## Emerald AI Conductor Pseudocode
The pseudocode below describes the inner-workings of Emerald AI's conductor software that intelligently reduces cluster-wide power through different interventions at the AI workload level.

The experiment\_run function launches a multi-hour experiment involving an AI workload ensemble (multiple AI jobs running concurrently) and a demand response event with a specific grid target that the AI workloads are intelligently orchestrated to meet while meeting workload performance thresholds.

```python
# Inputs:
#    job_info: dict[job, tuple[num_gpus, job_flexibility, avg_throughput, uses_dvfs]]
#        A mapping of jobs on the cluster to the number of GPUs
#        they start out with and its flexibility-defined by how tolerant it is to
#        throughput reduction.
#
#        Note: the experimenter must profile jobs for avg_throughput across
#              several GPU allocations before a grid event occurs.
#
#    demand_response_signal: dict[str, Any]
#        maps keys (event_start, event_end, ramp_duration, target_power)
#        to respective values for demand response event
#
#    policy: str
#        name of algorithm used to set job power allocations and control knob usage
FUNCTION experiment_run (job_info, demand_response_signal, policy):
    # get_power_target_timeline() uses the event duration, ramp_duration, and
    # power target in the demand response signal to create a list of power targets of
    # the form:
    #
    #     CLASS PowerTarget:
    #         start_time: datetime
    #         duration: timedelta
    #         power: float
    #         segment_type: SegmentType
    #         power_margin: PowerMargin
    #
    # Return: a list of PowerTarget objects which serve as a power limits that
    # the experiment cluster must meet over the defined periods. Includes a final
    # entry to return to the system's default power target.
    power_schedule = get_power_target_timeline(demand_response_signal):

    # calculate_job_min_throughput() iterates through the jobs in job_info and
    # uses job flexibility, average throughput, and segment durations to calculate
    # the minimum throughput a job needs to operate at during a grid event period
    # such that the average throughput drop across the entire job remains within the
    # allowed flexibility factor.
    job_min_throughput = calculate_job_min_throughput(job_info, power_schedule)

    # Check the designated policy and trigger its corresponding
    # function to get a dictionary of structure
    # Dict[jobs, # tuple[timestamp, power_cap | gpu_count]] that provides a
    # mapping of jobs to the timestamped resource allocations over the experiment.
    IF policy == "DVFS, Fair":
        # dvfs_fair uses the DVFS power capping knob and a fair-share algorithm
        # to fairly allocate power reductions across all three flexibility
        # buckets. Taking the same proportion of flexibility from jobs across
        # buckets to achieve the grid target.
        # (1) Initially allocate maximum DVFS power to each job.
        # (2) Repeat until total allocated power is near the budget:
        #     - Compute the headroom for all remaining jobs (i.e., how much power they can each shed)
        #     - Repeatedly distribute an equal fraction of total headroom to all jobs, clamped above a
        #       lower bound power that meets the job's flexibility constraint.
        per_job_allocation_timeline = dvfs_fair(job_info, power_schedule)
    ELIF policy == "DVFS + Job Pausing, Fair":
        # The dvfs_plus_job_pausing_fair function uses a combination of pausing and DVFS adjustments
        # to reduce power consumption across jobs. It identifies pause-eligible jobs
        # based on their flexibility and pauses them strategically while applying DVFS
        # fair-share policies to the remaining jobs. This hybrid approach ensures the
        # power budget is met while minimizing the impact on workload performance.
        # (1) Identify pause candidates as jobs with high event flexibility, which 
        #     refers to the amount of performance flexibility the job has during
        #     the event time window such that the job's overall performance does not violate
        #     its overall flexibility constraint.
        # (2) FOR each set of ≥ 2 pause-eligible jobs:
        #     - Calculate the power that DVFS will be able to control (power
        #       from non-pausing, flexible jobs)
        #     - Run dvfs_fair on the remaining DVFS jobs, with the calculated range of DVFS-manageable power
        #     - Flag the pause-eligible set that has the least total (pausing + dvfs)
        #       power reduced while staying under the power budget
        # (3) Divide the DR event time evenly among jobs so that one job is paused at a time,
        #     staggering pauses across the timeline. Use the dvfs_fair policy for non-paused
        #     DVFS-enabled jobs.
        per_job_allocation_timeline = dvfs_plus_job_pausing_fair(job_info, power_schedule)

    ELIF policy == "Job Pausing, Greedy":
        # The job_pausing_greedy function uses a pause-only greedy algorithm to reduce
        # power consumption across jobs. It calculates each job's pause allowance based
        # on its flexibility and iteratively pauses the most flexible jobs to bring the
        # cluster's power within the target budget. This approach ensures that power
        # reductions are achieved by strategically pausing jobs while minimizing the
        # impact on less flexible workloads.
        # (1) Compute each job's pause allowance (# timesteps it may be fully stopped) from its flexibility
        # (2) Create a timeline table of with length equal to the least-common multiple of all allowances,
        #     initialized to run all jobs at their GPU count
        # (3) Walk the timeline row by row. Whenever the current row’s power exceeds the budget,
        #     pause (set GPU count to 0) the next most-flexible job for exactly its allowed
        #     number of consecutive rows, continuing through flexible jobs until within power budget.
        per_job_allocation_timeline = job_pausing_greedy(job_info, power_schedule)

    ELSE: # "Job Pausing + Resource Allocation, Fair"
        # The job_pausing_plus_resource_allocation_fair function uses a combination of
        # GPU resizing and pausing in a fair-share algorithm to reduce power consumption
        # across jobs. It identifies jobs that can be fully paused based on their
        # flexibility and tests each pause-eligible job individually. For each candidate,
        # it pauses the job and resizes the remaining jobs using a fair-share GPU
        # allocation policy. The algorithm selects the plan that achieves the required
        # power reduction with the smallest count of jobs that need to be impacted,
        # ensuring the cluster meets the target budget while minimizing disruption to
        # workloads.
        # (1) Find jobs that can be completely paused. A job qualifies if its
        #     flexibility allows for it to stop for the full demand response window
        # (2) FOR each candidate job in pause-eligible jobs, estimate GPU
        #     counts needed to meet the power budget while fairly distributing
        #     GPU reductions across other jobs (fixing the paused job at GPUs=0).
        #     Keep the plan that has the greatest total power within the budget
        per_job_allocation_timeline = job_pausing_plus_resource_allocation_fair(job_info, power_schedule)

    # execute_job_schedule() is a per job power allocation enforcer. It
    # iterates through an allocation timeline, waits until the current timestamp
    # occurs, then either power caps, or changes the GPU allocation depending on what
    # action is prescribed.
    FUNCTION execute_job_schedule(job, allocation_timeline):
        FOR each (timestamp, target) in allocation_timeline:
            wait_until(timestamp)
            IF job.uses_dvfs:
                job.set_power_cap(target) # Refer to definition in orchestrator_commands.py
            ELSE:
                job.checkpoint(stop=True) # Refer to definition in orchestrator_commands.py
                job.start(target)


    # Executes threads for each job that operate in parallel, enforcing power
    # allocations via DVFS, GPU resizing, or pausing and starting based on the
    # allocation timeline generated by the policies. the final event in grid_events
    # returns to the cluster's default power and it continues operating untouched
    # once we reach the end of the timeline.
    With ThreadPoolExecutor(max_workers=num_jobs) as executor:
        for job, allocation_timeline in per_job_allocation_timeline.items():
            executor.submit(
                execute_job_schedule,
                job,
                allocation_timeline,
            )
```
