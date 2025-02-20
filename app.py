from flask import Flask, render_template, request
import datetime
from datetime import datetime, timedelta, time
import portion as P
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
import logging
from io import StringIO

# Set up logging
log_buffer = StringIO()
logging.basicConfig(level=logging.DEBUG, stream=log_buffer)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define global variables
all_tasks = []
all_resources = []
task_predecessors = {}
max_date = datetime(2033, 12, 31, 23, 59)

# Define working hours (9 AM to 5 PM, Monday to Friday)
def get_working_hours(date):
    if date.weekday() < 5:
        start = datetime.combine(date, time(9, 0))
        end = datetime.combine(date, time(17, 0))
        return start, end
    return None

# Find earliest start time for a task segment
def find_start_time(predecessors_end_times, required_resources, duration, busy_times, start_date, task_id, segment_num):
    if predecessors_end_times:
        predecessors_end = max(predecessors_end_times)
    else:
        predecessors_end = start_date

    duration_minutes = duration.total_seconds() / 60
    logger.debug(f"Task {task_id}-Segment {segment_num}: Duration = {duration_minutes} minutes")
    if duration_minutes > 480:
        raise ValueError(f"Task {task_id}-Segment {segment_num} duration {duration_minutes} exceeds 480 minutes after splitting.")

    current_date = predecessors_end.date()
    while True:
        if datetime.combine(current_date, time(0, 0)) > max_date:
            logger.error(f"Task {task_id}-Segment {segment_num}: Exceeded max date {max_date}. Resources: {', '.join(r.name for r in required_resources)}")
            raise ValueError(f"Task {task_id}-Segment {segment_num} cannot be scheduled; exceeds {max_date}.")
        
        working_hours = get_working_hours(current_date)
        if working_hours:
            start_work, end_work = working_hours
            t_min = max(start_work, predecessors_end)
            if t_min.date() > current_date:
                current_date = t_min.date()
                continue
            t_max = end_work - duration
            if t_min <= t_max:
                t = t_min
                while t <= t_max:
                    slot = P.closed(t, t + duration)
                    if all(slot & busy_times.get(res, P.empty()) == P.empty() for res in required_resources):
                        logger.debug(f"Task {task_id}-Segment {segment_num}: Scheduled at {t}")
                        return t
                    t += timedelta(minutes=1)
        current_date += timedelta(days=1)
        predecessors_end = datetime.combine(current_date, time(0, 0))

# Decode chromosome to a schedule with task splitting
def decode(individual, all_tasks, task_predecessors, start_date):
    task_indices = {task: i for i, task in enumerate(all_tasks)}
    busy_times = {res: P.empty() for res in all_resources}
    task_end_times = {}
    segment_times = []
    ready_tasks = set(task for task in all_tasks if not task_predecessors[task])
    scheduled_tasks = set()

    while ready_tasks:
        ready_indices = [task_indices[task] for task in ready_tasks]
        selected_index = min(ready_indices, key=lambda x: individual.index(x))
        selected_task = all_tasks[selected_index]
        
        total_duration_minutes = selected_task.duration.total_seconds() / 60
        daily_max = 480
        num_full_days = int(total_duration_minutes // daily_max)
        remaining_minutes = total_duration_minutes % daily_max
        segments = [timedelta(minutes=daily_max)] * num_full_days
        if remaining_minutes > 0:
            segments.append(timedelta(minutes=remaining_minutes))
        
        logger.debug(f"Task {selected_task.id}: Total duration = {total_duration_minutes} minutes, split into {len(segments)} segments")

        segment_start = None
        predecessors_end_times = [task_end_times[pred] for pred in task_predecessors[selected_task] 
                                  if pred in task_end_times]
        for i, segment_duration in enumerate(segments, 1):
            segment_start = find_start_time(
                predecessors_end_times if i == 1 else [segment_end],
                selected_task.required_resources,
                segment_duration,
                busy_times,
                start_date,
                selected_task.id,
                i
            )
            segment_end = segment_start + segment_duration
            segment_times.append((selected_task, i, segment_start, segment_end))
            for res in selected_task.required_resources:
                busy_times[res] |= P.closed(segment_start, segment_end)
            predecessors_end_times = [segment_end]
        
        task_end_times[selected_task] = segment_end
        
        ready_tasks.remove(selected_task)
        scheduled_tasks.add(selected_task)
        for task in all_tasks:
            if task not in scheduled_tasks and all(pred in task_end_times for pred in task_predecessors[task]):
                ready_tasks.add(task)

    makespan_datetime = max(task_end_times.values(), key=lambda x: x.timestamp())
    makespan = makespan_datetime.timestamp()
    return makespan, segment_times

# Class Definitions
class Resource:
    def __init__(self, name, type):
        self.name = name
        self.type = type  # "H" for human, "M" for machine

class Task:
    def __init__(self, id, setup_time, process_time_per_unit, required_resources, description, predecessors=None):
        self.id = id
        self.setup_time = setup_time
        self.process_time_per_unit = process_time_per_unit
        self.required_resources = required_resources
        self.description = description
        self.predecessors = predecessors or []
        self.duration = None
        self.job_id = None

class Job:
    def __init__(self, job_id, tasks, quantity, unit_price, description, client, promised_date):
        self.job_id = job_id
        self.tasks = tasks
        self.quantity = int(quantity)
        self.unit_price = unit_price
        self.description = description
        self.client = client
        self.promised_date = promised_date  # New field
        for task in tasks:
            setup_time = int(task.setup_time)
            process_time_per_unit = int(task.process_time_per_unit)
            total_minutes = setup_time + process_time_per_unit * self.quantity
            if total_minutes > 5256000:
                raise ValueError(f"Task {task.id} duration {total_minutes} minutes exceeds reasonable limit.")
            task.duration = timedelta(minutes=total_minutes)
            task.job_id = self.job_id
            logger.debug(f"Task {task.id}: Total duration = {total_minutes} minutes")

# Load data from Excel
def load_data(file_path):
    global all_tasks, all_resources, task_predecessors
    resources_df = pd.read_excel(file_path, sheet_name='Resources')
    jobs_df = pd.read_excel(file_path, sheet_name='Jobs')
    tasks_df = pd.read_excel(file_path, sheet_name='Tasks')

    resources = {row['Resource Name']: Resource(row['Resource Name'], row['Type']) 
                 for _, row in resources_df.iterrows()}
    all_resources = list(resources.values())

    tasks = {}
    for _, row in tasks_df.iterrows():
        task_id = row['Task ID']
        required_resources = [resources[name.strip()] for name in row['Required Resources'].split(',')]
        predecessors_str = row['Predecessors'] if pd.notna(row['Predecessors']) else ''
        description = row['Description'] if pd.notna(row['Description']) else 'No description'
        task = Task(task_id, row['Setup Time (min)'], row['Process Time per Unit (min)'], 
                    required_resources, description)
        tasks[task_id] = task

    for _, row in tasks_df.iterrows():
        task_id = row['Task ID']
        predecessors_str = row['Predecessors'] if pd.notna(row['Predecessors']) else ''
        if predecessors_str:
            predecessors = [tasks[pred.strip()] for pred in str(predecessors_str).split(',') 
                            if pred.strip() and pred.strip() in tasks]
            tasks[task_id].predecessors = predecessors

    jobs = []
    for _, row in jobs_df.iterrows():
        job_id = row['Job ID']
        quantity = row['Quantity']
        unit_price = row['Unit Price']
        description = row['Description'] if pd.notna(row['Description']) else 'No description'
        client = row['Client'] if pd.notna(row['Client']) else 'No client'
        promised_date = pd.to_datetime(row['Promised date']) if pd.notna(row['Promised date']) else datetime.now()
        job_tasks = [tasks[task_id] for task_id in tasks_df[tasks_df['Job ID'] == job_id]['Task ID']]
        job = Job(job_id, job_tasks, quantity, unit_price, description, client, promised_date)
        jobs.append(job)

    all_tasks = list(tasks.values())
    task_predecessors = {task: task.predecessors for task in all_tasks}

    return all_resources, jobs, all_tasks, task_predecessors

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(all_tasks)), len(all_tasks))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: (decode(ind, all_tasks, task_predecessors, start_date)[0],))
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Routes
@app.route('/', methods=['GET'])
def index():
    default_start_date = (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    return render_template('index.html', default_date=default_start_date.strftime('%Y-%m-%dT%H:%M'))

@app.route('/schedule', methods=['POST'])
def schedule():
    global start_date
    start_date_str = request.form['start_date']
    start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M')
    logger.debug(f"Selected start date: {start_date}")

    # Clear previous log output
    log_buffer.truncate(0)
    log_buffer.seek(0)

    file_path = 'InputData.xlsx'
    all_resources, jobs, all_tasks, task_predecessors = load_data(file_path)

    toolbox.unregister("indices")
    toolbox.unregister("individual")
    toolbox.unregister("population")
    toolbox.register("indices", random.sample, range(len(all_tasks)), len(all_tasks))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (decode(ind, all_tasks, task_predecessors, start_date)[0],))

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, 
                        halloffame=hof, verbose=False)
    best_individual = hof[0]
    
    makespan, segment_times = decode(best_individual, all_tasks, task_predecessors, start_date)
    
    scheduled_segments = [
        {"task_id": f"{task.id}-{seg_num}", "job_id": task.job_id, 
         "machines": ", ".join(res.name for res in task.required_resources if res.type == 'M'),
         "humans": ", ".join(res.name for res in task.required_resources if res.type == 'H'),
         "start": start, "end": end, "description": task.description}
        for task, seg_num, start, end in segment_times
    ]
    logger.debug(f"Scheduled segments: {scheduled_segments}")
    scheduled_segments.sort(key=lambda x: x["start"])
    
    job_completions = {
        job.job_id: {
            "completion": max(end for task, _, _, end in segment_times if task in job.tasks), 
            "description": job.description, 
            "client": job.client,
            "promised_date": job.promised_date,
            "unit_price": job.unit_price,
            "quantity": job.quantity
        } 
        for job in jobs
    }

    # Capture log output
    log_output = log_buffer.getvalue()
    
    return render_template("schedule.html", segments=scheduled_segments, job_completions=job_completions, log_output=log_output)

if __name__ == "__main__":
    app.run(debug=True)