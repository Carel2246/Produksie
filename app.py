from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import datetime
from datetime import datetime, timedelta, time
import portion as P
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
import logging
from io import StringIO
from werkzeug.utils import secure_filename
import os

# Set up logging
log_buffer = StringIO()
logging.basicConfig(level=logging.DEBUG, stream=log_buffer)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobshop.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# Scheduling Classes (moved to top)
class ResourceScheduler:
    def __init__(self, id, name, type):
        self.id = id
        self.name = name
        self.type = type  # H or M

class TaskScheduler:
    def __init__(self, id, setup_time, process_time_per_unit, required_resources, description, predecessors=None):
        self.id = id
        self.setup_time = setup_time
        self.process_time_per_unit = process_time_per_unit
        self.required_resources = required_resources
        self.description = description
        self.predecessors = predecessors or []
        self.duration = None  # Will be set in load_data
        self.job_id = None

class JobScheduler:
    def __init__(self, job_id, tasks, quantity, unit_price, description, client, promised_date):
        self.job_id = job_id
        self.tasks = tasks
        self.quantity = int(quantity)
        self.unit_price = unit_price
        self.description = description
        self.client = client
        self.promised_date = promised_date

# Define global variables
all_tasks = []
all_resources = []
task_predecessors = {}
max_date = datetime(2033, 12, 31, 23, 59)
RESOURCE_POOL = {}  # Global resource pool

# Database Models (for SQLAlchemy)
class ResourceModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    type = db.Column(db.String(1), nullable=False)  # H or M

class JobModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(50), unique=True, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255), nullable=True)
    client = db.Column(db.String(100), nullable=True)
    promised_date = db.Column(db.DateTime, nullable=True)

class TaskModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(50), unique=True, nullable=False)
    job_id = db.Column(db.String(50), db.ForeignKey('job_model.job_id'), nullable=False)
    setup_time = db.Column(db.Integer, nullable=False)
    process_time_per_unit = db.Column(db.Integer, nullable=False)
    required_resources = db.Column(db.String(255), nullable=False)  # Comma-separated
    predecessors = db.Column(db.String(255), nullable=True)  # Comma-separated
    description = db.Column(db.String(255), nullable=True)

# Initialize RESOURCE_POOL at app startup
with app.app_context():
    db.create_all()
    resources = ResourceModel.query.all()
    RESOURCE_POOL.update({r.name: ResourceScheduler(r.id, r.name, r.type) for r in resources})

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

# Decode chromosome to a schedule and calculate total Rand-days late
def decode(individual, all_tasks, task_predecessors, start_date, jobs):
    task_indices = {task: i for i, task in enumerate(all_tasks)}
    busy_times = {res: P.empty() for res in RESOURCE_POOL.values()}
    task_end_times = {}
    segment_times = []
    ready_tasks = set(task for task in all_tasks if not task_predecessors[task])
    scheduled_tasks = set()

    while ready_tasks:
        ready_indices = [task_indices[task] for task in ready_tasks]
        valid_ready_indices = [idx for idx in ready_indices if idx in individual]
        if not valid_ready_indices:
            logger.error(f"No valid ready indices found. Ready: {ready_indices}, Individual: {individual}")
            break
        selected_index = min(valid_ready_indices, key=lambda x: individual.index(x))
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
                if res not in busy_times:
                    logger.error(f"Resource {res.name} not in busy_times. Pool: {list(busy_times.keys())}")
                    raise KeyError(f"Resource {res.name} not found in busy_times")
                busy_times[res] |= P.closed(segment_start, segment_end)
            predecessors_end_times = [segment_end]
        
        task_end_times[selected_task] = segment_end
        
        ready_tasks.remove(selected_task)
        scheduled_tasks.add(selected_task)
        for task in all_tasks:
            if task not in scheduled_tasks and all(pred in task_end_times for pred in task_predecessors[task]):
                ready_tasks.add(task)

    # Calculate total Rand-days late
    job_completions = {
        job.job_id: max(end for task, _, _, end in segment_times if task in job.tasks)
        for job in jobs
    }
    total_rand_days_late = 0
    for job in jobs:
        completion = job_completions[job.job_id]
        days_late = (completion - job.promised_date).total_seconds() / 86400  # Days
        rand_days_late = max(0, days_late) * job.unit_price * job.quantity  # 0 if early/on-time
        total_rand_days_late += rand_days_late

    return total_rand_days_late, segment_times

# Load data from database
def load_data():
    jobs = JobModel.query.all()
    tasks = TaskModel.query.all()

    # Create task objects using shared resource pool
    task_dict = {}
    for task in tasks:
        required_resource_names = [r.strip() for r in task.required_resources.split(',')]
        missing_resources = [r for r in required_resource_names if r not in RESOURCE_POOL]
        if missing_resources:
            logger.error(f"Task {task.task_id} requires missing resources: {missing_resources}")
        required_resources = [RESOURCE_POOL[r] for r in required_resource_names if r in RESOURCE_POOL]
        predecessors = [task_dict[pred.strip()] for pred in task.predecessors.split(',') if pred.strip() and pred.strip() in task_dict] if task.predecessors else []
        task_obj = TaskScheduler(task.task_id, task.setup_time, task.process_time_per_unit, required_resources, task.description, predecessors)
        task_dict[task.task_id] = task_obj

    # Create job objects and set task durations
    job_dict = {}
    for job in jobs:
        job_tasks = [task_dict[task.task_id] for task in tasks if task.job_id == job.job_id]
        for task in job_tasks:
            total_minutes = task.setup_time + task.process_time_per_unit * job.quantity
            task.duration = timedelta(minutes=total_minutes)
            task.job_id = job.job_id
        job_obj = JobScheduler(job.job_id, job_tasks, job.quantity, job.unit_price, job.description, job.client, job.promised_date)
        job_dict[job.job_id] = job_obj

    all_tasks = list(task_dict.values())
    task_predecessors = {task: task.predecessors for task in all_tasks}

    return list(RESOURCE_POOL.values()), list(job_dict.values()), all_tasks, task_predecessors

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Routes
@app.route('/', methods=['GET'])
def index():
    default_start_date = (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    all_resources, jobs, all_tasks, task_predecessors = load_data()
    job_list = [
        {"job_id": job.job_id, "description": job.description, "client": job.client, "quantity": job.quantity}
        for job in jobs
    ]
    return render_template('index.html', default_date=default_start_date.strftime('%Y-%m-%dT%H:%M'), jobs=job_list)

@app.route('/schedule', methods=['POST'])
def schedule():
    global start_date
    start_date_str = request.form['start_date']
    start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M')
    logger.debug(f"Selected start date: {start_date}")

    log_buffer.truncate(0)
    log_buffer.seek(0)

    all_resources, jobs, all_tasks, task_predecessors = load_data()

    toolbox.register("indices", random.sample, range(len(all_tasks)), len(all_tasks))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (decode(ind, all_tasks, task_predecessors, start_date, jobs)[0],))
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, 
                        halloffame=hof, verbose=False)
    best_individual = hof[0]
    
    total_rand_days_late, segment_times = decode(best_individual, all_tasks, task_predecessors, start_date, jobs)
    
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

    log_output = log_buffer.getvalue()
    
    return render_template("schedule.html", segments=scheduled_segments, job_completions=job_completions, log_output=log_output)

@app.route('/add_resource', methods=['GET', 'POST'])
def add_resource():
    if request.method == 'POST':
        if 'add' in request.form:
            name = request.form['name']
            type = request.form['type']
            resource = ResourceModel(name=name, type=type)
            db.session.add(resource)
            db.session.commit()
            RESOURCE_POOL[name] = ResourceScheduler(resource.id, name, type)
        elif 'update' in request.form:
            resource_id = request.form['id']
            resource = ResourceModel.query.get(resource_id)
            old_name = resource.name
            resource.name = request.form['name']
            resource.type = request.form['type']
            db.session.commit()
            if old_name != resource.name:
                del RESOURCE_POOL[old_name]
            RESOURCE_POOL[resource.name] = ResourceScheduler(resource.id, resource.name, resource.type)
        elif 'delete' in request.form:
            resource_id = request.form['id']
            resource = ResourceModel.query.get(resource_id)
            db.session.delete(resource)
            db.session.commit()
            RESOURCE_POOL.pop(resource.name, None)
    resources = ResourceModel.query.all()
    return render_template('add_resource.html', resources=resources)

@app.route('/add_job', methods=['GET', 'POST'])
def add_job():
    if request.method == 'POST':
        if 'add' in request.form:
            job_id = request.form['job_id']
            quantity = request.form['quantity']
            unit_price = request.form['unit_price']
            description = request.form['description']
            client = request.form['client']
            promised_date = datetime.strptime(request.form['promised_date'], '%Y-%m-%dT%H:%M')
            job = JobModel(job_id=job_id, quantity=quantity, unit_price=unit_price, description=description, client=client, promised_date=promised_date)
            db.session.add(job)
            db.session.commit()
        elif 'update' in request.form:
            job_id = request.form['job_id']
            job = JobModel.query.get(job_id)
            job.quantity = int(request.form['quantity'])
            job.unit_price = float(request.form['unit_price'])
            job.description = request.form['description']
            job.client = request.form['client']
            job.promised_date = datetime.strptime(request.form['promised_date'], '%Y-%m-%d %I:%M %p')
            db.session.commit()
        elif 'delete' in request.form:
            job_id = request.form['job_id']
            job = JobModel.query.get(job_id)
            db.session.delete(job)
            db.session.commit()
    jobs = JobModel.query.all()
    return render_template('add_job.html', jobs=jobs)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add_task', methods=['GET', 'POST'])
def add_task():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '' and 'add_file' in request.form:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                tasks_df = pd.read_excel(file_path)
                for _, row in tasks_df.iterrows():
                    task = TaskModel(
                        task_id=row['Task ID'],
                        job_id=row['Job ID'],
                        setup_time=row['Setup Time (min)'],
                        process_time_per_unit=row['Process Time per Unit (min)'],
                        required_resources=row['Required Resources'],
                        predecessors=row['Predecessors'] if pd.notna(row['Predecessors']) else '',
                        description=row['Description'] if pd.notna(row['Description']) else 'No description'
                    )
                    db.session.add(task)
                db.session.commit()
        elif 'add' in request.form:
            task_id = request.form['task_id']
            job_id = request.form['job_id']
            setup_time = request.form['setup_time']
            process_time_per_unit = request.form['process_time_per_unit']
            required_resources = request.form['required_resources']
            predecessors = request.form['predecessors']
            description = request.form['description']
            task = TaskModel(
                task_id=task_id,
                job_id=job_id,
                setup_time=setup_time,
                process_time_per_unit=process_time_per_unit,
                required_resources=required_resources,
                predecessors=predecessors,
                description=description
            )
            db.session.add(task)
            db.session.commit()
        elif 'update' in request.form:
            task_id = request.form['task_id']
            task = TaskModel.query.get(task_id)
            task.job_id = request.form['job_id']
            task.setup_time = int(request.form['setup_time'])
            task.process_time_per_unit = int(request.form['process_time_per_unit'])
            task.required_resources = request.form['required_resources']
            task.predecessors = request.form['predecessors']
            task.description = request.form['description']
            db.session.commit()
        elif 'delete' in request.form:
            task_id = request.form['task_id']
            task = TaskModel.query.get(task_id)
            db.session.delete(task)
            db.session.commit()
    jobs = JobModel.query.all()
    tasks = TaskModel.query.all()
    return render_template('add_task.html', jobs=jobs, tasks=tasks)

if __name__ == "__main__":
    app.run(debug=True)