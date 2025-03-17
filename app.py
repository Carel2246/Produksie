from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, time
import random
from deap import base, creator, tools, algorithms
import logging
import pandas as pd
from io import BytesIO
import json
import time as timer
from sqlalchemy.sql import text
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost:5432/jobshop'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

with app.app_context():
    print(f"Database URL: {db.engine.url}")

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_number = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(255))
    order_date = db.Column(db.DateTime, nullable=True)
    promised_date = db.Column(db.DateTime, nullable=True)
    quantity = db.Column(db.Integer, nullable=False)
    price_each = db.Column(db.Float, nullable=False)
    customer = db.Column(db.String(100))
    completed = db.Column(db.Boolean, default=False, nullable=False)
    blocked = db.Column(db.Boolean, default=False, nullable=False)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_number = db.Column(db.String(50), unique=True, nullable=False)
    job_number = db.Column(db.String(50), db.ForeignKey('job.job_number'), nullable=False)
    description = db.Column(db.String(255))
    setup_time = db.Column(db.Float, nullable=False)
    time_each = db.Column(db.Float, nullable=False)
    predecessors = db.Column(db.String(255))
    resources = db.Column(db.String(255), nullable=False)
    completed = db.Column(db.Boolean, default=False, nullable=False)

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    type = db.Column(db.String(1), nullable=False)

class ResourceGroup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    type = db.Column(db.String(1), nullable=False)
    resources = db.relationship('Resource', secondary='resource_group_association')

class Calendar(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    weekday = db.Column(db.Integer, unique=True, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)

class Material(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Incremental ID
    job_number = db.Column(db.String(50), db.ForeignKey('job.job_number'), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Float, nullable=False)  # e.g., 5.5 units
    unit = db.Column(db.String(50), nullable=False)  # e.g., "kg", "meters"

class Template(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)  # e.g., "MLR10"
    description = db.Column(db.String(255))
    price_each = db.Column(db.Float, nullable=True)

class TemplateTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.Integer, db.ForeignKey('template.id'), nullable=False)
    task_number = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))
    setup_time = db.Column(db.Float, nullable=False)
    time_each = db.Column(db.Float, nullable=False)
    predecessors = db.Column(db.String(255))
    resources = db.Column(db.String(255), nullable=False)

class TemplateMaterial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.Integer, db.ForeignKey('template.id'), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(50), nullable=False)

resource_group_association = db.Table(
    'resource_group_association',
    db.Column('resource_id', db.Integer, db.ForeignKey('resource.id'), primary_key=True),
    db.Column('group_id', db.Integer, db.ForeignKey('resource_group.id'), primary_key=True)
)

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_number = db.Column(db.String(50), db.ForeignKey('task.task_number'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    resources_used = db.Column(db.String(255), nullable=False)

with app.app_context():
    db.create_all()
    if not Calendar.query.first():
        defaults = [
            (0, time(7, 0), time(16, 0)),
            (1, time(7, 0), time(16, 0)),
            (2, time(7, 0), time(16, 0)),
            (3, time(7, 0), time(16, 0)),
            (4, time(7, 0), time(13, 0)),
            (5, time(0, 0), time(0, 0)),
            (6, time(0, 0), time(0, 0))
        ]
        for weekday, start, end in defaults:
            db.session.add(Calendar(weekday=weekday, start_time=start, end_time=end))
        db.session.commit()

def load_data():
    jobs = Job.query.all()
    tasks = Task.query.all()
    resources = Resource.query.all()
    resource_groups = ResourceGroup.query.all()

    task_dict = {task.task_number: task for task in tasks}
    resource_dict = {res.name: res for res in resources}
    group_dict = {group.name: [res.name for res in group.resources] for group in resource_groups}

    for task in tasks:
        resource_items = [item.strip() for item in task.resources.split(',')]
        selected_resources = []
        for item in resource_items:
            if item in resource_dict:
                selected_resources.append(resource_dict[item])
            elif item in group_dict:
                group_resources = group_dict[item]
                if group_resources:
                    selected_resources.append(resource_dict[random.choice(group_resources)])
            else:
                logger.warning(f"Resource or group '{item}' not found for task {task.task_number}")
        task.selected_resources = selected_resources

        pred_ids = [p.strip() for p in task.predecessors.split(',')] if task.predecessors else []
        task.predecessor_tasks = [task_dict.get(p) for p in pred_ids if p in task_dict]

    return jobs, tasks

def update_job_completion(job_number):
    job = Job.query.filter_by(job_number=job_number).first()
    if not job:
        return
    
    all_tasks_done = Task.query.filter_by(job_number=job_number, completed=False).count() == 0
    job.completed = all_tasks_done
    db.session.commit()

def generate_schedule(task_order, tasks, start_date):
    resource_busy = {res.name: [] for res in Resource.query.all()}
    task_times = {}
    remaining_tasks = set(task.task_number for task in tasks if not task.completed)

    while remaining_tasks:
        scheduled = False
        for task in task_order:
            if task.task_number not in remaining_tasks:
                continue
            preds_done = all(pred.task_number in task_times or pred.completed for pred in task.predecessor_tasks)
            if not preds_done:
                continue

            earliest_start_minutes = 0
            if task.predecessor_tasks:
                pred_end_times = [task_times[pred.task_number][1] for pred in task.predecessor_tasks if pred.task_number in task_times]
                earliest_start_minutes = max(pred_end_times) if pred_end_times else 0

            duration = task.setup_time + (task.time_each * Job.query.filter_by(job_number=task.job_number).first().quantity)
            required_res = [res.name for res in task.selected_resources]

            latest_busy = earliest_start_minutes
            for res_name in required_res:
                busy_times = resource_busy[res_name]
                for start, end in sorted(busy_times):
                    if start <= latest_busy < end:
                        latest_busy = end
                    elif start > latest_busy:
                        if start - latest_busy >= duration:
                            break
                        latest_busy = end

            start_minutes = latest_busy
            end_minutes = start_minutes + duration

            task_times[task.task_number] = (start_minutes, end_minutes)
            for res_name in required_res:
                resource_busy[res_name].append((start_minutes, end_minutes))
            remaining_tasks.remove(task.task_number)
            scheduled = True

        if not scheduled and remaining_tasks:
            logger.error("Deadlock detected")
            break

    return task_times

def adjust_to_working_hours(start_date, task_times):
    calendar = {c.weekday: (c.start_time, c.end_time) for c in Calendar.query.all()}
    adjusted_times = {}

    for task_num, (start_min, end_min) in task_times.items():
        start_dt = start_date
        remaining_start = start_min
        remaining_end = end_min

        while remaining_start > 0 or remaining_end > 0:
            weekday = start_dt.weekday()
            work_start, work_end = calendar.get(weekday, (time(0, 0), time(0, 0)))
            day_start = datetime.combine(start_dt.date(), work_start)
            day_end = datetime.combine(start_dt.date(), work_end)
            day_minutes = (day_end - day_start).total_seconds() / 60 if work_start != work_end else 0

            if day_minutes == 0:
                start_dt = start_dt.replace(hour=0, minute=0) + timedelta(days=1)
                continue

            if start_dt < day_start:
                start_dt = day_start

            minutes_until_day_end = (day_end - start_dt).total_seconds() / 60
            if remaining_start > 0:
                if minutes_until_day_end >= remaining_start:
                    start_dt += timedelta(minutes=remaining_start)
                    remaining_start = 0
                else:
                    remaining_start -= minutes_until_day_end
                    start_dt = day_start + timedelta(days=1)
                continue

            if remaining_end > 0:
                if minutes_until_day_end >= remaining_end:
                    end_dt = start_dt + timedelta(minutes=remaining_end)
                    remaining_end = 0
                else:
                    remaining_end -= minutes_until_day_end
                    start_dt = day_start + timedelta(days=1)
                continue

        adjusted_times[task_num] = (start_dt, end_dt)
    return adjusted_times

def evaluate(individual, tasks, start_date, jobs):
    task_order = [tasks[i] for i in individual if not tasks[i].completed]
    task_times = generate_schedule(task_order, tasks, start_date)
    adjusted_times = adjust_to_working_hours(start_date, task_times)
    total_rand_days_late = 0
    for job in jobs:
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
        if all(t.task_number in adjusted_times or t.completed for t in job_tasks):
            end_time = max(adjusted_times[t.task_number][1] for t in job_tasks if t.task_number in adjusted_times) if any(t.task_number in adjusted_times for t in job_tasks) else start_date
            days_late = max(0, (end_time - job.promised_date).total_seconds() / 86400) if job.promised_date else 0
            total_rand_days_late += days_late * job.price_each * job.quantity
    return (total_rand_days_late,)

@app.route('/')
def index():
    return render_template('index.html', default_date=datetime.now().strftime('%Y-%m-%dT%H:%M'))

@app.route('/add_job', methods=['GET', 'POST'])
def add_job():
    if request.method == 'POST':
        if 'excel_file' in request.files:
            file = request.files['excel_file']
            if file and file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
                required_cols = {'Job Number', 'Quantity', 'Price Each'}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    return render_template('add_job.html', jobs=Job.query.all(), error=f"Missing required columns: {', '.join(missing_cols)}")
                for _, row in df.iterrows():
                    order_date = pd.to_datetime(row['Order Date']).to_pydatetime() if 'Order Date' in row and pd.notna(row['Order Date']) else None
                    promised_date = pd.to_datetime(row['Promised Date']).to_pydatetime() if 'Promised Date' in row and pd.notna(row['Promised Date']) else None
                    job = Job(
                        job_number=str(row['Job Number']),
                        description=str(row.get('Description', '')),
                        order_date=order_date,
                        promised_date=promised_date,
                        quantity=int(row['Quantity']),
                        price_each=float(row['Price Each']),
                        customer=str(row.get('Customer', ''))
                    )
                    if not Job.query.filter_by(job_number=job.job_number).first():
                        db.session.add(job)
                db.session.commit()
        else:
            order_date = request.form.get('order_date')
            promised_date = request.form.get('promised_date')
            job = Job(
                job_number=request.form['job_number'],
                description=request.form['description'],
                order_date=datetime.strptime(order_date, '%Y-%m-%dT%H:%M') if order_date else None,
                promised_date=datetime.strptime(promised_date, '%Y-%m-%dT%H:%M') if promised_date else None,
                quantity=int(request.form['quantity']),
                price_each=float(request.form['price_each']),
                customer=request.form['customer']
            )
            if not Job.query.filter_by(job_number=job.job_number).first():
                db.session.add(job)
            db.session.commit()
        return redirect(url_for('add_job'))
    return render_template('add_job.html', jobs=Job.query.all())

@app.route('/update_job/<job_number>', methods=['POST'])
def update_job(job_number):
    job = Job.query.filter_by(job_number=job_number).first_or_404()
    order_date = request.form.get('order_date')
    promised_date = request.form.get('promised_date')
    job.job_number = request.form['job_number']
    job.description = request.form['description']
    job.order_date = datetime.strptime(order_date, '%Y-%m-%dT%H:%M') if order_date else None
    job.promised_date = datetime.strptime(promised_date, '%Y-%m-%dT%H:%M') if promised_date else None
    job.quantity = int(request.form['quantity'])
    job.price_each = float(request.form['price_each'].replace('R', '').replace(' ', '').replace(',', '.'))
    job.customer = request.form['customer']
    db.session.commit()
    return redirect(url_for('add_job'))

@app.route('/delete_job/<job_number>', methods=['POST'])
def delete_job(job_number):
    job = Job.query.filter_by(job_number=job_number).first_or_404()
    Task.query.filter_by(job_number=job_number).delete()
    Schedule.query.filter_by(task_number=job_number).delete()
    db.session.delete(job)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/add_task', methods=['GET', 'POST'])
def add_task():
    if request.method == 'POST':
        if 'excel_file' in request.files:
            file = request.files['excel_file']
            if file and file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
                for _, row in df.iterrows():
                    task = Task(
                        task_number=str(row['Task Number']),
                        job_number=str(row['Job Number']),
                        description=str(row.get('Description', '')),
                        setup_time=float(row['Setup Time']),
                        time_each=float(row['Time Each']),
                        predecessors=str(row.get('Predecessors', '')),
                        resources=str(row['Resources']),
                        completed=bool(row.get('Completed', False))
                    )
                    if not Task.query.filter_by(task_number=task.task_number).first():
                        db.session.add(task)
                db.session.commit()
        else:
            task = Task(
                task_number=request.form['task_number'],
                job_number=request.form['job_number'],
                description=request.form['description'],
                setup_time=float(request.form['setup_time']),
                time_each=float(request.form['time_each']),
                predecessors=request.form['predecessors'],
                resources=request.form['resources'],
                completed='completed' in request.form
            )
            if not Task.query.filter_by(task_number=task.task_number).first():
                db.session.add(task)
            db.session.commit()
        return redirect(url_for('add_task'))
    return render_template('add_task.html', tasks=Task.query.all(), jobs=Job.query.all())

@app.route('/add_single_task/<job_number>', methods=['POST'])
def add_single_task(job_number):
    data = request.get_json()
    task = Task(
        task_number=data['task_number'],
        job_number=job_number,
        description=data.get('description', ''),
        setup_time=float(data['setup_time']),
        time_each=float(data['time_each']),
        predecessors=data.get('predecessors', ''),
        resources=data['resources'],
        completed=data.get('completed', False)
    )
    if Task.query.filter_by(task_number=task.task_number).first():
        return jsonify({"success": False, "error": f"Task number '{task.task_number}' already exists."}), 400
    db.session.add(task)
    db.session.commit()
    update_job_completion(job_number)
    return jsonify({"success": True})

@app.route('/add_single_material/<job_number>', methods=['POST'])
def add_single_material(job_number):
    data = request.get_json()
    material = Material(
        job_number=job_number,
        description=data['description'],
        quantity=float(data['quantity']),
        unit=data['unit']
    )
    db.session.add(material)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/update_material/<int:material_id>', methods=['POST'])
def update_material(material_id):
    material = Material.query.get_or_404(material_id)
    data = request.get_json()
    material.description = data['description']
    material.quantity = float(data['quantity'])
    material.unit = data['unit']
    db.session.commit()
    return jsonify({"success": True})

@app.route('/delete_material/<int:material_id>', methods=['POST'])
def delete_material(material_id):
    material = Material.query.get_or_404(material_id)
    db.session.delete(material)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/update_task/<task_number>', methods=['POST'])
def update_task(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    data = request.get_json()  # Parse JSON data from the request
    task.task_number = data['task_number']
    task.job_number = data['job_number']
    task.description = data['description']
    task.setup_time = float(data['setup_time'])
    task.time_each = float(data['time_each'])
    task.predecessors = data['predecessors']
    task.resources = data['resources']
    task.completed = data.get('completed', task.completed)  # Preserve existing value if not provided
    db.session.commit()
    update_job_completion(task.job_number)
    return jsonify({"success": True})

@app.route('/delete_task/<task_number>', methods=['POST'])
def delete_task(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    job_number = task.job_number
    Schedule.query.filter_by(task_number=task_number).delete()
    db.session.delete(task)
    db.session.commit()
    update_job_completion(job_number)
    return jsonify({"success": True})

@app.route('/toggle_task_completed/<task_number>', methods=['POST'])
def toggle_task_completed(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    data = request.get_json()
    task.completed = data.get('completed', False)
    db.session.commit()
    update_job_completion(task.job_number)
    return jsonify({"success": True})

@app.route('/add_resource', methods=['GET', 'POST'])
def add_resource():
    if request.method == 'POST':
        if 'excel_file' in request.files:
            file = request.files['excel_file']
            if file and file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
                for _, row in df.iterrows():
                    resource = Resource(
                        name=str(row['Name']),
                        type=str(row['Type'])
                    )
                    if not Resource.query.filter_by(name=resource.name).first():
                        db.session.add(resource)
                db.session.commit()
        else:
            resource = Resource(
                name=request.form['name'],
                type=request.form['type']
            )
            if not Resource.query.filter_by(name=resource.name).first():
                db.session.add(resource)
            db.session.commit()
        return redirect(url_for('add_resource'))
    return render_template('add_resource.html', resources=Resource.query.all())

@app.route('/update_resource/<int:id>', methods=['POST'])
def update_resource(id):
    resource = Resource.query.get_or_404(id)
    resource.name = request.form['name']
    resource.type = request.form['type']
    db.session.commit()
    return redirect(url_for('add_resource'))

@app.route('/delete_resource/<int:id>', methods=['POST'])
def delete_resource(id):
    resource = Resource.query.get_or_404(id)
    db.session.delete(resource)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/add_resource_group', methods=['GET', 'POST'])
def add_resource_group():
    if request.method == 'POST':
        if 'excel_file' in request.files:
            file = request.files['excel_file']
            if file and file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
                for _, row in df.iterrows():
                    name = str(row['Name'])
                    resource_names = [r.strip() for r in str(row['Resources']).split(',')]
                    resources = Resource.query.filter(Resource.name.in_(resource_names)).all()
                    group = ResourceGroup(name=name)
                    group.resources = resources
                    if not ResourceGroup.query.filter_by(name=name).first():
                        db.session.add(group)
                db.session.commit()
        else:
            name = request.form['name']
            resource_names = [r.strip() for r in request.form['resources'].split(',')]
            resources = Resource.query.filter(Resource.name.in_(resource_names)).all()
            group = ResourceGroup(name=name)
            group.resources = resources
            if not ResourceGroup.query.filter_by(name=name).first():
                db.session.add(group)
            db.session.commit()
        return redirect(url_for('add_resource_group'))
    return render_template('add_resource_group.html', groups=ResourceGroup.query.all(), all_resources=Resource.query.all())

@app.route('/update_resource_group/<int:id>', methods=['POST'])
def update_resource_group(id):
    group = ResourceGroup.query.get_or_404(id)
    group.name = request.form['name']
    resource_names = [r.strip() for r in request.form['resources'].split(',')]
    group.resources = Resource.query.filter(Resource.name.in_(resource_names)).all()
    db.session.commit()
    return redirect(url_for('add_resource_group'))

@app.route('/delete_resource_group/<int:id>', methods=['POST'])
def delete_resource_group(id):
    group = ResourceGroup.query.get_or_404(id)
    db.session.delete(group)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/add_calendar', methods=['GET', 'POST'])
def add_calendar():
    if request.method == 'POST':
        weekday = int(request.form['weekday'])
        start_time = datetime.strptime(request.form['start_time'], '%H:%M').time()
        end_time = datetime.strptime(request.form['end_time'], '%H:%M').time()
        cal = Calendar.query.filter_by(weekday=weekday).first()
        if cal:
            cal.start_time = start_time
            cal.end_time = end_time
        else:
            cal = Calendar(weekday=weekday, start_time=start_time, end_time=end_time)
            db.session.add(cal)
        db.session.commit()
        return redirect(url_for('add_calendar'))
    return render_template('add_calendar.html', calendar=Calendar.query.all())

@app.route('/delete_calendar/<int:weekday>', methods=['POST'])
def delete_calendar(weekday):
    cal = Calendar.query.filter_by(weekday=weekday).first_or_404()
    db.session.delete(cal)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/schedule', methods=['POST'])
def schedule():
    start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%dT%H:%M')
    jobs, tasks = load_data()

    if not tasks:
        return render_template('index.html', error="No tasks to schedule", default_date=start_date.strftime('%Y-%m-%dT%H:%M'))

    # Filter out tasks from blocked jobs
    active_jobs = [job for job in jobs if not job.blocked]
    active_tasks = [task for task in tasks if task.job_number in [job.job_number for job in active_jobs]]

    if not active_tasks:
        return render_template('index.html', error="No active tasks to schedule (all jobs blocked)", default_date=start_date.strftime('%Y-%m-%dT%H:%M'))

    start_time = timer.time()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(active_tasks)), len(active_tasks))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, tasks=active_tasks, start_date=start_date, jobs=active_jobs)

    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, halloffame=hof, verbose=False)
    best_individual = hof[0]

    task_order = [active_tasks[i] for i in best_individual]
    task_times = generate_schedule(task_order, active_tasks, start_date)
    adjusted_times = adjust_to_working_hours(start_date, task_times)
    total_rand_days_late = evaluate(best_individual, active_tasks, start_date, active_jobs)[0]

    # Rest of the function remains unchanged, just use active_jobs and active_tasks
    end_time = timer.time()
    elapsed_time = end_time - start_time

    db.session.query(Schedule).delete()
    db.session.commit()

    segments = []
    job_dict = {job.job_number: job for job in jobs}  # Use all jobs for lookup
    resource_dict = {res.name: res.type for res in Resource.query.all()}
    for task in active_tasks:  # Only active tasks
        if task.task_number in adjusted_times:
            start, end = adjusted_times[task.task_number]
            machines = ', '.join(res.name for res in task.selected_resources if resource_dict[res.name] == 'M')
            people = ', '.join(res.name for res in task.selected_resources if resource_dict[res.name] == 'H')
            job = job_dict.get(task.job_number)
            schedule_entry = Schedule(
                task_number=task.task_number,
                start_time=start,
                end_time=end,
                resources_used=', '.join(res.name for res in task.selected_resources)
            )
            db.session.add(schedule_entry)
            segments.append({
                'task_id': task.task_number,
                'job_id': task.job_number,
                'job_description': job.description if job else '',
                'job_quantity': job.quantity if job else 0,
                'description': task.description,
                'machines': machines,
                'people': people,
                'start': start,
                'end': end
            })

    job_data = []
    for job in active_jobs:  # Only active jobs
        job_tasks = [t for t in active_tasks if t.job_number == job.job_number]
        if all(t.task_number in adjusted_times or t.completed for t in job_tasks):
            expected_finish = max(adjusted_times[t.task_number][1] for t in job_tasks if t.task_number in adjusted_times) if any(t.task_number in adjusted_times for t in job_tasks) else start_date
            days_late = max(0, (expected_finish - job.promised_date).total_seconds() / 86400) if job.promised_date else 0
            job_data.append({
                'job_id': job.job_number,
                'promised_date': job.promised_date,
                'total_value': job.quantity * job.price_each,
                'expected_finish': expected_finish,
                'rand_days_late': days_late
            })
    db.session.commit()

    return render_template('schedule.html', segments=segments, total_rand_days_late=total_rand_days_late, jobs=job_data, elapsed_time=elapsed_time)

@app.route('/view_schedule')
def view_schedule():
    schedules = Schedule.query.all()
    segments = []
    jobs = Job.query.all()
    tasks = Task.query.all()
    job_dict = {job.job_number: job for job in jobs}
    task_dict = {task.task_number: task for task in tasks}
    resource_dict = {res.name: res.type for res in Resource.query.all()}
    task_times = {s.task_number: (s.start_time, s.end_time) for s in schedules}
    
    for s in schedules:
        task = task_dict.get(s.task_number)
        job = job_dict.get(task.job_number if task else '')
        if task:
            resources = [res.strip() for res in s.resources_used.split(',')]
            machines = ', '.join(res for res in resources if resource_dict.get(res, '') == 'M')
            people = ', '.join(res for res in resources if resource_dict.get(res, '') == 'H')
            segments.append({
                'task_id': s.task_number,
                'job_id': task.job_number,
                'job_description': job.description if job else '',
                'job_quantity': job.quantity if job else 0,
                'description': task.description,
                'machines': machines,
                'people': people,
                'start': s.start_time,
                'end': s.end_time
            })

    total_rand_days_late = 0
    job_data = []
    for job in jobs:
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
        if all(t.task_number in task_times or t.completed for t in job_tasks):
            expected_finish = max(task_times[t.task_number][1] for t in job_tasks if t.task_number in task_times) if any(t.task_number in task_times for t in job_tasks) else datetime.now()
            days_late = max(0, (expected_finish - job.promised_date).total_seconds() / 86400) if job.promised_date else 0
            total_rand_days_late += days_late * job.price_each * job.quantity
            job_data.append({
                'job_id': job.job_number,
                'promised_date': job.promised_date,
                'total_value': job.quantity * job.price_each,
                'expected_finish': expected_finish,
                'rand_days_late': days_late
            })

    return render_template('schedule.html', segments=segments, total_rand_days_late=total_rand_days_late, jobs=job_data, elapsed_time=None)

@app.route('/gantt')
def gantt():
    schedules = Schedule.query.all()
    if not schedules:
        return render_template('gantt.html', error="No schedule available. Please generate a schedule first.")
    
    segments = []
    tasks = Task.query.all()
    for s in schedules:
        task = next((t for t in tasks if t.task_number == s.task_number), None)
        if task:
            segments.append({
                'task_id': s.task_number,
                'job_id': task.job_number,
                'description': task.description,
                'resources': s.resources_used,
                'start': s.start_time.isoformat(),
                'end': s.end_time.isoformat()
            })
    
    if not segments:
        return render_template('gantt.html', error="No valid tasks found in the schedule.")
    
    segments_json = json.dumps(segments)
    return render_template('gantt.html', segments_json=segments_json)

@app.route('/production_schedule')
def production_schedule():
    start_date_str = request.args.get('start_date')
    if start_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    else:
        earliest_schedule = Schedule.query.order_by(Schedule.start_time).first()
        start_date = earliest_schedule.start_time.date() if earliest_schedule else datetime.now().date()
    
    days = int(request.args.get('days', 14))
    if days < 1 or days > 90:
        days = 14
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    today = datetime.now().date()
    
    prev_start = (start_date - timedelta(days=7)).strftime('%Y-%m-%d')
    next_start = (start_date + timedelta(days=7)).strftime('%Y-%m-%d')

    humans = Resource.query.filter_by(type='H').all()
    if not humans:
        return render_template('production_schedule.html', error="No human resources defined.")

    tasks = Task.query.all()
    jobs = Job.query.all()
    blocked_jobs = [job for job in jobs if job.blocked]  # Get blocked jobs
    task_dict = {task.task_number: task for task in tasks}
    job_dict = {job.job_number: job for job in jobs}

    schedule = {human.name: {date: [] for date in dates} for human in humans}
    schedules = Schedule.query.all()

    for s in schedules:
        task = task_dict.get(s.task_number)
        if not task:
            continue
        job = job_dict.get(task.job_number)
        task_date = s.start_time.date()
        if task_date not in dates:
            continue
        
        resources = [res.strip() for res in s.resources_used.split(',')]
        for res_name in resources:
            if res_name in schedule:
                schedule[res_name][task_date].append({
                    'task_description': task.description,
                    'job_description': job.description if job else task.job_number
                })

    return render_template('production_schedule.html', 
                           humans=humans, 
                           dates=dates, 
                           schedule=schedule, 
                           today=today, 
                           prev_start=prev_start, 
                           next_start=next_start, 
                           start_date=start_date, 
                           days=days,
                           blocked_jobs=blocked_jobs)  # Pass blocked jobs

@app.route('/export_production_schedule')
def export_production_schedule():
    # Get start date and days from query parameters
    start_date_str = request.args.get('start_date')
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else datetime.now().date()
    days = int(request.args.get('days', 14))
    if days < 1 or days > 90:
        days = 14
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Get human resources
    humans = Resource.query.filter_by(type='H').all()
    if not humans:
        return "No human resources defined.", 400

    # Load tasks and jobs for lookup
    tasks = Task.query.all()
    jobs = Job.query.all()
    task_dict = {task.task_number: task for task in tasks}
    job_dict = {job.job_number: job for job in jobs}

    # Build schedule dictionary
    schedule = {human.name: {date: [] for date in dates} for human in humans}
    schedules = Schedule.query.all()

    for s in schedules:
        task = task_dict.get(s.task_number)
        if not task:
            continue
        job = job_dict.get(task.job_number)
        task_date = s.start_time.date()
        if task_date not in dates:
            continue
        
        resources = [res.strip() for res in s.resources_used.split(',')]
        for res_name in resources:
            if res_name in schedule:
                schedule[res_name][task_date].append({
                    'task_description': task.description,
                    'job_description': job.description if job else task.job_number
                })

    # Create DataFrame with blank A1
    data = {'': [''] + [human.name for human in humans]}  # Empty header, then names
    for date in dates:
        column = [date.strftime('%d %b %a')] + ['' for _ in humans]  # Date header, then empty cells
        for i, human in enumerate(humans, start=1):  # Start=1 to skip header row
            tasks = schedule[human.name][date]
            cell = '\n'.join(f"* {t['job_description']} - {t['task_description']}" for t in tasks)
            column[i] = cell
        data[date.strftime('%d %b %a')] = column
    
    df = pd.DataFrame(data)
    
    # Export to Excel with formatting
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Production Schedule', index=False, header=False)  # No header row from DF
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Production Schedule']
        
        # Define formats
        bold_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
        cell_format = workbook.add_format({'border': 1, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter'})
        
        # Apply bold to date headers (row 0) and resource names (column 0, rows 1+)
        worksheet.set_row(0, None, bold_format)  # Date headers
        for row, human in enumerate(humans, start=1):
            worksheet.write(row, 0, human.name, bold_format)  # Resource names
        
        # Set column widths
        worksheet.set_column(0, 0, 10, bold_format)  # Column A: 10 pixels, bold
        worksheet.set_column(1, days, 17.5, cell_format)  # Date columns: 17.5 units
        
        # Page setup
        worksheet.set_landscape()  # Landscape orientation
        worksheet.set_paper(9)  # A4 paper size
        worksheet.set_margins(left=0.197, right=0.197, top=0.197, bottom=0.197)  # 0.5cm ≈ 0.197 inches
        pages = math.ceil(days / 7)  # Number of pages based on days/7
        worksheet.fit_to_pages(pages, 0)  # Fit to ceil(days/7) pages wide
        worksheet.repeat_rows(0)  # Repeat headers on each page
        worksheet.repeat_columns(0)  # Repeat column A on each page
    
    output.seek(0)
    filename = f"Produksieskedule {datetime.now().strftime('%Y-%m-%d')}.xlsx"
    return send_file(output, download_name=filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/add_job_with_tasks', methods=['GET', 'POST'])
def add_job_with_tasks():
    templates = Template.query.all()
    jobs = Job.query.all()
    # Split resources and groups by type
    machine_resources = Resource.query.filter_by(type='M').all()
    human_resources = Resource.query.filter_by(type='H').all()
    machine_groups = ResourceGroup.query.filter_by(type='M').all()
    human_groups = ResourceGroup.query.filter_by(type='H').all()

    if request.method == 'POST':
        order_date_str = request.form.get('order_date')
        promised_date_str = request.form.get('promised_date')
        order_date = datetime.fromisoformat(order_date_str) if order_date_str else None
        promised_date = datetime.fromisoformat(promised_date_str) if promised_date_str else None

        job = Job(
            job_number=request.form['job_number'],
            description=request.form.get('description', ''),
            order_date=order_date,
            promised_date=promised_date,
            quantity=int(request.form['quantity']),
            price_each=float(request.form['price_each']),
            customer=request.form.get('customer', ''),
            blocked='blocked' in request.form
        )
        if Job.query.filter_by(job_number=job.job_number).first():
            return render_template('add_job_with_tasks.html', templates=templates, jobs=jobs, 
                                 machine_resources=machine_resources, human_resources=human_resources,
                                 machine_groups=machine_groups, human_groups=human_groups,
                                 error=f"Job number '{job.job_number}' already exists.")

        Task.query.filter_by(job_number=job.job_number).delete()
        Material.query.filter_by(job_number=job.job_number).delete()

        tasks_data = {}
        materials_data = {}
        for key in request.form:
            if key.startswith('tasks['):
                parts = key[6:-1].split('][')
                index, field = parts[0], parts[1]
                if index not in tasks_data:
                    tasks_data[index] = {}
                tasks_data[index][field] = request.form[key]
            elif key.startswith('materials['):
                parts = key[10:-1].split('][')
                index, field = parts[0], parts[1]
                if index not in materials_data:
                    materials_data[index] = {}
                materials_data[index][field] = request.form[key]

        # Combine machine and human resources into a single string for validation
        valid_resources = {r.name for r in Resource.query.all()}
        valid_groups = {g.name for g in ResourceGroup.query.all()}
        all_valid = valid_resources.union(valid_groups)

        for index, task_info in tasks_data.items():
            # Combine machines and humans for validation
            resources = ','.join(filter(None, [task_info.get('machines', ''), task_info.get('humans', '')])).split(',')
            invalid = [r.strip() for r in resources if r.strip() and r.strip() not in all_valid]
            if invalid:
                return render_template('add_job_with_tasks.html', templates=templates, jobs=jobs, 
                                     machine_resources=machine_resources, human_resources=human_resources,
                                     machine_groups=machine_groups, human_groups=human_groups,
                                     error=f"Invalid resources in task {task_info['task_number']}: {', '.join(invalid)}")

        db.session.add(job)
        for index, task_info in tasks_data.items():
            combined_resources = ','.join(filter(None, [task_info.get('machines', ''), task_info.get('humans', '')]))
            task = Task(
                task_number=task_info['task_number'],
                job_number=job.job_number,
                description=task_info.get('description', ''),
                setup_time=float(task_info['setup_time']),
                time_each=float(task_info['time_each']),
                predecessors=task_info.get('predecessors', ''),
                resources=combined_resources,
                completed='completed' in task_info
            )
            if Task.query.filter_by(task_number=task.task_number).first():
                db.session.rollback()
                return render_template('add_job_with_tasks.html', templates=templates, jobs=jobs, 
                                     machine_resources=machine_resources, human_resources=human_resources,
                                     machine_groups=machine_groups, human_groups=human_groups,
                                     error=f"Task number '{task.task_number}' already exists.")
            db.session.add(task)

        for index, material_info in materials_data.items():
            material = Material(
                job_number=job.job_number,
                description=material_info['description'],
                quantity=float(material_info['quantity']),
                unit=material_info['unit']
            )
            db.session.add(material)

        db.session.commit()
        return redirect(url_for('add_job_with_tasks'))

    return render_template('add_job_with_tasks.html', templates=templates, jobs=jobs, 
                         machine_resources=machine_resources, human_resources=human_resources,
                         machine_groups=machine_groups, human_groups=human_groups)

@app.route('/get_template_data/<int:template_id>', methods=['GET'])
def get_template_data(template_id):
    template = Template.query.get_or_404(template_id)
    job_number = request.args.get('job_number', '')
    tasks = TemplateTask.query.filter_by(template_id=template.id).all()
    materials = TemplateMaterial.query.filter_by(template_id=template.id).all()
    data = {
        'template_description': template.description or template.name,
        'template_price_each': template.price_each,
        'tasks': [{
            'task_number': f"{job_number}-{t.task_number}",
            'description': t.description or '',
            'setup_time': t.setup_time,
            'time_each': t.time_each,
            'predecessors': ', '.join(f"{job_number}-{p.strip()}" for p in t.predecessors.split(',')) if t.predecessors else '',
            'resources': t.resources,
            'completed': False
        } for t in tasks],
        'materials': [{
            'description': m.description,
            'quantity': m.quantity,  # Base quantity, scaled in JS
            'unit': m.unit
        } for m in materials]
    }
    return jsonify(data)

@app.route('/add_template', methods=['GET', 'POST'])
def add_template():
    if request.method == 'POST':
        template = Template(
            name=request.form['name'],
            description=request.form['description'],
            price_each=float(request.form['price_each']) if request.form.get('price_each') else None
        )
        if Template.query.filter_by(name=template.name).first():
            return render_template('add_template.html', templates=Template.query.all(), error=f"Template '{template.name}' already exists.")
        
        db.session.add(template)
        db.session.flush()  # Get template.id before commit

        tasks_data = {}
        materials_data = {}
        for key in request.form:
            if key.startswith('tasks['):
                parts = key.split('[')
                index = int(parts[1].rstrip(']'))
                field = parts[2].rstrip(']')
                if index not in tasks_data:
                    tasks_data[index] = {}
                tasks_data[index][field] = request.form[key]
            elif key.startswith('materials['):
                parts = key.split('[')
                index = int(parts[1].rstrip(']'))
                field = parts[2].rstrip(']')
                if index not in materials_data:
                    materials_data[index] = {}
                materials_data[index][field] = request.form[key]

        for index, task_info in tasks_data.items():
            task = TemplateTask(
                template_id=template.id,
                task_number=task_info['task_number'],
                description=task_info.get('description', ''),
                setup_time=float(task_info['setup_time']),
                time_each=float(task_info['time_each']),
                predecessors=task_info.get('predecessors', ''),
                resources=task_info['resources']
            )
            db.session.add(task)

        for index, material_info in materials_data.items():
            material = TemplateMaterial(
                template_id=template.id,
                description=material_info['description'],
                quantity=float(material_info['quantity']),
                unit=material_info['unit']
            )
            db.session.add(material)

        db.session.commit()
        return redirect(url_for('add_template'))

    return render_template('add_template.html', templates=Template.query.all())

@app.route('/edit_template/<int:template_id>', methods=['GET', 'POST'])
def edit_template(template_id):
    template = Template.query.get_or_404(template_id)
    
    if request.method == 'POST':
        template.name = request.form['name']
        template.description = request.form['description']
        template.price_each = float(request.form['price_each']) if request.form.get('price_each') else None        
        # Delete existing tasks and materials
        TemplateTask.query.filter_by(template_id=template.id).delete()
        TemplateMaterial.query.filter_by(template_id=template.id).delete()

        tasks_data = {}
        materials_data = {}
        for key in request.form:
            if key.startswith('tasks['):
                parts = key.split('[')
                index = int(parts[1].rstrip(']'))
                field = parts[2].rstrip(']')
                if index not in tasks_data:
                    tasks_data[index] = {}
                tasks_data[index][field] = request.form[key]
            elif key.startswith('materials['):
                parts = key.split('[')
                index = int(parts[1].rstrip(']'))
                field = parts[2].rstrip(']')
                if index not in materials_data:
                    materials_data[index] = {}
                materials_data[index][field] = request.form[key]

        for index, task_info in tasks_data.items():
            task = TemplateTask(
                template_id=template.id,
                task_number=task_info['task_number'],
                description=task_info.get('description', ''),
                setup_time=float(task_info['setup_time']),
                time_each=float(task_info['time_each']),
                predecessors=task_info.get('predecessors', ''),
                resources=task_info['resources']
            )
            db.session.add(task)

        for index, material_info in materials_data.items():
            material = TemplateMaterial(
                template_id=template.id,
                description=material_info['description'],
                quantity=float(material_info['quantity']),
                unit=material_info['unit']
            )
            db.session.add(material)

        db.session.commit()
        return redirect(url_for('add_template'))

    tasks = TemplateTask.query.filter_by(template_id=template.id).all()
    materials = TemplateMaterial.query.filter_by(template_id=template.id).all()
    return render_template('edit_template.html', template=template, tasks=tasks, materials=materials)

@app.route('/validate_resources', methods=['POST'])
def validate_resources():
    data = request.get_json()
    resource_data = data.get('resources', [])
    
    # Get all valid resource and group names
    valid_resources = {res.name for res in Resource.query.all()}
    valid_groups = {group.name for group in ResourceGroup.query.all()}
    valid_names = valid_resources.union(valid_groups)
    
    # Check each row's resources
    invalid_entries = []
    for entry in resource_data:
        resources = entry['resources']
        invalid = [r for r in resources if r and r not in valid_names]
        if invalid:
            invalid_entries.append({'row': entry['row'], 'invalid': invalid})
    
    return jsonify({'invalid': invalid_entries})

@app.route('/review_jobs', methods=['GET', 'POST'])
def review_jobs():
    include_completed = request.args.get('include_completed', 'false').lower() == 'true'
    if include_completed:
        jobs = Job.query.all()
    else:
        jobs = Job.query.filter_by(completed=False).all()

    machine_resources = Resource.query.filter_by(type='M').all()
    human_resources = Resource.query.filter_by(type='H').all()
    machine_groups = ResourceGroup.query.filter_by(type='M').all()
    human_groups = ResourceGroup.query.filter_by(type='H').all()

    selected_job = None
    tasks = []
    materials = []

    # Handle job selection via POST or GET
    job_number = request.form.get('job_number') if request.method == 'POST' else request.args.get('job_number')
    print(f"Method: {request.method}, Job number: {job_number}")
    if job_number:
        selected_job = Job.query.filter_by(job_number=job_number).first()
        print(f"Selected job: {selected_job.job_number if selected_job else 'None'}")
        if selected_job:
            tasks = Task.query.filter_by(job_number=job_number).all()
            materials = Material.query.filter_by(job_number=job_number).all()

    if request.method == 'POST' and 'update' in request.form and selected_job:
        # Update job fields
        selected_job.description = request.form.get('description', '')
        order_date_str = request.form.get('order_date')
        promised_date_str = request.form.get('promised_date')
        selected_job.order_date = datetime.fromisoformat(order_date_str) if order_date_str else None
        selected_job.promised_date = datetime.fromisoformat(promised_date_str) if promised_date_str else None
        selected_job.quantity = int(request.form['quantity'])
        selected_job.price_each = float(request.form['price_each'])
        selected_job.customer = request.form.get('customer', '')
        selected_job.blocked = 'blocked' in request.form
        selected_job.completed = 'completed' in request.form

        # Parse tasks and materials from form
        tasks_data = {}
        materials_data = {}
        for key in request.form:
            if key.startswith('tasks['):
                parts = key[6:-1].split('][')
                index, field = parts[0], parts[1]
                if index not in tasks_data:
                    tasks_data[index] = {}
                tasks_data[index][field] = request.form[key]
            elif key.startswith('materials['):
                parts = key[10:-1].split('][')
                index, field = parts[0], parts[1]
                if index not in materials_data:
                    materials_data[index] = {}
                materials_data[index][field] = request.form[key]

        # Update or add tasks
        existing_tasks = {task.task_number: task for task in tasks}
        valid_resources = {r.name for r in Resource.query.all()}
        valid_groups = {g.name for g in ResourceGroup.query.all()}
        all_valid = valid_resources.union(valid_groups)

        submitted_task_numbers = {task_info['task_number'] for task_info in tasks_data.values()}
        for task in tasks:
            if task.task_number not in submitted_task_numbers:
                db.session.delete(task)

        for index, task_info in tasks_data.items():
            resources = ','.join(filter(None, [task_info.get('machines', ''), task_info.get('humans', '')]))
            invalid = [r.strip() for r in resources.split(',') if r.strip() and r.strip() not in all_valid]
            if invalid:
                return render_template('review_jobs.html', jobs=jobs, selected_job=selected_job, tasks=tasks, materials=materials,
                                     include_completed=include_completed, machine_resources=machine_resources, human_resources=human_resources,
                                     machine_groups=machine_groups, human_groups=human_groups,
                                     error=f"Invalid resources in task {task_info['task_number']}: {', '.join(invalid)}")
            task_number = task_info['task_number']
            if task_number in existing_tasks:
                # Update existing task
                task = existing_tasks[task_number]
                task.description = task_info.get('description', '')
                task.setup_time = float(task_info['setup_time'])
                task.time_each = float(task_info['time_each'])
                task.predecessors = task_info.get('predecessors', '')
                task.resources = resources
                task.completed = 'completed' in task_info
            else:
                # Add new task
                task = Task(
                    task_number=task_number,
                    job_number=job_number,
                    description=task_info.get('description', ''),
                    setup_time=float(task_info['setup_time']),
                    time_each=float(task_info['time_each']),
                    predecessors=task_info.get('predecessors', ''),
                    resources=resources,
                    completed='completed' in task_info
                )
                db.session.add(task)

        # Update materials
        Material.query.filter_by(job_number=job_number).delete()
        for index, material_info in materials_data.items():
            material = Material(
                job_number=job_number,
                description=material_info['description'],
                quantity=float(material_info['quantity']),
                unit=material_info['unit']
            )
            db.session.add(material)

        db.session.commit()
        return redirect(url_for('review_jobs', include_completed=include_completed, job_number=job_number))

    return render_template('review_jobs.html', jobs=jobs, selected_job=selected_job, tasks=tasks, materials=materials,
                          include_completed=include_completed, machine_resources=machine_resources, human_resources=human_resources,
                          machine_groups=machine_groups, human_groups=human_groups)

@app.route('/cash_flow', methods=['GET'])
def cash_flow():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    # Determine default date range from scheduled tasks
    first_scheduled = Schedule.query.order_by(Schedule.start_time).first()
    last_scheduled = Schedule.query.order_by(Schedule.end_time.desc()).first()

    default_start_date = first_scheduled.start_time.date() if first_scheduled else datetime.today().date()
    default_end_date = last_scheduled.end_time.date() if last_scheduled else datetime.today().date()

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str else default_start_date
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else default_end_date

    logger.debug(f"Cash Flow Projection Date Range: Start={start_date}, End={end_date}")

    # Get only jobs that are NOT completed
    pending_jobs = Job.query.filter_by(completed=False).all()
    logger.debug(f"Total Unfinished Jobs: {len(pending_jobs)}")

    job_values = {}
    schedules = Schedule.query.all()

    for job in pending_jobs:
        job_tasks = Task.query.filter_by(job_number=job.job_number).all()
        if not job_tasks:
            logger.debug(f"Job {job.job_number} has no tasks.")
            continue

        # Find estimated completion date (last scheduled task's end time)
        estimated_completion = max(
            (s.end_time for s in schedules if s.task_number in [t.task_number for t in job_tasks]),
            default=None
        )

        if estimated_completion:
            logger.debug(f"Job {job.job_number} - Estimated Completion on {estimated_completion.date()}")
        else:
            logger.debug(f"Job {job.job_number} has no scheduled tasks.")

        if estimated_completion and start_date <= estimated_completion.date() <= end_date:
            date_key = estimated_completion.date().strftime("%Y-%m-%d")
            job_value = job.quantity * job.price_each
            job_values[date_key] = job_values.get(date_key, 0) + job_value
        else:
            logger.debug(f"Job {job.job_number} is out of the selected range.")

    logger.debug(f"Projected Job Values by Date: {job_values}")

    # Generate cumulative cash flow projection
    sorted_dates = sorted(job_values.keys())
    cumulative_values = []
    total = 0

    for date in sorted_dates:
        total += job_values[date]
        cumulative_values.append({"date": date, "value": total})

    logger.debug(f"Cumulative Projection Values: {cumulative_values}")

    return render_template("cash_flow.html", data=cumulative_values, start_date=start_date, end_date=end_date)

@app.route('/delivery_schedule', methods=['GET'])
def delivery_schedule():
    schedules = Schedule.query.all()
    if not schedules:
        return render_template('delivery_schedule.html', error="No scheduled tasks available.")

    jobs = Job.query.all()
    tasks = Task.query.all()
    blocked_jobs = [job for job in jobs if job.blocked]  # Get blocked jobs
    job_dict = {job.job_number: job for job in jobs}
    task_dict = {task.task_number: task for task in tasks}

    start_date = min(s.start_time for s in schedules).date()
    end_date = max(s.end_time for s in schedules).date() + timedelta(days=1)

    job_completion = {}
    max_delivery_date = end_date
    for job in jobs:
        if job.blocked:  # Skip blocked jobs for scheduling
            continue
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
        scheduled_tasks = [t for t in job_tasks if t.task_number in [s.task_number for s in schedules]]
        
        if scheduled_tasks and all(t.completed or t.task_number in [s.task_number for s in schedules] for t in job_tasks):
            completion_date = max(s.end_time for s in schedules if s.task_number in [t.task_number for t in scheduled_tasks]).date()
            delivery_date = completion_date + timedelta(days=1)
            while delivery_date.weekday() >= 5:
                delivery_date += timedelta(days=1)
            is_late = job.promised_date and delivery_date > job.promised_date.date() if job.promised_date else False
            description = f"{job.customer or 'Unknown'} - {job.description} ({job.job_number})"
            if is_late and job.promised_date:
                description += f" (Promised: {job.promised_date.date().strftime('%Y-%m-%d')})"
            job_completion[job.job_number] = {
                'delivery_date': delivery_date,
                'is_late': is_late,
                'description': description
            }
            if delivery_date > max_delivery_date:
                max_delivery_date = delivery_date

    dates = [start_date + timedelta(days=i) for i in range((max_delivery_date - start_date).days + 1)]
    delivery_schedule = {date: [] for date in dates}
    for job_num, info in job_completion.items():
        delivery_schedule[info['delivery_date']].append({
            'description': info['description'],
            'is_late': info['is_late']
        })

    return render_template('delivery_schedule.html',
                          dates=dates,
                          delivery_schedule=delivery_schedule,
                          start_date=start_date,
                          end_date=max_delivery_date,
                          blocked_jobs=blocked_jobs)  # Pass blocked jobs

@app.route('/toggle_job_blocked/<job_number>', methods=['POST'])
def toggle_job_blocked(job_number):
    job = Job.query.filter_by(job_number=job_number).first_or_404()
    data = request.get_json()
    job.blocked = data.get('blocked', False)
    db.session.commit()
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True)