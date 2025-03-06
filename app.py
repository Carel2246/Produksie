from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file  # Added send_file
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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobshop.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_number = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(255))
    order_date = db.Column(db.DateTime, nullable=True)
    promised_date = db.Column(db.DateTime, nullable=True)
    quantity = db.Column(db.Integer, nullable=False)
    price_each = db.Column(db.Float, nullable=False)
    customer = db.Column(db.String(100))

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_number = db.Column(db.String(50), unique=True, nullable=False)
    job_number = db.Column(db.String(50), db.ForeignKey('job.job_number'), nullable=False)
    description = db.Column(db.String(255))
    setup_time = db.Column(db.Integer, nullable=False)
    time_each = db.Column(db.Integer, nullable=False)
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
    resources = db.relationship('Resource', secondary='resource_group_association')

class Calendar(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    weekday = db.Column(db.Integer, unique=True, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)

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
                        setup_time=int(row['Setup Time']),
                        time_each=int(row['Time Each']),
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
                setup_time=int(request.form['setup_time']),
                time_each=int(request.form['time_each']),
                predecessors=request.form['predecessors'],
                resources=request.form['resources'],
                completed='completed' in request.form
            )
            if not Task.query.filter_by(task_number=task.task_number).first():
                db.session.add(task)
            db.session.commit()
        return redirect(url_for('add_task'))
    return render_template('add_task.html', tasks=Task.query.all(), jobs=Job.query.all())

@app.route('/update_task/<task_number>', methods=['POST'])
def update_task(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    task.task_number = request.form['task_number']
    task.job_number = request.form['job_number']
    task.description = request.form['description']
    task.setup_time = int(request.form['setup_time'])
    task.time_each = int(request.form['time_each'])
    task.predecessors = request.form['predecessors']
    task.resources = request.form['resources']
    task.completed = 'completed' in request.form
    db.session.commit()
    return redirect(url_for('add_task'))

@app.route('/delete_task/<task_number>', methods=['POST'])
def delete_task(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    Schedule.query.filter_by(task_number=task_number).delete()
    db.session.delete(task)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/toggle_task_completed/<task_number>', methods=['POST'])
def toggle_task_completed(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    data = request.get_json()
    task.completed = data.get('completed', False)
    db.session.commit()
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

    start_time = timer.time()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(tasks)), len(tasks))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, tasks=tasks, start_date=start_date, jobs=jobs)

    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, halloffame=hof, verbose=False)
    best_individual = hof[0]

    task_order = [tasks[i] for i in best_individual]
    task_times = generate_schedule(task_order, tasks, start_date)
    adjusted_times = adjust_to_working_hours(start_date, task_times)
    total_rand_days_late = evaluate(best_individual, tasks, start_date, jobs)[0]

    end_time = timer.time()
    elapsed_time = end_time - start_time

    db.session.query(Schedule).delete()
    db.session.commit()

    segments = []
    job_dict = {job.job_number: job for job in jobs}
    resource_dict = {res.name: res.type for res in Resource.query.all()}
    for task in tasks:
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
    for job in jobs:
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
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
    # Get start date from query parameter or default to earliest scheduled task
    start_date_str = request.args.get('start_date')
    if start_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    else:
        earliest_schedule = Schedule.query.order_by(Schedule.start_time).first()
        start_date = earliest_schedule.start_time.date() if earliest_schedule else datetime.now().date()
    
    # Get number of days from query parameter, default to 14
    days = int(request.args.get('days', 14))
    if days < 1 or days > 90:  # Reasonable bounds
        days = 14
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    today = datetime.now().date()
    
    # Navigation dates
    prev_start = (start_date - timedelta(days=7)).strftime('%Y-%m-%d')
    next_start = (start_date + timedelta(days=7)).strftime('%Y-%m-%d')

    # Get human resources
    humans = Resource.query.filter_by(type='H').all()
    if not humans:
        return render_template('production_schedule.html', error="No human resources defined.")

    # Load tasks and jobs for lookup
    tasks = Task.query.all()
    jobs = Job.query.all()
    task_dict = {task.task_number: task for task in tasks}
    job_dict = {job.job_number: job for job in jobs}

    # Build schedule dictionary: {human: {date: [tasks]}}
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
        
        # Get human resources assigned to this task
        resources = [res.strip() for res in s.resources_used.split(',')]
        for res_name in resources:
            if res_name in schedule:  # If it's a human resource
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
                           days=days)

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

if __name__ == '__main__':
    app.run(debug=True)