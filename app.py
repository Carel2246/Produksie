from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, time
import random
from deap import base, creator, tools, algorithms
import logging
import pandas as pd
from io import BytesIO
import json
import time as timer  # For timing the scheduling process

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobshop.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database Models
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_number = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(255))
    order_date = db.Column(db.DateTime, nullable=False)
    promised_date = db.Column(db.DateTime, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price_each = db.Column(db.Float, nullable=False)  # In ZAR
    customer = db.Column(db.String(100))

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_number = db.Column(db.String(50), unique=True, nullable=False)
    job_number = db.Column(db.String(50), db.ForeignKey('job.job_number'), nullable=False)
    description = db.Column(db.String(255))
    setup_time = db.Column(db.Integer, nullable=False)  # Minutes
    time_each = db.Column(db.Integer, nullable=False)   # Minutes per unit
    predecessors = db.Column(db.String(255))            # Comma-separated task_numbers
    resources = db.Column(db.String(255), nullable=False)  # Comma-separated resource or group names

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    type = db.Column(db.String(1), nullable=False)  # 'H' or 'M'

class ResourceGroup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    resources = db.relationship('Resource', secondary='resource_group_association')

class Calendar(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    weekday = db.Column(db.Integer, unique=True, nullable=False)  # 0=Monday, 6=Sunday
    start_time = db.Column(db.Time, nullable=False)              # e.g., 07:00
    end_time = db.Column(db.Time, nullable=False)                # e.g., 16:00

resource_group_association = db.Table(
    'resource_group_association',
    db.Column('resource_id', db.Integer, db.ForeignKey('resource.id'), primary_key=True),
    db.Column('group_id', db.Integer, db.ForeignKey('resource_group.id'), primary_key=True)
)

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_number = db.Column(db.String(50), db.ForeignKey('task.task_number'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)  # Adjusted to working hours
    end_time = db.Column(db.DateTime, nullable=False)    # Adjusted to working hours
    resources_used = db.Column(db.String(255), nullable=False)  # Comma-separated resource names

# Initialize database
with app.app_context():
    db.create_all()
    if not Calendar.query.first():
        defaults = [
            (0, time(7, 0), time(16, 0)),  # Monday
            (1, time(7, 0), time(16, 0)),  # Tuesday
            (2, time(7, 0), time(16, 0)),  # Wednesday
            (3, time(7, 0), time(16, 0)),  # Thursday
            (4, time(7, 0), time(13, 0)),  # Friday
            (5, time(0, 0), time(0, 0)),   # Saturday (non-working)
            (6, time(0, 0), time(0, 0))    # Sunday (non-working)
        ]
        for weekday, start, end in defaults:
            db.session.add(Calendar(weekday=weekday, start_time=start, end_time=end))
        db.session.commit()

# Helper Functions
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
    remaining_tasks = set(task.task_number for task in tasks)

    while remaining_tasks:
        scheduled = False
        for task in task_order:
            if task.task_number not in remaining_tasks:
                continue
            preds_done = all(pred.task_number in task_times for pred in task.predecessor_tasks)
            if not preds_done:
                continue

            earliest_start_minutes = 0
            if task.predecessor_tasks:
                earliest_start_minutes = max(task_times[pred.task_number][1] for pred in task.predecessor_tasks)

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
    task_order = [tasks[i] for i in individual]
    task_times = generate_schedule(task_order, tasks, start_date)
    adjusted_times = adjust_to_working_hours(start_date, task_times)
    total_rand_days_late = 0
    for job in jobs:
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
        if all(t.task_number in adjusted_times for t in job_tasks):
            end_time = max(adjusted_times[t.task_number][1] for t in job_tasks)
            days_late = max(0, (end_time - job.promised_date).total_seconds() / 86400)
            total_rand_days_late += days_late * job.price_each * job.quantity
    return (total_rand_days_late,)

# Routes
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
                required_cols = {'Job Number', 'Order Date', 'Promised Date', 'Quantity', 'Price Each'}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    return render_template('add_job.html', jobs=Job.query.all(), error=f"Missing required columns: {', '.join(missing_cols)}")
                df['Order Date'] = pd.to_datetime(df['Order Date'])
                df['Promised Date'] = pd.to_datetime(df['Promised Date'])
                for _, row in df.iterrows():
                    job = Job(
                        job_number=str(row['Job Number']),
                        description=str(row.get('Description', '')),
                        order_date=row['Order Date'].to_pydatetime(),
                        promised_date=row['Promised Date'].to_pydatetime(),
                        quantity=int(row['Quantity']),
                        price_each=float(row['Price Each']),
                        customer=str(row.get('Customer', ''))
                    )
                    if not Job.query.filter_by(job_number=job.job_number).first():
                        db.session.add(job)
                db.session.commit()
        else:
            job = Job(
                job_number=request.form['job_number'],
                description=request.form['description'],
                order_date=datetime.strptime(request.form['order_date'], '%Y-%m-%dT%H:%M'),
                promised_date=datetime.strptime(request.form['promised_date'], '%Y-%m-%dT%H:%M'),
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
    job.job_number = request.form['job_number']
    job.description = request.form['description']
    job.order_date = datetime.strptime(request.form['order_date'], '%Y-%m-%dT%H:%M')
    job.promised_date = datetime.strptime(request.form['promised_date'], '%Y-%m-%dT%H:%M')
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
                        resources=str(row['Resources'])
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
                resources=request.form['resources']
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
    db.session.commit()
    return redirect(url_for('add_task'))

@app.route('/delete_task/<task_number>', methods=['POST'])
def delete_task(task_number):
    task = Task.query.filter_by(task_number=task_number).first_or_404()
    Schedule.query.filter_by(task_number=task_number).delete()
    db.session.delete(task)
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

    # Start timer
    start_time = timer.time()

    # Genetic Algorithm Setup
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

    # End timer
    end_time = timer.time()
    elapsed_time = end_time - start_time  # In seconds

    # Clear previous schedule
    db.session.query(Schedule).delete()
    db.session.commit()

    # Save new schedule with adjusted times
    segments = []
    for task in tasks:
        if task.task_number in adjusted_times:
            start, end = adjusted_times[task.task_number]
            res_used = ', '.join(res.name for res in task.selected_resources)
            schedule_entry = Schedule(
                task_number=task.task_number,
                start_time=start,
                end_time=end,
                resources_used=res_used
            )
            db.session.add(schedule_entry)
            segments.append({
                'task_id': task.task_number,
                'job_id': task.job_number,
                'description': task.description,
                'resources': res_used,
                'start': start,
                'end': end
            })

    job_data = []
    for job in jobs:
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
        if all(t.task_number in adjusted_times for t in job_tasks):
            expected_finish = max(adjusted_times[t.task_number][1] for t in job_tasks)
            days_late = max(0, (expected_finish - job.promised_date).total_seconds() / 86400) * job.price_each * job.quantity
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
    segments = [{
        'task_id': s.task_number,
        'job_id': Task.query.filter_by(task_number=s.task_number).first().job_number,
        'description': Task.query.filter_by(task_number=s.task_number).first().description,
        'resources': s.resources_used,
        'start': s.start_time,
        'end': s.end_time
    } for s in schedules]
    total_rand_days_late = 0
    jobs = Job.query.all()
    tasks = Task.query.all()
    task_times = {s.task_number: (s.start_time, s.end_time) for s in schedules}
    
    job_data = []
    for job in jobs:
        job_tasks = [t for t in tasks if t.job_number == job.job_number]
        if all(t.task_number in task_times for t in job_tasks):
            expected_finish = max(task_times[t.task_number][1] for t in job_tasks)
            days_late = max(0, (expected_finish - job.promised_date).total_seconds() / 86400) * job.price_each * job.quantity
            total_rand_days_late += days_late
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

if __name__ == '__main__':
    app.run(debug=True)