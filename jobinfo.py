#!/usr/bin/env python

import json
import os
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from optparse import OptionParser
from subprocess import getoutput

from hostlist import collect_hostlist


class Fields:
    def __init__(self):
        self.fields = OrderedDict()

    def add(self, fields):
        for field in fields.split(","):
            p = field.split(":")

            if len(p) > 1:
                tag = p[1].strip().lower()
            else:
                tag = p[0].strip().lower()

            self.fields[tag] = p[0]

    def get_header(self):
        return [self.fields[tag] for tag in self.fields]

    def get_items(self, job: dict) -> list:
        data = []
        for tag in self.fields:
            data.append(job[tag])

        return data


# Config section
slurm_base_path = "/usr/bin/"
sinfo = slurm_base_path + "sinfo"
squeue = slurm_base_path + "squeue"
scontrol = slurm_base_path + "scontrol"

feature_nodes = defaultdict(list)
for line in getoutput('sinfo -h -o "%n|%f"').split("\n"):
    node, features = line.strip().split("|")
    for feature in features.split(","):
        feature_nodes[feature].append(node)

block_reasons = [
    "QOSResourceLimit",
    "AssocGrpBillingRunMinutes",
    "AssocGrpCPURunMinutesLimit",
    "QOSMaxJobsPerUserLimit",
    "JobHeldAdmin",
]
# End config section

# Parse options
parser = OptionParser()
parser.add_option("-u", "--user", metavar="user", help="only list jobs for the specified user")
parser.add_option(
    "-A",
    "--account",
    metavar="account",
    help="only list jobs for the specified account / project",
)
parser.add_option(
    "-p",
    "--partition",
    metavar="partition",
    help="only list jobs in the specified partition / queue",
)
parser.add_option(
    "-f",
    "--feature",
    metavar="feature",
    help="Specify the feature to limit listing for",
)
parser.add_option(
    "-r",
    "--running",
    action="store_false",
    default=True,
    help="Only display running jobs",
)
parser.add_option(
    "-w",
    "--waiting",
    action="store_false",
    default=True,
    help="Only display waiting jobs",
)
parser.add_option(
    "-b",
    "--blocked",
    action="store_false",
    default=True,
    help="Only display blocked jobs",
)
parser.add_option("-s", "--summary", action="store_false", default=True, help="Display summary only")
(options, args) = parser.parse_args()

# Check options: If any option specified, turn off the ones not specified!
if not options.running or not options.waiting or not options.summary or not options.blocked:
    options.running = not options.running
    options.waiting = not options.waiting
    options.blocked = not options.blocked
    options.summary = not options.summary

# Check commands
for command in [sinfo, squeue, scontrol]:
    if not os.access(command, os.X_OK):
        print('Error accessing "' + command + '"!')
        sys.exit(1)

# Get clustername
slurm_conf = getoutput(scontrol + " show config 2>&1").split("\n")
clustername = ""
for line in slurm_conf:
    if line.split("=")[0].strip() == "ClusterName":
        clustername = line.split("=")[1].strip()
        break

if not clustername:
    print("Error: ClusterName not found in slurm config!")
    sys.exit(2)

# If specific feature was specified, build the corresponding node_spec
node_spec = ""
if options.feature:
    if options.feature not in feature_nodes:
        print('Feature "' + options.feature + '" not present or not possible to use as selection.')
        print("Allowed options are: " + ", ".join(sorted(feature_nodes.keys())))
        sys.exit(3)

    node_spec = " -w " + collect_hostlist(feature_nodes[options.feature]) + " "

print("CLUSTER: " + clustername)


def print_table(lines):
    if len(lines) < 1:
        return

    lens = []
    for item in lines[0]:
        lens.append(len(item))

    for line in lines[1:]:
        for col, item in enumerate(line):
            item = str(item)
            if len(item) > lens[col]:
                lens[col] = len(item)

    form = ""
    for col in lens[:-1]:
        form += "%" + str(col) + "s "
    form += "%s"  # "Left align" last column

    for line in lines:
        print(form % tuple(line))


def shorten_name(name):
    MAX_LEN = 45

    while len(name) > MAX_LEN:
        idx = name.find("/")
        if idx < 0:
            break
        name = name[idx + 1 :]

    if len(name) > MAX_LEN:
        name = "..." + name[-(MAX_LEN - 3) :]

    return name


def format_date_time(days: int, hours: int, minutes: int, seconds: int) -> str:
    if days > 0:
        return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"

    if hours > 0:
        return f"{hours:2}:{minutes:02}:{seconds:02}"

    if minutes > 0:
        return f"{minutes:2}:{seconds:02}"

    return f"{seconds:2}"


def format_timedelta(delta):
    hours = int(delta.seconds / 3600)
    minutes = int((delta.seconds - hours * 3600) / 60)
    seconds = delta.seconds - 3600 * hours - 60 * minutes

    return format_date_time(delta.days, hours, minutes, seconds)


def format_minutes(total_minutes):
    days = int(total_minutes / 60 / 24)
    hours = int((total_minutes - 60 * 24 * days) / 60)
    minutes = int(total_minutes - 60 * (24 * days + hours))
    seconds = 60 * (total_minutes - 60 * (24 * days + hours) - minutes)

    return format_date_time(days, hours, minutes, seconds)


def fix_fields(job):
    job["name"] = shorten_name(job["name"])
    job["cpus"] = job["cpus"]["number"]
    job["node_count"] = job["node_count"]["number"]
    if job["array_task_id"]["set"]:
        job["job_id"] = f"{job['array_job_id']['number']}_{job['array_task_id']['number']}"
    elif job["array_task_string"]:
        job["job_id"] = f"{job['array_job_id']['number']}_[{job['array_task_string']}]"
    job["start_time"] = datetime.fromtimestamp(job["start_time"])
    job["end_time"] = datetime.fromtimestamp(job["end_time"])
    if job["tres_per_node"].startswith("gres:"):
        job["tres_per_node"] = job["tres_per_node"][5:]


command = squeue + " -M " + clustername + node_spec + " --noheader --json"
squeue_output = json.loads(getoutput(command))

if "jobs" not in squeue_output:
    print("No jobs found, exiting")
    sys.exit(1)

hosts_used = []
if options.running or options.summary:  # Print running jobs
    fields = Fields()
    fields.add("JobId:job_id,Partition,Job Name:name,User Name:user_name,Account")
    fields.add("Start Time:start_time,Wall Time Left:time_left,# Nodes:node_count,# Cores:cpus")
    fields.add("TRES per node:tres_per_node,Nodes")

    running_node_count = 0
    running_job_count = 0
    running_node_count = 0

    jobs = sorted(
        [job for job in squeue_output["jobs"] if job["job_state"] in ["RUNNING", "COMPLETING"]],
        key=lambda job: job["end_time"],
    )

    completing_lines = []
    lines = []
    heading = fields.get_header()

    if options.running:
        lines.append(heading)
        completing_lines.append(heading)

    for job in jobs:
        if options.user and not job["user_name"] == options.user:
            continue
        if options.account and not job["account"] == options.account.lower():
            continue
        if options.partition and not job["partition"] == options.partition:
            continue

        fix_fields(job)

        job["time_left"] = format_timedelta(job["end_time"] - datetime.today())

        if options.running:
            if job["job_state"] == "RUNNING":
                lines.append(fields.get_items(job))
            else:
                completing_lines.append(fields.get_items(job))

        running_job_count += 1
        if job["nodes"] not in hosts_used:
            running_node_count += job["node_count"]
            hosts_used.append(job["nodes"])

    if len(completing_lines) > 1 and options.running:
        print("Completing jobs:")
        print_table(completing_lines)
        print()

    if len(lines) > 1 and options.running:
        print("Running jobs:")
        print_table(lines)

if options.waiting or options.blocked or options.summary:  # Print queued jobs
    fields = Fields()
    fields.add("JobId:job_id,Partition,Job Name:name,User Name:user_name,Account")
    fields.add("Prel. Start Time:start_time,Wall Time Limit:time_limit,Priority")
    fields.add("# Nodes:node_count,# Cores:cpus,Reason:state_reason,TRES per node:tres_per_node,Dependency")

    jobs = sorted(
        [job for job in squeue_output["jobs"] if job["job_state"] in ["PENDING"]],
        key=lambda job: job["priority"]["number"],
        reverse=True,
    )

    heading = fields.get_header()

    waiting_job_count = 0
    waiting_node_count = 0
    found_waiting_job = False
    bonus_job_count = 0
    bonus_node_count = 0
    found_bonus_job = [False, False, False]
    min_bonus_level = 1
    blocked_jobs = []
    blocked_jobs_count = 0

    lines = []
    for job in jobs:
        if options.user and not job["user_name"] == options.user:
            continue
        if options.account and not job["account"] == options.account.lower():
            continue
        if options.partition and not job["partition"] == options.partition:
            continue

        fix_fields(job)
        if job["start_time"] < datetime.today():
            job["start_time"] = "N/A"
        job["time_limit"] = format_minutes(job["time_limit"]["number"])
        job["priority"] = job["priority"]["number"]
        nodes = job["node_count"]

        # Find array jobs and add upp no. of nodes wanted
        if job["array_task_string"] != "":
            if job["array_task_string"].find(",") > 0:
                nodes *= job["array_task_string"].count(",") + 1
            elif job["array_task_string"].find(":") > 0:
                tmp = job["array_task_string"].split(":")
                jobs = [int(i) for i in tmp[0].split("-")]
                if len(jobs) > 1:
                    jobs = jobs[1] - jobs[0] + 1
                nodes *= jobs / int(tmp[1].split("%")[0])  # Divide by stepping (:), ignore max concurrent jobs (%)
            else:
                jobs = [int(i) for i in job["array_task_string"].split("%")[0].split("-")]

                if len(jobs) > 1:
                    nodes *= jobs[1] - jobs[0] + 1

            if job["array_task_string"].find("%") > 0:  # Handle limit on max concurrent jobs in array
                nodes = min(nodes, int(job["array_task_string"].split("%")[1].strip("]")))

        # Sort out "blocked" jobs
        if job["state_reason"] in block_reasons:
            if not blocked_jobs_count:
                blocked_jobs.append(heading)
            blocked_jobs.append(fields.get_items(job))
            blocked_jobs_count += 1
            continue

        try:
            try:
                bonus_level = int(job["qos"][-1:])
            except:
                bonus_level = 0
            if job["qos"][-6:-1] == "bonus":
                bonus_job_count += 1
                bonus_node_count += nodes
                if not found_bonus_job[bonus_level - 1]:
                    found_bonus_job[bonus_level - 1] = True
                    if options.waiting:
                        print_table(lines)
                        lines = []
                        print("\nWaiting bonus level " + str(bonus_level) + " jobs:")
                        lines.append(heading)
            else:
                waiting_job_count += 1
                waiting_node_count += nodes

        except ValueError:
            pass

        if not options.waiting:
            continue
        if not found_waiting_job and waiting_job_count > 0:
            print("\nWaiting jobs:")
            lines.append(heading)
            found_waiting_job = True

        lines.append(fields.get_items(job))

    if lines:
        print_table(lines)

    if options.blocked and blocked_jobs_count > 0:
        print("\nBlocked jobs:")
        print_table(blocked_jobs)

if not options.summary:
    sys.exit(0)

# Print job summary
tmp = "\nSummary: %d running jobs using %d nodes" % (
    running_job_count,
    running_node_count,
)
if waiting_job_count > 0:
    tmp += ", %d waiting normal jobs wanting <= %d nodes" % (
        waiting_job_count,
        waiting_node_count,
    )
if bonus_job_count > 0:
    tmp += ", %d waiting bonus jobs wanting <= %d nodes" % (
        bonus_job_count,
        bonus_node_count,
    )
if blocked_jobs_count > 0:
    tmp += ", %d blocked jobs" % blocked_jobs_count
print(tmp)

if options.feature:
    sys.exit(0)

# Print node usage summary
print("\nTotal node usage:")
print("%-15s %10s %10s %10s %10s" % ("PARTITION", "ALLOCATED", "IDLE", "OFFLINE", "TOTAL"))
sinfo_output = getoutput(sinfo + " -M" + clustername + " -s --noheader").split("\n")
for line in sinfo_output:
    i = line.split()
    i[0] = i[0].strip("*")
    if options.partition:
        if not i[0] == options.partition:
            continue
    j = i[3].split("/")
    print("%-15s %10s %10s %10s %10s" % (i[0], j[0], j[1], j[2], j[3]))

# Print node type usage
print("\nNode type usage on main partition:")
print("%-20s %10s %10s %10s %10s" % ("TYPE", "ALLOCATED", "IDLE", "OFFLINE", "TOTAL"))
sinfo_output = getoutput(sinfo + " -M" + clustername + " -p" + "main" + ' -s --noheader -o "%f|%F"').split("\n")
for line in sorted(sinfo_output):
    features, count = line.split("|")
    if features == "(null)":
        continue
    features = [f for f in features.split(",") if f != "NOGPU" and f != "25"]
    j = count.split("/")
    print("%-20s %10s %10s %10s %10s" % (",".join(features), j[0], j[1], j[2], j[3]))

print("\nTotal GPU usage:")
sinfo_output = getoutput(
    sinfo + " -M" + clustername + " --noheader -OPartition,Nodes,StateLong,GRES:100,gresused:100"
).split("\n")

gpus = dict()
gpu_nodes = defaultdict(dict)
for line in sinfo_output:
    p, count, state, gres, used = line.split()
    p = p.strip("* \t")
    state = state.strip("$* \t")
    Gtype = ""
    count = int(count)
    avail = 0
    for part in gres.split(","):
        pp = part.split(":")
        if len(pp) < 3:
            continue
        if pp[0] == "gpu":
            avail = int(pp[2].split("(")[0])
            Gtype = pp[1]
            break
    for part in used.split(","):
        pp = part.split(":")
        if len(pp) < 3:
            continue
        if pp[0] == "gpu":
            use = int(pp[2].split("(")[0])
            break
    if not Gtype:
        continue

    free = avail - use

    if Gtype not in gpus:
        gpus[Gtype] = {"avail": 0, "used": 0, "idle": 0, "offline": 0}

    gpus[Gtype]["avail"] += avail * count

    if (
        state.startswith("idle")
        or state.startswith("mixed")
        or state.startswith("allocated")
        or state.startswith("draining")
        or state.startswith("completed")
        or state.startswith("completing")
    ):
        gpus[Gtype]["used"] += use * count
        gpus[Gtype]["idle"] += free * count

    elif (
        state.startswith("drained")
        or state.startswith("reserved")
        or state.startswith("maint")
        or state.startswith("down")
        or state.startswith("invalid")
    ):
        gpus[Gtype]["offline"] += avail * count

    if not free:
        continue

    Gavail = Gtype + ":" + str(free)
    if Gavail not in gpu_nodes[p]:
        gpu_nodes[p][Gavail] = [count, free, p]
    else:
        gpu_nodes[p][Gavail][0] += count

if not gpus:
    print("None!")
else:
    print("TYPE           ALLOCATED IDLE OFFLINE TOTAL")
    for gpu in gpus:
        d = gpus[gpu]
        print("{:14}   {:7} {:4} {:7} {:5}".format(gpu, d["used"], d["idle"], d["offline"], d["avail"]))

if gpu_nodes:
    print("\nFree nodes per number of GPU:s:")
    print("PARTITION  # NODES  GPU:s")
    for p in sorted(gpu_nodes):
        for gavail in sorted(gpu_nodes[p]):
            print("{:9}  {:7}  {:<7}".format(p, gpu_nodes[p][gavail][0], gavail))
