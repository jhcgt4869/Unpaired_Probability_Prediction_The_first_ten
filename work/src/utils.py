import sys
import time

def out(logfile, x="\n", end="\n"):
    print(x, end=end)
    sys.stdout.flush()
    logfile.write(str(x) + end)
    logfile.flush()

def format_elapsed(start_time):
    elapsed_time     = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes   = divmod(minutes, 60)
    days, hours      = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string
