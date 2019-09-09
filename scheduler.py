import os
import subprocess
from time import sleep

class Launcher(object):

    def __init__(self, num_procs=30, capture_stdout=False):

        self.num_procs = num_procs
        self.proc_list = {}
        self.capture_stdout = capture_stdout

    def add(self, name, cmd, folder='./'):

        self.proc_list[name] = {'cmd': cmd, 'folder': folder, 'pid': None}

    def run_script(self, script_path):

        executable = "bash"
        subprocess.call([executable, script_path])

    def run(self, name):

        stdout_file = os.path.join(self.proc_list[name]['folder'], name + '.out')
        stderr_file = os.path.join(self.proc_list[name]['folder'], name + '.err')
        script_file = os.path.join(self.proc_list[name]['folder'], name + '.sh')

        with open(stdout_file, 'w') as f_stdout, open(stderr_file, 'w') as f_stderr, open(script_file, 'w') as f_cmd:

            executable = "bash"

            f_cmd.write(self.proc_list[name]['cmd'])

            self.__wait_slot()
            self.proc_list[name]['pid'] = subprocess.Popen([executable, script_file], stdout=f_stdout, stderr=f_stderr)

    def n_running(self):

        return sum([self.proc_list[name]['pid'].poll() is None for name in self.proc_list.keys() if self.proc_list[name]['pid'] is not None])

    def wait(self):

        while self.n_running() > 0: sleep(5.)

    def __wait_slot(self):

        # print("(launcher) currently %d procs running" % self.n_running())

        while self.n_running() >= self.num_procs: sleep(5.)



def check_file_repeat(filename,n_repeats=5,wait_seconds=5):

    i = 0
    f = None
    while i < n_repeats:
        try:
            f = open(filename)
        except:
            i += 1
            print "Failed attempt at reading {}".format(filename)
            sleep(wait_seconds)
            print "Retrying..."
            continue
        break
    assert f, "Cannot open qsub output file: {}".format(filename)

