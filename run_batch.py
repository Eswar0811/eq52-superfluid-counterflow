"""usage: run_batch.py <parallel> <config:seed> ...  -- runs src/inverse.py jobs, N at a time; skips finished ones."""
import os, subprocess, sys, time
N = int(sys.argv[1]); jobs = [j.split(":") for a in sys.argv[2:] for j in a.split()]
root = os.path.dirname(os.path.abspath(__file__)); running = []
env = {**os.environ, "PINN_THREADS": "2"}
while jobs or running:
    running = [p for p in running if p.poll() is None]
    while jobs and len(running) < N:
        c, s = jobs.pop(0)
        if os.path.exists(f"{root}/results/inv_{c}{'_nist' if os.environ.get('HE2_PROPS') == 'nist' else ''}_seed{s}.json"):
            continue
        running.append(subprocess.Popen([sys.executable, "inverse.py", c, s], cwd=f"{root}/src", env=env,
                                        stdout=open(f"{root}/logs/inv_{c}{'_nist' if os.environ.get('HE2_PROPS') == 'nist' else ''}_s{s}.log", "w"), stderr=subprocess.STDOUT))
    time.sleep(2)
