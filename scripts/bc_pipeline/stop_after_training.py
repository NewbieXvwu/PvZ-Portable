import os, signal, time
result='/tmp/pvz_bc_2b/training_results.json'
pid=49458
while not os.path.exists(result):
    time.sleep(0.1)
try:
    os.kill(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
