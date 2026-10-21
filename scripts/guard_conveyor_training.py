"""Yield task20's training to another trainer without changing any frozen study source."""
import json
import os
import signal
import time
from pathlib import Path

from run_conveyor_study import OUT, descendants, guard, read, trainers


def stopped(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] in ('T', 't')
    except FileNotFoundError:
        return True


if __name__ == '__main__':
    controller = read(OUT/'study_controller.process.json')['pid']
    paused = set()
    marker = OUT/'training_gpu_yield.json'
    with (OUT/'training_gpu_yield.events.jsonl').open('a') as events:
        while guard.live(controller):
            process = OUT/'t20_training.process.json'
            own = set(descendants(controller))
            training = process.exists() and guard.live(read(process)['pid'])
            foreign = sorted(set(trainers())-own) if training else []
            if training and foreign:
                # Stop the controller first, so its memory guard cannot undo this GPU pause.
                # Preserve already-stopped descendants: only resume processes stopped here.
                candidates = [controller]+[p for p in descendants(controller) if p != controller]
                changed = False
                for pid in candidates:
                    if pid not in paused and not stopped(pid):
                        try:
                            os.kill(pid, signal.SIGSTOP); paused.add(pid); changed = True
                        except ProcessLookupError:
                            pass
                if changed:
                    event = dict(time=time.time(), paused=True, foreign_trainers=foreign, own_pids=sorted(paused))
                    marker.write_text(json.dumps(event)+'\n')
                    events.write(json.dumps(event)+'\n'); events.flush()
            elif paused and not foreign and guard.available() >= 3*1024*1024:
                for pid in sorted(paused-{controller}):
                    try: os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError: pass
                if controller in paused:
                    try: os.kill(controller, signal.SIGCONT)
                    except ProcessLookupError: pass
                paused.clear()
                event = dict(time=time.time(), paused=False, foreign_trainers=[])
                marker.write_text(json.dumps(event)+'\n')
                events.write(json.dumps(event)+'\n'); events.flush()
            time.sleep(1)
