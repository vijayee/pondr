"""Two-handle concurrency repro for the WaveDB data-rewind bug.

Writer A: opens the DB, writes 30 records, idles 10s (window), writes 30 more,
closes, reopens, checks presence of all 60.
Reader B: opens the SAME directory CONCURRENTLY during A's idle window, reads,
closes. (This is what the mid-run diagnostic did to the live pilot.)

Expected if safe: A's final check = 60/60.
Observed suspect if B's close-time snapshot rewrites the section index from
B's recovered trie subset.
"""

import shutil
import subprocess
import sys
import time

ROOT = r"C:\Users\victor morrow\Git-projects\Pondr"
DB = ROOT + r"\data\_scratch\icm\conc_repro"

A = f"""import sys, time; sys.path.insert(0, r"{ROOT}")
from icm.ledger import ICMLedger; import wavedb; wavedb.set_quiet(True)
led = ICMLedger(r"{DB}")
for i in range(30):
    led.put_session("conc-" + str(i).zfill(3))
print("A: first 30 written", flush=True)
time.sleep(10)
for i in range(30, 60):
    led.put_session("conc-" + str(i).zfill(3))
print("A: last 30 written", flush=True)
led.close()
led2 = ICMLedger(r"{DB}")
pres = sum(1 for r in led2.ids("session") if led2._get_raw(led2.rec_key("session", r)) is not None)
print("A RESULT: listed=%d present=%d" % (len(led2.ids("session")), pres), flush=True)
led2.close()
"""

B = f"""import sys, time; sys.path.insert(0, r"{ROOT}")
from icm.ledger import ICMLedger; import wavedb; wavedb.set_quiet(True)
led = ICMLedger(r"{DB}")
ids = led.ids("session")
pres = sum(1 for r in ids if led._get_raw(led.rec_key("session", r)) is not None)
print("B: concurrent read listed=%d present=%d" % (len(ids), pres), flush=True)
time.sleep(2)
led.close()
print("B closed", flush=True)
"""


def main() -> None:
    shutil.rmtree(DB, ignore_errors=True)
    pa = subprocess.Popen([sys.executable, "-c", A], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(6)
    pb = subprocess.Popen([sys.executable, "-c", B], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_a = pa.communicate(timeout=180)[0].decode(errors="replace")
    out_b = pb.communicate(timeout=180)[0].decode(errors="replace")
    print("=== A ===")
    print(out_a[-500:])
    print("=== B ===")
    print(out_b[-300:])


if __name__ == "__main__":
    main()