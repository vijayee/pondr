"""Duration/concurrency repro for the WaveDB rewind bug.

H3a: pilot write profile (4 kinds + graph triples) over ~25 min, one handle,
     no concurrent opens. If A alone loses data -> timer/WAL bug.
H3b: same, PLUS one concurrent second-handle open+close at ~12 min
     (emulating the diagnostic read). Loss only in B -> two-handle interplay.
"""

import shutil
import subprocess
import sys
import time

ROOT = r"C:\Users\victor morrow\Git-projects\Pondr"

TEMPLATE = r'''import sys, time; sys.path.insert(0, "__ROOT__")
from icm.ledger import ICMLedger
from icm.instrument import ICMSession, ExpectationDecl, OutcomeDecl
import wavedb; wavedb.set_quiet(True)
led = ICMLedger(r"__DB__")
s0 = time.time()
for si in range(60):
    sid = "hb-" + str(si).zfill(3)
    sess = ICMSession(led, sid)
    sess.begin(meta={"pilot": "__VARIANT__"})
    sess.declare_intent("aim " + str(si), intent_id="hint-" + str(si).zfill(3),
                        slots={"subject": "SELF", "attitude": "intend",
                               "relation": "do " + str(si), "object": "out",
                               "residue": None, "propositions": 1})
    sess.step()
    e1 = sess.expect(ExpectationDecl(intent_id="hint-" + str(si).zfill(3),
                                     statement="p " + str(si),
                                     branches={"desired": "d"},
                                     horizon={"kind": "manual"}),
                     exp_id="hexp-" + str(si).zfill(3) + "-e1")
    sess.step()
    e2 = sess.expect(ExpectationDecl(intent_id="hint-" + str(si).zfill(3),
                                     statement="q " + str(si),
                                     branches={"desired": "d"},
                                     horizon={"kind": "manual"}),
                     exp_id="hexp-" + str(si).zfill(3) + "-e2")
    sess.step()
    sess.resolve(OutcomeDecl(e1, "o1 " + str(si), "desired"),
                 outcome_id="hout-" + str(si).zfill(3) + "-1")
    sess.resolve(OutcomeDecl(e2, "o2 " + str(si), "desired"),
                 outcome_id="hout-" + str(si).zfill(3) + "-2")
    sess.finish()
    if si % 15 == 0:
        pres = sum(1 for r in led.ids("intent") if led._get_raw(led.rec_key("intent", r)) is not None)
        print("__VARIANT__ [%02d/60] intents present=%d/60 elapsed=%.0fs" % (si, pres, time.time() - s0), flush=True)
    time.sleep(20)
led.close()
led2 = ICMLedger(r"__DB__")
pres = sum(1 for r in led2.ids("intent") if led2._get_raw(led2.rec_key("intent", r)) is not None)
print("__VARIANT__ RESULT: listed=%d present=%d" % (len(led2.ids("intent")), pres), flush=True)
led2.close()
'''


def writer(db: str, variant: str) -> str:
    return TEMPLATE.replace("__ROOT__", ROOT.replace(chr(92), "/")).replace("__DB__", db.replace(chr(92), "/")).replace("__VARIANT__", variant)


READER = r'''import sys; sys.path.insert(0, "__ROOT__")
from icm.ledger import ICMLedger; import wavedb; wavedb.set_quiet(True)
led = ICMLedger(r"__DB__")
ids = led.ids("intent")
pres = sum(1 for r in ids if led._get_raw(led.rec_key("intent", r)) is not None)
print("B-READ: listed=%d present=%d" % (len(ids), pres), flush=True)
led.close()
'''


def main() -> None:
    db_a = ROOT + r"\data\_scratch\icm\hammer3a"
    db_b = ROOT + r"\data\_scratch\icm\hammer3b"
    shutil.rmtree(db_a, ignore_errors=True)
    shutil.rmtree(db_b, ignore_errors=True)
    pa = subprocess.Popen([sys.executable, "-c", writer(db_a, "a")],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pb = subprocess.Popen([sys.executable, "-c", writer(db_b, "b")],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(12 * 60)  # 36 sessions in (~20s per session), ~12 min
    reader = READER.replace("__ROOT__", ROOT.replace(chr(92), "/")).replace("__DB__", db_b.replace(chr(92), "/"))
    pr = subprocess.Popen([sys.executable, "-c", reader],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_r = pr.communicate(timeout=120)[0].decode(errors="replace")
    out_a = pa.communicate(timeout=2400)[0].decode(errors="replace")
    out_b = pb.communicate(timeout=2400)[0].decode(errors="replace")
    print("=== B diagnostic mid-run read (at ~12 min) ===")
    print(out_r[-200:])
    print("=== A (no concurrency, 20 min total) ===")
    print(out_a[-800:])
    print("=== B (with concurrent read at 12 min, 20+ min total) ===")
    print(out_b[-800:])


if __name__ == "__main__":
    main()