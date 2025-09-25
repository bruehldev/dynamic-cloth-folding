# tools/jsonl_tap.py
import json, os, time

class StepTap:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def log_step(self, *, backend, episode, t, action, reward, done, info):
        row = {
            "ts": time.time(),
            "backend": backend,
            "ep": int(episode),
            "t": int(t),
            "a": [float(x) for x in action],
            "r": float(reward),
            "d": bool(done),
            # keep these small & comparable:
            "ctrl_error": float(info.get("ctrl_error", 0.0)),
            "dsum": float(info.get("dsum", 0.0)),
            "success": bool(info.get("is_success", False)),
        }
        self.f.write(json.dumps(row) + "\n")

    def close(self):
        try: self.f.flush(); self.f.close()
        except Exception: pass
