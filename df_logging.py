# df_logging.py
import os, json, time, hashlib, threading
from datetime import datetime
from typing import Any, Dict, List, Optional

def _now(): return time.time()
def _sha1_bytes(b: bytes) -> str:
    import hashlib; h = hashlib.sha1(); h.update(b); return h.hexdigest()

class RunLogger:
    def __init__(self, root="logs", project="dynamic-cloth-folding"):
        self.root = root; self.project = project
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()
        self._open = False

    def start_episode(self, meta: Dict[str, Any]):
        with self._lock:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            self.episode_id = f"{self.project}_{ts}"
            self.ep_dir = os.path.join(self.root, self.episode_id)
            os.makedirs(self.ep_dir, exist_ok=True)
            self.img_dir = os.path.join(self.ep_dir, "images")
            os.makedirs(self.img_dir, exist_ok=True)
            self.path = os.path.join(self.ep_dir, "episode.jsonl")
            self.f = open(self.path, "a", encoding="utf-8", buffering=1)
            self._open = True
            self._write({"type":"episode_start","t":_now(),"episode_id":self.episode_id,"meta":meta})

    def end_episode(self, summary: Dict[str, Any]):
        with self._lock:
            if not self._open: return
            self._write({"type":"episode_end","t":_now(),"episode_id":self.episode_id,"summary":summary})
            self.f.close(); self._open=False

    def save_image_gray(self, img_np):
        import imageio.v2 as iio
        assert img_np.ndim==2 and img_np.dtype=="uint8"
        name = f"img_{int(_now()*1e6)}.png"
        path = os.path.join(self.img_dir, name)
        iio.imwrite(path, img_np)
        with open(path,"rb") as fh:
            h = _sha1_bytes(fh.read())
        return {"path": os.path.relpath(path, self.root), "sha1": h, "shape": list(img_np.shape)}

    def log_policy_step(self, policy_step: int, data: Dict[str, Any], controller_block: Optional[Dict[str, Any]] = None) -> None:
        rec = {"type":"policy_step","t":_now(),"episode_id":self.episode_id,
               "policy_step":policy_step, **data}
        if controller_block is not None: rec["controller"]=controller_block
        self._write(rec)

    def build_controller_block(self, policy_step: int, substeps: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"policy_step":policy_step, "n":len(substeps), "sub":substeps}

    def _write(self, obj: Dict[str, Any]):
        self.f.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
