# G:\RAEZ-Next\hand_bone_tracker\gui\server.py
import sys
import os
from pathlib import Path

# === CRITICAL: Fix PYTHONPATH so 'models', 'training', etc. are importable ===
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))  # Also set cwd to project root

# Force UTF-8 output on Windows to avoid emoji encoding errors
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import signal
import asyncio
import csv
import collections
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import time
from models.hand_tracker import HandBoneTracker


# ── Lifespan (modern FastAPI, replaces deprecated @app.on_event) ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("[RAEZ BACKEND] Server started at http://localhost:8000")
    print(f"[RAEZ BACKEND] Project root: {PROJECT_ROOT}")
    print("="*60 + "\n")
    yield
    print("\n[RAEZ BACKEND] Server shutting down.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anti-cache Middleware to prevent browser disk/memory cache growth ─────────
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ── Global state ───────────────────────────────────────────────────────────────
training_process: subprocess.Popen | None = None
logs_queue = collections.deque(maxlen=2000)
active_websockets = set()
log_reader_task: asyncio.Task | None = None

async def consume_logs():
    global training_process, logs_queue, active_websockets
    proc = training_process
    if not proc:
        return
    print("[RAEZ] Background log reader started.")
    try:
        while proc.poll() is None:
            line = await asyncio.to_thread(proc.stdout.readline)
            if line:
                line = line.strip()
                logs_queue.append(line)
                print(f"[LOG IA] {line}")
                for ws in list(active_websockets):
                    try:
                        await ws.send_text(line)
                    except Exception:
                        active_websockets.discard(ws)
            else:
                await asyncio.sleep(0.01)
        # Flush remaining
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:
                break
            line = line.strip()
            logs_queue.append(line)
            print(f"[LOG IA] {line}")
            for ws in list(active_websockets):
                try:
                    await ws.send_text(line)
                except Exception:
                    active_websockets.discard(ws)
        # Notify
        for ws in list(active_websockets):
            try:
                await ws.send_text("--- Process Finished ---")
            except Exception:
                active_websockets.discard(ws)
    except Exception as e:
        print(f"[RAEZ] Error in background log reader: {e}")
    finally:
        print("[RAEZ] Background log reader stopped.")


# ── /status ───────────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    global training_process
    is_running = bool(training_process and training_process.poll() is None)
    pid = training_process.pid if is_running else None
    print(f"[RAEZ] Status: {'RUNNING (PID ' + str(pid) + ')' if is_running else 'IDLE'}")
    return {
        "is_running": is_running,
        "device": "RTX 4070 (CUDA)",
        "project": "Hand & Bone Tracker",
        "pid": pid,
    }


# ── /start-train ──────────────────────────────────────────────────────────────
@app.post("/start-train")
async def start_train(stage: int = 1, resume: bool = True):
    global training_process, logs_queue, log_reader_task
    if training_process and training_process.poll() is None:
        print("[RAEZ] Training already running, ignoring start request.")
        return {"error": "Training already running"}

    # Cancel previous log reader task if running
    if log_reader_task and not log_reader_task.done():
        log_reader_task.cancel()
        try:
            await log_reader_task
        except asyncio.CancelledError:
            pass

    logs_queue.clear()

    # Offload the inference model to CPU and clear CUDA cache to free up VRAM for training!
    global _model_instance
    if _model_instance is not None:
        print("[RAEZ] Offloading backend model to CPU to free GPU memory for training...")
        try:
            _model_instance = _model_instance.cpu()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[RAEZ] Failed to offload model: {e}")

    if stage == 1:
        config_name = "stage1_pretrain.yaml"
    elif stage == 2:
        config_name = "stage2_main.yaml"
    elif stage == 3:
        config_name = "stage3_finetune.yaml"
    else:
        config_name = f"stage{stage}_pretrain.yaml"
        
    config_path = f"configs/training/{config_name}"
    python_exe = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    cmd = [python_exe, "-u", "scripts/train.py", "--config", config_path]
    if resume:
        cmd.append("--resume")

    print(f"[RAEZ] Starting stage {stage} (resume={resume}): {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    training_process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    print(f"[RAEZ] Process started with PID: {training_process.pid}")
    
    # Start background log reader
    log_reader_task = asyncio.create_task(consume_logs())
    
    return {"message": f"Training stage {stage} started", "pid": training_process.pid}


# ── /stop-train ───────────────────────────────────────────────────────────────
@app.post("/stop-train")
def stop_train():
    global training_process
    if not training_process or training_process.poll() is not None:
        print("[RAEZ] Stop requested but no process running.")
        return {"error": "No training process running"}

    pid = training_process.pid
    print(f"[RAEZ] Stopping PID {pid}...")
    try:
        os.kill(pid, signal.CTRL_BREAK_EVENT)
        print(f"[RAEZ] CTRL_BREAK sent to process group {pid}")
    except Exception as e:
        print(f"[RAEZ] Error stopping process: {e}")
        training_process.kill()

    training_process = None
    return {"message": "Training stopped"}


# ── /logs (WebSocket) ─────────────────────────────────────────────────────────
@app.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    print("[RAEZ] WebSocket client connected to log stream")
    active_websockets.add(websocket)
    try:
        # Replay past logs
        for line in list(logs_queue):
            await websocket.send_text(line)
            
        # Keep connection open until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("[RAEZ] WebSocket client disconnected (normal)")
    except Exception as e:
        print(f"[RAEZ] WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)
        print("[RAEZ] WebSocket client removed from active listeners")


# ── /metrics ──────────────────────────────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return {"error": "No logs folder found"}

    csv_files = list(logs_dir.glob("**/metrics.csv"))
    if not csv_files:
        return {"error": "No metrics.csv found yet. Launch training first."}

    # Find the latest modified CSV to identify the active stage
    latest_csv = max(csv_files, key=os.path.getmtime)
    stage_dir = latest_csv.parent.parent
    
    # Find all metrics.csv belonging to the same stage directory
    stage_csvs = list(stage_dir.glob("**/metrics.csv"))
    
    def get_version_num(path):
        try:
            return int(path.parent.name.split("_")[-1])
        except Exception:
            return 0

    # Parse and organize metrics by version folder
    version_data = []
    for csv_path in stage_csvs:
        v_num = get_version_num(csv_path)
        steps_dict = {}
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step_str = row.get("step")
                    if not step_str:
                        continue
                    try:
                        step = int(float(step_str))
                    except ValueError:
                        continue
                    
                    clean = {}
                    for k, v in row.items():
                        try:
                            clean[k] = float(v) if (v is not None and v.strip() != "") else None
                        except (ValueError, AttributeError):
                            clean[k] = v
                    
                    if step not in steps_dict:
                        steps_dict[step] = {}
                    for k, v in clean.items():
                        if v is not None:
                            steps_dict[step][k] = v
            if steps_dict:
                version_data.append((v_num, steps_dict))
        except Exception as e:
            print(f"[RAEZ] Error reading {csv_path}: {e}")

    if not version_data:
        return []

    # Sort versions chronologically
    version_data.sort(key=lambda x: x[0])

    # Merge chronologically: discard older data that was overwritten/rolled back by a resume checkpoint
    merged_data = {}
    for v_num, steps_dict in version_data:
        min_step = min(steps_dict.keys())
        
        # Discard any steps >= min_step from already merged data
        keys_to_delete = [k for k in merged_data.keys() if k >= min_step]
        for k in keys_to_delete:
            del merged_data[k]
            
        # Merge the new run's steps
        for step, row in steps_dict.items():
            if step not in merged_data:
                merged_data[step] = {}
            merged_data[step].update(row)

    # Convert back to sorted list of rows
    sorted_steps = sorted(merged_data.keys())
    data = []
    for step in sorted_steps:
        # Reconstruct the row dict including the step key
        row = {"step": step}
        row.update(merged_data[step])
        data.append(row)

    print(f"[RAEZ] Returned {len(data)} merged metrics rows for {stage_dir.name} (merged {len(stage_csvs)} runs)")
    return data


# ── /checkpoints ──────────────────────────────────────────────────────────────
@app.get("/checkpoints")
def list_checkpoints():
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    ckpts = sorted(ckpt_dir.glob("**/*.ckpt"), key=os.path.getmtime, reverse=True)
    print(f"[RAEZ] Found {len(ckpts)} checkpoint(s).")
    return [{"name": c.name, "path": str(c), "size": c.stat().st_size} for c in ckpts]


# ── Model loading and checkpoint helpers (lazy imports) ─────────────────────
_model_instance = None

def get_latest_checkpoint_path():
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("**/*.ckpt"), key=os.path.getmtime, reverse=True)
    return str(ckpts[0]) if ckpts else None

def load_checkpoint_into_model(checkpoint_path: str):
    global _model_instance
    try:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RAEZ] Loading checkpoint: {checkpoint_path} on {device}...")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get('hyper_parameters', {}).get('config', {})
        backbone_name = config.get('backbone', 'efficientnet_b0')
        print(f"[RAEZ] Detected backbone from checkpoint: {backbone_name}")
        
        model = HandBoneTracker(backbone_name=backbone_name, pretrained=False)
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    clean_state_dict[k[6:]] = v
                else:
                    clean_state_dict[k] = v
            model.load_state_dict(clean_state_dict)
            print("[RAEZ] State dict loaded successfully from 'state_dict'.")
        else:
            model.load_state_dict(checkpoint)
            print("[RAEZ] State dict loaded directly.")
            
        model = model.to(device)
        model.eval()
        _model_instance = model
        print(f"[RAEZ] Model loaded on {device} and ready for inference.")
        return True
    except Exception as e:
        import traceback
        print(f"[RAEZ] Error loading checkpoint: {traceback.format_exc()}")
        return False

# ── /load-checkpoint ──────────────────────────────────────────────────────────
@app.post("/load-checkpoint")
def load_checkpoint(checkpoint_data: dict):
    path = checkpoint_data.get("path")
    if not path or not os.path.exists(path):
        return {"error": f"Checkpoint path '{path}' not found."}
    
    success = load_checkpoint_into_model(path)
    if success:
        return {"message": f"Successfully loaded checkpoint: {os.path.basename(path)}"}
    else:
        return {"error": "Failed to load checkpoint. Check backend console logs."}

# ── /save-custom-label ────────────────────────────────────────────────────────
@app.post("/save-custom-label")
async def save_custom_label(data: dict):
    print("[RAEZ] Saving custom labeled data...")
    try:
        import base64
        from io import BytesIO
        from PIL import Image
        import time
        import json
        
        img_b64 = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        pose_name = data.get("pose_name", "unknown")
        keypoints_2d = data.get("keypoints_2d", [])
        keypoints_3d = data.get("keypoints_3d", [])
        
        # Paths
        custom_dir = PROJECT_ROOT / "data" / "raw" / "custom"
        images_dir = custom_dir / "images"
        labels_dir = custom_dir / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        filename = f"{pose_name}_{timestamp}"
        
        img_path = images_dir / f"{filename}.png"
        img.save(img_path)
        
        meta = {
            "pose_name": pose_name,
            "timestamp": timestamp,
            "image_file": f"images/{filename}.png",
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "handedness": data.get("handedness", "unknown")
        }
        
        meta_path = labels_dir / f"{filename}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            
        print(f"[RAEZ] Custom data saved: {img_path.name}")
        return {"message": "Success", "filename": filename}
    except Exception as e:
        import traceback
        print(f"[RAEZ] Save custom label error: {traceback.format_exc()}")
        return {"error": str(e)}

# ── /custom-dataset-counts ───────────────────────────────────────────────────
@app.get("/custom-dataset-counts")
async def get_custom_dataset_counts():
    try:
        custom_dir = PROJECT_ROOT / "data" / "raw" / "custom"
        labels_dir = custom_dir / "labels"
        counts = {
            "middle_finger": 0,
            "fist": 0,
            "open_hand": 0,
            "thumbs_up": 0,
            "peace": 0,
            "ok_sign": 0
        }
        if labels_dir.exists():
            for f in labels_dir.glob("*.json"):
                name = f.stem
                for pose_id in counts.keys():
                    if name.startswith(pose_id):
                        counts[pose_id] += 1
                        break
        return counts
    except Exception as e:
        return {"error": str(e)}

_onnx_sessions = {}

def get_onnx_session(mode: str):
    global _onnx_sessions
    if mode not in _onnx_sessions:
        model_path = ""
        if mode == "onnx_fp32":
            model_path = str(PROJECT_ROOT / "exports" / "hand_tracker_simplified.onnx")
        elif mode == "onnx_int8":
            model_path = str(PROJECT_ROOT / "exports" / "hand_tracker_int8.onnx")
        
        if not model_path or not os.path.exists(model_path):
            print(f"[RAEZ] ONNX model path not found: {model_path}")
            return None
        
        try:
            print(f"[RAEZ] Loading ONNX session for {mode} from {model_path}...")
            import onnxruntime as ort
            providers = []
            if torch.cuda.is_available():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
            providers.append('CPUExecutionProvider')
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            _onnx_sessions[mode] = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            print(f"[RAEZ] ONNX session {mode} loaded successfully.")
        except Exception as e:
            print(f"[RAEZ] Error loading ONNX session: {e}")
            return None
            
    return _onnx_sessions[mode]

# ── /test-image ──────────────────────────────────────────────────────────────────────────────
@app.post("/test-image")
async def test_image(image_data: dict):
    global _model_instance
    try:

        img_b64 = image_data["image"].split(",")[-1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        start_time = time.time()

        img_np = np.array(img.resize((256, 256))) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
        
        # ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mode = image_data.get("mode", "pytorch")

        if mode in ("onnx_fp32", "onnx_int8"):
            session = get_onnx_session(mode)
            if session is not None:
                input_name = session.get_inputs()[0].name
                img_onnx = img_tensor.cpu().numpy()
                outputs = session.run(None, {input_name: img_onnx})
                coords_2d = outputs[1][0] * 256.0
                joints_3d = outputs[2][0]
            else:
                return {"error": f"ONNX Engine '{mode}' could not be initialized."}
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_tensor = img_tensor.to(device)

            if _model_instance is None:
                print("[RAEZ] Loading model for inference...")
                latest_ckpt = get_latest_checkpoint_path()
                if latest_ckpt:
                    success = load_checkpoint_into_model(latest_ckpt)
                    if not success:
                        print("[RAEZ] Failed to load latest checkpoint, fallback to raw model.")
                        _model_instance = HandBoneTracker(pretrained=False).to(device)
                        _model_instance.eval()
                else:
                    print("[RAEZ] No checkpoint found. Loading default untrained model.")
                    _model_instance = HandBoneTracker(pretrained=False).to(device)
                    _model_instance.eval()
            else:
                _model_instance = _model_instance.to(device)

            with torch.no_grad():
                preds = _model_instance(img_tensor)

            coords_2d = preds["coords_2d"][0].cpu().numpy() * 256.0
            joints_3d = preds["joints_3d"][0].cpu().numpy()
        
        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "keypoints": coords_2d.tolist(),
            "joints_3d": joints_3d.tolist(),
            "latency_ms": latency_ms
        }

    except Exception as e:
        import traceback
        print(f"[RAEZ] Inference error: {traceback.format_exc()}")
        return {"error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
