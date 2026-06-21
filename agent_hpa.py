"""
HPA Intelligent Training Agent
Reads all historical run metrics, reasons about patterns, decides next hyperparameters.
Does NOT run a fixed grid — every config decision is based on observed evidence.
Dashboard reads: logs/agent_status.json + logs/agent_log.jsonl
"""
import os, sys, json, time, signal, subprocess, shutil, random, math
from pathlib import Path
from datetime import datetime

BASE         = Path('/home/hnguye24/morphogene')
LOGS         = BASE / 'logs'
CKPT_DIR     = LOGS / 'checkpoints'
RUNS_DIR     = LOGS / 'runs'
TRAIN_SCRIPT = BASE / 'train_hpa.py'
CONDA_INIT   = '/share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh'
CONDA_ENV    = 'bm_seg2'

AGENT_STATUS  = LOGS / 'agent_status.json'
AGENT_LOG     = LOGS / 'agent_log.jsonl'
AGENT_PID_FILE= LOGS / 'agent.pid'
TRAIN_STATUS  = LOGS / 'train_status.json'
TRAIN_CFG     = LOGS / 'train_config.json'
TRAIN_METRICS = LOGS / 'train_log.jsonl'
TRAIN_STDOUT  = LOGS / 'train_stdout.log'

LOGS.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

# ── State ─────────────────────────────────────────────────────────────────────
_agent_running = True
_current_proc  = None
_total_runs    = 0
_start_time    = datetime.now()

# Known models to explore before pure exploitation
KNOWN_MODELS = ['efficientnet_b0', 'efficientnet_b3', 'resnet50', 'resnet34']


def ts():
    return datetime.now().isoformat(timespec='seconds')


def log(msg, level='INFO', extra=None):
    entry = {'ts': ts(), 'level': level, 'msg': msg}
    if extra:
        entry.update(extra)
    with open(AGENT_LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f"[{entry['ts']}] [{level}] {msg}")


def write_status(**kw):
    with open(AGENT_STATUS, 'w') as f:
        json.dump({'ts': ts(), 'agent_pid': os.getpid(),
                   'uptime': str(datetime.now() - _start_time).split('.')[0],
                   **kw}, f, indent=2)


# ── Data readers ──────────────────────────────────────────────────────────────
def read_run_metrics(run_dir):
    """Read per-epoch metrics from a run directory."""
    p = run_dir / 'metrics.jsonl'
    if not p.exists():
        return []
    eps = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    eps.append(json.loads(line))
                except:
                    pass
    return eps


def read_run_meta(run_dir):
    p = run_dir / 'meta.json'
    if not p.exists():
        # fallback: reconstruct from config + status
        meta = {}
        cp = run_dir / 'config.json'
        sp = run_dir / 'status.json'
        if cp.exists():
            try:
                meta['config'] = json.load(open(cp))
                meta['model'] = meta['config'].get('model', '?')
            except:
                pass
        if sp.exists():
            try:
                s = json.load(open(sp))
                meta['best_f1'] = s.get('best_f1', 0.0)
                meta['epochs_done'] = s.get('epoch', 0)
            except:
                pass
        return meta
    try:
        return json.load(open(p))
    except:
        return {}


def load_all_runs():
    """Return list of dicts with run summary + epoch data for every completed run."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = read_run_meta(d)
        eps  = read_run_metrics(d)
        if not eps:
            continue
        meta['epochs'] = eps
        meta['run_id'] = d.name
        runs.append(meta)
    return runs


def read_live_metrics():
    """Read current in-progress run metrics."""
    if not TRAIN_METRICS.exists():
        return []
    eps = []
    try:
        with open(TRAIN_METRICS) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        eps.append(json.loads(line))
                    except:
                        pass
    except:
        pass
    return eps


# ── Analysis functions ────────────────────────────────────────────────────────
def analyze_run(epochs, cfg=None):
    """Deeply analyze a list of epoch dicts. Returns a structured insight dict."""
    if not epochs:
        return {'status': 'no_data'}

    final   = epochs[-1]
    best_f1 = max(e.get('macro_f1', 0) for e in epochs)
    best_ep = next((i+1 for i, e in enumerate(epochs) if e.get('macro_f1', 0) == best_f1), len(epochs))
    n_ep    = len(epochs)

    # Convergence speed: how many epochs to reach 80% of best F1
    target_f1 = best_f1 * 0.8
    conv_ep = n_ep
    for i, e in enumerate(epochs):
        if e.get('macro_f1', 0) >= target_f1:
            conv_ep = i + 1
            break

    # Overfitting: compare train vs val loss in last 20% of epochs
    tail = epochs[max(0, int(n_ep * 0.8)):]
    if len(tail) >= 2:
        avg_train_tail = sum(e.get('train_loss', 0) for e in tail) / len(tail)
        avg_val_tail   = sum(e.get('val_loss', 0) for e in tail) / len(tail)
        overfit_ratio  = avg_val_tail / (avg_train_tail + 1e-9)
    else:
        avg_train_tail, avg_val_tail, overfit_ratio = 0, 0, 1.0

    # Val loss trend in tail: is it still decreasing or increasing?
    if len(tail) >= 3:
        val_losses = [e.get('val_loss', 0) for e in tail]
        val_trend  = (val_losses[-1] - val_losses[0]) / (len(val_losses) - 1)  # positive = worsening
    else:
        val_trend = 0.0

    # F1 trend: still improving at end?
    if len(epochs) >= 4:
        f1s = [e.get('macro_f1', 0) for e in epochs]
        last_quarter = f1s[max(0, int(n_ep * 0.75)):]
        f1_trend = (last_quarter[-1] - last_quarter[0]) / max(1, len(last_quarter) - 1)
    else:
        f1_trend = 0.0

    # Per-class weaknesses: classes with F1 < 0.1 in last epoch
    pcf1 = final.get('per_class_f1', [])
    weak_classes = [i for i, v in enumerate(pcf1) if v < 0.1]
    strong_classes = [i for i, v in enumerate(pcf1) if v > 0.5]

    # LR at end of training
    final_lr = final.get('lr', None)

    return {
        'best_f1':       round(best_f1, 4),
        'best_ep':       best_ep,
        'n_ep':          n_ep,
        'conv_ep':       conv_ep,           # epochs to reach 80% of best F1
        'overfit_ratio': round(overfit_ratio, 3),  # val/train loss ratio (>1.5 = overfitting)
        'val_trend':     round(val_trend, 5),       # positive = val loss worsening
        'f1_trend':      round(f1_trend, 5),        # positive = F1 still improving at end
        'final_lr':      final_lr,
        'weak_classes':  weak_classes,      # indices with F1 < 0.1
        'strong_classes': strong_classes,
        'final_train_loss': round(avg_train_tail, 4),
        'final_val_loss':   round(avg_val_tail, 4),
    }


def summarize_portfolio(runs):
    """Build a summary of what has been tried and what works best."""
    if not runs:
        return {}

    by_model = {}
    for r in runs:
        model = r.get('model') or r.get('config', {}).get('model', '?')
        cfg   = r.get('config', {})
        eps   = r.get('epochs', [])
        insight = analyze_run(eps, cfg)
        entry = {
            'run_id':   r['run_id'],
            'best_f1':  insight['best_f1'],
            'n_ep':     insight['n_ep'],
            'overfit':  insight['overfit_ratio'],
            'f1_trend': insight['f1_trend'],
            'val_trend':insight['val_trend'],
            'lr':       cfg.get('lr', None),
            'bs':       cfg.get('batch_size', None),
            'img_size': cfg.get('img_size', 224),
            'epochs':   cfg.get('epochs', None),
            'config':   cfg,
        }
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(entry)

    # Best overall
    all_results = [(r.get('model') or r.get('config', {}).get('model'), r.get('best_f1', 0), r)
                   for r in runs if r.get('epochs')]
    best = max(all_results, key=lambda x: x[1]) if all_results else None

    return {'by_model': by_model, 'best': best, 'n_runs': len(runs)}


# ── Intelligent decision making ───────────────────────────────────────────────
def decide_next_config(runs):
    """
    Core reasoning function. Reads all historical runs, reasons about patterns,
    and returns (config_dict, reasoning_string).
    """
    portfolio = summarize_portfolio(runs)
    n_runs = portfolio.get('n_runs', 0)
    by_model = portfolio.get('by_model', {})
    best = portfolio.get('best')  # (model, f1, run_dict)

    reasons = []

    # ── PHASE 1: Exploration ──────────────────────────────────────────────────
    # If we haven't tried all models at least once, explore systematically
    models_tried = set(by_model.keys())
    models_not_tried = [m for m in KNOWN_MODELS if m not in models_tried]

    if models_not_tried:
        model = models_not_tried[0]
        # Pick a safe LR for cold start
        lr = 1e-4
        bs = 16 if 'b3' in model else 32
        epochs = 20
        reasons.append(f"EXPLORE: '{model}' not yet tested. Starting with baseline config.")
        reasons.append(f"Decision: lr={lr:.1e}, bs={bs}, epochs={epochs}, img=224")
        cfg = dict(model=model, lr=lr, batch_size=bs, epochs=epochs,
                   img_size=224, pretrained=True, val_split=0.1, num_workers=4,
                   phase='explore')
        return cfg, '\n'.join(reasons)

    # ── PHASE 2: Analyze best model so far ────────────────────────────────────
    best_model, best_f1_ever, best_run = best if best else ('efficientnet_b0', 0.0, {})
    best_cfg = best_run.get('config', {}) if best_run else {}
    best_eps = best_run.get('epochs', []) if best_run else []
    best_insight = analyze_run(best_eps, best_cfg) if best_eps else {}

    reasons.append(f"PORTFOLIO: {n_runs} runs done. Best so far: {best_model} F1={best_f1_ever:.4f}")

    # Summarize model ranking
    for m, runs_m in sorted(by_model.items(), key=lambda x: -max(r['best_f1'] for r in x[1])):
        top = max(runs_m, key=lambda r: r['best_f1'])
        reasons.append(f"  {m}: best_f1={top['best_f1']:.4f} ({len(runs_m)} runs)")

    # ── PHASE 3: Diagnose last best run ──────────────────────────────────────
    if best_insight:
        overfit = best_insight.get('overfit_ratio', 1.0)
        f1_trend = best_insight.get('f1_trend', 0.0)
        val_trend = best_insight.get('val_trend', 0.0)
        conv_ep = best_insight.get('conv_ep', 0)
        n_ep = best_insight.get('n_ep', 0)
        weak = best_insight.get('weak_classes', [])
        best_ep = best_insight.get('best_ep', n_ep)

        reasons.append(f"\nDIAGNOSIS of best run ({best_model}):")
        reasons.append(f"  overfit_ratio={overfit:.2f} (val/train loss; >1.5 = overfitting)")
        reasons.append(f"  f1_trend={f1_trend:.5f} (positive = still improving at end)")
        reasons.append(f"  val_trend={val_trend:.5f} (positive = val loss worsening)")
        reasons.append(f"  best_ep={best_ep}/{n_ep}  conv_ep={conv_ep}")
        if weak:
            reasons.append(f"  weak classes ({len(weak)}): {weak[:10]}{'...' if len(weak)>10 else ''}")
    else:
        overfit, f1_trend, val_trend, conv_ep, n_ep, weak, best_ep = 1.0, 0.0, 0.0, 5, 20, [], 10
        reasons.append("\nNo deep insight available for best run.")

    # ── PHASE 4: Generate next config based on diagnosis ──────────────────────
    cur_lr = float(best_cfg.get('lr', 1e-4))
    cur_bs = int(best_cfg.get('batch_size', 32))
    cur_img = int(best_cfg.get('img_size', 224))
    cur_epochs = int(best_cfg.get('epochs', 20))

    next_model  = best_model
    next_lr     = cur_lr
    next_bs     = cur_bs
    next_img    = cur_img
    next_epochs = 30
    decision_reason = ""

    # Check if we have enough data to exploit
    best_model_runs = by_model.get(best_model, [])
    lrs_tried = {r['lr'] for r in best_model_runs if r['lr'] is not None}

    # Case A: Strongly overfitting — reduce LR, keep model
    if overfit > 1.6 and val_trend > 0:
        new_lr = cur_lr * 0.3
        decision_reason = (f"OVERFIT detected (ratio={overfit:.2f}, val_loss rising). "
                           f"Cutting LR from {cur_lr:.1e} to {new_lr:.1e}. "
                           f"Also reducing epochs to {max(20, best_ep+5)} to stop before overfit.")
        next_lr     = new_lr
        next_epochs = max(20, best_ep + 5)

    # Case B: Still improving at end — more epochs, same config
    elif f1_trend > 0.001 and overfit < 1.4:
        new_epochs = min(80, cur_epochs + 20)
        decision_reason = (f"Model still IMPROVING at end (f1_trend={f1_trend:.4f}). "
                           f"Extending epochs from {cur_epochs} to {new_epochs} with same config.")
        next_epochs = new_epochs

    # Case C: Converged fast, low LR might be suboptimal — try higher
    elif conv_ep < n_ep * 0.3 and overfit < 1.3 and 1e-3 not in lrs_tried:
        new_lr = min(cur_lr * 3.0, 5e-4)
        decision_reason = (f"Fast convergence (reached 80% best F1 in {conv_ep}/{n_ep} epochs). "
                           f"LR={cur_lr:.1e} may be too low. Trying {new_lr:.1e}.")
        next_lr = new_lr
        next_epochs = 25

    # Case D: Try larger image size if we haven't
    elif cur_img == 224 and best_f1_ever > 0.25:
        imgs_tried = {r['img_size'] for r in best_model_runs if r.get('img_size')}
        if 384 not in imgs_tried:
            decision_reason = (f"Good F1={best_f1_ever:.4f} at 224px. "
                               f"Trying 384px for {best_model} — more spatial detail may help weak classes.")
            next_img    = 384
            next_bs     = max(8, cur_bs // 2)  # larger images → smaller batch
            next_epochs = 30

    # Case E: Try second-best model with different LR
    elif n_runs >= 6:
        # Find model with best F1 that isn't best_model
        model_bests = {m: max(r['best_f1'] for r in rs)
                       for m, rs in by_model.items()}
        alt_models = sorted([(f1, m) for m, f1 in model_bests.items() if m != best_model], reverse=True)
        if alt_models:
            _, alt_model = alt_models[0]
            alt_runs = by_model[alt_model]
            alt_lrs_tried = {r['lr'] for r in alt_runs}
            # Try LR that wasn't tested for this model
            candidate_lrs = [5e-5, 1e-4, 2e-4, 5e-4]
            untried_lrs = [lr for lr in candidate_lrs if lr not in alt_lrs_tried]
            if untried_lrs:
                new_lr = untried_lrs[0]
                next_model = alt_model
                next_lr    = new_lr
                next_bs    = 16 if 'b3' in alt_model else 32
                next_epochs = 25
                decision_reason = (f"Investigating {alt_model} (best_f1={model_bests[alt_model]:.4f}) "
                                   f"with untried LR={new_lr:.1e}.")
            else:
                decision_reason = "All alt models and LRs explored. Perturbing best config."
                next_lr = cur_lr * random.choice([0.5, 0.7, 1.5, 2.0])
                next_epochs = 30
        else:
            decision_reason = "Only one model in portfolio. Perturbing LR."
            next_lr = cur_lr * random.choice([0.5, 2.0])
    else:
        # General: try nearby LR not yet tested on best model
        candidate_lrs = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        untried = [lr for lr in candidate_lrs if lr not in lrs_tried]
        if untried:
            next_lr = untried[0]
            decision_reason = (f"Exploring untried LR={next_lr:.1e} for {best_model} "
                               f"(already tried: {sorted(lrs_tried)}).")
        else:
            next_lr = cur_lr * random.choice([0.5, 1.5])
            decision_reason = f"All candidate LRs tried for {best_model}. Random perturbation: LR→{next_lr:.1e}."
        next_epochs = 25

    reasons.append(f"\nDECISION: {decision_reason}")

    # Safety clamps
    next_lr     = float(max(1e-6, min(1e-2, next_lr)))
    next_bs     = int(max(8, min(64, next_bs)))
    next_epochs = int(max(10, min(100, next_epochs)))

    cfg = dict(
        model=next_model, lr=round(next_lr, 8), batch_size=next_bs,
        epochs=next_epochs, img_size=next_img,
        pretrained=True, val_split=0.1, num_workers=4,
        phase='reason'
    )
    reasons.append(f"Config: {cfg}")
    return cfg, '\n'.join(reasons)


# ── Training subprocess ───────────────────────────────────────────────────────
def start_training(cfg):
    global _current_proc
    if TRAIN_METRICS.exists():
        TRAIN_METRICS.unlink()
    with open(TRAIN_CFG, 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(TRAIN_STATUS, 'w') as f:
        json.dump({'status': 'starting', 'epoch': 0, 'total': cfg['epochs'],
                   'pid': None, 'model': cfg['model'], 'device': None, 'best_f1': 0.0}, f)
    cmd = (f'source {CONDA_INIT} && conda activate {CONDA_ENV} && '
           f'python {TRAIN_SCRIPT} --config {TRAIN_CFG}')
    stdout_f = open(TRAIN_STDOUT, 'w')
    proc = subprocess.Popen(['bash', '-c', cmd],
                            stdout=stdout_f, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    stdout_f.close()  # parent closes after fork; child inherits the fd
    _current_proc = proc
    log(f"Started: {cfg['model']} lr={cfg['lr']:.1e} bs={cfg['batch_size']} "
        f"ep={cfg['epochs']} sz={cfg.get('img_size',224)}",
        level='START', extra={'config': cfg, 'pid': proc.pid})
    return proc


def archive_run(run_id, cfg, epochs_done, best_f1):
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [(TRAIN_METRICS, 'metrics.jsonl'),
                     (TRAIN_STATUS,  'status.json'),
                     (TRAIN_CFG,     'config.json'),
                     (TRAIN_STDOUT,  'stdout.log')]:
        if src.exists():
            shutil.copy2(src, run_dir / dst)
    with open(run_dir / 'meta.json', 'w') as f:
        json.dump({'run_id': run_id, 'model': cfg.get('model'),
                   'epochs_done': epochs_done, 'best_f1': round(best_f1, 4),
                   'phase': cfg.get('phase', '?'), 'config': cfg,
                   'archived_at': ts()}, f, indent=2)
    log(f"Archived → runs/{run_id}/  best_f1={best_f1:.4f}",
        extra={'run_id': run_id, 'best_f1': round(best_f1, 4)})


def epoch_comment(eps, cfg, best_f1_ever):
    """
    Analyze live epoch metrics and return a comment string (or None if nothing notable).
    Called after each new epoch completes.
    """
    if not eps:
        return None
    n = len(eps)
    last = eps[-1]
    ep   = last['epoch']
    mf1  = last.get('macro_f1', 0)
    tl   = last.get('train_loss', 0)
    vl   = last.get('val_loss', 0)
    total = cfg.get('epochs', 1)
    comments = []

    # ── First epoch baseline ──────────────────────────────────────────────────
    if ep == 1:
        comments.append(f"Ep1 baseline: F1={mf1:.4f} train_loss={tl:.4f} val_loss={vl:.4f}.")
        if mf1 < 0.05:
            comments.append("Very low F1 at ep1 — model may be struggling with class imbalance. Expected to improve.")
        return ' '.join(comments) if comments else None

    # ── Overfit signal ────────────────────────────────────────────────────────
    if n >= 3:
        gap = vl - tl
        prev_gap = eps[-2].get('val_loss', 0) - eps[-2].get('train_loss', 0)
        if gap > 0.05 and gap > prev_gap * 1.3:
            comments.append(f"⚠ Overfit widening: train={tl:.4f} val={vl:.4f} gap={gap:.4f} (was {prev_gap:.4f}).")

    # ── Val loss suddenly spiked ──────────────────────────────────────────────
    if n >= 2:
        prev_vl = eps[-2].get('val_loss', 0)
        if prev_vl > 0 and vl > prev_vl * 1.25:
            comments.append(f"⚠ Val loss jumped {prev_vl:.4f}→{vl:.4f} (+{((vl/prev_vl-1)*100):.0f}%).")

    # ── F1 stagnation in middle of training ───────────────────────────────────
    if n >= 5 and ep < int(total * 0.75):
        recent_f1s = [e.get('macro_f1', 0) for e in eps[-5:]]
        f1_range   = max(recent_f1s) - min(recent_f1s)
        if f1_range < 0.002 and mf1 < 0.3:
            comments.append(f"F1 stagnant in last 5 epochs (range={f1_range:.4f}, F1={mf1:.4f}). May need LR change.")

    # ── New all-time best ─────────────────────────────────────────────────────
    run_best = max(e.get('macro_f1', 0) for e in eps)
    if mf1 >= run_best and mf1 > best_f1_ever and ep > 1:
        comments.append(f"★ New all-time best! F1={mf1:.4f} beats previous best {best_f1_ever:.4f}.")

    # ── Periodic milestone every 5 epochs (independent of whether it's a new best) ──
    if ep > 1 and ep % 5 == 0:
        pct_done = round(ep / total * 100)
        comments.append(f"Ep{ep}/{total} ({pct_done}%): F1={mf1:.4f} tl={tl:.4f} vl={vl:.4f} "
                        f"run_best={run_best:.4f}.")

    # ── Halfway point summary (use >= to survive polling skips) ──────────────
    halfway = total // 2
    if halfway > 0 and ep >= halfway and (n < 2 or eps[-2]['epoch'] < halfway):
        pcf1 = last.get('per_class_f1', [])
        weak = [i for i, v in enumerate(pcf1) if v < 0.05]
        comments.append(f"Halfway ({ep}/{total}): F1={mf1:.4f}. "
                         f"{len(weak)} classes still at near-zero F1.")

    # ── Last epoch summary (use >= to survive polling skips) ─────────────────
    if ep >= total:
        pcf1 = last.get('per_class_f1', [])
        n_cls = len(pcf1)
        if n_cls > 0:
            good = sum(1 for v in pcf1 if v > 0.3)
            weak = [i for i, v in enumerate(pcf1) if v < 0.05]
            comments.append(f"Final ep{ep}: F1={mf1:.4f}. {good}/{n_cls} classes >0.3. "
                             f"{len(weak)} near-zero: {weak[:8]}{'...' if len(weak)>8 else ''}.")
        else:
            comments.append(f"Final ep{ep}: F1={mf1:.4f}. (per_class_f1 unavailable)")

    return ' | '.join(comments) if comments else None


def wait_for_training(proc, cfg, best_f1_ever, best_cfg):
    """Wait for training, logging per-epoch comments. Returns (status, best_f1, epochs_done)."""
    last_epoch = 0
    last_progress = time.time()
    timeout_no_progress = 1800  # 30 min
    epoch_times = []  # wall-clock times at each new epoch (for ETA)

    while True:
        time.sleep(15)  # poll every 15s for faster per-epoch feedback

        ret = proc.poll()
        if ret is not None:
            eps = read_live_metrics()
            run_f1 = max((e.get('macro_f1', 0) for e in eps), default=0.0)
            done_ep = eps[-1]['epoch'] if eps else 0
            if ret == 0:
                log(f"Training finished. ep={done_ep} f1={run_f1:.4f}", level='DONE',
                    extra={'best_f1': round(run_f1, 4), 'epochs': done_ep})
                return 'done', run_f1, done_ep
            else:
                log(f"Training crashed (exit={ret}). ep={done_ep}", level='CRASH',
                    extra={'exit_code': ret, 'epochs': done_ep})
                return 'crash', run_f1, done_ep

        eps = read_live_metrics()
        cur_ep = eps[-1]['epoch'] if eps else 0
        cur_f1 = eps[-1].get('macro_f1', 0) if eps else 0

        if cur_ep > last_epoch:
            now = time.time()
            epoch_times.append(now)
            last_epoch = cur_ep
            last_progress = now

            # Compute ETA
            sec_per_ep = None
            if len(epoch_times) >= 2:
                sec_per_ep = (epoch_times[-1] - epoch_times[0]) / (len(epoch_times) - 1)
            remaining = cfg['epochs'] - cur_ep
            eta_str = ''
            if sec_per_ep and remaining > 0:
                secs = int(sec_per_ep * remaining)
                if secs < 60:   eta_str = f'{secs}s'
                elif secs < 3600: eta_str = f'{secs//60}min'
                else:           eta_str = f'{secs/3600:.1f}h'

            write_status(state='training', model=cfg.get('model'), phase=cfg.get('phase'),
                         epoch=cur_ep, total=cfg['epochs'],
                         current_f1=round(cur_f1, 4),
                         best_f1_ever=round(best_f1_ever, 4),
                         best_model=best_cfg.get('model') if best_cfg else None,
                         total_runs=_total_runs,
                         eta=eta_str,
                         sec_per_epoch=round(sec_per_ep, 1) if sec_per_ep else None)

            # Per-epoch commentary
            comment = epoch_comment(eps, cfg, best_f1_ever)
            if comment:
                log(comment, level='WATCH',
                    extra={'epoch': cur_ep, 'total': cfg['epochs'],
                           'macro_f1': round(cur_f1, 4),
                           'eta': eta_str})

        if time.time() - last_progress > timeout_no_progress and cur_ep > 0:
            log(f"Stalled ({timeout_no_progress//60}min no progress) — killing", level='STALL')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
            time.sleep(5)
            return 'stall', cur_f1, cur_ep

        if not _agent_running:
            log("Stop requested — killing training", level='STOP')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
            return 'stopped', cur_f1, cur_ep


def signal_handler(sig, frame):
    global _agent_running
    log("Agent received stop signal", level='STOP')
    _agent_running = False
    # Remove PID file so dashboard knows we stopped
    if AGENT_PID_FILE.exists():
        AGENT_PID_FILE.unlink()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global _total_runs, _agent_running

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT,  signal_handler)

    log("=" * 60, level='BOOT')
    log(f"Intelligent HPA Agent started. PID={os.getpid()}", level='BOOT')
    log("Strategy: evidence-based reasoning, no fixed grid.", level='BOOT')
    log("=" * 60, level='BOOT')

    # Write PID file so dashboard can reconnect after its own restart
    AGENT_PID_FILE.write_text(str(os.getpid()))

    write_status(state='starting', total_runs=0, best_f1_ever=0.0,
                 best_model=None, phase='init')

    best_f1_ever = 0.0
    best_cfg     = None
    consecutive_crashes = 0

    while _agent_running:
        # ── THINK: read all historical runs and decide ─────────────────────
        write_status(state='thinking', total_runs=_total_runs,
                     best_f1_ever=round(best_f1_ever, 4),
                     best_model=best_cfg.get('model') if best_cfg else None)
        log("Reading portfolio and reasoning about next config...", level='THINK')

        runs = load_all_runs()
        cfg, reasoning = decide_next_config(runs)
        _total_runs += 1
        run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log(f"Run #{_total_runs} decision for: {cfg['model']} lr={cfg['lr']:.1e} "
            f"bs={cfg['batch_size']} ep={cfg['epochs']} sz={cfg.get('img_size',224)}",
            level='REASON', extra={'reasoning': reasoning, 'config': cfg,
                                   'run_index': _total_runs})

        write_status(state='launching', total_runs=_total_runs,
                     model=cfg.get('model'), phase=cfg.get('phase'),
                     best_f1_ever=round(best_f1_ever, 4),
                     best_model=best_cfg.get('model') if best_cfg else None,
                     current_reasoning=reasoning[:400])

        # ── ACT: launch training ──────────────────────────────────────────
        try:
            proc = start_training(cfg)
        except Exception as e:
            log(f"Failed to start training: {e}", level='ERROR')
            consecutive_crashes += 1
            if consecutive_crashes >= 3:
                log("3 launch failures — sleeping 5min", level='ERROR')
                time.sleep(300)
                consecutive_crashes = 0
            continue

        # ── WAIT: monitor progress ────────────────────────────────────────
        status, run_f1, epochs_done = wait_for_training(proc, cfg, best_f1_ever, best_cfg)

        # Archive regardless of exit status — partial runs count too
        # so the next agent restart doesn't re-run the same model from scratch
        if epochs_done > 0:
            archive_run(run_id, cfg, epochs_done, run_f1)

        if status == 'stopped':
            log("Agent stopping cleanly.", level='STOP')
            break

        if run_f1 > best_f1_ever:
            best_f1_ever = run_f1
            best_cfg = cfg.copy()
            log(f"New best! F1={run_f1:.4f} with {cfg['model']} lr={cfg['lr']:.1e}",
                level='BEST', extra={'f1': round(run_f1, 4), 'config': cfg})
            # Copy best checkpoint
            for pt in CKPT_DIR.glob('best_ep*.pt'):
                dest = CKPT_DIR / f'BEST_OVERALL_{cfg["model"]}_f1{run_f1:.4f}.pt'
                if not dest.exists():
                    shutil.copy2(pt, dest)

        if status == 'crash':
            consecutive_crashes += 1
            if consecutive_crashes >= 3:
                log("3 consecutive crashes — sleeping 2min", level='WARN')
                time.sleep(120)
                consecutive_crashes = 0
            else:
                log(f"Crash #{consecutive_crashes} — retrying in 30s", level='WARN')
                time.sleep(30)
        else:
            consecutive_crashes = 0
            time.sleep(10)

    write_status(state='stopped', total_runs=_total_runs,
                 best_f1_ever=round(best_f1_ever, 4),
                 best_model=best_cfg.get('model') if best_cfg else None)
    log(f"Agent stopped. Total runs: {_total_runs}. Best F1: {best_f1_ever:.4f}", level='STOP')
    if AGENT_PID_FILE.exists():
        AGENT_PID_FILE.unlink()


if __name__ == '__main__':
    main()
