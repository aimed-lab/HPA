#!/usr/bin/env python3
"""
HPA Dataset Workstation Dashboard
Flask app with 5 tabs: Explorer, Analysis, Train, Inference, Annotate
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, Response

# ── paths ──────────────────────────────────────────────────────────────────────
BASE        = Path("/home/hnguye24/morphogene")
TRAIN_DIR   = BASE / "train"
TRAIN_CSV   = BASE / "train.csv"
LOGS_DIR    = BASE / "logs"
CKPT_DIR    = LOGS_DIR / "checkpoints"
RUNS_DIR    = LOGS_DIR / "runs"
TRAIN_SCRIPT = BASE / "train_hpa.py"
CONDA_INIT  = "/share/apps/rc/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh"
CONDA_ENV   = "bm_seg2"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── class names ────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0:'Nucleoplasm', 1:'Nuclear membrane', 2:'Nucleoli',
    3:'Nucleoli fibrillar center', 4:'Nuclear speckles', 5:'Nuclear bodies',
    6:'Endoplasmic reticulum', 7:'Golgi apparatus', 8:'Peroxisomes',
    9:'Endosomes', 10:'Lysosomes', 11:'Intermediate filaments',
    12:'Actin filaments', 13:'Focal adhesion sites', 14:'Microtubules',
    15:'Microtubule ends', 16:'Cytokinetic bridge', 17:'Mitotic spindle',
    18:'Microtubule organizing center', 19:'Centrosome', 20:'Lipid droplets',
    21:'Plasma membrane', 22:'Cell junctions', 23:'Mitochondria',
    24:'Aggresome', 25:'Cytosol', 26:'Cytoplasmic bodies', 27:'Rods & rings'
}

# ── global caches ──────────────────────────────────────────────────────────────
_csv_cache = None
_csv_lock  = threading.Lock()
_model_cache = {}
_model_cache_lock = threading.Lock()

def load_csv():
    global _csv_cache
    with _csv_lock:
        if _csv_cache is not None:
            return _csv_cache
        rows = {}
        with open(TRAIN_CSV) as f:
            reader = csv.DictReader(f)
            for r in reader:
                labels = [int(x) for x in r['Target'].split()]
                rows[r['Id']] = labels
        _csv_cache = rows
    return _csv_cache

# ── image helpers ──────────────────────────────────────────────────────────────
def load_channels(image_id):
    """Return dict with keys blue, green, red, yellow as numpy arrays (H,W)."""
    ch = {}
    for color in ['blue', 'green', 'red', 'yellow']:
        p = TRAIN_DIR / f"{image_id}_{color}.png"
        if not p.exists():
            return None
        from PIL import Image
        arr = np.array(Image.open(p).convert('L'), dtype=np.uint8)
        ch[color] = arr
    return ch

def make_composite_png(image_id):
    """Return PNG bytes for composite image."""
    import io
    from PIL import Image
    ch = load_channels(image_id)
    if ch is None:
        return None
    R = np.clip(ch['red'].astype(np.uint16) + (ch['yellow'].astype(np.uint16) // 2), 0, 255).astype(np.uint8)
    G = np.clip(ch['green'].astype(np.uint16) + (ch['yellow'].astype(np.uint16) // 2), 0, 255).astype(np.uint8)
    B = ch['blue']
    composite = np.stack([R, G, B], axis=2)
    img = Image.fromarray(composite, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()

def make_channel_png(image_id, channel, colorize=True):
    """Return PNG bytes for a single channel, optionally false-colored."""
    import io
    from PIL import Image
    ch = load_channels(image_id)
    if ch is None:
        return None
    arr = ch[channel]
    if colorize:
        color_map = {
            'blue':   (0,   0,   255),
            'green':  (0,   255, 0),
            'red':    (255, 0,   0),
            'yellow': (255, 255, 0),
        }
        r, g, b = color_map[channel]
        rgb = np.stack([
            (arr.astype(np.uint16) * r // 255).astype(np.uint8),
            (arr.astype(np.uint16) * g // 255).astype(np.uint8),
            (arr.astype(np.uint16) * b // 255).astype(np.uint8),
        ], axis=2)
        img = Image.fromarray(rgb, 'RGB')
    else:
        img = Image.fromarray(arr, 'L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# HTML / Frontend
# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HPA Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;font-size:14px}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#1a1d27}
::-webkit-scrollbar-thumb{background:#2d3148;border-radius:3px}

/* layout */
#app{display:flex;flex-direction:column;height:100vh}
.topbar{display:flex;align-items:center;gap:16px;padding:10px 20px;background:#1a1d27;border-bottom:1px solid #2d3148;flex-shrink:0}
.topbar h1{font-size:18px;color:#a78bfa;font-weight:700;letter-spacing:.5px}
.topbar .badge{padding:2px 8px;border-radius:12px;font-size:11px;background:#2d3148;color:#94a3b8}
.tabs{display:flex;gap:4px;padding:8px 20px;background:#1a1d27;border-bottom:1px solid #2d3148;flex-shrink:0}
.tab-btn{padding:6px 16px;border:1px solid #2d3148;border-radius:6px;background:transparent;color:#94a3b8;cursor:pointer;font-size:13px;transition:all .15s}
.tab-btn:hover{border-color:#a78bfa;color:#a78bfa}
.tab-btn.active{background:#a78bfa22;border-color:#a78bfa;color:#a78bfa;font-weight:600}
.main{display:flex;flex:1;overflow:hidden;min-height:0}
.sidebar{width:220px;background:#1a1d27;border-right:1px solid #2d3148;padding:12px;overflow-y:auto;flex-shrink:0}
.content{flex:1;padding:16px;overflow-y:auto}
.tab-pane{display:none}
.tab-pane.active{display:block}

/* cards */
.card{background:#1a1d27;border:1px solid #2d3148;border-radius:8px;padding:14px;margin-bottom:14px}
.card-title{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;font-weight:600}

/* form controls */
label{font-size:12px;color:#94a3b8;display:block;margin-bottom:4px}
input[type=text],input[type=number],select{width:100%;padding:6px 10px;background:#0f1117;border:1px solid #2d3148;border-radius:6px;color:#e2e8f0;font-size:13px}
input[type=text]:focus,input[type=number]:focus,select:focus{outline:none;border-color:#a78bfa}
.btn{padding:7px 16px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}
.btn-primary{background:#a78bfa;color:#0f1117}
.btn-primary:hover{background:#c4b5fd}
.btn-danger{background:#ef4444;color:#fff}
.btn-danger:hover{background:#dc2626}
.btn-secondary{background:#2d3148;color:#e2e8f0}
.btn-secondary:hover{background:#3d4163}
.btn-sm{padding:4px 10px;font-size:12px}
.btn-success{background:#22c55e;color:#0f1117}
.btn-success:hover{background:#16a34a}
.btn-warning{background:#f59e0b;color:#0f1117}
.btn-warning:hover{background:#d97706}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}

/* status badge */
.status-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
.status-idle{background:#1e293b;color:#64748b}
.status-running{background:#7c3aed22;color:#a78bfa;animation:pulse 2s infinite}
.status-done{background:#16a34a22;color:#22c55e}
.status-error{background:#dc262622;color:#ef4444}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}

/* class sidebar */
.class-item{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:4px;cursor:pointer;margin-bottom:2px;transition:background .1s}
.class-item:hover{background:#2d3148}
.class-item.selected{background:#a78bfa22;border-left:2px solid #a78bfa;padding-left:4px}
.class-label{font-size:11px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#cbd5e1}
.class-count{font-size:10px;color:#64748b;width:32px;text-align:right;flex-shrink:0}
.class-bar-bg{flex:1;height:4px;background:#1e293b;border-radius:2px;overflow:hidden;max-width:54px}
.class-bar-fill{height:100%;background:#a78bfa;border-radius:2px}

/* image grid */
.img-grid{display:grid;gap:12px}
.grid-c4{grid-template-columns:repeat(4,1fr)}
.grid-c3{grid-template-columns:repeat(3,1fr)}
.grid-c2{grid-template-columns:repeat(2,1fr)}
.img-card{background:#1a1d27;border:1px solid #2d3148;border-radius:8px;overflow:hidden;transition:border-color .15s}
.img-card:hover{border-color:#a78bfa}
.img-composite{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#0f1117}
.img-channels{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;padding:4px;background:#0f1117}
.img-channels img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:2px}
.img-meta{padding:6px 8px}
.img-id{font-size:10px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.img-labels{display:flex;flex-wrap:wrap;gap:2px;margin-top:3px}
.lbl-chip{font-size:9px;padding:1px 5px;background:#2d3148;border-radius:10px;color:#a78bfa}

/* table */
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#1e293b;color:#94a3b8;padding:6px 10px;text-align:left;font-weight:600;border-bottom:1px solid #2d3148}
td{padding:5px 10px;border-bottom:1px solid #1a1d27;color:#cbd5e1}
tr:hover td{background:#1e293b}
.flag-dark{color:#f59e0b}
.flag-blown{color:#ef4444}

/* heatmap */
#heatmap-canvas{display:block;image-rendering:pixelated}

/* ── Analysis tab extras ──────────────────────────────────────────────── */
.analysis-stat-card{background:#0f1117;border:1px solid #2d3148;border-radius:10px;padding:16px 20px;flex:1;min-width:160px;position:relative;overflow:hidden}
.analysis-stat-card .asc-accent{position:absolute;top:0;left:0;width:4px;height:100%;border-radius:10px 0 0 10px}
.analysis-stat-card .asc-val{font-size:28px;font-weight:800;color:#e2e8f0;line-height:1.1;letter-spacing:-1px}
.analysis-stat-card .asc-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.7px;margin-top:4px;font-weight:600}
.analysis-stat-card .asc-sub{font-size:10px;color:#475569;margin-top:2px}
.cooc-table{border-collapse:collapse;font-size:9px;white-space:nowrap;width:100%}
.cooc-table th{padding:2px 3px;text-align:center;background:#1e293b;color:#64748b;position:sticky;top:0;z-index:2}
.cooc-table .cooc-row-hdr{position:sticky;left:0;background:#1a1d27;color:#94a3b8;font-size:9px;padding:2px 6px 2px 2px;z-index:1;white-space:nowrap;max-width:120px;overflow:hidden;text-overflow:ellipsis}
.cooc-table td{padding:1px 2px;text-align:center;min-width:20px;cursor:default}
.quality-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle}
.quality-dot-dark{background:#f59e0b}
.quality-dot-lowcontrast{background:#f87171}
.quality-dot-saturated{background:#60a5fa}
.quality-dot-good{background:#22c55e}
.imbalance-tbl tr.weight-severe td{background:#ef444411;border-left:3px solid #ef4444}
.imbalance-tbl tr.weight-warn td{background:#f59e0b11;border-left:3px solid #f59e0b}
.imbalance-tbl tr.weight-ok td{background:#22c55e11;border-left:3px solid #22c55e}
.log-scale-btn{padding:3px 10px;border:1px solid #2d3148;border-radius:4px;background:transparent;color:#94a3b8;cursor:pointer;font-size:11px;margin-left:8px;transition:all .15s}
.log-scale-btn.active{background:#a78bfa22;border-color:#a78bfa;color:#a78bfa}
.ch-stat-row{display:flex;gap:6px;font-size:11px;color:#94a3b8;margin-top:3px;flex-wrap:wrap}
.ch-stat-row span{background:#1e293b;padding:2px 6px;border-radius:4px}

/* inference */
.pred-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.pred-label{width:185px;font-size:12px;color:#cbd5e1;flex-shrink:0}
.pred-bar-bg{flex:1;height:13px;background:#1e293b;border-radius:3px;overflow:hidden}
.pred-bar-fill{height:100%;border-radius:3px;transition:width .3s}
.pred-score{font-size:11px;color:#94a3b8;width:45px;text-align:right;flex-shrink:0}
.pred-correct{color:#22c55e!important}
.pred-missed{color:#ef4444!important}
.pred-fp{color:#f59e0b!important}

/* annotate */
.checkbox-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.cb-item{display:flex;align-items:center;gap:5px;padding:5px 7px;border:1px solid #2d3148;border-radius:6px;cursor:pointer;transition:border-color .15s}
.cb-item:hover{border-color:#a78bfa}
.cb-item input{accent-color:#a78bfa;cursor:pointer;flex-shrink:0}
.cb-item label{font-size:11px;color:#cbd5e1;cursor:pointer;line-height:1.2}

/* misc */
.separator{height:1px;background:#2d3148;margin:10px 0}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #2d3148;border-top-color:#a78bfa;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.toast{position:fixed;bottom:20px;right:20px;padding:10px 18px;background:#1a1d27;border:1px solid #2d3148;border-radius:8px;color:#e2e8f0;z-index:9999;font-size:13px;animation:fadeIn .2s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.progress-bar-bg{height:8px;background:#1e293b;border-radius:4px;overflow:hidden;margin-top:4px}
.progress-bar-fill{height:100%;background:#a78bfa;border-radius:4px;transition:width .3s}
.chart-container{position:relative;height:260px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.flex-row{display:flex;gap:12px;flex-wrap:wrap}
.flex-col{flex:1;min-width:180px}

/* train tab extras */
.stat-box{background:#0f1117;border:1px solid #2d3148;border-radius:8px;padding:10px 14px;flex:1;min-width:110px;text-align:center}
.stat-box .sv{font-size:20px;font-weight:700;color:#a78bfa;line-height:1.2}
.stat-box .sl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-top:2px}
.status-overview{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:10px}
.status-dot{font-size:26px;line-height:1}
.status-plain{font-size:14px;color:#cbd5e1;font-weight:500}
.progress-bar-bg-lg{height:14px;background:#1e293b;border-radius:7px;overflow:hidden;position:relative;flex:1;min-width:120px}
.progress-bar-fill-lg{height:100%;background:linear-gradient(90deg,#7c3aed,#a78bfa);border-radius:7px;transition:width .4s;position:relative}
.progress-pct{position:absolute;right:6px;top:50%;transform:translateY(-50%);font-size:10px;font-weight:700;color:#fff;pointer-events:none}
.gpu-bar-bg{height:10px;background:#1e293b;border-radius:5px;overflow:hidden;flex:1}
.gpu-bar-fill{height:100%;border-radius:5px;transition:width .4s}
.gpu-row{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px}
.gpu-label{width:90px;color:#94a3b8;flex-shrink:0}
.gpu-val{width:70px;text-align:right;color:#cbd5e1;flex-shrink:0;font-size:11px}
.f1-poor{color:#ef4444}
.f1-fair{color:#f59e0b}
.f1-good{color:#22c55e}
.f1-excellent{color:#a78bfa}
.terminal-box{background:#0a0a0f;border:1px solid #2d3148;border-radius:6px;padding:10px 14px;font-family:'Courier New',monospace;font-size:11px;color:#4ade80;line-height:1.55;min-height:80px;max-height:260px;overflow-y:auto;white-space:pre-wrap;word-break:break-all}
.metrics-tbl th{cursor:pointer;user-select:none}
.metrics-tbl th:hover{color:#a78bfa}
.metrics-tbl .row-best-val{background:#1e3a5f22;border-left:2px solid #60a5fa}
.metrics-tbl .row-best-f1{background:#3b1d6022;border-left:2px solid #a78bfa}
.heatmap-tbl{border-collapse:collapse;font-size:9px;white-space:nowrap}
.heatmap-tbl th{padding:4px 2px;background:#1e293b;color:#64748b;position:sticky;top:0;z-index:2;white-space:nowrap}
.heatmap-tbl td{padding:2px 4px;text-align:center;min-width:28px}
.heatmap-tbl .epoch-col{position:sticky;left:0;background:#1a1d27;color:#64748b;z-index:1;padding-right:8px}
.epoch-window-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:12px;color:#94a3b8}
.epoch-window-row select{width:90px}
#chat-bubble {
  position:fixed;bottom:24px;right:24px;width:52px;height:52px;
  background:#a78bfa;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:22px;cursor:pointer;z-index:1000;
  box-shadow:0 4px 16px #0006;transition:transform .2s;user-select:none;
}
#chat-bubble:hover{transform:scale(1.1)}
#chat-panel {
  position:fixed;bottom:88px;right:24px;width:380px;max-height:520px;
  background:#1a1d27;border:1px solid #2d3148;border-radius:12px;
  display:none;flex-direction:column;z-index:1000;
  box-shadow:0 8px 32px #0008;overflow:hidden;
}
#chat-panel.open{display:flex}
#chat-header {
  padding:12px 16px;background:#22263a;border-bottom:1px solid #2d3148;
  display:flex;justify-content:space-between;align-items:center;
  font-weight:600;color:#a78bfa;font-size:14px;
}
#chat-messages {
  flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;
  min-height:200px;max-height:340px;
}
.chat-msg-user {
  align-self:flex-end;background:#a78bfa22;border:1px solid #a78bfa44;
  color:#e2e8f0;padding:7px 12px;border-radius:12px 12px 2px 12px;
  max-width:85%;font-size:13px;
}
.chat-msg-bot {
  align-self:flex-start;background:#22263a;border:1px solid #2d3148;
  color:#e2e8f0;padding:7px 12px;border-radius:12px 12px 12px 2px;
  max-width:92%;font-size:13px;line-height:1.5;
}
.chat-msg-bot b{color:#a78bfa}
#chat-suggestions {
  padding:6px 10px;display:flex;flex-wrap:wrap;gap:5px;border-top:1px solid #2d3148;
}
.chat-chip {
  font-size:11px;padding:3px 9px;border-radius:10px;background:#2d3148;
  color:#94a3b8;cursor:pointer;border:none;transition:all .15s;
}
.chat-chip:hover{background:#a78bfa22;color:#a78bfa;border-color:#a78bfa}
#chat-input-row {
  display:flex;gap:6px;padding:10px 12px;border-top:1px solid #2d3148;
}
#chat-input-row input {
  flex:1;padding:7px 10px;background:#0f1117;border:1px solid #2d3148;
  border-radius:6px;color:#e2e8f0;font-size:13px;
}
#chat-input-row input:focus{outline:none;border-color:#a78bfa}
#chat-input-row button {
  padding:7px 12px;background:#a78bfa;border:none;border-radius:6px;
  color:#0f1117;cursor:pointer;font-size:15px;font-weight:700;
}
#chat-input-row button:hover{background:#c4b5fd}
</style>
</head>
<body>
<div id="app">

<!-- topbar -->
<div class="topbar">
  <h1>HPA Dashboard</h1>
  <span class="badge" id="ds-stats">loading...</span>
  <span style="flex:1"></span>
  <span id="train-status-badge-top" class="status-badge status-idle">Idle</span>
</div>

<!-- tabs -->
<div class="tabs">
  <button class="tab-btn active"  onclick="switchTab('explorer',this)">Explorer</button>
  <button class="tab-btn"         onclick="switchTab('analysis',this)">Analysis</button>
  <button class="tab-btn"         onclick="switchTab('train',this)">Train</button>
  <button class="tab-btn"         onclick="switchTab('inference',this)">Inference</button>
  <button class="tab-btn"         onclick="switchTab('annotate',this)">Annotate</button>
  <button class="tab-btn"         onclick="switchTab('postanalysis',this)">Post Analysis</button>
</div>

<!-- main -->
<div class="main">

<!-- SIDEBAR (only visible on Explorer tab) -->
<div class="sidebar" id="sidebar-explorer">
  <div class="card-title">Filter by Class</div>
  <div style="display:flex;gap:6px;margin-bottom:8px">
    <select id="class-mode" style="flex:1">
      <option value="OR">OR</option>
      <option value="AND">AND</option>
    </select>
    <button class="btn btn-sm btn-secondary" onclick="clearClassFilter()">Clear</button>
  </div>
  <div id="class-list"></div>
</div>

<!-- CONTENT -->
<div class="content">

<!-- ===== TAB 1: Explorer ===== -->
<div class="tab-pane active" id="tab-explorer">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap">
    <label style="margin:0;white-space:nowrap">Grid size:</label>
    <select id="grid-size" onchange="loadImages()" style="width:70px">
      <option value="4">4</option>
      <option value="6">6</option>
      <option value="9" selected>9</option>
      <option value="12">12</option>
    </select>
    <label style="margin:0;white-space:nowrap">Split:</label>
    <select id="split-sel" onchange="loadImages()" style="width:90px">
      <option value="train">Train</option>
      <option value="all">All</option>
    </select>
    <button class="btn btn-secondary btn-sm" onclick="loadImages()">Shuffle</button>
    <span id="explorer-count" style="font-size:12px;color:#64748b"></span>
  </div>
  <div id="img-grid" class="img-grid grid-c3"></div>
</div>

<!-- ===== TAB 2: Analysis ===== -->
<div class="tab-pane" id="tab-analysis">

  <!-- ROW 1: Dataset Overview stat cards -->
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">
    <div class="analysis-stat-card">
      <div class="asc-accent" style="background:#a78bfa"></div>
      <div class="asc-val" id="asc-total-images">—</div>
      <div class="asc-label">Total Train Images</div>
      <div class="asc-sub" id="asc-total-sub"></div>
    </div>
    <div class="analysis-stat-card">
      <div class="asc-accent" style="background:#60a5fa"></div>
      <div class="asc-val" id="asc-multilabel">—</div>
      <div class="asc-label">Multi-label %</div>
      <div class="asc-sub" id="asc-multilabel-sub"></div>
    </div>
    <div class="analysis-stat-card">
      <div class="asc-accent" style="background:#4ade80"></div>
      <div class="asc-val" id="asc-avg-labels">—</div>
      <div class="asc-label">Avg Labels / Image</div>
      <div class="asc-sub" id="asc-avg-labels-sub"></div>
    </div>
    <div class="analysis-stat-card">
      <div class="asc-accent" style="background:#f59e0b"></div>
      <div class="asc-val" id="asc-annotated">—</div>
      <div class="asc-label">Total Annotated</div>
      <div class="asc-sub" id="asc-annotated-sub"></div>
    </div>
  </div>

  <!-- ROW 2: Class Distribution full width -->
  <div class="card" style="margin-bottom:14px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
      <span class="card-title" style="margin:0">Class Distribution — 28 Protein Localizations</span>
      <button id="dist-log-btn" class="log-scale-btn" onclick="toggleDistLog()">Log Scale</button>
    </div>
    <div style="position:relative;height:320px"><canvas id="dist-chart"></canvas></div>
  </div>

  <!-- ROW 3: Cardinality + Top Pairs -->
  <div class="two-col" style="margin-bottom:14px">
    <div class="card">
      <div class="card-title">Label Cardinality — Images by # of Labels</div>
      <div style="position:relative;height:240px"><canvas id="cardinality-chart"></canvas></div>
    </div>
    <div class="card" style="overflow-x:auto">
      <div class="card-title">Top Co-occurring Pairs</div>
      <div id="pairs-table-wrap" style="max-height:280px;overflow-y:auto">
        <div style="color:#64748b;font-size:12px">Loading...</div>
      </div>
    </div>
  </div>

  <!-- ROW 4: Co-occurrence Heatmap full width -->
  <div class="card" style="margin-bottom:14px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
      <span class="card-title" style="margin:0">Co-occurrence Matrix — How Often Classes Appear Together</span>
      <button id="cooc-log-btn" class="log-scale-btn" onclick="toggleCoocLog()">Log Scale</button>
      <span style="font-size:11px;color:#475569;margin-left:auto">Hover cells for details</span>
    </div>
    <div id="cooc-heatmap-wrap" style="overflow:auto;max-height:520px">
      <div style="color:#64748b;font-size:12px">Loading...</div>
    </div>
  </div>

  <!-- ROW 5: Histogram + Channel Stats -->
  <div class="two-col" style="margin-bottom:14px">
    <div class="card">
      <div class="card-title">Channel Histogram</div>
      <div style="display:flex;gap:6px;margin-bottom:8px;align-items:center">
        <input type="text" id="hist-id" placeholder="Image ID (blank=random)" style="flex:1"/>
        <button class="btn btn-secondary btn-sm" onclick="randomHistImage()">Random</button>
        <button class="btn btn-primary btn-sm" onclick="loadHistogram()">Load</button>
      </div>
      <div style="position:relative;height:220px"><canvas id="hist-chart"></canvas></div>
      <div id="hist-stats" style="margin-top:8px"></div>
    </div>
    <div class="card">
      <div class="card-title">Per-Channel Intensity Distribution <span style="font-weight:400;color:#475569">(n=300 sample)</span></div>
      <div id="ch-stats-wrap">
        <button class="btn btn-secondary btn-sm" onclick="loadChannelStats()">Compute (n=300)</button>
      </div>
    </div>
  </div>

  <!-- ROW 6: Quality Scatter full width -->
  <div class="card" style="margin-bottom:14px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
      <span class="card-title" style="margin:0">Image Quality Report <span style="font-weight:400;color:#475569">(n=300 sample)</span></span>
      <button class="btn btn-secondary btn-sm" onclick="loadQuality()">Run Quality Check</button>
      <span id="quality-summary" style="font-size:12px;color:#94a3b8;margin-left:8px"></span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div>
        <div style="position:relative;height:260px"><canvas id="quality-scatter"></canvas></div>
        <div style="display:flex;gap:12px;margin-top:6px;font-size:11px;flex-wrap:wrap">
          <span><span class="quality-dot quality-dot-dark"></span>Dark (mean&lt;15)</span>
          <span><span class="quality-dot quality-dot-lowcontrast"></span>Low contrast (std&lt;10)</span>
          <span><span class="quality-dot quality-dot-saturated"></span>Saturated (mean&gt;240)</span>
          <span><span class="quality-dot quality-dot-good"></span>Good</span>
        </div>
      </div>
      <div style="overflow-y:auto;max-height:300px" id="quality-flagged-table">
        <div style="color:#64748b;font-size:12px;padding:20px">Run quality check to see flagged images</div>
      </div>
    </div>
  </div>

  <!-- ROW 7: Class Imbalance full width -->
  <div class="card" style="margin-bottom:14px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
      <span class="card-title" style="margin:0">Class Imbalance Analysis — Recommended BCE Weights</span>
      <button class="btn btn-secondary btn-sm" onclick="exportImbalanceCSV()">Export CSV</button>
    </div>
    <div style="overflow-x:auto;max-height:400px;overflow-y:auto" id="imbalance-table-wrap">
      <div style="color:#64748b;font-size:12px">Loading...</div>
    </div>
  </div>

</div>

<!-- ===== TAB 3: Train ===== -->
<div class="tab-pane" id="tab-train">
  <!-- Auto-refresh control bar -->
  <div style="display:flex;align-items:center;gap:10px;padding:8px 14px;background:#1a1d27;border:1px solid #2d3148;border-radius:8px;margin-bottom:14px;flex-wrap:wrap">
    <span style="font-size:12px;color:#94a3b8;font-weight:600">Auto-refresh:</span>
    <button id="autorefresh-toggle" class="btn btn-success btn-sm" onclick="toggleAutoRefresh()" style="min-width:72px">&#9646;&#9646; Pause</button>
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:12px;color:#64748b">Metrics every</span>
      <input type="number" id="metrics-interval" value="10" min="1" max="300" style="width:58px;padding:4px 6px;font-size:12px" onchange="applyRefreshIntervals()"/>
      <span style="font-size:12px;color:#64748b">s</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:12px;color:#64748b">GPU every</span>
      <input type="number" id="gpu-interval" value="5" min="1" max="120" style="width:58px;padding:4px 6px;font-size:12px" onchange="applyRefreshIntervals()"/>
      <span style="font-size:12px;color:#64748b">s</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:12px;color:#64748b">Log every</span>
      <input type="number" id="log-interval" value="5" min="1" max="120" style="width:58px;padding:4px 6px;font-size:12px" onchange="applyRefreshIntervals()"/>
      <span style="font-size:12px;color:#64748b">s</span>
    </div>
    <button class="btn btn-secondary btn-sm" onclick="refreshTrainAll();refreshGPU();refreshTrainLog()">&#8635; Refresh Now</button>
    <span id="last-refresh-time" style="font-size:11px;color:#475569;margin-left:auto"></span>
  </div>

  <!-- ===== AGENT PANEL (top of Train tab) ===== -->
  <div class="card" id="agent-card" style="border:1px solid #7c3aed66;background:linear-gradient(135deg,#1e1b4b33,#0f172a);margin-bottom:14px">
    <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
      <span style="display:flex;align-items:center;gap:10px">
        <span id="agent-status-dot" style="font-size:20px">🟣</span>
        Intelligent Training Agent
        <span id="agent-badge" class="status-badge status-idle">Idle</span>
      </span>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn btn-sm" id="agent-start-btn" style="background:#7c3aed;color:#fff" onclick="agentStart()">&#9654; Start Agent</button>
        <button class="btn btn-secondary btn-sm" id="agent-stop-btn" onclick="agentStop()">&#9632; Stop</button>
        <button class="btn btn-secondary btn-sm" onclick="refreshAgent()">&#8635;</button>
      </div>
    </div>

    <!-- Compact two-column layout: stats + reasoning side by side -->
    <div style="display:grid;grid-template-columns:auto 1fr;gap:12px;margin-top:8px">
      <!-- Stats pills -->
      <div style="display:flex;flex-direction:column;gap:6px;min-width:160px">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
          <div style="background:#0f172a;border-radius:7px;padding:8px 10px;text-align:center">
            <div style="font-size:18px;font-weight:700;color:#a78bfa" id="agent-runs">—</div>
            <div style="font-size:10px;color:#64748b;margin-top:1px">Runs Done</div>
          </div>
          <div style="background:#0f172a;border-radius:7px;padding:8px 10px;text-align:center">
            <div style="font-size:18px;font-weight:700;color:#34d399" id="agent-best-f1">—</div>
            <div style="font-size:10px;color:#64748b;margin-top:1px">Best F1</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
          <div style="background:#0f172a;border-radius:7px;padding:6px 8px;text-align:center">
            <div style="font-size:12px;font-weight:700;color:#60a5fa" id="agent-best-model">—</div>
            <div style="font-size:9px;color:#64748b;margin-top:1px">Best Model</div>
          </div>
          <div style="background:#0f172a;border-radius:7px;padding:6px 8px;text-align:center">
            <div style="font-size:12px;font-weight:700;color:#fbbf24" id="agent-phase">—</div>
            <div style="font-size:9px;color:#64748b;margin-top:1px">Phase</div>
          </div>
          <div style="background:#0f172a;border-radius:7px;padding:6px 8px;text-align:center">
            <div style="font-size:12px;font-weight:700;color:#e2e8f0" id="agent-uptime">—</div>
            <div style="font-size:9px;color:#64748b;margin-top:1px">Uptime</div>
          </div>
        </div>
      </div>
      <!-- Reasoning + log stacked -->
      <div style="display:flex;flex-direction:column;gap:6px;min-width:0">
        <div>
          <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Current Reasoning</div>
          <div id="agent-reasoning" style="background:#0f172a;border-radius:6px;padding:8px 10px;font-size:11px;color:#a78bfa;font-family:monospace;white-space:pre-wrap;max-height:56px;overflow-y:auto">— waiting for agent —</div>
        </div>
        <div>
          <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Decision Log</div>
          <div id="agent-log-box" style="background:#0f172a;border-radius:6px;padding:8px 10px;font-size:11px;font-family:monospace;max-height:80px;overflow-y:auto;color:#cbd5e1">— no log yet —</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ROW 1: Status Overview -->
  <div class="card" style="margin-bottom:14px">
    <div class="card-title">Status Overview</div>
    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:10px">
      <span id="train-status-dot" style="font-size:26px;line-height:1">🟣</span>
      <div style="flex:1;min-width:200px">
        <div id="train-status-plain" style="font-size:14px;color:#cbd5e1;font-weight:500">Idle — configure and click Start</div>
        <div style="display:flex;align-items:center;gap:10px;margin-top:6px">
          <div style="flex:1;height:14px;background:#1e293b;border-radius:7px;overflow:hidden;position:relative">
            <div id="train-progress-fill" style="height:100%;background:linear-gradient(90deg,#7c3aed,#a78bfa);border-radius:7px;transition:width .4s;width:0%"></div>
            <span id="train-progress-pct" style="position:absolute;right:6px;top:50%;transform:translateY(-50%);font-size:10px;font-weight:700;color:#fff;pointer-events:none">0%</span>
          </div>
          <span id="train-status-big" class="status-badge status-idle" style="flex-shrink:0">Idle</span>
        </div>
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <div style="background:#0f1117;border:1px solid #2d3148;border-radius:8px;padding:10px 14px;flex:1;min-width:110px;text-align:center">
        <div id="stat-epoch" style="font-size:20px;font-weight:700;color:#a78bfa;line-height:1.2">—</div>
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-top:2px">Current Epoch</div>
      </div>
      <div style="background:#0f1117;border:1px solid #2d3148;border-radius:8px;padding:10px 14px;flex:1;min-width:110px;text-align:center">
        <div id="stat-best-f1" style="font-size:18px;font-weight:700;color:#a78bfa;line-height:1.2">—</div>
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-top:2px">Best F1 <span id="stat-f1-label" style="font-size:9px"></span></div>
      </div>
      <div style="background:#0f1117;border:1px solid #2d3148;border-radius:8px;padding:10px 14px;flex:1;min-width:110px;text-align:center">
        <div id="stat-lr" style="font-size:14px;font-weight:700;color:#a78bfa;line-height:1.2">—</div>
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-top:2px">Current LR</div>
      </div>
      <div style="background:#0f1117;border:1px solid #2d3148;border-radius:8px;padding:10px 14px;flex:1;min-width:110px;text-align:center">
        <div id="stat-eta" style="font-size:14px;font-weight:700;color:#a78bfa;line-height:1.2">—</div>
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-top:2px">ETA</div>
      </div>
    </div>
  </div>

  <!-- ROW 2: Config + GPU -->
  <div class="two-col">
    <div class="card">
      <div class="card-title">Configuration &amp; Controls</div>
      <div class="flex-row">
        <div class="flex-col">
          <label>Model</label>
          <select id="cfg-model">
            <option>efficientnet_b0</option>
            <option>efficientnet_b3</option>
            <option>resnet34</option>
            <option>resnet50</option>
            <option>densenet121</option>
          </select>
        </div>
        <div class="flex-col">
          <label>Image Size</label>
          <select id="cfg-imgsize">
            <option value="224">224</option>
            <option value="384">384</option>
            <option value="512">512</option>
          </select>
        </div>
      </div>
      <div class="flex-row" style="margin-top:8px">
        <div class="flex-col">
          <label>Learning Rate</label>
          <input type="number" id="cfg-lr" value="0.0001" step="0.00001"/>
        </div>
        <div class="flex-col">
          <label>Batch Size</label>
          <input type="number" id="cfg-bs" value="32"/>
        </div>
      </div>
      <div class="flex-row" style="margin-top:8px">
        <div class="flex-col">
          <label>Epochs</label>
          <input type="number" id="cfg-epochs" value="30"/>
        </div>
        <div class="flex-col">
          <label>Val Split</label>
          <input type="number" id="cfg-val" value="0.1" step="0.05"/>
        </div>
      </div>
      <div class="flex-row" style="margin-top:8px">
        <div class="flex-col">
          <label>Num Workers</label>
          <input type="number" id="cfg-workers" value="4"/>
        </div>
        <div class="flex-col">
          <label style="display:flex;align-items:center;gap:6px;margin-top:18px;cursor:pointer">
            <input type="checkbox" id="cfg-pretrained" checked style="accent-color:#a78bfa"/>
            Pretrained
          </label>
        </div>
      </div>
      <div class="btn-row" style="margin-top:12px">
        <button class="btn btn-primary" onclick="startTraining()">Start</button>
        <button class="btn btn-danger"  onclick="stopTraining()">Stop</button>
        <button class="btn btn-secondary" onclick="downloadCSV()">Download Metrics CSV</button>
      </div>
      <div class="separator"></div>
      <div style="display:flex;gap:8px;align-items:flex-end;margin-top:4px">
        <div style="flex:1"><label>Live LR Update</label><input type="number" id="live-lr" step="0.00001" placeholder="e.g. 0.00005"/></div>
        <button class="btn btn-secondary btn-sm" onclick="updateLR()">Update LR</button>
      </div>
      <div style="font-size:11px;color:#64748b;margin-top:8px">
        Model: <span id="status-model">—</span> &nbsp;|&nbsp; Device: <span id="status-device">—</span>
        &nbsp;|&nbsp; <span id="train-status-detail"></span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">GPU Monitor</div>
      <div id="gpu-name" style="font-size:13px;color:#a78bfa;font-weight:600;margin-bottom:10px">Detecting GPU...</div>
      <div id="gpu-no-gpu" style="display:none;background:#7c3aed22;border:1px solid #7c3aed55;border-radius:6px;padding:8px;color:#f59e0b;font-size:12px;margin-bottom:8px">
        No GPU detected — running in CPU Mode
      </div>
      <div id="gpu-stats">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px">
          <span style="width:90px;color:#94a3b8;flex-shrink:0">GPU Utilization</span>
          <div style="flex:1;height:10px;background:#1e293b;border-radius:5px;overflow:hidden"><div id="gpu-util-bar" style="height:100%;border-radius:5px;transition:width .4s;background:#a78bfa;width:0%"></div></div>
          <span style="width:70px;text-align:right;color:#cbd5e1;flex-shrink:0;font-size:11px" id="gpu-util-val">—</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px">
          <span style="width:90px;color:#94a3b8;flex-shrink:0">Memory</span>
          <div style="flex:1;height:10px;background:#1e293b;border-radius:5px;overflow:hidden"><div id="gpu-mem-bar" style="height:100%;border-radius:5px;transition:width .4s;background:#60a5fa;width:0%"></div></div>
          <span style="width:70px;text-align:right;color:#cbd5e1;flex-shrink:0;font-size:11px" id="gpu-mem-val">—</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px">
          <span style="width:90px;color:#94a3b8;flex-shrink:0">Temperature</span>
          <div style="flex:1;height:10px;background:#1e293b;border-radius:5px;overflow:hidden"><div id="gpu-temp-bar" style="height:100%;border-radius:5px;transition:width .4s;background:#f87171;width:0%"></div></div>
          <span style="width:70px;text-align:right;color:#cbd5e1;flex-shrink:0;font-size:11px" id="gpu-temp-val">—</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px">
          <span style="width:90px;color:#94a3b8;flex-shrink:0">Power</span>
          <div style="flex:1;height:10px;background:#1e293b;border-radius:5px;overflow:hidden"><div id="gpu-pwr-bar" style="height:100%;border-radius:5px;transition:width .4s;background:#fbbf24;width:0%"></div></div>
          <span style="width:70px;text-align:right;color:#cbd5e1;flex-shrink:0;font-size:11px" id="gpu-pwr-val">—</span>
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:6px">Process: <span id="gpu-proc">—</span></div>
      </div>
    </div>
  </div>

  <!-- ROW 3: Loss + F1 charts -->
  <div class="two-col">
    <div class="card">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
        <span class="card-title" style="margin:0">Loss Curves</span>
        <div style="flex:1;display:flex;justify-content:flex-end;align-items:center;gap:6px;font-size:12px;color:#94a3b8">
          Show last <select id="loss-window" onchange="refreshTrainMetrics()" style="width:75px;display:inline-block">
            <option value="10">10</option><option value="25">25</option><option value="0" selected>All</option>
          </select> epochs
        </div>
      </div>
      <div class="chart-container"><canvas id="loss-chart"></canvas></div>
    </div>
    <div class="card">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
        <span class="card-title" style="margin:0">Macro F1</span>
        <div style="flex:1;display:flex;justify-content:flex-end;align-items:center;gap:6px;font-size:12px;color:#94a3b8">
          Show last <select id="f1-window" onchange="refreshTrainMetrics()" style="width:75px;display:inline-block">
            <option value="10">10</option><option value="25">25</option><option value="0" selected>All</option>
          </select> epochs
        </div>
      </div>
      <div class="chart-container"><canvas id="f1-chart"></canvas></div>
    </div>
  </div>

  <!-- ROW 4: Per-Class F1 bar (last epoch, sorted) -->
  <div class="card">
    <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
      <span>Per-Class F1 — Last Epoch (sorted descending)</span>
      <button class="btn btn-secondary btn-sm" id="perclass-toggle-btn" onclick="togglePerClassAll()">Show All 28</button>
    </div>
    <div id="perclass-chart-wrap" style="height:340px"><canvas id="perclass-chart"></canvas></div>
  </div>

  <!-- ROW 5: Per-Class F1 Heatmap -->
  <div class="card" id="heatmap-card" style="display:none">
    <div class="card-title">Per-Class F1 Heatmap (epochs × classes)</div>
    <div style="overflow-x:auto;max-height:340px;overflow-y:auto">
      <div id="f1-heatmap"></div>
    </div>
  </div>

  <!-- ROW 6: Full Metrics Table -->
  <div class="card">
    <div class="card-title">Full Metrics Table</div>
    <div style="overflow-x:auto;max-height:320px;overflow-y:auto">
      <table class="metrics-tbl" id="metrics-table">
        <thead><tr>
          <th onclick="sortMetrics('epoch')">Epoch</th>
          <th onclick="sortMetrics('train_loss')">Train Loss</th>
          <th onclick="sortMetrics('val_loss')">Val Loss</th>
          <th onclick="sortMetrics('macro_f1')">Macro F1</th>
          <th onclick="sortMetrics('lr')">LR</th>
          <th onclick="sortMetrics('ts')">Time</th>
        </tr></thead>
        <tbody id="metrics-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ROW 7: Training Log -->
  <div class="card">
    <div class="card-title">Training Log <span style="font-size:10px;color:#64748b">(last 40 lines)</span></div>
    <div class="terminal-box" id="train-log-box">— no log yet —</div>
  </div>

  <!-- Previous Runs -->
  <div class="card" id="history-card">
    <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
      <span>Previous Runs</span>
      <div style="display:flex;gap:8px;align-items:center">
        <span id="viewing-run-badge" style="display:none;font-size:11px;padding:2px 8px;border-radius:10px;background:#7c3aed33;color:#a78bfa">Viewing archived run</span>
        <button class="btn btn-secondary btn-sm" onclick="loadLiveRun()">&#8635; Back to Live</button>
        <button class="btn btn-secondary btn-sm" onclick="loadRunsList()">&#8635; Refresh</button>
      </div>
    </div>
    <div id="runs-list" style="max-height:260px;overflow-y:auto;margin-top:8px">
      <div style="color:#64748b;font-size:13px">Loading runs...</div>
    </div>
  </div>

</div>

<!-- ===== TAB 4: Inference ===== -->
<div class="tab-pane" id="tab-inference">
  <div class="two-col">
    <div class="card">
      <div class="card-title">Input</div>
      <div style="display:flex;gap:8px;margin-bottom:8px">
        <input type="text" id="inf-id" placeholder="Image ID" style="flex:1"/>
        <button class="btn btn-secondary btn-sm" onclick="randomInfImage()">Random</button>
      </div>
      <div style="margin-bottom:10px">
        <label>Checkpoint</label>
        <select id="inf-ckpt" style="width:100%">
          <option value="">-- select --</option>
        </select>
      </div>
      <button class="btn btn-primary" onclick="runInference()">Run Inference</button>
      <div id="inf-status" style="margin-top:8px;font-size:12px;color:#64748b"></div>
    </div>
    <div class="card">
      <div class="card-title">Image Preview</div>
      <div id="inf-preview" style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start">
        <div style="color:#64748b;font-size:12px">No image loaded</div>
      </div>
    </div>
  </div>
  <div class="card" id="inf-results" style="display:none">
    <div class="card-title">Predictions vs Ground Truth
      <span style="font-size:11px;color:#64748b;margin-left:8px">
        <span style="color:#22c55e">correct</span> /
        <span style="color:#ef4444">missed GT</span> /
        <span style="color:#f59e0b">false positive</span>
      </span>
    </div>
    <div id="inf-pred-bars"></div>
  </div>
</div>

<!-- ===== TAB 5: Annotate ===== -->
<div class="tab-pane" id="tab-annotate">
  <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap">
    <span id="annot-progress" style="font-size:13px;color:#94a3b8">Loading...</span>
    <label style="display:flex;align-items:center;gap:5px;cursor:pointer;margin:0">
      <input type="checkbox" id="show-flagged" onchange="loadAnnotateNext()" style="accent-color:#a78bfa"/>
      Show only flagged
    </label>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Image <span id="annot-img-id" style="color:#a78bfa;font-size:11px;font-weight:400"></span></div>
      <div id="annot-preview" style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start"></div>
      <div style="margin-top:10px">
        <label>Note</label>
        <input type="text" id="annot-note" placeholder="optional note..."/>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Labels</div>
      <div class="checkbox-grid" id="annot-checkboxes"></div>
      <div class="btn-row" style="margin-top:14px">
        <button class="btn btn-success"   onclick="saveAnnotation()">Save</button>
        <button class="btn btn-secondary" onclick="loadAnnotateNext()">Skip</button>
        <button class="btn btn-warning"   onclick="flagAnnotation()">Flag</button>
      </div>
    </div>
  </div>
</div>

<!-- ===== TAB 6: Post Analysis ===== -->
<div class="tab-pane" id="tab-postanalysis">

  <!-- Control bar -->
  <div style="display:flex;align-items:center;gap:10px;padding:8px 14px;background:#1a1d27;border:1px solid #2d3148;border-radius:8px;margin-bottom:14px;flex-wrap:wrap">
    <span style="font-size:12px;color:#94a3b8;font-weight:600">Protein Atlas — Biological Analysis</span>
    <button class="btn btn-secondary btn-sm" onclick="initPostAnalysis()">&#8635; Refresh</button>
    <span style="font-size:11px;color:#475569;margin-left:auto" id="pa-last-refresh"></span>
  </div>

  <!-- ROW 1: Atlas Curation Readiness Banner -->
  <div class="card" style="margin-bottom:14px;padding:18px 20px">
    <div style="font-size:13px;color:#94a3b8;font-weight:600;letter-spacing:.04em;margin-bottom:14px;text-transform:uppercase">Atlas Annotation Readiness — <span id="pa-best-model-label" style="color:#a78bfa;text-transform:none;font-weight:700"></span></div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px">
      <div style="background:#052e16;border:1px solid #16a34a44;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:28px;font-weight:800;color:#4ade80" id="pa-ready-count">—</div>
        <div style="font-size:10px;color:#16a34a;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Auto-Annotate Ready</div>
        <div style="font-size:10px;color:#475569;margin-top:3px">F1 &gt; 0.50 · high confidence</div>
      </div>
      <div style="background:#1c1203;border:1px solid #ca8a0444;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:28px;font-weight:800;color:#fbbf24" id="pa-review-count">—</div>
        <div style="font-size:10px;color:#ca8a04;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Needs Expert Review</div>
        <div style="font-size:10px;color:#475569;margin-top:3px">F1 0.15–0.50 · use as lead</div>
      </div>
      <div style="background:#1f0a0a;border:1px solid #dc262644;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:28px;font-weight:800;color:#f87171" id="pa-weak-count">—</div>
        <div style="font-size:10px;color:#dc2626;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Insufficient Confidence</div>
        <div style="font-size:10px;color:#475569;margin-top:3px">F1 &lt; 0.15 · model fails</div>
      </div>
      <div style="background:#0f0a1f;border:1px solid #7c3aed44;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:28px;font-weight:800;color:#a78bfa" id="pa-starved-count">—</div>
        <div style="font-size:10px;color:#7c3aed;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Data-Starved (&lt;100 samples)</div>
        <div style="font-size:10px;color:#475569;margin-top:3px">not a model failure — needs data</div>
      </div>
    </div>
    <!-- Readiness bar -->
    <div style="font-size:10px;color:#475569;margin-bottom:5px">Annotation confidence across all 28 compartments</div>
    <div style="height:16px;border-radius:8px;overflow:hidden;display:flex;background:#1e2235" id="pa-readiness-bar"></div>
    <div style="display:flex;gap:16px;margin-top:6px;font-size:10px">
      <span style="color:#4ade80">&#9632; Auto-annotate</span>
      <span style="color:#fbbf24">&#9632; Expert review</span>
      <span style="color:#f87171">&#9632; Insufficient</span>
      <span style="color:#a78bfa">&#9632; Data-starved</span>
    </div>
  </div>

  <!-- ROW 2: Compartment group performance + Data scarcity scatter -->
  <div class="two-col" style="margin-bottom:14px">
    <div class="card">
      <div class="card-title">Performance by Subcellular Compartment Group</div>
      <div style="position:relative;height:280px"><canvas id="pa-group-chart"></canvas></div>
      <div style="font-size:10px;color:#475569;margin-top:8px">Average F1 at best epoch, grouped by organelle category. Gray dashed = 0.50 annotation threshold.</div>
    </div>
    <div class="card">
      <div class="card-title">Data Scarcity vs Detection Confidence
        <span style="font-size:10px;color:#475569;font-weight:400;margin-left:6px">each point = one compartment</span>
      </div>
      <div style="position:relative;height:260px"><canvas id="pa-scarcity-chart"></canvas></div>
      <div style="font-size:10px;color:#475569;margin-top:8px">Poor F1 on rare structures is primarily a <em>data problem</em>, not a model failure — the slope shows strong correlation with training set size.</div>
    </div>
  </div>

  <!-- ROW 3: Per-compartment biological assessment table -->
  <div class="card" style="margin-bottom:14px">
    <div class="card-title">Per-Compartment Biological Assessment</div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="border-bottom:1px solid #2d3148;color:#64748b;text-transform:uppercase;font-size:10px;letter-spacing:.04em">
            <th style="padding:6px 10px;text-align:left">Compartment</th>
            <th style="padding:6px 8px;text-align:left">Group</th>
            <th style="padding:6px 8px;text-align:center">Training<br>Samples</th>
            <th style="padding:6px 8px;text-align:center">Model F1</th>
            <th style="padding:6px 8px;text-align:left">Biological Challenge</th>
            <th style="padding:6px 8px;text-align:center">Annotation<br>Status</th>
          </tr>
        </thead>
        <tbody id="pa-bio-tbody">
          <tr><td colspan="6" style="color:#64748b;padding:20px;text-align:center">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ROW 4: Visual confounds + Learning curve -->
  <div class="two-col" style="margin-bottom:14px">
    <div class="card">
      <div class="card-title">Visually Similar Structures — Known Confounds</div>
      <div id="pa-confounds" style="font-size:12px"></div>
      <div style="font-size:10px;color:#475569;margin-top:10px">These compartments share morphological features under fluorescence microscopy, making them inherently difficult to separate even for human annotators.</div>
    </div>
    <div class="card">
      <div class="card-title">F1 Learning Curve — Best Model
        <span id="pa-curve-label" style="font-size:10px;color:#475569;font-weight:400;margin-left:6px"></span>
      </div>
      <div style="position:relative;height:220px"><canvas id="pa-f1-curves-chart"></canvas></div>
      <div id="pa-curve-notes" style="font-size:11px;color:#94a3b8;margin-top:8px;line-height:1.7"></div>
    </div>
  </div>

  <!-- ROW 5: Multi-label analysis -->
  <div class="card" style="margin-bottom:14px">
    <div class="card-title">Multi-Label Co-localization — 51.3% of proteins localize to multiple compartments</div>
    <div class="two-col" style="gap:20px">
      <div>
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px">Most frequent co-localizations in training data</div>
        <div id="pa-coloc-list" style="font-size:12px;color:#cbd5e1;line-height:2.1"></div>
      </div>
      <div>
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px">Multi-label challenge for the model</div>
        <div id="pa-multilabel-notes" style="font-size:12px;color:#cbd5e1;line-height:1.8"></div>
      </div>
    </div>
  </div>

  <!-- ROW 6: Scientific interpretation -->
  <div class="card">
    <div class="card-title">Scientific Interpretation &amp; Next Steps</div>
    <div id="pa-scientific" style="font-size:12px;color:#cbd5e1;line-height:1.9"></div>
  </div>

</div>

</div><!-- /content -->
</div><!-- /main -->
</div><!-- /app -->

<script>
// =====================================================
// Constants
// =====================================================
const CLASS_NAMES = {
  0:'Nucleoplasm',1:'Nuclear membrane',2:'Nucleoli',3:'Nucleoli fibrillar center',
  4:'Nuclear speckles',5:'Nuclear bodies',6:'Endoplasmic reticulum',7:'Golgi apparatus',
  8:'Peroxisomes',9:'Endosomes',10:'Lysosomes',11:'Intermediate filaments',
  12:'Actin filaments',13:'Focal adhesion sites',14:'Microtubules',15:'Microtubule ends',
  16:'Cytokinetic bridge',17:'Mitotic spindle',18:'Microtubule organizing center',
  19:'Centrosome',20:'Lipid droplets',21:'Plasma membrane',22:'Cell junctions',
  23:'Mitochondria',24:'Aggresome',25:'Cytosol',26:'Cytoplasmic bodies',27:'Rods & rings'
};

// =====================================================
// State
// =====================================================
let selectedClasses = new Set();
let classCounts = {};
let lossChart=null, f1Chart=null, perClassChart=null, distChart=null, histChart=null;
let trainRefreshTimer = null;
let agentRefreshTimer = null;
let currentAnnotId = null;
let statsData = {};
let analysisInited = false;

// =====================================================
// Utilities
// =====================================================
function toast(msg, duration=2800) {
  const t = document.createElement('div');
  t.className='toast'; t.textContent=msg;
  document.body.appendChild(t);
  setTimeout(()=>t.remove(), duration);
}

function switchTab(name, btn) {
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if (btn) btn.classList.add('active');
  const sb = document.getElementById('sidebar-explorer');
  sb.style.display = (name==='explorer') ? 'block' : 'none';
  if (name==='analysis')  initAnalysis();
  if (name==='train')     initTrainTab();
  if (name==='inference') loadCheckpoints();
  if (name==='annotate')     { buildAnnotCheckboxes(); loadAnnotateNext(); loadAnnotateStats(); }
  if (name==='postanalysis') initPostAnalysis();
}

async function apiFetch(url, opts={}) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || r.statusText);
  }
  return r.json();
}

// =====================================================
// Init
// =====================================================
async function init() {
  try {
    statsData = await apiFetch('/api/stats');
    document.getElementById('ds-stats').textContent =
      `${statsData.total_images.toLocaleString()} images · ${statsData.total_labels.toLocaleString()} labels · 28 classes`;
    classCounts = statsData.class_counts || {};
    buildClassSidebar();
    loadImages();
  } catch(e) {
    console.error('init error', e);
    document.getElementById('ds-stats').textContent = 'Error loading stats';
  }
}

// =====================================================
// TAB 1 — Explorer
// =====================================================
function buildClassSidebar() {
  const maxCount = Math.max(...Object.values(classCounts), 1);
  const list = document.getElementById('class-list');
  list.innerHTML = '';
  for (let i = 0; i < 28; i++) {
    const count = classCounts[i] || 0;
    const pct = Math.round(count / maxCount * 100);
    const div = document.createElement('div');
    div.className = 'class-item';
    div.dataset.cls = i;
    div.innerHTML = `
      <span class="class-label" title="${CLASS_NAMES[i]}">${CLASS_NAMES[i]}</span>
      <div class="class-bar-bg"><div class="class-bar-fill" style="width:${pct}%"></div></div>
      <span class="class-count">${count}</span>`;
    div.onclick = () => toggleClass(i, div);
    list.appendChild(div);
  }
}

function toggleClass(cls, el) {
  if (selectedClasses.has(cls)) {
    selectedClasses.delete(cls);
    el.classList.remove('selected');
  } else {
    selectedClasses.add(cls);
    el.classList.add('selected');
  }
  loadImages();
}

function clearClassFilter() {
  selectedClasses.clear();
  document.querySelectorAll('.class-item').forEach(el=>el.classList.remove('selected'));
  loadImages();
}

async function loadImages() {
  const n = parseInt(document.getElementById('grid-size').value);
  const split = document.getElementById('split-sel').value;
  const mode  = document.getElementById('class-mode').value;
  const grid  = document.getElementById('img-grid');
  grid.innerHTML = '<div style="color:#64748b;padding:20px;grid-column:1/-1">Loading... <span class="spinner"></span></div>';

  const colClass = n<=4 ? 'grid-c4' : n<=6 ? 'grid-c3' : n<=9 ? 'grid-c3' : 'grid-c4';
  grid.className = 'img-grid ' + colClass;

  let url = `/api/sample?n=${n}&split=${split}&mode=${mode}`;
  if (selectedClasses.size > 0) url += `&classes=${[...selectedClasses].join(',')}`;

  try {
    const data = await apiFetch(url);
    document.getElementById('explorer-count').textContent =
      `Showing ${data.items.length} of ${data.total_matching.toLocaleString()} matching`;
    grid.innerHTML = '';
    for (const item of data.items) {
      grid.appendChild(buildImageCard(item));
    }
    if (data.items.length === 0) {
      grid.innerHTML = '<div style="color:#64748b;padding:20px;grid-column:1/-1">No images match the current filter.</div>';
    }
  } catch(e) {
    grid.innerHTML = `<div style="color:#ef4444;padding:20px;grid-column:1/-1">Error: ${e.message}</div>`;
  }
}

function buildImageCard(item) {
  const div = document.createElement('div');
  div.className = 'img-card';
  const shortId = item.id.length > 12 ? item.id.slice(0,12)+'...' : item.id;
  const chips = item.labels.slice(0, 5).map(l =>
    `<span class="lbl-chip">${CLASS_NAMES[l] || l}</span>`).join('');
  const extra = item.labels.length > 5 ? `<span class="lbl-chip">+${item.labels.length-5}</span>` : '';
  div.innerHTML = `
    <img class="img-composite" src="/api/image/composite?id=${encodeURIComponent(item.id)}" loading="lazy"
         alt="composite" onerror="this.style.background='#1e293b'"/>
    <div class="img-channels">
      <img src="/api/image/channel?id=${encodeURIComponent(item.id)}&ch=blue"   loading="lazy" title="Blue (nucleus)"/>
      <img src="/api/image/channel?id=${encodeURIComponent(item.id)}&ch=green"  loading="lazy" title="Green (protein)"/>
      <img src="/api/image/channel?id=${encodeURIComponent(item.id)}&ch=red"    loading="lazy" title="Red (microtubules)"/>
      <img src="/api/image/channel?id=${encodeURIComponent(item.id)}&ch=yellow" loading="lazy" title="Yellow (ER)"/>
    </div>
    <div class="img-meta">
      <div class="img-id" title="${item.id}">${shortId}</div>
      <div class="img-labels">${chips}${extra}</div>
    </div>`;
  return div;
}

// =====================================================
// TAB 2 — Analysis  (professional rewrite)
// =====================================================
let _distLogScale = false;
let _coocLogScale = false;
let _coocMatrix = null;
let _qualityChart = null;
let _cardinalityChart = null;
let _imbalanceData = null;

async function initAnalysis() {
  if (analysisInited) return;
  analysisInited = true;
  buildOverviewCards();
  buildDistChart();
  buildCardinalityChart();
  buildPairsTable();
  buildCooccurrenceTable();
  loadHistogram();
  loadImbalanceTable();
}

// ── ROW 1: Overview Stat Cards ───────────────────────────────────────────────
async function buildOverviewCards() {
  try {
    const [statsR, annR] = await Promise.all([
      apiFetch('/api/stats'),
      apiFetch('/api/annotate/stats').catch(()=>({verified:0,total:0,flagged:0}))
    ]);
    const rows = statsR.total_images || 0;
    const labels = statsR.total_labels || 0;
    const cc = statsR.class_counts || {};
    const avgL = rows > 0 ? (labels / rows).toFixed(2) : '—';
    // multi-label % requires per-image label count — approximate from class_counts
    // we can fetch from cardinality endpoint instead; for now use annotation
    const multiPct = statsR.multi_label_pct != null
      ? (statsR.multi_label_pct * 100).toFixed(1) + '%'
      : '—';
    document.getElementById('asc-total-images').textContent = rows.toLocaleString();
    document.getElementById('asc-total-sub').textContent = labels.toLocaleString() + ' total label assignments';
    document.getElementById('asc-multilabel').textContent = multiPct;
    document.getElementById('asc-multilabel-sub').textContent = 'images with 2+ classes';
    document.getElementById('asc-avg-labels').textContent = avgL;
    document.getElementById('asc-avg-labels-sub').textContent = 'mean labels per image';
    document.getElementById('asc-annotated').textContent = (annR.verified || 0).toLocaleString();
    document.getElementById('asc-annotated-sub').textContent =
      `of ${(annR.total||rows).toLocaleString()} · ${annR.flagged||0} flagged`;
    // fill multi-label from cardinality
    apiFetch('/api/analysis/cardinality').then(d=>{
      const counts = d.counts || {};
      const total = Object.values(counts).reduce((a,b)=>a+b,0);
      const single = counts['1'] || counts[1] || 0;
      const multi = total - single;
      if (total > 0) {
        document.getElementById('asc-multilabel').textContent = (multi/total*100).toFixed(1)+'%';
        document.getElementById('asc-multilabel-sub').textContent =
          multi.toLocaleString() + ' of ' + total.toLocaleString() + ' images';
      }
    }).catch(()=>{});
  } catch(e) { console.error('overview cards', e); }
}

// ── ROW 2: Class Distribution ─────────────────────────────────────────────────
function toggleDistLog() {
  _distLogScale = !_distLogScale;
  const btn = document.getElementById('dist-log-btn');
  btn.classList.toggle('active', _distLogScale);
  btn.textContent = _distLogScale ? 'Linear Scale' : 'Log Scale';
  buildDistChart();
}

function buildDistChart() {
  const ctx = document.getElementById('dist-chart').getContext('2d');
  // Sort descending by count
  const pairs = Array.from({length:28}, (_,i) => ({i, count: classCounts[i]||0, name: CLASS_NAMES[i]}));
  pairs.sort((a,b)=>b.count-a.count);
  const total = pairs.reduce((s,p)=>s+p.count,0);

  const bgColors = pairs.map((_,idx)=>{
    const t = idx / 27;
    // gradient: purple(167,139,250) → blue(96,165,250) → gray(100,116,139)
    let r,g,b;
    if (t < 0.5) {
      const u = t*2;
      r = Math.round(167+(96-167)*u); g = Math.round(139+(165-139)*u); b = Math.round(250+0*u);
    } else {
      const u = (t-0.5)*2;
      r = Math.round(96+(100-96)*u); g = Math.round(165+(116-165)*u); b = Math.round(250+(139-250)*u);
    }
    return `rgba(${r},${g},${b},0.8)`;
  });

  if (distChart) distChart.destroy();
  distChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: pairs.map(p=>p.name),
      datasets:[{
        label:'Count', data: pairs.map(p=>p.count),
        backgroundColor: bgColors, borderWidth:0,
        borderRadius: 3,
      }]
    },
    options:{
      indexAxis:'y',
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{
          callbacks:{
            label: c => {
              const pct = total>0?(c.raw/total*100).toFixed(1):'0';
              return ` ${c.raw.toLocaleString()} images (${pct}%)`;
            }
          }
        }
      },
      scales:{
        x:{
          type: _distLogScale ? 'logarithmic' : 'linear',
          ticks:{color:'#64748b'}, grid:{color:'#1e293b'}
        },
        y:{ticks:{color:'#e2e8f0',font:{size:10}}, grid:{color:'#1e293b'}}
      }
    },
    plugins:[{
      id:'barLabels',
      afterDatasetsDraw(chart){
        const {ctx:c2, scales:{x,y}} = chart;
        chart.data.datasets[0].data.forEach((val,i)=>{
          const pct = total>0?(val/total*100).toFixed(1):'0';
          const xPos = x.getPixelForValue(val);
          const yPos = y.getPixelForValue(i);
          c2.save();
          c2.fillStyle='#94a3b8';
          c2.font='10px sans-serif';
          c2.textAlign='left';
          c2.fillText(`${val.toLocaleString()} (${pct}%)`, xPos+4, yPos+4);
          c2.restore();
        });
      }
    }]
  });
}

// ── ROW 3a: Label Cardinality ─────────────────────────────────────────────────
async function buildCardinalityChart() {
  try {
    const data = await apiFetch('/api/analysis/cardinality');
    const counts = data.counts || {};
    const labels = ['1','2','3','4','5+'];
    const vals = labels.map(k => counts[k] || counts[parseInt(k)] || 0);
    const total = vals.reduce((a,b)=>a+b,0);
    const ctx = document.getElementById('cardinality-chart').getContext('2d');
    if (_cardinalityChart) _cardinalityChart.destroy();
    _cardinalityChart = new Chart(ctx, {
      type:'bar',
      data:{
        labels: labels.map(l=>`${l} label${l==='1'?'':'s'}`),
        datasets:[{
          label:'Images',
          data: vals,
          backgroundColor:['#a78bfa','#818cf8','#60a5fa','#38bdf8','#2dd4bf'],
          borderWidth:0, borderRadius:4
        }]
      },
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{
          legend:{display:false},
          tooltip:{callbacks:{label:c=>`${c.raw.toLocaleString()} images (${total>0?(c.raw/total*100).toFixed(1):0}%)`}}
        },
        scales:{
          x:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},
          y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},title:{display:true,text:'Image count',color:'#64748b'}}
        }
      }
    });
  } catch(e) { console.error('cardinality', e); }
}

// ── ROW 3b: Top Co-occurring Pairs ────────────────────────────────────────────
async function buildPairsTable() {
  const wrap = document.getElementById('pairs-table-wrap');
  try {
    const data = await apiFetch('/api/analysis/pairs');
    const pairs = (data.pairs || []).slice(0,15);
    if (!pairs.length) { wrap.innerHTML='<div style="color:#64748b;font-size:12px">No data</div>'; return; }
    let html = `<table><thead><tr>
      <th style="min-width:120px">Class A</th>
      <th style="min-width:120px">Class B</th>
      <th style="text-align:right">Count</th>
      <th style="text-align:right">% Images</th>
    </tr></thead><tbody>`;
    for (const p of pairs) {
      html += `<tr>
        <td style="font-size:11px">${p.a_name}</td>
        <td style="font-size:11px">${p.b_name}</td>
        <td style="text-align:right;color:#a78bfa;font-weight:600">${p.count.toLocaleString()}</td>
        <td style="text-align:right;color:#64748b">${p.pct.toFixed(1)}%</td>
      </tr>`;
    }
    html += '</tbody></table>';
    wrap.innerHTML = html;
  } catch(e) {
    wrap.innerHTML = `<span style="color:#ef4444;font-size:12px">${e.message}</span>`;
  }
}

// ── ROW 4: Co-occurrence Heatmap (HTML table) ─────────────────────────────────
async function buildCooccurrenceTable() {
  const wrap = document.getElementById('cooc-heatmap-wrap');
  wrap.innerHTML = 'Loading co-occurrence... <span class="spinner"></span>';
  try {
    const data = await apiFetch('/api/analysis/cooccurrence');
    _coocMatrix = data.matrix;
    renderCoocTable();
  } catch(e) {
    wrap.innerHTML = `<span style="color:#ef4444;font-size:12px">${e.message}</span>`;
  }
}

function toggleCoocLog() {
  _coocLogScale = !_coocLogScale;
  const btn = document.getElementById('cooc-log-btn');
  btn.classList.toggle('active', _coocLogScale);
  btn.textContent = _coocLogScale ? 'Linear Scale' : 'Log Scale';
  if (_coocMatrix) renderCoocTable();
}

function renderCoocTable() {
  if (!_coocMatrix) return;
  const wrap = document.getElementById('cooc-heatmap-wrap');
  const N = 28;
  const matrix = _coocMatrix;
  // find max off-diagonal
  let maxVal = 1;
  for (let i=0;i<N;i++) for (let j=0;j<N;j++) if(i!==j) maxVal = Math.max(maxVal, matrix[i][j]);
  const logMax = Math.log1p(maxVal);

  let html = '<table class="cooc-table"><thead><tr><th style="min-width:90px"></th>';
  for (let j=0;j<N;j++) {
    html += `<th style="writing-mode:vertical-rl;transform:rotate(180deg);height:90px;vertical-align:bottom;padding:2px 1px;font-size:9px;color:#94a3b8;cursor:default" title="${CLASS_NAMES[j]}">${CLASS_NAMES[j]}</th>`;
  }
  html += '</tr></thead><tbody>';
  for (let i=0;i<N;i++) {
    html += `<tr><td class="cooc-row-hdr" title="${CLASS_NAMES[i]}">${CLASS_NAMES[i]}</td>`;
    for (let j=0;j<N;j++) {
      const raw = matrix[i][j];
      let v;
      if (i===j) {
        v = -1; // diagonal marker
      } else {
        v = _coocLogScale ? Math.log1p(raw)/logMax : raw/maxVal;
      }
      let bg, fg='transparent';
      if (i===j) {
        bg = '#2d3148';
      } else if (v <= 0) {
        bg = '#0f1117';
      } else {
        const r = Math.round(167*v), g2 = Math.round(139*v), b = Math.round(250*v);
        bg = `rgb(${r},${g2},${b})`;
        if (v > 0.6) fg = '#fff';
      }
      const title = i===j ? CLASS_NAMES[i] : `${CLASS_NAMES[i]} × ${CLASS_NAMES[j]}: ${raw.toLocaleString()} images`;
      html += `<td style="background:${bg};width:20px;height:20px" title="${title}"></td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ── ROW 5a: Channel Histogram ─────────────────────────────────────────────────
async function randomHistImage() {
  try {
    const d = await apiFetch('/api/sample?n=1&split=train');
    if (d.items && d.items.length>0) document.getElementById('hist-id').value = d.items[0].id;
  } catch(e) {}
}

async function loadHistogram() {
  const id = document.getElementById('hist-id').value.trim() || null;
  let url = '/api/analysis/histogram';
  if (id) url += '?id=' + encodeURIComponent(id);
  try {
    const data = await apiFetch(url);
    if (data.id) document.getElementById('hist-id').value = data.id;
    const ctx = document.getElementById('hist-chart').getContext('2d');
    const bins = Array.from({length:256}, (_,i)=>i);
    if (histChart) histChart.destroy();

    const chColors = {blue:'#60a5fa',green:'#4ade80',red:'#f87171',yellow:'#fbbf24'};
    const datasets = ['blue','green','red','yellow'].map(ch=>({
      label: ch.charAt(0).toUpperCase()+ch.slice(1),
      data: data[ch],
      borderColor: chColors[ch], borderWidth:1.5, pointRadius:0, fill:false,
      tension:0.1
    }));

    histChart = new Chart(ctx, {
      type:'line',
      data:{labels:bins, datasets},
      options:{
        responsive:true, maintainAspectRatio:false, animation:false,
        plugins:{legend:{labels:{color:'#94a3b8',boxWidth:12,font:{size:11}}}},
        scales:{
          x:{ticks:{color:'#64748b',maxTicksLimit:16}, grid:{color:'#1e293b'},
             title:{display:true,text:'Pixel intensity (0–255)',color:'#64748b',font:{size:10}}},
          y:{ticks:{color:'#64748b'}, grid:{color:'#1e293b'},
             title:{display:true,text:'Pixel count',color:'#64748b',font:{size:10}}}
        }
      }
    });

    // Channel stats below chart
    const stats = data.stats || {};
    const statEl = document.getElementById('hist-stats');
    if (Object.keys(stats).length) {
      let html = '';
      for (const ch of ['blue','green','red','yellow']) {
        const s = stats[ch] || {};
        const satPct = s.sat_pct != null ? s.sat_pct.toFixed(1) : '—';
        html += `<div class="ch-stat-row" style="margin-bottom:3px">
          <span style="color:${chColors[ch]};min-width:50px;font-weight:600">${ch}</span>
          <span>mean: ${s.mean!=null?s.mean.toFixed(1):'—'}</span>
          <span>std: ${s.std!=null?s.std.toFixed(1):'—'}</span>
          <span>min: ${s.min!=null?s.min:'—'}</span>
          <span>max: ${s.max!=null?s.max:'—'}</span>
          <span>sat: ${satPct}%</span>
        </div>`;
      }
      statEl.innerHTML = html;
    }
  } catch(e) { console.error('histogram', e); }
}

// ── ROW 5b: Channel Stats (dataset sample) ────────────────────────────────────
async function loadChannelStats() {
  const wrap = document.getElementById('ch-stats-wrap');
  wrap.innerHTML = 'Sampling 300 images... <span class="spinner"></span>';
  try {
    const data = await apiFetch('/api/analysis/channel_stats');
    const ch = data.channels || {};
    const chColors = {blue:'#60a5fa',green:'#4ade80',red:'#f87171',yellow:'#fbbf24'};
    const chList = ['blue','green','red','yellow'];

    // Fixed-scale bar chart: mean ± std, Y always 0–255
    const container = document.createElement('div');
    container.style.cssText = 'position:relative;height:220px;width:100%';
    const canvas = document.createElement('canvas');
    wrap.innerHTML = '';
    wrap.appendChild(container);
    container.appendChild(canvas);

    const means = chList.map(c=>(ch[c]||{}).mean||0);
    const stds  = chList.map(c=>(ch[c]||{}).std||0);
    const q25s  = chList.map(c=>(ch[c]||{}).q25||0);
    const q75s  = chList.map(c=>(ch[c]||{}).q75||0);
    const q50s  = chList.map(c=>(ch[c]||{}).q50||0);
    new Chart(canvas.getContext('2d'), {
      type:'bar',
      data:{
        labels: ['DAPI (Blue)','Protein (Green)','Microtubules (Red)','ER (Yellow)'],
        datasets:[
          {
            label:'IQR (Q25–Q75)',
            data: chList.map((_,i)=>({x:i, y:q25s[i], y2:q75s[i]})),
            backgroundColor: chList.map(c=>chColors[c]+'33'),
            borderColor: chList.map(c=>chColors[c]+'55'),
            borderWidth:0, borderRadius:3,
            // render as floating bars
            data: chList.map((_,i)=>[q25s[i], q75s[i]]),
          },
          {
            label:'Mean intensity',
            data: means,
            backgroundColor: chList.map(c=>chColors[c]+'bb'),
            borderColor: chList.map(c=>chColors[c]),
            borderWidth:2, borderRadius:4,
          }
        ]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{
            display:true,
            labels:{color:'#94a3b8',font:{size:11},
              generateLabels: chart => [
                {text:'Mean', fillStyle:'#a78bfa99', strokeStyle:'#a78bfa', lineWidth:2},
                {text:'IQR (Q25–Q75)', fillStyle:'#a78bfa33', strokeStyle:'#a78bfa55', lineWidth:1},
              ]
            }
          },
          tooltip:{callbacks:{
            label: ctx => {
              const i = ctx.dataIndex;
              if (ctx.datasetIndex===0) return `IQR: ${q25s[i].toFixed(1)} – ${q75s[i].toFixed(1)}`;
              return `Mean: ${means[i].toFixed(1)}  Median: ${q50s[i].toFixed(1)}  Std: ±${stds[i].toFixed(1)}`;
            }
          }}
        },
        scales:{
          x:{ticks:{color:'#94a3b8', font:{size:11}}, grid:{color:'#1e293b'}},
          y:{
            min:0, max:255,
            ticks:{color:'#64748b', stepSize:51,
              callback: v => v === 0 ? '0 (black)' : v === 255 ? '255 (white)' : v
            },
            grid:{color:'#1e293b'},
            title:{display:true, text:'Pixel Intensity (0–255)', color:'#64748b', font:{size:11}}
          }
        }
      },
      plugins:[{
        id:'errorbars',
        afterDatasetsDraw(chart){
          const {ctx:c2, scales:{x,y}} = chart;
          means.forEach((val,i)=>{
            const std = stds[i];
            const xPos = x.getPixelForValue(i);
            const yTop = y.getPixelForValue(Math.min(255, val+std));
            const yBot = y.getPixelForValue(Math.max(0,   val-std));
            c2.save();
            c2.strokeStyle = chList.map(c=>chColors[c])[i];
            c2.lineWidth = 2;
            c2.beginPath(); c2.moveTo(xPos,yTop); c2.lineTo(xPos,yBot); c2.stroke();
            [-5,5].forEach(d=>{
              c2.beginPath(); c2.moveTo(xPos+d,yTop); c2.lineTo(xPos-d,yTop); c2.stroke();
              c2.beginPath(); c2.moveTo(xPos+d,yBot); c2.lineTo(xPos-d,yBot); c2.stroke();
            });
            c2.restore();
          });
        }
      }]
    });

    // Stats table below
    let tbl = `<table style="margin-top:10px;font-size:11px"><thead><tr>
      <th>Channel</th><th>Mean</th><th>Std</th><th>Q25</th><th>Q50</th><th>Q75</th>
    </tr></thead><tbody>`;
    for (const c of chList) {
      const s = ch[c] || {};
      tbl += `<tr>
        <td style="color:${chColors[c]};font-weight:600">${c}</td>
        <td>${s.mean!=null?s.mean.toFixed(1):'—'}</td>
        <td>${s.std!=null?s.std.toFixed(1):'—'}</td>
        <td>${s.q25!=null?s.q25.toFixed(1):'—'}</td>
        <td>${s.q50!=null?s.q50.toFixed(1):'—'}</td>
        <td>${s.q75!=null?s.q75.toFixed(1):'—'}</td>
      </tr>`;
    }
    tbl += '</tbody></table>';
    wrap.insertAdjacentHTML('beforeend', tbl);
  } catch(e) {
    wrap.innerHTML = `<span style="color:#ef4444;font-size:12px">${e.message}</span>`;
  }
}

// ── ROW 6: Image Quality Scatter ──────────────────────────────────────────────
async function loadQuality() {
  document.getElementById('quality-summary').textContent = 'Sampling 300 images... ';
  document.getElementById('quality-flagged-table').innerHTML =
    '<div style="color:#64748b;font-size:12px;padding:20px">Running... <span class="spinner"></span></div>';
  try {
    const data = await apiFetch('/api/analysis/quality');
    const rows = data.rows || [];

    // classify each image
    const pts = rows.map(r=>{
      const brightness = (r.blue+r.green+r.red+r.yellow)/4;
      const contrast   = r.contrast || 0;
      let flag = 'good';
      if (brightness < 15)  flag = 'dark';
      else if (brightness > 240) flag = 'saturated';
      else if (contrast < 10) flag = 'lowcontrast';
      return {...r, brightness, contrast, flag};
    });

    // summary
    const cnts = {dark:0,lowcontrast:0,saturated:0,good:0};
    pts.forEach(p=>cnts[p.flag]=(cnts[p.flag]||0)+1);
    document.getElementById('quality-summary').textContent =
      `${cnts.dark} dark · ${cnts.lowcontrast} low-contrast · ${cnts.saturated} saturated · ${cnts.good} good`;

    // scatter chart
    const colorMap = {dark:'#f59e0b',lowcontrast:'#f87171',saturated:'#60a5fa',good:'#22c55e'};
    const flagOrder = ['dark','lowcontrast','saturated','good'];
    const datasets = flagOrder.map(flag=>({
      label: flag==='lowcontrast'?'Low contrast':flag.charAt(0).toUpperCase()+flag.slice(1),
      data: pts.filter(p=>p.flag===flag).map(p=>({x:p.brightness,y:p.contrast,id:p.id})),
      backgroundColor: colorMap[flag]+'aa',
      pointRadius:4, pointHoverRadius:6
    }));

    const ctx = document.getElementById('quality-scatter').getContext('2d');
    if (_qualityChart) _qualityChart.destroy();
    _qualityChart = new Chart(ctx, {
      type:'scatter',
      data:{datasets},
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{
          legend:{labels:{color:'#94a3b8',boxWidth:12}},
          tooltip:{callbacks:{label:c=>`${c.raw.id||''} — B:${c.raw.x.toFixed(1)} C:${c.raw.y.toFixed(1)}`}}
        },
        scales:{
          x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},
             title:{display:true,text:'Brightness (mean pixel value)',color:'#64748b',font:{size:10}}},
          y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},
             title:{display:true,text:'Contrast (std pixel value)',color:'#64748b',font:{size:10}}}
        }
      }
    });

    // Flagged images table
    const flagged = pts.filter(p=>p.flag!=='good');
    let html = `<div style="font-size:11px;color:#64748b;margin-bottom:6px">
      ${flagged.length} flagged images (dark + low-contrast + saturated)</div>`;
    if (flagged.length) {
      html += `<table style="font-size:11px"><thead><tr>
        <th>Image ID</th><th>Flag</th><th>Brightness</th><th>Contrast</th>
        <th>Blue</th><th>Green</th><th>Red</th><th>Yellow</th>
      </tr></thead><tbody>`;
      for (const r of flagged.slice(0,60)) {
        const dotCls = r.flag==='lowcontrast'?'quality-dot-lowcontrast':'quality-dot-'+r.flag;
        html += `<tr>
          <td style="max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px" title="${r.id}">${r.id.slice(0,20)}…</td>
          <td><span class="quality-dot ${dotCls}"></span>${r.flag}</td>
          <td>${r.brightness.toFixed(1)}</td>
          <td>${r.contrast.toFixed(1)}</td>
          <td>${r.blue.toFixed(1)}</td>
          <td>${r.green.toFixed(1)}</td>
          <td>${r.red.toFixed(1)}</td>
          <td>${r.yellow.toFixed(1)}</td>
        </tr>`;
      }
      html += '</tbody></table>';
    } else {
      html += '<div style="color:#22c55e;font-size:12px;padding:10px">All images pass quality thresholds</div>';
    }
    document.getElementById('quality-flagged-table').innerHTML = html;
  } catch(e) {
    document.getElementById('quality-summary').textContent = 'Error: ' + e.message;
    document.getElementById('quality-flagged-table').innerHTML =
      `<span style="color:#ef4444;font-size:12px">${e.message}</span>`;
  }
}

// ── ROW 7: Imbalance Table ────────────────────────────────────────────────────
async function loadImbalanceTable() {
  const wrap = document.getElementById('imbalance-table-wrap');
  try {
    const data = await apiFetch('/api/analysis/imbalance');
    _imbalanceData = data.classes || [];
    renderImbalanceTable();
  } catch(e) {
    wrap.innerHTML = `<span style="color:#ef4444;font-size:12px">${e.message}</span>`;
  }
}

function renderImbalanceTable() {
  const wrap = document.getElementById('imbalance-table-wrap');
  const classes = _imbalanceData || [];
  let html = `<table class="imbalance-tbl"><thead><tr>
    <th>Class</th>
    <th style="text-align:right">Count</th>
    <th style="text-align:right">% Dataset</th>
    <th style="text-align:right">Imbalance Ratio</th>
    <th style="text-align:right">Rec. BCE Weight</th>
    <th>Status</th>
  </tr></thead><tbody>`;
  for (const c of classes) {
    const rowCls = c.weight >= 10 ? 'weight-severe' : c.weight >= 5 ? 'weight-warn' : 'weight-ok';
    const statusColor = c.weight >= 10 ? '#ef4444' : c.weight >= 5 ? '#f59e0b' : '#22c55e';
    const statusText  = c.weight >= 10 ? 'Severely underrepresented' : c.weight >= 5 ? 'Underrepresented' : 'Balanced';
    html += `<tr class="${rowCls}">
      <td style="font-size:11px">${c.name}</td>
      <td style="text-align:right;font-weight:600;color:#a78bfa">${(c.count||0).toLocaleString()}</td>
      <td style="text-align:right;color:#94a3b8">${(c.pct||0).toFixed(2)}%</td>
      <td style="text-align:right;color:#94a3b8">${(c.ratio||0).toFixed(1)}×</td>
      <td style="text-align:right;font-weight:600;color:${statusColor}">${(c.weight||0).toFixed(2)}</td>
      <td style="font-size:10px;color:${statusColor}">${statusText}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function exportImbalanceCSV() {
  if (!_imbalanceData || !_imbalanceData.length) { toast('Load imbalance data first'); return; }
  const hdr = 'id,name,count,pct,ratio,weight\n';
  const rows = _imbalanceData.map(c=>`${c.id},"${c.name}",${c.count},${c.pct.toFixed(4)},${c.ratio.toFixed(4)},${c.weight.toFixed(4)}`);
  const csv = hdr + rows.join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href=url; a.download='hpa_class_weights.csv'; a.click();
  URL.revokeObjectURL(url);
  toast('Exported hpa_class_weights.csv');
}

// =====================================================
// TAB 3 — Train
// =====================================================
let gpuRefreshTimer = null;
let logRefreshTimer = null;
let metricsSortKey = 'epoch';
let metricsSortAsc = true;
let _lastEpochsData = [];

let _autoRefreshEnabled = true;

function getInterval(id, fallback) {
  const v = parseInt(document.getElementById(id)?.value || fallback);
  return Math.max(1, v) * 1000;
}

function applyRefreshIntervals() {
  if (!_autoRefreshEnabled) return;
  if (trainRefreshTimer) { clearInterval(trainRefreshTimer); trainRefreshTimer = null; }
  if (gpuRefreshTimer)   { clearInterval(gpuRefreshTimer);   gpuRefreshTimer   = null; }
  if (logRefreshTimer)   { clearInterval(logRefreshTimer);   logRefreshTimer   = null; }
  if (agentRefreshTimer) { clearInterval(agentRefreshTimer); agentRefreshTimer = null; }
  trainRefreshTimer = setInterval(() => { refreshTrainAll(); updateLastRefreshTime(); }, getInterval('metrics-interval', 10));
  gpuRefreshTimer   = setInterval(refreshGPU,      getInterval('gpu-interval', 5));
  logRefreshTimer   = setInterval(refreshTrainLog, getInterval('log-interval', 5));
  agentRefreshTimer = setInterval(refreshAgent, 8000);
}

function toggleAutoRefresh() {
  _autoRefreshEnabled = !_autoRefreshEnabled;
  const btn = document.getElementById('autorefresh-toggle');
  if (_autoRefreshEnabled) {
    btn.textContent = '⏸ Pause';
    btn.className = 'btn btn-success btn-sm';
    applyRefreshIntervals();
  } else {
    btn.textContent = '▶ Resume';
    btn.className = 'btn btn-secondary btn-sm';
    if (trainRefreshTimer) { clearInterval(trainRefreshTimer); trainRefreshTimer = null; }
    if (gpuRefreshTimer)   { clearInterval(gpuRefreshTimer);   gpuRefreshTimer   = null; }
    if (logRefreshTimer)   { clearInterval(logRefreshTimer);   logRefreshTimer   = null; }
    if (agentRefreshTimer) { clearInterval(agentRefreshTimer); agentRefreshTimer = null; }
  }
}

function updateLastRefreshTime() {
  const el = document.getElementById('last-refresh-time');
  if (el) el.textContent = 'Last refresh: ' + new Date().toLocaleTimeString();
}

function initTrainTab() {
  loadRunsList();
  refreshTrainAll();
  refreshGPU();
  refreshTrainLog();
  refreshAgent();
  updateLastRefreshTime();
  applyRefreshIntervals();
}

async function startTraining() {
  const cfg = {
    model:       document.getElementById('cfg-model').value,
    lr:          parseFloat(document.getElementById('cfg-lr').value),
    batch_size:  parseInt(document.getElementById('cfg-bs').value),
    epochs:      parseInt(document.getElementById('cfg-epochs').value),
    img_size:    parseInt(document.getElementById('cfg-imgsize').value),
    val_split:   parseFloat(document.getElementById('cfg-val').value),
    pretrained:  document.getElementById('cfg-pretrained').checked,
    num_workers: parseInt(document.getElementById('cfg-workers').value),
  };
  try {
    const r = await apiFetch('/api/train/start', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(cfg)
    });
    toast('Training started (PID ' + r.pid + ')');
    setTimeout(refreshTrainStatus, 1500);
  } catch(e) { toast('Error: ' + e.message); }
}

async function stopTraining() {
  try {
    await apiFetch('/api/train/stop', {method:'POST'});
    toast('Stop signal sent');
    setTimeout(refreshTrainStatus, 1200);
  } catch(e) { toast('Error: ' + e.message); }
}

async function updateLR() {
  const lr = parseFloat(document.getElementById('live-lr').value);
  if (!lr || lr <= 0) { toast('Enter a valid LR value'); return; }
  try {
    await apiFetch('/api/train/update_lr', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({lr})
    });
    toast('LR updated to ' + lr);
  } catch(e) { toast('Error: ' + e.message); }
}

async function downloadCSV() {
  try {
    const resp = await fetch('/api/train/metrics/csv');
    if (!resp.ok) { toast('No metrics available'); return; }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'train_metrics.csv';
    a.click();
    URL.revokeObjectURL(url);
  } catch(e) { toast('Error: ' + e.message); }
}

async function refreshGPU() {
  try {
    const g = await apiFetch('/api/system/gpu');
    if (!g.available) {
      document.getElementById('gpu-name').textContent = 'No GPU';
      document.getElementById('gpu-no-gpu').style.display = 'block';
      document.getElementById('gpu-stats').style.opacity = '0.4';
      return;
    }
    document.getElementById('gpu-no-gpu').style.display = 'none';
    document.getElementById('gpu-stats').style.opacity = '1';
    document.getElementById('gpu-name').textContent = g.name || 'GPU';

    const util = parseFloat(g.utilization_gpu) || 0;
    document.getElementById('gpu-util-bar').style.width = util + '%';
    document.getElementById('gpu-util-val').textContent = util.toFixed(0) + '%';

    const memUsed = parseFloat(g.memory_used) || 0;
    const memTotal = parseFloat(g.memory_total) || 1;
    const memPct = Math.min(100, memUsed / memTotal * 100);
    document.getElementById('gpu-mem-bar').style.width = memPct.toFixed(0) + '%';
    document.getElementById('gpu-mem-val').textContent = memUsed.toFixed(0) + '/' + memTotal.toFixed(0) + ' MiB';

    const temp = parseFloat(g.temperature_gpu) || 0;
    document.getElementById('gpu-temp-bar').style.width = Math.min(100, temp / 100 * 100).toFixed(0) + '%';
    document.getElementById('gpu-temp-val').textContent = temp.toFixed(0) + ' °C';

    const pwr = parseFloat(g.power_draw) || 0;
    const pwrPct = Math.min(100, pwr / 300 * 100);  // assume 300W TDP max
    document.getElementById('gpu-pwr-bar').style.width = pwrPct.toFixed(0) + '%';
    document.getElementById('gpu-pwr-val').textContent = pwr.toFixed(0) + ' W';

    document.getElementById('gpu-proc').textContent = g.process || '—';
  } catch(e) {
    document.getElementById('gpu-name').textContent = 'GPU unavailable';
  }
}

async function refreshTrainLog() {
  try {
    const data = await apiFetch('/api/train/log');
    const box = document.getElementById('train-log-box');
    if (data.lines && data.lines.length > 0) {
      box.textContent = data.lines.join('\n');
      box.scrollTop = box.scrollHeight;
    }
  } catch(e) {}
}

function computeETA(eps, totalEpochs) {
  if (!eps || eps.length < 2) return null;
  // Use last 5 epochs to estimate seconds/epoch
  const recent = eps.slice(-5);
  const times = recent.map(e => e.ts ? new Date(e.ts).getTime() : null).filter(Boolean);
  if (times.length < 2) return null;
  const dtMs = times[times.length-1] - times[0];
  if (dtMs <= 0) return null;
  const secPerEp = dtMs / 1000 / (times.length - 1);
  const done = eps[eps.length-1].epoch;
  const remaining = totalEpochs - done;
  if (remaining <= 0) return 'Done';
  const secs = Math.round(secPerEp * remaining);
  const fmtDur = s => s < 60 ? s+'s' : s < 3600 ? Math.round(s/60)+'min' : (s/3600).toFixed(1)+'h';
  return `${fmtDur(secs)} (${fmtDur(Math.round(secPerEp))}/ep)`;
}

function f1Label(f1) {
  if (f1 == null || isNaN(f1)) return '';
  if (f1 > 0.7)  return '<span style="color:#a78bfa">Excellent</span>';
  if (f1 > 0.5)  return '<span style="color:#22c55e">Good</span>';
  if (f1 > 0.3)  return '<span style="color:#f59e0b">Fair</span>';
  return '<span style="color:#ef4444">Poor</span>';
}

async function refreshTrainAll() {
  refreshTrainStatus();
  refreshTrainMetrics();
}

async function refreshTrainStatus() {
  try {
    const s = await apiFetch('/api/train/status');
    const st = s.status || 'idle';
    const clsMap = {running:'status-running', done:'status-done', error:'status-error', stopped:'status-idle'};
    const cls = clsMap[st] || 'status-idle';
    const dotMap = {running:'🟡', done:'🟢', error:'🔴', stopped:'🟣', idle:'🟣'};
    const dot = dotMap[st] || '🟣';
    const lbl = st==='running' ? `Running ${s.epoch}/${s.total}` : st.charAt(0).toUpperCase()+st.slice(1);

    document.getElementById('train-status-big').className = 'status-badge ' + cls;
    document.getElementById('train-status-big').textContent = lbl;
    document.getElementById('train-status-badge-top').className = 'status-badge ' + cls;
    document.getElementById('train-status-badge-top').textContent = lbl;
    document.getElementById('train-status-dot').textContent = dot;
    document.getElementById('train-status-detail').textContent = s.pid ? `PID: ${s.pid}` : '';
    document.getElementById('status-model').textContent = s.model || '—';
    document.getElementById('status-device').textContent = s.device || '—';

    // plain english status
    let plainMsg = 'Idle — configure and click Start';
    if (st === 'running') {
      const pct = s.total > 0 ? Math.round(s.epoch / s.total * 100) : 0;
      const eta = computeETA(_lastEpochsData, s.total || 0);
      plainMsg = `Epoch ${s.epoch}/${s.total} — ${pct}%` + (eta ? `  ·  ETA: ${eta}` : '');
    } else if (st === 'done') {
      plainMsg = 'Training complete';
    } else if (st === 'error') {
      plainMsg = 'Error — check training log';
    } else if (st === 'stopped') {
      plainMsg = 'Stopped — click Start to resume';
    }
    document.getElementById('train-status-plain').textContent = plainMsg;

    const pct = s.total > 0 && s.epoch != null ? Math.round(s.epoch / s.total * 100) : 0;
    document.getElementById('train-progress-fill').style.width = pct + '%';
    document.getElementById('train-progress-pct').textContent = pct + '%';

    // stat boxes
    document.getElementById('stat-epoch').textContent = (s.epoch != null && s.total > 0) ? `${s.epoch}/${s.total}` : '—';
    const bf1 = s.best_f1 != null ? Number(s.best_f1) : null;
    document.getElementById('stat-best-f1').textContent = bf1 != null ? bf1.toFixed(4) : '—';
    document.getElementById('stat-f1-label').innerHTML = bf1 != null ? f1Label(bf1) : '';

    // LR from last epoch in metrics
    const lr = _lastEpochsData.length > 0 ? _lastEpochsData[_lastEpochsData.length-1].lr : null;
    document.getElementById('stat-lr').textContent = lr != null ? Number(lr).toExponential(2) : '—';

    // ETA
    const eta = computeETA(_lastEpochsData, s.total || 0);
    document.getElementById('stat-eta').textContent = eta || '—';
  } catch(e) {}
}

async function refreshTrainMetrics() {
  try {
    const data = await apiFetch('/api/train/metrics');
    if (!data.epochs || data.epochs.length===0) return;
    _lastEpochsData = data.epochs;
    const eps = data.epochs;

    // apply window filter
    const lossWin = parseInt(document.getElementById('loss-window')?.value || '0');
    const f1Win   = parseInt(document.getElementById('f1-window')?.value   || '0');
    const lossEps = lossWin > 0 ? eps.slice(-lossWin) : eps;
    const f1Eps   = f1Win   > 0 ? eps.slice(-f1Win)   : eps;

    updateLossChart(lossEps.map(e=>e.epoch), lossEps.map(e=>e.train_loss), lossEps.map(e=>e.val_loss), eps);
    updateF1Chart(f1Eps.map(e=>e.epoch), f1Eps.map(e=>e.macro_f1), eps);

    const last = eps[eps.length-1];
    if (last && last.per_class_f1) updatePerClassChart(last.per_class_f1);

    // heatmap (only if >=2 epochs)
    if (eps.length >= 2) {
      document.getElementById('heatmap-card').style.display = 'block';
      renderHeatmap(eps);
    }

    renderMetricsTable(eps);
  } catch(e) {}
}

function updateLossChart(xs, trainLoss, valLoss, allEps) {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  // find min val_loss epoch for annotation
  let minValIdx = 0, minVal = Infinity;
  if (allEps) {
    allEps.forEach((e,i) => { if ((e.val_loss||Infinity) < minVal) { minVal=e.val_loss; minValIdx=i; } });
  }
  const annotations = {};
  if (allEps && allEps[minValIdx]) {
    annotations['bestVal'] = {
      type:'line', scaleID:'x', value: xs.indexOf(allEps[minValIdx].epoch),
      borderColor:'#60a5fa55', borderWidth:1.5, borderDash:[4,3],
      label:{content:'best val', display:true, color:'#60a5fa', font:{size:9}, position:'start'}
    };
  }
  const opts = {
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#94a3b8'}}, annotation: Object.keys(annotations).length ? {annotations} : {}},
    scales:{x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}},y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}}}
  };
  if (!lossChart) {
    lossChart = new Chart(ctx, {
      type:'line',
      data:{
        labels:xs,
        datasets:[
          {label:'Train Loss',data:trainLoss,borderColor:'#f87171',borderWidth:2,pointRadius:3,fill:false},
          {label:'Val Loss',  data:valLoss,  borderColor:'#60a5fa',borderWidth:2,pointRadius:3,fill:false},
        ]
      },
      options:opts
    });
  } else {
    lossChart.data.labels = xs;
    lossChart.data.datasets[0].data = trainLoss;
    lossChart.data.datasets[1].data = valLoss;
    lossChart.update('none');
  }
}

function updateF1Chart(xs, f1s, allEps) {
  const ctx = document.getElementById('f1-chart').getContext('2d');
  let bestF1 = 0, bestF1Idx = 0;
  if (allEps) {
    allEps.forEach((e,i) => { if ((e.macro_f1||0) > bestF1) { bestF1=e.macro_f1; bestF1Idx=i; } });
  }
  const datasets = [
    {label:'Macro F1',data:f1s,borderColor:'#a78bfa',backgroundColor:'#a78bfa22',borderWidth:2,pointRadius:3,fill:true}
  ];
  if (bestF1 > 0) {
    datasets.push({
      label:'Best F1',
      data: xs.map(()=>bestF1),
      borderColor:'#a78bfa55', borderDash:[6,4], borderWidth:1.5,
      pointRadius:0, fill:false
    });
  }
  if (!f1Chart) {
    f1Chart = new Chart(ctx, {
      type:'line',
      data:{labels:xs, datasets},
      options:{
        responsive:true,maintainAspectRatio:false,
        plugins:{legend:{labels:{color:'#94a3b8',filter:i=>i.text!=='Best F1'||bestF1>0}}},
        scales:{
          x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}},
          y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},min:0,max:1}
        }
      }
    });
  } else {
    f1Chart.data.labels = xs;
    f1Chart.data.datasets[0].data = f1s;
    if (f1Chart.data.datasets[1]) f1Chart.data.datasets[1].data = xs.map(()=>bestF1);
    f1Chart.update('none');
  }
}

let _lastPcf1 = null;
let _perClassShowAll = false;

function togglePerClassAll() {
  _perClassShowAll = !_perClassShowAll;
  const btn = document.getElementById('perclass-toggle-btn');
  btn.textContent = _perClassShowAll ? 'Show Top 14' : 'Show All 28';
  if (_lastPcf1) updatePerClassChart(_lastPcf1);
}

function updatePerClassChart(pcf1Array) {
  if (!pcf1Array || pcf1Array.length===0) return;
  _lastPcf1 = pcf1Array;
  // Accept either array of values or array of epochs — handle both
  let pcf1 = Array.isArray(pcf1Array) && typeof pcf1Array[0] === 'object' && pcf1Array[0].per_class_f1
    ? pcf1Array[pcf1Array.length-1].per_class_f1
    : pcf1Array;
  if (!pcf1) return;

  const ctx = document.getElementById('perclass-chart').getContext('2d');
  const indexed = pcf1.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v);
  const display = _perClassShowAll ? indexed : indexed.slice(0, 14);

  const labels = display.map(x=>CLASS_NAMES[x.i]);
  const values = display.map(x=>x.v);
  const colors = values.map(v => v >= 0.7 ? '#22c55e' : v >= 0.5 ? '#f59e0b' : '#ef4444');

  // Dynamic height: ~24px per bar
  const h = Math.max(340, display.length * 26);
  document.getElementById('perclass-chart-wrap').style.height = h + 'px';

  // Always destroy+recreate when label count changes (Chart.js doesn't handle shrink/grow well)
  if (perClassChart) { perClassChart.destroy(); perClassChart = null; }
  perClassChart = new Chart(ctx, {
    type:'bar',
    data:{labels,datasets:[{label:'F1',data:values,backgroundColor:colors,borderWidth:0}]},
    options:{
      responsive:true,maintainAspectRatio:false,indexAxis:'y',
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`F1: ${c.raw.toFixed(3)}`}}},
      scales:{
        x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},min:0,max:1,title:{display:true,text:'F1 Score',color:'#64748b'}},
        y:{ticks:{color:'#e2e8f0',font:{size:11}},grid:{color:'#1e293b'}}
      }
    }
  });
}

function renderHeatmap(eps) {
  const container = document.getElementById('f1-heatmap');
  const N = 28;
  let html = '<table class="heatmap-tbl"><thead><tr><th class="epoch-col">Ep</th>';
  for (let c=0; c<N; c++) {
    html += `<th title="${CLASS_NAMES[c]}" style="writing-mode:vertical-rl;transform:rotate(180deg);max-height:120px;vertical-align:bottom;padding:4px 2px;font-size:10px;color:#94a3b8">${CLASS_NAMES[c]}</th>`;
  }
  html += '</tr></thead><tbody>';
  for (const ep of eps) {
    const pcf1 = ep.per_class_f1 || [];
    html += `<tr><td class="epoch-col heatmap-tbl">${ep.epoch}</td>`;
    for (let c=0; c<N; c++) {
      const v = pcf1[c] != null ? pcf1[c] : 0;
      const r = Math.round(167 * v);
      const g = Math.round(139 * v);
      const b = Math.round(250 * v);
      const fg = v > 0.5 ? '#fff' : '#64748b';
      html += `<td style="background:rgb(${r},${g},${b});color:${fg}">${v.toFixed(2)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

function sortMetrics(key) {
  if (metricsSortKey === key) metricsSortAsc = !metricsSortAsc;
  else { metricsSortKey = key; metricsSortAsc = true; }
  renderMetricsTable(_lastEpochsData);
}

function renderMetricsTable(eps) {
  if (!eps || eps.length===0) return;
  const tbody = document.getElementById('metrics-tbody');
  if (!tbody) return;

  // find best val_loss and best macro_f1
  let bestValLoss = Infinity, bestF1 = -1;
  let bestValEp = null, bestF1Ep = null;
  eps.forEach(e => {
    if ((e.val_loss||Infinity) < bestValLoss) { bestValLoss=e.val_loss; bestValEp=e.epoch; }
    if ((e.macro_f1||0) > bestF1) { bestF1=e.macro_f1; bestF1Ep=e.epoch; }
  });

  // sort
  const sorted = [...eps].sort((a,b) => {
    const av = a[metricsSortKey] ?? 0;
    const bv = b[metricsSortKey] ?? 0;
    return metricsSortAsc ? (av>bv?1:-1) : (av<bv?1:-1);
  });

  let html = '';
  for (const e of sorted) {
    let rowCls = '';
    if (e.epoch === bestValEp && e.epoch === bestF1Ep) rowCls = 'row-best-val';
    else if (e.epoch === bestValEp) rowCls = 'row-best-val';
    else if (e.epoch === bestF1Ep)  rowCls = 'row-best-f1';
    const ts = e.ts ? new Date(e.ts*1000).toLocaleTimeString() : '—';
    const lr = e.lr != null ? Number(e.lr).toExponential(1) : '—';
    html += `<tr class="${rowCls}">
      <td>${e.epoch}</td>
      <td>${e.train_loss != null ? Number(e.train_loss).toFixed(4) : '—'}</td>
      <td>${e.val_loss   != null ? Number(e.val_loss).toFixed(4)   : '—'}</td>
      <td>${e.macro_f1   != null ? Number(e.macro_f1).toFixed(4)   : '—'}</td>
      <td>${lr}</td>
      <td>${ts}</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}

// =====================================================

// ── Run History ────────────────────────────────────────────────────────────────
let _viewingRunId = null;

async function loadRunsList() {
  const runs = await apiFetch('/api/train/runs').catch(() => []);
  const el = document.getElementById('runs-list');
  if (!runs || runs.length === 0) {
    el.innerHTML = '<div style="color:#64748b;font-size:13px;padding:8px 0">No previous runs yet. Start training to create one.</div>';
    return;
  }
  el.innerHTML = runs.map(r => {
    const f1   = r.best_f1 != null ? (r.best_f1 * 100).toFixed(1) + '%' : '—';
    const qual = r.best_f1 > 0.7 ? '#22c55e' : r.best_f1 > 0.5 ? '#f59e0b' : r.best_f1 > 0.3 ? '#f87171' : '#64748b';
    const active = _viewingRunId === r.run_id ? 'border-color:#a78bfa;background:#a78bfa11;' : '';
    return `<div onclick="loadRun('${r.run_id}')" style="display:flex;align-items:center;gap:12px;padding:8px 10px;border:1px solid #2d3148;border-radius:6px;margin-bottom:6px;cursor:pointer;transition:all .15s;${active}"
      onmouseover="this.style.borderColor='#a78bfa'" onmouseout="this.style.borderColor='${_viewingRunId===r.run_id?'#a78bfa':'#2d3148'}'">
      <div style="flex:1;min-width:0">
        <div style="font-size:12px;color:#e2e8f0;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
          ${r.model || '?'} &nbsp;·&nbsp; ${r.epochs_done || 0} epochs
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">${(r.run_id||'').replace('_',' ')}</div>
      </div>
      <div style="text-align:right;flex-shrink:0">
        <div style="font-size:13px;font-weight:700;color:${qual}">F1 ${f1}</div>
        <div style="font-size:10px;color:#475569">${r.archived_at ? r.archived_at.slice(0,10) : ''}</div>
      </div>
      <button class="btn btn-secondary btn-sm" style="flex-shrink:0" onclick="event.stopPropagation();loadRun('${r.run_id}')">Load</button>
    </div>`;
  }).join('');
}

async function loadRun(runId) {
  _viewingRunId = runId;
  document.getElementById('viewing-run-badge').style.display = 'inline-block';
  const data = await apiFetch('/api/train/runs/' + runId).catch(() => null);
  if (!data) { toast('Failed to load run'); return; }

  // Restore config form
  if (data.config) {
    const c = data.config;
    const setVal = (id, v) => { const el = document.getElementById(id); if(el) el.value = v; };
    setVal('cfg-model',   c.model || 'efficientnet_b0');
    setVal('cfg-lr',      c.lr || 0.0001);
    setVal('cfg-bs',      c.batch_size || 32);
    setVal('cfg-epochs',  c.epochs || 30);
    setVal('cfg-imgsize', c.img_size || 224);
    setVal('cfg-val',     c.val_split || 0.1);
    setVal('cfg-workers', c.num_workers || 4);
    const pt = document.getElementById('cfg-pretrained');
    if (pt) pt.checked = c.pretrained !== false;
  }

  // Render metrics as if live
  if (data.epochs && data.epochs.length > 0) {
    renderLossChart(data.epochs);
    renderF1Chart(data.epochs);
    updatePerClassChart(data.epochs);
    renderHeatmap(data.epochs);
    renderMetricsTable(data.epochs);
  }

  // Show status from archived status
  if (data.status) {
    const s = data.status;
    const badge = document.getElementById('train-status-big');
    if (badge) { badge.className = 'status-badge status-done'; badge.textContent = `Done — ${s.epoch||0}/${s.total||0} epochs`; }
    const bf = document.getElementById('best-f1');
    if (bf) bf.textContent = s.best_f1 != null ? (s.best_f1*100).toFixed(1)+'%' : '—';
  }

  // Highlight selected row
  loadRunsList();
  toast('Loaded run: ' + runId);
}

function loadLiveRun() {
  _viewingRunId = null;
  document.getElementById('viewing-run-badge').style.display = 'none';
  refreshTrainAll();
  loadRunsList();
  toast('Back to live data');
}

// ── Agent functions ──────────────────────────────────
const AGENT_LEVEL_COLORS = {
  THINK: '#a78bfa', REASON: '#60a5fa', START: '#34d399',
  DONE: '#22c55e', BEST: '#fbbf24', CRASH: '#f87171',
  STALL: '#f87171', WARN: '#f59e0b', STOP: '#94a3b8',
  BOOT: '#64748b', ERROR: '#ef4444', INFO: '#cbd5e1',
  WATCH: '#38bdf8',  // per-epoch observations
};

async function agentStart() {
  const r = await apiFetch('/api/agent/start', {method:'POST'}).catch(e=>({error:e.message}));
  if (r.error) { toast('Agent start failed: ' + r.error, true); return; }
  toast('Agent started (PID ' + (r.pid||'?') + ')');
  setTimeout(refreshAgent, 1500);
}

async function agentStop() {
  const r = await apiFetch('/api/agent/stop', {method:'POST'}).catch(e=>({error:e.message}));
  if (r.error) { toast('Agent stop failed: ' + r.error, true); return; }
  toast('Agent stop signal sent');
  setTimeout(refreshAgent, 2000);
}

async function refreshAgent() {
  // Status
  const s = await apiFetch('/api/agent/status').catch(()=>null);
  if (s) {
    const running = s.running;
    const dot  = document.getElementById('agent-status-dot');
    const badge = document.getElementById('agent-badge');
    if (dot)  dot.textContent = running ? '🟢' : (s.state==='stopped'?'🔴':'🟣');
    if (badge) {
      badge.textContent = running ? (s.state||'running').toUpperCase() : (s.state||'IDLE').toUpperCase();
      badge.className   = 'status-badge ' + (running ? 'status-running' : (s.state==='stopped'?'status-done':'status-idle'));
    }
    const setEl = (id, val) => { const e=document.getElementById(id); if(e) e.textContent=val; };
    setEl('agent-runs',       s.total_runs != null ? s.total_runs : '—');
    setEl('agent-best-f1',    s.best_f1_ever != null ? (s.best_f1_ever*100).toFixed(1)+'%' : '—');
    setEl('agent-best-model', s.best_model || '—');
    setEl('agent-phase',      (s.phase||s.state||'—').toUpperCase());
    setEl('agent-uptime',     s.uptime || '—');

    const reas = document.getElementById('agent-reasoning');
    if (reas && s.current_reasoning) reas.textContent = s.current_reasoning;

    const sb = document.getElementById('agent-start-btn');
    const stb = document.getElementById('agent-stop-btn');
    if (sb)  sb.disabled = running;
    if (stb) stb.disabled = !running;
  }

  // Log
  const logData = await apiFetch('/api/agent/log?n=80').catch(()=>null);
  if (logData && logData.entries) {
    const box = document.getElementById('agent-log-box');
    if (box) {
      if (logData.entries.length === 0) {
        box.innerHTML = '— no log yet —';
      } else {
        box.innerHTML = [...logData.entries].reverse().map(e => {
          const col = AGENT_LEVEL_COLORS[e.level] || '#cbd5e1';
          const ts  = (e.ts||'').slice(11,19);
          let msg = e.msg || '';
          // For REASON entries, show reasoning excerpt inline
          if (e.level==='REASON' && e.reasoning) {
            const lines = e.reasoning.split('\n').slice(0,4).join(' | ');
            msg += ' → ' + lines.slice(0,200);
          }
          return `<div style="padding:2px 0;border-bottom:1px solid #1e293b22">
            <span style="color:#475569;font-size:10px">${ts}</span>
            <span style="color:${col};font-weight:600;margin:0 6px">[${e.level}]</span>
            <span style="color:#e2e8f0">${msg}</span>
          </div>`;
        }).join('');
        box.scrollTop = 0;
      }
    }
  }
}

// TAB 4 — Inference
// =====================================================
async function loadCheckpoints() {
  try {
    const data = await apiFetch('/api/checkpoints');
    const sel = document.getElementById('inf-ckpt');
    sel.innerHTML = '<option value="">-- select --</option>';
    for (const ckpt of data.checkpoints) {
      const opt = document.createElement('option');
      opt.value = ckpt; opt.textContent = ckpt;
      sel.appendChild(opt);
    }
  } catch(e) {}
}

async function randomInfImage() {
  try {
    const data = await apiFetch('/api/sample?n=1&split=all');
    if (data.items && data.items.length > 0) {
      document.getElementById('inf-id').value = data.items[0].id;
    }
  } catch(e) {}
}

async function runInference() {
  const imageId    = document.getElementById('inf-id').value.trim();
  const checkpoint = document.getElementById('inf-ckpt').value;
  if (!imageId)    { toast('Enter an image ID'); return; }
  if (!checkpoint) { toast('Select a checkpoint'); return; }

  document.getElementById('inf-status').innerHTML = 'Running... <span class="spinner"></span>';
  document.getElementById('inf-results').style.display = 'none';

  // Show preview immediately
  const prev = document.getElementById('inf-preview');
  prev.innerHTML = `
    <img src="/api/image/composite?id=${encodeURIComponent(imageId)}"
         style="width:130px;height:130px;object-fit:cover;border-radius:6px;border:1px solid #2d3148"/>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
      <img src="/api/image/channel?id=${encodeURIComponent(imageId)}&ch=blue"
           style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Blue"/>
      <img src="/api/image/channel?id=${encodeURIComponent(imageId)}&ch=green"
           style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Green"/>
      <img src="/api/image/channel?id=${encodeURIComponent(imageId)}&ch=red"
           style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Red"/>
      <img src="/api/image/channel?id=${encodeURIComponent(imageId)}&ch=yellow"
           style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Yellow"/>
    </div>`;

  try {
    const result = await apiFetch('/api/inference/run', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image_id:imageId, checkpoint})
    });
    document.getElementById('inf-status').textContent = result.error || '';
    if (!result.error) renderInfResults(result);
  } catch(e) {
    document.getElementById('inf-status').textContent = 'Error: ' + e.message;
  }
}

function renderInfResults(result) {
  const container = document.getElementById('inf-results');
  container.style.display = 'block';
  const bars = document.getElementById('inf-pred-bars');
  bars.innerHTML = '';

  const gtSet   = new Set(result.gt_labels || []);
  const predSet = new Set((result.predicted_labels || []).map(p=>p.class_id));
  const scores  = result.all_scores || Array(28).fill(0);
  const thresh  = result.threshold || 0.5;

  const header = document.createElement('div');
  header.style.cssText = 'font-size:12px;color:#64748b;margin-bottom:10px';
  header.textContent = `Threshold: ${thresh.toFixed(2)}  |  GT labels: ${[...gtSet].map(i=>CLASS_NAMES[i]).join(', ') || 'none'}`;
  bars.appendChild(header);

  // Sort all 28 by score descending
  const sorted = scores.map((s,i)=>({s,i})).sort((a,b)=>b.s-a.s);

  for (const {s, i} of sorted) {
    const isPred = predSet.has(i);
    const isGT   = gtSet.has(i);
    let barColor    = '#2d3148';
    let labelCls    = '';
    if      (isPred && isGT)   { barColor='#22c55e'; labelCls='pred-correct'; }
    else if (isGT && !isPred)  { barColor='#ef4444'; labelCls='pred-missed'; }
    else if (isPred && !isGT)  { barColor='#f59e0b'; labelCls='pred-fp'; }

    const row = document.createElement('div');
    row.className = 'pred-bar-row';
    row.innerHTML = `
      <span class="pred-label ${labelCls}">${CLASS_NAMES[i]}</span>
      <div class="pred-bar-bg">
        <div class="pred-bar-fill" style="width:${(s*100).toFixed(1)}%;background:${barColor || '#a78bfa'}"></div>
      </div>
      <span class="pred-score">${(s*100).toFixed(1)}%</span>`;
    bars.appendChild(row);
  }
}

// =====================================================
// TAB 5 — Annotate
// =====================================================
function buildAnnotCheckboxes() {
  const el = document.getElementById('annot-checkboxes');
  if (el.children.length > 0) return;
  for (let i=0; i<28; i++) {
    const div = document.createElement('div');
    div.className = 'cb-item';
    div.innerHTML = `<input type="checkbox" id="acb_${i}" value="${i}"/>
      <label for="acb_${i}">${CLASS_NAMES[i]}</label>`;
    el.appendChild(div);
  }
}

async function loadAnnotateNext() {
  const flaggedOnly = document.getElementById('show-flagged').checked;
  try {
    const url = `/api/annotate/next?flagged_only=${flaggedOnly}&current=${encodeURIComponent(currentAnnotId||'')}`;
    const data = await apiFetch(url);
    currentAnnotId = data.id;
    document.getElementById('annot-img-id').textContent = data.id;

    const prev = document.getElementById('annot-preview');
    prev.innerHTML = `
      <img src="/api/image/composite?id=${encodeURIComponent(data.id)}"
           style="width:130px;height:130px;object-fit:cover;border-radius:6px;border:1px solid #2d3148"/>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
        <img src="/api/image/channel?id=${encodeURIComponent(data.id)}&ch=blue"
             style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Blue"/>
        <img src="/api/image/channel?id=${encodeURIComponent(data.id)}&ch=green"
             style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Green"/>
        <img src="/api/image/channel?id=${encodeURIComponent(data.id)}&ch=red"
             style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Red"/>
        <img src="/api/image/channel?id=${encodeURIComponent(data.id)}&ch=yellow"
             style="width:60px;height:60px;object-fit:cover;border-radius:4px" title="Yellow"/>
      </div>`;

    for (let i=0; i<28; i++) {
      const cb = document.getElementById('acb_'+i);
      if (cb) cb.checked = (data.labels||[]).includes(i);
    }
    document.getElementById('annot-note').value = data.note || '';
    loadAnnotateStats();
  } catch(e) { toast('Error: ' + e.message); }
}

async function saveAnnotation(flagged=false) {
  if (!currentAnnotId) return;
  const labels = [];
  for (let i=0; i<28; i++) {
    const cb = document.getElementById('acb_'+i);
    if (cb && cb.checked) labels.push(i);
  }
  const note = document.getElementById('annot-note').value;
  try {
    await apiFetch('/api/annotate/save', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id:currentAnnotId, verified_labels:labels, flagged, note})
    });
    toast(flagged ? 'Flagged & saved' : 'Saved!');
    loadAnnotateNext();
  } catch(e) { toast('Error: ' + e.message); }
}

function flagAnnotation() { saveAnnotation(true); }

async function loadAnnotateStats() {
  try {
    const s = await apiFetch('/api/annotate/stats');
    document.getElementById('annot-progress').textContent =
      `Verified: ${s.verified} / ${s.total.toLocaleString()}  |  Flagged: ${s.flagged}`;
  } catch(e) {}
}

// =====================================================
// Boot
// =====================================================
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('sidebar-explorer').style.display = 'block';
  init();
});

// ── Chat Widget ───────────────────────────────────────────────────────────────
const SUGGESTIONS = [
  'How many training images?',
  'Which class has most samples?',
  'What is the rarest class?',
  'Is training running?',
  'What is the best F1 so far?',
  'Which classes have F1 below 0.3?',
  'What classes co-occur with Nucleoplasm?',
  'How many multi-label images?',
  'What is the current learning rate?',
  'GPU status?',
];

function toggleChat() {
  const p = document.getElementById('chat-panel');
  p.classList.toggle('open');
  if (p.classList.contains('open')) {
    const msgs = document.getElementById('chat-messages');
    if (!msgs.children.length) {
      addBotMsg('👋 Hi! I can answer questions about your HPA dataset and training. Try one of the suggestions below or type your own question.');
      renderSuggestions();
    }
    document.getElementById('chat-input').focus();
  }
}

function clearChat() {
  document.getElementById('chat-messages').innerHTML = '';
  addBotMsg('Chat cleared. Ask me anything about your dataset or training!');
  renderSuggestions();
}

function renderSuggestions() {
  const el = document.getElementById('chat-suggestions');
  el.innerHTML = SUGGESTIONS.slice(0,5).map(s =>
    `<button class="chat-chip" onclick="askSuggestion('${s}')">${s}</button>`
  ).join('');
}

function askSuggestion(q) {
  document.getElementById('chat-input').value = q;
  sendChat();
}

function addUserMsg(text) {
  const el = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'chat-msg-user';
  d.textContent = text;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

function addBotMsg(html) {
  const el = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'chat-msg-bot';
  d.innerHTML = html;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

function addTyping() {
  const el = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'chat-msg-bot';
  d.id = 'chat-typing';
  d.innerHTML = '<span class="spinner"></span>';
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('chat-typing');
  if (t) t.remove();
}

async function sendChat() {
  const inp = document.getElementById('chat-input');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  addUserMsg(q);
  addTyping();
  try {
    const r = await apiFetch('/api/chat', { method:'POST', body: JSON.stringify({q}) });
    removeTyping();
    addBotMsg(r.answer);
    if (r.suggestions) {
      const el = document.getElementById('chat-suggestions');
      el.innerHTML = r.suggestions.map(s =>
        `<button class="chat-chip" onclick="askSuggestion('${s}')">${s}</button>`
      ).join('');
    }
  } catch(e) {
    removeTyping();
    addBotMsg('Sorry, something went wrong. Try again.');
  }
}

// =====================================================
// Post Analysis Tab — Biological / Domain-Expert View
// =====================================================

// Biological groupings of the 28 HPA compartments
const BIO_GROUPS = {
  'Nuclear': { ids:[0,1,2,3,4,5], color:'#60a5fa' },
  'Secretory Pathway': { ids:[6,7], color:'#34d399' },
  'Vesicular / Degradative': { ids:[8,9,10,20], color:'#fb923c' },
  'Cytoskeletal': { ids:[11,12,13,14,15], color:'#a78bfa' },
  'Cell Division': { ids:[16,17,18,19], color:'#f472b6' },
  'Plasma Membrane': { ids:[21,22], color:'#38bdf8' },
  'Cytoplasmic / Misc': { ids:[23,24,25,26,27], color:'#fbbf24' },
};

// Training sample counts from the full dataset (31,072 images)
const CLASS_SAMPLES = {
  0:12885,1:1254,2:3621,3:1561,4:1858,5:2513,
  6:1008,7:2822,8:53,9:45,10:28,
  11:1093,12:688,13:537,14:1066,15:21,
  16:530,17:210,18:902,19:1482,20:172,
  21:3777,22:802,23:2965,24:322,25:8228,26:328,27:11
};

// Known biological challenges per class (expert knowledge)
const CLASS_CHALLENGES = {
  0:'Diffuse nuclear signal; often co-labels with other nuclear compartments',
  1:'Thin rim pattern; confused with nucleoli rim and nuclear interior',
  2:'Bright, discrete nucleolar foci; variable in number and size per cell',
  3:'Sub-nucleolar sub-structure; visible only at high resolution',
  4:'Irregular speckled nuclear pattern; overlaps with nuclear bodies',
  5:'Discrete nuclear foci; morphologically similar to nuclear speckles',
  6:'Reticular perinuclear network; can be confused with Golgi in 2D',
  7:'Perinuclear stack or ribbon; easily confused with ER at low resolution',
  8:'Tiny puncta (~0.1–0.5 µm); extremely rare in dataset (53 images)',
  9:'Small puncta overlapping with lysosomes; only 45 training images',
  10:'Highly similar to peroxisomes/endosomes; 28 images — critically starved',
  11:'Long fibrous cytoplasmic bundles; variable expression across cell lines',
  12:'Cortical and stress-fiber patterns; variable with cell state',
  13:'Discrete puncta at cytoskeletal termini; easily confused with centrosome',
  14:'Linear filaments throughout cell; variable density across mitotic stages',
  15:'Tips of microtubule plus-ends; tiny puncta, only 21 training images',
  16:'Thin cytoplasmic canal between late dividing cells; rare event',
  17:'Barrel-shaped structure only in mitosis; ~1–2% of cells at any time',
  18:'Spot near nucleus; confused with centrosome, focal adhesion',
  19:'Paired foci near nucleus; confused with MTOC and nuclear bodies',
  20:'Round lipid-filled organelles; variable size and number; 172 images',
  21:'Uniform cell boundary; often co-labels with cell junctions',
  22:'Discrete spots at cell-cell contacts; requires multi-cell context',
  23:'Punctate cytoplasmic network; generally reliable due to large dataset',
  24:'Peri-nuclear inclusion when proteasome overwhelmed; 322 images',
  25:'Diffuse cytoplasm; hardest to distinguish from nucleoplasm leakage',
  26:'Heterogeneous cytoplasmic foci; poorly defined morphology; 328 images',
  27:'Extremely rare rod-shaped crystalline structures; only 11 training images',
};

// Known visual confound groups
const CONFOUND_GROUPS = [
  { label:'Small puncta — impossible to separate at standard resolution',
    classes:[8,9,10], note:'Peroxisomes, Endosomes and Lysosomes all appear as small round puncta. Even trained pathologists require IHC co-staining to distinguish them.' },
  { label:'Nuclear foci — morphologically overlapping',
    classes:[4,5], note:'Nuclear speckles and nuclear bodies both appear as irregular foci inside the nucleus. Distinguishing them requires marker co-staining.' },
  { label:'Diffuse cytoplasm vs nucleus — ambiguous boundaries',
    classes:[0,25], note:'Nucleoplasm and cytosol both show diffuse, non-punctate signal. Classification depends on cell segmentation quality.' },
  { label:'Perinuclear membranes — 2D projection overlap',
    classes:[6,7], note:'ER and Golgi are adjacent in 3D but collapse onto each other in 2D fluorescence projections, especially in flat cells.' },
  { label:'Mitotic-stage only structures — rare in asynchronous cultures',
    classes:[16,17,18], note:'Cytokinetic bridge, mitotic spindle and MTOC are only visible in actively dividing cells (<2% of cells in standard culture).' },
  { label:'Sub-micron centrosomal structures',
    classes:[13,15,18,19], note:'Focal adhesion sites, microtubule ends, MTOC and centrosome all produce small, bright puncta near the cell periphery or nucleus.' },
];

let paCharts = {};

function paDestroyAll() {
  Object.values(paCharts).forEach(c => { try { c.destroy(); } catch(e){} });
  paCharts = {};
}

function paAnnotStatus(f1, samples) {
  if (samples < 100) return { label:'Data-Starved', color:'#a78bfa', bg:'#2e1065' };
  if (f1 >= 0.50)   return { label:'Auto-Annotate', color:'#4ade80', bg:'#052e16' };
  if (f1 >= 0.15)   return { label:'Expert Review', color:'#fbbf24', bg:'#1c1203' };
  return { label:'Insufficient', color:'#f87171', bg:'#1f0a0a' };
}

async function initPostAnalysis() {
  document.getElementById('pa-last-refresh').textContent = 'Loading…';
  paDestroyAll();

  document.getElementById('pa-last-refresh').textContent = 'Loading…';
  paDestroyAll();

  let d;
  try {
    d = await apiFetch('/api/postanalysis/data');
  } catch(e) {
    document.getElementById('pa-bio-tbody').innerHTML =
      `<tr><td colspan="6" style="color:#f87171;padding:16px;text-align:center">Error loading data: ${e.message}</td></tr>`;
    return;
  }

  const runs = d.runs || [];
  const s    = d.summary || {};
  const classAvg = d.class_avg_f1 || [];

  if (!runs.length) {
    document.getElementById('pa-bio-tbody').innerHTML =
      '<tr><td colspan="6" style="color:#64748b;padding:20px;text-align:center">No completed runs found. Start training to populate this view.</td></tr>';
    document.getElementById('pa-last-refresh').textContent = 'No runs yet';
    return;
  }

  const bestRun = runs.reduce((a, b) => a.best_f1 > b.best_f1 ? a : b);
  const bestPcf1 = bestRun.pcf1_at_best || [];
  const nCls = 28;

  // Compute per-class F1 — use best run as the primary signal
  const clsF1 = Array.from({length: nCls}, (_, i) => classAvg[i] ?? (bestPcf1[i] || 0));

  // ── ROW 1: Atlas Readiness Banner ─────────────────────────────────────────
  document.getElementById('pa-best-model-label').textContent = `Best model: ${bestRun.model}`;

  let nReady=0, nReview=0, nWeak=0, nStarved=0;
  for (let i=0; i<nCls; i++) {
    const st = paAnnotStatus(clsF1[i], CLASS_SAMPLES[i]||0);
    if (st.label==='Auto-Annotate') nReady++;
    else if (st.label==='Expert Review') nReview++;
    else if (st.label==='Data-Starved') nStarved++;
    else nWeak++;
  }
  document.getElementById('pa-ready-count').textContent   = nReady;
  document.getElementById('pa-review-count').textContent  = nReview;
  document.getElementById('pa-weak-count').textContent    = nWeak;
  document.getElementById('pa-starved-count').textContent = nStarved;

  const barEl = document.getElementById('pa-readiness-bar');
  barEl.innerHTML = [
    { n:nReady,   c:'#4ade80' },
    { n:nReview,  c:'#fbbf24' },
    { n:nWeak,    c:'#f87171' },
    { n:nStarved, c:'#a78bfa' },
  ].map(seg => `<div style="flex:${seg.n};background:${seg.c}" title="${seg.n}"></div>`).join('');

  // ── ROW 2a: Compartment Group Chart ──────────────────────────────────────
  const groupNames  = Object.keys(BIO_GROUPS);
  const groupColors = groupNames.map(g => BIO_GROUPS[g].color);
  const groupAvgF1  = groupNames.map(g => {
    const ids = BIO_GROUPS[g].ids;
    return +(ids.reduce((s, i) => s + (clsF1[i]||0), 0) / ids.length).toFixed(3);
  });
  const gcCtx = document.getElementById('pa-group-chart').getContext('2d');
  paCharts.grp = new Chart(gcCtx, {
    type: 'bar',
    data: {
      labels: groupNames,
      datasets: [{
        data: groupAvgF1,
        backgroundColor: groupColors.map(c => c + 'cc'),
        borderColor:     groupColors,
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display:false },
        annotation: { annotations: { line1: {
          type:'line', yMin:0.5, yMax:0.5,
          borderColor:'#94a3b866', borderWidth:1, borderDash:[5,4],
        }}}
      },
      scales: {
        x: { ticks:{ color:'#94a3b8', font:{size:10} }, grid:{ color:'#1e2235' } },
        y: { min:0, max:1, ticks:{ color:'#64748b' }, grid:{ color:'#1e2235' },
             title:{ display:true, text:'Avg F1', color:'#64748b' } }
      }
    }
  });

  // ── ROW 2b: Data Scarcity Scatter ─────────────────────────────────────────
  function clsGroupColor(id) {
    for (const [g, v] of Object.entries(BIO_GROUPS)) if (v.ids.includes(id)) return v.color;
    return '#94a3b8';
  }
  const scCtx = document.getElementById('pa-scarcity-chart').getContext('2d');
  paCharts.sc = new Chart(scCtx, {
    type: 'scatter',
    data: {
      datasets: [{
        data: Array.from({length:nCls}, (_, i) => ({
          x: Math.log10(Math.max(1, CLASS_SAMPLES[i]||1)),
          y: clsF1[i],
          _id: i,
          _name: CLASS_NAMES[i],
          _n: CLASS_SAMPLES[i]||0,
        })),
        backgroundColor: Array.from({length:nCls}, (_,i) => clsGroupColor(i) + 'cc'),
        borderColor:     Array.from({length:nCls}, (_,i) => clsGroupColor(i)),
        pointRadius: 6, pointHoverRadius: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display:false },
        tooltip: { callbacks: {
          label: ctx => `${ctx.raw._name}: F1=${ctx.raw.y.toFixed(3)}, n=${ctx.raw._n.toLocaleString()}`
        }}
      },
      scales: {
        x: { title:{ display:true, text:'log₁₀(training samples)', color:'#64748b' },
             ticks:{ color:'#64748b', callback: v => `10^${v.toFixed(0)}` }, grid:{ color:'#1e2235' } },
        y: { min:0, max:1, title:{ display:true, text:'F1 Score', color:'#64748b' },
             ticks:{ color:'#64748b' }, grid:{ color:'#1e2235' } }
      }
    }
  });

  // ── ROW 3: Biological Assessment Table ────────────────────────────────────
  const sortedByF1 = Array.from({length:nCls}, (_,i)=>i).sort((a,b) => clsF1[b]-clsF1[a]);
  function clsGroupName(id) {
    for (const [g,v] of Object.entries(BIO_GROUPS)) if (v.ids.includes(id)) return g;
    return '—';
  }
  const bioTbody = document.getElementById('pa-bio-tbody');
  bioTbody.innerHTML = sortedByF1.map((c, rowIdx) => {
    const f1   = clsF1[c];
    const n    = CLASS_SAMPLES[c] || 0;
    const st   = paAnnotStatus(f1, n);
    const grp  = clsGroupName(c);
    const grpC = BIO_GROUPS[grp]?.color || '#94a3b8';
    const ch   = CLASS_CHALLENGES[c] || '—';
    const bg   = rowIdx%2===0 ? '#141624' : '#1a1d2a';
    const f1c  = f1>=0.5 ? '#4ade80' : f1>=0.15 ? '#fbbf24' : '#f87171';
    const nc   = n<100 ? '#a78bfa' : n<500 ? '#fbbf24' : '#94a3b8';
    return `<tr style="background:${bg}">
      <td style="padding:6px 10px;color:#e2e8f0;font-weight:600">${CLASS_NAMES[c]||c}</td>
      <td style="padding:6px 8px"><span style="color:${grpC};font-size:11px">${grp}</span></td>
      <td style="padding:6px 8px;text-align:center;color:${nc};font-weight:${n<100?'700':'400'}">${n.toLocaleString()}${n<100?' ⚠':''}${n<30?' ‼':''}</td>
      <td style="padding:6px 8px;text-align:center;color:${f1c};font-weight:700">${f1.toFixed(3)}</td>
      <td style="padding:6px 8px;color:#64748b;font-size:11px;max-width:320px">${ch}</td>
      <td style="padding:6px 8px;text-align:center">
        <span style="background:${st.bg};color:${st.color};border:1px solid ${st.color}55;border-radius:4px;padding:2px 8px;font-size:10px;font-weight:700;white-space:nowrap">${st.label}</span>
      </td>
    </tr>`;
  }).join('');

  // ── ROW 4a: Visual Confounds ──────────────────────────────────────────────
  const confEl = document.getElementById('pa-confounds');
  confEl.innerHTML = CONFOUND_GROUPS.map(g => {
    const names = g.classes.map(i => CLASS_NAMES[i]).join(', ');
    const f1s   = g.classes.map(i => `F1=${clsF1[i].toFixed(2)}`).join(' / ');
    return `<div style="padding:9px 0;border-bottom:1px solid #1e2235">
      <div style="color:#fbbf24;font-size:11px;font-weight:600;margin-bottom:3px">${g.label}</div>
      <div style="color:#94a3b8;font-size:11px;margin-bottom:3px">${names} · <span style="color:#64748b">${f1s}</span></div>
      <div style="color:#475569;font-size:10px;line-height:1.6">${g.note}</div>
    </div>`;
  }).join('');

  // ── ROW 4b: F1 Learning Curve ─────────────────────────────────────────────
  document.getElementById('pa-curve-label').textContent =
    `${bestRun.model} · best F1 = ${bestRun.best_f1.toFixed(4)} @ epoch ${bestRun.best_ep}`;
  const f1Ctx = document.getElementById('pa-f1-curves-chart').getContext('2d');
  paCharts.f1 = new Chart(f1Ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Macro F1 (val)',
        data: bestRun.f1_curve.map((v,ep) => ({x:ep+1,y:v})),
        borderColor: '#34d399', backgroundColor: '#34d39911',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3,
      }, {
        label: 'Val Loss (scaled)',
        data: bestRun.val_loss_curve.map((v,ep) => ({x:ep+1,y:Math.min(1,v)})),
        borderColor: '#f8717166', backgroundColor: 'transparent',
        borderWidth: 1.5, pointRadius: 0, borderDash:[4,3], tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type:'linear', title:{display:true,text:'Epoch',color:'#64748b'}, ticks:{color:'#64748b'}, grid:{color:'#1e2235'} },
        y: { min:0, ticks:{color:'#64748b'}, grid:{color:'#1e2235'} }
      },
      plugins: { legend:{ labels:{ color:'#94a3b8', boxWidth:12, font:{size:11} } } }
    }
  });
  // Biological milestones as annotations in the notes div
  const f1c = bestRun.f1_curve;
  const ep10pct = f1c.findIndex(v => v >= bestRun.best_f1*0.1) + 1;
  const ep50pct = f1c.findIndex(v => v >= bestRun.best_f1*0.5) + 1;
  document.getElementById('pa-curve-notes').innerHTML =
    `Model crossed 10% of peak F1 at epoch <b>${ep10pct>0?ep10pct:'?'}</b> · ` +
    `50% of peak at epoch <b>${ep50pct>0?ep50pct:'?'}</b> · ` +
    `Peak at epoch <b>${bestRun.best_ep}</b>. ` +
    `${bestRun.overfit_ratio > 1.3 ? 'Val loss diverged from training loss — model started memorizing staining patterns rather than learning true localizations.' :
       'Train and val loss tracked closely — model is generalizing well.'}`;

  // ── ROW 5: Multi-label co-localization ────────────────────────────────────
  // Known most-frequent co-localizations from the HPA training set
  const colocPairs = [
    { pair:'Nucleoplasm + Cytosol',           n:3841, note:'Proteins with dual nuclear-cytoplasmic shuttling (e.g., transcription factors)' },
    { pair:'Nucleoplasm + Nuclear membrane',  n:672,  note:'Proteins anchored at the nuclear envelope' },
    { pair:'Nucleoplasm + Nucleoli',          n:618,  note:'Proteins involved in ribosome biogenesis' },
    { pair:'Cytosol + Plasma membrane',       n:549,  note:'Peripheral membrane proteins / cytoskeletal anchors' },
    { pair:'Mitochondria + Cytosol',          n:484,  note:'Proteins translocating between cytoplasm and mitochondria' },
    { pair:'Nucleoplasm + Mitochondria',      n:421,  note:'Dual-targeted proteins with nuclear and mitochondrial roles' },
    { pair:'ER + Golgi apparatus',            n:318,  note:'Secretory pathway proteins in transit' },
    { pair:'Nucleoplasm + ER',                n:290,  note:'Proteins with roles in both transcription and protein synthesis' },
  ];
  document.getElementById('pa-coloc-list').innerHTML = colocPairs.map(p =>
    `<div style="display:flex;gap:10px;align-items:baseline;border-bottom:1px solid #1e2235;padding:4px 0">
      <span style="color:#60a5fa;font-weight:600;min-width:190px;font-size:11px">${p.pair}</span>
      <span style="color:#475569;font-size:11px">${p.n.toLocaleString()} images</span>
      <span style="color:#64748b;font-size:10px">${p.note}</span>
    </div>`
  ).join('');

  const multiLabelF1penalty = clsF1.filter((_,i) => CLASS_SAMPLES[i]>200)
    .reduce((s,v)=>s+v,0) / clsF1.filter((_,i)=>CLASS_SAMPLES[i]>200).length;
  document.getElementById('pa-multilabel-notes').innerHTML = `
    <p>51.3% of the 31,072 training images carry more than one label. For a classifier trained with independent binary cross-entropy per class (BCEWithLogitsLoss), co-localized structures are treated as independent — the model never learns that Nucleoplasm and Cytosol frequently co-occur, so it may suppress one prediction when it sees both.</p>
    <p style="margin-top:8px">Average F1 on well-represented classes (n≥200): <b style="color:#34d399">${multiLabelF1penalty.toFixed(3)}</b>. Classes expected to appear together (e.g., ER+Golgi) benefit from co-occurrence signal in training — consider adding co-occurrence regularization or a graph-based label dependency layer in future iterations.</p>`;

  // ── ROW 6: Scientific Interpretation ─────────────────────────────────────
  const autoReady  = sortedByF1.filter(i => paAnnotStatus(clsF1[i],CLASS_SAMPLES[i]||0).label==='Auto-Annotate')
                                .map(i => CLASS_NAMES[i]);
  const dataStarve = sortedByF1.filter(i => paAnnotStatus(clsF1[i],CLASS_SAMPLES[i]||0).label==='Data-Starved')
                                .map(i => CLASS_NAMES[i]);
  const trueWeak   = sortedByF1.filter(i => {
    const s = paAnnotStatus(clsF1[i],CLASS_SAMPLES[i]||0);
    return s.label==='Insufficient' || s.label==='Expert Review';
  }).filter(i => (CLASS_SAMPLES[i]||0)>=100).map(i => CLASS_NAMES[i]);

  document.getElementById('pa-scientific').innerHTML = `
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
    <div>
      <div style="font-size:11px;color:#4ade80;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">Compartments ready for semi-automated atlas annotation</div>
      <p>${autoReady.length ? autoReady.join(', ') + '. These compartments show consistent, characteristic staining patterns across cell lines and can be reliably flagged in new protein images without expert review.' : 'No compartments have reached the 0.50 F1 threshold yet. Continue training to expand coverage.'}</p>
      <div style="font-size:11px;color:#a78bfa;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin:12px 0 8px">Data-starved structures — a data collection problem, not a model failure</div>
      <p>${dataStarve.length ? dataStarve.join(', ') + '. These structures have fewer than 100 training images. The model physically cannot learn their morphology with this sample count — no amount of architecture tuning will fix this. The priority action is targeted data collection or use of external databases (e.g., OpenCell, Allen Cell).' : 'All classes have ≥100 training samples.'}</p>
    </div>
    <div>
      <div style="font-size:11px;color:#fbbf24;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">Structures with sufficient data but low model confidence</div>
      <p>${trueWeak.length ? trueWeak.slice(0,8).join(', ') + ` (${trueWeak.length} total). These have adequate training data but remain challenging due to visual ambiguity, cell-state dependence, or sub-resolution size. Candidate improvements: higher input resolution (384px+), focal loss to address residual imbalance, or contrastive pretraining on single-organelle marker images.` : 'All well-represented classes are at review-level confidence or better.'}</p>
      <div style="font-size:11px;color:#60a5fa;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin:12px 0 8px">Recommended next experiments</div>
      <ol style="margin:0;padding-left:18px;line-height:2.2;color:#94a3b8">
        <li>Acquire ≥200 additional images for: <b style="color:#a78bfa">${dataStarve.slice(0,4).join(', ')}${dataStarve.length>4?'…':''}</b></li>
        <li>Try 384px input resolution — small puncta (peroxisomes, microtubule ends, focal adhesions) require pixel-level detail</li>
        <li>Add co-occurrence loss or label graph to model co-localizing pairs</li>
        <li>Evaluate on a held-out set stratified by number of co-labels per image</li>
      </ol>
    </div>
  </div>`;

  document.getElementById('pa-last-refresh').textContent =
    'Updated ' + new Date().toLocaleTimeString();
}
</script>

<!-- Floating Chat Widget -->
<div id="chat-bubble" onclick="toggleChat()" title="Ask a question about your data">
  💬
</div>
<div id="chat-panel">
  <div id="chat-header">
    <span>HPA Assistant</span>
    <div style="display:flex;gap:8px;align-items:center">
      <button onclick="clearChat()" style="background:none;border:none;color:#64748b;cursor:pointer;font-size:11px">Clear</button>
      <button onclick="toggleChat()" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:16px">✕</button>
    </div>
  </div>
  <div id="chat-messages"></div>
  <div id="chat-suggestions"></div>
  <div id="chat-input-row">
    <input type="text" id="chat-input" placeholder="Ask about your dataset or training..." onkeydown="if(event.key==='Enter')sendChat()"/>
    <button onclick="sendChat()">➤</button>
  </div>
</div>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

# ── Dataset stats ──────────────────────────────────────────────────────────────
@app.route('/api/stats')
def api_stats():
    rows = load_csv()
    class_counts = defaultdict(int)
    total_labels = 0
    multi_label  = 0
    for labels in rows.values():
        if len(labels) > 1:
            multi_label += 1
        for l in labels:
            class_counts[l] += 1
            total_labels += 1
    n = len(rows)
    return jsonify({
        'total_images':    n,
        'total_labels':    total_labels,
        'multi_label_pct': round(multi_label / n, 4) if n > 0 else 0.0,
        'class_counts': {int(k): int(v) for k, v in class_counts.items()}
    })

# ── Sample images ──────────────────────────────────────────────────────────────
@app.route('/api/sample')
def api_sample():
    import random
    n           = int(request.args.get('n', 9))
    split       = request.args.get('split', 'train')
    mode        = request.args.get('mode', 'OR')
    classes_str = request.args.get('classes', '')

    rows = load_csv()
    all_ids = list(rows.keys())

    if classes_str:
        filter_cls = set(int(c) for c in classes_str.split(',') if c.strip())
        if mode == 'AND':
            filtered = [i for i in all_ids if filter_cls.issubset(set(rows[i]))]
        else:
            filtered = [i for i in all_ids if bool(filter_cls & set(rows[i]))]
    else:
        filtered = all_ids

    total_matching = len(filtered)
    n = min(n, len(filtered))
    sample = random.sample(filtered, n) if n > 0 else []

    items = [{'id': sid, 'labels': rows[sid]} for sid in sample]
    return jsonify({'items': items, 'total_matching': total_matching})

# ── Image serving ──────────────────────────────────────────────────────────────
@app.route('/api/image/composite')
def api_image_composite():
    image_id = request.args.get('id', '')
    png = make_composite_png(image_id)
    if png is None:
        return Response('Not found', status=404)
    return Response(png, mimetype='image/png')

@app.route('/api/image/channel')
def api_image_channel():
    image_id = request.args.get('id', '')
    ch = request.args.get('ch', 'blue')
    if ch not in ('blue', 'green', 'red', 'yellow'):
        return Response('Invalid channel', status=400)
    png = make_channel_png(image_id, ch, colorize=True)
    if png is None:
        return Response('Not found', status=404)
    return Response(png, mimetype='image/png')

# ── Analysis: cardinality ──────────────────────────────────────────────────────
@app.route('/api/analysis/cardinality')
def api_analysis_cardinality():
    rows = load_csv()
    counts = defaultdict(int)
    for labels in rows.values():
        n = len(labels)
        key = str(n) if n <= 4 else '5+'
        counts[key] += 1
    # ensure keys exist
    for k in ['1','2','3','4','5+']:
        counts.setdefault(k, 0)
    return jsonify({'counts': dict(counts)})

# ── Analysis: co-occurring pairs ───────────────────────────────────────────────
@app.route('/api/analysis/pairs')
def api_analysis_pairs():
    rows = load_csv()
    total = len(rows)
    pair_counts = defaultdict(int)
    for labels in rows.values():
        lbls = sorted(set(l for l in labels if 0 <= l < 28))
        for idx in range(len(lbls)):
            for jdx in range(idx+1, len(lbls)):
                pair_counts[(lbls[idx], lbls[jdx])] += 1
    # sort by count descending, take top 20
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:20]
    pairs = []
    for (a, b), cnt in sorted_pairs:
        pairs.append({
            'a': a, 'b': b,
            'a_name': CLASS_NAMES.get(a, str(a)),
            'b_name': CLASS_NAMES.get(b, str(b)),
            'count': cnt,
            'pct': round(cnt / total * 100, 2) if total > 0 else 0.0
        })
    return jsonify({'pairs': pairs})

# ── Analysis: channel stats (cached) ───────────────────────────────────────────
_channel_stats_cache = None
_channel_stats_lock  = threading.Lock()

@app.route('/api/analysis/channel_stats')
def api_analysis_channel_stats():
    global _channel_stats_cache
    with _channel_stats_lock:
        if _channel_stats_cache is not None:
            return jsonify(_channel_stats_cache)
    import random
    rows = load_csv()
    sample_ids = random.sample(list(rows.keys()), min(300, len(rows)))
    ch_data = {c: [] for c in ('blue', 'green', 'red', 'yellow')}
    for sid in sample_ids:
        ch = load_channels(sid)
        if ch is None:
            continue
        for color in ('blue', 'green', 'red', 'yellow'):
            ch_data[color].append(float(np.mean(ch[color])))
    result = {}
    for color, vals in ch_data.items():
        if not vals:
            result[color] = {}
            continue
        arr = np.array(vals)
        result[color] = {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'q25':  float(np.percentile(arr, 25)),
            'q50':  float(np.percentile(arr, 50)),
            'q75':  float(np.percentile(arr, 75)),
        }
    out = {'channels': result, 'n': len(sample_ids)}
    with _channel_stats_lock:
        _channel_stats_cache = out
    return jsonify(out)

# ── Analysis: class imbalance ──────────────────────────────────────────────────
@app.route('/api/analysis/imbalance')
def api_analysis_imbalance():
    rows = load_csv()
    total_images = len(rows)
    class_counts = defaultdict(int)
    for labels in rows.values():
        for l in labels:
            if 0 <= l < 28:
                class_counts[l] += 1
    count_max = max(class_counts.values()) if class_counts else 1
    classes = []
    for i in range(28):
        cnt = class_counts.get(i, 0)
        ratio = count_max / max(cnt, 1)
        classes.append({
            'id':     i,
            'name':   CLASS_NAMES.get(i, str(i)),
            'count':  cnt,
            'pct':    round(cnt / total_images * 100, 4) if total_images > 0 else 0.0,
            'ratio':  round(ratio, 4),
            'weight': round(ratio, 4),  # pos_weight for BCE = count_max / count_i
        })
    # sort by count descending
    classes.sort(key=lambda x: -x['count'])
    return jsonify({'classes': classes})

# ── Analysis: co-occurrence ────────────────────────────────────────────────────
@app.route('/api/analysis/cooccurrence')
def api_cooccurrence():
    rows = load_csv()
    N = 28
    matrix = [[0]*N for _ in range(N)]
    for labels in rows.values():
        for i in labels:
            for j in labels:
                if 0 <= i < N and 0 <= j < N:
                    matrix[i][j] += 1
    return jsonify({'matrix': matrix})

# ── Analysis: histogram ────────────────────────────────────────────────────────
@app.route('/api/analysis/histogram')
def api_histogram():
    import random
    image_id = request.args.get('id', '')
    if not image_id:
        rows = load_csv()
        image_id = random.choice(list(rows.keys()))
    ch = load_channels(image_id)
    if ch is None:
        return jsonify({'error': 'image not found'}), 404
    result = {'id': image_id, 'stats': {}}
    for color in ('blue', 'green', 'red', 'yellow'):
        arr = ch[color]
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        result[color] = hist.tolist()
        result['stats'][color] = {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'min':  int(np.min(arr)),
            'max':  int(np.max(arr)),
            'sat_pct': float(np.sum(arr >= 250) / arr.size * 100),
        }
    return jsonify(result)

# ── Analysis: quality ──────────────────────────────────────────────────────────
@app.route('/api/analysis/quality')
def api_quality():
    import random
    rows = load_csv()
    sample_ids = random.sample(list(rows.keys()), min(300, len(rows)))
    result = []
    for sid in sample_ids:
        ch = load_channels(sid)
        if ch is None:
            continue
        row = {'id': sid}
        all_pixels = []
        for color in ('blue', 'green', 'red', 'yellow'):
            arr = ch[color]
            row[color] = float(np.mean(arr))
            all_pixels.append(arr.astype(np.float32))
        # contrast = std across all channels
        combined = np.stack(all_pixels, axis=0)
        row['contrast'] = float(np.std(combined))
        row['snr'] = float(np.mean(combined) / (np.std(combined) + 1e-6))
        result.append(row)
    return jsonify({'rows': result})

# ── Train: start ───────────────────────────────────────────────────────────────
@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    import shutil as _shutil
    cfg = request.get_json(force=True)

    # ── archive previous run ───────────────────────────────────────────────────
    prev_metrics = LOGS_DIR / 'train_log.jsonl'
    prev_status  = LOGS_DIR / 'train_status.json'
    prev_cfg     = LOGS_DIR / 'train_config.json'
    prev_stdout  = LOGS_DIR / 'train_stdout.log'
    if prev_metrics.exists() and prev_metrics.stat().st_size > 0:
        # build run_id from first epoch timestamp or file mtime
        try:
            import json as _json
            with open(prev_metrics) as _f:
                first = _json.loads(_f.readline())
            run_id = first.get('ts', '')[:19].replace(':', '-').replace('T', '_')
        except Exception:
            import datetime as _dt
            run_id = _dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(prev_metrics, run_dir / 'metrics.jsonl')
        if prev_status.exists():  _shutil.copy2(prev_status, run_dir / 'status.json')
        if prev_cfg.exists():     _shutil.copy2(prev_cfg,    run_dir / 'config.json')
        if prev_stdout.exists():  _shutil.copy2(prev_stdout, run_dir / 'stdout.log')
        # read summary for meta
        epochs_done, best_f1, model = 0, 0.0, ''
        try:
            with open(prev_status) as _f: _s = _json.load(_f)
            epochs_done = _s.get('epoch', 0)
            best_f1     = _s.get('best_f1', 0.0)
            model       = _s.get('model', '')
        except Exception: pass
        import datetime as _dt
        with open(run_dir / 'meta.json', 'w') as _f:
            _json.dump({'run_id': run_id, 'model': model,
                        'epochs_done': epochs_done, 'best_f1': best_f1,
                        'archived_at': _dt.datetime.now().isoformat()}, _f)

    cfg_path = LOGS_DIR / 'train_config.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    status_path = LOGS_DIR / 'train_status.json'
    with open(status_path, 'w') as f:
        json.dump({
            'status': 'starting', 'epoch': 0, 'total': cfg.get('epochs', 30),
            'pid': None, 'model': cfg.get('model'), 'device': None, 'best_f1': 0.0
        }, f)

    cmd = (
        f'source {CONDA_INIT} && '
        f'conda activate {CONDA_ENV} && '
        f'python {TRAIN_SCRIPT} --config {cfg_path}'
    )
    log_out = open(LOGS_DIR / 'train_stdout.log', 'w')
    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        stdout=log_out, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    log_out.close()  # parent closes after fork; child inherits the fd

    with open(status_path, 'w') as f:
        json.dump({
            'status': 'running', 'epoch': 0, 'total': cfg.get('epochs', 30),
            'pid': proc.pid, 'model': cfg.get('model'), 'device': None, 'best_f1': 0.0
        }, f)

    return jsonify({'status': 'started', 'pid': proc.pid})

# ── Train: stop ────────────────────────────────────────────────────────────────
@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    status_path = LOGS_DIR / 'train_status.json'
    if not status_path.exists():
        return jsonify({'error': 'No status file'}), 404
    with open(status_path) as f:
        s = json.load(f)
    pid = s.get('pid')
    if not pid:
        return jsonify({'error': 'No PID'}), 400
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    s['status'] = 'stopped'
    with open(status_path, 'w') as f:
        json.dump(s, f)
    return jsonify({'status': 'stopped', 'pid': pid})

# ── Train: status ──────────────────────────────────────────────────────────────
@app.route('/api/train/status')
def api_train_status():
    status_path = LOGS_DIR / 'train_status.json'
    if not status_path.exists():
        return jsonify({'status':'idle','epoch':0,'total':0,'pid':None,'model':None,'device':None,'best_f1':0.0})
    with open(status_path) as f:
        return jsonify(json.load(f))

# ── Train: metrics ─────────────────────────────────────────────────────────────
@app.route('/api/train/metrics')
def api_train_metrics():
    log_path = LOGS_DIR / 'train_log.jsonl'
    if not log_path.exists():
        return jsonify({'epochs': []})
    epochs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    epochs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return jsonify({'epochs': epochs})

# ── Train: update LR ──────────────────────────────────────────────────────────
@app.route('/api/train/update_lr', methods=['POST'])
def api_train_update_lr():
    data = request.get_json(force=True)
    lr = data.get('lr')
    if lr is None:
        return jsonify({'error': 'lr required'}), 400
    with open(LOGS_DIR / 'live_config.json', 'w') as f:
        json.dump({'lr': float(lr)}, f)
    return jsonify({'status': 'ok', 'lr': lr})

# ── System: GPU stats ──────────────────────────────────────────────────────────
@app.route('/api/system/gpu')
def api_system_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            return jsonify({'available': False})
        parts = [p.strip() for p in result.stdout.strip().split(',')]
        if len(parts) < 6:
            return jsonify({'available': False})
        # get current GPU process name
        proc_result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        proc_str = proc_result.stdout.strip().split('\n')[0] if proc_result.stdout.strip() else ''
        return jsonify({
            'available': True,
            'name': parts[0],
            'utilization_gpu': parts[1],
            'memory_used': parts[2],
            'memory_total': parts[3],
            'temperature_gpu': parts[4],
            'power_draw': parts[5],
            'process': proc_str or 'None',
        })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return jsonify({'available': False})
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})

# ── Train: log ─────────────────────────────────────────────────────────────────
@app.route('/api/train/log')
def api_train_log():
    log_path = LOGS_DIR / 'train_stdout.log'
    if not log_path.exists():
        return jsonify({'lines': []})
    try:
        with open(log_path) as f:
            all_lines = f.readlines()
        lines = [l.rstrip('\n') for l in all_lines[-40:]]
        return jsonify({'lines': lines})
    except Exception as e:
        return jsonify({'lines': [f'Error reading log: {e}']})

# ── Train: metrics CSV download ────────────────────────────────────────────────
@app.route('/api/train/metrics/csv')
def api_train_metrics_csv():
    import io as _io
    log_path = LOGS_DIR / 'train_log.jsonl'
    if not log_path.exists():
        return Response('No metrics file', status=404)
    epochs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    epochs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not epochs:
        return Response('No data', status=404)
    buf = _io.StringIO()
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'macro_f1', 'lr', 'ts']
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for e in epochs:
        writer.writerow({k: e.get(k, '') for k in fieldnames})
    csv_bytes = buf.getvalue().encode('utf-8')
    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=train_metrics.csv'}
    )

# ── Train: runs (history) ─────────────────────────────────────────────────────
@app.route('/api/train/runs')
def api_train_runs():
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir(): continue
        meta_path = d / 'meta.json'
        try:
            with open(meta_path) as f: meta = json.load(f)
        except Exception:
            meta = {'run_id': d.name, 'model': '?', 'epochs_done': 0, 'best_f1': 0.0}
        runs.append(meta)
    return jsonify(runs)

@app.route('/api/train/runs/<run_id>')
def api_train_run(run_id):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return jsonify({'error': 'Run not found'}), 404
    result = {}
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f: result['config'] = json.load(f)
    status_path = run_dir / 'status.json'
    if status_path.exists():
        with open(status_path) as f: result['status'] = json.load(f)
    metrics_path = run_dir / 'metrics.jsonl'
    epochs = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: epochs.append(json.loads(line))
                    except: pass
    result['epochs'] = epochs
    meta_path = run_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f: result['meta'] = json.load(f)
    return jsonify(result)

# ── Checkpoints ────────────────────────────────────────────────────────────────
@app.route('/api/checkpoints')
def api_checkpoints():
    if not CKPT_DIR.exists():
        return jsonify({'checkpoints': []})
    ckpts = sorted([p.name for p in CKPT_DIR.glob('*.pt')], reverse=True)
    return jsonify({'checkpoints': ckpts})

# ── Inference ──────────────────────────────────────────────────────────────────
@app.route('/api/inference/run', methods=['POST'])
def api_inference_run():
    data = request.get_json(force=True)
    image_id  = data.get('image_id', '').strip()
    ckpt_name = data.get('checkpoint', '').strip()

    if not image_id:
        return jsonify({'error': 'image_id required'}), 400
    if not ckpt_name:
        return jsonify({'error': 'checkpoint required'}), 400

    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        return jsonify({'error': f'Checkpoint not found: {ckpt_name}'}), 404

    model_key = str(ckpt_path)
    with _model_cache_lock:
        if model_key not in _model_cache:
            _model_cache[model_key] = _load_model(ckpt_path)
        model_info = _model_cache.get(model_key)

    if model_info is None:
        return jsonify({'error': 'Failed to load model checkpoint.'}), 500

    result = _run_model_inference(model_info, image_id)
    rows = load_csv()
    result['gt_labels'] = rows.get(image_id, [])
    return jsonify(result)

def _load_model(ckpt_path):
    try:
        import torch
        import timm
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        cfg_sub = ckpt.get('cfg', {})
        model_name = ckpt.get('model') or cfg_sub.get('model', 'efficientnet_b0')
        in_chans = ckpt.get('in_chans') or cfg_sub.get('in_chans', 4)
        model = timm.create_model(model_name, pretrained=False, num_classes=28, in_chans=in_chans)
        state = (ckpt.get('state_dict') or ckpt.get('model_state_dict') or
                 ckpt.get('state') or
                 {k: v for k, v in ckpt.items() if isinstance(v, __import__('torch').Tensor)})
        model.load_state_dict(state, strict=False)
        model.eval()
        return {'model': model, 'model_name': model_name, 'in_chans': in_chans}
    except Exception as e:
        print(f"[inference] load error: {e}", file=sys.stderr)
        return None

def _run_model_inference(model_info, image_id):
    try:
        import torch
        import torch.nn.functional as F

        ch = load_channels(image_id)
        if ch is None:
            return {'error': 'image not found', 'all_scores': [0.0]*28, 'predicted_labels': [], 'threshold': 0.5}

        in_chans = model_info.get('in_chans', 4)
        channels = ['blue', 'green', 'red', 'yellow'][:in_chans]
        arr = np.stack([ch[c] for c in channels], axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1, C, H, W

        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # per-channel normalization
        mn = tensor.mean(dim=[0,2,3], keepdim=True)
        sd = tensor.std(dim=[0,2,3], keepdim=True).clamp(min=1e-6)
        tensor = (tensor - mn) / sd

        model = model_info['model']
        with torch.no_grad():
            scores = torch.sigmoid(model(tensor))[0].numpy()

        threshold = 0.5
        predicted_labels = [
            {'class_id': int(i), 'score': float(s), 'name': CLASS_NAMES.get(i, f'Class_{i}')}
            for i, s in sorted(enumerate(scores), key=lambda x: -x[1])
            if s >= threshold
        ]
        return {
            'image_id': image_id,
            'all_scores': [float(s) for s in scores],
            'predicted_labels': predicted_labels,
            'threshold': threshold,
        }
    except Exception as e:
        print(f"[inference] run error: {e}", file=sys.stderr)
        return {'error': str(e), 'all_scores': [0.0]*28, 'predicted_labels': [], 'threshold': 0.5}

# ── Annotate ───────────────────────────────────────────────────────────────────
def load_annotations():
    ann_path = LOGS_DIR / 'annotations.json'
    if not ann_path.exists():
        return {}
    with open(ann_path) as f:
        return json.load(f)

def save_annotations_file(ann):
    ann_path = LOGS_DIR / 'annotations.json'
    with open(ann_path, 'w') as f:
        json.dump(ann, f, indent=2)

@app.route('/api/annotate/next')
def api_annotate_next():
    import random
    flagged_only = request.args.get('flagged_only', 'false').lower() == 'true'
    current      = request.args.get('current', '')
    rows = load_csv()
    ann  = load_annotations()
    all_ids = list(rows.keys())

    if flagged_only:
        pool = [i for i in all_ids if ann.get(i, {}).get('flagged', False)]
        if not pool:
            pool = all_ids
    else:
        unannotated = [i for i in all_ids if i not in ann]
        pool = unannotated if unannotated else all_ids

    pool = [i for i in pool if i != current]
    if not pool:
        pool = all_ids

    chosen = random.choice(pool)
    gt     = rows.get(chosen, [])
    exist  = ann.get(chosen, {})
    return jsonify({
        'id':               chosen,
        'labels':           exist.get('verified_labels', gt),
        'gt_labels':        gt,
        'flagged':          exist.get('flagged', False),
        'note':             exist.get('note', ''),
        'already_annotated': chosen in ann
    })

@app.route('/api/annotate/save', methods=['POST'])
def api_annotate_save():
    data = request.get_json(force=True)
    image_id = data.get('id')
    if not image_id:
        return jsonify({'error': 'id required'}), 400
    ann = load_annotations()
    ann[image_id] = {
        'verified_labels': data.get('verified_labels', []),
        'flagged':         data.get('flagged', False),
        'note':            data.get('note', ''),
        'ts':              time.time()
    }
    save_annotations_file(ann)
    return jsonify({'status': 'saved'})

@app.route('/api/annotate/stats')
def api_annotate_stats():
    rows = load_csv()
    ann  = load_annotations()
    verified = len(ann)
    flagged  = sum(1 for v in ann.values() if v.get('flagged', False))
    return jsonify({'total': len(rows), 'verified': verified, 'flagged': flagged})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    import re as _re
    data = request.get_json(force=True)
    q = data.get('q', '').lower().strip()

    def fmt_num(n): return f"{n:,}"
    def pct(v): return f"{v*100:.1f}%"

    rows = load_csv()  # {id: [label_ids]}

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_class_counts():
        from collections import Counter
        c = Counter()
        for lbls in rows.values(): c.update(lbls)
        return c

    def get_metrics():
        lp = LOGS_DIR / 'train_log.jsonl'
        if not lp.exists(): return []
        eps = []
        with open(lp) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: eps.append(json.loads(line))
                    except: pass
        return eps

    def get_status():
        sp = LOGS_DIR / 'train_status.json'
        if not sp.exists(): return {}
        with open(sp) as f: return json.load(f)

    def get_cfg():
        cp = LOGS_DIR / 'train_config.json'
        if not cp.exists(): return {}
        with open(cp) as f: return json.load(f)

    def find_class_id(q):
        # find class by name fragment or number
        for i, name in CLASS_NAMES.items():
            if name.lower() in q or str(i) in q.split():
                return i, name
        return None, None

    answer = None
    suggestions = None

    # ── intent matching ──────────────────────────────────────────────────────

    # Total images
    if any(x in q for x in ['how many', 'total', 'count', 'size', 'dataset size']):
        if any(x in q for x in ['train', 'image', 'sample', 'data']):
            n_train = len(rows)
            n_multi = sum(1 for v in rows.values() if len(v) > 1)
            answer = (f"<b>{fmt_num(n_train)}</b> training images total.<br>"
                      f"<b>{fmt_num(n_multi)}</b> ({pct(n_multi/n_train)}) have multiple labels.")
            suggestions = ['Which class has most samples?', 'What is the rarest class?', 'How many classes are there?']

    # Class count / num classes
    if answer is None and any(x in q for x in ['how many class', 'number of class', 'classes are']):
        answer = "There are <b>28 classes</b> of protein subcellular localization in the HPA dataset."
        suggestions = ['Which class has most samples?', 'What is the rarest class?']

    # Most common class
    if answer is None and any(x in q for x in ['most common', 'most sample', 'largest class', 'most frequent', 'most images']):
        counts = get_class_counts()
        top = counts.most_common(5)
        rows_txt = ''.join(f"<br>&nbsp;&nbsp;{i+1}. <b>{CLASS_NAMES.get(c, str(c))}</b> — {fmt_num(n)} samples" for i,(c,n) in enumerate(top))
        answer = f"Top 5 most common classes:{rows_txt}"
        suggestions = ['What is the rarest class?', 'How many multi-label images?']

    # Rarest class
    if answer is None and any(x in q for x in ['rare', 'least', 'smallest class', 'fewest', 'uncommon']):
        counts = get_class_counts()
        bottom = counts.most_common()[-5:][::-1]
        rows_txt = ''.join(f"<br>&nbsp;&nbsp;{i+1}. <b>{CLASS_NAMES.get(c, str(c))}</b> — {fmt_num(n)} samples" for i,(c,n) in enumerate(bottom))
        answer = f"5 rarest classes:{rows_txt}"
        suggestions = ['Which class has most samples?', 'What classes co-occur with Rods & rings?']

    # Multi-label
    if answer is None and any(x in q for x in ['multi-label', 'multi label', 'multiple label', 'more than one']):
        n_multi = sum(1 for v in rows.values() if len(v) > 1)
        n = len(rows)
        answer = (f"<b>{fmt_num(n_multi)}</b> out of {fmt_num(n)} images ({pct(n_multi/n)}) "
                  f"have multiple labels. The average is {sum(len(v) for v in rows.values())/n:.2f} labels per image.")
        suggestions = ['Which class has most samples?', 'Is training running?']

    # Training status
    if answer is None and any(x in q for x in ['training', 'is train', 'running', 'status', 'progress']):
        s = get_status()
        st = s.get('status', 'idle')
        if st == 'running':
            ep, tot = s.get('epoch',0), s.get('total',0)
            pct_done = f"{100*ep/tot:.0f}%" if tot else "?"
            answer = (f"Training is <b>running</b> \u2705<br>"
                      f"Epoch <b>{ep}/{tot}</b> ({pct_done} complete)<br>"
                      f"Model: <b>{s.get('model','?')}</b> on <b>{s.get('device','?')}</b><br>"
                      f"Best F1 so far: <b>{pct(s.get('best_f1',0))}</b>")
        elif st == 'done':
            answer = f"Training is <b>done</b> \u2705 Best F1: <b>{pct(s.get('best_f1',0))}</b>"
        else:
            answer = "Training is <b>idle</b> — go to the Train tab to start."
        suggestions = ['What is the best F1 so far?', 'What is the current learning rate?']

    # Best F1
    if answer is None and any(x in q for x in ['best f1', 'highest f1', 'best score', 'best result', 'f1 score']):
        eps = get_metrics()
        if eps:
            best = max(eps, key=lambda e: e.get('macro_f1', 0))
            answer = (f"Best macro F1: <b>{pct(best['macro_f1'])}</b> at epoch <b>{best['epoch']}</b><br>"
                      f"Val loss at that point: <b>{best.get('val_loss',0):.4f}</b>")
        else:
            answer = "No training metrics yet. Start training first!"
        suggestions = ['Which classes have F1 below 0.3?', 'What is the current learning rate?']

    # F1 below threshold
    if answer is None and _re.search(r'f1.*(below|under|less than|<)\s*([\d.]+)', q):
        m = _re.search(r'([\d.]+)', q.split('below' if 'below' in q else 'under' if 'under' in q else '<')[-1])
        thresh = float(m.group(1)) if m else 0.3
        eps = get_metrics()
        if eps:
            last = eps[-1]
            pcf1 = last.get('per_class_f1', [])
            poor = [(i, v) for i, v in enumerate(pcf1) if v < thresh]
            poor.sort(key=lambda x: x[1])
            if poor:
                rows_txt = ''.join(f"<br>&nbsp;&nbsp;\u2022 <b>{CLASS_NAMES.get(i, str(i))}</b>: {pct(v)}" for i,v in poor[:10])
                answer = f"{len(poor)} classes with F1 < {thresh} (last epoch):{rows_txt}"
            else:
                answer = f"All classes have F1 \u2265 {thresh} in the last epoch. \U0001f389"
        else:
            answer = "No training metrics yet."
        suggestions = ['What is the best F1 so far?', 'Which classes have F1 above 0.7?']

    # F1 above threshold
    if answer is None and _re.search(r'f1.*(above|over|greater|>)\s*([\d.]+)', q):
        m = _re.search(r'([\d.]+)', q.split('above' if 'above' in q else 'over' if 'over' in q else '>')[-1])
        thresh = float(m.group(1)) if m else 0.7
        eps = get_metrics()
        if eps:
            last = eps[-1]
            pcf1 = last.get('per_class_f1', [])
            good = [(i, v) for i, v in enumerate(pcf1) if v >= thresh]
            good.sort(key=lambda x: -x[1])
            if good:
                rows_txt = ''.join(f"<br>&nbsp;&nbsp;\u2022 <b>{CLASS_NAMES.get(i, str(i))}</b>: {pct(v)}" for i,v in good)
                answer = f"{len(good)} classes with F1 \u2265 {thresh}:{rows_txt}"
            else:
                answer = f"No classes have F1 \u2265 {thresh} yet."
        else:
            answer = "No training metrics yet."

    # Current LR
    if answer is None and any(x in q for x in ['learning rate', 'lr', 'current lr']):
        eps = get_metrics()
        cfg = get_cfg()
        if eps:
            last_lr = eps[-1].get('lr', cfg.get('lr','?'))
            answer = f"Current learning rate: <b>{last_lr:.2e}</b> (started at {cfg.get('lr','?')})"
        else:
            answer = f"Configured learning rate: <b>{cfg.get('lr', '?')}</b> (training not started yet)"
        suggestions = ['What is the best F1 so far?', 'Is training running?']

    # Co-occurrence
    if answer is None and any(x in q for x in ['co-occur', 'appear with', 'with', 'together']):
        cls_id, cls_name = find_class_id(q)
        if cls_id is not None:
            from collections import Counter
            co = Counter()
            for lbls in rows.values():
                if cls_id in lbls:
                    for l in lbls:
                        if l != cls_id: co[l] += 1
            total_with = sum(1 for lbls in rows.values() if cls_id in lbls)
            top_co = co.most_common(5)
            rows_txt = ''.join(f"<br>&nbsp;&nbsp;\u2022 <b>{CLASS_NAMES.get(c, str(c))}</b>: {n} times ({pct(n/max(total_with,1))})" for c,n in top_co)
            answer = f"Classes most co-occurring with <b>{cls_name}</b> ({fmt_num(total_with)} images):{rows_txt}"
        else:
            answer = "Please specify a class name. E.g. 'What classes co-occur with Nucleoplasm?'"
        suggestions = ['Which class has most samples?', 'What is the rarest class?']

    # Specific class info
    if answer is None and any(x in q for x in ['what is class', 'tell me about', 'info about', 'class info', 'describe']):
        cls_id, cls_name = find_class_id(q)
        if cls_id is not None:
            counts = get_class_counts()
            n = counts.get(cls_id, 0)
            total = len(rows)
            eps = get_metrics()
            f1_txt = ''
            if eps:
                last_pcf1 = eps[-1].get('per_class_f1', [])
                if cls_id < len(last_pcf1):
                    f1_txt = f"<br>Last epoch F1: <b>{pct(last_pcf1[cls_id])}</b>"
            answer = (f"<b>Class {cls_id}: {cls_name}</b><br>"
                      f"Samples: <b>{fmt_num(n)}</b> ({pct(n/total)} of training set){f1_txt}")
            suggestions = [f'What classes co-occur with {cls_name}?', 'Is training running?']
        else:
            counts = get_class_counts()
            answer = "Available classes:<br>" + '<br>'.join(f"&nbsp;{i}: {name} ({fmt_num(counts.get(i,0))})" for i,name in CLASS_NAMES.items())

    # GPU
    if answer is None and any(x in q for x in ['gpu', 'memory', 'vram', 'cuda', 'hardware']):
        try:
            import subprocess as _sp
            r = _sp.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = [x.strip() for x in r.stdout.strip().split(',')]
                answer = (f"GPU: <b>{parts[0]}</b><br>"
                          f"Utilization: <b>{parts[1]}%</b><br>"
                          f"Memory: <b>{parts[2]} / {parts[3]} MiB</b><br>"
                          f"Temperature: <b>{parts[4]}\u00b0C</b>")
            else:
                answer = "No GPU detected or nvidia-smi not available."
        except Exception as e:
            answer = f"Could not query GPU: {e}"
        suggestions = ['Is training running?', 'What is the best F1 so far?']

    # Epoch info
    if answer is None and any(x in q for x in ['epoch', 'how many epoch', 'current epoch']):
        eps = get_metrics()
        s = get_status()
        if eps:
            last = eps[-1]
            answer = (f"Completed <b>{last['epoch']}</b> of {s.get('total','?')} epochs.<br>"
                      f"Last train loss: <b>{last.get('train_loss',0):.4f}</b><br>"
                      f"Last val loss: <b>{last.get('val_loss',0):.4f}</b><br>"
                      f"Last macro F1: <b>{pct(last.get('macro_f1',0))}</b>")
        else:
            answer = "No epochs completed yet."
        suggestions = ['What is the best F1 so far?', 'Is training running?']

    # Model info
    if answer is None and any(x in q for x in ['model', 'architecture', 'which model', 'backbone']):
        cfg = get_cfg()
        s = get_status()
        if cfg:
            answer = (f"Model: <b>{cfg.get('model','?')}</b><br>"
                      f"Image size: <b>{cfg.get('img_size','?')}px</b><br>"
                      f"Batch size: <b>{cfg.get('batch_size','?')}</b><br>"
                      f"Pretrained: <b>{'Yes' if cfg.get('pretrained') else 'No'}</b>")
        else:
            answer = "No training config found yet."
        suggestions = ['Is training running?', 'What is the best F1 so far?']

    # Help / fallback
    if answer is None:
        answer = ("I can answer questions about:<br>"
                  "\u2022 <b>Dataset</b>: image counts, class distribution, multi-label stats<br>"
                  "\u2022 <b>Classes</b>: info about any of the 28 protein localization classes<br>"
                  "\u2022 <b>Co-occurrence</b>: which classes appear together<br>"
                  "\u2022 <b>Training</b>: status, epochs, loss, F1 scores<br>"
                  "\u2022 <b>GPU</b>: utilization, memory, temperature<br><br>"
                  "Try: <i>\"Which class has the most samples?\"</i>")
        suggestions = ['How many training images?', 'Is training running?', 'Which class has most samples?', 'GPU status?']

    return jsonify({'answer': answer, 'suggestions': suggestions or []})


# ══════════════════════════════════════════════════════════════════════════════
# Agent API routes
# ══════════════════════════════════════════════════════════════════════════════
AGENT_SCRIPT  = BASE / 'agent_hpa.py'
AGENT_STATUS  = LOGS_DIR / 'agent_status.json'
AGENT_LOG     = LOGS_DIR / 'agent_log.jsonl'
AGENT_STDOUT  = LOGS_DIR / 'agent_stdout.log'
AGENT_PID_FILE = LOGS_DIR / 'agent.pid'
_agent_proc   = None  # subprocess handle (valid only in this dashboard process)


def _pid_alive(pid):
    """Check if a PID is alive without killing it."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _agent_pid_from_file():
    """Read PID saved by the agent itself, survives dashboard restarts."""
    if AGENT_PID_FILE.exists():
        try:
            return int(AGENT_PID_FILE.read_text().strip())
        except Exception:
            pass
    # Fallback: read from agent_status.json written by the agent
    if AGENT_STATUS.exists():
        try:
            return int(json.load(open(AGENT_STATUS)).get('agent_pid', 0))
        except Exception:
            pass
    return None


def _agent_is_running():
    global _agent_proc
    # 1. Check our own subprocess handle
    if _agent_proc is not None and _agent_proc.poll() is None:
        return True
    # 2. Reconnect via saved PID (survives dashboard restart)
    pid = _agent_pid_from_file()
    if pid and _pid_alive(pid):
        return True
    return False


def _agent_running_pid():
    """Return the PID of the running agent, or None."""
    global _agent_proc
    if _agent_proc is not None and _agent_proc.poll() is None:
        return _agent_proc.pid
    pid = _agent_pid_from_file()
    if pid and _pid_alive(pid):
        return pid
    return None


@app.route('/api/agent/status')
def api_agent_status():
    running = _agent_is_running()
    pid = _agent_running_pid()
    status = {'running': running, 'pid': pid}
    if AGENT_STATUS.exists():
        try:
            with open(AGENT_STATUS) as f:
                status.update(json.load(f))
        except Exception:
            pass
    # Override running field with live check (agent_status.json may lag)
    status['running'] = running
    return jsonify(status)


@app.route('/api/agent/log')
def api_agent_log():
    n = int(request.args.get('n', 50))
    if not AGENT_LOG.exists():
        return jsonify({'entries': []})
    entries = []
    try:
        with open(AGENT_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return jsonify({'entries': entries[-n:]})


@app.route('/api/agent/start', methods=['POST'])
def api_agent_start():
    global _agent_proc
    pid = _agent_running_pid()
    if pid:
        return jsonify({'status': 'already_running', 'pid': pid})
    cmd = (f'source {CONDA_INIT} && conda activate {CONDA_ENV} && '
           f'python {AGENT_SCRIPT}')
    out = open(AGENT_STDOUT, 'a')   # append so history is not lost on restart
    _agent_proc = subprocess.Popen(
        ['bash', '-c', cmd], stdout=out, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    out.close()  # parent closes after fork; child inherits the fd
    # Save PID immediately so reconnect works after dashboard restart
    AGENT_PID_FILE.write_text(str(_agent_proc.pid))
    return jsonify({'status': 'started', 'pid': _agent_proc.pid})


@app.route('/api/agent/stop', methods=['POST'])
def api_agent_stop():
    global _agent_proc
    pid = _agent_running_pid()
    if not pid:
        return jsonify({'status': 'not_running'})
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception as e:
        return jsonify({'status': 'error', 'detail': str(e)})
    # Clear PID file so dashboard knows it's stopped
    if AGENT_PID_FILE.exists():
        AGENT_PID_FILE.unlink()
    return jsonify({'status': 'stopped'})


# ══════════════════════════════════════════════════════════════════════════════
# Post Analysis API
# ══════════════════════════════════════════════════════════════════════════════
RUNS_DIR = LOGS_DIR / 'runs'


def _load_run(run_dir):
    """Load a single archived run. Returns dict or None."""
    try:
        meta   = json.loads((run_dir / 'meta.json').read_text())
        cfg    = json.loads((run_dir / 'config.json').read_text())
        epochs = []
        for line in (run_dir / 'metrics.jsonl').read_text().splitlines():
            line = line.strip()
            if line:
                try: epochs.append(json.loads(line))
                except: pass
        if not epochs:
            return None
        return {'meta': meta, 'cfg': cfg, 'epochs': epochs}
    except Exception:
        return None


def _analyze_run_pa(run):
    """Compute analytics for the post-analysis view."""
    epochs = run['epochs']
    cfg    = run['cfg']
    meta   = run['meta']

    f1s    = [e['macro_f1'] for e in epochs]
    vloss  = [e['val_loss'] for e in epochs]
    tloss  = [e['train_loss'] for e in epochs]

    best_f1  = max(f1s)
    best_ep  = f1s.index(best_f1) + 1
    pcf1_at_best = epochs[best_ep - 1].get('per_class_f1', [])

    # Overfit ratio: avg(val_loss last 20%) / avg(train_loss last 20%)
    tail = max(1, len(epochs) // 5)
    avg_vl_tail = float(np.mean(vloss[-tail:]))
    avg_tl_tail = float(np.mean(tloss[-tail:]))
    overfit_ratio = round(avg_vl_tail / (avg_tl_tail + 1e-9), 3)

    # Convergence speed: epochs to reach 80% of best F1
    target = 0.8 * best_f1
    conv_ep = next((e['epoch'] for e in epochs if e['macro_f1'] >= target), len(epochs))

    # Val trend (last 25%): negative = improving
    qt = max(1, len(epochs) // 4)
    val_trend = round(float(np.mean(vloss[-qt:]) - np.mean(vloss[:qt])), 5)

    # Verdict
    if overfit_ratio > 1.4:
        verdict = 'overfit'
    elif best_f1 > 0.28:
        verdict = 'good'
    elif best_f1 > 0.18:
        verdict = 'ok'
    else:
        verdict = 'weak'

    return {
        'run_id':       meta.get('run_id', ''),
        'model':        cfg.get('model', '?'),
        'lr':           cfg.get('lr', 0),
        'bs':           cfg.get('batch_size', 0),
        'epochs_done':  len(epochs),
        'best_f1':      round(best_f1, 4),
        'best_ep':      best_ep,
        'overfit_ratio': overfit_ratio,
        'conv_ep':      conv_ep,
        'val_trend':    val_trend,
        'verdict':      verdict,
        'pcf1_at_best': [round(v, 4) for v in pcf1_at_best],
        'f1_curve':     [round(v, 4) for v in f1s],
        'val_loss_curve': [round(v, 5) for v in vloss],
        'train_loss_curve': [round(v, 5) for v in tloss],
    }


@app.route('/api/postanalysis/data')
def api_postanalysis_data():
    if not RUNS_DIR.exists():
        return jsonify({'runs': [], 'summary': {}, 'class_avg_f1': [], 'recommendations': []})

    runs_raw = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir():
            r = _load_run(d)
            if r:
                runs_raw.append(r)

    if not runs_raw:
        return jsonify({'runs': [], 'summary': {}, 'class_avg_f1': [], 'recommendations': []})

    analyzed = [_analyze_run_pa(r) for r in runs_raw]

    # Summary stats
    best_run = max(analyzed, key=lambda x: x['best_f1'])
    total_epochs = sum(a['epochs_done'] for a in analyzed)

    # Per-class average F1 across all runs (at each run's best epoch)
    nc = 28
    class_sums  = [0.0] * nc
    class_count = [0]   * nc
    for a in analyzed:
        for i, v in enumerate(a['pcf1_at_best'][:nc]):
            class_sums[i]  += v
            class_count[i] += 1
    class_avg_f1 = [round(class_sums[i] / class_count[i], 4) if class_count[i] else 0.0
                    for i in range(nc)]

    # Recommendations
    recs = []
    if best_run['overfit_ratio'] > 1.35:
        recs.append('Strong overfitting detected in best run — try stronger augmentation, weight decay, or dropout.')
    if best_run['conv_ep'] < best_run['epochs_done'] * 0.4:
        recs.append('Model converges quickly — consider shorter runs or larger learning rate schedules.')
    hard_classes = [i for i, v in enumerate(class_avg_f1) if v < 0.05]
    if hard_classes:
        recs.append(f'{len(hard_classes)} classes have avg F1 < 0.05 — consider focal loss or targeted oversampling.')
    models_tried = list({a['model'] for a in analyzed})
    if len(models_tried) < 3:
        recs.append(f'Only {len(models_tried)} model(s) tested — try EfficientNet-B2/B3 or ResNet101 for higher capacity.')
    if best_run['best_f1'] < 0.25:
        recs.append('Overall F1 still below 0.25 — dataset is challenging; verify label quality and consider test-time augmentation.')
    if not recs:
        recs.append('Training looks healthy. Consider ensembling top-2 runs for further gains.')

    return jsonify({
        'runs': analyzed,
        'summary': {
            'total_runs':   len(analyzed),
            'best_f1':      best_run['best_f1'],
            'best_model':   best_run['model'],
            'total_epochs': total_epochs,
            'best_overfit': best_run['overfit_ratio'],
            'best_run_id':  best_run['run_id'],
        },
        'class_avg_f1': class_avg_f1,
        'recommendations': recs,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPA Workstation Dashboard')
    parser.add_argument('--port',  type=int, default=8768)
    parser.add_argument('--host',  type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"[HPA Dashboard] http://{args.host}:{args.port}")
    print(f"  Train dir:  {TRAIN_DIR}")
    print(f"  Logs dir:   {LOGS_DIR}")
    print(f"  Train CSV:  {TRAIN_CSV}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
