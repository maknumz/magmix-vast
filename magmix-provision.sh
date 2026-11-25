#!/bin/bash
# =====================================================================
#  MagMix — Krea 2 + LoRA + หน้าแอป MagMix Studio สำหรับ vast.ai
#  ใช้กับเทมเพลท ComfyUI ทางการของ vast (image: vastai/comfyui)
#  vast จะรันสคริปต์นี้ให้เองตอนเปิดเครื่องครั้งแรก (ตัวแปร PROVISIONING_SCRIPT)
#  ไม่ต้องใช้ HF token — ไฟล์ทั้งหมดเป็น public
# =====================================================================
set -uo pipefail

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
M="$COMFY/models"
mkdir -p "$M/diffusion_models" "$M/text_encoders" "$M/vae" "$M/loras" "$COMFY/custom_nodes"

FAILED=0
dl() {  # dl <url> <ไฟล์ปลายทาง>
  local url="$1" dest="$2" name
  name="$(basename "$dest")"
  if [ -s "$dest" ]; then echo "✓ มีแล้ว: $name"; return 0; fi
  echo "⬇ กำลังโหลด: $name"
  for i in 1 2 3 4 5; do
    if curl -fL --retry 3 --retry-delay 5 -C - -o "$dest.part" "$url"; then
      mv "$dest.part" "$dest"; echo "✓ เสร็จ: $name"; return 0
    fi
    echo "  โหลดสะดุด ลองใหม่ครั้งที่ $i ..."; sleep 5
  done
  echo "✗ โหลดไม่สำเร็จ: $name ($url)"; return 1
}

# ---------- โมเดล Krea 2 Turbo (Comfy-Org) ----------
K="https://huggingface.co/Comfy-Org/Krea-2/resolve/main"
dl "$K/diffusion_models/krea2_turbo_fp8_scaled.safetensors" "$M/diffusion_models/krea2_turbo_fp8_scaled.safetensors" & P1=$!
dl "$K/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"     "$M/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"     & P2=$!
dl "$K/vae/qwen_image_vae.safetensors"                       "$M/vae/qwen_image_vae.safetensors"                       & P3=$!

# ---------- LoRA ของ MagMix ----------
L="https://huggingface.co/Khanes"
dl "$L/kreansfw/resolve/main/NSFW.safetensors" "$M/loras/NSFW.safetensors" & P4=$!
dl "$L/kreapro/resolve/main/Pro.safetensors"   "$M/loras/Pro.safetensors"  & P5=$!
dl "$L/90s/resolve/main/90s.safetensors"       "$M/loras/90s.safetensors"  & P6=$!
dl "$L/Krea2Film/resolve/main/FILM.safetensors" "$M/loras/FILM.safetensors" & P7=$!

for p in $P1 $P2 $P3 $P4 $P5 $P6 $P7; do wait "$p" || FAILED=1; done

# ---------- หน้าแอป MagMix Studio (เปิดที่ /magmix) ----------
cat > "$COMFY/custom_nodes/magmix_studio.py" <<'MAGMIX_PLUGIN_EOF'
"""MagMix Studio — หน้าแอปครอบ ComfyUI สำหรับ Krea 2
วางไฟล์นี้ใน ComfyUI/custom_nodes/ แล้วรีสตาร์ต ComfyUI
เปิดใช้: http(s)://<ที่อยู่ ComfyUI>/magmix
"""
import os
import re
import time
import uuid
import asyncio
import json
import urllib.parse

from aiohttp import web, ClientSession, ClientTimeout
from server import PromptServer
import folder_paths

HTML = r'''<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MagMix Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0d0d0d; --card:#161616; --card2:#1e1e1e; --card3:#252525; --line:#262626; --line2:#363636;
  --text:#e6e6e6; --muted:#8c8c8c; --accent:#e8192c; --accent2:#ff3040; --accent-dim:#3a0d12;
  --knob:#ffffff; --ok:#3fbf7f; --err:#ff4d5e;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
  font-family:"DM Sans","Noto Sans Thai",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:14px}
button{font-family:inherit}
.app{display:grid;grid-template-columns:340px minmax(0,1fr) 400px;gap:14px;height:100vh;padding:14px}
.col{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:12px;padding-right:2px}
.col::-webkit-scrollbar,.gscroll::-webkit-scrollbar,.lmenu::-webkit-scrollbar{width:6px}
.col::-webkit-scrollbar-thumb,.gscroll::-webkit-scrollbar-thumb,.lmenu::-webkit-scrollbar-thumb{background:#333;border-radius:3px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h3{margin:0 0 12px;font-size:16px;font-weight:700;display:flex;align-items:center;justify-content:space-between;gap:8px}
.card h3 small{color:var(--muted);font-weight:500;font-size:13px;margin-left:6px}
.brand{display:flex;align-items:center;gap:10px;padding:2px 4px 0}
.brand b{font-size:20px;font-weight:800;letter-spacing:.5px}
.brand span{background:var(--accent);color:#fff;font-size:11px;font-weight:700;letter-spacing:.6px;padding:3px 7px;border-radius:5px;text-transform:uppercase}
.brand small{color:var(--muted);font-size:12px;margin-left:auto}
select,textarea,input[type=number],input[type=text],input[type=password]{width:100%;background:var(--card2);color:var(--text);border:1px solid var(--line2);
  border-radius:10px;padding:11px 12px;font-size:15px;font-family:inherit;outline:none}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),linear-gradient(135deg,var(--muted) 50%,transparent 50%);
  background-position:calc(100% - 18px) 50%,calc(100% - 13px) 50%;background-size:5px 5px;background-repeat:no-repeat;padding-right:34px}
select:focus,textarea:focus,input:focus{border-color:var(--accent)}
.row{display:flex;align-items:center;justify-content:space-between;color:var(--muted);margin:14px 0 6px;font-size:14px}
.row b{color:var(--text);font-weight:600}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:22px;background:transparent;margin:0;cursor:pointer;--p:50%}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--accent) var(--p),#2c2c2c var(--p))}
input[type=range]::-moz-range-track{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--accent) var(--p),#2c2c2c var(--p))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:var(--knob);margin-top:-7px;
  box-shadow:0 0 0 3px rgba(232,25,44,.35),0 1px 4px rgba(0,0,0,.6);border:0}
input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--knob);border:0;box-shadow:0 0 0 3px rgba(232,25,44,.35)}
.aspects{display:flex;gap:6px;overflow-x:auto;padding:12px 2px 6px;scrollbar-width:none}
.aspects::-webkit-scrollbar{display:none}
.chip{flex:0 0 auto;width:64px;border-radius:12px;padding:8px 4px 6px;display:flex;flex-direction:column;align-items:center;gap:6px;
  cursor:pointer;color:var(--muted);user-select:none;border:1px solid transparent}
.chip .box{border:1.5px solid #666;border-radius:3px}
.chip span{font-size:14px}
.chip:hover{color:var(--text)}
.chip.on{background:var(--accent-dim);color:#fff;border-color:var(--accent)}
.chip.on .box{border-color:#fff;border-width:2px}
.seg{display:flex;background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:3px;margin-top:6px}
.seg button{flex:1;background:transparent;border:0;color:var(--muted);font-size:15px;padding:9px 0;border-radius:9px;cursor:pointer}
.seg button:hover{color:var(--text)}
.seg button.on{background:var(--accent);color:#fff;font-weight:700}
.seg.big button{font-size:20px;padding:10px 0}
.lora{border-top:1px solid var(--line);padding-top:14px;margin-top:14px}
.lora:first-of-type{border-top:0;margin-top:0;padding-top:0}
.lora-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;font-size:16px;font-weight:700}
.iconbtn{width:40px;height:40px;border-radius:10px;background:var(--card2);border:1px solid var(--line2);color:var(--text);
  font-size:22px;line-height:1;cursor:pointer;display:grid;place-items:center;flex:0 0 auto}
.iconbtn:hover{border-color:var(--accent);color:#fff}
.pill{background:var(--card2);border:1px solid var(--line2);color:var(--text);border-radius:10px;padding:8px 12px;font-size:14px;
  cursor:pointer;display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.pill:hover{border-color:var(--accent)}
.pill.red{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700}
.pill.red:hover{background:var(--accent2)}
.pill.danger:hover{border-color:var(--err);color:#ffb3ba}
.pill.on{border-color:var(--accent);background:var(--accent-dim);color:#fff}
.pill:disabled{opacity:.5;cursor:not-allowed}
.muted{color:var(--muted)}
.check{display:flex;align-items:center;gap:8px;color:var(--muted);margin-top:8px;cursor:pointer;user-select:none}
.check input{accent-color:var(--accent)}
.seedrow{display:flex;gap:8px}.seedrow input{flex:1}
textarea{min-height:150px;resize:vertical;line-height:1.5}
.gen{width:100%;margin-top:12px;background:var(--accent);border:0;color:#fff;font-size:17px;font-weight:700;padding:14px;border-radius:12px;cursor:pointer}
.gen:hover{background:var(--accent2)}.gen:disabled{opacity:.55;cursor:wait}
.bar{height:6px;background:#2a2a2a;border-radius:3px;overflow:hidden;margin-top:12px}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
.status{margin-top:8px;color:var(--muted);font-size:13px;min-height:18px;white-space:pre-wrap}
.status.err{color:var(--err)}
.stage{flex:1;min-height:420px;display:flex;align-items:center;justify-content:center;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;position:relative}
.stage > img{width:100%;height:100%;object-fit:contain;display:block}
.stage img.preview{opacity:.75;filter:saturate(.8)}
.stage .empty{color:var(--muted);text-align:center;padding:30px;line-height:1.7}
.stage .tools{position:absolute;right:10px;top:10px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.stage .tools a,.stage .tools button{background:rgba(0,0,0,.6);color:#fff;text-decoration:none;padding:7px 11px;border-radius:9px;font-size:13px;border:1px solid rgba(255,255,255,.12);cursor:pointer}
.stage .tools a:hover,.stage .tools button:hover{background:var(--accent);border-color:var(--accent)}
.stage .multi{display:grid;grid-template-columns:repeat(2,1fr);grid-auto-rows:1fr;gap:8px;width:100%;height:100%;padding:8px}
.stage .multi img{width:100%;height:100%;object-fit:contain;background:#0a0a0a;border-radius:10px;cursor:zoom-in;min-height:0}
.meta{color:var(--muted);font-size:12px;text-align:center;line-height:1.6}
.meta button{background:none;border:0;color:var(--accent2);cursor:pointer;font-size:12px;padding:0 4px;text-decoration:underline}
/* ---- gallery ---- */
#c3{overflow:hidden}
.gcard{flex:1;min-height:0;display:flex;flex-direction:column;padding-bottom:10px}
.gbar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;min-height:38px;margin-bottom:10px}
.gbar .hint{flex:1}
.gscroll{flex:1;min-height:0;overflow:auto;margin:0 -6px;padding:0 6px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.grid .cell{position:relative;aspect-ratio:3/4;border-radius:10px;overflow:hidden;background:var(--card2);cursor:pointer;border:2px solid transparent}
.grid .cell:hover{border-color:#444}
.grid .cell.on{border-color:var(--accent)}
.grid .cell.up::after{content:"UPSCALE";position:absolute;left:6px;bottom:6px;background:rgba(0,0,0,.7);color:#fff;font-size:10px;font-weight:700;letter-spacing:.5px;padding:2px 6px;border-radius:5px}
.grid img{width:100%;height:100%;object-fit:cover;display:block}
.grid .tick{display:none;position:absolute;right:7px;top:7px;width:24px;height:24px;border-radius:50%;border:2px solid #fff;background:rgba(0,0,0,.45);
  color:#fff;font-size:14px;font-weight:800;place-items:center}
.selmode .grid .tick{display:grid}
.grid .cell.picked{border-color:var(--accent)}
.grid .cell.picked .tick{background:var(--accent);border-color:var(--accent)}
.gempty{color:var(--muted);text-align:center;padding:40px 10px;line-height:1.7}
.upbox{background:var(--card2);border:1px solid var(--accent);border-radius:12px;padding:14px;margin-bottom:10px}
.upbox .uphead{display:flex;gap:12px;align-items:center}
.upbox .uphead img{width:56px;height:72px;object-fit:cover;border-radius:8px;flex:0 0 auto}
.upbox .uphead b{display:block;font-size:15px}
.upbox .uphead div{font-size:13px;color:var(--muted);line-height:1.5}
.upbox .btns{display:flex;gap:8px;margin-top:12px}
.upbox .btns .pill{flex:1;justify-content:center;padding:10px}
/* ---- LoRA picker ---- */
.lbtn{width:100%;display:flex;align-items:center;gap:10px;background:var(--card2);border:1px solid var(--line2);border-radius:10px;
  padding:7px 12px 7px 7px;color:var(--text);font-size:15px;cursor:pointer;text-align:left;min-height:46px}
.lbtn:hover,.lbtn.open{border-color:var(--accent)}
.lbtn .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lbtn .nm.off{color:var(--muted)}
.lbtn .car{color:var(--muted);font-size:11px}
.lic{width:32px;height:32px;border-radius:7px;object-fit:cover;flex:0 0 auto;display:block}
.lic.none{display:grid;place-items:center;background:var(--card3);color:var(--muted);font-size:15px}
.lbtn .noicon{width:5px}
.lmenu{position:fixed;z-index:60;background:#1a1a1a;border:1px solid var(--line2);border-radius:12px;padding:6px;overflow:auto;
  box-shadow:0 12px 32px rgba(0,0,0,.6);display:none}
.lmenu.open{display:block}
.lmenu .tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;padding:2px 2px 6px}
.lmenu .tile{display:flex;flex-direction:column;align-items:stretch;gap:6px;background:var(--card2);border:2px solid transparent;border-radius:10px;padding:6px;cursor:pointer;color:var(--text);font-size:14px;font-weight:600}
.lmenu .tile img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:7px;display:block}
.lmenu .tile:hover{border-color:#555}
.lmenu .tile.on{border-color:var(--accent)}
.lmenu .tile.used{opacity:.45}
.lmenu .item{display:flex;align-items:center;gap:10px;padding:10px 10px;border-radius:8px;cursor:pointer;font-size:14px;border:0;background:none;color:var(--text);width:100%;text-align:left}
.lmenu .item:hover{background:var(--card3)}
.lmenu .item.on{background:var(--accent-dim);color:#fff}
.lmenu .item.used{opacity:.45}
.lmenu .sep{color:var(--muted);font-size:12px;padding:8px 10px 4px}
/* ---- misc ---- */
.banner{background:#2a0f12;border:1px solid #6b1c25;color:#ffb3ba;border-radius:12px;padding:10px 12px;font-size:13px;display:none;white-space:pre-wrap}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.65);display:none;align-items:center;justify-content:center;z-index:70;padding:16px}
.modal.open{display:flex}
.dialog{background:var(--card);border:1px solid var(--line2);border-radius:16px;width:min(560px,100%);max-height:90vh;overflow:auto;padding:20px}
.dialog.small{width:min(420px,100%)}
.dialog h3{margin:0 0 4px;font-size:19px;display:flex;justify-content:space-between;align-items:center}
.dialog h4{margin:18px 0 8px;font-size:15px}
.dialog .btns{display:flex;gap:8px;justify-content:flex-end;margin-top:18px}
.dialog .btns .pill{padding:10px 18px}
.hint{color:var(--muted);font-size:13px;line-height:1.6}
.urlrow{display:flex;gap:8px}.urlrow input{flex:1}
.btn{background:var(--accent);border:0;color:#fff;font-weight:700;border-radius:10px;padding:0 16px;cursor:pointer;font-size:15px}
.btn:hover{background:var(--accent2)}
.drop{display:block;border:2px dashed #3a3a3a;border-radius:14px;padding:22px;text-align:center;color:var(--muted);cursor:pointer;line-height:1.6}
.drop.over{border-color:var(--accent);color:var(--text);background:var(--accent-dim)}
.job{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-top:8px;font-size:13px}
.job .t{display:flex;justify-content:space-between;gap:8px}
.job .t b{font-weight:600;word-break:break-all}
.job .bar{margin-top:8px;height:5px}
.job.done{border-color:#22543a}.job.err{border-color:#6b1c25}
.job .e{color:var(--err);margin-top:6px;white-space:pre-wrap}
details summary{cursor:pointer;color:var(--muted);margin-top:16px}
details .row{margin-top:10px}
@media (max-width:1100px){.app{grid-template-columns:1fr;height:auto}.col,#c3{overflow:visible}.stage{min-height:360px}.gscroll{max-height:75vh}}
</style>
</head>
<body>
<div class="app">
  <div class="col" id="c1">
    <div class="brand"><b>MAGMIX</b><span>Studio</span><small>Krea 2 · ComfyUI</small></div>
    <div class="banner" id="banner"></div>

    <div class="card">
      <h3><span>LoRA</span><button class="pill" id="openImport" title="เพิ่มไฟล์ LoRA เข้าเครื่อง">⬆ Import LoRA</button></h3>
      <div id="loraList"></div>
    </div>

    <div class="card">
      <h3>Image Size</h3>
      <select id="preset"></select>
      <div class="aspects" id="aspects"></div>
      <div class="seg" id="tiers"></div>
      <div class="row"><span>Width</span><b id="wv"></b></div>
      <input type="range" id="w" min="512" max="2048" step="16">
      <div class="row"><span>Height</span><b id="hv"></b></div>
      <input type="range" id="h" min="512" max="2048" step="16">
    </div>

    <div class="card">
      <h3>Batch Size</h3>
      <div class="seg big" id="batch"></div>
    </div>

    <div class="card">
      <h3>Sampler</h3>
      <div class="row" style="margin-top:0"><span>Steps</span><b id="stv"></b></div>
      <input type="range" id="steps" min="1" max="50" step="1">
      <div class="row"><span>CFG</span><b id="cfv"></b></div>
      <input type="range" id="cfg" min="1" max="10" step="0.1">
      <div class="row"><span>Sampler</span></div>
      <select id="sampler"></select>
      <div class="row"><span>Scheduler</span></div>
      <select id="scheduler"></select>
      <div class="row"><span>Seed</span></div>
      <div class="seedrow"><input type="number" id="seed" min="0"><button class="iconbtn" id="dice" title="สุ่ม seed">🎲</button></div>
      <label class="check"><input type="checkbox" id="rand"> สุ่ม seed ใหม่ทุกครั้งที่กดสร้าง</label>
    </div>

    <div class="card">
      <h3>Models</h3>
      <div class="row" style="margin-top:0"><span>Diffusion model</span></div>
      <select id="unet"></select>
      <div class="row"><span>Text encoder</span></div>
      <select id="clip"></select>
      <div class="row"><span>VAE</span></div>
      <select id="vae"></select>
    </div>
  </div>

  <div class="col" id="c2">
    <div class="card">
      <h3><span>Prompt</span><button class="pill" id="copyPrompt" title="คัดลอก prompt">📋 คัดลอก</button></h3>
      <textarea id="prompt" placeholder="พิมพ์สิ่งที่อยากได้ เช่น a young Thai woman, long hair, soft smile, cozy room, night vibe"></textarea>
      <button class="gen" id="go">✨ สร้างภาพ</button>
      <div class="bar"><i id="bar"></i></div>
      <div class="status" id="status"></div>
    </div>
    <div class="stage" id="stage"><div class="empty">ภาพที่สร้างจะแสดงตรงนี้<br><span style="font-size:12px">กด ✨ สร้างภาพ หรือ Ctrl/⌘ + Enter</span></div></div>
    <div class="meta" id="meta"></div>
  </div>

  <div class="col" id="c3">
    <div class="card gcard" id="gcard">
      <h3><span>Gallery<small id="gcount"></small></span><button class="pill" id="selMode">☑ เลือกหลายรูป</button></h3>
      <div class="gbar" id="gbar"></div>
      <div class="upbox" id="upbox" hidden></div>
      <div class="gscroll" id="gscroll"><div class="grid" id="grid"></div></div>
    </div>
  </div>
</div>

<div class="lmenu" id="lmenu"></div>

<div class="modal" id="confirmModal">
  <div class="dialog small">
    <h3 id="cfTitle"></h3>
    <div class="hint" id="cfText" style="margin-top:8px;font-size:14px"></div>
    <div class="btns"><button class="pill" id="cfNo">ยกเลิก</button><button class="pill red" id="cfYes">ลบ</button></div>
  </div>
</div>

<div class="modal" id="importModal">
  <div class="dialog">
    <h3>Import LoRA <button class="iconbtn" id="closeImport" title="ปิด">×</button></h3>
    <div class="hint">ไฟล์จะถูกเก็บในเครื่อง vast (ComfyUI/models/loras) แล้วเลือกใช้ได้ทันที</div>

    <h4>วางลิงก์ huggingface.co หรือ civitai (.com / .red)</h4>
    <div class="urlrow"><input type="text" id="impUrl" placeholder="https://civitai.com/models/12345  หรือ  https://huggingface.co/…/xxx.safetensors"><button class="btn" id="impGo">ดาวน์โหลด</button></div>
    <div class="hint" style="margin-top:6px">• Civitai: ใช้ลิงก์หน้าโมเดลได้เลย ทั้ง civitai.com และ civitai.red (ถ้าเลือกเวอร์ชันไว้ จะโหลดเวอร์ชันนั้น)<br>• HuggingFace: เปิดหน้าไฟล์ .safetensors แล้วก๊อปลิงก์</div>

    <h4>หรืออัปโหลดไฟล์จากเครื่องคุณ</h4>
    <label class="drop" id="drop"><input type="file" id="impFile" accept=".safetensors" multiple hidden>
      ลากไฟล์ <b>.safetensors</b> มาวางตรงนี้ หรือคลิกเพื่อเลือกไฟล์</label>

    <div id="jobs"></div>

    <details>
      <summary>ตัวเลือกเพิ่มเติม (ไฟล์ที่ต้องล็อกอินก่อนโหลด)</summary>
      <div class="row"><span>Civitai API key</span></div>
      <input type="password" id="civKey" placeholder="ใส่เฉพาะเมื่อ Civitai แจ้งว่าต้องล็อกอิน">
      <div class="row"><span>HuggingFace token</span></div>
      <input type="password" id="hfTok" placeholder="ใส่เฉพาะไฟล์ใน repo ที่ล็อกไว้">
      <div class="hint" style="margin-top:6px">เก็บไว้ในเบราว์เซอร์นี้เท่านั้น และส่งไปที่เครื่อง vast ของคุณตอนกดดาวน์โหลด</div>
    </details>
  </div>
</div>

<script>
(() => {
"use strict";
const API = location.origin + "/api";
const BASE = location.origin + "/magmix";
const $ = (id) => document.getElementById(id);
const CLIENT_ID = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));
const MAX_LORA = 6;
const CHUNK = 32 * 1024 * 1024;
const ICONS = {"90s":"data:image/jpeg;base64,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","film":"data:image/jpeg;base64,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","nsfw":"data:image/jpeg;base64,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","pro":"data:image/jpeg;base64,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"};          // ไอคอน LoRA ของ MagMix (ชื่อไฟล์ → รูป)
const UP_MAX = 4096;              // ด้านยาวสุดของภาพหลัง Upscale

const ASPECTS = [
  ["21:9", 1904, 816], ["16:9", 1648, 928], ["3:2", 1536, 1024], ["4:3", 1408, 1088], ["5:4", 1376, 1104],
  ["1:1", 1216, 1216], ["4:5", 1104, 1376], ["3:4", 1088, 1408], ["2:3", 1024, 1536], ["9:16", 928, 1648]
];
const TIERS = [["Small", 0.75], ["Normal", 1], ["Large", 1.25]];
const r16 = (v) => Math.max(512, Math.min(2048, Math.round(v / 16) * 16));
const dims = (a, t) => {
  const A = ASPECTS.find(x => x[0] === a), T = TIERS.find(x => x[0] === t);
  return [r16(A[1] * T[1]), r16(A[2] * T[1])];
};

const DEF = {
  prompt: "", aspect: "3:4", tier: "Normal", width: 1088, height: 1408, batch: 1,
  loras: [{ name: "Disabled", w: 100 }], steps: 8, cfg: 1, sampler: "euler", scheduler: "simple",
  seed: Math.floor(Math.random() * 1e14), rand: true, unet: "", clip: "", vae: "", civKey: "", hfTok: "",
  upScale: 2, upDen: 35
};
let S = Object.assign({}, DEF);
try { Object.assign(S, JSON.parse(localStorage.getItem("magmix_studio") || "{}")); } catch (e) {}
if (![1, 2, 3, 4].includes(S.batch)) S.batch = 1;
if (![1.5, 2].includes(S.upScale)) S.upScale = 2;
if (!Array.isArray(S.loras) || !S.loras.length) S.loras = [{ name: "Disabled", w: 100 }];
const save = () => { try { localStorage.setItem("magmix_studio", JSON.stringify(S)); } catch (e) {} };
let LISTS = { unet: [], clip: [], vae: [], lora: [], sampler: ["euler"], scheduler: ["simple"] };

const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fillRange = (el) => el.style.setProperty("--p", ((el.value - el.min) / (el.max - el.min) * 100) + "%");
const pretty = (v) => v.replace(/\.(safetensors|ckpt|pt|bin)$/i, "").replace(/__/g, " / ");
const opt = (sel, values, cur, nice) => {
  sel.innerHTML = "";
  values.forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = nice ? pretty(v) : v; sel.appendChild(o); });
  if (values.includes(cur)) sel.value = cur;
};
const status = (msg, err) => { $("status").textContent = msg || ""; $("status").className = "status" + (err ? " err" : ""); };
const banner = (msg) => { const b = $("banner"); b.style.display = msg ? "block" : "none"; b.textContent = msg || ""; };
const comboOptions = (spec) => {
  if (!spec) return [];
  if (Array.isArray(spec[0])) return spec[0];
  if (spec[0] === "COMBO" && spec[1] && Array.isArray(spec[1].options)) return spec[1].options;
  return [];
};
const pick = (list, cur, hints) => {
  if (list.includes(cur)) return cur;
  for (const h of hints) { const f = list.find(x => x.toLowerCase().includes(h)); if (f) return f; }
  return list[0] || "";
};
const fmtMB = (b) => (b / 1048576).toFixed(b > 1e9 ? 0 : 1) + " MB";
const newSeed = () => Math.floor(Math.random() * 1e14);

// ---------- ยืนยันก่อนลบ ----------
function confirmBox(title, text, yes) {
  return new Promise(res => {
    $("cfTitle").textContent = title; $("cfText").textContent = text; $("cfYes").textContent = yes || "ตกลง";
    $("confirmModal").classList.add("open");
    const done = (v) => { $("confirmModal").classList.remove("open"); $("cfYes").onclick = $("cfNo").onclick = null; res(v); };
    $("cfYes").onclick = () => done(true); $("cfNo").onclick = () => done(false);
  });
}

// ---------- LoRA ----------
const iconOf = (name) => {
  if (!name || name === "Disabled") return null;
  const stem = name.split(/[\\/]/).pop().replace(/\.[^.]+$/, "").toLowerCase();
  return ICONS[stem] || null;
};
let menuFor = -1;
function closeMenu() {
  $("lmenu").classList.remove("open"); menuFor = -1;
  document.querySelectorAll(".lbtn.open").forEach(b => b.classList.remove("open"));
}
function openMenu(i, btn) {
  if (menuFor === i) return closeMenu();
  closeMenu(); menuFor = i; btn.classList.add("open");
  const cur = S.loras[i].name;
  const used = new Set(S.loras.filter((L, k) => k !== i).map(L => L.name));
  const withIcon = LISTS.lora.filter(iconOf), plain = LISTS.lora.filter(n => !iconOf(n));
  const cls = (n) => (n === cur ? " on" : "") + (used.has(n) ? " used" : "");
  let h = `<button class="item${cur === "Disabled" ? " on" : ""}" data-v="Disabled"><span class="lic none">⦸</span>Disabled (ไม่ใช้)</button>`;
  if (withIcon.length) h += `<div class="sep">LoRA ของ MagMix</div><div class="tiles">` +
    withIcon.map(n => `<div class="tile${cls(n)}" data-v="${esc(n)}"><img src="${iconOf(n)}" alt="">${esc(pretty(n))}</div>`).join("") + `</div>`;
  if (plain.length) h += `<div class="sep">LoRA อื่น ๆ</div>` +
    plain.map(n => `<button class="item${cls(n)}" data-v="${esc(n)}">${esc(pretty(n))}</button>`).join("");
  if (!LISTS.lora.length) h += `<div class="sep">ยังไม่มีไฟล์ LoRA — กด ⬆ Import LoRA</div>`;
  const m = $("lmenu"); m.innerHTML = h;
  m.querySelectorAll("[data-v]").forEach(el => el.onclick = (e) => {
    e.stopPropagation(); S.loras[i].name = el.dataset.v; closeMenu(); save(); renderLoras();
  });
  const r = btn.getBoundingClientRect(), below = innerHeight - r.bottom - 12, above = r.top - 12;
  m.style.left = r.left + "px"; m.style.width = Math.max(r.width, 260) + "px";
  m.classList.add("open");
  const want = Math.min(m.scrollHeight, 460);
  if (below >= Math.min(want, 300) || below >= above) { m.style.top = (r.bottom + 6) + "px"; m.style.bottom = "auto"; m.style.maxHeight = Math.max(160, below) + "px"; }
  else { m.style.top = "auto"; m.style.bottom = (innerHeight - r.top + 6) + "px"; m.style.maxHeight = above + "px"; }
}
document.addEventListener("click", (e) => { if (menuFor >= 0 && !e.target.closest("#lmenu") && !e.target.closest(".lbtn")) closeMenu(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeMenu(); if (selMode) setSelMode(false); } });
window.addEventListener("resize", closeMenu);
$("c1").addEventListener("scroll", closeMenu);

function renderLoras() {
  const box = $("loraList"); box.innerHTML = "";
  S.loras.forEach((L, i) => {
    if (L.name !== "Disabled" && LISTS.lora.length && !LISTS.lora.includes(L.name)) L.name = "Disabled";
    const d = document.createElement("div"); d.className = "lora";
    const btn = i === 0 ? `<button class="iconbtn" data-act="add" title="เพิ่มช่อง LoRA">+</button>`
                        : `<button class="iconbtn" data-act="del" title="ลบช่องนี้">−</button>`;
    const ic = iconOf(L.name), off = L.name === "Disabled";
    const lead = ic ? `<img class="lic" src="${ic}" alt="">` : off ? `<span class="lic none">⦸</span>` : `<span class="noicon"></span>`;
    d.innerHTML = `<div class="lora-head"><span>LoRA ${i + 1}</span>${btn}</div>
      <button class="lbtn" type="button">${lead}<span class="nm${off ? " off" : ""}">${esc(off ? "Disabled" : pretty(L.name))}</span><span class="car">▼</span></button>
      <div class="wrap" style="display:${off ? "none" : "block"}">
        <div class="row"><span>Weight</span><b>${Math.round(L.w)}%</b></div>
        <input type="range" min="0" max="200" step="5" value="${L.w}"></div>`;
    const lb = d.querySelector(".lbtn");
    lb.onclick = (e) => { e.stopPropagation(); openMenu(i, lb); };
    const rg = d.querySelector("input[type=range]"); fillRange(rg);
    rg.oninput = () => { L.w = Number(rg.value); d.querySelector(".row b").textContent = L.w + "%"; fillRange(rg); save(); };
    d.querySelector("[data-act]").onclick = (e) => {
      if (e.currentTarget.dataset.act === "add") { if (S.loras.length < MAX_LORA) S.loras.push({ name: "Disabled", w: 100 }); }
      else S.loras.splice(i, 1);
      closeMenu(); save(); renderLoras();
    };
    box.appendChild(d);
  });
  if (!LISTS.lora.length) {
    const p = document.createElement("div"); p.className = "hint"; p.style.marginTop = "10px";
    p.textContent = "ยังไม่มีไฟล์ LoRA ในเครื่อง — กด ⬆ Import LoRA เพื่อเพิ่ม"; box.appendChild(p);
  }
}
function useLora(name) {
  if (S.loras.some(L => L.name === name)) return;
  const slot = S.loras.find(L => L.name === "Disabled");
  if (slot) { slot.name = name; slot.w = 100; }
  else if (S.loras.length < MAX_LORA) S.loras.push({ name, w: 100 });
  save(); renderLoras();
}

// ---------- Image Size ----------
function renderSize() {
  const sel = $("preset"); sel.innerHTML = "";
  const cur = `${S.width}x${S.height}`;
  let found = false;
  TIERS.forEach(([t]) => ASPECTS.forEach(([a]) => {
    const [w, h] = dims(a, t), o = document.createElement("option");
    o.value = `${a}|${t}`; o.textContent = `${w}x${h}   ·  ${a}  ${t}`;
    if (`${w}x${h}` === cur && a === S.aspect && t === S.tier) { o.selected = true; found = true; }
    sel.appendChild(o);
  }));
  if (!found) { const o = document.createElement("option"); o.value = "custom"; o.textContent = `${cur}   ·  กำหนดเอง`; o.selected = true; sel.prepend(o); }
  const box = $("aspects"); box.innerHTML = "";
  ASPECTS.forEach(([a, w, h]) => {
    const c = document.createElement("div"), k = 34 / Math.max(w, h);
    c.className = "chip" + (a === S.aspect ? " on" : "");
    c.innerHTML = `<div class="box" style="width:${Math.round(w * k)}px;height:${Math.round(h * k)}px"></div><span>${a}</span>`;
    c.onclick = () => { S.aspect = a; if (!S.tier) S.tier = "Normal"; [S.width, S.height] = dims(a, S.tier); save(); renderSize(); };
    box.appendChild(c);
  });
  const on = box.querySelector(".chip.on"); if (on) on.scrollIntoView({ block: "nearest", inline: "center" });
  const seg = $("tiers"); seg.innerHTML = "";
  TIERS.forEach(([t]) => {
    const b = document.createElement("button"); b.textContent = t; b.className = t === S.tier ? "on" : "";
    b.onclick = () => { S.tier = t; if (!S.aspect) S.aspect = "3:4"; [S.width, S.height] = dims(S.aspect, t); save(); renderSize(); };
    seg.appendChild(b);
  });
  $("w").value = S.width; $("h").value = S.height; fillRange($("w")); fillRange($("h"));
  $("wv").textContent = Number(S.width).toLocaleString(); $("hv").textContent = Number(S.height).toLocaleString();
}
function matchSize() {
  const match = TIERS.flatMap(([t]) => ASPECTS.map(([a]) => [a, t])).find(([a, t]) => { const [w, h] = dims(a, t); return w === S.width && h === S.height; });
  if (match) { S.aspect = match[0]; S.tier = match[1]; } else { S.aspect = ""; S.tier = ""; }
}
$("preset").onchange = (e) => {
  if (e.target.value === "custom") return;
  const [a, t] = e.target.value.split("|"); S.aspect = a; S.tier = t; [S.width, S.height] = dims(a, t); save(); renderSize();
};
["w", "h"].forEach(id => $(id).addEventListener("input", (e) => {
  S[id === "w" ? "width" : "height"] = Number(e.target.value); matchSize(); save(); renderSize();
}));

// ---------- Batch Size ----------
function renderBatch() {
  const seg = $("batch"); seg.innerHTML = "";
  [1, 2, 3, 4].forEach(n => {
    const b = document.createElement("button"); b.textContent = n; b.className = n === S.batch ? "on" : "";
    b.onclick = () => { S.batch = n; save(); renderBatch(); };
    seg.appendChild(b);
  });
}

// ---------- sampler / models ----------
function renderSampler() {
  $("steps").value = S.steps; $("stv").textContent = S.steps; fillRange($("steps"));
  $("cfg").value = S.cfg; $("cfv").textContent = Number(S.cfg).toFixed(1); fillRange($("cfg"));
  opt($("sampler"), LISTS.sampler, S.sampler); opt($("scheduler"), LISTS.scheduler, S.scheduler);
  $("seed").value = S.seed; $("rand").checked = !!S.rand;
  opt($("unet"), LISTS.unet, S.unet); opt($("clip"), LISTS.clip, S.clip); opt($("vae"), LISTS.vae, S.vae);
}
$("steps").oninput = (e) => { S.steps = Number(e.target.value); $("stv").textContent = S.steps; fillRange(e.target); save(); };
$("cfg").oninput = (e) => { S.cfg = Number(e.target.value); $("cfv").textContent = S.cfg.toFixed(1); fillRange(e.target); save(); };
["sampler", "scheduler", "unet", "clip", "vae"].forEach(k => $(k).onchange = (e) => { S[k] = e.target.value; save(); });
$("seed").onchange = (e) => { S.seed = Math.max(0, Math.floor(Number(e.target.value) || 0)); save(); };
$("dice").onclick = () => { S.seed = newSeed(); $("seed").value = S.seed; save(); };
$("rand").onchange = (e) => { S.rand = e.target.checked; save(); };
const setPrompt = (t) => { S.prompt = t; $("prompt").value = t; save(); };
$("prompt").value = S.prompt || "";
$("prompt").oninput = (e) => { S.prompt = e.target.value; save(); };

// ---------- คัดลอก prompt ----------
async function copyText(t) {
  try { await navigator.clipboard.writeText(t); return true; } catch (e) {}
  const ta = document.createElement("textarea"); ta.value = t; ta.style.cssText = "position:fixed;opacity:0;left:-9999px";
  document.body.appendChild(ta); ta.select();
  let ok = false; try { ok = document.execCommand("copy"); } catch (e) {}
  ta.remove(); return ok;
}
$("copyPrompt").onclick = async () => {
  const b = $("copyPrompt"), t = $("prompt").value;
  if (!t.trim()) { b.textContent = "ยังไม่มี prompt"; }
  else b.textContent = (await copyText(t)) ? "✓ คัดลอกแล้ว" : "คัดลอกไม่ได้";
  clearTimeout(b._t); b._t = setTimeout(() => { b.textContent = "📋 คัดลอก"; }, 1600);
};

// ---------- load lists ----------
async function info(node) {
  const r = await fetch(`${API}/object_info/${node}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`object_info ${node}: HTTP ${r.status}`);
  const j = await r.json(); return (j[node] && j[node].input && j[node].input.required) || {};
}
async function reloadLoraList() {
  try { LISTS.lora = comboOptions((await info("LoraLoaderModelOnly")).lora_name); } catch (e) {}
  renderLoras();
}
async function loadLists() {
  try {
    const [u, c, v, l, k] = await Promise.all(["UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "KSampler"].map(info));
    LISTS.unet = comboOptions(u.unet_name); LISTS.clip = comboOptions(c.clip_name); LISTS.vae = comboOptions(v.vae_name);
    LISTS.lora = comboOptions(l.lora_name); LISTS.sampler = comboOptions(k.sampler_name); LISTS.scheduler = comboOptions(k.scheduler);
    LISTS.clipTypes = comboOptions(c.type);
    S.unet = pick(LISTS.unet, S.unet, ["krea2_turbo", "krea2"]);
    S.clip = pick(LISTS.clip, S.clip, ["qwen3vl_4b", "qwen3vl"]);
    S.vae = pick(LISTS.vae, S.vae, ["qwen_image_vae", "qwen"]);
    S.sampler = pick(LISTS.sampler, S.sampler, ["euler"]); S.scheduler = pick(LISTS.scheduler, S.scheduler, ["simple"]);
    const miss = [];
    if (!LISTS.unet.length) miss.push("diffusion model"); if (!LISTS.clip.length) miss.push("text encoder"); if (!LISTS.vae.length) miss.push("VAE");
    if (LISTS.clipTypes.length && !LISTS.clipTypes.includes("krea2")) miss.push("ComfyUI รุ่นนี้ยังไม่รองรับ Krea 2 (อัปเดต ComfyUI)");
    banner(miss.length ? "⚠️ ยังขาด: " + miss.join(", ") + "\nถ้าเพิ่งเปิดเครื่อง อาจกำลังโหลดโมเดลอยู่ — รอสักครู่แล้วรีเฟรชหน้า" : "");
    save();
  } catch (e) {
    banner("⚠️ ต่อ ComfyUI ไม่ได้ — รอ ComfyUI เปิดเสร็จแล้วรีเฟรชหน้า\n" + e.message);
  }
  renderLoras(); renderSampler();
}

// ---------- workflow ----------
function baseNodes(loras, text) {
  const P = {
    "10": { class_type: "UNETLoader", inputs: { unet_name: S.unet, weight_dtype: "default" } },
    "11": { class_type: "CLIPLoader", inputs: { clip_name: S.clip, type: "krea2", device: "default" } },
    "12": { class_type: "VAELoader", inputs: { vae_name: S.vae } }
  };
  let model = ["10", 0];
  loras.forEach((L, i) => {
    const id = String(30 + i);
    P[id] = { class_type: "LoraLoaderModelOnly", inputs: { model, lora_name: L.name, strength_model: L.w / 100 } };
    model = [id, 0];
  });
  P["13"] = { class_type: "CLIPTextEncode", inputs: { text: text || "", clip: ["11", 0] } };
  P["14"] = { class_type: "ConditioningZeroOut", inputs: { conditioning: ["13", 0] } };
  return { P, model };
}
const activeLoras = () => S.loras.filter(L => L.name !== "Disabled" && L.w > 0).map(L => ({ name: L.name, w: L.w }));
function buildGenerate(seed) {
  const loras = activeLoras();
  const { P, model } = baseNodes(loras, S.prompt);
  P["15"] = { class_type: "EmptyLatentImage", inputs: { width: S.width, height: S.height, batch_size: S.batch } };
  P["16"] = { class_type: "KSampler", inputs: { seed, steps: S.steps, cfg: S.cfg, sampler_name: S.sampler, scheduler: S.scheduler,
                denoise: 1, model, positive: ["13", 0], negative: ["14", 0], latent_image: ["15", 0] } };
  P["17"] = { class_type: "VAEDecode", inputs: { samples: ["16", 0], vae: ["12", 0] } };
  P["18"] = { class_type: "SaveImage", inputs: { filename_prefix: "MagMix", images: ["17", 0] } };
  const meta = { v: 1, prompt: S.prompt || "", loras, gen_w: S.width, gen_h: S.height, seed, steps: S.steps, cfg: S.cfg,
                 sampler: S.sampler, scheduler: S.scheduler, batch: S.batch, unet: S.unet };
  return { P, meta };
}
const upSize = (m, scale) => {
  let k = scale; const long = Math.max(m.width, m.height);
  if (long * k > UP_MAX) k = UP_MAX / long;
  return [Math.round(m.width * k / 16) * 16, Math.round(m.height * k / 16) * 16];
};
function buildUpscale(src, m, scale, den, seed) {
  const loras = Array.isArray(m.loras) ? m.loras.filter(L => L.w > 0 && LISTS.lora.includes(L.name)) : activeLoras();
  const text = typeof m.prompt === "string" ? m.prompt : S.prompt;
  const { P, model } = baseNodes(loras, text);
  const [W, H] = upSize(m, scale);
  P["20"] = { class_type: "LoadImage", inputs: { image: `${src} [output]` } };
  P["21"] = { class_type: "ImageScale", inputs: { image: ["20", 0], upscale_method: "lanczos", width: W, height: H, crop: "disabled" } };
  P["22"] = { class_type: "VAEEncode", inputs: { pixels: ["21", 0], vae: ["12", 0] } };
  P["16"] = { class_type: "KSampler", inputs: { seed, steps: S.steps, cfg: S.cfg, sampler_name: S.sampler, scheduler: S.scheduler,
                denoise: den / 100, model, positive: ["13", 0], negative: ["14", 0], latent_image: ["22", 0] } };
  P["17"] = { class_type: "VAEDecode", inputs: { samples: ["16", 0], vae: ["12", 0] } };
  P["18"] = { class_type: "SaveImage", inputs: { filename_prefix: "MagMix_upscale", images: ["17", 0] } };
  const meta = { v: 1, prompt: text || "", loras, gen_w: m.gen_w || m.width, gen_h: m.gen_h || m.height, seed: m.seed,
                 steps: m.steps, cfg: m.cfg, sampler: m.sampler, scheduler: m.scheduler, unet: S.unet,
                 upscale: { from: src, scale, denoise: den, seed, w: W, h: H } };
  return { P, meta, W, H, loras };
}

// ---------- display ----------
const relOf = (im) => (im.subfolder ? im.subfolder + "/" : "") + im.filename;
const viewUrl = (rel) => {
  const i = rel.lastIndexOf("/");
  return `${API}/view?filename=${encodeURIComponent(rel.slice(i + 1))}&subfolder=${encodeURIComponent(i < 0 ? "" : rel.slice(0, i))}&type=output`;
};
let GALLERY = [];                 // [{f, t}] ใหม่สุดก่อน
const tOf = (rel) => { const g = GALLERY.find(x => x.f === rel); return g ? Math.floor(g.t) : 0; };
const thumbUrl = (rel) => `${BASE}/thumb?f=${encodeURIComponent(rel)}&t=${tOf(rel)}`;
let current = null, lastSet = null, curMeta = null;
const META = {};
async function getMeta(rel) {
  if (META[rel]) return META[rel];
  try { const r = await fetch(`${BASE}/meta?f=${encodeURIComponent(rel)}`, { cache: "no-store" }); if (r.ok) return (META[rel] = await r.json()); } catch (e) {}
  return null;
}
function metaText(m) {
  if (!m) return "";
  const p = [`${m.width}×${m.height}`];
  if (m.upscale) p.push(`Upscale ×${m.upscale.scale} · รายละเอียด ${m.upscale.denoise}%`);
  if (m.seed != null) p.push(`seed ${m.seed}`);
  if (m.steps != null) p.push(`${m.steps} steps · cfg ${m.cfg}`);
  if (m.loras && m.loras.length) p.push("LoRA: " + m.loras.map(L => `${pretty(L.name)} ${L.w}%`).join(", "));
  return p.join(" · ");
}
function renderMeta(m, extra) {
  const canApply = m && (m.prompt != null || m.seed != null);
  $("meta").innerHTML = esc(extra || metaText(m) || (current || "")) +
    (canApply ? ` <button id="applyMeta" title="ใส่ขนาด LoRA seed และค่า Sampler ของภาพนี้กลับไปที่แผงตั้งค่า">↺ ใช้ค่าตั้งของภาพนี้</button>` : "");
  if (canApply) $("applyMeta").onclick = () => applyMeta(m);
}
function show(rel, meta) {
  current = rel;
  const url = viewUrl(rel);
  const back = lastSet && lastSet.length > 1 && lastSet.includes(rel) ? `<button id="backSet">◀ ดูทั้งชุด</button>` : "";
  $("stage").innerHTML = `<img src="${url}" alt=""><div class="tools">${back}<button id="stUp">🔍 Upscale</button><a href="${url}" target="_blank" rel="noopener">เปิดเต็ม</a><a href="${url}" download="${esc(rel.split("/").pop())}">ดาวน์โหลด</a></div>`;
  if (back) $("backSet").onclick = () => showSet(lastSet);
  $("stUp").onclick = () => { selectImage(rel, { keepPrompt: true }).then(() => openUpscale()); };
  renderMeta(meta || META[rel] || null, meta ? null : rel);
  markGallery();
}
function showSet(rels, metaStr) {
  if (rels.length === 1) return show(rels[0]);
  lastSet = rels; current = null;
  $("stage").innerHTML = `<div class="multi">${rels.map((r, i) => `<img data-i="${i}" src="${viewUrl(r)}" alt="">`).join("")}</div>`;
  $("stage").querySelectorAll(".multi img").forEach(img => img.onclick = () => selectImage(rels[Number(img.dataset.i)], { keepPrompt: true }));
  $("meta").textContent = metaStr || `${rels.length} ภาพ — คลิกภาพเพื่อดูใหญ่`;
  markGallery();
}
function showEmpty() {
  current = null; curMeta = null;
  $("stage").innerHTML = `<div class="empty">ภาพที่สร้างจะแสดงตรงนี้<br><span style="font-size:12px">กด ✨ สร้างภาพ หรือ Ctrl/⌘ + Enter</span></div>`;
  $("meta").textContent = ""; markGallery(); renderGBar();
}
function showPreview(blobUrl) {
  const img = $("stage").querySelector("img.preview");
  if (img) { const old = img.src; img.src = blobUrl; URL.revokeObjectURL(old); }
  else $("stage").innerHTML = `<img class="preview" src="${blobUrl}" alt="">`;
}
async function selectImage(rel, o) {
  o = o || {};
  closeUpscale();
  show(rel);
  const m = await getMeta(rel);
  if (current !== rel) return;
  curMeta = m;
  renderMeta(m);
  if (!o.keepPrompt) {
    if (m && typeof m.prompt === "string") { setPrompt(m.prompt); status("ใส่ prompt ของภาพนี้ในช่อง Prompt แล้ว — แก้ไขหรือกดสร้างซ้ำได้เลย"); }
    else status("ภาพนี้ไม่มีข้อมูล prompt ติดมา", true);
  }
  renderGBar();
}
function applyMeta(m) {
  const done = [], miss = [];
  if (typeof m.prompt === "string") { setPrompt(m.prompt); done.push("prompt"); }
  const gw = m.gen_w || m.width, gh = m.gen_h || m.height;
  if (gw >= 512 && gw <= 2048 && gh >= 512 && gh <= 2048) { S.width = gw; S.height = gh; matchSize(); done.push("ขนาด"); }
  if (Array.isArray(m.loras)) {
    const ok = m.loras.filter(L => LISTS.lora.includes(L.name)).slice(0, MAX_LORA);
    m.loras.filter(L => !LISTS.lora.includes(L.name)).forEach(L => miss.push(pretty(L.name)));
    S.loras = ok.length ? ok.map(L => ({ name: L.name, w: L.w })) : [{ name: "Disabled", w: 100 }];
    done.push("LoRA");
  }
  if (m.seed != null) { S.seed = m.seed; S.rand = false; done.push("seed (ปิดสุ่ม seed แล้ว)"); }
  if (m.steps != null) S.steps = m.steps;
  if (m.cfg != null) S.cfg = m.cfg;
  if (m.sampler && LISTS.sampler.includes(m.sampler)) S.sampler = m.sampler;
  if (m.scheduler && LISTS.scheduler.includes(m.scheduler)) S.scheduler = m.scheduler;
  save(); renderLoras(); renderSize(); renderSampler();
  status("✓ ใช้ค่าของภาพนี้แล้ว: " + done.concat(m.steps != null ? ["Sampler"] : []).join(", ") +
         (miss.length ? `\n⚠️ ไม่มี LoRA ในเครื่อง: ${miss.join(", ")}` : ""), miss.length > 0);
}

// ---------- gallery ----------
let selMode = false;
const picked = new Set();
function markGallery() {
  const on = new Set(current ? [current] : (lastSet || []));
  document.querySelectorAll(".grid .cell").forEach(c => {
    c.classList.toggle("on", !selMode && on.has(c.dataset.f));
    c.classList.toggle("picked", selMode && picked.has(c.dataset.f));
  });
}
function renderGrid() {
  const g = $("grid"), frag = document.createDocumentFragment();
  g.innerHTML = "";
  GALLERY.forEach(({ f }) => {
    const c = document.createElement("div");
    c.className = "cell" + (/(^|\/)MagMix_upscale/i.test(f) ? " up" : ""); c.dataset.f = f;
    c.innerHTML = `<img loading="lazy" decoding="async" src="${thumbUrl(f)}" alt=""><span class="tick">✓</span>`;
    c.onclick = () => {
      if (selMode) { picked.has(f) ? picked.delete(f) : picked.add(f); markGallery(); renderGBar(); }
      else selectImage(f);
    };
    frag.appendChild(c);
  });
  g.appendChild(frag);
  if (!GALLERY.length) g.innerHTML = `<div class="gempty" style="grid-column:1/-1">ยังไม่มีภาพ<br>ภาพที่สร้างทั้งหมดจะเก็บไว้ตรงนี้</div>`;
  $("gcount").textContent = GALLERY.length ? `${GALLERY.length.toLocaleString()} ภาพ` : "";
  markGallery();
}
async function loadGallery() {
  try {
    const r = await fetch(`${BASE}/gallery`, { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const items = await r.json();
    const sig = (a) => a.map(x => x.f + "@" + x.t).join("|");
    const changed = sig(items) !== sig(GALLERY);
    GALLERY = items;
    const have = new Set(items.map(x => x.f));
    [...picked].forEach(f => { if (!have.has(f)) picked.delete(f); });
    if (changed) renderGrid();
    renderGBar();
  } catch (e) {}
  return GALLERY;
}
function setSelMode(on) {
  selMode = on; picked.clear(); closeUpscale();
  $("gcard").classList.toggle("selmode", on);
  $("selMode").classList.toggle("on", on);
  $("selMode").textContent = on ? "✕ เลิกเลือก" : "☑ เลือกหลายรูป";
  markGallery(); renderGBar();
}
$("selMode").onclick = () => setSelMode(!selMode);
function renderGBar() {
  const b = $("gbar");
  if (selMode) {
    b.innerHTML = `<span class="hint">เลือกแล้ว <b style="color:var(--text)">${picked.size}</b> รูป</span>
      <button class="pill" id="gAll">${picked.size === GALLERY.length && GALLERY.length ? "ไม่เลือกเลย" : "เลือกทั้งหมด"}</button>
      <button class="pill danger" id="gDelMany" ${picked.size ? "" : "disabled"}>🗑 ลบ ${picked.size || ""} รูป</button>`;
    $("gAll").onclick = () => { if (picked.size === GALLERY.length) picked.clear(); else GALLERY.forEach(x => picked.add(x.f)); markGallery(); renderGBar(); };
    $("gDelMany").onclick = () => deleteImages([...picked]);
    return;
  }
  if (current && GALLERY.some(x => x.f === current)) {
    b.innerHTML = `<button class="pill red" id="gUp" ${running ? "disabled" : ""}>🔍 Upscale</button>
      <button class="pill danger" id="gDel">🗑 ลบ</button>
      <span class="hint" style="text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(current.split("/").pop())}</span>`;
    $("gUp").onclick = () => openUpscale();
    $("gDel").onclick = () => deleteImages([current]);
  } else {
    b.innerHTML = `<span class="hint">คลิกภาพเพื่อดูใหญ่ ดึง prompt กลับมา หรือ Upscale / ลบ</span>`;
  }
}
async function deleteImages(files) {
  if (!files.length) return;
  const ok = await confirmBox(files.length === 1 ? "ลบภาพนี้?" : `ลบ ${files.length} ภาพ?`,
    "ภาพจะถูกลบออกจากเครื่อง vast ถาวร กู้คืนไม่ได้ (ถ้าอยากเก็บ ให้กดดาวน์โหลดก่อน)", "ลบถาวร");
  if (!ok) return;
  try {
    const r = await fetch(`${BASE}/delete`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ files }) });
    const j = await r.json();
    (j.deleted || []).forEach(f => { delete META[f]; picked.delete(f); });
    const gone = new Set(j.deleted || []);
    const idx = current ? GALLERY.findIndex(x => x.f === current) : -1;
    await loadGallery();
    if (lastSet && lastSet.some(f => gone.has(f))) lastSet = lastSet.filter(f => !gone.has(f));
    if (current && gone.has(current)) {
      const next = GALLERY[Math.min(Math.max(idx, 0), GALLERY.length - 1)];
      if (next) selectImage(next.f, { keepPrompt: true }); else showEmpty();
    }
    if (selMode && !GALLERY.length) setSelMode(false);
    status(j.errors && j.errors.length ? "❌ ลบบางภาพไม่ได้:\n" + j.errors.join("\n") : `🗑 ลบแล้ว ${gone.size} ภาพ`, !!(j.errors && j.errors.length));
  } catch (e) { status("❌ ลบไม่สำเร็จ: " + e.message, true); }
}

// ---------- Upscale ----------
function closeUpscale() { $("upbox").hidden = true; $("upbox").innerHTML = ""; }
async function openUpscale() {
  const rel = current; if (!rel) return;
  const m = curMeta && curMeta.f === rel ? curMeta : await getMeta(rel);
  if (!m) { status("❌ อ่านข้อมูลภาพนี้ไม่ได้", true); return; }
  const box = $("upbox"); box.hidden = false;
  const draw = () => {
    const [W, H] = upSize(m, S.upScale);
    const capped = Math.max(m.width, m.height) * S.upScale > UP_MAX;
    const hasLoras = Array.isArray(m.loras);
    const missing = hasLoras ? m.loras.filter(L => !LISTS.lora.includes(L.name)).map(L => pretty(L.name)) : [];
    box.innerHTML = `<div class="uphead"><img src="${thumbUrl(rel)}" alt=""><div><b>Upscale ภาพนี้</b>
        ${m.width}×${m.height} → <b style="display:inline;color:var(--text)">${W}×${H}</b>${capped ? " (จำกัดด้านยาว " + UP_MAX + ")" : ""}</div></div>
      <div class="row"><span>ขยาย</span></div>
      <div class="seg" id="upScale">${[1.5, 2].map(k => `<button data-k="${k}" class="${k === S.upScale ? "on" : ""}">×${k}</button>`).join("")}</div>
      <div class="row"><span>เพิ่มรายละเอียด</span><b id="upDv">${S.upDen}%</b></div>
      <input type="range" id="upDen" min="15" max="60" step="5" value="${S.upDen}">
      <div class="hint" style="margin-top:4px">น้อย = เหมือนเดิม แค่คมขึ้น · กลาง 30–40% = เติมรายละเอียดผิว/ผ้า (แนะนำ) · มากเกิน 50% = หน้าอาจเปลี่ยน</div>
      <div class="hint" style="margin-top:8px">ใช้ prompt ${hasLoras ? "และ LoRA ชุดเดียวกับตอนสร้างภาพนี้" : "ในช่อง Prompt และ LoRA ที่ตั้งไว้ตอนนี้ (ภาพนี้ไม่มีข้อมูลเดิม)"}${missing.length ? `<br><span style="color:var(--err)">ไม่มี LoRA ในเครื่อง (จะข้าม): ${esc(missing.join(", "))}</span>` : ""}</div>
      <div class="btns"><button class="pill" id="upCancel">ยกเลิก</button><button class="pill red" id="upGo" ${running ? "disabled" : ""}>🔍 เริ่ม Upscale</button></div>`;
    box.querySelectorAll("#upScale button").forEach(b => b.onclick = () => { S.upScale = Number(b.dataset.k); save(); draw(); });
    const rg = $("upDen"); fillRange(rg);
    rg.oninput = () => { S.upDen = Number(rg.value); $("upDv").textContent = S.upDen + "%"; fillRange(rg); save(); };
    $("upCancel").onclick = closeUpscale;
    $("upGo").onclick = () => startUpscale(rel, m);
  };
  draw();
}
async function startUpscale(rel, m) {
  if (running) return;
  if (!S.unet || !S.clip || !S.vae) { status("ยังโหลดรายชื่อโมเดลไม่ได้ — ตรวจว่า ComfyUI เปิดอยู่", true); return; }
  const seed = newSeed();
  const { P, meta, W, H } = buildUpscale(rel, m, S.upScale, S.upDen, seed);
  closeUpscale();
  await submit(P, meta, { kind: "up", w: W, h: H, seed, batch: 1, label: `Upscale ${W}×${H}` });
}

// ---------- websocket ----------
let ws = null, wsOk = false, running = null;
function connectWS() {
  try {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?clientId=${CLIENT_ID}`);
    ws.binaryType = "blob";
    ws.onopen = () => { wsOk = true; };
    ws.onclose = () => { wsOk = false; setTimeout(connectWS, 3000); };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        if (running) showPreview(URL.createObjectURL(ev.data.slice(8, ev.data.size, "image/jpeg")));
        return;
      }
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      const d = m.data || {};
      if (!running || (d.prompt_id && d.prompt_id !== running.id)) return;
      if (m.type === "progress") {
        $("bar").style.width = (d.value / d.max * 100) + "%";
        status(`${running.kind === "up" ? "กำลัง Upscale" : "กำลังสร้าง" + (running.batch > 1 ? " " + running.batch + " ภาพ" : "")}… ${d.value}/${d.max}`);
      } else if (m.type === "executing" && d.node) {
        const n = Number(d.node);
        status({ 10: "โหลดโมเดล…", 11: "โหลด text encoder…", 12: "โหลด VAE…", 13: "อ่าน prompt…", 17: "แปลงเป็นภาพ…",
                 20: "โหลดภาพต้นฉบับ…", 21: "ขยายภาพ…", 22: "เตรียมภาพเข้าโมเดล…" }[n] || (n >= 30 && n < 40 ? "โหลด LoRA…" : $("status").textContent));
      } else if (m.type === "executed" && d.node === "18" && d.output && d.output.images) {
        finish(d.output.images);
      } else if (m.type === "execution_error") {
        fail(`❌ ${d.node_type || ""}: ${d.exception_message || "เกิดข้อผิดพลาด"}`);
      } else if (m.type === "execution_cached" && (d.nodes || []).includes("18")) {
        poll(running.id);
      }
    };
  } catch (e) { wsOk = false; }
}
function setBusy(b) { $("go").disabled = b; const u = $("gUp"); if (u) u.disabled = b; const g = $("upGo"); if (g) g.disabled = b; }
async function finish(images) {
  if (!running || !images || !images.length) return;
  const R = running, sec = ((Date.now() - R.t0) / 1000).toFixed(1);
  running = null; setBusy(false); $("bar").style.width = "100%";
  status(R.kind === "up" ? `✅ Upscale เสร็จใน ${sec} วินาที — ภาพใหม่อยู่บนสุดของ Gallery` : `✅ เสร็จใน ${sec} วินาที`);
  const rels = images.map(relOf);
  await loadGallery();
  lastSet = rels;
  if (rels.length === 1) selectImage(rels[0], { keepPrompt: true });
  else showSet(rels, `${R.w}×${R.h} · ${rels.length} ภาพ · seed ${R.seed} · ${sec}s — คลิกภาพเพื่อดูใหญ่`);
  $("gscroll").scrollTop = 0;
}
function fail(msg) { running = null; setBusy(false); $("bar").style.width = "0"; status(msg, true); renderGBar(); }
async function poll(id) {
  for (let i = 0; i < 900 && running && running.id === id; i++) {
    try {
      const r = await fetch(`${API}/history/${id}`, { cache: "no-store" }); const j = await r.json(); const h = j[id];
      if (h) {
        const st = h.status || {};
        if (st.status_str === "error") {
          const em = (st.messages || []).find(x => x[0] === "execution_error");
          return fail("❌ " + (em ? `${em[1].node_type}: ${em[1].exception_message}` : "เกิดข้อผิดพลาด"));
        }
        const out = h.outputs && h.outputs["18"];
        if (out && out.images && out.images.length) return finish(out.images);
      }
    } catch (e) {}
    await new Promise(r => setTimeout(r, wsOk ? 4000 : 1500));
  }
}
async function submit(P, meta, R) {
  setBusy(true); $("bar").style.width = "0"; status("ส่งงาน…");
  try {
    const r = await fetch(`${API}/prompt`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: P, client_id: CLIENT_ID, extra_data: { extra_pnginfo: { magmix: meta } } }) });
    const j = await r.json();
    if (!r.ok || j.error) {
      const ne = j.node_errors ? Object.values(j.node_errors).map(n => typeof n === "string" ? n : `${n.class_type}: ${(n.errors || []).map(e => e.details || e.message).join(", ")}`).join("\n") : "";
      return fail("❌ " + ((j.error && (j.error.message || j.error)) || `HTTP ${r.status}`) + (ne ? "\n" + ne : ""));
    }
    running = Object.assign({ id: j.prompt_id, t0: Date.now() }, R);
    status(j.number ? `เข้าคิวแล้ว (ลำดับ ${j.number})` : "เข้าคิวแล้ว");
    poll(j.prompt_id);
  } catch (e) { fail("❌ ส่งงานไม่สำเร็จ: " + e.message); }
}
async function generate() {
  if (running) return;
  if (!S.unet || !S.clip || !S.vae) { status("ยังโหลดรายชื่อโมเดลไม่ได้ — ตรวจว่า ComfyUI เปิดอยู่", true); return; }
  if (S.rand) { S.seed = newSeed(); $("seed").value = S.seed; save(); }
  const seed = S.seed;
  const { P, meta } = buildGenerate(seed);
  await submit(P, meta, { kind: "gen", seed, batch: S.batch, w: S.width, h: S.height });
}
$("go").onclick = generate;
$("prompt").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate(); });

// ---------- Import LoRA ----------
const UP = {};
let jobTimer = null;
const seenDone = new Set();
$("civKey").value = S.civKey || ""; $("hfTok").value = S.hfTok || "";
$("civKey").onchange = (e) => { S.civKey = e.target.value.trim(); save(); };
$("hfTok").onchange = (e) => { S.hfTok = e.target.value.trim(); save(); };
$("openImport").onclick = () => { $("importModal").classList.add("open"); refreshJobs(); $("impUrl").focus(); };
$("closeImport").onclick = () => $("importModal").classList.remove("open");
$("importModal").onclick = (e) => { if (e.target.id === "importModal") $("importModal").classList.remove("open"); };

function jobHtml(j) {
  const pct = j.total ? Math.min(100, j.done / j.total * 100) : (j.status === "done" ? 100 : 0);
  const st = j.status === "done" ? "✅ " + (j.note || "เสร็จแล้ว") : j.status === "error" ? "❌ ไม่สำเร็จ"
           : j.total ? `${fmtMB(j.done)} / ${fmtMB(j.total)}` : (j.done ? fmtMB(j.done) : "กำลังเริ่ม…");
  const cancel = j.status === "running" && j.kind !== "upload" ? ` <a href="#" data-cancel="${esc(j.id)}" class="muted">ยกเลิก</a>` : "";
  return `<div class="job ${j.status === "done" ? "done" : j.status === "error" ? "err" : ""}">
    <div class="t"><b>${esc(j.name ? pretty(j.name) : j.url || "")}</b><span class="muted">${st}${cancel}</span></div>
    ${j.status === "running" ? `<div class="bar"><i style="width:${pct}%"></i></div>` : ""}
    ${j.error ? `<div class="e">${esc(j.error)}</div>` : ""}</div>`;
}
async function refreshJobs() {
  let server = [];
  try { const r = await fetch(`${BASE}/lora/jobs`, { cache: "no-store" }); if (r.ok) server = await r.json(); } catch (e) {}
  const all = Object.values(UP).concat(server).sort((a, b) => (b.t || 0) - (a.t || 0));
  $("jobs").innerHTML = all.length ? `<h4>สถานะ</h4>` + all.map(jobHtml).join("") : "";
  $("jobs").querySelectorAll("[data-cancel]").forEach(a => a.onclick = async (e) => {
    e.preventDefault(); await fetch(`${BASE}/lora/cancel/${a.dataset.cancel}`, { method: "POST" }); refreshJobs();
  });
  const newlyDone = all.filter(j => j.status === "done" && j.name && !seenDone.has(j.id));
  newlyDone.forEach(j => seenDone.add(j.id));
  if (newlyDone.length) { await reloadLoraList(); newlyDone.forEach(j => { if (LISTS.lora.includes(j.name)) useLora(j.name); }); }
  const busy = all.some(j => j.status === "running");
  clearTimeout(jobTimer);
  if (busy || $("importModal").classList.contains("open")) jobTimer = setTimeout(refreshJobs, busy ? 1200 : 4000);
}
(async () => { try { const r = await fetch(`${BASE}/lora/jobs`); if (r.ok) (await r.json()).forEach(j => { if (j.status === "done") seenDone.add(j.id); }); } catch (e) {} })();

const addErr = (url, msg) => { const id = "e" + Date.now(); UP[id] = { id, url, status: "error", error: msg, t: Date.now() / 1000 }; };
$("impGo").onclick = async () => {
  const url = $("impUrl").value.trim();
  if (!url) return;
  try {
    const r = await fetch(`${BASE}/lora/url`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, civitai_key: S.civKey || "", hf_token: S.hfTok || "" }) });
    const j = await r.json();
    if (!r.ok) addErr(url, j.error || `HTTP ${r.status}`); else $("impUrl").value = "";
  } catch (e) { addErr(url, e.message); }
  refreshJobs();
};
$("impUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") $("impGo").click(); });

async function uploadFile(file) {
  const id = "u" + Date.now() + Math.random().toString(16).slice(2, 6);
  const J = UP[id] = { id, kind: "upload", name: file.name, status: "running", done: 0, total: file.size, t: Date.now() / 1000 };
  if (!/\.safetensors$/i.test(file.name)) { Object.assign(J, { status: "error", error: "รองรับเฉพาะไฟล์ .safetensors" }); return refreshJobs(); }
  refreshJobs();
  try {
    let offset = 0;
    for (;;) {
      const part = file.slice(offset, offset + CHUNK);
      const q = `name=${encodeURIComponent(file.name)}&offset=${offset}&total=${file.size}`;
      let r = null, j = {}, tries = 0;
      for (;;) {
        try { r = await fetch(`${BASE}/lora/upload?${q}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: part }); j = await r.json(); }
        catch (e) { r = null; j = { error: e.message }; }
        if (r && r.ok) break;
        if (++tries >= 3 || (r && r.status >= 400 && r.status < 500)) throw new Error(j.error || `HTTP ${r ? r.status : "?"}`);
        await new Promise(res => setTimeout(res, 2000));
      }
      J.name = j.name || J.name;
      if (j.done) { J.done = file.size; J.status = "done"; if (j.exists) J.note = "มีไฟล์นี้อยู่แล้ว"; break; }
      offset += part.size; J.done = offset;
      refreshJobs();
      if (offset >= file.size) throw new Error("เซิร์ฟเวอร์ไม่ยืนยันว่าได้รับไฟล์ครบ");
    }
  } catch (e) { Object.assign(J, { status: "error", error: e.message }); }
  refreshJobs();
}
$("impFile").onchange = (e) => { [...e.target.files].forEach(uploadFile); e.target.value = ""; };
const drop = $("drop");
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { [...e.dataTransfer.files].forEach(uploadFile); });

// ---------- start ----------
renderLoras(); renderSize(); renderBatch(); renderSampler(); renderGBar();
connectWS(); loadLists();
loadGallery().then(g => { if (g.length) selectImage(g[0].f, { keepPrompt: true }); });
document.addEventListener("visibilitychange", () => { if (!document.hidden && !running) loadGallery(); });
})();
</script>
</body>
</html>
'''

ROUTES = PromptServer.instance.routes
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
META_CACHE = {}    # (path, mtime) -> metadata ของภาพ
ALLOWED_HOSTS = ("huggingface.co", "civitai.com", "civitai.red", "civitai.green")
JOBS = {}          # งานดาวน์โหลด LoRA จากลิงก์
UA = {"User-Agent": "MagMix-Studio/1.0"}


# ---------------------------------------------------------------- helpers
def _lora_dir():
    d = folder_paths.get_folder_paths("loras")[0]
    os.makedirs(d, exist_ok=True)
    return d


def _safe_name(name):
    name = os.path.basename(urllib.parse.unquote(name or "")).strip()
    name = re.sub(r"[^\w.\-()+ ]", "_", name).strip(" .")
    stem = os.path.splitext(name)[0] if name.lower().endswith(".safetensors") else name
    stem = stem or "lora"
    return stem[:140] + ".safetensors"


def _name_from_cd(cd):
    if not cd:
        return None
    m = re.search(r"filename\*\s*=\s*[^']*''([^;]+)", cd)
    if m:
        return urllib.parse.unquote(m.group(1).strip().strip('"'))
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', cd)
    return m.group(1).strip() if m else None


def _host_ok(host):
    host = (host or "").lower()
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


async def _resolve(url, session, civitai_key):
    """แปลงลิงก์ที่ผู้ใช้วาง → (ลิงก์ดาวน์โหลดจริง, headers, ชื่อไฟล์ที่เดาได้)"""
    u = urllib.parse.urlparse(url.strip())
    if u.scheme != "https" or not _host_ok(u.hostname):
        raise ValueError("รองรับเฉพาะลิงก์ https จาก huggingface.co หรือ civitai (.com / .red / .green)")
    host = u.hostname.lower()
    if host.endswith("huggingface.co"):
        path = u.path.replace("/blob/", "/resolve/", 1)
        if "/resolve/" not in path or not path.lower().endswith(".safetensors"):
            raise ValueError("ลิงก์ HuggingFace ต้องเป็นลิงก์ของไฟล์ .safetensors "
                             "(เปิดหน้าไฟล์แล้วก๊อปลิงก์ ไม่ใช่หน้าแรกของ repo)")
        return urllib.parse.urlunparse(("https", "huggingface.co", path, "", "", "")), {}, os.path.basename(path)
    # ---- civitai (civitai.com / civitai.red / civitai.green ใช้ระบบเดียวกัน) ----
    cbase = "https://civitai.red" if host.endswith("civitai.red") else "https://civitai.com"
    q = urllib.parse.parse_qs(u.query)
    m = re.search(r"/api/download/models/(\d+)", u.path)
    if m:
        vid = m.group(1)
    else:
        m = re.search(r"/models/(\d+)", u.path)
        if not m:
            raise ValueError("ลิงก์ Civitai ต้องเป็นหน้าโมเดล เช่น https://civitai.com/models/12345")
        vid = (q.get("modelVersionId") or [None])[0]
        if not vid:
            h = dict(UA)
            if civitai_key:
                h["Authorization"] = "Bearer " + civitai_key
            async with session.get(cbase + "/api/v1/models/" + m.group(1), headers=h) as r:
                if r.status != 200:
                    raise ValueError("อ่านข้อมูลโมเดลจาก Civitai ไม่ได้ (HTTP %d)" % r.status)
                j = await r.json()
            versions = j.get("modelVersions") or []
            if not versions:
                raise ValueError("โมเดลนี้ไม่มีไฟล์ให้ดาวน์โหลด")
            vid = str(versions[0]["id"])
    headers = {}
    if civitai_key:
        headers["Authorization"] = "Bearer " + civitai_key
    return cbase + "/api/download/models/%s?type=Model&format=SafeTensor" % vid, headers, None


async def _run_job(jid, url, civitai_key, hf_token):
    J = JOBS[jid]
    tmp = None
    try:
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=30, sock_read=180)) as s:
            dl, headers, guess = await _resolve(url, s, civitai_key)
            headers = dict(UA, **headers)
            if hf_token and "huggingface.co" in dl:
                headers["Authorization"] = "Bearer " + hf_token
            async with s.get(dl, headers=headers, allow_redirects=True) as r:
                if r.status in (401, 403):
                    raise ValueError("ไฟล์นี้ต้องล็อกอินก่อนโหลด — ใส่ %s ใน 'ตัวเลือกเพิ่มเติม' แล้วลองใหม่"
                                     % ("Civitai API key" if "civitai" in dl else "HuggingFace token"))
                if r.status == 404:
                    raise ValueError("ไม่พบไฟล์ (404) — ตรวจลิงก์อีกครั้ง")
                if r.status != 200:
                    raise ValueError("ดาวน์โหลดไม่ได้ (HTTP %d)" % r.status)
                if "text/html" in (r.headers.get("Content-Type") or ""):
                    raise ValueError("ได้หน้าเว็บแทนไฟล์ — ลิงก์ไม่ถูก หรือไฟล์นี้ต้องใช้ API key")
                name = _safe_name(_name_from_cd(r.headers.get("Content-Disposition")) or guess
                                  or os.path.basename(r.url.path))
                J["name"] = name
                dest = os.path.join(_lora_dir(), name)
                if os.path.exists(dest):
                    J.update(status="done", note="มีไฟล์นี้อยู่แล้ว", done=os.path.getsize(dest),
                             total=os.path.getsize(dest))
                    return
                J["total"] = int(r.headers.get("Content-Length") or 0)
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    async for chunk in r.content.iter_chunked(1 << 20):
                        if J.get("cancel"):
                            raise ValueError("ยกเลิกแล้ว")
                        f.write(chunk)
                        J["done"] += len(chunk)
                with open(tmp, "rb") as f:
                    head = f.read(8)
                if len(head) < 8 or int.from_bytes(head, "little") > 100_000_000:
                    raise ValueError("ไฟล์ที่ได้ไม่ใช่ .safetensors")
                os.replace(tmp, dest)
                tmp = None
        J["status"] = "done"
    except Exception as e:  # noqa
        J.update(status="error", error=str(e) or e.__class__.__name__)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------- gallery
def _out_dir():
    return os.path.realpath(folder_paths.get_output_directory())


def _thumb_dir():
    base = getattr(folder_paths, "get_user_directory", None)
    base = base() if base else folder_paths.get_temp_directory()
    d = os.path.join(base, "magmix_thumbs")
    os.makedirs(d, exist_ok=True)
    return d


def _out_path(rel):
    """แปลงชื่อภาพ (เช่น MagMix_00001_.png หรือ sub/x.png) → path จริงในโฟลเดอร์ output เท่านั้น"""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    root = _out_dir()
    p = os.path.realpath(os.path.join(root, rel))
    if not rel or not p.startswith(root + os.sep) or not p.lower().endswith(IMG_EXT):
        raise ValueError("ชื่อไฟล์ไม่ถูกต้อง")
    return p


def _list_images():
    root = _out_dir()
    items = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            if f.startswith(".") or not f.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            items.append({"f": os.path.relpath(p, root).replace(os.sep, "/"), "t": st.st_mtime})
    items.sort(key=lambda x: -x["t"])
    return items


def _thumb_file(rel):
    from PIL import Image
    src = _out_path(rel)
    st = os.stat(src)
    key = re.sub(r"[^\w.-]", "_", rel) + "_%d.webp" % int(st.st_mtime)
    dst = os.path.join(_thumb_dir(), key)
    if not os.path.exists(dst):
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((420, 640))
            im.save(dst + ".tmp", "WEBP", quality=80)
        os.replace(dst + ".tmp", dst)
    return dst


def _txt(v):
    return v if isinstance(v, str) else None


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _meta_from_graph(g):
    """อ่านค่าจาก workflow ของ ComfyUI ที่ฝังในภาพ (ภาพที่ไม่ได้สร้างจาก MagMix Studio)"""
    m = {"loras": []}
    nodes = {k: v for k, v in g.items() if isinstance(v, dict)}
    ks = next((v for v in nodes.values() if v.get("class_type") in ("KSampler", "KSamplerAdvanced")), None)
    if ks:
        i = ks.get("inputs", {})
        m.update(seed=_num(i.get("seed", i.get("noise_seed"))), steps=_num(i.get("steps")), cfg=_num(i.get("cfg")),
                 sampler=_txt(i.get("sampler_name")), scheduler=_txt(i.get("scheduler")))
        pos = i.get("positive")
        if isinstance(pos, list) and str(pos[0]) in nodes:
            m["prompt"] = _txt(nodes[str(pos[0])].get("inputs", {}).get("text"))
    if not m.get("prompt"):
        for v in nodes.values():
            t = _txt(v.get("inputs", {}).get("text"))
            if v.get("class_type") == "CLIPTextEncode" and t:
                m["prompt"] = t
                break
    for v in nodes.values():
        i = v.get("inputs", {})
        if v.get("class_type") in ("LoraLoaderModelOnly", "LoraLoader") and _txt(i.get("lora_name")):
            w = _num(i.get("strength_model"))
            m["loras"].append({"name": i["lora_name"], "w": round((1 if w is None else w) * 100)})
        if v.get("class_type") == "EmptyLatentImage":
            m.update(gen_w=_num(i.get("width")), gen_h=_num(i.get("height")))
    return m


def _read_meta(rel):
    from PIL import Image
    p = _out_path(rel)
    st = os.stat(p)
    ck = (p, st.st_mtime)
    if ck in META_CACHE:
        return META_CACHE[ck]
    with Image.open(p) as im:
        info = dict(getattr(im, "text", None) or im.info or {})
        w, h = im.size
    m = {}
    try:
        if info.get("magmix"):
            m = json.loads(info["magmix"])
        elif info.get("prompt"):
            m = _meta_from_graph(json.loads(info["prompt"]))
    except Exception:
        m = {}
    m.update(f=rel, width=w, height=h, t=st.st_mtime)
    if len(META_CACHE) > 5000:
        META_CACHE.clear()
    META_CACHE[ck] = m
    return m


async def _gallery(request):
    items = await asyncio.get_running_loop().run_in_executor(None, _list_images)
    return web.json_response(items, headers={"Cache-Control": "no-store"})


async def _thumb(request):
    try:
        p = await asyncio.get_running_loop().run_in_executor(None, _thumb_file, request.query.get("f", ""))
    except Exception as e:  # noqa
        return web.json_response({"error": str(e)}, status=404)
    return web.FileResponse(p, headers={"Cache-Control": "public, max-age=31536000, immutable",
                                        "Content-Type": "image/webp"})


async def _meta(request):
    try:
        m = await asyncio.get_running_loop().run_in_executor(None, _read_meta, request.query.get("f", ""))
    except Exception as e:  # noqa
        return web.json_response({"error": str(e)}, status=404)
    return web.json_response(m)


async def _delete(request):
    try:
        body = await request.json()
        files = list(body.get("files") or [])
    except Exception:
        return web.json_response({"error": "ข้อมูลไม่ถูกต้อง"}, status=400)
    deleted, errors = [], []
    for rel in files[:5000]:
        try:
            p = _out_path(rel)
            os.remove(p)
            deleted.append(rel)
        except FileNotFoundError:
            deleted.append(rel)
        except Exception as e:  # noqa
            errors.append("%s: %s" % (rel, e))
    try:  # ลบภาพย่อของไฟล์ที่ลบไปแล้ว
        td = _thumb_dir()
        for rel in deleted:
            pre = re.sub(r"[^\w.-]", "_", rel) + "_"
            for f in os.listdir(td):
                if f.startswith(pre):
                    os.remove(os.path.join(td, f))
    except OSError:
        pass
    return web.json_response({"deleted": deleted, "errors": errors})


# ---------------------------------------------------------------- routes
async def _page(request):
    return web.Response(text=HTML, content_type="text/html", charset="utf-8",
                        headers={"Cache-Control": "no-store"})


async def _lora_url(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "ข้อมูลไม่ถูกต้อง"}, status=400)
    url = (body.get("url") or "").strip()
    if not url:
        return web.json_response({"error": "ยังไม่ได้ใส่ลิงก์"}, status=400)
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or not _host_ok(u.hostname):
        return web.json_response({"error": "รองรับเฉพาะลิงก์ https จาก huggingface.co หรือ civitai (.com / .red / .green)"}, status=400)
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"id": jid, "url": url, "name": "", "status": "running", "done": 0, "total": 0,
                 "error": "", "note": "", "t": time.time()}
    asyncio.ensure_future(_run_job(jid, url, (body.get("civitai_key") or "").strip(),
                                   (body.get("hf_token") or "").strip()))
    return web.json_response({"id": jid})


async def _lora_jobs(request):
    jobs = sorted(JOBS.values(), key=lambda j: -j["t"])[:20]
    return web.json_response([{k: v for k, v in j.items() if k != "cancel"} for j in jobs])


async def _lora_cancel(request):
    J = JOBS.get(request.match_info["jid"])
    if J and J["status"] == "running":
        J["cancel"] = True
    return web.json_response({"ok": True})


async def _lora_upload(request):
    """อัปโหลดไฟล์ทีละชิ้น (ผ่าน cloudflare ได้ แม้ไฟล์ใหญ่กว่า 100MB)"""
    name = _safe_name(request.query.get("name", ""))
    try:
        offset = int(request.query.get("offset", "0"))
        total = int(request.query.get("total", "0"))
    except ValueError:
        return web.json_response({"error": "ข้อมูลไม่ถูกต้อง"}, status=400)
    dest = os.path.join(_lora_dir(), name)
    tmp = dest + ".upload"
    if offset == 0:
        if os.path.exists(dest):
            return web.json_response({"ok": True, "done": True, "exists": True, "name": name})
        open(tmp, "wb").close()
    size = os.path.getsize(tmp) if os.path.exists(tmp) else -1
    if size != offset:
        return web.json_response({"error": "ชิ้นไฟล์ไม่ต่อเนื่อง (%d ≠ %d) — อัปโหลดใหม่" % (size, offset)}, status=409)
    with open(tmp, "ab") as f:
        while True:
            chunk = await request.content.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(tmp)
    if total and size >= total:
        with open(tmp, "rb") as f:
            head = f.read(8)
        if len(head) < 8 or int.from_bytes(head, "little") > 100_000_000:
            os.remove(tmp)
            return web.json_response({"error": "ไฟล์นี้ไม่ใช่ .safetensors"}, status=400)
        os.replace(tmp, dest)
        return web.json_response({"ok": True, "done": True, "name": name})
    return web.json_response({"ok": True, "size": size, "name": name})


ROUTES.get("/magmix")(_page)
ROUTES.get("/magmix/")(_page)
ROUTES.post("/magmix/lora/url")(_lora_url)
ROUTES.get("/magmix/lora/jobs")(_lora_jobs)
ROUTES.post("/magmix/lora/cancel/{jid}")(_lora_cancel)
ROUTES.post("/magmix/lora/upload")(_lora_upload)
ROUTES.get("/magmix/gallery")(_gallery)
ROUTES.get("/magmix/thumb")(_thumb)
ROUTES.get("/magmix/meta")(_meta)
ROUTES.post("/magmix/delete")(_delete)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
print("[MagMix Studio] พร้อมใช้งาน → เปิด /magmix")
MAGMIX_PLUGIN_EOF
echo "✓ ติดตั้งหน้าแอป MagMix Studio แล้ว"

# ---------- รีสตาร์ต ComfyUI ให้โหลดหน้าแอป ----------
supervisorctl restart comfyui || true

if [ "$FAILED" -ne 0 ]; then
  echo "⚠️ บางไฟล์โหลดไม่สำเร็จ — vast จะลองใหม่ตอนรีสตาร์ตเครื่อง"
  exit 1
fi
echo "✅ MagMix พร้อมใช้งาน — เปิด ComfyUI แล้วเติม /magmix ต่อท้ายที่อยู่"
