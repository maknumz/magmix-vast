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
<style>
:root{
  --bg:#0d0e11; --card:#15161a; --card2:#1c1d22; --line:#2a2b31; --text:#e7e7ea; --muted:#9a9ba3;
  --accent:#e2533c; --accent2:#ff6b52; --knob:#d9dade; --ok:#4cc38a; --err:#ff5d5d;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans Thai",Roboto,sans-serif;font-size:14px}
.app{display:grid;grid-template-columns:340px minmax(0,1fr) 380px;gap:14px;height:100vh;padding:14px}
.col{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:12px;padding-right:2px}
.col::-webkit-scrollbar{width:6px}.col::-webkit-scrollbar-thumb{background:#2c2d33;border-radius:3px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px}
.card h3{margin:0 0 12px;font-size:17px;font-weight:700;display:flex;align-items:center;justify-content:space-between;gap:8px}
.brand{font-size:18px;font-weight:800;letter-spacing:.3px;padding:4px 4px 0}
.brand small{display:block;font-weight:500;color:var(--muted);font-size:12px;margin-top:2px}
select,textarea,input[type=number],input[type=text],input[type=password]{width:100%;background:var(--card2);color:var(--text);border:1px solid var(--line);
  border-radius:10px;padding:11px 12px;font-size:15px;font-family:inherit;outline:none}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),linear-gradient(135deg,var(--muted) 50%,transparent 50%);
  background-position:calc(100% - 18px) 50%,calc(100% - 13px) 50%;background-size:5px 5px;background-repeat:no-repeat;padding-right:34px}
select:focus,textarea:focus,input:focus{border-color:#4a4b53}
.row{display:flex;align-items:center;justify-content:space-between;color:var(--muted);margin:14px 0 6px;font-size:14px}
.row b{color:var(--text);font-weight:600}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:22px;background:transparent;margin:0;cursor:pointer;--p:50%}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--accent) var(--p),#2e2f35 var(--p))}
input[type=range]::-moz-range-track{height:6px;border-radius:3px;background:linear-gradient(90deg,var(--accent) var(--p),#2e2f35 var(--p))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:var(--knob);margin-top:-8px;
  box-shadow:0 1px 4px rgba(0,0,0,.6);border:2px solid #f4f4f6}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:var(--knob);border:2px solid #f4f4f6}
.aspects{display:flex;gap:6px;overflow-x:auto;padding:12px 2px 6px;scrollbar-width:none}
.aspects::-webkit-scrollbar{display:none}
.chip{flex:0 0 auto;width:64px;border-radius:12px;padding:8px 4px 6px;display:flex;flex-direction:column;align-items:center;gap:6px;
  cursor:pointer;color:var(--muted);user-select:none;border:1px solid transparent}
.chip .box{border:1.5px solid #6f7078;border-radius:3px}
.chip span{font-size:14px}
.chip.on{background:var(--card2);color:var(--text);border-color:#33343a}
.chip.on .box{border-color:#fff;border-width:2px}
.seg{display:flex;background:var(--card2);border-radius:12px;padding:3px;margin-top:6px}
.seg button{flex:1;background:transparent;border:0;color:var(--muted);font-size:15px;padding:9px 0;border-radius:9px;cursor:pointer;font-family:inherit}
.seg button.on{background:#3a3b42;color:var(--text);font-weight:700}
.seg.big button{font-size:20px;padding:10px 0}
.lora{border-top:1px solid var(--line);padding-top:14px;margin-top:14px}
.lora:first-of-type{border-top:0;margin-top:0;padding-top:0}
.lora-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;font-size:17px;font-weight:700}
.iconbtn{width:40px;height:40px;border-radius:10px;background:var(--card2);border:1px solid var(--line);color:var(--text);
  font-size:22px;line-height:1;cursor:pointer;display:grid;place-items:center;flex:0 0 auto}
.iconbtn:hover{border-color:#4a4b53}
.pill{background:var(--card2);border:1px solid var(--line);color:var(--text);border-radius:10px;padding:8px 12px;font-size:14px;
  cursor:pointer;font-family:inherit;display:inline-flex;align-items:center;gap:6px}
.pill:hover{border-color:#4a4b53}
.muted{color:var(--muted)}
.check{display:flex;align-items:center;gap:8px;color:var(--muted);margin-top:8px;cursor:pointer;user-select:none}
.seedrow{display:flex;gap:8px}.seedrow input{flex:1}
textarea{min-height:150px;resize:vertical;line-height:1.5}
.gen{width:100%;margin-top:12px;background:var(--accent);border:0;color:#fff;font-size:17px;font-weight:700;padding:14px;border-radius:12px;cursor:pointer}
.gen:hover{background:var(--accent2)}.gen:disabled{opacity:.55;cursor:wait}
.bar{height:6px;background:#2a2b31;border-radius:3px;overflow:hidden;margin-top:12px}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
.status{margin-top:8px;color:var(--muted);font-size:13px;min-height:18px;white-space:pre-wrap}
.status.err{color:var(--err)}
.stage{flex:1;min-height:420px;display:flex;align-items:center;justify-content:center;background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden;position:relative}
.stage > img{width:100%;height:100%;object-fit:contain;display:block}
.stage img.preview{opacity:.75;filter:saturate(.8)}
.stage .empty{color:var(--muted);text-align:center;padding:30px;line-height:1.7}
.stage .tools{position:absolute;right:10px;top:10px;display:flex;gap:6px}
.stage .tools a,.stage .tools button{background:rgba(0,0,0,.55);color:#fff;text-decoration:none;padding:7px 11px;border-radius:9px;font-size:13px;border:0;cursor:pointer;font-family:inherit}
.stage .multi{display:grid;grid-template-columns:repeat(2,1fr);grid-auto-rows:1fr;gap:8px;width:100%;height:100%;padding:8px}
.stage .multi img{width:100%;height:100%;object-fit:contain;background:#0f1013;border-radius:10px;cursor:zoom-in;min-height:0}
.meta{color:var(--muted);font-size:12px;text-align:center}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.grid .cell{aspect-ratio:3/4;border-radius:10px;overflow:hidden;background:var(--card2);cursor:pointer;border:2px solid transparent}
.grid .cell.on{border-color:var(--accent)}
.grid img{width:100%;height:100%;object-fit:cover;display:block}
.grid .ph{aspect-ratio:3/4;border-radius:10px;background:var(--card2);opacity:.45}
.banner{background:#3a1d1d;border:1px solid #6b2b2b;color:#ffb4b4;border-radius:12px;padding:10px 12px;font-size:13px;display:none;white-space:pre-wrap}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:50;padding:16px}
.modal.open{display:flex}
.dialog{background:var(--card);border:1px solid var(--line);border-radius:18px;width:min(560px,100%);max-height:90vh;overflow:auto;padding:20px}
.dialog h3{margin:0 0 4px;font-size:19px;display:flex;justify-content:space-between;align-items:center}
.dialog h4{margin:18px 0 8px;font-size:15px}
.hint{color:var(--muted);font-size:13px;line-height:1.6}
.urlrow{display:flex;gap:8px}.urlrow input{flex:1}
.btn{background:var(--accent);border:0;color:#fff;font-weight:700;border-radius:10px;padding:0 16px;cursor:pointer;font-size:15px;font-family:inherit}
.btn:hover{background:var(--accent2)}
.drop{display:block;border:2px dashed #3a3b42;border-radius:14px;padding:22px;text-align:center;color:var(--muted);cursor:pointer;line-height:1.6}
.drop.over{border-color:var(--accent);color:var(--text);background:#1d1416}
.job{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-top:8px;font-size:13px}
.job .t{display:flex;justify-content:space-between;gap:8px}
.job .t b{font-weight:600;word-break:break-all}
.job .bar{margin-top:8px;height:5px}
.job.done{border-color:#285c43}.job.err{border-color:#6b2b2b}
.job .e{color:var(--err);margin-top:6px;white-space:pre-wrap}
details summary{cursor:pointer;color:var(--muted);margin-top:16px}
details .row{margin-top:10px}
@media (max-width:1100px){.app{grid-template-columns:1fr;height:auto}.col{overflow:visible}.stage{min-height:360px}}
</style>
</head>
<body>
<div class="app">
  <div class="col" id="c1">
    <div class="brand">MagMix Studio<small>Krea 2 · ComfyUI</small></div>
    <div class="banner" id="banner"></div>

    <div class="card">
      <h3><span>LoRA</span><button class="pill" id="openImport" title="เพิ่มไฟล์ LoRA เข้าเครื่อง">⬆ Import LoRA</button></h3>
      <div id="loraList"></div>
    </div>

    <div class="card">
      <h3>Image Size <span class="muted" title="เลือกสัดส่วน แล้วเลือก Small / Normal / Large หรือลากปรับกว้าง-สูงเองได้">ⓘ</span></h3>
      <select id="preset"></select>
      <div class="aspects" id="aspects"></div>
      <div class="seg" id="tiers"></div>
      <div class="row"><span>Width</span><b id="wv"></b></div>
      <input type="range" id="w" min="512" max="2048" step="16">
      <div class="row"><span>Height</span><b id="hv"></b></div>
      <input type="range" id="h" min="512" max="2048" step="16">
    </div>

    <div class="card">
      <h3>Batch Size <span class="muted" title="จำนวนภาพที่สร้างต่อการกด 1 ครั้ง">ⓘ</span></h3>
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
      <h3>Prompt</h3>
      <textarea id="prompt" placeholder="พิมพ์สิ่งที่อยากได้ เช่น a young Thai woman, long hair, soft smile, cozy room, night vibe"></textarea>
      <button class="gen" id="go">✨ สร้างภาพ</button>
      <div class="bar"><i id="bar"></i></div>
      <div class="status" id="status"></div>
    </div>
    <div class="stage" id="stage"><div class="empty">ภาพที่สร้างจะแสดงตรงนี้<br><span style="font-size:12px">กด ✨ สร้างภาพ หรือ Ctrl/⌘ + Enter</span></div></div>
    <div class="meta" id="meta"></div>
  </div>

  <div class="col" id="c3">
    <div class="card">
      <h3>ภาพล่าสุด <span class="muted" style="font-size:13px;font-weight:500" id="gcount"></span></h3>
      <div class="grid" id="grid"></div>
    </div>
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
  seed: Math.floor(Math.random() * 1e14), rand: true, unet: "", clip: "", vae: "", civKey: "", hfTok: ""
};
let S = Object.assign({}, DEF);
try { Object.assign(S, JSON.parse(localStorage.getItem("magmix_studio") || "{}")); } catch (e) {}
if (![1, 2, 3, 4].includes(S.batch)) S.batch = 1;
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

// ---------- LoRA ----------
function renderLoras() {
  const box = $("loraList"); box.innerHTML = "";
  S.loras.forEach((L, i) => {
    if (L.name !== "Disabled" && LISTS.lora.length && !LISTS.lora.includes(L.name)) L.name = "Disabled";
    const d = document.createElement("div"); d.className = "lora";
    const btn = i === 0 ? `<button class="iconbtn" data-act="add" title="เพิ่มช่อง LoRA">+</button>`
                        : `<button class="iconbtn" data-act="del" title="ลบช่องนี้">−</button>`;
    d.innerHTML = `<div class="lora-head"><span>LoRA ${i + 1}</span>${btn}</div><select></select>
      <div class="wrap" style="display:${L.name === "Disabled" ? "none" : "block"}">
        <div class="row"><span>Weight</span><b>${Math.round(L.w)}%</b></div>
        <input type="range" min="0" max="200" step="5" value="${L.w}"></div>`;
    const sel = d.querySelector("select");
    opt(sel, ["Disabled"].concat(LISTS.lora), L.name, true);
    sel.onchange = () => { L.name = sel.value; save(); renderLoras(); };
    const rg = d.querySelector("input[type=range]"); fillRange(rg);
    rg.oninput = () => { L.w = Number(rg.value); d.querySelector(".row b").textContent = L.w + "%"; fillRange(rg); save(); };
    d.querySelector("button").onclick = (e) => {
      if (e.currentTarget.dataset.act === "add") { if (S.loras.length < MAX_LORA) S.loras.push({ name: "Disabled", w: 100 }); }
      else S.loras.splice(i, 1);
      save(); renderLoras();
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
$("preset").onchange = (e) => {
  if (e.target.value === "custom") return;
  const [a, t] = e.target.value.split("|"); S.aspect = a; S.tier = t; [S.width, S.height] = dims(a, t); save(); renderSize();
};
["w", "h"].forEach(id => $(id).addEventListener("input", (e) => {
  S[id === "w" ? "width" : "height"] = Number(e.target.value);
  const match = TIERS.flatMap(([t]) => ASPECTS.map(([a]) => [a, t])).find(([a, t]) => { const [w, h] = dims(a, t); return w === S.width && h === S.height; });
  if (match) { S.aspect = match[0]; S.tier = match[1]; } else { S.aspect = ""; S.tier = ""; }
  save(); renderSize();
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
$("dice").onclick = () => { S.seed = Math.floor(Math.random() * 1e14); $("seed").value = S.seed; save(); };
$("rand").onchange = (e) => { S.rand = e.target.checked; save(); };
$("prompt").value = S.prompt || "";
$("prompt").oninput = (e) => { S.prompt = e.target.value; save(); };

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

// ---------- prompt ----------
function buildPrompt(seed) {
  const P = {
    "10": { class_type: "UNETLoader", inputs: { unet_name: S.unet, weight_dtype: "default" } },
    "11": { class_type: "CLIPLoader", inputs: { clip_name: S.clip, type: "krea2", device: "default" } },
    "12": { class_type: "VAELoader", inputs: { vae_name: S.vae } }
  };
  let model = ["10", 0];
  S.loras.filter(L => L.name !== "Disabled" && L.w > 0).forEach((L, i) => {
    const id = String(30 + i);
    P[id] = { class_type: "LoraLoaderModelOnly", inputs: { model, lora_name: L.name, strength_model: L.w / 100 } };
    model = [id, 0];
  });
  P["13"] = { class_type: "CLIPTextEncode", inputs: { text: S.prompt || "", clip: ["11", 0] } };
  P["14"] = { class_type: "ConditioningZeroOut", inputs: { conditioning: ["13", 0] } };
  P["15"] = { class_type: "EmptyLatentImage", inputs: { width: S.width, height: S.height, batch_size: S.batch } };
  P["16"] = { class_type: "KSampler", inputs: { seed, steps: S.steps, cfg: S.cfg, sampler_name: S.sampler, scheduler: S.scheduler,
                denoise: 1, model, positive: ["13", 0], negative: ["14", 0], latent_image: ["15", 0] } };
  P["17"] = { class_type: "VAEDecode", inputs: { samples: ["16", 0], vae: ["12", 0] } };
  P["18"] = { class_type: "SaveImage", inputs: { filename_prefix: "MagMix", images: ["17", 0] } };
  return P;
}

// ---------- display ----------
const viewUrl = (im) => `${API}/view?filename=${encodeURIComponent(im.filename)}&subfolder=${encodeURIComponent(im.subfolder || "")}&type=${encodeURIComponent(im.type || "output")}`;
let current = null, lastSet = null;
function show(im, meta) {
  current = im;
  const url = viewUrl(im);
  const back = lastSet && lastSet.length > 1 && lastSet.some(x => x.filename === im.filename) ? `<button id="backSet">◀ ดูทั้งชุด</button>` : "";
  $("stage").innerHTML = `<img src="${url}" alt=""><div class="tools">${back}<a href="${url}" target="_blank" rel="noopener">เปิดเต็ม</a><a href="${url}" download="${esc(im.filename)}">ดาวน์โหลด</a></div>`;
  if (back) $("backSet").onclick = () => showSet(lastSet);
  if (meta !== undefined) $("meta").textContent = meta || im.filename;
  markGallery();
}
function showSet(ims, meta) {
  if (ims.length === 1) return show(ims[0], meta);
  lastSet = ims; current = null;
  $("stage").innerHTML = `<div class="multi">${ims.map((im, i) => `<img data-i="${i}" src="${viewUrl(im)}" alt="">`).join("")}</div>`;
  $("stage").querySelectorAll(".multi img").forEach(img => img.onclick = () => show(ims[Number(img.dataset.i)]));
  if (meta !== undefined) $("meta").textContent = meta;
  markGallery();
}
function showPreview(blobUrl) {
  const img = $("stage").querySelector("img.preview");
  if (img) { const old = img.src; img.src = blobUrl; URL.revokeObjectURL(old); }
  else $("stage").innerHTML = `<img class="preview" src="${blobUrl}" alt="">`;
}
function markGallery() {
  const on = new Set(current ? [current.filename] : (lastSet || []).map(x => x.filename));
  document.querySelectorAll(".grid .cell").forEach(c => c.classList.toggle("on", on.has(c.dataset.f)));
}

// ---------- gallery ----------
async function loadGallery() {
  try {
    const r = await fetch(`${API}/history?max_items=200`, { cache: "no-store" });
    const H = await r.json();
    const items = []; let k = 0;
    Object.values(H).forEach(h => {
      const ts = ((h.status && h.status.messages) || []).map(m => m[1] && m[1].timestamp).filter(Boolean)[0] || 0;
      Object.values(h.outputs || {}).forEach(o => (o.images || []).forEach(im => { if (im.type === "output") items.push({ im, ts, k: k++ }); }));
    });
    items.sort((a, b) => (b.ts - a.ts) || (b.k - a.k));
    const last = items.slice(0, 12);
    const g = $("grid"); g.innerHTML = "";
    last.forEach(({ im }) => {
      const c = document.createElement("div"); c.className = "cell"; c.dataset.f = im.filename;
      c.innerHTML = `<img loading="lazy" src="${viewUrl(im)}" alt="">`;
      c.onclick = () => show(im, im.filename); g.appendChild(c);
    });
    for (let i = last.length; i < 12; i++) { const p = document.createElement("div"); p.className = "ph"; g.appendChild(p); }
    $("gcount").textContent = last.length ? `${last.length}/12` : "";
    markGallery();
    return last;
  } catch (e) { return []; }
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
        status(`กำลังสร้าง${running.batch > 1 ? " " + running.batch + " ภาพ" : ""}… ${d.value}/${d.max}`);
      } else if (m.type === "executing" && d.node) {
        const n = Number(d.node);
        status({ 10: "โหลดโมเดล…", 11: "โหลด text encoder…", 12: "โหลด VAE…", 13: "อ่าน prompt…", 17: "แปลงเป็นภาพ…" }[n] || (n >= 30 && n < 40 ? "โหลด LoRA…" : $("status").textContent));
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
function finish(images) {
  if (!running || !images || !images.length) return;
  const sec = ((Date.now() - running.t0) / 1000).toFixed(1);
  const meta = `${running.w}×${running.h} · ${images.length} ภาพ · seed ${running.seed} · ${sec}s`;
  running = null; $("go").disabled = false; $("bar").style.width = "100%";
  status(`✅ เสร็จใน ${sec} วินาที`);
  lastSet = images; showSet(images, meta); loadGallery();
}
function fail(msg) { running = null; $("go").disabled = false; $("bar").style.width = "0"; status(msg, true); }
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
async function generate() {
  if (running) return;
  if (!S.unet || !S.clip || !S.vae) { status("ยังโหลดรายชื่อโมเดลไม่ได้ — ตรวจว่า ComfyUI เปิดอยู่", true); return; }
  if (S.rand) { S.seed = Math.floor(Math.random() * 1e14); $("seed").value = S.seed; save(); }
  const seed = S.seed;
  $("go").disabled = true; $("bar").style.width = "0"; status("ส่งงาน…");
  try {
    const r = await fetch(`${API}/prompt`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: buildPrompt(seed), client_id: CLIENT_ID }) });
    const j = await r.json();
    if (!r.ok || j.error) {
      const ne = j.node_errors ? Object.values(j.node_errors).map(n => typeof n === "string" ? n : `${n.class_type}: ${(n.errors || []).map(e => e.details || e.message).join(", ")}`).join("\n") : "";
      return fail("❌ " + ((j.error && (j.error.message || j.error)) || `HTTP ${r.status}`) + (ne ? "\n" + ne : ""));
    }
    running = { id: j.prompt_id, seed, t0: Date.now(), batch: S.batch, w: S.width, h: S.height };
    status(j.number ? `เข้าคิวแล้ว (ลำดับ ${j.number})` : "เข้าคิวแล้ว");
    poll(j.prompt_id);
  } catch (e) { fail("❌ ส่งงานไม่สำเร็จ: " + e.message); }
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
renderLoras(); renderSize(); renderBatch(); renderSampler();
connectWS(); loadLists();
loadGallery().then(last => { if (last.length) show(last[0].im, last[0].im.filename); });
})();
</script>
</body>
</html>
'''

ROUTES = PromptServer.instance.routes
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
