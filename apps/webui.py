#!/usr/bin/env python3
"""webui.py — a local, Claude-style chat UI for your personal brain.

Run it, open the printed URL in a browser tab, and it's an app. Everything stays on your machine
(binds to localhost by default). No extra dependencies — Python standard library + your existing
personal_brain.py. Each person runs their own instance, which keeps it private and on-device.

    python3 apps/webui.py            # -> http://127.0.0.1:8765
    PORT=9000 python3 apps/webui.py  # custom port
    HOST=0.0.0.0 python3 apps/webui.py   # share on your LAN (only do this on a trusted network)

In the chat: ask anything; start a message with 'remember ' to teach it a fact.
Honest scope is the same as personal_brain.py — exact/flat memory, embedder recall with holes,
answers at your local model's quality.
"""
import sys, os, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import personal_brain as pb

brain = pb.Brain()
lock = threading.Lock()

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal Brain</title>
<style>
  :root{ --bg:#faf9f7; --panel:#ffffff; --ink:#1f1e1c; --muted:#8a857d; --line:#ece9e3;
         --user:#1f1e1c; --userink:#faf9f7; --ai:#f3f1ec; --accent:#c8613a; }
  @media (prefers-color-scheme:dark){ :root{ --bg:#1a1917; --panel:#211f1c; --ink:#ece9e3;
         --muted:#8f897f; --line:#302d29; --user:#ece9e3; --userink:#1a1917; --ai:#2a2724; --accent:#e0855c; } }
  *{ box-sizing:border-box } html,body{ height:100% }
  body{ margin:0; background:var(--bg); color:var(--ink);
        font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .app{ max-width:760px; margin:0 auto; height:100dvh; display:flex; flex-direction:column }
  header{ display:flex; align-items:center; justify-content:space-between; padding:16px 20px;
          border-bottom:1px solid var(--line); }
  .brand{ display:flex; align-items:center; gap:9px; font-weight:600; letter-spacing:-.01em }
  .brand .dot{ font-size:20px }
  .meta{ font-size:13px; color:var(--muted) }
  main{ flex:1; overflow-y:auto; padding:24px 20px; display:flex; flex-direction:column; gap:16px }
  .hello{ color:var(--muted); text-align:center; margin:auto 0; max-width:460px; align-self:center }
  .hello h1{ color:var(--ink); font-size:20px; font-weight:600; margin:0 0 8px }
  .msg{ display:flex; max-width:100% } .msg.me{ justify-content:flex-end }
  .bubble{ padding:11px 15px; border-radius:16px; max-width:82%; white-space:pre-wrap; word-wrap:break-word }
  .me .bubble{ background:var(--user); color:var(--userink); border-bottom-right-radius:5px }
  .ai .bubble{ background:var(--ai); border-bottom-left-radius:5px }
  .sys .bubble{ background:transparent; color:var(--muted); font-size:14px; padding:4px 6px; text-align:center; margin:0 auto }
  .think{ color:var(--muted); font-style:italic }
  footer{ border-top:1px solid var(--line); padding:12px 16px 16px }
  .composer{ display:flex; gap:10px; align-items:flex-end; background:var(--panel);
             border:1px solid var(--line); border-radius:18px; padding:8px 8px 8px 16px }
  textarea{ flex:1; border:0; resize:none; background:transparent; color:var(--ink); font:inherit;
            outline:none; max-height:160px; padding:6px 0 }
  button{ border:0; width:38px; height:38px; border-radius:12px; background:var(--accent); color:#fff;
          font-size:18px; cursor:pointer; flex:0 0 auto } button:disabled{ opacity:.4; cursor:default }
  .hint{ text-align:center; color:var(--muted); font-size:12px; margin-top:9px }
  .hint b{ color:var(--ink); font-weight:600 }
</style></head><body>
<div class="app">
  <header>
    <div class="brand"><span class="dot">🧠</span><span>Personal Brain</span></div>
    <div class="meta" id="meta">local · private</div>
  </header>
  <main id="chat">
    <div class="hello" id="hello">
      <h1>Your private, on-device memory</h1>
      Ask anything about what you've told it — in plain, messy English.
      Start a message with <b>remember&nbsp;…</b> to teach it a new fact.
    </div>
  </main>
  <footer>
    <div class="composer">
      <textarea id="input" rows="1" placeholder="Talk to me — tell me things or ask about them"></textarea>
      <button id="send" title="Send">↑</button>
    </div>
    <div class="hint">Runs entirely on your machine · <span id="facts">0 facts</span></div>
  </footer>
</div>
<script>
const chat=document.getElementById('chat'), input=document.getElementById('input'),
      send=document.getElementById('send'), factsEl=document.getElementById('facts'),
      hello=document.getElementById('hello');
function add(role,text){ if(hello)hello.remove();
  const m=document.createElement('div'); m.className='msg '+(role==='me'?'me':role==='sys'?'sys':'ai');
  const b=document.createElement('div'); b.className='bubble'; b.textContent=text; m.appendChild(b);
  chat.appendChild(m); chat.scrollTop=chat.scrollHeight; return b; }
function setFacts(n){ factsEl.textContent=n+' fact'+(n===1?'':'s'); }
async function state(){ try{ const r=await fetch('/api/state'); const j=await r.json();
  setFacts(j.facts); document.getElementById('meta').textContent='local · recall: '+j.mode; }catch(e){} }
async function submit(){ const msg=input.value.trim(); if(!msg) return;
  input.value=''; input.style.height='auto'; add('me',msg); send.disabled=true;
  const t=add('ai','…'); t.classList.add('think');
  try{ const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:msg})}); const j=await r.json();
    t.classList.remove('think'); t.textContent=j.text; t.parentElement.className='msg '+(j.role==='system'?'sys':'ai');
    setFacts(j.facts);
  }catch(e){ t.textContent='(error reaching the local server)'; }
  send.disabled=false; input.focus(); }
send.onclick=submit;
input.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); submit(); } });
input.addEventListener('input',()=>{ input.style.height='auto'; input.style.height=Math.min(input.scrollHeight,160)+'px'; });
state(); input.focus();
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps({"facts": len(brain.facts), "mode": brain.mode}))
        else:
            self._send(404, "not found", "text/plain")
    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, "not found", "text/plain"); return
        n = int(self.headers.get("Content-Length", 0) or 0)
        try: data = json.loads(self.rfile.read(n) or b"{}")
        except Exception: data = {}
        msg = (data.get("message") or "").strip()
        with lock:
            text = brain.chat(msg) if msg else "Say something — tell me things or ask about them."
            reply = {"role": "assistant", "text": text}
        self._send(200, json.dumps({**reply, "facts": len(brain.facts)}))
    def log_message(self, *a): pass

def main():
    host = os.environ.get("HOST", "127.0.0.1"); port = int(os.environ.get("PORT", "8765"))
    print(f"Personal Brain UI  ->  http://{host}:{port}")
    print(f"  {len(brain.facts)} facts loaded · recall: {brain.mode} · all local")
    print("  open that URL in a browser tab. Ctrl+C to stop.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()

if __name__ == "__main__":
    main()
