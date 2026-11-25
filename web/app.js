'use strict';
const $ = id => document.getElementById(id);
const video = $('comparison');
let session = null, polling = null, displayedRevision = null, messageCount = 0, pending = false;
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(typeof body.detail === 'string' ? body.detail : 'The request could not be completed.'); }
  return response.json();
}
const post = value => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
function showError(error) { $('error').textContent = error.message; $('error').hidden = false; }
function controls(busy) {
  pending = busy;
  $('send').disabled = busy || !session;
  $('new-session').disabled = busy;
  $('undo').disabled = busy || !session?.can_undo;
  document.querySelectorAll('[data-command]').forEach(button => button.disabled = busy || !session);
  $('chat-form').setAttribute('aria-busy', String(busy));
}
function messages(rows) {
  if (messageCount > rows.length) { $('messages').replaceChildren(); messageCount = 0; }
  for (const row of rows.slice(messageCount)) {
    const item = document.createElement('div'); item.className = `message ${row.role}`;
    const label = document.createElement('div'); label.className = 'who'; label.textContent = row.role === 'user' ? 'YOU' : 'WALKEDIT';
    const content = document.createElement('p'); content.textContent = row.text;
    item.append(label, content); $('messages').append(item);
  }
  if (rows.length !== messageCount) { messageCount = rows.length; $('messages').scrollTop = $('messages').scrollHeight; }
}
function display(data) {
  session = data;
  localStorage.setItem('walkedit-session', data.id);
  messages(data.messages);
  $('status-text').textContent = data.message;
  $('render-status').className = 'render-status' + (data.state === 'rendering' ? ' busy' : data.state === 'failed' ? ' failed' : '');
  $('source-duration').textContent = `${data.source.seconds.toFixed(2)} s`;
  $('speed').replaceChildren(document.createTextNode(String(Number(data.plan.speed.toFixed(3)))), Object.assign(document.createElement('span'), {textContent:'×'}));
  $('scope').textContent = data.plan.start === null ? 'Whole motion' : `Source ${data.plan.start}–${data.plan.end} s`;
  $('ramp').textContent = data.plan.transition ? `${data.plan.transition} s transition at each edge` : 'No transition ramp';
  if (data.result && data.revision !== displayedRevision) {
    const base = `/api/sessions/${data.id}/renders/${data.revision}/`;
    video.src = base + 'comparison.mp4'; video.classList.add('ready'); $('empty-stage').hidden = true;
    $('preview-kind').textContent = 'Rendered 3D edit / 30 FPS';
    $('edited-duration').textContent = `${data.result.edited_seconds.toFixed(2)} s`;
    $('render-details').textContent = `${data.result.edited_frames} edited frames · final pose held after completion`;
    for (const [id,file] of [['download-video','comparison.mp4'],['download-motion','edited.pt'],['download-plan','edit.json']]) { $(id).href = base + file + '?download=true'; $(id).removeAttribute('aria-disabled'); }
    const times = data.result.source_times;
    const maxOutput = Math.max((times.length - 1) / data.result.output_fps, data.source.span);
    // Retain original frame indices when decimating the plot.
    const step = Math.max(1,Math.floor(times.length/140));
    const path = times.map((t,i) => (i%step===0 || i===times.length-1) ? `${i===0?'M':'L'}${(12+696*(i/data.result.output_fps)/maxOutput).toFixed(2)} ${(88-76*t/data.source.span).toFixed(2)}` : '').join(' ');
    $('time-line').setAttribute('d',path);
    $('identity-line').setAttribute('d',`M12 88 L${12+696*data.source.span/maxOutput} 12`);
    displayedRevision = data.revision;
  }
  controls(data.state === 'rendering');
}
async function poll() {
  clearTimeout(polling);
  try { const data = await api(`/api/sessions/${session.id}`); display(data); if (data.state === 'rendering') polling = setTimeout(poll,1000); }
  catch(error) { controls(false); showError(error); $('status-text').textContent = 'Connection interrupted. Reload to check the render.'; }
}
async function send(text) {
  if (!text.trim() || pending || !session) return;
  $('error').hidden = true; controls(true); $('status-text').textContent = 'Interpreting your command…';
  try { const data = await api(`/api/sessions/${session.id}/messages`,post({text})); display(data); $('command').value = ''; if (data.state === 'rendering') polling = setTimeout(poll,700); }
  catch(error) { controls(false); showError(error); }
}
$('chat-form').addEventListener('submit',event => {event.preventDefault();send($('command').value);});
$('command').addEventListener('keydown',event => {if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();send($('command').value);}});
document.querySelectorAll('[data-command]').forEach(button => button.addEventListener('click',() => send(button.dataset.command)));
$('undo').addEventListener('click',() => send('Undo'));
async function newSession() {
  const data = await api('/api/sessions',{method:'POST'});
  clearTimeout(polling); messageCount = 0; $('messages').replaceChildren(); displayedRevision = null;
  video.pause(); video.removeAttribute('src');video.load();video.classList.remove('ready');$('empty-stage').hidden=false;
  $('edited-duration').textContent='—';$('preview-kind').textContent='Ready for your first edit';$('render-details').textContent='A command creates a new motion file and MP4.';
  document.querySelectorAll('.download').forEach(link=>{link.removeAttribute('href');link.setAttribute('aria-disabled','true');});
  $('time-line').setAttribute('d','M12 88 708 12');$('identity-line').setAttribute('d','M12 88 708 12');display(data);
}
$('new-session').addEventListener('click',()=>newSession().catch(showError));
async function initialize() {
  controls(true);
  try {
    const health = await api('/api/health');
    if (!health.ready) throw new Error(health.error);
    const saved = localStorage.getItem('walkedit-session');
    if (saved && /^[a-f0-9]{32}$/.test(saved)) { try { display(await api(`/api/sessions/${saved}`)); } catch { await newSession(); } }
    else await newSession();
    if (session.state==='rendering') poll();
    if (!session.result && health.reference_available) { video.src='/reference.mp4';video.classList.add('ready');$('empty-stage').hidden=true;$('preview-kind').textContent='Reference example / 1.5× — send a command to edit'; }
  } catch(error) { showError(error);$('status-text').textContent='Motion setup required';controls(false); }
}
initialize();
