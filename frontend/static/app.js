const $ = id => document.getElementById(id);
let session = sessionStorage.getItem('visionfind-session'), busy = false, pending = [], recording = null;
let activeJob = sessionStorage.getItem('visionfind-job');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function notice(text='') { $('notice').textContent = text; }
function controls(value) { busy=value; for(const id of ['send','attach','voice','new-chat']) $(id).disabled=value; }
async function api(path, options={}) {
  const response = await fetch('/api/' + path, options);
  let data; try { data=await response.json(); } catch { throw Error('Connection failed. Check the local server.'); }
  if(!response.ok) { const error=Error(typeof data.detail==='string' ? data.detail : 'Request could not complete.'); error.status=response.status; throw error; }
  return data;
}
function fit() { $('query').style.height='auto'; $('query').style.height=Math.min($('query').scrollHeight,150)+'px'; }
function openPreview(url, video=false) {
  const element=document.createElement(video?'video':'img'); element.src=url;
  if(video) element.controls=true;
  $('preview-content').replaceChildren(element); $('preview').showModal();
}
$('close-preview').onclick=()=>{ $('preview').close(); $('preview-content').replaceChildren(); };
$('preview').onclick=e=>{ if(e.target===$('preview')) $('close-preview').click(); };
function mediaNode(url, caption='', video=false, large=false) {
  const wrap=document.createElement('div'); wrap.className='thumb-wrap';
  const media=document.createElement(video?'video':'img'); media.src=url;
  media.className=large ? (video?'result-video':'result-image') : 'thumb';
  if(video && large) { media.controls=true; media.preload='metadata'; }
  else { media.tabIndex=0; media.onclick=()=>openPreview(url,video); media.onkeydown=e=>{ if(e.key==='Enter')openPreview(url,video); }; }
  if(!video)media.alt=caption || 'Conversation image';
  wrap.append(media);
  if(caption){ const label=document.createElement('div'); label.className='caption'; label.textContent=caption; wrap.append(label); }
  return wrap;
}
function pendingPreview() {
  $('attachments').replaceChildren();
  pending.forEach((item,index)=>{
    const element=mediaNode(item.url,item.file.name,item.file.type.startsWith('video/'));
    const remove=document.createElement('button'); remove.type='button'; remove.className='remove'; remove.textContent='×';
    remove.setAttribute('aria-label','Remove '+item.file.name);
    remove.onclick=()=>{ URL.revokeObjectURL(item.url); pending.splice(index,1); pendingPreview(); };
    element.append(remove); $('attachments').append(element);
  });
}
$('attach').onclick=()=>{ $('picker').value=''; $('picker').click(); };
$('picker').onchange=e=>{
  const files=[...e.target.files]; const combined=[...pending.map(p=>p.file),...files];
  const video=combined.some(f=>f.type.startsWith('video/') || /\.(mp4|mov|webm|avi|mkv)$/i.test(f.name));
  if(combined.length>10 || (video && combined.length>1)) return notice('Attach up to ten images or one video.');
  if(files.some(f=>f.size>(video?100:20)*1024*1024))return notice('Images: up to 20 MB each. Video: up to 100 MB.');
  for(const file of files)pending.push({file,url:URL.createObjectURL(file)});
  notice(); pendingPreview();
};
function textNode(text) {
  const div=document.createElement('div'); div.className='text';
  // Render emphasis with text nodes, never injected model HTML.
  text.split(/(\*\*[^*]+\*\*)/g).forEach(part=>{
    if(part.startsWith('**') && part.endsWith('**')){ const bold=document.createElement('strong');bold.textContent=part.slice(2,-2);div.append(bold); }
    else div.append(document.createTextNode(part));
  }); return div;
}
function render(messages) {
  const pane=$('conversation'); pane.replaceChildren();
  for(const message of messages){
    const div=document.createElement('div'); div.className='message '+message.role;
    const c=message.content;
    if(c.kind==='text'){ if(!c.text.trim())continue; div.append(textNode(c.text)); }
    else if(c.kind==='gallery'){
      const row=document.createElement('div'); row.className='media-row';
      for(const item of c.items)row.append(mediaNode('/api/assets/'+item.asset_id,item.caption || 'Image'));
      div.append(row);
    } else if(['image','video','audio','file'].includes(c.kind)){
      const url='/api/assets/'+c.asset_id;
      if(c.kind==='image'||c.kind==='video') div.append(mediaNode(url,'',c.kind==='video',message.role!=='user'));
      else if(c.kind==='audio'){const audio=document.createElement('audio');audio.controls=true;audio.src=url;div.append(audio);}
      else {const a=document.createElement('a');a.href=url;a.textContent='Download result';a.download='';div.append(a);}
    }
    pane.append(div);
  }
  pane.scrollTop=pane.scrollHeight;
}
function statusNode(text){ let element=$('job-status'); if(!element){element=document.createElement('div');element.id='job-status';element.className='pending';$('conversation').append(element);} element.textContent=text; $('conversation').scrollTop=$('conversation').scrollHeight; }
async function poll(jobId) {
  let errors=0;
  for(;;){
    let result;
    try{ result=await api('jobs/'+jobId); errors=0; }
    catch(error){if(error.status===404){activeJob=null;session=null;sessionStorage.removeItem('visionfind-job');sessionStorage.removeItem('visionfind-session');throw Error('The backend session ended. Start a new chat.');}if(++errors>4)throw error;statusNode('Reconnecting');await sleep(2500);continue;}
    if(result.status==='completed'){render(result.messages);sessionStorage.removeItem('visionfind-job');activeJob=null;return;}
    if(result.status==='failed'){sessionStorage.removeItem('visionfind-job');activeJob=null;throw Error(result.error || 'Processing failed.');}
    statusNode(result.status==='queued'?'Waiting for the model':'Working on your request');
    await sleep(1500);
  }
}
$('composer').onsubmit=async e=>{
  e.preventDefault(); if(busy)return; if(activeJob)return notice('Reload this page to resume the current request.');
  const text=$('query').value.trim(); if(!text && !pending.length)return;
  if(recording)recording.stop();
  controls(true); notice();
  try{
    if(!session){const data=await api('sessions',{method:'POST'});session=data.session_id;sessionStorage.setItem('visionfind-session',session);}
    let uploadIds=[];
    if(pending.length){notice('Uploading attachments…');const form=new FormData();pending.forEach(p=>form.append('files',p.file));const data=await api(`sessions/${session}/uploads`,{method:'POST',body:form});uploadIds=data.upload_ids;}
    const data=await api(`sessions/${session}/turns`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,upload_ids:uploadIds})});
    $('welcome')?.remove();
    const user=document.createElement('div');user.className='message user';if(text)user.append(textNode(text));
    if(pending.length){const row=document.createElement('div');row.className='media-row';pending.forEach(p=>row.append(mediaNode(p.url,p.file.name,p.file.type.startsWith('video/'))));user.append(row);}
    $('conversation').append(user);
    const oldPending=pending;pending=[];pendingPreview();$('query').value='';fit();notice();
    activeJob=data.job_id;sessionStorage.setItem('visionfind-job',activeJob);
    try{await poll(activeJob);}finally{oldPending.forEach(p=>URL.revokeObjectURL(p.url));}
  }catch(error){notice(error.message+(activeJob?' Reload to resume the pending job.':''));}
  finally{controls(false);$('query').focus();}
};
$('query').oninput=fit;
$('query').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('composer').requestSubmit();}};
$('new-chat').onclick=async()=>{
  if(busy||activeJob)return notice('Wait for the current request first.');
  if(session){try{await api('sessions/'+session,{method:'DELETE'});}catch{}}
  session=null;sessionStorage.removeItem('visionfind-session');location.reload();
};
for(const name of ['chat','help']) $(name+'-tab').onclick=()=>{
  $('chat-page').hidden=name!=='chat';$('help-page').hidden=name!=='help';
  $('chat-tab').classList.toggle('active',name==='chat');$('help-tab').classList.toggle('active',name==='help');
};
const Speech=window.SpeechRecognition || window.webkitSpeechRecognition;
$('voice').onclick=()=>{
  if(!Speech)return notice('Browser dictation is unavailable here. You can type your query.');
  if(recording){recording.stop();return;}
  const recognition=new Speech();recording=recognition;recognition.lang='en-US';recognition.continuous=false;recognition.interimResults=false;
  const original=$('query').value;
  recognition.onstart=()=>{$('voice').classList.add('recording');notice('Listening… Your words will be editable before sending.');};
  recognition.onresult=e=>{$('query').value=(original+' '+e.results[0][0].transcript).trim();fit();notice('Review the transcript, then send.');};
  recognition.onerror=e=>notice('Dictation stopped: '+e.error+'. You can type instead.');
  recognition.onend=()=>{recording=null;$('voice').classList.remove('recording');};
  try{recognition.start();}catch{recording=null;notice('Could not start dictation.');}
};
async function connect(){try{await api('connection');$('connection').textContent='Connected';$('connection').classList.add('ready');}catch{$('connection').textContent='Offline';notice('Check Kaggle, its tunnel, and your local configuration.');}}
connect();
if(activeJob){controls(true);statusNode('Resuming your request');poll(activeJob).catch(e=>notice(e.message)).finally(()=>controls(false));}
