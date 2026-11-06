'use strict';
(() => {
  let data = JSON.parse(document.getElementById('snapshot-data').textContent);
  // Page capabilities belong to the served application, never to replaceable reports.
  const livePage = data.live === true;
  let sourceMode = livePage ? 'server' : 'snapshot';
  let directory = null, polling = false, lastStamp = null, lastDelivery = 0, taskTick = null;
  let transcriptKey = null, taskKey = null, transcriptSession = null, connectionRevision = 0, followLatest = true;
  const savedSessions = new Map();
  let selectedSession = '';
  try {
    const savedSession = localStorage.getItem('swift-session-view');
    if (typeof savedSession === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(savedSession)) selectedSession = savedSession;
  } catch (_) {}
  function rememberSession(sessionId) {
    selectedSession = sessionId;
    try {
      if (sessionId) localStorage.setItem('swift-session-view', sessionId);
      else localStorage.removeItem('swift-session-view');
    } catch (_) {}
  }
  let recoveryGeneration = 0;
  const runtime = {available:false,connected:false,connecting:false,startupTimer:null,status:'disconnected',inputTurnOpen:null,ws:null,media:null,context:null,processor:null,input:null,mute:null,sources:new Map(),playEpoch:0,nextPlay:0,interrupting:false,liveReport:null,aec:null,engagement:null,runId:null,sessionId:null,wakeCueSessionId:null,wakeCueTimer:null,turnId:0,completedTurnId:null,eventSequence:0,statusSequence:0,micMuted:false,micLevel:0,micVisual:0,playLevel:0,toolActive:false,ended:false,mode:null,requestBudget:null,artifactsAvailable:false,deviceCapabilities:null,deviceActionPending:null,deviceActionMessage:'',music:{capabilities:null,capabilitiesLoaded:false,capabilitiesError:false,connectRequested:false,state:null,player:null,sdkLoaded:false,sdkLoading:false,sdkConnecting:false,sdkReady:false,activationRequired:false,deviceId:null,stateUpdatedAt:0,apiTimer:null,operation:0,reportedPlaybackStatus:null,reportedTrackKey:null}};
  const $ = id => document.getElementById(id);
  const spotifyPlayIcon='<svg class="control-svg" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 5v14l11-7z"/></svg>';
  const spotifyPauseIcon='<svg class="control-svg" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 5v14M15 5v14"/></svg>';
  $('aec-probe-link').hidden = !livePage;
  $('recording-lab-link').hidden = !livePage;
  const pageNonce=document.currentScript?.nonce||'';
  try{const mode=localStorage.getItem('swift-conversation-mode');if(['single','followup','extended'].includes(mode))$('conversation-mode').value=mode;}catch(_){}
  $('conversation-mode').addEventListener('change',()=>{try{localStorage.setItem('swift-conversation-mode',$('conversation-mode').value);}catch(_){}});
  const voiceNode=$('voice-status'), waveBars=[...document.querySelectorAll('.signal-bars i')];
  document.documentElement.dataset.hidden=String(document.hidden);
  document.addEventListener('visibilitychange',()=>{document.documentElement.dataset.hidden=String(document.hidden);});
  const escape = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const format = (v,d=0) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString(undefined,{maximumFractionDigits:d}) : 'Unavailable';
  const numeric = v => typeof v==='number' && Number.isFinite(v);
  const quantity = (v,unit='',d=0) => numeric(v)?format(v,d)+unit:'Unavailable';
  const text = (id,value) => {if($(id).textContent!==value)$(id).textContent=value;};
  const date = v => Number.isFinite(Date.parse(v))?new Date(v).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'Unknown';
  const sessionLabels = {idle:'Idle',listening:'Listening',thinking:'Processing',speaking:'Speaking',interrupted:'Interrupted',ended:'Session ended'};
  const friendlyPhrase = value => typeof value==='string'?(/^hey[_ ]swift$/i.test(value)?'Hey Swift':value):'Phrase event';
  const sourceLabel = d => d.source_kind==='pcm-stdin'?'PCM stream diagnostic':d.source_kind==='wav-file'?'File replay':'Audio report';
  const reports = k => data.reports.filter(r=>r.kind===k||(k==='trace'&&r.kind==='replay'&&Array.isArray(r.data.turns))), latest = k=>reports(k)[0];
  const filename = r=>r.source.split('/').pop();
  const labels = {'HPC-live':'HPC execution',mock:'Mock',review:'Review only',inventory:'Inventory',reported:'Unverified'};
  const taskKinds=new Set(['music_play','music_playlist','music_pause','music_resume','music_stop','music_next','music_previous','music_volume_up','music_volume_down','music_volume_set','light_on','light_off','set_color','set_brightness','set_timer','set_alarm','cancel_schedule','set_temperature','create_reminder','list_reminders','get_time','get_weather','agent']);
  const rgbNames={red:'#FF0000',green:'#00FF00',blue:'#0000FF',orange:'#FF8000',purple:'#8000FF',pink:'#FF69B4',yellow:'#FFFF00',cyan:'#00FFFF',magenta:'#FF00FF',white:'#FFFFFF','warm white':'#FFF0D1',teal:'#00A69E',amber:'#FFAB00',violet:'#8F00FF',lime:'#80FF00',navy:'#000080',coral:'#FF8050'};
  const safeRgbColor=value=>typeof value==='string'?(rgbNames[value.trim().toLowerCase()]||(/^#?[\da-f]{6}$/i.test(value.trim())?'#'+value.trim().replace(/^#/,'').toUpperCase():null)):null;
  const rgbColorLabel=value=>rgbNames[String(value).trim().toLowerCase()]?String(value).trim().toLowerCase():safeRgbColor(value)||'unknown';
  const validTaskWidget=w=>{
    if(!w||typeof w!=='object'||Array.isArray(w))return false;
    if(w.type==='lamp')return Object.keys(w).sort().join(',')==='brightness,color,on,type'&&typeof w.on==='boolean'&&safeRgbColor(w.color)!==null&&Number.isInteger(w.brightness)&&w.brightness>=0&&w.brightness<=100;
    if(w.type==='timer')return Object.keys(w).sort().join(',')==='active,deadline_utc,display_value,schedule_kind,target_local,type'&&['timer','alarm'].includes(w.schedule_kind)&&typeof w.active==='boolean'&&typeof w.deadline_utc==='string'&&w.deadline_utc.length<=60&&(!w.deadline_utc||Number.isFinite(Date.parse(w.deadline_utc)))&&typeof w.target_local==='string'&&w.target_local.length<=60&&typeof w.display_value==='string'&&w.display_value.length<=32;
    if(w.type==='temperature')return Object.keys(w).sort().join(',')==='measurement,setpoint_c,type'&&Number.isInteger(w.setpoint_c)&&w.setpoint_c>=16&&w.setpoint_c<=30&&w.measurement===false;
    if(w.type==='reminders')return Object.keys(w).sort().join(',')==='items,persistent,type'&&Array.isArray(w.items)&&w.items.length<=3&&w.items.every(x=>['drink water','study','exercise'].includes(x))&&typeof w.persistent==='boolean';
    if(w.type==='time')return Object.keys(w).sort().join(',')==='local_time,timezone,type'&&typeof w.local_time==='string'&&w.local_time.length<=20&&typeof w.timezone==='string'&&w.timezone.length<=80;
    if(w.type==='weather')return Object.keys(w).sort().join(',')==='condition,location,observed_utc,provider,temperature_c,type'&&typeof w.location==='string'&&w.location.length<=80&&typeof w.condition==='string'&&w.condition.length<=50&&numeric(w.temperature_c)&&w.temperature_c>=-100&&w.temperature_c<=70&&typeof w.observed_utc==='string'&&w.observed_utc.length<=60&&Number.isFinite(Date.parse(w.observed_utc))&&w.provider==='Open-Meteo';
    return false;
  };
  const pill = (text,color='subtle')=>`<span class="pill ${color}">${escape(text)}</span>`;
  function deviceAdapterLabel(capability){
    if(!capability)return 'Device adapter · not checked';
    if(capability.target==='mock')return 'Device adapter · simulation';
    if(capability.target==='pi'&&capability.available===true)return 'Pi adapter ready · output unverified';
    if(capability.target==='pi'&&capability.available===false)return 'Raspberry Pi is not connected';
    return 'Pi adapter · checked when voice starts';
  }
  function renderDeviceAdapterStatus(){
    const node=$('device-adapter-status');
    if(node)text('device-adapter-status',deviceAdapterLabel(runtime.deviceCapabilities));
  }
  function acceptDeviceCapabilities(value){
    if(!value||!['mock','pi','unavailable'].includes(value.target)||!['simulated','ready_unverified','unavailable','not_initialized'].includes(value.status)||!(value.available===true||value.available===false||value.available===null)||value.physical_readback!==false)return null;
    return {target:value.target,status:value.status,available:value.available,physical_readback:false,message:typeof value.message==='string'?value.message.slice(0,120):''};
  }
  function rgbModuleIllustration(color, opacity){
    const ledColor=safeRgbColor(color)||'#94A3B8';
    const level=Math.max(0.08,Math.min(1,Number(opacity)||0.08));
    return `<span class="rgb-module-illustration" aria-hidden="true"><svg viewBox="0 0 72 72"><rect class="module-board" x="5" y="5" width="62" height="62" rx="10"/><path class="module-trace" d="M14 13h8v5m36-5h-8v5M14 51h8m28 0h8"/><path class="module-lens-shell" d="M24 25a12 12 0 1 1 24 0c0 5-3 8-5 10H29c-2-2-5-5-5-10Z"/><ellipse class="module-lens-color" cx="36" cy="23" rx="10" ry="10" fill="${ledColor}" opacity="${level}"/><path class="module-highlight" d="M30 19c1-3 3-4 6-4"/><path class="module-led-leg" d="M31 35v5m10-5v5"/><rect class="module-resistor" x="15" y="42" width="9" height="6" rx="2"/><path class="module-resistor-mark" d="M19 43v4"/><rect class="module-resistor" x="31.5" y="42" width="9" height="6" rx="2"/><path class="module-resistor-mark" d="M36 43v4"/><rect class="module-resistor" x="48" y="42" width="9" height="6" rx="2"/><path class="module-resistor-mark" d="M52.5 43v4"/><circle class="module-pad" cx="14" cy="56" r="2.5"/><circle class="module-pad" cx="29" cy="56" r="2.5"/><circle class="module-pad" cx="44" cy="56" r="2.5"/><circle class="module-pad" cx="59" cy="56" r="2.5"/><path class="module-pin" d="M14 58v7m15-7v7m15-7v7m15-7v7"/></svg></span>`;
  }
  function deviceControlsMarkup(widget){
    if(!livePage||!runtime.connected)return '';
    const capability=runtime.deviceCapabilities;
    const mock=capability?.target==='mock';
    const available=mock||capability?.target==='pi'&&capability.available===true;
    const disabled=!available||runtime.deviceActionPending!==null;
    const target=mock?'mock simulation':'Raspberry Pi';
    const note=capability?.target==='pi'&&capability.available===false?'Raspberry Pi is not connected. Controls are disabled.':mock?'Mock simulation · no physical hardware is changed.':'Pi command · physical output is not sensor-verified.';
    const color=safeRgbColor(widget?.color)||'#0000FF';
    const brightness=Number.isInteger(widget?.brightness)?widget.brightness:60;
    const button=(label,action,value,source)=>`<button type="button" class="device-control" data-device-action="${action}"${value===undefined?'':` data-device-value="${escape(value)}"`}${source?` data-device-value-source="${source}"`:''} ${disabled?'disabled':''}>${label}</button>`;
    return `<details class="device-controls"><summary>RGB controls · ${target}</summary><div class="device-control-groups" data-light-controls><div class="device-power-row"><span>Output</span>${button('On','light_on')}${button('Off','light_off')}</div><div class="device-color-row"><label>Color<input class="device-color-input" type="color" aria-label="Choose RGB LED color" value="${color}" data-light-color ${disabled?'disabled':''}><output data-color-output>${color}</output></label>${button('Apply color','set_color',undefined,'color')}</div><div class="device-brightness-row"><label>Brightness <output data-brightness-output>${brightness}%</output><input class="device-brightness-input" type="range" min="0" max="100" step="1" value="${brightness}" aria-label="RGB LED brightness, 0 to 100 percent" data-light-brightness ${disabled?'disabled':''}></label>${button('Apply brightness','set_brightness',undefined,'brightness')}</div></div><small>${escape(runtime.deviceActionMessage||note)}</small></details>`;
  }
  function sendDeviceAction(button){
    const action=button.dataset.deviceAction;
    let value=button.dataset.deviceValue;
    if(button.dataset.deviceValueSource){
      const controls=button.closest('[data-light-controls]');
      const field=button.dataset.deviceValueSource==='color'?controls?.querySelector('[data-light-color]'):controls?.querySelector('[data-light-brightness]');
      value=field?.value;
      if(button.dataset.deviceValueSource==='color')value=safeRgbColor(value);
      if(button.dataset.deviceValueSource==='brightness'&&value!==undefined)value=String(Number(value));
      if(value===undefined||value===null){runtime.deviceActionMessage='Choose a valid light setting before applying it.';refreshTranscriptProjection();return;}
    }
    if(!runtime.connected||runtime.deviceActionPending!==null||runtime.ws?.readyState!==WebSocket.OPEN)return;
    const commandId=`ui-${crypto.randomUUID().replaceAll('-','')}`;
    const command={type:'device_ui_action',command_id:commandId,action};
    if(value!==undefined)command.value=value;
    try{
      runtime.ws.send(JSON.stringify(command));
      runtime.deviceActionPending=commandId;
      const target=runtime.deviceCapabilities?.target==='mock'?'mock adapter':'Raspberry Pi adapter';
      runtime.deviceActionMessage=`Sending command to ${target}…`;
      refreshTranscriptProjection();
    }catch(_){runtime.deviceActionPending=null;runtime.deviceActionMessage='The control could not reach the voice session. No retry was sent.';refreshTranscriptProjection();}
  }
  const evidence = r=>pill(r.data.agent_mode==='external-services'?'CPU frontend / configured services':r.data.agent_mode==='mock'?'CPU frontend / mock agent':labels[r.evidence]||'Unverified',r.evidence==='HPC-live'?'teal':r.evidence==='mock'?'blue':'subtle');
  const showNotice = (text,tone='warning')=>{$('notice').className=tone==='error'?'notice-error':'';if($('notice').textContent!==text)$('notice').textContent=text;$('notice').hidden=!text;};
  const metric = (label,value,note='')=>`<div><div class="metric-label">${escape(label)}</div><div class="metric-value${value==='Unavailable'?' metric-unavailable':''}">${escape(value)}</div>${note?`<div class="metric-note">${escape(note)}</div>`:''}</div>`;
  function validate(next) {
    if (!next || next.schema_version!==1 || !Array.isArray(next.reports) || next.reports.length>221 || !Array.isArray(next.warnings) || typeof next.generated_utc!=='string') throw new Error('Unsupported report format');
    for(const r of next.reports){
      if(!r || typeof r.source!=='string' || typeof r.id!=='string' || typeof r.evidence!=='string' || !r.data || typeof r.data!=='object')throw new Error('Invalid report');
      if(r.kind==='replay' && (!Array.isArray(r.data.wake_events) || !r.data.wake_events.every(e=>e && typeof e==='object')))throw new Error('Invalid audio report');
      if(r.kind==='scenarios' && (!Array.isArray(r.data.scenarios)||!r.data.scenarios.every(s=>s&&typeof s.name==='string'&&(s.passed===null||typeof s.passed==='boolean'))))throw new Error('Invalid scenario report');
      if(r.kind==='trace' && (!Array.isArray(r.data.turns) || !r.data.turns.every(t=>t && typeof t==='object')))throw new Error('Invalid timing report');
      if(r.kind==='transcript'){const d=r.data;if(!['mock','HPC-live','Pi-verified'].includes(d.mode)||!['idle','listening','thinking','speaking','interrupted','ended'].includes(d.state)||typeof d.updated_utc!=='string'||d.kind!=='conversation-transcript'||d.display_authorized!==true||typeof d.session_id!=='string'||!Number.isInteger(d.revision)||d.revision<0||!Array.isArray(d.messages)||d.messages.length>200)throw new Error('Invalid transcript');const ids=new Set();for(const m of d.messages){if(!m||!['user','assistant'].includes(m.role)||!Number.isInteger(m.turn_id)||m.turn_id<1||m.id!==`${m.turn_id}:${m.role}`||ids.has(m.id)||typeof m.text!=='string'||m.text.length>8000||typeof m.confirmed_text!=='string'||m.confirmed_text.length>8000||!['asr_partial','asr_final','generated','mock_sink_confirmed','sink_confirmed'].includes(m.delivery)||!['partial','final','interrupted','error'].includes(m.status))throw new Error('Invalid transcript message');ids.add(m.id);if(m.role==='assistant'&&(!m.text.startsWith(m.confirmed_text)||(m.status!=='partial'&&m.text!==m.confirmed_text)))throw new Error('Invalid delivery prefix');if(m.role==='assistant'&&(['asr_partial','asr_final'].includes(m.delivery)||(m.status==='final'&&m.delivery==='generated')))throw new Error('Invalid assistant delivery');if(m.role==='user'&&(!['asr_partial','asr_final'].includes(m.delivery)||m.confirmed_text||(m.status==='partial'&&m.delivery!=='asr_partial')||(m.status==='final'&&m.delivery!=='asr_final')))throw new Error('Invalid user delivery');}}
      if(r.kind==='transcript'&&r.data.messages.some(m=>m.role==='assistant'&&'generation_complete' in m&&typeof m.generation_complete!=='boolean'))throw new Error('Invalid text finalization');
      if(r.kind==='transcript'&&'tasks' in r.data){const rows=r.data.tasks;if(!Array.isArray(rows)||rows.length>50)throw new Error('Invalid task activity');const ids=new Set();for(const t of rows){if(!t||!Number.isInteger(t.turn_id)||t.turn_id<1||typeof t.call_id!=='string'||!/^[-_a-zA-Z0-9]{1,80}$/.test(t.call_id)||t.id!==`${t.turn_id}:${t.call_id}`||ids.has(t.id)||!taskKinds.has(t.kind)||!['running','simulated','accepted','awaiting_input','completed','reported','unavailable','replaced','cancelled','failed','unknown'].includes(t.status)||!['mock','reported','Pi-verified'].includes(t.mode)||t.status==='simulated'&&t.mode!=='mock'||t.mode==='Pi-verified'&&r.data.mode!=='Pi-verified'||typeof t.title!=='string'||!t.title||t.title.length>80||typeof t.summary!=='string'||t.summary.length>200||typeof t.display!=='string'||t.display.length>32||'widget' in t&&!validTaskWidget(t.widget))throw new Error('Invalid task activity');const locationKeys=['execution_target','absolute_path','verification_level'];const hasLocation=locationKeys.some(key=>key in t);if(hasLocation&&(!locationKeys.every(key=>typeof t[key]==='string')||!t.execution_target||t.execution_target.length>48||!t.absolute_path.startsWith('/')||t.absolute_path.length>512||/[\u0000-\u001f]/.test(t.absolute_path)||!t.verification_level||t.verification_level.length>120))throw new Error('Invalid task location');if('artifacts' in t){if(t.kind!=='agent'||t.status!=='reported'||!/^([a-f0-9]{24})$/.test(t.artifact_id||'')||!Array.isArray(t.artifacts)||!t.artifacts.length||t.artifacts.length>8||new Set(t.artifacts).size!==t.artifacts.length||!t.artifacts.every(name=>typeof name==='string'&&/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(html|css|js|md|txt|json)$/.test(name)))throw new Error('Invalid task artifacts');}ids.add(t.id);}}
    }
    return next;
  }
  function selected(id,kind){
    const node=$(id), previous=node.value;
    node.innerHTML=reports(kind).map(r=>`<option value="${escape(r.id)}">${escape(filename(r))}</option>`).join('');
    if([...node.options].some(o=>o.value===previous))node.value=previous;
    else if(kind==='trace'){
      const usable=reports(kind).find(r=>r.data.turns?.some(t=>t.turn_id>0));
      if(usable)node.value=usable.id;
    }
    node.disabled=!node.options.length;
  }
  function assistantReport(){
    return runtime.connected?runtime.liveReport:savedSessions.get(selectedSession);
  }
  function history(){
    for(const report of reports('transcript'))savedSessions.set(report.data.session_id,report);
    if(runtime.liveReport)savedSessions.set(runtime.liveReport.data.session_id,runtime.liveReport);
    while(savedSessions.size>50){const key=[...savedSessions.keys()].find(key=>key!==selectedSession);if(!key)break;savedSessions.delete(key);}
    const select=$('history-select');
    const options='<option value="">New session</option>'+[...savedSessions.values()].reverse().map(report=>`<option value="${escape(report.data.session_id)}">${escape((report.data.mode==='mock'?'Mock · ':'Saved · ')+date(report.data.updated_utc))}</option>`).join('');
    if(select.innerHTML!==options)select.innerHTML=options;
    select.value=selectedSession;
    select.disabled=runtime.connected||!savedSessions.size;
  }
  function showMusicPanel(){
    if(!livePage)return false;
    const m=runtime.music,task=assistantReport()?.data?.tasks?.at(-1);
    const media=Boolean(m.state?.track&&['playing','paused'].includes(m.state.status));
    const musicRequest=Boolean(task?.kind?.startsWith('music_'));
    return media||musicRequest;
  }
  const visibleTasks = d => (Array.isArray(d?.tasks)?d.tasks:[]).filter(t=>t.kind!=='agent'||t.answer_only!==true||t.status!=='reported'||t.artifacts?.length||t.widget||t.absolute_path);
  function syncAssistantLayout(){
    const report=assistantReport(), hasMessages=Boolean(report?.data?.messages?.length), hasTasks=Boolean(visibleTasks(report?.data).length);
    const active=runtime.connected;
    const conversation=active||hasMessages||hasTasks;
    $('view-assistant').dataset.conversation=String(conversation);
    $('welcome').hidden=conversation;
    $('transcript-panel').hidden=!conversation;
    const showMusicSetup=showMusicPanel();
    $('conversation-workspace').dataset.hasTasks=String(hasTasks||showMusicSetup);
    $('transcript-empty').hidden=hasMessages||!conversation;
    $('transcript-messages').hidden=!hasMessages;
    const emptyText=runtime.status==='wake_listening'?'Say Hey Swift to begin.':runtime.status==='listening'?"Go ahead, I’m listening.":'Your exchange will appear here.';
    text('transcript-empty',emptyText);
  }
  function transcript(){
    const report=assistantReport(), d=report?.data;
    const visible=Boolean(d?.messages.length);
    syncAssistantLayout();
    voiceStatus();
    const taskRows=visibleTasks(d);
    const activeRows=taskRows.filter(task=>['accepted','running','awaiting_input'].includes(task.status));
    const activeTaskRows=[...new Map(activeRows.map(task=>[task.task_id||task.id,task])).values()];
    const historyRows=taskRows.filter(task=>!['accepted','running','awaiting_input'].includes(task.status)).sort((left,right)=>{
      const stamp=task=>{const value=Date.parse(task.finished_utc||task.updated_utc||task.started_utc||'');return Number.isFinite(value)?value:0;};
      return stamp(right)-stamp(left) || taskRows.indexOf(right)-taskRows.indexOf(left);
    });
    const showMusicSetup=showMusicPanel();
    const liveTranscript=runtime.connected&&report?.id==='runtime-live';
    const taskSnapshotAt=liveTranscript?'':d?.updated_utc||'';
    $('task-activity').hidden=!taskRows.length&&!showMusicSetup;
    $('task-panel-heading').hidden=!activeTaskRows.length;
    const taskHeading=$('task-panel-heading').querySelector('h2');
    if(activeTaskRows.length&&taskHeading)taskHeading.textContent=liveTranscript?'Current task':'Task at snapshot';
    $('active-task-items').hidden=!activeTaskRows.length;
    $('task-history-heading').hidden=!historyRows.length;
    $('task-state-summary').hidden=!taskRows.some(task=>task.kind.startsWith('music_')||['light_on','light_off','set_color','set_brightness'].includes(task.kind));
    if(taskRows.length){
      const nextTaskKey=report.id+':'+d.session_id+':'+JSON.stringify(taskRows);
      if(nextTaskKey!==taskKey){
        taskKey=nextTaskKey;
        const icon=t=>['light_on','light_off','set_color','set_brightness'].includes(t.kind)?'●':['set_timer','set_alarm','cancel_schedule','get_time'].includes(t.kind)?'◷':t.kind==='set_temperature'?'°':['create_reminder','list_reminders'].includes(t.kind)||t.kind==='agent'&&t.artifacts?.length?'▤':t.kind==='get_weather'?'☁':'↗';
        const observedPlayerState=t=>{const match=/(?:Spotify|the browser player) now reports (paused|not playing|playing)\b/i.exec(t.summary||'');return match?match[1].toLowerCase():'';};
        const evidence=t=>t.status==='running'?(liveTranscript?'Working · '+(t.mode==='mock'?'Mock':'Service'):'Recorded · was running'+(t.mode==='mock'?' · Mock':'')):t.status==='simulated'?'Simulated':t.status==='accepted'?(liveTranscript?'Accepted · output not sensor-verified':'At snapshot · accepted'):t.status==='awaiting_input'?(liveTranscript?'Waiting for your input':'At snapshot · input needed'):t.status==='completed'?(t.mode==='mock'?'Completed · mock':'Completed · service-reported'):t.status==='unavailable'?'Unavailable':t.status==='replaced'?'Replaced by a newer schedule':t.status==='cancelled'?'Cancelled':t.status==='failed'?'Failed · '+(t.mode==='mock'?'Mock':'Service'):t.status==='unknown'&&t.kind.startsWith('music_')?(observedPlayerState(t)?`Request uncertain · player ${observedPlayerState(t)}`:'Request unconfirmed'):t.status==='unknown'?'Outcome unknown':t.status==='reported'?(t.mode==='mock'?'Mock · unverified':'Reported · unverified'):'Unknown';
        const taskUrl=(id,name)=>`/api/voice/tasks/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(name)}`;
        const newest=taskRows.at(-1)?.id, newestUpdate=historyRows[0]?.id;
        const recency=t=>{
          if(t.id===newestUpdate&&t.status==='completed')return `Completed most recently · Turn ${t.turn_id}`;
          if(t.id===newestUpdate&&t.status==='failed')return `Failed most recently · Turn ${t.turn_id}`;
          if(t.id===newestUpdate&&t.status==='cancelled')return `Cancelled most recently · Turn ${t.turn_id}`;
          return t.id!==newest?`Earlier result · Turn ${t.turn_id}`:t.status==='running'?'Current request':t.status==='failed'?'Latest request · Failed':t.status==='unknown'?'Latest request · Outcome unknown':'Latest result';
        };
        const formatTimer=seconds=>`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
        const elapsedText=seconds=>seconds<60?`${seconds}s`:`${Math.floor(seconds/60)}m ${String(seconds%60).padStart(2,'0')}s`;
        const ageSeconds=(stamp,reference=Date.now())=>Number.isFinite(Date.parse(stamp))?Math.max(0,Math.floor((reference-Date.parse(stamp))/1000)):null;
        const updateTimerClocks=()=>{
          const nodes=[...$('task-items').querySelectorAll('[data-timer-deadline]')];
          for(const node of nodes){const due=Date.parse(node.dataset.timerDeadline);if(!Number.isFinite(due))continue;const seconds=Math.max(0,Math.ceil((due-Date.now())/1000));node.textContent=seconds?formatTimer(seconds):node.dataset.timerKind==='alarm'?'Time reached':'Elapsed';}
          const taskNodes=[...$('active-task-items').querySelectorAll('[data-task-started]')];
          for(const node of taskNodes){
            const started=Date.parse(node.dataset.taskStarted),finished=Date.parse(node.dataset.taskFinished||''),snapshotMs=Date.parse(node.dataset.taskSnapshotAt||''),recorded=Number.isFinite(snapshotMs),now=recorded?snapshotMs:Date.now();
            if(Number.isFinite(started))textNode(node.querySelector('[data-task-elapsed]'),`Elapsed ${elapsedText(Math.max(0,Math.floor(((Number.isFinite(finished)?finished:now)-started)/1000)))}${recorded?' at snapshot':''}`);
            const progressAge=ageSeconds(node.dataset.taskProgressAt||'',now);
            textNode(node.querySelector('[data-task-progress-age]'),progressAge===null?'Progress not reported':recorded?`Progress updated ${elapsedText(progressAge)} before snapshot`:`Progress updated ${elapsedText(progressAge)} ago`);
            const heartbeatAge=ageSeconds(node.dataset.taskHeartbeat||'',now);
            const live=heartbeatAge!==null&&heartbeatAge<=35;
            textNode(node.querySelector('[data-task-heartbeat]'),recorded?live?'Worker active at snapshot':heartbeatAge===null?'Worker heartbeat unavailable at snapshot':'No recent worker signal at snapshot':live?'Worker active · no new phase':heartbeatAge===null?'Worker heartbeat unavailable':'No recent worker signal');
          }
        };
        const textNode=(node,value)=>{if(node&&node.textContent!==value)node.textContent=value;};
        const widget=t=>{
          const simulation=t.mode==='mock'&&t.status==='simulated';
          if(t.kind.startsWith('music_')){
            const observation=observedPlayerState(t);
            const state=t.status==='running'?'Requesting':t.status==='failed'?'Playback unavailable':t.status==='unknown'?observation?`Player reports ${observation}`:'Request outcome uncertain':simulation?(t.kind==='music_volume_set'||t.kind==='music_volume_up'||t.kind==='music_volume_down'?`Simulated volume · ${t.display}`:t.display==='pause'||t.display==='stop'?'Simulated pause':'Simulated playback'):t.kind==='music_volume_set'||t.kind==='music_volume_up'||t.kind==='music_volume_down'?`Volume · ${t.display}`:t.kind==='music_next'?'Next track requested':t.kind==='music_previous'?'Previous track requested':t.display==='paused'?'Paused · device reported':t.display==='pause'||t.display==='stop'?'Pause requested':t.display==='playing'?'Playing · device reported':'Playback requested';
            const summary=t.status==='unknown'&&observation?`Audible sound is not verified. The request may or may not have caused this player state.`:(t.summary||'Waiting for the music service.').replace(/ (?:Audible playback is unverified|Playback is unverified)\.$/,'');
            const kicker=simulation?'SIMULATED MUSIC · NO AUDIO':t.status==='reported'?'SPOTIFY · SERVICE REPORTED':t.status==='unknown'?'SPOTIFY · REQUEST UNCERTAIN':'MUSIC REQUEST';
            return `<div class="task-widget task-music" role="group" aria-label="Music result"><span class="music-note" aria-hidden="true">♫</span><span class="task-widget-copy"><span class="task-widget-kicker">${kicker}</span><strong>${escape(state)}</strong><span>${escape(summary)}</span></span></div>`;
          }
          const structured=t.widget;
          if(structured?.type==='lamp'){
            const state=structured.on?'on':'off', color=structured.color, brightness=structured.brightness;
            const stateText=`${state.toUpperCase()} · ${color} · ${brightness}%`;
            const note=simulation?'Simulation only · no physical lamp connected':'Pi command accepted · no sensor confirms the LED state';
            const opacity=structured.on?brightness/100:0.08;
            return `<div class="task-widget task-light" data-light-state="${state}" data-light-color="${escape(color)}" data-brightness="${brightness}" role="group" aria-label="${escape((simulation?'Simulated RGB LED ':'Commanded RGB LED ')+stateText)}">${rgbModuleIllustration(color,opacity)}<span class="task-widget-copy"><span class="task-widget-kicker">${simulation?'SIMULATED RGB LED':'RGB LED · GPIO COMMAND'}</span><strong>${escape(stateText)}</strong><span>${escape(note)}</span></span>${deviceControlsMarkup(structured)}</div>`;
          }
          const lampKinds=['light_on','light_off','set_color','set_brightness'];
          if(lampKinds.includes(t.kind)){
            const command=t.kind==='light_on'?'on':t.kind==='light_off'?'off':null;
            const state=command?(simulation?command:'unknown'):'setting';
            const color=rgbColorLabel(t.display);
            const brightness=/^(?:100|[1-9]?\d)%$/.test(t.display)?t.display.slice(0,-1):'unknown';
            const value=command?(simulation?command==='on'?'ON':'OFF':`Command · ${command}`):t.kind==='set_color'?`Color · ${color==='unknown'?'unavailable':color}`:`Brightness · ${brightness==='unknown'?'unavailable':brightness+'%'}`;
            const note=simulation?'Simulation only · no physical lamp connected':'Command reported · physical state unverified';
            const label=command&&simulation?`Simulated lamp ${command}`:`Light setting ${value}`;
            const opacity=state==='on'&&brightness!=='unknown'?Number(brightness)/100:0.08;
            return `<div class="task-widget task-light" data-light-state="${state}" data-light-color="${escape(color)}" data-brightness="${brightness}" role="group" aria-label="${escape(label)}">${rgbModuleIllustration(color,opacity)}<span class="task-widget-copy"><span class="task-widget-kicker">${simulation?'SIMULATED RGB LED':t.status==='unavailable'?'PI DEVICE UNAVAILABLE':'RGB LED · GPIO COMMAND'}</span><strong>${escape(value)}</strong><span>${escape(t.status==='unavailable'&&t.summary? t.summary:note)}</span></span>${deviceControlsMarkup()}</div>`;
          }
          if(t.kind==='set_timer'||t.kind==='set_alarm'){
            const details=structured?.type==='timer'?structured:null;
            const active=Boolean(liveTranscript&&details?.active&&t.status!=='replaced'&&t.status!=='cancelled');
            const kind=details?.schedule_kind||(t.kind==='set_alarm'?'alarm':'timer');
            const deadline=active&&details?.deadline_utc?` data-timer-deadline="${escape(details.deadline_utc)}" data-timer-kind="${kind}"`:'';
            const fallback=details?.display_value||t.display||'Value unavailable';
            const initial=active&&details.deadline_utc?formatTimer(Math.max(0,Math.ceil((Date.parse(details.deadline_utc)-Date.now())/1000))):fallback;
            const label=kind==='alarm'?'ALARM':'TIMER';
            const kicker=simulation?`SIMULATED ${label} · NO ALERT`:t.mode==='reported'?`TM1637 ${label} · ACCEPTED`:`RECORDED ${label}`;
            const sourceNote=simulation?'Countdown is software simulated; no sound will alert you.':t.mode==='reported'?'The display command was accepted; no buzzer is configured.':'Saved result; current schedule state is unavailable.';
            return `<div class="task-widget task-timer" role="group" aria-label="${escape((simulation?'Simulated ':t.mode==='reported'?'Pi display ':'Recorded ')+label.toLowerCase()+': '+fallback)}"><span class="timer-face" aria-hidden="true"><span>◷</span></span><span class="task-widget-copy"><span class="task-widget-kicker">${kicker}</span><strong class="timer-value"${deadline}>${escape(initial)}</strong><span>${escape(sourceNote)}</span></span></div>`;
          }
          if(t.kind==='set_temperature'){
            const degrees=structured?.type==='temperature'?structured.setpoint_c:t.display||'Unavailable';
            return `<div class="task-widget task-timer task-temperature" role="group" aria-label="${escape((simulation?'Simulated ':'Pi display ')+`demo setpoint ${degrees} degrees Celsius`)}"><span class="timer-face" aria-hidden="true"><span>°</span></span><span class="task-widget-copy"><span class="task-widget-kicker">${simulation?'SIMULATED SETPOINT':'TM1637 DEMO SETPOINT'}</span><strong>${escape(String(degrees))}°C</strong><span>Display only · not measured temperature or HVAC control.</span></span></div>`;
          }
          if(structured?.type==='reminders'){
            const items=structured.items;
            const textValue=items.length?items.join(' · '):'No reminders saved';
            return `<div class="task-widget task-timer task-reminders" role="group" aria-label="${escape('Reminder list: '+textValue)}"><span class="timer-face" aria-hidden="true"><span>▤</span></span><span class="task-widget-copy"><span class="task-widget-kicker">${structured.persistent?'SAVED REMINDERS':'SESSION REMINDERS · SIMULATED'}</span><strong>${items.length} reminder${items.length===1?'':'s'}</strong><span>${escape(textValue)}</span></span></div>`;
          }
          if(structured?.type==='time'){
            return `<div class="task-widget task-timer" role="group" aria-label="${escape(`Reported local time ${structured.local_time} ${structured.timezone}`)}"><span class="timer-face" aria-hidden="true"><span>◷</span></span><span class="task-widget-copy"><span class="task-widget-kicker">SYSTEM CLOCK</span><strong>${escape(structured.local_time)}</strong><span>${escape(structured.timezone)} · verify Pi clock on device.</span></span></div>`;
          }
          if(structured?.type==='weather'){
            return `<div class="task-widget task-timer task-weather" role="group" aria-label="${escape(`Current weather in ${structured.location}: ${structured.condition}, ${structured.temperature_c} degrees Celsius`)}"><span class="timer-face" aria-hidden="true"><span>☁</span></span><span class="task-widget-copy"><span class="task-widget-kicker">CURRENT CONDITIONS · ${escape(structured.provider)}</span><strong>${escape(structured.condition)} · ${escape(String(structured.temperature_c))}°C</strong><span>${escape(structured.location)} · reported ${escape(date(structured.observed_utc))} · not a local sensor.</span></span></div>`;
          }
          if(t.kind==='cancel_schedule')return `<div class="task-widget task-timer" role="group" aria-label="Timer or alarm cancellation"><span class="timer-face" aria-hidden="true"><span>×</span></span><span class="task-widget-copy"><span class="task-widget-kicker">${simulation?'SIMULATED SCHEDULE':'SCHEDULE'}</span><strong>${escape(t.status==='cancelled'||t.status==='simulated'?t.display||'Cancelled':'No active schedule')}</strong><span>${escape(t.summary||'No active timer or alarm.')}</span></span></div>`;
          return '';
        };
        const card=(t,isActive=false)=>{
          const files=Array.isArray(t.artifacts)?t.artifacts:[];
          const fileMarkup=files.length?`<div class="task-files"><div class="task-files-heading"><strong>Task output</strong><span>${files.length} file${files.length===1?'':'s'}</span></div><ul>${files.map(name=>{const ext=name.split('.').at(-1).toUpperCase();return `<li><span class="file-kind" aria-hidden="true">${escape(ext)}</span>${runtime.artifactsAvailable?`<button class="task-file-preview" type="button" data-preview-artifact="${escape(t.artifact_id)}" data-preview-name="${escape(name)}" aria-label="Preview ${escape(name)}">${escape(name)}</button><a href="${taskUrl(t.artifact_id,name)}" download="${escape(name)}">Download</a>`:`<span class="task-file-name">${escape(name)}</span>`}</li>`;}).join('')}</ul>${runtime.artifactsAvailable?'':`<p>File access is unavailable in this saved view.</p>`}</div>`:'';
          const location=t.absolute_path?`<dl class="task-location"><div><dt>Execution target</dt><dd>${escape(t.execution_target)}</dd></div><div><dt>Absolute path</dt><dd><code>${escape(t.absolute_path)}</code></dd></div><div><dt>Verification status</dt><dd>${escape(t.verification_level)}</dd></div></dl>`:'';
          const visual=widget(t);
          const musicUnknown=t.kind.startsWith('music_')&&t.status==='unknown';
          const recovery=t.id===newestUpdate&&(t.status==='failed'||t.status==='unknown'&&!musicUnknown)?`<div class="task-recovery"><strong>${t.status==='failed'?'Needs attention':'Outcome unavailable'}</strong><p>${escape(t.kind.startsWith('music_')?(t.status==='unknown'?'Check current Spotify playback before retrying; the command may already have taken effect.':t.summary?.includes('active playback device')?'Start the configured Spotify receiver, then run the secured terminal device check. Manual song selection is not required.':t.summary?.includes('token expired')?'Renew Spotify access in the secured terminal setup before trying again.':/receiver|Spotify player/.test(t.summary||'')?'Start the configured Spotify receiver, then use the secured terminal device check before trying again.':'Check the music service in Diagnostics before making another request.'):(t.status==='unknown'?'Outcome unknown. Check before retrying.':'Action failed. See details for this request.'))}</p></div>`:'';
          const taskId=t.task_id||t.id;
          const progress=isActive?`<div class="background-task-progress" data-task-started="${escape(t.started_utc||'')}" data-task-finished="${escape(t.finished_utc||'')}" data-task-progress-at="${escape(t.last_progress_utc||'')}" data-task-heartbeat="${escape(t.heartbeat_utc||'')}" data-task-snapshot-at="${escape(taskSnapshotAt)}"><div class="background-task-phase"><strong>${escape(t.phase|| (t.status==='awaiting_input'?'Waiting for your input':t.status==='accepted'?'Accepted · worker starting':'Waiting for worker progress'))}</strong></div>${Array.isArray(t.completed_steps)&&t.completed_steps.length?`<ul class="task-steps">${t.completed_steps.map(step=>`<li>${escape(step)}</li>`).join('')}</ul>`:''}<div class="background-task-meta"><span data-task-elapsed>Elapsed —</span><span data-task-progress-age>Progress not reported</span><span data-task-heartbeat>Worker heartbeat unavailable</span></div></div>`:'';
          return `<article class="task-card${isActive?' task-card-active':''}" role="listitem" data-task-id="${escape(taskId)}" data-kind="${escape(t.kind)}" data-status="${escape(t.status)}" data-latest="${String(t.id===newest)}"><span class="task-art" aria-hidden="true" data-color="${escape(['red','blue','green'].includes(t.display)?t.display:'none')}">${icon(t)}</span><div class="task-copy"><span class="task-recency">${escape(isActive?'Task '+taskId.slice(0,8):recency(t))}</span><strong>${escape(t.title)}</strong>${visual||t.kind.startsWith('music_')||t.kind==='set_temperature'?'':`<span>${escape(t.summary||'In progress')}</span>`}</div><span class="task-evidence">${escape(evidence(t))}</span>${progress}${visual}${recovery}${fileMarkup}${location}</article>`;
        };
        const openDeviceControlTasks=new Set([...document.querySelectorAll('.device-controls[open]')].map(node=>node.closest('[data-task-id]')?.dataset.taskId).filter(Boolean));
        const previousScroll=$('task-items').scrollTop;
        const previousIds=[...$('task-items').querySelectorAll('.task-card')].map(node=>node.dataset.taskId);
        $('active-task-items').innerHTML=activeTaskRows.map(task=>card(task,true)).join('');
        $('task-items').innerHTML=historyRows.map(task=>card(task,false)).join('');
        for(const cardNode of document.querySelectorAll('.task-card[data-task-id]'))if(openDeviceControlTasks.has(cardNode.dataset.taskId)){const details=cardNode.querySelector('.device-controls');if(details)details.open=true;}
        const nextIds=[...$('task-items').querySelectorAll('.task-card')].map(node=>node.dataset.taskId);
        $('task-items').scrollTop=JSON.stringify(previousIds)===JSON.stringify(nextIds)?previousScroll:0;
        const latestMusic=taskRows.filter(task=>task.kind.startsWith('music_')).at(-1);
        const latestLight=taskRows.filter(task=>['light_on','light_off','set_color','set_brightness'].includes(task.kind)).at(-1);
        const compact=[];
        if(latestMusic){
          const simulated=latestMusic.mode==='mock'&&['simulated','completed'].includes(latestMusic.status);
          const action=latestMusic.display||latestMusic.kind.replace('music_','');
          const state=simulated?`Simulated · ${action}`:latestMusic.status==='reported'?`Spotify · ${action} reported`:latestMusic.status==='failed'?'Playback unavailable':latestMusic.status==='unknown'?'Playback state unknown':'Music request in progress';
          compact.push(`<div class="resource-state-item"><span class="resource-state-icon" aria-hidden="true">♫</span><span><small>${simulated?'MUSIC · SIMULATION':'MUSIC · DEVICE STATUS'}</small><strong>${escape(state)}</strong></span></div>`);
        }
        if(latestLight){
          const lamp=latestLight.widget?.type==='lamp'?latestLight.widget:null;
          const simulated=latestLight.mode==='mock'&&latestLight.status==='simulated';
          const state=lamp?`${lamp.on?'On':'Off'} · ${lamp.color} · ${lamp.brightness}%`:latestLight.status==='failed'?'Light command failed':latestLight.status==='unknown'?'Light state unknown':latestLight.display||latestLight.title;
          compact.push(`<div class="resource-state-item"><span class="resource-state-icon" aria-hidden="true">●</span><span><small>${simulated?'LIGHT · SIMULATION':'LIGHT · COMMAND STATUS'}</small><strong>${escape(state)}</strong></span></div>`);
        }
        $('task-state-summary').innerHTML=compact.join('');
        updateTimerClocks();
        const needsTaskClock=Boolean($('active-task-items').querySelector('[data-task-started]'));
        const needsTimerClock=Boolean($('task-items').querySelector('[data-timer-deadline]'));
        if((needsTaskClock||needsTimerClock)&&taskTick===null)taskTick=setInterval(updateTimerClocks,1000);
        if(!needsTaskClock&&!needsTimerClock&&taskTick!==null){clearInterval(taskTick);taskTick=null;}
      }
    }else{taskKey=null;$('active-task-items').textContent='';$('task-items').textContent='';$('task-state-summary').textContent='';$('task-panel-heading').hidden=true;$('task-history-heading').hidden=true;if(taskTick!==null){clearInterval(taskTick);taskTick=null;}}
    $('transcript-mode').textContent=runtime.connected?(runtime.mode==='mock'?'Mock voice runtime':'Live browser session'):d?.mode==='mock'?'Mock recording':'Recorded session';
    if(!visible){$('transcript-messages').textContent='';transcriptKey=null;transcriptSession=null;followLatest=true;$('jump-latest').hidden=true;return;}
    const backends=d.backend_modes&&typeof d.backend_modes==='object'?['asr','llm','tts'].filter(role=>['mock','external'].includes(d.backend_modes[role])).map(role=>role.toUpperCase()+' '+d.backend_modes[role]):[];
    $('transcript-mode').textContent=runtime.connected&&report?.id==='runtime-live'?(d.mode==='mock'?'Mock voice runtime':'Live browser session'):d.mode==='mock'?'Mock recording':'Recorded session';
    $('transcript-mode').title=backends.join(' · ');

    const key=report.id+':'+report.sha256+':'+d.session_id+':'+d.revision;
    if(key===transcriptKey)return;
    const log=$('transcript-messages');
    const previousScroll=log.scrollTop;
    const follow=transcriptSession!==d.session_id || (log.getClientRects().length?log.scrollHeight-log.scrollTop-log.clientHeight<70:followLatest);
    followLatest=follow;
    transcriptSession=d.session_id;
    transcriptKey=key;
    const markup=d.messages.map(m=>{
      const user=m.role==='user';
      const draft=m.status==='partial';
      const label=m.status==='interrupted'?'Interrupted':m.status==='error'?'Stopped':draft?(user?'Listening…':m.generation_complete?'':'Draft'):'';
      let content=escape(m.text);
      if(!user&&draft&&!m.generation_complete){content=escape(m.confirmed_text)+'<span class="message-draft">'+escape(m.text.slice(m.confirmed_text.length))+'</span>';}
      if(!m.text&&!user)content='<span class="message-empty">Response stopped before playback.</span>';
      return `<article data-status="${escape(m.status)}" class="message ${user?'message-user':'message-assistant'}" data-message-id="${escape(m.id)}"><div class="message-header"><strong>${user?'You':'Swift'}</strong>${label?pill(label,m.status==='error'?'red':m.status==='interrupted'?'amber':'subtle'):''}</div><div class="message-text">${content}</div></article>`;
    }).join('');
    const template=document.createElement('template');
    template.innerHTML=markup;
    const existing=new Map([...log.children].map(node=>[node.dataset.messageId,node]));
    const wanted=new Set(d.messages.map(m=>m.id));
    for(const node of [...log.children])if(!wanted.has(node.dataset.messageId))node.remove();
    for(const [index,next] of [...template.content.children].entries()){
      const previous=existing.get(next.dataset.messageId);
      const node=previous?.outerHTML===next.outerHTML?previous:next;
      if(previous&&node!==previous)previous.replaceWith(node);
      if(log.children[index]!==node)log.insertBefore(node,log.children[index]||null);
    }
    log.scrollTop=follow?log.scrollHeight:previousScroll;
    requestAnimationFrame(()=>{
      if(transcriptKey!==key)return;
      if(follow)log.scrollTop=log.scrollHeight;
      updateScroll();
    });
  }
  function refreshTranscriptProjection(){transcriptKey=null;taskKey=null;transcript();}
  function voiceStatus(stale=false){
    const report=assistantReport(), d=report?.data;
    const state=runtime.connected?(runtime.micMuted&&runtime.status==='wake_listening'?'muted':runtime.status==='wake_listening'?'ready':runtime.status):runtime.ended?'ended':'disconnected';
    $('conversation-mode').disabled=runtime.connected||runtime.connecting;
    const node=$('voice-status');node.style.setProperty('--followup-left',runtime.connected&&state==='listening'&&runtime.inputTurnOpen===false&&numeric(runtime.engagement?.remaining_seconds)?Math.max(0,Math.min(1,runtime.engagement.remaining_seconds/runtime.engagement.window_seconds)):0);node.dataset.waiting=String(runtime.connected&&state==='listening'&&runtime.inputTurnOpen===false&&numeric(runtime.engagement?.remaining_seconds));node.dataset.state=state;node.dataset.live=String(runtime.connected);node.dataset.mode=runtime.connected?runtime.mode||'unknown':d?.mode==='mock'?'mock':d?'recorded':'none';
    const label=runtime.connected?(runtime.interrupting?'Stopping speech…':state==='listening'?runtime.inputTurnOpen===false?(runtime.engagement?.awaiting_answer?'Waiting for your answer':runtime.engagement?.mode?'Listening for a follow-up':'Ready for your next request'):'Listening…':{ready:'Waiting for ‘Hey Swift’',activating:'Starting conversation',thinking:runtime.toolActive?'Working on your request…':'Thinking…',speaking:'Speaking…',interrupted:'Interrupted',muted:'Microphone muted'}[state]||'Voice connected'):runtime.ended?'Conversation ended':d?'Recorded session':runtime.available?'Voice disconnected':'Voice not connected';
    text('session-summary',label);
    const replyComplete=state==='listening'&&runtime.completedTurnId===runtime.turnId;
    const activeContext=replyComplete?'Reply complete · ask a follow-up':'';
    text('voice-status-context',runtime.connected?activeContext:stale?'Saved reports are stale':d?date(d.updated_utc):'');
    document.querySelector('#welcome p').textContent=runtime.available?'Start a conversation to enable voice.':'Connect the voice runtime to begin.';
    text('mic-indicator',runtime.connected?(runtime.micMuted?'Microphone muted':runtime.media?.getAudioTracks()?.[0]?.readyState==='live'?'Microphone on':'Microphone unavailable'):'Microphone not connected');
    node.dataset.muted=String(runtime.micMuted);node.dataset.input=String(runtime.connected&&!runtime.micMuted&&runtime.media?.getAudioTracks()?.[0]?.readyState==='live');
    $('voice-start').classList.toggle('primary',!runtime.connected);
    $('voice-start').classList.toggle('end-session',runtime.connected);
    $('voice-interrupt').disabled=!runtime.connected||runtime.status!=='speaking'||runtime.interrupting;
    $('voice-interrupt').hidden=!runtime.connected||runtime.ended||runtime.status!=='speaking'||runtime.interrupting;
    $('voice-mute').hidden=!runtime.connected||runtime.ended;
    syncAssistantLayout();
    text('voice-action-hint','Open session details');
    text('footer-note',runtime.connected?(runtime.mode==='mock'?'Mock voice session':'Voice session connected'):'Reports only · voice not connected');
    if($('session-dialog').open)sessionDetails();
    updateWave();
  }
  const themePreference=matchMedia('(prefers-color-scheme: dark)');
  let theme='system';
  try{const saved=localStorage.getItem('swift-theme');if(['system','light','dark'].includes(saved))theme=saved;}catch(_){}
  function applyTheme(){
    document.documentElement.dataset.theme=theme==='system'?(themePreference.matches?'dark':'light'):theme;
    $('theme-select').value=theme;
    $('theme-select').querySelector('[value=system]').textContent='System ('+(themePreference.matches?'Dark':'Light')+')';
    $('theme-select').title='System follows your device. It matches Light when your device uses a light theme.';
  }
  $('theme-select').addEventListener('change',()=>{theme=$('theme-select').value;try{localStorage.setItem('swift-theme',theme);}catch(_){}applyTheme();});
  themePreference.addEventListener('change',applyTheme);
  applyTheme();
  function updateScroll(){
    const log=$('transcript-messages');
    if(!log.getClientRects().length)return;
    followLatest=log.scrollHeight-log.scrollTop-log.clientHeight<70;
    $('jump-latest').hidden=followLatest || !log.children.length;
  }
  function sourceCaption(r){
    const runValue=r.data.run_id||r.data.turns?.[0]?.run_id;
    const run=typeof runValue==='string'?runValue:'';
    const sessionValue=r.data.session_id||(Array.isArray(r.data.session_ids)?r.data.session_ids.at(-1):'')||r.data.turns?.[0]?.session_id;
    const session=typeof sessionValue==='string'?sessionValue:'';
    return filename(r)+(run?' · run '+run.slice(0,8):'')+(session?' · session '+session.slice(0,8):'');
  }
  function activity(){
    const replay=latest('replay'), trace=latest('trace');
    const items=[];
    if(replay){
      const d=replay.data;
      const wake=d.wake_events[0];
      items.push(['≋',sourceLabel(d),`${wake?friendlyPhrase(wake.phrase)+' detected · ':''}${format(d.voiced_frames)} / ${format(d.vad_frames)} frames voiced · ${d.audio_origin||'origin unknown'}`,replay]);
    }
    if(trace && trace.id!==replay?.id){
      const completed=trace.data.turns.filter(t=>t.turn_id>0&&t.outcome==='completed').length;
      items.push(['↗','Session software trace',`${format(completed)} completed turn(s)`,trace]);
    }
    const suite=latest('scenarios');
    if(suite)items.push(['✓','Mock integration scenarios',`${suite.data.scenarios.filter(s=>s.passed===true).length} / ${suite.data.scenarios.length} lifecycle checks passed`,suite]);
    text('activity-count',data.reports.length?data.reports.length+' reports':'No reports');
    $('activity-list').innerHTML=items.length?items.map(([icon,title,description,r])=>`<div class="activity-item"><span class="activity-icon" aria-hidden="true">${icon}</span><div><strong>${escape(title)}</strong><p>${escape(description)}</p>${evidence(r)}<p class="source-caption">${escape(sourceCaption(r))}</p></div></div>`).join(''):'<p class="activity-empty">No report activity yet.</p>';
  }
  function audio(){
    const r=data.reports.find(r=>r.id===$('replay-select').value);
    if(!r){$('audio-content').innerHTML='<div class="empty">No audio report yet.</div>';return;}
    const d=r.data, trial=latest('wake_trial')?.data;
    const score=typeof r.sha256==='string' && /^[a-f0-9]{64}$/.test(r.sha256)?reports('frontend_score').find(s=>s.data.replay_sha256===r.sha256)?.data:null;
    const fraction=numeric(d.vad_frames)&&d.vad_frames>0&&numeric(d.voiced_frames)&&d.voiced_frames>=0?Math.min(100,Math.max(0,100*d.voiced_frames/d.vad_frames)):null;
    const activationCount=numeric(d.activation_count)?d.activation_count:d.wake_events_available===true||(d.wake_events_available!==false&&d.wake_frames>0)?d.wake_events.length:null;
    const phrase=(d.wake_events.find(e=>e.phrase)?.phrase?friendlyPhrase(d.wake_events.find(e=>e.phrase).phrase):'') || (d.wake_backend==='not_configured'?'Wake not configured':'No phrase reported');
    const origin=typeof d.audio_origin==='string'&&d.audio_origin?d.audio_origin:'origin unknown';
    const vadBar=numeric(fraction)?`<div class="voice-bar" role="img" aria-label="${format(fraction,1)} percent voiced"><span id="vad-fill"></span></div><p>${format(fraction,1)}% of scored frames</p>`:'<p>Voiced fraction unavailable</p>';
    $('audio-content').innerHTML=`<div class="metrics">${metric('Input',quantity(numeric(d.input_rate)?d.input_rate/1000:null,' kHz',1),d.source_kind==='pcm-stdin'?'Stream sample rate':'Reported sample rate')}${metric('Model input',quantity(numeric(d.model_rate)?d.model_rate/1000:null,' kHz',1))}${metric('Activations',format(activationCount),phrase)}${metric('Clipping',quantity(numeric(d.clipped_fraction)?d.clipped_fraction*100:null,'%',2))}</div>
    <div class="audio-details"><div><h3>Wake word ${evidence(r)}</h3><p>${escape(d.wake_backend||'Unknown')}<br>Input offset <strong>${quantity(d.wake_events[0]?.audio_seconds,' s',2)}</strong><br>Model score <strong>${quantity(d.wake_peak_score,'',4)}</strong></p><p class="compact-note">A score is not real-world accuracy.</p>${trial&&d.wake_backend==='sherpa-onnx-kws'&&origin==='synthetic'?`<details class="synthetic-trials"><summary>Synthetic development trials</summary><div class="trial-line"><strong>${format(trial.positive_trials_detected)}/${format(trial.positive_trials)}</strong> positives detected<br><strong>${format(trial.negative_trials_triggered)}/${format(trial.negative_trials)}</strong> negatives triggered</div><p class="compact-note">Earlier generated-audio cases, not this clip or real-speaker accuracy.</p></details>`:''}</div><div><h3>Voice activity</h3><p><strong>${format(d.voiced_frames)} / ${format(d.vad_frames)}</strong> frames voiced</p>${vadBar}</div><div><h3>Echo control</h3><p>${d.aec==='not_run'?'Not run in this report':'Unknown'}<br>${latest('configuration')?'Configuration review only':'No configuration report'}</p></div></div>
    ${score?`<p class="compact-note">Annotated clip · ${escape(score.audio_origin)} · ${escape(score.split)} · not population accuracy</p><div class="metrics">${metric('Wake windows',format(score.wake_detected)+' / '+format(score.wake_expected))}${metric('False activations',format(score.false_activations))}${metric('VAD precision',quantity(numeric(score.vad_precision)?score.vad_precision*100:null,'%',1))}${metric('VAD recall',quantity(numeric(score.vad_recall)?score.vad_recall*100:null,'%',1))}</div>`:''}
    ${typeof d.state==='string'&&d.state?`<div class="audio-details"><div><h3>Diagnostic session</h3><p>${escape(d.state==='external_session'?'Session-controlled frontend':d.state.replaceAll('_',' '))} · ${escape(d.run_status||'Unknown')}<br>${d.incomplete_utterance?'Speech unfinished at input end':d.finished?'No unfinished utterance at input end':'Input not finalized'}</p></div><div><h3>CPU inference · p95</h3><p>Wake ${quantity(d.compute_ms?.wake?.p95,' ms / call',2)}<br>VAD ${quantity(d.compute_ms?.vad?.p95,' ms / call',2)}</p><p class="compact-note">Software compute only; excludes capture, model startup and acoustic latency.</p></div><div><h3>Input</h3><p>${escape(sourceLabel(d))} · ${escape(origin)}<br>Report updated ${escape(date(r.modified_utc))}</p></div></div><details><summary>Speech boundaries & events</summary><div class="table-wrap"><table><thead><tr><th scope="col">Input offset · s</th><th scope="col">Event</th></tr></thead><tbody>${(Array.isArray(d.frontend_events)?d.frontend_events:[]).filter(e=>e&&typeof e==='object').slice(-30).map(e=>`<tr><td>${format(e.audio_seconds,3)}</td><td>${escape(e.event)} ${escape(e.reason)}</td></tr>`).join('')}</tbody></table></div><p class="compact-note">${Array.isArray(d.vad_segments)?format(d.vad_segments.length):'Unavailable'} voiced ranges retained${d.events_truncated?' · older events omitted':''}. Raw VAD decisions, not annotated accuracy.</p></details>`:''}
    <p class="compact-note">${escape(sourceLabel(d))} · ${escape(origin)} · acoustic latency unavailable · ${escape(filename(r))}</p>`;
    if(numeric(fraction))$('vad-fill').style.width=fraction+'%';
  }
  function conversation(){
    const r=data.reports.find(r=>r.id===$('trace-select').value);
    if(!r){$('conversation-content').innerHTML='<div class="empty">No timing report yet.</div>';return;}
    const turns=r.data.turns.filter(t=>t.turn_id>0);
    if(!turns.length){$('conversation-content').innerHTML='<div class="empty">No conversation turns in this report.</div>';return;}
    const timing=(t,k)=>Array.isArray(t.durations_ms?.[k])&&t.durations_ms[k].some(numeric)?t.durations_ms[k].filter(numeric).map(n=>format(n,2)).join(' / '):'Unavailable';
    const timingDetail=(t,k)=>{const value=timing(t,k);return value==='Unavailable'?value:value+' ms';};
    $('conversation-content').innerHTML=`<div class="table-wrap" tabindex="0" role="region" aria-label="Session software timings"><table><thead><tr><th>Session</th><th>Turn</th><th>Type</th><th>Outcome</th><th>ASR span · ms</th><th>First text segment · ms</th><th>First TTS PCM · ms</th></tr></thead><tbody>${turns.map(t=>`<tr><td>${escape(typeof t.session_id==='string'&&t.session_id?t.session_id.slice(0,8):'Unknown')}</td><td>${escape(format(t.turn_id))}</td><td>${escape(t.mode)}</td><td class="${t.outcome==='error'?'scenario-fail':t.outcome==='completed'?'scenario-pass':''}">${escape(t.outcome||'Unavailable')}</td><td>${timing(t,'asr_request_ms')}</td><td>${timing(t,'llm_first_segment_ms')}</td><td>${timing(t,'tts_first_chunk_ms')}</td></tr>`).join('')}</tbody></table></div>
    <details><summary>Stage details</summary>${turns.map(t=>`<div class="audio-details stage-details"><div><h3>Turn ${escape(format(t.turn_id))} · ${escape(t.mode)} · ASR</h3><p>Submitted audio ${numeric(t.asr_input_duration_ms)?quantity(t.asr_input_duration_ms/1000,' s',2):'Unavailable'}<br>MCP input chunks ${t.asr_input_incremental?'Incremental':'Unavailable or buffered'}<br>ASR partials ${t.asr_partial_observed?'Observed':'None observed'}<br>Endpoint → ASR final ${timingDetail(t,'endpoint_to_asr_final_ms')}</p></div><div><h3>Reasoning</h3><p>Request → first text delta ${timingDetail(t,'llm_request_to_first_delta_ms')}<br>Request total ${timingDetail(t,'llm_request_total_ms')}<br>Tokens/second Unavailable</p></div><div><h3>Synthesis</h3><p>Request → first PCM ${timingDetail(t,'tts_first_chunk_ms')}<br>Render total ${timingDetail(t,'tts_render_total_ms')}<br>First network byte Unavailable</p></div></div>`).join('')}</details>
    <p class="compact-note">Software timings only. Submitted audio includes pre-roll; ASR span includes input delivery. “First text delta” is not necessarily one token. First TTS PCM is decoded audio, not an audible response. Unavailable values are not zero.</p>`;
  }
  function components(){
    const replay=data.reports.find(r=>r.id===$('replay-select').value), d=replay?.data;
    const inventory=latest('inventory')?.data;
    const context=d?.execution_mode==='HPC-live-file-inference'?'HPC CPU':d?.execution_mode==='Pi-local-inference'?'Pi-reported':replay?labels[replay.evidence]||'Unverified':'Unknown';
    const rows=[
      ['Microphone capture',inventory?.audio_device_directory?'Device listed · untested':'Unknown'],
      ['Echo cancellation',d?.aec==='not_run'?'Not run in report':'Unknown'],
      ['Preprocessing',numeric(d?.model_rate)&&numeric(d?.input_rate)?context+' report':'Unknown'],
      ['Wake word',d?.wake_backend==='not_configured'?'Not configured in report':d?.wake_frames>0?context+' report':'Unknown'],
      ['VAD',d?.vad_frames>0?context+' report':'Unknown'],
      ['ASR / TTS',d?.agent_mode==='mock'?'Mock via our MCP pair':d?.agent_mode==='external-services'?'Configured via our MCP pair · inspect outcomes':'Unknown'],
      ['Reasoning',d?.backends?.llm==='mock'?'Mock response adapter':d?.backends?.llm==='external'?'External adapter configured':'Unknown'],
      ['Task execution',numeric(d?.mock_effects)?format(d.mock_effects)+' in-memory mock effect(s)':'Unknown']
    ];
    text('component-scope',replay?'Selected audio report':'Available reports');
    text('pi-status',data.reports.some(r=>r.data.execution_mode==='Pi-local-inference')?'Pi execution reported · unverified':inventory?.pi_detected?'Pi inventory only · unverified':'Pi verification unknown');
    $('component-list').innerHTML=rows.map(([k,v])=>`<div class="component"><span>${k}</span>${pill(v)}</div>`).join('');
  }
  function services(){
    const report=latest('services');
    const protocols={asr:['vllm-qwen-audio-v1','vllm-qwen3-asr-v1'],tts:['vllm-qwen-tts-v1']};
    const trials=reports('replay').filter(r=>Object.entries(protocols).some(([role,values])=>r.data.backends?.[role]==='external'&&values.includes(r.data.backend_contracts?.[role])));
    $('service-panel').hidden=!report&&!trials.length;
    if(!report&&!trials.length)return;
    const checks=report?.data.checks||[];
    $('service-content').innerHTML='<div class="service-grid">'+[['asr',8000,'Qwen2 Audio · 7B','endpoint_to_asr_final_ms'],['tts',8001,'Qwen3 TTS · CustomVoice','tts_first_chunk_ms']].map(([role,port,defaultModel,metricName])=>{
      const auth=checks.find(row=>row.port===port&&row.path==='/v1/models')?.status;
      const health=checks.find(row=>row.port===port&&row.path==='/health')?.status;
      const trial=trials.find(r=>r.data.backends?.[role]==='external'&&protocols[role].includes(r.data.backend_contracts?.[role]));
      const model=role==='asr'&&trial?.data.backend_contracts?.asr==='vllm-qwen3-asr-v1'?'Qwen3-ASR · dedicated transcription':defaultModel;
      // Correlate evidence within this application run/session, never another trace.
      const turns=(trial?.data.turns||[]).filter(t=>t.turn_id>0&&t.run_id===trial.data.run_id&&trial.data.session_ids?.includes(t.session_id));
      const successful=turns.filter(t=>t.outcome==='completed'&&Array.isArray(t.durations_ms?.[metricName])&&t.durations_ms[metricName].some(v=>numeric(v)&&v>=0));
      const passed=successful.length>0&&(role!=='tts'||trial.data.sink_chunks>0);
      const failed=turns.some(t=>t.outcome==='error');
      const label=passed?'Trial passed':failed?'Trial failed':trial?'Trial incomplete':'No inference trial';
      const values=successful.flatMap(t=>t.durations_ms[metricName]).filter(v=>numeric(v)&&v>=0);
      const latestValue=values.at(-1);
      const timing=passed?`<p class="service-timing">${quantity(latestValue,' ms',1)} <span>${role==='asr'?'endpoint → final text':'latest segment · request → first PCM'}</span></p>`:'';
      const probe=auth===200?'Catalog authenticated':auth===401?'Key required':health===200?'Health check reached':'Access unknown';
      const context=trial?`<p class="compact-note">${escape(date(trial.modified_utc))} · ${escape(sourceCaption(trial))}</p><button class="inspect-button" data-inspect="${escape(trial.id)}">Inspect ${role.toUpperCase()} trial</button>`:'';
      return `<div><h3>${role.toUpperCase()} ${pill(label,passed?'teal':failed?'red':'subtle')}</h3><p>${model}</p>${timing}${context}<p class="compact-note">${escape(probe)} · ${report?escape(date(report.modified_utc)):'no inspection'}</p></div>`;
    }).join('')+'</div><p class="compact-note">Historical file trials, not current service or microphone connectivity. Timings are software observations; inspect each source for reasoning and playback modes.</p>';
  }
  function scenarios(){
    const report=latest('scenarios');
    $('scenario-panel').hidden=!report;
    if(!report)return;
    const pause=report.data.scenarios.find(row=>row.observations?.['800ms_pause_endpoint_640ms_utterances']>1);
    $('scenario-content').innerHTML=`<p class="compact-note">${escape(sourceCaption(report))}</p><div class="table-wrap"><table><thead><tr><th scope="col">Scenario</th><th scope="col">Outcome</th><th scope="col">Observed elapsed</th></tr></thead><tbody>${report.data.scenarios.map(row=>`<tr><td>${escape(row.name.replaceAll('_',' '))}</td><td class="${row.passed===true?'scenario-pass':row.passed===false?'scenario-fail':''}">${row.passed===true?'✓ Passed':row.passed===false?'× Failed':'Unavailable'}</td><td>${quantity(row.elapsed_seconds,' s',2)}</td></tr>`).join('')}</tbody></table></div>${pause?'<p class="warning">Known turn-taking tradeoff: an 800 ms simulated pause splits at the 640 ms silence policy. The 1024 ms profile preserves one utterance but adds 384 ms of waiting. Real-recording validation is still needed.</p>':''}<p class="compact-note">Passed means the expected lifecycle/recovery occurred, including controlled faults. Mock barriers/delays are test inputs, not model speed or acoustic latency. Inspect the report for configuration.</p>`;
  }
  function evidenceList(){
    const filter=$('evidence-filter').value, rows=data.reports.filter(r=>filter==='all'||r.evidence===filter);
    $('report-count').textContent=data.reports.length+' sources';
    $('evidence-rows').innerHTML=rows.length?rows.map(r=>`<tr><td>${escape(filename(r))}</td><td>${evidence(r)}</td><td>${escape(date(r.modified_utc))}</td><td><button class="inspect-button" data-inspect="${escape(r.id)}" aria-label="Inspect ${escape(filename(r))}">Inspect</button></td></tr>`).join(''):'<tr><td colspan="4">No matching reports.</td></tr>';
    $('report-warnings').hidden=!data.warnings.length;
    $('report-warnings').textContent=data.warnings.length+' report(s) unavailable. Other data is retained.';
  }
  function route(){
    const key=location.hash==='#diagnostics'?'diagnostics':'assistant';
    $('view-assistant').hidden=key!=='assistant';$('view-diagnostics').hidden=key!=='diagnostics';
    if(key==='assistant')requestAnimationFrame(()=>{if(followLatest)$('transcript-messages').scrollTop=$('transcript-messages').scrollHeight;updateScroll();});
    document.querySelectorAll('nav [data-view]').forEach(a=>{a.classList.toggle('active',a.dataset.view===key);if(a.dataset.view===key)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  }
  function connection(state){
    const folder=sourceMode==='folder',server=sourceMode==='server';
    const stale=state==='stale'||state==='error';
    voiceStatus(stale);
    text('feed-state',stale?'Reports · Stale':folder?'Reports · Folder feed':server?'Reports · Local feed':sourceMode==='import'?'Reports · Imported file':'Reports · Snapshot');
    for(const id of ['feed-dot','footer-dot']){$(id).className='dot '+(stale?'stale':folder||server?'connected':'');}
    $('snapshot-description').textContent=stale?'Last saved reports':folder?'Terminal reports':server?'Local reports':'Saved reports';
    text('snapshot-time','Snapshot generated '+date(data.generated_utc));
    $('connection-description').textContent=stale?'Updates interrupted':folder?'Report folder connected':server?'Local report service':sourceMode==='import'?'Report file opened':'Saved snapshot';
    $('connection-note').textContent=stale?'Last report retained.':folder||server?'Report updates every 2 seconds.':'Saved report · no updates.';
    $('refresh-button').disabled=sourceMode!=='server'&&sourceMode!=='folder';
  }
  function paint(){history();transcript();activity();renderMusic();selected('replay-select','replay');selected('trace-select','trace');audio();conversation();components();services();scenarios();evidenceList();connection();route();}
  function accept(next,{clearNotice=false}={}){validate(next);for(const report of next.reports)if(report.kind==='transcript')savedSessions.set(report.data.session_id,report);data=next;if(runtime.liveReport)data.reports=[runtime.liveReport,...data.reports.filter(r=>r.kind!=='transcript')].slice(0,221);paint();if(clearNotice)showNotice('');}
  async function readFolder(revision){
    const handle=await directory.getFileHandle('me3-dashboard.json');
    const file=await handle.getFile();
    if(file.size>16*1024*1024)throw new Error('Report file exceeds 16 MiB');
    const next=validate(JSON.parse(await file.text()));
    if(revision!==connectionRevision)return;
    if(next.generated_utc!==lastStamp){lastDelivery=Date.now();lastStamp=next.generated_utc;accept(next);}
    if(Date.now()-lastDelivery>12000){connection('stale');showNotice('Terminal updates paused. Showing the last report.');}
    else{connection();showNotice('');}
  }
  async function refresh(){
    if(polling)return;
    polling=true;
    const revision=connectionRevision;
    try{
      if(sourceMode==='folder')await readFolder(revision);
      else if(sourceMode==='server'){
        const response=await fetch('/api/snapshot',{cache:'no-store',signal:AbortSignal.timeout(6000)});
        if(!response.ok)throw new Error('Report service unavailable');
        const next=await response.json();
        if(revision===connectionRevision)accept(next);
      }
    }catch(_){if(revision===connectionRevision){connection('error');showNotice('Updates unavailable. Showing the last report.','error');}}
    finally{polling=false;}
  }
  $('history-select').addEventListener('change',()=>{rememberSession($('history-select').value);runtime.ended=false;transcriptKey=null;taskKey=null;followLatest=true;paint();});
  $('transcript-messages').addEventListener('scroll',updateScroll);
  $('jump-latest').addEventListener('click',()=>{followLatest=true;$('transcript-messages').scrollTop=$('transcript-messages').scrollHeight;updateScroll();});
  new ResizeObserver(()=>{if(followLatest)$('transcript-messages').scrollTop=$('transcript-messages').scrollHeight;updateScroll();}).observe($('transcript-messages'));
  $('connect-feed').addEventListener('click',()=>{
    $('folder-support').textContent=window.showDirectoryPicker?'Only the selected folder is read.':'Folder updates require Chrome or Edge. Open report file is available instead.';
    $('choose-folder').disabled=!window.showDirectoryPicker;
    $('connection-dialog').showModal();
  });
  $('choose-folder').addEventListener('click',async()=>{
    try{directory=await window.showDirectoryPicker({mode:'read'});sourceMode='folder';connectionRevision++;lastStamp=null;lastDelivery=0;$('connection-dialog').close();await refresh();}
    catch(error){if(error.name!=='AbortError')showNotice('Could not open the feed folder. Try Open report file.');}
  });
  $('disconnect-feed').addEventListener('click',()=>{directory=null;sourceMode='snapshot';connectionRevision++;connection();showNotice('');$('connection-dialog').close();});
  $('close-connection').addEventListener('click',()=>$('connection-dialog').close());
  $('load-file').addEventListener('click',()=>$('file-input').click());
  $('file-input').addEventListener('change',async()=>{
    try{const file=$('file-input').files[0];if(!file)return;if(file.size>16*1024*1024)throw new Error('Too large');const next=validate(JSON.parse(await file.text()));sourceMode='import';connectionRevision++;directory=null;accept(next,{clearNotice:true});}
    catch(_){showNotice('That file is not a supported report snapshot. Previous data is unchanged.','error');}
    finally{$('file-input').value='';}
  });
  $('refresh-button').addEventListener('click',refresh);
  $('replay-select').addEventListener('change',()=>{audio();components();});$('trace-select').addEventListener('change',conversation);$('evidence-filter').addEventListener('change',evidenceList);
  function inspectReport(r){if(!r)return;$('dialog-title').textContent=filename(r);$('dialog-source').textContent=r.source+(r.sha256?' · SHA-256 '+r.sha256:' · current runtime state');$('dialog-content').textContent=JSON.stringify(r.data,null,2);$('evidence-dialog').showModal();}
  for(const id of ['evidence-rows','service-content'])$(id).addEventListener('click',event=>{const button=event.target.closest('[data-inspect]');if(button)inspectReport(data.reports.find(r=>r.id===button.dataset.inspect));});
  function sessionDetails(){
    const report=assistantReport(), d=report?.data;
    const rows=[['Model services',runtime.connected&&runtime.providerRoles?Object.entries(runtime.providerRoles).filter(([role,value])=>['asr','llm','tts'].includes(role)&&typeof value==='string'&&value.length<=200).map(([role,value])=>role.toUpperCase()+': '+value).join(' · '):'See the selected run configuration'],['Conversation context',runtime.connected?'Current engagement only; a new wake starts fresh':'Saved transcript; not restored as model context'],['Action records','Separate from chat; do not resume tasks or timers'],['Voice runtime',runtime.connected?'Browser microphone connected':'Not connected'],['Reported state',runtime.connected?(runtime.status==='wake_listening'?'Wake listening':sessionLabels[runtime.status]||'Connected'):d?sessionLabels[d.state]||'Unknown':'Unknown'],['Echo cancellation',runtime.connected?(runtime.aec===true?'Browser reports on':runtime.aec===false?'Browser reports off':'Browser did not report'): 'Unavailable'],['Reasoning requests',runtime.requestBudget?(runtime.requestBudget.limit===null?`${runtime.requestBudget.used} submitted; provider limits apply`:`${runtime.requestBudget.used} of ${runtime.requestBudget.limit} used this bridge run`):'Unavailable'],['Evidence',runtime.connected?'Live browser transport · acoustic behavior unverified':d?.mode==='mock'?'Simulated session':d?'Reported software session':'No session report'],['Report source',sourceMode==='folder'?'Folder feed':sourceMode==='server'?'Local report feed':sourceMode==='import'?'Imported snapshot':'Saved snapshot'],['Updated',d?date(d.updated_utc):'Unavailable']];
    $('session-details').innerHTML=rows.map(([key,value])=>`<div><dt>${escape(key)}</dt><dd>${escape(value)}</dd></div>`).join('');
    const aecNote=runtime.aec===true?'the browser reports echo cancellation enabled':runtime.aec===false?'the browser reports echo cancellation disabled':'the browser did not report its echo cancellation setting';
    text('session-dialog-note',runtime.connected?`Browser microphone capture and playback are connected; ${aecNote}. This setting does not measure acoustic cancellation, and Pi behavior is unverified.`:runtime.available?'Start conversation connects the browser microphone to the loopback voice runtime. Saved reports do not indicate a live connection.':'This is saved report evidence. No browser microphone or voice runtime is connected.');
    $('inspect-session-report').disabled=!report;
  }
  function openSession(){sessionDetails();$('session-dialog').showModal();}
  $('task-items').addEventListener('click',async event=>{
    const button=event.target.closest('[data-preview-artifact]');
    if(!button)return;
    $('artifact-title').textContent=button.dataset.previewName;
    $('artifact-source').textContent='Generated file · displayed as text only; not executed';
    $('artifact-content').textContent='Loading file…';
    $('artifact-dialog').showModal();
    try{
      if(!runtime.artifactsAvailable)throw new Error('Task file access is unavailable in this view.');
      const response=await fetch(`/api/voice/tasks/${encodeURIComponent(button.dataset.previewArtifact)}/artifacts/${encodeURIComponent(button.dataset.previewName)}`,{cache:'no-store',signal:AbortSignal.timeout(5000)});
      if(!response.ok)throw new Error('This task file is no longer available from the connected runtime.');
      $('artifact-content').textContent=await response.text();
    }catch(error){$('artifact-content').textContent=error.message||'Could not load the task file.';}
  });
  $('close-artifact').addEventListener('click',()=>$('artifact-dialog').close());
  $('inspect-session').addEventListener('click',openSession);
  $('voice-help').addEventListener('click',openSession);
  $('close-session').addEventListener('click',()=>$('session-dialog').close());
  $('inspect-session-report').addEventListener('click',()=>{$('session-dialog').close();$('inspect-session').focus();inspectReport(assistantReport());});
  $('close-dialog').addEventListener('click',()=>$('evidence-dialog').close());
  $('export-button').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='me3-dashboard.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  for(const dialog of [$('connection-dialog'),$('evidence-dialog'),$('session-dialog'),$('artifact-dialog')])dialog.addEventListener('keydown',event=>{
    if(event.key!=='Tab')return;
    const controls=[...dialog.querySelectorAll('button:not(:disabled),[tabindex="0"]')];
    const first=controls[0],last=controls.at(-1);
    if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
    else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
  });
  document.querySelector('.skip').addEventListener('click',event=>{event.preventDefault();$('main').focus();});
  window.addEventListener('hashchange',route);
  function updateWave(){
    const playback=runtime.connected&&runtime.status==='speaking'&&runtime.sources.size>0;
    const microphone=runtime.connected&&!runtime.micMuted&&['listening','wake_listening'].includes(runtime.status)&&runtime.media?.getAudioTracks()?.[0]?.readyState==='live';
    const source=playback?'playback':microphone?'microphone':'none';
    if(voiceNode.dataset.levelSource!==source)voiceNode.dataset.levelSource=source;
    const hasPlaybackLevel=source==='playback'&&runtime.playLevel>.012;
    if(voiceNode.dataset.wave!==String(hasPlaybackLevel))voiceNode.dataset.wave=String(hasPlaybackLevel);
    const level=source==='microphone'?Math.min(1,runtime.micVisual*5):source==='playback'?Math.min(1,runtime.playLevel*3):0;
    voiceNode.style.setProperty('--mic-level',source==='microphone'?level:0);
    const base=[14,22,34,42,34,22,14],weight=[.38,.58,.78,1,.78,.58,.38];
    waveBars.forEach((bar,index)=>{const height=`${base[index]+Math.round(level*28*weight[index])}px`;if(bar.style.height!==height)bar.style.height=height;});
    const activity=String(runtime.connected&&!runtime.micMuted&&runtime.micLevel>.025);
    if($('mic-indicator').dataset.activity!==activity)$('mic-indicator').dataset.activity=activity;
  }
  function stopPlayback(){
    runtime.playEpoch++;
    for(const source of runtime.sources.values())try{source.stop();}catch(_){}
    runtime.sources.clear();runtime.nextPlay=0;runtime.playLevel=0;updateWave();
  }
  function playWakeCue(sessionId){
    const context=runtime.context;
    if(!runtime.connected||!sessionId||runtime.wakeCueSessionId===sessionId)return;
    runtime.wakeCueSessionId=sessionId;
    voiceNode.dataset.wakeConfirmed='true';
    if(runtime.wakeCueTimer!==null)clearTimeout(runtime.wakeCueTimer);
    runtime.wakeCueTimer=setTimeout(()=>{
      if(runtime.wakeCueSessionId===sessionId)delete voiceNode.dataset.wakeConfirmed;
      runtime.wakeCueTimer=null;
    },850);
    if(!context||context.state==='closed')return;
    try{
      const start=context.currentTime+.015;
      [[587.33,0,.105],[880,.12,.13]].forEach(([frequency,offset,duration])=>{
        const oscillator=context.createOscillator(),gain=context.createGain(),when=start+offset;
        oscillator.type='sine';oscillator.frequency.setValueAtTime(frequency,when);
        gain.gain.setValueAtTime(.0001,when);gain.gain.exponentialRampToValueAtTime(.052,when+.01);gain.gain.exponentialRampToValueAtTime(.0001,when+duration);
        oscillator.connect(gain);gain.connect(context.destination);oscillator.start(when);oscillator.stop(when+duration+.012);
      });
    }catch(_){/* The wake confirmation is optional; voice capture must continue if output is unavailable. */}
  }
  function interruptVoice(){
    if(!runtime.connected||runtime.status!=='speaking'||runtime.interrupting)return;
    runtime.interrupting=true;stopPlayback();text('voice-control-note','');voiceStatus();
    if(runtime.ws?.readyState===WebSocket.OPEN)runtime.ws.send(JSON.stringify({type:'interrupt'}));
  }
  async function probeBridgeRecovery(){
    const generation=++recoveryGeneration;
    for(const delay of [500,1000,2000,4000,8000]){
      await new Promise(resolve=>setTimeout(resolve,delay));
      if(generation!==recoveryGeneration||runtime.connected||runtime.connecting)return;
      try{
        const response=await fetch('/api/health',{cache:'no-store',signal:AbortSignal.timeout(2000)});
        const health=response.ok?await response.json():null;
        if(health?.bridge==='ready'&&health.voice_session==='idle'){
          showNotice('Voice bridge is reachable again. Click Start conversation to open a new session; the previous session was not resumed and no command was replayed.');
          return;
        }
      }catch(_){}
    }
    if(generation===recoveryGeneration&&!runtime.connected&&!runtime.connecting)
      showNotice('Voice bridge is still unavailable. Check the HPC bridge terminal and VS Code port forwarding, then click Start conversation to retry. No command was replayed.','error');
  }
  function closeVoice(ended=false){
    recoveryGeneration++;
    if(runtime.startupTimer!==null){clearTimeout(runtime.startupTimer);runtime.startupTimer=null;}
    if(runtime.wakeCueTimer!==null){clearTimeout(runtime.wakeCueTimer);runtime.wakeCueTimer=null;}
    stopPlayback();
    if(runtime.ws?.readyState===WebSocket.OPEN)runtime.ws.send(JSON.stringify({type:'stop'}));
    runtime.ws?.close();runtime.processor?.disconnect();runtime.input?.disconnect();runtime.mute?.disconnect();
    runtime.media?.getTracks().forEach(track=>track.stop());
    runtime.context?.close().catch(()=>{});
    runtime.ws=runtime.media=runtime.context=runtime.processor=runtime.input=runtime.mute=null;
    $('fresh-context').hidden=true;
    runtime.connected=false;runtime.connecting=false;runtime.status='disconnected';runtime.inputTurnOpen=null;runtime.interrupting=false;runtime.aec=null;runtime.micMuted=false;runtime.micLevel=0;runtime.micVisual=0;runtime.playLevel=0;runtime.toolActive=false;runtime.ended=ended;runtime.runId=null;runtime.sessionId=null;runtime.wakeCueSessionId=null;delete voiceNode.dataset.wakeConfirmed;runtime.turnId=0;runtime.completedTurnId=null;runtime.eventSequence=0;runtime.statusSequence=0;
    $('voice-start').textContent='Start conversation';$('voice-start').disabled=false;$('voice-start').setAttribute('aria-busy','false');$('voice-interrupt').disabled=true;$('voice-mute').hidden=true;$('voice-mute').textContent='Mute microphone';
    text('voice-control-note','');history();voiceStatus();
    if(sourceMode==='server')refresh();
  }
  function receiveAudio(bytes){
    if(!runtime.connected||runtime.interrupting||!runtime.context||bytes.byteLength<6||bytes.byteLength>8196||bytes.byteLength%2)return;
    const view=new DataView(bytes),id=view.getUint32(0,true),frames=(bytes.byteLength-4)/2;
    if(runtime.sources.size>=12){stopPlayback();showNotice('Playback buffer limit reached. Audio stopped.','error');return;}
    const buffer=runtime.context.createBuffer(1,frames,16000),out=buffer.getChannelData(0);
    for(let i=0;i<frames;i++)out[i]=view.getInt16(4+i*2,true)/32768;
    const source=runtime.context.createBufferSource();source.buffer=buffer;source.connect(runtime.context.destination);
    runtime.playLevel=Math.sqrt(out.reduce((sum,value)=>sum+value*value,0)/frames);updateWave();
    const epoch=runtime.playEpoch,when=Math.max(runtime.context.currentTime+.035,runtime.nextPlay);
    runtime.nextPlay=when+buffer.duration;runtime.sources.set(id,source);updateWave();
    source.onended=()=>{if(runtime.sources.get(id)===source)runtime.sources.delete(id);if(!runtime.sources.size){runtime.playLevel=0;updateWave();}if(epoch===runtime.playEpoch&&runtime.ws?.readyState===WebSocket.OPEN)runtime.ws.send(JSON.stringify({type:'audio_ack',id}));};
    source.start(when);
  }
  function captureBlock(event){
    if(!runtime.connected||runtime.ws?.readyState!==WebSocket.OPEN)return;
    const input=event.inputBuffer.getChannelData(0),pcm=new Int16Array(input.length);
    let energy=0;
    for(let i=0;i<input.length;i++){const value=Math.max(-1,Math.min(1,input[i]));energy+=value*value;pcm[i]=Math.round(value<0?value*32768:value*32767);}
    runtime.micLevel=runtime.micMuted?0:Math.sqrt(energy/input.length);
    runtime.micVisual=runtime.micMuted?0:runtime.micVisual*.78+runtime.micLevel*.22;
    updateWave();
    if(runtime.ws.bufferedAmount>131072){showNotice('Microphone transport fell behind. Voice stopped.','error');closeVoice();return;}
    runtime.ws.send(pcm.buffer);
  }
  function musicTime(ms){
    if(!numeric(ms)||ms<0)return '—:—';
    const seconds=Math.floor(ms/1000);
    return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;
  }
  function safeArtwork(value){
    try{const url=new URL(value);return url.protocol==='https:'&&url.hostname==='i.scdn.co'?url.href:'';}catch(_){return '';}
  }
  function reportMusicPlayback(value,source){
    if(!runtime.connected||runtime.ws?.readyState!==WebSocket.OPEN)return;
    const status=value?.status;
    if(!['playing','paused','not_playing','buffering'].includes(status))return;
    const m=runtime.music,track=value?.track;
    const trackKey=typeof track?.id==='string'&&track.id?track.id:track?`${track.name||''}\u0000${track.artists||''}`:'';
    const trackChanged=m.reportedTrackKey!==null&&trackKey!==m.reportedTrackKey;
    if(status===m.reportedPlaybackStatus&&!trackChanged)return;
    m.reportedPlaybackStatus=status;m.reportedTrackKey=trackKey;
    runtime.ws.send(JSON.stringify({type:'music_playback_observed',status,source:source==='sdk'?'web_playback_sdk':'spotify_api',track_changed:trackChanged}));
  }
  function setMusicState(value,source='api'){
    if(!value||!['playing','paused','not_playing','buffering','unavailable'].includes(value.status))return;
    const m=runtime.music;
    if(source==='api'&&m.capabilities?.output_backend==='web_playback_sdk'&&(!m.sdkReady||value.target_matched!==true))return;
    if(source==='api'&&m.sdkReady&&m.state?.source==='sdk'){
      m.state.volume_percent=numeric(value.volume_percent)?value.volume_percent:null;
      m.state.supports_volume=value.supports_volume===true;
      renderMusic();
      return;
    }
    const track=value.track&&typeof value.track==='object'?value.track:null;
    const artists=typeof track?.artists==='string'?track.artists.slice(0,240):'';
    runtime.music.state={
      status:value.status,
      output_backend:source==='sdk'?'web_playback_sdk':value.output_backend,
      device_name:typeof value.device_name==='string'?value.device_name.slice(0,80):null,
      track:track?{name:typeof track.name==='string'?track.name.slice(0,160):'Unknown track',artists,artwork:safeArtwork(track.artwork)}:null,
      progress_ms:numeric(value.progress_ms)&&value.progress_ms>=0?value.progress_ms:null,
      duration_ms:numeric(value.duration_ms)&&value.duration_ms>0?value.duration_ms:null,
      volume_percent:numeric(value.volume_percent)&&value.volume_percent>=0&&value.volume_percent<=100?value.volume_percent:null,
      supports_volume:value.supports_volume===true,
      audible:null,
      reported:true,
      disallows:value.disallows&&typeof value.disallows==='object'?{skipping_next:value.disallows.skipping_next===true,skipping_prev:value.disallows.skipping_prev===true,seeking:value.disallows.seeking===true}:{},
      source
    };
    runtime.music.statusError='';runtime.music.stateStale=false;
    runtime.music.stateUpdatedAt=performance.now();
    reportMusicPlayback(value,source);
    if(source==='sdk'||runtime.music.sdkReady&&runtime.music.message==='Spotify player connected inside Swift.'&&['playing','paused'].includes(value.status))runtime.music.message='';
    renderMusic();
  }
  function renderMusic(){
    const m=runtime.music,cap=m.capabilities,panel=$('spotify-player');
    const visible=showMusicPanel();
    const footerVisible=livePage&&m.capabilitiesLoaded&&cap?.configured;
    $('music-setup').hidden=!footerVisible;
    const footerLabel=cap?.setup_required?'Spotify · permission needed':m.sdkReady?(m.activationRequired?'Spotify · enable audio':'Spotify connected'):m.sdkConnecting?'Spotify connecting…':m.message?.includes('timed out')?'Spotify · timed out':m.message?.includes('failed')?'Spotify · connection failed':'Spotify not connected';
    text('music-setup-status',footerLabel);
    $('music-setup-status').title=m.message||footerLabel;
    $('music-setup-connect').hidden=m.sdkReady&&!m.activationRequired||!cap?.browser_player;
    $('music-setup-connect').disabled=m.sdkConnecting;
    $('music-setup-connect').textContent=m.sdkConnecting?'Connecting…':m.activationRequired?'Enable audio':'Connect Spotify';
    $('spotify-diagnostics').hidden=!footerVisible;
    text('spotify-diagnostic-detail',cap?.setup_required?'Spotify SDK needs an updated consent grant with streaming, user-read-email and user-read-private. Use the secured terminal setup.':m.message||m.statusError||footerLabel);
    panel.hidden=!visible;
    $('task-activity').hidden=!visible&&!Boolean(visibleTasks(assistantReport()?.data).length);
    $('conversation-workspace').dataset.hasTasks=String(visible||Boolean(visibleTasks(assistantReport()?.data).length));
    if(!livePage||!m.capabilitiesLoaded&&!m.capabilitiesError)return;
    const connect=$('spotify-connect');
    if(!cap?.configured){
      const missingCapability=m.capabilitiesError&&!m.capabilitiesLoaded;
      text('spotify-output-label',missingCapability?'Bridge status unavailable':'Not configured in this bridge');
      text('spotify-track-title',missingCapability?'Spotify setup could not be checked':'Spotify is not connected to Swift');
      text('spotify-track-artist',missingCapability?'Refresh after reconnecting the voice bridge.':'Restart the voice bridge with ME3_MUSIC_CONFIG=var/cache/music.json.');
      $('spotify-art-image').removeAttribute('src');$('spotify-art-image').hidden=true;$('spotify-art-fallback').hidden=false;
      $('spotify-seek').disabled=true;$('spotify-seek').max='1';$('spotify-seek').value='0';$('spotify-progress-fill').style.width='0%';
      text('spotify-current-time','—:—');text('spotify-total-time','—:—');
      $('spotify-previous').disabled=true;$('spotify-next').disabled=true;$('spotify-toggle').disabled=true;
      $('spotify-toggle').innerHTML=spotifyPlayIcon;$('spotify-toggle').setAttribute('aria-label','Play or pause');
      connect.hidden=true;connect.disabled=true;
      text('spotify-caption',m.capabilitiesError?'Swift could not read Spotify capability from the running bridge. Restart the bridge, then refresh this page.':'Spotify controls will appear here once the bridge loads the ME3 Spotify configuration.');
      return;
    }
    const browser=cap.output_backend==='web_playback_sdk';
    const state=m.state,track=state?.track;
    const hasReceiver=browser?m.sdkReady:Boolean(state?.device_name||cap.target_configured);
    let outputLabel;
    if(browser){
      outputLabel=state?.source==='sdk'?'Swift · this browser':m.sdkReady&&state?.device_name?`Swift ready · active Spotify: ${state.device_name}`:m.sdkReady?'Swift · this browser ready':m.sdkConnecting?'Connecting this browser…':'Browser player ready to connect';
    }else if(cap.device_name&&state?.device_name&&cap.device_name!==state.device_name){outputLabel=`Active: ${state.device_name} · Target: ${cap.device_name}`;}
    else outputLabel=state?.device_name?`Spotify Connect · ${state.device_name}`:cap.device_name?`Spotify Connect · ${cap.device_name}`:'Spotify Connect · waiting for player';
    text('spotify-output-label',outputLabel);
    text('spotify-track-title',track?.name||(state?.status==='unavailable'?'Playback status unavailable':'Nothing playing'));
    text('spotify-track-artist',track?.artists||'Spotify playback will appear here.');
    const art=$('spotify-art-image'),artUrl=safeArtwork(track?.artwork);
    if(artUrl){if(art.src!==artUrl)art.src=artUrl;art.hidden=false;$('spotify-art-fallback').hidden=true;}
    else{art.removeAttribute('src');art.hidden=true;$('spotify-art-fallback').hidden=false;}
    const duration=state?.duration_ms||0;
    let position=state?.progress_ms||0;
    if(state?.status==='playing'&&!m.stateStale&&m.stateUpdatedAt)position+=Math.max(0,performance.now()-m.stateUpdatedAt);
    position=duration?Math.min(duration,position):0;
    const seek=$('spotify-seek');
    seek.max=String(duration||1);
    if(document.activeElement!==seek)seek.value=String(position);
    $('spotify-progress-fill').style.width=(duration?`${position/duration*100}%`:'0%');
    text('spotify-current-time',musicTime(position));text('spotify-total-time',musicTime(duration));
    const volume=$('spotify-volume'),volumeValue=state?.volume_percent;
    if(document.activeElement!==volume)volume.value=numeric(volumeValue)?String(volumeValue):'50';
    text('spotify-volume-value',numeric(volumeValue)?`${volumeValue}%`:'Unavailable');
    const canControl=hasReceiver&&!m.stateStale&&Boolean(track)&&['playing','paused'].includes(state?.status);
    $('spotify-previous').disabled=!canControl||m.commandBusy||state?.disallows?.skipping_prev===true;
    $('spotify-previous').title=state?.disallows?.skipping_prev?'Spotify reports no available previous track.':'Previous track';
    $('spotify-next').disabled=!canControl||m.commandBusy||state?.disallows?.skipping_next===true;
    $('spotify-next').title=state?.disallows?.skipping_next?'Spotify reports no available next track.':'Next track';
    $('spotify-toggle').disabled=!canControl||m.commandBusy;
    $('spotify-toggle').innerHTML=state?.status==='playing'?spotifyPauseIcon:spotifyPlayIcon;
    $('spotify-toggle').setAttribute('aria-label',state?.status==='playing'?'Pause Spotify':'Resume Spotify');
    seek.disabled=!canControl||!duration||m.commandBusy||state?.disallows?.seeking===true;
    volume.disabled=!canControl||state?.supports_volume!==true||!numeric(volumeValue)||m.commandBusy;
    connect.hidden=!browser||!cap.browser_player||(m.sdkReady&&!m.activationRequired);
    connect.disabled=m.sdkConnecting;
    connect.textContent=m.sdkConnecting?'Connecting…':m.activationRequired?'Enable Spotify audio':'Connect Spotify';
    let caption=m.message||m.statusError;
    if(!caption&&cap.setup_required)caption='Spotify browser playback needs streaming and profile permissions. See Diagnostics.';
    if(!caption&&browser&&!cap.browser_player)caption='Spotify browser playback is not authorized yet.';
    if(!caption&&browser&&!m.sdkReady)caption='Connect Spotify here to play and control music inside Swift.';
    if(!caption&&state?.status==='unavailable')caption='Spotify status is unavailable. Check authorization and network access.';
    if(!caption&&state?.status==='playing')caption=`Spotify reports playback on ${state.device_name||'the selected player'}. Sound is not acoustically verified.`;
    if(!caption&&state?.status==='paused')caption='Spotify reports playback paused.';
    if(!caption&&state?.status==='not_playing')caption=cap.target_configured?'The configured Spotify player is ready when it appears online.':'No active Spotify playback is reported.';
    text('spotify-caption',caption||'Waiting for Spotify playback state.');
  }
  async function registerSpotifyDevice(type,deviceId){
    const response=await fetch('/api/voice/music/device',{method:'POST',headers:{'Content-Type':'application/json'},cache:'no-store',signal:AbortSignal.timeout(8000),body:JSON.stringify({type,device_id:deviceId})});
    if(!response.ok)throw new Error('Spotify player registration failed.');
    return response.json();
  }
  function loadSpotifySdk(){
    const m=runtime.music;
    if(m.sdkLoaded||m.sdkLoading)return;
    if(window.Spotify?.Player){m.sdkLoaded=true;if(m.connectRequested)connectSpotifyFromGesture();return;}
    m.sdkLoading=true;
    window.onSpotifyWebPlaybackSDKReady=()=>{m.sdkLoading=false;m.sdkLoaded=true;renderMusic();if(m.connectRequested)connectSpotifyFromGesture();};
    const script=document.createElement('script');script.src='https://sdk.scdn.co/spotify-player.js';script.async=true;if(pageNonce)script.nonce=pageNonce;
    script.onerror=()=>{m.sdkLoading=false;m.connectRequested=false;m.message='Spotify player library could not be loaded. Check network access, then press Connect Spotify to retry.';renderMusic();};
    document.head.append(script);
  }
  function spotifySdkState(value){
    if(!value){setMusicState({status:'not_playing',output_backend:'web_playback_sdk',device_name:'Swift · this browser'},'sdk');return;}
    const item=value.track_window?.current_track,artists=Array.isArray(item?.artists)?item.artists.slice(0,5).map(row=>row.name).filter(name=>typeof name==='string').join(', '):'';
    const image=Array.isArray(item?.album?.images)?item.album.images.find(row=>safeArtwork(row?.url)):null;
    setMusicState({status:value.paused?'paused':'playing',output_backend:'web_playback_sdk',device_name:'Swift · this browser',disallows:value.disallows,track:item?{id:item.id,name:item.name,artists,artwork:image?.url||''}:null,progress_ms:value.position,duration_ms:value.duration},'sdk');
  }
  function connectSpotifyFromGesture(){
    const m=runtime.music,cap=m.capabilities;
    if(!livePage)return;
    if(!m.capabilitiesLoaded){m.connectRequested=true;m.message='Checking Spotify setup…';renderMusic();return;}
    if(!cap?.configured){m.connectRequested=false;m.message='Spotify is not configured in the running voice bridge.';renderMusic();return;}
    if(cap.output_backend!=='web_playback_sdk'||!cap.browser_player)return;
    if(m.sdkReady){if(m.activationRequired)void activateSpotifyFromGesture();return;}
    if(m.sdkConnecting)return;
    if(!m.sdkLoaded&&!window.Spotify){m.connectRequested=true;loadSpotifySdk();m.message='Loading Spotify in Swift…';renderMusic();return;}
    if(!window.Spotify?.Player){m.connectRequested=true;m.message='Spotify is still loading in Swift…';renderMusic();return;}
    m.connectRequested=false;
    m.sdkConnecting=true;m.message='Connecting Spotify inside Swift…';
    if(m.player){
      const player=m.player;
      m.connectTimer=setTimeout(()=>failSpotifyConnect(player,'Spotify connection timed out. Retry the footer Connect Spotify action; see Diagnostics for details.'),15000);
      try{void player.activateElement();}catch(_){}
      Promise.resolve(m.disconnectPending).then(()=>player.connect()).then(ok=>{if(!ok)failSpotifyConnect(player,'Spotify connection failed. Retry Connect Spotify.');}).catch(()=>failSpotifyConnect(player,'Spotify connection failed. Retry Connect Spotify.'));
      renderMusic();return;
    }
    const player=new window.Spotify.Player({
      name:'Swift Voice Assistant',
      volume:0.65,
      getOAuthToken:callback=>{fetch('/api/voice/music/token',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(12000)}).then(async response=>{const value=await response.json();if(!response.ok||!value.available||typeof value.access_token!=='string')throw new Error('Spotify authorization needs attention.');if(m.player===player)callback(value.access_token);}).catch(()=>{failSpotifyConnect(player,'Spotify could not get a playback token. Check Spotify authorization in the secured terminal setup.');});}
    });
    m.player=player;
    m.connectTimer=setTimeout(()=>failSpotifyConnect(player,'Spotify connection timed out. Check browser privacy or network restrictions, then retry Connect Spotify.'),15000);
    player.addListener('ready',async({device_id})=>{
      if(m.player!==player||!m.sdkConnecting)return;
      clearTimeout(m.connectTimer);
      m.deviceId=device_id;m.sdkReady=true;m.sdkConnecting=false;m.message='Spotify player connected inside Swift.';
      try{const value=await registerSpotifyDevice('ready',device_id);m.capabilities=value;}catch(_){m.message='Spotify is connected in this browser, but voice commands cannot target it yet.';}
      renderMusic();refreshMusicState(true);
    });
    player.addListener('not_ready',async({device_id})=>{
      if(m.player!==player)return;
      if(m.deviceId===device_id){m.sdkReady=false;m.deviceId=null;try{await registerSpotifyDevice('not_ready',device_id);}catch(_){}m.message='Spotify browser player disconnected.';renderMusic();}
    });
    player.addListener('player_state_changed',value=>{if(m.player===player)spotifySdkState(value);});
    player.addListener('autoplay_failed',()=>{if(m.player!==player)return;m.activationRequired=true;m.message='Click Enable Spotify audio so this browser can play music.';renderMusic();});
    player.addListener('initialization_error',()=>failSpotifyConnect(player,'Spotify could not initialize in this browser.'));
    player.addListener('authentication_error',error=>failSpotifyConnect(player,/scope/i.test(String(error?.message||''))?'Spotify authorization failed: SDK permissions rejected. See Diagnostics.':'Spotify authentication failed. See Diagnostics and the secured setup.'));
    player.addListener('account_error',()=>failSpotifyConnect(player,'Spotify requires a Premium account for in-page playback.'));
    player.addListener('playback_error',()=>{if(m.player!==player)return;m.message='Spotify reported a playback error. Check this browser’s audio output.';renderMusic();});
    try{void player.activateElement();}catch(_){}
    void player.connect().then(connected=>{if(!connected)failSpotifyConnect(player,'Spotify could not connect this browser as a player. Retry Connect Spotify.');}).catch(()=>failSpotifyConnect(player,'Spotify could not connect this browser as a player. Retry Connect Spotify.'));
    renderMusic();
  }
  function failSpotifyConnect(player,message){
    const m=runtime.music;if(m.player!==player)return;
    clearTimeout(m.connectTimer);m.sdkConnecting=false;m.sdkReady=false;m.connectRequested=false;m.message=message;
    try{m.disconnectPending=Promise.resolve(player.disconnect()).catch(()=>{});}catch(_){}
    renderMusic();
  }
  async function activateSpotifyFromGesture(){
    const m=runtime.music;
    if(!m.player?.activateElement)return;
    try{const activated=await m.player.activateElement();if(activated===false)throw new Error('activation rejected');m.activationRequired=false;m.message='Spotify audio is enabled in this browser.';}
    catch(_){m.message='Spotify still needs browser audio permission. Check this tab’s site settings.';}
    renderMusic();
  }
  let musicStatusBusy=false;
  async function refreshMusicState(force=false){
    const m=runtime.music;if(!livePage||!m.capabilities?.configured||musicStatusBusy)return;
    musicStatusBusy=true;
    try{
      const response=await fetch('/api/voice/music/status',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(4000)});
      const value=await response.json();
      if(!response.ok){musicStatusUnavailable();return;}
      if(!force&&m.state?.source==='sdk'&&performance.now()-m.stateUpdatedAt<5000)return;
      setMusicState(value,'api');
    }catch(_){musicStatusUnavailable();}
    finally{musicStatusBusy=false;}
  }
  function musicStatusUnavailable(){
    const m=runtime.music;
    if(m.sdkReady&&m.state?.source==='sdk')return;
    if(m.state&&!m.stateStale&&m.state.status==='playing'&&m.state.progress_ms!==null)m.state.progress_ms+=Math.max(0,performance.now()-m.stateUpdatedAt);
    m.stateStale=true;m.statusError='Playback status temporarily unavailable. Showing the last confirmed state.';renderMusic();
  }
  function recordMusicControl(action,outcome){
    if(runtime.connected&&runtime.ws?.readyState===WebSocket.OPEN)runtime.ws.send(JSON.stringify({type:'music_ui_control',action,outcome}));
  }
  async function spotifyControl(action,positionMs=null){
    const m=runtime.music,cap=m.capabilities;if(m.commandBusy||!cap?.configured)return;
    m.commandBusy=true;m.message='Sending playback control to Spotify…';renderMusic();
    try{
      const command={command_id:'ui-'+crypto.randomUUID().replaceAll('-',''),action};
      if(action==='seek')command.position_ms=positionMs;
      if(action==='volume_set')command.volume_percent=positionMs;
      const response=await fetch('/api/voice/music/control',{method:'POST',headers:{'Content-Type':'application/json'},cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(8000),body:JSON.stringify(command)});
      const result=await response.json();if(!response.ok||!result.available)throw new Error(result.message||'Spotify rejected the control.');
      setMusicState(result.state,'api');m.message=result.receipt?.summary||'Spotify accepted the control.';
      recordMusicControl(action,'accepted');
    }catch(error){recordMusicControl(action,'failed');m.message=String(error.message||'Spotify control failed.').slice(0,180);}
    finally{m.commandBusy=false;renderMusic();}
  }
  $('music-setup-connect').addEventListener('click',connectSpotifyFromGesture);
  $('spotify-connect').addEventListener('click',connectSpotifyFromGesture);
  $('spotify-previous').addEventListener('click',()=>spotifyControl('previous'));
  $('spotify-next').addEventListener('click',()=>spotifyControl('next'));
  $('spotify-toggle').addEventListener('click',()=>spotifyControl(runtime.music.state?.status==='playing'?'pause':'resume'));
  $('spotify-seek').addEventListener('input',event=>{
    const state=runtime.music.state,duration=state?.duration_ms||0,value=Number(event.currentTarget.value);
    if(!duration||!Number.isFinite(value))return;
    $('spotify-current-time').textContent=musicTime(value);$('spotify-progress-fill').style.width=`${value/duration*100}%`;
  });
  $('spotify-seek').addEventListener('change',event=>spotifyControl('seek',Number(event.currentTarget.value)));
  $('spotify-volume').addEventListener('input',event=>text('spotify-volume-value',`${Number(event.currentTarget.value)}%`));
  $('spotify-volume').addEventListener('change',event=>spotifyControl('volume_set',Number(event.currentTarget.value)));
  function microphoneFailure(error){
    const name=error?.name||'';
    if(name==='NotAllowedError'||name==='SecurityError')return 'Microphone permission was denied. Allow microphone access for this localhost page in the browser site settings, then try again.';
    if(name==='NotFoundError'||name==='DevicesNotFoundError')return 'No microphone is available to this browser. Connect or enable the intended USB microphone, then retry.';
    if(name==='NotReadableError'||name==='TrackStartError')return 'The microphone is present but could not be opened. Close other apps using it and check the selected input device.';
    if(name==='OverconstrainedError'||name==='ConstraintNotSatisfiedError')return 'The microphone does not support the requested capture format. Check the browser input device and retry.';
    if(name==='AbortError')return 'Microphone startup was interrupted. Check the browser device state and retry.';
    return 'The browser could not open its microphone. Check site permission, device selection, and whether another app owns the device.';
  }
  function voiceStartupStep(label){
    const button=$('voice-start');
    button.disabled=true;button.textContent=label;button.setAttribute('aria-busy','true');
    text('voice-control-note',label);
  }
  async function startVoice(){
    if(runtime.connected){closeVoice(true);return;}
    if(runtime.connecting)return;
    recoveryGeneration++;
    runtime.connecting=true;
    connectSpotifyFromGesture();
    voiceStartupStep('Checking bridge…');
    try{
      if(!livePage)throw new Error('Voice controls require the live Swift bridge page. Open the forwarded localhost URL, not an exported report.');
      let health;
      try{
        const response=await fetch('/api/health',{cache:'no-store',signal:AbortSignal.timeout(4000)});
        if(!response.ok)throw new Error('health response');
        health=await response.json();
      }catch(_){throw new Error('Swift voice bridge is unreachable at this forwarded port. Keep the bridge running on HPC and confirm VS Code forwards local 127.0.0.1:8765 to remote 127.0.0.1:8765. No microphone was opened.');}
      if(health.bridge!=='ready')throw new Error('The page server responded but did not identify as the Swift voice bridge. Check the forwarded remote port.');
      if(health.voice_session==='active')throw new Error('A voice session is already active on this bridge. Use that browser tab, or end it before starting another.');
      if(!navigator.mediaDevices?.getUserMedia||!window.AudioContext)throw new Error('This browser cannot capture audio on the current origin. Use Edge or Chrome at the forwarded localhost URL.');
      voiceStartupStep('Waiting for mic…');
      let media;
      try{media=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:{ideal:true},noiseSuppression:{ideal:false},autoGainControl:{ideal:false}},video:false});}
      catch(error){throw new Error(microphoneFailure(error));}
      runtime.media=media;
      media.getAudioTracks()[0]?.addEventListener('ended',()=>{if(runtime.media===media){closeVoice();showNotice('The microphone device disconnected. Reconnect the USB input or select another browser input, then start a new conversation.','error');}});
      let context;
      voiceStartupStep('Starting audio…');
      try{context=new AudioContext({sampleRate:48000,latencyHint:'interactive'});runtime.context=context;await context.resume();}
      catch(_){throw new Error('The microphone opened, but browser audio could not start. Check site audio permission and retry from this tab.');}
      if(![16000,44100,48000].includes(context.sampleRate))throw new Error('Unsupported browser audio sample rate.');
      voiceStartupStep('Connecting voice…');
      const ws=new WebSocket(`ws://${location.host}/api/voice/ws`);runtime.ws=ws;ws.binaryType='arraybuffer';
      await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(new Error('Swift voice bridge did not accept the WebSocket. Check the forwarded port and bridge terminal.')),6000);
        ws.onopen=()=>{clearTimeout(timeout);resolve();};
        ws.onerror=()=>{clearTimeout(timeout);reject(new Error('The bridge port is reachable, but its voice WebSocket failed to open. Refresh the live page and check for another active session.'));};
        ws.onclose=()=>{clearTimeout(timeout);reject(new Error('The bridge closed before voice setup completed. Check its terminal for startup errors.'));};
      });
      runtime.aec=media.getAudioTracks()[0]?.getSettings()?.echoCancellation;runtime.ended=false;
      const input=context.createMediaStreamSource(media),processor=context.createScriptProcessor(1024,1,1),mute=context.createGain();mute.gain.value=0;
      runtime.input=input;runtime.processor=processor;runtime.mute=mute;
      processor.onaudioprocess=captureBlock;input.connect(processor);processor.connect(mute);mute.connect(context.destination);
      ws.onmessage=event=>{
        if(event.data instanceof ArrayBuffer){receiveAudio(event.data);return;}
        let message;try{message=JSON.parse(event.data);}catch(_){return;}
        if(message.type==='audio_stop'){stopPlayback();runtime.interrupting=false;return;}
        if(message.type==='ready'&&typeof message.run_id==='string'&&typeof message.session_id==='string'){
          if(runtime.startupTimer!==null){clearTimeout(runtime.startupTimer);runtime.startupTimer=null;}
          $('audio-diagnostic-capture').hidden=message.diagnostic_audio!==true;text('audio-diagnostic-capture',message.diagnostic_audio===true?'Audio diagnostics enabled: up to 60 seconds of received and ASR audio retained privately when this voice connection ends.':'');rememberSession(message.session_id);savedSessions.delete(message.session_id);runtime.runId=message.run_id;runtime.sessionId=message.session_id;runtime.wakeCueSessionId=null;runtime.turnId=0;runtime.completedTurnId=null;runtime.eventSequence=0;runtime.statusSequence=0;runtime.mode=message.mode;runtime.requestBudget=message.reasoning_budget||null;runtime.providerRoles=message.provider_roles||null;runtime.artifactsAvailable=message.task_artifact_access===true;runtime.deviceCapabilities=acceptDeviceCapabilities(message.devices);runtime.deviceActionPending=null;runtime.deviceActionMessage='';runtime.liveReport=null;runtime.connected=true;runtime.connecting=false;runtime.status='wake_listening';runtime.inputTurnOpen=null;runtime.interrupting=false;runtime.toolActive=false;runtime.music.reportedPlaybackStatus=null;runtime.music.reportedTrackKey=null;renderDeviceAdapterStatus();
          if(runtime.music.state&&!runtime.music.stateStale&&performance.now()-runtime.music.stateUpdatedAt<5000)reportMusicPlayback(runtime.music.state,runtime.music.state.source);
          data.reports=data.reports.filter(r=>r.kind!=='transcript');showNotice('');paint();$('voice-start').textContent='End conversation';$('voice-start').disabled=false;$('voice-start').setAttribute('aria-busy','false');$('voice-mute').hidden=false;text('voice-control-note','');voiceStatus();return;
        }
        if(message.type==='runtime_event'&&runtime.connected&&message.data?.run_id===runtime.runId&&message.sequence>runtime.eventSequence){
          runtime.eventSequence=message.sequence;
          if(['wake_detected_file','wake_injected_file','wake_injected'].includes(message.data.event)&&typeof message.data.session_id==='string'&&message.data.session_id!==runtime.sessionId){
            rememberSession(message.data.session_id);runtime.sessionId=message.data.session_id;runtime.turnId=0;runtime.completedTurnId=null;runtime.toolActive=false;runtime.liveReport=null;runtime.status='activating';data.reports=data.reports.filter(r=>r.kind!=='transcript');paint();
          }
          if(message.data.event==='wake_detected_file')playWakeCue(message.data.session_id);
          if(message.data.session_id!==runtime.sessionId)return;
          const eventTurn=message.data.turn_id;
          if(Number.isInteger(eventTurn)&&eventTurn<runtime.turnId)return;
          if(message.data.reasoning_budget)runtime.requestBudget=message.data.reasoning_budget;
          if(message.data.event==='llm_empty_response')showNotice('Reasoning returned no answer. Swift asked you to try again.');
          if(message.data.event==='turn_error'){
            if(message.data.error_code==='context_capacity')$('fresh-context').hidden=false;
            const service=message.data.error_stage==='tool'?'Task':({asr:'Speech recognition',tts:'Speech synthesis',llm:'Reasoning'}[message.data.error_role]||'Model');
            const messages={request_allowance_exhausted:'This test run has reached its configured model-request limit. No new request was sent. Review the run limit before continuing.',context_capacity:'Context is full. End this conversation, then use Start conversation for fresh context. Saved results remain available.',invalid_context:'Swift could not read the task history safely. Start a new conversation.',request_limit_reached:`${service} request allowance is exhausted for this bridge run. Review the authorized request limit before continuing.`,provider_rate_limited:`${service} is rate-limiting requests. Wait before trying again; Swift did not retry automatically.`,provider_unavailable:`${service} is temporarily unavailable. Any completed task remains recorded; no automatic retry was made.`,provider_timeout:`${service} timed out. Any completed task remains recorded; no automatic retry was made.`,provider_incomplete_response:`${service} response stream ended early. Check task results before retrying.`,turn_deadline_exceeded:'Swift reached the response deadline. Any completed task remains recorded; no automatic retry was made.',provider_rejected:`${service} rejected this request. Check its configuration; Swift did not retry automatically.`,provider_error:`${service} request failed. Any completed task remains recorded; no automatic retry was made.`,english_only_asr:'Speech recognition could not produce a usable English transcript. No task was started.',english_only_output:'Swift blocked text outside its English-script policy. Please ask again in English; no speech was played.',english_only_tts:'Speech synthesis blocked text outside its English-script policy; no audio was generated.'};
            const timeoutBoundary={response_headers:'while waiting for response headers',first_sse_event:'before the first streamed event',first_text_delta:'after streaming began but before assistant text',sse_stream:'while receiving the response stream'}[message.data.error_boundary];
            const parseDetails={invalid_sse_json:'a malformed stream event',incomplete_sse:'a response stream that ended early',invalid_event_shape:'an unexpected response structure',event_not_object:'a non-object response event',provider_error_event:'an error event from the reasoning service',missing_choices:'a response event with no choices',choices_not_array:'an invalid choices list',unexpected_choice_count:'an unexpected number of choices',choice_not_object:'an invalid choice item',invalid_choice_index:'an unexpected choice index',invalid_delta_shape:'an invalid response segment',invalid_tool_call_shape:'an invalid task-call structure',invalid_tool_arguments:'invalid task arguments',invalid_tool_identity:'an invalid task-call identifier',unadvertised_tool_name:'an unsupported task name',duplicate_tool_identity:'a duplicate task-call identifier',invalid_finish_reason:'an incomplete model response'};
            const parseDetail=parseDetails[message.data.error_detail];
            if(message.data.error_code==='music_outcome_uncertain'){
              // The task card is the durable, action-specific receipt. Avoid a
              // second generic warning that can contradict later player state.
              showNotice('');
            }else{
              const notice=message.data.error_code==='provider_invalid_response'&&parseDetail?`${service} returned ${parseDetail}. Check task results before retrying.`:message.data.error_code==='provider_invalid_response'?`${service} returned an invalid response. Check task results before retrying.`:message.data.error_code==='provider_timeout'&&timeoutBoundary?`${service} timed out ${timeoutBoundary}. Any completed task remains recorded; no automatic retry was made.`:messages[message.data.error_code]||'Swift could not complete this response. Any completed task remains recorded; no automatic retry was made.';
              showNotice(notice,'error');
            }
            if($('session-dialog').open)sessionDetails();
          }
          if(message.data.event==='playback_stopped_browser'){
            const hadPlayback=runtime.interrupting||runtime.status==='speaking'||runtime.sources.size>0;
            runtime.interrupting=false;
            // audio_stop owns queue clearing. This polled event can arrive
            // after newer audio; replaying the stop here loses its ACKs.
            if(hadPlayback){runtime.status='interrupted';voiceStatus();}
          }
          if(message.data.event==='turn_completed'&&Number.isInteger(eventTurn))runtime.completedTurnId=eventTurn;
          if(eventTurn===runtime.turnId&&message.data.event==='tool_started'){runtime.toolActive=true;voiceStatus();}
          if(eventTurn===runtime.turnId&&message.data.event==='tool_completed'){
            runtime.toolActive=false;
            if(message.data.tool_call_id===runtime.deviceActionPending){runtime.deviceActionPending=null;runtime.deviceActionMessage='';refreshTranscriptProjection();}
            voiceStatus();
          }
          if(message.data.event==='turn_error'&&runtime.deviceActionPending){runtime.deviceActionPending=null;runtime.deviceActionMessage='The action response failed. Check its task receipt before retrying.';refreshTranscriptProjection();}
          return;
        }
        if(message.type==='device_action_ack'){
          if(message.command_id!==runtime.deviceActionPending)return;
          if(message.status==='accepted'||message.status==='duplicate')runtime.deviceActionMessage='Command accepted by the voice session; waiting for its action receipt.';
          else{runtime.deviceActionPending=null;runtime.deviceActionMessage=String(message.message||'The device action could not be submitted.').slice(0,120);}
          refreshTranscriptProjection();return;
        }
        if(message.type==='status'&&runtime.connected&&message.run_id===runtime.runId&&message.session_id===runtime.sessionId&&['wake_listening','activating','listening','thinking','speaking'].includes(message.state)&&Number.isInteger(message.turn_id)&&message.turn_id>=runtime.turnId&&Number.isInteger(message.status_sequence)&&message.status_sequence>runtime.statusSequence&&message.event_sequence>=runtime.eventSequence&&!(runtime.interrupting&&['thinking','speaking'].includes(message.state))){
          runtime.turnId=message.turn_id;runtime.sessionId=message.session_id;runtime.statusSequence=message.status_sequence;runtime.status=message.state;runtime.engagement=message.engagement||null;runtime.inputTurnOpen=typeof message.input_open==='boolean'?message.input_open:null;
          if(message.state!=='thinking')runtime.toolActive=false;
          if(runtime.completedTurnId!==null&&message.turn_id>runtime.completedTurnId)runtime.completedTurnId=null;
          if(message.state==='listening'||message.state==='wake_listening')runtime.interrupting=false;
          voiceStatus();text('voice-control-note','');return;
        }
        if(message.type==='transcript'&&message.data){
          try{if(!runtime.connected||message.data.session_id!==runtime.sessionId||runtime.liveReport&&message.data.revision<=runtime.liveReport.data.revision)return;const report={id:'runtime-live',source:'Live browser transport',kind:'transcript',evidence:message.data.mode==='mock'?'mock':'HPC-live',modified_utc:message.data.updated_utc,sha256:'',data:message.data};
            validate({schema_version:1,reports:[report],warnings:[],generated_utc:message.data.updated_utc});
            runtime.liveReport=report;data.reports=[report,...data.reports.filter(r=>r.kind!=='transcript')].slice(0,221);paint();
          }catch(_){showNotice('Voice transcript update was invalid.','error');}
        }
        if(message.type==='error'){
          const reason=String(message.reason||'runtime setup failed').slice(0,120);
          if(!runtime.connected){closeVoice();showNotice(`Swift bridge accepted the connection but could not start the voice runtime (${reason}). Check the bridge terminal and wake assets.`, 'error');}
          else showNotice('Voice bridge: '+reason,'error');
        }
      };
      ws.onclose=()=>{if(runtime.ws===ws){const lost=runtime.connected||runtime.connecting;closeVoice();if(lost){showNotice('Voice bridge connection ended. Checking for recovery; the session will not resume and no command will be replayed.','error');void probeBridgeRecovery();}}};
      ws.send(JSON.stringify({type:'hello',rate:context.sampleRate,channels:1,conversation_mode:$('conversation-mode').value,aec_reported:typeof runtime.aec==='boolean'?runtime.aec:null}));
      runtime.startupTimer=setTimeout(()=>{if(runtime.ws===ws&&!runtime.connected){closeVoice();showNotice('Voice runtime did not become ready within 15 seconds. Check the bridge terminal for speech-service or wake-model startup errors.','error');}},15000);
      $('voice-interrupt').disabled=true;voiceStartupStep('Starting voice…');voiceStatus();
    }catch(error){closeVoice();showNotice(error.message||'Could not start voice.','error');}
  }
  $('voice-start').addEventListener('click',startVoice);
  $('fresh-context').addEventListener('click',()=>{if(!runtime.connected)return;closeVoice(true);showNotice('Conversation ended. Click Start conversation for fresh context. Saved results remain available.');});
  for(const container of [$('task-items'),$('active-task-items')]){
    container.addEventListener('input',event=>{
      if(event.target.matches('[data-light-brightness]')){const output=event.target.closest('[data-light-controls]')?.querySelector('[data-brightness-output]'),value=`${Number(event.target.value)}%`;if(output&&output.textContent!==value)output.textContent=value;}
      if(event.target.matches('[data-light-color]')){const output=event.target.closest('[data-light-controls]')?.querySelector('[data-color-output]'),value=safeRgbColor(event.target.value)||'Choose a color';if(output&&output.textContent!==value)output.textContent=value;}
    });
    container.addEventListener('click',event=>{
      const button=event.target.closest('[data-device-action]');
      if(button&&!button.disabled)sendDeviceAction(button);
    });
  }
  $('voice-interrupt').addEventListener('click',interruptVoice);
  $('voice-mute').addEventListener('click',()=>{if(!runtime.connected)return;runtime.micMuted=!runtime.micMuted;runtime.media?.getAudioTracks().forEach(track=>{track.enabled=!runtime.micMuted;});runtime.micLevel=0;runtime.micVisual=0;$('voice-mute').textContent=runtime.micMuted?'Unmute microphone':'Mute microphone';voiceStatus();});
  if(livePage)fetch('/api/voice/capabilities',{cache:'no-store',signal:AbortSignal.timeout(5000)}).then(response=>response.ok?response.json():null).then(capability=>{
    if(!capability?.available){runtime.music.capabilitiesError=true;runtime.music.connectRequested=false;renderMusic();return;}
    runtime.available=true;runtime.artifactsAvailable=capability.task_artifact_access===true;runtime.deviceCapabilities=acceptDeviceCapabilities(capability.devices);renderDeviceAdapterStatus();document.querySelector('.controls-unavailable').hidden=true;$('voice-controls').hidden=false;
    if(capability.music&&typeof capability.music==='object'){
      runtime.music.capabilities=capability.music;runtime.music.capabilitiesLoaded=true;
      if(runtime.music.capabilities.configured){if(runtime.music.capabilities.output_backend==='web_playback_sdk'&&runtime.music.capabilities.browser_player)loadSpotifySdk();void refreshMusicState(true);runtime.music.apiTimer=setInterval(()=>{if(!document.hidden)void refreshMusicState();},3000);runtime.music.uiTimer=setInterval(()=>{if(!document.hidden&&runtime.music.state?.status==='playing')renderMusic();},1000);}
      else runtime.music.message='Spotify is not configured in the running voice bridge. Restart it with ME3_MUSIC_CONFIG=var/cache/music.json.';
    }else{runtime.music.capabilitiesError=true;runtime.music.connectRequested=false;runtime.music.message='This voice bridge does not report Spotify setup. Restart the updated bridge.';}
    text('voice-control-note','');voiceStatus();
    transcript();renderMusic();
    if(runtime.music.connectRequested&&runtime.music.capabilitiesLoaded){if(runtime.music.capabilities?.configured&&runtime.music.capabilities.browser_player)loadSpotifySdk();else runtime.music.connectRequested=false;}
  }).catch(()=>{runtime.music.capabilitiesError=true;runtime.music.message='Swift could not read Spotify setup from the voice bridge. Check its connection and refresh.';renderMusic();});
  setInterval(()=>{if(!document.hidden&&(sourceMode==='server'||sourceMode==='folder'))refresh();},2000);
  paint();
})();
