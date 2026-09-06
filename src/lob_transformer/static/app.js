const $ = id => document.getElementById(id);
let connected = false, chatBusy = false, job = null, pollTimer, activeTask = 'text';
let splits = {train:null, validation:null, test:null};
const pct = value => `${(value * 100).toFixed(2)}%`;
async function api(path, body) {
  const response = await fetch(path, {method:body === undefined ? 'GET' : 'POST',
    headers:body === undefined ? {} : {'Content-Type':'application/json'},
    body:body === undefined ? undefined : JSON.stringify(body), signal:AbortSignal.timeout(120000)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
  return data;
}
function page(name) {
  if (!['chat','train','models'].includes(name)) name = 'chat';
  document.querySelectorAll('nav [data-page]').forEach(el => el.classList.toggle('active', el.dataset.page === name));
  for (const key of ['chat','train','models']) $(`${key}-page`).classList.toggle('hidden', key !== name);
  $('page-title').textContent = {chat:'模型测试', train:'模型训练', models:'模型管理'}[name];
  location.hash = name;
  if (name === 'models') refreshModels();
}
document.querySelectorAll('[data-page]').forEach(el => el.onclick = () => page(el.dataset.page));
function showModel(data) {
  connected = true; activeTask = data.task || 'text';
  $('connection').textContent = '● 已连接 · ' + data.model;
  $('model-name').textContent = data.model;
  $('model-task').textContent = activeTask === 'time' ? '中文时间转换' : '普通文本续写';
  $('model-size').textContent = `${data.vocab_size} / ${data.context_length}`;
  $('model-shape').textContent = `${data.layers} / ${data.dimensions}`;
  $('vocabulary').textContent = data.vocabulary.map(c => c === '\n' ? '↵' : c === ' ' ? '␣' : c).join(' ');
  $('token-control').classList.toggle('hidden', activeTask === 'time');
  $('time-examples').classList.toggle('hidden', activeTask !== 'time');
  $('prompt').placeholder = activeTask === 'time' ? '例如：下午三点半' : '输入训练词表中的文字';
  $('prompt-label').textContent = activeTask === 'time' ? '中文时间' : '提示词';
  $('test-hint').textContent = activeTask === 'time' ? '输入中文时间，模型输出 HH:MM；无需添加等号。' : '单轮续写 · 历史消息不自动拼接上下文';
  $('scope').textContent = activeTask === 'time' ? '支持上午一至十一点、中午十二点、下午一至六点、晚上七至十一点，搭配整、半或零至五十九分。暂不支持“两点”、日期和凌晨。模型可能出错，请查看测试报告。' : '这是字符级续写模型，词表之外的输入会被拒绝。';
  $('send').disabled = chatBusy; $('benchmark').disabled = chatBusy;
}
async function health() {
  try { showModel(await api('/health')); }
  catch { connected = false; $('connection').textContent = '连接失败，请检查本地服务'; $('send').disabled = true; $('benchmark').disabled = true; }
}
function message(role, text, meta = '') {
  const el = document.createElement('div'); el.className = 'message ' + role;
  const label = document.createElement('strong'); label.textContent = role === 'user' ? '你' : role === 'error' ? '测试失败' : `模型 · ${$('model-name').textContent}`;
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
  el.append(label, bubble);
  if (meta) {const note = document.createElement('div'); note.className = 'meta'; note.textContent = meta; el.append(note);}
  $('messages').append(el); $('messages').scrollTop = $('messages').scrollHeight;
  return el;
}
$('clear').onclick = () => $('messages').replaceChildren();
document.querySelectorAll('[data-prompt]').forEach(el => el.onclick = () => {$('prompt').value = el.dataset.prompt; $('prompt').focus();});
$('prompt').onkeydown = event => {if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); $('chat-form').requestSubmit();}};
$('chat-form').onsubmit = async event => {
  event.preventDefault(); if (chatBusy || !connected) return;
  if (activeTask !== 'time' && !$('tokens').reportValidity()) return;
  const prompt = $('prompt').value; const task = activeTask;
  chatBusy = true; $('send').disabled = true; $('benchmark').disabled = true; $('clear').disabled = true; $('empty')?.remove();
  message('user', prompt); const pending = message('assistant', '正在生成…');
  try {
    const data = await api(task === 'time' ? '/convert-time' : '/generate', task === 'time' ? {prompt, use_cache:$('use-cache').value === 'true'} : {prompt, tokens:Number($('tokens').value), use_cache:$('use-cache').value === 'true'});
    pending.remove(); message('assistant', data.completion || '（未生成新字符）', `${data.metrics.elapsed_ms.toFixed(2)} ms · ${data.metrics.tokens_per_second.toFixed(1)} 字符/秒 · 缓存${data.metrics.use_cache ? '开启' : '关闭'}`);
    $('prompt').value = '';
  } catch (error) {pending.remove(); message('error', error.message);}
  finally {chatBusy = false; $('send').disabled = !connected; $('benchmark').disabled = !connected; $('clear').disabled = false;}
};
$('benchmark').onclick = async () => {
  if (chatBusy || !connected) return;
  if (!$('prompt').reportValidity() || (activeTask !== 'time' && !$('tokens').reportValidity())) return;
  chatBusy = true; $('send').disabled = true; $('benchmark').disabled = true;
  $('benchmark-result').textContent = '正在比较两种推理方式…';
  try {
    const result = await api('/benchmark', {prompt:$('prompt').value, tokens:Number($('tokens').value)});
    $('benchmark-result').textContent = `输出${result.identical ? '完全一致' : '不一致，请检查'}。关闭：${result.uncached.elapsed_ms.toFixed(2)} ms / ${result.uncached.tokens_per_second.toFixed(1)} 字符/秒；开启：${result.cached.elapsed_ms.toFixed(2)} ms / ${result.cached.tokens_per_second.toFixed(1)} 字符/秒。速度比 ${result.speedup.toFixed(2)}×（大于 1 表示加速）。窗口重建 ${result.window_rebuilds} 次。`;
  } catch (error) {$('benchmark-result').textContent = error.message;}
  finally {chatBusy = false; $('send').disabled = !connected; $('benchmark').disabled = !connected;}
};
function mode() {
  const time = $('task').value === 'time';
  $('time-data').classList.toggle('hidden', !time); $('text-data').classList.toggle('hidden', time);
  $('time-data').querySelectorAll('input').forEach(el => el.disabled = !time);
  $('text-data').querySelectorAll('input,select,textarea').forEach(el => el.disabled = time);
  $('corpus').required = !time;
}
$('task').onchange = mode;
function updateData() {
  for (const key of Object.keys(splits)) $(`${key}-count`).textContent = splits[key] ? `${splits[key].length} 条样例` : '尚未导入';
  document.querySelectorAll('[data-download]').forEach(el => el.disabled = !splits[el.dataset.download]);
  $('data-preview').textContent = splits.train ? splits.train.slice(0,3).map(r => `${r.input} → ${r.output}`).join('\n') : '点击“生成示例语料”，或上传三份 JSONL。';
}
function download(name, content, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('builtin').onclick = async () => {
  $('builtin').disabled = true;
  try {splits = await api('/time-data'); updateData(); $('train-error').textContent = '';}
  catch (error) {$('train-error').textContent = error.message;}
  finally {$('builtin').disabled = false;}
};
document.querySelectorAll('[data-download]').forEach(el => el.onclick = () => {
  const name = el.dataset.download;
  if (splits[name]) download(`${name}.jsonl`, splits[name].map(r => JSON.stringify(r)).join('\n')+'\n', 'application/x-ndjson');
});
async function parseJSONL(file) {
  if (!file || !file.name.toLowerCase().endsWith('.jsonl')) throw Error('请选择 .jsonl 文件');
  if (file.size > 2*1024*1024) throw Error('单份语料不能超过 2 MiB');
  const text = new TextDecoder('utf-8', {fatal:true}).decode(await file.arrayBuffer());
  const rows = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    let row; try {row = JSON.parse(line);} catch {throw Error(`${file.name} 第 ${index+1} 行不是有效 JSON`);}
    if (!row || typeof row.input !== 'string' || typeof row.output !== 'string') throw Error(`${file.name} 第 ${index+1} 行需要 input 和 output 文本字段`);
    rows.push(row);
  }
  if (!rows.length) throw Error('语料为空');
  return rows;
}
for (const key of Object.keys(splits)) $(`${key}-file`).onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  try {splits[key] = await parseJSONL(file); $('train-error').textContent = '';}
  catch (error) {splits[key] = null; $('train-error').textContent = error.message;}
  updateData();
};
$('dataset-folder').onchange = async event => {
  if (!event.target.files.length) return;
  try {
    const result = {};
    for (const key of Object.keys(splits)) {
      const files = Array.from(event.target.files).filter(file => file.name === `${key}.jsonl`);
      if (files.length !== 1) throw Error(`文件夹需要且只能有一份 ${key}.jsonl`);
      result[key] = await parseJSONL(files[0]);
    }
    splits = result; updateData(); $('train-error').textContent = '';
  } catch (error) {$('train-error').textContent = error.message;}
};
function count() {const chars = Array.from($('corpus').value); $('corpus-count').textContent = `${chars.length} 字符 · ${new Set(chars).size} 种字符`;}
$('corpus').oninput = count;
$('example').onclick = () => {$('corpus').value = '你好，世界。今天我们一起学习。\n模型读取字符，预测下一个字符。\n'.repeat(8); count();};
async function importFiles(fileList) {
  const all = Array.from(fileList);
  const files = all.filter(file => !(file.webkitRelativePath || file.name).split('/').some(part => part.startsWith('.')) && (/\.(txt|text|md|markdown|csv|tsv|json|jsonl|xml|html?|ya?ml|log)$/i.test(file.name) || file.type.startsWith('text/'))).sort((a,b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, 'zh-CN'));
  if (!files.length) throw Error('没有找到文本文件');
  if (files.some(file => /^(train|validation|test)\.jsonl$/i.test(file.name))) throw Error('这套配对语料请切换到“中文时间转换”模式分别导入，避免把测试答案用于训练。');
  if (files.reduce((n,f) => n+f.size,0) > 8*1024*1024) throw Error('文本文件总大小不能超过 8 MiB');
  const texts = [];
  for (const file of files) texts.push(new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer()));
  const text = texts.join('\n'); if (Array.from(text).length > 1000000) throw Error('语料不能超过 100 万字符');
  $('corpus').value = text; count(); $('import-summary').textContent = `已导入 ${files.length} 个文本文件，跳过 ${all.length-files.length} 个其他文件`;
}
for (const id of ['file','folder']) $(id).onchange = async () => {
  if (!$(id).files.length) return;
  try {await importFiles($(id).files); $('train-error').textContent = '';}
  catch (error) {$('train-error').textContent = error.message;}
};
function renderReport(container, report) {
  container.replaceChildren(); if (!report?.test) return;
  const title = document.createElement('h2'); const passed = report.test.accuracy >= .95;
  title.className = passed ? 'success' : 'warning';
  title.textContent = `${passed ? '达到 95% 目标' : '尚未达到 95% 目标'} · 测试准确率 ${pct(report.test.accuracy)}`;
  const summary = document.createElement('p');
  summary.textContent = `测试正确 ${report.test.correct} / ${report.test.total} 条${report.validation ? `；验证准确率 ${pct(report.validation.accuracy)}` : ''}${report.best_step ? `；最佳步数 ${report.best_step}` : ''}。这是当前测试集的成绩，不保证每条输入正确。`;
  container.append(title, summary);
  if (report.test.failures.length) {
    const table = document.createElement('table'); table.className = 'report-table';
    const head = document.createElement('tr');
    for (const label of ['输入','正确答案','模型输出']) {const th = document.createElement('th'); th.textContent = label; head.append(th);}
    table.append(head);
    for (const row of report.test.failures.slice(0,30)) {
      const tr = document.createElement('tr');
      for (const text of [row.input,row.output,row.actual]) {const td = document.createElement('td'); td.textContent = text; tr.append(td);}
      table.append(tr);
    }
    container.append(table);
    if (report.test.failures.length > 30) {const note = document.createElement('p'); note.textContent = '显示前 30 条失败案例，完整结果请下载报告。'; container.append(note);}
  }
  const button = document.createElement('button'); button.className = 'button'; button.textContent = '下载这份评估报告';
  button.onclick = () => download('report.json', JSON.stringify(report,null,2)); container.append(button);
}
function renderJob(data) {
  const previousStatus = job?.status; job = data;
  const running = data.status === 'running', time = data.task === 'time';
  $('train-button').disabled = running; $('stop').disabled = !running || data.stopping;
  $('stop').textContent = data.stopping ? '正在停止…' : '停止训练';
  $('train-status').textContent = {idle:'尚未开始',running:'训练中',completed:'训练完成',failed:'训练失败',cancelled:'已停止'}[data.status] || data.status;
  $('activate').disabled = data.status !== 'completed' || data.activated === true;
  $('activate').textContent = data.activated ? '已启用此模型' : '启用模型并测试';
  $('step-value').textContent = data.steps ? `${data.step} / ${data.steps}` : '—';
  $('progress').value = data.status === 'completed' ? 100 : data.steps ? data.step/data.steps*100 : 0;
  $('loss-label').textContent = time ? '当前训练批次损失' : '固定训练样本损失';
  $('val-label').textContent = time ? '验证准确率' : '验证样本损失';
  const loss = time ? data.train_batch_loss : data.loss;
  $('loss-value').textContent = loss === undefined ? '—' : loss.toFixed(4);
  $('val-value').textContent = time ? (data.validation_accuracy === undefined ? '待评估' : pct(data.validation_accuracy)) : (data.val_loss === undefined ? '—' : data.val_loss.toFixed(4));
  $('elapsed').textContent = data.elapsed ?? '—';
  const rows = data.history || [];
  $('log').textContent = rows.map(r => time ? `step ${r.step}  loss ${r.train_batch_loss.toFixed(5)}${r.validation_accuracy === undefined ? '' : `  验证 ${pct(r.validation_accuracy)}`}` : `step ${r.step}  train ${r.loss.toFixed(5)}  val ${r.val_loss.toFixed(5)}`).join('\n') || (running ? '初始化模型…' : '等待训练任务…');
  if (data.error) $('log').textContent += '\n' + data.error;
  if (data.comparison) $('log').textContent += `\n最佳步数 ${data.best_step}\n训练前：${data.comparison.before}\n最佳模型：${data.comparison.after}`;
  $('log').scrollTop = $('log').scrollHeight;
  $('checkpoint').textContent = data.checkpoint ? '已自动保存，可在模型管理中下载。' : '';
  $('chart-note').textContent = time ? '橙线：验证准确率，纵轴固定 0～100%。每 100 步评估；每 10 步刷新训练损失。' : '绿线：训练损失；橙线：验证损失。纵轴按两条曲线的共同范围缩放。';
  if (time) {
    $('curve').setAttribute('points','');
    $('val-curve').setAttribute('points', rows.filter(r => r.validation_accuracy !== undefined).map(r => `${10+r.step/Math.max(1,data.step)*380},${135-r.validation_accuracy*120}`).join(' '));
  } else {
    const values = rows.flatMap(r => [r.loss,r.val_loss]), min = Math.min(...values), max = Math.max(...values);
    for (const [id,key] of [['curve','loss'],['val-curve','val_loss']]) $(id).setAttribute('points', rows.map(r => `${10+r.step/Math.max(1,data.step)*380},${135-(r[key]-min)/Math.max(.000001,max-min)*120}`).join(' '));
  }
  if (data.status === 'completed' && previousStatus !== 'completed') refreshModels();
  if (data.report && !$('training-report').dataset.rendered) {renderReport($('training-report'), data.report); $('training-report').dataset.rendered = 'yes';}
  if (data.status !== 'completed') {$('training-report').replaceChildren(); delete $('training-report').dataset.rendered;}
}
async function poll() {
  clearTimeout(pollTimer);
  try {renderJob(await api('/training'));} catch {$('train-status').textContent = '状态连接失败，稍后重试';}
  pollTimer = setTimeout(poll, job?.status === 'running' ? 1000 : 5000);
}
$('retry').onclick = () => {health(); poll();};
$('train-form').onsubmit = async event => {
  event.preventDefault(); $('train-error').textContent = ''; $('train-button').disabled = true;
  try {
    let request;
    if ($('task').value === 'time') {
      if (Object.values(splits).some(rows => !rows?.length)) throw Error('请先生成示例语料，或分别上传训练、验证和测试集。');
      request = {task:'time',steps:Number($('time-steps').value),splits};
    } else {
      request = {task:'text',text:$('corpus').value};
      for (const key of ['steps','batch_size','context_length','learning_rate','dimensions','heads','layers']) request[key] = Number($(key).value);
    }
    renderJob(await api('/train',request)); poll();
  } catch (error) {$('train-error').textContent = error.message; $('train-button').disabled = job?.status === 'running';}
};
$('stop').onclick = async () => {try {renderJob(await api('/stop',{}));} catch (error) {$('train-error').textContent = error.message;}};
$('activate').onclick = async () => {
  if (!job) return; $('activate').disabled = true;
  try {showModel(await api('/activate',{id:job.id})); $('messages').replaceChildren(); $('prompt').value = activeTask === 'time' ? '下午三点半' : ''; page('chat'); poll();}
  catch (error) {$('train-error').textContent = error.message; $('activate').disabled = false;}
};
async function refreshModels() {
  try {
    const models = await api('/models'); $('models-list').replaceChildren();
    for (const model of models) {
      const card = document.createElement('div'); card.className = 'panel model-card';
      const title = document.createElement('b'); title.textContent = model.name;
      const note = document.createElement('div'); note.className = 'muted'; note.textContent = `${model.task === 'time' ? '中文时间转换' : '普通文本续写'}${model.active ? ' · 当前使用' : ''}`;
      const actions = document.createElement('div'); actions.className = 'actions';
      const use = document.createElement('button'); use.className = 'button primary'; use.textContent = model.active ? '当前使用 · 去测试' : '启用并测试';
      use.onclick = async () => {
        use.disabled = true;
        try {showModel(await api('/select-model',{id:model.id})); $('messages').replaceChildren(); $('prompt').value = activeTask === 'time' ? '下午三点半' : ''; page('chat'); refreshModels();}
        catch (error) {$('model-error').textContent = error.message; use.disabled = false;}
      };
      actions.append(use);
      const link = document.createElement('a'); link.className = 'button'; link.href = `/artifacts/${model.id}/model`; link.textContent = '下载模型'; actions.append(link);
      if (model.has_report) {
        const report = document.createElement('button'); report.className = 'button'; report.textContent = '查看报告';
        report.onclick = async () => {try {const data = await api(`/artifacts/${model.id}/report`); $('model-report').classList.remove('hidden'); renderReport($('model-report'),data); $('model-report').scrollIntoView({behavior:'smooth'});} catch(error) {$('model-error').textContent = error.message;}};
        const evaluate = document.createElement('button'); evaluate.className = 'button'; evaluate.textContent = '重新评估原测试集';
        evaluate.onclick = async () => {evaluate.disabled = true; evaluate.textContent = '评估中…'; try {const data = await api('/evaluate',{id:model.id}); $('model-report').classList.remove('hidden'); renderReport($('model-report'),data);} catch(error) {$('model-error').textContent = error.message;} finally {evaluate.disabled = false; evaluate.textContent = '重新评估原测试集';}};
        actions.append(report,evaluate);
        for (const [kind,label] of [['train','训练集'],['validation','验证集'],['test','测试集']]) {const a = document.createElement('a'); a.className = 'button'; a.href = `/artifacts/${model.id}/${kind}`; a.textContent = `下载${label}`; actions.append(a);}
      }
      card.append(title,note,actions); $('models-list').append(card);
    }
  } catch (error) {$('model-error').textContent = error.message;}
}
$('refresh-models').onclick = refreshModels;
$('upload-model').onclick = async () => {
  $('upload-model').disabled = true; $('model-error').textContent = '';
  try {
    const file = $('model-file').files[0];
    if (!file || !file.name.endsWith('.npz')) throw Error('请选择 .npz 模型文件');
    if (file.size > 8*1024*1024) throw Error('模型不能超过 8 MiB');
    const bytes = new Uint8Array(await file.arrayBuffer()); let binary = '';
    for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i,i+8192));
    await api('/upload-model',{task:$('upload-task').value,data:btoa(binary)});
    $('model-file').value = ''; await refreshModels();
  } catch (error) {$('model-error').textContent = error.message;}
  finally {$('upload-model').disabled = false;}
};
mode(); updateData(); page(location.hash.slice(1) || 'chat'); health(); poll();
