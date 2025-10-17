// Uyarlanmış frontend için script
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendMessage(text, sender='bot'){
  const container = document.createElement('div');
  container.className = 'message ' + (sender==='user' ? 'user' : 'bot');

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  // use textContent for user input to avoid XSS; bot content comes from trusted source
  if(sender === 'user') bubble.textContent = text;
  else bubble.textContent = text;

  container.appendChild(bubble);
  messagesEl.appendChild(container);
  scrollToBottom();
}

async function sendQuestion(){
  const q = inputEl.value.trim();
  if(!q) return;
  appendMessage(q, 'user');
  inputEl.value = '';
  inputEl.focus();
  sendBtn.disabled = true;
  try{
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q, chat_history: []})
    });
    if(!resp.ok){
      appendMessage('Sunucu hatası: ' + resp.statusText, 'bot');
      return;
    }
    const data = await resp.json();
    appendMessage(data.answer || 'Cevap yok', 'bot');
  }catch(err){
    console.error(err);
    appendMessage('Sunucuya ulaşılamıyor', 'bot');
  }finally{
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', sendQuestion);
inputEl.addEventListener('keydown', (e)=>{ if(e.key === 'Enter'){ e.preventDefault(); sendQuestion(); }});


