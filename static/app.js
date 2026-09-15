'use strict';
const $=id=>document.getElementById(id), video=$('camera'), canvas=$('output'), ctx=canvas.getContext('2d');
const capture=document.createElement('canvas'), cctx=capture.getContext('2d');
let stream=null,token=null,running=false,generation=0,controller=null;
const messages={'Move closer':'Đưa mặt lại gần hơn','Center your full face':'Đưa toàn bộ mặt vào giữa khung','Face camera directly':'Nhìn thẳng vào camera','Need more light':'Cần thêm ánh sáng','Reduce glare':'Giảm ánh sáng chói','Hold still / improve focus':'Giữ yên hoặc điều chỉnh độ nét'};
function status(title,detail){$('status').textContent=title;$('detail').textContent=detail;}
async function session(){const r=await fetch('/api/session',{method:'POST'});const data=await r.json();if(!r.ok)throw Error(data.error||'Không kết nối được máy chủ');token=data.token;}
function stop(){generation++;running=false;controller?.abort();if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;if(token)fetch('/api/session',{method:'DELETE',headers:{'X-Session-Token':token},keepalive:true}).catch(()=>{});token=null;$('start').disabled=false;$('stop').disabled=true;$('device').disabled=false;$('empty').hidden=false;$('dot').classList.remove('on');$('live').textContent='ĐÃ DỪNG CAMERA';ctx.clearRect(0,0,canvas.width,canvas.height);$('count').textContent='—';$('latency').textContent='—';status('Đã dừng','Camera đã tắt. Bạn có thể bắt đầu phiên mới.');}
async function loop(id){
 while(running&&generation===id){
  const t=performance.now();
  try{
   const scale=Math.min(1,640/video.videoWidth);capture.width=Math.round(video.videoWidth*scale);capture.height=Math.round(video.videoHeight*scale);
   cctx.save();cctx.translate(capture.width,0);cctx.scale(-1,1);cctx.drawImage(video,0,0,capture.width,capture.height);cctx.restore();
   const blob=await new Promise(resolve=>capture.toBlob(resolve,'image/jpeg',.8));
   if(!running||generation!==id)break;
   controller=new AbortController();const timer=setTimeout(()=>controller.abort(),15000);let r;
   try{r=await fetch('/api/frame',{method:'POST',headers:{'Content-Type':'image/jpeg','X-Session-Token':token},body:blob,signal:controller.signal});}finally{clearTimeout(timer);}
   if(!running||generation!==id)break;
   if(r.status===401){await session();continue;}
   if(r.status===429){status('Máy chủ đang bận','Đang thử lại, bạn giữ khuôn mặt trong khung.');await new Promise(r=>setTimeout(r,350));continue;}
   const data=await r.json();if(!r.ok)throw Error(data.error||'Lỗi nhận diện');
   canvas.width=capture.width;canvas.height=capture.height;ctx.drawImage(capture,0,0);
   for(const f of data.faces){const [x,y,r,b]=f.box;const color=f.status==='real'?'#47d99b':f.status==='fake'?'#ff727b':'#f2c464';ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(x,y,r-x,b-y);let text=messages[f.quality]||({real:'Dự đoán thật',fake:'Dự đoán giả',checking:'Đang kiểm tra',uncertain:'Chưa chắc chắn'}[f.status]||'Điều chỉnh');if(['real','fake'].includes(f.status))text+=` · ${f.scores[f.status].toFixed(3)}`;ctx.font='13px sans-serif';const width=ctx.measureText(text).width+10;const tx=Math.max(0,Math.min(x,canvas.width-width)),ty=Math.max(23,y);ctx.fillStyle='#101a29';ctx.fillRect(tx,ty-23,width,21);ctx.fillStyle=color;ctx.fillText(text,tx+5,ty-8);}
   $('count').textContent=data.faces.length;$('latency').textContent=data.processing_ms+' ms';
   const states=data.faces.map(f=>f.status);
   if(!states.length)status('Chưa thấy khuôn mặt','Đưa toàn bộ mặt vào khung, nhìn thẳng và tránh che mặt.');
   else if(states.includes('fake'))status('Phát hiện dấu hiệu giả','Model dự đoán giả mạo. Đây là kết quả demo, cần kiểm chứng.');
   else if(states.every(s=>s==='real'))status('Dự đoán người thật','Kết quả nhất quán qua nhiều khung hình. Điểm không bảo đảm đúng.');
   else status('Đang kiểm tra',messages[data.faces.find(f=>f.quality)?.quality]||'Giữ yên và chờ các khung hình nhất quán.');
  }catch(e){if(generation!==id)break;stop();status('Kết nối bị gián đoạn','Camera đã tắt. Bấm Bật camera để thử lại.');break;}
  await new Promise(r=>setTimeout(r,Math.max(0,180-(performance.now()-t))));
 }
}
$('start').onclick=async()=>{
 $('start').disabled=true;const id=++generation;
 try{
  if(!navigator.mediaDevices?.getUserMedia)throw Error('Trình duyệt cần HTTPS và hỗ trợ camera.');
  const device=$('device').value;
  stream=await navigator.mediaDevices.getUserMedia({video:device?{deviceId:{exact:device},width:{ideal:640},height:{ideal:480}}:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
  video.srcObject=stream;await video.play();
  const devices=await navigator.mediaDevices.enumerateDevices();$('device').replaceChildren(new Option('Camera mặc định',''));devices.filter(d=>d.kind==='videoinput').forEach((d,i)=>$('device').add(new Option(d.label||`Camera ${i+1}`,d.deviceId)));$('device').value=stream.getVideoTracks()[0].getSettings().deviceId||'';
  status('Đang kết nối','Máy chủ miễn phí có thể cần thời gian khởi động.');await session();
  if(generation!==id){stop();return;}
  running=true;$('stop').disabled=false;$('device').disabled=true;$('empty').hidden=true;$('live').textContent='CAMERA ĐANG BẬT';$('dot').classList.add('on');loop(id);
 }catch(e){stop();status('Không thể bắt đầu',e.name==='NotAllowedError'?'Bạn chưa cấp quyền camera. Hãy cho phép rồi thử lại.':e.message);}
};
$('stop').onclick=stop;window.addEventListener('pagehide',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden&&stream)stop();});
