'use strict';

const $ = (id) => document.getElementById(id);
const UNKNOWN_COLOR = '#E2711D';
const KNOWN_COLOR = '#5CC8FF';
const REAL_COLOR = '#00E676';
const FAKE_COLOR = '#FF3B30';

const state = {
  view: 'live',
  info: null,
  threshold: 0.55,
  paused: false,
  logOn: true,
  livenessOn: true, // Mặc định BẬT chống giả mạo
  selectedDeviceId: '', // Mặc định luôn ưu tiên camera máy (Laptop Webcam)
  lastFaces: [],
  ws: null,
  busy: false,
  lastReply: 0,
  fps: 0,
  enroll: { step: 1, name: '', session: null, clips: [], summary: null },
  reportT: null,
};

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'style') Object.assign(node.style, value);
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

const pct = (x) => (x * 100).toFixed(1) + '%';

// ---- Điều hướng ----------------------------------------------------------------------------

function showView() {
  const view = (location.hash || '#live').slice(1);
  state.view = ['live', 'enroll', 'manage', 'report'].includes(view) ? view : 'live';
  for (const name of ['live', 'enroll', 'manage', 'report']) $('view-' + name).hidden = name !== state.view;
  document.querySelectorAll('.tabs a').forEach((a) => {
    if (a.dataset.view === state.view) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  if (state.view === 'manage') loadManageView();
  if (state.view === 'report') loadInfo();
}
window.addEventListener('hashchange', showView);

// ---- Thông tin chung -------------------------------------------------------------------------

async function loadInfo() {
  const response = await fetch('/api/info');
  state.info = await response.json();
  const info = state.info;
  $('chip-provider').textContent = (info.providers[0] || 'CPU').replace('ExecutionProvider', '') + ' · ONNX Runtime';
  $('chip-people').textContent = info.people.length + ' người đã đăng ký';
  if (state.reportT === null) {
    state.threshold = info.threshold;
    state.reportT = Math.round(info.threshold * 20) / 20;
    $('thr').value = info.threshold.toFixed(2);
  }
  $('people').replaceChildren(...info.people.map((p) => el('span', { text: p })));
  renderThreshold();
  renderLog(info.log);
  renderReport();
}

function calibrationAt(t) {
  const rows = (state.info && state.info.report.calibration) || [];
  let best = null;
  for (const row of rows) if (!best || Math.abs(row.t - t) < Math.abs(best.t - t)) best = row;
  return best;
}

// ---- 1. Nhận diện trực tiếp & Quản lý Camera ------------------------------------------------

let stream = null;
let isSwitchingCamera = false;

function isSCM200Camera(rawLabel = '') {
  const l = (rawLabel || '').toLowerCase();
  // UVC Camera 0 là camera ngoài SC-M200 cắm qua cổng USB
  return /uvc\s*camera\s*0|sc[-_]?m200|m200/i.test(l);
}

function isLaptopCamera(rawLabel = '') {
  const l = (rawLabel || '').toLowerCase();
  // USB2.0 HD UVC WebCam là camera máy tích hợp sẵn của laptop
  return /usb2\.0\s*hd|integrated|built[-_]?in|internal|laptop|facetime|easycamera/i.test(l);
}

function formatCameraLabel(rawLabel = '', index = 0) {
  if (isSCM200Camera(rawLabel)) {
    return `📷 Camera ngoài SC-M200 (USB rời)`;
  }
  if (isLaptopCamera(rawLabel)) {
    return `💻 Camera máy (Laptop Webcam)`;
  }
  return `📷 ${rawLabel || `Camera ${index + 1}`}`;
}

async function populateCameraList(activeDeviceId = '') {
  const select = $('camera-select');
  if (!select) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');

    select.innerHTML = '';
    if (videoDevices.length === 0) {
      select.innerHTML = '<option value="">Không tìm thấy camera</option>';
      return;
    }

    let defaultLaptopId = '';
    let scM200Id = '';

    videoDevices.forEach((device, index) => {
      const opt = document.createElement('option');
      opt.value = device.deviceId;
      const rawLabel = device.label || `Camera ${index + 1}`;
      opt.textContent = formatCameraLabel(rawLabel, index);

      if (isSCM200Camera(rawLabel)) {
        scM200Id = device.deviceId;
      } else if (isLaptopCamera(rawLabel)) {
        if (!defaultLaptopId) defaultLaptopId = device.deviceId;
      }

      select.appendChild(opt);
    });

    let chosenId = activeDeviceId || state.selectedDeviceId || localStorage.getItem('faceid_camera_id_v2');
    const exists = videoDevices.some((d) => d.deviceId === chosenId);
    if (!chosenId || !exists) {
      chosenId = defaultLaptopId || (videoDevices[0] ? videoDevices[0].deviceId : '');
    }

    if (chosenId) {
      select.value = chosenId;
      state.selectedDeviceId = chosenId;
      localStorage.setItem('faceid_camera_id_v2', chosenId);
    }
  } catch (err) {
    console.warn('Không thể liệt kê thiết bị camera:', err);
  }
}

async function startCamera(preferredDeviceId = null) {
  if (isSwitchingCamera) return;
  isSwitchingCamera = true;

  try {
    // 1. Dừng hoàn toàn stream cũ trước khi xin stream mới
    if (stream) {
      try {
        stream.getTracks().forEach((track) => track.stop());
      } catch (e) {
        console.warn('Lỗi khi dừng track:', e);
      }
      stream = null;
      // Tránh nghẽn phần cứng DirectShow/MediaFoundation trên Windows khi chuyển đổi camera USB
      await new Promise((resolve) => setTimeout(resolve, 200));
    }

    // 2. Xác định camera mục tiêu
    let targetDeviceId = preferredDeviceId || state.selectedDeviceId || localStorage.getItem('faceid_camera_id_v2') || '';

    // MẶC ĐỊNH: Nếu chưa chọn camera, chủ động ưu tiên tìm camera máy trước
    if (!targetDeviceId) {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((d) => d.kind === 'videoinput');
        const laptopCam = videoDevices.find((d) => isLaptopCamera(d.label));
        if (laptopCam) {
          targetDeviceId = laptopCam.deviceId;
        } else if (videoDevices.length > 0) {
          targetDeviceId = videoDevices[0].deviceId;
        }
      } catch (e) {
        // Tiếp tục bình thường nếu chưa enumerate được
      }
    }

    // 3. Sử dụng exact constraint khi người dùng chỉ định đích danh thiết bị
    const exactConstraints = targetDeviceId
      ? {
          audio: false,
          video: {
            deviceId: { exact: targetDeviceId },
            width: { ideal: 1920, max: 1920 },
            height: { ideal: 1080, max: 1080 },
          },
        }
      : {
          audio: false,
          video: {
            width: { ideal: 1280, max: 1920 },
            height: { ideal: 720, max: 1080 },
          },
        };

    try {
      stream = await navigator.mediaDevices.getUserMedia(exactConstraints);
    } catch (exactErr) {
      console.warn('Exact device constraint không thành công, thử ideal fallback:', exactErr);
      const idealConstraints = targetDeviceId
        ? {
            audio: false,
            video: {
              deviceId: { ideal: targetDeviceId },
              width: { ideal: 1280, max: 1920 },
              height: { ideal: 720, max: 1080 },
            },
          }
        : { audio: false, video: true };

      try {
        stream = await navigator.mediaDevices.getUserMedia(idealConstraints);
      } catch (idealErr) {
        console.warn('Ideal constraint thất bại, mở camera mặc định:', idealErr);
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      }
    }

    const video = $('live-video');
    const enrollVideo = $('enroll-video');

    if (video) video.srcObject = stream;
    if (enrollVideo) enrollVideo.srcObject = stream;

    // Xử lý hiển thị ngay khi có video
    const onReady = () => {
      $('live-placeholder').hidden = true;
      if (video && video.videoWidth && video.videoHeight) {
        $('live-cam').style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
      }
    };

    if (video) {
      video.play().catch(() => {});
      if (video.readyState >= 1) onReady();
      else {
        video.onloadedmetadata = onReady;
        video.onplaying = onReady;
      }
    }

    const currentTrack = stream.getVideoTracks()[0];
    let actualDeviceId = '';
    if (currentTrack) {
      const settings = currentTrack.getSettings();
      if (settings.deviceId) {
        actualDeviceId = settings.deviceId;
        state.selectedDeviceId = actualDeviceId;
        localStorage.setItem('faceid_camera_id_v2', actualDeviceId);
      }
    }

    await populateCameraList(actualDeviceId || targetDeviceId);
  } catch (error) {
    console.error('Không thể mở camera:', error);
    $('live-placeholder').hidden = false;
    $('live-placeholder').textContent =
      'Không thể mở camera: ' + error.message + '. Vui lòng cho phép quyền truy cập camera trong trình duyệt hoặc cắm lại cáp USB.';
  } finally {
    isSwitchingCamera = false;
  }
}

// Bắt sự kiện người dùng chọn camera từ Dropdown
$('camera-select').addEventListener('change', async (event) => {
  const chosenId = event.target.value;
  if (!chosenId) return;
  state.selectedDeviceId = chosenId;
  localStorage.setItem('faceid_camera_id_v2', chosenId);
  await startCamera(chosenId);
});

// Bắt sự kiện nhấn nút "Đổi Camera" 🔄 để chuyển đổi luân phiên
const btnSwitchCam = $('btn-switch-cam');
if (btnSwitchCam) {
  btnSwitchCam.addEventListener('click', async () => {
    const select = $('camera-select');
    if (!select || select.options.length <= 1) {
      alert('Chỉ có 1 camera được nhận diện trên máy tính.');
      return;
    }
    const nextIndex = (select.selectedIndex + 1) % select.options.length;
    select.selectedIndex = nextIndex;
    const nextDeviceId = select.options[nextIndex].value;
    if (nextDeviceId) {
      state.selectedDeviceId = nextDeviceId;
      localStorage.setItem('faceid_camera_id_v2', nextDeviceId);
      btnSwitchCam.disabled = true;
      btnSwitchCam.textContent = '⏳ Đang đổi...';
      try {
        await startCamera(nextDeviceId);
      } finally {
        btnSwitchCam.disabled = false;
        btnSwitchCam.textContent = '🔄 Đổi Camera';
      }
    }
  });
}

if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', async () => {
    await populateCameraList(state.selectedDeviceId);
  });
}

function grabCanvas(video) {
  const canvas = $('grab');
  const width = Math.min(960, video.videoWidth);
  canvas.width = width;
  canvas.height = Math.round((width * video.videoHeight) / video.videoWidth);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function connect() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    state.busy = false;
    sendSettings();
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    state.busy = false;
    const now = performance.now();
    if (state.lastReply) state.fps = state.fps * 0.8 + (1000 / (now - state.lastReply)) * 0.2;
    state.lastReply = now;
    $('live-ms').textContent = data.ms + ' ms/frame';
    $('chip-fps').textContent = state.fps.toFixed(1) + ' FPS';
    renderFaces(data.faces);
    renderLog(data.log);
    pump();
  };
  ws.onclose = () => {
    state.busy = false;
    setTimeout(connect, 1000);
  };
  state.ws = ws;
}

function sendSettings() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(
      JSON.stringify({
        threshold: state.threshold,
        log: state.logOn,
        liveness: state.livenessOn,
      })
    );
  }
}

function pump() {
  const video = $('live-video');
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN || state.busy || state.paused) return;
  if (state.view !== 'live' || !video.videoWidth) return;
  state.busy = true;
  grabCanvas(video).toBlob(
    (blob) => {
      if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
      else state.busy = false;
    },
    'image/jpeg',
    0.88
  );
}
setInterval(pump, 200);

function renderFaces(faces) {
  state.lastFaces = faces;
  const boxes = faces.map((f) => {
    let color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    let boxClass = 'box';
    let tagText = f.name;

    if (state.livenessOn) {
      if (f.is_spoof) {
        color = FAKE_COLOR;
        boxClass = 'box box-spoof';
        const fakePct = f.liveness ? (f.liveness.fake_score * 100).toFixed(0) : 99;
        tagText = `🚨 [FAKE ${fakePct}%] ${f.raw_name || 'Giả mạo'}`;
      } else {
        color = f.known ? REAL_COLOR : UNKNOWN_COLOR;
        boxClass = f.known ? 'box box-real' : 'box';
        tagText = f.known ? `✅ [REAL] ${f.name}` : `❓ [REAL] Người lạ`;
      }
    } else {
      tagText = f.known ? `⚠️ ${f.name} (Chưa bật liveness)` : UNKNOWN;
    }

    return el(
      'div',
      {
        class: boxClass,
        style: {
          left: (1 - f.x - f.w) * 100 + '%',
          top: f.y * 100 + '%',
          width: f.w * 100 + '%',
          height: f.h * 100 + '%',
          borderColor: color,
        },
      },
      el('div', { class: 'tag', text: tagText, style: { background: color } })
    );
  });
  $('live-boxes').replaceChildren(...boxes);

  const cards = faces.map((f) => {
    let color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    let subText = 'Gần nhất: ' + f.best;

    if (state.livenessOn) {
      if (f.is_spoof) {
        color = FAKE_COLOR;
        const fakePct = f.liveness ? (f.liveness.fake_score * 100).toFixed(1) : 99;
        subText = `🚨 PHÁT HIỆN GIẢ MẠO (${fakePct}%) · TỪ CHỐI ĐIỂM DANH`;
      } else {
        color = f.known ? REAL_COLOR : UNKNOWN_COLOR;
        const realPct = f.liveness ? (f.liveness.real_score * 100).toFixed(1) : 100;
        subText = `✓ Thực thể sống (${realPct}%) · Hợp lệ`;
      }
    } else {
      subText = `⚠️ Đang tắt chống giả mạo (Nguy cơ bị gian lận)`;
    }

    return el('div', { class: 'face-card', style: { borderColor: f.is_spoof && state.livenessOn ? FAKE_COLOR : '' } }, [
      el('div', { class: 'card-head' }, [
        el('span', { text: f.name, style: { fontWeight: 600, fontSize: '15px', color } }),
        el('span', { class: 'mono small', text: f.score.toFixed(2), style: { color: '#B4C1C9' } }),
      ]),
      el('div', { class: 'meter' }, [
        el('div', { class: 'fill', style: { width: Math.max(0, f.score) * 100 + '%', background: color } }),
        el('div', { class: 'mark', style: { left: state.threshold * 100 + '%' } }),
      ]),
      el('div', { class: 'small muted', style: { color: f.is_spoof && state.livenessOn ? FAKE_COLOR : '' }, text: subText }),
    ]);
  });
  $('live-cards').replaceChildren(...cards);
}

function renderLog(rows) {
  if (!rows || !rows.length) return;
  $('log-list').replaceChildren(
    ...rows.map((row) =>
      el('div', { class: 'log-row' }, [
        el('span', { class: 'mono small muted', text: row.time }),
        el('span', { class: 'name', text: row.name, style: { color: row.name === 'Người lạ' ? UNKNOWN_COLOR : '' } }),
        el('span', { class: 'mono small', text: Number(row.score).toFixed(2), style: { color: '#B4C1C9' } }),
      ])
    )
  );
}

function renderThreshold() {
  $('thr-value').textContent = state.threshold.toFixed(2);
  const row = calibrationAt(state.threshold);
  if (!row) return;
  $('thr-tar').textContent = pct(row.tar);
  $('thr-far').textContent = pct(row.far);
  $('thr-far').style.color = row.far > 0 ? UNKNOWN_COLOR : '';
  $('thr-note').textContent =
    'Số liệu tại ngưỡng ' +
    row.t.toFixed(2) +
    ' từ đánh giá trên face_embeddings.npz. Ngưỡng lưu trong mô hình: ' +
    state.info.threshold.toFixed(2) +
    '.';
}

function updateLivenessUI() {
  const on = state.livenessOn;
  $('btn-liveness').setAttribute('aria-checked', String(on));
  $('btn-liveness').className = 'btn btn-liveness ' + (on ? 'active' : 'inactive');
  $('liveness-btn-text').textContent = on ? '🛡️ Chống giả mạo: BẬT' : '⚠️ Chống giả mạo: TẮT';

  const chip = $('chip-liveness');
  if (chip) {
    chip.textContent = on ? '🛡️ Chống Giả Mạo: BẬT' : '⚠️ Chống Giả Mạo: TẮT';
    chip.className = 'chip chip-liveness ' + (on ? 'on' : 'off');
  }

  const badge = $('liveness-badge-status');
  if (badge) {
    badge.textContent = on ? 'BẬT (AN TOÀN)' : 'TẮT (DỄ BỊ GIAN LẬN)';
    badge.style.background = on ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 82, 82, 0.15)';
    badge.style.color = on ? '#00E676' : '#FF5252';
  }

  const banner = $('liveness-banner');
  if (banner) {
    banner.className = 'card liveness-banner ' + (on ? '' : 'off');
  }

  const desc = $('liveness-desc');
  if (desc) {
    desc.innerHTML = on
      ? 'Mô hình <b>liveness_cam4k</b> đang hoạt động. Khuôn mặt trên màn hình điện thoại hoặc ảnh in sẽ bị phát hiện và từ chối chấm công.'
      : '<b style="color: #FF5252">CẢNH BÁO NGUY HIỂM:</b> Đang tắt kiểm tra thật/giả. Ảnh chụp hoặc màn hình điện thoại có thể qua mặt hệ thống để chấm công gian lận!';
  }
}

// Bật / tắt Liveness
$('btn-liveness').addEventListener('click', () => {
  state.livenessOn = !state.livenessOn;
  updateLivenessUI();
  sendSettings();
});

$('thr').addEventListener('input', (event) => {
  state.threshold = parseFloat(event.target.value);
  renderThreshold();
  sendSettings();
});

$('btn-pause').addEventListener('click', () => {
  state.paused = !state.paused;
  $('btn-pause').textContent = state.paused ? 'Tiếp tục' : 'Tạm dừng';
  $('live-state').replaceChildren(...(state.paused ? ['TẠM DỪNG'] : [el('span', { class: 'dot' }), 'LIVE']));
  if (state.paused) $('chip-fps').textContent = '0 FPS';
});

$('btn-log').addEventListener('click', () => {
  state.logOn = !state.logOn;
  $('btn-log').setAttribute('aria-checked', String(state.logOn));
  sendSettings();
});

$('btn-snap').addEventListener('click', () => {
  const video = $('live-video');
  if (!video.videoWidth) return;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  ctx.lineWidth = 3;
  ctx.font = '600 20px sans-serif';
  for (const f of state.lastFaces) {
    let color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    if (state.livenessOn && f.is_spoof) color = FAKE_COLOR;
    else if (state.livenessOn && f.known) color = REAL_COLOR;

    const [x, y, w, h] = [f.x * canvas.width, f.y * canvas.height, f.w * canvas.width, f.h * canvas.height];
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 28, ctx.measureText(f.name).width + 16, 26);
    ctx.fillStyle = '#0F1417';
    ctx.fillText(f.name, x + 8, y - 8);
  }
  const link = el('a', { href: canvas.toDataURL('image/png'), download: 'faceid_' + Date.now() + '.png' });
  link.click();
});

// ---- 2. Đăng ký người mới ----------------------------------------------------------------

function setStep(step) {
  const enroll = state.enroll;
  if (step >= 2 && !enroll.name) step = 1;
  if (step === 3 && !(enroll.summary && enroll.summary.embeddings)) step = 2;
  enroll.step = step;
  for (const n of [1, 2, 3]) $('enroll-step' + n).hidden = n !== step;
  document.querySelectorAll('.step').forEach((button) => {
    const n = Number(button.dataset.step);
    if (n === step) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
    button.classList.toggle('done', n < step);
  });
  if (step === 3) renderSummary();
}

document.querySelectorAll('.step').forEach((button) =>
  button.addEventListener('click', () => setStep(Number(button.dataset.step)))
);

$('enroll-next').addEventListener('click', () => {
  const name = $('enroll-name').value.trim();
  const error = !name ? 'Hãy nhập tên.' : /[\\/]/.test(name) ? 'Tên không được chứa / hoặc \\.' : '';
  $('enroll-name-error').textContent = error;
  $('enroll-name-error').hidden = !error;
  if (error) return;
  if (name !== state.enroll.name) Object.assign(state.enroll, { name, session: null, clips: [], summary: null });
  renderClips();
  setStep(2);
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

$('enroll-record').addEventListener('click', async () => {
  const video = $('enroll-video');
  if (!video.videoWidth) return;
  const button = $('enroll-record');
  button.disabled = true;
  $('enroll-rec').hidden = false;
  const frames = [];
  for (let i = 0; i < 30; i++) {
    $('enroll-count').textContent = Math.ceil((30 - i) / 5) + 's';
    frames.push(grabCanvas(video).toDataURL('image/jpeg', 0.85));
    await sleep(200);
  }
  $('enroll-count').textContent = 'đang phân tích…';
  try {
    const response = await fetch('/api/enroll/clip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: state.enroll.session, name: state.enroll.name, frames }),
    });
    const summary = await response.json();
    if (!response.ok) throw new Error(summary.detail || response.statusText);
    state.enroll.session = summary.session;
    state.enroll.summary = summary;
    state.enroll.clips.push(summary);
    renderClips();
  } catch (error) {
    $('clip-list').prepend(el('p', { class: 'small', text: 'Lỗi: ' + error.message, style: { color: 'var(--warn)' } }));
  } finally {
    button.disabled = false;
    $('enroll-rec').hidden = true;
  }
});

function renderClips() {
  const clips = state.enroll.clips;
  $('enroll-review').disabled = !(state.enroll.summary && state.enroll.summary.embeddings);
  if (!clips.length) {
    $('clip-list').replaceChildren(
      el('p', { class: 'small muted', text: 'Chưa quay đoạn nào. Chỉ để một người trong khung hình.' })
    );
    return;
  }
  $('clip-list').replaceChildren(
    ...clips.map((clip) => {
      const s = clip.stats;
      const total = clip.total || 1;
      const poor = s.blurry + s.low_score + s.small + s.pose;
      const missing = s.no_face + s.multiple_faces;
      const tips = [];
      if (s.blurry / total > 0.4) tips.push('Nhiều frame bị mờ — quay nơi sáng hơn, giữ máy yên.');
      if (s.multiple_faces / total > 0.2) tips.push('Có người khác lọt vào khung hình.');
      if (s.no_face / total > 0.3) tips.push('Nhiều frame không thấy mặt — ngồi gần camera hơn.');
      return el(
        'div',
        {
          style: {
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            paddingTop: '10px',
            borderTop: '1px solid var(--line-soft)',
          },
        },
        [
          el('div', { class: 'card-head', style: { fontSize: '14px' } }, [
            el('span', { text: 'Đoạn ' + clip.clip, style: { fontWeight: 500 } }),
            el('span', { class: 'mono', text: s.ok + '/' + total + ' frame tốt', style: { color: '#B4C1C9' } }),
          ]),
          el('div', { class: 'qbar' }, [
            el('div', { style: { width: (s.ok / total) * 100 + '%', background: 'var(--known)' } }),
            el('div', { style: { width: (poor / total) * 100 + '%', background: 'var(--unknown)' } }),
            el('div', { style: { width: (missing / total) * 100 + '%', background: 'var(--pose)' } }),
          ]),
          ...tips.map((tip) => el('div', { class: 'small', text: tip, style: { color: 'var(--warn)' } })),
        ]
      );
    })
  );
}

$('enroll-review').addEventListener('click', () => setStep(3));
$('enroll-more').addEventListener('click', () => setStep(2));

function renderSummary() {
  const enroll = state.enroll;
  const summary = enroll.summary;
  const valid = enroll.clips.filter((clip) => clip.stats.ok > 0).length;
  $('sum-clips').textContent = valid + ' / ' + enroll.clips.length;
  $('sum-emb').textContent = summary.embeddings;
  $('sum-nearest').textContent = summary.nearest ? summary.nearest.name + ' ' + summary.nearest.score.toFixed(2) : '—';
  const threshold = state.info ? state.info.threshold : 0.55;
  const notice = $('sum-notice');
  const lines = [];
  let good = true;
  if (summary.nearest && summary.nearest.score >= threshold) {
    good = false;
    lines.push(
      'Giống ' +
        summary.nearest.name +
        ' (' +
        summary.nearest.score.toFixed(2) +
        ' ≥ ' +
        threshold.toFixed(2) +
        '): có thể người này đã được đăng ký dưới tên khác.'
    );
  } else {
    lines.push('Dưới ngưỡng ' + threshold.toFixed(2) + ' với mọi người đã đăng ký — không trùng hồ sơ cũ.');
  }
  if (summary.exists) lines.push('Tên "' + enroll.name + '" đã có: dữ liệu mới sẽ được thêm vào hồ sơ cũ.');
  if (valid < 2) lines.push('Nên quay ít nhất 2 đoạn để hồ sơ ổn định và đánh giá được.');
  notice.className = 'notice ' + (good ? 'good' : 'bad');
  notice.replaceChildren(el('div', {}, lines.map((line) => el('div', { text: line }))));
  $('enroll-result').hidden = true;
}

$('enroll-save').addEventListener('click', async () => {
  const button = $('enroll-save');
  button.disabled = true;
  try {
    const response = await fetch('/api/enroll/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: state.enroll.session }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || response.statusText);
    $('enroll-result').textContent =
      'Đã lưu ' + result.name + ' (' + result.embeddings + ' embedding). Danh sách hiện có ' + result.people + ' người.';
    $('enroll-result').style.color = 'var(--ok)';
    Object.assign(state.enroll, { session: null, clips: [], summary: null });
    await loadInfo();
  } catch (error) {
    $('enroll-result').textContent = 'Lỗi: ' + error.message;
    $('enroll-result').style.color = 'var(--warn)';
  } finally {
    $('enroll-result').hidden = false;
    button.disabled = false;
  }
});

// ---- 3. Quản lý dữ liệu nhân viên & Điểm danh ---------------------------------------------

async function loadManageView() {
  const container = $('manage-people-list');
  if (!container) return;
  container.innerHTML = '<p class="small muted">Đang tải danh sách nhân viên...</p>';

  try {
    const res = await fetch('/api/people/list');
    const data = await res.json();
    const people = data.people || [];

    $('manage-count').textContent = people.length;

    if (people.length === 0) {
      container.innerHTML =
        '<p class="small muted" style="padding: 16px 0;">Chưa có nhân viên nào trong cơ sở dữ liệu. Hãy bấm "Đăng ký người mới" để thêm.</p>';
      return;
    }

    container.innerHTML = '';
    people.forEach((p) => {
      const item = el('div', { class: 'person-item' }, [
        el('div', { class: 'person-info' }, [
          el('span', { class: 'person-name', text: p.name }),
          el('span', { class: 'person-meta', text: `${p.frames} mẫu khuôn mặt (frames) · ${p.videos} video clips` }),
        ]),
        el('div', { class: 'person-actions' }, [
          el(
            'button',
            {
              class: 'btn small',
              type: 'button',
              onclick: async () => {
                const newName = prompt(`Nhập tên mới thay thế cho "${p.name}":`, p.name);
                if (!newName || newName.trim() === p.name) return;
                try {
                  const renameRes = await fetch('/api/people/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ old_name: p.name, new_name: newName.trim() }),
                  });
                  const renameData = await renameRes.json();
                  if (!renameRes.ok) throw new Error(renameData.detail || 'Không thể đổi tên');
                  await loadManageView();
                  await loadInfo();
                } catch (e) {
                  alert('Lỗi: ' + e.message);
                }
              },
            },
            ['✏️ Đổi tên']
          ),
          el(
            'button',
            {
              class: 'btn small btn-danger',
              type: 'button',
              onclick: async () => {
                if (!confirm(`Bạn có chắc chắn muốn xóa hồ sơ của "${p.name}" khỏi hệ thống không?`)) return;
                try {
                  const delRes = await fetch('/api/people/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: p.name }),
                  });
                  const delData = await delRes.json();
                  if (!delRes.ok) throw new Error(delData.detail || 'Không thể xóa hồ sơ');
                  await loadManageView();
                  await loadInfo();
                } catch (e) {
                  alert('Lỗi: ' + e.message);
                }
              },
            },
            ['🗑️ Xóa']
          ),
        ]),
      ]);
      container.appendChild(item);
    });
  } catch (err) {
    container.innerHTML = `<p class="small" style="color: var(--warn)">Lỗi tải danh sách: ${err.message}</p>`;
  }
}

// Nút xóa trắng nhật ký điểm danh hôm nay
$('btn-clear-log').addEventListener('click', async () => {
  if (!confirm('Bạn có chắc chắn muốn xóa sạch toàn bộ lịch sử điểm danh hôm nay không?')) return;
  try {
    const res = await fetch('/api/log/clear', { method: 'POST' });
    if (!res.ok) throw new Error('Không thể xóa nhật ký');
    await loadInfo();
    alert('Đã xóa trắng lịch sử điểm danh hôm nay.');
  } catch (e) {
    alert('Lỗi: ' + e.message);
  }
});

// Nút xóa sạch toàn bộ hồ sơ tất cả nhân viên (Reset)
$('btn-clear-all').addEventListener('click', async () => {
  if (
    !confirm(
      '⚠️ CẢNH BÁO QUAN TRỌNG:\nThao tác này sẽ XÓA SẠCH toàn bộ dữ liệu khuôn mặt của TẤT CẢ nhân viên!\nBạn có chắc chắn muốn tiếp tục?'
    )
  )
    return;
  try {
    const res = await fetch('/api/people/clear_all', { method: 'POST' });
    if (!res.ok) throw new Error('Không thể xóa toàn bộ dữ liệu');
    await loadManageView();
    await loadInfo();
    alert('Đã xóa sạch toàn bộ dữ liệu nhân viên. Bạn có thể chuyển sang tab Đăng ký để thêm người mới.');
  } catch (e) {
    alert('Lỗi: ' + e.message);
  }
});

// ---- 4. Đánh giá mô hình -----------------------------------------------------------------

const THRESHOLD_NOTES = {
  '0.45': 'Ngưỡng cũ: một phần frame người lạ bị gọi bằng tên người giống nhất.',
  '0.50': 'Điểm người lạ vừa hết bị nhận nhầm, chưa có khoảng dự phòng.',
  '0.55': 'Ngưỡng đề xuất: thêm 0.05 dự phòng cho frame webcam mờ hơn dữ liệu hiệu chỉnh.',
};

function renderReport() {
  const info = state.info;
  if (!info) return;
  const report = info.report;
  $('kpi-people').textContent = info.people.length;
  $('kpi-videos').textContent = report.videos || '—';
  $('kpi-frames').textContent = report.frames ? report.frames.toLocaleString('vi-VN') : '—';
  $('kpi-top1').textContent = report.top1 != null ? pct(report.top1) : '—';

  const rows = report.calibration || [];
  $('chart').replaceChildren(
    ...rows.map((row) =>
      el(
        'button',
        {
          type: 'button',
          'aria-pressed': String(Math.abs(row.t - state.reportT) < 0.001),
          'aria-label': 'Ngưỡng ' + row.t.toFixed(2),
          onclick: () => {
            state.reportT = row.t;
            renderReport();
          },
        },
        [
          el('span', { class: 'bars' }, [
            el('span', { style: { height: row.tar * 100 + '%', background: 'var(--known)' } }),
            el('span', { style: { height: row.far * 100 + '%', background: 'var(--unknown)' } }),
          ]),
          el('span', { class: 't', text: row.t.toFixed(2) }),
        ]
      )
    )
  );

  const sel = calibrationAt(state.reportT);
  if (sel) {
    $('sel-t').textContent = sel.t.toFixed(2);
    $('sel-tar').textContent = pct(sel.tar);
    $('sel-far').textContent = pct(sel.far);
    $('sel-far').style.color = sel.far > 0 ? UNKNOWN_COLOR : '';
    $('sel-note').textContent =
      THRESHOLD_NOTES[sel.t.toFixed(2)] ||
      (sel.far > 0
        ? 'Quá dễ: người lạ thường bị gọi bằng tên người khác.'
        : 'An toàn với người lạ; ngưỡng càng cao càng dễ bỏ sót người quen.');
  }

  const maxFrames = Math.max(1, ...report.people.map((p) => p.frames));
  $('people-table').replaceChildren(
    ...report.people.map((p) => {
      const close = p.nearestScore >= 0.4;
      return el('tr', {}, [
        el('td', { text: p.name, style: { fontWeight: 500 } }),
        el('td', { class: 'mono', text: String(p.videos) }),
        el('td', {}, el('span', { style: { display: 'flex', alignItems: 'center', gap: '10px' } }, [
          el('span', { class: 'mono', text: String(p.frames), style: { width: '44px' } }),
          el('span', {
            class: 'mini',
          }, el('span', {
            style: { width: (p.frames / maxFrames) * 100 + '%', background: p.frames < 60 ? UNKNOWN_COLOR : KNOWN_COLOR },
          })),
        ])),
        el('td', {
          text: p.nearest ? p.nearest + ' ' + p.nearestScore.toFixed(2) : '—',
          style: { color: close ? '#F2B544' : '#B4C1C9' },
        }),
      ]);
    })
  );
}

// ---- Khởi động ---------------------------------------------------------------------------

showView();
setStep(1);
loadInfo();
startCamera();
connect();
updateLivenessUI();
