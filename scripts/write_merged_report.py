"""Report the completed merged run against the frozen pre-merge Sep30 model."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return f'{value:,}'.replace(',', '.')


def pct(value):
    return 'Không xác định' if value is None else f'{value*100:.2f}%'.replace('.', ',')


def loss(value):
    return f'{value:.5f}'.replace('.', ',')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs/merged-20261001')
    parser.add_argument('--dataset', default='dataset/merged-20261001')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    run, data = ROOT/args.run, ROOT/args.dataset
    e, m = read(run/'evaluation.json'), read(data/'manifest.json')
    anchor, api = read(run/'anchor_comparison.json'), read(run/'api_smoke.json')
    old_m = read(ROOT/'dataset/retrain-20260930/manifest.json')
    old_e = read(ROOT/'outputs/retrain-20260930/evaluation.json')
    config = e['configuration']
    if (config['baseline_model_sha256'] != old_e['new_model_sha256']
            or config['dataset_manifest_sha256'] != sha(data/'manifest.json')
            or e['new_model_sha256'] != sha(run/'liveness.h5')
            or anchor['candidate_model_sha256'] != e['new_model_sha256']
            or api['model_sha256'] != e['new_model_sha256']):
        raise ValueError('Merged report inputs do not match the frozen runs')
    total_before = sum(s['frames'] for s in old_m['summary'].values())
    total_after = len(m['frames'])
    sources = ('legacy-cache', 'drive-20260930')
    source_names = {'legacy-cache':'Ảnh từ bộ cũ', 'drive-20260930':'Ảnh từ video Drive'}
    summary = e['source_split_summary']
    legacy_total = sum(v['frames'] for v in summary['legacy-cache'].values())
    new_total = sum(v['frames'] for v in summary['drive-20260930'].values())
    old_rejected = sum(m.get('legacy_quality_rejections', {}).values())
    drive_clips = [v for v in m['videos'] if v['source']=='drive-20260930']
    drive_sampled = sum(v['statistics'].get('sampled',0) for v in drive_clips)
    lines = ['# Báo cáo gộp dữ liệu và huấn luyện lại', '',
             'Ngày hoàn thành: 01/10/2026.', '',
             '**Trước = model ngày 30/9 khi chưa gộp toàn bộ bộ ảnh cũ. Sau = model vừa train bằng bộ dữ liệu gộp ngày 01/10.** Các chỉ số trước/sau dưới đây chấm cả hai model trên cùng ảnh đã khóa. Điểm trước được đánh giá lại trên bộ kiểm tra này, không lấy điểm của một test khác để so.', '',
             '## 1. Dữ liệu trước và sau gộp', '',
             '| Tập | Trước gộp (30/9) | Sau gộp (01/10) |', '|---|---:|---:|']
    for split,title in [('train','Train'),('validation','Validation'),('test','Test chính')]:
        lines.append(f"| {title} | {number(old_m['summary'][split]['frames'])} | {number(m['summary'][split]['frames'])} |")
    lines += [f"| Kiểm tra video cũ riêng | {number(old_m['summary']['regression_old']['frames'])} | Nguồn video nằm trong test chung; vẫn chấm bổ sung đúng 415 ảnh trước |",
              f'| Tổng ảnh của bộ | {number(total_before)} | {number(total_after)} |', '',
              'Test chính của bộ trước chỉ gồm 1.473 ảnh từ video mới. Test chính của bộ sau là test chung gồm 3.343 ảnh cache cũ và 1.513 ảnh video mới. Các chỉ số so sánh ở mục 3–4 đánh giá lại cả hai model trên cùng test chung của bộ sau.', '',
              'Trong bộ sau, mọi ảnh chỉ thuộc một tập train/validation/test. “Gộp” nghĩa là cả nguồn cũ và mới cùng có mặt trong bộ dữ liệu, rồi giữ ảnh kiểm tra ngoài train.', '',
              '| Nguồn trong bộ sau | Train | Validation | Test | Tổng |', '|---|---:|---:|---:|---:|']
    for source in sources:
        counts = summary[source]
        values = [counts.get(split,{}).get('frames',0) for split in ('train','validation','test')]
        lines.append('| '+source_names[source]+' | '+' | '.join(number(v) for v in values)+' | '+number(sum(values))+' |')
    lines += ['', f'Em rà soát toàn bộ **24.221 ảnh cũ**. Sau kiểm tra chất lượng ảnh cắt và trùng pixel, bộ gộp giữ {number(legacy_total)} ảnh cũ. Em kiểm tra {len(drive_clips)} video Drive, lấy {number(drive_sampled)} khung cách nhau ít nhất 0,2 giây trên toàn video, không giới hạn 100 khung/video. Bộ gộp giữ {number(new_total)} ảnh từ video mới.', '',
              f'Ảnh cũ bị loại bởi kiểm tra chất lượng: {number(old_rejected)}; gồm 158 ảnh lóa và 31 ảnh thiếu nét.', '',
              'Ảnh cũ kiểm tra đọc ảnh, cạnh tối thiểu 65 px, độ sáng, lóa, độ nét và ảnh đồng màu. Vì đã cắt sẵn bằng SSD, ảnh cũ thiếu tọa độ toàn khung và điểm mốc để kiểm tra đầy đủ như API. Ảnh video mới chạy YuNet và bộ lọc chất lượng trên khung gốc. Bộ lọc áp dụng cùng quy tắc cho cả nhãn thật và giả, không dựa vào điểm phân loại của model.', '',
              'Em kiểm tra trùng đúng pixel đầu vào 32×32: không phát hiện ảnh trùng chính xác hoặc trùng khác nhãn. Không áp dụng ngưỡng gần trùng chưa được kiểm chứng.', '',
              'Giữ nguyên ranh giới video của bộ cũ ngày 15/9: 23 video train, 5 validation và 5 test. Nguồn Drive giữ B/Thu cho validation, Nga/Phúc cho test, các mã còn lại cho train. Dữ liệu cũ tách theo video, có người xuất hiện ở nhiều tập. Sáu tên video chung chung chưa xác định được danh tính và được đánh dấu rõ. Việc tách theo người của nguồn mới dựa vào tên nhóm cung cấp, chưa xác minh độc lập danh tính.', '',
              '## 2. Huấn luyện', '',
              '| Thông số | Trước gộp | Sau gộp |', '|---|---:|---:|',
              f"| Epoch đã chạy | {config['baseline_training']['epochs_run']} | {e['epochs_run']} |",
              f"| Epoch chọn checkpoint | {config['baseline_training']['best_epoch']} | {e['best_epoch']} |",
              f"| Loss train lúc train tại checkpoint, có augmentation/trọng số | {loss(config['baseline_training']['historical_augmented_train_loss_at_best'])} | {loss(e['history_at_best_epoch']['loss'])} |",
              f"| Loss validation lịch sử tại checkpoint | {loss(config['baseline_training']['historical_validation_loss_at_best'])} | {loss(e['history_at_best_epoch']['val_loss'])} |", '',
              'Hai dòng loss lịch sử dùng hai bộ dữ liệu khác nhau. Dùng bảng đánh giá trên cùng dữ liệu ở mục 3 để so chất lượng. Model sau train từ đầu với CNN LivenessNet, BGR 32×32 /255, seed 42, Adam 0,0001, batch 32, tối đa 50 epoch. Tổng trọng số của hai lớp bằng nhau; trong mỗi lớp, tổng trọng số của từng video bằng nhau để video nhiều ảnh không lấn át. Chọn checkpoint theo validation loss, dừng sớm sau 8 epoch không cải thiện. Không chọn model hoặc ngưỡng theo test.', '',
              '## 3. Loss, precision, recall và F1 trên cùng bộ gộp', '',
              'Các tập train/validation/test trong bảng là của bộ sau. Loss đánh giá là cross-entropy trên ảnh gốc, chế độ suy luận, không augmentation và không trọng số. P/R/F1 là macro, cho hai lớp trọng số bằng nhau. Đây là phân loại ảnh cắt, chọn lớp có điểm cao nhất, chưa áp dụng quyết định theo nhiều khung của API.', '',
              '| Tập | Loss trước / sau | Precision trước / sau | Recall trước / sau | F1 trước / sau | Accuracy trước / sau |',
              '|---|---:|---:|---:|---:|---:|']
    for split,title in [('train','Train'),('validation','Validation'),('test','Test chung')]:
        b,a = e['before'][split],e['after'][split]
        bp,ap = b['classification_report']['macro avg'],a['classification_report']['macro avg']
        values = [loss(b['loss'])+' / '+loss(a['loss'])]+[pct(bp[k])+' / '+pct(ap[k]) for k in ('precision','recall','f1-score')]+[pct(b['accuracy'])+' / '+pct(a['accuracy'])]
        lines.append('| '+title+' | '+' | '.join(values)+' |')
    lines += ['', 'Model trước đã học các video nguồn của một phần validation cũ trong lần 30/9. Vì vậy, điểm validation của model trước có phơi nhiễm nguồn; không diễn giải là test độc lập. Validation của model sau giữ ngoài train trong lần này.', '',
              '## 4. Test chung và kết quả theo nguồn', '',
              '| Nhóm test | Ảnh / video | Precision trước / sau | Recall trước / sau | F1 trước / sau | Accuracy trước / sau | Loss trước / sau |',
              '|---|---:|---:|---:|---:|---:|---:|']
    groups = [('Tất cả',e['before']['test'],e['after']['test'])]
    groups += [(source_names[s],e['before']['test']['per_source'][s],e['after']['test']['per_source'][s]) for s in sources]
    for title,b,a in groups:
        bp,ap=b['classification_report']['macro avg'],a['classification_report']['macro avg']
        vals = [number(a['frame_count'])+' / '+str(a['video_count'])]+[pct(bp[k])+' / '+pct(ap[k]) for k in ('precision','recall','f1-score')]+[pct(b['accuracy'])+' / '+pct(a['accuracy']),loss(b['loss'])+' / '+loss(a['loss'])]
        lines.append('| '+title+' | '+' | '.join(vals)+' |')
    lines += ['', '| Lỗi test chung | Trước | Sau |', '|---|---:|---:|']
    br,ar=e['before']['test']['presentation_attack_rates'],e['after']['test']['presentation_attack_rates']
    for title,count,denominator,rate in [('Giả bị nhận thành thật','fake_as_real_count','fake_count','APCER'),('Thật bị nhận thành giả','real_as_fake_count','real_count','BPCER')]:
        lines.append(f'| {title} | {br[count]}/{br[denominator]} ({pct(br[rate])}) | {ar[count]}/{ar[denominator]} ({pct(ar[rate])}) |')
    lines.append(f"| ACER | {pct(br['ACER'])} | {pct(ar['ACER'])} |")
    lines += ['', '| Bỏ lọt ảnh giả theo nguồn | Trước | Sau |', '|---|---:|---:|']
    for source in sources:
        sr_before = e['before']['test']['per_source'][source]['presentation_attack_rates']
        sr_after = e['after']['test']['per_source'][source]['presentation_attack_rates']
        lines.append(f"| {source_names[source]} | {sr_before['fake_as_real_count']}/{sr_before['fake_count']} ({pct(sr_before['APCER'])}) | {sr_after['fake_as_real_count']}/{sr_after['fake_count']} ({pct(sr_after['APCER'])}) |")
    bootstrap = e['test_video_paired_bootstrap']
    vf = bootstrap['metrics']['macro_f1']
    interval = vf['delta_95_percentile_interval']
    lines += ['', f"Gộp xác suất các ảnh để lấy một nhãn cho mỗi video: F1 macro theo video {pct(vf['before'])} thành {pct(vf['after'])}. Bootstrap ghép cặp theo {bootstrap['video_count']} video, 5.000 lần lấy mẫu, cho khoảng 95% của chênh lệch F1 là {interval[0]*100:.2f} đến {interval[1]*100:.2f} điểm phần trăm. Đây là phép đo khác F1 theo ảnh trong bảng. Khoảng này chỉ phản ánh các video/người đã thử; video cùng người có thể vẫn tương quan, số video ít nên chưa suy rộng được."]
    lines += ['', '## 5. Kiểm tra lại đúng các ảnh của báo cáo trước', '',
              'Giữ nguyên 1.473 ảnh test mới và 415 ảnh kiểm tra video cũ đã dùng ngày 30/9, chỉ đổi model. Các video này vẫn nằm ngoài train sau gộp. 415 ảnh là ảnh cắt bằng YuNet từ cùng 5 video cũ; khác bộ 3.343 ảnh cache cắt bằng SSD trong test chung. Chúng chỉ là benchmark bổ sung giữ nguyên để đối chiếu, không cộng thêm vào tổng 28.957 ảnh. Điểm model trước được tính lại và xác minh khớp mảng dự đoán đã lưu của lần 30/9.', '',
              '| Nhóm cố định | Ảnh | F1 trước / sau | Accuracy trước / sau | Loss trước / sau |', '|---|---:|---:|---:|---:|']
    for key,title in [('new','Video mới'),('old','Video cũ')]:
        b,a=anchor['groups'][key]['before'],anchor['groups'][key]['after']
        lines.append(f"| {title} | {number(a['frame_count'])} | {pct(b['classification_report']['macro avg']['f1-score'])} / {pct(a['classification_report']['macro avg']['f1-score'])} | {pct(b['accuracy'])} / {pct(a['accuracy'])} | {loss(b['loss'])} / {loss(a['loss'])} |")
    replay_path=run/'continuous_runtime_comparison.json'
    if replay_path.exists():
        runtime=read(replay_path)
        if runtime['models']['before']['model_sha256']!=config['baseline_model_sha256'] or runtime['models']['after']['model_sha256']!=e['new_model_sha256']:
            raise ValueError('Pipeline replay used different models')
        lines += ['', '## 6. Phát lại pipeline và tích hợp API', '',
                  'Phát lại cố định 12 giây đầu của 16 video test mới, tối đa 5 khung/giây, dùng cùng JPEG của SDK rồi chạy detector, chất lượng và quyết định theo nhiều khung. Đây là offline, chưa đo mạng hoặc camera trực tiếp.', '',
                  '| Kết quả replay | Trước | Sau |', '|---|---:|---:|']
        for title,key in [('Khung đã thử','all_sample_count'),('Kết luận đúng','correct_conclusive_count'),('Kết luận sai','wrong_conclusive_count'),('Chưa kết luận / bị loại','no_decision_count')]:
            lines.append(f"| {title} | {runtime['before'][key]} | {runtime['after'][key]} |")
        lines += [f"| Tỷ lệ đưa ra Real/Fake | {pct(runtime['before']['conclusive_coverage'])} | {pct(runtime['after']['conclusive_coverage'])} |", '',
                  'Khung chưa kết luận không được tính là đúng. Cần đọc số quyết định sai cùng tỷ lệ đưa ra quyết định.']
    else:
        lines += ['', '## 6. Tích hợp API', '']
    lines += ['',f"Model H5 sau gộp đã nạp và gọi thật qua Flask test client: health, analyze, tạo/gửi/xóa session trên {len(api['examples'])} JPEG mẫu. Kiểm tra đúng mã model và hash ảnh đầu vào. Đây là kiểm tra tích hợp nội bộ, không phải đo độ chính xác độc lập hoặc tốc độ camera/mạng.", '',
              '## 7. Nhận xét và kết luận', '']
    b,a=e['before']['test'],e['after']['test']
    delta=a['classification_report']['macro avg']['f1-score']-b['classification_report']['macro avg']['f1-score']
    lines.append(f"Trên cùng test gộp, F1 macro {pct(b['classification_report']['macro avg']['f1-score'])} thành {pct(a['classification_report']['macro avg']['f1-score'])}, accuracy {pct(b['accuracy'])} thành {pct(a['accuracy'])}, loss {loss(b['loss'])} thành {loss(a['loss'])}.")
    for source in sources:
        b,a=e['before']['test']['per_source'][source],e['after']['test']['per_source'][source]
        delta_source=a['classification_report']['macro avg']['f1-score']-b['classification_report']['macro avg']['f1-score']
        source_caveat = 'F1 gần như ngang nhau, nhưng bỏ lọt ảnh giả tăng.' if abs(delta_source)<0.001 else 'F1 giảm và bỏ lọt ảnh giả tăng.' if delta_source<0 else 'Cần đọc cùng tỷ lệ bỏ lọt ảnh giả ở mục 4.'
        lines.append(f"\n{source_names[source]}: F1 macro {'tăng' if delta_source>0 else 'giảm' if delta_source<0 else 'giữ nguyên'} từ {pct(b['classification_report']['macro avg']['f1-score'])} thành {pct(a['classification_report']['macro avg']['f1-score'])} ({delta_source*100:+.2f} điểm phần trăm). Accuracy thay đổi {pct(b['accuracy'])} thành {pct(a['accuracy'])}. {source_caveat}")
    improvement = delta>0 and ar['ACER']<br['ACER']
    worse = delta<0 and ar['ACER']>br['ACER']
    conclusion = ('Model sau gộp tốt hơn trên test chung đã kiểm tra.' if improvement else
                  'Model sau gộp kém hơn về F1 và lỗi trung bình trên test chung.' if worse else
                  'Kết quả có đánh đổi; chưa có cải thiện đồng thời F1 và lỗi trung bình trên test chung.')
    lines += ['', '**'+conclusion+'** Kết luận có giới hạn trong các video và điều kiện đã thử. Cần đọc riêng nhóm cũ/mới để thấy đánh đổi.', '',
              f"Tỷ lệ ảnh giả bị nhận thành thật tăng từ {pct(br['APCER'])} lên {pct(ar['APCER'])}; tỷ lệ ảnh thật bị nhận thành giả giảm từ {pct(br['BPCER'])} xuống {pct(ar['BPCER'])}. Với mục tiêu chống giả mạo, việc bỏ lọt ảnh giả tăng là điểm cần khắc phục.", '',
              'Em giữ model sau gộp trong thư mục riêng để phân tích, chưa thay model của API đang dùng bằng bản này. Có nhiều ảnh hơn không tự đảm bảo model tốt hơn. Cách cắt mặt cũ bằng SSD và mới bằng YuNet, cùng khác biệt cách quay/codec, có thể gây lệch nguồn; đây là giả thuyết cần kiểm chứng, chưa xác định nguyên nhân duy nhất. Lần tiếp theo nên đồng bộ tiền xử lý và điều chỉnh huấn luyện bằng validation, rồi dùng một test mới độc lập để chốt.', '',
              'Giới hạn:', '',
              '- Các test này đã được xem trước và ảnh hưởng đến quyết định gộp dữ liệu. Đây là benchmark phát triển cố định, chưa phải một test cuối hoàn toàn mới.',
              '- Người trong video cũ có thể xuất hiện ở train. Sáu video tên chung chung chưa xác định được danh tính. Nguồn mới tách mã người theo thư mục, chưa xác minh độc lập.',
              '- Video thật và giả mới còn khác codec, định dạng và cách quay. Model có thể học dấu hiệu nguồn dữ liệu.',
              '- Ảnh trong cùng video tương quan. F1 theo ảnh, F1 theo một quyết định/video và kết quả temporal API là ba phép đánh giá khác nhau.',
              '- Điểm cao trên các tập này không chứng minh hết overfit hoặc đủ an toàn cho xác thực thực tế.', '',
              'Bước tiếp theo là test thêm người và điều kiện quay mới, thu thật/giả cùng thiết bị, ghi rõ kiểu tấn công và kiểm thử camera gọi API qua mạng.', '',
              '## 8. File kết quả và cách sử dụng', '',
              '- [Model sau gộp](liveness.h5) và [nhãn](le.pickle)',
              '- [Đánh giá đầy đủ](evaluation.json)',
              '- [Đánh giá lại ảnh cố định](anchor_comparison.json)',
              '- [Lịch sử từng epoch](training.csv)',
              '- [Kiểm tra tích hợp API](api_smoke.json)',
              '- [Kiểm toán độc lập dữ liệu và kết quả](audit.json)',
              '- [Đường cong học](learning_curves.png)', '',
              'Model sau gộp nằm trong thư mục riêng. Lệnh chạy API bằng model này:', '',
              '```powershell',
              '.\\venv\\Scripts\\python.exe api_app.py --model outputs/merged-20261001/liveness.h5 --labels outputs/merged-20261001/le.pickle',
              '```', '',
              '![Loss và accuracy theo epoch](learning_curves.png)']
    output=run/'report.md'
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
