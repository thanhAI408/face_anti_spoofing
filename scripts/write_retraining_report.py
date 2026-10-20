"""Write a Vietnamese report from completed, paired training/evaluation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(value):
    return f'{value*100:.2f}%'.replace('.', ',')


def decimal(value):
    return f'{value:.5f}'.replace('.', ',')


def integer(value):
    return f'{value:,}'.replace(',', '.')


def points(value, signed=False):
    return (f'{value*100:+.2f}' if signed else f'{value*100:.2f}').replace('.', ',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='outputs/retrain-20260930')
    parser.add_argument('--dataset', default='dataset/retrain-20260930')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild this run report after review')
    args = parser.parse_args()
    run = ROOT/args.run
    report = json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/args.dataset/'manifest.json').read_text(encoding='utf-8'))
    manifest_hash=hashlib.sha256((ROOT/args.dataset/'manifest.json').read_bytes()).hexdigest()
    if manifest_hash != report['configuration']['dataset_manifest_sha256']:
        raise ValueError('Report manifest differs from the evaluated training run')
    before, after = report['before'], report['after']
    baseline = report['configuration']['baseline_training']
    bp, ap = before['test']['classification_report'], after['test']['classification_report']
    br, ar = before['test']['presentation_attack_rates'], after['test']['presentation_attack_rates']
    f1_delta = ap['macro avg']['f1-score']-bp['macro avg']['f1-score']
    accuracy_delta = after['test']['accuracy']-before['test']['accuracy']
    old_image_counts = {label: sum(1 for _ in (ROOT/'dataset'/label).rglob('*.png'))
                        for label in ('fake', 'real')}
    old_videos = [v for v in manifest['videos'] if v['source'] == 'original']
    old_stats = {key: sum(v['statistics'].get(key, 0) for v in old_videos)
                 for key in ('sampled', 'accepted', 'no_face', 'multiple_faces', 'quality_rejected')}
    source_frames = {
        source: {split: sum(v['accepted_frames'] for v in manifest['videos']
                            if v['source'] == source and v['split'] == split)
                 for split in ('train', 'validation', 'test', 'regression_old')}
        for source in ('original', 'drive-20260930')
    }
    prior_split_path = ROOT / report['configuration']['baseline_model_path']
    prior_split_path = prior_split_path.parent / 'split.json'
    prior_split = json.loads(prior_split_path.read_text(encoding='utf-8'))
    prior_evaluation = json.loads((prior_split_path.parent/'evaluation.json').read_text(encoding='utf-8'))
    prior_test = prior_evaluation['new_model']['classification_report']
    prior_total = sum(stat['frames'] for stat in prior_split.values())
    current_total = sum(stat['frames'] for stat in manifest['summary'].values())
    lines = ['# Báo cáo bổ sung dữ liệu và huấn luyện lại model', '',
             'Ngày thực hiện: 30/09/2026. Em so sánh model ngày 15/9 với model huấn luyện lại từ đầu, trên cùng dữ liệu đã khóa.', '',
             '## 1. Dữ liệu và cách chia', '',
             'Em tải đủ 56 video mới từ Google Drive (30 thật, 26 giả, khoảng 1,58 GB), kiểm tra kích thước và SHA256. Thư mục A MOV dùng nhãn Fake sau khi sếp chỉnh lại trên Drive.', '',
             '### So sánh bộ dữ liệu trước và sau bổ sung video', '',
             'Trong phần so sánh trước–sau, “bộ trước” là bộ dùng ngày 15/9, trước khi tải video mới. “Bộ sau” là toàn bộ bộ được chuẩn bị cho lần train ngày 30/9, gồm ảnh từ video cũ và video vừa bổ sung. Đây là hai mốc thời gian, không phải hai nguồn video.', '',
             '| Tập | Trước bổ sung (15/9) | Sau bổ sung, bộ trích lại (30/9) |',
             '|---|---:|---:|']
    for split, title in [('train', 'Train'), ('validation', 'Validation'), ('test', 'Test')]:
        lines.append(f"| {title} | {integer(prior_split[split]['frames'])} | {integer(manifest['summary'][split]['frames'])} |")
    lines += [f"| Hồi quy riêng | Chưa tách riêng | {integer(manifest['summary']['regression_old']['frames'])} |",
              f"| Tổng ảnh của bộ dữ liệu | {integer(prior_total)} | {integer(current_total)} |", '',
             f"**Lần train sau chưa dùng toàn bộ {integer(prior_total)} ảnh cũ rồi cộng thêm ảnh mới.** Em tạo một bộ riêng bằng cách lấy mẫu lại từ video cũ và video mới. Vì thay cách trích ảnh, giới hạn số khung/video và lọc đầu vào, bộ sau chỉ có {integer(current_total)} ảnh. Cách làm này tăng nguồn video nhưng giảm số ảnh dùng so với lần trước.", '',
             'Ngày 15/9, 5 video dùng cho hồi quy hiện tại nằm trong tập test cũ. Ngày 30/9, em trích lại 415 ảnh từ chúng và tách thành tập hồi quy riêng. Vì vậy, 415 ảnh không phải tổng dữ liệu cũ và không cộng thêm vào tổng 24.221 của lần trước.', '',
             f"**Bộ ảnh gốc vẫn còn {integer(sum(old_image_counts.values()))} PNG trên máy:** {integer(old_image_counts['real'])} ảnh thật và {integer(old_image_counts['fake'])} ảnh giả. Em giữ bộ ảnh gốc, không ghi đè bằng bộ trích lại.", '',
             'Em dùng 28 video cũ có mã người từ tên video. Sáu video cũ có tên chung chung bị loại khỏi lần train này vì chưa xác định được nhóm người. Năm video test cũ giữ riêng để kiểm tra hồi quy và không đưa vào train lần này.', '',
             '### Chi tiết nguồn ảnh trong bộ sau (30/9)', '',
             '| Nguồn | Train | Validation | Test mới | Hồi quy cũ | Tổng chọn lại |',
             '|---|---:|---:|---:|---:|---:|']
    for source, title in [('original', 'Trích lại từ video cũ'), ('drive-20260930', 'Trích từ video mới')]:
        counts = source_frames[source]
        lines.append('| '+title+' | '+' | '.join(integer(counts[k]) for k in ('train', 'validation', 'test', 'regression_old'))+' | '+integer(sum(counts.values()))+' |')
    totals = {split: sum(counts[split] for counts in source_frames.values()) for split in ('train', 'validation', 'test', 'regression_old')}
    lines += ['| Tổng lần này | '+' | '.join(integer(totals[k]) for k in ('train', 'validation', 'test', 'regression_old'))+' | '+integer(sum(totals.values()))+' |', '',
             '**415 ảnh là phần hồi quy từ 5 video cũ.** Tổng ảnh chọn lại từ video cũ là 2.165, trong đó 1.750 ảnh dùng train. Tổng 7.029 ảnh của lần này bao gồm cả train và các tập đánh giá, chỉ 4.113 ảnh thực sự vào train.', '',
             '| Tập | Video có ảnh hợp lệ | Ảnh hợp lệ | Người/mã người |',
             '|---|---:|---:|---|']
    for name, title in [('train','Train'),('validation','Validation'),('test','Test mới'),('regression_old','Hồi quy 5 video cũ, không train')]:
        stat=manifest['summary'][name]
        lines.append(f"| {title} | {stat['videos']} | {stat['frames']} | {', '.join(stat['subjects'])} |")
    lines += ['', 'Nga và Phúc nằm trong test mới, B và Thu nằm trong validation. Model mới không học ảnh của bốn mã người này. Danh tính nhóm dựa vào tên thư mục/tên video do nhóm cung cấp; chưa xác nhận độc lập rằng mỗi tên khác nhau là một người riêng.', '',
              'Em trích lại mặt bằng YuNet và bộ lọc chất lượng của API, giữ đầu vào BGR 32×32 chia 255. Em lấy mẫu trải trên video, tối đa 100 khung/video và cách nhau ít nhất 0,2 giây. Giới hạn lấy mẫu làm số ảnh chọn lại nhỏ hơn nhiều so với bộ ảnh cũ. Sự chênh lệch này không có nghĩa phần lớn ảnh cũ bị loại vì chất lượng kém.', '',
              f"Riêng 28 video cũ: lấy mẫu {integer(old_stats['sampled'])} khung, giữ {integer(old_stats['accepted'])} và loại {integer(old_stats['sampled']-old_stats['accepted'])}. Phần bị loại gồm {old_stats['no_face']} khung không có mặt, {old_stats['multiple_faces']} khung có nhiều mặt và {old_stats['quality_rejected']} khung không đạt chất lượng.", '',
              'Mỗi video có trọng số bằng nhau trong cùng lớp để video nhiều ảnh không chiếm ưu thế trong loss train. Giới hạn 100 khung có thể bỏ qua một số biến thiên hữu ích. Đây là điểm cần thử lại, chưa có bằng chứng nó là nguyên nhân làm kết quả hồi quy giảm.', '',
              '## 2. Quá trình huấn luyện', '',
              '| Thông số | Trước | Sau |', '|---|---:|---:|',
              f"| Epoch đã chạy | {baseline['epochs_run']} | {report['epochs_run']} |",
              f"| Epoch chọn model | {baseline['best_epoch']} | {report['best_epoch']} |",
              f"| Loss train tại epoch chọn model, có augmentation | {decimal(baseline['historical_augmented_train_loss_at_best'])} | {decimal(report['history_at_best_epoch']['loss'])} |",
              f"| Loss validation tại epoch chọn model | {decimal(baseline['historical_validation_loss_at_best'])} | {decimal(report['history_at_best_epoch']['val_loss'])} |", '',
              'Model mới giữ kiến trúc CNN LivenessNet tương thích API. Train từ đầu với seed 42, Adam 0,0001, batch 32, tối đa 50 epoch. Chọn checkpoint theo validation loss, dừng sớm sau 8 epoch không cải thiện. Test không tham gia chọn model/ngưỡng.', '',
              '**Loss lịch sử ở bảng trên dùng hai bộ dữ liệu khác nhau. Bảng dưới mới là so sánh công bằng: cùng khung gốc, không augmentation, cùng công thức cross-entropy và chế độ inference.**', '',
              '## 3. Loss đánh giá trên cùng dữ liệu', '',
              'Cả hai model được đánh giá trên các tập train/validation/test/hồi quy của bộ sau ngày 30/9. Loss trong mục này tính ở chế độ suy luận trên ảnh gốc của từng tập, không augmentation. Đây không phải loss khi đang train ở epoch 11. Với model mới, loss train có augmentation tại epoch 11 là 0,13709, còn loss đánh giá trên tập train là 0,04635.', '',
              '| Tập đánh giá | Trước | Sau |', '|---|---:|---:|']
    for name,title in [('train','Train'),('validation','Validation'),('test','Test mới'),('regression_old','Hồi quy cũ')]:
        lines.append(f"| {title} | {decimal(before[name]['loss'])} | {decimal(after[name]['loss'])} |")
    lines += ['', '### Precision, recall và F1 macro trên từng tập', '',
              '| Tập | Precision trước/sau | Recall trước/sau | F1 trước/sau |', '|---|---:|---:|---:|']
    for name,title in [('train','Train'),('validation','Validation'),('test','Test mới'),('regression_old','Hồi quy cũ')]:
        left=before[name]['classification_report']['macro avg']
        right=after[name]['classification_report']['macro avg']
        lines.append('| '+title+' | '+' | '.join(f"{pct(left[key])} / {pct(right[key])}" for key in ('precision','recall','f1-score'))+' |')
    lines += ['', 'Macro cho hai lớp trọng số bằng nhau. Đây là suy luận trên ảnh gốc, không augmentation.']
    lines += ['', '## 4. Precision, recall và F1 trên cùng test mới', '',
              '**Hai cột là hai model, cùng đánh giá lại trên test ngày 30/9 gồm 1.473 ảnh.** Đây không phải điểm của model cũ trên test 15/9 đem so với điểm model mới trên test 30/9.', '',
              f"Để đối chiếu lịch sử, model ngày 15/9 từng đạt F1 macro {pct(prior_test['macro avg']['f1-score'])} và accuracy {pct(prior_test['accuracy'])} trên test cũ {integer(prior_split['test']['frames'])} ảnh. Các điểm lịch sử này dùng một tập test khác, không so trực tiếp với điểm test mới để kết luận tốt hơn.", '',
              '| Lớp | Precision trước | Precision sau | Recall trước | Recall sau | F1 trước | F1 sau |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key,title in [('fake','Giả mạo'),('real','Người thật'),('macro avg','Macro (hai lớp bằng trọng số)')]:
        values=[bp[key]['precision'],ap[key]['precision'],bp[key]['recall'],ap[key]['recall'],bp[key]['f1-score'],ap[key]['f1-score']]
        lines.append('| '+title+' | '+' | '.join(pct(v) for v in values)+' |')
    lines += ['', '| Chỉ số bổ sung | Trước | Sau |', '|---|---:|---:|',
              f"| Accuracy ảnh | {pct(before['test']['accuracy'])} | {pct(after['test']['accuracy'])} |",
              f"| Giả bị nhận thành thật (APCER) | {br['fake_as_real_count']}/{br['fake_count']} ({pct(br['APCER'])}) | {ar['fake_as_real_count']}/{ar['fake_count']} ({pct(ar['APCER'])}) |",
              f"| Thật bị nhận thành giả (BPCER) | {br['real_as_fake_count']}/{br['real_count']} ({pct(br['BPCER'])}) | {ar['real_as_fake_count']}/{ar['real_count']} ({pct(ar['BPCER'])}) |",
              f"| ACER (trung bình hai tỷ lệ lỗi) | {pct(br['ACER'])} | {pct(ar['ACER'])} |",
              f"| Accuracy theo video, trung bình điểm trong video | {pct(before['test']['video_level']['accuracy'])} | {pct(after['test']['video_level']['accuracy'])} |", '',
              'Các chỉ số theo ảnh chỉ dùng ảnh mặt đạt bộ lọc và chọn lớp có điểm cao nhất (argmax). Dòng accuracy theo video lấy trung bình điểm trên các ảnh hợp lệ trong video rồi chọn lớp có điểm cao nhất. Cả hai phép đánh giá chưa áp dụng ngưỡng 0,8 hay quy tắc quyết định theo thời gian của API. Khung không có mặt hoặc không đạt chất lượng được báo riêng ở mục 6.', '',
              'Precision lớp Fake là tỷ lệ đúng trong các ảnh bị kết luận Fake. Recall lớp Fake là tỷ lệ ảnh giả được phát hiện. Precision cao vẫn có thể đi cùng bỏ sót nhiều giả mạo, nên cần đọc cả recall và số lỗi giả thành thật.', '',
              '## 5. Kết quả theo từng người test', '',
              '| Mã người | F1 macro trước | F1 macro sau | Accuracy trước | Accuracy sau |', '|---|---:|---:|---:|---:|']
    subjects_before={s['subject_id']:s for s in before['test']['per_subject']}
    for stat in after['test']['per_subject']:
        prior=subjects_before[stat['subject_id']]
        lines.append(f"| {stat['subject_id']} | {pct(prior['classification_report']['macro avg']['f1-score'])} | {pct(stat['classification_report']['macro avg']['f1-score'])} | {pct(prior['accuracy'])} | {pct(stat['accuracy'])} |")
    lines += ['', '## 6. Khung bị loại trước phân loại', '',
              '| Nhãn test | Khung đã lấy mẫu | Khung qua chất lượng | Không mặt/nhiều mặt/chất lượng không đạt |', '|---|---:|---:|---:|']
    for label in ('fake','real'):
        clips=[v for v in manifest['videos'] if v['split']=='test' and v['label']==label]
        sampled=sum(v['statistics'].get('sampled',0) for v in clips)
        valid=sum(v['statistics'].get('quality_valid',0) for v in clips)
        lines.append(f'| {label} | {sampled} | {valid} ({pct(valid/sampled)}) | {sampled-valid} |')
    runtime_path=run/'runtime_comparison.json'
    if runtime_path.exists():
        lines += ['', 'Đánh giá tổng hợp nhiều khung hình lưu riêng tại [runtime_comparison.json](runtime_comparison.json). Đây là phát lại offline theo mốc thời gian video đã lấy mẫu và cùng quy tắc temporal của API, không phải số đo FPS camera hay HTTP.']
    continuous_path=run/'continuous_runtime_comparison.json'
    if continuous_path.exists():
        continuous=json.loads(continuous_path.read_text(encoding='utf-8'))
        if (continuous['input_hashes']['manifest_sha256'] != manifest_hash
                or continuous['models']['before']['model_sha256'] != report['configuration']['baseline_model_sha256']
                or continuous['models']['after']['model_sha256'] != report['new_model_sha256']):
            raise ValueError('Continuous runtime artifacts do not match this training run')
        cb,ca=continuous['before'],continuous['after']
        lines += ['', '### Kiểm tra luồng giống đầu vào API', '',
                  'Em cố định 12 giây đầu của cả 16 video test, lấy tối đa 5 khung/giây. Mỗi khung đi qua cùng hàm resize/JPEG của SDK (cạnh tối đa 960, chất lượng 90), rồi chạy lại YuNet, bộ lọc chất lượng và tổng hợp nhiều khung cho cả hai model. Đây là phát lại video offline, không đo độ trễ HTTP hay FPS camera.', '',
                  '| Chỉ số trên tất cả khung lấy mẫu | Trước | Sau |', '|---|---:|---:|',
                  f"| Khung đã thử | {cb['all_sample_count']} | {ca['all_sample_count']} |",
                  f"| Đưa ra được Real/Fake (coverage) | {cb['conclusive_count']} ({pct(cb['conclusive_coverage'])}) | {ca['conclusive_count']} ({pct(ca['conclusive_coverage'])}) |",
                  f"| Kết luận đúng | {cb['correct_conclusive_count']} | {ca['correct_conclusive_count']} |",
                  f"| Kết luận sai | {cb['wrong_conclusive_count']} | {ca['wrong_conclusive_count']} |",
                  f"| Chưa kết luận/bị loại | {cb['no_decision_count']} | {ca['no_decision_count']} |", '',
                  '| Lỗi theo nhãn thật, mẫu số gồm cả khung chưa kết luận | Trước | Sau |', '|---|---:|---:|',
                  f"| Giả bị nhận thành thật | {cb['fake_as_real_count']}/{cb['per_true_class']['fake']['all_sample_count']} | {ca['fake_as_real_count']}/{ca['per_true_class']['fake']['all_sample_count']} |",
                  f"| Thật bị nhận thành giả | {cb['real_as_fake_count']}/{cb['per_true_class']['real']['all_sample_count']} | {ca['real_as_fake_count']}/{ca['per_true_class']['real']['all_sample_count']} |", '',
                  'Khung chưa kết luận không được tính là đúng. Lỗi thấp cần đọc cùng coverage: model từ chối kết luận mọi khung cũng sẽ có số quyết định sai thấp.', '',
                  'Chi tiết từng khung và từng video: [continuous_runtime_comparison.json](continuous_runtime_comparison.json).']
        lines += ['', f"Trong phép phát lại, tỷ lệ kết luận đúng trên tất cả khung thay đổi từ {pct(cb['correct_fraction_of_all_samples'])} thành {pct(ca['correct_fraction_of_all_samples'])}. Coverage thay đổi {points(ca['conclusive_coverage']-cb['conclusive_coverage'], signed=True)} điểm phần trăm. Đây là đánh giá luồng riêng, khác với F1 của ảnh mặt đã qua bộ lọc."]
    lines += ['', '## 7. Nhận xét và kết luận', '']
    if f1_delta > 0 and ar['ACER'] < br['ACER'] and ar['APCER'] <= br['APCER']:
        lines.append(f'**Model phân loại ảnh mặt đạt chất lượng tốt hơn trên tập test mới đã kiểm tra:** F1 macro tăng {points(f1_delta)} điểm phần trăm, accuracy thay đổi {points(accuracy_delta, signed=True)} điểm phần trăm và ACER giảm. Kết luận này áp dụng cho các video/người trong phép thử. Em chưa coi đây là bằng chứng đủ để dùng cho xác thực thực tế.')
    elif f1_delta <= 0 and ar['ACER'] >= br['ACER']:
        lines.append('**Chưa thấy model mới tốt hơn trên tập test mới:** F1 macro không tăng và ACER không giảm. Em giữ model mới riêng để phân tích lỗi, không đề nghị thay model đang dùng chỉ vì đã thêm dữ liệu.')
    else:
        lines.append('**Kết quả có đánh đổi:** một số chỉ số tăng nhưng mức lỗi thật/giả chưa cùng cải thiện. Cần cân nhắc riêng lỗi bỏ sót giả mạo và lỗi từ chối người thật trước khi chọn model.')
    if ar['APCER'] > br['APCER']:
        lines.append(f"\n**Lỗi giả lọt thành thật tăng từ {pct(br['APCER'])} thành {pct(ar['APCER'])}.** Dù một số chỉ số trung bình có thể tăng, kết quả này chưa phù hợp để thay model cho mục tiêu chống giả mạo.")
    regression_delta=after['regression_old']['accuracy']-before['regression_old']['accuracy']
    lines.append(f"\nTrên 415 ảnh của 5 video cũ giữ riêng, accuracy ảnh thay đổi từ {pct(before['regression_old']['accuracy'])} thành {pct(after['regression_old']['accuracy'])} ({points(regression_delta, signed=True)} điểm phần trăm). Các người trong nhóm hồi quy vẫn có thể xuất hiện ở train, nên đây là kiểm tra hồi quy theo video.")
    if regression_delta < 0:
        lines.append('Kết quả hồi quy giảm: model mới chưa giữ được toàn bộ chất lượng trên dữ liệu cũ; cần xem lỗi từng video trước khi đưa vào API đang dùng.')
        rb=before['regression_old']['presentation_attack_rates']
        ra=after['regression_old']['presentation_attack_rates']
        lines += ['', '| Lỗi trên ảnh mặt của 5 video cũ | Trước | Sau |', '|---|---:|---:|',
                  f"| Giả bị nhận thành thật | {rb['fake_as_real_count']}/{rb['fake_count']} | {ra['fake_as_real_count']}/{ra['fake_count']} |",
                  f"| Thật bị nhận thành giả | {rb['real_as_fake_count']}/{rb['real_count']} | {ra['real_as_fake_count']}/{ra['real_count']} |", '',
                  '**Kết luận tổng thể: cải thiện rõ trên dữ liệu mới, nhưng chưa tốt hơn đồng đều trên dữ liệu cũ và mới. Em giữ model mới làm bản ứng viên, chưa thay model cho API hiện tại.**', '',
                  'Lần thí nghiệm tiếp theo nên rà giới hạn lấy mẫu 100 khung/video, cân bằng dữ liệu cũ/mới và có cả điều kiện quay cũ lẫn mới trong validation. Sau khi chọn model theo validation, dùng một test độc lập mới tách theo người. Em không train nhiều lần rồi chọn bản theo kết quả test đã xem.']
    video_f1=report['test_video_paired_bootstrap']['metrics']['macro_f1']
    ci=video_f1['delta_95_percentile_interval']
    lines += ['', '### Đánh giá F1 theo video và độ bất định', '',
              f"Mỗi video có một quyết định, lấy trung bình điểm model trên các ảnh hợp lệ rồi chọn lớp có điểm cao nhất. F1 macro theo video tăng từ {pct(video_f1['before'])} lên {pct(video_f1['after'])}, chênh lệch {points(video_f1['delta'])} điểm phần trăm. Đây là đơn vị video, khác F1 theo ảnh ở mục 4.", '',
              f'Khoảng percentile 95% cho chênh lệch F1 theo video: [{points(ci[0])}; {points(ci[1])}] điểm phần trăm, dùng 5.000 lần lấy mẫu video ghép cặp. Khoảng này có điều kiện trên 16 video của 2 mã người đã thử. Video cùng người còn tương quan. Khoảng này không đo khả năng tổng quát hóa sang người hoặc điều kiện quay mới.', '',
              'Giới hạn còn lại:', '',
              '- Tập test mới chỉ gồm 2 mã người. Những ảnh trong cùng video tương quan với nhau.',
              '- Video giả mới dùng MOV, video thật mới dùng MP4. Model có thể học một phần đặc điểm thiết bị/định dạng quay. Resize về cùng kích thước không tự loại bỏ sai lệch này.',
              '- Nhãn lấy từ thư mục Real/Fake do nhóm cung cấp. Chưa có metadata đầy đủ về thiết bị, ảnh in hay video phát lại để phân tích từng kiểu tấn công.',
              '- Điểm softmax không phải xác suất đúng đã hiệu chỉnh. Các số liệu này không chứng minh hết overfit.', '',
              'Lần huấn luyện này thay cả nguồn dữ liệu, cách lấy mẫu, bộ lọc và cách chia tập. Kết quả trước–sau đánh giá hai model trên cùng ảnh, nhưng chưa tách riêng tác động của việc bổ sung video so với các thay đổi chuẩn bị dữ liệu khác.', '',
              'Bước tiếp theo: thu cả thật và giả bằng cùng thiết bị/cài đặt quay, ghi nhãn kiểu tấn công, thêm người test độc lập và kiểm tra toàn bộ luồng API/camera.', '',
              '## 8. File kết quả', '',
              '- [Model mới](liveness.h5) và [nhãn](le.pickle)',
              '- [Toàn bộ đánh giá JSON](evaluation.json)',
              '- [Lịch sử từng epoch](training.csv)',
              '- [Cấu hình và hash](configuration.json)',
              '- [Đường cong học](learning_curves.png)', '',
              '![Loss và accuracy theo epoch](learning_curves.png)', '',
              'Model gốc và model ngày 15/9 giữ nguyên. Model mới nằm trong thư mục lần train này; chưa đổi model cho API hiện tại.']
    if (run/'api_smoke.json').exists():
        smoke=json.loads((run/'api_smoke.json').read_text(encoding='utf-8'))
        if smoke['model_sha256'] != report['new_model_sha256']:
            raise ValueError('API smoke used a different candidate model')
        lines += ['', f"Model H5 mới đã nạp qua API thật bằng Flask test client: health, analyze, tạo/gửi/xóa session hoạt động trên {len(smoke['examples'])} ảnh mẫu. Đây là kiểm tra tích hợp nội bộ, không đo tốc độ mạng hay độ chính xác độc lập. [Chi tiết kiểm tra API](api_smoke.json)."]
    output=run/'report.md'
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
