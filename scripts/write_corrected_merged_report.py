"""Write a source-backed Vietnamese report after corrected-run evidence is complete.

No TensorFlow, predictions, test selection, dataset writes, or API activation.
Missing evidence, failed audit, or mismatched model provenance aborts reporting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("before", "failed", "after")
PHASE_NAMES = {"before": "Trước: chưa gộp 30/9", "failed": "Gộp ban đầu", "after": "Sau: đã sửa"}
SOURCES = ("legacy-cache", "drive-20260930")
SOURCE_NAMES = {"legacy-cache": "Ảnh cache cũ", "drive-20260930": "Ảnh từ video Drive"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return f"{value:,}".replace(",", ".")


def decimal(value, places=5):
    return f"{value:.{places}f}".replace(".", ",")


def percent(value):
    return "Không xác định" if value is None else decimal(value * 100, 2) + "%"


def direction(before, after):
    return "giảm" if after < before else "tăng" if after > before else "giữ nguyên"


def link(path, title):
    target = Path(path).resolve().as_posix()
    return f"[{title}](<{target}>)"


def metric_row(title, result):
    macro = result["classification_report"]["macro avg"]
    return "| " + title + " | " + " | ".join([
        decimal(result["loss"]), percent(macro["precision"]),
        percent(macro["recall"]), percent(macro["f1-score"]), percent(result["accuracy"]),
    ]) + " |"


def rate_cell(result, count, denominator, rate):
    return f"{number(result[count])}/{number(result[denominator])} ({percent(result[rate])})"


def optional_http_evidence(run: Path, model_hash: str):
    smoke_path, benchmark_path = run / "http_smoke.json", run / "http_benchmark.json"
    if not smoke_path.exists() and not benchmark_path.exists():
        return None
    require(smoke_path.is_file() and benchmark_path.is_file(), "HTTP evidence is only partially written")
    smoke, benchmark = read(smoke_path), read(benchmark_path)
    require(smoke["passed"] is True and smoke["server_stopped"] is True, "HTTP smoke failed or test server was not stopped")
    require(smoke["model_activated_for_default_api"] is False, "HTTP test activated the default API model")
    require(smoke["model_sha256"] == model_hash == smoke["health"]["model"]["sha256"] == benchmark["model"]["sha256"], "HTTP benchmark/smoke used a different model")
    require(benchmark["kind"] == "local_single_frame_serial_latency", "Unexpected benchmark scope")
    require(benchmark["iterations"] == 10 and benchmark["warmup_per_path"] == 5, "HTTP benchmark sampling plan changed")
    require(all(benchmark["comparison"].values()), "Direct and HTTP inputs/results do not match")
    require(len(smoke["examples"]) == 4, "HTTP integration example count changed")
    for example in smoke["examples"]:
        require(example["analyze_and_session_routes_passed"] is True, "HTTP route integration failed")
        require(len(example["statuses"]) == 8, "HTTP sample response count changed")
        require(all(value == model_hash for value in example["response_model_hashes"]), "HTTP response model binding failed")
        require(all(value == example["jpeg_sha256"] for value in example["response_image_hashes"]), "HTTP response image binding failed")
    require(benchmark["input"]["sha256"] == smoke["examples"][0]["jpeg_sha256"], "HTTP benchmark image changed")
    for key in ("direct_decode_and_pipeline", "http_client", "server_processing", "server_total"):
        result = benchmark[key]
        require(result["count"] == 10, "HTTP latency observation count changed")
        require(all(math.isfinite(result[field]) and result[field] >= 0 for field in ("median_ms", "p95_ms", "min_ms", "max_ms")), "Invalid measured HTTP latency")
    require(sum(benchmark["status_counts"].values()) == 10, "HTTP benchmark status denominator changed")
    return smoke, benchmark


def validate_inputs(run: Path):
    e = read(run / "evaluation.json")
    selection = read(run / "selection.json")
    training = read(run / "training_summary.json")
    anchor = read(run / "anchor_comparison.json")
    api = read(run / "api_smoke.json")
    replay = read(run / "continuous_runtime_comparison.json")
    audit = read(run / "audit.json")
    failed_run = ROOT / "outputs/merged-20261001"
    prior_run = ROOT / "outputs/retrain-20260930"
    failed = read(failed_run / "evaluation.json")
    failed_anchor = read(failed_run / "anchor_comparison.json")
    failed_replay = read(failed_run / "continuous_runtime_comparison.json")
    investigation = ROOT / "outputs/merged-investigation-20261001"
    data_audit = read(investigation / "independent_data_audit.json")
    controls = read(investigation / "bn/diagnosis.json")
    groupnorm_protocol = read(investigation / "groupnorm_protocol.json")
    config = e["configuration"]
    model_hashes = {
        "before": sha(prior_run / "liveness.h5"),
        "failed": sha(failed_run / "liveness.h5"),
        "after": sha(run / "liveness.h5"),
    }
    require(audit.get("passed") is True, "Corrected-run independent audit is not passed")
    require(selection == e["selection"], "Evaluation and saved selection differ")
    require(training == e["training_summary"], "Evaluation and training summary differ")
    require(config["selection_sha256"] == sha(run / "selection.json"), "Selection file changed")
    require(selection["test_used_for_selection"] is False, "Test was used to choose candidate")
    require(len(selection["candidates"]) == 2, "Expected the two predeclared validation trials")
    require(selection["candidate_model_sha256"] == model_hashes["after"] == e["new_model_sha256"], "Corrected model mismatch")
    require(training["model_sha256"] == model_hashes["after"], "Training model hash mismatch")
    require(training["configuration_sha256"] == sha(run / "configuration.json"), "Training configuration changed")
    require(config["baseline_model_sha256"] == model_hashes["before"], "Before model mismatch")
    require(config["failed_merge_model_sha256"] == model_hashes["failed"] == failed["new_model_sha256"], "Failed-merge model mismatch")
    require(config["dataset_manifest_sha256"] == selection["dataset_manifest_sha256"] == failed["configuration"]["dataset_manifest_sha256"], "Merged data provenance changed")
    require(controls["model_sha256"] == model_hashes["failed"] and controls["dataset_manifest_sha256"] == selection["dataset_manifest_sha256"], "BN controls used different model/data")
    require(controls["test_evaluated"] is False, "BN controls unexpectedly used test")
    original_control_loss = controls["original"]["balanced_validation_loss"]["mean"]
    for trial in controls["trials"].values():
        require(trial["trainable_weights_unchanged"] is True, "BN control changed classifier weights")
        require(trial["balanced_validation_loss"]["mean"] > original_control_loss, "BN controls no longer support the stated rejection")
    initial_checks = data_audit["checks"]
    for key in ("npz_sha256_matches_manifest", "array_shape_matches_manifest", "pixel_hash_rows_aligned", "unique_pixel_hashes"):
        require(initial_checks[key] is True, "Initial numerical/data audit failed")
    for boundaries in ("video_split_overlaps", "drive_subject_split_overlaps"):
        require(all(not value for value in initial_checks[boundaries].values()), "Initial data boundary audit failed")
    require(data_audit["metrics"]["test"]["after"]["all"]["confusion_matrix"] == failed["after"]["test"]["confusion_matrix"], "Initial independent confusion matrix does not match failed merge")
    require(groupnorm_protocol["test_used_for_new_trial_selection"] is False, "Normalization protocol used test")
    expected_counts = {"train": 20404, "validation": 3697, "test": 4856}
    require(sum(part["frames"] for part in e["dataset_summary"].values()) == 28957, "Merged total changed")
    for split, count in expected_counts.items():
        require(e["dataset_summary"][split]["frames"] == count, f"{split} count changed")
        for phase in PHASES:
            require(e[phase][split]["frame_count"] == count, f"{phase}/{split} count mismatch")
            require(e[phase][split]["label_order"] == ["fake", "real"], "Unexpected model label order")
    candidates = selection["candidates"]
    for item in candidates:
        path = ROOT / item["run_directory"]
        trial_summary = read(path / "training_summary.json")
        require(sha(path / "liveness.h5") == item["candidate_model_sha256"], "Trial model changed")
        require(sha(path / "training_summary.json") == item["training_summary_sha256"], "Trial summary changed")
        require(sha(path / "configuration.json") == item["configuration_sha256"], "Trial configuration changed")
        require(trial_summary["validation_only_trial"] is True and trial_summary["test_predictions_or_metrics_created"] is False, "Trial used test")
        require(math.isclose(trial_summary["minimum_weighted_validation_ce"], item["validation_source_class_video_loss"], abs_tol=1e-6), "Trial validation score changed")
    chosen = min(candidates, key=lambda row: row["validation_source_class_video_loss"])
    require(chosen["run_directory"] == selection["selected_run_directory"], "Candidate did not minimize validation criterion")
    require(chosen["candidate_model_sha256"] == model_hashes["after"], "Selected trial was not exported")
    for phase in ("before", "after"):
        require(replay["models"][phase]["model_sha256"] == model_hashes[phase], "Replay model mismatch")
        require(replay[phase]["all_sample_count"] == 942, "Replay sample count changed")
    require(failed_replay["models"]["before"]["model_sha256"] == model_hashes["before"], "Failed replay baseline mismatch")
    require(failed_replay["models"]["after"]["model_sha256"] == model_hashes["failed"], "Failed replay model mismatch")
    require(failed_replay["after"]["all_sample_count"] == 942, "Failed replay count changed")
    require(anchor["baseline_model_sha256"] == model_hashes["before"] and anchor["candidate_model_sha256"] == model_hashes["after"], "Anchor model mismatch")
    require(failed_anchor["candidate_model_sha256"] == model_hashes["failed"], "Failed anchor mismatch")
    for group, count in (("new", 1473), ("old", 415)):
        require(anchor["groups"][group]["after"]["frame_count"] == count, "Anchor count changed")
        require(failed_anchor["groups"][group]["after"]["frame_count"] == count, "Failed anchor count changed")
    require(api["model_sha256"] == model_hashes["after"] and api["health_http_status"] == 200, "API model/health mismatch")
    require(len(api["examples"]) == 4, "Expected four integration examples")
    require(api["model_activated_for_current_api"] is False, "Unexpected default API activation")
    for example in api["examples"]:
        require(example["all_http_statuses_ok"] and example["exact_image_hash_binding_ok"], "API integration sample failed")
    optional_http_evidence(run, model_hashes["after"])
    return e, selection, training, anchor, api, replay, audit, failed, failed_anchor, failed_replay, data_audit, controls, groupnorm_protocol


def build_report(run: Path):
    (e, selection, training, anchor, api, replay, audit, failed,
     failed_anchor, failed_replay, data_audit, controls, groupnorm_protocol) = validate_inputs(run)
    config = e["configuration"]
    sources = e["source_split_summary"]
    before_training = config["baseline_training"]
    prior_evaluation = read(ROOT / "outputs/retrain-20260930/evaluation.json")
    require(prior_evaluation["new_model_sha256"] == config["baseline_model_sha256"], "Historical dataset summary belongs to another model")
    prior_counts = {split: summary["frames"] for split, summary in prior_evaluation["dataset_summary"].items()}
    require(prior_counts == {"train": 4113, "validation": 1028, "test": 1473, "regression_old": 415}, "Before-run dataset summary changed")
    legacy_exclusions = [record for record in failed["dataset_exclusions"] if record.get("source") == "legacy-cache" and record.get("reason") == "legacy_crop_quality_rejected"]
    quality_counts = Counter(record["quality_reason"] for record in legacy_exclusions)
    require(len(legacy_exclusions) == 189 and quality_counts == {"Reduce glare": 158, "Hold still / improve focus": 31}, "Legacy quality counts changed")
    selected_groupnorm = "groupnorm" in selection["selected_run_directory"]
    test = {phase: e[phase]["test"] for phase in PHASES}
    rates = {phase: test[phase]["presentation_attack_rates"] for phase in PHASES}
    f1 = {phase: test[phase]["classification_report"]["macro avg"]["f1-score"] for phase in PHASES}
    lines = ["# Báo cáo rà soát và sửa lần huấn luyện dữ liệu gộp", "",
             "Ngày báo cáo: 01/10/2026.", "",
             "Sếp, em đã kiểm tra lại số liệu, dữ liệu và cách huấn luyện. Báo cáo này phân biệt rõ ba mốc:", "",
             "- **Trước:** model ngày 30/9, trước khi gộp toàn bộ ảnh cũ.",
             "- **Gộp ban đầu:** model gộp ngày 01/10 đã báo cáo, có chất lượng giảm.",
             "- **Sau:** model sửa cách huấn luyện, chọn bằng validation trong hai phương án đã định trước.", "",
             "Ba model được đánh giá trên cùng ảnh và ranh giới tập. Em giữ nguyên nhãn, toàn bộ dữ liệu gộp và tập kiểm tra; không bỏ ảnh vì model đoán sai.", "",
             "## 1. Rà số liệu và tìm nguyên nhân", "",
             "Em tính lại precision, recall, F1 và cross-entropy trực tiếp từ mảng dự đoán đã lưu. Kết quả trước đây đúng; nhãn của hai model đều theo thứ tự `fake, real`. Toàn bộ 28.957 hàng ảnh khớp hash pixel, hash archive, thứ tự và kích thước. Không có pixel trùng chính xác hoặc video giao giữa các tập.", "",
             "Lỗi chủ yếu là ảnh giả bị nhận thành thật. Những video replay khó đã được xem lại toàn khung: có điện thoại phát lại, nên không đổi nhãn hoặc loại chúng để làm đẹp kết quả.", "",
             "| Ví dụ khó | Sai trước | Sai gộp ban đầu | Tổng ảnh |", "|---|---:|---:|---:|",
             "| Train: tuyen_fake_213 | 585 | 723 | 1.863 |",
             "| Validation: truong_fake_213 | 41 | 157 | 642 |",
             "| Test: thang_fake_732025 | 135 | 258 | 310 |",
             "| Test: Nga 3.MOV | 1 | 52 | 91 |", "",
             "**Vấn đề trong cách train:** batch lấy ảnh theo số lượng, nên 88,36% ảnh là cache cũ; nhưng sample weight lại giao 54,77% ảnh hưởng của loss cho nguồn mới. Trọng số mỗi ảnh chênh tới 52,75 lần. CNN có 6 lớp BatchNorm, trong khi thống kê BatchNorm được cập nhật từ ảnh trong batch, không theo sample weight.", "",
             "BatchNorm dùng thống kê batch khi train và trung bình tích lũy khi suy luận. Vì vậy, cách lấy batch có thể ảnh hưởng đến kết quả chạy thật. Đây là cơ chế giải thích có cơ sở; chưa chứng minh nó là nguyên nhân duy nhất. [Tài liệu Keras về BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/).", "",
             "### Kiểm chứng riêng giả thuyết BatchNorm", "",
             "Em thử chỉ hiệu chỉnh thống kê BatchNorm bằng ảnh train, giữ nguyên trọng số học được, tắt Dropout và chỉ chấm validation. Không dùng test. Cả hai cách hiệu chỉnh đều làm validation kém hơn nên bị loại:", "",
             "| Phương án | Loss validation cân bằng nguồn/lớp/video | Loss validation theo ảnh |", "|---|---:|---:|"]
    lines.append(f"| Không hiệu chỉnh | {decimal(controls['original']['balanced_validation_loss']['mean'])} | {decimal(controls['original']['validation']['loss'])} |")
    for key, title in (("uniform", "Hiệu chỉnh bằng batch ảnh thông thường"), ("source_class_video_balanced", "Hiệu chỉnh bằng batch cân bằng")):
        result = controls["trials"][key]
        lines.append(f"| {title} | {decimal(result['balanced_validation_loss']['mean'])} | {decimal(result['validation']['loss'])} |")
    lines += ["", "Kết quả này bác bỏ cách sửa đơn giản bằng hiệu chỉnh BatchNorm. Việc học đặc trưng và cách tối ưu cũng cần sửa; em không quy toàn bộ lỗi cho thống kê BatchNorm.", "",
              "## 2. Sửa cách huấn luyện, giữ nguyên dữ liệu", "",
              "### Dữ liệu model trước đã dùng và dữ liệu để so sánh", "",
              "Model trước ngày 30/9 dùng bộ **7.029 ảnh**; trong đó chỉ **4.113 ảnh dùng train**, 1.028 validation, 1.473 test mới và 415 ảnh kiểm tra video cũ riêng. Model trước chưa train bằng 20.404 ảnh của bộ gộp.", "",
              "| Tập | Bộ trước ngày 30/9 | Bộ gộp dùng ở hai lần train sau |", "|---|---:|---:|",
              "| Train | 4.113 | 20.404 |", "| Validation | 1.028 | 3.697 |",
              "| Test chính | 1.473 | 4.856 |", "| Kiểm tra video cũ riêng | 415 | Nằm trong nguồn test chung; chấm bổ sung 415 ảnh cố định |",
              "| Tổng bộ dữ liệu | 7.029 | 28.957 |", "",
              "Để so trước/sau công bằng, em nạp lại model trước và chấm trên cùng các tập của bộ gộp: 20.404 train, 3.697 validation và 4.856 test. Các hàng “Trước” trong bảng đánh giá là chấm lại, không phải số ảnh model trước đã học hoặc điểm của test khác.", "",
              "Bộ cache gốc **24.221 ảnh vẫn được giữ nguyên**. Bộ gộp giữ 24.032 ảnh cũ; 189 ảnh không đưa vào bộ gộp gồm 158 ảnh lóa và 31 ảnh thiếu nét, không xóa khỏi dữ liệu gốc. Với 56 video mới, lấy 5.138 khung ứng viên và giữ 4.925 ảnh sau kiểm tra; đã bỏ giới hạn 100 khung/video.", "",
              "### Dữ liệu gộp giữ nguyên giữa lần lỗi và lần sửa", "",
              "| Tập dữ liệu gộp | Số ảnh |", "|---|---:|",
              "| Train | 20.404 |", "| Validation | 3.697 |", "| Test | 4.856 |", "| Tổng | 28.957 |", "",
              "| Nguồn | Train | Validation | Test | Tổng |", "|---|---:|---:|---:|---:|"]
    for source in SOURCES:
        counts = [sources[source][split]["frames"] for split in ("train", "validation", "test")]
        lines.append("| " + SOURCE_NAMES[source] + " | " + " | ".join(number(value) for value in counts) + " | " + number(sum(counts)) + " |")
    lines += ["", "Em đã thay batch bằng bốn nhóm: cũ–giả, cũ–thật, mới–giả và mới–thật. Mỗi batch 32 ảnh có đúng 8 ảnh mỗi nhóm. Trong từng nhóm, lấy lượt đều theo video và dùng vòng ảnh xáo trộn liên tục; không cộng thêm sample weight. Như vậy, nguồn cũ/mới và nhãn thật/giả đều có 50% trong batch, thay cho sự lệch giữa phân phối ảnh và loss.", "",
              "Hai phương án đều khởi tạo model mới với seed 42, giữ đầu vào BGR 32×32 /255, augmentation, optimizer và cách chọn checkpoint. Mỗi phương án có một lượt warmup qua toàn bộ 20.404 ảnh train. Warmup không được chọn làm checkpoint. Sau đó chạy các epoch cân bằng; một epoch cân bằng có 638 batch, tương đương 20.416 lượt lấy ảnh, có thể lặp ảnh ngắn và chưa đi hết ảnh dài. Vòng ảnh được giữ qua các epoch; warmup đảm bảo không bỏ ảnh train nào.", "",
              "- **Phương án 1:** giữ LivenessNet với BatchNorm, sửa cách lấy batch và tiêu chí validation.",
              "- **Phương án 2:** giữ cùng quy trình phương án 1; chỉ đổi phần chuẩn hóa: 5 lớp BatchNorm tích chập thành GroupNorm (8 nhóm), lớp BatchNorm ở Dense thành LayerNorm.", "",
              "GroupNorm chuẩn hóa theo nhóm kênh trong từng mẫu thay vì dùng thống kê chung của batch. Việc thử phương án này nhằm kiểm tra ảnh hưởng của phần chuẩn hóa; không mặc định nó luôn tốt hơn. [Tài liệu Keras về GroupNormalization](https://keras.io/api/layers/normalization_layers/group_normalization/).", "",
              "### Chọn model bằng validation trước khi chấm test", "",
              "Tiêu chí là cross-entropy trung bình cân bằng theo nguồn, nhãn và video: trong mỗi nhóm nguồn/nhãn, lấy loss trung bình từng video rồi lấy trung bình các video; cuối cùng lấy trung bình bốn nhóm. Tiêu chí này tránh video dài hoặc nguồn nhiều ảnh lấn át. Loss này khác loss trung bình mọi ảnh ở bảng đánh giá bên dưới.", "",
              "| Phương án | Warmup | Epoch cân bằng chạy / chọn | Loss validation để chọn | Được chọn |", "|---|---:|---:|---:|---:|"]
    for candidate in selection["candidates"]:
        directory = ROOT / candidate["run_directory"]
        summary = read(directory / "training_summary.json")
        title = "Batch cân bằng + GroupNorm/LayerNorm" if "groupnorm" in candidate["run_directory"] else "Batch cân bằng + BatchNorm"
        chosen = candidate["run_directory"] == selection["selected_run_directory"]
        lines.append(f"| {title} | {summary['warmup_epochs']} | {summary['balanced_epochs_run']} / {summary['selected_balanced_epoch']} | {decimal(candidate['validation_source_class_video_loss'])} | {'Có' if chosen else 'Không'} |")
    lines += ["", f"Model sau được chọn: **{'batch cân bằng + GroupNorm/LayerNorm' if selected_groupnorm else 'batch cân bằng + BatchNorm'}**. Quyết định được ghi vào `selection.json` trước khi chấm test của model mới; không chọn ngưỡng hoặc checkpoint theo test.", "",
              "| Mốc model | Epoch chạy | Epoch checkpoint |", "|---|---:|---:|",
              f"| Trước gộp 30/9 | {before_training['epochs_run']} | {before_training['best_epoch']} |",
              f"| Gộp ban đầu | {failed['epochs_run']} | {failed['best_epoch']} |",
              f"| Sau sửa: epoch cân bằng | {training['balanced_epochs_run']} | {training['selected_balanced_epoch']} |",
              f"| Sau sửa: cộng warmup | {training['total_passes_including_warmup']} lượt | {training['warmup_epochs']} warmup + {training['selected_balanced_epoch']} cân bằng |", "",
              "Số epoch của các quy trình không tương đương hoàn toàn: model sau có warmup riêng và lấy batch cân bằng. Không coi việc chạy nhiều epoch hơn là bằng chứng tốt hơn.", "",
              "## 3. Loss, precision, recall và F1 trên cùng bộ dữ liệu", "",
              "P/R/F1 là macro: hai lớp có trọng số bằng nhau. Loss là cross-entropy trung bình ảnh gốc khi suy luận, không augmentation và không sample weight. Các tập trong bảng đều là của bộ gộp. Nhãn là argmax trên từng ảnh cắt, chưa áp dụng quyết định nhiều khung của API."]
    for split, title in (("train", "Train — 20.404 ảnh"), ("validation", "Validation — 3.697 ảnh"), ("test", "Test chung — 4.856 ảnh")):
        lines += ["", "### " + title, "", "| Model | Loss | Precision | Recall | F1 | Accuracy |", "|---|---:|---:|---:|---:|---:|"]
        lines += [metric_row(PHASE_NAMES[phase], e[phase][split]) for phase in PHASES]
    lines += ["", "Model trước đã train trên nguồn của bốn video đang nằm trong validation cũ. Điểm validation trước có phơi nhiễm nguồn và không phải đánh giá độc lập. Validation của hai phương án sửa nằm ngoài train của chính các lần train đó.", "",
              "## 4. Test theo nguồn và lỗi chống giả mạo", ""]
    for source in SOURCES:
        group = test["after"]["per_source"][source]
        lines += ["### " + SOURCE_NAMES[source] + f" — {number(group['frame_count'])} ảnh / {group['video_count']} video", "",
                  "| Model | Loss | Precision | Recall | F1 | Accuracy |", "|---|---:|---:|---:|---:|---:|"]
        lines += [metric_row(PHASE_NAMES[phase], test[phase]["per_source"][source]) for phase in PHASES]
        lines.append("")
    lines += ["| Lỗi trên test chung | Trước | Gộp ban đầu | Sau sửa |", "|---|---:|---:|---:|"]
    for title, count, denominator, rate in (("Giả bị nhận thành thật", "fake_as_real_count", "fake_count", "APCER"), ("Thật bị nhận thành giả", "real_as_fake_count", "real_count", "BPCER")):
        lines.append("| " + title + " | " + " | ".join(rate_cell(rates[phase], count, denominator, rate) for phase in PHASES) + " |")
    lines.append("| ACER — trung bình hai tỷ lệ lỗi | " + " | ".join(percent(rates[phase]["ACER"]) for phase in PHASES) + " |")
    lines += ["", "| Giả bị nhận thành thật theo nguồn | Trước | Gộp ban đầu | Sau sửa |", "|---|---:|---:|---:|"]
    for source in SOURCES:
        source_rates = [test[phase]["per_source"][source]["presentation_attack_rates"] for phase in PHASES]
        lines.append("| " + SOURCE_NAMES[source] + " | " + " | ".join(rate_cell(value, "fake_as_real_count", "fake_count", "APCER") for value in source_rates) + " |")
    lines += ["", "Các tỷ lệ trên là lỗi argmax của ảnh cắt; không có nghĩa API cho phép xác thực thật với cùng tỷ lệ. API còn bộ lọc đầu vào, trạng thái chưa kết luận, tracking và tổng hợp nhiều khung.", ""]
    bootstrap = e["test_video_paired_bootstrap"]
    interval = bootstrap["metrics"]["macro_f1"]["delta_95_percentile_interval"]
    lines += [f"Đánh giá một nhãn/video bằng trung bình xác suất: F1 {percent(test['before']['video_level']['classification_report']['macro avg']['f1-score'])} → {percent(test['after']['video_level']['classification_report']['macro avg']['f1-score'])}. Bootstrap ghép cặp theo {bootstrap['video_count']} video cho khoảng 95% của chênh lệch F1 theo video: {decimal(interval[0]*100, 2)} đến {decimal(interval[1]*100, 2)} điểm phần trăm. Đây là phép đo khác F1 theo ảnh; số video ít và video cùng người có thể tương quan."]
    if interval == [0.0, 0.0]:
        lines += ["", "Khoảng 0 đến 0 xuất hiện vì cả hai model đều đoán đúng nhãn trung bình của toàn bộ 21 video, nên kết quả bootstrap hiện tại không thay đổi. Điều này không chứng minh độ bất định bằng không hoặc model chắc chắn đúng trên video/người mới."]
    lines += ["",
              "## 5. Chấm lại đúng ảnh đã dùng ngày 30/9", "",
              "Giữ nguyên 1.473 ảnh video mới và 415 ảnh video cũ của báo cáo trước, chỉ đổi model. 415 ảnh cũ được cắt bằng YuNet, khác 3.343 ảnh cache SSD trong test chung. Đây là benchmark bổ sung; không cộng vào tổng 28.957 ảnh. Video và pixel của các nhóm này được kiểm tra nằm ngoài train.", "",
              "| Nhóm / model | Ảnh | F1 | Accuracy | Loss |", "|---|---:|---:|---:|---:|"]
    for key, title in (("new", "Video mới"), ("old", "Video cũ")):
        results = {"before": anchor["groups"][key]["before"], "failed": failed_anchor["groups"][key]["after"], "after": anchor["groups"][key]["after"]}
        for phase in PHASES:
            result = results[phase]
            lines.append(f"| {title}: {PHASE_NAMES[phase]} | {number(result['frame_count'])} | {percent(result['classification_report']['macro avg']['f1-score'])} | {percent(result['accuracy'])} | {decimal(result['loss'])} |")
    lines += ["", "## 6. Phát lại pipeline và kiểm tra API", "",
              "Phát lại cùng 12 giây đầu của 16 video test mới, tối đa 5 khung/giây. Ba model nhận cùng JPEG từ SDK, rồi detector, kiểm tra chất lượng và tổng hợp nhiều khung được chạy lại. Mỗi video bắt đầu với lịch sử mới. Tổng là 942 khung; đây là kiểm thử offline, chưa đo camera trực tiếp hoặc tốc độ mạng.", "",
              "| Kết quả replay | Trước | Gộp ban đầu | Sau sửa |", "|---|---:|---:|---:|"]
    replay_results = {"before": replay["before"], "failed": failed_replay["after"], "after": replay["after"]}
    for title, key in (("Tổng khung", "all_sample_count"), ("Kết luận đúng", "correct_conclusive_count"), ("Kết luận sai", "wrong_conclusive_count"), ("Chưa kết luận / bị loại", "no_decision_count")):
        lines.append("| " + title + " | " + " | ".join(number(replay_results[phase][key]) for phase in PHASES) + " |")
    lines.append("| Tỷ lệ đưa ra Real/Fake | " + " | ".join(percent(replay_results[phase]["conclusive_coverage"]) for phase in PHASES) + " |")
    lines += ["", "Khung chưa kết luận không được tính là đúng. Cần đọc lỗi sai cùng tỷ lệ đưa ra quyết định, vì từ chối nhiều có thể làm số lỗi thấp hơn.", "",
              f"Model sau được nạp thật qua Flask test client: health, analyze, tạo/gửi/xóa session trên {len(api['examples'])} JPEG mẫu. Các mã HTTP, hash model và hash ảnh đầu vào đều được kiểm tra. Đây chỉ là kiểm tra tích hợp, không phải bốn ca chứng minh độ chính xác hoặc khả năng chống giả mạo. JPEG lặp trong session dùng để kiểm tra trạng thái, không phải camera đang quay.", "",
              ]
    http = optional_http_evidence(run, e["new_model_sha256"])
    if http is not None:
        http_smoke, benchmark = http
        lines += ["### Kiểm tra qua HTTP thật trên máy local", "",
                  f"Em khởi động riêng server thử tại `{http_smoke['url']}`, gọi bằng SDK qua HTTP loopback trên 4 JPEG, kiểm tra analyze/session cùng hash ảnh và model. Server thử đã dừng sau kiểm tra; không đổi model API mặc định.", "",
                  "Phép đo độ trễ dùng **một ảnh lặp 10 lần tuần tự**, mỗi đường gọi warmup 5 lần, cấu hình 2 CPU thread. Ảnh JPEG được mã hóa một lần trước phần đo và dùng giống nhau cho gọi trực tiếp/HTTP. Đây không phải FPS camera, tốc độ mạng bên ngoài hoặc đo tải đồng thời.", "",
                  "| Đường đo | Số lần | Median (ms) | P95 (ms) |", "|---|---:|---:|---:|"]
        for key, title in (("direct_decode_and_pipeline", "Giải mã JPEG + pipeline trực tiếp"), ("http_client", "Client HTTP: gửi + nhận"), ("server_processing", "Phần xử lý pipeline trên server"), ("server_total", "Tổng thời gian phía server")):
            result = benchmark[key]
            lines.append(f"| {title} | {result['count']} | {decimal(result['median_ms'], 2)} | {decimal(result['p95_ms'], 2)} |")
        statuses = ", ".join(f"`{key}`: {count}" for key, count in sorted(benchmark["status_counts"].items()))
        lines += ["", "Trạng thái của 10 phản hồi benchmark: " + statuses + ". Độ trễ không chứng minh độ chính xác; trạng thái chưa kết luận không được hiểu là phát hiện đúng. Kết quả còn phụ thuộc tải CPU và cấu hình ở lần đo.", "",
                  "Median/P95 được trích từ số đo trong `http_benchmark.json`. File này không lưu toàn bộ vector thời gian từng lượt nên kiểm toán chỉ kiểm tra phạm vi, số lượng và tính hợp lý của bản tổng hợp; chưa tính lại độc lập các phân vị. Mẫu 10 lần còn nhỏ.", ""]
    lines += ["## 7. Nhận xét và kết luận", "",
              f"Trên cùng test gộp, F1: **{percent(f1['before'])} trước gộp → {percent(f1['failed'])} gộp ban đầu → {percent(f1['after'])} sau sửa**. Loss: {decimal(test['before']['loss'])} → {decimal(test['failed']['loss'])} → {decimal(test['after']['loss'])}.", ""]
    improved_from_failed = f1["after"] > f1["failed"] and rates["after"]["ACER"] < rates["failed"]["ACER"]
    better_than_before = f1["after"] > f1["before"] and rates["after"]["ACER"] < rates["before"]["ACER"] and rates["after"]["APCER"] <= rates["before"]["APCER"]
    if improved_from_failed:
        lines.append("**Cách train đã sửa giúp cải thiện đồng thời F1 và ACER so với lần gộp ban đầu trên benchmark đã kiểm tra.**")
    else:
        lines.append("**Cách train đã sửa chưa cải thiện đồng thời F1 và ACER so với lần gộp ban đầu; không kết luận đã giải quyết hoàn toàn.**")
    lines.append("")
    lines.append("**Trên test chung, model sau tốt hơn model trước gộp về F1 và ACER, đồng thời không tăng tỷ lệ bỏ lọt giả chung.** Kết quả này chưa có nghĩa tốt hơn ở mọi nguồn hoặc mọi cách sử dụng." if better_than_before else "**So với model trước gộp, chưa có cải thiện đồng thời F1, ACER và tỷ lệ bỏ lọt giả. Cần xem rõ phần đánh đổi.**")
    lines += ["", f"Tỷ lệ giả bị nhận thành thật {direction(rates['before']['APCER'], rates['after']['APCER'])}: {percent(rates['before']['APCER'])} trước gộp, {percent(rates['failed']['APCER'])} gộp ban đầu và {percent(rates['after']['APCER'])} sau sửa. Tỷ lệ thật bị nhận thành giả: {percent(rates['before']['BPCER'])} → {percent(rates['failed']['BPCER'])} → {percent(rates['after']['BPCER'])}."]
    for source in SOURCES:
        values = {phase: test[phase]["per_source"][source] for phase in PHASES}
        sf = {phase: values[phase]["classification_report"]["macro avg"]["f1-score"] for phase in PHASES}
        sr = {phase: values[phase]["presentation_attack_rates"] for phase in PHASES}
        lines += ["", f"{SOURCE_NAMES[source]}: F1 {percent(sf['before'])} → {percent(sf['failed'])} → {percent(sf['after'])}; giả bị nhận thành thật {percent(sr['before']['APCER'])} → {percent(sr['failed']['APCER'])} → {percent(sr['after']['APCER'])}. Đây là kết quả riêng nguồn, không che bằng điểm chung."]
    drive_before, drive_after = test["before"]["per_source"]["drive-20260930"], test["after"]["per_source"]["drive-20260930"]
    if (drive_after["classification_report"]["macro avg"]["f1-score"] < drive_before["classification_report"]["macro avg"]["f1-score"]
            or drive_after["presentation_attack_rates"]["APCER"] > drive_before["presentation_attack_rates"]["APCER"]):
        drive_f1_worse = drive_after["classification_report"]["macro avg"]["f1-score"] < drive_before["classification_report"]["macro avg"]["f1-score"]
        drive_apcer_worse = drive_after["presentation_attack_rates"]["APCER"] > drive_before["presentation_attack_rates"]["APCER"]
        worsened_metrics = "cả F1 và tỷ lệ bỏ lọt giả" if drive_f1_worse and drive_apcer_worse else "F1" if drive_f1_worse else "tỷ lệ bỏ lọt giả"
        lines += ["", f"**Riêng nguồn video Drive vẫn kém hơn model trước gộp về {worsened_metrics}**, dù đã phục hồi so với lần gộp ban đầu. Cải thiện điểm chung chủ yếu đến từ nguồn cache cũ."]
    rb, ra, rf = replay_results["before"], replay_results["after"], replay_results["failed"]
    lines += ["", f"Replay sau sửa có {number(ra['correct_conclusive_count'])} kết luận đúng, {number(ra['wrong_conclusive_count'])} kết luận sai và {number(ra['no_decision_count'])} khung chưa kết luận; so với trước gộp là {number(rb['correct_conclusive_count'])}/{number(rb['wrong_conclusive_count'])}/{number(rb['no_decision_count'])}. Tỷ lệ đưa ra Real/Fake: {percent(rb['conclusive_coverage'])} trước gộp → {percent(rf['conclusive_coverage'])} gộp ban đầu → {percent(ra['conclusive_coverage'])} sau sửa."]
    if ra["conclusive_coverage"] < rb["conclusive_coverage"]:
        lines += ["", "**Pipeline sau sửa còn đưa ra ít kết luận hơn model trước gộp trên replay này.** Dù F1 argmax test chung tăng, chưa thể nói trải nghiệm camera/API đã tốt hơn toàn diện."]
    if rates["after"]["BPCER"] > rates["failed"]["BPCER"]:
        lines += ["", f"So với lần gộp ban đầu, lỗi thật bị nhận thành giả tăng từ {percent(rates['failed']['BPCER'])} lên {percent(rates['after']['BPCER'])}; đây là phần đánh đổi khi giảm bỏ lọt giả. So với trước gộp, tỷ lệ này là {percent(rates['before']['BPCER'])} → {percent(rates['after']['BPCER'])}. Không kết luận mọi chỉ số đều cải thiện."]
    lines += ["", "Bằng chứng hiện tại cho thấy sửa cách lấy batch/tiêu chí chọn model và thử phần chuẩn hóa có thể thay đổi chất lượng. So sánh phương án 1 với 2 chỉ khác phần chuẩn hóa; so với lần gộp ban đầu còn thay cả cách lấy mẫu và chọn checkpoint. Vì vậy, không tách riêng phần đóng góp của từng thay đổi chỉ từ ba điểm test.", "",
              "**Model sau vẫn lưu riêng; em chưa thay model mặc định của camera/API.** Muốn dùng bản này cần chỉ định đúng model và nhãn trong lệnh khởi động ở cuối báo cáo.", "",
              "## 8. Giới hạn và việc tiếp theo", "",
              "- Test đã được xem trước và đã ảnh hưởng đến quyết định cải tiến; đây là benchmark phát triển cố định, chưa phải test cuối độc lập.",
              "- Người của dữ liệu cũ có thể xuất hiện ở nhiều tập; sáu tên video chung chung chưa xác định danh tính. Nguồn mới chỉ tách theo mã thư mục do người cung cấp đặt, chưa xác minh độc lập danh tính.",
              "- Validation trước phơi nhiễm bốn nguồn video cũ từng thuộc train; không dùng điểm đó để khẳng định khả năng tổng quát.",
              "- Ảnh cùng video tương quan. Lấy thêm nhiều ảnh từ cùng video chưa tăng số người, thiết bị và kiểu tấn công tương ứng.",
              "- Cache cũ dùng SSD, ảnh mới dùng YuNet; cách quay và codec thật/giả còn khác nhau. Model vẫn có thể học dấu hiệu nguồn hoặc thiết bị.",
              "- F1 argmax theo ảnh, F1 theo video và quyết định temporal của API là các phép đo khác nhau. Không gọi lỗi argmax là tỷ lệ API xác thực sai.",
              ("- Kiểm tra API gồm test client và HTTP loopback trên cùng máy; chưa đo gọi API từ máy khác, độ trễ mạng bên ngoài hoặc FPS camera trực tiếp của model sau." if http is not None else "- Kiểm tra API chỉ gồm test client nội bộ; chưa đo gọi API từ máy khác, độ trễ mạng hoặc FPS camera trực tiếp của model sau."), "",
              "Bước tiếp theo: sếp cho thu một tập hoàn toàn mới với người/thiết bị/ánh sáng mới, quay thật và giả cùng điều kiện, ghi kiểu tấn công. Giữ tập này ngoài mọi lần chỉnh model rồi đánh giá trước khi đổi model mặc định hoặc đưa API ra ngoài.", "",
              "## 9. Bằng chứng và sử dụng model", "",
              "Kiểm toán độc lập của lần sửa đã trả `passed: true`. Kiểm toán xác minh dữ liệu, ranh giới, nhãn, các mảng dự đoán, phép tính, lựa chọn validation, replay và mẫu tích hợp API; không thay thế test độc lập.", "",
              "- " + link(run / "liveness.h5", "Model sau sửa") + " và " + link(run / "le.pickle", "nhãn"),
              "- " + link(run / "evaluation.json", "Đánh giá ba mốc trên cùng dữ liệu"),
              "- " + link(run / "selection.json", "Quyết định chọn giữa hai phương án"),
              "- " + link(run / "training_summary.json", "Epoch, checkpoint và mức phủ ảnh train"),
              "- " + link(run / "anchor_comparison.json", "Ảnh cố định ngày 30/9"),
              "- " + link(run / "continuous_runtime_comparison.json", "Replay 942 khung"),
              "- " + link(run / "api_smoke.json", "Kiểm tra tích hợp API"),
              "- " + link(run / "audit.json", "Kiểm toán độc lập của lần sửa"),
              "- " + link(ROOT / "outputs/merged-investigation-20261001/independent_data_audit.json", "Rà dữ liệu và tính lại số liệu lần gộp ban đầu"),
              "- " + link(ROOT / "outputs/merged-investigation-20261001/bn/diagnosis.json", "Thử hiệu chỉnh BatchNorm bị loại"), "",
              "Lệnh chạy API với model sau sửa:", "", "```powershell",
              f".\\venv\\Scripts\\python.exe api_app.py --model {run.relative_to(ROOT).as_posix()}/liveness.h5 --labels {run.relative_to(ROOT).as_posix()}/le.pickle",
              "```", "",
              "![Đường cong huấn luyện](" + (run / "learning_curves.png").resolve().as_posix() + ")"]
    if http is not None:
        lines += ["", "Bằng chứng HTTP local: " + link(run / "http_smoke.json", "kiểm tra route/hash") + " và " + link(run / "http_benchmark.json", "đo độ trễ một ảnh lặp 10 lần") + "."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/merged-corrected-20261001")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    require(run.is_relative_to(ROOT), "Report directory must stay inside the workspace")
    output = run / "report.md"
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    report = build_report(run)
    require((run / "learning_curves.png").is_file(), "Learning curves artifact missing")
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
