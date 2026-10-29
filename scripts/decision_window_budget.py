"""48-hour production-window accounting and a uniform measured training budget."""
import json
import math
import os
from pathlib import Path
import time


def merged(intervals, start, end):
    result = []
    for lo, hi in sorted((max(start, lo), min(end, hi)) for lo, hi in intervals if hi > start and lo < end):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([lo, hi])
    return result


def subtract(intervals, busy):
    result = []
    for lo, hi in intervals:
        cursor = lo
        for a, b in busy:
            if b <= cursor or a >= hi:
                continue
            if a > cursor:
                result.append([cursor, a])
            cursor = max(cursor, b)
        if cursor < hi:
            result.append([cursor, hi])
    return result


def proc_started(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    return boot + int(fields[19]) / os.sysconf('SC_CLK_TCK')


def windows_48h(pipeline, now=None):
    from gpu_window_runner import processes
    now = now or time.time()
    start = now - 48 * 3600
    busy, eligible = [], []
    def record(document):
        if not isinstance(document, dict):
            return
        for phase, target in (('train', busy), ('pack', eligible), ('selection', eligible), ('evaluation', eligible)):
            lo, hi = document.get(phase+'_started'), document.get(phase+'_finished')
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                target.append((lo, hi))
        for candidate in document.get('candidates', []):
            record(candidate)
    for pattern in ('round_*/packed.json', 'round_*/trained.json', 'round_*/complete.json',
                    'round_*/seeds/*/trained.json', 'round_*/seeds/*/failed.json'):
        for path in Path(pipeline).glob(pattern):
            record(json.loads(path.read_text()))
    for pid, info in processes().items():
        if not any(str(pipeline) in arg for arg in info['argv']):
            continue
        if info['script'] in ('train_rollout_policy.py', 'pack_training_rows.py', 'select_student.py', 'promotion_gate.py'):
            try:
                target = busy if info['script'] == 'train_rollout_policy.py' else eligible
                target.append((proc_started(pid), now))
            except OSError:
                continue
    busy = merged(busy, start, now)
    eligible = subtract(merged(eligible, start, now), busy)
    busy_seconds = sum(hi-lo for lo, hi in busy)
    window_seconds = sum(hi-lo for lo, hi in eligible)
    return dict(start=start, end=now, hours=48, training_process_hours=busy_seconds/3600,
        gpu_idle_fraction_upper_bound=1-busy_seconds/(48*3600),
        eligible_window_hours=window_seconds/3600, eligible_fraction=window_seconds/(48*3600),
        definition='union(pack,selection,gate) minus training processes; unobserved controller gaps excluded; historical memory admission not subtracted')


def choose_budget(rates, training_rows, duty, epochs=12, margin=1.25, max_days=3):
    if duty <= 0 or len(rates) != 8 or any(rate <= 0 for rate in rates.values()):
        raise ValueError('eight measured rates and a positive observed window fraction are required')
    epoch_seconds = sum(training_rows / rate for rate in rates.values()) * margin
    original_days = epoch_seconds * epochs / (86400 * duty)
    chosen_epochs = epochs
    fraction = 1.0
    if original_days > max_days:
        chosen_epochs = max(1, min(epochs, math.floor(max_days * 86400 * duty / epoch_seconds)))
        if epoch_seconds * chosen_epochs > max_days * 86400 * duty:
            fraction = min(1.0, .98 * max_days * 86400 * duty / (epoch_seconds * chosen_epochs))
    per_job = {name: dict(rows_per_second=rate, full_gpu_hours=training_rows*epochs/rate/3600,
        planned_gpu_hours=training_rows*chosen_epochs*fraction/rate/3600) for name, rate in rates.items()}
    return dict(original_epochs=epochs, epochs=chosen_epochs, train_row_fraction=fraction, margin=margin,
        measured_training_rows=training_rows, original_seed1_days=original_days,
        planned_seed1_days=epoch_seconds*chosen_epochs*fraction/(86400*duty),
        original_seed1_gpu_hours=epoch_seconds*epochs/3600,
        planned_seed1_gpu_hours=epoch_seconds*chosen_epochs*fraction/3600,
        original_all_gpu_hours=epoch_seconds*epochs*2/3600,
        planned_all_gpu_hours=epoch_seconds*chosen_epochs*fraction*2/3600,
        planned_all_days=epoch_seconds*chosen_epochs*fraction*2/(86400*duty), per_job=per_job,
        reason='keep full row/terrain coverage and reduce epochs uniformly first; if one epoch still exceeds three days, apply the same stable row hash fraction to all arms/seeds')


def markdown(report):
    budget, windows = report['budget'], report['windows']
    lines = ['\n## 下一空档吞吐实测与统一预算（自动记录）\n',
        f"测量代码：`{report['revision']}`；近48小时记录到的可调度空档 {windows['eligible_window_hours']:.2f} h，比例 {windows['eligible_fraction']:.2%}。生产训练进程 {windows['training_process_hours']:.2f} h；GPU无训练进程比例上限 {windows['gpu_idle_fraction_upper_bound']:.2%}。折算采用可调度pack/挑选/门禁窗口比例，未把控制器空白算作可用时间；历史内存准入可能进一步缩短窗口。\n",
        '| 结构 | 实测行/秒 | 原12轮单作业GPU h | 预算后单作业GPU h |', '|---|---:|---:|---:|']
    for name, entry in budget['per_job'].items():
        lines.append(f"| {name} | {entry['rows_per_second']:.0f} | {entry['full_gpu_hours']:.2f} | {entry['planned_gpu_hours']:.2f} |")
    lines += ['', f"共同有效训练行 {budget['measured_training_rows']:,}；原seed1合计 {budget['original_seed1_gpu_hours']:.2f} GPU h / {budget['original_seed1_days']:.2f}日；16作业原合计 {budget['original_all_gpu_hours']:.2f} GPU h。总量含25%工作集/装载/续跑余量，表内单作业为纯实测速率推算。",
        f"统一选择 **{budget['epochs']} epoch、训练行比例 {budget['train_row_fraction']:.6f}**，所有8臂、两个种子完全相同。先减少轮数保留稀有地形和卡组覆盖；若一轮仍超过三天，再用同一固定行哈希抽样，留出集保持完整。预算后seed1约 {budget['planned_seed1_gpu_hours']:.2f} GPU h / {budget['planned_seed1_days']:.2f}日，全16作业约 {budget['planned_all_gpu_hours']:.2f} GPU h / {budget['planned_all_days']:.2f}日。",
        '测量使用按地形抽取的r42真实行、真实前后向/AdamW/微批累积与公开实体token，包含批次构造和传输；固定小工作集不能代表全语料缺页、全量token准备及CPU评测时间，后者另计。', '']
    if 'sidecar_max_bytes' in budget:
        lines += [f"磁盘约束进一步统一限制训练行比例：token缓存上限 {budget['sidecar_max_bytes']/(1<<30):.2f} GiB，按样本P95×1.5预计 {budget['estimated_sidecar_bytes']/(1<<30):.2f} GiB；预算时宿主最小余量 {budget['host_free_bytes']/(1<<30):.2f} GiB。仅缓存被采用的训练行，CPU留出逐档案生成token；全部decision作业完成后删除缓存和原帧硬链接视图，保留原始数据与模型交接。", '']
    if budget.get('disk_blocked'):
        lines += ['**宿主磁盘余量不足，以上训练预算尚未放行；吞吐测量已完成并发布，未创建训练ready标记。需释放磁盘后重新确定预算。**', '']
    return '\n'.join(lines)
