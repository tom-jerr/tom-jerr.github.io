"""DSpark execution panels. Geometry shares the article's Figure primitives.

Panel 1 (inference) follows DSparkWorkerV2._forward_decode at the pinned
commit: host picks K from the lagged relay ring during forward prepare, the
current block's confidence decides who gets the K on device, and the executor
verifies / accepts / commits. Panel 2 (host_budget) zooms into the relay and
the host-side budget search; function names annotate the corresponding boxes.
"""

MUTED = "#637083"


def inference(Figure):
    f = Figure(
        "SGLang DSpark：一轮 decode 的三个组件与滞后预算",
        "host 在 forward prepare 用滞后两槽的历史 confidence 选出预算 K；当前 decode 中 proposer 提出候选，planner 分配 K 并定布局，executor 验证、接受并提交；发布的结果进入下一轮的滞后读取。",
        768,
    )
    f.note(78, "固定源码 822e73ccddc0 · 箭头表示数据依赖，框宽不代表耗时 · 默认 min_verify_len = 1")
    # ── Host row ──────────────────────────────────────────────────────────
    f.text(32, 119, "Host / forward prepare（overlap 下先于本轮 GPU 执行）", 20, weight=650)
    f.chain(193, xs=[32, 304, 576, 848])
    f.box(32, 137, "滞后读取", ["slot = (ring_pos − 2) % 3", "copy_done.query() 非阻塞"], "neutral", h=112)
    f.text(152, 244, "ConfidenceRelay.resolve()", 16, color=MUTED, anchor="middle")
    f.box(304, 137, "历史置信度预算", ["_shift_to_lag → cumprod", "generation 失效 → 全 1"], "anchor", h=112)
    f.text(424, 244, "compute_budget()", 16, color=MUTED, anchor="middle")
    f.box(576, 137, "预算搜索", ["展平过滤排序 → 前缀和", "argmax(τ̂ × SPS) → K"], "anchor", h=112)
    f.text(696, 244, "compute_verify_token_budget()", 16, color=MUTED, anchor="middle")
    f.box(848, 137, "预算预先写入", ["draft_input", "verify_token_budget = K"], "neutral", h=112)
    f.text(968, 244, "overlap 下由 scheduler 调用", 16, color=MUTED, anchor="middle")
    # ── Device row ────────────────────────────────────────────────────────
    f.text(32, 330, "Device / 当前 decode step（DSparkWorkerV2._forward_decode）", 20, weight=650)
    # The prewritten K enters the planner without a fresh host<->device sync.
    f.arrow((968, 249), (968, 290), (434, 290), (434, 345))
    f.text(610, 282, "预先写入的 K；本轮无 host↔device 同步", 17)
    f.chain(420, xs=[32, 307, 582, 857], width=253)
    f.box(32, 345, "Proposer", ["一次并行 backbone", "序列头修正 → 采样"], "draft", w=253, h=150)
    f.text(158, 460, "propose()", 16, color=MUTED, anchor="middle")
    f.box(307, 345, "Planner", ["取回预算 → top-K 分配", "verify_lens + TP 广播"], "seq", w=253, h=150)
    f.text(433, 462, "resolve_verify_token_budget()", 15, color=MUTED, anchor="middle")
    f.text(433, 478, "schedule_layout()", 15, color=MUTED, anchor="middle")
    f.box(582, 345, "Executor · verify", ["ragged 行压紧 → bucket", "grammar mask → eager"], "target", w=253, h=150)
    f.text(708, 460, "run_compact / run_non_compact", 15, color=MUTED, anchor="middle")
    f.box(857, 345, "接受与提交", ["accept_and_finalize", "commit_hidden → KV 注入", "on_publish → relay"], "accept", w=253, h=150)
    # Candidates and corrected q bypass the allocator; only confidence and K
    # determine verify lengths. They meet the accept kernels at the executor.
    f.arrow((158, 495), (158, 585), (983, 585), (983, 495))
    f.text(168, 566, "候选 token 与 corrected q → 接受判定，不参与分配", 17)
    # Cross-round closure: publish writes the D2H ring; the next round's
    # prepare reads a two-slot-old entry.
    f.arrow((1108, 420), (1108, 92), (152, 92), (152, 137))
    f.text(520, 100, "on_publish(confidence) → D2H ring 写入；下一轮 prepare 滞后读取", 17)
    # ── Example strip ─────────────────────────────────────────────────────
    f.text(32, 615, "示例：R = 2，host 从历史分数选出 K = 4", 21, weight=650)
    f.text(32, 647, "当前 survival 分配 [3,1] → verify_lens = [4,2]（各含 1 个 anchor）", 20)
    f.text(32, 679, "target 有效行 M = R + K = 6；capture bucket 可能 replay 8 行", 20)
    f.note(720, "预算决定额度，当前块决定归属；两者在 schedule_layout() 汇合。发布的是本轮置信度，读回的是滞后两槽的历史。")
    f.note(752, "TP：draft 采样与 verify_lens 分别经 DSPARK_DRAFT_* / DSPARK_PLAN 从 rank 0 广播；预算本身无 collective。")
    return f


def host_budget(Figure):
    f = Figure(
        "Host 如何得到 top-k 的 K：relay 可见性与历史搜索",
        "三槽 pinned host ring 默认两步滞后；query 完成后按请求槽位读取，generation 失效取 ones，成本搜索只返回数量 K。",
        625,
    )
    f.note(78, "ConfidenceRelay 搬运预测；HostConfidenceBudgetPlanner 补足 lag 并选预算；GPU 决定当前赢家。")
    f.chain(167)
    f.box(32, 111, "GPU publish", ["scatter 当前 confidence", "记录 publish_ready"], "draft")
    f.box(304, 111, "专用 D2H stream", ["wait_event → async copy", "issue_ring_copy + copy_done"], "neutral")
    f.box(576, 111, "Host 非阻塞读取", ["slot = (ring_pos − 2) % 3", "copy_done.query()"], "anchor")
    f.text(696, 218, "ConfidenceRelay.resolve()", 16, color=MUTED, anchor="middle")
    f.box(848, 111, "历史请求行", ["按 req_pool_indices 取行", "携带 generation 快照"], "neutral")
    f.text(32, 276, "默认 ring depth = 3，relay lag = 2", 21, weight=650)
    f.text(32, 311, "发布 c0：pos=1 → None", 20)
    f.text(32, 345, "发布 c1：pos=2 → 读 c0", 20)
    f.text(32, 379, "发布 c2：pos=3 → 读 c1", 20)
    f.text(389, 311, "query 未完成 → None", 20)
    f.text(389, 345, "预算不可用 → 布局回退", 20)
    f.text(389, 379, "不在这里等待当前 GPU", 20)
    f.arrow((968, 223), (968, 405), (152, 405), (152, 435))
    f.chain(491)
    f.box(32, 435, "可选 host carry", ["长度 max(lag − relay, 0)", "默认 overlap：无需追加"], "neutral")
    f.text(152, 542, "_shift_to_lag()", 16, color=MUTED, anchor="middle")
    f.box(304, 435, "generation 校验", ["匹配 → cumprod(history)", "失效 → survival 全 1"], "anchor")
    f.text(424, 542, "_two_steps_prior_survival()", 16, color=MUTED, anchor="middle")
    f.box(576, 435, "历史收益 × SPS", ["展平过滤排序 → 前缀和", "argmax(τ̂ × SPS) → K"], "anchor")
    f.text(696, 542, "compute_verify_token_budget()", 16, color=MUTED, anchor="middle")
    f.box(848, 435, "当前 GPU top-K", ["使用本轮 survival", "重新决定请求 / 位置归属"], "seq")
    f.text(968, 542, "ScheduleVerifyLensTopk.execute()", 16, color=MUTED, anchor="middle")
    f.note(593, "K 是额外草稿数；默认 target 总行数为 R + K。历史排序索引不会作为当前分配结果传递。")
    return f
