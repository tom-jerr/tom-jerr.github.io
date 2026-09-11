#!/usr/bin/env python3
"""Deterministic process diagrams for the HiCache request-timing article.

Uses the established HiCache palette/primitives; all stages are semantic data.
Usage: python scripts/draw_hicache_timing.py [output_directory]
"""
from pathlib import Path
import argparse
from draw_hicache_figures import (SVG, INK, MUTED, LINE, WHITE, CONTROL,
    CONTROL_BG, CPU, CPU_BG, GPU, GPU_BG, STORE, STORE_BG, WARN, WARN_BG)


def box(s, x, y, w, lines, fill=CONTROL_BG, color=CONTROL, h=62):
    s.rect(x, y, w, h, fill, color, radius=8, stroke_width=1.5)
    s.multiline(x+w/2, y+h/2-8 if len(lines)>1 else y+h/2+6,
                lines, 17, INK, 550, 'middle', 24)


def sequence(title, participants, events, foot):
    w = 1520
    xs = [170 + i * (1180/(len(participants)-1)) for i in range(len(participants))]
    h = 185 + len(events)*83 + 60
    s = SVG(w, h, title, '时间向下；实线为调用或提交，虚线为完成通知；不是等比例耗时。')
    s.text(42, 42, title, 28, INK, 700)
    s.text(42, 74, '时间向下 ↓    实线：调用 / 提交    虚线：完成通知    同一泳道：本线程操作', 17, MUTED)
    for x, p in zip(xs, participants):
        s.line(x, 148, x, h-72, LINE, 1.5, False, True)
        box(s, x-135, 102, 270, [p], h=48)
    for i, (a,b,lines,color) in enumerate(events):
        y = 188+i*83
        s.text(42, y+19, str(i+1).zfill(2), 16, MUTED)
        if a == b:
            box(s, xs[a]-137, y-12, 274, lines, WHITE, color)
        else:
            dashed = color == MUTED
            s.line(xs[a], y+40, xs[b], y+40, color, 2, True, dashed)
            s.multiline((xs[a]+xs[b])/2, y+3, lines, 16, INK, 500, 'middle', 22)
    s.text(42, h-25, foot, 17, MUTED)
    return s


def read_sequence():
    return sequence('L3 预取：请求入队、后台 I/O、完成发布、重新匹配',
        ['Scheduler', 'HiRadix / Controller', 'Query / Sync 线程', 'I/O 线程 / L3 Backend'], [
        (0,1,['入队前 init_next_round_input()', 'match_prefix() 同时识别 L1 / L2'],CONTROL),
        (1,0,['GPU indices + host_hit_length', '保留 last_host_node 挂载点'],MUTED),
        (0,1,['_prefetch_kvcache()', '仅提交允许匹配范围内的 suffix'],CONTROL),
        (1,2,['保护 Host 挂载点', 'prefetch_queue 入队'],CONTROL),
        (2,3,['hash chain → batch_exists()', '查询连续 page prefix'],STORE),
        (3,2,['存在性查询结果', '各 Rank 取 MIN'],MUTED),
        (2,1,['prefetch_hit_queue', '返回 storage_hit_count'],MUTED),
        (1,1,['Scheduler drain：分配 L2', '不足则驱逐 / 缩短 / 撤销'],CPU),
        (1,3,['prefetch_buffer → batch_get', '数据写入已分配的 Host slots'],STORE),
        (0,0,['预取期间可调度其他请求', '本请求尚未获准入批'],CONTROL),
        (3,2,['PrefetchAck：连续完成长度', '最后发送 completed_req'],MUTED),
        (2,1,['跨 Rank 对齐完成长度', 'ack_prefetch_queue'],MUTED),
        (1,1,['_handle_prefetch_result()', '发布 Host 节点，处理重复页'],CPU),
        (0,1,['check_prefetch_progress()', '已完成，或按策略终止并采纳部分'],CONTROL),
        (0,1,['再次 init_next_round_input()', '重新 match_prefix()'],CONTROL),
        (1,0,['更新后的 L1 indices / L2 命中', '交给 PrefillAdder 做入批判断'],MUTED),
        ], '完整完成可由 drain 直接发布；提前终止由 check_prefetch_progress 发布。此图按完整成功路径排列。')


def read_pipeline():
    s=SVG(1690, 655, '从入队到 Prefill 的读 pipeline', '时间向右；预取期间其他请求可运行；当前请求先发布 L2 再申请 H2D。')
    s.text(38, 43, '读 pipeline：先扩展可复用前缀，再启动本请求的 Prefill', 27, INK, 700)
    s.text(38, 76, '顺序示意，不按耗时比例；成功且达到预取 / 回填阈值的 cache-mode 路径', 17, MUTED)
    ys=[145,245,345,445,545]
    for y,label in zip(ys,['Scheduler','L3 Query','L3 I/O → L2','H2D stream','GPU compute']):
        s.text(30,y+38,label,18,INK,600)
        s.line(210,y+72,1640,y+72,LINE,1,False)
    stages=[(0,220,190,['本地 L1 / L2 match','提交 L3 suffix'],CONTROL_BG,CONTROL),
        (1,440,175,['hash + exists','跨 Rank MIN'],STORE_BG,STORE),
        (0,650,170,['drain hit 结果','按命中分配 L2'],CPU_BG,CPU),
        (2,850,180,['batch_get → Host','ACK / 跨 Rank MIN'],STORE_BG,STORE),
        (0,1060,170,['发布 L2 + 重匹配','预算检查'],CONTROL_BG,CONTROL),
        (0,1260,175,['init_load_back','申请 L1 / load_queue'],CPU_BG,CPU),
        (3,1460,180,['start_loading','按层恢复 prefix'],CPU_BG,CPU),
        (4,1530,110,['Prefill','真 miss'],GPU_BG,GPU)]
    for left,right in zip(stages,stages[1:]):
        la,x,w,*_=left; lb,x2,w2,*_=right
        s.path(f'M {x+w} {ys[la]+31} L {x2-14} {ys[la]+31} L {x2-14} {ys[lb]+31} L {x2} {ys[lb]+31}',INK,1.8)
    for lane,x,w,labels,fill,color in stages: box(s,x,ys[lane],w,labels,fill,color)
    box(s,440,ys[4],930,['已有请求继续 Prefill / Decode；不依赖本请求的 L3 prefetch 完成'],GPU_BG,GPU)
    s.text(1456,534,'逐层 event →',16,CPU)
    s.line(210,632,1640,632,INK,1.5)
    s.text(1540,620,'时间 →',16,MUTED)
    return s


def layer_pipeline():
    s=SVG(1410,500,'H2D 与 Prefill 按层重叠','每层先完成历史 prefix 的 H2D，随后该层 attention 才能读取；全量 ACK 只做生命周期回收。')
    s.text(38,43,'H2D 按层搬运；Attention 在读取该层 KV 前等待 event',27,INK,700)
    s.text(38,77,'本图示意 4 层；箱宽不是测量耗时，重叠机会不等于一定隐藏全部传输',17,MUTED)
    for y,l in [(150,'H2D stream'),(280,'Compute stream'),(390,'Scheduler')]:
        s.text(30,y+34,l,18,INK,600)
    transfers=[(230,150),(435,150),(640,150),(845,150)]
    computes=[(435,280),(640,280),(845,280),(1050,280)]
    for i,((x,y),(cx,cy)) in enumerate(zip(transfers,computes)):
        s.path(f'M {x+170} {y+31} L {x+187} {y+31} L {x+187} {cy+31} L {cx} {cy+31}',CPU,2)
        if i<3:
            s.line(x+170,y+31,x+205,y+31,INK,1.6)
            s.line(cx+170,cy+31,cx+205,cy+31,INK,1.6)
        box(s,x,y,170,[f'Layer {i} H2D','全部待恢复 pages'],CPU_BG,CPU)
        box(s,cx,cy,170,[f'Layer {i} Forward','计算未缓存 suffix'],GPU_BG,GPU)
        s.text(cx-4,249,f'wait event {i}',15,CPU)
    s.path('M 1015 181 L 1270 181 L 1270 420 L 1150 420',MUTED,1.8,True,True)
    box(s,720,390,430,['loading_check：全量 H2D ACK 完成后释放搬运锁'],WHITE,CONTROL)
    s.text(230,435,'与 Forward 的逐层读取等待分开',17,MUTED)
    s.line(210,476,1360,476,INK,1.5)
    s.text(1270,465,'时间 →',16,MUTED)
    return s


def write_sequence():
    return sequence('写穿时序：L1 登记、D2H 完成、L3 backup、释放保护',
        ['Scheduler', 'HiRadix / Controller', 'D2H stream', 'Backup 线程 / L3'],[
        (0,1,['Prefill 后 cache_unfinished_req', '或结束后 cache_finished_req'],CONTROL),
        (1,1,['insert：登记 GPU indices', '去重 / 页对齐 / 写策略判断'],CONTROL),
        (1,2,['write_backup → write()', '分配 L2，提交 D2H'],CPU),
        (1,1,['host_value 已分配 + pending', '保留 GPU 源数据保护'],CPU),
        (0,0,['请求继续 Decode，或结束清理', '无需等待 L3 backup'],CONTROL),
        (2,1,['ack_write.finish_event 就绪', '确认所有层 D2H 完成'],MUTED),
        (0,1,['writing_check()', '消费 D2H ACK'],CONTROL),
        (1,1,['_finish_write_through_ack', '清 pending / 发布 CPU 完成'],CPU),
        (1,3,['write_backup_storage', 'backup_queue；保护 Host 源页'],STORE),
        (1,1,['释放 D2H 的 GPU 保护锁', 'L3 读取源已在 Host'],CONTROL),
        (3,3,['逐批 page_set_func', '成功批次累计 completed_tokens'],STORE),
        (3,1,['ack_backup_queue', '操作结束，可含部分写入失败'],MUTED),
        (0,1,['Scheduler drain backup ACK', '移除 ongoing_backup'],CONTROL),
        (1,1,['release_host()', '允许驱逐，通常不立即释放页'],CPU),
        ], '写穿主路径；选择性写穿多一道计数阈值。write_back 的 GPU 释放顺序见下一图。')


def write_pipeline():
    s=SVG(1610,590,'写策略决定备份何时进入 pipeline','写穿由插入和计数触发；写回由驱逐触发，释放 GPU 之前必须确认 D2H 完成。')
    s.text(38,44,'三种写策略：触发点不同，D2H → L3 的数据依赖相同',27,INK,700)
    s.text(38,80,'普通 cache-mode / dense HiRadixCache；L3 已启用且 backup 未跳过',17,MUTED)
    lanes=[(145,'write_through',['insert','计数 ≥ 1']),
           (275,'selective',['insert / 再插入','计数 ≥ 2']),
           (405,'write_back',['GPU eviction','选中无 Host 副本节点'])]
    for y,name,first in lanes:
        s.text(28,y+35,name,18,INK,650)
        texts=[first,['申请 L2','D2H 所有层'],['D2H finish','确认 Host 有效'],['L2 → L3','batch_set'],['Storage ACK','释放 Host 保护']]
        xs=[215,480,745,1010,1275]
        for j in range(4): s.line(xs[j]+225,y+31,xs[j+1],y+31,INK,1.8)
        for j,(x,t) in enumerate(zip(xs,texts)):
            box(s,x,y,225,t,CONTROL_BG if j==0 else CPU_BG if j<3 else STORE_BG,
                CONTROL if j==0 else CPU if j<3 else STORE)
        if name=='write_back':
            s.line(857,y+62,857,y+85,CPU,1.8)
            s.text(690,y+111,'到这里才 free 源 GPU slots；L3 不必完成',17,CPU)
        else:
            s.text(215,y+92,'chunked=True 跳过计数；父节点 Host 前缀检查通过后才提交',16,MUTED)
    s.text(38,566,'写穿：D2H ACK 后 GPU 变得可驱逐。写回：可先清 node.value，但物理 GPU slots 要保留到 D2H 完成。',17,MUTED)
    return s


FIGURES={'hicache-request-read-sequence':read_sequence,
         'hicache-request-read-pipeline':read_pipeline,
         'hicache-layer-load-pipeline':layer_pipeline,
         'hicache-request-write-sequence':write_sequence,
         'hicache-write-policy-pipeline':write_pipeline}

def main():
    p=argparse.ArgumentParser();p.add_argument('output',nargs='?',type=Path,
        default=Path(__file__).resolve().parents[1]/'content/sglang/img')
    root=p.parse_args().output;root.mkdir(parents=True,exist_ok=True)
    for name,fn in FIGURES.items():
        s=fn();path=root/(name+'.svg');path.write_text(s.finish(),encoding='utf-8')
        print(f'{path}: {s.width} x {s.height}')

if __name__=='__main__': main()
