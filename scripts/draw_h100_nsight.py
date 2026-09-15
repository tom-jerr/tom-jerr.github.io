"""Editable H100 data-path and async timeline figures; no measured timing data."""
from pathlib import Path
from html import escape
import argparse

FONT = "'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif"
INK = '#263238'
BLUE = '#e9eefb'
TEAL = '#e1f3ef'
GRAY = '#f4f5f7'

class SVG:
    def __init__(self, w, h, title, desc):
        self.p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
                  f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
                  '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#52606d"/></marker></defs>',
                  f'<rect width="{w}" height="{h}" fill="white"/>']
    def rect(self, x, y, w, h, fill=GRAY, stroke='#b3bec7'):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    def text(self, x, y, s, size=18, anchor='start', weight=400, color=INK):
        self.p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')
    def arrow(self, points, dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        self.p.append(f'<polyline points="{points}" fill="none" stroke="#52606d" stroke-width="2"{dash} marker-end="url(#arrow)"/>')
    def save(self, path):
        path.write_text('\n'.join(self.p + ['</svg>'])+'\n', encoding='utf-8')

def architecture(out):
    s=SVG(1200,700,'H100 SXM 的计算与数据路径','左侧为 HBM 和全 GPU 共享的 L2，右侧展开一个 SM 的四个子分区；普通 load 与 TMA 走不同的寄存器供给路径。')
    s.text(40,43,'H100 SXM：沿数据路径理解硬件',27,weight=700)
    s.text(40,74,'逻辑示意；右侧只展开 132 个 SM 中的一个，箭头表示主要读取方向。',17,color='#607080')
    s.rect(35,112,340,482,'#fafbfc')
    s.text(58,147,'GPU 共享的存储层次',21,weight=700)
    s.rect(65,179,280,96,BLUE)
    s.text(205,211,'HBM3 · 80 GB',22,'middle',700)
    s.text(205,243,'SXM 标称最高 3.35 TB/s',17,'middle')
    s.arrow('205,276 205,333')
    s.text(222,310,'内存控制器',16)
    s.rect(65,335,280,111,TEAL)
    s.text(205,369,'L2 · 50 MB',23,'middle',700)
    s.text(205,400,'分区缓存 / SM 间数据复用',17,'middle')
    s.text(205,426,'实际路径还包含片上互连',16,'middle')
    s.text(58,495,'global / local 是地址空间语义。',17)
    s.text(58,525,'local 并不代表片上存储；',17)
    s.text(58,555,'cache hit 也不代表没有端口成本。',17)

    s.rect(433,112,732,482,'#fafbfc')
    s.text(458,147,'一个 SM：4 个 SMSP + 共享的数据存储',21,weight=700)
    for i in range(4):
        x=456+i*174
        s.rect(x,172,159,208,'white')
        s.text(x+79.5,201,f'SMSP {i}',19,'middle',700)
        s.text(x+79.5,238,'Warp Scheduler',16,'middle')
        s.text(x+79.5,270,'Register File',16,'middle')
        s.text(x+79.5,306,'FP / INT / LSU',16,'middle')
        s.text(x+79.5,343,'Tensor Core 等',16,'middle')
        s.arrow(f'{x+79.5},417 {x+79.5},382')
    s.rect(456,421,681,106,TEAL)
    s.text(796,455,'统一 L1 / Shared 存储资源：256 KiB',22,'middle',700)
    s.text(796,486,'普通 load 可经 L1；Shared 由软件管理，按 bank 组织',17,'middle')
    s.text(796,513,'Shared 每 SM 最多 228 KiB；WGMMA 可直接读取 Shared operand',16,'middle')
    s.arrow('346,374 400,374 400,452 454,452')
    s.text(394,349,'load',15,'middle')
    s.arrow('346,417 390,417 390,561 700,561 700,529',True)
    s.text(755,568,'TMA：L2 → Shared，省去显式寄存器中转',16)
    s.text(42,639,'发射指令 ≠ 指令完成。普通 load 的结果要返回寄存器；TMA/WGMMA 还需要各自的完成协议。',18)
    s.text(42,672,'来源：NVIDIA H100 白皮书、Hopper Tuning Guide、PTX；布局为本文重绘。',16,color='#607080')
    s.save(out/'hardware-path.svg')

def pipeline(out):
    s=SVG(1200,660,'从立即等待到双缓冲流水','上半部每次搬运后立即等待并计算；下半部让下一个 tile 的搬运与当前 tile 的计算重叠，只有消费者读完后才能复用 buffer。时间长度仅示意。')
    s.text(40,43,'异步 API 只有和独立工作重叠，才能隐藏等待',26,weight=700)
    s.text(40,74,'时间轴仅表示依赖与重叠，不代表 H100 实测 cycle 或加速比。',17,color='#607080')
    s.text(40,124,'每轮立即等待',21,weight=700)
    def box(x,y,w,label,kind):
        s.rect(x,y,w,50,BLUE if kind=='copy' else TEAL)
        s.text(x+w/2,y+32,label,18,'middle',600)
    for x,w,label,kind in [(175,165,'TMA tile 0','copy'),(365,205,'MMA tile 0','compute'),(595,165,'TMA tile 1','copy'),(785,205,'MMA tile 1','compute')]:
        box(x,148,w,label,kind)
    for a,b in [(340,365),(570,595),(760,785)]: s.arrow(f'{a},173 {b-2},173')
    s.text(180,229,'copy 完成后才能读；MMA 读完之后才能覆写同一缓冲区。',17)
    s.text(40,298,'两个 Shared slot',21,weight=700)
    s.text(42,354,'TMA',19,weight=700)
    s.text(42,441,'Tensor Core',19,weight=700)
    box(175,325,165,'tile 0 → slot 0','copy')
    box(365,325,165,'tile 1 → slot 1','copy')
    box(620,325,165,'tile 2 → slot 0','copy')
    box(365,412,230,'MMA tile 0','compute')
    box(620,412,230,'MMA tile 1','compute')
    box(875,412,230,'MMA tile 2','compute')
    s.arrow('340,351 352,351 352,437 363,437')
    s.arrow('530,375 530,394 610,394 610,437 618,437')
    s.arrow('785,375 785,394 865,394 865,437 873,437')
    s.arrow('595,437 607,437 607,350 618,350',True)
    s.text(790,351,'← slot 0 复用前，tile 0 的读取已完成',16)
    s.arrow('176,491 1126,491')
    s.text(1127,519,'时间',17,'end')
    s.rect(40,545,1120,77,'#fafbfc')
    s.text(60,574,'两种完成条件：ready = 数据已经写好；reusable = 消费者已经不再读取。',19,weight=600)
    s.text(60,603,'生产者提前搬下一块；消费者只等待当前块；group 深度、Shared stage 数和寄存器预算共同约束流水。',17)
    s.save(out/'async-pipeline.svg')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out', nargs='?', type=Path, default=Path(__file__).resolve().parents[1]/'content/gpu/img/h100-nsight')
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    architecture(args.out)
    pipeline(args.out)
