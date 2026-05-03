import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from PIL import Image

# 由于NTSB的报告中的图表并没有给出全部数据
# 使用一段简单的代码 绘制ExactSample.csv和Tableresolution.csv文件中的数据将其可视化
# 最后合并图片 生成一个pdf


# ─────────────────────────────────────────────
# CONFIG / 配置参数
# ─────────────────────────────────────────────
CSV_PATH1 = "ExactSample.csv"
CSV_PATH2 = "TableResolution.csv"
OUTPUT_DIR = "."
DPI = 200
GROUP_SIZE = 10

# 时间范围过滤：设置为 [x, y] 即可绘制 这一区间内的所有数据
TIME_RANGE = [288200, float('inf')]  

def load_and_clean_data(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    lines = [l.strip() for l in raw_lines if l.strip()]

    try:
        data_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("DATA"))
    except StopIteration:
        raise ValueError("CSV 文件中未找到 'DATA' 标记行")

    names = [n.strip() for n in lines[data_idx+1].split(",")]
    units = [u.strip() for u in lines[data_idx+2].split(",")]
    types = [t.strip() for t in lines[data_idx+3].split(",")]

    data_rows = []
    expected_cols = len(names)
    for l in lines[data_idx+4:]:
        row = [val.strip() for val in l.split(",")]
        if len(row) < expected_cols:
            row.extend([""] * (expected_cols - len(row)))
        else:
            row = row[:expected_cols]
        data_rows.append(row)

    df_raw = pd.DataFrame(data_rows, columns=names)
    time_col = names[0]
    
    # 转换时间列
    df_raw[time_col] = pd.to_numeric(df_raw[time_col], errors='coerce')
    df_raw = df_raw.dropna(subset=[time_col])

    # 智能处理每一列
    processed_data = {time_col: df_raw[time_col].values}
    col_meta_updates = {}

    for col in names[1:]:
        # 尝试转数字
        numeric_series = pd.to_numeric(df_raw[col].replace('', np.nan), errors='coerce')
        
        # 处理非数值字符（如 NORMAL/UNAVAIL）
        if numeric_series.isnull().all() and not df_raw[col].replace('', np.nan).isnull().all():
            unique_words = sorted([str(w) for w in df_raw[col].unique() if w != ''])
            word_map = {word: float(i) for i, word in enumerate(unique_words)}
            processed_data[col] = df_raw[col].map(word_map).values
            col_meta_updates[col] = {"is_forced_enum": True, "forced_map": {v: k for k, v in word_map.items()}}
        else:
            processed_data[col] = numeric_series.values
            col_meta_updates[col] = {"is_forced_enum": False}

    df = pd.DataFrame(processed_data)
    
    # 按时间聚合（合并稀疏行）
    df = df.groupby(time_col, as_index=False).first()
    
    # 【时间范围过滤逻辑】
    if TIME_RANGE is not None:
        df = df[(df[time_col] >= TIME_RANGE[0]) & (df[time_col] <= TIME_RANGE[1])]
        print(f"时间过滤完成，起始时间: {TIME_RANGE[0]}, 保留数据点: {len(df)}")

    return df, names, units, types, col_meta_updates

ENUM_RE = re.compile(r'([\d.eE+\-]+)\s*:\s*[\d.eE+\-]+\s*="([^"]+)"')

def parse_enum_meta(names, units, types, forced_updates):
    meta = {}
    for i, c in enumerate(names):
        t_str = types[i]
        is_enum = t_str.startswith("%N") or forced_updates.get(c, {}).get("is_forced_enum", False)
        
        enum_map = {}
        # 解析表头的 %N 定义
        matches = ENUM_RE.findall(t_str)
        for v, label in matches:
            try: enum_map[float(v)] = label
            except: pass
        
        # 合并强制映射
        if forced_updates.get(c, {}).get("is_forced_enum"):
            enum_map.update(forced_updates[c]["forced_map"])
        
        meta[c] = {
            "unit": units[i],
            "is_enum": is_enum,
            "enum": enum_map
        }
    return meta

def plot_fdr_stack(df, cols, time_col, meta_info, title="FDR Plot"):
    if df.empty:
        print("Warning: 所选时间范围内没有数据点。")
        return None

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, max(2.2 * n, 5)), sharex=True, dpi=DPI)
    if n == 1: axes = [axes]

    t = df[time_col].values

    for i, col in enumerate(cols):
        ax = axes[i]
        m = meta_info[col]
        
        # 填充缺失值以保持阶跃线段连续
        y_raw = df[col].ffill() 
        y_vals = y_raw.values
        
        if np.all(np.isnan(y_vals)):
            ax.text(0.5, 0.5, f"No Data: {col}", transform=ax.transAxes, ha='center', color='gray')
        else:
            if m["is_enum"]:
                # 状态类数据：强制阶跃函数样式
                ax.step(t, y_vals, where="post", color='#d62728', linewidth=1.8)
                
                # 动态设置纵轴刻度标签
                present_vals = np.unique(y_vals[~np.isnan(y_vals)])
                tick_vals = sorted(list(set(present_vals) | set(m["enum"].keys())))
                if tick_vals:
                    ax.set_yticks(tick_vals)
                    ax.set_yticklabels([m["enum"].get(v, str(v)) for v in tick_vals])
                    ax.set_ylim(min(tick_vals) - 0.5, max(tick_vals) + 0.5)
            else:
                # 数值类数据
                ax.plot(t, y_vals, color='#1f77b4', linewidth=1.2, alpha=0.9)
                valid_y = y_vals[~np.isnan(y_vals)]
                if len(valid_y) > 0:
                    ymin, ymax = np.min(valid_y), np.max(valid_y)
                    pad = (ymax - ymin) * 0.2 if ymax != ymin else 1.0
                    ax.set_ylim(ymin - pad, ymax + pad)

        # 设置每个子图的背景网格和标签
        unit_label = f"\n({m['unit']})" if m['unit'] and m['unit'] != '()' else ""
        ax.set_ylabel(f"{col}{unit_label}", rotation=0, ha="right", va="center", fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)

    # 【核心修复：双重保障禁用偏移量和科学计数法】
    last_ax = axes[-1]
    last_ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    last_ax.xaxis.set_major_formatter(fmt)
    last_ax.set_xlabel(f"Time (sec)", fontsize=10, fontweight='bold')
    plt.xlim(t.min(), t.max())
    plt.setp(last_ax.get_xticklabels(), rotation=0)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────
# 核心：对单个 CSV 执行完整绘图流程，返回生成的 PNG 文件路径列表
# ─────────────────────────────────────────────
def process_csv(csv_path, tag):
    """
    读取 csv_path，绘制所有分组图，将 PNG 保存至 OUTPUT_DIR。
    tag: 用于区分两套文件的前缀标识，如 "CSV1" / "CSV2"
    返回：按顺序排列的 PNG 路径列表
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found，跳过。")
        return []

    print(f"\n{'='*60}")
    print(f"开始处理：{csv_path}  [标识: {tag}]")
    print(f"{'='*60}")

    df, names, units, types, forced_updates = load_and_clean_data(csv_path)
    meta_info = parse_enum_meta(names, units, types, forced_updates)

    time_col = names[0]
    plot_cols = [c for c in names if c != time_col]
    groups = [plot_cols[i:i+GROUP_SIZE] for i in range(0, len(plot_cols), GROUP_SIZE)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generated_files = []

    suffix = "FULL" if TIME_RANGE is None else f"START_{TIME_RANGE[0]}"

    for i, g in enumerate(groups):
        print(f"  正在绘制 [{tag}] 组 {i+1}/{len(groups)} ...")
        fig = plot_fdr_stack(
            df, g, time_col, meta_info,
            title=f"FDR Analysis ({tag}) - Group {i+1}"
        )
        if fig:
            fname = f"FDR_STACK_{tag}_{suffix}_{i+1:02d}.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            generated_files.append(fpath)
            print(f"    已保存：{fname}")

    print(f"[{tag}] 共生成 {len(generated_files)} 张图片。")
    return generated_files


# ─────────────────────────────────────────────
# 将图片列表合并为单个 PDF
# ─────────────────────────────────────────────
def merge_images_to_pdf(image_paths, output_pdf):
    image_list = []
    for img_path in image_paths:
        img = Image.open(img_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        image_list.append(img)

    if image_list:
        image_list[0].save(
            output_pdf,
            save_all=True,
            append_images=image_list[1:]
        )
        print(f"\n成功！共合并 {len(image_list)} 张图片。")
        print(f"PDF 保存至：{output_pdf}")
    else:
        print("错误：没有可合并的图片。")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    # ① 分别处理两个 CSV，生成两套 PNG
    files_csv1 = process_csv(CSV_PATH1, tag="CSV1")
    files_csv2 = process_csv(CSV_PATH2, tag="CSV2")

    # ② 按 CSV1 → CSV2 的顺序合并所有图片
    all_files = files_csv1 + files_csv2

    # ③ 输出 PDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_pdf  = os.path.join(current_dir, "MU5735FDR_latest_time.pdf")
    merge_images_to_pdf(all_files, output_pdf)


if __name__ == "__main__":
    main()