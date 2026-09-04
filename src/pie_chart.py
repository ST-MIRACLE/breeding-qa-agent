# -*- coding: utf-8 -*-
"""
基因型分布饼图生成器 - 独立工具
用法: python src/pie_chart.py "252" "ty1"
功能: 统计指定季节、指定基因的 R/H/S 分布，生成HTML饼图文件
"""
import io
import os
import sys
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RAGEngine

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts")


def generate_pie_chart(season_code, gene_name, label=""):
    """生成基因型分布饼图，返回(html_path, stats_json)"""
    engine = RAGEngine()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 统计: 遍历匹配季节的文件，找目标基因列，统计 R/H/S
    season_matchers = engine._parse_season(season_code)
    gene_lower = gene_name.lower()
    gene_col = None
    file_name = None
    distribution = {"R": 0, "H": 0, "S": 0, "无条带": 0}

    for filename, df in engine.breeding_data.items():
        if not engine._match_season(filename, season_matchers):
            continue
        # 每个文件独立找目标基因列
        gene_col = None
        for col in df.columns:
            cl = str(col).lower()
            if (gene_lower == "ty" and cl.startswith("ty")) or (gene_lower in cl):
                gene_col = col
                file_name = filename
                break
        if gene_col is None:
            continue
        # 统计分布
        for raw in df[gene_col]:
            val = str(raw).strip().upper()
            if val in ("", "NAN", "NONE", "NA", "NULL", "无", "无带", "无条带", "NO", "NONE条带"):
                distribution["无条带"] += 1
            elif val in distribution:
                distribution[val] += 1
            else:
                distribution["无条带"] += 1

    total = sum(distribution.values())
    if total == 0:
        return None, None

    # 只保留有数据的类别
    stats = {k: v for k, v in distribution.items() if v > 0}

    # 生成 HTML 饼图（内联 ECharts CDN）
    chart_data = [
        {"name": {"R": "抗病 (R)", "H": "杂合 (H)", "S": "感病 (S)", "无条带": "无条带"}.get(k, k),
         "value": v}
        for k, v in stats.items()
    ]

    title = f"{label or season_code} {gene_name} 基因型分布"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="../portfolio/_shared/js/echarts.min.js"></script>
<style>
body {{ background:#0f1117; color:#e0e0e0; font-family: "Microsoft YaHei", sans-serif; margin:0; padding:30px; text-align:center; }}
h2 {{ color:#fff; margin-bottom:20px; }}
#chart {{ width:720px; height:480px; margin:0 auto; background:#1a1a2e; border-radius:12px; }}
.stats {{ margin-top:20px; color:#6c7293; font-size:14px; }}
.stats b {{ color:#e63946; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div id="chart"></div>
<div class="stats">总样本 <b>{total}</b> 个单株 · 数据来源: {file_name or "多文件"}</div>
<script>
var chart = echarts.init(document.getElementById('chart'));
chart.setOption({{
  tooltip: {{ trigger: 'item', formatter: '{{b}}: {{c}} 个 ({{d}}%)' }},
  legend: {{ orient: 'horizontal', bottom: 10, textStyle: {{ color: '#a0a0b0' }} }},
  series: [{{
    name: '基因型',
    type: 'pie',
    radius: ['40%', '68%'],
    center: ['50%', '45%'],
    avoidLabelOverlap: true,
    itemStyle: {{ borderRadius: 8, borderColor: '#1a1a2e', borderWidth: 2 }},
    label: {{ color: '#e0e0e0', formatter: '{{b}} {{c}} ({{d}}%)' }},
    data: {json.dumps(chart_data, ensure_ascii=False)}
  }}]
}});
window.addEventListener('resize', function() {{ chart.resize(); }});
</script>
</body>
</html>"""

    # 安全文件名
    safe_season = season_code.replace(" ", "").replace("/", "_")
    safe_gene = gene_name.replace(" ", "").replace("/", "_")
    html_path = os.path.join(OUTPUT_DIR, f"pie_{safe_season}_{safe_gene}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    stats_for_reply = {
        "基因": gene_name,
        "总单株": total,
        "分布": {k: v for k, v in stats.items()},
        "文件": file_name,
        "图片路径": html_path
    }
    return html_path, stats_for_reply


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pie_chart.py <季节> <基因>")
        print("示例: python pie_chart.py 252 ty1")
        sys.exit(1)
    season = sys.argv[1]
    gene = sys.argv[2]
    path, stats = generate_pie_chart(season, gene)
    if path:
        print("✅ 饼图已生成:", path)
        print("统计:", json.dumps(stats, ensure_ascii=False, indent=2))
        os.startfile(path)  # 自动打开浏览器查看
    else:
        print("❌ 未找到数据")
