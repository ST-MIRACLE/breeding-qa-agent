"""
RAG引擎 - 混合检索 + 知识库约束 + LLM生成
结构化知识(JSON) + 非结构化文档(向量检索) + 用户数据查询
"""
import os
import json
import requests
import pandas as pd
from config import API_KEY, CHAT_MODEL, STRUCTURED_KB_PATH, BREEDING_DATA_DIR
from embedding_manager import EmbeddingManager


class RAGEngine:
    """育种知识库RAG引擎"""

    CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self):
        self.llm_api_key = API_KEY
        self.chat_model = CHAT_MODEL
        self.embedding = EmbeddingManager()
        self.structured_kb = self._load_structured_kb()
        self.breeding_data = self._load_breeding_data()

    def _load_structured_kb(self):
        with open(STRUCTURED_KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_breeding_data(self):
        """递归加载用户育种数据，用于数据查询"""
        data = {}
        if not os.path.exists(BREEDING_DATA_DIR):
            return data

        for root, dirs, files in os.walk(BREEDING_DATA_DIR):
            for filename in files:
                if not (filename.endswith(".xlsx") or filename.endswith(".xls")):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    df = pd.read_excel(filepath)
                    df = self._fix_dataframe_header(df, filepath)
                    data[filename] = df
                except Exception:
                    pass
        return data

    def _fix_dataframe_header(self, df, filepath):
        """修复表头：处理列名全为Unnamed的文件（表头可能在第0行或第1行）"""
        if len(df) == 0:
            return df
        # 情况1：表头在第二行（第0行是"名称"标记）
        if str(df.columns[0]).startswith("Unnamed") and str(df.iloc[0, 1]) == "名称":
            return pd.read_excel(filepath, header=1)

        # 情况2：列名全是Unnamed，但第0行含基因型表头（Ty1/Ty3/Frl等）
        all_unnamed = all(str(c).startswith("Unnamed") for c in df.columns)
        if all_unnamed and len(df) > 1:
            first_row = df.iloc[0].astype(str).tolist()
            gene_kws = ["Ty", "Tm", "Frl", "Sm", "Mi", "SW", "Tobrfv", "TY", "编号"]
            has_gene_header = any(any(k in str(v) for k in gene_kws) for v in first_row if str(v) not in ("nan", ""))
            if has_gene_header:
                # 用第0行做列名，删除第0行表头行，数据从第1行开始
                new_cols = []
                for i, v in enumerate(first_row):
                    if str(v) in ("nan", "") or str(v).startswith("Unnamed"):
                        new_cols.append(f"col{i}")
                    else:
                        new_cols.append(str(v))
                df = df.iloc[1:].copy()
                df.columns = new_cols
                return df
        return df

    def query(self, question, chat_history=None):
        """主查询入口：意图判断 → 混合检索 → LLM生成"""
        # 自我介绍/寒暄类问题，直接返回固定回答（不检索不调API）
        greeting_intro = self._handle_greeting(question)
        if greeting_intro is not None:
            return {
                "answer": greeting_intro,
                "sources": ["系统自我介绍"],
                "context_count": 1,
                "intent": "greeting"
            }

        intent = self._classify_intent(question)

        context_parts = []
        sources = []

        if intent == "chart_query":
            chart_result = self._generate_pie_chart(question)
            if chart_result:
                context_parts.append(chart_result)
                sources.append("基因型分布图")
            else:
                context_parts.append("未能生成图表，请确认问题中包含季节（如252）和基因（如ty1）。")
        elif intent == "report_query":
            report_result = self._generate_data_report(question)
            if report_result:
                context_parts.append(report_result)
                sources.append("育种数据报告")
            else:
                context_parts.append("未能找到匹配的数据文件，请确认季节编号（如252、261）。")
        elif intent == "data_query":
            data_result = self._query_breeding_data(question)
            if data_result:
                context_parts.append(data_result)
                sources.append("用户育种数据")
        elif intent == "hybrid_query":
            hybrid_result = self._query_hybrid_data(question)
            if hybrid_result:
                context_parts.append(hybrid_result)
                sources.append("杂交组合数据")

        rag_results = self.embedding.search(question, top_k=5)
        for r in rag_results:
            if r["score"] > 0.3:
                context_parts.append(f"【来源: {r['source']}】\n{r['content']}")
                sources.append(r["source"])

        structured_context = self._query_structured_kb(question)
        if structured_context:
            context_parts.append(f"【来源: 育种知识库(结构化)】\n{structured_context}")
            sources.append("结构化知识库")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "无相关知识"

        answer = self._generate_answer(question, context, sources, chat_history)
        return {
            "answer": answer,
            "sources": list(set(sources)),
            "context_count": len(context_parts),
            "intent": intent
        }

    def _handle_greeting(self, question):
        """识别自我介绍/寒暄类问题，返回固定回答"""
        import re
        greetings = [
            r"你是谁", r"你是什么", r"自我介绍", r"介绍一下你自己",
            r"介绍一下你", r"你能做什么", r"你会做什么", r"你有什么功能",
            r"你好", r"您好", r"hi", r"hello", r"在吗",
            r"你叫什么", r"你叫什么名字", r"你是谁呀", r"你是谁啊"
        ]
        q = question.strip().lower()
        for g in greetings:
            if re.search(g, question) or g in q:
                return (
                    "🤝 你好！我是**育种知识助手**，专注于番茄育种领域的智能问答系统。\n\n"
                    "我可以帮你做这些事：\n"
                    "1️⃣ **数据查询**：查你的育种数据，比如\"25年糖度低于5的品种有哪些\"\n"
                    "2️⃣ **图表分析**：生成基因型分布饼图，比如\"把252的Ty1抗感杂合做成饼图\"\n"
                    "3️⃣ **报告生成**：生成品质/性状分析报告，比如\"生成252季度的品质报告\"\n"
                    "4️⃣ **杂交选配**：推荐杂交组合方案\n"
                    "5️⃣ **知识问答**：回答番茄育种、抗病基因等专业问题\n\n"
                    "你可以直接问我育种相关的问题，我会基于知识库和数据给你准确回答。"
                )
        return None

    def _classify_intent(self, question):
        """意图分类：报告生成 / 图表 / 数据查询 / 知识查询"""
        import re
        # 图表生成意图（饼图/统计图/分布图/可视化）
        chart_patterns = [
            r"饼图", r"饼状图", r"统计图", r"分布图", r"图表", r"可视化",
            r"做成图", r"画个", r"画一个", r"出图", r"做成饼"
        ]
        if any(k in question for k in chart_patterns):
            return "chart_query"
        # 报告生成意图（正则匹配，支持"生成252季度的品质报告"这类中间带季节的表述）
        report_patterns = [
            r"生成.{0,12}报告", r"制作.{0,12}报告", r"分析.{0,8}报告",
            r"品质分析", r"性状分析", r"品质报告", r"性状报告",
            r"出.{0,6}报告", r"写.{0,6}报告"
        ]
        if any(re.search(p, question) for p in report_patterns):
            return "report_query"
        if any(kw in question for kw in ["杂交组合", "推荐", "配组", "选配"]):
            return "hybrid_query"
        data_patterns = [
            r'\d+材料', r'哪些品种', r'哪些材料', r'多少份', r'排名', r'排序', r'TOP',
            r'最高', r'最低', r'平均', r'统计', r'数据里', r'材料里',
            r'大于\d', r'小于\d', r'大于\d+', r'小于\d+',
            r'糖度.*\d', r'硬度.*\d', r'果重.*\d',
            r'\d+.*糖度', r'\d+.*硬度', r'\d+.*果重',
            r'\d+季', r'\d{3}.*品质', r'\d{3}.*性状',
            r'筛选', r'查询', r'查找', r'列出',
            r'报告', r'分析',
            # 基因型查询触发词
            r'抗病材料', r'感病材料', r'杂合', r'分子标记', r'回交材料',
            r'抗病', r'感病', r'ty', r'Ty', r'TY', r'Tm', r'tm', r'frl', r'Frl'
        ]
        for pattern in data_patterns:
            if re.search(pattern, question):
                return "data_query"
        return "knowledge_query"

    def _parse_season(self, question):
        """解析季节：支持 252/261 代号 和 25年/26年/2025年 说法。
        返回 [(regex_pattern, label)] 列表，用于匹配文件名。"""
        import re
        matchers = []
        # 1) 3位代号：252、261
        for m in re.findall(r'(?<!\d)(\d{3})(?!\d)', question):
            matchers.append((re.compile(re.escape(m)), m))
        # 2) 完整年份：2025年/2026年
        for m in re.findall(r'(20\d{2})年', question):
            matchers.append((re.compile(re.escape(m)), m))
            short = m[2:]  # 2025 -> 25
            matchers.append((re.compile(short + r'\d'), short))  # 25开头的代号 252/253...
        # 3) 短年份：25年/26年 -> 完整年份 + 25开头的代号
        for m in re.findall(r'(?<!\d)(\d{2})年', question):
            if not m.startswith("20"):
                matchers.append((re.compile('20' + m), '20' + m))
                matchers.append((re.compile(m + r'\d'), m))
        # 去重
        seen = set()
        unique = []
        for pat, lbl in matchers:
            key = (pat.pattern, lbl)
            if key not in seen:
                seen.add(key)
                unique.append((pat, lbl))
        return unique

    def _match_season(self, filename, matchers):
        """判断文件名是否匹配季节，返回标签（如252/2025/25）。
        当 matchers 为空时返回 True（表示不限制季节）。"""
        if not matchers:
            return True
        for pat, lbl in matchers:
            if pat.search(filename):
                return lbl
        return None

    def _parse_conditions(self, question):
        """解析数值条件筛选，支持 大于/高于/超过/以上 和 小于/低于/以下 等说法"""
        import re
        conditions = []
        traits = ["糖度", "硬度", "果重", "单果重"]
        for trait in traits:
            # 大于类
            matched = False
            for pat, op in [
                (r'{}.*?(?:大于|高于|超过|以上|不少于|不低于)\s*(\d+\.?\d*)'.format(trait), '>'),
                (r'{}\s*(?:>=|≥)\s*(\d+\.?\d*)'.format(trait), '>='),
                (r'{}\s*>\s*(\d+\.?\d*)'.format(trait), '>'),
            ]:
                m = re.search(pat, question)
                if m:
                    conditions.append((trait, op, float(m.group(1))))
                    matched = True
                    break
            if matched:
                continue
            # 小于类
            for pat, op in [
                (r'{}.*?(?:小于|低于|以下|不超过|不高于)\s*(\d+\.?\d*)'.format(trait), '<'),
                (r'{}\s*(?:<=|≤)\s*(\d+\.?\d*)'.format(trait), '<='),
                (r'{}\s*<\s*(\d+\.?\d*)'.format(trait), '<'),
            ]:
                m = re.search(pat, question)
                if m:
                    conditions.append((trait, op, float(m.group(1))))
                    break
        return conditions

    def _parse_gene_query(self, question):
        """解析基因型查询（如 ty抗病/感病/杂合、R/H/S），返回 (genes, phenotype) 或 None"""
        import re
        q = question.lower()
        gene_names = ["ty1", "ty2", "ty3", "tm-2a", "tm-1", "tm-2", "frl", "sm", "mi-1", "tobrfv", "sw"]
        has_gene = any(g in q for g in gene_names)
        is_gene_q = any(k in q for k in ["抗病", "感病", "杂合", "分子标记", "基因型", "抗性"])
        if not (has_gene or is_gene_q):
            return None
        # 表型：R=抗病 H=杂合 S=感病
        phenotype = None
        # "抗感杂合"表示要看全部三种分布，不应限定单一表型
        if "抗感杂合" in q or "抗感" in q or "rhs" in q or "r/h/s" in q or "分布" in q or "统计" in q:
            phenotype = None  # 全部
        elif "抗病" in q or "抗性" in q:
            phenotype = 'R'
        elif "杂合" in q:
            phenotype = 'H'
        elif "感病" in q:
            phenotype = 'S'
        # 目标基因
        genes = [g for g in gene_names if g in q]
        if not genes and ("ty" in q or is_gene_q):
            genes = ["ty"]  # 泛指ty（匹配所有ty列）
        return (genes, phenotype)

    def _find_name_col(self, df):
        """找到名称/材料列"""
        import re
        # 1) 列名含名称类关键词
        for col in df.columns:
            if any(kw in str(col) for kw in ["名称", "编号", "品种", "材料", "组合"]):
                # 但"col0/col1"这种修复列名不能算，需确认列里有实际值
                return col
        # 2) 表头修复后的文件：找第一个含组合特征（括号/×/BC）且非全空的列
        best = None
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            sample = non_null.astype(str)
            # 含组合名特征（（×BC）的材料名列
            if sample.str.contains('（|\(|×|BC', regex=True).any():
                return col
            # 记录第一个有较多非空值的文本列作为后备
            if best is None and len(non_null) > 5:
                best = col
        if best is not None:
            return best
        return df.columns[0]

    def _query_breeding_data(self, question):
        """从用户育种数据中查询：支持季节(25年/252)、数值条件、基因型(R/H/S)"""
        import re
        results = []

        # 1. 解析季节
        season_matchers = self._parse_season(question)

        # 2. 解析数值条件（糖度/硬度/果重 + 大于/小于/低于/以上等）
        conditions = self._parse_conditions(question)

        # 3. 解析基因型查询（ty抗病/感病/杂合）
        gene_query = self._parse_gene_query(question)

        # 4. TOP查询 / 统计 / 份数
        top_match = re.search(r'TOP\s*(\d+)', question, re.IGNORECASE)
        top_n = int(top_match.group(1)) if top_match else (5 if any(kw in question for kw in ["排名", "排序", "最高", "最低"]) else 0)
        want_stats = any(kw in question for kw in ["平均", "均值", "统计", "分布"])
        want_count = any(kw in question for kw in ["多少份", "多少材料", "多少品种", "几份", "数量", "有多少"])

        # 5. 遍历数据文件
        for filename, df in self.breeding_data.items():
            # 季节过滤
            season_label = self._match_season(filename, season_matchers)
            if season_matchers and season_label is None:
                continue

            name_col = self._find_name_col(df)

            # ==== 基因型查询分支 ====
            if gene_query is not None:
                genes, phenotype = gene_query
                # 找到目标基因列
                gene_col = None
                for col in df.columns:
                    cl = str(col).lower()
                    if any((g == "ty" and cl.startswith("ty")) or (g in cl) for g in genes):
                        gene_col = col
                        break
                if gene_col is None:
                    continue
                series = df[gene_col].astype(str).str.upper()
                if phenotype:
                    filtered = df[series == phenotype]
                    pheno_str = {"R": "抗病(R)", "H": "杂合(H)", "S": "感病(S)"}.get(phenotype, phenotype)
                else:
                    filtered = df[series.isin(["R", "H", "S"])]
                    pheno_str = "R/H/S"
                if len(filtered) > 0:
                    # 合并单元格前向填充：让同一材料的编号行显示同一材料名
                    fill_df = df.copy()
                    if name_col in fill_df.columns:
                        fill_df[name_col] = fill_df[name_col].ffill()
                    season_info = f"（{season_label}季）" if season_label else ""
                    results.append(f"🧬 文件: {filename}{season_info}")
                    results.append(f"   基因: {gene_col} | 表型: {pheno_str}")
                    # 按材料去重：同一材料只显示一次，标注该材料的单株数量
                    name_series = fill_df[name_col].astype(str) if name_col in fill_df.columns else fill_df.index.astype(str)
                    material_groups = {}
                    for idx, row in filtered.iterrows():
                        name = str(name_series.loc[idx]) if idx in name_series.index else ""
                        if name == "nan":
                            name = "(未标注)"
                        material_groups.setdefault(name, 0)
                        material_groups[name] += 1
                    results.append(f"   符合条件的材料共 {len(material_groups)} 个（{len(filtered)} 个单株）:")
                    shown = 0
                    for name, count in list(material_groups.items())[:30]:
                        if count > 1:
                            results.append(f"   • {name} — {gene_col}: {pheno_str}（{count} 个单株）")
                        else:
                            results.append(f"   • {name} — {gene_col}: {pheno_str}")
                        shown += 1
                    if len(material_groups) > 30:
                        results.append(f"   ...还有 {len(material_groups)-30} 个材料")
                    results.append("")
                continue

            # ==== 数值条件查询分支 ====
            for trait, op, threshold in conditions:
                matching_cols = [col for col in df.columns if trait in str(col) and df[col].dtype in ['float64', 'int64']]
                if not matching_cols:
                    continue
                col = matching_cols[0]
                col_data = pd.to_numeric(df[col], errors='coerce')
                if op == '>':
                    filtered = df[col_data > threshold]
                    op_str = f"{trait} > {threshold}"
                elif op == '>=':
                    filtered = df[col_data >= threshold]
                    op_str = f"{trait} >= {threshold}"
                elif op == '<':
                    filtered = df[col_data < threshold]
                    op_str = f"{trait} < {threshold}"
                elif op == '<=':
                    filtered = df[col_data <= threshold]
                    op_str = f"{trait} <= {threshold}"
                else:
                    continue
                if len(filtered) > 0:
                    season_info = f"（{season_label}季）" if season_label else ""
                    results.append(f"📁 文件: {filename}{season_info}")
                    results.append(f"   条件: {op_str}")
                    results.append(f"   符合条件的材料共 {len(filtered)} 份:")
                    name_col = name_col if name_col in filtered.columns else filtered.columns[0]
                    for idx, row in filtered.head(30).iterrows():
                        name = str(row.get(name_col, ''))
                        value = row.get(col, '')
                        results.append(f"   • {name} — {trait}: {value}")
                    if len(filtered) > 30:
                        results.append(f"   ...还有 {len(filtered)-30} 份材料")
                    results.append("")

            # ==== TOP查询 ====
            if top_n > 0:
                # 只查用户提到的指标列（避免"糖度"误匹配"单果重"）
                top_targets = [t for t in ["糖度", "硬度", "果重"] if t in question] or ["糖度", "硬度", "果重"]
                for col in df.columns:
                    if any(kw in str(col) for kw in top_targets) and df[col].dtype in ['float64', 'int64']:
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        if "最低" in question:
                            top_df = df.loc[col_data.nsmallest(top_n).index]
                            top_str = f"最低{top_n}"
                        else:
                            top_df = df.loc[col_data.nlargest(top_n).index]
                            top_str = f"最高{top_n}"
                        season_info = f"（{season_label}季）" if season_label else ""
                        results.append(f"📁 文件: {filename}{season_info}")
                        results.append(f"   {col} {top_str}:")
                        name_col = name_col if name_col in top_df.columns else top_df.columns[0]
                        for idx, row in top_df.iterrows():
                            results.append(f"   {top_df.index.get_loc(idx)+1}. {row.get(name_col, '')} — {col}: {row[col]}")
                        results.append("")

            # ==== 份数统计 ====
            if want_count:
                season_info = f"（{season_label}季）" if season_label else ""
                results.append(f"📁 文件: {filename}{season_info}")
                results.append(f"   材料份数: {len(df)} 行")
                results.append("")

            # ==== 统计查询 ====
            if want_stats:
                for col in df.columns:
                    if any(kw in str(col) for kw in ["糖度", "硬度", "果重", "单果重"]) and df[col].dtype in ['float64', 'int64']:
                        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(col_data) > 0:
                            season_info = f"（{season_label}季）" if season_label else ""
                            results.append(f"📁 文件: {filename}{season_info}")
                            results.append(f"   {col} 统计: 均值={col_data.mean():.2f}, "
                                         f"最大={col_data.max():.2f}, 最小={col_data.min():.2f}, "
                                         f"标准差={col_data.std():.2f}, 样本数={len(col_data)}")
                            results.append("")

        return "\n".join(results) if results else None

    def _generate_data_report(self, question):
        """生成数据报告：汇总某季节的品质/性状数据"""
        import re
        results = []

        # 识别季节（252/261代号，或25年/26年说法）
        season_matchers = self._parse_season(question)

        want_quality = "品质" in question or "果实" in question
        want_trait = "性状" in question
        if not want_quality and not want_trait:
            want_quality = True
            want_trait = True

        for filename, df in self.breeding_data.items():
            # 匹配季节
            matched_season = self._match_season(filename, season_matchers)
            if season_matchers and matched_season is None:
                continue

            name_col = None
            for col in df.columns:
                if any(kw in str(col) for kw in ["名称", "编号", "品种", "材料", "组合"]):
                    name_col = col
                    break
            if name_col is None:
                name_col = df.columns[0]

            is_quality_file = any(kw in str(df.columns).lower() for kw in ["糖度", "硬度", "果重", "品质"])
            is_trait_file = any(kw in str(df.columns) for kw in ["生长习性", "裂果", "果肩", "熟前果色"])

            if want_quality and is_quality_file:
                season_info = f"（{matched_season}季）" if matched_season else ""
                results.append(f"{'═' * 40}")
                results.append(f"📊 品质数据报告 {season_info}")
                results.append(f"📁 文件: {filename}")
                results.append(f"   样本数: {len(df)}")

                for col in df.columns:
                    if any(kw in str(col) for kw in ["糖度", "硬度", "果重", "单果重"]) and df[col].dtype in ['float64', 'int64']:
                        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(col_data) > 0:
                            results.append(f"")
                            results.append(f"   【{col}】")
                            results.append(f"     均值: {col_data.mean():.2f}")
                            results.append(f"     最大值: {col_data.max():.2f} (材料: {df.loc[col_data.idxmax(), name_col]})")
                            results.append(f"     最小值: {col_data.min():.2f} (材料: {df.loc[col_data.idxmin(), name_col]})")
                            results.append(f"     标准差: {col_data.std():.2f}")
                            if "糖度" in str(col):
                                high = len(col_data[col_data >= 10])
                                results.append(f"     高糖材料(≥10): {high}份 ({high/len(col_data)*100:.1f}%)")
                            if "硬度" in str(col):
                                good = len(col_data[(col_data >= 8) & (col_data <= 10)])
                                results.append(f"     硬度最佳(8-10): {good}份 ({good/len(col_data)*100:.1f}%)")

                # TOP5
                for col in df.columns:
                    if "糖度" in str(col) and df[col].dtype in ['float64', 'int64']:
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        top5 = df.loc[col_data.nlargest(5).index]
                        results.append(f"")
                        results.append(f"   【糖度TOP5】")
                        for i, (idx, row) in enumerate(top5.iterrows()):
                            results.append(f"     {i+1}. {row.get(name_col, '')} — 糖度: {row[col]}")
                        break

                results.append("")

            if want_trait and is_trait_file:
                season_info = f"（{matched_season}季）" if matched_season else ""
                results.append(f"{'═' * 40}")
                results.append(f"📋 性状数据报告 {season_info}")
                results.append(f"📁 文件: {filename}")
                results.append(f"   样本数: {len(df)}")

                for col in df.columns:
                    if col == name_col:
                        continue
                    col_str = str(col)
                    if any(kw in col_str for kw in ["生长习性", "裂果", "果肩", "熟前果色", "萼片", "花序", "熟性", "生长势"]):
                        value_counts = df[col].value_counts()
                        if len(value_counts) > 0 and len(value_counts) <= 10:
                            results.append(f"")
                            results.append(f"   【{col}】分布:")
                            for val, count in value_counts.items():
                                results.append(f"     {val}: {count}份 ({count/len(df)*100:.1f}%)")

                results.append("")

        if not results:
            season_labels = [lbl for _, lbl in season_matchers]
            if season_labels:
                return f"未找到包含 '{season_labels}' 的数据文件。请确认季节编号（如252、261）。可用文件：\n" + "\n".join(self.breeding_data.keys())
            return None

        return "\n".join(results)

    def _generate_pie_chart(self, question):
        """生成基因型分布饼图（HTML/ECharts）。识别季节+基因，统计 R/H/S 分布。"""
        import re
        import json
        import webbrowser

        # 解析季节和基因（季节可为空=统计全部文件）
        season_matchers = self._parse_season(question)
        gene_query = self._parse_gene_query(question)
        if gene_query is None:
            return None
        genes, _ = gene_query
        gene_name = genes[0] if genes else "ty"
        gene_lower = gene_name.lower()

        # 统计分布：抗病/杂合/感病/无条带（"无/无带/无条带"及空值都归为"无条带"）
        distribution = {"R": 0, "H": 0, "S": 0, "无条带": 0}
        file_name = None
        for filename, df in self.breeding_data.items():
            if not self._match_season(filename, season_matchers):
                continue
            gene_col = None
            for col in df.columns:
                cl = str(col).lower()
                if (gene_lower == "ty" and cl.startswith("ty")) or (gene_lower in cl):
                    gene_col = col
                    file_name = filename
                    break
            if gene_col is None:
                continue
            for raw in df[gene_col]:
                val = str(raw).strip().upper()
                # 归一化："无/无带/无条带"及空值（缺失）都归为"无条带"
                if val in ("", "NAN", "NONE", "NA", "NULL", "无", "无带", "无条带", "NO", "NONE条带"):
                    distribution["无条带"] += 1
                elif val in distribution:
                    distribution[val] += 1
                else:
                    distribution["无条带"] += 1

        total = sum(distribution.values())
        if total == 0:
            return None

        stats = {k: v for k, v in distribution.items() if v > 0}
        chart_data = [
            {"name": {"R": "抗病 (R)", "H": "杂合 (H)", "S": "感病 (S)", "无条带": "无条带"}.get(k, k),
             "value": v}
            for k, v in stats.items()
        ]

        season_label = season_matchers[0][1] if season_matchers else "全部数据"
        title = f"{season_label} {gene_name} 基因型分布"

        # 保存 HTML 到项目 charts 目录
        charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts")
        os.makedirs(charts_dir, exist_ok=True)
        safe_season = str(season_label).replace(" ", "").replace("/", "_")
        safe_gene = gene_name.replace(" ", "").replace("/", "_")
        html_path = os.path.join(charts_dir, f"pie_{safe_season}_{safe_gene}.html")

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
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # 自动打开浏览器查看
        try:
            webbrowser.open("file:///" + html_path.replace("\\", "/").replace("\\", "/"))
        except Exception:
            pass

        # 返回可展示的文本结果
        dist_lines = "\n".join(
            f"   • {label} {cnt} 个单株 ({cnt/total*100:.1f}%)"
            for label, cnt in stats.items()
        )
        return (f"📊 {title}\n"
                f"📁 文件: {file_name}\n"
                f"   总样本: {total} 个单株\n"
                f"{dist_lines}\n"
                f"🖼 饼图已生成并打开: {html_path}")

    def _query_hybrid_data(self, question):
        """查询杂交组合数据"""
        import re
        results = []

        for filename, df in self.breeding_data.items():
            if "杂交" not in filename and "组合" not in filename:
                continue

            results.append(f"📁 文件: {filename}")
            results.append(f"   样本数: {len(df)}")

            # 显示评级分布
            for col in df.columns:
                if "评级" in str(col):
                    value_counts = df[col].value_counts()
                    results.append(f"")
                    results.append(f"   【评级分布】")
                    for val, count in value_counts.items():
                        if pd.notna(val) and str(val).strip():
                            results.append(f"     {val}: {count}个组合")
                    break

            # 显示糖度TOP
            for col in df.columns:
                if "糖度" in str(col) and df[col].dtype in ['float64', 'int64']:
                    col_data = pd.to_numeric(df[col], errors='coerce')
                    top5 = df.loc[col_data.nlargest(5).index]
                    results.append(f"")
                    results.append(f"   【糖度TOP5组合】")
                    name_col = df.columns[0]
                    for i, (idx, row) in enumerate(top5.iterrows()):
                        results.append(f"     {i+1}. {row.get(name_col, '')} — 糖度: {row[col]}")
                    break

            results.append("")

        return "\n".join(results) if results else None

    def _query_structured_kb(self, question):
        """从结构化知识库中查询"""
        results = []

        if "Ty1" in question or "Ty3" in question or "Ty2" in question:
            for gene in ["Ty1", "Ty2", "Ty3"]:
                if gene in self.structured_kb.get("抗病基因知识", {}):
                    info = self.structured_kb["抗病基因知识"][gene]
                    results.append(f"{gene}: {info.get('全称', '')}, "
                                   f"抗病类型: {info.get('抗病类型', '')}, "
                                   f"遗传方式: {info.get('遗传方式', '')}")

        if "Tm" in question:
            for gene in ["Tm-1", "Tm-2", "Tm-2a"]:
                if gene in self.structured_kb.get("抗病基因知识", {}):
                    info = self.structured_kb["抗病基因知识"][gene]
                    results.append(f"{gene}: {info.get('全称', '')}, "
                                   f"抗病类型: {info.get('抗病类型', '')}")

        if "杂交" in question or "组合" in question or "配组" in question:
            principles = self.structured_kb.get("杂交组合原则", {})
            for key, val in principles.items():
                results.append(f"{key}: {val}")

        if "糖度" in question:
            standards = self.structured_kb.get("评价标准", {}).get("糖度", {})
            results.append(f"糖度评价标准: {json.dumps(standards, ensure_ascii=False)}")

        if "硬度" in question:
            standards = self.structured_kb.get("评价标准", {}).get("硬度", {})
            results.append(f"硬度评价标准: {json.dumps(standards, ensure_ascii=False)}")

        return "\n".join(results) if results else None

    def _generate_answer(self, question, context, sources, chat_history=None):
        """调用大模型生成回答，失败时使用本地检索结果"""
        system_prompt = """你是一位资深的番茄育种专家和AI助手。请基于以下知识库内容回答用户问题。

【重要规则】
1. 只基于提供的知识库内容回答，不要编造信息
2. 如果知识库中没有相关信息，明确告知用户"知识库中暂无相关信息"
3. 回答中引用知识来源，如"根据抗病基因详解文档..."
4. 涉及数据时，引用具体数值
5. 用育种专业语言回答，但确保清晰易懂"""

        user_prompt = f"""【知识库内容】
{context}

【用户问题】
{question}

请基于知识库内容回答上述问题。如果知识库中有数据支撑，请引用具体数据。"""

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for h in chat_history[-4:]:
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_prompt})

        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 2000
        }

        try:
            resp = requests.post(self.CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return self._local_answer(question, context, sources, str(e))

    def _local_answer(self, question, context, sources, error_msg):
        """API不可用时的本地回答（基于检索结果直接组织）"""
        if context == "无相关知识":
            return "抱歉，知识库中暂无与您问题直接相关的内容。请尝试换个问法，或检查知识库是否已更新。"

        lines = []

        context_blocks = context.split("\n\n---\n\n")
        has_data = False
        for i, block in enumerate(context_blocks):
            if "📊" in block or "📋" in block or "📁" in block or "🧬" in block or "═" in block or "条件:" in block:
                lines.append(block)
                has_data = True
            elif "【来源:" in block:
                end = block.find("】")
                source_match = block[:end+1]
                block_content = block[end+1:].strip()
                if block_content:
                    lines.append(f"{'─' * 40}")
                    lines.append(source_match)
                    lines.append(block_content[:800])
                    lines.append("")

        if not has_data:
            lines.insert(0, f"📌 问题：{question}")
            lines.insert(1, "")
            lines.append("─" * 40)

        lines.append(f"📚 知识来源：{', '.join(sources)}")
        lines.append("")
        lines.append("⚠️ 注：大模型API暂不可用，以上为知识库+数据检索结果。")

        return "\n".join(lines)


if __name__ == "__main__":
    engine = RAGEngine()

    print("\n" + "=" * 60)
    print("  育种知识库问答系统 - 测试")
    print("=" * 60)

    test_questions = [
        "Ty1基因和Ty3基因同时携带的材料抗病性会怎样？",
        "糖度大于10的品种怎么评价？",
        "杂交组合选配的原则是什么？",
        "硬度和口感有什么关系？",
    ]

    for q in test_questions:
        print(f"\n问：{q}")
        result = engine.query(q)
        print(f"答：{result['answer'][:200]}...")
        print(f"来源：{result['sources']}")
        print("-" * 40)
