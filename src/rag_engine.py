"""
RAG引擎 - 混合检索 + 知识库约束 + LLM生成
结构化知识(JSON) + 非结构化文档(向量检索) + 用户数据查询
"""
import os
import json
import requests
import pandas as pd
from config import API_KEY, CHAT_MODEL, STRUCTURED_KB_PATH, BREEDING_DATA_DIR, VECTOR_STORE_DIR
from embedding_manager import EmbeddingManager


class RAGEngine:
    """育种知识库RAG引擎"""

    CHAT_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def __init__(self):
        self.llm_api_key = API_KEY
        self.chat_model = CHAT_MODEL
        self.embedding = EmbeddingManager()
        self._ensure_index()
        self.structured_kb = self._load_structured_kb()
        self.breeding_data = self._load_breeding_data()

    def _ensure_index(self):
        """确保TF-IDF索引存在，不存在则自动构建"""
        store_path = os.path.join(VECTOR_STORE_DIR, "tfidf_store.npz")
        if not os.path.exists(store_path):
            print("[初始化] 未检测到TF-IDF索引，正在构建...")
            self.embedding.build_index()
            print("[初始化] 索引构建完成")

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
                    if df.columns[0].startswith("Unnamed") and str(df.iloc[0, 1]) == "名称":
                        df = pd.read_excel(filepath, header=1)
                    data[filename] = df
                except Exception:
                    pass
        return data

    def query(self, question, chat_history=None):
        """主查询入口：意图判断 → 混合检索 → LLM生成"""
        intent = self._classify_intent(question)

        context_parts = []
        sources = []

        if intent == "report_query":
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

    def _classify_intent(self, question):
        """意图分类：报告生成 / 数据查询 / 知识查询"""
        import re
        if any(kw in question for kw in ["生成报告", "分析报告", "生成.*报告", "品质分析", "性状分析"]):
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
            r'报告', r'分析'
        ]
        for pattern in data_patterns:
            if re.search(pattern, question):
                return "data_query"
        return "knowledge_query"

    def _query_breeding_data(self, question):
        """从用户育种数据中查询"""
        import re
        results = []

        # 1. 识别季节关键词（如 "252"、"261"）
        season_keywords = re.findall(r'(\d{3})', question)
        season_keywords = [s for s in season_keywords if s in question]

        # 2. 识别条件筛选（如 "糖度大于10"、"硬度>9"）
        conditions = []
        condition_patterns = [
            (r'糖度.*?大于\s*(\d+\.?\d*)', '糖度', '>'),
            (r'糖度.*?超过\s*(\d+\.?\d*)', '糖度', '>'),
            (r'糖度.*?小于\s*(\d+\.?\d*)', '糖度', '<'),
            (r'硬度.*?大于\s*(\d+\.?\d*)', '硬度', '>'),
            (r'硬度.*?大于等于\s*(\d+\.?\d*)', '硬度', '>='),
            (r'硬度.*?小于\s*(\d+\.?\d*)', '硬度', '<'),
            (r'果重.*?大于\s*(\d+\.?\d*)', '果重', '>'),
            (r'果重.*?小于\s*(\d+\.?\d*)', '果重', '<'),
            (r'糖度\s*>=?\s*(\d+\.?\d*)', '糖度', '>'),
            (r'糖度\s*<=?\s*(\d+\.?\d*)', '糖度', '<'),
            (r'硬度\s*>=?\s*(\d+\.?\d*)', '硬度', '>'),
            (r'硬度\s*<=?\s*(\d+\.?\d*)', '硬度', '<'),
        ]
        for pattern, trait, op in condition_patterns:
            match = re.search(pattern, question)
            if match:
                conditions.append((trait, op, float(match.group(1))))

        # 3. 识别TOP查询
        top_match = re.search(r'TOP\s*(\d+)', question, re.IGNORECASE)
        top_n = int(top_match.group(1)) if top_match else (5 if any(kw in question for kw in ["排名", "排序", "最高", "最低"]) else 0)

        # 4. 识别统计查询
        want_stats = any(kw in question for kw in ["平均", "均值", "统计", "分布"])

        # 5. 遍历数据文件查询
        for filename, df in self.breeding_data.items():
            # 检查是否需要按季节过滤
            file_season = None
            for sk in season_keywords:
                if sk in filename:
                    file_season = sk
                    break

            # 如果用户问了特定季节但这个文件不属于该季节，跳过
            if season_keywords and file_season is None:
                continue

            # 找到名称列
            name_col = None
            for col in df.columns:
                if any(kw in str(col) for kw in ["名称", "编号", "品种", "材料", "组合"]):
                    name_col = col
                    break
            if name_col is None:
                name_col = df.columns[0]

            # 找到符合条件的性状列
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
                else:
                    continue

                if len(filtered) > 0:
                    season_info = f"（{file_season}季）" if file_season else ""
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

            # TOP查询
            if top_n > 0:
                for col in df.columns:
                    if any(kw in str(col) for kw in ["糖度", "硬度", "果重"]) and df[col].dtype in ['float64', 'int64']:
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        if "最低" in question:
                            top_df = df.loc[col_data.nsmallest(top_n).index]
                            top_str = f"最低{top_n}"
                        else:
                            top_df = df.loc[col_data.nlargest(top_n).index]
                            top_str = f"最高{top_n}"

                        season_info = f"（{file_season}季）" if file_season else ""
                        results.append(f"📁 文件: {filename}{season_info}")
                        results.append(f"   {col} {top_str}:")
                        name_col = name_col if name_col in top_df.columns else top_df.columns[0]
                        for idx, row in top_df.iterrows():
                            results.append(f"   {top_df.index.get_loc(idx)+1}. {row.get(name_col, '')} — {col}: {row[col]}")
                        results.append("")

            # 统计查询
            if want_stats:
                for col in df.columns:
                    if any(kw in str(col) for kw in ["糖度", "硬度", "果重", "单果重"]) and df[col].dtype in ['float64', 'int64']:
                        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(col_data) > 0:
                            season_info = f"（{file_season}季）" if file_season else ""
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

        # 识别季节（252、261等）或关键词（品质、性状）
        season_match = re.findall(r'(\d{3})', question)
        season_keywords = season_match if season_match else []

        want_quality = "品质" in question or "果实" in question
        want_trait = "性状" in question
        if not want_quality and not want_trait:
            want_quality = True
            want_trait = True

        for filename, df in self.breeding_data.items():
            # 匹配季节
            matched_season = None
            for sk in season_keywords:
                if sk in filename:
                    matched_season = sk
                    break
            if season_keywords and matched_season is None:
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
            if season_keywords:
                return f"未找到包含 '{season_keywords}' 的数据文件。请确认季节编号（如252、261）。可用文件：\n" + "\n".join(self.breeding_data.keys())
            return None

        return "\n".join(results)

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
            "input": {"messages": messages},
            "parameters": {"temperature": 0.3, "top_p": 0.8, "result_format": "message"}
        }

        try:
            resp = requests.post(self.CHAT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            return result["output"]["choices"][0]["message"]["content"]
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
            if "📊" in block or "📋" in block or "📁" in block or "═" in block or "条件:" in block:
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
