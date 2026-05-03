import pandas as pd
from collections import Counter
import re
import spacy
import string

# 1. 极速加载：仅保留词性标注(tagger)和属性分配(attribute_ruler)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'lemmatizer'])

def main():
    print(">>> 正在启动系统级词频统计（已适配拼音与逻辑还原版）...")
    
    try:
        df = pd.read_csv('3 词元化语料库_黑话还原版.csv', encoding='utf-8-sig')
        texts = df['Lemmatized_Content'].fillna("").astype(str).tolist()
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # 2. 配置过滤与映射名单
    # 核心情感/逻辑保留词
    SENTIMENT_RETAIN = {'not', 'no', 'never', 'but', 'however', 'so', 'very', 'too', 'because'}
    # 平台噪音
    PLATFORM_NOISE = {'fyp', 'foryou', 'pov', 'page', 'video', 'tiktok', 'post'}
    # 业务映射 (根据你的研究需求调整)
    SPECIAL_MAPPING = {'chinese': 'china', 'jujube': 'red_date', 'goji': 'goji_berry'}
    # 严苛标点过滤
    PUNCT_STRICT = set(string.punctuation) | {'…', '—', '’', '‘', '“', '”', '...'}

    all_pairs = []

    print(f">>> 正在处理 {len(texts)} 条语料，统计深度词频...")
    
    for doc in nlp.pipe(texts, batch_size=500):
        for token in doc:
            word = token.text.strip().lower()
            
            # --- 核心过滤逻辑 ---
            
            # 1. 过滤空值与纯标点
            if not word or word in PUNCT_STRICT:
                continue

            # 2. 处理非英文字符 (Emoji, 连续符号如 !!!, 或拼音/特殊符号)
            # 如果不含字母，归类为情感信号
            if not re.search(r'[a-z]', word, re.I):
                # 只保留长度 > 0 的有效符号（如 ❤️）
                all_pairs.append((word, '情感符号 (Signal)'))
                continue
            
            # 3. 基础长度过滤 (跳过 i, a 等单字母)
            if len(word) <= 1:
                continue
            
            # 4. 平台噪音过滤
            if word in PLATFORM_NOISE:
                continue

            # 5. 停用词处理 (避开情感保留词)
            if token.is_stop and word not in SENTIMENT_RETAIN:
                continue
            
            # 6. 业务同义词合并
            word = SPECIAL_MAPPING.get(word, word)
            
            # 7. 词性分类优化
            # 拼音词（如 xiexie）常被标为 X (foreign word) 或 NOUN
            pos_label = {
                'NOUN': '名词 (Noun)', 
                'VERB': '动词 (Verb)', 
                'ADJ': '形容词 (Adj)', 
                'ADV': '副词 (Adv)',
                'INTJ': '感叹词 (Interj)', # ugh, wow, xiexie(有时)
                'X': '外来语/其他 (X/Other)' # 拼音常客
            }.get(token.pos_, '其他 (Other)')
            
            all_pairs.append((word, pos_label))

    # 4. 聚合数据
    counts = Counter(all_pairs)
    result_data = [
        {'Word': k[0], 'POS': k[1], 'Frequency': v} 
        for k, v in counts.items()
    ]
    
    # 排序：先按频率降序，再按单词升序
    final_df = pd.DataFrame(result_data).sort_values(by=['Frequency', 'Word'], ascending=[False, True])
    
    # 5. 导出结果
    final_df.to_csv('4 词频统计结果_黑话还原版.csv', index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"✅ 第 4 步完成！共统计 {len(final_df)} 个独立词汇。")
    print("【词频报表预览】")
    # 预览时排除“其他”分类，看更有意义的词
    print(final_df[final_df['POS'] != '其他 (Other)'].head(15).to_string(index=False))

if __name__ == "__main__":
    main()