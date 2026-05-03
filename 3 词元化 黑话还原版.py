import pandas as pd
import spacy
import re
import ast
import unicodedata
from spacy.tokenizer import Tokenizer # 新增导入

# 1. 加载模型，禁用不需要的组件以提速
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# --- 核心优化：防止 Emoji 被切分 ---
# 自定义分词器：只按空格切分，不按标点或 Emoji 内部结构切分
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
# --------------------------------

# --- 深度系统化词典 ---

# 语气与拟声词归一化
INTERJECTION_MAP = {
    r"\bu+g+h+\b": "ugh", r"\be+w+w*\b": "ew",
    r"\ba+h+a+\b": "aha", r"\bm+h+m+\b": "mhm",
    r"\bw+o+w+\b": "wow", r"\by+a+y+\b": "yay",
}

# 平台特有黑话
PLATFORM_MAP = {
    r"\bfyps?\b": "for you page", r"\bpovs?\b": "point of view",
    r"\bomw\b": "on my way", r"\bootd\b": "outfit of the day",
    r"\bcap\b": "lie", r"\bslay\b": "amazing",
}

# 网络社交缩写 (含 bc, w/, ig 等逻辑词)
SLANG_MAP = {
    r"\bbc\b": "because", r"\bb/c\b": "because",
    r"\bw/\b": "with", r"\bw/o\b": "without",
    r"\big\b": "i guess",
    r"\bw[au]s+up\b": "what is up", r"\bsup\b": "what is up", 
    r"\bfr\b": "for real", r"\bistg\b": "i swear to god",
    r"\blm?f?ao\b": "laugh", r"\blol\b": "laugh", 
    r"\baf\b": "as fuck", r"\btbh\b": "to be honest",
    r"\bwth\b": "what the hell", r"\btf\b": "the fuck",
    r"\bu\b": "you", r"\bur\b": "your", r"\br\b": "are",
    r"\bppl\b": "people", r"\bthx\b": "thanks",
    r"\bbtw\b": "by the way", r"\bidk\b": "i do not know",
    r"\bik\b": "i know", r"\brn\b": "right now",
    r"\bngl\b": "not going to lie",
}

# 常见口语缩合与语法还原
INFORMAL_AND_GRAMMAR_MAP = {
    r"\bwanna\b": "want to", r"\bgonna\b": "going to", 
    r"\bkinda\b": "kind of", r"\btryna\b": "trying to",
    r"\bgotta\b": "got to", r"\baint\b": "is not",
    r"\bcuz\b": "because", r"\bcause\b": "because",
    r"n't\b": " not", r"'s\b": " is", r"'m\b": " am", 
    r"'re\b": " are", r"'ve\b": " have", r"'ll\b": " will", r"'d\b": " would"
}

# 保护名单
PROTECT_WORDS = {'chilling'}

def strip_accents(text):
    """
    处理拼音：去除声调符号 (例如: Xièxiè -> XieXie)
    """
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')

def lemmatize_for_antconc(tokens):
    if not tokens: return ""
    
    # 1. 拼回句子并去除拼音声调
    text = " ".join(tokens)
    text = strip_accents(text)
    
    # 2. 执行正则词典替换 (拟声词 -> 平台词 -> 缩写 -> 语法)
    all_maps = [INTERJECTION_MAP, PLATFORM_MAP, SLANG_MAP, INFORMAL_AND_GRAMMAR_MAP]
    for mapping in all_maps:
        for pattern, replacement in mapping.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 3. 压缩重复字母 (sooooo -> soo)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    
    # 4. spaCy 词元还原
    doc = nlp(text)
    final_tokens = []
    
    for token in doc:
        word = token.text.lower()
        
        # A. 保护非英文字符 (标点、Emoji、信号)
        if not re.search(r'[a-z]', word):
            final_tokens.append(word)
            continue
            
        # B. 保护特定梗词
        if word in PROTECT_WORDS:
            final_tokens.append(word)
            continue
            
        # C. 词元还原 (been -> be)
        lemma = token.lemma_.lower()
        if lemma == "-pron-":
            final_tokens.append(word)
        else:
            final_tokens.append(lemma)
            
    return " ".join(final_tokens)

def main():
    print(">>> 正在启动系统级词元化：")
    print("- 逻辑还原 (bc -> because)")
    print("- 拼音脱敏 (Xièxiè -> xiexie)")
    print("- 结构保留 (保留标点与 Emoji)")

    try:
        df = pd.read_csv('2 分词原始列表.csv', encoding='utf-8-sig')
        df['Tokens'] = df['Tokens'].apply(ast.literal_eval)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # 执行处理
    df['Lemmatized_Content'] = df['Tokens'].apply(lemmatize_for_antconc)
    
    # 导出核心列
    output_df = df[['Content', 'LikeCount', 'Lemmatized_Content']]
    output_df.to_csv('3 词元化语料库.csv', index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"✅ 第 3 步完成！已成功生成：3 词元化语料库_黑话还原版.csv")

if __name__ == "__main__":
    main()