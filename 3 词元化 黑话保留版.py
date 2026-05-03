import pandas as pd
import spacy
import re
import ast
import unicodedata
from spacy.tokenizer import Tokenizer

# 1. 加载模型，禁用不需要的组件以提速
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# 自定义分词器：只按空格切分，保护 Emoji 和特殊符号
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

# --- 深度系统化词典 ---

# 【保留】仅保留基础语法还原，确保 AntConc 统计时动词一致性
# 注意：不再包含 PLATFORM_MAP 和 SLANG_MAP，因为我们要保留黑话原貌
INFORMAL_AND_GRAMMAR_MAP = {
    r"n't\b": " not", r"'s\b": " is", r"'m\b": " am", 
    r"'re\b": " are", r"'ve\b": " have", r"'ll\b": " will", r"'d\b": " would"
}

# 【新增】将黑话加入保护名单，防止 spaCy 将其错误还原（例如把 slaying 还原成 slay 是对的，但要把 slay 还原成别的就不行）
# 这里可以放入你特别关心的词
PROTECT_WORDS = {'slay', 'cap', 'fr', 'bc', 'pov', 'fyp', 'chilling'}

def strip_accents(text):
    """去除拼音声调"""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')

def lemmatize_for_antconc(tokens):
    if not tokens: return ""
    
    # 1. 拼回句子并去除拼音声调
    text = " ".join(tokens)
    text = strip_accents(text)
    
    # 2. 执行语法还原（仅处理缩合词，不处理黑话）
    for pattern, replacement in INFORMAL_AND_GRAMMAR_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 3. 压缩重复字母 (sooooo -> soo)
    # 这对保留黑话很有用，能把 "slayyyyy" 统一为 "slay"
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1", text) 
    
    # 4. spaCy 词元还原
    doc = nlp(text)
    final_tokens = []
    
    for token in doc:
        word = token.text.lower()
        
        # A. 保护非英文字符 (标点、Emoji)
        if not re.search(r'[a-z]', word):
            final_tokens.append(word)
            continue
            
        # B. 保护特定黑话原型
        if word in PROTECT_WORDS:
            final_tokens.append(word)
            continue
            
        # C. 基础词元还原 (例如: running -> run)
        lemma = token.lemma_.lower()
        if lemma == "-pron-":
            final_tokens.append(word)
        else:
            final_tokens.append(lemma)
            
    return " ".join(final_tokens)

def main():
    print(">>> 正在启动系统级词元化（黑话保留版）：")
    print("- 仅还原语法 (n't -> not)")
    print("- 保留原始黑话 (slay, fr, bc, cap 等)")
    print("- 压缩重复字母 (sooo -> so)")

    try:
        df = pd.read_csv('2 分词原始列表.csv', encoding='utf-8-sig')
        df['Tokens'] = df['Tokens'].apply(ast.literal_eval)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # 执行处理
    df['Lemmatized_Content'] = df['Tokens'].apply(lemmatize_for_antconc)
    
    # 导出
    output_df = df[['Content', 'LikeCount', 'Lemmatized_Content']]
    output_df.to_csv('3 词元化语料库_黑话保留版.csv', index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"✅ 第 3 步完成！已生成：3 词元化语料库_黑话保留版.csv")

if __name__ == "__main__":
    main()