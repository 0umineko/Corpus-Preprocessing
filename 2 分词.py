import pandas as pd
import re

def tokenize_all_reserved(text):
    if not isinstance(text, str): return []
    
    # 1. 预处理：规范化撇号
    text = text.replace("’", "'").replace("‘", "'")
    # 注意：建议不要在第一步强行 .lower()，除非你确定不需要区分大小写
    # text = text.lower() 
    
    # 2. 增强型正则
    patterns = [
        # a. 标点表情 (T_T, ^_^, :))
        r"[t_t><\^]{3,}",                   
        r"[:;=x][\-\^]?[pd\)\(\/\\\|dox]",
        r"<3",                              
        
        # b. 动作描述 (*sigh*)
        r"\*[a-zA-Z]+\*",
        
        # c. 重复的情感标点 (..., !!!, ???, ~~~)
        r"\.{2,}",                          
        r"\!+",                             
        r"\?+",                             
        r"\~+",                             
        
        # d. Emoji 综合方案
        r"[\U0001f1e6-\U0001f1ff]{2}" 
        r"|(?:[\U00010000-\U0010ffff]|[\u2600-\u27ff])[\U0001f3fb-\U0001f3ff]?(?:\u200d(?:[\U00010000-\U0010ffff]|[\u2600-\u27ff])[\U0001f3fb-\U0001f3ff]?)*"
        r"|[\u2600-\u27ff]",

        # e. 数字支持 (整数和浮点数)
        r"\d+\.?\d*",
        
        # f. 【核心修复】：支持中文、多国文字及带撇号的单词
        # \w 包含字母、数字、下划线及各语种文字；\u4e00-\u9fa5 是中文范围
        r"[\w\u4e00-\u9fa5']+",
        
        # g. 所有其他单个标点和特殊符号
        r"[^\w\s]"
    ]
    
    combined_regex = "|".join(patterns)
    
    # 提取所有匹配项
    tokens = re.findall(combined_regex, text)
    
    return tokens

def main():
    try:
        # 注意：如果文件里有中文，建议用 utf-8-sig
        df = pd.read_csv('1 最终纯净语料.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("❌ 未找到 '1 最终纯净语料.csv'")
        return

    print(">>> 正在执行【全保留】分词（修复中文及各语种文字识别问题）...")
    
    df['Tokens'] = df['Content'].apply(tokenize_all_reserved)
    
    output_df = df[['Content', 'LikeCount', 'Tokens']]
    output_df.to_csv('2 分词原始列表.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ 第 2 步完成：已保存至 '2 分词原始列表.csv'")

if __name__ == "__main__":
    main()