import pandas as pd
import re
import unicodedata

def full_cleaning(text):
    if not isinstance(text, str): return ""
    # 1. 规范化：全角转半角
    text = unicodedata.normalize('NFKC', text)
    # 2. 原始逻辑：剔除贴纸、更多等噪声
    text = text.replace(r'[贴纸]', '').replace('更多', '')
    # 3. 原始逻辑：换行/回车/制表符处理成空格
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # 4. 原始逻辑：标点归一化
    text = re.sub(r'\s*([!?,;])\1+', r'\1', text)
    text = re.sub(r'\s*\.{2,}', ' ...', text)
    return text.strip()

def get_reject_reason(text):
    """【完整保留】所有过滤判断功能"""
    if not text or len(text.strip()) < 2: 
        return "太短"
    if re.search(r'[\u4e00-\u9fa5]', text): 
        return "包含中文"
    if re.match(r'^(@\w+\s?)+$', text): 
        return "纯社交提及"
    if re.match(r'^https?://\S+$', text): 
        return "纯链接/广告"
    if not re.search(r'[a-zA-Z]', text): 
        return "无有效英文内容"
    return None

def main():
    # 读取第0步（排序后）的数据
    try:
        df = pd.read_csv('0 排序后数据.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('0 排序后数据.csv', encoding='gb18030')

    print(f">>> 正在执行清洗：保留 Content 和 LikeCount，剔除冗余列...")

    # 执行清洗
    df['Content'] = df['Content'].apply(full_cleaning)
    
    # 执行过滤判断
    df['Reject_Reason'] = df['Content'].apply(get_reject_reason)
    
    # 去重（由于第0步排过序，这里会保留点赞最高的那一条）
    df = df.drop_duplicates(subset=['Content'])
    
    # --- 1. 处理合格语料 ---
    valid_df = df[df['Reject_Reason'].isna()].copy()
    final_valid_df = valid_df[['Content', 'LikeCount']]
    final_valid_df.to_csv('1 最终纯净语料.csv', index=False, encoding='utf-8-sig')
    
    # --- 2. 处理剔除数据备份（核心修改点） ---
    rejected_df = df[df['Reject_Reason'].notna()].copy()
    
    # 自动识别并删除所有 'Unnamed' 开头的列
    # 这样可以动态去掉那四个多余的空列，同时保留 UserName, Content, LikeCount, Reject_Reason 等
    cols_to_keep = [col for col in rejected_df.columns if 'Unnamed' not in col]
    rejected_df = rejected_df[cols_to_keep]
    
    rejected_df.to_csv('1 剔除数据备份.csv', index=False, encoding='utf-8-sig')
    
    print("-" * 40)
    print(f"✅ 第1步完成：")
    print(f" - 合格语料: {len(final_valid_df)} 条 (已保留 Content & LikeCount)")
    print(f" - 剔除备份: {len(rejected_df)} 条 (已移除多余空列，保留有效字段)")
    print("-" * 40)

if __name__ == "__main__":
    main()