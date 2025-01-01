import pandas as pd
from snownlp import SnowNLP
import os

# 设置预处理数据目录和情感分析结果保存目录
pre_data_dir = 'data/precessed_data'  # 使用正斜杠更具可移植性
analysis_data_dir = 'data/snownlp_result'

# 如果情感分析结果保存目录不存在，则创建
os.makedirs(analysis_data_dir, exist_ok=True)

# 情感分析函数
def sentiment_analysis(text):
    s = SnowNLP(text)
    score = s.sentiments  # 得到情感分数，范围从 0 到 1
    # 根据情感分数进行二分类
    sentiment_category = 'pos' if score > 0.5 else 'neg'
    return sentiment_category, score  # 返回情感分类和情感得分

# 读取pre_data目录下的所有CSV文件
for file_name in os.listdir(pre_data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(pre_data_dir, file_name)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 确保'comments'列存在
            if 'comments' not in df.columns:
                print(f"文件 {file_name} 缺少 'comments' 列，跳过该文件。")
                continue

            # 处理空值（NaN）情况，确保不会出错
            df['comments'] = df['comments'].fillna('')

            # 应用情感分析函数
            df['sentiment_category'], df['sentiment_score'] = zip(*df['comments'].apply(sentiment_analysis))

            # 重命名列
            df.rename(columns={'评分': 'score', 'comments': 'comment'}, inplace=True)

            # 重新排序列
            df = df[['score', 'sentiment_category', 'sentiment_score', 'comment']]

            # 保存情感分析后的数据到analysis_data目录
            analysis_file_path = os.path.join(analysis_data_dir, file_name)
            df.to_csv(analysis_file_path, index=False)

            print(f"情感分析结果已保存到: {analysis_file_path}")
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
