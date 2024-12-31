import pandas as pd
from snownlp import SnowNLP
import os

# 设置预处理数据目录和情感分析结果保存目录
pre_data_dir = 'data\processed_data'
analysis_data_dir = 'data\snownlp_result'

# 如果情感分析结果保存目录不存在，则创建
if not os.path.exists(analysis_data_dir):
    os.makedirs(analysis_data_dir)

# 情感分析函数
def sentiment_analysis(text):
    s = SnowNLP(text)
    score = s.sentiments  # 得到情感分数，范围从 0 到 1
    # 根据情感分数进行二分类
    sentiment_category = 'positive' if score > 0.5 else 'negative'
    return sentiment_category, score  # 返回情感分类和情感得分

# 读取pre_data目录下的所有CSV文件
for file_name in os.listdir(pre_data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(pre_data_dir, file_name)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 应用情感分析函数
        df['sentiment_category'], df['sentiment_score'] = zip(*df['comments'].apply(sentiment_analysis))

        # 保存情感分析后的数据到analysis_data目录
        analysis_file_path = os.path.join(analysis_data_dir, file_name)
        df.to_csv(analysis_file_path, index=False)

        print(f"情感分析结果已保存到: {analysis_file_path}")
