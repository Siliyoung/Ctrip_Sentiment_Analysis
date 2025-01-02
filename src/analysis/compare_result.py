import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter

## 1. 加载数据并合并两种方法的结果
# 加载两种情感分析结果
snow_df = pd.read_csv('data/snownlp_result/result_遇龙河景区.csv')  # SnowNLP方法结果
roberta_df = pd.read_csv('data/roberta_results/sentiment_遇龙河景区.csv')  # RoBERTa方法结果

# 合并两种方法的结果
merged_df = pd.merge(snow_df[['score', 'sentiment_category', 'sentiment_score', 'comment']], 
                     roberta_df[['sentiment_category', 'sentiment_score', 'comment']], 
                     on='comment', 
                     suffixes=('_snow', '_roberta'))

# 检查合并后的数据
print(merged_df.head())

## 2. 计算每种情感分析方法的情感分类分布（正面、负面）
# 统计情感分析结果
snow_sentiment_counts = merged_df['sentiment_category_snow'].value_counts()
roberta_sentiment_counts = merged_df['sentiment_category_roberta'].value_counts()

# 可视化情感分布（饼图）
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# SnowNLP情感分析饼图
axes[0].pie(snow_sentiment_counts, labels=snow_sentiment_counts.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('SnowNLP Sentiment Distribution')

# RoBERTa情感分析饼图
axes[1].pie(roberta_sentiment_counts, labels=roberta_sentiment_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('RoBERTa Sentiment Distribution')

plt.show()

## 3. 词云图：每个景点评论的关键词
# 函数：生成评论的词云
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 选择一个景点进行词云分析
scenic_spot = '七星景区'  # 修改为你感兴趣的景区名称
scenic_comments = merged_df[merged_df['scenic_spot'] == scenic_spot]['comment'].tolist()

# 生成词云
generate_wordcloud(scenic_comments)


## 4. 比较两个方法对每个景点的情感分析结果
# 计算每个景区的正负面评论数
snow_sentiment_count_by_spot = merged_df.groupby('scenic_spot')['sentiment_category_snow'].value_counts().unstack(fill_value=0)
roberta_sentiment_count_by_spot = merged_df.groupby('scenic_spot')['sentiment_category_roberta'].value_counts().unstack(fill_value=0)

# 可视化比较
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# SnowNLP的正负面评论数量条形图
snow_sentiment_count_by_spot.plot(kind='bar', stacked=True, ax=axes[0], color=['red', 'green'])
axes[0].set_title('SnowNLP Sentiment Count by Scenic Spot')
axes[0].set_ylabel('Number of Comments')

# RoBERTa的正负面评论数量条形图
roberta_sentiment_count_by_spot.plot(kind='bar', stacked=True, ax=axes[1], color=['red', 'green'])
axes[1].set_title('RoBERTa Sentiment Count by Scenic Spot')
axes[1].set_ylabel('Number of Comments')

plt.tight_layout()
plt.show()

## 5 计算一致性：两种方法分类一致的评论数量
merged_df['sentiment_consistency'] = merged_df['sentiment_category_snow'] == merged_df['sentiment_category_roberta']
consistency_rate = merged_df['sentiment_consistency'].mean()
print(f"两种方法的情感分类一致性：{consistency_rate:.2f}")


