import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import jieba  # 中文分词库
from collections import Counter

# 1. 读取并合并所有景区的情感分析结果
def load_and_merge_all_data(snow_dir, roberta_dir, scenic_spots_info):
    all_scenic_comments = []
    snow_sentiment_summary = {'pos': 0, 'neg': 0}
    roberta_sentiment_summary = {'pos': 0, 'neg': 0}

    for spot_info in scenic_spots_info:
        scenic_spot = spot_info['scenic_spot']
        snow_file_path = os.path.join(snow_dir, f'result_{scenic_spot}.csv')
        roberta_file_path = os.path.join(roberta_dir, f'sentiment_{scenic_spot}.csv')
        
        # 如果文件不存在，则跳过该景区
        if not os.path.exists(snow_file_path) or not os.path.exists(roberta_file_path):
            continue
        
        snow_df = pd.read_csv(snow_file_path)
        roberta_df = pd.read_csv(roberta_file_path)

        # 合并两种情感分析结果
        merged_df = pd.merge(
            snow_df[['score', 'sentiment_category', 'sentiment_score', 'comment']],
            roberta_df[['sentiment_category', 'sentiment_score', 'comment']],
            on='comment',
            suffixes=('_snow', '_roberta')
        )
        
        # 汇总情感分类分布
        snow_sentiment_counts = merged_df['sentiment_category_snow'].value_counts()
        roberta_sentiment_counts = merged_df['sentiment_category_roberta'].value_counts()

        snow_sentiment_summary['pos'] += snow_sentiment_counts.get('pos', 0)
        snow_sentiment_summary['neg'] += snow_sentiment_counts.get('neg', 0)

        roberta_sentiment_summary['pos'] += roberta_sentiment_counts.get('pos', 0)
        roberta_sentiment_summary['neg'] += roberta_sentiment_counts.get('neg', 0)

        # 汇总评论
        scenic_comments = merged_df['comment'].tolist()
        all_scenic_comments.extend(scenic_comments)

    return all_scenic_comments, snow_sentiment_summary, roberta_sentiment_summary

# 2. 绘制情感分布饼图
def plot_sentiment_distribution(snow_sentiment_summary, roberta_sentiment_summary, output_dir):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # 处理 SnowNLP 的情感分析分布
    snow_sentiment_counts = pd.Series(snow_sentiment_summary).fillna(0)
    snow_sentiment_counts = snow_sentiment_counts[snow_sentiment_counts > 0]  # 删除0值的情感类别
    if len(snow_sentiment_counts) > 0:
        axes[0].pie(snow_sentiment_counts, labels=snow_sentiment_counts.index, autopct='%1.1f%%', startangle=90,
                    colors=['#97A675', '#F2F7F3'], pctdistance=0.85, textprops={'fontsize': 14})
    else:
        axes[0].pie([1], labels=["No Data"], colors=['#F2F7F3'], pctdistance=0.85)  # 如果没有数据，显示 "No Data"
    axes[0].set_title('SnowNLP Sentiment Distribution')

    # 处理 RoBERTa 的情感分析分布
    roberta_sentiment_counts = pd.Series(roberta_sentiment_summary).fillna(0)
    roberta_sentiment_counts = roberta_sentiment_counts[roberta_sentiment_counts > 0]  # 删除0值的情感类别
    if len(roberta_sentiment_counts) > 0:
        axes[1].pie(roberta_sentiment_counts, labels=roberta_sentiment_counts.index, autopct='%1.1f%%', startangle=90,
                    colors=['#97A675', '#F2F7F3'], pctdistance=0.85, textprops={'fontsize': 14})
    else:
        axes[1].pie([1], labels=["No Data"], colors=['#F2F7F3'], pctdistance=0.85)  # 如果没有数据，显示 "No Data"
    axes[1].set_title('RoBERTa Sentiment Distribution')

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution_summary.png'))
    plt.close()


# 3. 生成所有景区评论的词云图
def generate_wordcloud(all_scenic_comments, output_dir):
    stop_words = set([
        "的", "了", "是", "我", "不", "一个", "在", "有", "和", "就", "人", "都", "也", "很", "什么", "没有", 
        "可以", "会", "对", "他", "这", "它", "我们", "你", "你们", "她", "他们", "她们", "的", "不", "但是",
        "为", "怎么", "已经", "而且", "更", "来", "自己", "怎么", "这样"
    ])
    
    # 对所有评论进行分词处理
    text_cut = jieba.cut(' '.join(all_scenic_comments))
    filtered_text = [word for word in text_cut if word not in stop_words and len(word.strip()) > 1]

    # 统计词频
    word_counts = Counter(filtered_text)

    # 生成词云
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='C:\\Windows\\Fonts\\msyh.ttc'
    ).generate_from_frequencies(word_counts)

    # 绘制词云图
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # 保存词云图
    plt.savefig(os.path.join(output_dir, 'all_scenic_spots_wordcloud.png'), bbox_inches='tight')
    plt.close()

# 4. 汇总分析并生成结果
def generate_summary(snow_dir, roberta_dir, scenic_spots_info, output_dir):
    all_scenic_comments, snow_sentiment_summary, roberta_sentiment_summary = load_and_merge_all_data(snow_dir, roberta_dir, scenic_spots_info)

    # 绘制情感分布饼图
    plot_sentiment_distribution(snow_sentiment_summary, roberta_sentiment_summary, output_dir)

    # 生成所有景区评论的词云图
    generate_wordcloud(all_scenic_comments, output_dir)

# 主程序
def main():
    # 文件路径
    final_data_info_file = 'data/precessed_data/final_data_info.txt'
    snow_dir = 'data/snownlp_result'
    roberta_dir = 'data/roberta_results'
    output_dir = 'data/images'  # 输出目录，保存结果

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取景区信息
    scenic_spots_info = []
    with open(final_data_info_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(', ')
            scenic_spot = parts[0].split(': ')[1]
            address = parts[1].split(': ')[1]
            data_size = int(parts[2].split(': ')[1])
            scenic_spots_info.append({
                'scenic_spot': scenic_spot,
                'address': address,
                'data_size': data_size
            })

    # 汇总分析并生成结果
    generate_summary(snow_dir, roberta_dir, scenic_spots_info, output_dir)

# 执行主程序
if __name__ == '__main__':
    main()
