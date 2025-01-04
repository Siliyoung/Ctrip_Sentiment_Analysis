import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import jieba  # 中文分词库
from collections import Counter



# 1. 读取 final_data_info.txt 文件获取景区信息
def load_scenic_spots_info(file_path):
    scenic_spots_info = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 提取每一行的景区名称及其他信息
            parts = line.strip().split(', ')
            scenic_spot = parts[0].split(': ')[1]
            address = parts[1].split(': ')[1]
            data_size = int(parts[2].split(': ')[1])
            scenic_spots_info.append({
                'scenic_spot': scenic_spot,
                'address': address,
                'data_size': data_size
            })
    return scenic_spots_info

# 2. 读取两种情感分析结果并合并数据
def load_and_merge_data(scenic_spot, snow_dir, roberta_dir):
    # 拼接文件路径
    snow_file_path = os.path.join(snow_dir, f'result_{scenic_spot}.csv')
    roberta_file_path = os.path.join(roberta_dir, f'sentiment_{scenic_spot}.csv')
    
    if not os.path.exists(snow_file_path) or not os.path.exists(roberta_file_path):
        return None  # 文件不存在时跳过

    snow_df = pd.read_csv(snow_file_path)
    roberta_df = pd.read_csv(roberta_file_path)

    # 合并数据
    merged_df = pd.merge(
        snow_df[['score', 'sentiment_category', 'sentiment_score', 'comment']],
        roberta_df[['sentiment_category', 'sentiment_score', 'comment']],
        on='comment',
        suffixes=('_snow', '_roberta')
    )
    return merged_df

# 3. 绘制情感分布饼图['#97A675', '#F2F7F3']
def plot_sentiment_distribution(snow_sentiment_counts, roberta_sentiment_counts, scenic_spot, output_dir):
    # 设置字体，确保中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建一个画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))  # 增加图像的高度，确保标题不被遮挡

    # SnowNLP情感分析饼图
    axes[0].pie(snow_sentiment_counts, labels=snow_sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors= ['#97A675', '#F2F7F3'])
    axes[0].set_title(f'SnowNLP Sentiment Distribution ({scenic_spot})')

    # RoBERTa情感分析饼图
    axes[1].pie(roberta_sentiment_counts, labels=roberta_sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#97A675', '#F2F7F3'])
    axes[1].set_title(f'RoBERTa Sentiment Distribution ({scenic_spot})')
    

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{scenic_spot}_sentiment_distribution.png'))
    plt.show()

# 4. 生成词云图，使用jieba进行中文分词
def generate_wordcloud(text, scenic_spot, output_dir):
    # 停用词列表 (可以根据需要扩展)
    stop_words = set([
        "的", "了", "是", "我", "不", "一个", "在", "有", "和", "就", "人", "都", "也", "很", "什么", "没有", 
        "可以", "会", "对", "他", "这", "它", "我们", "你", "你们", "她", "他们", "她们", "的", "不", "但是",
        "为", "怎么", "已经", "而且", "更", "来", "自己", "怎么", "这样"
    ])
    # 使用jieba进行中文分词，并过滤停用词
    text_cut = jieba.cut(' '.join(text))  # 对文本进行分词
    filtered_text = [word for word in text_cut if word not in stop_words and len(word.strip()) > 1]  # 去掉停用词和单字符的词

    # 统计每个词的出现频率
    word_counts = Counter(filtered_text)

    # 使用wordcloud的generate_from_frequencies方法生成词云
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='C:\\Windows\\Fonts\\msyh.ttc'  # 微软雅黑字体
    ).generate_from_frequencies(word_counts)  # 使用词频来生成词云
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # 保存词云图
    plt.savefig(os.path.join(output_dir, f'{scenic_spot}_wordcloud.png'), bbox_inches='tight')
    plt.show()


# 5. 绘制每个景区的情感分析结果
def analyze_and_plot(scenic_spots_info, snow_dir, roberta_dir, output_dir):
    for spot_info in scenic_spots_info:
        scenic_spot = spot_info['scenic_spot']
        print(f"正在处理: {scenic_spot}")
        
        # 加载并合并两种情感分析结果
        merged_df = load_and_merge_data(scenic_spot, snow_dir, roberta_dir)
        if merged_df is None:
            print(f"未找到相关数据文件，跳过: {scenic_spot}")
            continue
        
        # 计算情感分类分布
        snow_sentiment_counts = merged_df['sentiment_category_snow'].value_counts()
        roberta_sentiment_counts = merged_df['sentiment_category_roberta'].value_counts()

        # 绘制并保存饼图
        plot_sentiment_distribution(snow_sentiment_counts, roberta_sentiment_counts, scenic_spot, output_dir)

        # 生成并保存评论词云图
        scenic_comments = merged_df['comment'].tolist()
        generate_wordcloud(scenic_comments, scenic_spot, output_dir)

# 主程序
def main():
    # 文件路径
    final_data_info_file = 'data/precessed_data/final_data_info.txt'  # 这里确保路径没有拼写错误
    snow_dir = 'data/snownlp_result'
    roberta_dir = 'data/roberta_results'
    output_dir = 'data/images'  # 输出目录，保存结果

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取景区信息
    scenic_spots_info = load_scenic_spots_info(final_data_info_file)
    
    # 分析并绘制图表
    analyze_and_plot(scenic_spots_info, snow_dir, roberta_dir, output_dir)
# 执行主程序
if __name__ == '__main__':
    main()
