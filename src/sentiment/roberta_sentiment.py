import os
import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import traceback

# 设置模型路径
model_path = os.path.join("models", "roberta-base-finetuned-dianping-chinese")

# 使用中文微调模型进行情感分析
classifier = pipeline("sentiment-analysis", model=model_path, device=0)  # 如果没有 GPU，将 device 设置为 -1

# 加载中文模型的分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 定义最大 token 长度（与模型一致，通常为 512）
MAX_TOKEN_LENGTH = 512

# 对评论进行截断或分割
def preprocess_reviews(reviews):
    processed_reviews = []
    for review in reviews:
        tokens = tokenizer.encode(review, truncation=True, max_length=MAX_TOKEN_LENGTH)
        truncated_review = tokenizer.decode(tokens, skip_special_tokens=True)
        processed_reviews.append(truncated_review)
    return processed_reviews

# 修改情感分析函数，逐条分析并添加调试信息
def analyze_sentiment(reviews):
    results = []
    processed_reviews = preprocess_reviews(reviews)
    for i, review in enumerate(processed_reviews):
        try:
            sentiment = classifier(review)[0]
            original_label = sentiment['label']
            score = sentiment['score']

            # 提取正负面情感标签
            if "positive" in original_label:
                sentiment_category = 'pos'
            elif "negative" in original_label:
                sentiment_category = 'neg'
            else:
                sentiment_category = 'unknown'

            # # 打印调试信息
            # print(f"评论索引 {i} 的原始预测结果: {sentiment}")

            results.append((reviews[i], sentiment_category, score))
        except Exception as e:
            print(f"评论索引 {i} 处理失败: {e}")
            print(f"评论长度: {len(reviews[i])}")
            print(f"预处理后评论长度: {len(review)}")
            print(traceback.format_exc())
    return results

# 处理单个文件并保存结果为单独的文件
def process_file(file_path, output_dir, scenic_area):
    try:
        # 读取文件
        df = pd.read_csv(file_path)

        # 检查是否包含 comments 列
        if 'comments' not in df.columns:
            raise ValueError(f"文件 {file_path} 缺少 'comments' 列")

        # 过滤空值
        reviews = df['comments'].dropna().tolist()
        print(f"正在分析 {scenic_area} 的评论数据，共 {len(reviews)} 条")

        # 分析情感
        analysis_results = analyze_sentiment(reviews)

        # 保存结果
        results = [
            {
                'score': df['评分'].iloc[i] if '评分' in df.columns else None,
                'sentiment_category': sentiment_category,
                'sentiment_score': sentiment_score,
                'comment': review
            }
            for i, (review, sentiment_category, sentiment_score) in enumerate(analysis_results)
        ]
        output_df = pd.DataFrame(results)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 输出到文件
        output_file_path = os.path.join(output_dir, f"sentiment_{scenic_area}.csv")
        output_df.to_csv(output_file_path, index=False)
        print(f"分析结果已保存至 {output_file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        print(traceback.format_exc())

# 批量处理目录中的所有文件
def process_all_files(input_dir, output_dir):
    for file_name in tqdm(os.listdir(input_dir), desc="处理文件"):
        if file_name.endswith(".csv") and file_name.startswith("result_"):
            scenic_area = file_name.split("result_")[1].replace(".csv", "")
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, output_dir, scenic_area)

# 指定输入目录和输出目录
input_directory = 'data\processed_data'
output_directory = 'data\roberta_results'

# 执行批量处理
process_all_files(input_directory, output_directory)
