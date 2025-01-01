import os
import pandas as pd
import re
import csv  # 引入 csv 模块

# === 配置参数 ===
data_dir = "data\\comment_data"  # 原始数据目录
output_dir = "data\\precessed_data"  # 处理后保存的目录
min_reviews = 100  # 每个景区最少评论数量
max_total_reviews = 60000  # 总评论数量上限
allowable_excess = 3000  # 允许的超出数据量

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# === Step 1: 加载数据 ===
file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
all_data = []

for file in file_list:
    file_path = os.path.join(data_dir, file)
    scenic_spot, address = file.replace("result_", "").replace(".csv", "").split("_")
    df = pd.read_csv(file_path)
    df["scenic_spot"] = scenic_spot
    df["address"] = address
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# 输出未处理前的总数据量
print(f"原始数据总量: {len(data)} 条")

# === Step 2: 数据清理和预处理函数 ===
def preprocess_review(review):
    # 去除多余空格
    review = review.strip()
    return review

data["comments"] = data["comments"].apply(preprocess_review)

# === Step 3: 统计景区评论数量 ===
scenic_counts = data["scenic_spot"].value_counts()

# === Step 4: 过滤数据量过少的景区 ===
valid_scenic_spots = scenic_counts[scenic_counts >= min_reviews].index
filtered_data = data[data["scenic_spot"].isin(valid_scenic_spots)]

# === Step 5: 筛选满足数据量需求的景区数据 ===
# 按景区评论数量降序排列
scenic_counts_sorted = scenic_counts.loc[valid_scenic_spots].sort_values(ascending=False)

# 按优先级保留景区
final_data = pd.DataFrame()
selected_scenic_spots = []
current_total = 0

for scenic_spot in scenic_counts_sorted.index:
    spot_data = filtered_data[filtered_data["scenic_spot"] == scenic_spot]
    if current_total + len(spot_data) > max_total_reviews + allowable_excess:
        continue  # 跳过超出限制的景区
    final_data = pd.concat([final_data, spot_data])
    selected_scenic_spots.append((scenic_spot, len(spot_data)))
    current_total += len(spot_data)
    if current_total >= max_total_reviews:
        break

# 输出处理后的数据量
print(f"处理后数据总量: {len(final_data)} 条")

# === Step 6: 按景区分文件保存结果 ===
for scenic_spot in final_data["scenic_spot"].unique():
    spot_data = final_data[final_data["scenic_spot"] == scenic_spot]
    
    # 删除景区名称和地址列
    spot_data = spot_data.drop(columns=["scenic_spot", "address"])
    
    # 数据预处理
    spot_data["comments"] = spot_data["comments"].fillna("").astype(str).apply(preprocess_review)
    
    # 格式化数据为目标样式
    formatted_data = spot_data[["评分", "comments"]]
    
    # 保存为文件，确保所有字段用双引号括起来
    file_name = f"result_{scenic_spot}.csv"
    formatted_data.to_csv(os.path.join(output_dir, file_name), index=False, quoting=csv.QUOTE_ALL)

# === Step 7: 保存附加信息 ===
info_message = "\n".join([f"景区: {scenic_spot}, 地址: {filtered_data[filtered_data['scenic_spot'] == scenic_spot]['address'].iloc[0]}, 数据量: {count}"
                          for scenic_spot, count in selected_scenic_spots])

with open(os.path.join(output_dir, "final_data_info.txt"), "w", encoding="utf-8") as info_file:
    info_file.write("数据处理附加信息:\n")
    info_file.write(info_message)

print(f"数据预处理完成，按景区保存的数据文件已保存在 {output_dir} 文件夹中。")
