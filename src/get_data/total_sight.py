import time
import random
import re
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 数据列表
timeList = []  # 发表时间
ip = []  # IP 属地
scoreList = []  # 评分
comments = []  # 评论文本

def getData(driver, ddl1, j):
    '''获取数据'''
    times = driver.find_elements(By.CSS_SELECTOR, '.commentTime')
    scores = driver.find_elements(By.CSS_SELECTOR, '.averageScore')[1:]  # 评分可能从第二个元素开始
    comment = driver.find_elements(By.CSS_SELECTOR, '.commentDetail')

    for c, t, s in zip(comment, times, scores):
        try:
            # 提取评论时间
            timeList.append(re.findall(r'(\d{4}-\d{1,2}-\d{1,2})', t.text)[0])
            # 提取IP属地
            ip.append(re.findall(r"：(.*)", t.text)[0])
            # 提取评分
            scoreList.append(re.findall(r"(.*)分", s.text)[0])
            # 提取评论内容
            comments.append(c.text)
        except Exception as e:
            print(f"Error on page {j}: {e}")
    print(f"共{ddl1}页，第{j}页下载完成...")

def read_sight_data(csv_file):
    """读取CSV文件并返回景区名称和链接的列表"""
    sight_data = pd.read_csv(csv_file, encoding='utf-8')
    return sight_data[['景区名称', '景区链接']]

def sanitize_filename(filename):
    """替换文件名中的非法字符"""
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def is_next_page_button_present(driver):
    """检查翻页按钮是否存在并且可点击"""
    try:
        # 显式等待，最多等待10秒钟，直到翻页按钮可点击
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.ant-pagination-next a'))
        )
        return True
    except:
        return False

def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_address(driver):
    """获取景区地址，增加异常处理"""
    try:
        address_elements = driver.find_elements(By.CSS_SELECTOR, '.baseInfoText')
        if address_elements:
            return address_elements[0].text
        else:
            return "无地址"  # 如果没有找到该元素，返回默认值
    except Exception as e:
        print(f"获取地址时出错: {e}")
        return "无地址"  # 如果出现异常，返回默认值

if __name__ == '__main__':
    # 读取sight_data.csv文件
    sight_data = read_sight_data(r'data\sight_data.csv')  # 读取CSV文件

    # 确保存储数据的目录存在
    ensure_directory_exists(r"data\comment_data")

    # 遍历CSV中的每个景区
    for index, row in sight_data.iterrows():
        id = row['景区名称']  # 景区名称
        url = row['景区链接']  # 景区链接

        # 创建新的浏览器实例
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 无头模式
        options.add_argument('--disable-gpu')  # 禁用 GPU 加速
        driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=options)

        try:
            driver.get(url)  # 进入景区页面
            # 生成合法的文件名
            # 获取地址信息（如果需要）
            address_text = get_address(driver)  # 使用新的 get_address 方法            address_text = sanitize_filename(address_text)  # 清理地址中的非法字符

            filename = f"data/comment_data/result_{id}_{address_text}.csv"  # 生成文件名

            # print(filename)
            # 检查文件是否已经存在
            if os.path.exists(filename):
                print(f"文件 {filename} 已存在，跳过该景区...")
                continue  # 跳过该景区，处理下一个景区

            time.sleep(4)

            # 获取总页数
            ddl = driver.find_elements(By.CSS_SELECTOR, '.ant-pagination')
            
            if not ddl:
                continue
            for t in ddl:
                ddl1 = t.text.split("\n")[-2]
            j = 1
            
            while True:
                t1 = random.uniform(1, 2)
                getData(driver, ddl1, j)  # 获取数据
                j += 1

                # 如果页数大于总页数，退出
                if j > int(ddl1):
                    print(f"已爬取完 {id}，共 {ddl1} 页，结束爬取...")
                    break  # 结束循环


                # 翻页
                element = driver.find_element(By.CSS_SELECTOR, '.ant-pagination-next a')
                driver.execute_script("arguments[0].click();", element)
                time.sleep(t1)

            # 保存数据到 CSV
            result_data = pd.DataFrame({
                "date": timeList,
                "ip 属地": ip,
                "评分": scoreList,
                "comments": comments
            })

            # 保存文件
            result_data.to_csv(filename, encoding='utf-8', index=False)
            print(f"数据已保存至 CSV 文件: {filename}")

            # 清空数据列表，准备抓取下一个景区
            timeList.clear()
            ip.clear()
            scoreList.clear()
            comments.clear()

        finally:
            driver.quit()  # 确保退出当前浏览器会话

    print("所有景区爬取完成")
