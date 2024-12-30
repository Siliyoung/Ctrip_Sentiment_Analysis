import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# todo 停不下来
def save_to_csv(data, mode="a"):
    # 保存数据到 CSV 文件，默认以追加模式打开
    with open(r"data\sight_data.csv", mode=mode, newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if mode == "w":
            writer.writerow(["景区名称", "景区链接", "景区等级"])  # 写入表头
        for row in data:
            writer.writerow(row)  # 写入数据行

def test_sight_data_extraction():
    # 设置 Chrome 无头模式
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--disable-gpu')  # 禁用 GPU 加速
    driver = webdriver.Chrome(service=Service('chromedriver.exe'), options=options)

    sight_data = []  # 用于存储爬取的数据
    page_num = 1  # 记录当前页数

    try:
        # 设置初始 URL，第1页
        url = "https://you.ctrip.com/sight/guilin28/s0-p1.html"
        driver.get(url)
        time.sleep(3)  # 等待页面加载

        # 保存数据表头
        save_to_csv([], mode="w")  # 只写表头
        # 循环翻页抓取
        while True:
            print(f"正在抓取第 {page_num} 页数据...")

            # 显式等待：等待至少 5 个景区名称元素加载出来
            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.titleModule_name__Li4Tv')))
            
            # 查找当前页的景区元素
            div_elements = driver.find_elements(By.CSS_SELECTOR, '.titleModule_name__Li4Tv')

            # 抓取当前页的所有景区信息
            for div in div_elements:
                try:
                    # 获取景区名称和链接
                    a_tag = div.find_element(By.TAG_NAME, 'a')
                    sight_name = a_tag.text
                    sight_url = a_tag.get_attribute('href')

                    # 获取景区等级，加入异常处理，若无等级则默认显示“无等级”
                    try:
                        level_span = div.find_element(By.CLASS_NAME, 'titleModule_level-text-view__40Dbg')
                        sight_level = level_span.text
                    except:
                        sight_level = "无等级"  # 如果没有找到等级信息，赋予默认值

                    # 打印当前景区数据
                    print(f"景区名称: {sight_name}")
                    print(f"景区链接: {sight_url}")
                    print(f"景区等级: {sight_level}")
                    print("-" * 40)

                    sight_data.append([sight_name, sight_url, sight_level])  # 将数据保存到列表中

                except Exception as e:
                    print(f"Error extracting data: {e}")

            # 保存当前页数据到文件
            save_to_csv(sight_data)
            sight_data.clear()  # 清空数据，防止内存溢出

            # 查找并点击下一页按钮
            try:
                next_page_button = driver.find_element(By.CSS_SELECTOR, '.ant-pagination-next a')
                driver.execute_script("arguments[0].click();", next_page_button)
                time.sleep(3)  # 等待新页面加载
                page_num += 1  # 增加页数
            except:
                print("已经是最后一页，退出翻页")
                break  # 如果没有找到“下一页”按钮，则退出循环

        print("数据抓取完成，所有数据已保存到 'sight_data.csv' 文件中。")
        
    finally:
        driver.quit()

# 运行测试
if __name__ == "__main__":
    test_sight_data_extraction()
