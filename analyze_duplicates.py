import os
import re
from collections import defaultdict
import shutil
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import logging
import warnings
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 未安裝，嘗試安裝...")
    os.system("pip3 install PyPDF2")
    from PyPDF2 import PdfReader
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("scikit-learn 未安裝，嘗試安裝...")
    os.system("pip3 install scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 設置日誌配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("duplicate_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略 PyPDF2 的警告訊息
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")

def extract_info(filename):
    """從檔案名稱中提取主題和日期/版本資訊"""
    # 移除副檔名
    name = os.path.splitext(filename)[0]
    # 尋找日期格式 (YYYYMM) 從檔名倒數第六碼起
    date_match = re.search(r'_(\d{6})$', name)
    date = date_match.group(1) if date_match else "000000"
    # 提取主題 (移除日期部分)
    topic = name.split('_')[0] if date_match else name
    # 提取版本號 (如果有)
    version_match = re.search(r'v(\d+\.\d+(\.\d+)?)', name)
    version = version_match.group(1) if version_match else "0.0"
    return topic, date, version

def format_date(date_str):
    """格式化日期字符串為可比較的格式"""
    try:
        if len(date_str) == 6:
            return datetime.strptime(date_str, "%Y%m")
        else:
            return datetime(1, 1, 1)  # 默認日期為最早可能的日期
    except ValueError:
        return datetime(1, 1, 1)  # 默認日期為最早可能的日期

async def extract_pdf_content_async(filename, pages=10, executor=None):
    """異步提取PDF檔案的前幾頁文字內容"""
    loop = asyncio.get_event_loop()
    try:
        # 使用線程池執行阻塞的PDF提取操作
        text = await loop.run_in_executor(executor, extract_pdf_content_sync, filename, pages)
        return text
    except Exception as e:
        logger.error(f"無法提取 {filename} 的內容: {e}")
        return ""

def extract_pdf_content_sync(filename, pages=10):
    """同步提取PDF檔案的前幾頁文字內容"""
    try:
        reader = PdfReader(filename)
        text = ""
        for i, page in enumerate(reader.pages):
            if i < pages:
                text += page.extract_text()
            else:
                break
        return text
    except Exception as e:
        raise e

def compute_similarity_sync(contents):
    """同步計算內容之間的相似度"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except ValueError as e:
        logger.error(f"計算相似度時出現錯誤: {e}")
        logger.info("使用默認相似度矩陣（全為0）")
        return np.zeros((len(contents), len(contents)))

async def compute_similarity_async(contents, executor=None):
    """異步計算內容之間的相似度"""
    loop = asyncio.get_event_loop()
    try:
        # 使用進程池執行阻塞的相似度計算操作
        similarity_matrix = await loop.run_in_executor(executor, compute_similarity_sync, contents)
        return similarity_matrix
    except Exception as e:
        logger.error(f"計算相似度時出現錯誤: {e}")
        logger.info("使用默認相似度矩陣（全為0）")
        return np.zeros((len(contents), len(contents)))

def get_all_files(directory):
    """遞歸獲取目錄及其子目錄中的所有PDF檔案，排除備份目錄"""
    pdf_files = []
    backup_dir_name = "backup_duplicates_content_similarity"
    for root, dirs, files in os.walk(directory):
        # 排除備份目錄
        if backup_dir_name in dirs:
            dirs.remove(backup_dir_name)
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

async def main(base_directory=None):
    # 如果未提供目錄，則使用默認目錄
    if base_directory is None:
        base_directory = "/Users/lio/Desktop/HPE 產品小幫手 - 資料準備"
    
    logger.info(f"開始處理目錄: {base_directory}")
    # 獲取所有PDF檔案
    logger.info("正在獲取所有PDF檔案路徑...")
    files = get_all_files(base_directory)
    
    if not files:
        logger.warning("未找到任何PDF檔案。")
        return
    
    # 用於存儲檔案資訊的分組
    file_groups = defaultdict(list)
    contents = []
    file_to_index = {}
    file_paths = []
    
    logger.info(f"找到 {len(files)} 個PDF檔案。正在異步提取PDF前10頁內容，這可能需要一些時間...")
    # 使用線程池來並行處理PDF內容提取
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as thread_executor:
        tasks = [extract_pdf_content_async(f, pages=10, executor=thread_executor) for f in files]
        results = await asyncio.gather(*tasks)
        
        for i, (f, content) in enumerate(zip(files, results)):
            contents.append(content if content else " ")
            file_to_index[f] = i
            file_paths.append(f)
            topic, date, version = extract_info(os.path.basename(f))
            file_groups[topic].append((f, date, version))
    
    logger.info("正在異步計算內容相似度，這可能需要一些時間...")
    # 使用進程池來並行處理相似度計算
    with ProcessPoolExecutor(max_workers=1) as process_executor:
        content_similarity_matrix = await compute_similarity_async(contents, executor=process_executor)
    
    # 第一階段：內容相似度比對，保留較新檔案
    to_keep_content = []
    to_move_content = []
    
    for topic, group in file_groups.items():
        if len(group) > 1:
            logger.info(f"主題 '{topic}' 有 {len(group)} 個檔案進行內容相似度分析:")
            # 按日期和版本排序
            sorted_group = sorted(group, key=lambda x: (format_date(x[1]), x[2]), reverse=True)
            seen_indices = set()
            for i, (filename, date, version) in enumerate(sorted_group):
                file_index = file_to_index[filename]
                if i == 0:
                    logger.info(f"  保留: {os.path.basename(filename)} (最新日期: {date}, 版本: {version})")
                    to_keep_content.append(filename)
                    seen_indices.add(file_index)
                else:
                    similar_to_kept = False
                    for kept_index in seen_indices:
                        similarity = content_similarity_matrix[file_index][kept_index]
                        if similarity >= 0.8:  # 內容相似度閾值設為0.8
                            similar_to_kept = True
                            break
                    if similar_to_kept:
                        logger.info(f"  移動: {os.path.basename(filename)} (內容相似, 日期: {date}, 版本: {version})")
                        to_move_content.append(filename)
                    else:
                        logger.info(f"  保留: {os.path.basename(filename)} (日期: {date}, 版本: {version})")
                        to_keep_content.append(filename)
                        seen_indices.add(file_index)
        else:
            if group:
                to_keep_content.append(group[0][0])
    
    # 移動重覆檔案到子目錄下的備份目錄
    for file_to_move in to_move_content:
        src = file_to_move
        sub_dir = os.path.dirname(file_to_move)
        # 確保只使用最上層的備份目錄，避免嵌套
        backup_dir_content = os.path.join(sub_dir, "backup_duplicates_content_similarity")
        if not os.path.exists(backup_dir_content):
            os.makedirs(backup_dir_content)
        dst = os.path.join(backup_dir_content, os.path.basename(file_to_move))
        if os.path.exists(src):  # 檢查源檔案是否存在
            if os.path.exists(dst):  # 檢查目標檔案是否已存在
                logger.warning(f"目標檔案 {dst} 已存在，跳過移動 {src}。")
            else:
                try:
                    logger.info(f"移動 {src} 到 {dst} (內容相似)")
                    shutil.move(src, dst)
                except Exception as e:
                    logger.error(f"移動檔案 {src} 到 {dst} 時發生錯誤: {e}")
        else:
            logger.warning(f"源檔案 {src} 不存在，無法移動。")

    logger.info("完成分析和移動檔案。")
    logger.info(f"保留檔案數量: {len(to_keep_content)}")
    logger.info(f"移動檔案數量: {len(to_move_content)}")

if __name__ == "__main__":
    # 檢查是否有提供目錄路徑作為命令行參數
    base_directory = None
    if len(sys.argv) > 1:
        base_directory = sys.argv[1]
        logger.info(f"使用命令行提供的目錄路徑: {base_directory}")
    else:
        logger.info("未提供目錄路徑，使用默認目錄。")
    
    asyncio.run(main(base_directory))