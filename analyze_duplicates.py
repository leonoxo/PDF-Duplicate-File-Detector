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
from typing import List, Tuple, Dict, Optional, Set
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

# 配置常量
DEFAULT_DIRECTORY: str = "/Users/lio/Desktop/HPE 產品小幫手 - 資料準備"
BACKUP_DIR_NAME: str = "backup_duplicates_content_similarity"
SIMILARITY_THRESHOLD: float = 0.8
PDF_PAGES_TO_EXTRACT: int = 10


def extract_info(filename: str) -> Tuple[str, str, str]:
    """從檔案名稱中提取主題和日期/版本資訊。
    
    Args:
        filename: 檔案名稱。
    
    Returns:
        包含主題、日期和版本的元組。
    """
    name = os.path.splitext(filename)[0]
    date_match = re.search(r'_(\d{6})$', name)
    date = date_match.group(1) if date_match else "000000"
    topic = name.split('_')[0] if date_match else name
    version_match = re.search(r'v(\d+\.\d+(\.\d+)?)', name)
    version = version_match.group(1) if version_match else "0.0"
    return topic, date, version


def format_date(date_str: str) -> datetime:
    """格式化日期字符串為可比較的格式。
    
    Args:
        date_str: 日期字符串。
    
    Returns:
        格式化後的日期時間對象，若格式錯誤則返回默認最早日期。
    """
    try:
        if len(date_str) == 6:
            return datetime.strptime(date_str, "%Y%m")
        return datetime(1, 1, 1)
    except ValueError:
        logger.error(f"日期格式錯誤: {date_str}，使用默認最早日期")
        return datetime(1, 1, 1)


async def extract_pdf_content_async(filename: str, pages: int = PDF_PAGES_TO_EXTRACT, 
                                   executor: Optional[ThreadPoolExecutor] = None) -> str:
    """異步提取PDF檔案的前幾頁文字內容。
    
    Args:
        filename: PDF檔案路徑。
        pages: 要提取的頁數，默認為常量PDF_PAGES_TO_EXTRACT。
        executor: 線程池執行器。
    
    Returns:
        提取的文本內容，若失敗則返回空字符串。
    """
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(executor, extract_pdf_content_sync, filename, pages)
        return text
    except Exception as e:
        logger.error(f"無法提取 {filename} 的內容: {str(e)}")
        return ""


def extract_pdf_content_sync(filename: str, pages: int = PDF_PAGES_TO_EXTRACT) -> str:
    """同步提取PDF檔案的前幾頁文字內容。
    
    Args:
        filename: PDF檔案路徑。
        pages: 要提取的頁數，默認為常量PDF_PAGES_TO_EXTRACT。
    
    Returns:
        提取的文本內容。
    """
    try:
        reader = PdfReader(filename)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= pages:
                break
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        logger.error(f"同步提取 {filename} 內容時發生錯誤: {str(e)}")
        raise


def compute_similarity_sync(contents: List[str]) -> np.ndarray:
    """同步計算內容之間的相似度。
    
    Args:
        contents: 文本內容列表。
    
    Returns:
        相似度矩陣，若計算失敗則返回全零矩陣。
    """
    try:
        if len(contents) < 2:
            return np.zeros((len(contents), len(contents)))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        return cosine_similarity(tfidf_matrix)
    except ValueError as e:
        logger.error(f"計算相似度時出現錯誤: {str(e)}")
        logger.info("使用默認相似度矩陣（全為0）")
        return np.zeros((len(contents), len(contents)))


async def compute_similarity_async(contents: List[str], 
                                 executor: Optional[ProcessPoolExecutor] = None) -> np.ndarray:
    """異步計算內容之間的相似度。
    
    Args:
        contents: 文本內容列表。
        executor: 進程池執行器。
    
    Returns:
        相似度矩陣，若計算失敗則返回全零矩陣。
    """
    loop = asyncio.get_event_loop()
    try:
        similarity_matrix = await loop.run_in_executor(executor, compute_similarity_sync, contents)
        return similarity_matrix
    except Exception as e:
        logger.error(f"異步計算相似度時出現錯誤: {str(e)}")
        logger.info("使用默認相似度矩陣（全為0）")
        return np.zeros((len(contents), len(contents)))


def get_all_files(directory: str) -> List[str]:
    """遞歸獲取目錄及其子目錄中的所有PDF檔案，排除備份目錄。
    
    Args:
        directory: 目錄路徑。
    
    Returns:
        PDF檔案路徑列表。
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        if BACKUP_DIR_NAME in dirs:
            dirs.remove(BACKUP_DIR_NAME)
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def move_duplicate_files(files_to_move: List[str], backup_dir_name: str = BACKUP_DIR_NAME) -> None:
    """移動重複檔案到備份目錄。
    
    Args:
        files_to_move: 要移動的檔案列表。
        backup_dir_name: 備份目錄名稱，默認為常量BACKUP_DIR_NAME。
    """
    for file_to_move in files_to_move:
        src = file_to_move
        sub_dir = os.path.dirname(file_to_move)
        backup_dir = os.path.join(sub_dir, backup_dir_name)
        try:
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            dst = os.path.join(backup_dir, os.path.basename(file_to_move))
            if os.path.exists(src):
                if os.path.exists(dst):
                    logger.warning(f"目標檔案 {dst} 已存在，跳過移動 {src}。")
                else:
                    logger.info(f"移動 {src} 到 {dst} (內容相似)")
                    shutil.move(src, dst)
            else:
                logger.warning(f"源檔案 {src} 不存在，無法移動。")
        except Exception as e:
            logger.error(f"移動檔案 {src} 到 {dst} 時發生錯誤: {str(e)}")


def group_files_by_topic(files: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    """將檔案按主題分組。
    
    Args:
        files: 檔案路徑列表。
    
    Returns:
        按主題分組的檔案列表字典。
    """
    file_groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for f in files:
        topic, date, version = extract_info(os.path.basename(f))
        file_groups[topic].append((f, date, version))
    return file_groups


def process_topic_group(topic: str, group: List[Tuple[str, str, str]], 
                       file_to_index: Dict[str, int], content_similarity_matrix: np.ndarray, 
                       similarity_threshold: float = SIMILARITY_THRESHOLD) -> Tuple[List[str], List[str]]:
    """處理單個主題組的檔案，找出相似檔案並決定保留和移動的檔案。
    
    Args:
        topic: 主題名稱。
        group: 主題下的檔案列表。
        file_to_index: 檔案到索引的映射。
        content_similarity_matrix: 內容相似度矩陣。
        similarity_threshold: 相似度閾值。
    
    Returns:
        保留的檔案列表和移動的檔案列表。
    """
    to_keep: List[str] = []
    to_move: List[str] = []
    
    if len(group) <= 1:
        if group:
            to_keep.append(group[0][0])
        return to_keep, to_move
    
    logger.info(f"主題 '{topic}' 有 {len(group)} 個檔案進行內容相似度分析:")
    sorted_group = sorted(group, key=lambda x: (format_date(x[1]), x[2]), reverse=True)
    processed_files: Set[str] = set()
    
    for filename, date, version in sorted_group:
        if filename in processed_files:
            continue
        
        file_index = file_to_index[filename]
        similar_files = []
        
        for other_filename, other_date, other_version in sorted_group:
            if other_filename in processed_files or filename == other_filename:
                continue
            other_index = file_to_index[other_filename]
            similarity = content_similarity_matrix[file_index][other_index]
            if similarity >= similarity_threshold:
                similar_files.append((other_filename, other_date, other_version))
        
        if similar_files:
            similar_files.append((filename, date, version))
            similar_files.sort(key=lambda x: format_date(x[1]), reverse=True)
            newest_date = format_date(similar_files[0][1])
            candidates = [f for f in similar_files if format_date(f[1]) == newest_date]
            
            if len(candidates) > 1:
                candidates.sort(key=lambda x: os.path.getsize(x[0]) if os.path.exists(x[0]) else 0, reverse=True)
                logger.info(f"  保留: {os.path.basename(candidates[0][0])} (日期: {candidates[0][1]}, 版本: {candidates[0][2]}, 容量最大)")
            else:
                logger.info(f"  保留: {os.path.basename(candidates[0][0])} (最新日期: {candidates[0][1]}, 版本: {candidates[0][2]})")
            
            to_keep.append(candidates[0][0])
            processed_files.add(candidates[0][0])
            
            for f in similar_files:
                if f[0] != candidates[0][0]:
                    logger.info(f"  移動: {os.path.basename(f[0])} (內容相似, 日期: {f[1]}, 版本: {f[2]})")
                    to_move.append(f[0])
                    processed_files.add(f[0])
        else:
            logger.info(f"  保留: {os.path.basename(filename)} (日期: {date}, 版本: {version})")
            to_keep.append(filename)
            processed_files.add(filename)
    
    return to_keep, to_move


async def analyze_files(base_directory: str = DEFAULT_DIRECTORY) -> None:
    """分析檔案並處理重複檔案。
    
    Args:
        base_directory: 基礎目錄路徑，默認為常量DEFAULT_DIRECTORY。
    """
    logger.info(f"開始處理目錄: {base_directory}")
    logger.info("正在獲取所有PDF檔案路徑...")
    files = get_all_files(base_directory)
    
    if not files:
        logger.warning("未找到任何PDF檔案。")
        return
    
    contents: List[str] = []
    file_to_index: Dict[str, int] = {}
    file_paths: List[str] = []
    
    logger.info(f"找到 {len(files)} 個PDF檔案。正在異步提取PDF前{PDF_PAGES_TO_EXTRACT}頁內容，這可能需要一些時間...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as thread_executor:
        tasks = [extract_pdf_content_async(f, executor=thread_executor) for f in files]
        results = await asyncio.gather(*tasks)
        
        for i, (f, content) in enumerate(zip(files, results)):
            contents.append(content if content else " ")
            file_to_index[f] = i
            file_paths.append(f)
    
    logger.info("正在異步計算內容相似度，這可能需要一些時間...")
    with ProcessPoolExecutor(max_workers=1) as process_executor:
        content_similarity_matrix = await compute_similarity_async(contents, executor=process_executor)
    
    file_groups = group_files_by_topic(files)
    to_keep_content: List[str] = []
    to_move_content: List[str] = []
    
    for topic, group in file_groups.items():
        keep, move = process_topic_group(topic, group, file_to_index, content_similarity_matrix)
        to_keep_content.extend(keep)
        to_move_content.extend(move)
    
    move_duplicate_files(to_move_content)
    
    logger.info("完成分析和移動檔案。")
    logger.info(f"保留檔案數量: {len(to_keep_content)}")
    logger.info(f"移動檔案數量: {len(to_move_content)}")


async def main(base_directory: Optional[str] = None) -> None:
    """主函數，處理命令行參數並啟動分析。
    
    Args:
        base_directory: 基礎目錄路徑，若未提供則使用默認值。
    """
    if base_directory is None:
        if len(sys.argv) > 1:
            base_directory = sys.argv[1]
            logger.info(f"使用命令行提供的目錄路徑: {base_directory}")
        else:
            base_directory = DEFAULT_DIRECTORY
            logger.info("未提供目錄路徑，使用默認目錄。")
    
    await analyze_files(base_directory)


if __name__ == "__main__":
    asyncio.run(main())