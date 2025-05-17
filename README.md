# PDF Duplicate File Detector

這個專案提供了一個 Python 腳本 `analyze_duplicates.py`，用於識別目錄及其子目錄中是否有重複的 PDF 檔案，特別適合與 Dify LLMOps 配合使用。

## 功能

- 掃描指定目錄及其子目錄以查找重複的 PDF 檔案
- 生成重複 PDF 檔案的分析報告
- 記錄分析結果到日誌檔案 `duplicate_analysis.log`
- 與 Dify LLMOps 整合以增強檔案管理流程

## 安裝

1. 確保您已安裝 Python 3.x
2. 克隆此儲存庫到您的本地機器：
   ```
   git clone https://github.com/leonoxo/PDF-Duplicate-File-Detector.git
   ```
3. 進入專案目錄：
   ```
   cd PDF-Duplicate-File-Detector
   ```

## 使用方法

1. 執行腳本以分析重複的 PDF 檔案：
   ```
   python analyze_duplicates.py
   ```
2. 查看生成的 `duplicate_analysis.log` 檔案以了解分析結果。
3. 可選：將此工具與 Dify LLMOps 平台整合，以自動化重複檔案的識別和管理。

## 貢獻

如果您有任何改進建議或發現了問題，請提交一個 issue 或 pull request。

## 許可證

此專案採用 MIT 許可證 - 詳情請見 LICENSE 檔案。