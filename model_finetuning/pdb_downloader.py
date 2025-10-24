import pandas as pd
import requests
import os
from tqdm import tqdm # 진행 상황을 보여주는 라이브러리

# --- 설정 (사용자 환경에 맞게 수정) ---
CSV_FILE_PATH = './data/DEDUP90_FINETUNE/2nd_total_except_prup3.csv' # 새로 받은 CSV 파일 경로
PDB_DOWNLOAD_FOLDER = './data/DEDUP90_FINETUNE/PDB/'       # PDB 파일을 저장할 폴더 경로
# -----------------------------------------

def download_pdb(pdb_id, output_folder):
    """지정된 PDB ID의 파일을 다운로드합니다."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_path = os.path.join(output_folder, f"{pdb_id}.pdb")

    # 파일이 이미 존재하면 건너뛰기
    if os.path.exists(save_path):
        # print(f"File {pdb_id}.pdb already exists. Skipping.")
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # HTTP 에러가 발생하면 예외 발생
        
        with open(save_path, 'w') as f:
            f.write(response.text)
        return "Success"
    except requests.exceptions.RequestException as e:
        return f"Failed: {e}"

def main():
    # PDB 저장 폴더가 없으면 생성
    if not os.path.exists(PDB_DOWNLOAD_FOLDER):
        os.makedirs(PDB_DOWNLOAD_FOLDER)

    # CSV 파일 읽기
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully read {CSV_FILE_PATH}.")
    except FileNotFoundError:
        print(f"Error: Cannot find the file '{CSV_FILE_PATH}'. Please check the path.")
        return

    # 'PDB chain' 열이 있는지 확인
    if 'PDB chain' not in df.columns:
        print("Error: 'PDB chain' column not found in the CSV file.")
        return
        
    # PDB ID 목록 추출 (예: '7S2R_A'에서 '_A'를 제외하고 '7S2R'만 사용)
    pdb_ids_with_chains = df['PDB chain'].unique()
    unique_pdb_ids = sorted(list(set([item.split('_')[0] for item in pdb_ids_with_chains])))
    
    print(f"Found {len(unique_pdb_ids)} unique PDB IDs to download.")

    # tqdm을 사용하여 다운로드 진행 상황 표시
    results = {}
    with tqdm(total=len(unique_pdb_ids), desc="Downloading PDB files") as pbar:
        for pdb_id in unique_pdb_ids:
            status = download_pdb(pdb_id, PDB_DOWNLOAD_FOLDER)
            results[pdb_id] = status
            pbar.update(1)

    # 다운로드 결과 요약
    success_count = list(results.values()).count("Success")
    failed_ids = [pid for pid, status in results.items() if status != "Success"]
    
    print("\n--- Download Complete ---")
    print(f"Successfully downloaded: {success_count} files")
    if failed_ids:
        print(f"Failed to download: {len(failed_ids)} files")
        print("Failed IDs:", failed_ids)

if __name__ == "__main__":
    main()