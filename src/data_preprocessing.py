import pandas as pd
import os

# --- Configuration ---
# กำหนด Path ของไฟล์ (โดยอ้างอิงจาก Root Directory ของโปรเจกต์)
RAW_PATH = 'data/raw/'
PROCESSED_PATH = 'data/processed/'
OUTPUT_FILENAME = 'merged_oil_prices.csv'

def load_and_clean_data():
    """
    ฟังก์ชันสำหรับโหลดข้อมูลดิบ, ทำความสะอาด, และรวมไฟล์
    """
    print("🔄 Loading raw data...")
    
    # 1. เช็กก่อนว่าไฟล์มีอยู่จริงไหม
    brent_path = os.path.join(RAW_PATH, 'brent_prices.csv')
    wti_path = os.path.join(RAW_PATH, 'wti_prices.csv')
    
    if not os.path.exists(brent_path) or not os.path.exists(wti_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ข้อมูลใน {RAW_PATH} กรุณาเช็กชื่อไฟล์หรือตำแหน่งโฟลเดอร์")

    # 2. อ่านไฟล์ CSV
    df_brent = pd.read_csv(brent_path)
    df_wti = pd.read_csv(wti_path)

    # 3. แปลงคอลัมน์ Date ให้เป็น format วันที่จริงๆ (Datetime Object)
    # เพื่อให้ Python เข้าใจว่านี่คือ "เวลา" ไม่ใช่แค่ตัวหนังสือ
    df_brent['date'] = pd.to_datetime(df_brent['date'])
    df_wti['date'] = pd.to_datetime(df_wti['date'])

    # 4. เปลี่ยนชื่อคอลัมน์ราคา (Price) ให้ชัดเจนก่อนรวม
    # จากเดิมชื่อ 'Price' เหมือนกันทั้งคู่ เดี๋ยวจะงง
    df_brent = df_brent.rename(columns={'Price': 'Brent_Price'})
    df_wti = df_wti.rename(columns={'Price': 'WTI_Price'})

    print(f"   - Brent data points: {len(df_brent)}")
    print(f"   - WTI data points:   {len(df_wti)}")

    # 5. Merge ข้อมูล (Inner Join)
    # ใช้ 'inner' เพื่อเอาเฉพาะวันที่ 'มีข้อมูลทั้งคู่' เท่านั้น
    # (ตัดวันที่ตลาดฝั่งใดฝั่งหนึ่งปิดออกไป เพื่อป้องกัน Missing Value)
    print("🔄 Merging datasets...")
    df_merged = pd.merge(df_brent, df_wti, on='date', how='inner')

    # 6. เรียงลำดับตามวันที่ (เก่า -> ใหม่)
    df_merged = df_merged.sort_values(by='date').reset_index(drop=True)
    
    print(f"✅ Merge complete! Total matched records: {len(df_merged)}")
    return df_merged

def save_data(df):
    """
    ฟังก์ชันสำหรับบันทึกไฟล์ลงโฟลเดอร์ processed
    """
    # สร้างโฟลเดอร์ processed ถ้ายังไม่มี
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
        print(f"📁 Created folder: {PROCESSED_PATH}")

    # บันทึกเป็น CSV
    output_path = os.path.join(PROCESSED_PATH, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved processed data to: {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # เรียกใช้ฟังก์ชัน
        merged_df = load_and_clean_data()
        
        # (Optional) เช็กดูหน้าตาข้อมูล 5 บรรทัดแรก
        print("\n--- Preview Data ---")
        print(merged_df.head())
        
        save_data(merged_df)
        print("\n✨ Data preprocessing finished successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")