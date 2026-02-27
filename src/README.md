# ⚙️ Source Code (Production Pipeline)

This folder contains the **finalized, executable scripts** for the project. These scripts automate the processes developed in the `notebooks/` folder.

## 🚀 Key Scripts
* **`data_preprocessing.py`**:
    * Loads raw data from `data/raw/`.
    * Performs merging, cleaning, and date alignment.
    * Saves the result to `data/processed/`.

> ⚠️ **Important:** Do not modify these files unless you are finalizing the system logic. These are the core scripts for the project workflow.

---

# 🇹🇭 โค้ดระบบหลัก (Final Production)

โฟลเดอร์นี้เก็บโค้ดฉบับสมบูรณ์ (The Real Deal) ที่ผ่านการคัดกรองมาจาก Notebooks แล้ว

## 🚀 ไฟล์สำคัญ
* **`data_preprocessing.py`**: สคริปต์สำหรับ "เตรียมข้อมูล" (รวมไฟล์, คลีนค่าว่าง) เพื่อให้เพื่อนทุกคนได้ไฟล์ข้อมูลมาตรฐานเดียวกัน

> ⚠️ **คำเตือน:** **"ไม่ต้องไปยุ่งหรือแก้ไข"** นอกเสียจากว่าต้องการปรับแก้ระบบหลักของโปรเจกต์ เพราะไฟล์นี้คือขั้นตอนสุดท้ายของการทำงาน (Pipeline)