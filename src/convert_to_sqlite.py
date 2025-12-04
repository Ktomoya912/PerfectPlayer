import sqlite3
from pathlib import Path

from utils import readDB2

base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "perfect.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

data = readDB2()

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS board_data (
board_index INTEGER PRIMARY KEY,
evaluation_score REAL);
"""

cursor.execute(CREATE_TABLE_SQL)
conn.commit()

for board_index, evaluation_score in data.items():
    cursor.execute(
        "INSERT OR REPLACE INTO board_data (board_index, evaluation_score) VALUES (?, ?)",
        (board_index, evaluation_score),
    )
conn.commit()
# 最初の10行を表示して確認
cursor.execute("SELECT * FROM board_data LIMIT 10")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
print("Data has been successfully converted and stored in SQLite database.")
