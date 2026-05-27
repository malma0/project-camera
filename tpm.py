import sqlite3
c = sqlite3.connect('data/stats.db')
print(c.execute("SELECT sql FROM sqlite_master WHERE name='sessions'").fetchone()[0])
print()
print("Примеры записей:")
for row in c.execute("SELECT * FROM sessions LIMIT 3").fetchall():
    print(row)