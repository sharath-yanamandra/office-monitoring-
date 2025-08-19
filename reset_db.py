import mysql.connector

def reset_db():
    try:
        conn = mysql.connector.connect(
            host='34.93.87.255',
            user='insighteye', 
            password='insighteye0411',
            port=3306
        )
        cursor = conn.cursor()
        cursor.execute("DROP DATABASE IF EXISTS dc_test")
        cursor.execute("CREATE DATABASE dc_test")
        print("✅ Database reset complete!")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    reset_db()
