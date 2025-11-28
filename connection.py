import psycopg2

conn = psycopg2.connect(
    dbname="bank_reviews",
    user="Moggy",
    password="MoGGy8080",
    host="localhost",
    port=5432
)

print("Connected successfully!")

conn.close()
