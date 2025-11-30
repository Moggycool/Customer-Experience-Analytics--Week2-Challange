"""
Database connection helper.
Use get_connection() to connect to PostgreSQL.
"""

import psycopg2
from psycopg2 import pool

# -------------------------------------------
# PostgreSQL Connection Pool (recommended)
# -------------------------------------------
connection_pool = pool.SimpleConnectionPool(
    1,              # min connections
    10,             # max connections
    dbname="bank_reviews",
    user="moggy",
    password="MoGGy8080",
    host="localhost",
    port=5432
)


def get_connection():
    """Return a new database connection from the pool."""
    return connection_pool.getconn()


def release_connection(conn):
    """Release a connection back to the pool."""
    connection_pool.putconn(conn)


# Test only when running this file directly, not when importing
if __name__ == "__main__":
    conn = None
    try:
        conn = get_connection()
        print("Connected successfully!")
    except Exception as e:
        print("Connection failed:", e)
    finally:
        if conn:
            release_connection(conn)
