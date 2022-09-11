import pymysql
import pymysql.cursors
import dotenv
import os

from typing import Tuple, List

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)


class MysqlIO:
    def __init__(self):
        self.connection = pymysql.connect(
            host=os.environ.get("HOST"),
            user=os.environ.get("USER"),
            password=os.environ.get("PASSWORD"),
            database=os.environ.get("DATABASE"),
            cursorclass=pymysql.cursors.Cursor
        )

    def _commit(self, cursor: pymysql.cursors.Cursor):
        self.connection.commit()
        cursor.close()
        return self.connection.cursor()

    def insert(self, table: str, columns: List[str], value_list: List[Tuple]):
        cursor = self.connection.cursor()
        chunk_size = 10000
        sublist = [value_list[i:i + chunk_size] for i in range(0, len(value_list), chunk_size)]

        column_str = f"`{'`,`'.join(columns)}`"
        value_alt_char_list = ["%s"] * len(value_list[0])
        value_alt_chars = ",".join(value_alt_char_list)
        sql = f"Insert IGNORE INTO `{table}` ({column_str}) VALUES ({value_alt_chars})"

        for i, values in enumerate(sublist):
            cursor.executemany(sql, values)
            cursor = self._commit(cursor)
            print(f"Insert {i * chunk_size}/{len(value_list)} to {table}")

    def insert_one(self, table: str, columns: List[str], values: Tuple):
        cursor = self.connection.cursor()

        column_str = f"`{'`,`'.join(columns)}`"
        value_alt_char_list = ["%s"] * len(values)
        value_alt_chars = ",".join(value_alt_char_list)
        sql = f"Insert IGNORE INTO `{table}` ({column_str}) VALUES ({value_alt_chars})"
        cursor.execute(sql, values)
        insert_id = cursor.lastrowid
        self.connection.commit()
        cursor.close()
        return insert_id

    def update(self, table: str, columns: List[str], key: str, value_list: List[Tuple]):
        cursor = self.connection.cursor()
        chunk_size = 10000
        sublist = [value_list[i:i + chunk_size] for i in range(0, len(value_list), chunk_size)]

        column_str = ', '.join([f"`{x}`=%s" for x in columns])
        sql = f"Update `{table}` Set {column_str} where {key}=%s"

        for i, values in enumerate(sublist):
            cursor.executemany(sql, values)
            cursor = self._commit(cursor)
            print(f"Update {i * chunk_size}/{len(value_list)} to {table}")

    def select(self, table: str, key: str, key_value):
        sql = f"Select * From `{table}` where {key} = %s"
        cursor = self.connection.cursor()
        cursor.execute(sql, key_value)
        data = cursor.fetchall()
        cursor.close()

        return data

    def select_all(self, table: str, column="*"):
        if column != "*":
            sql = f"Select `{column}` From `{table}`"
        else:
            sql = f"Select * From `{table}`"
        cursor = self.connection.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        cursor.close()

        return data

    def execute(self, sql):
        cursor = self.connection.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        cursor.close()

        return data
