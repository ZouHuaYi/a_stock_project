from src.utils.db_manager import DatabaseManager

# 初始化数据库
if __name__ == "__main__":
    db_manager = DatabaseManager()
    db_manager.drop_tables()
    db_manager.create_tables()
    db_manager.close()

