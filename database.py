from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

class Database:
    def __init__(self):
        self.CONNECTION_STRING = "mongodb+srv://hyagobora:mongodb@cluster0.lgmoa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        self.database_name = "vehicles_data_db"
        self.collection_name = "vehicles_data_collections"
        self.connect = MongoClient(
            self.CONNECTION_STRING, server_api=ServerApi("1")
        ).get_database(self.database_name)[self.collection_name]

    def insert_result(self, json_data):
        try:
            return self.connect.insert_one(json_data)
        except Exception as e:
            print(f"Erro ao inserir no MongoDB: {e}")

