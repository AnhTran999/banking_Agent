from pathlib import Path
import json
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


# Xác định thư mục gốc của project
BASE_DIR = Path(__file__).resolve().parents[1]


# Trỏ tới database Chroma đã được data_loader.py lưu xuống ổ đĩa
CHROMA_DIR = BASE_DIR / "chroma_db"


# Dùng cùng embedding function với lúc tạo collection
embedding_function = ONNXMiniLM_L6_V2()


# Kết nối tới database Chroma đã tồn tại
client = chromadb.PersistentClient(path=str(CHROMA_DIR))


# Lấy collection bank_faq đã được tạo trước đó
collection = client.get_collection(
    name="bank_faq",
    embedding_function=embedding_function,
)


# Tìm các documents gần với câu hỏi truyền vào
results = collection.query(
    query_texts=["how to open a bank account"],
    n_results=3,
)


preaty_json = json.dumps(results, indent=4)
print(preaty_json)