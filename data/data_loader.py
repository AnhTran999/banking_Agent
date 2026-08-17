import pandas as pd 
from config import Settings
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from pathlib import Path


settings = Settings()


FAQ_path = settings.FAQ_file
df_qa = pd.read_csv(FAQ_path)
df_qa["combie_text"] = (
    "Question: " + df_qa["Question"].astype(str) + ". " +
    "Answer: " + df_qa["Answer"].astype(str) + ". " +
    "Class: " + df_qa["Class"].astype(str) + ". " 
    ) 

bank_branch_path = settings.bank_branch_file
df_bank_branch = pd.read_csv(bank_branch_path)


# Dùng embedding local ONNX, không gọi Gemini API ở bước này
ef = ONNXMiniLM_L6_V2()


# Xác định thư mục gốc của project
BASE_DIR = Path(__file__).resolve().parents[1]


# Lưu database Chroma xuống ổ đĩa thay vì chỉ giữ trong RAM
CHROMA_DIR = BASE_DIR / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_DIR))


# Lấy hoặc tạo collection có tên bank_faq
collection = client.get_or_create_collection(
    name="bank_faq",
    embedding_function=ef,
)


# Thêm documents, metadata và id vào collection
collection.add(
    documents=df_qa["combie_text"].tolist(),
    metadatas=df_qa.to_dict(orient="records"),
    ids=df_qa.index.astype(str).tolist(),
)
