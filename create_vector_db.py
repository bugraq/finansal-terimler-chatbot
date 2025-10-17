import os
import pandas as pd
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")

csv_file_path = "financial_terms_final.csv"
try:
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    df.dropna(subset=['Terim', 'Tanım'], inplace=True)
    df['Tanım'] = df['Tanım'].astype(str)

    texts = df['Tanım'].tolist()
    metadatas = [{'terim': terim} for terim in df['Terim']]
    print(f"{len(texts)} adet temizlenmiş tanım başarıyla yüklendi.")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
    exit()

print("Google Embedding Modeli başlatılıyor...")
# DÜZELTME: Daha yeni bir embedding modeli kullanıyoruz.
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

print("FAISS veritabanı oluşturuluyor...")

chunk_size = 50
text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
metadata_chunks = [metadatas[i:i + chunk_size] for i in range(0, len(metadatas), chunk_size)]

print(f"İlk parça ({len(text_chunks[0])} adet tanım) işleniyor...")
vector_store = FAISS.from_texts(
    texts=text_chunks[0], 
    embedding=embeddings, 
    metadatas=metadata_chunks[0]
)

for i in range(1, len(text_chunks)):
    print(f"Parça {i+1}/{len(text_chunks)} ({len(text_chunks[i])} adet tanım) işleniyor...")
    time.sleep(5) 
    vector_store.add_texts(
        texts=text_chunks[i],
        metadatas=metadata_chunks[i]
    )

db_name = "faiss_index_financial_terms"
vector_store.save_local(db_name)

print(f"\n✅ BAŞARILI! Vektör veritabanı oluşturuldu ve '{db_name}' adıyla kaydedildi.")