from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import os

print("🔵 Loading Arabic + English datasets...")

df_fake = pd.read_csv("data/fake/Fake.csv", encoding="ISO-8859-1")
df_real = pd.read_csv("data/real/True.csv", encoding="ISO-8859-1")

df_fake["label"] = 0
df_real["label"] = 1

df = pd.concat([df_fake, df_real], ignore_index=True)

print("📊 Total samples:", len(df))

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"🔧 Loading embedding model: {model_name}")

embedder = SentenceTransformer(model_name)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import pickle

print("Loading Arabic + English dataset...")

df_fake = pd.read_csv("data_arabic/fake_arabic.csv", encoding="utf-8")
df_real = pd.read_csv("data_arabic/real_arabic.csv", encoding="utf-8")

df_fake["label"] = 0
df_real["label"] = 1

df = pd.concat([df_fake, df_real], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

X = df["text"].astype(str).tolist()
y = df["label"].tolist()

print("Loading SMALL multilingual model (~50MB)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L3-v2")

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import os

print("🔵 Loading Arabic + English datasets...")

df_fake = pd.read_csv("data/fake/Fake.csv", encoding="ISO-8859-1")
df_real = pd.read_csv("data/real/True.csv", encoding="ISO-8859-1")

df_fake["label"] = 0
df_real["label"] = 1

df = pd.concat([df_fake, df_real], ignore_index=True)

print("📊 Total samples:", len(df))

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"🔧 Loading embedding model: {model_name}")

embedder = SentenceTransformer(model_name)

print("🧠 Generating embeddings (this may take 2–5 minutes)...")
X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("🤖 Training classifier...")
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

print("📈 Evaluating model...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("💾 Saving model...")
pickle.dump(clf, open("classifier.pkl", "wb"))
pickle.dump(embedder, open("embedder.pkl", "wb"))

print("✅ DONE! Model saved as classifier.pkl + embedder.pkl")
