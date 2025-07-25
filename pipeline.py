from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from catboost import CatBoostClassifier
import pandas as pd

sentenceModel = SentenceTransformer("sentence-transformers/LaBSE")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
items = pd.read_parquet("items.parquet", engine="fastparquet") 

items_dict = {}

for item_id, title, content in zip(items["itemId"], items["title"], items["content"]):
    items_dict[item_id] = f"{title} {content}"
    
train["leftItemText"] = train["leftItemId"].map(items_dict)
train["rightItemText"] = train["rightItemId"].map(items_dict)

left_embeddings = sentenceModel.encode(train["leftItemText"].tolist(), show_progress_bar=True)
right_embeddings = sentenceModel.encode(train["rightItemText"].tolist(), show_progress_bar=True)

cos_sim = [cosine_similarity(left_emb.reshape(1,-1), right_emb.reshape(1,-1))[0,0]
           for left_emb, right_emb in zip(left_embeddings, right_embeddings)]

train["cos_sim"] = cos_sim
train = train.drop(["leftItemText", "rightItemText", "leftItemId", "rightItemId"], axis=1)

test["leftItemText"] = test["leftItemId"].map(items_dict)
test["rightItemText"] = test["rightItemId"].map(items_dict)

left_embeddings = sentenceModel.encode(test["leftItemText"].tolist(), show_progress_bar=True)
right_embeddings = sentenceModel.encode(test["rightItemText"].tolist(), show_progress_bar=True)

cos_sim = [cosine_similarity(left_emb.reshape(1,-1), right_emb.reshape(1,-1))[0,0]
           for left_emb, right_emb in zip(left_embeddings, right_embeddings)]

test["cos_sim"] = cos_sim
test = test.drop(["leftItemText", "rightItemText", "leftItemId", "rightItemId"], axis=1)

X_train = train["cos_sim"]
y_train = train["target"]

model = CatBoostClassifier(
    verbose=100,
    random_seed=42
)

model.fit(X_train, y_train)
preds = model.predict(test)

submission = pd.DataFrame({"target": preds})
submission.to_csv("submission.csv", index=True)
print("Done!")



