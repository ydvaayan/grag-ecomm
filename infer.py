import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.prompts import ChatPromptTemplate

# Load env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")          # neo4j+s://<id>.databases.neo4j.io
NEO4J_USER = os.getenv("NEO4J_USERNAME")    # Aura DB username (not Google SSO)
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")    # Aura DB password
print(f"Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}")
# Connect (Aura requires TLS scheme neo4j+s)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))  # [Aura TLS]
driver.verify_connectivity()

# Embedder
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Schema (Aura supports vector indexes + constraints)
schema = """
CREATE CONSTRAINT item_id IF NOT EXISTS FOR (i:Item) REQUIRE i.item_id IS UNIQUE;
CREATE CONSTRAINT seller_id IF NOT EXISTS FOR (s:Seller) REQUIRE s.seller_id IS UNIQUE;
CREATE CONSTRAINT cat_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE;
CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT coll_id IF NOT EXISTS FOR (c:Collection) REQUIRE c.collection_id IS UNIQUE;
CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;

CREATE VECTOR INDEX item_desc_embed IF NOT EXISTS
FOR (i:Item) ON (i.desc_vec)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}};
"""

with driver.session() as sess:
    for stmt in [s.strip() for s in schema.split(";") if s.strip()]:
        sess.run(stmt)

# Load CSVs
products = pd.read_csv("products.csv")
sellers = pd.read_csv("sellers.csv")
collections = pd.read_csv("collections.csv")
collection_items = pd.read_csv("collection_items.csv")
user_events = pd.read_csv("users.csv")

# LLM (your local/vLLM endpoint stays the same)

llm = ChatOpenAI(
    model="Qwen2.5-1.5B-Instruct",         # or your local model id
    openai_api_key="EMPTY",              # vLLM can ignore/validate as configured
    openai_api_base="https://8081-01k5hptbeey4vhfxjxcg6rbp2g.cloudspaces.litng.ai/v1",
    temperature=0,
)

def upsert_products(df: pd.DataFrame):
    with driver.session() as sess:
        for _, r in df.iterrows():
            doc = f"{r.title}. {r.description}. Category: {r.category}. Brand: {r.brand}. Tags: {r.tags}"
            vec = embedder.encode([doc])[0].astype(np.float32).tolist()
            cypher = """
            MERGE (i:Item {item_id:$id})
              SET i.title=$title, i.description=$description, i.price=$price,
                  i.rating=$rating, i.popularity=$popularity, i.desc_vec=$vec
            MERGE (b:Brand {name:$brand})
            MERGE (c:Category {name:$category})
            MERGE (s:Seller {seller_id:$seller_id})
            MERGE (i)-[:OF_BRAND]->(b)
            MERGE (i)-[:IN_CATEGORY]->(c)
            MERGE (i)-[:SOLD_BY]->(s)
            WITH i
            UNWIND $tags AS tg
              MERGE (t:Tag {name:tg})
              MERGE (i)-[:HAS_TAG]->(t)
            """
            sess.run(cypher, parameters={
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "price": int(r.price),
                "rating": float(r.rating),
                "popularity": float(r.popularity),
                "brand": r.brand,
                "category": r.category,
                "seller_id": r.seller_id,
                "tags": [t.strip() for t in str(r.tags).split(";") if t.strip()],
                "vec": vec,
            })

def upsert_sellers(df: pd.DataFrame):
    with driver.session() as sess:
        for _, r in df.iterrows():
            sess.run("""
            MERGE (s:Seller {seller_id:$seller_id})
              SET s.name=$name, s.region=$region
            """, parameters={"seller_id": r.seller_id, "name": r.name, "region": r.region})

def upsert_collections(cdf: pd.DataFrame, cidf: pd.DataFrame):
    with driver.session() as sess:
        for _, r in cdf.iterrows():
            sess.run("""
            MERGE (c:Collection {collection_id:$cid})
              SET c.name=$name, c.description=$desc
            """, parameters={"cid": r.collection_id, "name": r.name, "desc": r.description})
        for _, r in cidf.iterrows():
            sess.run("""
            MATCH (c:Collection {collection_id:$cid}), (i:Item {item_id:$iid})
            MERGE (c)-[:HAS_ITEM]->(i)
            """, parameters={"cid": r.collection_id, "iid": r.item_id})

def upsert_user_events(df: pd.DataFrame):
    # APOC Core is supported in Aura; apoc.create.relationship is allowed.
    with driver.session() as sess:
        for uid, grp in df.groupby("user_id"):
            sess.run("MERGE (u:User {user_id:$u})", parameters={"u": uid})
            for _, r in grp.iterrows():
                sess.run("""
                MATCH (u:User {user_id:$u}), (i:Item {item_id:$iid})
                CALL apoc.create.relationship(u, $rt, {ts:$ts}, i) YIELD rel
                RETURN rel
                """, parameters={"u": r.user_id, "iid": r.item_id, "rt": r.type, "ts": r.ts})

def create_similarity(k=3):
    # Aura-friendly: use Cypher vector function to compute pairwise cosine on stored embeddings.
    # For large catalogs, do this offline or with batching to avoid O(N^2) cost.
    with driver.session() as sess:
        sess.run("""
        // Compute top-k neighbors per item using vector.similarity.cosine
        MATCH (i:Item)
        WITH collect(i) AS items
        UNWIND items AS i1
        WITH i1, [i IN items WHERE i<>i1] AS others
        UNWIND others AS i2
        WITH i1, i2, vector.similarity.cosine(i1.desc_vec, i2.desc_vec) AS sim
        ORDER BY i1.item_id, sim DESC
        WITH i1, collect({n:i2, s:sim})[0..$k] AS nbrs
        UNWIND nbrs AS nb
        MERGE (i1)-[r:SIMILAR_TO]->(nb.n)
        SET r.score = nb.s
        """, parameters={"k": k})

# Ingest
upsert_sellers(sellers)
upsert_products(products)
upsert_collections(collections, collection_items)
upsert_user_events(user_events)
create_similarity(k=3)

# Retrieval utilities (Aura connection reused)
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)
vector = Neo4jVector(
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
    node_label="Item", text_node_property="description", embedding_node_property="desc_vec"
)

route_prompt = ChatPromptTemplate.from_template(
    "Classify intent for recommendation: '{q}'. Reply exactly 'GRAPH' if constraints/categories/tags/collections dominate; else 'VECTOR'."
)

def route(q: str) -> str:
    resp = llm.invoke(route_prompt.format_messages(q=q))
    t = resp.content.strip().upper()
    return "GRAPH" if "GRAPH" in t else "VECTOR"

def graph_candidates(user_id: str, q: str, limit: int = 20):
    cypher = """
    OPTIONAL MATCH (col:Collection)-[:HAS_ITEM]->(i:Item)
    WHERE toLower(col.name) CONTAINS 'father'
    WITH collect(i) AS col_items
    OPTIONAL MATCH (i2:Item)-[:HAS_TAG]->(t:Tag)
    WHERE toLower(t.name) IN ['gift','father','men']
    WITH col_items + collect(i2) AS cands
    UNWIND cands AS c
    WITH DISTINCT c
    OPTIONAL MATCH (:User {user_id:$uid})-[:VIEWED|PURCHASED]->(h:Item)
    OPTIONAL MATCH (h)-[:HAS_TAG]->(ht)<-[:HAS_TAG]-(c)
    OPTIONAL MATCH (h)-[:IN_CATEGORY]->(hc)<-[:IN_CATEGORY]-(c)
    WITH c, count(DISTINCT ht)+count(DISTINCT hc) AS overlap
    WITH c, (0.6*overlap + 0.25*coalesce(c.popularity,0) + 0.15*coalesce(c.rating,0)) AS score
    RETURN c.item_id AS item_id, c.title AS title, c.description AS description, c.price AS price, score
    ORDER BY score DESC LIMIT $limit
    """
    with driver.session() as sess:
        return sess.run(cypher, parameters={"uid": user_id, "limit": limit}).data()

def vector_candidates(q: str, limit: int = 20):
    docs = vector.similarity_search_with_score(q, k=limit)
    out = []
    for d, s in docs:
        out.append({
            "item_id": d.metadata.get("item_id"),
            "title": d.metadata.get("title") or d.page_content[:60],
            "description": d.page_content,
            "price": d.metadata.get("price"),
            "score": float(s)
        })
    return out

def explain_paths(user_id: str, item_id: str, k: int = 2):
    cypher = """
    MATCH (u:User {user_id:$u}), (i:Item {item_id:$i})
    OPTIONAL MATCH p1=(u)-[:VIEWED|PURCHASED]->(:Item)-[:SIMILAR_TO]->(i)
    OPTIONAL MATCH p2=(u)-[:VIEWED|PURCHASED]->(:Item)-[:HAS_TAG]->(:Tag)<-[:HAS_TAG]-(i)
    OPTIONAL MATCH p3=(:Collection)-[:HAS_ITEM]->(i)
    WITH [p IN [p1,p2,p3] WHERE p IS NOT NULL][..$k] AS paths
    RETURN paths
    """
    with driver.session() as sess:
        res = sess.run(cypher, parameters={"u": user_id, "i": item_id, "k": k}).data()
    return res[0]["paths"] if res else []

def recommend_for_query(user_id: str, query: str, top_k: int = 5):
    mode = route(query)
    cands = graph_candidates(user_id, query, 30) if mode == "GRAPH" else vector_candidates(query, 30)
    sel_prompt = ChatPromptTemplate.from_template(
        "Intent: {q}\nCandidates: {c}\nPick top {k} items covering diverse categories and budgets; "
        "prefer tags gift/father/men; output JSON list with item_id and a one-sentence reason."
    )
    selection = llm.invoke(sel_prompt.format_messages(q=query, c=str(cands), k=top_k)).content
    import json
    try:
        picks = json.loads(selection)
    except Exception:
        picks = []
    enriched = []
    for p in picks[:top_k]:
        paths = explain_paths(user_id, p.get("item_id"))
        enriched.append({**p, "paths": str(paths)[:400]})
    final_prompt = ChatPromptTemplate.from_template(
        "Intent: {q}\nItems: {items}\nWrite a concise recommendation list (bullet points): "
        "name, price, reason; include a short explanation when paths indicate a match with tags/collection."
    )
    return llm.invoke(final_prompt.format_messages(q=query, items=str(enriched))).content

print(recommend_for_query("U100", "I want to buy gift for fathers day", top_k=5))
