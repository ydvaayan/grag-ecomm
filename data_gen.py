import csv, random, math, itertools
from pathlib import Path
random.seed(42)

OUT = Path(".")

# Category universe (Meesho-like, India-focused)
categories = {
    "Women Ethnic": ["Sarees", "Kurtis", "Dress Materials", "Leggings"],
    "Women Western": ["Tops", "Jeans", "Dresses", "Co-ords"],
    "Men": ["T-shirts", "Shirts", "Jeans", "Kurta"],
    "Kids": ["Frocks", "Sets", "T-Shirts", "Ethnic"],
    "Jewellery": ["Earrings", "Necklaces", "Bangles", "Rings"],
    "Home": ["Bedsheets", "Curtains", "Decor", "Organizer"],
    "Kitchen": ["Cookware", "Storage", "Tools", "Appliances"],
    "Beauty": ["Skincare", "Haircare", "Makeup", "Grooming"],
    "Footwear": ["Women Sandals", "Men Sneakers", "Flats", "Heels"],
    "Electronics": ["Earbuds", "Smartwatch", "Power Bank", "Cables"],
    "Bags & Luggage": ["Handbags", "Backpacks", "Duffle", "Wallets"],
    "Sports & Fitness": ["Dumbbells", "Yoga Mat", "Resistance Bands", "Bottles"]
}

# Brands and tags
brands = [
    "UrbanHide","SareeSaheli","DesiWeave","SteelPro","SonicBeat","Oak&Co",
    "GlowHerb","FitBolt","KidJoy","CraftKart","MaxStyle","NeoTrend","BharatHome",
    "ChefPro","AuraCare","TrailTech","DenimWorks","RoyalRoots","SparkLite","BeechCraft"
]

generic_tags = [
    "gift","budget","premium","eco","handmade","trending","bestseller","new",
    "festival","office","travel","party","dailywear","ethnic","western","men","women","kids",
    "home","kitchen","grooming","fitness","gadgets","accessories","decor","wedding","summer","winter"
]

# Sellers (15)
regions = ["IN-W", "IN-N", "IN-S", "IN-E", "IN-C", "IN-NE", "IN-MH", "IN-KA", "IN-DL", "IN-GJ"]
seller_names = [
    "CuppaCraft","LeatherWorks","GroomingHub","SoundKart","OakWood",
    "DesiCrafts","FashionVilla","UrbanBazaar","SareeMandir","GlowNest",
    "ChefBazaar","FitNation","KidsKart","TechPulse","HomeBliss"
]
sellers = []
for i, name in enumerate(seller_names, start=1):
    sellers.append({
        "seller_id": f"S{i:03d}",
        "name": name,
        "region": random.choice(regions),
        "rating": round(random.uniform(3.7, 4.9), 2)
    })

# Collections (25): seasonal, thematic, price bands, audience
collections_seed = [
    ("C001","Fathers Day Gifts","Curated gifts for dads across hobbies","Summer","Men"),
    ("C002","Mothers Day Gifts","Thoughtful gifts for moms","Summer","Women"),
    ("C003","Wedding Season Picks","Festive and ethnic bestsellers","All","All"),
    ("C004","Budget Under 499","Great value under 499","All","All"),
    ("C005","Budget 500-999","Value picks","All","All"),
    ("C006","Premium 2000+","Premium and designer picks","All","All"),
    ("C007","Back To School","Bags and kids essentials","Monsoon","Kids"),
    ("C008","Home Makeover","Bedsheets, curtains, decor","All","All"),
    ("C009","Kitchen Essentials","Cookware, storage, tools","All","All"),
    ("C010","Monsoon Must-Haves","Quick-dry, umbrellas, organizers","Monsoon","All"),
    ("C011","Winter Warmers","Thermals and cozy home","Winter","All"),
    ("C012","Pujo Ethnic Edit","Festive ethnic wear","Autumn","All"),
    ("C013","College Fits","Trendy fashion picks","All","Women"),
    ("C014","Office Staples","Formal shirts, handbags, organizers","All","All"),
    ("C015","Grooming For Men","Beard kits, perfumes","All","Men"),
    ("C016","Beauty Bestsellers","Skincare & makeup","All","Women"),
    ("C017","Fitness At Home","Mats, bands, bottles","All","All"),
    ("C018","Travel Ready","Backpacks, organizers","All","All"),
    ("C019","Wedding Gifts","Home & decor gifts","All","All"),
    ("C020","Jewellery Trends","Earrings, necklaces","All","Women"),
    ("C021","Electronics Deals","Earbuds, smartwatches","All","All"),
    ("C022","Kids Festive Wear","Ethnic sets & frocks","Festive","Kids"),
    ("C023","Handmade & Eco","Sustainable crafts","All","All"),
    ("C024","Party Night Out","Party dresses & heels","All","Women"),
    ("C025","Daily Essentials","Everyday best value","All","All"),
]
collections = [{"collection_id": cid, "name": nm, "description": desc, "season": ssn, "audience": aud}
               for cid, nm, desc, ssn, aud in collections_seed]

# Users (10)
user_names = ["Aarav","Vihaan","Aditya","Vivaan","Arjun","Sai","Aanya","Ananya","Diya","Isha"]
user_regions = ["IN-MH","IN-KA","IN-DL","IN-TN","IN-WB","IN-GJ","IN-TG","IN-RJ","IN-UP","IN-KL"]
all_cats = list(categories.keys())
budget_bands = ["<500","500-999","1000-1999","2000-4999","5000+"]

users = []
for i in range(10):
    users.append({
        "user_id": f"U{i+1:03d}",
        "name": user_names[i],
        "region": user_regions[i],
        "preferred_categories": "|".join(sorted(random.sample(all_cats, k=3))),
        "budget_band": random.choice(budget_bands)
    })

# Helper: title/description templates
adjectives = ["Classic","Trendy","Premium","Elegant","Stylish","Comfy","Durable","Handcrafted","Eco","Lightweight","Vibrant","Minimal"]
materials = ["Cotton","Chiffon","Silk","Denim","Leather","Wood","Stainless Steel","Alloy","Bamboo","Polycarbonate","Canvas","Terracotta"]
colors = ["Black","Blue","Red","Green","Pink","Purple","Beige","Brown","Grey","Navy","Maroon","Teal","White","Yellow","Orange"]
features = ["RFID Blocking","Quick-Dry","Nonstick","Leak-Proof","Noise Cancellation","Long Battery","Anti-Theft","Organizer Pockets","Skin-Friendly","Stretchable","Anti-Slip"]

def make_title(cat, sub, brand):
    return f"{random.choice(adjectives)} {random.choice(colors)} {sub} by {brand}"

def make_desc(cat, sub):
    fs = ", ".join(random.sample(features, k=2))
    mat = random.choice(materials)
    return f"{sub} in {mat} with {fs}. Ideal for {cat.lower()} use."

def make_price(cat):
    # category-based typical price ranges
    base = {
        "Women Ethnic": (399, 2999),
        "Women Western": (299, 2499),
        "Men": (299, 2499),
        "Kids": (199, 1499),
        "Jewellery": (149, 1999),
        "Home": (299, 2999),
        "Kitchen": (199, 2499),
        "Beauty": (149, 1499),
        "Footwear": (299, 2499),
        "Electronics": (499, 4999),
        "Bags & Luggage": (299, 2999),
        "Sports & Fitness": (199, 2499),
    }[cat]
    low, high = base
    # skew towards affordable with a log-like distribution
    r = random.random()
    price = int(low + (high - low) * (r**2))
    return max(low, min(price, high))

def pick_tags(cat, sub):
    pool = set(generic_tags)
    if cat in ["Women Ethnic","Women Western","Jewellery","Footwear"]:
        pool.update(["women","fashion","style","ethnic","western","party","wedding"])
    if cat in ["Men"]:
        pool.update(["men","grooming","office","casual"])
    if cat in ["Kids"]:
        pool.update(["kids","school","festival"])
    if cat in ["Home","Kitchen"]:
        pool.update(["home","kitchen","organizer","decor"])
    if cat in ["Electronics"]:
        pool.update(["gadgets","tech","charging","audio"])
    if cat in ["Sports & Fitness"]:
        pool.update(["fitness","yoga","gym"])
    tags = random.sample(sorted(pool), k=min(6, len(pool)))
    return ";".join(tags)

def popularity_from_price(price):
    # cheaper tends to be more popular; normalize to [0.2, 0.98]
    score = 1.0 / (1.0 + math.log1p(price))
    return round(0.2 + 0.78 * score, 3)

# Build products (1000)
products = []
pid = 1
cat_cycle = []
for cat, subs in categories.items():
    cat_cycle.extend([cat] * (1000 // len(categories)))
# fill remainder
while len(cat_cycle) < 1000:
    cat_cycle.append(random.choice(list(categories.keys())))
random.shuffle(cat_cycle)

for cat in cat_cycle[:1000]:
    sub = random.choice(categories[cat])
    brand = random.choice(brands)
    seller = random.choice(sellers)["seller_id"]
    title = make_title(cat, sub, brand)
    desc = make_desc(cat, sub)
    price = make_price(cat)
    rating = round(random.uniform(3.5, 4.9) + random.random()*0.1, 2)
    rating = min(rating, 5.0)
    popularity = popularity_from_price(price)
    in_stock = random.random() > 0.06  # ~6% OOS
    tag_str = pick_tags(cat, sub)

    products.append({
        "id": f"I{pid:04d}",
        "title": title,
        "description": desc,
        "price": price,
        "brand": brand,
        "category": cat,
        "subcategory": sub,
        "tags": tag_str,
        "seller_id": seller,
        "rating": rating,
        "popularity": popularity,
        "in_stock": "true" if in_stock else "false"
    })
    pid += 1

# Collection assignment rules:
# - Map by price bands, seasonal, thematic, and audience signals inferred from tags/category.
def assign_collections(p):
    cols = []
    price = p["price"]
    cat = p["category"]
    tags = p["tags"].split(";")

    # price bands
    if price < 500: cols.append("C004")
    if 500 <= price <= 999: cols.append("C005")
    if price >= 2000: cols.append("C006")

    # seasonal/thematic
    if "wedding" in tags or cat in ["Women Ethnic","Jewellery"]:
        cols.append("C003"); cols.append("C019"); cols.append("C012")
    if "home" in tags or cat=="Home":
        cols.append("C008"); cols.append("C025")
    if "kitchen" in tags or cat=="Kitchen":
        cols.append("C009"); cols.append("C025")
    if "gadgets" in tags or cat=="Electronics":
        cols.append("C021")
    if "fitness" in tags or cat=="Sports & Fitness":
        cols.append("C017")
    if "kids" in tags or cat=="Kids":
        cols.append("C007"); cols.append("C022")
    if "men" in tags or cat=="Men":
        cols.append("C015")
    if "women" in tags or cat in ["Women Western","Women Ethnic","Jewellery","Footwear"]:
        cols.append("C013"); cols.append("C020"); cols.append("C024")

    # eco/handmade
    if "eco" in tags or "handmade" in tags:
        cols.append("C023")

    # father/mother gift heuristics via tags
    if "gift" in tags and ("men" in tags or cat=="Men" or "gadgets" in tags or "wallet" in p["title"].lower()):
        cols.append("C001")
    if "gift" in tags and ("women" in tags or "beauty" in tags or cat in ["Women Western","Women Ethnic","Jewellery"]):
        cols.append("C002")

    # travel / office themes
    if "travel" in tags:
        cols.append("C018")
    if "office" in tags:
        cols.append("C014")

    # keep unique and limit 1–3 collections per item
    uniq = list(dict.fromkeys(cols))
    random.shuffle(uniq)
    return uniq[:max(1, min(3, len(uniq)))]

collection_items = []
for p in products:
    for cid in assign_collections(p):
        collection_items.append({"collection_id": cid, "item_id": p["id"]})

# Write CSVs
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

write_csv(OUT/"products.csv",
          ["id","title","description","price","brand","category","subcategory","tags","seller_id","rating","popularity","in_stock"],
          products)

write_csv(OUT/"sellers.csv",
          ["seller_id","name","region","rating"],
          sellers)

write_csv(OUT/"collections.csv",
          ["collection_id","name","description","season","audience"],
          collections)

write_csv(OUT/"collection_items.csv",
          ["collection_id","item_id"],
          collection_items)

write_csv(OUT/"users.csv",
          ["user_id","name","region","preferred_categories","budget_band"],
          users)

print("Generated: products.csv (1000), sellers.csv (15), collections.csv (25), collection_items.csv (~{}), users.csv (10)".format(len(collection_items)))
