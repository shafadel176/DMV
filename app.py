"""
Streamlit App - Book Recommendation System (Book-Crossing dataset)

Tugas DMV Week 10 - Recommendation System
Nama  : Shafadel Lestariyan Lainy
NRP   : 5003241060
Kelas : Data Mining and Visualization K

Jalankan dengan:
    streamlit run app.py
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
)

CSV_KWARGS = dict(sep=";", encoding="latin-1", escapechar="\\")
FILE_NAMES = {
    "ratings": "BX-Book-Ratings.csv",
    "books": "BX-Books.csv",
    "users": "BX-Users.csv",
}


# =====================================================================
# A. DATA LOADING, CLEANING & PREPROCESSING
# =====================================================================
def _src(x):
    """Upload (bytes) -> BytesIO, path (str) -> str."""
    return io.BytesIO(x) if isinstance(x, (bytes, bytearray)) else x


@st.cache_resource(show_spinner="Membaca & membersihkan dataset...")
def load_and_clean(ratings_src, books_src, users_src):
    ratings = pd.read_csv(_src(ratings_src), dtype={"ISBN": str}, **CSV_KWARGS)
    books = pd.read_csv(
        _src(books_src),
        usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"],
        dtype={"ISBN": str},
        low_memory=False,
        **CSV_KWARGS,
    )
    users = pd.read_csv(_src(users_src), **CSV_KWARGS)

    stats = {
        "shape_ratings": ratings.shape,
        "shape_books": books.shape,
        "shape_users": users.shape,
        "missing_ratings": ratings.isnull().sum(),
        "missing_books": books.isnull().sum(),
        "missing_users": users.isnull().sum(),
        "dup_ratings": int(ratings.duplicated().sum()),
        "dup_books": int(books.duplicated().sum()),
        "dup_users": int(users.duplicated().sum()),
    }

    # Menangani missing value pada books
    books["Book-Author"] = books["Book-Author"].fillna("Unknown")
    books["Publisher"] = books["Publisher"].fillna("Unknown")

    # Menggabungkan dataset
    data = ratings.merge(books, on="ISBN", how="left").merge(users, on="User-ID", how="left")
    stats["shape_merged"] = data.shape
    stats["missing_merged"] = data.isnull().sum()

    # Menghapus baris tanpa informasi buku
    data_clean = data.dropna(subset=["Book-Title"]).reset_index(drop=True)
    stats["shape_clean"] = data_clean.shape
    stats["rating_dist"] = data_clean["Book-Rating"].value_counts().sort_index()

    # Rating eksplisit (> 0)
    data_explicit = data_clean[data_clean["Book-Rating"] > 0].copy()
    stats["shape_explicit"] = data_explicit.shape

    return {"data_clean": data_clean, "data_explicit": data_explicit, "stats": stats}


# =====================================================================
# MODEL BUILDING (CF, Content-Based, Train/Test)
# =====================================================================
def make_bundle(df):
    """Bangun user-item matrix (sparse) + item similarity dari sebuah DataFrame rating."""
    agg = df.groupby(["User-ID", "ISBN"], as_index=False)["Book-Rating"].mean()
    users = pd.Index(np.sort(agg["User-ID"].unique()))
    isbns = pd.Index(np.sort(agg["ISBN"].unique()))
    rows = users.get_indexer(agg["User-ID"])
    cols = isbns.get_indexer(agg["ISBN"])
    mat = csr_matrix(
        (agg["Book-Rating"].to_numpy(dtype=float), (rows, cols)),
        shape=(len(users), len(isbns)),
    )
    return {
        "df": df,
        "users": users,
        "isbns": isbns,
        "mat": mat,
        "item_sim": cosine_similarity(mat.T),
    }


@st.cache_resource(show_spinner="Membangun model rekomendasi...")
def build_models(ratings_src, books_src, users_src, top_n_books, test_size, seed):
    data_explicit = load_and_clean(ratings_src, books_src, users_src)["data_explicit"]

    # Membatasi ke N buku dengan rating terbanyak
    top_isbn = data_explicit["ISBN"].value_counts().head(top_n_books).index
    data_cf = data_explicit[data_explicit["ISBN"].isin(top_isbn)].copy()

    # Content-based: TF-IDF judul + penulis
    books_content = (
        data_cf[["ISBN", "Book-Title", "Book-Author"]]
        .drop_duplicates(subset="ISBN")
        .reset_index(drop=True)
    )
    books_content["Book-Author"] = books_content["Book-Author"].fillna("Unknown")
    books_content["features"] = books_content["Book-Title"] + " " + books_content["Book-Author"]
    tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(books_content["features"])
    content_sim = cosine_similarity(tfidf_matrix)
    isbn_index = dict(zip(books_content["ISBN"], range(len(books_content))))

    # Train / test split
    train_data, test_data = train_test_split(data_cf, test_size=test_size, random_state=seed)
    test_data = test_data[test_data["Book-Rating"] >= 7]
    relevant = test_data.groupby("User-ID")["ISBN"].apply(set).to_dict()

    title_map = dict(zip(data_cf["ISBN"], data_cf["Book-Title"]))

    return {
        "data_cf": data_cf,
        "full": make_bundle(data_cf),
        "train": make_bundle(train_data),
        "train_data": train_data,
        "test_data": test_data,
        "relevant": relevant,
        "books_content": books_content,
        "content_sim": content_sim,
        "isbn_index": isbn_index,
        "title_map": title_map,
    }


# =====================================================================
# C. POPULARITY-BASED
# =====================================================================
def popularity_table(df):
    pop = (
        df.groupby(["ISBN", "Book-Title"])
        .agg(rating_count=("Book-Rating", "count"), average_rating=("Book-Rating", "mean"))
        .reset_index()
    )
    return pop.sort_values("rating_count", ascending=False)


# =====================================================================
# D. COLLABORATIVE FILTERING
# =====================================================================
def _user_row(b, user_id):
    i = b["users"].get_loc(user_id)
    row = b["mat"][i]
    return i, row.indices, row.data


def recommend_user_based(b, user_id, n=10, k_neighbors=10):
    i, idx, _ = _user_row(b, user_id)
    sims = cosine_similarity(b["mat"][i], b["mat"]).ravel()
    sims[i] = -np.inf  # jangan pilih diri sendiri
    neighbors = np.argsort(sims)[::-1][:k_neighbors]
    neighbor_ids = b["users"][neighbors]
    user_books = b["isbns"][idx]

    df = b["df"]
    rec = (
        df[df["User-ID"].isin(neighbor_ids) & ~df["ISBN"].isin(user_books)]
        .groupby(["ISBN", "Book-Title"])
        .agg(rating_count=("Book-Rating", "count"), average_rating=("Book-Rating", "mean"))
        .reset_index()
        .sort_values("average_rating", ascending=False)
    )
    return rec.head(n)


def recommend_item_based(b, user_id, title_map, n=10, k_similar=10):
    _, idx, ratings = _user_row(b, user_id)
    rated = set(idx.tolist())
    scores = {}
    for j, rating in zip(idx, ratings):
        sims = b["item_sim"][j].copy()
        sims[j] = -np.inf
        for t in np.argsort(sims)[::-1][:k_similar]:
            if t not in rated:
                scores[t] = max(scores.get(t, 0), sims[t] * rating)

    top = sorted(scores, key=scores.get, reverse=True)[:n]
    isbn_list = [b["isbns"][t] for t in top]
    return pd.DataFrame(
        {
            "ISBN": isbn_list,
            "Book-Title": [title_map.get(i, "") for i in isbn_list],
            "score": [scores[t] for t in top],
        }
    )


# =====================================================================
# E. CONTENT-BASED
# =====================================================================
def recommend_content_based(M, isbn, n=10):
    bc = M["books_content"]
    pos = M["isbn_index"][isbn]
    similar = M["content_sim"][pos].argsort()[::-1][1 : n + 1]
    rec = bc.iloc[similar][["ISBN", "Book-Title", "Book-Author"]].copy()
    rec["score"] = M["content_sim"][pos, similar]
    return rec


def recommend_content_user(M, b, user_id, n=10, min_rating=7):
    df = b["df"]
    user_books = df[(df["User-ID"] == user_id) & (df["Book-Rating"] >= min_rating)]["ISBN"].tolist()
    positions = [M["isbn_index"][i] for i in user_books if i in M["isbn_index"]]
    cols = ["ISBN", "Book-Title", "score"]
    if not positions:
        return pd.DataFrame(columns=cols)
    result = M["books_content"].copy()
    result["score"] = M["content_sim"][:, positions].max(axis=1)
    result = result[~result["ISBN"].isin(user_books)]
    return result.sort_values("score", ascending=False)[cols].head(n)


# =====================================================================
# F. WEIGHTED HYBRID
# =====================================================================
def recommend_hybrid(M, b, user_id, n=10, alpha=0.5):
    cf = recommend_item_based(b, user_id, M["title_map"], n=50)
    if cf.empty:
        return cf.assign(content_score=[], hybrid_score=[])

    _, idx, _ = _user_row(b, user_id)
    user_pos = [M["isbn_index"][i] for i in b["isbns"][idx] if i in M["isbn_index"]]

    cf["content_score"] = cf["ISBN"].apply(
        lambda isbn: M["content_sim"][M["isbn_index"][isbn], user_pos].max()
        if isbn in M["isbn_index"] and user_pos
        else 0
    )
    max_score = cf["score"].max()
    cf_norm = cf["score"] / max_score if max_score > 0 else 0
    cf["hybrid_score"] = alpha * cf_norm + (1 - alpha) * cf["content_score"]
    return cf.sort_values("hybrid_score", ascending=False).head(n)


# =====================================================================
# G. EVALUATION
# =====================================================================
def precision_recall(rec, relevant_set, k=10):
    recommended = set(rec.head(k)["ISBN"])
    hit = len(relevant_set & recommended)
    precision = hit / k
    recall = hit / len(relevant_set) if relevant_set else 0
    return precision, recall


def evaluate_all_methods(M, user_id, k=10, alpha=0.5):
    """Evaluasi 5 metode untuk satu user pada bundle TRAIN. Return dict nama -> (P, R)."""
    b = M["train"]
    relevant = M["relevant"].get(user_id, set())

    pop_train = (
        b["df"].groupby(["ISBN", "Book-Title"]).agg(rating_count=("Book-Rating", "count")).reset_index()
        .sort_values("rating_count", ascending=False)
    )
    recs = {
        "User-Based CF": recommend_user_based(b, user_id, n=k),
        "Item-Based CF": recommend_item_based(b, user_id, M["title_map"], n=k),
        "Content-Based": recommend_content_user(M, b, user_id, n=k),
        "Weighted Hybrid": recommend_hybrid(M, b, user_id, n=k, alpha=alpha),
        "Popularity-Based": pop_train.head(k),
    }
    return {name: precision_recall(r, relevant, k) for name, r in recs.items()}


# =====================================================================
# UI HELPERS
# =====================================================================
def show_df(df, **kwargs):
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    st.dataframe(df, width="stretch", **kwargs)


def barh(labels, values, title, xlabel, ylabel="Judul Buku"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def fmt_rules(rules):
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return out[["antecedents", "consequents", "support", "confidence", "lift"]]


@st.cache_resource(show_spinner="Menjalankan Apriori...")
def run_apriori(ratings_src, books_src, users_src, top_n, min_support, min_conf, max_len):
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    data_clean = load_and_clean(ratings_src, books_src, users_src)["data_clean"]
    assoc = data_clean[["User-ID", "Book-Title"]]
    top_books = assoc["Book-Title"].value_counts().head(top_n).index
    filtered = assoc[assoc["Book-Title"].isin(top_books)]
    transactions = filtered.groupby("User-ID")["Book-Title"].apply(list).tolist()

    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    df_tx = pd.DataFrame(arr, columns=te.columns_)

    freq = apriori(df_tx, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=True)
    if freq.empty:
        return len(transactions), freq, pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    return len(transactions), freq, rules


# =====================================================================
# SIDEBAR - DATA SOURCE & PARAMETERS
# =====================================================================
st.sidebar.title("📚 Book Recommender")
st.sidebar.caption("Book-Crossing dataset")

source_mode = st.sidebar.radio("Sumber dataset", ["Upload CSV", "Folder lokal"])

sources = None
if source_mode == "Upload CSV":
    up_r = st.sidebar.file_uploader(FILE_NAMES["ratings"], type="csv", key="up_r")
    up_b = st.sidebar.file_uploader(FILE_NAMES["books"], type="csv", key="up_b")
    up_u = st.sidebar.file_uploader(FILE_NAMES["users"], type="csv", key="up_u")
    if up_r and up_b and up_u:
        sources = (up_r.getvalue(), up_b.getvalue(), up_u.getvalue())
else:
    folder = st.sidebar.text_input("Path folder berisi 3 file BX-*.csv", value="")
    if folder:
        paths = [os.path.join(folder, FILE_NAMES[k]) for k in ("ratings", "books", "users")]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            st.sidebar.error("File tidak ditemukan:\n" + "\n".join(missing))
        else:
            sources = tuple(paths)

st.sidebar.divider()
st.sidebar.subheader("Parameter Model")
top_n_books = st.sidebar.slider("Jumlah buku teratas (CF & Content)", 200, 2000, 1000, 100)
test_size = st.sidebar.slider("Proporsi data testing", 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random state", value=42, step=1)

# =====================================================================
# MAIN
# =====================================================================
st.title("📚 Book Recommendation System")
st.caption(
    "Tugas DMV Week 10 · Shafadel Lestariyan Lainy · NRP 5003241060 · "
    "Association Rule, Popularity, Collaborative Filtering, Content-Based, Hybrid & Evaluasi"
)

if sources is None:
    st.info(
        "Silakan upload **BX-Book-Ratings.csv**, **BX-Books.csv**, dan **BX-Users.csv** "
        "di sidebar (atau isi path folder lokal) untuk memulai."
    )
    st.stop()

loaded = load_and_clean(*sources)
data_clean, data_explicit, stats = loaded["data_clean"], loaded["data_explicit"], loaded["stats"]

M = build_models(*sources, top_n_books, test_size, int(seed))
full, train = M["full"], M["train"]
default_user = int(full["users"][np.diff(full["mat"].indptr).argmax()])

tabs = st.tabs(
    [
        "📊 Data & Cleaning",
        "🔗 Association Rule",
        "🔥 Popularity",
        "👥 Collaborative Filtering",
        "📖 Content-Based",
        "🧬 Weighted Hybrid",
        "📏 Evaluasi",
    ]
)

# ---------------------------------------------------------------- A
with tabs[0]:
    st.header("A. Data Cleaning and Preprocessing")

    c1, c2, c3 = st.columns(3)
    c1.metric("Ratings", f"{stats['shape_ratings'][0]:,} baris")
    c2.metric("Books", f"{stats['shape_books'][0]:,} baris")
    c3.metric("Users", f"{stats['shape_users'][0]:,} baris")

    with st.expander("Missing value & duplikat dataset asli", expanded=False):
        a, b_, c = st.columns(3)
        a.write("**Ratings**")
        a.dataframe(stats["missing_ratings"].rename("missing"))
        a.write(f"Duplikat: {stats['dup_ratings']}")
        b_.write("**Books**")
        b_.dataframe(stats["missing_books"].rename("missing"))
        b_.write(f"Duplikat: {stats['dup_books']}")
        c.write("**Users**")
        c.dataframe(stats["missing_users"].rename("missing"))
        c.write(f"Duplikat: {stats['dup_users']}")

    st.subheader("Hasil penggabungan & pembersihan")
    d1, d2, d3 = st.columns(3)
    d1.metric("Data gabungan", f"{stats['shape_merged'][0]:,}")
    d2.metric("Setelah drop buku tanpa judul", f"{stats['shape_clean'][0]:,}")
    d3.metric("Rating eksplisit (>0)", f"{stats['shape_explicit'][0]:,}")

    with st.expander("Missing value data gabungan"):
        st.dataframe(stats["missing_merged"].rename("missing"))

    st.subheader("Distribusi Book-Rating")
    dist = stats["rating_dist"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(dist.index.astype(str), dist.values)
    ax.set_xlabel("Book-Rating")
    ax.set_ylabel("Jumlah")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Rating 0 adalah rating implisit (pengguna berinteraksi tanpa memberi nilai), sehingga "
        "metode berbasis rating hanya memakai rating eksplisit (> 0)."
    )

    st.subheader("Sampel data bersih")
    st.dataframe(data_clean.head(20), width="stretch")

# ---------------------------------------------------------------- B
with tabs[1]:
    st.header("B. Association Rule — Apriori")
    st.write("Pola buku yang sering dibaca/di-rating bersamaan oleh pengguna yang sama.")

    c1, c2, c3, c4 = st.columns(4)
    ap_top = c1.slider("Jumlah buku terpopuler", 200, 2000, 1000, 100, key="ap_top")
    ap_sup = c2.number_input("Min support", 0.0005, 0.05, 0.001, 0.0005, format="%.4f")
    ap_conf = c3.slider("Min confidence", 0.05, 0.9, 0.1, 0.05)
    ap_len = c4.selectbox("Panjang itemset maks.", [2, 3, 4], index=0)

    if st.button("▶️ Jalankan Apriori", type="primary"):
        st.session_state["apriori_params"] = (ap_top, ap_sup, ap_conf, ap_len)

    if "apriori_params" in st.session_state:
        try:
            n_tx, freq, rules = run_apriori(*sources, *st.session_state["apriori_params"])
        except ImportError:
            st.error("Library `mlxtend` belum terpasang. Jalankan: pip install mlxtend")
            st.stop()

        m1, m2, m3 = st.columns(3)
        m1.metric("Jumlah transaksi (user)", f"{n_tx:,}")
        m2.metric("Frequent itemsets", f"{len(freq):,}")
        m3.metric("Association rules", f"{len(rules):,}")

        if rules.empty:
            st.warning("Tidak ada rule yang ditemukan. Turunkan min support / min confidence.")
        else:
            sort_by = st.selectbox("Urutkan berdasarkan", ["lift", "confidence", "support"])
            query = st.text_input("Cari judul buku pada rule (opsional)")
            show = fmt_rules(rules).sort_values(sort_by, ascending=False)
            if query:
                mask = show["antecedents"].str.contains(query, case=False) | show[
                    "consequents"
                ].str.contains(query, case=False)
                show = show[mask]
            n_show = st.slider("Jumlah rule ditampilkan", 5, 100, 10)
            show_df(show.head(n_show))

            if len(show):
                top = show.iloc[0]
                st.info(
                    f"**Interpretasi:** rule teratas berdasarkan *{sort_by}* adalah "
                    f"**{top['antecedents']} → {top['consequents']}** dengan support "
                    f"{top['support']:.4%}, confidence {top['confidence']:.2%}, dan lift {top['lift']:.2f}. "
                    "Lift > 1 berarti kedua buku muncul bersama lebih sering daripada yang diharapkan secara acak."
                )
    else:
        st.caption("Atur parameter lalu tekan tombol untuk menjalankan Apriori.")

# ---------------------------------------------------------------- C
with tabs[2]:
    st.header("C. Popularity-Based Recommendation")
    st.write("Buku direkomendasikan berdasarkan jumlah pengguna yang memberi rating.")

    c1, c2 = st.columns(2)
    n_pop = c1.slider("Jumlah buku ditampilkan", 5, 30, 10, key="n_pop")
    min_cnt = c2.slider("Minimal jumlah rating", 1, 200, 1, key="min_cnt")

    pop = popularity_table(data_explicit)
    pop = pop[pop["rating_count"] >= min_cnt]
    top_pop = pop.head(n_pop)
    show_df(top_pop)

    barh(
        top_pop.sort_values("rating_count")["Book-Title"],
        top_pop.sort_values("rating_count")["rating_count"],
        f"Top {n_pop} Buku Terpopuler",
        "Jumlah Rating",
    )
    if len(top_pop):
        r = top_pop.iloc[0]
        st.info(
            f"**Interpretasi:** *{r['Book-Title']}* adalah buku dengan rating terbanyak "
            f"({int(r['rating_count'])} rating, rata-rata {r['average_rating']:.2f}). "
            "Popularitas tidak selalu sejalan dengan rata-rata rating yang tinggi."
        )

# ---------------------------------------------------------------- D
with tabs[3]:
    st.header("D. Collaborative Filtering — User-Based & Item-Based")
    st.caption(
        f"User-item matrix: {full['mat'].shape[0]:,} pengguna × {full['mat'].shape[1]:,} buku"
    )

    c1, c2 = st.columns(2)
    cf_user = c1.number_input("User-ID", value=default_user, step=1, key="cf_user")
    n_cf = c2.slider("Jumlah rekomendasi", 5, 30, 10, key="n_cf")
    cf_user = int(cf_user)

    if cf_user not in full["users"]:
        st.error(
            f"User-ID {cf_user} tidak ada pada matrix. Default (paling aktif): {default_user}"
        )
    else:
        _, idx, _ = _user_row(full, cf_user)
        st.write(f"User **{cf_user}** telah memberi rating pada **{len(idx)}** buku.")

        left, right = st.columns(2)
        with left:
            st.subheader("User-Based")
            k_nb = st.slider("Jumlah tetangga (similar users)", 3, 50, 10)
            show_df(recommend_user_based(full, cf_user, n=n_cf, k_neighbors=k_nb))
            st.caption(
                "Berdasarkan rata-rata rating dari pengguna-pengguna yang mirip. Rata-rata 10 dengan "
                "rating_count = 1 belum tentu kuat karena dukungan rating masih sedikit."
            )
        with right:
            st.subheader("Item-Based")
            k_sim = st.slider("Jumlah buku mirip per buku", 3, 50, 10)
            show_df(recommend_item_based(full, cf_user, M["title_map"], n=n_cf, k_similar=k_sim))
            st.caption("Skor = similarity buku × rating yang diberikan user pada buku acuan.")

        with st.expander("Buku yang sudah di-rating user ini"):
            rated = full["df"][full["df"]["User-ID"] == cf_user][
                ["ISBN", "Book-Title", "Book-Rating"]
            ].sort_values("Book-Rating", ascending=False)
            show_df(rated)

# ---------------------------------------------------------------- E
with tabs[4]:
    st.header("E. Content-Based Filtering")
    st.write("Kemiripan buku berdasarkan **judul** dan **penulis** (TF-IDF + cosine similarity).")

    bc = M["books_content"]
    labels = (bc["Book-Title"] + " — " + bc["Book-Author"] + " (" + bc["ISBN"] + ")").tolist()
    choice = st.selectbox("Pilih buku acuan", labels)
    n_cb = st.slider("Jumlah rekomendasi", 5, 30, 10, key="n_cb")
    sel_isbn = bc["ISBN"].iloc[labels.index(choice)]
    show_df(recommend_content_based(M, sel_isbn, n=n_cb))
    st.caption(
        "Skor 1.0 menandakan edisi (ISBN) lain dengan judul dan penulis yang sama persis dengan buku acuan."
    )

# ---------------------------------------------------------------- F
with tabs[5]:
    st.header("F. Weighted Hybrid Recommendation")
    st.write("Gabungan **Item-Based CF** dan **Content-Based** dengan pembobotan skor.")

    c1, c2, c3 = st.columns(3)
    hy_user = int(c1.number_input("User-ID", value=default_user, step=1, key="hy_user"))
    alpha = c2.slider("Bobot Item-Based CF (α)", 0.0, 1.0, 0.5, 0.05)
    n_hy = c3.slider("Jumlah rekomendasi", 5, 30, 10, key="n_hy")
    st.caption(f"hybrid_score = {alpha:.2f} × skor CF ternormalisasi + {1 - alpha:.2f} × skor content")

    if hy_user not in full["users"]:
        st.error(f"User-ID {hy_user} tidak ada pada matrix.")
    else:
        hybrid = recommend_hybrid(M, full, hy_user, n=n_hy, alpha=alpha)
        show_df(hybrid)
        if len(hybrid):
            top = hybrid.iloc[0]
            st.info(
                f"**Interpretasi:** *{top['Book-Title']}* berada di peringkat pertama dengan "
                f"hybrid score {top['hybrid_score']:.4f} (skor item-based {top['score']:.4f}, "
                f"skor content {top['content_score']:.4f})."
            )

# ---------------------------------------------------------------- G
with tabs[6]:
    st.header("G. Evaluation and Comparison")
    st.write(
        f"Evaluasi pada data training ({len(M['train_data']):,} baris) dan testing "
        f"({len(M['test_data']):,} baris, rating ≥ 7 dianggap relevan)."
    )

    c1, c2, c3 = st.columns(3)
    k_eval = c1.slider("K (Precision@K / Recall@K)", 5, 30, 10)
    alpha_eval = c2.slider("α Hybrid", 0.0, 1.0, 0.5, 0.05, key="alpha_eval")
    mode = c3.radio("Mode", ["Satu user", "Rata-rata banyak user"], horizontal=True)

    if mode == "Satu user":
        ev_user = int(st.number_input("User-ID", value=default_user, step=1, key="ev_user"))
        if ev_user not in train["users"]:
            st.error(f"User-ID {ev_user} tidak ada pada data training.")
            res = None
        else:
            n_rel = len(M["relevant"].get(ev_user, set()))
            st.write(f"Jumlah buku relevan pada data testing user {ev_user}: **{n_rel}**")
            if n_rel == 0:
                st.warning("User ini tidak memiliki buku relevan pada data testing.")
            res = evaluate_all_methods(M, ev_user, k=k_eval, alpha=alpha_eval)
    else:
        n_users = st.slider("Jumlah user (paling aktif di training)", 5, 200, 30)
        if st.button("▶️ Hitung evaluasi", type="primary"):
            activity = pd.Series(np.diff(train["mat"].indptr), index=train["users"])
            candidates = [u for u in activity.sort_values(ascending=False).index if u in M["relevant"]][
                :n_users
            ]
            bar = st.progress(0.0)
            acc = []
            for n_done, u in enumerate(candidates, 1):
                acc.append(evaluate_all_methods(M, int(u), k=k_eval, alpha=alpha_eval))
                bar.progress(n_done / len(candidates))
            bar.empty()
            names = list(acc[0].keys())
            res = {nm: tuple(np.mean([a[nm][i] for a in acc]) for i in (0, 1)) for nm in names}
            st.session_state["multi_eval"] = (res, len(candidates))
        res = None
        if "multi_eval" in st.session_state:
            res, n_eval = st.session_state["multi_eval"]
            st.caption(f"Rata-rata dari {n_eval} user.")

    if res:
        table = pd.DataFrame(res, index=[f"Precision@{k_eval}", f"Recall@{k_eval}"]).T
        table = table.sort_values(f"Precision@{k_eval}", ascending=False)
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#cdeccd"),
            width="stretch",
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(table))
        ax.bar(x - 0.2, table.iloc[:, 0], 0.4, label=table.columns[0])
        ax.bar(x + 0.2, table.iloc[:, 1], 0.4, label=table.columns[1])
        ax.set_xticks(x)
        ax.set_xticklabels(table.index, rotation=15)
        ax.set_title("Perbandingan Metode Rekomendasi")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        best = table.index[0]
        worst = table.index[-1]
        st.success(
            f"**Kesimpulan:** metode **{best}** memperoleh performa terbaik "
            f"(Precision@{k_eval} = {table.iloc[0, 0]:.2%}, Recall@{k_eval} = {table.iloc[0, 1]:.2%}), "
            f"sedangkan **{worst}** terendah (Precision = {table.iloc[-1, 0]:.2%}). "
            "Rekomendasi yang dipersonalisasi umumnya lebih sesuai dibanding popularitas umum."
        )
