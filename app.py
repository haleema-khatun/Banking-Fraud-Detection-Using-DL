import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Folders
UPLOAD_FOLDER = "uploads"
MODELS_FOLDER = "models"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

DATASET_PATH = os.path.join(UPLOAD_FOLDER, "bank_transactions_data1.csv")
GRAPH_PATH = os.path.join(STATIC_FOLDER, "model_comparison.png")
BEST_MODEL_PATH = os.path.join(MODELS_FOLDER, "best_model.pkl")
PREPROCESS_PATH = os.path.join(MODELS_FOLDER, "preprocess.pkl")
METRICS_PATH = os.path.join(MODELS_FOLDER, "metrics.pkl")


# =============== DB INIT ===============

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            username TEXT UNIQUE,
            password TEXT,
            phone TEXT
        )
    """)
    conn.commit()
    conn.close()


@app.before_first_request
def setup():
    init_db()


def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn


# =============== AUTH ROUTES ===============

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        username = request.form["username"]
        phone = request.form["phone"]
        password = request.form["password"]

        hashed = generate_password_hash(password)

        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users(name,email,username,password,phone) VALUES (?,?,?,?,?)",
                (name, email, username, hashed, phone),
            )
            conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("user_home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    # default: admin / admin123
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin":
            session.clear()
            session["admin"] = True
            flash("Admin logged in.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials.", "danger")

    return render_template("admin_login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))


# =============== ADMIN SECTION ===============

@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    dataset_exists = os.path.exists(DATASET_PATH)
    metrics_exists = os.path.exists(METRICS_PATH)

    return render_template(
        "admin_dashboard.html",
        dataset_exists=dataset_exists,
        metrics_exists=metrics_exists,
    )


import pandas as pd
from werkzeug.utils import secure_filename

@app.route("/admin/upload_dataset", methods=["GET", "POST"])
def admin_upload_dataset():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    preview = None

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("❌ Please select a CSV file.", "danger")
            return redirect(url_for("admin_upload_dataset"))

        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # ✅ Read CSV and get first 5 rows
        df = pd.read_csv(file_path)
        preview = df.head().to_html(classes="table table-striped", index=False)

        flash("✅ Dataset uploaded successfully.", "success")

    return render_template("upload.html", preview=preview)

import os
import pandas as pd
from sklearn.model_selection import train_test_split

@app.route("/admin/split_dataset", methods=["GET"])
def admin_split_dataset():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not os.path.exists(DATASET_PATH):
        flash("❌ Upload dataset first.", "danger")
        return redirect(url_for("admin_dashboard"))

    # ✅ Read dataset
    df = pd.read_csv(DATASET_PATH)

    total_rows = len(df)

    # ✅ Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_rows = len(train_df)
    test_rows = len(test_df)

    # ✅ Save files
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    train_path = os.path.join(UPLOAD_FOLDER, "train.csv")
    test_path = os.path.join(UPLOAD_FOLDER, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    flash("✅ Dataset successfully split into Train & Test sets.", "success")

    # ✅ Send row counts to HTML
    return render_template(
        "split.html",
        total_rows=total_rows,
        train_rows=train_rows,
        test_rows=test_rows
    )

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
BEST_MODEL_PATH = "models/best_model.pkl"
PREPROCESS_PATH = "models/preprocess.pkl"
METRICS_PATH = "models/metrics.pkl"
GRAPH_PATH = "static/model_accuracy.png"

os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
# ======================================================
# GNN MODEL
# ======================================================
class GNNModel(torch.nn.Module):

    def __init__(self, input_dim):
        super(GNNModel, self).__init__()

        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)
# ======================================================
# TRAIN GNN FUNCTION
# ======================================================
def train_gnn(X, y):

    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    num_nodes = X_tensor.shape[0]

    # Create simple transaction graph
    edge_list = []

    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

    model = GNNModel(X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ---------- TRAIN ----------
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    # ---------- EVALUATE ----------
    model.eval()
    _, pred = model(data).max(dim=1)

    acc = (pred == data.y).sum().item() / len(y_tensor)

    return model, round(acc * 100, 2)
# ======================================================
# AUTOENCODER MODEL
# ======================================================
# -------- AUTOENCODER IMPORTS --------
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# ======================================================
# IMPROVED AUTOENCODER
# ======================================================
# ======================================================
# DEEP AUTOENCODER MODEL
# ======================================================
class AutoEncoder(nn.Module):

    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16)
        )

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
# ======================================================
# TRAIN AUTOENCODER (PAPER IMPLEMENTATION)
# ======================================================
def train_autoencoder(X, y):

    device = torch.device("cpu")

    # ---------------- Convert to Tensor ----------------
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Train ONLY on normal transactions
    normal_idx = (y == 0).values
    X_normal = X_tensor[normal_idx]

    # ---------------- DataLoader ----------------
    dataset = TensorDataset(X_normal)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # ---------------- Model ----------------
    model = AutoEncoder(X.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # ---------------- Epoch Selection ----------------
    data_size = X.shape[0]

    if data_size < 5000:
        epochs = 60
    elif data_size < 20000:
        epochs = 100
    else:
        epochs = 150

    # Early stopping parameters
    best_loss = float("inf")
    patience = 10
    counter = 0

    # ================= TRAINING =================
    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:
            batch_x = batch[0]

            optimizer.zero_grad()

            output = model(batch_x)
            loss = criterion(output, batch_x)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.6f}")

        # -------- Early Stopping --------
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    # ================= FRAUD DETECTION =================
    model.eval()

    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean(
            (X_tensor - reconstructed) ** 2,
            dim=1
        ).cpu().numpy()

    # -------- Dynamic Threshold (Paper Method) --------
    threshold = np.percentile(errors, 95)

    preds = (errors > threshold).astype(int)

    accuracy = accuracy_score(y, preds)

    print("Autoencoder Accuracy:", accuracy)

    return model, round(accuracy * 100, 2)
# ================= PATHS =================
MODEL_DIR = "models"

LOGISTIC_PATH = f"{MODEL_DIR}/logistic.pkl"
GB_PATH = f"{MODEL_DIR}/gradient.pkl"
XGB_PATH = f"{MODEL_DIR}/xgboost.pkl"
GNN_PATH = f"{MODEL_DIR}/gnn.pth"
AUTO_PATH = f"{MODEL_DIR}/autoencoder.pth"

PREPROCESS_PATH = f"{MODEL_DIR}/preprocess.pkl"
BEST_MODEL_NAME_PATH = f"{MODEL_DIR}/best_model.txt"
@app.route("/admin/train_models")
def admin_train_models():

    # ================= ADMIN CHECK =================
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    import os
    import pickle
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier

    # ================= PATHS =================
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    LOGISTIC_PATH = f"{MODEL_DIR}/logistic.pkl"
    GB_PATH = f"{MODEL_DIR}/gradient.pkl"
    XGB_PATH = f"{MODEL_DIR}/xgboost.pkl"
    GNN_PATH = f"{MODEL_DIR}/gnn.pth"
    AUTO_PATH = f"{MODEL_DIR}/autoencoder.pth"

    PREPROCESS_PATH = f"{MODEL_DIR}/preprocess.pkl"
    METRICS_PATH = f"{MODEL_DIR}/metrics.pkl"
    BEST_MODEL_PATH = f"{MODEL_DIR}/best_model.txt"
    GRAPH_PATH = "static/model_accuracy.png"

    csv_path = "bank_transactions_data1.csv"

    # ======================================================
    # 1. LOAD DATA
    # ======================================================
    df = pd.read_csv(csv_path)

    # ======================================================
    # 2. DROP UNUSED COLUMNS
    # ======================================================
    drop_cols = ["TransactionID", "TransactionDate", "IP Address", "DeviceID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ======================================================
    # 3. ENCODE CATEGORICAL DATA
    # ======================================================
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_mappings = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_mappings[col] = {
            cls: int(i) for i, cls in enumerate(le.classes_)
        }

    # ======================================================
    # 4. SPLIT FEATURES & LABEL
    # ======================================================
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    feature_cols = X.columns.tolist()

    # ======================================================
    # 5. SCALE DATA
    # ======================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================================================
    # 6. TRAIN TEST SPLIT
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ======================================================
    # 7. HANDLE CLASS IMBALANCE
    # ======================================================
    class_weights = {
        0: 1,
        1: np.sum(y_train == 0) / np.sum(y_train == 1)
    }

    # ======================================================
    # 8. TRAIN SKLEARN MODELS
    # ======================================================
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight=class_weights
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=class_weights[1],
            eval_metric="logloss"
        )
    }

    accuracies = {}
    best_acc = -1
    best_name = ""

    model_paths = {
        "Logistic Regression": LOGISTIC_PATH,
        "Gradient Boosting": GB_PATH,
        "XGBoost": XGB_PATH
    }

    for name, model in models.items():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds) * 100, 2)
        accuracies[name] = acc

        # SAVE MODEL
        with open(model_paths[name], "wb") as f:
            pickle.dump(model, f)

        if acc > best_acc:
            best_acc = acc
            best_name = name

    # ======================================================
    # 9. TRAIN GNN
    # ======================================================
    gnn_model, gnn_acc = train_gnn(X_scaled, y)

    accuracies["GNN"] = gnn_acc
    torch.save(gnn_model.state_dict(), GNN_PATH)

    if gnn_acc > best_acc:
        best_acc = gnn_acc
        best_name = "GNN"

    # ======================================================
    # 10. TRAIN AUTOENCODER
    # ======================================================
    auto_model, auto_acc = train_autoencoder(X_scaled, y)

    accuracies["AutoEncoder"] = auto_acc
    torch.save(auto_model.state_dict(), AUTO_PATH)

    if auto_acc > best_acc:
        best_acc = auto_acc
        best_name = "AutoEncoder"

    # ======================================================
    # 11. SAVE PREPROCESS INFO
    # ======================================================
    preprocess_info = {
        "drop_cols": drop_cols,
        "cat_cols": cat_cols,
        "feature_cols": feature_cols,
        "cat_mappings": cat_mappings,
        "scaler": scaler
    }

    with open(PREPROCESS_PATH, "wb") as f:
        pickle.dump(preprocess_info, f)

    # ======================================================
    # 12. SAVE BEST MODEL NAME
    # ======================================================
    with open(BEST_MODEL_PATH, "w") as f:
        f.write(best_name)

    # ======================================================
    # 13. SAVE METRICS
    # ======================================================
    metrics = {
        "accuracies": accuracies,
        "best_model": best_name,
        "best_acc": best_acc
    }

    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    # ======================================================
    # 14. SAVE GRAPH
    # ======================================================
    plt.figure(figsize=(8,5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()

    # ======================================================
    # 15. SUCCESS MESSAGE
    # ======================================================
    flash(
        f"✅ Training Complete. Best Model = {best_name} ({best_acc:.2f}%)",
        "success"
    )

    return redirect(url_for("admin_algorithms"))
@app.route("/admin/algorithms")
def admin_algorithms():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not os.path.exists(METRICS_PATH):
        flash("Train models first.", "danger")
        return redirect(url_for("admin_dashboard"))

    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)

    accuracies = metrics["accuracies"]
    best_model = metrics["best_model"]
    best_acc = metrics["best_acc"]

    return render_template(
        "algorithms.html",
        accuracies=accuracies,
        best_model=best_model,
        best_acc=best_acc,
    )

@app.route("/admin/reports")
def admin_reports():

    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not os.path.exists(METRICS_PATH):
        flash("Train models first.", "danger")
        return redirect(url_for("admin_dashboard"))

    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)

    accuracies = metrics["accuracies"]
    best_model = metrics["best_model"]
    best_acc = metrics["best_acc"]

    return render_template(
        "admin_reports.html",
        accuracies=accuracies,
        best_model=best_model,
        best_acc=best_acc
    )

# =============== USER SECTION ===============

import os
import pickle
import torch

# ------------------------------------------------------
# LOAD MODEL + PREPROCESS FUNCTION (FINAL VERSION)
# ------------------------------------------------------
import os
import pickle
import torch

def load_gradient_boosting_model():

    import pickle
    import os

    MODEL_PATH = "models/gradient.pkl"
    PREPROCESS_PATH = "models/preprocess.pkl"

    # check files
    if not os.path.exists(MODEL_PATH):
        print("❌ gradient.pkl not found")
        return None, None

    if not os.path.exists(PREPROCESS_PATH):
        print("❌ preprocess.pkl not found")
        return None, None

    # load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # load preprocessing
    with open(PREPROCESS_PATH, "rb") as f:
        prep = pickle.load(f)

    print("✅ Gradient Boosting model loaded")

    return model, prep
@app.route("/user/home")
def user_home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("user_home.html")
@app.route("/user/predict", methods=["GET", "POST"])
def user_predict():

    # ================= LOGIN CHECK =================
    if "user" not in session:
        return redirect(url_for("login"))

    # ================= LOAD MODEL =================
    model, prep = load_gradient_boosting_model()

    if model is None or prep is None:
        flash("Admin must train models first.", "danger")
        return redirect(url_for("user_home"))

    feature_cols = prep["feature_cols"]
    cat_cols = prep.get("cat_cols", [])
    cat_mappings = prep.get("cat_mappings", {})

    # ================= FORM SUBMIT =================
    if request.method == "POST":

        row = {}

        try:
            for col in feature_cols:

                val = request.form.get(col)

                if val is None or val.strip() == "":
                    flash(f"Enter value for {col}", "danger")
                    return render_template(
                        "user_predict.html",
                        feature_cols=feature_cols,
                        cat_cols=cat_cols,
                        cat_mappings=cat_mappings
                    )

                # Categorical Encoding
                if col in cat_cols:
                    mapping = cat_mappings[col]

                    if val not in mapping:
                        flash(f"Invalid value for {col}", "danger")
                        return render_template(
                            "user_predict.html",
                            feature_cols=feature_cols,
                            cat_cols=cat_cols,
                            cat_mappings=cat_mappings
                        )

                    row[col] = mapping[val]

                else:
                    row[col] = float(val)

        except Exception as e:
            flash(str(e), "danger")
            return render_template(
                "user_predict.html",
                feature_cols=feature_cols,
                cat_cols=cat_cols,
                cat_mappings=cat_mappings
            )

        # Create DataFrame
        df_input = pd.DataFrame([row])[feature_cols]

        # Scale
        X_input = prep["scaler"].transform(df_input)

        # Prediction
        pred = int(model.predict(X_input)[0])

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(X_input)[0][1] * 100
        else:
            confidence = 90.0

        # Result
        if pred == 1:
            result = "🚨 FRAUD TRANSACTION DETECTED"
            prevention_msg = """
            ⚠️ Prevention Tips:
            • Immediately block your card.
            • Contact your bank.
            • Change passwords.
            • Enable transaction alerts.
            """
            alert_type = "danger"
        else:
            result = "✅ NORMAL TRANSACTION"
            prevention_msg = "Transaction appears safe."
            alert_type = "success"

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(confidence, 2),
            input_data=row,
            prevention=prevention_msg,
            alert_type=alert_type
        )

    # ================= PAGE LOAD =================
    return render_template(
        "user_predict.html",
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        cat_mappings=cat_mappings
    )
@app.route("/about")
def about():
    return render_template("about.html")


# =============== RUN ===============
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)

