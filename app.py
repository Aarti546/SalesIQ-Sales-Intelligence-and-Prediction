
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SalesIQ | Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

pio.templates.default = "plotly_white"

PLOT_FONT = {
    "family": "Arial, sans-serif",
    "color": "#334155",
    "size": 13
}


def style_plot(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=PLOT_FONT,
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 18, "color": "#0F172A"}
        } if title else None,
        margin=dict(l=55, r=30, t=55, b=55),
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "font": {"color": "#0F172A"}
        }
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#E8EDF4", linecolor="#CBD5E1",
        zeroline=False, tickfont={"color": "#64748B"}
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#E8EDF4", linecolor="#CBD5E1",
        zeroline=False, tickfont={"color": "#64748B"}
    )
    return fig

st.markdown("""
<style>

/* Overall page */
.stApp {
    background: #FFFFFF;
    color: #1F2937;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8FAFC 0%, #EEF4FA 100%);
    border-right: 1px solid #DCE5F0;
}

/* Headings */
h1 {
    color: #0F172A !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}
h2 {
    color: #1E293B !important;
    font-weight: 700 !important;
}
h3 {
    color: #334155 !important;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #DDE6F1;
    border-top: 3px solid #2563EB;
    border-radius: 18px;
    padding: 20px 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
}

div[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

div[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-weight: 750;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(120deg, #2563EB, #7C3AED);
    color: #FFFFFF !important;
    border: 1px solid #2563EB;
    border-radius: 12px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(120deg, #1D4ED8, #7C3AED, #DB2777);
    box-shadow: 0 6px 16px rgba(236, 72, 153, 0.24);
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stDateInput input {
    background-color: #FFFFFF;
    color: #1F2937;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    color: #1F2937;
    border-radius: 10px;
    border-color: #CBD5E1;
}

/* Radio navigation */
div[role="radiogroup"] {
    gap: 0.35rem;
}

div[role="radiogroup"] label {
    background-color: #F1F5F9;
    color: #334155;
    border-radius: 10px;
    padding: 8px 10px;
    transition: background 0.2s ease;
}

div[role="radiogroup"] label:hover {
    background-color: #E8EEF6;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #64748B;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB;
}

/* Expanders and bordered containers */
div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #DDE6F1;
    background-color: #FFFFFF;
    border-radius: 15px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 13px;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #DDE6F1;
    border-radius: 12px;
    overflow: hidden;
}

/* Dividers */
hr {
    border-color: #E2E8F0;
}

/* Footer text */
small {
    color: #64748B;
}

</style>
""", unsafe_allow_html=True)



# ---------------- DATA ----------------
DATA_FILE = "Amazon India Sales Data.csv"


@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)

    # Standardize headers so raw and previously cleaned CSVs can work
    data.columns = (
        data.columns.astype(str).str.strip().str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    aliases = {
        "ship_service_level": ["ship_service_level", "shipservicelevel"],
        "fulfilment": ["fulfilment", "fulfillment"],
        "ship_state": ["ship_state", "shipstate"],
        "qty": ["qty", "quantity"],
        "amount": ["amount", "sales_amount"],
        "b2b": ["b2b", "business_to_business"],
        "category": ["category"],
        "size": ["size"],
        "status": ["status"],
        "date": ["date"],
    }

    rename = {}
    for standard, choices in aliases.items():
        for choice in choices:
            if choice in data.columns:
                rename[choice] = standard
                break

    data = data.rename(columns=rename)

    required = [
        "status", "qty", "amount", "category", "size",
        "fulfilment", "ship_service_level", "b2b"
    ]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "Required columns not found: " + ", ".join(missing)
            + ". Check the CSV headers."
        )

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for col in ["qty", "amount"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Convert common B2B values into 0/1
    b2b_values = data["b2b"].astype(str).str.strip().str.lower()
    data["b2b"] = b2b_values.map({
        "true": 1, "1": 1, "yes": 1,
        "false": 0, "0": 0, "no": 0
    })

    # Create the cancellation target
    data["is_cancelled"] = (
        data["status"].astype(str).str.strip().str.lower()
        .eq("cancelled").astype(int)
    )

    for col in [
        "category", "size", "fulfilment", "ship_service_level"
    ]:
        data[col] = data[col].fillna("Unknown").astype(str)

    return data


try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info(
        "Place your Amazon CSV beside app.py and check DATA_FILE "
        "and the column names."
    )
    st.stop()

# ---------------- TRAIN MODELS ----------------
CLASS_FEATURES = [
    "qty", "category", "size", "fulfilment",
    "ship_service_level", "b2b"
]
NUMERIC_FEATURES = ["qty", "b2b"]
CATEGORICAL_FEATURES = [
    "category", "size", "fulfilment", "ship_service_level"
]


def make_classifier(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    return Pipeline([("preprocessing", prep), ("model", model)])


@st.cache_resource
def train_models(data):
    # Regression models use rows with a known amount.
    reg = data.dropna(subset=["amount", "qty"]).copy()
    X_reg = reg[["qty"]]
    y_reg = reg["amount"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    }

    reg_results = {}
    for name, model in regression_models.items():
        model.fit(X_train_r, y_train_r)
        pred = model.predict(X_test_r)
        reg_results[name] = {
            "model": model,
            "actual": y_test_r,
            "predictions": pred,
            "MAE": mean_absolute_error(y_test_r, pred),
            "R²": r2_score(y_test_r, pred)
        }

    # Classification uses all rows and excludes status and amount
    # so the target is not accidentally given to the model.
    cls = data.copy()
    X_cls = cls[CLASS_FEATURES]
    y_cls = cls["is_cancelled"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42,
        stratify=y_cls
    )

    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=3, random_state=42
        ),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5)
    }

    cls_results = {}
    for name, estimator in classification_models.items():
        model = make_classifier(estimator)
        model.fit(X_train_c, y_train_c)
        pred = model.predict(X_test_c)

        cls_results[name] = {
            "model": model,
            "actual": y_test_c,
            "predictions": pred,
            "Accuracy": accuracy_score(y_test_c, pred),
            "Precision": precision_score(
                y_test_c, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test_c, pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test_c, pred, zero_division=0
            ),
            "Confusion matrix": confusion_matrix(y_test_c, pred)
        }

    return reg_results, cls_results


with st.spinner("Preparing data and training the five models..."):
    reg_results, cls_results = train_models(df)

# ---------------- SIDEBAR ----------------
st.sidebar.title("SalesIQ")
st.sidebar.caption("Sales Intelligence & Prediction")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Sales EDA",
        "Model Comparison",
        "Model Details",
        "Cancellation Prediction",
        "Dataset Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Amazon India Sales Data | 2022")

# ---------------- HEADER ----------------
st.title("SalesIQ")
st.caption("Understand sales. Explore patterns. Evaluate predictions.")

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Business Overview")

    total_sales = df["amount"].sum()
    cancellation_rate = df["is_cancelled"].mean() * 100
    average_order_value = df["amount"].mean()
    total_records = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded sales", f"₹{total_sales:,.0f}")
    c2.metric("Dataset records", f"{total_records:,}")
    c3.metric("Cancellation rate", f"{cancellation_rate:.2f}%")
    c4.metric(
        "Average recorded amount",
        f"₹{average_order_value:,.2f}"
        if pd.notna(average_order_value) else "N/A"
    )

    st.info(
        "Sales totals are calculated from non-missing amount values. "
        "Check the dataset's meaning and currency before interpreting "
        "these figures as realized revenue."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Sales by Category")
        category_sales = (
            df.groupby("category", dropna=False)["amount"]
            .sum().sort_values(ascending=False).head(10)
            .reset_index()
        )
        fig = px.bar(
            category_sales, x="category", y="amount",
            color_discrete_sequence=["#22D3EE"]
        )
        fig.update_layout(
            xaxis_title="Category", yaxis_title="Recorded amount",
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with right:
        st.markdown("### Order Cancellation")
        counts = df["is_cancelled"].map({
            0: "Not cancelled", 1: "Cancelled"
        }).value_counts().rename_axis("Status").reset_index(name="Orders")
        fig = px.pie(
            counts, names="Status", values="Orders", hole=0.55,
            color_discrete_sequence=["#F472B6", "#FBBF24"]
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown("### Model Highlights")
    st.write(
        "The regression models estimate amount using quantity only. "
        "Their performance should be judged using MAE and R². "
        "The classification models estimate cancellation using order "
        "characteristics. Compare recall as well as accuracy because "
        "missed cancellations matter."
    )

# ---------------- EDA ----------------
elif page == "Sales EDA":
    st.subheader("Exploratory Data Analysis")

    available_categories = sorted(df["category"].unique().tolist())
    chosen_categories = st.multiselect(
        "Filter categories",
        available_categories,
        default=available_categories
    )
    eda = df[df["category"].isin(chosen_categories)].copy()

    left, right = st.columns(2)

    with left:
        st.markdown("### Category-wise Amount")
        category_sales = (
            eda.groupby("category")["amount"].sum()
            .sort_values(ascending=False).reset_index()
        )
        fig = px.bar(
            category_sales, x="category", y="amount",
            color_discrete_sequence=["#8B5CF6"]
        )
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with right:
        st.markdown("### Quantity Distribution")
        fig = px.histogram(
            eda, x="qty", nbins=20,
            color_discrete_sequence=["#22D3EE"]
        )
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Fulfilment Method")
        fulfilment_counts = (
            eda["fulfilment"].value_counts()
            .rename_axis("Fulfilment").reset_index(name="Orders")
        )
        fig = px.bar(
            fulfilment_counts, x="Fulfilment", y="Orders",
            color_discrete_sequence=["#F472B6"]
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with right:
        st.markdown("### B2B vs B2C")
        b2b_data = eda.copy()
        b2b_data["Customer type"] = b2b_data["b2b"].map({
            1: "B2B", 0: "B2C"
        }).fillna("Unknown")
        b2b_counts = (
            b2b_data["Customer type"].value_counts()
            .rename_axis("Customer type")
            .reset_index(name="Orders")
        )
        fig = px.pie(
            b2b_counts, names="Customer type", values="Orders",
            hole=0.5,
            color_discrete_sequence=["#22D3EE", "#F472B6", "#FBBF24"]
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    if "date" in eda.columns and eda["date"].notna().any():
        st.markdown("### Monthly Sales Trend")
        monthly = (
            eda.dropna(subset=["date"])
            .assign(month=lambda x: x["date"].dt.to_period("M").astype(str))
            .groupby("month")["amount"].sum().reset_index()
        )
        fig = px.line(
            monthly, x="month", y="amount", markers=True,
            color_discrete_sequence=["#A78BFA"]
        )
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown("### Missing Values")
    missing = (
        df.isna().sum().rename("Missing values")
        .reset_index().rename(columns={"index": "Column"})
    )
    st.dataframe(missing, use_container_width=True, hide_index=True)

# ---------------- MODEL COMPARISON ----------------
elif page == "Model Comparison":
    st.subheader("Compare All Five Models")

    regression_table = pd.DataFrame([
        {
            "Model": name,
            "MAE": result["MAE"],
            "R²": result["R²"]
        }
        for name, result in reg_results.items()
    ])

    classification_table = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1-score": result["F1-score"]
        }
        for name, result in cls_results.items()
    ])

    st.markdown("### Regression Models")
    st.caption("Lower MAE is better; R² closer to 1 is better.")
    st.dataframe(
        regression_table.style.format({
            "MAE": "{:.3f}", "R²": "{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    fig = px.bar(
        regression_table, x="Model", y="MAE",
        color_discrete_sequence=["#F472B6"],
        title="Regression: Mean Absolute Error (lower is better)"
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown("### Classification Models")
    st.caption(
        "Precision measures the reliability of cancellation alerts; "
        "recall measures how many actual cancellations were detected."
    )
    st.dataframe(
        classification_table.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1-score": "{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    fig = px.bar(
        classification_table, x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-score"],
        barmode="group",
        color_discrete_sequence=[
            "#22D3EE", "#34D399", "#F472B6", "#FBBF24"
        ],
        title="Classification Metrics"
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)

# ---------------- MODEL DETAILS ----------------
elif page == "Model Details":
    st.subheader("Model Evaluation Details")

    model_type = st.radio(
        "Choose model family",
        ["Regression", "Classification"],
        horizontal=True
    )

    if model_type == "Regression":
        selected = st.selectbox(
            "Select regression model", list(reg_results.keys())
        )
        result = reg_results[selected]

        c1, c2 = st.columns(2)
        c1.metric("MAE", f"{result['MAE']:.3f}")
        c2.metric("R²", f"{result['R²']:.4f}")

        plot_data = pd.DataFrame({
            "Actual amount": np.asarray(result["actual"]),
            "Predicted amount": result["predictions"]
        })
        fig = px.scatter(
            plot_data, x="Actual amount", y="Predicted amount",
            opacity=0.5, color_discrete_sequence=["#A78BFA"]
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)
        st.dataframe(plot_data.head(20), use_container_width=True)

    else:
        selected = st.selectbox(
            "Select classification model", list(cls_results.keys())
        )
        result = cls_results[selected]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{result['Accuracy']:.3f}")
        c2.metric("Precision", f"{result['Precision']:.3f}")
        c3.metric("Recall", f"{result['Recall']:.3f}")
        c4.metric("F1-score", f"{result['F1-score']:.3f}")

        cm = result["Confusion matrix"]
        cm_data = pd.DataFrame(
            cm,
            index=["Actual: Not cancelled", "Actual: Cancelled"],
            columns=["Predicted: Not cancelled", "Predicted: Cancelled"]
        )
        st.markdown("### Confusion Matrix")
        st.dataframe(cm_data, use_container_width=True)

        st.caption(
            "The top-left and bottom-right cells are correct predictions. "
            "The bottom-left cell represents cancellations the model missed."
        )

# ---------------- CANCELLATION PREDICTION ----------------
elif page == "Cancellation Prediction":
    st.subheader("Predict Order Cancellation")
    st.write(
        "Enter order characteristics to estimate the cancellation class. "
        "This is a model prediction, not a guarantee of the outcome."
    )

    with st.form("prediction_form"):
        qty = st.number_input(
            "Quantity", min_value=0, max_value=100, value=1, step=1
        )
        category = st.selectbox(
            "Category", sorted(df["category"].unique().tolist())
        )
        size = st.selectbox(
            "Size", sorted(df["size"].unique().tolist())
        )
        fulfilment = st.selectbox(
            "Fulfilment", sorted(df["fulfilment"].unique().tolist())
        )
        service = st.selectbox(
            "Shipping service level",
            sorted(df["ship_service_level"].unique().tolist())
        )
        b2b = st.selectbox(
            "Customer type", ["B2C", "B2B"]
        )
        chosen_model = st.selectbox(
            "Classification model", list(cls_results.keys())
        )

        submitted = st.form_submit_button("Predict Cancellation")

    if submitted:
        input_row = pd.DataFrame([{
            "qty": qty,
            "category": category,
            "size": size,
            "fulfilment": fulfilment,
            "ship_service_level": service,
            "b2b": 1 if b2b == "B2B" else 0
        }])

        model = cls_results[chosen_model]["model"]
        prediction = int(model.predict(input_row)[0])

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_row)[0][1]
            st.metric("Estimated cancellation probability", f"{probability:.1%}")

        if prediction == 1:
            st.warning("Model result: Cancellation predicted.")
        else:
            st.success("Model result: Cancellation not predicted.")

        st.caption(
            "This probability is model-based and should not be interpreted "
            "as a calibrated real-world risk without further validation."
        )

# ---------------- DATASET EXPLORER ----------------
elif page == "Dataset Explorer":
    st.subheader("Explore the Dataset")

    search = st.text_input("Search across displayed records")
    view = df.copy()

    if search:
        mask = view.astype(str).apply(
            lambda column: column.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)
        view = view[mask]

    st.write(f"Showing {len(view):,} records")
    st.dataframe(view, use_container_width=True, height=420)

    st.download_button(
        "Download displayed data as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="salesiq_filtered_data.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("SalesIQ • Built with Python & Streamlit")
