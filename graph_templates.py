"""
Graph Templates — Separation of Concerns
Pre-built graph code templates. The AI only fills in parameters (columns, file, title).
No code generation from scratch = minimal token usage.
"""

import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Directory to store generated graph images
GRAPH_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "graphs")
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# ============================================================
# STYLE CONFIG — Premium dark theme
# ============================================================
DARK_STYLE = {
    "bg_color": "#1a1a2e",
    "text_color": "#e0e0e0",
    "grid_color": "#333355",
    "accent_colors": ["#6c63ff", "#ff6584", "#43e97b", "#f9d423", "#38f9d7", "#fa709a", "#fee140", "#a18cd1"],
    "font_size_title": 14,
    "font_size_label": 11,
    "font_size_tick": 9,
}

def _apply_dark_style(fig, ax):
    """Apply premium dark theme to any chart."""
    s = DARK_STYLE
    fig.patch.set_facecolor(s["bg_color"])
    ax.set_facecolor(s["bg_color"])
    ax.title.set_color(s["text_color"])
    ax.xaxis.label.set_color(s["text_color"])
    ax.yaxis.label.set_color(s["text_color"])
    ax.tick_params(colors=s["text_color"], labelsize=s["font_size_tick"])
    ax.grid(True, color=s["grid_color"], alpha=0.3, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color(s["grid_color"])

def _save_graph(fig, prefix="graph"):
    """Save figure and return the filename."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(GRAPH_OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename


# ============================================================
# TEMPLATE: Bar Chart
# ============================================================
def bar_chart(file_path, x_col, y_col, title="Bar Chart", top_n=15):
    """Generate a bar chart from CSV/Excel data."""
    df = _load_data(file_path)
    if x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found. Available: {list(df.columns)}"

    data = df[[x_col, y_col]].dropna()
    # If y is numeric, aggregate; otherwise count
    if pd.api.types.is_numeric_dtype(data[y_col]):
        plot_data = data.groupby(x_col)[y_col].sum().nlargest(top_n)
    else:
        plot_data = data[x_col].value_counts().head(top_n)
        y_col = "Count"

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = DARK_STYLE["accent_colors"]
    bars = ax.bar(range(len(plot_data)), plot_data.values,
                  color=[colors[i % len(colors)] for i in range(len(plot_data))])
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels([str(l)[:20] for l in plot_data.index], rotation=45, ha="right")
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=DARK_STYLE["font_size_title"], fontweight="bold")
    _apply_dark_style(fig, ax)
    filename = _save_graph(fig, "bar")
    return filename, f"Bar chart: {title} ({len(plot_data)} items)"


# ============================================================
# TEMPLATE: Line Chart
# ============================================================
def line_chart(file_path, x_col, y_col, title="Line Chart"):
    """Generate a line chart from CSV/Excel data."""
    df = _load_data(file_path)
    if x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found. Available: {list(df.columns)}"

    data = df[[x_col, y_col]].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data[x_col], data[y_col], color=DARK_STYLE["accent_colors"][0],
            linewidth=2, marker="o", markersize=4)
    ax.fill_between(data[x_col].values, data[y_col].values, alpha=0.15,
                    color=DARK_STYLE["accent_colors"][0])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=DARK_STYLE["font_size_title"], fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    _apply_dark_style(fig, ax)
    filename = _save_graph(fig, "line")
    return filename, f"Line chart: {title}"


# ============================================================
# TEMPLATE: Pie Chart
# ============================================================
def pie_chart(file_path, label_col, value_col=None, title="Pie Chart", top_n=8):
    """Generate a pie chart from CSV/Excel data."""
    df = _load_data(file_path)
    if label_col not in df.columns:
        return None, f"Column '{label_col}' not found. Available: {list(df.columns)}"

    if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        data = df.groupby(label_col)[value_col].sum().nlargest(top_n)
    else:
        data = df[label_col].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = DARK_STYLE["accent_colors"][:len(data)]
    wedges, texts, autotexts = ax.pie(data.values, labels=[str(l)[:20] for l in data.index],
                                       autopct="%1.1f%%", colors=colors,
                                       textprops={"color": DARK_STYLE["text_color"], "fontsize": 10})
    ax.set_title(title, fontsize=DARK_STYLE["font_size_title"], fontweight="bold",
                 color=DARK_STYLE["text_color"])
    fig.patch.set_facecolor(DARK_STYLE["bg_color"])
    filename = _save_graph(fig, "pie")
    return filename, f"Pie chart: {title} ({len(data)} slices)"


# ============================================================
# TEMPLATE: Scatter Plot
# ============================================================
def scatter_plot(file_path, x_col, y_col, title="Scatter Plot"):
    """Generate a scatter plot from CSV/Excel data."""
    df = _load_data(file_path)
    if x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found. Available: {list(df.columns)}"

    data = df[[x_col, y_col]].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data[x_col], data[y_col], color=DARK_STYLE["accent_colors"][1],
               alpha=0.7, s=40, edgecolors=DARK_STYLE["accent_colors"][0])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=DARK_STYLE["font_size_title"], fontweight="bold")
    _apply_dark_style(fig, ax)
    filename = _save_graph(fig, "scatter")
    return filename, f"Scatter plot: {title}"


# ============================================================
# TEMPLATE: Histogram
# ============================================================
def histogram(file_path, col, title="Histogram", bins=20):
    """Generate a histogram from CSV/Excel data."""
    df = _load_data(file_path)
    if col not in df.columns:
        return None, f"Column '{col}' not found. Available: {list(df.columns)}"

    data = df[col].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=bins, color=DARK_STYLE["accent_colors"][4], edgecolor=DARK_STYLE["bg_color"],
            alpha=0.85)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=DARK_STYLE["font_size_title"], fontweight="bold")
    _apply_dark_style(fig, ax)
    filename = _save_graph(fig, "hist")
    return filename, f"Histogram: {title}"


# ============================================================
# TEMPLATE: Horizontal Bar Chart
# ============================================================
def hbar_chart(file_path, label_col, value_col, title="Horizontal Bar Chart", top_n=15):
    """Generate a horizontal bar chart from CSV/Excel data."""
    df = _load_data(file_path)
    if label_col not in df.columns or value_col not in df.columns:
        return None, f"Columns '{label_col}' or '{value_col}' not found. Available: {list(df.columns)}"

    if pd.api.types.is_numeric_dtype(df[value_col]):
        data = df.groupby(label_col)[value_col].sum().nlargest(top_n)
    else:
        data = df[label_col].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = DARK_STYLE["accent_colors"]
    ax.barh(range(len(data)), data.values,
            color=[colors[i % len(colors)] for i in range(len(data))])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([str(l)[:25] for l in data.index])
    ax.set_xlabel(value_col)
    ax.set_title(title, fontsize=DARK_STYLE["font_size_title"], fontweight="bold")
    ax.invert_yaxis()
    _apply_dark_style(fig, ax)
    filename = _save_graph(fig, "hbar")
    return filename, f"Horizontal bar: {title}"


# ============================================================
# DATA LOADER — supports CSV and Excel
# ============================================================
def _load_data(file_path):
    """Load CSV or Excel file into a DataFrame."""
    if file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    else:
        # Try multiple encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(file_path, encoding="utf-8", errors="ignore")


# ============================================================
# DISPATCHER — called by the AI tool
# ============================================================
GRAPH_TYPES = {
    "bar": bar_chart,
    "line": line_chart,
    "pie": pie_chart,
    "scatter": scatter_plot,
    "histogram": histogram,
    "hbar": hbar_chart,
}

def generate_graph(graph_type, file_path, x_col, y_col=None, title=None):
    """
    Main entry point. The AI calls this with minimal params.
    Returns (filename, description) or (None, error_message).
    """
    func = GRAPH_TYPES.get(graph_type)
    if not func:
        return None, f"Unknown graph type '{graph_type}'. Available: {list(GRAPH_TYPES.keys())}"

    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    if not title:
        title = f"{graph_type.title()} Chart"

    try:
        if graph_type == "pie":
            return func(file_path, x_col, y_col, title)
        elif graph_type == "histogram":
            return func(file_path, x_col, title)
        else:
            if not y_col:
                return None, f"'{graph_type}' chart needs both x_col and y_col."
            return func(file_path, x_col, y_col, title)
    except Exception as e:
        return None, f"Graph error: {str(e)}"


def get_file_columns(file_path):
    """Return column names for a data file."""
    try:
        df = _load_data(file_path)
        return list(df.columns)
    except Exception as e:
        return f"Error: {e}"
