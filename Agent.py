import re
import os
import io
import base64
import json
import traceback
from numbers import Number

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, session
from google.cloud import bigquery
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------
# App setup
# -------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")

bq_client = bigquery.Client()

llm_global = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.2,
    max_output_tokens=512,
    google_api_key=os.environ.get("GOOGLE_CLOUD_API_KEY")
)

# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def get_dataset_schema(project_id, dataset_id):
    dataset_ref = bq_client.dataset(dataset_id, project=project_id)
    tables = list(bq_client.list_tables(dataset_ref))

    schema_info = f"You have access to the following tables in `{project_id}.{dataset_id}`:\n"
    for table in tables:
        table_ref = bq_client.get_table(f"{project_id}.{dataset_id}.{table.table_id}")
        columns = [field.name for field in table_ref.schema]
        schema_info += f"- `{project_id}.{dataset_id}.{table.table_id}`: {', '.join(columns)}\n"

    return schema_info

def clean_sql(sql_text):
    if not sql_text:
        return ""
    sql_text = re.sub(r"```[a-zA-Z]*\n?", "", sql_text)
    sql_text = sql_text.replace("```", "")
    return sql_text.strip()

def normalize_response_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

def is_safe_select_sql(sql_query: str) -> bool:
    """
    Minimal safety gate: only allow SELECT/WITH queries.
    """
    if not sql_query:
        return False
    s = sql_query.strip().lower()
    # allow WITH ... SELECT ... or SELECT ...
    if not (s.startswith("select") or s.startswith("with")):
        return False
    # block common DML/DDL keywords
    blocked = ["delete", "update", "insert", "merge", "drop", "alter", "truncate", "create"]
    return not any(re.search(rf"\b{k}\b", s) for k in blocked)

def detect_followup(user_query: str) -> bool:
    """
    Heuristic: short refinement questions typically used in follow-ups.
    """
    if not user_query:
        return False

    q = user_query.strip().lower()

    # common follow-up patterns
    followup_starts = [
        "give me for", "for ", "only ", "just ", "what about", "and for", "now for",
        "show for", "filter", "same for", "for the", "for year", "for segment"
    ]

    if any(q.startswith(p) for p in followup_starts):
        return True

    # very short questions are usually follow-ups
    # e.g. "for corporate", "for 2024", "only uk"
    word_count = len(q.split())
    if word_count <= 4:
        return True

    return False

def rewrite_followup_question(user_query: str, last_context: dict) -> str:
    """
    Converts follow-up into a context-rich question.
    """
    if not last_context:
        return user_query

    last_question = last_context.get("question", "")
    last_sql = last_context.get("sql", "")
    last_columns = last_context.get("columns", [])

    # Make it explicit to LLM that this is a refinement
    rewritten = f"""
This is a follow-up refinement to the previous successful query.

Previous question:
{last_question}

Previous SQL:
{last_sql}

Previous result columns:
{last_columns}

Follow-up user request:
{user_query}

Task:
- Treat the follow-up as a refinement/filter of the previous question.
- If user mentions a category value (example: Corporate), apply it as a filter on the most likely dimension (example: Segment).
- If user mentions a year/time, apply it as a filter on the most likely date/year field.
- If follow-up is ambiguous, make the smallest reasonable refinement rather than changing the metric.
"""
    return rewritten.strip()

# -------------------------------------------------
# Insight logic (CACHED DATA ONLY)
# -------------------------------------------------

def generate_insights_from_data(rows):
    if not rows:
        return "No data available to generate insights."

    limited_rows = rows[:10]

    insight_prompt = f"""
You are a senior business data analyst.

Analyze the query result data below and generate business insights.

Instructions:
- Do NOT repeat raw rows
- Use simple, decision-oriented business language

Provide:
1. Executive Summary (2–3 sentences)
2. Key Insights (bullet points)
3. Business Implication or Recommendation

Data:
{limited_rows}
"""
    response = llm_global.invoke(insight_prompt)
    return normalize_response_content(getattr(response, "content", ""))

# -------------------------------------------------
# Chart logic (PYTHON GENERATED IMAGE – FAST)
# -------------------------------------------------

def generate_chart_image(rows):
    if not rows:
        return None

    columns = list(rows[0].keys())

    # detect numeric / categorical
    numeric_cols = [c for c in columns if isinstance(rows[0].get(c), Number)]
    categorical_cols = [c for c in columns if c not in numeric_cols]

    if not numeric_cols or not categorical_cols:
        return None

    x_col = categorical_cols[0]
    y_col = numeric_cols[0]

    x_vals = [str(r.get(x_col)) for r in rows]
    y_vals = [float(r.get(y_col)) for r in rows]

    plt.figure(figsize=(9, 4.5))
    plt.bar(x_vals, y_vals, color="#17a2b8")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} by {x_col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("html_form.html")

@app.route("/ping", methods=["GET"])
def ping():
    try:
        llm_global.invoke("Ping")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.json or {}
        user_query = payload.get("query")
        action = payload.get("action", "data")

        # -------------------------------------------------
        # ✅ INSIGHT MODE
        # -------------------------------------------------
        if action == "insight":
            rows = session.get("last_result")
            if not rows:
                return jsonify({"insights": "No data available for insights."})
            insights = generate_insights_from_data(rows)
            return jsonify({"insights": insights})

        # -------------------------------------------------
        # ✅ CHART MODE
        # -------------------------------------------------
        if action == "chart":
            rows = session.get("last_result")
            if not rows:
                return jsonify({"chart_image": None})
            img_b64 = generate_chart_image(rows)
            return jsonify({"chart_image": img_b64})

        # -------------------------------------------------
        # ✅ DATA MODE – NEW USER QUESTION (with follow-up support)
        # -------------------------------------------------

        if not user_query:
            return jsonify({"error": "Query missing"}), 400

        # 🔥 Reset only “output caches” for a new data query
        # DO NOT reset conversational context keys.
        session.pop("last_result", None)
        session.pop("last_sql", None)

        schema_info = get_dataset_schema("my-ai-sql-project", "sales")

        # last successful context (for follow-ups)
        last_context = session.get("last_context")  # dict: {question, sql, columns}

        # if user asks follow-up, rewrite query to make intent explicit
        is_follow = detect_followup(user_query) and bool(last_context)
        question_for_llm = rewrite_followup_question(user_query, last_context) if is_follow else user_query

        # Keep a small history for LLM (optional but helps)
        # Store only user questions (not raw assistant SQL) to avoid prompt pollution
        if "chat_history" not in session:
            session["chat_history"] = []
        history = session["chat_history"]
        history.append(user_query)
        # keep last 6 user turns only
        history = history[-6:]
        session["chat_history"] = history
        history_text = "\n".join([f"- {h}" for h in history])

        # ✅ Prompt: explicit instruction to use previous successful context
        previous_context_text = ""
        if last_context:
            previous_context_text = f"""
Previous successful query context (use ONLY if current question is a follow-up):
- Previous question: {last_context.get("question", "")}
- Previous SQL: {last_context.get("sql", "")}
- Previous result columns: {last_context.get("columns", [])}
"""

        sql_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""
You are a helpful assistant that generates BigQuery SQL.

Strict rules:
- Use ONLY the schema provided
- Do NOT hallucinate tables or columns
- Output ONLY SQL (no explanations)
- Generate syntactically correct BigQuery SQL
- SQL must be read-only: ONLY SELECT/WITH queries

Follow-up handling:
- If the user question is a follow-up refinement (examples: "for corporate", "only 2024", "same but for UK"),
  use the Previous successful query context to refine/filter the previous intent.
- Do NOT reuse old results. Always generate a fresh SQL query.
- Preserve the same metric unless the user explicitly changes it.

Schema:
{schema_info}

Recent user questions:
{history_text}

{previous_context_text}
"""
            ),
            ("user", "{question}")
        ])

        chain = sql_prompt | llm_global
        response = chain.invoke({"question": question_for_llm})

        raw_output = normalize_response_content(getattr(response, "content", ""))
        sql_query = clean_sql(raw_output)

        # Basic safety check
        if not is_safe_select_sql(sql_query):
            # Do not overwrite last_context on failure
            return jsonify({
                "sql": "",
                "results": [],
                "error": "Unable to generate a safe SELECT query for this question."
            })

        # Execute SQL
        query_job = bq_client.query(sql_query)
        rows = [dict(row) for row in query_job.result()]

        # ✅ Cache ONLY if rows exist (prevents state poisoning)
        if rows and len(rows) > 0:
            session["last_result"] = rows
            session["last_sql"] = sql_query

            # Save minimal intent context for follow-ups
            session["last_context"] = {
                "question": user_query,
                "sql": sql_query,
                "columns": list(rows[0].keys()) if rows else []
            }
        else:
            # No rows — keep last_context unchanged so follow-ups still refer to last successful query
            session.pop("last_result", None)
            session.pop("last_sql", None)

        return jsonify({
            "sql": sql_query,
            "results": rows
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# App runner
# -------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
