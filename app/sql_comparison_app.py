"""
SQL Comparison Streamlit App

Displays ground truth questions and SQL results alongside user-input SQL for comparison.

Usage:
    streamlit run sql_comparison_app.py
"""

import streamlit as st
import sqlite3
import json
import pandas as pd
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="SQL Query Comparison",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for SQL display without horizontal scroll
st.markdown("""
<style>
    /* Make code blocks wrap text */
    .stCodeBlock code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* SQL display box styling */
    .sql-display {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Source Code Pro', monospace;
        font-size: 14px;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-x: hidden;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }

    .sql-display-user {
        background-color: #e8f4f8;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Source Code Pro', monospace;
        font-size: 14px;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-x: hidden;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Paths - Support both local and Docker environments
# In Docker: DATA_DIR=/app/data, In local: uses relative path from file
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent.parent / "data"))
DB_PATH = DATA_DIR / "bird_data" / "dev_databases" / "california_schools" / "california_schools.sqlite"
DEV_JSON_PATH = DATA_DIR / "bird_data" / "dev.json"


@st.cache_data
def load_questions(db_id: str = "california_schools") -> list[dict]:
    """Load questions for a specific database from dev.json."""
    with open(DEV_JSON_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    return [q for q in all_questions if q["db_id"] == db_id]


def execute_sql(sql: str, db_path: Path = DB_PATH) -> tuple[pd.DataFrame | None, str | None]:
    """
    Execute SQL query and return results as DataFrame.

    Returns:
        Tuple of (DataFrame or None, error message or None)
    """
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


def compare_results(df1: pd.DataFrame | None, df2: pd.DataFrame | None) -> dict:
    """Compare two DataFrames and return comparison metrics."""
    if df1 is None or df2 is None:
        return {"match": False, "reason": "One or both queries failed"}

    # Check if shapes match
    if df1.shape != df2.shape:
        return {
            "match": False,
            "reason": f"Shape mismatch: Ground Truth {df1.shape} vs Your Query {df2.shape}"
        }

    # Check if values match (allowing for column name differences)
    try:
        df1_values = df1.values.tolist()
        df2_values = df2.values.tolist()

        # Sort both for comparison (in case order differs)
        df1_sorted = sorted([tuple(row) for row in df1_values])
        df2_sorted = sorted([tuple(row) for row in df2_values])

        if df1_sorted == df2_sorted:
            return {"match": True, "reason": "Results match!"}
        else:
            return {"match": False, "reason": "Values differ between results"}
    except Exception as e:
        return {"match": False, "reason": f"Comparison error: {str(e)}"}


def display_result(df: pd.DataFrame | None, error: str | None, title: str):
    """Display query result or error."""
    if error:
        st.error(f"Error: {error}")
    elif df is not None:
        st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.warning("No results")


def main():
    st.title("🔍 SQL Query Comparison Tool")
    st.markdown("Compare your SQL queries against ground truth from the BIRD benchmark")

    # Load questions
    questions = load_questions("california_schools")

    if not questions:
        st.error("No questions found for california_schools database")
        return

    # Sidebar - Question selector
    st.sidebar.header("📋 Question Selection")

    # Create question options
    question_options = {
        f"Q{q['question_id']}: {q['question'][:50]}...": q['question_id']
        for q in questions
    }

    selected_label = st.sidebar.selectbox(
        "Select a question:",
        options=list(question_options.keys()),
        index=0
    )

    selected_id = question_options[selected_label]
    selected_question = next(q for q in questions if q["question_id"] == selected_id)

    # Filter by difficulty
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter by Difficulty")
    difficulties = list(set(q["difficulty"] for q in questions))
    selected_difficulty = st.sidebar.multiselect(
        "Difficulty levels:",
        difficulties,
        default=difficulties
    )

    filtered_questions = [q for q in questions if q["difficulty"] in selected_difficulty]
    st.sidebar.write(f"Showing {len(filtered_questions)} of {len(questions)} questions")

    # Main content - Tabs
    tab1, tab2 = st.tabs(["📊 Compare SQL", "📝 Question Browser"])

    with tab1:
        # Question display
        st.header("Question")
        st.info(selected_question["question"])

        if selected_question.get("evidence"):
            st.caption(f"💡 **Hint:** {selected_question['evidence']}")

        st.caption(f"**Difficulty:** {selected_question['difficulty']} | **Question ID:** {selected_question['question_id']}")

        st.markdown("---")

        # Two columns for comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Ground Truth")

            # Ground truth SQL with custom styling (no horizontal scroll)
            st.markdown("**SQL Query:**")
            st.markdown(
                f'<div class="sql-display">{selected_question["SQL"]}</div>',
                unsafe_allow_html=True
            )

            # Execute ground truth
            gt_df, gt_error = execute_sql(selected_question["SQL"])

            st.markdown("**Result:**")
            display_result(gt_df, gt_error, "Ground Truth")

        with col2:
            st.subheader("🔧 Your Query")

            # User SQL input
            st.markdown("**Enter SQL Query:**")
            user_sql = st.text_area(
                "Enter your SQL query:",
                value="",
                height=150,
                placeholder="SELECT * FROM ...",
                label_visibility="collapsed"
            )

            # Execute button
            execute_btn = st.button("▶️ Execute Query", type="primary", use_container_width=True)

            # Process execution
            if execute_btn and user_sql.strip():
                user_df, user_error = execute_sql(user_sql)

                # Store results in session state
                st.session_state['user_df'] = user_df
                st.session_state['user_error'] = user_error
                st.session_state['last_user_sql'] = user_sql

            # Display executed query for readability
            if 'last_user_sql' in st.session_state and st.session_state['last_user_sql']:
                st.markdown("**Executed SQL Query:**")
                st.markdown(
                    f'<div class="sql-display-user">{st.session_state["last_user_sql"]}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("**Result:**")

            # Display user results
            if 'user_df' in st.session_state or 'user_error' in st.session_state:
                display_result(
                    st.session_state.get('user_df'),
                    st.session_state.get('user_error'),
                    "Your Query"
                )
            else:
                st.info("Enter a SQL query and click Execute to see results")

        # Comparison section
        st.markdown("---")
        st.subheader("📈 Comparison Result")

        if 'user_df' in st.session_state and gt_df is not None:
            comparison = compare_results(gt_df, st.session_state.get('user_df'))

            if comparison["match"]:
                st.success(f"✅ {comparison['reason']}")
            else:
                st.error(f"❌ {comparison['reason']}")

            # Show side-by-side statistics
            stat_col1, stat_col2, stat_col3 = st.columns(3)

            with stat_col1:
                st.metric(
                    "Ground Truth Rows",
                    len(gt_df) if gt_df is not None else "Error"
                )

            with stat_col2:
                user_df = st.session_state.get('user_df')
                st.metric(
                    "Your Query Rows",
                    len(user_df) if user_df is not None else "Error"
                )

            with stat_col3:
                st.metric(
                    "Match Status",
                    "✅ Match" if comparison["match"] else "❌ Mismatch"
                )
        else:
            st.info("Execute your query to see comparison results")

    with tab2:
        st.header("Question Browser")
        st.markdown("Browse all questions for the California Schools database")

        # Display questions table
        questions_df = pd.DataFrame([
            {
                "ID": q["question_id"],
                "Question": q["question"],
                "Difficulty": q["difficulty"],
                "Has Evidence": "Yes" if q.get("evidence") else "No"
            }
            for q in filtered_questions
        ])

        st.dataframe(
            questions_df,
            use_container_width=True,
            height=400,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Difficulty": st.column_config.TextColumn("Difficulty", width="small"),
                "Has Evidence": st.column_config.TextColumn("Evidence", width="small")
            }
        )

        # Question detail viewer
        st.markdown("---")
        st.subheader("Question Detail")

        detail_id = st.number_input(
            "Enter Question ID to view details:",
            min_value=0,
            max_value=max(q["question_id"] for q in questions),
            value=0
        )

        detail_question = next((q for q in questions if q["question_id"] == detail_id), None)

        if detail_question:
            st.markdown(f"**Question:** {detail_question['question']}")
            if detail_question.get("evidence"):
                st.markdown(f"**Evidence:** {detail_question['evidence']}")
            st.markdown(f"**Difficulty:** {detail_question['difficulty']}")
            st.markdown("**Ground Truth SQL:**")
            st.markdown(
                f'<div class="sql-display">{detail_question["SQL"]}</div>',
                unsafe_allow_html=True
            )

            if st.button("Execute this SQL", key="browser_execute"):
                df, error = execute_sql(detail_question["SQL"])
                if error:
                    st.error(error)
                else:
                    st.dataframe(df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Database: California Schools | Source: BIRD Benchmark")


if __name__ == "__main__":
    main()
