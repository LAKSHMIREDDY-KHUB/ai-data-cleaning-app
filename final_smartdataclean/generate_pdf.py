# generate_pdf.py

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
import datetime


def generate_pdf_report(
    df,
    cleaned_df,
    before_score,
    after_score,
    strategy_scores,
    best_strategy,
    feature_importance_dict,
    cleaning_summary,
    ai_text
):

    file_path = "SmartPrep_Report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # ==================================================
    # TITLE
    # ==================================================
    elements.append(Paragraph("SmartPrep AI - Data Cleaning Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        f"Generated On: {datetime.datetime.now()}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.4 * inch))


    # ==================================================
    # 1️⃣ DATASET METADATA
    # ==================================================
    elements.append(Paragraph("1. Dataset Metadata", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    metadata_data = [
        ["Total Rows", df.shape[0]],
        ["Total Columns", df.shape[1]],
        ["Missing Values (Before)", df.isnull().sum().sum()],
        ["Missing Values (After)", cleaned_df.isnull().sum().sum()],
        ["Duplicate Rows", df.duplicated().sum()],
    ]

    metadata_table = Table(metadata_data)
    metadata_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)
    ]))

    elements.append(metadata_table)
    elements.append(Spacer(1, 0.4 * inch))


    # ==================================================
    # 2️⃣ COLUMN DATA TYPES
    # ==================================================
    elements.append(Paragraph("2. Column Data Types", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    dtype_data = [["Column Name", "Data Type"]]
    for col in df.columns:
        dtype_data.append([col, str(df[col].dtype)])

    dtype_table = Table(dtype_data, repeatRows=1)
    dtype_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)
    ]))

    elements.append(dtype_table)
    elements.append(Spacer(1, 0.4 * inch))


    # ==================================================
    # 3️⃣ HEALTH SCORE (TABLE)
    # ==================================================
    elements.append(Paragraph("3. Health Score Comparison", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    health_data = [
        ["Before Cleaning", round(before_score, 2)],
        ["After Cleaning", round(after_score, 2)],
        ["Improvement", round(after_score - before_score, 2)]
    ]

    health_table = Table(health_data)
    health_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)
    ]))

    elements.append(health_table)
    elements.append(Spacer(1, 0.3 * inch))


    # ==================================================
    # HEALTH SCORE CHART
    # ==================================================
    fig1 = plt.figure()
    plt.bar(["Before", "After"], [before_score, after_score])
    plt.title("Health Score Comparison")

    buffer1 = BytesIO()
    plt.savefig(buffer1, format="png")
    plt.close(fig1)
    buffer1.seek(0)

    elements.append(Image(buffer1, width=4*inch, height=3*inch))
    elements.append(Spacer(1, 0.4 * inch))


    # ==================================================
    # 4️⃣ STRATEGY COMPARISON
    # ==================================================
    elements.append(Paragraph("4. Cleaning Strategy Comparison", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    strategy_data = [["Strategy", "Model Score"]]
    for strategy, score in strategy_scores.items():
        strategy_data.append([strategy, round(score, 4)])

    strategy_data.append(["Best Strategy", best_strategy])

    strategy_table = Table(strategy_data, repeatRows=1)
    strategy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)
    ]))

    elements.append(strategy_table)
    elements.append(Spacer(1, 0.3 * inch))


    # STRATEGY CHART
    if strategy_scores:

        fig2 = plt.figure()
        plt.bar(strategy_scores.keys(), strategy_scores.values())
        plt.title("Strategy Performance Comparison")
        plt.xticks(rotation=30)

        buffer2 = BytesIO()
        plt.savefig(buffer2, format="png")
        plt.close(fig2)
        buffer2.seek(0)

        elements.append(Image(buffer2, width=4*inch, height=3*inch))
        elements.append(Spacer(1, 0.4 * inch))


    # ==================================================
    # 5️⃣ FEATURE IMPORTANCE
    # ==================================================
    if feature_importance_dict:

        elements.append(Paragraph("5. Feature Importance", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        features = list(feature_importance_dict.keys())
        importances = list(feature_importance_dict.values())

        fig3 = plt.figure()
        plt.barh(features, importances)
        plt.title("Feature Importance")

        buffer3 = BytesIO()
        plt.savefig(buffer3, format="png")
        plt.close(fig3)
        buffer3.seek(0)

        elements.append(Image(buffer3, width=4*inch, height=3*inch))
        elements.append(Spacer(1, 0.4 * inch))


    # ==================================================
    # 6️⃣ CLEANING SUMMARY
    # ==================================================
    elements.append(Paragraph("6. Cleaning Summary", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(cleaning_summary, styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))


    # ==================================================
    # 7️⃣ AI RECOMMENDATIONS
    # ==================================================
    elements.append(Paragraph("7. AI Recommendations", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if ai_text:
        for line in ai_text.split("\n"):
            elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
    else:
        elements.append(Paragraph("No AI recommendations available.", styles["Normal"]))


    doc.build(elements)

    return file_path