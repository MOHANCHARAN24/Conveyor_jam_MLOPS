import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "conveyor_jam_mlop_report.pdf"
CI_EVIDENCE_PATH = ROOT / "reports" / "ci_evidence.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _table(rows: list[list], widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B56")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F6F9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(18 * mm, 12 * mm, "Conveyor Jam Prediction | Synthetic-data demonstration")
    canvas.drawRightString(192 * mm, 12 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test_run.returncode != 0:
        raise RuntimeError(f"Automated tests failed:\n{test_run.stdout}\n{test_run.stderr}")

    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    matrices = json.loads((ROOT / "reports" / "confusion_matrices.json").read_text())
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    repository_url = remote.stdout.strip() if remote.returncode == 0 else ""
    ci_evidence = (
        json.loads(CI_EVIDENCE_PATH.read_text(encoding="utf-8"))
        if CI_EVIDENCE_PATH.exists()
        else None
    )
    versions = [
        json.loads(
            (ROOT / "data" / "versions" / "conveyor" / f"v{index:03d}" / "metadata.json").read_text()
        )
        for index in (1, 2)
    ]
    frames = [
        pd.read_csv(ROOT / "data" / "versions" / "conveyor" / f"v{index:03d}" / version["file"])
        for index, version in enumerate(versions, start=1)
    ]
    frame_stats = [
        {
            "rows": len(frame),
            "throughput": frame["parcels_per_minute"].mean(),
            "jam_rate": frame["next_shift_jam"].mean(),
        }
        for frame in frames
    ]

    from api import main as api_main

    api_main.load_model()
    sample = {
        "features": {
            "motor_current_a": 18.0,
            "rated_motor_current_a": 20.0,
            "vibration_mm_s": 7.5,
            "belt_age_years": 8.0,
            "parcels_per_minute": 110.0,
            "dust_accumulation_index": 72.0,
            "days_since_maintenance": 95.0,
            "humidity_pct": 68.0,
        }
    }
    api_response = TestClient(api_main.app).post("/predict", json=sample)
    if api_response.status_code != 200:
        raise RuntimeError(f"API report example failed: {api_response.text}")
    prediction = api_response.json()
    (ROOT / "reports" / "api_prediction.json").write_text(
        json.dumps({"request": sample, "response": prediction}, indent=2),
        encoding="utf-8",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCentered",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183B56"),
            fontSize=22,
            leading=27,
            spaceAfter=12,
        )
    )
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11))
    story = [
        Spacer(1, 18 * mm),
        Paragraph("Predicting Conveyor Belt Jamming", styles["TitleCentered"]),
        Paragraph("MLOps Implementation Report", styles["Heading2"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Automated Parcel Sorting Centre | Generated "
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            styles["BodyText"],
        ),
        Spacer(1, 10 * mm),
        Paragraph("1. Problem and prediction target", styles["Heading1"]),
        Paragraph(
            "Predict whether a conveyor belt is likely to jam during its next operating "
            "shift using current motor, vibration, throughput, dust, belt-age, maintenance, "
            "and humidity measurements. The target is <b>next_shift_jam</b> (1 = jam, "
            "0 = normal operation). These synthetic records support an end-to-end "
            "demonstration only and are not evidence of production performance.",
            styles["BodyText"],
        ),
        Spacer(1, 4 * mm),
        Paragraph("2. Workflow architecture", styles["Heading1"]),
        Preformatted(
            "Synthetic sensor + maintenance records\n"
            "  -> Dataset V1 and festival V2 snapshots (SHA-256)\n"
            "  -> Sensor validation and deterministic feature engineering\n"
            "  -> Reusable engineered feature store + stratified holdout split\n"
            "  -> Logistic Regression / Random Forest -> metrics + confusion matrices\n"
            "  -> MLflow experiment + selected registered model\n"
            "  -> FastAPI /health and /predict <- GitHub Actions pytest checks",
            styles["Code"],
        ),
        Paragraph("3. Dataset versions", styles["Heading1"]),
        _table(
            [["Version", "Rows", "Mean parcels/min", "Jam rate", "SHA-256 prefix"]]
            + [
                [
                    f"V{index}",
                    str(stats["rows"]),
                    f'{stats["throughput"]:.2f}',
                    f'{stats["jam_rate"]:.1%}',
                    version["sha256"][:16] + "...",
                ]
                for index, (version, stats) in enumerate(zip(versions, frame_stats), start=1)
            ],
            [23 * mm, 17 * mm, 34 * mm, 25 * mm, 55 * mm],
        ),
        Paragraph(
            "V1 contains baseline shifts. V2 uses the same paired synthetic shifts with "
            "1.30x parcels/minute, increased motor load, higher vibration, dust, and humidity, "
            "and a three-day maintenance delay. Version metadata files retain full checksums "
            "and generation notes.",
            styles["Small"],
        ),
        PageBreak(),
        Paragraph("4. Preprocessing and feature store", styles["Heading1"]),
        Paragraph(
            "Sensor columns are checked for presence, finite numeric values, allowed ranges, "
            "and valid binary labels. Numeric values are median-imputed and standardized "
            "inside the training pipeline. The split is stratified (80% training / 20% test, "
            "random_state=42). The file features/conveyor_feature_store.csv stores all "
            "versioned records and engineered values for reuse; train/test transformed "
            "matrices are also generated.",
            styles["BodyText"],
        ),
        Spacer(1, 3 * mm),
        _table(
            [
                ["Feature", "Definition"],
                ["motor_load_ratio", "motor_current_a / rated_motor_current_a"],
                [
                    "belt_wear_pressure_index",
                    "(belt_age_years / 10) x (vibration_mm_s / 10) x (days_since_maintenance / 180)",
                ],
            ],
            [55 * mm, 115 * mm],
        ),
        Paragraph("5. Model comparison (positive class: Jam Risk)", styles["Heading1"]),
        _table(
            [["Model", "Accuracy", "Precision", "Recall", "F1"]]
            + [
                [
                    row["model"],
                    f'{row["accuracy"]:.3f}',
                    f'{row["precision"]:.3f}',
                    f'{row["recall"]:.3f}',
                    f'{row["f1"]:.3f}',
                ]
                for _, row in comparison.iterrows()
            ],
            [48 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm],
        ),
        Paragraph(
            f'Selection criterion: maximum held-out Jam Risk F1. Selected: '
            f'<b>{metadata["model_name"]}</b>. These scores are synthetic-data split '
            "metrics and should not be interpreted as expected live-site performance.",
            styles["BodyText"],
        ),
        Spacer(1, 3 * mm),
        _table(
            [["Model", "Confusion matrix (rows=true, columns=predicted; [Normal, Jam])"]]
            + [
                [name, json.dumps(value["matrix"])]
                for name, value in matrices.items()
            ],
            [48 * mm, 122 * mm],
        ),
        Paragraph("6. MLflow registry evidence", styles["Heading1"]),
        Paragraph(
            f'Experiment: <b>conveyor-jam-prediction</b><br/>'
            f'Registered name: <b>{metadata["registered_model_name"]}</b><br/>'
            f'Registered version: <b>{metadata["registered_model_version"]}</b><br/>'
            f'Selected run metrics: {json.dumps(metadata["metrics"])}<br/>'
            "Tracking backend: local SQLite (mlflow.db); use MLflow UI for run-level evidence.",
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph("7. CI/CD execution evidence", styles["Heading1"]),
        Paragraph(
            "The committed GitHub Actions workflow installs project dependencies and runs "
            "pytest on pushes and pull requests to main/master. Local command executed while "
            "building this report: <b>python -m pytest -q</b>.",
            styles["BodyText"],
        ),
        Preformatted(test_run.stdout.strip() or "pytest completed successfully.", styles["Code"]),
        Paragraph(
            (
                f'Hosted GitHub Actions: <b>{ci_evidence["status"]}</b>; '
                f'run {ci_evidence["run_id"]}, commit {ci_evidence["commit"]}. '
                f'<link href="{ci_evidence["url"]}" color="blue">View run</link>'
                if ci_evidence
                else (
                    "Hosted GitHub Actions status is not recorded in this report yet. "
                    "See the repository Actions tab for the current run."
                )
            ),
            styles["BodyText"],
        ),
        Paragraph("8. FastAPI serving and prediction result", styles["Heading1"]),
        Paragraph(
            "GET /health reports service/model readiness. POST /predict validates a complete "
            "sensor payload and returns Jam Risk or Normal Operation with the predicted jam "
            "probability and selected model.",
            styles["BodyText"],
        ),
        Preformatted(
            json.dumps({"request": sample, "response": prediction}, indent=2),
            styles["Code"],
        ),
        Paragraph("9. Conclusion and limitations", styles["Heading1"]),
        Paragraph(
            "The repository provides versioned synthetic datasets, reproducible preprocessing "
            "and engineered features, two candidate classifiers, held-out comparisons, MLflow "
            "registration, a validated API, and automated tests. Deployment requires real "
            "labeled shifts, sensor calibration, maintenance-system integration, operational "
            "threshold review, monitoring, and an actual hosted CI run.",
            styles["BodyText"],
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Repository: "
            + (
                f'<link href="{repository_url}" color="blue">{repository_url}</link>'
                if repository_url
                else "Git remote not configured"
            ),
            styles["Small"],
        ),
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="Predicting Conveyor Belt Jamming - MLOps Implementation Report",
        author="Universal MLOps Pipeline",
    )
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    print(f"Created {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    build_report()
