"""Fixtures partagées par la suite de tests."""

from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Génère un PDF de 2 pages avec du texte connu."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.drawString(72, 750, "Page un : introduction au systeme RAG.")
    pdf.showPage()
    pdf.drawString(72, 750, "Page deux : la recherche semantique vectorielle.")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
