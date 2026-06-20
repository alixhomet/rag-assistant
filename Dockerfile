# ──────────────────────────────────────────────────────────────
# Étape 1 : builder — installe les dépendances dans un venv isolé
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Bonnes pratiques Python en conteneur
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# On installe les dépendances AVANT de copier le code :
# ce layer est mis en cache tant que requirements.txt ne change pas.
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Étape 2 : runtime — image finale, légère, sans outils de build
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Utilisateur non-root : on n'exécute jamais une app en root dans un conteneur
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# On récupère uniquement le venv déjà construit (pas les outils de compilation)
COPY --from=builder /opt/venv /opt/venv

# Puis le code applicatif
COPY src/ ./src/
COPY ui/ ./ui/

# Dossier de données persistant, possédé par l'utilisateur non-root
RUN mkdir -p /app/data/uploads /app/data/chroma && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Vérification de santé : Streamlit expose un endpoint /_stcore/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Démarrage : on écoute sur 0.0.0.0 pour être joignable hors du conteneur
CMD ["streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
