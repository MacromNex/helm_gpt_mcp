FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/macronex/helm_gpt_mcp"
LABEL org.opencontainers.image.description="GPT-based de novo macrocyclic peptide design with HELM support"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# Core dependencies
RUN pip install --no-cache-dir \
    fastmcp loguru click pandas numpy tqdm \
    scikit-learn scipy

# Install RDKit
RUN conda install -y -c conda-forge rdkit && conda clean -afy

# Clone HELM-GPT repo
RUN git clone https://github.com/macronex/helm_gpt_mcp.git /tmp/helm_gpt_mcp || true

# Copy MCP server source
COPY src/ src/

CMD ["python", "src/server.py"]
