# HELM-GPT MCP Quick Start Guide for Claude Code

## Installation & Setup

### 1. Prerequisites
- Conda or Mamba package manager
- Claude Code CLI installed
- Git (for cloning the repository)

### 2. Quick Installation
```bash
# Clone the repository
git clone <repository-url>
cd helm_gpt_mcp

# Install dependencies
mamba env create -f environment.yml
# OR manually create environment
mamba create -p ./env python=3.10 fastmcp loguru pandas numpy scikit-learn

# Register with Claude Code
claude mcp add cycpep-tools -- "$(pwd)/env/bin/python" "$(pwd)/src/server.py"

# Verify installation
claude mcp list
```

### 3. Test Installation
```bash
# Run integration tests
python tests/run_integration_tests.py

# Should output: "🎉 All tests passed! MCP server is ready for use."
```

## Available Tools

### 🧪 Quick Analysis Tools (< 30 seconds)
- **`helm_to_smiles`** - Convert HELM notation to SMILES
- **`predict_permeability`** - Predict membrane permeability
- **`predict_kras_binding`** - Predict KRAS protein binding
- **`validate_helm_notation`** - Validate HELM syntax

### ⏳ Batch Processing Tools (2-15 minutes)
- **`submit_helm_to_smiles_batch`** - Batch HELM conversion
- **`submit_permeability_batch`** - Batch permeability prediction
- **`submit_kras_binding_batch`** - Batch KRAS binding prediction

### 📊 Job Management Tools
- **`get_job_status`** - Check job progress
- **`get_job_result`** - Get completed results
- **`list_jobs`** - List all jobs
- **`cancel_job`** - Cancel running jobs

### ℹ️ Information Tools
- **`get_server_info`** - Server information
- **`get_model_info`** - ML model details

## Quick Test Prompts

### Test 1: Basic Tool Discovery
```
What MCP tools do you have available for cyclic peptides?
```

### Test 2: Simple HELM Conversion
```
Convert this HELM notation to SMILES: PEPTIDE1{G.R.G.D.S.P}$$$$
```

### Test 3: Permeability Prediction
```
Predict the membrane permeability for cyclic peptide: PEPTIDE1{G.R.G.D.S.P}$$$$
```

### Test 4: Full Drug Discovery Workflow
```
Analyze this cyclic peptide for drug potential: PEPTIDE1{G.R.G.D.S.P}$$$$

Please:
1. Convert to SMILES
2. Predict membrane permeability
3. Predict KRAS binding affinity
4. Summarize drug-likeness
```

### Test 5: Batch Processing
```
Run a batch analysis on these cyclic peptides:
- PEPTIDE1{G.R.G.D.S.P}$$$$ (RGD peptide)
- PEPTIDE1{Y.I.G.S.R}$$$$ (laminin peptide)
- PEPTIDE1{R.G.D.F.V}$$$$ (modified RGD)

For each, predict both permeability and KRAS binding, then rank by drug potential.
```

## Expected Response Format

### Successful Tool Response
```json
{
  "status": "success",
  "data": {
    "input": "PEPTIDE1{G.R.G.D.S.P}$$$$",
    "smiles": "CC(C)[C@H](N)C(=O)N[C@@H](C)C(=O)N...",
    "success_count": 1,
    "total_count": 1
  },
  "execution_time": "0.8s",
  "message": "HELM to SMILES conversion completed successfully"
}
```

### Job Submission Response
```json
{
  "status": "submitted",
  "job_id": "job_20250101_120000_abc123",
  "estimated_completion": "2-5 minutes",
  "message": "Job submitted successfully. Use get_job_status('job_id') to check progress."
}
```

## Common Use Cases

### 🧬 Drug Discovery Pipeline
1. **Input**: Cyclic peptide sequences or HELM notation
2. **Convert**: HELM → SMILES for computational analysis
3. **Predict**: Membrane permeability (oral bioavailability)
4. **Predict**: Target binding affinity (KRAS protein)
5. **Analyze**: Drug-likeness assessment and optimization suggestions

### 🔬 Virtual Screening
1. **Input**: Large library of cyclic peptides
2. **Batch Process**: Submit batch jobs for parallel processing
3. **Filter**: By permeability and binding criteria
4. **Rank**: Top candidates for experimental validation

### 📊 Comparative Analysis
1. **Input**: Multiple peptide variants
2. **Compare**: Properties across all variants
3. **Optimize**: Identify best-performing sequences
4. **Report**: Detailed comparison tables

## Performance Expectations

| Operation Type | Processing Speed | Typical Response Time |
|---------------|------------------|----------------------|
| Single HELM conversion | ~33 sequences/second | < 30 seconds |
| Single permeability prediction | ~10 sequences/second | < 30 seconds |
| Single KRAS binding prediction | ~10 sequences/second | < 30 seconds |
| Batch processing (10-100 sequences) | Background job | 2-15 minutes |
| Job status check | Instant | < 5 seconds |

## Troubleshooting

### Server Not Found
```bash
# Re-register the server
claude mcp remove cycpep-tools  # if exists
claude mcp add cycpep-tools -- "$(pwd)/env/bin/python" "$(pwd)/src/server.py"
claude mcp list  # verify registration
```

### Tool Execution Errors
```bash
# Check server logs
tail -f logs/mcp_server.log

# Test server manually
mamba run -p ./env python tests/test_server_start.py
```

### Job Failures
```
Use the get_job_log tool to view error details:
"Show me the logs for job [job_id]"

Common issues:
- Invalid HELM notation: Check syntax
- Missing model files: Verify models directory
- Resource limits: Large batches may need splitting
```

### Environment Issues
```bash
# Rebuild environment
mamba env remove -p ./env
mamba create -p ./env python=3.10 fastmcp loguru pandas numpy scikit-learn

# Reinstall server
pip install -e .
```

## Support & Documentation

- **Full Documentation**: See `reports/step6_mcp_tools.md`
- **Test Prompts**: See `tests/test_prompts.md`
- **Integration Report**: See `reports/step7_integration.md`
- **Server Details**: See `README.md`

## Quick Commands

```bash
# Check server status
claude mcp list

# Test server manually
python tests/test_server_start.py

# Run full integration tests
python tests/run_integration_tests.py

# Start development server
mamba run -p ./env fastmcp dev src/server.py

# View job directory
ls -la jobs/

# View server logs
tail -f logs/mcp_server.log
```

---

**Ready to start analyzing cyclic peptides with Claude Code!** 🚀

Simply start Claude Code and begin using the test prompts above. The system will guide you through each step of the computational analysis process.