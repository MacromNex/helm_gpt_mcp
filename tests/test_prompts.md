# MCP Integration Test Prompts for Cyclic Peptide Tools

## Tool Discovery Tests

### Prompt 1: List All Tools
```
What MCP tools are available for cyclic peptides? Give me a brief description of each tool and what it does.
```

### Prompt 2: Tool Details
```
Explain how to use the helm_to_smiles tool, including all required and optional parameters.
```

### Prompt 3: Server Information
```
Show me information about the cycpep-tools MCP server, including version and capabilities.
```

## Sync Tool Tests

### Prompt 4: HELM to SMILES Conversion
```
Convert this HELM notation to SMILES: PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$
```

### Prompt 5: Permeability Prediction
```
Predict the membrane permeability for this HELM sequence: PEPTIDE1{G.R.G.D.S.P}$$$$
```

### Prompt 6: KRAS Binding Prediction
```
Predict the KRAS binding affinity for this HELM sequence: PEPTIDE1{G.R.G.D.S.P}$$$$
```

### Prompt 7: HELM Validation
```
Validate if this HELM notation is correct: PEPTIDE1{A.G.C.D.E.F.G}$$$$
```

### Prompt 8: Error Handling - Invalid HELM
```
Try to convert this invalid HELM notation to SMILES: invalid_helm_string_123
```

### Prompt 9: Error Handling - Empty Input
```
What happens if I try to predict permeability for an empty HELM sequence?
```

## Submit API Tests (Long-Running Operations)

### Prompt 10: Submit HELM to SMILES Batch
```
Submit a batch job to convert these HELM sequences to SMILES:
- PEPTIDE1{G.R.G.D.S.P}$$$$
- PEPTIDE1{R.G.D.F.V}$$$$
- PEPTIDE1{Y.I.G.S.R}$$$$
```

### Prompt 11: Submit Permeability Batch
```
Submit a batch job to predict membrane permeability for these HELM sequences:
- PEPTIDE1{G.R.G.D.S.P}$$$$
- PEPTIDE1{R.G.D.F.V}$$$$
- PEPTIDE1{Y.I.G.S.R}$$$$
```

### Prompt 12: Submit KRAS Binding Batch
```
Submit a batch job to predict KRAS binding for these HELM sequences:
- PEPTIDE1{G.R.G.D.S.P}$$$$
- PEPTIDE1{R.G.D.F.V}$$$$
- PEPTIDE1{Y.I.G.S.R}$$$$
```

### Prompt 13: Check Job Status
```
Check the status of job ID: [job_id_from_previous_submission]
```

### Prompt 14: Get Job Results
```
Get the results for completed job ID: [job_id_from_completed_job]
```

### Prompt 15: View Job Logs
```
Show me the last 20 lines of logs for job ID: [job_id]
```

### Prompt 16: List All Jobs
```
List all jobs and their current status.
```

### Prompt 17: Cancel Job
```
Cancel the running job with ID: [job_id]
```

### Prompt 18: Cleanup Completed Jobs
```
Clean up all completed jobs that are older than 1 day.
```

## Batch Processing Tests

### Prompt 19: Multi-Tool Batch Processing
```
For these cyclic peptides, I want to:
1. Convert HELM to SMILES
2. Predict membrane permeability
3. Predict KRAS binding affinity

HELM sequences:
- PEPTIDE1{G.R.G.D.S.P}$$$$
- PEPTIDE1{R.G.D.F.V}$$$$
- PEPTIDE1{K.L.D.L.K.L.D.L}$$$$

Process them all in batch and give me a summary when complete.
```

### Prompt 20: Large Batch Test
```
Submit a large batch job with 10 HELM sequences for permeability prediction:
- PEPTIDE1{A.A.A.A.A}$$$$
- PEPTIDE1{G.G.G.G.G}$$$$
- PEPTIDE1{L.L.L.L.L}$$$$
- PEPTIDE1{P.P.P.P.P}$$$$
- PEPTIDE1{F.F.F.F.F}$$$$
- PEPTIDE1{W.W.W.W.W}$$$$
- PEPTIDE1{Y.Y.Y.Y.Y}$$$$
- PEPTIDE1{H.H.H.H.H}$$$$
- PEPTIDE1{K.K.K.K.K}$$$$
- PEPTIDE1{R.R.R.R.R}$$$$
```

## End-to-End Workflow Scenarios

### Prompt 21: Full Drug Discovery Pipeline
```
I'm screening cyclic peptides for drug candidates. For this HELM sequence PEPTIDE1{G.R.G.D.S.P}$$$$:

1. First convert it to SMILES
2. Then predict its membrane permeability
3. Then predict its KRAS binding affinity
4. Finally, give me a summary of whether this would be a good drug candidate

Walk me through the complete workflow.
```

### Prompt 22: Comparative Analysis
```
I want to compare the drug-likeness of these three cyclic peptides:
- PEPTIDE1{G.R.G.D.S.P}$$$$ (RGD peptide)
- PEPTIDE1{Y.I.G.S.R}$$$$ (laminin peptide)
- PEPTIDE1{R.G.D.F.V}$$$$ (modified RGD)

For each one:
1. Convert to SMILES
2. Predict permeability (higher is better for oral drugs)
3. Predict KRAS binding (lower KD is better)
4. Rank them by overall drug potential

Give me a comparative table with all the results.
```

### Prompt 23: Optimization Workflow
```
I have a lead compound PEPTIDE1{G.R.G.D.S.P}$$$$. Help me analyze it and suggest next steps:

1. Get its current properties (SMILES, permeability, KRAS binding)
2. Assess if the permeability is sufficient (>-5 log units)
3. Assess if KRAS binding is promising (<100 μM KD)
4. Based on results, tell me what modifications might improve it
```

### Prompt 24: Batch Screening Workflow
```
I'm doing a virtual screen of a peptide library. Submit batch jobs for:

1. HELM to SMILES conversion for all these peptides:
   - PEPTIDE1{G.R.G.D.S.P}$$$$
   - PEPTIDE1{R.G.D.F.V}$$$$
   - PEPTIDE1{Y.I.G.S.R}$$$$
   - PEPTIDE1{K.L.D.L.K.L.D.L}$$$$
   - PEPTIDE1{P.L.G.F.A}$$$$

2. Once SMILES conversion is complete, predict permeability for all
3. For peptides with good permeability (>-5), predict KRAS binding
4. Generate a final ranked list of the top candidates

Coordinate the entire workflow and keep me updated on progress.
```

### Prompt 25: Troubleshooting Workflow
```
I submitted some jobs but I'm having issues. Help me troubleshoot:

1. List all my recent jobs
2. Check which ones failed and why
3. For any failed jobs, show me the error logs
4. Suggest how to fix the issues and resubmit
```

## Model Information and Validation Tests

### Prompt 26: Model Information
```
Tell me about the machine learning models used for permeability and KRAS binding prediction. What are their architectures, training data, and expected accuracy?
```

### Prompt 27: Validation Test
```
Run validation tests on known peptides to check model performance:

Test these with known properties:
- PEPTIDE1{G.R.G.D.S.P}$$$$ (known RGD peptide - should have moderate permeability)
- PEPTIDE1{A.A.A.A.A}$$$$ (simple peptide - should have poor permeability)

Do the predictions match expectations?
```

## Stress Tests

### Prompt 28: Concurrent Job Submission
```
Submit 5 different batch jobs simultaneously to test concurrent processing:
1. HELM to SMILES batch (5 peptides)
2. Permeability prediction batch (5 peptides)
3. KRAS binding batch (5 peptides)
4. Another HELM to SMILES batch (3 peptides)
5. Another permeability batch (4 peptides)

Monitor all jobs and report when each completes.
```

### Prompt 29: Resource Limits Test
```
Submit a very large batch job with 50 HELM sequences for SMILES conversion to test system limits. Monitor resource usage and completion time.
```

### Prompt 30: Error Recovery Test
```
Submit a batch job with a mix of valid and invalid HELM sequences:
- PEPTIDE1{G.R.G.D.S.P}$$$$ (valid)
- invalid_helm_123 (invalid)
- PEPTIDE1{Y.I.G.S.R}$$$$ (valid)
- another_invalid_helm (invalid)
- PEPTIDE1{R.G.D.F.V}$$$$ (valid)

Verify that:
1. Valid sequences are processed successfully
2. Invalid sequences are handled gracefully with error messages
3. The batch doesn't fail completely due to individual errors
4. Final results clearly indicate which sequences succeeded/failed
```

## Expected Response Formats

### Successful Tool Response
```json
{
  "status": "success",
  "data": {
    "input": "PEPTIDE1{G.R.G.D.S.P}$$$$",
    "output": "...",
    "additional_info": "..."
  },
  "execution_time": "0.5s",
  "message": "Operation completed successfully"
}
```

### Job Submission Response
```json
{
  "status": "submitted",
  "job_id": "abc123",
  "estimated_completion": "2-5 minutes",
  "message": "Job submitted successfully. Use get_job_status('abc123') to check progress."
}
```

### Job Status Response
```json
{
  "job_id": "abc123",
  "status": "running|completed|failed",
  "submitted_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:05Z",
  "completed_at": "2024-01-01T12:03:15Z",
  "progress": "75%",
  "message": "Processing 3 of 4 sequences"
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Invalid HELM notation: missing peptide definition",
  "input": "invalid_helm_123",
  "suggestion": "Please check HELM syntax. Valid format: PEPTIDE1{A.G.C}$$$$"
}
```

## Test Execution Instructions

1. Run these tests in order, starting with tool discovery
2. For submit API tests, wait for jobs to complete before proceeding
3. Record response times for sync tools (should be <30 seconds)
4. Record job completion times for batch operations
5. Verify all error messages are helpful and actionable
6. Check that invalid inputs don't crash the server
7. Confirm concurrent job submission works properly
8. Validate that job cleanup functions work correctly

## Success Criteria

- All tool discovery tests pass
- All sync tools respond within 30 seconds
- All batch operations complete successfully
- Job management functions work correctly
- Error handling provides clear, helpful messages
- Concurrent operations don't interfere with each other
- Server remains stable throughout all tests
- Resource usage stays within reasonable limits