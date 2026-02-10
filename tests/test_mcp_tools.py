#!/usr/bin/env python3
"""Test MCP server tools directly."""

import sys
from pathlib import Path

# Add src directory to path
TEST_DIR = Path(__file__).parent
MCP_ROOT = TEST_DIR.parent
sys.path.insert(0, str(MCP_ROOT / "src"))
sys.path.insert(0, str(MCP_ROOT / "scripts"))


def get_tool_fn(tool):
    """Extract the underlying function from a FastMCP FunctionTool."""
    if hasattr(tool, 'fn'):
        return tool.fn
    return tool


def test_helm_to_smiles_tool():
    """Test the helm_to_smiles MCP tool."""
    print("Testing helm_to_smiles tool...")
    try:
        from server import helm_to_smiles
        fn = get_tool_fn(helm_to_smiles)

        # Test with a simple cyclic peptide HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = fn(helm_input=test_helm)

        print(f"✅ helm_to_smiles: {result['status']}")

        if result['status'] == 'success':
            print(f"   Total conversions: {result.get('total_count', 0)}")
            print(f"   Successful: {result.get('success_count', 0)}")

            if result.get('results') and len(result['results']) > 0:
                conv = result['results'][0]
                print(f"   SMILES: {conv.get('smiles', 'N/A')}")
                print(f"   Success: {conv.get('success', False)}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ helm_to_smiles failed: {e}")
        return False

def test_predict_permeability_tool():
    """Test the predict_permeability MCP tool."""
    print("Testing predict_permeability tool...")
    try:
        from server import predict_permeability
        fn = get_tool_fn(predict_permeability)

        # Test with a simple cyclic peptide HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = fn(helm_input=test_helm)

        print(f"✅ predict_permeability: {result['status']}")

        if result['status'] == 'success':
            print(f"   Total predictions: {result.get('total_count', 0)}")
            print(f"   Successful: {result.get('success_count', 0)}")

            if result.get('results') and len(result['results']) > 0:
                pred = result['results'][0]
                print(f"   Permeability score: {pred.get('permeability_score', 'N/A')}")
                print(f"   Interpretation: {pred.get('interpretation', 'N/A')}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ predict_permeability failed: {e}")
        return False

def test_predict_kras_binding_tool():
    """Test the predict_kras_binding MCP tool."""
    print("Testing predict_kras_binding tool...")
    try:
        from server import predict_kras_binding
        fn = get_tool_fn(predict_kras_binding)

        # Test with a simple cyclic peptide HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = fn(helm_input=test_helm)

        print(f"✅ predict_kras_binding: {result['status']}")

        if result['status'] == 'success':
            print(f"   Total predictions: {result.get('total_count', 0)}")
            print(f"   Successful: {result.get('success_count', 0)}")

            if result.get('results') and len(result['results']) > 0:
                pred = result['results'][0]
                print(f"   Binding score: {pred.get('binding_score', 'N/A')}")
                print(f"   KD value (μM): {pred.get('kd_value_um', 'N/A')}")
                print(f"   Interpretation: {pred.get('interpretation', 'N/A')}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ predict_kras_binding failed: {e}")
        return False

def test_validate_helm_tool():
    """Test the validate_helm_notation MCP tool."""
    print("Testing validate_helm_notation tool...")
    try:
        from server import validate_helm_notation
        fn = get_tool_fn(validate_helm_notation)

        # Test with a simple cyclic peptide HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = fn(test_helm)

        print(f"✅ validate_helm_notation: {result['status']}")

        if result['status'] == 'success':
            print(f"   Valid: {result.get('valid')}")
            print(f"   HELM: {result.get('helm', 'N/A')}")
            print(f"   SMILES: {result.get('smiles', 'N/A')}")
            if result.get('error'):
                print(f"   Error: {result.get('error')}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ validate_helm_notation failed: {e}")
        return False

def test_server_info_tool():
    """Test the get_server_info MCP tool."""
    print("Testing get_server_info tool...")
    try:
        from server import get_server_info
        fn = get_tool_fn(get_server_info)

        result = fn()

        print(f"✅ get_server_info: {result['status']}")

        if result['status'] == 'success':
            print(f"   Server: {result.get('server_name')}")
            print(f"   Version: {result.get('version')}")
            print(f"   Sync tools: {len(result.get('sync_tools', []))}")
            print(f"   Submit tools: {len(result.get('submit_tools', []))}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ get_server_info failed: {e}")
        return False

def test_model_info_tool():
    """Test the get_model_info MCP tool."""
    print("Testing get_model_info tool...")
    try:
        from server import get_model_info
        fn = get_tool_fn(get_model_info)

        result = fn()

        print(f"✅ get_model_info: {result['status']}")

        if result['status'] == 'success':
            models = result.get('models', {})
            for model_name, model_data in models.items():
                status = '✅' if model_data['available'] else '❌'
                print(f"   {model_name}: {status} ({model_data['size_mb']:.1f} MB)")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ get_model_info failed: {e}")
        return False

def test_job_management_tools():
    """Test the job management MCP tools."""
    print("Testing job management tools...")
    try:
        from server import list_jobs, cleanup_completed_jobs
        list_jobs_fn = get_tool_fn(list_jobs)
        cleanup_fn = get_tool_fn(cleanup_completed_jobs)

        # Test list_jobs
        result = list_jobs_fn()
        print(f"✅ list_jobs: {result['status']}")
        print(f"   Total jobs: {result.get('total', 0)}")

        # Test cleanup
        cleanup_result = cleanup_fn(keep_days=1)
        print(f"✅ cleanup_completed_jobs: {cleanup_result['status']}")

        return result['status'] == 'success' and cleanup_result['status'] == 'success'
    except Exception as e:
        print(f"❌ job management tools failed: {e}")
        return False

def test_batch_csv_processing():
    """Test processing a CSV file."""
    print("Testing CSV file processing...")
    try:
        from server import helm_to_smiles
        fn = get_tool_fn(helm_to_smiles)

        # Use the example data file
        test_file = MCP_ROOT / "examples" / "data" / "sequences" / "CycPeptMPDB_Peptide_All.csv"

        if not test_file.exists():
            print(f"⚠️  Test file not found: {test_file}")
            return True  # Skip this test but don't fail

        result = fn(input_file=str(test_file), limit=5)

        print(f"✅ CSV processing: {result['status']}")

        if result['status'] == 'success':
            print(f"   Processed: {result.get('total_count', 0)} sequences")
            print(f"   Successful: {result.get('success_count', 0)}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ CSV processing failed: {e}")
        return False

def main():
    """Run all MCP tool tests."""
    print("🧪 Testing MCP Server Tools\n")

    tests = [
        ("HELM to SMILES", test_helm_to_smiles_tool),
        ("Permeability Prediction", test_predict_permeability_tool),
        ("KRAS Binding Prediction", test_predict_kras_binding_tool),
        ("HELM Validation", test_validate_helm_tool),
        ("Server Info", test_server_info_tool),
        ("Model Info", test_model_info_tool),
        ("Job Management", test_job_management_tools),
        ("CSV Processing", test_batch_csv_processing)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print('='*60)
        try:
            if test_func():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ CRASHED: {e}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print('='*60)

    if passed == total:
        print("🎉 All MCP tools working correctly!")
        return 0
    else:
        print("⚠️  Some tools need attention")
        return 1

if __name__ == "__main__":
    exit(main())