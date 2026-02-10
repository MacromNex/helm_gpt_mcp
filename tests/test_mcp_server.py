#!/usr/bin/env python3
"""Test the MCP server functionality."""

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


def test_server_import():
    """Test that the server can be imported without errors."""
    try:
        from server import mcp
        print("✅ Server imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import server: {e}")
        return False

def test_server_info():
    """Test the get_server_info tool."""
    try:
        from server import get_server_info
        fn = get_tool_fn(get_server_info)
        result = fn()
        print(f"✅ get_server_info: {result['status']}")
        print(f"   Server: {result.get('server_name')}")
        print(f"   Sync tools: {len(result.get('sync_tools', []))}")
        return True
    except Exception as e:
        print(f"❌ get_server_info failed: {e}")
        return False

def test_model_info():
    """Test the get_model_info tool."""
    try:
        from server import get_model_info
        fn = get_tool_fn(get_model_info)
        result = fn()
        print(f"✅ get_model_info: {result['status']}")
        for model_name, model_data in result.get('models', {}).items():
            print(f"   {model_name}: {'✅' if model_data['available'] else '❌'} ({model_data['size_mb']:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ get_model_info failed: {e}")
        return False

def test_helm_validation():
    """Test HELM validation."""
    try:
        from server import validate_helm_notation
        fn = get_tool_fn(validate_helm_notation)

        # Test with a simple cyclic peptide HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = fn(test_helm)
        print(f"✅ validate_helm_notation: {result['status']}")
        if result['status'] == 'success':
            print(f"   Valid: {result.get('valid')}")
        else:
            print(f"   Error: {result.get('error')}")
        return True
    except Exception as e:
        print(f"❌ validate_helm_notation failed: {e}")
        return False

def test_job_manager():
    """Test job management functions."""
    try:
        from server import list_jobs
        fn = get_tool_fn(list_jobs)
        result = fn()
        print(f"✅ list_jobs: {result['status']}")
        print(f"   Total jobs: {result.get('total', 0)}")
        return True
    except Exception as e:
        print(f"❌ list_jobs failed: {e}")
        return False

def test_single_helm_conversion():
    """Test single HELM to SMILES conversion."""
    try:
        from server import helm_to_smiles
        fn = get_tool_fn(helm_to_smiles)

        # Test with a simple cyclic peptide HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = fn(helm_input=test_helm)
        print(f"✅ helm_to_smiles (single): {result['status']}")

        if result['status'] == 'success':
            conversions = result.get('conversions', [])
            if conversions:
                print(f"   Converted {len(conversions)} sequence(s)")
                print(f"   SMILES: {conversions[0].get('smiles', 'N/A')}")
            else:
                print("   No conversions returned")
        else:
            print(f"   Error: {result.get('error')}")
        return True
    except Exception as e:
        print(f"❌ helm_to_smiles failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing MCP Server for Cyclic Peptide Tools\n")

    tests = [
        test_server_import,
        test_server_info,
        test_model_info,
        test_helm_validation,
        test_job_manager,
        test_single_helm_conversion
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test.__name__}")
        print('='*50)
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print('='*50)

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())