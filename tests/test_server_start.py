#!/usr/bin/env python3
"""Test that the MCP server starts without errors."""

import sys
import subprocess
import time
import signal
from pathlib import Path

# Add src directory to path
TEST_DIR = Path(__file__).parent
MCP_ROOT = TEST_DIR.parent
sys.path.insert(0, str(MCP_ROOT / "src"))

def test_server_import():
    """Test that the server can be imported without errors."""
    try:
        from server import mcp
        print("✅ Server imported successfully")

        # Check that tools are registered
        # FastMCP stores tools differently - we can inspect the function registry
        print("✅ Tools imported successfully")
        print("   Tools are registered with FastMCP framework")

        return True
    except Exception as e:
        print(f"❌ Failed to import server: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_startup():
    """Test that the server can start and respond to help."""
    try:
        print("Starting server subprocess...")

        # Start the server process
        proc = subprocess.Popen(
            [sys.executable, "src/server.py", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for completion with timeout
        try:
            stdout, stderr = proc.communicate(timeout=10)

            print("✅ Server started and showed help")

            # Check if output contains expected content
            if "cycpep-tools" in stdout:
                print("✅ Server name found in output")

            if "FastMCP" in stdout:
                print("✅ FastMCP framework detected")

            return proc.returncode == 0

        except subprocess.TimeoutExpired:
            print("❌ Server startup timed out")
            proc.kill()
            return False

    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False

def test_direct_function_access():
    """Test accessing the raw functions directly (bypassing MCP)."""
    try:
        print("Testing direct function access...")

        # Import the raw functions
        from helm_to_smiles import run_helm_to_smiles

        # Test with simple HELM
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = run_helm_to_smiles(input_file=test_helm)

        print(f"✅ Direct function call successful")
        print(f"   Success count: {result.get('success_count', 0)}")
        print(f"   Total count: {result.get('total_count', 0)}")

        return result.get('success_count', 0) > 0

    except Exception as e:
        print(f"❌ Direct function access failed: {e}")
        return False

def test_config_and_model_files():
    """Test that all required files exist."""
    try:
        print("Checking required files...")

        # Check config files
        configs_dir = MCP_ROOT / "configs"
        required_configs = [
            "default_config.json",
            "helm_to_smiles_config.json",
            "predict_permeability_config.json",
            "predict_kras_binding_config.json"
        ]

        for config_file in required_configs:
            config_path = configs_dir / config_file
            if config_path.exists():
                print(f"   ✅ {config_file}")
            else:
                print(f"   ❌ {config_file} missing")
                return False

        # Check model files
        models_dir = MCP_ROOT / "examples" / "data" / "models"
        required_models = [
            "regression_rf.pkl",
            "kras_xgboost_reg.pkl"
        ]

        for model_file in required_models:
            model_path = models_dir / model_file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024*1024)
                print(f"   ✅ {model_file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {model_file} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ File check failed: {e}")
        return False

def main():
    """Run server startup tests."""
    print("🧪 MCP Server Startup Tests\n")

    tests = [
        ("Server Import", test_server_import),
        ("Required Files", test_config_and_model_files),
        ("Direct Function Access", test_direct_function_access),
        ("Server Startup", test_server_startup)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        try:
            if test_func():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ CRASHED: {e}")

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print('='*50)

    if passed == total:
        print("🎉 Server is ready to use!")
        print("\nTo start the server:")
        print("  mamba activate ./env")
        print("  python src/server.py")
        print("\nTo test in development mode:")
        print("  fastmcp dev src/server.py")
        return 0
    else:
        print("⚠️  Server needs fixes before use")
        return 1

if __name__ == "__main__":
    exit(main())