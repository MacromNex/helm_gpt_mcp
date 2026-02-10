#!/usr/bin/env python3
"""Simple test of individual functions."""

import sys
from pathlib import Path

# Add src directory to path
TEST_DIR = Path(__file__).parent
MCP_ROOT = TEST_DIR.parent
sys.path.insert(0, str(MCP_ROOT / "src"))
sys.path.insert(0, str(MCP_ROOT / "scripts"))

def test_script_imports():
    """Test that we can import the original scripts."""
    try:
        import helm_to_smiles
        import predict_permeability
        import predict_kras_binding
        print("✅ All script modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import scripts: {e}")
        return False

def test_direct_function_calls():
    """Test calling script functions directly."""
    try:
        from helm_to_smiles import run_helm_to_smiles

        # Test with a simple HELM - pass as input_file (will be treated as single string)
        test_helm = "PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$"
        result = run_helm_to_smiles(input_file=test_helm)

        print(f"✅ Direct function call successful")
        print(f"   Result keys: {list(result.keys())}")

        if 'results' in result and result['results']:
            conv = result['results'][0]
            print(f"   HELM: {conv.get('helm', 'N/A')}")
            print(f"   Valid: {conv.get('valid', 'N/A')}")
            print(f"   SMILES: {conv.get('smiles', 'N/A')}")

        return True
    except Exception as e:
        print(f"❌ Direct function call failed: {e}")
        return False

def test_model_files():
    """Test that model files exist."""
    models_dir = MCP_ROOT / "examples" / "data" / "models"

    permeability_model = models_dir / "regression_rf.pkl"
    kras_model = models_dir / "kras_xgboost_reg.pkl"

    print(f"Models directory: {models_dir}")
    print(f"Permeability model: {'✅' if permeability_model.exists() else '❌'} {permeability_model}")
    print(f"KRAS model: {'✅' if kras_model.exists() else '❌'} {kras_model}")

    return permeability_model.exists() and kras_model.exists()

def test_config_files():
    """Test that config files exist."""
    configs_dir = MCP_ROOT / "configs"

    config_files = [
        "default_config.json",
        "helm_to_smiles_config.json",
        "predict_permeability_config.json",
        "predict_kras_binding_config.json"
    ]

    print(f"Configs directory: {configs_dir}")
    for config_file in config_files:
        config_path = configs_dir / config_file
        print(f"{config_file}: {'✅' if config_path.exists() else '❌'}")

    return all((configs_dir / cf).exists() for cf in config_files)

def main():
    """Run simple tests."""
    print("🧪 Simple Tests for MCP Components\n")

    tests = [
        ("Script imports", test_script_imports),
        ("Direct function calls", test_direct_function_calls),
        ("Model files", test_model_files),
        ("Config files", test_config_files)
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

    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())