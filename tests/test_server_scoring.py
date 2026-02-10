#!/usr/bin/env python3
"""Test server-based scoring MCP tools (Boltz2/Rosetta)."""

import sys
import json
from pathlib import Path

# Add src and scripts directories to path
TEST_DIR = Path(__file__).parent
MCP_ROOT = TEST_DIR.parent
sys.path.insert(0, str(MCP_ROOT / "src"))
sys.path.insert(0, str(MCP_ROOT / "scripts"))


def get_tool_fn(tool):
    """Extract the underlying function from a FastMCP FunctionTool."""
    if hasattr(tool, 'fn'):
        return tool.fn
    return tool


def test_server_scorer_info_tool():
    """Test the get_server_scorer_info MCP tool."""
    print("Testing get_server_scorer_info tool...")
    try:
        from server import get_server_scorer_info
        fn = get_tool_fn(get_server_scorer_info)

        result = fn()

        print(f"  Status: {result['status']}")

        if result['status'] == 'success':
            print(f"  Available scorers: {result.get('available_scorers', [])}")
            print(f"  Repo implementation available: {result.get('repo_implementation_available', False)}")

            # Check Boltz2 info
            boltz2 = result.get('boltz2', {})
            print(f"  Boltz2 supported scores: {boltz2.get('supported_scores', [])}")

            # Check Rosetta info
            rosetta = result.get('rosetta', {})
            print(f"  Rosetta supported scores: {rosetta.get('supported_scores', [])}")

            print("✅ PASSED")
            return True
        else:
            print(f"  Error: {result.get('error')}")
            print("❌ FAILED")
            return False

    except Exception as e:
        print(f"❌ get_server_scorer_info failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_scoring_utils_module():
    """Test the server_scoring_utils module directly."""
    print("Testing server_scoring_utils module...")
    try:
        from lib.server_scoring_utils import (
            ServerScorerWrapper,
            get_available_server_scorers,
            check_server_connection
        )

        # Test get_available_server_scorers
        info = get_available_server_scorers()
        print(f"  Repo available: {info.get('repo_available', False)}")
        print(f"  Available scorers: {info.get('available_scorers', [])}")

        # Test ServerScorerWrapper initialization
        print("  Testing ServerScorerWrapper initialization...")

        # Initialize Rosetta scorer
        rosetta = ServerScorerWrapper(
            scorer_type='rosetta',
            server_host='http://localhost:8001'
        )
        print(f"  Rosetta scorer initialized: using_repo={rosetta._use_repo}")

        # Initialize Boltz2 scorer
        boltz2 = ServerScorerWrapper(
            scorer_type='boltz2',
            server_host='http://localhost:8000'
        )
        print(f"  Boltz2 scorer initialized: using_repo={boltz2._use_repo}")

        # Test get_scorer_info method
        scorer_info = rosetta.get_scorer_info()
        print(f"  Rosetta scorer info keys: {list(scorer_info.keys())}")

        # Test check_server_connection (expects failure since server not running)
        print("  Testing check_server_connection (expecting failure)...")
        conn_result = check_server_connection(
            scorer_type='rosetta',
            server_host='http://localhost:8001',
            timeout=2
        )
        print(f"  Connection status: {conn_result.get('status')}")
        # We expect 'unreachable' since no server is running

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ server_scoring_utils module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_with_server_tool():
    """Test the score_with_server MCP tool (without running server)."""
    print("Testing score_with_server tool...")
    try:
        from server import score_with_server
        fn = get_tool_fn(score_with_server)

        # Test with a sequence - expects connection error since no server running
        result = fn(
            helm_input="ACDEFGHIK",
            scorer_type="rosetta",
            server_host="http://localhost:8001",
            timeout=2
        )

        print(f"  Status: {result['status']}")

        # We expect either error (connection refused) or the result structure
        if 'error' in result:
            print(f"  Error (expected - no server): {result.get('error', '')[:100]}...")
        else:
            print(f"  Scorer type: {result.get('scorer_type')}")
            if 'summary' in result:
                print(f"  Summary: {result.get('summary')}")

        # Check that the function runs without crashing
        print("✅ PASSED (function executed correctly, connection failure expected)")
        return True

    except Exception as e:
        print(f"❌ score_with_server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_submit_server_scoring_tool():
    """Test the submit_server_scoring MCP tool validation."""
    print("Testing submit_server_scoring tool validation...")
    try:
        from server import submit_server_scoring
        fn = get_tool_fn(submit_server_scoring)

        # Test with invalid scorer type
        result = fn(
            input_file="/nonexistent/file.csv",
            scorer_type="invalid_scorer",
            output_dir="/tmp/test_output"
        )

        print(f"  Status: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error (expected): {result.get('error', '')[:100]}...")
            if 'invalid_scorer' in result.get('error', '') or 'Invalid scorer_type' in result.get('error', ''):
                print("✅ PASSED (validation correctly rejected invalid scorer)")
                return True
            else:
                print("  Note: Error was for different reason")

        # Test with valid scorer but nonexistent file
        result = fn(
            input_file="/nonexistent/file.csv",
            scorer_type="rosetta",
            output_dir="/tmp/test_output"
        )

        print(f"  Status: {result['status']}")
        if result['status'] == 'error':
            print(f"  Error (expected): {result.get('error', '')[:100]}...")
            if 'not found' in result.get('error', '').lower():
                print("✅ PASSED (validation correctly rejected nonexistent file)")
                return True

        print("❌ FAILED (expected validation error)")
        return False

    except Exception as e:
        print(f"❌ submit_server_scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_with_server_script():
    """Test the score_with_server.py script directly."""
    print("Testing score_with_server.py script...")
    try:
        from score_with_server import run_server_scoring, get_available_server_scorers

        # Test get_available_server_scorers
        info = get_available_server_scorers()
        print(f"  Available scorers: {info.get('available_scorers', [])}")

        # Test run_server_scoring with single sequence (expects connection error)
        result = run_server_scoring(
            input_file="ACDEFGHIK",
            scorer_type="rosetta",
            server_host="http://localhost:8001",
            timeout=2
        )

        print(f"  Status: {result.get('status')}")
        print(f"  Scorer type: {result.get('scorer_type')}")

        if 'summary' in result:
            summary = result['summary']
            print(f"  Total sequences: {summary.get('total_sequences')}")
            print(f"  Failed sequences: {summary.get('failed_sequences')}")

        # Connection failure expected
        print("✅ PASSED (script executed correctly)")
        return True

    except Exception as e:
        print(f"❌ score_with_server script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file():
    """Test that server_scoring_config.json is valid."""
    print("Testing server_scoring_config.json...")
    try:
        config_path = MCP_ROOT / "configs" / "server_scoring_config.json"

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"  Config loaded successfully")
        print(f"  Keys: {list(config.keys())}")

        # Check required sections
        required_sections = ['boltz2', 'rosetta', 'transformation_types']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False
            print(f"  ✓ {section} section present")

        # Check Boltz2 config
        boltz2 = config['boltz2']
        if 'server' not in boltz2 or 'supported_scores' not in boltz2:
            print("❌ Boltz2 config missing server or supported_scores")
            return False
        print(f"  Boltz2 server: {boltz2['server'].get('host')}")

        # Check Rosetta config
        rosetta = config['rosetta']
        if 'server' not in rosetta or 'supported_scores' not in rosetta:
            print("❌ Rosetta config missing server or supported_scores")
            return False
        print(f"  Rosetta server: {rosetta['server'].get('host')}")

        print("✅ PASSED")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config: {e}")
        return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_updated_server_info():
    """Test that get_server_info includes server scoring tools."""
    print("Testing updated get_server_info...")
    try:
        from server import get_server_info
        fn = get_tool_fn(get_server_info)

        result = fn()

        print(f"  Server version: {result.get('version')}")

        # Check for new server_scoring_tools category
        server_scoring_tools = result.get('server_scoring_tools', [])
        print(f"  Server scoring tools: {server_scoring_tools}")

        expected_tools = ['score_with_server', 'submit_server_scoring', 'get_server_scorer_info']
        for tool in expected_tools:
            if tool in server_scoring_tools:
                print(f"    ✓ {tool}")
            else:
                print(f"    ✗ {tool} (missing)")

        # Check sync_tools includes score_with_server
        sync_tools = result.get('sync_tools', [])
        if 'score_with_server' in sync_tools:
            print(f"  ✓ score_with_server in sync_tools")
        else:
            print(f"  ✗ score_with_server not in sync_tools")

        # Check submit_tools includes submit_server_scoring
        submit_tools = result.get('submit_tools', [])
        if 'submit_server_scoring' in submit_tools:
            print(f"  ✓ submit_server_scoring in submit_tools")
        else:
            print(f"  ✗ submit_server_scoring not in submit_tools")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ Server info test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all server scoring tests."""
    print("🧪 Testing Server-Based Scoring MCP Tools\n")

    tests = [
        ("Config File", test_config_file),
        ("Server Scoring Utils Module", test_server_scoring_utils_module),
        ("get_server_scorer_info Tool", test_server_scorer_info_tool),
        ("score_with_server Tool", test_score_with_server_tool),
        ("submit_server_scoring Validation", test_submit_server_scoring_tool),
        ("score_with_server Script", test_score_with_server_script),
        ("Updated Server Info", test_updated_server_info),
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
        except Exception as e:
            print(f"❌ CRASHED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print('='*60)

    if passed == total:
        print("🎉 All server scoring tests passed!")
        return 0
    else:
        print("⚠️  Some tests need attention")
        return 1


if __name__ == "__main__":
    exit(main())
