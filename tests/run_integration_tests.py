#!/usr/bin/env python3
"""Automated integration test runner for Cyclic Peptide MCP server."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class MCPTestRunner:
    """Test runner for MCP server integration tests."""

    def __init__(self, server_path: str, env_path: str):
        self.server_path = Path(server_path).resolve()
        self.env_path = Path(env_path).resolve()
        self.results = {
            "test_date": datetime.now().isoformat(),
            "server_path": str(server_path),
            "env_path": str(env_path),
            "tests": {},
            "issues": [],
            "summary": {},
            "performance": {}
        }

    def run_command(self, cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Run a command and capture output."""
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time": execution_time
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "execution_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def test_server_startup(self) -> bool:
        """Test that server starts without errors."""
        print("🧪 Testing server startup...")

        cmd = [
            "mamba", "run", "-p", str(self.env_path),
            "python", "-c", "from src.server import mcp; print('Server started successfully')"
        ]

        result = self.run_command(cmd, timeout=60)

        self.results["tests"]["server_startup"] = {
            "status": "passed" if result["success"] else "failed",
            "execution_time": result["execution_time"],
            "output": result.get("stdout", ""),
            "error": result.get("stderr", "") or result.get("error", "")
        }

        if result["success"]:
            print("✅ Server startup test passed")
        else:
            print(f"❌ Server startup test failed: {result.get('stderr', result.get('error'))}")
            self.results["issues"].append({
                "test": "server_startup",
                "severity": "high",
                "error": result.get("stderr", result.get("error")),
                "suggested_fix": "Check server.py syntax and dependencies"
            })

        return result["success"]

    def test_dependencies(self) -> bool:
        """Test that all required dependencies are available."""
        print("🧪 Testing dependencies...")

        dependencies = ["fastmcp", "loguru", "pandas", "numpy", "sklearn"]
        all_passed = True

        for dep in dependencies:
            cmd = [
                "mamba", "run", "-p", str(self.env_path),
                "python", "-c", f"import {dep}; print('{dep} imported successfully')"
            ]

            result = self.run_command(cmd, timeout=30)

            self.results["tests"][f"dependency_{dep}"] = {
                "status": "passed" if result["success"] else "failed",
                "execution_time": result["execution_time"],
                "output": result.get("stdout", ""),
                "error": result.get("stderr", "") or result.get("error", "")
            }

            if result["success"]:
                print(f"✅ {dep} import test passed")
            else:
                print(f"❌ {dep} import test failed: {result.get('stderr', result.get('error'))}")
                all_passed = False
                self.results["issues"].append({
                    "test": f"dependency_{dep}",
                    "severity": "high",
                    "error": result.get("stderr", result.get("error")),
                    "suggested_fix": f"Install {dep} with: mamba install {dep}"
                })

        return all_passed

    def test_claude_cli_registration(self) -> bool:
        """Test that Claude CLI can see the registered server."""
        print("🧪 Testing Claude CLI registration...")

        cmd = ["claude", "mcp", "list"]
        result = self.run_command(cmd, timeout=30)

        if result["success"] and "cycpep-tools" in result["stdout"]:
            status = "passed"
            print("✅ Claude CLI registration test passed")
        else:
            status = "failed"
            print(f"❌ Claude CLI registration test failed")
            self.results["issues"].append({
                "test": "claude_cli_registration",
                "severity": "medium",
                "error": "Server not found in Claude CLI",
                "suggested_fix": "Re-register with: claude mcp add cycpep-tools ..."
            })

        self.results["tests"]["claude_cli_registration"] = {
            "status": status,
            "execution_time": result["execution_time"],
            "output": result.get("stdout", ""),
            "error": result.get("stderr", "") or result.get("error", "")
        }

        return status == "passed"

    def test_fastmcp_dev_mode(self) -> bool:
        """Test that fastmcp dev mode starts."""
        print("🧪 Testing FastMCP dev mode...")

        cmd = [
            "mamba", "run", "-p", str(self.env_path),
            "timeout", "10s", "fastmcp", "dev", str(self.server_path)
        ]

        result = self.run_command(cmd, timeout=15)

        # Dev mode should start and then timeout - that's expected
        if "MCP inspector" in result.get("stdout", "") or "MCP Inspector" in result.get("stdout", ""):
            status = "passed"
            print("✅ FastMCP dev mode test passed")
        else:
            status = "failed"
            print(f"❌ FastMCP dev mode test failed")
            self.results["issues"].append({
                "test": "fastmcp_dev_mode",
                "severity": "medium",
                "error": "FastMCP dev mode didn't start properly",
                "suggested_fix": "Check FastMCP installation and server.py"
            })

        self.results["tests"]["fastmcp_dev_mode"] = {
            "status": status,
            "execution_time": result["execution_time"],
            "output": result.get("stdout", ""),
            "error": result.get("stderr", "") or result.get("error", "")
        }

        return status == "passed"

    def test_model_files(self) -> bool:
        """Test that required model files exist."""
        print("🧪 Testing model files...")

        model_files = [
            "examples/data/models/regression_rf.pkl",
            "examples/data/models/kras_xgboost_reg.pkl"
        ]

        all_exist = True

        for model_file in model_files:
            file_path = Path(model_file)
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✅ {model_file} exists ({file_size:.1f} MB)")

                self.results["tests"][f"model_file_{file_path.name}"] = {
                    "status": "passed",
                    "file_path": str(file_path),
                    "file_size_mb": file_size
                }
            else:
                print(f"❌ {model_file} not found")
                all_exist = False

                self.results["tests"][f"model_file_{file_path.name}"] = {
                    "status": "failed",
                    "file_path": str(file_path),
                    "error": "File not found"
                }

                self.results["issues"].append({
                    "test": f"model_file_{file_path.name}",
                    "severity": "high",
                    "error": f"Model file {model_file} not found",
                    "suggested_fix": "Check if models were downloaded correctly"
                })

        return all_exist

    def test_config_files(self) -> bool:
        """Test that all config files exist and are valid JSON."""
        print("🧪 Testing configuration files...")

        config_files = [
            "configs/default_config.json",
            "configs/helm_to_smiles_config.json",
            "configs/predict_permeability_config.json",
            "configs/predict_kras_binding_config.json"
        ]

        all_valid = True

        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        config = json.load(f)
                    print(f"✅ {config_file} is valid JSON")

                    self.results["tests"][f"config_file_{file_path.name}"] = {
                        "status": "passed",
                        "file_path": str(file_path),
                        "config_keys": list(config.keys()) if isinstance(config, dict) else "not_dict"
                    }
                except json.JSONDecodeError as e:
                    print(f"❌ {config_file} invalid JSON: {e}")
                    all_valid = False

                    self.results["tests"][f"config_file_{file_path.name}"] = {
                        "status": "failed",
                        "file_path": str(file_path),
                        "error": f"Invalid JSON: {e}"
                    }

                    self.results["issues"].append({
                        "test": f"config_file_{file_path.name}",
                        "severity": "medium",
                        "error": f"Config file {config_file} has invalid JSON: {e}",
                        "suggested_fix": "Fix JSON syntax in config file"
                    })
            else:
                print(f"❌ {config_file} not found")
                all_valid = False

                self.results["tests"][f"config_file_{file_path.name}"] = {
                    "status": "failed",
                    "file_path": str(file_path),
                    "error": "File not found"
                }

                self.results["issues"].append({
                    "test": f"config_file_{file_path.name}",
                    "severity": "medium",
                    "error": f"Config file {config_file} not found",
                    "suggested_fix": "Create missing config file"
                })

        return all_valid

    def generate_summary(self) -> None:
        """Generate test summary."""
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values()
                          if test.get("status") == "passed")
        failed_tests = total_tests - passed_tests

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": f"{passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A",
            "total_issues": len(self.results["issues"]),
            "high_severity_issues": len([i for i in self.results["issues"] if i["severity"] == "high"]),
            "medium_severity_issues": len([i for i in self.results["issues"] if i["severity"] == "medium"])
        }

        # Performance metrics
        execution_times = [test.get("execution_time", 0) for test in self.results["tests"].values()
                          if "execution_time" in test]
        if execution_times:
            self.results["performance"] = {
                "total_execution_time": sum(execution_times),
                "average_test_time": sum(execution_times) / len(execution_times),
                "fastest_test": min(execution_times),
                "slowest_test": max(execution_times)
            }

    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🚀 Starting MCP Server Integration Tests")
        print("=" * 60)

        test_methods = [
            self.test_dependencies,
            self.test_model_files,
            self.test_config_files,
            self.test_server_startup,
            self.test_claude_cli_registration,
            self.test_fastmcp_dev_mode
        ]

        all_passed = True

        for test_method in test_methods:
            try:
                result = test_method()
                all_passed = all_passed and result
                print()  # Add spacing between tests
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed with exception: {e}")
                all_passed = False
                self.results["issues"].append({
                    "test": test_method.__name__,
                    "severity": "high",
                    "error": f"Test exception: {e}",
                    "suggested_fix": "Check test implementation"
                })

        self.generate_summary()

        print("=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"Total Tests: {self.results['summary']['total_tests']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Failed: {self.results['summary']['failed']}")
        print(f"Pass Rate: {self.results['summary']['pass_rate']}")
        print(f"Total Issues: {self.results['summary']['total_issues']}")

        if self.results["summary"]["total_issues"] > 0:
            print("\n🔍 Issues Found:")
            for issue in self.results["issues"]:
                severity_emoji = "🔴" if issue["severity"] == "high" else "🟡"
                print(f"{severity_emoji} {issue['test']}: {issue['error']}")
                print(f"   💡 Suggested fix: {issue['suggested_fix']}")

        if "performance" in self.results:
            print(f"\n⏱️ Performance:")
            print(f"Total execution time: {self.results['performance']['total_execution_time']:.1f}s")
            print(f"Average test time: {self.results['performance']['average_test_time']:.1f}s")

        if all_passed:
            print("\n🎉 All tests passed! MCP server is ready for use.")
        else:
            print(f"\n❌ {self.results['summary']['failed']} test(s) failed. See issues above.")

        return all_passed

    def save_report(self, output_file: str) -> None:
        """Save test results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n📄 Full test report saved to: {output_path}")


def main():
    """Main test runner entry point."""
    server_path = "src/server.py"
    env_path = "./env"

    if not Path(server_path).exists():
        print(f"❌ Server file not found: {server_path}")
        sys.exit(1)

    if not Path(env_path).exists():
        print(f"❌ Environment directory not found: {env_path}")
        sys.exit(1)

    runner = MCPTestRunner(server_path, env_path)
    success = runner.run_all_tests()

    # Save detailed report
    runner.save_report("reports/step7_integration_test_results.json")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()