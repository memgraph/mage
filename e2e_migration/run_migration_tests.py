#!/usr/bin/env python3
"""
Migration test runner for e2e_migration tests.
This script can be called directly or through pytest.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_test_directories():
    """Find all test_* directories in the current directory."""
    current_dir = Path(__file__).parent
    test_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("test_")]
    return sorted(test_dirs)


def filter_test_directories(test_dirs, filter_pattern):
    """Filter test directories based on the -k pattern."""
    if not filter_pattern:
        return test_dirs
    
    filtered_dirs = []
    for test_dir in test_dirs:
        # Extract database type from directory name (e.g., "test_mysql" -> "mysql")
        db_type = test_dir.name.replace("test_", "")
        
        # Check if the filter pattern matches the database type
        if filter_pattern.lower() in db_type.lower():
            filtered_dirs.append(test_dir)
    
    return filtered_dirs


def run_tests_for_directory(test_dir):
    """Run tests for a specific test directory."""
    print(f"\n{'='*50}")
    print(f"Running tests for {test_dir.name}")
    print(f"{'='*50}")
    
    # Use docker-compose to start services and run tests
    os.chdir(test_dir)
    try:
        # Start services
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        # Wait for services
        print("Waiting for services to be ready...")
        import time
        time.sleep(30)
        
        # Check service health
        subprocess.run(["docker-compose", "ps"], check=True)
        
        # Run tests from parent directory
        os.chdir("..")
        
        # Build pytest command with test directory context
        pytest_cmd = [
            "python3", "-m", "pytest", "test_migration.py", "-v", "--tb=short",
            f"--test-dir={test_dir.name}"
        ]
        
        result = subprocess.run(pytest_cmd)
        
        # Cleanup
        os.chdir(test_dir)
        subprocess.run(["docker-compose", "down", "-v"], check=True)
        os.chdir("..")
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tests for {test_dir.name}: {e}")
        # Try to cleanup even if tests failed
        try:
            subprocess.run(["docker-compose", "down", "-v"])
        except:
            pass
        return e.returncode


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run e2e migration tests")
    parser.add_argument(
        "-k",
        dest="test_filter",
        help="Filter tests by database type (e.g., 'mysql', 'postgresql')",
        type=str,
        required=False,
    )
    return parser.parse_args()


def main():
    """Main function to run all migration tests."""
    args = parse_arguments()
    
    test_dirs = find_test_directories()
    
    if not test_dirs:
        print("No test_* directories found!")
        return 1
    
    # Apply filter if provided
    if args.test_filter:
        test_dirs = filter_test_directories(test_dirs, args.test_filter)
        if not test_dirs:
            print(f"No test directories found matching filter: {args.test_filter}")
            return 1
        print(f"Filtered test directories: {[d.name for d in test_dirs]}")
    else:
        print(f"Found test directories: {[d.name for d in test_dirs]}")
    
    overall_exit_code = 0
    
    for test_dir in test_dirs:
        exit_code = run_tests_for_directory(test_dir)
        if exit_code != 0:
            overall_exit_code = exit_code
        print(f"Completed tests for {test_dir.name} (exit code: {exit_code})")
    
    print(f"\n{'='*50}")
    print(f"All tests completed (overall exit code: {overall_exit_code})")
    print(f"{'='*50}")
    
    return overall_exit_code


if __name__ == "__main__":
    sys.exit(main())
