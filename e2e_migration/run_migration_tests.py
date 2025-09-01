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


def find_test_files(test_dir):
    """Find all test.yml files in the test directory."""
    test_files = []
    test_path = test_dir / "test"
    
    if not test_path.exists():
        return test_files
    
    # Find all .yml files in test subdirectories
    for subdir in test_path.iterdir():
        if subdir.is_dir():
            for yml_file in subdir.glob("*.yml"):
                test_files.append(yml_file)
    
    return sorted(test_files)


def run_tests_for_directory(test_dir):
    """Run tests for a specific test directory."""
    print(f"\n{'='*50}")
    print(f"Running tests for {test_dir.name}")
    print(f"{'='*50}")
    
    # Find all test files
    test_files = find_test_files(test_dir)
    if not test_files:
        print(f"No test files found in {test_dir.name}/test/")
        return 1
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.relative_to(test_dir)}")
    
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
        
        overall_exit_code = 0
        
        # Run each test file
        for test_file in test_files:
            test_name = test_file.stem
            print(f"\n--- Running test: {test_name} ---")
            
            # Build pytest command with test directory context
            pytest_cmd = [
                "python3", "-m", "pytest", "test_migration.py", "-v", "--tb=short",
                f"--test-dir={test_dir.name}",
                f"--test-file={test_file.relative_to(test_dir)}"
            ]
            
            result = subprocess.run(pytest_cmd)
            if result.returncode != 0:
                overall_exit_code = result.returncode
                print(f"Test {test_name} failed (exit code: {result.returncode})")
            else:
                print(f"Test {test_name} passed")
        
        # Cleanup
        os.chdir(test_dir)
        subprocess.run(["docker-compose", "down", "-v"], check=True)
        os.chdir("..")
        
        return overall_exit_code
        
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
