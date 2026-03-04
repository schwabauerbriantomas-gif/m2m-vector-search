#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Validation Script - Manual Execution Required

This script validates that all M2M files exist and were created successfully.
Executes this script manually from CMD/PowerShell to see results.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_DIR = r"C:\Users\Brian\.openclaw\workspace\projects\m2m"
REPORT_FILE = r"C:\Users\Brian\.openclaw\workspace\m2m_validation_report.json"

def validate_directory(directory, description):
    """Validate directory exists and list contents."""
    path = Path(directory)
    result = {
        'exists': False,
        'is_directory': False,
        'file_count': 0,
        'file_list': []
    }
    
    if path.exists():
        result['exists'] = True
        
        if path.is_dir():
            result['is_directory'] = True
            
            # List all files
            files = []
            for item in path.rglob('*'):
                files.append({
                    'name': item.name,
                    'path': str(item),
                    'size_bytes': item.stat().st_size if item.is_file() else 0,
                    'is_file': item.is_file()
                })
            
            result['file_count'] = len(files)
            result['file_list'] = sorted(files, key=lambda x: x['name'])
    
    return result

def validate_file(filepath, description, check_content=False):
    """Validate file exists, size, and optionally content."""
    path = Path(filepath)
    result = {
        'exists': False,
        'is_file': False,
        'size_bytes': 0,
        'content_preview': None,
        'has_syntax_error': False
    }
    
    if path.exists():
        result['exists'] = True
        
        if path.is_file():
            result['is_file'] = True
            result['size_bytes'] = path.stat().st_size
            
            if check_content:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Basic syntax check for Python files
                        if path.suffix == '.py':
                            if content.count('def ') < 1:
                                result['has_syntax_error'] = True
                        result['content_preview'] = content[:200] + "..." if len(content) > 200 else content
                except Exception as e:
                    result['content_preview'] = f"Error reading file: {str(e)}"
    
    return result

def validate_m2m_files():
    """Validate all M2M files created in Phase 1."""
    print("=" * 70)
    print("M2M VALIDATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'project_directory': PROJECT_DIR,
        'validation_results': {},
        'summary': {
            'total_files_checked': 0,
            'files_created': [],
            'files_failed': [],
            'critical_files': {}
        }
    }
    
    files_to_check = [
        ('README.md', 'Main documentation', True),
        ('LICENSE', 'Apache 2.0 license', True),
        ('m2m.py', 'Main Python module', True),
        ('config.py', 'Configuration module', True),
        ('examples/basic_usage.py', 'Basic usage example', True),
        ('examples/langchain_rag.py', 'LangChain RAG example', True),
        ('examples/llamaindex_rag.py', 'LlamaIndex RAG example', True),
        ('benchmarks/benchmark_m2m.py', 'Benchmark script', True),
        ('tests/README.md', 'Test suite documentation', False)
    ]
    
    total_checked = 0
    
    print("VALIDATING FILES...")
    print("-" * 70)
    
    for filename, description, check_content in files_to_check:
        total_checked += 1
        filepath = os.path.join(PROJECT_DIR, filename)
        
        print(f"Checking: {filename:50}")
        
        result = validate_file(filepath, description, check_content)
        
        report['validation_results'][filename] = result
        
        if result['exists']:
            print(f"  [OK] File exists ({result['size_bytes']} bytes)")
            
            if result['is_file']:
                print(f"  [OK] Valid file")
                
                if check_content and result['content_preview']:
                    print(f"  [OK] Content preview: {result['content_preview'][:100]}")
                
                if check_content and result['has_syntax_error']:
                    print(f"  [WARNING] Possible syntax error (fewer 'def ')")
                    report['summary']['files_failed'].append(filename)
                else:
                    report['summary']['files_created'].append(filename)
            
            if description in ['README.md', 'LICENSE', 'm2m.py']:
                report['summary']['critical_files'][filename] = result
        else:
            print(f"  [FAILED] File NOT FOUND")
            report['summary']['files_failed'].append(filename)
        
        print()
    
    report['summary']['total_files_checked'] = total_checked
    
    return report

def main():
    """Main validation function."""
    print("M2M VALIDATION TOOL")
    print("This tool validates all M2M files created in Phase 1.")
    print()
    print("INSTRUCTIONS:")
    print("1. This script will check all M2M files")
    print("2. Results will be saved to:")
    print(f"   {REPORT_FILE}")
    print("3. You can manually open this file to see results")
    print()
    
    input("Press Enter to start validation...")
    print()
    
    # Validate project directory
    dir_result = validate_directory(PROJECT_DIR, "M2M project directory")
    print(f"Directory: {'EXISTS' if dir_result['exists'] else 'NOT FOUND'}")
    print(f"Files: {dir_result['file_count']}")
    print()
    
    # Validate M2M files
    file_report = validate_m2m_files()
    
    # Add directory result to file report
    file_report['validation_results']['project_directory'] = dir_result
    
    # Save report
    try:
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(file_report, f, indent=2)
        
        print()
        print("=" * 70)
        print("VALIDATION COMPLETED")
        print("=" * 70)
        print(f"Report saved to: {REPORT_FILE}")
        print()
        
        # Print summary
        print("SUMMARY:")
        print(f"Total files checked: {file_report['summary']['total_files_checked']}")
        print(f"Files created: {len(file_report['summary']['files_created'])}")
        print(f"Files failed: {len(file_report['summary']['files_failed'])}")
        print()
        
        # Check critical files
        critical_status = []
        for filename, result in file_report['summary']['critical_files'].items():
            if result['exists'] and result['is_file']:
                status = "✅ OK"
                critical_status.append(True)
            else:
                status = "❌ MISSING"
                critical_status.append(False)
            
            print(f"  {filename}: {status}")
        
        print()
        print("CRITICAL FILES STATUS:")
        if all(critical_status):
            print("  [SUCCESS] All critical files (README.md, LICENSE, m2m.py) are present!")
        else:
            print("  [FAILED] Some critical files are missing. Please check manually.")
        
        print()
        print("=" * 70)
        print("NEXT STEPS:")
        print("1. Open report file to see detailed results")
        print("2. If all critical files are present, proceed to GitHub release")
        print("3. If any files are missing, report back to AI assistant")
        print()
        print(f"To open report: notepad \"{REPORT_FILE}\"")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save report: {str(e)}")
        return False

if __name__ == "__main__":
    main()
