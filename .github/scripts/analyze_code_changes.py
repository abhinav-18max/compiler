#!/usr/bin/env python3
"""
Code Analyzer Script for GitHub Action

This script analyzes code changes in a repository:
- Detects changes between commits in a push
- Identifies changed/added/deleted functions, classes, and methods
- Generates structured JSON output for further processing by LLM backend
"""

import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from git import Repo, GitCommandError
from tree_sitter import Language, Parser
import argparse
from fnmatch import fnmatch

# Configure paths
REPO_PATH = os.getcwd()
DOCAI_DIR = os.path.join(REPO_PATH, ".docai")
ELEMENTS_DB_PATH = os.path.join(DOCAI_DIR, "code_elements.json")

# Default excluded paths (can be overridden by .docai/config.json)
DEFAULT_EXCLUDED_PATHS = [
    ".git/*",
    ".github/*",
    "node_modules/*",
    "venv/*",
    ".env/*",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.class",
    "*.log",
    "*.cache",
    "build/*",
    "dist/*",
    ".next/*",
    ".docai/*",
]


def load_config():
    """Load configuration from .docai/config.json or use defaults."""
    config_path = os.path.join(DOCAI_DIR, "config.json")
    config = {"excluded_paths": DEFAULT_EXCLUDED_PATHS}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                if "excluded_paths" in user_config:
                    config["excluded_paths"] = user_config["excluded_paths"]
        except Exception as e:
            print(f"Warning: Error loading config from {config_path}: {e}")
            print("Using default configuration")

    return config


# Load configuration
CONFIG = load_config()

# Ensure .docai directory exists
os.makedirs(DOCAI_DIR, exist_ok=True)

# Language extensions mapping
LANGUAGE_EXTENSIONS = {
    # Python
    ".py": "python",
    # JavaScript/TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Java
    ".java": "java",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
}

# Tree-sitter language modules mapping
try:
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_java
    import tree_sitter_go
    import tree_sitter_rust

    LANGUAGE_MODULES = {
        "python": tree_sitter_python.language,
        "javascript": tree_sitter_javascript.language,
        "typescript": tree_sitter_typescript.language_typescript,
        "tsx": tree_sitter_typescript.language_tsx,
        "java": tree_sitter_java.language,
        "go": tree_sitter_go.language,
        "rust": tree_sitter_rust.language,
    }
except ImportError as e:
    print(f"Warning: Error importing tree-sitter modules: {e}")
    print("Some language support may be limited")
    LANGUAGE_MODULES = {}


class CodeAnalyzer:
    """Analyzes code changes in a git repository."""

    def __init__(self, repo_path: str):
        """
        Initialize the CodeAnalyzer.

        Args:
            repo_path: Path to the git repository
        """
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.languages = self._setup_tree_sitter()
        self.parser = Parser()
        self.code_elements_db = self._load_code_elements_db()

    def _setup_tree_sitter(self) -> Dict[str, Language]:
        """Initialize and load tree-sitter languages."""
        lang_objects = {}
        for lang_name, lang_module in LANGUAGE_MODULES.items():
            try:
                if lang_module():
                    lang_objects[lang_name] = Language(lang_module())
            except Exception as e:
                print(f"Error loading language {lang_name}: {e}")
        return lang_objects

    def _load_code_elements_db(self) -> Dict:
        """Load existing code elements database if it exists."""
        if os.path.exists(ELEMENTS_DB_PATH):
            try:
                with open(ELEMENTS_DB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Invalid JSON in {ELEMENTS_DB_PATH}, creating new database"
                )

        # Return empty database structure if file doesn't exist or is invalid
        return {"elements": {}, "metadata": {"last_processed_commit": None}}

    def _save_code_elements_db(self):
        """Save the code elements database."""
        with open(ELEMENTS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(self.code_elements_db, f, indent=2)

    def _get_file_language(self, file_path: str) -> Optional[str]:
        """
        Determine the programming language of a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None if not supported
        """
        extension = os.path.splitext(file_path)[1].lower()
        return LANGUAGE_EXTENSIONS.get(extension)

    def _should_exclude_path(self, file_path: str) -> bool:
        """
        Check if a file path should be excluded based on configuration.

        Args:
            file_path: Path to check

        Returns:
            True if path should be excluded, False otherwise
        """
        # Convert to relative path for matching
        rel_path = os.path.relpath(file_path, self.repo_path)

        # Check against each exclude pattern
        for pattern in CONFIG["excluded_paths"]:
            if fnmatch(rel_path, pattern):
                print(f"Excluding {rel_path} (matched pattern {pattern})")
                return True
        return False

    def _get_file_content_at_commit(
        self, file_path: str, commit: str
    ) -> Optional[bytes]:
        """
        Get the content of a file at a specific commit.

        Args:
            file_path: Path to the file
            commit: Commit hash

        Returns:
            File content as bytes or None if not found
        """
        rel_path = os.path.relpath(file_path, self.repo_path)

        try:
            # Get the file content at the specific commit
            content = self.repo.git.show(f"{commit}:{rel_path}")
            return content.encode("utf-8", errors="replace")
        except GitCommandError:
            # File might not exist at this commit
            return None
        except Exception as e:
            print(f"Error getting file content for {file_path} at {commit}: {e}")
            return None

    def _format_code(self, code: str, lang_name: str) -> str:
        """Format code based on language."""
        # Remove extra newlines at start and end
        code = code.strip()

        # Basic indentation fix
        lines = code.split("\n")
        if len(lines) > 1:
            # Find the minimum indentation level
            min_indent = float("inf")
            for line in lines[1:]:  # Skip first line
                if line.strip():  # Only check non-empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            # Remove common indentation
            if min_indent < float("inf"):
                formatted_lines = [lines[0]]  # Keep first line as is
                for line in lines[1:]:
                    if line.strip():  # Only process non-empty lines
                        formatted_lines.append(line[min_indent:])
                    else:
                        formatted_lines.append("")
                code = "\n".join(formatted_lines)

        # Language-specific formatting
        if lang_name in ["javascript", "typescript"]:
            # Format arrow functions
            code = code.replace("( )", "()")
            code = code.replace("( ", "(")
            code = code.replace(" )", ")")
            code = code.replace("=> {", "=> {")
            code = code.replace("})", "}")

            # Format object properties
            code = code.replace(" :", ":")
            code = code.replace(": ", ":")

            # Format then/catch
            code = code.replace(".then (", ".then(")
            code = code.replace(".catch (", ".catch(")

        elif lang_name == "python":
            # Format Python-specific elements
            code = code.replace("def(", "def (")
            code = code.replace("class(", "class (")

            # Ensure proper spacing around operators
            code = code.replace("def  ", "def ")
            code = code.replace("class  ", "class ")

        return code

    def _get_element_type_and_name(
        self, node, content: bytes
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine the type and name of a code element based on AST structure.
        Makes minimal assumptions about coding style.

        Args:
            node: The AST node
            content: File content in bytes

        Returns:
            Tuple of (element_type, element_name)
        """
        element_type = None
        element_name = None

        try:
            # Handle variable declarations and assignments
            if node.type in ["variable_declarator", "assignment"]:
                # Get the name from the left side
                for child in node.children:
                    if child.type == "identifier":
                        element_name = content[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="replace")
                        break

                # Check the right side to determine type
                for child in node.children:
                    if child.type in ["arrow_function", "function", "call_expression"]:
                        element_type = "code_element"
                        break

            # Handle function-like declarations
            elif node.type in [
                "function_definition",  # Python
                "function_declaration",  # JavaScript
                "method_definition",  # JavaScript/TypeScript
                "arrow_function",  # JavaScript/TypeScript
            ]:
                element_type = "code_element"
                # Get function name if present
                for child in node.children:
                    if child.type in ["identifier", "property_identifier"]:
                        element_name = content[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="replace")
                        break

            # Handle class-like declarations
            elif node.type in [
                "class_definition",  # Python
                "class_declaration",  # JavaScript/TypeScript
            ]:
                element_type = "code_element"
                for child in node.children:
                    if child.type in ["identifier", "type_identifier"]:
                        element_name = content[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="replace")
                        break

            # Handle object properties and methods
            elif node.type == "object_property":
                for child in node.children:
                    if child.type == "property_identifier":
                        element_name = content[
                            child.start_byte : child.end_byte
                        ].decode("utf-8", errors="replace")
                    elif child.type in ["arrow_function", "function"]:
                        element_type = "code_element"

            # Handle decorators and their targets
            elif node.type == "decorated_definition":
                element_type = "code_element"
                # Look for the name in the decorated target
                for child in node.children:
                    if child.type in ["function_definition", "class_definition"]:
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                element_name = content[
                                    subchild.start_byte : subchild.end_byte
                                ].decode("utf-8", errors="replace")
                                break

            # Handle export declarations
            elif node.type == "export_statement":
                for child in node.children:
                    if child.type in [
                        "function_declaration",
                        "class_declaration",
                        "variable_declarator",
                    ]:
                        return self._get_element_type_and_name(child, content)

        except Exception as e:
            print(f"Error determining element type and name: {str(e)}")
            return None, None

        return element_type, element_name

    def _parse_file(
        self, file_path: str, content: Optional[bytes] = None
    ) -> List[Dict]:
        """
        Parse a file to extract code elements.
        Uses a generic approach that works with any coding style.

        Args:
            file_path: Path to the file
            content: Optional file content (bytes), reads from file if None

        Returns:
            List of code elements
        """
        lang_name = self._get_file_language(file_path)
        if not lang_name:
            print(f"No language detected for file: {file_path}")
            return []

        if lang_name not in self.languages:
            print(f"Language {lang_name} not supported for file: {file_path}")
            return []

        # Get file content if not provided
        if content is None:
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                print(f"Successfully read file: {file_path} ({len(content)} bytes)")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                return []

        if not content:
            print(f"Empty file content for: {file_path}")
            return []

        # Set the parser language
        try:
            self.parser.language = self.languages[lang_name]
        except Exception as e:
            print(f"Error setting parser language for {lang_name}: {str(e)}")
            return []

        # Parse the file
        try:
            tree = self.parser.parse(content)
            print(f"Successfully parsed file: {file_path}")
        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
            return []

        # Try direct approach using the node iterator
        code_elements = []
        seen_nodes = set()  # To avoid duplicates

        try:
            # Find code elements
            for node in self._iter_tree(tree.root_node):
                print(f"Found node type: {node.type}")  # Debug node types

                # Get element type and name based on AST structure
                element_type, element_name = self._get_element_type_and_name(
                    node, content
                )

                if element_type and element_name:
                    # Generate a unique key for this node
                    node_key = f"{element_name}:{node.start_point[0]}"
                    if node_key in seen_nodes:
                        continue
                    seen_nodes.add(node_key)

                    try:
                        # Get the full code
                        raw_code = content[node.start_byte : node.end_byte].decode(
                            "utf-8", errors="replace"
                        )

                        # Format the code
                        formatted_code = self._format_code(raw_code, lang_name)

                        # Use relative path for file_path
                        rel_file_path = os.path.relpath(file_path, self.repo_path)

                        # Create the element with additional context
                        code_elements.append(
                            {
                                "name": element_name,
                                "code": formatted_code,
                                "file_path": rel_file_path,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "node_type": node.type,  # Include original AST node type for reference
                            }
                        )
                        print(f"Found code element: {element_name} in {rel_file_path}")
                    except Exception as e:
                        print(f"Error creating element entry: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error processing nodes in {file_path}: {str(e)}")
            return code_elements

        if not code_elements:
            print(f"No code elements found in: {file_path}")
            print("\nAST Structure:")
            self._print_ast(tree.root_node, content)

        return code_elements

    def _iter_tree(self, node):
        """Helper method to iterate through all nodes in a tree."""
        yield node
        for child in node.children:
            yield from self._iter_tree(child)

    def _parse_file_at_commit(self, file_path: str, commit: str) -> List[Dict]:
        """
        Parse a file at a specific commit to extract code elements.

        Args:
            file_path: Path to the file
            commit: Commit hash

        Returns:
            List of code elements
        """
        content = self._get_file_content_at_commit(file_path, commit)
        if not content:
            return []

        return self._parse_file(file_path, content)

    def _generate_element_id(self, element: Dict) -> str:
        """
        Generate a unique ID for a code element.

        Args:
            element: Code element dictionary

        Returns:
            Unique ID string
        """
        rel_path = os.path.relpath(element["file_path"], self.repo_path)
        unique_str = f"{rel_path}:{element['name']}:{element['type']}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def analyze_repo_changes(self) -> Dict:
        """
        Analyze all files in the repository.

        Returns:
            Dictionary with code elements and metadata
        """
        print("\n=== Starting Repository Analysis ===")
        print(f"Repository path: {self.repo_path}")

        # Lists to track code elements
        all_elements = []
        scanned_files = set()

        # Walk through the repository and analyze all files
        for root, dirs, files in os.walk(self.repo_path):
            rel_root = os.path.relpath(root, self.repo_path)
            print(f"\nScanning directory: {rel_root}")

            # Skip excluded directories early
            dirs[:] = [
                d for d in dirs if not self._should_exclude_path(os.path.join(root, d))
            ]
            if dirs:
                print(f"Subdirectories to scan: {', '.join(dirs)}")

            # Filter and report files
            supported_files = []
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repo_path)

                # Skip if path should be excluded
                if self._should_exclude_path(file_path):
                    print(f"  Skipping excluded: {rel_path}")
                    continue

                # Check if file type is supported
                lang_name = self._get_file_language(file_path)
                if not lang_name or lang_name not in self.languages:
                    print(
                        f"  Skipping unsupported: {rel_path} (type: {lang_name if lang_name else 'unknown'})"
                    )
                    continue

                supported_files.append((file_path, rel_path, lang_name))

            if supported_files:
                print(f"\nFound {len(supported_files)} supported files in {rel_root}:")
                for file_path, rel_path, lang_name in supported_files:
                    print(f"  → {rel_path} ({lang_name})")
                    scanned_files.add(rel_path)

                    try:
                        # Parse the current state of the file
                        elements = self._parse_file(file_path)

                        # Add elements to our list
                        for element in elements:
                            element_id = self._generate_element_id(element)
                            all_elements.append(element)

                            # Update the elements database
                            self.code_elements_db["elements"][element_id] = {
                                "type": element["type"],
                                "name": element["name"],
                                "file_path": element["file_path"],
                                "code": element["code"],
                                "last_analyzed": self.repo.head.commit.hexsha,
                            }
                    except Exception as e:
                        print(f"Error analyzing file {rel_path}: {str(e)}")
                        continue

        # Update metadata
        self.code_elements_db["metadata"].update(
            {
                "last_processed_commit": self.repo.head.commit.hexsha,
                "total_elements": len(self.code_elements_db["elements"]),
                "last_analysis_timestamp": self.repo.head.commit.committed_datetime.isoformat(),
                "scanned_files": list(scanned_files),
            }
        )

        # Save the updated database
        self._save_code_elements_db()

        # Generate report JSON
        report = {
            "elements": all_elements,
            "stats": {
                "total_elements": len(all_elements),
                "elements_by_type": {},
                "files_scanned": len(scanned_files),
                "scanned_files": list(scanned_files),
            },
            "metadata": {
                "repository": os.path.basename(self.repo_path),
                "commit": self.repo.head.commit.hexsha,
                "timestamp": self.repo.head.commit.committed_datetime.isoformat(),
            },
        }

        # Count elements by type
        for element in all_elements:
            element_type = element["type"]
            if element_type not in report["stats"]["elements_by_type"]:
                report["stats"]["elements_by_type"][element_type] = 0
            report["stats"]["elements_by_type"][element_type] += 1

        # Save the report
        report_path = os.path.join(
            DOCAI_DIR, f"analysis_report_{self.repo.head.commit.hexsha[:7]}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n=== Analysis Summary ===")
        print(f"Files scanned: {len(scanned_files)}")
        print("Scanned files:")
        for file in sorted(scanned_files):
            print(f"  • {file}")
        print(f"\nTotal elements found: {len(all_elements)}")
        if all_elements:
            print("\nElement counts by type:")
            for type_name, count in report["stats"]["elements_by_type"].items():
                print(f"  • {type_name}: {count}")
        else:
            print("\nWarning: No code elements were found!")
            print("Supported file extensions:", ", ".join(LANGUAGE_EXTENSIONS.keys()))

        return report

    def _print_ast(self, node, content: bytes, level: int = 0):
        """Debug helper to print the AST structure."""
        indent = "  " * level
        node_text = content[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )
        node_text = node_text.replace("\n", "\\n")[:50]  # Truncate long lines
        print(f"{indent}{node.type}: {node_text}")
        for child in node.children:
            self._print_ast(child, content, level + 1)


def main():
    try:
        print("Starting code analysis...")
        analyzer = CodeAnalyzer(REPO_PATH)
        analyzer.analyze_repo_changes()
        print("Code analysis completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
