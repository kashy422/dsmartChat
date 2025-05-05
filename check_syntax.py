import ast

filename = "src/agent.py"
try:
    with open(filename, "r", encoding="utf-8") as file:
        source = file.read()
    
    ast.parse(source, filename)
    print(f"✅ No syntax errors found in {filename}")
except SyntaxError as e:
    line_num = e.lineno
    col_num = e.offset
    line = source.splitlines()[line_num - 1] if line_num <= len(source.splitlines()) else "LINE NOT FOUND"
    print(f"❌ Syntax error in {filename} at line {line_num}, column {col_num}:")
    print(f"Line {line_num}: {line}")
    print(f"Error message: {e}")
    
    # Show context (5 lines before and after)
    start = max(0, line_num - 6)
    end = min(len(source.splitlines()), line_num + 5)
    print("\nContext:")
    for i in range(start, end):
        prefix = "→ " if i + 1 == line_num else "  "
        print(f"{prefix}{i+1}: {source.splitlines()[i]}")
except Exception as e:
    print(f"❌ Error checking {filename}: {e}") 