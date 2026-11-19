import ast
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nb_dir = Path(__file__).resolve().parent.parent / "notebooks"
for nb_path in sorted(nb_dir.glob("*.ipynb")):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    print(f"\n=== {nb_path.name} (nbformat={nb['nbformat']}.{nb['nbformat_minor']}, cells={len(nb['cells'])}) ===")
    for i, c in enumerate(nb["cells"]):
        cell_type = c["cell_type"]
        cell_id = c.get("id", f"cell_{i}")
        src = "".join(c["source"])
        src_lines = len(c["source"])
        preview = c["source"][0].strip()[:60] if c["source"] else "(empty)"
        if cell_type == "code" and not preview.startswith("%%writefile .env"):
            body = "\n".join(
                line for line in src.splitlines()
                if not line.startswith("%%writefile") and not line.startswith("!")
            )
            ast.parse(body, filename=f"{nb_path.name}:{cell_id}")
        print(f"  [{i:2d}] {cell_type:8s} | id={cell_id:20s} | lines={src_lines:3d} | {preview}")
print("\nAll code cells passed Python AST syntax check.")
