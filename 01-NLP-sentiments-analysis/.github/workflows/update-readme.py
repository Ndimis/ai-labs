import os
import yaml

ROOT = "."
README = "README.md"

projects = []

for folder in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, folder)
    meta = os.path.join(path, "project.yaml")

    if os.path.isdir(path) and os.path.isfile(meta):
        with open(meta, "r") as f:
            data = yaml.safe_load(f)

        projects.append({
            "category": data["category"],
            "name": data["name"],
            "description": data["description"],
            "link": f"./{folder}"
        })

# Build markdown table
lines = [
    "| Category | Project Name | Description | Link |",
    "|---------|--------------|-------------|------|"
]

for p in projects:
    lines.append(
        f"| {p['category']} | {p['name']} | {p['description']} | [{p['link']}]({p['link']}) |"
    )

table = "\n".join(lines)

with open(README, "r") as f:
    content = f.read()

new_content = content.split("<!-- PROJECT_TABLE_START -->")[0] + \
    "<!-- PROJECT_TABLE_START -->\n" + table + "\n<!-- PROJECT_TABLE_END -->" + \
    content.split("<!-- PROJECT_TABLE_END -->")[1]

with open(README, "w") as f:
    f.write(new_content)