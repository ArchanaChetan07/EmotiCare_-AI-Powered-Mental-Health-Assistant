import os

# Base folders
structure = {
    "data": ["raw", "processed", "external"],
    "notebooks": [
        "01_eda.ipynb",
        "02_preprocessing.ipynb",
        "03_modeling.ipynb"
    ],
    "src": [
        "__init__.py",
        "config.py",
        "preprocessing.py",
        "modeling.py",
        "chatbot_engine.py",
        "crisis_detector.py",
        "trend_visualizer.py",
        "dashboard.py"
    ],
    "app": [
        "main.py",
        "chat_ui.py",
        "journal_ui.py"
    ],
    "tests": [
        "test_model.py"
    ],
    "outputs": ["model", "logs", "visualizations"]
}

# Create folders and files
for folder, contents in structure.items():
    base = os.path.join("EmotiCare", folder)
    os.makedirs(base, exist_ok=True)
    for item in contents:
        path = os.path.join(base, item)
        if path.endswith(".py") or path.endswith(".ipynb"):
            open(path, "a").close()
        else:
            os.makedirs(path, exist_ok=True)

print("✅ EmotiCare project file structure created successfully.")
