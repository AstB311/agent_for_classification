# Equipment Classification

**Utility** for determining the condition of equipment based on sensor data.

## Installation

1. Clone the repository:

    ```bash
    git clone <URL>
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running the tool, you need to start the local server.  
It will start automatically when running the utility via `__main__.py`.

You can use the built-in example requests.

1. Navigate to the folder:

    ```bash
    cd test
    ```

2. Run and select a file for testing.

## Modules Overview

**Folder `main_scripts`:**
1. `main.py` — command-line interface.  
2. `rules` — folder containing the rules for the utility.  

**Folder `src`:**
1. `classification_methods.py` — all methods related to data classification.  
2. `clusterization_methods.py` — all methods related to data clustering.  
3. `connector.py` — database connection class.  
4. `two_methods_include.py` — methods related to optimization and processing of classification and clustering results.

---

**Run with parameter:**
```bash
python -m project
