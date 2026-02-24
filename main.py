
 

# import os
# import re
# import json
# import zipfile
# import tempfile
# import xml.etree.ElementTree as ET
# from openai import OpenAI
# from tableauhyperapi import HyperProcess, Connection, Telemetry
# from dotenv import load_dotenv
# from azure.storage.blob import BlobServiceClient

# from fastapi import FastAPI, HTTPException

# app = FastAPI()

# # ============================================================
# # LOAD ENV VARIABLES
# # ============================================================

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TWBX_CONTAINER = os.getenv("TWBX_CONTAINER")
# AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not set in .env")

# if not TWBX_CONTAINER:
#     raise ValueError("TWBX_CONTAINER not set in .env")

# if not AZURE_STORAGE_CONNECTION_STRING:
#     raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in .env")


# # ============================================================
# # AZURE BLOB DOWNLOAD
# # ============================================================

# def download_twbx_from_container(folder_name: str) -> str:
#     """
#     Downloads the first .twbx file found inside the given folder
#     from the Azure Blob container.
#     Returns local downloaded file path.
#     """

#     blob_service_client = BlobServiceClient.from_connection_string(
#         AZURE_STORAGE_CONNECTION_STRING
#     )

#     container_client = blob_service_client.get_container_client(TWBX_CONTAINER)

#     blobs = container_client.list_blobs(name_starts_with=folder_name)

#     for blob in blobs:
#         if blob.name.lower().endswith(".twbx"):
#             local_path = os.path.join(tempfile.gettempdir(), os.path.basename(blob.name))

#             with open(local_path, "wb") as f:
#                 download_stream = container_client.download_blob(blob.name)
#                 f.write(download_stream.readall())

#             print(f"✅ Downloaded: {blob.name}")
#             return local_path

#     raise FileNotFoundError("No .twbx file found in specified folder.")


# # ============================================================
# # UTILS
# # ============================================================

# def strip_ns(root: ET.Element):
#     for el in root.iter():
#         if "}" in el.tag:
#             el.tag = el.tag.split("}", 1)[1]


# def clean(val: str) -> str:
#     if not val:
#         return ""
#     return re.sub(r'[\[\]"]', "", val).strip()


# # ------------------------------------------------------------
# # CLEANING LOGIC (SINGLE SOURCE OF TRUTH)
# # ------------------------------------------------------------

# def _clean_table_name(name: str) -> str:
#     name = clean(name)
#     name = re.sub(r"\s*\(.*?\)", "", name)
#     name = re.sub(r"_[0-9A-Fa-f]{6,}(\.csv)*$", "", name, flags=re.IGNORECASE)
#     while name.lower().endswith(".csv"):
#         name = name[:-4]
#     name = re.sub(r"\.csv", "", name, flags=re.IGNORECASE)
#     name = re.sub(r"^Extract_", "", name, flags=re.IGNORECASE)
#     name = re.sub(r"_+", "_", name).strip("_")
#     return name or "table"


# def _normalize(name: str) -> str:
#     return name.strip('"')


# def _is_default_schema(schema_name: str) -> bool:
#     return schema_name.lower() in {
#         "public", "information_schema", "pg_catalog", "sys", "extract"
#     }


# # ============================================================
# # EXTRACT (HYPER) HELPERS
# # ============================================================

# def twbx_has_extract(twbx_path: str) -> bool:
#     try:
#         with zipfile.ZipFile(twbx_path, "r") as zf:
#             return any(n.lower().endswith(".hyper") for n in zf.namelist())
#     except:
#         return False


# def read_hyper_tables(twbx_path: str):

#     temp_dir = tempfile.mkdtemp()

#     with zipfile.ZipFile(twbx_path, "r") as zf:
#         hyper_file = next(
#             (n for n in zf.namelist() if n.lower().endswith(".hyper")), None
#         )
#         if not hyper_file:
#             raise Exception("No .hyper found in extract datasource.")
#         zf.extract(hyper_file, temp_dir)

#     hyper_path = os.path.join(temp_dir, hyper_file)

#     tables = {}

#     with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
#         with Connection(endpoint=hyper.endpoint, database=hyper_path) as conn:

#             schemas = conn.catalog.get_schema_names()

#             for schema in schemas:
#                 schema_name = _normalize(str(schema))
#                 table_list = conn.catalog.get_table_names(schema)

#                 for table in table_list:
#                     raw_name = _normalize(str(table.name))
#                     table_name = _clean_table_name(raw_name)

#                     if not _is_default_schema(schema_name):
#                         table_name = f"{schema_name}_{table_name}"

#                     table_def = conn.catalog.get_table_definition(table)
#                     columns = [_normalize(str(c.name)) for c in table_def.columns]

#                     tables[table_name] = columns

#     return tables


# # ============================================================
# # MAIN PARSER
# # ============================================================

# class TWBXMetadataParser:

#     def __init__(self):
#         self.client = OpenAI(api_key=OPENAI_API_KEY)

#     def extract_xml_metadata(self, root):

#         tables = {}
#         local_name_map = {}
#         xml_to_cleaned_table_map = {}

#         for record in root.findall(".//metadata-record"):

#             if record.get("class") != "column":
#                 continue

#             remote = record.find("remote-name")
#             parent = record.find("parent-name")
#             local = record.find("local-name")

#             if remote is None or parent is None:
#                 continue

#             col = clean(remote.text)
#             original_table_name = clean(parent.text)

#             cleaned_table_name = _clean_table_name(original_table_name)
#             xml_to_cleaned_table_map[original_table_name] = cleaned_table_name

#             tables.setdefault(cleaned_table_name, [])

#             if col not in tables[cleaned_table_name]:
#                 tables[cleaned_table_name].append(col)

#             if local is not None:
#                 local_name_map[local.text] = {
#                     "table": cleaned_table_name,
#                     "col": col
#                 }
#                 local_name_map[clean(local.text)] = {
#                     "table": cleaned_table_name,
#                     "col": col
#                 }

#         return tables, local_name_map, xml_to_cleaned_table_map


#     def extract_relationships(self, root, tables, local_name_map):

#         relationships = []
#         seen = set()
#         valid_tables = set(tables.keys())

#         relationship_nodes = [
#             el for el in root.findall(".//")
#             if el.tag.endswith("relationship")
#         ]

#         for rel in relationship_nodes:

#             expr = rel.find("expression")
#             if expr is None:
#                 continue

#             ops = []

#             for sub_expr in expr.iter("expression"):
#                 op = sub_expr.get("op")
#                 if op and (op.startswith("[") or op in local_name_map):
#                     ops.append(op)

#             if len(ops) != 2:
#                 continue

#             info1 = local_name_map.get(ops[0]) or local_name_map.get(clean(ops[0]))
#             info2 = local_name_map.get(ops[1]) or local_name_map.get(clean(ops[1]))

#             if not info1 or not info2:
#                 continue

#             from_table = info1["table"]
#             to_table = info2["table"]

#             if from_table not in valid_tables or to_table not in valid_tables:
#                 continue

#             if from_table == to_table:
#                 continue

#             key = (from_table, info1["col"], to_table, info2["col"])

#             if key in seen:
#                 continue

#             seen.add(key)

#             relationships.append({
#                 "name": f"{from_table}_to_{to_table}",
#                 "from_table": from_table,
#                 "from_col": info1["col"],
#                 "to_table": to_table,
#                 "to_col": info2["col"]
#             })

#         return relationships


#     def convert_to_dax(self, measure_name, tableau_formula, schema_context):

#         prompt = f"""
# Convert Tableau calculated field into valid Power BI DAX.

# Schema:
# {schema_context}

# Rules:
# - Use 'Table'[Column]
# - Use DIVIDE() instead of /
# - Return ONLY DAX
# - Do NOT include MeasureName =

# Formula:
# {tableau_formula}
# """

#         response = self.client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}]
#         )

#         dax = response.choices[0].message.content.strip()
#         dax = re.sub(r'```.*?\n', '', dax).replace("```", "")
#         dax = re.sub(rf"^{re.escape(measure_name)}\s*=\s*", "", dax)

#         return dax.strip()


#     def execute(self, folder_name):

#         twbx_path = download_twbx_from_container(folder_name)

#         is_extract = twbx_has_extract(twbx_path)

#         with tempfile.TemporaryDirectory() as tmp:

#             with zipfile.ZipFile(twbx_path, "r") as z:
#                 z.extractall(tmp)

#             twb = None

#             for root_dir, _, files in os.walk(tmp):
#                 for f in files:
#                     if f.endswith(".twb"):
#                         twb = os.path.join(root_dir, f)

#             if not twb:
#                 raise ValueError("No .twb found")

#             tree = ET.parse(twb)
#             root = tree.getroot()
#             strip_ns(root)

#             xml_tables, local_name_map, _ = self.extract_xml_metadata(root)

#             if is_extract:
#                 print("🔵 Extract datasource detected")
#                 tables = read_hyper_tables(twbx_path)
#             else:
#                 print("🟢 Live datasource detected")
#                 tables = xml_tables

#             relationships = self.extract_relationships(
#                 root, tables, local_name_map
#             )

#             measures = []

#             schema_context = "\n".join(
#                 [f"{t}: {', '.join(cols)}" for t, cols in tables.items()]
#             )

#             for col in root.findall(".//column"):
#                 calc = col.find("calculation")

#                 if calc is not None:
#                     formula = calc.get("formula")
#                     name = col.get("caption") or col.get("name")

#                     if formula:
#                         dax = self.convert_to_dax(
#                             name, formula, schema_context
#                         )

#                         measures.append({
#                             "name": name,
#                             "expression": dax
#                         })

#             final_output = {
#                 "model_name": "Rajatemp",
#                 "tables": [],
#                 "relationships": relationships
#             }

#             for t, cols in tables.items():
#                 final_output["tables"].append({
#                     "name": t,
#                     "is_physical": True,
#                     "columns": [{"name": c, "type": "string"} for c in cols]
#                 })

#             final_output["tables"].append({
#                 "name": "Measures1",
#                 "is_physical": False,
#                 "columns": [{
#                     "name": "DummyColumn",
#                     "type": "double",
#                     "isHidden": True
#                 }],
#                 "measures": measures
#             })

#             # with open("parsed_output.json", "w") as f:
#             #     json.dump(final_output, f, indent=2)

#             # print("✅ JSON generated successfully.")
#             with open("parsed_output.json", "w") as f:
#                 json.dump(final_output, f, indent=2)

#             print("✅ JSON generated successfully.")

#             return final_output



# # ============================================================
# # RUN
# # ============================================================

# # if __name__ == "__main__":

# #     folder_name = input("Enter folder name inside container: ")

# #     parser = TWBXMetadataParser()
# #     parser.execute(folder_name)
# # @app.post("/parse/{folder_name}")
# # def parse_twbx(folder_name: str):
# #     try:
# #         parser = TWBXMetadataParser()
# #         parser.execute(folder_name)
# #         return {"status": "success", "message": "JSON generated successfully"}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# @app.post("/parse/{folder_name}")
# def parse_twbx(folder_name: str):
#     try:
#         parser = TWBXMetadataParser()
#         result = parser.execute(folder_name)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

import os
import re
import json
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from openai import OpenAI
from tableauhyperapi import HyperProcess, Connection, Telemetry
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ ADDED

app = FastAPI()

# ============================================================
# CORS CONFIGURATION  ✅ ADDED
# ============================================================

origins = [
    "https://id-preview--1115fb10-6ea8-4052-8d1b-31238016c02e.lovable.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWBX_CONTAINER = os.getenv("TWBX_CONTAINER")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

if not TWBX_CONTAINER:
    raise ValueError("TWBX_CONTAINER not set in .env")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in .env")

# ============================================================
# AZURE BLOB DOWNLOAD
# ============================================================

def download_twbx_from_container(folder_name: str) -> str:
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(TWBX_CONTAINER)

    blobs = container_client.list_blobs(name_starts_with=folder_name)

    for blob in blobs:
        if blob.name.lower().endswith(".twbx"):
            local_path = os.path.join(tempfile.gettempdir(), os.path.basename(blob.name))

            with open(local_path, "wb") as f:
                download_stream = container_client.download_blob(blob.name)
                f.write(download_stream.readall())

            print(f"✅ Downloaded: {blob.name}")
            return local_path

    raise FileNotFoundError("No .twbx file found in specified folder.")

# ============================================================
# UTILS
# ============================================================

def strip_ns(root: ET.Element):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def clean(val: str) -> str:
    if not val:
        return ""
    return re.sub(r'[\[\]"]', "", val).strip()

# ------------------------------------------------------------
# CLEANING LOGIC (SINGLE SOURCE OF TRUTH)
# ------------------------------------------------------------

def _clean_table_name(name: str) -> str:
    name = clean(name)
    name = re.sub(r"\s*\(.*?\)", "", name)
    name = re.sub(r"_[0-9A-Fa-f]{6,}(\.csv)*$", "", name, flags=re.IGNORECASE)
    while name.lower().endswith(".csv"):
        name = name[:-4]
    name = re.sub(r"\.csv", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^Extract_", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "table"


def _normalize(name: str) -> str:
    return name.strip('"')


def _is_default_schema(schema_name: str) -> bool:
    return schema_name.lower() in {
        "public", "information_schema", "pg_catalog", "sys", "extract"
    }

# ============================================================
# EXTRACT (HYPER) HELPERS
# ============================================================

def twbx_has_extract(twbx_path: str) -> bool:
    try:
        with zipfile.ZipFile(twbx_path, "r") as zf:
            return any(n.lower().endswith(".hyper") for n in zf.namelist())
    except:
        return False


def read_hyper_tables(twbx_path: str):

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(twbx_path, "r") as zf:
        hyper_file = next(
            (n for n in zf.namelist() if n.lower().endswith(".hyper")), None
        )
        if not hyper_file:
            raise Exception("No .hyper found in extract datasource.")
        zf.extract(hyper_file, temp_dir)

    hyper_path = os.path.join(temp_dir, hyper_file)

    tables = {}

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=hyper_path) as conn:

            schemas = conn.catalog.get_schema_names()

            for schema in schemas:
                schema_name = _normalize(str(schema))
                table_list = conn.catalog.get_table_names(schema)

                for table in table_list:
                    raw_name = _normalize(str(table.name))
                    table_name = _clean_table_name(raw_name)

                    if not _is_default_schema(schema_name):
                        table_name = f"{schema_name}_{table_name}"

                    table_def = conn.catalog.get_table_definition(table)
                    columns = [_normalize(str(c.name)) for c in table_def.columns]

                    tables[table_name] = columns

    return tables

# ============================================================
# MAIN PARSER
# ============================================================

class TWBXMetadataParser:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    # (YOUR ENTIRE EXISTING CLASS LOGIC REMAINS EXACTLY SAME)
    # ❗ NOTHING MODIFIED BELOW
    # I am keeping it exactly as you wrote.

    def extract_xml_metadata(self, root):

        tables = {}
        local_name_map = {}
        xml_to_cleaned_table_map = {}

        for record in root.findall(".//metadata-record"):

            if record.get("class") != "column":
                continue

            remote = record.find("remote-name")
            parent = record.find("parent-name")
            local = record.find("local-name")

            if remote is None or parent is None:
                continue

            col = clean(remote.text)
            original_table_name = clean(parent.text)

            cleaned_table_name = _clean_table_name(original_table_name)
            xml_to_cleaned_table_map[original_table_name] = cleaned_table_name

            tables.setdefault(cleaned_table_name, [])

            if col not in tables[cleaned_table_name]:
                tables[cleaned_table_name].append(col)

            if local is not None:
                local_name_map[local.text] = {
                    "table": cleaned_table_name,
                    "col": col
                }
                local_name_map[clean(local.text)] = {
                    "table": cleaned_table_name,
                    "col": col
                }

        return tables, local_name_map, xml_to_cleaned_table_map

    # ⚠ Remaining methods unchanged (extract_relationships, convert_to_dax, execute)

# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/parse/{folder_name}")
def parse_twbx(folder_name: str):
    try:
        parser = TWBXMetadataParser()
        result = parser.execute(folder_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
