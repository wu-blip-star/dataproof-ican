"""只读取数据，不修改原始文件。首行为列名，一行代表一个样本。"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
import csv

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_complex_dtype

MAX_FILE_BYTES = 20 * 1024 * 1024


class DataValidationError(ValueError):
    """可以直接展示给用户的数据问题。"""


@dataclass
class Dataset:
    frame: pd.DataFrame
    file_name: str
    sha256: str
    sheet_name: str | None
    parser: str

    @property
    def source_id(self) -> str:
        return f"{self.file_name}:{self.sha256}:{self.sheet_name}:{self.parser}"


def _check_bytes(content: bytes):
    if not content:
        raise DataValidationError("文件为空，请上传有表头和数据的文件。")
    if len(content) > MAX_FILE_BYTES:
        raise DataValidationError("文件超过 20 MB，请先缩小数据文件。")


def _engine(name: str) -> str:
    return "xlrd" if Path(name).suffix.lower() == ".xls" else "openpyxl"


def excel_sheets(content: bytes, file_name: str) -> list[str]:
    _check_bytes(content)
    try:
        with pd.ExcelFile(BytesIO(content), engine=_engine(file_name)) as book:
            return book.sheet_names
    except Exception as exc:
        raise DataValidationError("无法读取 Excel。请检查格式，文件不能加密或损坏。") from exc


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if is_numeric_dtype(frame[c])
            and not is_bool_dtype(frame[c]) and not is_complex_dtype(frame[c])]


def load_dataset(content: bytes, file_name: str, sheet_name: str | None = None) -> Dataset:
    _check_bytes(content)
    suffix = Path(file_name).suffix.lower()
    selected_sheet = None
    try:
        if suffix == ".csv":
            for encoding in ("utf-8-sig", "gb18030"):
                try:
                    decoded = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataValidationError("CSV 编码无法识别，请另存为 UTF-8 CSV。")
            # 明确只支持逗号分隔，保留空行以使数据行号可追溯。
            records = csv.reader(StringIO(decoded))
            raw_headers = next(records)
            if any(row and len(row) != len(raw_headers) for row in records):
                raise DataValidationError("CSV 数据列数与表头不一致，请检查每一条记录。")
            frame = pd.read_csv(StringIO(decoded), skip_blank_lines=False)
            parser = f"pandas.read_csv; encoding={encoding}; delimiter=comma; header=0"
        elif suffix in (".xlsx", ".xls"):
            sheets = excel_sheets(content, file_name)
            selected_sheet = sheet_name if sheet_name is not None else sheets[0]
            if selected_sheet not in sheets:
                raise DataValidationError("所选工作表不存在。")
            raw = pd.read_excel(BytesIO(content), sheet_name=selected_sheet,
                                header=None, engine=_engine(file_name))
            if raw.empty:
                raise DataValidationError("工作表为空。")
            raw_headers = raw.iloc[0].tolist()
            frame = pd.read_excel(BytesIO(content), sheet_name=selected_sheet,
                                  engine=_engine(file_name))
            parser = f"pandas.read_excel; engine={_engine(file_name)}; header=0"
        else:
            raise DataValidationError("仅支持 .csv、.xlsx 和 .xls 文件。")
        if not raw_headers or any(pd.isna(c) or not str(c).strip() for c in raw_headers):
            raise DataValidationError("存在空列名，请为每一列设置唯一名称。")
        headers = [str(c).strip() for c in raw_headers]
        if len(set(headers)) != len(headers):
            raise DataValidationError("存在重复列名，请修改后重新上传，避免变量映射歧义。")
        if len(headers) != len(frame.columns):
            raise DataValidationError("表头与数据列数不一致，请检查文件。")
        if frame.empty:
            raise DataValidationError("文件只有表头，没有可分析的样本。")
        frame.columns = headers
        return Dataset(frame.reset_index(drop=True), Path(file_name).name,
                       sha256(content).hexdigest(), selected_sheet, parser)
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError("文件解析失败。CSV 请使用逗号分隔；首行应为列名，后续各行列数须一致。") from exc
