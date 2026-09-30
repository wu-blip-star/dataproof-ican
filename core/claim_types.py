"""V3 structured claims. V4 may supply this same schema; no text parser is used."""
from dataclasses import asdict, dataclass, field
from typing import Any

from core.data import DataValidationError, numeric_columns


@dataclass
class StructuredClaim:
    claim_type: str
    variables: list[str]
    outcome: str
    predictors: list[str] = field(default_factory=list)
    group_variable: str | None = None
    comparison_groups: list[Any] = field(default_factory=list)
    direction: str = "positive"
    strength: str = "unspecified"
    significance_language: str = "statistical_significance"
    original_text: str = ""
    parsed_claim: dict | None = None

    def to_dict(self):
        import json
        # pandas category levels can be NumPy scalars, while the V4 interface is JSON.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False,
                                     default=lambda value: value.item() if hasattr(value, 'item') else str(value), allow_nan=False))

    def columns(self):
        return list(dict.fromkeys(self.variables + self.predictors + [self.outcome]
                                  + ([self.group_variable] if self.group_variable else [])))

    def validate(self, frame):
        kinds = {"correlation", "group_difference", "categorical_association", "regression_association"}
        if self.claim_type not in kinds:
            raise DataValidationError("未知研究问题类型。")
        if any(c not in frame.columns for c in self.columns()):
            raise DataValidationError("主张引用了不存在的变量。")
        if self.strength != "unspecified" or self.significance_language != "statistical_significance":
            raise DataValidationError("V3 仅核验方向及统计显著性，不核验强度或等效性主张。")
        numeric = numeric_columns(frame)
        if self.claim_type in ("correlation", "categorical_association"):
            if len(self.variables) != 2 or len(set(self.variables)) != 2 or self.outcome != self.variables[1]:
                raise DataValidationError("该问题需要两个不同变量，第二个变量应为结果变量。")
            if self.predictors or self.group_variable or self.comparison_groups:
                raise DataValidationError("相关/分类关联主张不能混入回归预测变量或分组条件。")
        if self.claim_type == "correlation":
            if any(v not in numeric for v in self.variables) or self.direction not in ("positive", "negative"):
                raise DataValidationError("相关主张要求两个数值变量和正/负方向。")
        if self.claim_type == "group_difference":
            if not self.group_variable or self.group_variable == self.outcome or self.outcome not in numeric:
                raise DataValidationError("分组比较需要分组变量及不同的数值结果变量。")
            if self.variables != [self.group_variable,self.outcome] or self.predictors:
                raise DataValidationError("分组主张的变量映射应严格对应分组变量及结果变量。")
            if self.direction not in ("positive", "negative", "different"):
                raise DataValidationError("请选择高于、低于或总体有差异。")
            if len(set(self.comparison_groups)) != len(self.comparison_groups):
                raise DataValidationError("比较组不能重复。")
            if self.direction != "different" and (len(self.comparison_groups) != 2 or self.comparison_groups[0] == self.comparison_groups[1]):
                raise DataValidationError("方向性两组比较必须明确两个不同组，方向为第一组减第二组。")
        if self.claim_type == "categorical_association" and self.direction != "associated":
            raise DataValidationError("分类变量独立性检验仅核验是否存在关联。")
        if self.claim_type == "regression_association":
            if self.group_variable or self.comparison_groups:
                raise DataValidationError("回归主张不能混入分组比较条件。")
            if (not self.predictors or len(set(self.predictors)) != len(self.predictors)
                    or self.outcome in self.predictors or self.outcome == "const"
                    or "const" in self.predictors or any(c not in numeric for c in self.predictors + [self.outcome])
                    or len(self.variables) != 1 or self.variables[0] not in self.predictors
                    or self.direction not in ("positive", "negative")):
                raise DataValidationError("回归要求不同的数值预测变量、数值结果及一个待核验系数；const 为保留名称。")


def claim_judgment(value, p, direction, alpha):
    if direction in ("positive", "negative"):
        expected = 1 if direction == "positive" else -1
        if value * expected < 0:
            return "Direction Conflict", ("样本方向与主张相反；" + ("反向证据达到显著。" if p < alpha else "反向证据也未达到显著，不能断言总体方向。"))
        if value == 0:
            return "Unsupported", "样本估计为零，未支持指定方向。"
    if p < alpha:
        return "Supported", "当前样本结果符合结构化主张，且 p < α；显著不等于效应强。"
    return "Evidence Insufficient", "当前证据不足（p ≥ α）；不显著不能证明没有关系或没有差异。"
